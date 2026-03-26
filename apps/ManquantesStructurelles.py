import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from scipy.stats import chi2_contingency

# ============================================================
# App de détection de questions conditionnelles
# autres définition pour conditionnelles : avec valeurs manquantes structurelles, skip patterns
# - Construit columns_infos (label, type, top10 modalités)
# - Appel LLM avec response_format={"type":"json_object"}
# - Parse robuste + affichage + enrichissement columns_infos
# ============================================================

MODE_KEY = "__NAV_MODE__"

# Client OpenAI (clé via variable d'env)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))



# ----------------------------
# (A) Parsing question_id / sub_id
# ----------------------------
QUESTION_PATTERNS = [
    # 18-a, 18-a_1, 18-a.1
    re.compile(r"(?P<qid>\d+)\s*[-_.]\s*(?P<sub>[A-Za-z]+|\d+)\b"),
    # Q18a, Q18_a, Q18-1, Q18.1
    re.compile(r"\b[Qq]\s*(?P<qid>\d+)\s*[-_.]?\s*(?P<sub>[A-Za-z]+|\d+)\b"),
    # 18a, 18b (sans séparateur)
    re.compile(r"\b(?P<qid>\d+)(?P<sub>[A-Za-z])\b"),
    # Q18 (sans sous-id)
    re.compile(r"\b[Qq]\s*(?P<qid>\d+)\b"),
    # 18 (sans sous-id) â€” dernier recours, mais attention faux positifs (année, etc.)
    re.compile(r"(?P<qid>\d{1,4})\b"),
]

def parse_question_id(colname: str) -> tuple[str | None, str | None]:
    """
    Extrait (question_id, sub_id) depuis un nom de colonne.
    Heuristique : on teste plusieurs patterns; on renvoie le premier match plausible.
    """
    s = str(colname).strip()
    for pat in QUESTION_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        qid = m.groupdict().get("qid")
        sub = m.groupdict().get("sub")
        # Petit garde-fou : éviter de prendre une année type 2023 comme qid si pas de Q ou de sous-id
        if pat.pattern == QUESTION_PATTERNS[-1].pattern:
            if qid and len(qid) == 4 and int(qid) >= 1900 and int(qid) <= 2099:
                continue
        return (qid, sub)
    return (None, None)


# ----------------------------
# Helpers types/modalités
# ----------------------------
def infer_col_type(s: pd.Series, cat_unique_threshold: int = 30) -> str:
    """Infère un type haut-niveau: 'categorical', 'categorical_code', 'integer', 'continuous'."""
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return "categorical"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"

    if pd.api.types.is_numeric_dtype(s):
        nunique = s.nunique(dropna=True)
        if nunique > 0 and nunique <= cat_unique_threshold:
            return "categorical_code"
        if pd.api.types.is_integer_dtype(s):
            return "integer"
        x = s.dropna()
        if len(x) == 0:
            return "continuous"
        if np.all(np.isclose(x, np.round(x), atol=1e-9)):
            return "integer"
        return "continuous"

    return "categorical"


def top_modalities(s: pd.Series, max_modalities: int = 10) -> list[str]:
    vc = s.dropna().astype(str).value_counts()
    return vc.index.tolist()[:max_modalities]


def build_columns_infos(
    df: pd.DataFrame,
    max_modalities: int = 10,
    cat_unique_threshold: int = 30
) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        inferred_type = infer_col_type(s, cat_unique_threshold=cat_unique_threshold)
        nunique = int(s.nunique(dropna=True)) if s.nunique(dropna=True) is not None else 0

        qid, sub = parse_question_id(col)

        mods = top_modalities(s, max_modalities=max_modalities) if inferred_type in ("categorical", "categorical_code") else []
        modalities_json = json.dumps([str(x) for x in mods], ensure_ascii=False)

        rows.append({
            "column": col,
            "label": col,  # si vous avez un dictionnaire libellés, remplacez ici
            "question_id": qid,
            "sub_id": sub,
            "inferred_type": inferred_type,
            "n_unique": nunique,
            "missing_rate": float(s.isna().mean()),
            "modalities_json": modalities_json,
        })

    return pd.DataFrame(rows)


def columns_infos_to_payload(columns_infos: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in columns_infos.iterrows():
        records.append({
            "column": row["column"],
            "label": row["label"],
            "question_id": row.get("question_id"),
            "sub_id": row.get("sub_id"),
            "inferred_type": row["inferred_type"],
            "n_unique": int(row["n_unique"]),
            "missing_rate": float(row["missing_rate"]),
            "modalities": json.loads(row["modalities_json"]) if row.get("modalities_json") else [],
        })
    return records


# ----------------------------
# Data-driven skip detection
# ----------------------------
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    tbl = pd.crosstab(x, y)
    if tbl.size == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(tbl, correction=False)
    n = tbl.to_numpy().sum()
    if n == 0:
        return 0.0
    r, k = tbl.shape
    phi2 = chi2 / n
    denom = max(1e-12, min(k - 1, r - 1))
    return float(np.sqrt(phi2 / denom))


def detect_skip_candidates_data(
    df: pd.DataFrame,
    columns_infos: pd.DataFrame,
    children_missing_only: bool = True,
    parent_max_levels: int = 40,
    cat_unique_threshold: int = 30,
    min_support: int = 50,
    tau_high: float = 0.95,
    tau_low: float = 0.20,
    v_min: float = 0.30,
    max_parents_per_child: int = 5,
    max_triggers_store: int = 30
) -> pd.DataFrame:
    """
    Détection data-driven:
    - R_child = 1 si observé, 0 si manquant
    - Parent candidat = catégoriel/bool/low-card, et nunique <= parent_max_levels
    - Triggers = modalités parent où p_miss >= tau_high et support >= min_support
    - Ajoute metrics + score composite (B)
    """
    cols = df.columns.tolist()

    # Parents candidats
    parent_candidates = []
    for c in cols:
        nunique = df[c].nunique(dropna=True)
        if (df[c].dtype == "O" or str(df[c].dtype).startswith("category") or pd.api.types.is_bool_dtype(df[c]) or nunique <= cat_unique_threshold):
            if nunique <= parent_max_levels:
                parent_candidates.append(c)

    # Map question_id/sub_id pour score/hints
    qinfo = columns_infos.set_index("column")[["question_id", "sub_id"]].to_dict(orient="index")

    results = []
    for child in cols:
        miss_rate = float(df[child].isna().mean())
        if children_missing_only and miss_rate == 0.0:
            continue

        R = df[child].notna().astype(int)

        scored = []
        for parent in parent_candidates:
            if parent == child:
                continue

            tmp = pd.DataFrame({"x": df[parent], "R": R}).dropna(subset=["x"])
            if len(tmp) < min_support:
                continue

            v = cramers_v(tmp["x"].astype(str), tmp["R"])
            if v < v_min:
                continue

            grp = tmp.groupby("x")["R"].agg(["count", "mean"])
            grp["p_miss"] = 1 - grp["mean"]

            triggers = grp[(grp["count"] >= min_support) & (grp["p_miss"] >= tau_high)].index.tolist()
            if not triggers:
                continue

            low = grp[(grp["count"] >= min_support) & (grp["p_miss"] <= tau_low)].index.tolist()
            if not low:
                continue

            mask_trig = tmp["x"].isin(triggers)
            purity = float((tmp.loc[mask_trig, "R"] == 0).mean()) if mask_trig.any() else 0.0  # P(missing|trigger)
            coverage = float(df[parent].isin(triggers).mean())
            coverage_missing = float((df[parent].isin(triggers) & df[child].isna()).mean())
            incoherence_rate = float((df[parent].isin(triggers) & df[child].notna()).mean())

            trigger_p_miss_mean = float(grp.loc[triggers, "p_miss"].mean()) if len(triggers) else 0.0
            trigger_support_min = int(grp.loc[triggers, "count"].min()) if len(triggers) else 0
            trigger_support_sum = int(grp.loc[triggers, "count"].sum()) if len(triggers) else 0

            # (B) Score composite EDA-first (simple, interprétable)
            # - priorise impact réel des manquants (coverage_missing)
            # - favorise pureté élevée
            # - pénalise incohérences
            score = coverage_missing * purity * (1.0 - incoherence_rate)

            # Bonus léger (optionnel) : si même question_id, c'est plus plausible
            child_qid = qinfo.get(child, {}).get("question_id")
            parent_qid = qinfo.get(parent, {}).get("question_id")
            same_block = (child_qid is not None and parent_qid is not None and child_qid == parent_qid)
            if same_block:
                score *= 1.10  # petit bonus, pas une décision

            scored.append((
                score, coverage_missing, v, purity,
                parent, triggers, coverage, incoherence_rate,
                trigger_p_miss_mean, trigger_support_min, trigger_support_sum,
                same_block
            ))

        scored.sort(reverse=True, key=lambda t: (t[0], t[1], t[2]))
        for (score, coverage_missing, v, purity,
             parent, triggers, coverage, incoherence_rate,
             trigger_p_miss_mean, trigger_support_min, trigger_support_sum,
             same_block) in scored[:max_parents_per_child]:
            results.append({
                "child": child,
                "parent": parent,
                "score": float(score),
                "cramers_v": float(v),
                "missing_rate_child": miss_rate,
                "purity": float(purity),
                "coverage": float(coverage),
                "coverage_missing": float(coverage_missing),
                "incoherence_rate": float(incoherence_rate),
                "trigger_p_miss_mean": float(trigger_p_miss_mean),
                "trigger_support_min": int(trigger_support_min),
                "trigger_support_sum": int(trigger_support_sum),
                "same_question_block": bool(same_block),
                "trigger_values_json": json.dumps([str(x) for x in triggers[:max_triggers_store]], ensure_ascii=False),
                "n_triggers": int(min(len(triggers), max_triggers_store)),
            })

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values(["score", "coverage_missing", "cramers_v"], ascending=[False, False, False]).reset_index(drop=True)
    return out


# ----------------------------
# LLM qualification (C)
# ----------------------------
def llm_classify_candidates(
    columns_infos_records: list[dict],
    candidates_df: pd.DataFrame,
    top_n: int = 50,
    model: str = "gpt-4o-mini",
) -> dict:

    use_df = candidates_df.head(top_n).copy()

    cand_records = []
    for _, r in use_df.iterrows():
        cand_records.append({
            "parent": r["parent"],
            "child": r["child"],
            "trigger_values": json.loads(r["trigger_values_json"]),
            "metrics": {
                "score": float(r["score"]),
                "cramers_v": float(r["cramers_v"]),
                "purity": float(r["purity"]),
                "coverage_missing": float(r["coverage_missing"]),
                "incoherence_rate": float(r["incoherence_rate"]),
                "same_question_block": bool(r.get("same_question_block", False)),
            }
        })

    payload = {
        "columns_infos": columns_infos_records,
        "candidates": cand_records
    }

    prompt = f"""
Tu es un expert en méthodologie dâ€™enquêtes (marketing / sociologie) et en data management.

Objectif
- On te donne :
  (1) des infos sur toutes les colonnes (label/type/modalités top + question_id/sub_id quand détectable),
  (2) des CANDIDATS skip issus des données (analyse des manquants : triggers, pureté, incohérences, etc.).
- Ta tâche : VALIDER/REJETER et QUALIFIER ces candidats en règles "parent -> child".

Point important : numérotation des questions
- Les colonnes ont souvent des numéros (ex: 18-a / 18-b / 18-c, Q18a, etc.).
- Utilise "question_id" et "sub_id" comme INDICE fort :
  - même question_id => plus plausible relation au sein d'un bloc,
  - suffixes a/b/c => souvent sous-questions (follow-up),
  - mais ce n'est pas une preuve : si le texte contredit, rejette.

Interprétation des métriques (guide)
- purity proche de 1 = très compatible avec un skip (enfant manquant quand parent=trigger)
- incoherence_rate proche de 0 = cohérent (peu de cas parent trigger mais enfant observé)
- coverage_missing = impact réel (proportion de lignes concernées parmi les manquants)
- score = agrégat EDA (priorise impact + pureté + cohérence). Aide à prioriser, pas à décider seul.

Sortie JSON strict (uniquement)
{{
  "skip_rules":[
    {{
      "parent":"...",
      "child":"...",
      "evidence_text":"...",
      "condition_description":"...",
      "parent_trigger_values":["..."],
      "child_expected_when_triggered":"NA_STRUCT",
      "recommended_encoding":{{"categorical_dataset":"NA_STRUCT","quantitative_dataset":0}},
      "rule_type":"LOGICAL_SKIP|SCOPE_RESTRICTION|FOLLOW_UP",
      "confidence":0.0,
      "decision":"ACCEPT|REJECT",
      "reject_reason":""
    }}
  ],
  "notes":[]
}}

Définitions
- LOGICAL_SKIP : le libellé indique explicitement "si oui/non" / condition logique.
- SCOPE_RESTRICTION : "uniquement pour ..." / restriction de population.
- FOLLOW_UP : question de suivi implicite au sein d'un même bloc.

Encodage
- COMPTAGE ⇒ 0 : si child est un comptage, recommander quantitative_dataset = 0 quand non applicable.
- Pour reconnaître un comptage, utilisez ces indices (cumulatifs) :
- inferred_type âˆˆ {{integer, categorical_code}} avec valeurs numériques discrètes
- libellé/nom contient des mots-clés : combien, nombre, nb, count, how many, qty, quantité
- modalités typiques : 0,1,2,... ou â€œaucunâ€/â€œ0â€
- ATTRIBUT â‡’ NA_STRUCT : si câ€™est un attribut dâ€™un objet non existant (âge du plus jeune enfant, marque du dernier achat, montant payéâ€¦), recommander categorical_dataset = NA_STRUCT et quantitative_dataset = null.

Règles de prudence
- Sois conservateur: rejette si ambigu.
- N'inclus pas de règle avec confidence < 0.6.
- Si le texte ne supporte pas clairement la relation, REJECT même si le score est bon (risque de faux positifs).
- Si la relation est plausible mais le sens parent/child est incertain, REJECT et explique.

Input (JSON)
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def enrich_columns_infos_with_rules(columns_infos: pd.DataFrame, llm_rules: list[dict]) -> pd.DataFrame:
    """
    Ajoute dans columns_infos enrichi:
    - NA_STRUCT / rule types (bool)
    - struct_parent (meilleure règle)
    - struct_trigger_values_json
    - struct_action_categorical (ex: NA_STRUCT)
    - struct_action_quantitative (ex: 0 ou null)
    - struct_condition_description
    - struct_confidence
    - parents_json / children_json (listes agrégées)
    """

    out = columns_infos.copy()

    # Flags globaux
    out["NA_STRUCT"] = False
    out["LOGICAL_SKIP"] = False
    out["SCOPE_RESTRICTION"] = False
    out["FOLLOW_UP"] = False

    # (NOUVEAU) colonnes "règle structurante retenue"
    out["struct_parent"] = None
    out["struct_trigger_values_json"] = "[]"
    out["struct_action_categorical"] = None
    out["struct_action_quantitative"] = None
    out["struct_condition_description"] = None
    out["struct_confidence"] = None
    out["struct_rule_type"] = None

    # Agrégats parent/enfant
    parents_map = {c: [] for c in out["column"].tolist()}
    children_map = {c: [] for c in out["column"].tolist()}

    # Garder la "meilleure" règle par child
    best_rule_by_child = {}

    for r in llm_rules:
        if r.get("decision") != "ACCEPT":
            continue

        parent = r.get("parent")
        child = r.get("child")
        if not parent or not child:
            continue

        # agrégats
        if parent in children_map:
            children_map[parent].append(child)
        if child in parents_map:
            parents_map[child].append(parent)

        conf = float(r.get("confidence", 0.0) or 0.0)

        # choix meilleure règle: max confidence
        if (child not in best_rule_by_child) or (conf > float(best_rule_by_child[child].get("confidence", 0.0) or 0.0)):
            best_rule_by_child[child] = r

    # Appliquer flags + meilleure règle
    for child, r in best_rule_by_child.items():
        parent = r.get("parent")
        rule_type = r.get("rule_type")
        child_expected = r.get("child_expected_when_triggered")
        trig = r.get("parent_trigger_values", []) or []
        conf = float(r.get("confidence", 0.0) or 0.0)
        cond_desc = r.get("condition_description")

        rec_enc = r.get("recommended_encoding", {}) or {}
        cat_enc = rec_enc.get("categorical_dataset", "NA_STRUCT")
        quant_enc = rec_enc.get("quantitative_dataset", None)

        if child_expected == "NA_STRUCT":
            out.loc[out["column"] == child, "NA_STRUCT"] = True

        if rule_type in ("LOGICAL_SKIP", "SCOPE_RESTRICTION", "FOLLOW_UP"):
            out.loc[out["column"] == child, rule_type] = True

        # (NOUVEAU) stocker la règle structurante retenue
        out.loc[out["column"] == child, "struct_parent"] = parent
        out.loc[out["column"] == child, "struct_trigger_values_json"] = json.dumps([str(x) for x in trig], ensure_ascii=False)
        out.loc[out["column"] == child, "struct_action_categorical"] = cat_enc
        out.loc[out["column"] == child, "struct_action_quantitative"] = quant_enc
        out.loc[out["column"] == child, "struct_condition_description"] = cond_desc
        out.loc[out["column"] == child, "struct_confidence"] = conf
        out.loc[out["column"] == child, "struct_rule_type"] = rule_type

    # sérialiser parents/enfants (évite erreurs arrow)
    out["parents_json"] = out["column"].map(lambda c: json.dumps(parents_map.get(c, []), ensure_ascii=False))
    out["children_json"] = out["column"].map(lambda c: json.dumps(children_map.get(c, []), ensure_ascii=False))

    return out


def apply_structural_imputation(
    df: pd.DataFrame,
    columns_infos_enriched: pd.DataFrame,
    use_explicit_code_for_struct_na: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute les manquants structurels (skip patterns) en s'appuyant sur columns_infos_enriched.

    Règle :
    - On remplit UNIQUEMENT quand :
        parent ∈ triggers ET child est NA
    - Si child est quantitative :
        remplit avec struct_action_quantitative si défini, sinon laisse NA
    - Si child n'est pas quantitative :
        laisse NA par défaut, ou met "NA_STRUCT" si use_explicit_code_for_struct_na=True

    Hypothèse :
    - columns_infos_enriched contient une colonne 'is_quantitative'
      (bool, 0/1, ou libellé interprétable)
    """

    df_out = df.copy()
    audit_rows = []

    if columns_infos_enriched is None or columns_infos_enriched.empty:
        return df_out, pd.DataFrame([])

    needed_cols = [
        "column",
        "struct_parent",
        "struct_trigger_values_json",
        "struct_action_quantitative",
        "struct_confidence",
    ]
    for c in needed_cols:
        if c not in columns_infos_enriched.columns:
            raise ValueError(f"columns_infos_enriched doit contenir la colonne '{c}'")


    def parse_is_quantitative(row: pd.Series, child: str) -> bool:
        """
        Détermine si la variable enfant est quantitative.
        Priorité à inferred_type construit dans build_columns_infos().
        """

        inferred_type = row.get("inferred_type", None)

        if isinstance(inferred_type, str):
            t = inferred_type.strip().lower()

            # Quantitatives
            if t in {"integer", "continuous"}:
                return True

            # Catégorielles, même si codées numériquement
            if t in {"categorical", "categorical_code"}:
                return False

        # Compatibilité si tu ajoutes plus tard d'autres colonnes metadata
        candidate_fields = [
            "is_quantitative",
            "is_numeric",
            "variable_type",
            "column_type",
            "dtype_group",
            "semantic_type",
        ]

        for field in candidate_fields:
            if field not in row.index:
                continue

            value = row[field]

            if pd.isna(value):
                continue

            if isinstance(value, (bool, np.bool_)):
                return bool(value)

            if isinstance(value, (int, float, np.integer, np.floating)):
                return bool(value)

            if isinstance(value, str):
                v = value.strip().lower()
                if v in {"quantitative", "numeric", "num", "continuous", "discrete", "integer"}:
                    return True
                if v in {"categorical", "category", "qualitative", "text", "ordinal", "binary", "categorical_code"}:
                    return False

        # Par prudence, si on ne sait pas, on considère NON quantitative
        return False

    for _, row in columns_infos_enriched.iterrows():
        child = row["column"]
        parent = row["struct_parent"]

        if not parent or pd.isna(parent):
            continue
        if child not in df_out.columns or parent not in df_out.columns:
            continue

        try:
            triggers = json.loads(row["struct_trigger_values_json"] or "[]")
        except Exception:
            triggers = []

        if not triggers:
            continue

        mask_struct = df_out[parent].isin(triggers) & df_out[child].isna()
        n_to_fill = int(mask_struct.sum())
        if n_to_fill == 0:
            continue

        is_quantitative = parse_is_quantitative(row, child)

        if is_quantitative:
            fill_value = row.get("struct_action_quantitative", None)

            if fill_value is None or (isinstance(fill_value, float) and np.isnan(fill_value)):
                audit_rows.append({
                    "child": child,
                    "parent": parent,
                    "n_filled": 0,
                    "filled_with": None,
                    "mode": "left_as_NA",
                    "reason": "quantitative_action_missing",
                    "is_quantitative": True,
                    "confidence": float(row.get("struct_confidence") or 0.0),
                })
                continue

            df_out.loc[mask_struct, child] = fill_value
            audit_rows.append({
                "child": child,
                "parent": parent,
                "n_filled": n_to_fill,
                "filled_with": fill_value,
                "mode": "quantitative_action",
                "is_quantitative": True,
                "confidence": float(row.get("struct_confidence") or 0.0),
            })

        else:
            if use_explicit_code_for_struct_na:
                df_out.loc[mask_struct, child] = "NA_STRUCT"
                audit_rows.append({
                    "child": child,
                    "parent": parent,
                    "n_filled": n_to_fill,
                    "filled_with": "NA_STRUCT",
                    "mode": "structural_na_code",
                    "is_quantitative": False,
                    "confidence": float(row.get("struct_confidence") or 0.0),
                })
            else:
                audit_rows.append({
                    "child": child,
                    "parent": parent,
                    "n_filled": 0,
                    "filled_with": None,
                    "mode": "left_as_NA",
                    "reason": "categorical_structural_missing",
                    "is_quantitative": False,
                    "confidence": float(row.get("struct_confidence") or 0.0),
                })

    audit = pd.DataFrame(audit_rows)
    return df_out, audit


def apply_structural_imputation_auto(
    df: pd.DataFrame,
    columns_infos_enriched: pd.DataFrame,
    use_explicit_code_for_struct_na: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_structural_imputation(
        df=df,
        columns_infos_enriched=columns_infos_enriched,
        use_explicit_code_for_struct_na=use_explicit_code_for_struct_na,
    )

# ============================================================
# UI
# ============================================================
def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    
    if "etape7_terminee" not in st.session_state:
        st.session_state["etape7_terminee"] = False
    
    st.title("Données manquantes structurelles")
    st.caption("Données mmanquantes structurelles = questions conditionnelles, skip patters.")
    st.caption("Ce module les identifie et les traite.")
        
    st.subheader("1) Import du fichier")


    # rechargement du dataset
    if "df_ex_multiples" in st.session_state:
        df = st.session_state.df_ex_multiples
    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")

    st.success("Fichier chargé.")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        sep = st.text_input("Séparateur CSV", value=";")
    with c2:
        encoding = st.text_input("Encodage", value="latin-1")
    with c3:
        header_opt = st.checkbox("Première ligne = header", value=True)


    st.subheader("2) Paramètres")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        max_modalities = st.slider("Modalités top-N (résumé) par variable", 3, 30, 10)
    with p2:
        cat_unique_threshold = st.slider("Seuil cardinalité â†’ categorical_code", 5, 100, 30)
    with p3:
        parent_max_levels = st.slider("Max niveaux parent (au-delà ignoré)", 5, 200, 40)
    with p4:
        min_support = st.slider("min_support (par modalité trigger)", 10, 500, 50)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        tau_high = st.slider("tau_high (p_miss trigger)", 0.70, 0.999, 0.95)
    with q2:
        tau_low = st.slider("tau_low (p_miss non-trigger)", 0.00, 0.50, 0.20)
    with q3:
        v_min = st.slider("v_min (Cramér's V min)", 0.00, 1.00, 0.30)
    with q4:
        max_parents_per_child = st.slider("Max parents par enfant", 1, 10, 5)

    children_missing_only = st.checkbox("Analyser uniquement les variables ayant des manquants", value=True)

    st.subheader("3) Détection à partir des obsservations")
    st.write("Détection à partir des observations (métriques + scores)")
    
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Lancer la détection par métriques"):
            proceed = True

    if proceed:
        columns_infos = build_columns_infos(
            df,
            max_modalities=max_modalities,
            cat_unique_threshold=cat_unique_threshold
        )

        st.session_state["columns_infos"] = columns_infos

        st.markdown("### Information sur les variables")
        st.dataframe(columns_infos.head(50))

        candidates_df = detect_skip_candidates_data(
            df=df,
            columns_infos=columns_infos,
            children_missing_only=children_missing_only,
            parent_max_levels=parent_max_levels,
            cat_unique_threshold=cat_unique_threshold,
            min_support=min_support,
            tau_high=tau_high,
            tau_low=tau_low,
            v_min=v_min,
            max_parents_per_child=max_parents_per_child,
        )

        st.session_state["candidates_df"] = candidates_df

        if candidates_df.empty:
            st.info("Aucun candidat détecté avec les seuils actuels. Essayez de baisser v_min ou tau_high, ou min_support.")
            st.session_state.df_imputed_structural = df
            st.session_state["etape7_terminee"] = True
            return
            
        else:
            # Tableau trié par score (déjà trié), affichage complet
            st.markdown("### Candidats identifiés par métriques + score")        
            st.dataframe(st.session_state.candidates_df)

        
    st.subheader("4) Qualification par LLM")
    if client is None:
        st.warning("Clé OpenAI absente : étape LLM désactivée.")
    else:
        if "candidates_df" not in st.session_state:
            st.info("Lancez la détection par métriques.")
        else:
            candidates_df = st.session_state["candidates_df"]
            columns_infos = st.session_state.get("columns_infos")

            if candidates_df is None or candidates_df.empty:
                st.info("Aucun candidat à qualifier.")
            else:
                r1, r2 = st.columns(2)
                with r1:
                    top_n = st.slider("Nombre de candidats envoyés au LLM (top N)", 5, 300, 50)
                with r2:
                    model = st.text_input("Modèle", value="gpt-4o-mini")

                proceed = False
                if mode == "automatique":
                    proceed = True
                else:
                    if st.button("Qualifier avec LLM"):
                        proceed = True

                if proceed:
                    
                    try:
                        cols_records = columns_infos_to_payload(columns_infos)
                        llm_result = llm_classify_candidates(
                            columns_infos_records=cols_records,
                            candidates_df=candidates_df,
                            top_n=top_n,
                            model=model,
                        )

                        st.session_state["llm_result"] = llm_result

                        rules = llm_result.get("skip_rules", []) or []
                        st.markdown("### Qualification des candidats par LLM")
                        if rules:
                            rules_df = pd.DataFrame(rules)
                            st.dataframe(rules_df)

                            accepted = rules_df[rules_df["decision"] == "ACCEPT"].copy()
                            rejected = rules_df[rules_df["decision"] == "REJECT"].copy()

                            st.markdown("#### ACCEPT")
                            st.dataframe(accepted if not accepted.empty else pd.DataFrame({"info": ["Aucune règle acceptée."]}))

                            st.markdown("#### REJECT")
                            st.dataframe(rejected if not rejected.empty else pd.DataFrame({"info": ["Aucune règle rejetée."]}))

                            enriched = enrich_columns_infos_with_rules(columns_infos, rules)
                            st.session_state["columns_infos_enriched"] = enriched

                        else:
                            st.info("Aucune règle retournée par le LLM.")

                        notes = llm_result.get("notes", []) or []
                        if notes:
                            st.markdown("### Notes")
                            for n in notes:
                                st.write(f"- {n}")

                    except Exception as e:
                        st.error(f"Erreur appel LLM / parsing JSON : {e}")
                        
    st.markdown("### Tableau complet des variables avec les règles")                        
    st.dataframe(st.session_state["columns_infos_enriched"])                

    st.subheader("5) Aide à lâ€™interprétation des métriques")
    st.markdown(
        """
    - **trigger_values** (valeurs déclencheuses) : modalités du parent pour lesquelles lâ€™enfant est *presque toujours manquant*  
    → condition : `p_miss(x)=P(enfant manquant | parent=x) >= tau_high` et `count(x) >= min_support`.

    - **purity** : `P(enfant manquant | parent ∈ triggers)`  
    â†’ proche de 1 = très compatible skip.

    - **coverage** : `P(parent ∈ triggers)`  
    → combien d’observations sont dans la zone “trigger”.

    - **coverage_missing** : `P(parent ∈ triggers AND enfant manquant)`  
    â†’ impact réel sur les manquants (priorisation EDA).

    - **incoherence_rate** : `P(parent âˆˆ triggers AND enfant observé)`  
    â†’ doit être proche de 0 pour un skip propre (sinon incohérences ou relation plus complexe).

    - **score** : `coverage_missing * purity * (1 - incoherence_rate)`  
    â†’ score EDA simple pour trier/prioriser, pas une preuve à lui seul.

    - **same_question_block** : vrai si parent et child partagent le même `question_id` détecté  
    â†’ indice utile (bloc 18-a/18-b/18-c), mais pas une règle.
    """
    )
    
    st.subheader("6) Imputation des manquantes structurelles")

    # 1) Préconditions
    if "df_ex_multiples" not in st.session_state:
        st.error("Aucun dataset en session.")
        st.stop()

    if "columns_infos_enriched" not in st.session_state:
        st.info("Aucune règle structurante disponible. Lance d'abord l'étape 4) Qualification par LLM pour créer columns_infos_enriched.")
    else:
        # 2) Choix UI (simple et explicite)
        use_explicit_code = st.checkbox(
            "Pour les variables sans action quantitative (pas de 0 recommandé), coder explicitement 'NA_STRUCT' au lieu de laisser NA",
            value=True
        )

        show_audit_only_filled = st.checkbox(
            "Audit : afficher uniquement les variables où au moins une imputation a été appliquée",
            value=True
        )

        # 3) Action
        proceed = False
        if mode == "automatique":
            proceed = True
        else:
            if st.button("Appliquer l'imputation structurelle"):
                proceed = True

        if proceed:
        
            df_in = st.session_state["df_ex_multiples"]
            enriched = st.session_state["columns_infos_enriched"]

            df_imputed, audit = apply_structural_imputation(
                df=df_in,
                columns_infos_enriched=enriched,
                use_explicit_code_for_struct_na=use_explicit_code
            )

            # 4) Stockage en session (clair et non ambigu)
            st.session_state["df_imputed_structural"] = df_imputed
            st.session_state["audit_imputation_structural"] = audit

            st.success("Imputation structurelle appliquée. Résultat stocké dans st.session_state['df_imputed_structural'].")

            # 5) Affichages
            st.markdown("### Résultat des imputations")
            if audit is None or audit.empty:
                st.info("Aucune cellule n'a été imputée (aucun NA structurel détecté, ou aucune action quantitative définie).")
            else:
                audit_view = audit.copy()
                if show_audit_only_filled and "n_filled" in audit_view.columns:
                    audit_view = audit_view[audit_view["n_filled"] > 0]

                st.dataframe(audit_view, use_container_width=True)

            st.markdown("### Aperçu du dataset imputé")
            st.dataframe(df_imputed.head(), use_container_width=True)
            st.session_state["etape7_terminee"] = True
            
            csv = df_imputed.to_csv(index=False, sep=';', encoding='utf-8')
            st.download_button("Télécharger le dataset", csv, "df_imputed.csv", "text/csv")


