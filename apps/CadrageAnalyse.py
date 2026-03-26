import os
import json
import pandas as pd
import streamlit as st
from typing import List
from openai import OpenAI

# --- Clé OpenAI
api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# ===================== Helpers =====================

def _infer_variable_type(series: pd.Series) -> str:
    """Retourne 'quantitative' ou 'categorielle' (heuristique simple)."""
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.dropna().nunique()
        return "categorielle" if nunique <= 6 else "quantitative"
    return "categorielle"

def _top_categories(series: pd.Series, max_modalites: int = 200) -> List[str]:
    """Liste de modalités uniques triées par fréquence, sur TOUT le jeu de données (pas d'échantillon)."""
    vals = series.dropna().astype(str)
    counts = vals.value_counts()
    return counts.index.tolist()[:max_modalites]

def _ensure_table_schema(df_tbl: pd.DataFrame, all_vars: List[str], inferred_types: dict, modalites_map: dict) -> pd.DataFrame:
    """
    Garantit la présence de toutes les colonnes attendues et des lignes pour chaque variable du jeu de données.
    Réconcilie ajouts/suppressions et remplit les valeurs manquantes avec des défauts sûrs.
    """
    expected_cols = ["variable", "type", "illustrative", "cible", "sens_cible", "modalité_cat_cible", "modalités_disponibles"]
    if df_tbl is None or df_tbl.empty:
        df_tbl = pd.DataFrame(columns=expected_cols)

    # Ajoute colonnes manquantes
    for c in expected_cols:
        if c not in df_tbl.columns:
            df_tbl[c] = "" if c in ["type", "sens_cible", "modalité_cat_cible", "modalités_disponibles"] else False

    # Remet en place toutes les variables (ajoute manquantes)
    current_vars = set(df_tbl["variable"].tolist()) if "variable" in df_tbl.columns else set()
    missing = [v for v in all_vars if v not in current_vars]
    if missing:
        add_rows = []
        for v in missing:
            vtype = "quantitative" if inferred_types.get(v, "categorielle") == "quantitative" else "catégorielle"
            add_rows.append({
                "variable": v,
                "type": vtype,
                "illustrative": False,
                "cible": False,
                "sens_cible": "",
                "modalité_cat_cible": "",
                "modalités_disponibles": ", ".join(modalites_map.get(v, [])) if vtype == "catégorielle" else "",
            })
        df_tbl = pd.concat([df_tbl, pd.DataFrame(add_rows)], ignore_index=True)

    # Retire lignes dont la variable n'existe plus
    df_tbl = df_tbl[df_tbl["variable"].isin(all_vars)].reset_index(drop=True)

    # Met à jour 'type' et 'modalités_disponibles' (source de vérité = jeu de données)
    df_tbl["type"] = [
        "quantitative" if inferred_types.get(row["variable"], "categorielle") == "quantitative" else "catégorielle"
        for _, row in df_tbl.iterrows()
    ]
    df_tbl["modalités_disponibles"] = [
        ", ".join(modalites_map.get(row["variable"], [])) if (row["type"] == "catégorielle") else ""
        for _, row in df_tbl.iterrows()
    ]

    # Normalise colonnes bool
    for b in ["illustrative", "cible"]:
        try:
            df_tbl[b] = df_tbl[b].astype(bool)
        except Exception:
            pass

    # Normalise sens_cible / modalité_cat_cible
    df_tbl["sens_cible"] = df_tbl["sens_cible"].fillna("").astype(str).str.lower()
    df_tbl["sens_cible"] = df_tbl["sens_cible"].where(df_tbl["sens_cible"].isin(["", "min", "max"]), "")
    df_tbl["modalité_cat_cible"] = df_tbl["modalité_cat_cible"].fillna("").astype(str)

    return df_tbl[expected_cols]

def _call_llm(df: pd.DataFrame):
    # requête groupée de 4 appels LLM
    model = "gpt-4o-mini"
    temperature = 0

    context = f'''Tu es un·e data analyst senior. Réponds en français, clair et concis.
Structure attendue en 5 sections:
1) Le secteur (ex.: telecom, immobilier, santé, media, luxe...), le métier (marketing, RH, opérations, recherche, ...), le type de jeu de données (web analytics, CRM, enquête) associés au jeu de données
2) L'unité d'observation (ex: personne (marketing b2c, RH,...), organisation (marketing b2b), session (web analyticsâ€¦), transaction, produit, process,...)
3) Variables cibles ou d'intérêt (à estimer à partir du contexte implicite du jeu de données) â€” et pour chaque variable, quelle valeur peut constituer une cible : une modalité pour une variable catégorielle (comme "positif" pour une variable binaire), ou le sens ("min" ou "max") pour une variable quantitative (ex.: "min" pour les coûts, "max" pour les revenus)
4) Variables illustratives, qui décrivent les unités d'observation (ex.: caractéristiques socio-démographiquee pour les personnes), par opposition aux variables actives (opinions, comportements). Il y a généralement, hors variable cible, un nombre relativement équilibré entre le nombre d'illustratives et actives.
5) Description des variables (liste à puces avec le nom de la variable, une description de 2-5 phrases, type/format de la variable si évident). Décris les toutes, n'en oublie aucune.
6) Si des libellés de variables ne sont pas explicites, recommandations de reformulation.'''

    recommendations = f'''Tu es un·e data analyst senior. Réponds en français,  clair et concis.
Structure attendue en 6 sections, avec une introduction:
1) 3 à 5 insights potentiels à produire (mentionne les colonnes utilisées)
2) Les variables absentes du jeu de données, qui seraient utiles dans son contexte (liste : nom â€” pourquoi utile)
3) Sources pour obtenir ces variables (liste : nom — URL)
4) Recommandations pour améliorer ce jeu de données pour une prochaine analyse
'''

    object = f'''Tu es un·e data analyst senior. Réponds  en français, clair et concis.
Définit le contexte global du jeu de données (3 phrases).
Utilise la présentation du contexte qui a déjà été réalisée, ci-joint.'''

    variables = '''Tu es un·e data analyst senior. Réponds UNIQUEMENT en JSON valide (aucun texte hors JSON).
À partir de la liste de colonnes fournie, identifie :
- target_variables : liste ORDONNÉE des variables cibles (libellés EXACTS parmi 'columns'), de la plus évidente à la moins évidente.
- target_modalities : pour chaque variable cible, la modalité cible si catégorielle, ou le sens "min"/"max" si quantitative (objet {variable: modalité|min|max}).
- illustrative_variables : variables illustratives (libellés EXACTS).
Utilise l'analyse déjà réalisée (objet, contexte, recommandations).

Règles strictes:
1) N'invente pas de noms qui ne sont pas dans 'columns'.
2) Respecte la casse et l’orthographe EXACTES des colonnes.
3) Si aucune n'est claire, renvoie [].

Schéma de sortie JSON:
{
  "target_variables": ["..."],
  "target_modalities": {"var1": "min", "var2": "positif"},
  "illustrative_variables": ["..."]
}'''

    client = OpenAI(api_key=api_key)
    preview = df.head(min(10, len(df))).to_csv(index=False)
    payload = {"columns": list(df.columns), "data_sample_preview_as_csv": preview}

    # 1) Objet du jeu de données (texte)
    r1 = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        max_tokens=2000,
    )
    dataset_context = r1.choices[0].message.content

    # 2) Contexte & recommandations (texte)
    context_blob2 = {**payload, "dataset_context": dataset_context}
    r2 = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": recommendations},
            {"role": "user", "content": json.dumps(context_blob2, ensure_ascii=False)},
        ],
        max_tokens=2000,
    )
    dataset_recommendations = r2.choices[0].message.content

    # 3) Synthèse (texte)
    context_blob3 = {**payload, "dataset_context": dataset_context}
    r3 = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": object},
            {"role": "user", "content": json.dumps(context_blob3, ensure_ascii=False)},
        ],
        max_tokens=2000,
    )
    dataset_object = r3.choices[0].message.content

    # 4) Variables (JSON)
    context_blob4 = {
        **payload,
        "dataset_object": dataset_object,
        "dataset_context": dataset_context,
        "dataset_recommendations": dataset_recommendations,
    }
    r4 = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": variables},
            {"role": "user", "content": json.dumps(context_blob4, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    variables_fonctions = json.loads(r4.choices[0].message.content)
    target_vars        = variables_fonctions.get("target_variables", []) or []
    target_modalities  = variables_fonctions.get("target_modalities", {}) or {}
    illustrative_vars  = variables_fonctions.get("illustrative_variables", []) or []

    # Priorité au brief utilisateur si une variable cible est citée explicitement ou par son libellé exact
    brief_raw = str(st.session_state.get("dataset_key_questions_value", "") or "")
    brief_norm = brief_raw.lower()
    for col in df.columns:
        col_norm = str(col).lower()
        if col_norm in brief_norm and col not in target_vars:
            target_vars = [col] + target_vars
            st.session_state["brief_target_variable"] = col
    # Supprimer doublons en conservant l'ordre
    seen = set()
    target_vars = [v for v in target_vars if not (v in seen or seen.add(v))]

    return {
        "dataset_object": dataset_object,
        "dataset_context": dataset_context,
        "dataset_recommendations": dataset_recommendations,
        "target_variables": target_vars,
        "target_modalities": target_modalities,
        "illustrative_variables": illustrative_vars,
    }

# ===================== App =====================

def run():
    st.title("Cadrage de l'analyse")

    # État global minimal (pas de mode auto ici)
    st.session_state.setdefault("etape8_terminee", False)
    st.session_state.setdefault("dataset_object", None)
    st.session_state.setdefault("dataset_context", None)
    st.session_state.setdefault("dataset_recommendations", None)
    st.session_state.setdefault("llm_cached", False)
    st.session_state.setdefault("llm_suggestions", {})
    st.session_state.setdefault("var_table_validated_ok", False)
    st.session_state.setdefault("dataset_key_questions", None)
    st.session_state.setdefault("dataset_key_questions_saved", False)
    st.session_state.setdefault("dataset_key_questions_value", "")  

    # 1) récupération du dataset

    if "df_imputed_structural" in st.session_state:
        df = st.session_state.df_imputed_structural
    else:
        st.warning("Aucun jeu de données trouvé. Veuillez d'abord passer par l'application Préparation 1.")
        st.stop()

    # 2) Attribution du rôle des variables par LLM
    st.header("Contexte du jeu de données & rôle des variables")

    if not st.session_state["llm_cached"]:
        with st.spinner("Analyse par LLM en cours (initiale)..."):
            try:
                res = _call_llm(df)
            except Exception as e:
                st.error(f"Erreur appel LLM : {e}")
                st.stop()

        st.session_state["dataset_object"]           = res["dataset_object"]
        st.session_state["dataset_context"]         = res["dataset_context"]
        st.session_state["dataset_recommendations"] = res["dataset_recommendations"]
        st.session_state["llm_suggestions"] = {
            "target_variables":       res["target_variables"],
            "illustrative_variables": res["illustrative_variables"],
            "target_modalities":      res["target_modalities"],
        }
        st.session_state["llm_cached"] = True
    else:
        st.caption("âœ… Contexte généré.")
        
    with st.expander("â„¹ï¸ Détails du contexte du jeu de données", expanded=False):
        st.subheader("Objet du jeu de données")
        st.markdown(st.session_state["dataset_object"] or "")
        st.subheader("Contexte du jeu de données")
        st.markdown(st.session_state["dataset_context"] or "")
        st.subheader("Recommandations sur le jeu de données")
        st.markdown(st.session_state["dataset_recommendations"] or "")

        st.markdown("---")
        st.write("**Variables cibles (suggestion LLM) :**",
                ", ".join(st.session_state["llm_suggestions"].get("target_variables", [])) or "Aucune.")

        # Affichage lisible des modalités cibles (dict)
        tm_disp = st.session_state["llm_suggestions"].get("target_modalities", {}) or {}
        if tm_disp:
            st.write("**Modalités cibles (suggestion LLM) :**")
            st.json(tm_disp)
        else:
            st.write("**Modalités cibles (suggestion LLM) :** Aucune.")
        st.write("**Variables illustratives (suggestion LLM) :**",
                ", ".join(st.session_state["llm_suggestions"].get("illustrative_variables", [])) or "Aucune.")

    # 3) Tableau de validation
    st.header("Attribution du rôle des variables")

    # Types & modalités (sur TOUT le df)
    all_vars = list(df.columns)
    inferred_types = {c: _infer_variable_type(df[c]) for c in all_vars}
    modalites_map = {c: _top_categories(df[c], 200) for c in all_vars if inferred_types[c] == "categorielle"}

    # Construction initiale
    def _initial_table():
        tgt = set(st.session_state["llm_suggestions"].get("target_variables", []))
        illu = set(st.session_state["llm_suggestions"].get("illustrative_variables", []))
        prev_tm = st.session_state["llm_suggestions"].get("target_modalities", {}) or {}
        rows = []
        for v in all_vars:
            vtype = "quantitative" if inferred_types[v] == "quantitative" else "catégorielle"
            sens = ""
            modcat = ""
            if v in prev_tm:
                if vtype == "quantitative" and str(prev_tm[v]).lower() in {"min", "max"}:
                    sens = str(prev_tm[v]).lower()
                elif vtype == "catégorielle":
                    modcat = str(prev_tm[v])
            # Pré-remplissage cat. si cible suggérée
            if vtype == "catégorielle" and not modcat and v in tgt and modalites_map.get(v):
                modcat = modalites_map[v][0]  # plus fréquente
            rows.append({
                "variable": v,
                "type": vtype,
                "illustrative": v in illu,
                "cible": v in tgt,
                "sens_cible": sens,
                "modalité_cat_cible": modcat,
                "modalités_disponibles": ", ".join(modalites_map.get(v, [])) if vtype == "catégorielle" else "",
            })
        return pd.DataFrame(rows)

    if "var_table_df" not in st.session_state:
        st.session_state["var_table_df"] = _initial_table()
    else:
        st.session_state["var_table_df"] = _ensure_table_schema(
            st.session_state["var_table_df"], all_vars, inferred_types, modalites_map
        )

    with st.expander("â„¹ï¸ Explications sur le rôle des variables", expanded=False):
        st.write("""
            Les variables peuvent jouer plusieurs rôles: être cible, illustratives, ou médiatrices.\n
            - **Variable cible** : la variable que l'on cherche à analyser ou prédire.\n
            - **Variable illustrative** : une variable descriptive qui aide à comprendre les unités d'observation (ex.: caractéristiques socio-démographiques).\n
            Veuillez lire attentivement le tableau et ajuster les rôles des variables si nécessaire.\n
            
            Pour la modalité cible de la variable cible:\n
            - **Quantitatives** â†’ *Sens cible* = « min » ou « max ».\n
            - **Catégorielles** â†’ *Modalité (cat.)* parmi les valeurs observées (voir « Modalités disponibles »).\n
            
            Les cases *illustrative* et *cible* sont exclusives.\n
            
            Le tableau a été pré-rempli automatiquement par LLM.\n
            Souvent la modalité cible n'est pas renseignée, merci de la compléter manuellement si besoin.
        """)

    sens_options = ["", "min", "max"]
    all_cat_options = [""] + sorted({m for mods in modalites_map.values() for m in mods})

    edited_df = st.data_editor(
        st.session_state["var_table_df"],
        key="var_table_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["variable", "type", "modalités_disponibles"],
        column_config={
            "variable": st.column_config.TextColumn("Variable"),
            "type": st.column_config.TextColumn("Type (inféré)"),
            "illustrative": st.column_config.CheckboxColumn("Variable illustrative"),
            "cible": st.column_config.CheckboxColumn("Variable cible"),
            "sens_cible": st.column_config.SelectboxColumn(
                "Sens cible (quant.)", options=sens_options,
                help="Pour variables quantitatives : choisir « min » ou « max ». Laisser vide sinon."
            ),
            "modalité_cat_cible": st.column_config.SelectboxColumn(
                "Modalité (cat.)", options=all_cat_options,
                help="Choisir une modalité si la variable est catégorielle. Laisser vide sinon."
            ),
            "modalités_disponibles": st.column_config.TextColumn(
                "Modalités disponibles (lecture seule)",
                help="Liste des modalités observées (échantillonnée si >200)."
            ),
        }
    )

    # Validation & persistance du tableau (manuel)
    if st.button("Valider l'état du tableau", type="primary", key="validate_var_table"):
        errors = []
        # ne retenir qu'une variable cible
        selected_targets = edited_df.loc[edited_df["cible"] == True, "variable"].tolist()
        if len(selected_targets) > 1:
            errors.append(
                "â€¢ Merci de ne sélectionner quâ€™UNE seule variable cible (une seule case « cible » cochée)."
            )
        
        for _, row in edited_df.iterrows():
            if not row["cible"]:
                continue
            var = row["variable"]
            vtype = "quantitative" if inferred_types.get(var, "categorielle") == "quantitative" else "catégorielle"
            sens = (row.get("sens_cible") or "").strip().lower()
            modcat = (row.get("modalité_cat_cible") or "").strip()
            if vtype == "quantitative" and sens not in {"min", "max"}:
                errors.append(f"â€¢ {var} (quant.) : choisir « min » ou « max ».") 
            if vtype == "catégorielle":
                known = set(modalites_map.get(var, []))
                if not modcat:
                    errors.append(f"{var} (cat.) : choisir une modalité.")
                elif known and modcat not in known:
                    errors.append(f"{var} : modalité « {modcat} » absente des valeurs observées.")

        if errors:
            st.error("Merci de corriger avant validation :\n\n" + "\n".join(errors))
        else:
            st.session_state["var_table_df"] = _ensure_table_schema(edited_df.copy(), all_vars, inferred_types, modalites_map)

            # Construit les sélections finales
            sel = st.session_state["var_table_df"]
            st.session_state["target_variables"]       = sel.loc[sel["cible"], "variable"].tolist()
            st.session_state["illustrative_variables"] = sel.loc[sel["illustrative"], "variable"].tolist()

            target_modalities = {}
            for _, row in sel.iterrows():
                if not row["cible"]:
                    continue
                var = row["variable"]
                vtype = "quantitative" if inferred_types.get(var, "categorielle") == "quantitative" else "catégorielle"
                if vtype == "quantitative":
                    target_modalities[var] = (row.get("sens_cible") or "").strip().lower()
                else:
                    target_modalities[var] = (row.get("modalité_cat_cible") or "").strip()

            st.session_state["target_modalities"] = target_modalities
            st.session_state["var_table_validated_ok"] = True
            st.success("âœ… Tableau validé et enregistré.")

    # Récap lecture seule
    with st.expander("Voir l'état enregistré (récapitulatif)"):
        st.write("**Variables cibles :**", ", ".join(st.session_state.get("target_variables", [])) or "—")
        st.write("**Variables illustratives :**", ", ".join(st.session_state.get("illustrative_variables", [])) or "—")
        tm = st.session_state.get("target_modalities", {}) or {}
        st.write("**Modalités cibles :**", ", ".join([f"{v} â†’ {m}" for v, m in tm.items()]) or "â€”")


    # Paramètres de l'analyse (2/2)
    
    with st.expander("Paramètres de l'analyse 2/2"):
        if "outliers_percent_target" not in st.session_state:
            st.session_state["outliers_percent_target"] = 0.01
        st.slider(
            "Pourcentage d'outliers (contamination)",
            min_value=0.0, 
            max_value=20.0, 
            value=st.session_state.outliers_percent_target, 
            step=0.1,
            key = "outliers_percent_target",
            help="Choisissez la contamination (en %)"
        )

        if "distinct_threshold_continuous" not in st.session_state:
            st.session_state["distinct_threshold_continuous"] = 5
        st.number_input(
            "Seuil (nb de modalités distinctes) à partir duquel une variable NUMÉRIQUE est CONTINUE",
            min_value=2, max_value=20, value=5,
            step=1,
            key = "distinct_threshold_continuous"
        )
        
        if "num_quantiles" not in st.session_state:
            st.session_state["num_quantiles"] = 5
        st.slider(
            "Nombre de quantiles pour la discrétisation",
            min_value=2, max_value=20, value=5, 
            key="num_quantiles"
        )
        
        if "mod_freq_min" not in st.session_state:
            st.session_state["mod_freq_min"] = 0.9
        st.slider(
            "Fréquence du mode à partir de laquelle la discrétisation est binaire",
            min_value=0.80, max_value=0.99, value=0.9, step=0.01, 
            key="mod_freq_min"
        )

        if "n_clusters_segmentation" not in st.session_state:
            st.session_state["n_clusters_segmentation"] = 10
        st.slider(
            "Nombre de segments pour le profiling de la population",
            min_value=2, max_value=20, value=10, step=1, 
            key="n_clusters_segmentation"
        )

        if "n_clusters_target" not in st.session_state:
            st.session_state["n_clusters_target"] = 3
        st.slider(
            "Nombre de segments pour le profiling de la cible",
            min_value=2, max_value=15, value=3, step=1, 
            key="n_clusters_target"
        )
        
        if "kmodes_n_init" not in st.session_state:
            st.session_state["kmodes_n_init"] = 2
        st.slider(
            "Nombre d'itérations du Kmode", 
            min_value=2, max_value=20, value=2, step=1,
            key="kmodes_n_init"
        )

    # 4) Définition de la question clé (manuel)
    st.header("Orientation de l'analyse")
    default_q = st.session_state.get("dataset_key_questions_value", "")
    st.text_input("Précisez l'orientation que vous souhaitez donner à l'analyse (brief). Sinon écrivez 'aucune' :", key="dataset_key_questions_input", value=default_q)
    if st.session_state["dataset_key_questions_value"]:
        st.info(f"Brief déjà renseigné : {st.session_state['dataset_key_questions_value']}")

    if st.button("Enregistrer le brief"):
        val = st.session_state.get("dataset_key_questions_input", "")
        if not val.strip():
            st.warning("Veuillez définir le brief.")
        else:
            st.session_state["dataset_key_questions_value"] = val.strip()
            st.session_state["dataset_key_questions_saved"] = True
            st.success("âœ… Brief enregistré.")


    # État final de l'étape
    if st.session_state.get("var_table_validated_ok", False) and st.session_state.get("dataset_key_questions_saved", False):
        st.session_state["etape8_terminee"] = True
        st.success("Étape terminée. Vous pouvez lancer la prochaine étape (diagnostic des valeurs manquantes) depuis le menu à gauche.")

if __name__ == "__main__":
    run()
