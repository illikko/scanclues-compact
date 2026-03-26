import streamlit as st
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


MODE_KEY = "__NAV_MODE__"

# ----------------------------
# 1) Encodage (inchangé, juste robuste si id_cols vide)
# ----------------------------
def encode_modalites_ponderees(
    df: pd.DataFrame,
    prefix: str = "sel",
    id_cols: Tuple[str, ...] = ("__row_id__",),
    poids_mode: str = "prefer_dataset",   # "prefer_dataset" | "dataset_only" | "rank_only"
    rank_base: str = "present_keys",      # "present_keys" | "max_index"
):
    pat = re.compile(rf"^{re.escape(prefix)}/(\d+)/(key|weight)$")
    key_cols, weight_cols = {}, {}

    for c in df.columns:
        m = pat.match(c)
        if m:
            i = int(m.group(1))
            kind = m.group(2)
            (key_cols if kind == "key" else weight_cols)[i] = c

    indices = sorted(set(key_cols) | set(weight_cols))
    if not indices:
        raise ValueError(f"Aucune colonne ne matche le pattern: {prefix}/{{i}}/(key|weight)")

    parts = []
    for i in indices:
        kcol = key_cols.get(i)
        wcol = weight_cols.get(i)

        tmp = df.loc[:, list(id_cols)].copy()
        tmp["i"] = i
        tmp["key"] = df[kcol] if kcol else np.nan
        tmp["weight_dataset"] = df[wcol] if wcol else np.nan
        parts.append(tmp)

    long = pd.concat(parts, ignore_index=True)
    long["key"] = long["key"].replace(r"^\s*$", np.nan, regex=True)
    long = long.dropna(subset=["key"]).copy()

    long = long.sort_values(list(id_cols) + ["i"])
    long["rank"] = long.groupby(list(id_cols)).cumcount() + 1

    if rank_base == "present_keys":
        n = long.groupby(list(id_cols))["key"].transform("size")
    elif rank_base == "max_index":
        n = long.groupby(list(id_cols))["i"].transform("max") + 1
    else:
        raise ValueError("rank_base doit être 'present_keys' ou 'max_index'")

    long["weight_rank"] = (n + 1 - long["rank"]).astype(float)
    long["weight_dataset"] = pd.to_numeric(long["weight_dataset"], errors="coerce")

    if poids_mode == "prefer_dataset":
        long["weight_final"] = long["weight_dataset"].fillna(long["weight_rank"])
    elif poids_mode == "dataset_only":
        long["weight_final"] = long["weight_dataset"]
    elif poids_mode == "rank_only":
        long["weight_final"] = long["weight_rank"]
    else:
        raise ValueError("poids_mode doit être 'prefer_dataset', 'dataset_only' ou 'rank_only'")

    X = (
        long.pivot_table(
            index=list(id_cols),
            columns="key",
            values="weight_final",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
    )
    X.columns.name = None
    return X


def build_standard_block(df: pd.DataFrame, key_cols: List[str], weight_cols: List[Optional[str]], prefix="sel"):
    out = df.copy()
    for i, kcol in enumerate(key_cols):
        out[f"{prefix}/{i}/key"] = out[kcol]
        wcol = weight_cols[i] if i < len(weight_cols) else None
        if wcol:
            out[f"{prefix}/{i}/weight"] = out[wcol]
    return out


# ----------------------------
# 2) Détection des groupes choix/poids (regex)
# ----------------------------
@dataclass
class RankedGroup:
    base: str
    key_cols_by_rank: Dict[int, str]
    weight_cols_by_rank: Dict[int, str]

def _clean_base(b: str) -> str:
    # retire espaces et séparateurs finaux
    b = (b or "").strip()
    b = re.sub(r"[\s_\-–—:/\\]+$", "", b).strip()
    return b

def detect_ranked_groups(
    df: pd.DataFrame,
    min_ranks: int = 2,
    require_consecutive: bool = True,
    consecutive_ratio_min: float = 0.7,
    max_rank: int = 200,
) -> List[RankedGroup]:
    """
    Détecte des groupes du type:
      - "Question ... ?_1", "...?_2", ... "...?_10"
      - "Question - 1", "Question - 2"
      - "Question (1)", "Question (2)"
    + poids optionnels:
      - "Question ... ?_1_weight" ou "... ?_1_poids"
      - "Question ... ?_weight_1" ou "... ?_poids_1"
    """

    cols = [str(c) for c in df.columns]

    # 1) KEY columns: base + rank
    key_patterns = [
        re.compile(r"^(?P<base>.+?)[\s_\-–—]+(?P<rank>\d{1,3})\s*$"),        # "..._10" "... - 2"
        re.compile(r"^(?P<base>.+?)\((?P<rank>\d{1,3})\)\s*$"),             # "...(2)"
        re.compile(r"^(?P<base>.+?)\[(?P<rank>\d{1,3})\]\s*$"),             # "...[2]"
        re.compile(r"^(?P<base>.*?)(?P<rank>\d{1,3})\s*$"),  # base = lettres/underscore etc., rank = chiffres finaux
    ]

    # 2) WEIGHT columns: same base + rank + (poids|weight|w)
    weight_patterns = [
        re.compile(r"^(?P<base>.+?)[\s_\-–—]+(?P<rank>\d{1,3})[\s_\-–—]+(?P<w>poids|weight|w)\s*$", re.I),
        re.compile(r"^(?P<base>.+?)[\s_\-–—]+(?P<w>poids|weight|w)[\s_\-–—]+(?P<rank>\d{1,3})\s*$", re.I),
        re.compile(r"^(?P<base>.+?)\((?P<rank>\d{1,3})\)[\s_\-–—]+(?P<w>poids|weight|w)\s*$", re.I),
        re.compile(r"^(?P<base>.+?)\[(?P<rank>\d{1,3})\][\s_\-–—]+(?P<w>poids|weight|w)\s*$", re.I),
    ]

    key_map: Dict[str, Dict[int, str]] = {}
    weight_map: Dict[str, Dict[int, str]] = {}

    # --- scan keys
    for c in cols:
        m = None
        for pat in key_patterns:
            m = pat.match(c)
            if m:
                break
        if not m:
            continue

        rank = int(m.group("rank"))
        if rank <= 0 or rank > max_rank:
            continue

        base = _clean_base(m.group("base"))
        if not base:
            continue

        key_map.setdefault(base, {})[rank] = c

    # --- scan weights
    for c in cols:
        for pat in weight_patterns:
            m = pat.match(c)
            if not m:
                continue
            rank = int(m.group("rank"))
            if rank <= 0 or rank > max_rank:
                continue
            base = _clean_base(m.group("base"))
            if not base:
                continue
            weight_map.setdefault(base, {})[rank] = c
            break

    groups: List[RankedGroup] = []
    for base, ranks_to_col in key_map.items():
        ranks = sorted(ranks_to_col.keys())
        if len(ranks) < min_ranks:
            continue

        # filtre "consécutivité" pour éviter des faux groupes (ex: base_2023, base_2024)
        if require_consecutive and len(ranks) >= 2:
            expected = set(range(min(ranks), max(ranks) + 1))
            ratio = len(set(ranks)) / len(expected) if expected else 1.0
            if ratio < consecutive_ratio_min:
                continue

        groups.append(
            RankedGroup(
                base=base,
                key_cols_by_rank=dict(sorted(ranks_to_col.items())),
                weight_cols_by_rank=dict(sorted(weight_map.get(base, {}).items())),
            )
        )

    groups.sort(key=lambda g: len(g.key_cols_by_rank), reverse=True)
    return groups


# ----------------------------
# 3) Application sur N groupes + merge + drop sources
# ----------------------------
def encode_all_ranked_groups(
    df: pd.DataFrame,
    groups: List[RankedGroup],
    id_cols: Optional[List[str]] = None,
    poids_mode: str = "prefer_dataset",
    rank_base: str = "present_keys",
    drop_sources: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Retourne:
      - df_final : df original + colonnes encodées pour chaque groupe
      - dropped_by_group : colonnes supprimées (debug)
    """
    work = df.copy()

    # ID optionnel -> fallback row id
    if not id_cols:
        work = work.reset_index(drop=True).copy()
        work["__row_id__"] = np.arange(len(work), dtype=np.int64)
        id_cols = ["__row_id__"]

    dropped_by_group: Dict[str, List[str]] = {}

    for gi, g in enumerate(groups, start=1):
        key_cols = [g.key_cols_by_rank[r] for r in sorted(g.key_cols_by_rank)]
        weight_cols = [g.weight_cols_by_rank.get(r) for r in sorted(g.key_cols_by_rank)]

        # construire bloc standard
        prefix = f"__selg{gi}__"
        df_std = build_standard_block(work, key_cols=key_cols, weight_cols=weight_cols, prefix=prefix)

        # encoder
        X = encode_modalites_ponderees(
            df_std,
            prefix=prefix,
            id_cols=tuple(id_cols),
            poids_mode=poids_mode,
            rank_base=rank_base,
        )

        # renommer colonnes encodées (éviter collisions inter-groupes)
        # colonnes non-id => features
        feat_cols = [c for c in X.columns if c not in id_cols]
        safe_base = re.sub(r"\s+", "_", (g.base.strip() if g.base else f"group{gi}"))
        safe_base = re.sub(r"[^\w\-]+", "_", safe_base).strip("_")
        X = X.rename(columns={c: f"{safe_base}__{c}" for c in feat_cols})

        # merge dans work
        work = work.merge(X, on=id_cols, how="left")

        # drop colonnes standard + sources
        to_drop = []
        # colonnes standard ajoutées
        for i in range(len(key_cols)):
            to_drop.append(f"{prefix}/{i}/key")
            # weight standard uniquement si existant
            if weight_cols[i]:
                to_drop.append(f"{prefix}/{i}/weight")

        if drop_sources:
            # sources = key cols + weight cols d'origine (si existantes)
            to_drop.extend(key_cols)
            to_drop.extend([w for w in weight_cols if w])

        # nettoyer (ignore si absentes)
        to_drop = [c for c in to_drop if c in work.columns]
        dropped_by_group[g.base or f"group{gi}"] = to_drop
        if to_drop:
            work = work.drop(columns=to_drop)

    # si on a créé __row_id__, on le retire à la fin
    if "__row_id__" in work.columns and ("__row_id__" not in (df.columns.tolist())):
        work = work.drop(columns=["__row_id__"])

    return work, dropped_by_group


# ----------------------------
# App Streamlit
# ----------------------------

def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    
    # déclaration des variables
    if "etape5_terminee" not in st.session_state:
        st.session_state["etape5_terminee"] = False
    
    st.header("Réponses multiples ordonnées")
    st.text("Concerne les groupes de colonnes de type réponse_1, réponse_2,...")
    st.text("Ce module les identifie et les traite: création d'indicatrices pour chaque modalité, avec pondération suivant le rang ou poids existant.")

    # rechargement du dataset
    if "df_shortlabels" in st.session_state:
        df = st.session_state.df_shortlabels
    elif "df_ex_verbatim" in st.session_state:
        df = st.session_state.df_ex_verbatim
    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")

    st.success("Fichier chargé.")
    st.dataframe(df.head(), use_container_width=True)

    # Détection groupes
    st.subheader("Détection automatique des groupes (choix_1, choix_2, ...)")
    groups = detect_ranked_groups(df, min_ranks=2)

    if not groups:
        st.warning("Aucun groupe de colonnes ordonnées détecté.")
        st.session_state.df_ex_ordonnees = df
        st.session_state["etape5_terminee"] = True
        return

    else:

        st.write(f"Groupes détectés : **{len(groups)}**")
        for g in groups:
            ranks = list(g.key_cols_by_rank.keys())
            st.write(f"- **{g.base or '(sans base)'}** : rangs {ranks} "
                    f"(poids trouvés: {len(g.weight_cols_by_rank)})")

        for g in groups:
            st.write("BASE:", g.base)
            st.write("KEY:", g.key_cols_by_rank)
            st.write("WEIGHT:", g.weight_cols_by_rank)

        # ID optionnel
        st.subheader("Colonnes identifiantes (optionnel)")
        id_cols = st.multiselect(
            "Si vide, l'encodage se fera ligne-par-ligne (index implicite).",
            options=df.columns.tolist(),
            default=[c for c in ["id", "label"] if c in df.columns]
        )

        # paramètres
        st.subheader("Paramètres")
        use_existing_weights = st.toggle("Utiliser les poids existants quand ils existent", value=True)
        poids_mode = "prefer_dataset" if use_existing_weights else "rank_only"

        rank_base = st.selectbox("Base du poids de rang", ["present_keys", "max_index"], index=0)
        drop_sources = st.toggle("Supprimer les colonnes sources (choix/poids) après encodage", value=True)
        
        proceed = False
        if mode == "automatique":
            proceed = True
        else:
            if st.button("Lancer l'encodage (tous les groupes)"):
                proceed = True

        if proceed:
            try:
                df_final, dropped = encode_all_ranked_groups(
                    df,
                    groups=groups,
                    id_cols=id_cols if id_cols else None,
                    poids_mode=poids_mode,
                    rank_base=rank_base,
                    drop_sources=drop_sources,
                )
                st.session_state.df_ex_ordonnees = df_final
                st.session_state.dropped = dropped

                st.success("Encodage terminé.")
                st.dataframe(st.session_state.df_ex_ordonnees.head(), use_container_width=True)

                with st.expander("Colonnes supprimées"):
                    st.json(dropped)

                st.session_state["etape5_terminee"] = True

            except Exception as e:
                st.error(f"Erreur pendant l'encodage : {e}")
                st.stop()



