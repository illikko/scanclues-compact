import os
import re
import unicodedata
from io import BytesIO

import pandas as pd
import streamlit as st
from openai import OpenAI
from core.df_registry import DFState, get_df, set_df
from utils import preparation_process

# ========= OpenAI =========
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODE_KEY = "__NAV_MODE__"


# ========= Détection colonnes verbatims =========
MISSING_STRINGS = {"", "nan", "na", "none", "nul", "null"}

# Diagnostic rapide (sans LLM) pour éviter les doubles passages
def run_diagnostic_only():
    df_src = get_df(DFState.RAW)
    if not isinstance(df_src, pd.DataFrame):
        st.session_state["has_verbatim_candidates"] = False
        st.session_state["verbatim_candidates"] = []
        st.session_state["verbatim_details"] = {}
        return
    df = df_src.copy()
    min_avg_len = 30
    min_unique_ratio = 0.7
    sample = df.head(200)
    sample_candidates, _ = detect_long_text_columns(sample, min_avg_len=min_avg_len, min_unique_ratio=min_unique_ratio)
    if not sample_candidates:
        st.session_state["has_verbatim_candidates"] = False
        st.session_state["verbatim_candidates"] = []
        st.session_state["verbatim_details"] = {}
        return
    candidates, details = detect_long_text_columns(df, min_avg_len=min_avg_len, min_unique_ratio=min_unique_ratio)
    st.session_state["has_verbatim_candidates"] = bool(candidates)
    st.session_state["verbatim_candidates"] = candidates
    st.session_state["verbatim_details"] = details

# -------- Init session_state --------
st.session_state.setdefault('syntheses_verbatim', None)


def is_non_empty_str(x) -> bool:
    if pd.isna(x):
        return False
    xs = str(x).strip()
    return xs.lower() not in MISSING_STRINGS

def _fingerprint(s: str) -> str:
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # retirer accents
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_long_text_columns(
    df: pd.DataFrame, 
    min_avg_len: int = 50, 
    min_unique_ratio: float = 0.7
    ):
    """
    Candidats si :
      - on ignore NA/vides
      - ratio d'unicité (sur non-vides normalisés) >= MIN_UNIQUE_RATIO
      - longueur moyenne (sur non-vides bruts) > min_avg_len
    Retourne (candidates, details)
    """
    candidates, details = [], {}
    for col in df.columns:
        s = df[col].astype(str)
        mask = s.apply(is_non_empty_str)
        s_ne = s[mask]
        n = len(s_ne)
        if n == 0:
            details[col] = {"n_non_empty": 0, "avg_len": 0.0, "unique_ratio": 0.0}
            continue

        avg_len = s_ne.str.len().mean()
        fp = s_ne.map(_fingerprint)
        unique_ratio = fp.nunique(dropna=False) / n
        is_candidate = (avg_len > min_avg_len) and (unique_ratio >= min_unique_ratio)
        if is_candidate:
            candidates.append(col)

        details[col] = {
            "n_non_empty": int(n),
            "avg_len": float(avg_len),
            "unique_ratio": float(unique_ratio),
        }
    return candidates, details

# ========= Comptage & concat =========
def count_words_chars(text: str):
    if not isinstance(text, str):
        text = str(text)
    words = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    chars = len(text)
    return words, chars

def concat_non_empty(series: pd.Series) -> str:
    non_empty = [str(x) for x in series if is_non_empty_str(x)]
    return "\n\n---\n\n".join(non_empty)  # séparateur clair

# ========= Lecture CSV =========
def try_read_csv(file_bytes: BytesIO, encoding: str, sep_choice: str):
    file_bytes.seek(0)
    if sep_choice != "auto":
        return pd.read_csv(file_bytes, encoding=encoding, sep=sep_choice)
    for sep in [";", ",", "\t"]:
        try:
            file_bytes.seek(0)
            return pd.read_csv(file_bytes, encoding=encoding, sep=sep)
        except Exception:
            pass
    file_bytes.seek(0)
    return pd.read_csv(file_bytes, encoding=encoding)  # fallback


# ========= App =========

def run_diagnostic_only():
    """
    Detection rapide des colonnes verbatims sans appel LLM.
    Stocke has_verbatim_candidates, verbatim_candidates, verbatim_details.
    """
    df_src = get_df(DFState.RAW)
    if not isinstance(df_src, pd.DataFrame):
        st.session_state["has_verbatim_candidates"] = False
        st.session_state["verbatim_candidates"] = []
        st.session_state["verbatim_details"] = {}
        return

    df = df_src.copy()
    min_avg_len = st.session_state.get("verbatim_min_avg_len", 30)
    min_unique_ratio = st.session_state.get("verbatim_min_unique_ratio", 0.7)

    sample = df.head(200)
    sample_candidates, _ = detect_long_text_columns(sample, min_avg_len=min_avg_len, min_unique_ratio=min_unique_ratio)
    if not sample_candidates:
        st.session_state["has_verbatim_candidates"] = False
        st.session_state["verbatim_candidates"] = []
        st.session_state["verbatim_details"] = {}
        return

    candidates, details = detect_long_text_columns(df, min_avg_len=min_avg_len, min_unique_ratio=min_unique_ratio)
    st.session_state["has_verbatim_candidates"] = bool(candidates)
    st.session_state["verbatim_candidates"] = candidates
    st.session_state["verbatim_details"] = details

def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    # déclaration des variables
    if "etape3_terminee" not in st.session_state:
        st.session_state["etape3_terminee"] = False
    if "syntheses_verbatim" not in st.session_state:
        st.session_state["syntheses_verbatim"] = None
    
    
    st.title("Synthèse de textes longs")
    st.write("Exemples : verbatims, questions ouvertes, descripions...")


    if st.session_state.syntheses_verbatim is not None:
        st.subheader("Synthèses déjà réalisées")
        st.text_area("Synthèse des verbatims", st.session_state.syntheses_verbatim, height=350)

    # upload du dataset
    df_src = get_df(DFState.RAW)
    if isinstance(df_src, pd.DataFrame):
        df = df_src.copy()
        st.success(f"Fichier chargé â€” {df.shape[0]} lignes Ã— {df.shape[1]} colonnes")
        st.write("#### Aperçu du jeu de données :", df)
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application Préparation 1.")
        st.stop()

    with st.expander("Paramètres de détection", expanded=False):
        min_avg_len = st.slider("Longueur moyenne minimale (caractères)", 10, 500, 30)
        MIN_UNIQUE_RATIO = st.slider("ratio de valeurs uniques", 0.4, 1.0, 0.7)       
            
    with st.expander("Paramètres de synthèse", expanded=False):
        temp = st.slider("Température", 0.0, 1.0, 0.2, 0.05)
        max_toks = st.slider("max_tokens", 200, 4000, 800, 50)
        limit_input = st.checkbox("Limiter la taille d'entrée envoyée au modèle ?", value=False)
        max_input_chars = st.number_input("Si oui, nombre max de caractères par colonne", min_value=1000, value=30000, step=1000)

    # Réutilisation d'un diagnostic pré-calculé
    if "has_verbatim_candidates" in st.session_state:
        candidates = st.session_state.get("verbatim_candidates", [])
        details = st.session_state.get("verbatim_details", {})
        if not candidates:
            st.info("Aucun verbatim détecté. Module ignoré.")
            set_df(DFState.VERBATIM_READY, df, step_name="VerbatimSummary/no-op-fast")
            st.session_state["etape3_terminee"] = True
            return
        goto_selection = True
    else:
        goto_selection = False
    # 1) Détection candidats
    if not goto_selection:
        sample = df.head(200)
        sample_candidates, _ = detect_long_text_columns(sample, min_avg_len=min_avg_len, min_unique_ratio=MIN_UNIQUE_RATIO)
        if not sample_candidates:
            st.info("Aucun verbatim détecté (échantillon). Module ignoré.")
            set_df(DFState.VERBATIM_READY, df, step_name="VerbatimSummary/no-op-fast")
            st.session_state["etape3_terminee"] = True
            st.session_state["has_verbatim_candidates"] = False
            st.session_state["verbatim_candidates"] = []
            st.session_state["verbatim_details"] = {}
            return
        candidates, details = detect_long_text_columns(df, min_avg_len=min_avg_len, min_unique_ratio=MIN_UNIQUE_RATIO)
        st.session_state["has_verbatim_candidates"] = bool(candidates)
        st.session_state["verbatim_candidates"] = candidates
        st.session_state["verbatim_details"] = details

    with st.expander("Détails détection colonnes"):
        det_table = []
        for col, d in details.items():
            det_table.append({
                "colonne": col,
                "non_vides": d["n_non_empty"],
                "longueur_moyenne_non_vides": round(d["avg_len"], 1),
                "ratio_unicite": round(d["unique_ratio"], 3),
                "candidat": col in candidates
            })
        st.dataframe(pd.DataFrame(det_table).sort_values(
            by=["candidat", "longueur_moyenne_non_vides"], ascending=[False, False]
        ), use_container_width=True)

    # 2) Sélection éditable
    st.markdown("### Sélectionner les colonnes à synthétiser")
    rows = []
    for col, d in details.items():
        rows.append({
            "colonne": col,
            "non_vides": d["n_non_empty"],
            "longueur_moyenne_non_vides": round(d["avg_len"], 1),
            "ratio_unicite": round(d["unique_ratio"], 3),
            "Sélection": col in candidates
        })
    sel_df = pd.DataFrame(rows)
    edited = st.data_editor(
        sel_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Sélection": st.column_config.CheckboxColumn(help="Inclure cette colonne dans l'analyse"),
            "ratio_unicite": st.column_config.NumberColumn(format="%.3f"),
            "longueur_moyenne_non_vides": st.column_config.NumberColumn(format="%.1f"),
        },
        hide_index=True,
        key="editor_select_cols"
    )
    selected_cols = edited.loc[edited["Sélection"], "colonne"].tolist()
    st.info(
        "Colonnes sélectionnées :\n" +
        ("\n".join(f"- {col}" for col in selected_cols) if selected_cols else "aucune")
    )

    if not selected_cols:
        st.info("Aucune colonne de verbatim.")
        set_df(DFState.VERBATIM_READY, df, step_name="VerbatimSummary/no-op")
        st.session_state["etape3_terminee"] = True
        return
    else:

        # 3) Comptages avant API
        stats_df = None
        if selected_cols:
            st.subheader("Comptage mots/caractères (avant appels API)")
            stats_rows = []
            for col in selected_cols:
                question = str(col)
                w_q, c_q = count_words_chars(question)
                concat_resp = concat_non_empty(df[col].astype(str))
                w_r, c_r = count_words_chars(concat_resp)
                stats_rows.append({
                    "colonne": col,
                    "mots_titre": w_q,
                    "caract_titre": c_q,
                    "mots_reponses_non_vides_total": w_r,
                    "caract_reponses_non_vides_total": c_r,
                    "nb_reponses_non_vides": int((df[col].astype(str).apply(is_non_empty_str)).sum())
                })
            stats_df = pd.DataFrame(stats_rows)
            st.dataframe(stats_df, use_container_width=True)

        # 4) Lancer les synthèses (1 appel/colonne)
        
        proceed = False
        if mode == "automatique":
            proceed = True
        else:
            if st.button("Lancer la synthèse par LLM"):
                proceed = True

        if proceed:
        
            parts = []
            prog = st.progress(0.0)

            for i, col in enumerate(selected_cols, start=1):
                question = str(col)
                responses_text = concat_non_empty(df[col].astype(str))
                if limit_input and len(responses_text) > max_input_chars:
                    responses_text = responses_text[:max_input_chars] + "\n\n[TRONQUÉE¦]"

                system_msg = (
                    "Tu es un analyste d'études qui produit des synthèses claires, structurées et actionnables. "
                    "Réponds en français, de manière concise."
                )
                user_msg = (
                    f"TITRE (question) : {question}\n\n"
                    f"RÉPONSES (non vides, brutes) :\n{responses_text}\n\n"
                    "TACHE : Produis une synthèse structurée (puces courtes) couvrant :\n"
                    "- Thèmes majeurs récurrents\n"
                    "- Points positifs et négatifs marquants\n"
                    "- Points d'opposition \n"
                    "- verbatims représentatifs (courts, entre guillemets)\n"
                    "N'invente pas au-delà des réponses. Pas de données personnelles."
                )

                with st.spinner(f"Synthèse en cours pour -{col}- "):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=float(temp),
                        max_tokens=int(max_toks),
                    )
                    summary = resp.choices[0].message.content
                block = f"{col}\n{summary}".strip()
                parts.append(block)
                prog.progress(i / max(1, len(selected_cols)))

            st.session_state.syntheses_verbatim = "\n\n".join(parts)
            st.success("Synthèses terminées âœ…")
            st.text_area("Synthèse des verbatims", st.session_state.syntheses_verbatim, height=350)

            # suppression des colonnes de verbatim
            df_ex_verbatim = df.drop(selected_cols, axis=1)
            st.subheader("Jeux de donnnées après élimination des colonnes de verbatims")
            set_df(DFState.VERBATIM_READY, df_ex_verbatim, step_name="VerbatimSummary")
            st.dataframe(df_ex_verbatim, use_container_width=True)
            st.session_state["etape3_terminee"] = True




