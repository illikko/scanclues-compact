import pandas as pd
import streamlit as st

from core.df_registry import DFState, get_df
from .ReponsesMultiples import detect_multimodal_config
from .ReponsesMultiplesOrdonnees import detect_ranked_groups


def _get_input_df() -> pd.DataFrame | None:
    for state in (
        DFState.IMPUTED_STRUCTURAL,
        DFState.MULTI_DONE,
        DFState.MULTI_ORD_DONE,
        DFState.SHORT_LABELS,
        DFState.VERBATIM_READY,
        DFState.RAW,
    ):
        df = get_df(state)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None


def _run_diagnostics(df: pd.DataFrame) -> dict:
    label_too_long = [c for c in df.columns if len(str(c)) > 50]
    missing_pct = float(df.isna().sum().sum() / max(1, df.size) * 100)

    ranked_groups = detect_ranked_groups(df, min_ranks=2)
    multi_det = detect_multimodal_config(df)

    # Proxy robuste pour "manquantes structurelles candidates".
    missing_cols = int((df.isna().sum() > 0).sum())
    skip_candidates_count = max(missing_cols, len(ranked_groups))

    return {
        "label_too_long_count": len(label_too_long),
        "label_too_long_cols": label_too_long,
        "missing_pct": missing_pct,
        "ranked_groups_count": len(ranked_groups),
        "multi_detected": bool(multi_det.ok),
        "multi_sep": multi_det.sep,
        "skip_candidates_count": int(skip_candidates_count),
    }


def run():
    st.title("Diagnostic Global")

    st.session_state.setdefault("etape2_terminee", False)
    st.session_state.setdefault(
        "pipeline_selection",
        {"preparation": True, "profilage": False, "analyse_descriptive": False},
    )
    st.session_state.setdefault("pipeline_ready_to_run", False)
    st.session_state.setdefault("pipeline_diagnostics", {})
    st.session_state.setdefault("dataset_key_questions_value", "")
    st.session_state.setdefault("dataset_key_questions_mode", "sb")  # sb=Sans brief, ab=Avec brief

    df = _get_input_df()
    if not isinstance(df, pd.DataFrame):
        st.warning("Aucun dataset disponible. Lancez d'abord l'etape Upload.")
        st.stop()

    st.success(f"Dataset charge: {df.shape[0]} lignes x {df.shape[1]} colonnes")

    diag = _run_diagnostics(df)
    st.session_state["pipeline_diagnostics"] = diag

    st.subheader("Taches a realiser")
    with st.expander("Voir le diagnostic de preparation detecte", expanded=False):
        if diag["label_too_long_count"] > 0:
            st.warning(f"Raccourcissement des libelles: {diag['label_too_long_count']} colonnes.")
        if diag["missing_pct"] > 0:
            st.warning(f"Valeurs manquantes detectees: {diag['missing_pct']:.2f}%.")
        if diag["ranked_groups_count"] > 0:
            st.warning(f"Reponses multiples ordonnees detectees: {diag['ranked_groups_count']} groupes.")
        if diag["multi_detected"]:
            st.warning(f"Reponses multiples detectees (separateur probable: {diag['multi_sep']}).")
        if diag["skip_candidates_count"] > 0:
            st.warning(
                f"Manquantes structurelles candidates: {diag['skip_candidates_count']} relations."
            )
        if (
            diag["label_too_long_count"] == 0
            and diag["missing_pct"] == 0
            and diag["ranked_groups_count"] == 0
            and not diag["multi_detected"]
            and diag["skip_candidates_count"] == 0
        ):
            st.info("Aucune tache de preparation detectee.")

    st.subheader("Choix des traitements a realiser")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state["pipeline_selection"]["preparation"] = st.checkbox(
            "Preparation",
            value=st.session_state["pipeline_selection"]["preparation"],
            help="Inclut: Preparation2, PreparationCorrelations, Outliers, CodificationOrdinales",
        )
    with col2:
        st.session_state["pipeline_selection"]["profilage"] = st.checkbox(
            "Profilage",
            value=st.session_state["pipeline_selection"]["profilage"],
            help="Inclut: Segmentation, Profils_y",
        )
    with col3:
        st.session_state["pipeline_selection"]["analyse_descriptive"] = st.checkbox(
            "Analyse descriptive",
            value=st.session_state["pipeline_selection"]["analyse_descriptive"],
        )

    st.subheader("Brief (optionnel)")
    brief_mode = st.radio(
        "Mode brief",
        options=["sb", "ab"],
        format_func=lambda x: "Sans brief" if x == "sb" else "Avec brief",
        index=0 if st.session_state.get("dataset_key_questions_mode", "sb") == "sb" else 1,
        horizontal=True,
        key="dataset_key_questions_mode",
    )

    brief = st.session_state.get("dataset_key_questions_value", "")
    if brief_mode == "ab":
        brief = st.text_area(
            "Saisir le brief d'analyse",
            value=st.session_state.get("dataset_key_questions_value", ""),
            placeholder="Ex: identifier les profils les plus lies a la satisfaction elevee et proposer 3 actions prioritaires.",
            height=120,
        )

    if st.button("Lancer", type="primary"):
        if brief_mode == "ab" and not str(brief).strip():
            st.error("Le brief est obligatoire en mode 'Avec brief'.")
            st.stop()

        if not any(st.session_state["pipeline_selection"].values()):
            st.error("Sélectionnez au moins un bloc de traitements.")
            st.stop()

        st.session_state["dataset_key_questions_value"] = str(brief).strip() if brief_mode == "ab" else ""
        st.session_state["dataset_key_questions_saved"] = True
        st.session_state["pipeline_ready_to_run"] = True
        st.session_state["pipeline_executed"] = False
        st.session_state["pipeline_status"] = "pending"
        st.session_state["pipeline_halt"] = None
        st.session_state["final_report_ready"] = False
        st.session_state["final_export_zip_bytes"] = None
        st.session_state["etape2_terminee"] = True
        st.success("Selection enregistree. Passez au Rapport Final.")
