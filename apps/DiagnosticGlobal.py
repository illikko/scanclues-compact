import pandas as pd
import streamlit as st

from core.df_registry import DFState, get_df
from .ReponsesMultiples import detect_multimodal_config
from .ReponsesMultiplesOrdonnees import detect_ranked_groups
from .VerbatimSummary import detect_long_text_columns
from utils import ensure_analysis_params


def _verbatim_only_dataset(df: pd.DataFrame | None) -> tuple[bool, list[str], list[str]]:
    """
    Détecte un dataset 100% verbatims (hors colonnes identifiants éventuelles).
    On s'appuie sur la typologie issue de Preparation1 (df_semantic_types) quand disponible,
    sinon on retombe sur une détection simple longueurs/unicité.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False, [], []

    sem = st.session_state.get("df_semantic_types")
    if isinstance(sem, pd.DataFrame) and not sem.empty and {"name", "semantic_type"} <= set(sem.columns):
        sem_ok = sem.copy()
        verb_cols = sem_ok.loc[sem_ok["semantic_type"] == "long_text", "name"].astype(str).tolist()
        id_cols = sem_ok.loc[sem_ok["semantic_type"] == "identifier", "name"].astype(str).tolist()
        other_cols = [
            str(n)
            for n in sem_ok["name"].astype(str).tolist()
            if n not in verb_cols + id_cols
        ]
        only_verbs = bool(verb_cols) and len(other_cols) == 0
        return only_verbs, verb_cols, id_cols

    # Fallback : toutes les colonnes sont longues selon heuristique
    candidates, _ = detect_long_text_columns(df, min_avg_len=50, min_unique_ratio=0.7)
    only_verbs = bool(candidates) and len(candidates) == len(df.columns)
    return only_verbs, candidates, []


def _init_session_state() -> None:
    """Initialise toutes les clés nécessaires au module."""
    defaults = {
        "etape2_terminee": False,
        "pipeline_selection": {
            "preparation": True,
            "profilage": True,
            "analyse_descriptive": True,
            "sankey_crosstabs": False,
            "distribution_figures": False,
        },
        "pipeline_ready_to_run": False,
        "pipeline_diagnostics": {},
        "dataset_key_questions_value": "",
        "dataset_key_questions_value_saved": "",
        "dataset_key_questions": "",
        "dataset_key_questions_mode": "sb",
        "dataset_key_questions_saved": False,
        "pipeline_executed": False,
        "pipeline_status": None,
        "pipeline_halt": None,
        "final_report_ready": False,
        "final_export_zip_bytes": None,
        "pipeline_config": None,
        "num_quantiles": 5,
        "distinct_threshold_continuous": 5,
        "mod_freq_min": 0.90,
        "correlation_threshold_v": 0.75,
        "outliers_percent_target": 1.0,
        "n_clusters_segmentation": 10,
        "n_clusters_target": 3,
        "kmodes_n_init": 2,
        "high_freq_threshold": 0.90,
        "run_sankey_crosstabs": False,
        "generate_distribution_figures": False,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


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


def _suggest_preparation_tasks(df: pd.DataFrame, diag: dict) -> list[str]:
    """Construit la liste des tâches de préparation à afficher."""
    tasks: list[str] = []

    # Cas 100% verbatims (hors identifiants) : on court-circuite les autres modules
    verb_only, verb_cols, id_cols = _verbatim_only_dataset(df)
    st.session_state["verbatim_only_dataset"] = verb_only
    if verb_only:
        count_txt = len(verb_cols)
        count_id = len(id_cols)
        tasks.append(f"Synthèse des verbatims (jeu 100% texte – {count_txt} colonnes verbatim, {count_id} identifiants ignorés)")
        st.session_state["verbatim_candidates"] = verb_cols
        st.session_state["verbatim_identifier_cols"] = id_cols
        return tasks
    else:
        st.session_state.pop("verbatim_only_dataset", None)

    if diag.get("label_too_long_count", 0) > 0:
        too_long = diag.get("label_too_long_count", 0)
        limit = 50
        tasks.append(f"Raccourcir les libellés trop longs ({too_long} colonne(s) > {limit} caractères)")

    missing_pct = diag.get("missing_pct", 0.0)
    tasks.append(f"Traiter les valeurs manquantes (~{missing_pct:.1f}% manquantes)")

    if diag.get("ranked_groups_count", 0) > 0:
        tasks.append("Identifier et traiter les réponses ordinales/multiples ordonnées")

    if diag.get("multi_detected", False):
        tasks.append("Traiter les réponses multiples (séparateur détecté)")

    # Détection verbatim : nombre de colonnes texte longues
    if isinstance(df, pd.DataFrame) and not df.empty:
        candidates, _ = detect_long_text_columns(df, min_avg_len=50, min_unique_ratio=0.7)
        if candidates:
            st.session_state["verbatim_candidates"] = candidates
            tasks.append(f"Synthèse des verbatims ({len(candidates)} colonne(s) texte détectée(s))")

    # Modules toujours exécutés
    tasks.append("Détection des valeurs aberrantes")
    tasks.append("Suppression des variables non informatives")

    # dédoublonnage
    return list(dict.fromkeys(tasks))


def validate_pipeline_form() -> bool:
    """Valide les choix utilisateur. Affiche les erreurs et retourne True si OK."""
    brief_mode = st.session_state.get("dataset_key_questions_mode", "sb")
    brief = st.session_state.get("dataset_key_questions_value", "")

    if brief_mode == "ab" and not str(brief).strip():
        st.error("Le brief est obligatoire en mode « Avec brief ».")
        return False

    pipeline_selection = st.session_state.get("pipeline_selection", {})
    if not any(pipeline_selection.values()):
        st.error("Sélectionnez au moins un bloc de traitements.")
        return False

    return True


def get_pipeline_config() -> dict:
    """Construit une config propre à transmettre au pipeline."""
    return {
        "pipeline_selection": st.session_state["pipeline_selection"].copy(),
        "num_quantiles": st.session_state["num_quantiles"],
        "distinct_threshold_continuous": st.session_state["distinct_threshold_continuous"],
        "mod_freq_min": st.session_state["mod_freq_min"],
        "correlation_threshold_v": st.session_state["correlation_threshold_v"],
        "outliers_percent_target": st.session_state["outliers_percent_target"],
        "n_clusters_segmentation": st.session_state["n_clusters_segmentation"],
        "n_clusters_target": st.session_state["n_clusters_target"],
        "kmodes_n_init": st.session_state["kmodes_n_init"],
        "high_freq_threshold": st.session_state["high_freq_threshold"],
        "run_sankey_crosstabs": st.session_state["run_sankey_crosstabs"],
        "generate_distribution_figures": st.session_state["generate_distribution_figures"],
        "dataset_key_questions_mode": st.session_state["dataset_key_questions_mode"],
        "dataset_key_questions_value": st.session_state["dataset_key_questions_value"],
    }


def render_pipeline_form() -> bool:
    """Affiche le formulaire."""
    st.subheader("Choix des traitements à réaliser")

    current_selection = st.session_state.get(
        "pipeline_selection",
        {
            "preparation": True,
            "profilage": True,
            "analyse_descriptive": True,
            "sankey_crosstabs": False,
            "distribution_figures": False,
        },
    )

    with st.container(border=True):
        with st.form("pipeline_form", clear_on_submit=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                preparation = st.checkbox(
                    "Préparation",
                    value=bool(current_selection.get("preparation", True)),
                    help="Inclut : toutes les étapes de préparation identifiées (valeurs manquantes et aberrantes, variables non informatives, codification des variables ordinales, libellées trop longs, colonnes à multi-modalités, manquantes structurantes, etc.)",
                )

            with col2:
                profilage = st.checkbox(
                    "Profilage",
                    value=bool(current_selection.get("profilage", True)),
                    help="Inclut : Profils des segments sur la population globale, profils des segments cibles, etc.",
                )

            with col3:
                analyse_descriptive = st.checkbox(
                    "Analyse descriptive",
                    value=bool(current_selection.get("analyse_descriptive", True)),
                    help="Inclut : relations statistiques et sémantique entre les variables, représentation graphique synthétique de ces relations (diagramme de Sankey et dendrogramme).",
                )

            sankey_crosstabs = st.checkbox(
                "Analyse détaillée des tris croisés",
                value=bool(current_selection.get("sankey_crosstabs", False)),
                help="Inclut : tris croisés détaillés pour les couples les pertinents de variables.",
            )

            distribution_figures = st.checkbox(
                "Analyse détaillée de la distribution des variables",
                value=bool(current_selection.get("distribution_figures", False)),
                help="Inclut : les histogrammes de toutes les variables.",
            )

            st.subheader("Brief (optionnel)")
            brief_mode = st.radio(
                "Mode brief",
                options=["sb", "ab"],
                format_func=lambda x: "Sans brief" if x == "sb" else "Avec brief",
                horizontal=True,
                key="dataset_key_questions_mode",
            )

            brief_value = st.text_area(
                "Saisir le brief d'analyse",
                key="dataset_key_questions_value",
                placeholder="Ex : identifier les profils les plus liés à la satisfaction élevée et proposer 3 actions prioritaires.",
                height=120,
                help='Le brief n\'est pris en compte que si "Avec brief" est sélectionné.',
            )

            with st.expander("Paramètres de l'analyse", expanded=False):
                st.caption(
                    "Valeurs utilisées par le pipeline auto. Vous pouvez les ajuster avant lancement."
                )

                colp1, colp2 = st.columns(2)

                with colp1:
                    st.number_input(
                        "Nombre de quantiles (discrétisation)",
                        min_value=2,
                        max_value=20,
                        step=1,
                        key="num_quantiles",
                    )

                    st.number_input(
                        "Seuil nb modalités pour les variables continues",
                        min_value=2,
                        max_value=50,
                        step=1,
                        key="distinct_threshold_continuous",
                    )

                    st.slider(
                        "Fréquence mini du mode (binarisation)",
                        min_value=0.50,
                        max_value=0.99,
                        step=0.01,
                        key="mod_freq_min",
                    )

                    st.slider(
                        "Seuil V de Cramer (corrélations fortes)",
                        min_value=0.50,
                        max_value=0.95,
                        step=0.01,
                        key="correlation_threshold_v",
                    )

                    st.slider(
                        "Pourcentage d'outliers (contamination)",
                        min_value=0.0,
                        max_value=20.0,
                        step=0.1,
                        key="outliers_percent_target",
                    )

                with colp2:
                    st.number_input(
                        "Clusters segmentation (Kmodes)",
                        min_value=2,
                        max_value=50,
                        step=1,
                        key="n_clusters_segmentation",
                    )

                    st.number_input(
                        "Clusters profils cible",
                        min_value=2,
                        max_value=20,
                        step=1,
                        key="n_clusters_target",
                    )

                    st.number_input(
                        "Kmodes n_init",
                        min_value=1,
                        max_value=20,
                        step=1,
                        key="kmodes_n_init",
                    )

                    st.slider(
                        "Seuil mode dominant (segmentation)",
                        min_value=0.50,
                        max_value=0.99,
                        step=0.01,
                        key="high_freq_threshold",
                    )

            submitted = st.form_submit_button("Lancer", type="primary")

    if submitted:
        st.session_state["pipeline_selection"] = {
            "preparation": bool(preparation),
            "profilage": bool(profilage),
            "analyse_descriptive": bool(analyse_descriptive),
            "sankey_crosstabs": bool(sankey_crosstabs),
            "distribution_figures": bool(distribution_figures),
        }
        st.session_state["run_sankey_crosstabs"] = bool(sankey_crosstabs)
        st.session_state["generate_distribution_figures"] = bool(distribution_figures)
        # Sauvegarde explicite du brief pour les écrans suivants (clé distincte pour éviter les conflits widgets)
        st.session_state["dataset_key_questions_value_saved"] = st.session_state.get("dataset_key_questions_value", "")
        # Copie non liée à un widget pour les modules en aval (RapportFinal / QA)
        st.session_state["dataset_key_questions"] = st.session_state.get("dataset_key_questions_value", "")

    return submitted

def run():
    st.subheader("Définition des objectifs")

    ensure_analysis_params(st.session_state)
    _init_session_state()

    df = _get_input_df()
    if not isinstance(df, pd.DataFrame):
        st.warning("Aucun dataset disponible. Lancez d'abord l'étape Upload.")
        st.stop()

    st.success(f"Dataset chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    current_upload_nonce = int(st.session_state.get("__UPLOAD_NONCE__", 0))
    cached_diag = st.session_state.get("pipeline_diagnostics")
    cached_tasks = st.session_state.get("pipeline_prep_tasks", [])
    revisit_without_changes = (
        isinstance(cached_diag, dict)
        and st.session_state.get("__DG_LAST_NONCE__", current_upload_nonce) == current_upload_nonce
        and not st.session_state.pop("__DG_FORCE_RERUN__", False)
    )

    if revisit_without_changes and cached_tasks:
        diag = cached_diag
        prep_tasks = cached_tasks
    else:
        diag = _run_diagnostics(df)
        st.session_state["pipeline_diagnostics"] = diag
        prep_tasks = _suggest_preparation_tasks(df, diag)
        st.session_state["pipeline_prep_tasks"] = prep_tasks
        st.session_state["__DG_LAST_NONCE__"] = current_upload_nonce

    st.markdown("##### Traitements de préparation à réaliser")
    st.markdown("\n".join([f"- {t}" for t in prep_tasks]))
    if st.session_state.get("verbatim_only_dataset"):
        st.info("Dataset 100% verbatim détecté : seule la synthèse des verbatims sera exécutée.")
    if diag.get("label_too_long_cols"):
        sample_cols = ", ".join(diag["label_too_long_cols"][:5])
        st.caption(
            f"Exemples de libellés longs: {sample_cols}"
            + (" ..." if len(diag["label_too_long_cols"]) > 5 else "")
        )

    submitted = render_pipeline_form()

    if submitted:
        if not validate_pipeline_form():
            st.stop()

        config = get_pipeline_config()
        st.session_state["pipeline_config"] = config

        st.session_state["dataset_key_questions_saved"] = True
        st.session_state["pipeline_ready_to_run"] = True
        st.session_state["pipeline_executed"] = False
        st.session_state["pipeline_status"] = "pending"
        st.session_state["pipeline_halt"] = None
        st.session_state["final_report_ready"] = False
        st.session_state["final_export_zip_bytes"] = None
        st.session_state["etape2_terminee"] = True

        st.success("Sélection enregistrée. Passez au Rapport Final.")

