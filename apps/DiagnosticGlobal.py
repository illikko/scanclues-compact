import pandas as pd
import streamlit as st

from core.analysis_context_resolver import (
    build_analysis_options,
    default_pipeline_selection,
    has_executable_selection,
    resolve_analysis_context,
)
from core.df_registry import DFState, get_df
from core.preparation_details import refresh_preparation_details_payload
from core.preparation_diagnostics import get_preparation_diagnostics, set_preparation_diagnostic
from .DiagnosticMissing import diagnose_missing_values
from .ManquantesStructurelles import diagnose_structural_missing_candidates
from .ReponsesMultiples import detect_multimodal_config
from .ReponsesMultiplesOrdonnees import detect_ranked_groups
from .VerbatimSummary import detect_long_text_columns, diagnose_verbatim_columns
from utils import ensure_analysis_params
from core.progress_state import reset_progress

NAV_CONTEXT_KEY = "__NAV_CONTEXT__"


def _diagnostic_signature(df: pd.DataFrame) -> dict:
    return {
        "upload_nonce": int(st.session_state.get("__UPLOAD_NONCE__", 0)),
        "shape": tuple(df.shape),
        "columns": tuple(str(c) for c in df.columns.tolist()),
    }


def _find_duplicate_columns(df: pd.DataFrame | None) -> list[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    cols = pd.Index(df.columns)
    return cols[cols.duplicated()].astype(str).tolist()


def _safe_detect_ranked_groups(df: pd.DataFrame) -> tuple[list, str | None]:
    try:
        return detect_ranked_groups(df, min_ranks=2), None
    except Exception as exc:
        return [], repr(exc)


def _safe_detect_multimodal_config(df: pd.DataFrame):
    try:
        return detect_multimodal_config(df), None
    except Exception as exc:
        return None, repr(exc)


def _safe_detect_long_text_columns(
    df: pd.DataFrame,
    *,
    min_avg_len: int,
    min_unique_ratio: float,
) -> tuple[list[str], dict, str | None]:
    try:
        candidates, details = detect_long_text_columns(
            df,
            min_avg_len=min_avg_len,
            min_unique_ratio=min_unique_ratio,
        )
        return candidates, details, None
    except Exception as exc:
        return [], {}, repr(exc)


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
    candidates, _, error = _safe_detect_long_text_columns(
        df,
        min_avg_len=50,
        min_unique_ratio=0.7,
    )
    if error:
        st.session_state["pipeline_diagnostics_errors"] = {
            **st.session_state.get("pipeline_diagnostics_errors", {}),
            "verbatim_only_detection": error,
        }
        return False, [], []
    only_verbs = bool(candidates) and len(candidates) == len(df.columns)
    return only_verbs, candidates, []


def _init_session_state() -> None:
    """Initialise toutes les clés nécessaires au module."""
    defaults = {
        "etape2_terminee": False,
        "pipeline_selection": default_pipeline_selection(),
        "analysis_options": build_analysis_options(default_pipeline_selection()),
        "details_preparation_selected": False,
        "pipeline_ready_to_run": False,
        "pipeline_diagnostics": {},
        "pipeline_diagnostics_ready": False,
        "pipeline_diagnostics_signature": None,
        "pipeline_prep_tasks": [],
        "pipeline_diagnostics_errors": {},
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
    duplicate_columns = _find_duplicate_columns(df)
    errors: dict[str, str] = {}

    ranked_groups, ranked_error = _safe_detect_ranked_groups(df)
    if ranked_error:
        errors["ranked_groups"] = ranked_error

    multi_det, multi_error = _safe_detect_multimodal_config(df)
    if multi_error:
        errors["multimodal"] = multi_error

    missing_cols = int((df.isna().sum() > 0).sum())
    skip_candidates_count = max(missing_cols, len(ranked_groups))

    st.session_state["pipeline_diagnostics_errors"] = errors

    return {
        "label_too_long_count": len(label_too_long),
        "label_too_long_cols": label_too_long,
        "missing_pct": missing_pct,
        "ranked_groups_count": len(ranked_groups),
        "multi_detected": bool(getattr(multi_det, "ok", False)),
        "multi_sep": getattr(multi_det, "sep", None),
        "skip_candidates_count": int(skip_candidates_count),
        "duplicate_columns": duplicate_columns,
        "diagnostic_errors": errors,
    }


def collect_preparation_diagnostics(df: pd.DataFrame, diag: dict | None = None) -> dict[str, dict]:
    diagnostics = dict(get_preparation_diagnostics())
    diag = diag or {}

    if "missing_values" not in diagnostics:
        missing_diag, _ = diagnose_missing_values(df)
        set_preparation_diagnostic(missing_diag)
        diagnostics["missing_values"] = missing_diag

    if "long_labels" not in diagnostics:
        long_labels_diag = {
            "id": "long_labels",
            "label": "Raccourcir les libellés trop longs",
            "needed": bool(diag.get("label_too_long_count", 0) > 0),
            "reason": (
                f"{diag.get('label_too_long_count', 0)} colonne(s) dépassent 50 caractères"
                if diag.get("label_too_long_count", 0) > 0
                else "Aucun libellé trop long détecté"
            ),
            "details": {
                "max_chars": 50,
                "columns": diag.get("label_too_long_cols", []),
            },
            "compute_module": "LabelShortening",
            "render_module": "LabelShortening",
            "available": True,
        }
        set_preparation_diagnostic(long_labels_diag)
        diagnostics["long_labels"] = long_labels_diag

    if "verbatim_summary" not in diagnostics and "semantic_verbatim" not in diagnostics:
        verbatim_diag, _ = diagnose_verbatim_columns(df, min_avg_len=50, min_unique_ratio=0.7)
        diagnostics["verbatim_summary"] = verbatim_diag

    if "ranked_multiple_responses" not in diagnostics:
        ranked_groups_count = int(diag.get("ranked_groups_count", 0) or 0)
        ranked_diag = {
            "id": "ranked_multiple_responses",
            "label": "Traiter les réponses multiples ordonnées",
            "needed": bool(ranked_groups_count > 0),
            "reason": (
                f"{ranked_groups_count} groupe(s) détecté(s)"
                if ranked_groups_count > 0
                else "Aucun groupe de réponses ordonnées détecté"
            ),
            "details": {
                "groups": [],
                "min_ranks": 2,
            },
            "compute_module": "ReponsesMultiplesOrdonnees",
            "render_module": "ReponsesMultiplesOrdonnees",
            "available": True,
        }
        set_preparation_diagnostic(ranked_diag)
        diagnostics["ranked_multiple_responses"] = ranked_diag

    if "multiple_responses" not in diagnostics:
        multi_diag = {
            "id": "multiple_responses",
            "label": "Traiter les réponses multiples",
            "needed": bool(diag.get("multi_detected", False)),
            "reason": (
                f"séparateur détecté '{diag.get('multi_sep')}'"
                if diag.get("multi_detected", False)
                else "Aucune configuration de réponses multiples détectée"
            ),
            "details": {
                "separator": diag.get("multi_sep"),
                "columns": [],
            },
            "compute_module": "ReponsesMultiples",
            "render_module": "ReponsesMultiples",
            "available": True,
        }
        set_preparation_diagnostic(multi_diag)
        diagnostics["multiple_responses"] = multi_diag

    if "structural_missing" not in diagnostics:
        structural_diag = diagnose_structural_missing_candidates(df)
        diagnostics["structural_missing"] = structural_diag

    return diagnostics


def normalize_preparation_tasks(df: pd.DataFrame, diag: dict, preparation_diagnostics: dict[str, dict]) -> list[str]:
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

    long_labels_diag = preparation_diagnostics.get("long_labels")
    if isinstance(long_labels_diag, dict) and long_labels_diag.get("needed"):
        tasks.append(f"{long_labels_diag.get('label')} ({long_labels_diag.get('reason')})")
    elif diag.get("label_too_long_count", 0) > 0:
        too_long = diag.get("label_too_long_count", 0)
        limit = 50
        tasks.append(f"Raccourcir les libellés trop longs ({too_long} colonne(s) > {limit} caractères)")

    missing_diag = preparation_diagnostics.get("missing_values")
    if isinstance(missing_diag, dict) and missing_diag.get("needed"):
        tasks.append(f"{missing_diag.get('label')} ({missing_diag.get('reason')})")
    else:
        missing_pct = diag.get("missing_pct", 0.0)
        tasks.append(f"Traiter les valeurs manquantes (~{missing_pct:.1f}% manquantes)")

    semantic_order = [
        "semantic_dates",
        "semantic_geo",
        "semantic_identifiers",
        "semantic_verbatim",
    ]
    for diag_id in semantic_order:
        semantic_diag = preparation_diagnostics.get(diag_id)
        if not isinstance(semantic_diag, dict) or not semantic_diag.get("needed"):
            continue
        suffix = ""
        if semantic_diag.get("available") is False:
            suffix = " (module à venir)"
        tasks.append(f"{semantic_diag.get('label')} ({semantic_diag.get('reason')}){suffix}")

    ranked_diag = preparation_diagnostics.get("ranked_multiple_responses")
    if isinstance(ranked_diag, dict) and ranked_diag.get("needed"):
        tasks.append(f"{ranked_diag.get('label')} ({ranked_diag.get('reason')})")
    elif diag.get("ranked_groups_count", 0) > 0:
        tasks.append("Identifier et traiter les réponses ordinales/multiples ordonnées")

    multi_diag = preparation_diagnostics.get("multiple_responses")
    if isinstance(multi_diag, dict) and multi_diag.get("needed"):
        tasks.append(f"{multi_diag.get('label')} ({multi_diag.get('reason')})")
    elif diag.get("multi_detected", False):
        tasks.append("Traiter les réponses multiples (séparateur détecté)")

    structural_diag = preparation_diagnostics.get("structural_missing")
    if isinstance(structural_diag, dict) and structural_diag.get("needed"):
        tasks.append(f"{structural_diag.get('label')} ({structural_diag.get('reason')})")

    verbatim_semantic_diag = preparation_diagnostics.get("semantic_verbatim")
    verbatim_diag = preparation_diagnostics.get("verbatim_summary")
    if (
        isinstance(verbatim_diag, dict)
        and verbatim_diag.get("needed")
        and not (isinstance(verbatim_semantic_diag, dict) and verbatim_semantic_diag.get("needed"))
    ):
        verbatim_cols = verbatim_diag.get("details", {}).get("columns", [])
        if verbatim_cols:
            st.session_state["verbatim_candidates"] = verbatim_cols
        tasks.append(f"{verbatim_diag.get('label')} ({verbatim_diag.get('reason')})")
    elif (
        not (isinstance(verbatim_semantic_diag, dict) and verbatim_semantic_diag.get("needed"))
        and not (isinstance(verbatim_diag, dict) and verbatim_diag.get("needed"))
        and isinstance(df, pd.DataFrame)
        and not df.empty
    ):
        candidates, _, error = _safe_detect_long_text_columns(
            df,
            min_avg_len=50,
            min_unique_ratio=0.7,
        )
        if error:
            st.session_state["pipeline_diagnostics_errors"] = {
                **st.session_state.get("pipeline_diagnostics_errors", {}),
                "verbatim_candidates": error,
            }
            candidates = []
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
    if not has_executable_selection(pipeline_selection):
        st.error("Sélectionnez au moins un bloc de traitements.")
        return False

    return True


def get_pipeline_config() -> dict:
    """Construit une config propre à transmettre au pipeline."""
    return {
        "pipeline_selection": st.session_state["pipeline_selection"].copy(),
        "analysis_options": st.session_state.get("analysis_options", build_analysis_options(st.session_state["pipeline_selection"])),
        "details_preparation_selected": bool(st.session_state.get("details_preparation_selected", False)),
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
    st.markdown("##### Choix des insights à produire")

    current_selection = st.session_state.get(
        "pipeline_selection",
        default_pipeline_selection(),
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
                    help="Inclut : les relations statistiques et sémantiques entre les variables, des représentations graphiques synthétiques de ces relations (diagramme de Sankey et dendrogramme).",
                )

            details_preparation = st.checkbox(
                "Détails de la préparation",
                value=bool(current_selection.get("details_preparation", False)),
                help="Affiche dans le rapport final les artefacts détaillés des modules de préparation : libellés avant/après, diagnostic des manquants, outliers supprimés, etc.",
            )

            sankey_crosstabs = st.checkbox(
                "Analyse détaillée des tris croisés",
                value=bool(current_selection.get("sankey_crosstabs", False)),
                help="Inclut : tris croisés détaillés pour les couples de variables les plus pertinents.",
            )

            distribution_figures = st.checkbox(
                "Analyse détaillée de la distribution des variables",
                value=bool(current_selection.get("distribution_figures", False)),
                help="Inclut : les histogrammes de toutes les variables.",
            )

            st.markdown("##### Brief (optionnel)")
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
                        help="Nombre de quantiles pour la discrétisation des variables continues (5 par défaut)."
                    )

                    st.number_input(
                        "Seuil nb modalités pour les variables continues",
                        min_value=2,
                        max_value=50,
                        step=1,
                        key="distinct_threshold_continuous",
                        help="Nombre maximum de modalités distinctes pour qu'une variable soit considérée comme continue (5 par défaut)."
                    )

                    st.slider(
                        "Fréquence mini du mode (binarisation)",
                        min_value=0.50,
                        max_value=0.99,
                        step=0.01,
                        key="mod_freq_min",
                        help="Seuil de fréquence du mode à partir duquel une variable est considérée comme binaire (90% par défaut)."
                    )

                    st.slider(
                        "Taux de corrélation au delà duquel 2 variables sont redondantes",
                        min_value=0.50,
                        max_value=0.95,
                        step=0.01,
                        key="correlation_threshold_v",
                        help="Seuil de redondance entre 2 variables mesuré par le taux d'information mutuelle (75% par défaut)."
                    )

                    st.slider(
                        "Pourcentage d'outliers (contamination)",
                        min_value=0.0,
                        max_value=20.0,
                        step=0.1,
                        key="outliers_percent_target",
                        help="Pourcentage de valeurs aberrantes à détecter par la méthode d'isolation forest (1% par défaut)."
                    )

                with colp2:
                    st.number_input(
                        "Clusters segmentation (Kmodes)",
                        min_value=2,
                        max_value=50,
                        step=1,
                        key="n_clusters_segmentation",
                        help="Nombre de clusters pour la segmentation globale par Kmodes (10 par défaut)."
                    )

                    st.number_input(
                        "Clusters profils cible",
                        min_value=2,
                        max_value=20,
                        step=1,
                        key="n_clusters_target",
                        help="Nombre de clusters pour l'identification des profils cibles par Kmodes (3 par défaut)."
                    )

                    st.number_input(
                        "Kmodes n_init",
                        min_value=1,
                        max_value=20,
                        step=1,
                        key="kmodes_n_init",
                        help="Nombre d'initialisations pour le clustering par Kmodes (2 par défaut)."
                    )

                    st.slider(
                        "Seuil mode dominant (segmentation)",
                        min_value=0.50,
                        max_value=0.99,
                        step=0.01,
                        key="high_freq_threshold",
                        help="Seuil de fréquence du mode dominant pour l'affichage des modalités dans les profils de segmentation (90% par défaut)."
                    )

            submitted = st.form_submit_button("Lancer", type="primary")

    if submitted:
        st.session_state["pipeline_selection"] = {
            "preparation": bool(preparation),
            "details_preparation": bool(details_preparation),
            "profilage": bool(profilage),
            "analyse_descriptive": bool(analyse_descriptive),
            "sankey_crosstabs": bool(sankey_crosstabs),
            "distribution_figures": bool(distribution_figures),
        }
        st.session_state["analysis_options"] = build_analysis_options(st.session_state["pipeline_selection"])
        st.session_state["details_preparation_selected"] = bool(details_preparation)
        st.session_state["run_sankey_crosstabs"] = bool(sankey_crosstabs)
        st.session_state["generate_distribution_figures"] = bool(distribution_figures)
        # Sauvegarde explicite du brief pour les écrans suivants (clé distincte pour éviter les conflits widgets)
        st.session_state["dataset_key_questions_value_saved"] = st.session_state.get("dataset_key_questions_value", "")
        # Copie non liée à un widget pour les modules en aval (RapportFinal / QA)
        st.session_state["dataset_key_questions"] = st.session_state.get("dataset_key_questions_value", "")

    return submitted

def run():
    st.subheader("Diagnostic et définition des objectifs")

    ensure_analysis_params(st.session_state)
    _init_session_state()
    st.session_state["analysis_options"] = build_analysis_options(st.session_state.get("pipeline_selection", default_pipeline_selection()))
    st.session_state["details_preparation_selected"] = bool(
        st.session_state["analysis_options"]["details_preparation_selected"]
    )

    df = _get_input_df()
    if not isinstance(df, pd.DataFrame):
        st.warning("Aucun dataset disponible. Lancez d'abord l'étape Upload.")
        st.stop()

    st.success(f"Dataset chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    current_upload_nonce = int(st.session_state.get("__UPLOAD_NONCE__", 0))
    cached_diag = st.session_state.get("pipeline_diagnostics")
    cached_tasks = st.session_state.get("pipeline_prep_tasks", [])
    current_signature = _diagnostic_signature(df)
    cached_signature = st.session_state.get("pipeline_diagnostics_signature")
    revisit_without_changes = (
        isinstance(cached_diag, dict)
        and bool(st.session_state.get("pipeline_diagnostics_ready", False))
        and isinstance(cached_tasks, list)
        and cached_signature == current_signature
        and not st.session_state.pop("__DG_FORCE_RERUN__", False)
    )

    if revisit_without_changes and cached_tasks:
        diag = cached_diag
        prep_tasks = cached_tasks
    else:
        diag = _run_diagnostics(df)
        st.session_state["pipeline_diagnostics"] = diag
        preparation_diagnostics = collect_preparation_diagnostics(df, diag)
        prep_tasks = normalize_preparation_tasks(df, diag, preparation_diagnostics)
        st.session_state["pipeline_prep_tasks"] = prep_tasks
        st.session_state["pipeline_diagnostics_ready"] = True
        st.session_state["pipeline_diagnostics_signature"] = current_signature
        st.session_state["__DG_LAST_NONCE__"] = current_upload_nonce
        refresh_preparation_details_payload()

    st.markdown("##### Traitements de préparation à réaliser")
    st.markdown("\n".join([f"- {t}" for t in prep_tasks]))
    if st.session_state.get("verbatim_only_dataset"):
        st.info("Dataset 100% verbatim détecté : seule la synthèse des verbatims sera exécutée.")
    duplicate_columns = diag.get("duplicate_columns", [])
    if duplicate_columns:
        sample_dups = ", ".join(duplicate_columns[:5])
        st.warning(
            "Colonnes dupliquées détectées dans le dataset. "
            "Certaines heuristiques automatiques ont été neutralisées pour éviter un plantage. "
            f"Exemples : {sample_dups}"
            + (" ..." if len(duplicate_columns) > 5 else "")
        )
    diag_errors = diag.get("diagnostic_errors", {})
    if diag_errors:
        st.warning(
            "Une ou plusieurs détections automatiques ont été ignorées car elles ont échoué sur ce dataset. "
            "Le pipeline reste lançable avec un diagnostic partiel."
        )
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
        st.session_state["analysis_context"] = resolve_analysis_context(st.session_state)

        st.session_state["dataset_key_questions_saved"] = True
        st.session_state["pipeline_ready_to_run"] = True
        st.session_state["pipeline_executed"] = False
        st.session_state["pipeline_status"] = "pending"
        st.session_state["pipeline_halt"] = None
        st.session_state["final_report_ready"] = False
        st.session_state["final_export_zip_bytes"] = None
        st.session_state["etape2_terminee"] = True
        st.session_state[NAV_CONTEXT_KEY] = "action"
        reset_progress("post_diagnostic", "Lancement de l'analyse finale")
        st.session_state["__NAV_SELECTED__"] = "3"
        try:
            st.query_params["step"] = "3"
        except Exception:
            st.experimental_set_query_params(step="3")
        st.rerun()

        st.success("Sélection enregistrée. Passez au Rapport Final.")

