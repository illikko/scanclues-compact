import streamlit as st
import pandas as pd
import matplotlib as mpl
from openai import OpenAI
import json
import os
import time
from core.df_registry import init_df_registry
from core.reset_state import reset_app_state
from .CadrageAnalyse import _call_llm
from .PipelineRunner import get_selected_module_plan, run_selected
from ._report import (
    add_from_state, add_text, add_table, add_figure_auto, build_html_report,
    add_text_html, add_table_html, reset_report
)
from core.progress_state import set_progress
from core.report_export import build_export_zip, build_final_report_html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components

MODE_KEY = "__NAV_MODE__"
NAV_CONTEXT_KEY = "__NAV_CONTEXT__"


# ========= OpenAI =========
api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Fonctions utilitaires ---
def is_non_empty(x):
    """Vérifie si un objet est non vide, compatible avec DataFrame / str / list."""
    if x is None:
        return False
    if isinstance(x, str):
        return x.strip() != ""
    if isinstance(x, (list, tuple, set, dict)):
        return len(x) > 0
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return not x.empty
    return True

def to_block(x):
    """Convertit un objet en texte lisible pour le prompt."""
    if isinstance(x, pd.DataFrame):
        return x.head(20).to_string(index=False)
    return str(x)


SECTION_INFO_TEXTS = {
    "verbatims": (
        "Cette section résume les colonnes de texte libre détectées dans le jeu de données. "
        "Elle met en avant les thèmes ou messages dominants extraits des verbatims."
    ),
    "insights": (
        "Cette section présente les enseignements les plus importants extraits automatiquement du jeu de données. "
        "Elle met en avant les constats prioritaires, les signaux saillants et les points d'attention à retenir rapidement."
    ),
    "profilage": (
        "Cette section décrit les profils ou segments les plus marquants observés dans les données. "
        "Elle aide à comprendre quelles caractéristiques distinguent les groupes et quels facteurs sont associés à la cible analysée."
    ),
    "analyse_descriptive": (
        "Cette section résume la structure du jeu de données et les principales relations entre variables. "
        "Elle permet de visualiser les distributions, regroupements et dépendances avant une interprétation plus détaillée."
    ),
    "contexte": (
        "Cette section précise le sujet du jeu de données et le rôle attendu de ses variables. "
        "Elle fournit aussi des recommandations utiles pour relire les résultats dans leur contexte métier."
    ),
    "tris_croises": (
        "Cette section détaille des croisements entre paires de variables jugées utiles pour l'analyse. "
        "Elle permet de comparer les répartitions observées et de lire leur interprétation associée."
    ),
    "analyse_technique": (
        "Cette section décrit les traitements de préparation appliqués au jeu de données. "
        "Elle synthétise notamment le nettoyage, les valeurs manquantes, les aberrants et les transformations techniques réalisées."
    ),
    "distributions": (
        "Cette section présente les distributions détaillées des variables sélectionnées sous forme de graphiques. "
        "Elle aide à repérer les niveaux, dispersions, asymétries et modalités dominantes."
    ),
    "etapes_preparation": (
        "Cette section liste les principales étapes de préparation appliquées au jeu de données. "
        "Elle permet de retracer de manière synthétique les transformations successives avant l'analyse finale."
    ),
}


SECTION_INFO_LABELS = {
    "verbatims": "Synthèse des verbatims",
    "insights": "Principaux insights",
    "profilage": "Profilage",
    "analyse_descriptive": "Analyse descriptive",
    "contexte": "Contexte du jeu de données",
    "tris_croises": "Tris croisés détaillés",
    "analyse_technique": "Analyse technique du jeu de données",
    "distributions": "Distributions détaillées",
    "etapes_preparation": "Etapes des préparations",
}


def _expander_label(section_key: str) -> str:
    base_label = SECTION_INFO_LABELS[section_key]
    tooltip = str(SECTION_INFO_TEXTS.get(section_key, "")).replace('"', "&quot;")
    return f'{base_label} [ℹ️](# "{tooltip}")' if tooltip else base_label

def _preview_df_from_payload(columns, rows):
    if not isinstance(columns, list) or not isinstance(rows, list) or not rows:
        return None
    try:
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return None


def _render_preparation2_details(details: dict) -> None:
    if not isinstance(details, dict) or not details:
        return

    missing_values = details.get("missing_values", {}) or {}
    if isinstance(missing_values, dict) and missing_values:
        st.markdown("**Traitement des valeurs manquantes**")

        dropped_columns = missing_values.get("dropped_columns") or []
        if dropped_columns:
            st.write("Colonnes supprimées car trop incomplètes : " + ", ".join(map(str, dropped_columns)))

        dropped_rows = missing_values.get("dropped_rows")
        if isinstance(dropped_rows, int) and dropped_rows > 0:
            st.write(f"Lignes supprimées car trop incomplètes : {dropped_rows}")

        remaining_missing_columns = missing_values.get("remaining_missing_columns") or []
        if remaining_missing_columns:
            st.write("Colonnes restant partiellement incomplètes : " + ", ".join(map(str, remaining_missing_columns)))

        simple_imputation_columns = missing_values.get("simple_imputation_columns") or []
        if simple_imputation_columns:
            st.write("Imputation simple appliquée sur : " + ", ".join(map(str, simple_imputation_columns)))

        hotdeck_stats = missing_values.get("hotdeck_stats") or {}
        if isinstance(hotdeck_stats, dict) and hotdeck_stats:
            st.markdown("Hot-deck simple")
            hotdeck_df = pd.DataFrame(
                [{"Colonne": str(col), "Valeurs imputées": int(count)} for col, count in hotdeck_stats.items()]
            )
            if not hotdeck_df.empty:
                st.dataframe(hotdeck_df, use_container_width=True, hide_index=True)

    rare_modalities = details.get("rare_modalities", {}) or {}
    if isinstance(rare_modalities, dict):
        grouped_columns = rare_modalities.get("grouped_columns") or []
        if grouped_columns:
            st.markdown("**Regroupement des modalités rares**")
            st.write(", ".join(map(str, grouped_columns)))

    second_pass = details.get("second_pass", {}) or {}
    if isinstance(second_pass, dict):
        dropped_columns = second_pass.get("dropped_columns") or []
        if dropped_columns:
            st.markdown("**Vérifications finales**")
            st.write(
                "Colonnes non informatives supprimées après nettoyage : "
                + ", ".join(map(str, dropped_columns))
            )


def _render_preparation_details(payload: dict) -> None:
    if not isinstance(payload, dict) or not payload:
        st.info("Aucun détail de préparation disponible.")
        return


    label_shortening = payload.get("label_shortening", {})
    mapping_preview = _preview_df_from_payload(
        label_shortening.get("mapping_columns"),
        label_shortening.get("mapping_preview"),
    )
    if isinstance(mapping_preview, pd.DataFrame) and not mapping_preview.empty:
        st.markdown("**Raccourcissement des noms des colonnes: libellés avant / après**")
        st.dataframe(mapping_preview, use_container_width=True)

    semantic_rows = label_shortening.get("semantic_types_preview") or []
    semantic_columns = label_shortening.get("semantic_types_columns")
    if not semantic_columns and semantic_rows:
        semantic_columns = [str(k) for k in semantic_rows[0].keys()]
    semantic_preview = _preview_df_from_payload(
        semantic_columns,
        semantic_rows,
    )
    if isinstance(semantic_preview, pd.DataFrame) and not semantic_preview.empty:
        st.markdown("**Types sémantiques détectés**")
        st.dataframe(semantic_preview, use_container_width=True)

    missing_values = payload.get("missing_values", {})
    if missing_values.get("diagnostic"):
        st.markdown("**Diagnostic des valeurs manquantes**")
        st.write(missing_values.get("diagnostic"))
    missing_preview = _preview_df_from_payload(
        missing_values.get("table_columns"),
        missing_values.get("table_preview"),
    )
    if isinstance(missing_preview, pd.DataFrame) and not missing_preview.empty:
        st.dataframe(missing_preview, use_container_width=True)
    if missing_values.get("little_test_result"):
        st.caption(str(missing_values.get("little_test_result")))

    structural_missing = payload.get("structural_missing", {})
    if structural_missing.get("diagnostic"):
        st.markdown("**Manquantes structurelles**")
        st.write(structural_missing.get("diagnostic"))
    structural_preview = _preview_df_from_payload(
        structural_missing.get("candidates_columns"),
        structural_missing.get("candidates_preview"),
    )
    if isinstance(structural_preview, pd.DataFrame) and not structural_preview.empty:
        st.caption("Colonnes repérées comme potentiellement concernées par un mécanisme de non-réponse structurelle.")
        st.dataframe(structural_preview, use_container_width=True)

    outliers = payload.get("outliers", {})
    outliers_preview = _preview_df_from_payload(
        outliers.get("table_columns"),
        outliers.get("table_preview"),
    )
    if outliers.get("removed") is not None:
        st.markdown("**Outliers supprimés**")
        removed_count = len(outliers.get("indices") or []) if outliers.get("removed") else 0
        st.caption(f"Nombre de lignes supprimées : {removed_count}")
    if isinstance(outliers_preview, pd.DataFrame) and not outliers_preview.empty:
        st.dataframe(outliers_preview, use_container_width=True)

    prep2_details = payload.get("preparation2", {}).get("details")
    if prep2_details:
        st.markdown("**Détails complémentaires de préparation**")
        _render_preparation2_details(prep2_details)

def _show_sankey_screen(fig_obj, sankey_base64: str | None = None) -> bool:
    if fig_obj is None:
        if isinstance(sankey_base64, str) and sankey_base64.strip():
            try:
                import base64 as _b64
                st.image(_b64.b64decode(sankey_base64), use_container_width=True)
                return True
            except Exception:
                return False
        return False


    # 1) Essai direct
    try:
        st.plotly_chart(fig_obj, use_container_width=True)
        return True
    except Exception:
        pass

    # 2) Coercion vers go.Figure
    coerced = None
    try:
        if isinstance(fig_obj, go.Figure) or hasattr(fig_obj, "to_plotly_json"):
            coerced = go.Figure(fig_obj)
        elif isinstance(fig_obj, dict) and fig_obj:
            coerced = go.Figure(fig_obj)
        elif isinstance(fig_obj, str) and fig_obj.strip():
            s = fig_obj.strip()
            if s.startswith("{") and s.endswith("}"):
                coerced = go.Figure(json.loads(s))
        if coerced is not None:
            st.plotly_chart(coerced, use_container_width=True)
            return True
    except Exception:
        pass

    # 3) Fallback HTML Plotly
    try:
        html = pio.to_html(coerced if coerced is not None else fig_obj, full_html=False, include_plotlyjs="cdn")
        components.html(html, height=640, scrolling=True)
        return True
    except Exception:
        pass

    # Fallback matplotlib uniquement si compatible.
    if hasattr(fig_obj, "savefig"):
        st.pyplot(fig_obj)
        return True

    # 4) Fallback image base64
    if isinstance(sankey_base64, str) and sankey_base64.strip():
        try:
            import base64 as _b64
            st.image(_b64.b64decode(sankey_base64), use_container_width=True)
            return True
        except Exception:
            pass

    st.warning(f"Diagramme Sankey non affichable dans l'écran (format inattendu: {type(fig_obj).__name__}).")
    return False


def _navigate_to_qa() -> None:
    st.session_state[NAV_CONTEXT_KEY] = "view"
    st.session_state["__NAV_SELECTED__"] = "4"
    try:
        st.query_params["step"] = "4"
    except Exception:
        st.experimental_set_query_params(step="4")


def _change_objectives() -> None:
    st.session_state[NAV_CONTEXT_KEY] = "view"
    st.session_state["etape2_terminee"] = False
    st.session_state["etape40_terminee"] = False
    st.session_state["etape41_terminee"] = False
    st.session_state["__NAV_SELECTED__"] = "2"
    try:
        st.query_params["step"] = "2"
    except Exception:
        st.experimental_set_query_params(step="2")

# gérer la taille des graphiques

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# ================================================================
# Application principale
# ================================================================
def run():
    st.subheader("Rapport")
    # Réinitialiser les blocs/flags de section détaillée à chaque exécution
    st.session_state["_rf_blocks"] = []
    st.session_state["__DETAIL_SECTION_ADDED__"] = False
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    nav_context = str(st.session_state.get(NAV_CONTEXT_KEY, "view"))
    passive_nav = nav_context == "view"
    
    # Initialisation des états
    if "final_report_ready" not in st.session_state:
        st.session_state["final_report_ready"] = False
    st.session_state.setdefault("final_export_zip_bytes", None)
    st.session_state.setdefault("pipeline_executed", False)
    st.session_state.setdefault(
        "pipeline_selection",
        {"preparation": True, "profilage": False, "analyse_descriptive": False},
    )
    # Forcer la réparation du rapport si dataset 100% verbatim
    if st.session_state.get("verbatim_only_dataset"):
        st.session_state["final_report_ready"] = False

    defaults = {
        'dataset_object': None,                      
        'dataset_context': None,               
        'dataset_key_questions': None,              
        'dataset_key_questions_value': "",         
        'dataset_key_questions_value_saved': "",
        'dataset_key_questions_mode': "sb",
        'global_synthesis': None, 
        'global_synthesis_generated': False,
        'data_preparation_synthesis_generated': False,        
        'dataset_recommendations': None,               
        'profil_dominant': None,           
        'interpretationACM': None,          
        'dendrogram_interpretation': None,           
        'segmentation_detailed_profiles': None,
        'process': None,                    
        'data_preparation_synthesis': None,                
        'variables_description': None,
        'latent_summary_text': None,
        'latent_table': None,
        'crosstabs_interpretation': None,
        'sankey_pair_results': None,
        'sankey_diagram': None,
        'sankey_diagram_base64': None,
        'ordinal_codification_mapping': None,
        'shortened_labels_mapping': None,
        'syntheses_verbatim': None,
        'brief_context_global_synthesis': None,
        'etape40_terminee': False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


    # chargement des informations nécessaires
    dataset_object = st.session_state.get('dataset_object')
    dataset_context = st.session_state.get('dataset_context')
    dataset_recommendations = st.session_state.get('dataset_recommendations')
    dataset_key_questions = (
        st.session_state.get('dataset_key_questions_value')
        or st.session_state.get("dataset_key_questions_value_saved")
        or st.session_state.get("dataset_key_questions")
    )
    dataset_key_questions_mode = st.session_state.get("dataset_key_questions_mode", "sb")
    profils_y_text = st.session_state.get('profils_y_text')
    profil_dominant_analysis = st.session_state.get('profil_dominant_analysis')
    dominant_discretes = st.session_state.get('dominant_discretes')
    dominant_continues = st.session_state.get('dominant_continues')    
    interpretationACM = st.session_state.get('interpretationACM')
    dendrogram_interpretation = st.session_state.get('dendrogram_interpretation')
    fig_dendro = st.session_state.get('dendrogram')
    segmentation_profiles_text = st.session_state.get('segmentation_profiles_text')
    profils_y_table = st.session_state.get('profils_y_table')
    segmentation_profiles_table = st.session_state.get('segmentation_profiles_table')
    segmentation_detailed_profiles = st.session_state.get('segmentation_detailed_profiles')
    process = st.session_state.get('process')
    figs_variables_distribution = st.session_state.get("figs_variables_distribution", [])
    dataset_characteristics  = st.session_state.get("dataset_characteristics")
    variables_raw = st.session_state.get("variables_raw")
    data_preparation_synthesis = st.session_state.get("data_preparation_synthesis")    
    fig_missing_percentages = st.session_state.get("fig_missing_percentages")
    fig_missing_correlation_heatmap = st.session_state.get("fig_missing_correlation_heatmap")
    fig_missing_correlation_dendrogram = st.session_state.get("fig_missing_correlation_dendrogram")
    little_test_result = st.session_state.get("little_test_result")
    results_store = st.session_state.get("sankey_pair_results", {})
    latent_summary_text = st.session_state.get("latent_summary_text")
    sankey_diagram = st.session_state.get("sankey_diagram", {})
    sankey_latents = st.session_state.get("sankey_latents", {})
    sankey_diagram_base64 = st.session_state.get("sankey_diagram_base64")
    crosstabs_interpretation = st.session_state.get("crosstabs_interpretation", {})
    sankey_pair_results = st.session_state.get("sankey_pair_results", {})
    tm = st.session_state.get("target_modalities", {})
    syntheses_verbatim = st.session_state.get("syntheses_verbatim")
        
    # import des CSVs non encore importés
    df_ready = st.session_state.get('df_ready')
    target_detailed_profiles = st.session_state.get('profils_y')
    shortened_labels_mapping = st.session_state.get('shortened_labels_mapping')
    df_shortlabels = st.session_state.get("df_shortlabels")   
    ordinal_codification_mapping = st.session_state.get('ordinal_codification_mapping')
    df_encoded = st.session_state.get('df_encoded')
    preparation_details_payload = st.session_state.get("preparation_details_payload", {})
    details_preparation_selected = bool(st.session_state.get("details_preparation_selected", False))

    if dataset_key_questions_mode == "ab" and str(dataset_key_questions or "").strip():
        st.caption(f"Brief saisi: {dataset_key_questions}")
    progress_slot = st.empty()

    def _set_rf_progress(value: int, label: str):
        set_progress(value, label, phase="post_diagnostic")
        progress_slot.progress(value, text=label)

    # Execution pipeline selectionne avant rendu final.
    if (not passive_nav) and st.session_state.get("pipeline_ready_to_run", False) and not st.session_state.get(
        "pipeline_executed", False
    ):
        st.session_state["final_report_ready"] = False
        st.session_state["final_export_zip_bytes"] = None
        st.session_state["__PIPELINE_LABEL_REFRESHED__"] = False

        pipeline_selection = st.session_state.get("pipeline_selection", {})
        module_plan = get_selected_module_plan(pipeline_selection)
        total_expected = sum(float(item.get("expected_seconds", 0.0)) for item in module_plan) or 1.0
        expected_by_label = {item["label"]: float(item.get("expected_seconds", 0.0)) for item in module_plan}
        offset_by_label = {}
        cumulative = 0.0
        for item in module_plan:
            offset_by_label[item["label"]] = cumulative
            cumulative += float(item.get("expected_seconds", 0.0))

        progress_bar = progress_slot.progress(0, text="Préparation de l'exécution...")
        current_module_start = {"t": None}

        def _refresh_progress(module_name: str | None = None, *, force_complete: bool = False):
            current_label = module_name or st.session_state.get("pipeline_current_module", "") or "Exécution du pipeline"
            if force_complete:
                ratio = 1.0
            else:
                offset = float(offset_by_label.get(current_label, 0.0))
                expected = float(expected_by_label.get(current_label, 0.0))
                ratio = offset / total_expected
                if expected > 0 and current_module_start["t"] is not None:
                    elapsed = max(0.0, time.monotonic() - current_module_start["t"])
                    ratio = min(1.0, (offset + min(elapsed, expected)) / total_expected)
            progress_bar.progress(int(max(0.0, min(1.0, ratio)) * 100), text=f"Traitements en cours - module : {current_label}")

        def _on_progress(module_name: str):
            st.session_state["pipeline_current_module"] = module_name
            st.session_state["pipeline_current_function"] = ""
            current_module_start["t"] = time.monotonic()
            _refresh_progress(module_name)

        def _on_function(module_name: str, _function_name: str):
            _refresh_progress(module_name)

        with st.spinner("Exécution des traitements en cours..."):
            run_selected(
                pipeline_selection,
                show_details=False,
                progress_callback=_on_progress,
                function_progress_callback=_on_function,
            )
        _refresh_progress(force_complete=True)
        _set_rf_progress(65, "Traitements terminés")

        status = st.session_state.get("pipeline_status", "completed")
        if status == "completed":
            st.session_state["__PIPELINE_SHOW_SUCCESS__"] = True
        elif status == "completed_with_skips":
            logs = st.session_state.get("pipeline_execution_logs", [])
            skipped = [x for x in logs if x.get("status") == "skipped"]
            skipped_names = {x.get("module") for x in skipped}
        else:
            halt = st.session_state.get("pipeline_halt") or {}
            st.error(
                "Exécution des traitements arrêtée. "
                f"Module: {halt.get('module', 'inconnu')} | "
                f"Cause: {halt.get('cause', 'inconnue')} | "
                f"Detail: {halt.get('error', 'non disponible')}"
            )

        # Rafraichit les sorties apres execution.
        df_ready = st.session_state.get("df_ready")
        process = st.session_state.get("process")
        sankey_diagram = st.session_state.get("sankey_diagram")
        sankey_latents = st.session_state.get("sankey_latents")
        fig_dendro = st.session_state.get("dendrogram")

        if not st.session_state.get("__PIPELINE_LABEL_REFRESHED__", False):
            st.session_state["__PIPELINE_LABEL_REFRESHED__"] = True
            st.rerun()

    verb_cols = st.session_state.get("verbatim_candidates", [])
    verb_ids = st.session_state.get("verbatim_identifier_cols", [])

    # Affichage de la synthàse verbatim uniquement si des colonnes verbatim sont détectées.
    if verb_cols or st.session_state.get("verbatim_only_dataset"):
        with st.expander(_expander_label("verbatims"), expanded=False):
            if syntheses_verbatim:
                st.text_area("Résumé", syntheses_verbatim, height=320)
            elif st.session_state.get("verbatim_only_dataset"):
                st.info("Dataset 100% verbatim : synthèse absente ou non générée.")
            else:
                st.info("Aucune synthèse verbatim disponible pour ce run.")
            st.caption(f"Colonnes verbatim détectées : {len(verb_cols)} | Identifiants ignorés : {len(verb_ids)}")
            if verb_cols:
                st.write(", ".join(map(str, verb_cols)))

    # Contexte LLM minimal si absent (une seule fois)
    if (not passive_nav) and isinstance(df_ready, pd.DataFrame) and not st.session_state.get("__RF_CONTEXT_DONE__", False) and (
        not st.session_state.get("dataset_context") or not st.session_state.get("dataset_object")
    ):
        try:
            res = _call_llm(df_ready)
            st.session_state["dataset_object"] = res.get("dataset_object")
            st.session_state["dataset_context"] = res.get("dataset_context")
            st.session_state["dataset_recommendations"] = res.get("dataset_recommendations")
            st.session_state["target_variables"] = res.get("target_variables", [])
            st.session_state["target_modalities"] = res.get("target_modalities", {})
            st.session_state["illustrative_variables"] = res.get("illustrative_variables", [])
            st.session_state["__RF_CONTEXT_DONE__"] = True
        except Exception:
            pass

    selection = st.session_state.get(
        "pipeline_selection",
        {"preparation": True, "profilage": False, "analyse_descriptive": False},
    )
    prep_selected = bool(selection.get("preparation", True))
    profilage_selected = bool(selection.get("profilage", False))
    descriptive_selected = bool(selection.get("analyse_descriptive", False))
    # Harmoniser avec les clés des checkboxes de DiagnosticGlobal et forcer l'affichage
    # si des artefacts existent déjà en session (évite de masquer les résultats produits).
    sankey_crosstabs_selected = bool(
        st.session_state.get("run_sankey_crosstabs", selection.get("sankey_crosstabs", False))
        or st.session_state.get("sankey_pair_results")
        or st.session_state.get("crosstabs_interpretation")
    )
    distribution_figures_selected = bool(
        st.session_state.get("generate_distribution_figures", selection.get("distribution_figures", False))
        or st.session_state.get("figs_variables_distribution")
        or st.session_state.get("figs_variables_distribution_detailed")
    )
    show_insights = profilage_selected or descriptive_selected
    show_technical_text = descriptive_selected

    report_introduction = """
Le rapport d'analyse du jeu de donnees se compose des sections principales:
1. Principaux insights et recommandations
2. Profilage
3. Analyse descriptive
4. Analyse technique
5. Exports CSV
    """
    st.session_state["report_introduction"] = report_introduction

    T100 = "Rapport d'analyse du jeu de donnees"
    T0 = "Principaux insights et recommandations"
    T10 = "Sommaire"
    T11 = "1.1 Sujet du jeu de données"
    T13 = "1.3 La route vers l'objectif"
    T131 = "Relations entre variables (Diagramme Sankey)"
    T134 = "Croisement entre les variables : "
    T135 = "Pourcentage en ligne"
    T136 = "Interprétation du tableau des tris croisés"
    T15 = "1.5 Synthèse des verbatims"
    T21 = "2.1 Profils associés aux cibles"
    T211 = "Profils détaillés associés à la cible"
    T212 = "Attributs présentés par ordre de différenciation croissant"
    T22 = "2.2 Profils de la population entière"
    T221 = "Profils détaillés de la segmentation"
    T222 = "Attributs présentés par ordre de differenciation decroissant"
    T31 = "3.1 Relation entre les variables"
    T32 = "3.2 Relations hiérarchiques entre variables"
    T33 = "3.3 Représentation graphique des corrélations entre variables"
    T331 = "Dendrogramme des relations hiérarchiques entre variables"
    T34 = "3.4 Regroupement des variables en dimensions latentes"
    T35 = "3.5 Analyse descriptive des variables"
    T352 = "3.5.2 Profil dominant"
    T351 = "3.5.1 Distribution des variables"
    T3521 = "Histogramme de la variable"
    T41 = "4.1 Synthèse sur la structure du jeu de donnees et de sa preparation"
    T42 = "4.2 Contexte du jeu de données et rôle des variables"
    T43 = "4.3 Caractéristiques du jeu de données"
    T44 = "4.4 Caractéristiques des variables"
    T45 = "4.5 Valeurs manquantes"
    T451 = "4.5.1 Pourcentage de valeurs manquantes par variable"
    T4611 = "Pourcentage de valeurs manquantes"
    T452 = "4.5.2 Corrélation entre variables avec des valeurs manquantes"
    T4521 = "Carte de chaleur des corrélations entre variables avec des valeurs manquantes"
    T453 = "4.5.3 Corrélation entre variables avec des valeurs manquantes"
    T4531 = "Dendrogramme des corrélations entre variables avec des valeurs manquantes"
    T454 = "4.5.4 Résultat du test de Little"
    T461 = "Etapes des préparations"
    T462 = "Tableau de correspondance des libellés originaux/raccourcis"
    T463 = "Tableau de correspondance d'encodage des variables ordinales"

    proceed = False
    if mode == "automatique":
        proceed = not passive_nav
    else:
        if st.button("Générer la synthèse par IA"):
            proceed = True

    if show_insights and proceed and not st.session_state.get("final_report_ready", False) and isinstance(df_ready, pd.DataFrame) and not df_ready.empty:
        try:
            _set_rf_progress(72, "Synthèse générale")
            with st.spinner("Rédaction de la synthèse générale par LLM en cours..."):
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                brief_context = None
                brief_question = str(dataset_key_questions or "").strip()
                if brief_question:
                    brief_context = {
                        "brief_question": brief_question[:1200],
                        "brief_target_variable": st.session_state.get("brief_target_variable"),
                        "brief_illustrative_variables": st.session_state.get("brief_illustrative_variables"),
                        "brief_reason": st.session_state.get("brief_reason"),
                        "brief_relevance": st.session_state.get("brief_relevance"),
                    }
                    st.session_state["brief_context_global_synthesis"] = brief_context
                else:
                    st.session_state["brief_context_global_synthesis"] = None

                # Bloc prompt principal avec prise en compte optionnelle du brief (texte en clair, sans doublon)
                brief_focus_instructions = ""
                if brief_context:
                    bq_clean = brief_question.replace("\n", " ").strip()
                    brief_focus_instructions = f"""
                PRISE EN COMPTE DU BRIEF (mode A/B) :
                - Question du brief (rappel) : {bq_clean}
                - OBLIGATOIRE : commencer la section "Principaux insights" par : "Question du brief : {bq_clean}".
                - Ajouter 2 à 3 bullet points qui répondent directement à cette question, ancrés dans les données et en citant la cible du brief si elle est détectée.
                - Si les données ne permettent pas de répondre complètement, écrire explicitement "Données insuffisantes pour répondre complètement au brief" dans ce bloc et dans les risques/limites.
                    """

                prompt_parts = []

                if brief_focus_instructions:
                    prompt_parts.append(brief_focus_instructions)

                prompt_parts.append('''
                Vous êtes un expert en analyse de données. Répondez en français, clair et concis.
                Un jeu de données tabulaire (typiquement un export CSV/Excel) a été fourni dans le but d'en extraire des insights et recommandations sur les variables cibles, souvent des KPIs métier.
                Les informations à extraire ont été identifiées, et des analyses pour comprendre le contexte du jeu de données ont déjà été réalisées et vous sont fournies plus bas :
                - dataset_subject : le sujet du jeu de données
                - dataset_context : description du jeu de données et de ses variables
                - dataset_recommendations : recommandations techniques pour l'analyse d'une nouvelle version
                - la cible est définie par la variable cible ({target_variables}) et sa modalité cible (target_modality) donnée ici : {tm}
                
                Des analyses statistiques ont déjà été réalisées :
                - sankey_diagram : relations des variables illustratives vers les variables cibles
                - target_profiles_text : profils associés aux cibles identifiées
                - profil_dominant_analysis : attributs dominants (médiane pour les continues, mode pour les catégorielles)
                - interpretationACM : dimensions sémantiques issues de l'ACM
                - segmentation_profiles_text : profils issus d'une segmentation
                - dendrogramm_interpretation : interprétation du dendrogramme des corrélations
                
                Rédigez les principaux insights du jeu de données :
                - à partir des analyses statistiques réalisées
                - sous forme de bullet points
                Sections attendues :
                - profil dominant (profil_dominant_analysis)
                - profils associés à la cible (profils_y_text) avec le titre contenant {tm} ; ne mentionner que les fréquences de la population globale.
                - profils issus de la segmentation (segmentation_profiles_text)
                - variables clés manquantes pour expliquer la cible.
                Commencer par ce qui est le plus important et lié à dataset_subject.
                Distinguer ce qui est attendu de ce qui est surprenant ou contre-intuitif.
                Répondre aux objectifs recherchés s'ils ont été précisés dans dataset_key_questions.
                ''')

                prompt_parts.append('''
                Dans un paragraphe à part, proposez de refaire l'analyse avec un nouvel angle, en :
                -- prenant les autres cibles (autres modalités ou autres variables que la variable et sa modalité utilisée pour cette analyse), parmi celles déjà identifiées (target_variables)
                -- augmentant la granularité de l'analyse (plus de segments, discrétisation des variables...)
                -- ajoutant les variables manquantes qui seraient pertinentes pour répondre aux questions posées (dataset_key_questions) et qui sont identifiées dans dataset_recommendations.
                
                Finissez enfin par indiquer que tous les détails de l'analyse sont fournis dans les sections suivantes.
                ''')

                global_synthesis_prompt = "\n".join(prompt_parts)
                preview = df_ready.head(10).to_csv(index=False)
                preview = preview[:20000]  # limiter la taille

                payload = {
                    "columns": [str(c) for c in df_ready.columns],
                    "data_sample_preview_as_csv": preview,
                }

                # Sanitize crosstabs to avoid huge payloads (drop images / heavy matrices)
                raw_ct = st.session_state.get("crosstabs_interpretation", []) or []
                crosstabs_light = []
                for item in raw_ct:
                    try:
                        crosstabs_light.append({
                            "var_x": item.get("var_x"),
                            "var_y": item.get("var_y"),
                            "interpretation": item.get("interpretation", ""),
                        })
                    except Exception:
                        continue

                context_blob_global_synthesis = {
                    **payload,
                    "dataset_object": dataset_object,
                    "dataset_context": dataset_context,
                    "crosstabs_interpretation": crosstabs_light,
                    "sankey_interpretation_synthesis": st.session_state.get("sankey_interpretation_synthesis", ""),
                    "latent_summary_text": st.session_state.get("latent_summary_text", ""),
                    "interpretationACM": interpretationACM,
                    "dendrogram_interpretation": dendrogram_interpretation,
                    "profil_dominant_analysis": profil_dominant_analysis,
                    "profils_y_text": profils_y_text,
                    "segmentation_profiles_text": segmentation_profiles_text,
                }
                if str(dataset_key_questions or "").strip():
                    context_blob_global_synthesis["dataset_key_questions"] = dataset_key_questions
                if brief_context and dataset_key_questions_mode == "ab":
                    context_blob_global_synthesis["brief_context"] = brief_context

                user_content = json.dumps(context_blob_global_synthesis, ensure_ascii=False, default=str)

                r1 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": global_synthesis_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )

                global_synthesis = r1.choices[0].message.content
                # Si brief en mode A/B, préfixer la synthèse par un bloc déterministe conforme aux consignes
                if brief_context and dataset_key_questions_mode == "ab":
                    bq_clean = brief_question.replace("\n", " ").strip()
                    prefix = (
                        f"Question du brief : {bq_clean}\n"
                        "- Les points suivants répondent directement à cette question en s'appuyant sur les analyses réalisées "
                        "tout en couvrant l'ensemble des insights standards.\n\n"
                    )
                    global_synthesis = prefix + (global_synthesis or "")

                st.session_state.global_synthesis = global_synthesis
                st.session_state.global_synthesis_generated = True

        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'appel à l'API : {e}")

                
    if mode == "automatique":
        proceed = not passive_nav
    else:
        if st.button("Générer l'analyse technique par IA"):
            proceed = True
    if show_technical_text and proceed and not st.session_state.get("final_report_ready", False) and isinstance(df_ready, pd.DataFrame) and not df_ready.empty:
        try:
            _set_rf_progress(86, "Synthèse technique")
            with st.spinner("Rédaction de la synthèse sur la préparation du jeu de données par LLM en cours..."):
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                data_preparation = f'''Vous êtes un expert en analyse de données. Réponds en français, clair et concis.
                Un jeu de données a été préparé dans le but de réaliser une analyse.
                Rédigez une synthèse en 5-7 phrases sur les caractéristiques du jeu de données (dimensions, types des variables) et les traitements qui ont ét réalisés (données manquantes, aberrantes, variables trop corrélées, raccourcissement des labels, codification des variables ordinales).
                S'il y en a, interprétez les corrélations entre les variables avec des valeurs manquantes: matrice, dendrogramme, et test de Little, et les implications pour le traitement des valeurs manquantes, tout particulièrement les résultats au test de Little.
                Ainsi que le traitement des valeurs aberrantes (par défaut le taux de contamination utilisé est de 0.1%.). 
                Les documents de la préparation sont fournis plus bas.
                Ne parlez pas du sens sémantique des variables, uniquement des traitements qui ont été réalisés.
                Si un traitement n'avait pas besoin d'être réalisé (le test de Little n'est fait que si il y a des valeurs manquantes), n'en parlez pas.
                '''
                
                preview = df_ready.head(10).to_csv(index=False)
                preview = preview[:20000] 

                payload = {
                    "columns": [str(c) for c in df_ready.columns],
                    "data_sample_preview_as_csv": preview,
                }

                context_blob_data_preparation = {
                    **payload,
                    "dataset_characteristics": dataset_characteristics,
                    "caractéristiques des variables": variables_raw,
                    "valeurs manquantes - percentages par variable": fig_missing_percentages,
                    "valeurs manquantes - matrice de corrélation": fig_missing_correlation_heatmap,
                    "valeurs manquantes - dendrogramme des corrélations": fig_missing_correlation_dendrogram,
                    "valeurs manquantes - Résultat du test de Little": little_test_result
                }

                # dump robuste (évite erreurs de types non sérialisables)
                user_content_2 = json.dumps(context_blob_data_preparation, ensure_ascii=False, default=str)

                r2 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": data_preparation},
                        {"role": "user", "content": user_content_2},
                    ],
                )

                data_preparation_synthesis = r2.choices[0].message.content
                st.session_state.data_preparation_synthesis = data_preparation_synthesis
                st.session_state.data_preparation_synthesis_generated = True

        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'appel à l'API : {e}")


    # affichage final (sections repliees par defaut)
    if show_insights:
        with st.expander(_expander_label("insights"), expanded=False):
            brief_intro = ""
            if dataset_key_questions_mode == "ab":
                bq = str(dataset_key_questions or "").strip()
                if bq:
                    target_hint = st.session_state.get("brief_target_variable")
                    target_txt = f" (cible détectée : {target_hint})" if target_hint else ""
                    brief_intro = f"**Question du brief :** {bq}{target_txt}"

            gs_text = st.session_state.get("global_synthesis")
            if not isinstance(gs_text, str) or not gs_text.strip():
                gs_text = "Aucune synthèse disponible"
            combined = "\n\n".join([x for x in [brief_intro, gs_text] if x])
            st.markdown(combined)

    if profilage_selected:
        with st.expander(_expander_label("profilage"), expanded=False):
            st.markdown("### Profils associés à la cible")
            st.markdown(st.session_state.get("profils_y_text", "Aucun texte disponible"))
            st.markdown("### Segmentation")
            st.markdown(st.session_state.get("segmentation_profiles_text", "Aucun texte disponible"))

    has_sankey = (sankey_diagram is not None and sankey_diagram != {}) or bool(str(sankey_diagram_base64 or "").strip())
    sankey_text = str(st.session_state.get("sankey_interpretation_synthesis") or "").strip()
    acm_text = str(st.session_state.get("interpretationACM") or "").strip()
    dendro_text = str(st.session_state.get("dendrogram_interpretation") or "").strip()
    latent_text = str(st.session_state.get("latent_summary_text") or "").strip()
    has_latent_df = isinstance(sankey_latents, pd.DataFrame) and not sankey_latents.empty
    has_descriptive_content = has_sankey or bool(sankey_text) or bool(acm_text) or bool(dendro_text) or bool(fig_dendro) or bool(latent_text) or has_latent_df

    if False and (has_sankey or sankey_text):
        if has_sankey:
            st.markdown("### Principales relations entre les variables")
            _show_sankey_screen(sankey_diagram, st.session_state.get("sankey_diagram_base64"))
        if sankey_text:
            st.markdown(sankey_text)
    if False and (latent_text or has_latent_df):
        st.markdown("### Dimensions latentes")
        if latent_text:
            st.markdown(latent_text)
        if has_latent_df:
            st.dataframe(sankey_latents)

    if descriptive_selected:
        with st.expander(_expander_label("analyse_descriptive"), expanded=False):
            if has_descriptive_content:
                st.markdown("### Relations multivariées :")
                if has_sankey or sankey_text:
                    st.markdown("### Principales relations entre les variables")
                    if has_sankey:
                        _show_sankey_screen(sankey_diagram, st.session_state.get("sankey_diagram_base64"))
                    if sankey_text:
                        st.markdown(sankey_text)
                if acm_text:
                    st.markdown(acm_text)
                st.markdown("### Relations bivariées :")
                if dendro_text:
                    st.markdown(dendro_text)
                if fig_dendro:
                    st.subheader("Dendrogramme des corrélations :")
                    st.pyplot(fig_dendro)
                st.subheader("Regroupement des variables par dimensions latentes")
                if latent_text:
                    st.markdown(latent_text)
                if has_latent_df:
                    st.dataframe(sankey_latents)
            else:
                sankey_log = next(
                    (x for x in st.session_state.get("pipeline_execution_logs", []) if x.get("module") == "DiagramSankey"),
                    {}
                )
                reason = sankey_log.get("reason", "aucune sortie descriptive produite")
                st.info(f"Analyse descriptive non disponible pour ce run ({reason}).")
                llm_err = str(st.session_state.get("diagram_sankey_llm_error") or "").strip()
                if llm_err:
                    st.error(f"Erreur LLM DiagramSankey: {llm_err}")
                    
    if descriptive_selected:
        with st.expander(_expander_label("contexte"), expanded=False):
            st.markdown(st.session_state.get("dataset_object", "Aucune synthèse disponible"))
            st.markdown(st.session_state.get("dataset_recommendations", ""))

    # Tris croisés détaillés (si cochés dans DiagnosticGlobal ou déjà produits)
    raw_crosstab_list = st.session_state.get("crosstabs_interpretation", [])
    crosstab_list = []
    if isinstance(raw_crosstab_list, list):
        for item in raw_crosstab_list:
            if isinstance(item, dict):
                crosstab_list.append(item)
    if sankey_crosstabs_selected or crosstab_list:
        if crosstab_list:
            with st.expander(_expander_label("tris_croises"), expanded=False):
                for item in crosstab_list:
                    var_x = item.get("var_x")
                    var_y = item.get("var_y")
                    interpretation = item.get("interpretation", "")
                    heatmap_png = item.get("heatmap_png")
                    ct_count = item.get("ct_count")
                    metrics_caption = str(item.get("metrics_caption") or "").strip()
                    if heatmap_png or interpretation:
                        st.write(f"{T134} {var_x} - {var_y}")
                        if isinstance(ct_count, pd.DataFrame) and not ct_count.empty:
                            st.dataframe(ct_count, use_container_width=True)
                        if isinstance(heatmap_png, str):
                            try:
                                import base64 as _b64
                                st.image(_b64.b64decode(heatmap_png), caption=T135)
                            except Exception:
                                st.image(heatmap_png, caption=T135)
                        elif heatmap_png:
                            st.image(heatmap_png, caption=T135)
                        if metrics_caption:
                            st.caption(metrics_caption)
                        if interpretation:
                            st.write(interpretation)

    if prep_selected and descriptive_selected:
        with st.expander(_expander_label("analyse_technique"), expanded=False):
            st.markdown(st.session_state.get("data_preparation_synthesis", "Aucune synthèse technique disponible"))
            st.markdown(st.session_state.get("dataset_context", ""))

    # Distributions détaillées (si cochées), placées après l'analyse technique
    if distribution_figures_selected and descriptive_selected:
        dist_items = (
            st.session_state.get("figs_variables_distribution", [])
            or st.session_state.get("figs_variables_distribution_detailed", [])
        )
        if dist_items:
            with st.expander(_expander_label("distributions"), expanded=False):
                for item in dist_items:
                    title = item.get("title", "Distribution")
                    png = item.get("png", b"")
                    st.subheader(title)
                    st.image(png, caption="Histogramme de la distribution de la variable")

    if prep_selected:
        with st.expander(_expander_label("etapes_preparation"), expanded=False):
            if isinstance(process, pd.DataFrame):
                proc = process.copy()
                if verb_cols:
                    synth_row = {
                        "Etape": "Synthèse verbatims",
                        "Nb observations": proc["Nb observations"].iloc[-1] if "Nb observations" in proc.columns else "",
                        "Nb variables": proc["Nb variables"].iloc[-1] if "Nb variables" in proc.columns else "",
                        "Traitement": f"Synthèse de {len(verb_cols)} colonne(s) verbatim",
                    }
                    proc = pd.concat([proc, pd.DataFrame([synth_row])], ignore_index=True)
                st.dataframe(proc, use_container_width=True)
            else:
                st.info("Aucune étape de préparation disponible.")
            if details_preparation_selected:
                st.markdown("##### Détails de la préparation")
                _render_preparation_details(preparation_details_payload)
                
            # Info verbatim si aucune synthèse
            if not syntheses_verbatim:
                if st.session_state.get("verbatim_only_dataset"):
                    st.info("étape verbatim : dataset 100% texte, synthèse absente ou non générée.")
                elif verb_cols:
                    st.info(f"étape verbatim : {len(verb_cols)} colonne(s) texte traitée(s).")
    # =============================================================
    # GENERATION DU HTML
    # =============================================================

    if not st.session_state["final_report_ready"]:
        html = build_final_report_html(st.session_state, locals())
        st.session_state["final_report_html"] = html
        st.session_state["final_report_ready"] = True
        _set_rf_progress(100, "Rapport final prêt")

    # =============================================================
    # TELECHARGEMENT DU PACKAGE
    # =============================================================
    st.markdown("##### Export des fichiers de l'analyse")
    if st.button("Préparer le zip des fichiers", use_container_width=True):
        html_report = st.session_state.get("final_report_html") or ""
        st.session_state["final_export_zip_bytes"] = build_export_zip(st.session_state, html_report)

    export_bundle = st.session_state.get("final_export_zip_bytes")
    if export_bundle:
        st.download_button(
            label="Técharger le zip des fichiers",
            data=export_bundle,
            file_name="export_bundle.zip",
            mime="application/zip"
        )
        
        with st.expander("Explications sur le contenu des fichiers", expanded=False):
            st.markdown("""
                Le fichier zip contient:\n
                - Le rapport en format HTML avec les textes, tableaux, et graphiques.\n
                - les fichiers csv, qui incluent.\n
                    - le jeu de données préparé, après nettoyage: "df_ready"\n
                    - le jeu de données préparé, après nettoyage, et encodages des variables ordinales: "df_encoded"\n
                    - le tableau de correspondance entre les modalités des variables ordinales et leur codification: "ordinal_codification_mapping"\n            
                    - le tableau de correspondance entre les libellés des variables d'origine et leur raccourcissement: "shortened_labels_mapping"\n
                    - le tableau des profils de la segmentation: "segmentation_profiles_table"\n
                    - le tableau détaillé des profils de la segmentation (avec tous les attributs significatifs pour chaque segment): "segmentation_detailed_profiles"\n
                Davantage de fichiers sont disponibles dans chaque module.\n
            """)

    logs = st.session_state.get("pipeline_execution_logs", [])
    if logs:
        with st.expander("Logs d'exécution des modules", expanded=False):
            st.write("Variables cibles:", st.session_state.get("target_variables", []))
            st.write("Variables illustratives:", st.session_state.get("illustrative_variables", []))
            logs_df = pd.DataFrame(logs)
            if not logs_df.empty:
                keep_cols = [c for c in ["module", "status", "elapsed_sec"] if c in logs_df.columns]
                st.dataframe(logs_df[keep_cols] if keep_cols else logs_df, use_container_width=True)
                sankey_log = next(
                    (
                        item for item in logs
                        if isinstance(item, dict)
                        and item.get("module") == "DiagramSankey"
                        and item.get("status") == "skipped"
                    ),
                    None,
                )
                if isinstance(sankey_log, dict):
                    st.markdown("**Détail du log DiagramSankey**")
                    st.json(sankey_log)
                sankey_exit_debug = st.session_state.get("diagram_sankey_exit_debug")
                if isinstance(sankey_exit_debug, dict) and sankey_exit_debug:
                    st.markdown("**Trace interne DiagramSankey**")
                    st.json(sankey_exit_debug)
            elapsed = st.session_state.get("pipeline_execution_seconds")
            if isinstance(elapsed, (int, float)):
                st.write("Temps execution pipeline (s):", round(float(elapsed), 2))

    if st.session_state.pop("__PIPELINE_SHOW_SUCCESS__", False):
        st.success("Exécution des traitements terminée.")
    # =============================================================
    # Fin de l'étape
    # =============================================================

    st.markdown("##### Actions suivantes")
    qa_col, change_col, reset_col = st.columns(3)
    with qa_col:
        st.button("Posez une question", use_container_width=True, on_click=_navigate_to_qa)
    with change_col:
        st.button("Changer les objectifs", use_container_width=True, on_click=_change_objectives)
    with reset_col:
        if st.button("Réinitialiser", use_container_width=True):
            reset_app_state()

    st.session_state["etape40_terminee"] = True
    st.stop()
