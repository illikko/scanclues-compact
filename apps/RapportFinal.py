import streamlit as st
import pandas as pd
import matplotlib as mpl
from openai import OpenAI
import json
import os
import io
import base64
import plotly.io as pio
import html as _html
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

from core.df_registry import init_df_registry
from core.reset_state import reset_app_state
from .CadrageAnalyse import _call_llm
from .PipelineRunner import run_selected
from ._report import (
    add_from_state, add_text, add_table, add_figure_auto, build_html_report,
    add_text_html, add_table_html, build_html_report_with_tables, reset_report
)

MODE_KEY = "__NAV_MODE__"


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

def _rf_add_text(title: str, text: str):
    """Ajoute un bloc texte en HTML simple, en conservant les puces/retours."""
    if not text:
        return
    title = _html.escape(title or "")
    # on échappe le contenu puis on le met dans <pre> pour garder les puces/lignes
    body = _html.escape(str(text))
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  <pre style="white-space:pre-wrap;margin:6px 0 10px 0;color:#444">{body}</pre>
</section>
"""
    st.session_state.setdefault("_rf_blocks", []).append(block)

def _rf_add_df(title: str, df: pd.DataFrame, intro: str | None = None, max_height: str | None = None):
    """Ajoute un DataFrame en HTML avec scroll, et gère l'index intelligemment."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    title = _html.escape(title or "")
    intro_html = ""
    if intro:
        intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{_html.escape(str(intro))}</div>'

    show_index = should_show_index(df)
    
    # Correction: styles CSS pour rendre le tableau responsive
    table_html = df.to_html(
        index=show_index, 
        border=1, 
        escape=False,
        classes='dataframe table table-striped',
    )

    style_wrap = "overflow-x:auto;"
    if max_height:
        style_wrap += f"max-height:{max_height};overflow-y:auto;"

    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  <div style="{style_wrap}">{table_html}</div>
</section>
"""
    st.session_state.setdefault("_rf_blocks", []).append(block)

def _rf_add_image(title: str, img, intro: str | None = None, dpi: int = 120):
    """
    Accepte soit des bytes PNG, soit une Figure matplotlib.
    Amélioré pour les images trop petites.
    """
    # 1) Normaliser en bytes
    if hasattr(img, 'savefig'):  # Matplotlib Figure
        buf = io.BytesIO()
        # Correction: augmentation de la taille pour les petites images
        original_size = img.get_size_inches()
        if original_size[0] < 8:  # Si l'image est trop petite
            img.set_size_inches(10, 8)  # Taille minimale pour la lisibilité
        
        img.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        png_bytes = buf.getvalue()
    elif isinstance(img, (bytes, bytearray, memoryview)):
        png_bytes = bytes(img)
    else:
        return

    # 2) Encoder et pousser le bloc HTML
    b64 = base64.b64encode(png_bytes).decode("ascii")
    intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{intro}</div>' if intro else ""
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  <img src="data:image/png;base64,{b64}" style="display:block;max-width:100%;height:auto;margin:8px 0 0 0"/>
</section>
"""
    st.session_state.setdefault("_rf_blocks", []).append(block)


def _rf_add_collapsible(title: str, inner_html: str):
    """Ajoute un bloc repliable (<details>) pour les contenus détaillés."""
    if not inner_html:
        return
    title = _html.escape(title or "")
    block = f"""
<section style="margin:24px 0">
  <details style="background:#fafafa;border:1px solid #ddd;border-radius:6px;padding:12px">
    <summary style="cursor:pointer;font-weight:600;font-size:16px">{title}</summary>
    <div style="margin-top:12px">{inner_html}</div>
  </details>
</section>
"""
    st.session_state.setdefault("_rf_blocks", []).append(block)


def _rf_add_sankey(title: str, sankey_base64: str, intro: str | None = None):
    """Ajoute un diagramme Sankey en base64"""
    if not sankey_base64:
        return
        
    intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{intro}</div>' if intro else ""
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  <img src="data:image/png;base64,{sankey_base64}" style="display:block;max-width:100%;height:auto;margin:8px 0 0 0"/>
</section>
"""
    st.session_state.setdefault("_rf_blocks", []).append(block)

def _rf_add_sankey_html(title: str, fig, intro: str | None = None):
    """Ajoute un diagramme Sankey interactif (HTML Plotly)"""
    if fig is None:
        return

    # HTML Plotly autonome (PlotlyJS chargé via CDN)
    sankey_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn"
    )

    intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{intro}</div>' if intro else ""
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  {sankey_html}
</section>
"""
    st.session_state.setdefault("_rf_blocks", []).append(block)


def _rf_add_crosstab(title: str, heatmap_base64: str, interpretation: str, intro: str | None = None):
    """Ajoute un crosstab complet avec heatmap et interprétation"""
    if not heatmap_base64:
        return
        
    intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{intro}</div>' if intro else ""
    
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  <img src="data:image/png;base64,{heatmap_base64}" style="display:block;max-width:100%;height:auto;margin:8px 0 0 0"/>
  <div style="margin-top:16px;padding:12px;background:#f8f9fa;border-left:4px solid #007cba">
    <strong>Interprétation :</strong><br>
    {interpretation}
  </div>
</section>
"""
    st.session_state.setdefault("_rf_blocks", []).append(block)

# gérer les index des dataframes (la 1ère colonne disparait lorsqu'elle est en index)
def should_show_index(df: pd.DataFrame) -> bool:
    idx = df.index
    is_default_range = isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1
    has_name = bool(idx.name)  # None ou "" => pas de nom
    return (not is_default_range) or has_name


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

    import json
    import plotly.graph_objects as go
    import plotly.io as pio
    import streamlit.components.v1 as components

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
    # Réinitialiser les blocs/flags de section détaillée à chaque exécution
    st.session_state["_rf_blocks"] = []
    st.session_state["__DETAIL_SECTION_ADDED__"] = False
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    
    # Initialisation des états
    if "final_report_ready" not in st.session_state:
        st.session_state["final_report_ready"] = False
    st.session_state.setdefault("final_export_zip_bytes", None)
    st.session_state.setdefault("pipeline_executed", False)
    st.session_state.setdefault(
        "pipeline_selection",
        {"preparation": True, "profilage": False, "analyse_descriptive": False},
    )
    # Forcer la régénération du rapport si dataset 100% verbatim
    if st.session_state.get("verbatim_only_dataset"):
        st.session_state["final_report_ready"] = False

    defaults = {
        'dataset_object': None,                      
        'dataset_context': None,               
        'dataset_key_questions': None,              
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
        'etape40_terminee': False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


    # chargement des informations nécessaires
    dataset_object = st.session_state.get('dataset_object')
    dataset_context = st.session_state.get('dataset_context')
    dataset_recommendations = st.session_state.get('dataset_recommendations')
    dataset_key_questions = st.session_state.get('dataset_key_questions_value')
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

    if dataset_key_questions_mode == "ab" and str(dataset_key_questions or "").strip():
        st.caption(f"Brief saisi: {dataset_key_questions}")
    progress_slot = st.empty()

    # Execution pipeline selectionne avant rendu final.
    if st.session_state.get("pipeline_ready_to_run", False) and not st.session_state.get(
        "pipeline_executed", False
    ):
        st.session_state["final_report_ready"] = False
        st.session_state["final_export_zip_bytes"] = None
        st.session_state["__PIPELINE_LABEL_REFRESHED__"] = False

        def _render_progress():
            module_name = st.session_state.get("pipeline_current_module", "")
            fn_name = st.session_state.get("pipeline_current_function", "")
            progress_slot.info(
                f"Module en cours: {module_name}\n"
                f"Fonction en cours: {fn_name or '-'}"
            )

        def _on_progress(module_name: str):
            st.session_state["pipeline_current_module"] = module_name
            st.session_state["pipeline_current_function"] = ""
            _render_progress()

        def _on_function(module_name: str, function_name: str):
            st.session_state["pipeline_current_module"] = module_name
            st.session_state["pipeline_current_function"] = function_name
            _render_progress()

        with st.spinner("Execution pipeline en cours..."):
            run_selected(
                st.session_state.get("pipeline_selection", {}),
                show_details=False,
                progress_callback=_on_progress,
                function_progress_callback=_on_function,
            )
        progress_slot.empty()

        status = st.session_state.get("pipeline_status", "completed")
        if status == "completed":
            st.success("Execution pipeline terminee.")
        elif status == "completed_with_skips":
            logs = st.session_state.get("pipeline_execution_logs", [])
            skipped = [x for x in logs if x.get("status") == "skipped"]
            skipped_names = {x.get("module") for x in skipped}
            if skipped_names and skipped_names.issubset({"DiagramSankey"}):
                st.info(
                    "Exécution pipeline terminée. "
                    "DiagramSankey non exécuté (variables cibles/illustratives non disponibles)."
                )
            else:
                st.warning("Execution pipeline terminee avec modules ignores (voir logs).")
        else:
            halt = st.session_state.get("pipeline_halt") or {}
            st.error(
                "Execution pipeline arretee. "
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

    logs = st.session_state.get("pipeline_execution_logs", [])
    if logs:
        with st.expander("Logs d'exécution des modules", expanded=False):
            st.write("Variables cibles:", st.session_state.get("target_variables", []))
            st.write("Variables illustratives:", st.session_state.get("illustrative_variables", []))
            logs_df = pd.DataFrame(logs)
            if not logs_df.empty:
                keep_cols = [c for c in ["module", "status", "elapsed_sec"] if c in logs_df.columns]
                st.dataframe(logs_df[keep_cols] if keep_cols else logs_df, use_container_width=True)
            elapsed = st.session_state.get("pipeline_execution_seconds")
            if isinstance(elapsed, (int, float)):
                st.write("Temps execution pipeline (s):", round(float(elapsed), 2))

    verb_cols = st.session_state.get("verbatim_candidates", [])
    verb_ids = st.session_state.get("verbatim_identifier_cols", [])

    # Affichage de la synthèse verbatim uniquement si des colonnes verbatim sont détectées
    if verb_cols or st.session_state.get("verbatim_only_dataset"):
        with st.expander("Synthèse des verbatims", expanded=False):
            if syntheses_verbatim:
                st.text_area("Résumé", syntheses_verbatim, height=320)
            elif st.session_state.get("verbatim_only_dataset"):
                st.info("Dataset 100% verbatim : synthèse absente ou non générée.")
            else:
                st.info("Aucune synthèse verbatim disponible pour ce run.")
            st.caption(f"Colonnes verbatim détectées : {len(verb_cols)} | Identifiants ignorés : {len(verb_ids)}")
            if verb_cols:
                st.write(", ".join(map(str, verb_cols)))

    # Contexte LLM minimal si absent.
    if isinstance(df_ready, pd.DataFrame) and (
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
    # si des artefacts existent déjà en session (évite de masquer les résultats produits).
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
    T11 = "1.1 Sujet du jeu de donnees"
    T13 = "1.3 La route vers l'objectif"
    T131 = "Relations entre variables (Diagramme Sankey)"
    T134 = "Croisement entre les variables : "
    T135 = "Pourcentage en ligne"
    T136 = "Interpretation du tableau des tris croises"
    T15 = "1.5 Synthese des verbatims"
    T21 = "2.1 Profils associes aux cibles"
    T211 = "Profils detailles associes a la cible"
    T212 = "Attributs presentes par ordre de differenciation croissant"
    T22 = "2.2 Profils de la population entiere"
    T221 = "Profils detailles de la segmentation"
    T222 = "Attributs presentes par ordre de differenciation decroissant"
    T31 = "3.1 Relation entre les variables"
    T32 = "3.2 Relations hierarchiques entre variables"
    T33 = "3.3 Representation graphique des correlations entre variables"
    T331 = "Dendrogramme des relations hierarchiques entre variables"
    T34 = "3.4 Regroupement des variables en dimensions latentes"
    T35 = "3.5 Analyse descriptive des variables"
    T352 = "3.5.2 Profil dominant"
    T351 = "3.5.1 Distribution des variables"
    T3521 = "Histogramme de la variable"
    T41 = "4.1 Synthese sur la structure du jeu de donnees et de sa preparation"
    T42 = "4.2 Contexte du jeu de donnees et role des variables"
    T43 = "4.3 Caracteristiques du jeu de donnees"
    T44 = "4.4 Caracteristiques des variables"
    T45 = "4.5 Valeurs manquantes"
    T451 = "4.5.1 Pourcentage de valeurs manquantes par variable"
    T4611 = "Pourcentage de valeurs manquantes"
    T452 = "4.5.2 Correlation entre variables avec des valeurs manquantes"
    T4521 = "Carte de chaleur des correlations entre variables avec des valeurs manquantes"
    T453 = "4.5.3 Correlation entre variables avec des valeurs manquantes"
    T4531 = "Dendrogramme des correlations entre variables avec des valeurs manquantes"
    T454 = "4.5.4 Resultat du test de Little"
    T461 = "Etapes des preparations"
    T462 = "Tableau de correspondance des libelles originaux/raccourcis"
    T463 = "Tableau de correspondance d'encodage des variables ordinales"

    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Générer la synthèse par IA"):
            proceed = True

    if show_insights and proceed and not st.session_state.get("final_report_ready", False) and isinstance(df_ready, pd.DataFrame) and not df_ready.empty:
        try:
            with st.spinner("Rédaction de la synthèse générale par LLM en cours..."):
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                global_synthesis_prompt = "\n".join(['''
                Vous êtes un expert en analyse de donnees. Repondez en français, clair et concis.
                Un jeu de données tabulaire (typiquement un export CSV/excel) a été fourrni dans le but d'en extraire des insights, recommandations, sur les variables cibles, souvent des KPIs métiers.
                Les informations à extraire ont été identifiées, et des analyses pour comprendre le contexte du jeu de données ont déjà été réalisées et vous sont fournies plus bas:
                - dataset_subject: pour définir le sujet sur lequel porte le jeu de données
                - dataset_context: une description du jeu de données et de ses variables
                - dataset_recommendations: des recommandations techniques pour l'analyse d'une nouvelle version de ce jeu de données.
                - la cible est définie par la variable cible ({target_variables}) et sa modalité cible (target_modality), et sont données ici : {tm}
                
                Des analyses statistiques ont déjà été réalisées:
                - sankey_diagram: le diagramme partant des variables illustratives (sociodémographiques si les unités d'obervations sont des personnes) vers les variables cibles
                - target_profiles_text: la description des profils associées aux cibles identifiées.
                - profil_dominant_analysis: les attributs dominants (médiane pour les variables continues, mode pour les variables catégorielles) 
                - interpretationACM: une analyse factorielle (ACM) qui permet de définir dimensions sémantiques (les axes de l'ACP) sur lesquelles des groupes d'attributs s'opposent
                - segmentation_profiles_text: la description des profils isssus d'une segmentation
                - dendrogramm_interpretation: l'interprétation du dendrogramme des corrélations entre les variables
                - ctas_rules_text: les règles d'obtention de la cible à partir des variables illustratives (sociodémographiques)
                
                Rédigez les principaux insights du jeu de données: 
                - à partir des analyses statistiques réalisées
                - sous formes de bullet points
                avec les sections suivantes:
                - avec la description du profil dominant (détaillé dans profil_dominant_analysis)
                - des profils associés à la cible (détaillé dans profils_y_text), que vous mentionnez dans le titre de cette section : {tm}. Ne mentionnez que les fréquences de la population globale, pas les fréquences cible.
                - les profils issus de la segmentation de la population globale (détaillé dans segmentation_profiles_text)
                - des attributs permettant d'obtenir la cible (détaillé dans ctas_rules_text)
                - s'il manque des variables clé pour expliquer la variable cible.
                Commencer les énumérations par ce qui est le plus important et en rapport avec le dataset_subject.
                Distinguez ce qui est attendus de ce qui est plus surprenant ou contre-intuitif.
                Répondez aux objectifs recherchés si ils ont été précisés dans dataset_key questions.
                
                Dans un paragraphe à part, proposez de refaire l'analyse avec un nouvel angle, en :
                -- prenant les autres cibles (autres modalités ou autres variables que la variable et sa modalité utilisée pour cette analyse), parmi celles déjà identifiées (target_variables) 
                -- augmentant la granularité de l'analyse (augmenter le nombre des segments, la discrétisation des variables...)
                -- ajouter les variables qui manquent actuellement dans le jeu de données mais qui seraient pertinentes pour répondre aux questions posées (dataset_key_questions) et qui sont identifiées dans dataset_recommendations.
                
                Finissez enfin par indiquer que tous les détails de l'analyse sont fournis dans les sections suivantes.
                '''])

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
                if dataset_key_questions_mode == "ab" and str(dataset_key_questions or "").strip():
                    context_blob_global_synthesis["dataset_key_questions"] = dataset_key_questions

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
                st.session_state.global_synthesis = global_synthesis
                st.session_state.global_synthesis_generated = True

        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'appel a l'API : {e}")

    if mode == "automatique":
        proceed = True
    else:
        if st.button("Générer l'analyse technique par IA"):
            proceed = True
    if show_technical_text and proceed and not st.session_state.get("final_report_ready", False) and isinstance(df_ready, pd.DataFrame) and not df_ready.empty:
        try:
            with st.spinner("Rédaction de la synthèse sur la préparation du jeu de données par LLM en cours..."):
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                data_preparation = f'''Vous êtes un expert en analyse de données. Réponds en français, clair et concis.
                Un jeu de données a été préparé dans le but de réaliser une analyse.
                Rédigez une synthèse en 5-7 phrases sur les caractéristiques du jeu de données (dimensions, types des variables) et les traitements qui ont été réalisés (données manquantes, aberrantes, variables trop corrélées, raccourcissement des labels, codification des variables ordinales).
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
        with st.expander("Principaux insights", expanded=False):
            st.markdown(st.session_state.get("global_synthesis", "Aucune synthèse disponible"))

    if profilage_selected:
        with st.expander("Profilage", expanded=False):
            st.markdown("### Profils associés à la cible")
            st.markdown(st.session_state.get("profils_y_text", "Aucun texte disponible"))
            st.markdown("### Segmentation")
            st.markdown(st.session_state.get("segmentation_profiles_text", "Aucun texte disponible"))

    has_sankey = sankey_diagram is not None and sankey_diagram != {}
    sankey_text = str(st.session_state.get("sankey_interpretation_synthesis") or "").strip()
    acm_text = str(st.session_state.get("interpretationACM") or "").strip()
    dendro_text = str(st.session_state.get("dendrogram_interpretation") or "").strip()
    latent_text = str(st.session_state.get("latent_summary_text") or "").strip()
    has_latent_df = isinstance(sankey_latents, pd.DataFrame) and not sankey_latents.empty
    has_descriptive_content = has_sankey or bool(sankey_text) or bool(acm_text) or bool(dendro_text) or bool(fig_dendro) or bool(latent_text) or has_latent_df


    if descriptive_selected:
        with st.expander("Analyse descriptive", expanded=False):
            if has_descriptive_content:
                if has_sankey:
                    st.markdown("### Principales relations entre les variables :")
                    _show_sankey_screen(sankey_diagram, st.session_state.get("sankey_diagram_base64"))
                if sankey_text:
                    st.markdown(sankey_text)
                st.markdown("### Relations multivariées :")
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
        with st.expander("Contexte du jeu de données", expanded=False):
            st.markdown(st.session_state.get("dataset_object", "Aucune synthèse disponible"))
            st.markdown(st.session_state.get("dataset_recommendations", ""))

        # Tris croisés détaillés (si cochés dans DiagnosticGlobal)
        if sankey_crosstabs_selected:
            results_store = st.session_state.get("sankey_pair_results", {})
            crosstab_list = st.session_state.get("crosstabs_interpretation", [])
            if (isinstance(results_store, dict) and results_store) or crosstab_list:
                with st.expander("Tris croisés détaillés", expanded=False):
                    if isinstance(results_store, dict) and results_store:
                        pairs_sorted = sorted(results_store.items(), key=lambda kv: kv[1].get("v", 0), reverse=True)
                        for _, res in pairs_sorted:
                            var_x = res.get("var_x")
                            var_y = res.get("var_y")
                            interpretation = res.get("interpretation", "")
                            heatmap_png = res.get("heatmap_png")
                            if heatmap_png or interpretation:
                                st.write(f"{T134} {var_x} - {var_y}")
                                if isinstance(heatmap_png, str):
                                    try:
                                        import base64 as _b64
                                        st.image(_b64.b64decode(heatmap_png), caption=T135)
                                    except Exception:
                                        st.image(heatmap_png, caption=T135)
                                elif heatmap_png:
                                    st.image(heatmap_png, caption=T135)
                                if interpretation:
                                    st.write(interpretation)
                    elif crosstab_list:
                        for item in crosstab_list:
                            var_x = item.get("var_x")
                            var_y = item.get("var_y")
                            interpretation = item.get("interpretation", "")
                            heatmap_png = item.get("heatmap_png")
                            if heatmap_png or interpretation:
                                st.write(f"{T134} {var_x} - {var_y}")
                                if isinstance(heatmap_png, str):
                                    try:
                                        import base64 as _b64
                                        st.image(_b64.b64decode(heatmap_png), caption=T135)
                                    except Exception:
                                        st.image(heatmap_png, caption=T135)
                                elif heatmap_png:
                                    st.image(heatmap_png, caption=T135)
                                if interpretation:
                                    st.write(interpretation)

    if prep_selected and descriptive_selected:
        with st.expander("Analyse technique du jeu de données", expanded=False):
            st.markdown(st.session_state.get("data_preparation_synthesis", "Aucune synthèse technique disponible"))
            st.markdown(st.session_state.get("dataset_context", ""))

    # Distributions détaillées (si cochées), placées après l’analyse technique
    if distribution_figures_selected and descriptive_selected:
        dist_items = (
            st.session_state.get("figs_variables_distribution", [])
            or st.session_state.get("figs_variables_distribution_detailed", [])
        )
        if dist_items:
            with st.expander("Distributions détaillées", expanded=False):
                for item in dist_items:
                    title = item.get("title", "Distribution")
                    png = item.get("png", b"")
                    st.subheader(title)
                    st.image(png, caption="Histogramme de la distribution de la variable")

    if prep_selected:
        with st.expander("Etapes des préparations", expanded=False):
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
            # Info verbatim si aucune synthèse
            if not syntheses_verbatim:
                if st.session_state.get("verbatim_only_dataset"):
                    st.info("Étape verbatim : dataset 100% texte, synthèse absente ou non générée.")
                elif verb_cols:
                    st.info(f"Étape verbatim : {len(verb_cols)} colonne(s) texte traitée(s).")
    # =============================================================
    # GENERATION DU HTML
    # =============================================================

    if not st.session_state["final_report_ready"]:
        # Principaux insights
        _rf_add_text(T0, st.session_state.get("global_synthesis"))        
        
        # Rapport
        _rf_add_text(T10, st.session_state.get("report_introduction"))
        _rf_add_text(T11, st.session_state.get("dataset_object"))      

        # AFFICHAGE DU DIAGRAMME SANKEY
        sankey_diagram = st.session_state.get("sankey_diagram")
        if sankey_diagram is not None:
            _rf_add_sankey_html(
                T13,
                sankey_diagram,
                intro=T131
            )

        _rf_add_text("", st.session_state.get("sankey_interpretation_synthesis"))

        # crosstabs (section séparée, si demandé ou si artefacts existants)
        if sankey_crosstabs_selected:
            results_store = st.session_state.get("sankey_pair_results", {})
            crosstab_list = st.session_state.get("crosstabs_interpretation", [])
            parts = []
            import base64 as _b64

            if isinstance(results_store, dict) and results_store:
                pairs_sorted = sorted(
                    results_store.items(),
                    key=lambda kv: kv[1].get("v", 0),
                    reverse=True
                )
                for pair_id, res in pairs_sorted:
                    var_x = res.get("var_x")
                    var_y = res.get("var_y")
                    interpretation = res.get("interpretation", "")
                    heatmap_png = res.get("heatmap_png")
                    img_html = ""
                    if isinstance(heatmap_png, str):
                        b64 = heatmap_png
                    elif isinstance(heatmap_png, (bytes, bytearray, memoryview)):
                        b64 = _b64.b64encode(bytes(heatmap_png)).decode("ascii")
                    else:
                        b64 = ""
                    if b64:
                        img_html = f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px'/>"
                    if img_html or interpretation:
                        parts.append(
                            f"<div style='margin-bottom:16px'><h4 style='margin:0 0 6px 0'>{_html.escape(str(var_x))} vs {_html.escape(str(var_y))}</h4>{img_html}<pre style='white-space:pre-wrap;color:#444;margin-top:6px'>{_html.escape(str(interpretation))}</pre></div>"
                        )

            if not parts and crosstab_list:
                for item in crosstab_list:
                    var_x = item.get("var_x")
                    var_y = item.get("var_y")
                    interpretation = item.get("interpretation", "")
                    heatmap_png = item.get("heatmap_png")
                    img_html = ""
                    if isinstance(heatmap_png, str):
                        b64 = heatmap_png
                    elif isinstance(heatmap_png, (bytes, bytearray, memoryview)):
                        b64 = _b64.b64encode(bytes(heatmap_png)).decode("ascii")
                    else:
                        b64 = ""
                    if b64:
                        img_html = f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px'/>"
                    if img_html or interpretation:
                        parts.append(
                            f"<div style='margin-bottom:16px'><h4 style='margin:0 0 6px 0'>{_html.escape(str(var_x))} vs {_html.escape(str(var_y))}</h4>{img_html}<pre style='white-space:pre-wrap;color:#444;margin-top:6px'>{_html.escape(str(interpretation))}</pre></div>"
                        )

            if parts:
                _rf_add_collapsible(
                    "Tris croisés détaillés",
                    "".join(parts),
                )

        if syntheses_verbatim:
            _rf_add_text(T15, syntheses_verbatim)
        elif st.session_state.get("verbatim_only_dataset"):
            _rf_add_text(T15, "Dataset 100% verbatim détecté, mais aucune synthèse n'a été générée.")

        if profils_y_text is not None:
            _rf_add_text(T21, st.session_state.get("profils_y_text"))

        if isinstance(st.session_state.get("profils_y"), pd.DataFrame):
            _rf_add_df(
                T211,
                st.session_state["profils_y"],
                intro=T212,
                max_height="480px",
            )  
              
        _rf_add_text(T22, st.session_state.get("segmentation_profiles_text"))
        
        segmentation_profiles_table = st.session_state.get("segmentation_profiles_table")
        if isinstance(segmentation_profiles_table, pd.DataFrame):
            _rf_add_df(
                T221,
                segmentation_profiles_table,
                intro=T222,
                max_height="480px",
            )

        # analyse descriptive (analyse univariée, bivariée, mutlivariée)
        _rf_add_text(T31, st.session_state.get("interpretationACM"))
        _rf_add_text(T32, st.session_state.get("dendrogram_interpretation"))
        
        # affichage du dendrogramme des corrélations
        if "dendrogram" in st.session_state:
            _rf_add_image(
                title=T33,
                img=st.session_state["dendrogram"],
                dpi=120,
                intro=T331
            )

        if "latent_summary_text" in st.session_state:
            _rf_add_text(T34, st.session_state.get("latent_summary_text"))
            

        if isinstance(st.session_state.get("sankey_latents"), pd.DataFrame):
            _rf_add_df(
                "",
                st.session_state["sankey_latents"],
                intro=T34,
                max_height="480px",
            )
        # Introduction de la section descriptive détaillée
        _rf_add_text(T35, T351)

        # Affichage des histogrammes de toutes les variables (si l'option est active ou si des artefacts existent)
        if distribution_figures_selected:
            dist_items = (
                st.session_state.get("figs_variables_distribution", [])
                or st.session_state.get("figs_variables_distribution_detailed", [])
            )
            for item in dist_items:
                _rf_add_image(
                    title=item.get("title", "Distribution"),
                    img=item.get("png", b""),
                    intro=T3521,
                )

        if isinstance(dominant_continues, pd.DataFrame):
            _rf_add_df(
                T351,
                dominant_continues,
                intro=T351,
                max_height="480px",
            )

        if isinstance(dominant_discretes, pd.DataFrame):
            _rf_add_df(
                T351,
                dominant_discretes,
                intro=T351,
                max_height="480px",
            )
        # profil dominant
        _rf_add_text(T352, st.session_state.get("profil_dominant_analysis"))

        # Analyse technique du jeu de données
        _rf_add_text(T41, st.session_state.get("data_preparation_synthesis"))    
        _rf_add_text(T42, st.session_state.get("dataset_context"))
        _rf_add_text(T43, st.session_state.get("dataset_characteristics"))

        if isinstance(st.session_state.get("variables_raw"), pd.DataFrame):
            _rf_add_df(
                T44,
                st.session_state["variables_raw"],
                intro=T451,
                max_height="600px", 
            )
            
        _rf_add_text(T45, st.session_state.get("missing_values"))
    
        if "fig_missing_percentages" in st.session_state:
            _rf_add_image(
                title=T451,
                img=st.session_state["fig_missing_percentages"],
                intro=T4611
            )
            
        if "fig_missing_correlation_heatmap" in st.session_state:
            _rf_add_image(
                title=T452,
                img=st.session_state["fig_missing_correlation_heatmap"],
                dpi=150,
                intro=T4521
            )

        if "fig_missing_correlation_dendrogram" in st.session_state:
            _rf_add_image(
                title=T453,
                img=st.session_state["fig_missing_correlation_dendrogram"],
                intro=T4531
            )           
        
        
        _rf_add_text(T454, st.session_state.get("little_test_result"))

        if isinstance(st.session_state.get("process"), pd.DataFrame):
            _rf_add_df(
                T461,
                st.session_state["process"],
                intro=T461,
                max_height="480px",
            )
        
        shortened_labels_mapping = st.session_state.get('shortened_labels_mapping')     
        if isinstance(shortened_labels_mapping, pd.DataFrame):
            _rf_add_df(
                T462,
                shortened_labels_mapping,
                intro=T462,
                max_height="480px",
            )

        ordinal_codification_mapping = st.session_state.get('ordinal_codification_mapping')     
        if isinstance(ordinal_codification_mapping, pd.DataFrame):
            _rf_add_df(
                T463,
                ordinal_codification_mapping,
                intro=T463,
                max_height="480px",
            )

        # construction une seule fois
        html = build_html_report_with_tables(title=T100)
        # CSS minimal pour une police lisible et pas de gros blancs autour des images
        _css = """
        <style>
        table { font-size:14px; border-collapse:collapse; }
        th, td { border:1px solid #e5e7eb; padding:6px 8px; vertical-align:top; }
        img { display:block; margin:0; max-width:100%; height:auto; }
        /* Correction: styles ameliores pour les tableaux */
        .dataframe { 
            width: 100%; 
            table-layout: fixed;
            word-wrap: break-word;
        }
        .dataframe th, .dataframe td {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        </style>
        """
        extra = "".join(st.session_state.get("_rf_blocks", []))
        st.session_state["_rf_blocks"] = []  # on purge pour éviter les doublons au rerun

        if "</body>" in html:
            html = html.replace("</body>", _css + extra + "</body>")
        else:
            html = _css + html + extra

        st.session_state["final_report_html"] = html    

        # =============================================================
        # CONSTRUCTION DU PACKETS DE FICHIERS
        # =============================================================
        
        html_report = st.session_state.get("final_report_html")

        
        # --- EXPORT DES FICHIERS ---

        # --- Construction du ZIP en mémoire ---
        zip_buffer = BytesIO()
        df_ready_zip = st.session_state.get("df_ready")
        segmentation_profiles_table_zip = st.session_state.get("segmentation_profiles_table")
        segmentation_detailed_profiles_zip = st.session_state.get("segmentation_detailed_profiles")
        profils_y_table_zip = st.session_state.get("profils_y_table")
        profils_y_detailed_zip = st.session_state.get("profils_y_detailed")
        shortened_labels_mapping_zip = st.session_state.get("shortened_labels_mapping")
        df_shortlabels_zip = st.session_state.get("df_shortlabels")
        ordinal_codification_mapping_zip = st.session_state.get("ordinal_codification_mapping")
        df_encoded_zip = st.session_state.get("df_encoded")

        with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED) as zf:
            # HTML
            zf.writestr("report.html", html_report.encode("utf-8"))
                    
            # CSV (UTF-8 avec BOM + séparateur)
            if isinstance(df_ready_zip, pd.DataFrame):
                zf.writestr(
                    "df_ready.csv",
                    df_ready_zip.to_csv(sep=";", index=False).encode("latin-1", errors="replace")
                )        
            
            if isinstance(segmentation_profiles_table_zip, pd.DataFrame):
                zf.writestr(
                    "segmentation_profiles_table.csv",
                    segmentation_profiles_table_zip.to_csv(sep=";", index=True).encode("latin-1", errors="replace")
                )
            
            if isinstance(segmentation_detailed_profiles_zip, pd.DataFrame):
                zf.writestr(
                    "segmentation_detailed_profiles.csv",
                    segmentation_detailed_profiles_zip.to_csv(sep=";", index=False).encode("latin-1", errors="replace")
                )

            if isinstance(profils_y_table_zip, pd.DataFrame):
                zf.writestr(
                    "profils_y_table.csv",
                    profils_y_table_zip.to_csv(sep=";", index=False).encode("latin-1", errors="replace")
                )

            if isinstance(profils_y_detailed_zip, pd.DataFrame):
                zf.writestr(
                    "profils_y_detailed.csv",
                    profils_y_detailed_zip.to_csv(sep=";", index=False).encode("latin-1", errors="replace")
                )
                
            if isinstance(shortened_labels_mapping_zip, pd.DataFrame):
                zf.writestr(
                    "shortened_labels_mapping.csv",
                    shortened_labels_mapping_zip.to_csv(sep=";", index=False).encode("latin-1", errors="replace")
                )
            
            if isinstance(df_shortlabels_zip, pd.DataFrame):
                zf.writestr(
                    "df_shortlabels.csv",
                    df_shortlabels_zip.to_csv(sep=";", index=False).encode("latin-1", errors="replace")
                )

            if isinstance(ordinal_codification_mapping_zip, pd.DataFrame):
                zf.writestr(
                    "ordinal_codification_mapping.csv",
                    ordinal_codification_mapping_zip.to_csv(sep=";", index=False).encode("latin-1", errors="replace")
                )

            if isinstance(df_encoded_zip, pd.DataFrame):
                zf.writestr(
                    "df_encoded.csv",
                    df_encoded_zip.to_csv(sep=";", index=False).encode("latin-1", errors="replace")
                )

        # Important : se repositionner au début
        zip_buffer.seek(0)
        st.session_state["final_export_zip_bytes"] = zip_buffer.getvalue()
        st.session_state["final_report_ready"] = True

    # =============================================================
    # TELECHARGEMENT DU PACKAGE
    # =============================================================
    export_bundle = st.session_state.get("final_export_zip_bytes")
    if export_bundle:
        st.header("Export des fichiers de l'analyse")
        st.download_button(
            label="Télécharger le zip des fichiers",
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
    # =============================================================
    # Fin de l'étape
    # =============================================================

    st.markdown("#### Actions suivantes")
    qa_col, change_col, reset_col = st.columns(3)
    with qa_col:
        if st.button("Posez une question", use_container_width=True):
            try:
                st.query_params["step"] = "4"
            except Exception:
                st.experimental_set_query_params(step="4")
            st.session_state["__NAV_SELECTED__"] = "4"
            st.rerun()
    with change_col:
        if st.button("Changer les objectifs", use_container_width=True):
            st.session_state["__DG_FORCE_RERUN__"] = True
            st.session_state["pipeline_ready_to_run"] = False
            st.session_state["pipeline_executed"] = False
            st.session_state["pipeline_status"] = None
            st.session_state["pipeline_halt"] = None
            st.session_state["final_report_ready"] = False
            st.session_state["final_export_zip_bytes"] = None
            st.session_state["etape2_terminee"] = False
            st.session_state["etape40_terminee"] = False
            st.session_state["etape41_terminee"] = False
            st.session_state["__NAV_SELECTED__"] = "2"
            try:
                st.query_params["step"] = "2"
            except Exception:
                st.experimental_set_query_params(step="2")
            st.rerun()
    with reset_col:
        if st.button("Réinitialiser", use_container_width=True):
            reset_app_state()

    st.session_state["etape40_terminee"] = True
    st.stop()
