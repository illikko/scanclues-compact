import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def val_or_default(val, default):
    """Retourne default si val est None, chaîne vide, ou pandas vide; sinon val."""
    if val is None:
        return default
    if isinstance(val, str):
        return val if val.strip() else default
    if isinstance(val, (pd.DataFrame, pd.Series)):
        return val if not val.empty else default
    # pour listes/dicts, optionnel : if hasattr(val, "__len__") and len(val)==0: return default
    return val

def to_text(x):
    """Sérialise pour le prompt."""
    if isinstance(x, pd.DataFrame):
        return x.to_csv(index=False)
    if isinstance(x, pd.Series):
        return x.to_csv(index=True)
    return str(x)


def summarize_sankey_pairs(results_store, max_items: int = 20):
    """Resume sankey_pair_results pour payload LLM (compact)."""
    if not isinstance(results_store, dict) or not results_store:
        return []

    rows = []
    for pair_id, r in results_store.items():
        interp = str(r.get("interpretation", "") or "").strip()
        if len(interp) > 500:
            interp = interp[:500] + "..."
        rows.append(
            {
                "pair_id": pair_id,
                "var_x": r.get("var_x"),
                "var_y": r.get("var_y"),
                "v": r.get("v"),
                "p": r.get("p"),
                "chi2": r.get("chi2"),
                "interpretation": interp,
            }
        )

    rows.sort(key=lambda x: float(x.get("v") or 0), reverse=True)
    return rows[:max_items]

def run():
    # États init
    st.session_state.setdefault("profil_dominant", None)
    st.session_state.setdefault("profils", None)
    st.session_state.setdefault("interpretationACM", None)

    # Récupération des données (sans 'or' ambigu)
    dataset_object = st.session_state.get('dataset_object')
    dataset_context = st.session_state.get('dataset_context')
    dataset_recommendations = st.session_state.get('dataset_recommendations')
    key_questions_answer = st.session_state.get('key_questions_answer')
    target_profiles_text = st.session_state.get('target_profiles_text')
    profil_dominant_analysis = st.session_state.get('profil_dominant_analysis')
    dominant_continues = st.session_state.get('dominant_continues')
    dominant_discretes = st.session_state.get('dominant_discretes')
    interpretationACM = st.session_state.get('interpretationACM')
    dendrogram_interpretation = st.session_state.get('dendrogram_interpretation')
    latent_summary_text = st.session_state.get('latent_summary_text')
    fig_dendro = st.session_state.get('dendrogram')
    segmentation_profiles_text = st.session_state.get('segmentation_profiles_text')
    target_profiles_table = st.session_state.get('target_profiles_table')
    segmentation_profiles_table = st.session_state.get('segmentation_profiles_table')
    segmentation_detailed_profiles = st.session_state.get('segmentation_detailed_profiles')
    ctas_rules_text = st.session_state.get('ctas_rules_text')
    process = st.session_state.get('process')
    figs_variables_distribution = st.session_state.get("figs_variables_distribution", [])
    dataset_characteristics  = st.session_state.get("dataset_characteristics")
    variables_raw = st.session_state.get("variables_raw")
    data_preparation_synthesis = st.session_state.get("data_preparation_synthesis")    
    fig_missing_percentages = st.session_state.get("fig_missing_percentages")
    fig_missing_correlation_heatmap = st.session_state.get("fig_missing_correlation_heatmap")
    fig_missing_correlation_dendrogram = st.session_state.get("fig_missing_correlation_dendrogram")
    little_test_result = st.session_state.get("little_test_result")
    df_ready = st.session_state.get("df_ready")
    sankey_interpretation_synthesis = st.session_state.get("sankey_interpretation_synthesis")
    crosstabs_interpretation = st.session_state.get("crosstabs_interpretation")
    sankey_pair_results = st.session_state.get("sankey_pair_results", {})
    sankey_latents = st.session_state.get("sankey_latents")
    global_synthesis = st.session_state.get("global_synthesis")
 
    st.header("Q&A")

    # Saisie
    question = st.text_input("Posez une question sur un aspect spécifique posé par le jeu de données :")

    if st.button("Envoyer"):
        if not question.strip():
            st.warning("Veuillez poser une question.")
            return

        try:
            with st.spinner("Analyse de la question par LLM en cours..."):
                
                sys_ctx_6 = f'''Vous êtes un expert en analyse de données, en marketing s'il s'agit de questionnaires. Réponds  en français, clair et concis.
                        Une analyse détaillée a été réalisée sur un jeu de données.
                        Les résultats vous sont fournis dans plusieurs documents :
                        profil dominant, personas de la segmentation, et interprétation de l'ACM,...
                        Une question vous a été posée: {question}                    
                        Commencez par répondre en vous limitant au jeu de données , en premier et c'est la priorité, de celles qui s'appuient sur des données externes. Au besoin séparez en 2 parties distinctes.
                '''

                preview = df_ready.head(10).to_csv(index=False)
                preview = preview[:20000]  # limiter la taille

                payload = {
                    "columns": [str(c) for c in df_ready.columns],
                    "data_sample_preview_as_csv": preview,
                }

                context_blob5 = {
                    **payload,
                    "dataset_object": dataset_object,
                    "dataset_context": dataset_context,
                    "dataset_recommendations": dataset_recommendations,
                    "target_profiles_text": target_profiles_text,
                    "profil_dominant_analysis": profil_dominant_analysis,
                    "interpretationACM": interpretationACM,
                    "latent_summary_text": latent_summary_text,
                    "segmentation_profiles_text": segmentation_profiles_text,
                    "key_questions_answer": key_questions_answer,
                    "data_preparation_synthesis": data_preparation_synthesis,
                    "dendrogram_interpretation" : dendrogram_interpretation,
                    "ctas_rules_text": ctas_rules_text,
                    "global_synthesis": global_synthesis,
                    "sankey_interpretation_synthesis": sankey_interpretation_synthesis,
                    "crosstabs_interpretation": crosstabs_interpretation,
                    "sankey_pair_results_summary": summarize_sankey_pairs(sankey_pair_results, max_items=20),
                    "sankey_latents_csv": to_text(sankey_latents) if isinstance(sankey_latents, pd.DataFrame) else "",
                }

                user_content = json.dumps(context_blob5, ensure_ascii=False, default=str)

                r6 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": sys_ctx_6},
                        {"role": "user", "content": user_content},
                    ],
                )

                answer = r6.choices[0].message.content
                st.text_area("Réponse :", answer, height=300)

        except Exception as e:
            st.error(f"Erreur d'appel à l'API OpenAI : {e}")
