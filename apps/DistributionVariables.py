import os
from contextlib import nullcontext

import pandas as pd
import streamlit as st
from openai import OpenAI


MODE_KEY = "__NAV_MODE__"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def run():
    mode = (
        "automatique"
        if st.session_state.get("__PIPELINE_FORCE_AUTO__", False)
        else st.session_state.get(MODE_KEY, "automatique")
    )
    pipeline_silent = bool(st.session_state.get("__PIPELINE_SILENT__", False))
    qa_silent = bool(st.session_state.get("__QA_SILENT__", False))
    render_ui = not pipeline_silent and not qa_silent

    # Les histogrammes détaillés sont exclusivement produits par DistributionsDetail.
    st.session_state["figs_variables_distribution"] = []

    if render_ui:
        st.header("Analyse de la distribution des variables")

    dominant_continues = []
    dominant_discretes = []
    st.session_state.setdefault("etape14_terminee", None)

    df = st.session_state.get("__QA_SEGMENT_DF__")
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = st.session_state.get("df_ready")
    output_key = str(st.session_state.get("__QA_PROFILE_OUTPUT_KEY__") or "profil_dominant_analysis")
    if not isinstance(df, pd.DataFrame) or df.empty:
        if render_ui:
            st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application Préparation 1.")
        return

    for variable in df.columns:
        valeurs_uniques = df[variable].nunique(dropna=True)
        est_numerique = pd.api.types.is_numeric_dtype(df[variable])

        if est_numerique and valeurs_uniques > 10:
            series = df[variable].dropna()
            if series.empty:
                continue
            dominant_continues.append({
                "Variable": variable,
                "Médiane": series.median(),
            })
            continue

        counts = (
            df[variable]
            .value_counts(normalize=True, dropna=True)
            .mul(100)
            .sort_values(ascending=False)
        )
        if counts.empty:
            continue

        cumul = counts.cumsum()
        idx_seuil = (cumul >= 50).idxmax()
        pos_seuil = cumul.index.get_loc(idx_seuil)
        modalites_principales = counts.iloc[:pos_seuil + 1]
        freq_cumulee = float(cumul.iloc[pos_seuil])

        dominant_discretes.append({
            "Variable": variable,
            "Modalités principales": ", ".join(map(str, modalites_principales.index)),
            "Fréquence cumulée (%)": round(freq_cumulee, 1),
        })

    if dominant_continues:
        st.session_state["dominant_continues"] = (
            pd.DataFrame(dominant_continues).sort_values(by="Médiane", ascending=False)
        )
        if render_ui:
            st.subheader("Profil dominant - variables continues (médianes)")
            st.dataframe(st.session_state["dominant_continues"], use_container_width=True)
    else:
        st.session_state["dominant_continues"] = pd.DataFrame(columns=["Variable", "Médiane"])

    if dominant_discretes:
        st.session_state["dominant_discretes"] = (
            pd.DataFrame(dominant_discretes).sort_values(by="Fréquence cumulée (%)", ascending=False)
        )
        if render_ui:
            st.subheader("Profil dominant - variables discrètes (modes)")
            st.dataframe(st.session_state["dominant_discretes"], use_container_width=True)
    else:
        st.session_state["dominant_discretes"] = pd.DataFrame(
            columns=["Variable", "Modalités principales", "Fréquence cumulée (%)"]
        )
        if render_ui:
            st.info("Aucun résumé de variables disponible.")

    if render_ui:
        st.subheader("Profil dominant")

    proceed = mode == "automatique"
    if not proceed and render_ui and st.button("Rédaction du profil dominant par LLM"):
        proceed = True

    if proceed:
        if df.shape[0] <= 100:
            ecart_acceptable = 10
        elif df.shape[0] <= 200:
            ecart_acceptable = 7
        elif df.shape[0] <= 400:
            ecart_acceptable = 5
        elif df.shape[0] <= 600:
            ecart_acceptable = 4
        else:
            ecart_acceptable = 3

        try:
            spinner_context = (
                st.spinner("Rédaction du profil dominant par LLM en cours...")
                if render_ui
                else nullcontext()
            )
            with spinner_context:
                df_cont = st.session_state.get("dominant_continues", pd.DataFrame())
                df_disc = st.session_state.get("dominant_discretes", pd.DataFrame())
                dataset_context = st.session_state.get("dataset_context", "Pas de contexte fourni.")
                dataset_characteristics = st.session_state.get(
                    "dataset_characteristics",
                    "Pas de caractéristiques du dataset fournies.",
                )

                system_msg = {
                    "role": "system",
                    "content": f"""
                        Vous êtes un expert en analyse de données. Réponds en français, clair et concis.
                        Un jeu de données a été analysé, dont le contexte est précisé dans cette description : {dataset_context}.
                        Ainsi que ses caractéristiques : {dataset_characteristics}.
                        Un profil sur l'unité d'observation (1 personne s'il s'agit des opinions d'une enquête, référez-vous à l'unité d'observation identifiée dans dataset_context) a été généré avec le mode (la fréquence la plus élevée) pour chaque variable.
                        Répondez en 2 sections:
                        1- portrait dominant: Faites le portrait du profil dominant avec des phrases (pas de bullet point). Pour les variables catégorielles, indiquez quand vous la/les citez entre parenthèses la fréquence de la/les modalités dominantes.
                        2- représentativité de l'échantillon : seulement si il y a des variables sociodémographiques (genre=homme/femme, tranches d'âge, CSP+ versus CSP-), évaluez les écarts entre les fréquences des catégories, évaluez si il y a un déséquilibre (plus de femmes que d'hommes, plus de > 60 ans que de 15-25,...), plus de CPS + alors que dans la population il y en a moins que de CSP-,...
                        Si il y a des écarts > 3%, décrivez ces écarts.
                        Si un écart est supérieur sur un attribut à {ecart_acceptable} points de pourcentage, concluez que l'échantillon présente un risque de non représentativité sur cet attribut.
                        Dans les autres cas, dites: "Il n'y a pas d'éléments pour évaluer la représentativité de l'échantillon."
                    """,
                }

                user_msg = {
                    "role": "user",
                    "content": (
                        "Voici les données (format CSV):\n"
                        f"{df_cont.to_csv(index=False)}\n"
                        f"{df_disc.to_csv(index=False)}"
                    ),
                }

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[system_msg, user_msg],
                    temperature=0,
                    max_tokens=4000,
                )

                st.session_state[output_key] = response.choices[0].message.content
        except Exception as e:
            if render_ui:
                st.error(f"Erreur d'appel à l'API OpenAI : {e}")

    if render_ui:
        st.text_area(
            "Profils associés aux segments :",
            st.session_state.get(output_key, ""),
            height=300,
        )
        st.write("Vous pouvez lancer la prochaine étape dans le menu à gauche: Analyse des corrélations.")

    st.session_state["etape14_terminee"] = True
