import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from openai import OpenAI
import os

# pour sauter en mode automatique
MODE_KEY = "__NAV_MODE__"

# ppour l'appel OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    pipeline_silent = bool(st.session_state.get("__PIPELINE_SILENT__", False))
    display_enabled = not pipeline_silent
    generate_distribution_figures = bool(
        st.session_state.get("generate_distribution_figures", not pipeline_silent)
    )

    # Éviter les doublons en mode pipeline silencieux (DistributionsDetail produit déjà les figures)
    st.session_state.setdefault("figs_variables_distribution", [])
    if not generate_distribution_figures:
        st.session_state["figs_variables_distribution"] = st.session_state.get("figs_variables_distribution", [])

    if display_enabled:
        st.header("Analyse de la distribution des variables")

    # Liste pour construire le tableau récapitulatif des modalités dominantes
    dominant_continues = []
    dominant_discretes = []

    # initialisation des états
    if "etape14_terminee" not in st.session_state:
        st.session_state["etape14_terminee"] = None

    st.session_state.setdefault("figs_variables_distribution", [])
    if not generate_distribution_figures:
        st.session_state["figs_variables_distribution"] = []

    # Charger le dataset
    if "df_ready" in st.session_state:
        df = st.session_state.df_ready
        
    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application Préparation 1.")
        # >>> indispensable pour éviter NameError sur df plus bas
        return
    
    # Traduction des textes
    T1 = "Fréquence (%)"
     
    # Boucle sur les variables

    for variable in df.columns:
        # test sur la présence de valeurs NAs sur le dataset préparé
        valeurs_uniques = df[variable].nunique(dropna=True)
        est_numerique = pd.api.types.is_numeric_dtype(df[variable])

        # -------- VARIABLES CONTINUES --------
        if est_numerique and valeurs_uniques > 10:
            series = df[variable].dropna()

            # 1/ Gestion des bins
            if valeurs_uniques < 30:
                # test : toutes les valeurs sont-elles entières ?
                est_entiere = np.all(np.equal(series, series.astype(int)))
                if est_entiere:
                    val_min = int(series.min())
                    val_max = int(series.max())
                    # nombre de valeurs consécutives entre min et max
                    bins = max(1, val_max - val_min + 1)
                else:
                    # moins de 50 valeurs distinctes non entières => 1 bin par valeur distincte
                    bins = int(valeurs_uniques)
            else:
                # beaucoup de valeurs distinctes => 50 bins
                bins = 50

            if generate_distribution_figures:
                fig, ax = plt.subplots()
                ax.hist(series, bins=bins, edgecolor='black')
                ax.set_title(f"{variable}")
                ax.set_ylabel("Occurrences")
                fig.tight_layout()
                st.pyplot(fig)
                
                # --- sauvegarde PNG pour le rapport
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                st.session_state.figs_variables_distribution.append({
                    "title": f"{variable}",
                    "png": buf.getvalue(),
                })
                plt.close(fig)

            # 2/ Ajout de la médiane dans dominant_continues
            mediane = series.median()
            dominant_continues.append({
                "Variable": variable,
                "Médiane": mediane,
            })
        
        # -------- VARIABLES DISCRÈTES --------
        else:
            # fréquences (%) triées DÉCROISSANT
            counts = (
                df[variable]
                .value_counts(normalize=True, dropna=True)
                .mul(100)
                .sort_values(ascending=False)
            )

            # si aucune modalité (variable vide)
            if counts.empty:
                continue

            nb_modalites = len(counts)

            # positions verticales : 0, 1, 2, ...
            y_pos = np.arange(nb_modalites)

            # -----------------------
            # PARAMÈTRES DE FIGURE
            # -----------------------
            fig_width = 6.0                    # largeur FIXE (important)
            bar_height = 0.6                   # épaisseur des barres
            height_per_bar = 0.45              # hauteur par barre
            fig_height = max(1.5, nb_modalites * height_per_bar)

            if generate_distribution_figures:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))

                # -----------------------
                ax.barh(y_pos, counts.values, height=bar_height)
                ax.invert_yaxis()  # le plus frequent en haut

                ax.set_title(f"{variable}")
                ax.set_xlabel(T1)

                # -----------------------
                # LABELS
                # -----------------------
                labels = counts.index.astype(str)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)

                # marge verticale
                ax.set_ylim(-1, nb_modalites)

                # axe X ajuste
                max_freq = counts.values.max()
                ax.set_xlim(0, max_freq * 1.15)

                # -----------------------
                # Gestion de la marge gauche (si libelles longs)
                # -----------------------
                max_label_len = max(len(s) for s in labels)
                left_margin = 0.25 + min(0.25, (max_label_len - 12) * 0.015)
                left_margin = max(0.25, min(0.50, left_margin))

                fig.subplots_adjust(left=left_margin, right=0.95)

                # -----------------------
                # Annotations des frequences
                # -----------------------
                for i, freq in enumerate(counts.values):
                    ax.text(freq + max_freq * 0.02, i, f"{freq:.1f}%", va='center')

                fig.tight_layout()
                st.pyplot(fig)

                # -----------------------
                # Sauvegarde PNG pour rapport
                # -----------------------
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                st.session_state.figs_variables_distribution.append({
                    "title": f"{variable}",
                    "png": buf.getvalue(),
                })
                plt.close(fig)

            # -----------------------
            # Résumé des modalités principales (cumul >= 50 %)
            # -----------------------
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

    # --- mémorisation et affichage---
    if dominant_continues:
        st.session_state.dominant_continues = pd.DataFrame(dominant_continues).sort_values(by="Médiane", ascending=False)
        st.subheader("Profil dominant - variables continues (médianes)")
        st.dataframe(st.session_state.dominant_continues, use_container_width=True)

    if dominant_discretes:
        st.session_state.dominant_discretes = pd.DataFrame(dominant_discretes).sort_values(by="Fréquence cumulée (%)", ascending=False)
        st.subheader("Profil dominant - variables discrètes (modes)")
        st.dataframe(st.session_state.dominant_discretes, use_container_width=True)

    else:
        st.info("Aucun résumé de variables disponible.")
    
    
    # génération du commentaire par LLM")
    st.subheader("Profil dominant")
    
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Rédaction du profil dominant par LLM"):
            proceed = True

    if proceed:
        
        # règles pour déterminer si les écarts pour des variables binaires théoriquement équilibrées sont significatifs
        if df.shape[0] <= 100:
            ecart_acceptable = 10  # points de pourcentage
        elif df.shape[0] <= 200:
            ecart_acceptable = 7
        elif df.shape[0] <= 400:
            ecart_acceptable = 5
        elif df.shape[0] <= 600:
            ecart_acceptable = 4
        else:
            ecart_acceptable = 3
 
        try:
            with st.spinner("Rédaction du profil dominant par LLM en cours..."):
                # récupération des données
                df_cont = st.session_state.get("dominant_continues", pd.DataFrame())
                df_disc = st.session_state.get("dominant_discretes", pd.DataFrame())
                dataset_context = st.session_state.get("dataset_context", "Pas de contexte fourni.")
                dataset_characteristics = st.session_state.get("dataset_characteristics", "Pas de caractéristiques du dataset fournies.")
                    
                system_msg = {
                    "role": "system",
                    "content": f'''
                        Vous êtes un expert en analyse de données. Réponds en français, clair et concis.
                        Un jeu de données a été analysé, dont le contexte est précisé dans cette description : {dataset_context}. 
                        Ainsi que ses caractéristiques : {dataset_characteristics}.
                        Un profil sur l'unité d'observation (1 personne s'il s'agit des opinions d'une enquête, référez-vous à l'unité d'observation identifiée dans dataset_context) a été généré avec le mode (la fréquence la plus élevée) pour chaque variable.
                        Répondez en 2 sections:
                        1- portrait dominant: Faites le portrait du profil dominant avec des phrases (pas de bullet point). Pour les variables catégorielles, indiquez quand vous la/les citez entre parenthèses la fréquence de la/les modalités dominantes.
                        2- représentativité de l'échantillon : seulement si il y a des variables sociodémographiques (genre=homme/femme, tranches d'âge, CSP+ versus CSP-), évaluez les écarts entre les fréquences des catégories, évaluez si il y a un déséquilibre (plus de femmes que d'hommes, plus de > 60 ans que de 15-25,...), plus de CPS + alors que dans la population il y en a moins que de CSP-,...
                        Si il y a des écarts > 3%, décrivez ces écarts.
                        Si un écart est supérieur sur un attribut à {ecart_acceptable} points de pourcentage, concluez que l'échantillon présente un risque de non représentativité sur cet attribut.
                        Dans les autres cas, dites: "Il n'y a pas d'éléments pour évaluer la représentativité de l'échantillon.
                    '''
                }

                user_msg = {
                    "role": "user",
                    "content": (
                        "Voici les donnéesâ€¦.(format CSV):\n"
                        f"{df_cont.to_csv(index=False)}\n"
                        f"{df_disc.to_csv(index=False)}"
                    )
                }

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[system_msg, user_msg],
                    temperature=0,
                    max_tokens=4000
                )

                profil_dominant_analysis = response.choices[0].message.content
                st.session_state.profil_dominant_analysis = profil_dominant_analysis            
            
        except Exception as e:
            st.error(f"Erreur d'appel à l'API OpenAI : {e}")

    st.text_area("Profils associés aux segments :", st.session_state.profil_dominant_analysis, height=300)

    st.write("Vous pouvez lancer la prochaine étape dans le menu à gauche: Analyse des corrélations.")
    st.session_state["etape14_terminee"] = True




