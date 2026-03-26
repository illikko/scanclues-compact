import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from openai import OpenAI
import os
import json, re


MODE_KEY = "__NAV_MODE__"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    # Initialisation des états
    if "etape30_terminee" not in st.session_state:
        st.session_state["etape30_terminee"] = False
    if 'bins_table' not in st.session_state:
        st.session_state.bins_table = None
    if 'clusters_df' not in st.session_state:
        st.session_state.clusters_df = None
    if 'segmentation_validated' not in st.session_state:
        st.session_state.segmentation_validated = False
    if 'characterization' not in st.session_state:
        st.session_state.characterization = None
    if 'segmentation_profiles_table' not in st.session_state:
        st.session_state.segmentation_profils_table = None
    if 'df_active' not in st.session_state:
        st.session_state.df_active = None
    if 'df_encoded' not in st.session_state:
        st.session_state.df_encoded = None
    if 'df_ready' not in st.session_state:
        st.session_state.df_ready = None
    if 'clusters_counts' not in st.session_state:
        st.session_state.clusters_counts = None       
    if 'clusters_frequencies' not in st.session_state:
        st.session_state.clusters_frequencies = None   
    if 'clusters_summary' not in st.session_state:
        st.session_state.clusters_summary = None 
    
    st.session_state.setdefault("segmentation_profiles_text", "")
    st.session_state.setdefault("profils_generated", False)

    st.title("Segmentation")
    st.subheader("Avec attribution de profils")
    
    url = "https://medium.com/@vincent.castaignet/a-comprensive-guide-for-analysing-rich-tabular-datasets-38e7c9fa9305"
    st.markdown(f"Pour une présentation de la segmentation, se référer à cet [article]({url}).")

    df_active = st.session_state.get("df_active")
    df_encoded = st.session_state.get("df_encoded")
    df_ready = st.session_state.get("df_ready")
    
    # Vérifier quels datasets sont disponibles
    datasets_disponibles = {}

    candidats = [
        ("df_ready",   "Toutes les variables"),
        ("df_active",  "Variables actives, ordinales encodées"),
        ("df_encoded", "Toutes les variables, ordinales encodées"),
    ]

    for key, label in candidats:
        val = st.session_state.get(key)
        if isinstance(val, pd.DataFrame) and not val.empty:
            datasets_disponibles[label] = val

    if not datasets_disponibles:
        st.warning("Aucun dataset *valide* trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()  # empêche la suite du script d'utiliser 'df' inexistant

    # Sélection et copie sûre
    choix = st.selectbox("Choisissez un dataset à utiliser :", list(datasets_disponibles.keys()), index=0)
    df = datasets_disponibles[choix].copy()

    st.success(f"{choix} chargé depuis l'application précédente.")
    st.write("Aperçu du dataset :")
    st.dataframe(df.head())


    # Étape 1 - Sélection des variables
    st.header("Sélection des variables")

    if "step3_validated" not in st.session_state:
        st.session_state.step3_validated = False

    selected_cols = st.multiselect(
        "Colonnes à garder",
        df.columns.tolist(),
        default=df.columns.tolist()
    )

    # IMPORTANT: on met à jour df en session dès maintenant
    st.session_state.df = df[selected_cols].copy()

    if st.button("Valider la sélection des variables"):
        st.session_state.step3_validated = True
        st.success("Sélection validée.")

    st.write("Variables sélectionnées (courantes) :")
    st.dataframe(st.session_state.df.head())

    # Étape 2 - Détermination du nombre de segments
    st.subheader("Détermination du nombre de segments optimal")
    st.write("Cette étape est optionnelle car elle peut prendre quelques minutes.")

    if st.button("Lancer lâ€™analyse de la méthode du coude"):
        with st.spinner("Calcul en cours..."):
            costs = []
            K = range(2, 11)

            for k in K:
                km = KModes(n_clusters=k, init='Huang', n_init=2, random_state=42, verbose=0)
                km.fit(df)  # `data` contient uniquement les colonnes catégorielles
                costs.append(km.cost_)

            fig, ax = plt.subplots()
            ax.plot(K, costs, marker='o')
            ax.set_xlabel('Nombre de clusters')
            ax.set_ylabel('Coût de dissimilarité')
            ax.set_title('Méthode du coude (K-Modes)')

            st.pyplot(fig)

            st.success("Analyse terminée. Identifiez le nb de clusters avec la méthode du coude")
            st.write("Valeurs du coût :")
            st.write(pd.DataFrame({"k": K, "coût": costs}))


    # Étape 3 - Segmentation
    st.subheader("Segmentation")

    df = st.session_state.df.copy()
    
    st.markdown("##### Paramètres de la segmentation")
    model_segmentation = st.selectbox("Modèles de clustering", ["Kmodes", "Kmeans", "CAH", "DBSCAN"], index=0)
    num_quantiles = st.slider("Nombre de quantiles pour la discrétisation", min_value=3, max_value=10, value=5)
    n_clusters = st.slider("Nombre de segments", min_value=2, max_value=20, value=10)
    n_init = st.slider("Nombre d'itérations du Kmode", min_value=2, max_value=10, value=5)
    high_freq_threshold = st.slider("Fréquence du mode à partir de laquelle la discrétisation est binaire", min_value=0.80, max_value=0.99, value=0.9, step=0.01)
    distinct_threshold_continuous = st.number_input("Seuil (nb de modalités distinctes) à partir duquel une variable numérique est continue", min_value=2, max_value=20, value=5, step=1)

    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Lancer la discrétisation"):
            proceed = True

    if proceed:
        # discrétisation
        num_cols = df.select_dtypes(include=['number'])
        distinct_counts = num_cols.nunique()
        continuous = distinct_counts[distinct_counts > distinct_threshold_continuous].index
        discrete = [col for col in df.columns if col not in continuous]

        # convertir les variables discrètes en str
        for var in discrete:
            if df[var].dtypes in ["int64", "float64"]:
                df[var] = df[var].astype(str)

        # traitement des variables continues à haute fréquence du mode
        cols_high_freq = []
        freq_rows = []

        for col in continuous:
            # Valeur dominante et fréquence (sur la colonne d'origine)
            mode_value = df[col].mode(dropna=True).iloc[0]
            mod_freq = df[col].value_counts(normalize=True, dropna=False).loc[mode_value]

            if mod_freq > high_freq_threshold:
                # Masque calculé AVANT conversion en string (sinon bug comparaison string vs numérique)
                mask_mode = df[col].eq(mode_value)

                # Gestion du cas où il n'y a aucune valeur "autre"
                if (~mask_mode).sum() == 0:
                    dominant_label = str(mode_value)
                    other_label = "others (none)"
                    df[col] = df[col].astype("string")
                    df[col] = dominant_label
                else:
                    non_mode = df.loc[~mask_mode, col]
                    min_value = non_mode.min()
                    max_value = non_mode.max()

                    dominant_label = str(mode_value)
                    other_label = f"{min_value} to {max_value}"

                    df[col] = df[col].astype("string")
                    df.loc[mask_mode, col] = dominant_label
                    df.loc[~mask_mode, col] = other_label

                cols_high_freq.append(col)

                # Fréquences des 2 modalités après transformation
                vc = df[col].value_counts(normalize=True, dropna=False)
                freq_rows.append({
                    "variable": col,
                    "modalité dominante": dominant_label,
                    "fréquence modalité dominante": float(vc.get(dominant_label, 0.0)),
                    "autres valeurs": other_label,
                    "fréquence autres valeurs": float(vc.get(other_label, 0.0)),
                })
        if freq_rows:
            freq_table = pd.DataFrame(freq_rows).sort_values("fréquence modalité dominante", ascending=False)
            st.subheader("Variables continues à haute fréquence du mode")
            st.dataframe(freq_table, use_container_width=True)

        # discretisation en bins
        continuous = [col for col in continuous if col not in cols_high_freq]
        bins_by_col = {}

        for col in continuous:
            q, bins = pd.qcut(df[col], num_quantiles, retbins=True, labels=False, duplicates='drop')
            bins = [round(b, 2) for b in bins]
            quantile_labels = [f"({bins[i]}, {bins[i+1]}]" for i in range(len(bins) - 1)]
            df[col] = q.map(dict(enumerate(quantile_labels)))
            bins_by_col[col] = quantile_labels
        
        
        # --- Construire le tableau récapitulatif d'affichage ---
        rows = []
        for col in continuous:
            s = df[col]

            row = {
                "Variable": col,
                "NA (n)": int(s.isna().sum()),
                "NA (%)": round(float(s.isna().mean() * 100), 1),
            }

            labels = bins_by_col.get(col, [])

            # tableau récapitulatif des intervalles de la discrétisation
            for i in range(num_quantiles):
                row[f"Bin_{i+1}"] = labels[i] if i < len(labels) else ""

            rows.append(row)

        bins_table = pd.DataFrame(rows)

        # ordre des colonnes
        bin_cols = [f"Bin_{i+1}" for i in range(num_quantiles)]
        cols = ["Variable", "NA (n)", "NA (%)"] + bin_cols
        bins_table = pd.DataFrame(rows)

        if bins_table.empty:
            st.info("Aucune variable continue à discrétiser (seulement des booléennes 0/1).")
            bins_table = pd.DataFrame(columns=cols)
        else:
            bins_table = bins_table[cols]
        
        st.session_state.bins_table = bins_table
        st.write("Aperçu du dataset discrétisé:")
        st.dataframe(df.head())
    
    if st.session_state.bins_table is not None:
        st.subheader("Intervalles (bins) générés par variable")
        st.dataframe(st.session_state.bins_table, use_container_width=True)


    # kmodes model
    if st.session_state.segmentation_validated == True:
        st.write("Segmentation déjà réalisée:")
        st.dataframe(st.session_state.segmentation_profiles_table)
                    
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Lancer la segmentation"):
            proceed = True

    if proceed:
        km = KModes(
            n_clusters= n_clusters,
            init='Huang',
            n_init=n_init,
            random_state=42
        )

        # 1) Fit + labels numériques
        clusters = km.fit_predict(df)
        df["Cluster_id"] = clusters

        # 2) Mapping numérique -> étiquette lisible G1, G2, ...
        label_map = {i: f"G{i+1}" for i in range(km.n_clusters)}

        # colonne lisible pour tous les affichages
        df["Cluster"] = df["Cluster_id"].map(label_map)

        # Sauvegarde dans la session
        st.session_state.clusters_df = df
        st.session_state.segmentation_validated = True
        st.session_state.label_map = label_map

        st.success("Segmentation réalisée.")

        # 3) Création du tableau des centroïdes
        # -> on enlève les colonnes ajoutées après le fit
        features_cols = [c for c in df.columns if c not in ("Cluster", "Cluster_id")]
        C = pd.DataFrame(
            km.cluster_centroids_,
            columns=features_cols
        )

        # Index des centroïdes en G1, G2, ...
        C.index = [label_map[i] for i in range(km.n_clusters)]
        C.index.name = "Cluster"

        # 4) Transposition
        C = C.T
        C.index.name = "Variable"

        # 5) Niveau de différenciation
        def count_distinct(row):
            return len(set(row))

        C["Differenciation"] = C.apply(lambda row: count_distinct(row), axis=1)
        C = C.sort_values(by="Differenciation", ascending=False)
        st.session_state.segmentation_profiles_table = C
        
        st.write("Voici les profils des groupes, avec les variables les plus discriminantes en premier :")
        st.dataframe(st.session_state.segmentation_profiles_table)

        # 6) Dataset avec groupes attribués
        st.write("Voici le dataset avec les groupes attribués :")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf8")
        st.download_button("Télécharger le dataset", csv, "df.csv", "text/csv")

        # 7) Comptage du nombre d'observations dans chaque groupe (en G1, G2, ...)
        # on force l'ordre G1, G2, ... même si un groupe vide
        all_labels = [label_map[i] for i in range(km.n_clusters)]
        clusters_counts = df["Cluster"].value_counts().reindex(all_labels, fill_value=0)

        clusters_frequencies = clusters_counts / df.shape[0] * 100
        clusters_frequencies = clusters_frequencies.map(lambda x: f"{x:.1f}%")

        st.session_state.clusters_counts = clusters_counts
        st.session_state.clusters_frequencies = clusters_frequencies
        
        clusters_summary = pd.DataFrame({
            "Groupe": clusters_counts.index,
            "Effectif": clusters_counts.values,
            "Fréquence": clusters_frequencies.values
        })
        st.session_state.clusters_summary = clusters_summary
        
    st.write("Effectifs et fréquence par cluster") 
    st.dataframe(st.session_state.clusters_summary)

    if not st.session_state.segmentation_validated:
        st.info("Lancez d’abord la segmentation.")

    # Étape 4 - Génération des profils détaillés
    st.subheader("Génération des profils détaillés")
    st.write(
        "Pour chaque segment, on calcule la fréquence de chaque attribut pour le segment "
        "et dans la population, puis une mesure de représentativité (TestValue, l'écart standardisé)."
    )

    # 1) Récupérer le dataset segmenté
    if "clusters_df" not in st.session_state or st.session_state.clusters_df is None:
        st.error("Dataset segmenté introuvable. Lancez la segmentation (étape 3) d'abord.")
        st.stop()

    df = st.session_state.clusters_df.copy()

    # 2) Fonction de caractérisation des groupes
    def group_characterization(data: pd.DataFrame, group_column: str, variables):
        results = []
        for var in variables:
            # fréquences globales (inclure NaN pour éviter de perdre la colonne si beaucoup de manquants)
            overall_freq = data[var].value_counts(normalize=True, dropna=False)
            # par groupe (group_column = "Cluster" -> G1, G2, ...)
            for group, gdf in data.groupby(group_column):
                n = len(gdf)
                if n == 0:
                    continue
                group_freq = gdf[var].value_counts(normalize=True, dropna=False)
                for category, p in overall_freq.items():
                    gp = float(group_freq.get(category, 0.0))
                    # éviter divisions par zéro
                    if p in (0.0, 1.0) or n == 0:
                        tv = 0.0
                    else:
                        tv = (gp - float(p)) / np.sqrt(p * (1 - p) / n)
                    results.append({
                        "Cluster": group,  # ici déjà G1, G2, ...
                        "Variable": f"{var}={category}",
                        "Group frequency": gp,
                        "Overall frequency": float(p),
                        "TestValue": float(tv),
                    })
        cols = ["Cluster", "Variable", "Group frequency", "Overall frequency", "TestValue"]
        return pd.DataFrame(results, columns=cols)

    # âš  exclure Cluster et Cluster_id des variables à caractériser
    variables_to_characterize = [c for c in df.columns if c not in ("Cluster", "Cluster_id")]
    charac = group_characterization(df, "Cluster", variables_to_characterize)

    # 3) (sécurité supplémentaire) s'assurer que les colonnes existent
    expected = ["Cluster", "Variable", "Group frequency", "Overall frequency", "TestValue"]
    for c in expected:
        if c not in charac.columns:
            if c in ("Cluster", "Variable"):
                charac[c] = pd.Series(dtype="object")
            else:
                charac[c] = pd.Series(dtype="float64")
    charac = charac[expected]

    # 4) Tri + filtre
    charac = charac.sort_values(["Cluster", "TestValue"], ascending=[True, False], ignore_index=True)
    charac = charac[charac["TestValue"] >= 2]
    
    if charac.empty:
        st.info("Aucune modalité avec TestValue â‰¥ 2 pour lâ€™instant.")

    st.session_state.segmentation_detailed_profiles = charac
    st.markdown("##### Caractérisation des groupes (par TestValue)")
    st.dataframe(st.session_state.segmentation_detailed_profiles)

    # 6) Export CSV
    csv2 = charac.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger la caractérisation", csv2, "segmentation_detailed_profiles.csv", "text/csv")

    # 7) Profils simplifiés pour lâ€™étape suivante
    threshold = 0.1
    profils_simples = charac.copy()
    profils_simples["Importance de l'attribut"] = np.where(
        profils_simples["Group frequency"] > threshold, "dominant", "rare"
    )
    profils_simples = profils_simples.drop(columns=["Group frequency", "Overall frequency", "TestValue"])
    st.session_state.profils_simples = profils_simples
    st.dataframe(profils_simples)


    # étape 5 - génération des profils
    st.subheader("Génération des profils")
    st.write("Rédaction des profils par LLM")

    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Lancer la génération des profils"):
            proceed = True
        
        if st.button("Effacer le texte généré"):
            st.session_state.profils_y_generated = False
            st.session_state.profils_y_text = ""
            st.rerun()

    if proceed:
        try:
            with st.spinner("Génération des profils par LLM en cours..."):
                profils_simples = st.session_state.profils_simples.copy()
                profils_simples_str = profils_simples.to_csv(index=False)
                dataset_context = st.session_state["dataset_context"]
                
                system_msg = {
                    "role": "system",
                    "content": f'''Vous êtes un expert en analyse de données, en marketing s'il s'agit de questionnaires. Réponds en français, clair et concis.
                        Un jeu de données a été analysé, dont le contexte est précisé dans cette description : {dataset_context}.
                        Une segmentation a été réalisée, et pour chaque segment, des attributs dominants et rares ont été identifiés.
                        Dans le fichier joint profils_simples_str, la 1ère colonne indique le numéro du groupe, la 2ème l'attribution (la modalité d'une variable catégorielle qui s'applique le plus au groupe), la 3ème l'importance pour le groupe de cet attribut (dominant ou rare).
                        Pour chaque segment, vous fournirez :
                        - Un nom de profil, 2â€“4 mots, cohérent avec l'unité d'observation du jeu de données (précisé dans dataset_context).
                        - Si l'unité d'observation correspond à une personne, le profil utilisera les variables d'opinion, comportement, et si présents les variables socio-démongraphiques (lâ€™âge, le genre...).
                        - Si l'unité d'observation n'est pas une personne (une entreprise, un animal, un processus...), cherchez les variables qui la décrit.                 
                        - Une justification en 4-6 phrases, évoquant uniquement les attributs dominants du segment. Vous compléterez éventuellement le profil avec les attributs 'rares', dans une phrase séparée.
                        - Veillez à nâ€™utiliser aucun prénom générique (évitez « Henry », etc.).
                        - vous finissez avec le nombre d'effectifs et la fréquence du clusters, fournis dans les documents joints (clusters_counts, clusters_frequencies).
                        Pas de tirets ou autre symbole de séparation entre les paragraphes.
                    '''
                }

                user_msg = {
                    "role": "user",
                    "content": (
                        "Voici les données à analyser (CSV) :\n"
                        f"{profils_simples_str}\n\n"
                        f"{st.session_state.clusters_counts}, {st.session_state.clusters_frequencies}"
                    )
                }

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[system_msg, user_msg],
                    temperature=0,
                    top_p=1,
                    n=1,
                    seed=42,
                    max_tokens=4000
                )

                segmentation_profiles_response = response.choices[0].message.content or ""
                st.session_state["segmentation_profiles_text"] = segmentation_profiles_response
                st.session_state["profils_generated"] = True
            
        except Exception as e:
            st.error(f"Erreur d'appel à l'API OpenAI : {e}")

    if st.session_state.segmentation_profiles_text:
        st.markdown(f"##### Profils associés aux segments")
        st.text_area("Profils", st.session_state.segmentation_profiles_text, height=350)
    

    st.write("Vous pouvez lancer la prochaine étape dans le menu à gauche: Profils associés à une cible.")
    st.session_state["etape30_terminee"] = True




