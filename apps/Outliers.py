import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from core.df_registry import DFState, get_df, set_df
from utils import preparation_process

MODE_KEY = "__NAV_MODE__"

def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    
    st.header("Préparation du dataset (4/4)")
    st.subheader("Détection de valeurs anormales et extrêmes")
    st.write("Détection d'anomalies avec le modèle Isolation Forest")

    # --- Init session state ---
    st.session_state.setdefault("etape13_terminee", False)
    st.session_state.setdefault("df_ex_corr", None)   
    st.session_state.setdefault("df_clean", pd.DataFrame())    
    st.session_state.setdefault("outliers_removed", False)
    st.session_state.setdefault("outliers_decision", None)
    st.session_state.setdefault("outliers_indices", [])
    st.session_state.setdefault("df_outliers_sorted", pd.DataFrame())
    st.session_state.setdefault("outliers_detected", False)
    st.session_state.setdefault("decision_applied_once", False)

    # --- Charger le dataset source ---
    df_ex_corr = get_df(DFState.EX_CORR)
    df_clean = get_df(DFState.CLEAN, default=pd.DataFrame())

    if isinstance(df_ex_corr, pd.DataFrame) and not df_ex_corr.empty:
        df = df_ex_corr
        st.success("Dataset chargé depuis l'application précédente.")
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())

    elif isinstance(df_clean, pd.DataFrame) and not df_clean.empty:
        df = df_clean
        st.success("Dataset chargé depuis l'application précédente.")
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())

    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()


    # ===============================
    # 1) FORMULAIRE : paramètres + détection
    # ===============================
    st.markdown("##### Sélection des paramètres de détection des outliers")
    
    outliers_percent_target = st.slider(
        "Pourcentage d'outliers (contamination)",
        min_value=0.0, 
        max_value=20.0, 
        value=1.0, 
        step=0.1,
        help="Choisissez la contamination (en %)"
    )
    
    proceed = False

    if mode == "automatique":
        # Aucun bouton n'est affiché en auto
        proceed = True
    else:
        # En manuel: on affiche le bouton et on procède uniquement s'il est cliqué
        if st.button("Appliquer les choix sélectionnés"):
            proceed = True

    if proceed:
        # Calculs déclenchés uniquement ici
        dfd = pd.get_dummies(df, dtype=int)
        fr = outliers_percent_target / 100
        model = IsolationForest(contamination=fr, random_state=42)
        model.fit(dfd)
        pred = model.predict(dfd)

        result = pd.DataFrame({'Predictions': pred}, index=dfd.index)
        outliers = result[result["Predictions"] == -1]
        outliers_indices = outliers.index.tolist()

        scores = model.decision_function(dfd)
        df_scores = pd.DataFrame({'score': scores}, index=df.index)

        df_outliers = df.loc[outliers_indices].copy()
        df_outliers["score"] = df_scores.loc[outliers_indices]["score"]
        df_outliers_sorted = df_outliers.sort_values(by="score")

        # Mémoriser en session
        st.session_state.outliers_indices = outliers_indices
        st.session_state.df_outliers_sorted = df_outliers_sorted
        st.session_state.outliers_detected = True
        # Réinitialiser la décision à chaque nouvelle détection
        st.session_state.outliers_decision = None
        st.session_state.decision_applied_once = False

    # Affichage des résultats si une détection a été faite
    if st.session_state.outliers_detected:
        st.subheader("Outliers identifiés :")
        st.write(
            f"Nombre d'outliers détectés : {len(st.session_state.outliers_indices)}. "
            "Un score d'anomalie a été calculé pour chaque observation (colonne 'score')."
        )
        st.dataframe(st.session_state.df_outliers_sorted.head(10))

        # ===============================
        # 2) FORMULAIRE : décision appliquer / conserver
        # ===============================
        proceed2 = False
        if mode == "automatique":
            proceed2 = True
        else:
            if st.button("Supprimer ces outliers"):
                proceed2 = True

        # Valeurs par défaut pour éviter UnboundLocalError
        df_ready = df.copy()
        action = ""

        if proceed2:
            # Supprimer les outliers
            df_ready = df.drop(index=st.session_state.outliers_indices)
            set_df(DFState.READY, df_ready, step_name="Outliers")
            st.session_state.outliers_removed = True

            st.subheader("Dataset après traitement des outliers :")
            st.write("Le dataset est prêt pour les analyses.")
            st.dataframe(df_ready)

            nb_outliers = len(st.session_state.outliers_indices)
            action = f"{nb_outliers} valeurs anormales/extrêmes supprimées avec le modèle Isolation Forest."
            st.success(action)

            csv = df_ready.to_csv(sep=';', index=False, encoding='latin-1')
            st.download_button(
                "Télécharger le dataset préparé",
                data=csv,
                file_name="df_ready.csv",
                mime="text/csv",
                key="download_prepared_df"
            )
        else:
            # On ne touche pas au dataset
            st.info("Aucune modification n'a été appliquée au dataset (outliers conservés).")
            st.session_state.outliers_removed = False
            set_df(DFState.READY, df_ready, step_name="Outliers")  # garde la cohérence
            action = "outliers conservés."

        # Mettre à jour le tableau de process
        preparation_process(df_ready, action)
        st.markdown("##### État d'avancement de la préparation du dataset :")
        process = st.session_state.process
        st.dataframe(st.session_state.process)

        # déterminer la dimension du dataset après tous les traitements de préparation
        n_obs_ready, n_var_ready = df_ready.shape
        dimensions_dataset_ready = st.write(f"Après préparation il comporte {n_obs_ready} observations et {n_var_ready} variables. Le détail des traitements (données manquantes, variables non informatives,...) est précisé plus bas.")

        st.write("Vous pouvez lancer la prochaine étape dans le menu à gauche : Analyse des corrélations.")
        st.session_state["etape13_terminee"] = True




