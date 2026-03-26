import streamlit as st
import pandas as pd
import os
from core.df_registry import DFState, get_df, set_df

MODE_KEY = "__NAV_MODE__"

def _get_illustrative_list():
    """
    Récupère la liste des variables illustratives depuis différents emplacements possibles,
    pour être robuste à la façon dont elle a été stockée auparavant.
    """
    # Cas 1 : clé standard dans le session_state
    if isinstance(st.session_state.get("illustrative_variables"), (list, set, tuple)):
        return list(st.session_state["illustrative_variables"])

    # Cas 2 : clé alternative dans le session_state
    if isinstance(st.session_state.get("st.session_state_illustrative_variables"), (list, set, tuple)):
        return list(st.session_state["st.session_state_illustrative_variables"])

    # Cas 3 : attribut posé sur st (peu probable mais on couvre le cas mentionné)
    if hasattr(st, "session_state_illustrative_variables") and isinstance(getattr(st, "session_state_illustrative_variables"), (list, set, tuple)):
        return list(getattr(st, "session_state_illustrative_variables"))

    return []

def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    st.header("Séparation des variables actives/illustratives")

    # déclaration des variables
    if "etape16_terminee" not in st.session_state:
        st.session_state["etape16_terminee"] = None

    # Vérifier quels datasets sont disponibles
    datasets_disponibles = {}

    candidats = [
        (DFState.ENCODED, "Toutes les variables, ordinales encodées"),
        (DFState.READY, "Toutes les variables"),
    ]

    for state, label in candidats:
        val = get_df(state)
        if isinstance(val, pd.DataFrame) and not val.empty:
            datasets_disponibles[label] = val

    if not datasets_disponibles:
        st.warning("Aucun dataset *valide* trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()

    # Sélection du dataset
    choix = st.selectbox("Choisissez un dataset à utiliser :", list(datasets_disponibles.keys()), index=0)
    df = datasets_disponibles[choix].copy()

    st.success(f"{choix} chargé depuis l'application précédente.")
    st.write("Aperçu du dataset :")
    st.dataframe(df.head())

    st.write("Choisissez le statut de chaque variable")

    # Pré-remplissage à partir de la liste illustrative
    illustrative_list = set(_get_illustrative_list())
    options = ["Active", "Illustrative", "Ignorer"]

    # variable_status contiendra le choix courant pour chaque variable
    variable_status = {}

    # MODE AUTOMATIQUE : pas de formulaire, on “soumet” directement
    if mode == "automatique":
        for col in df.columns:
            # Valeur par défaut : "Illustrative" si dans la liste, sinon "Active"
            default_value = st.session_state.get(col, "Illustrative" if col in illustrative_list else "Active")
            default_index = options.index(default_value) if default_value in options else 0

            choice = st.radio(
                label=f"Variable : {col}",
                options=options,
                key=col,
                index=default_index
            )
            variable_status[col] = choice

        submitted = True  # auto-validation

    # MODE MANUEL : on garde le formulaire mais avec les valeurs pré-remplies
    else:
        with st.form("statut_form"):
            for col in df.columns:
                default_value = st.session_state.get(col, "Illustrative" if col in illustrative_list else "Active")
                default_index = options.index(default_value) if default_value in options else 0

                choice = st.radio(
                    label=f"Variable : {col}",
                    options=options,
                    horizontal=True,
                    key=col,
                    index=default_index
                )
                variable_status[col] = choice

            submitted = st.form_submit_button("Valider les statuts")

    if submitted:
        # --- Séparation des colonnes ---
        active_cols = [col for col, cat in variable_status.items() if cat == "Active"]
        illustrative_cols = [col for col, cat in variable_status.items() if cat == "Illustrative"]

        df_active = df[active_cols] if active_cols else pd.DataFrame(index=df.index)
        df_illustrative = df[illustrative_cols] if illustrative_cols else pd.DataFrame(index=df.index)

        csv_a = df_active.to_csv(index=False).encode("utf-8")
        csv_i = df_illustrative.to_csv(index=False).encode("utf-8")

        # --- Aperçu ---
        st.write("Aperçu des DataFrames générés")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Variables actives**")
            st.dataframe(df_active)
            st.download_button("Télécharger le dataset", csv_a, "df_active.csv", "text/csv")

        with col2:
            st.write("**Variables Illustratives**")
            st.dataframe(df_illustrative)
            st.download_button("Télécharger le dataset", csv_i, "df_illustrative.csv", "text/csv")

        # Stockage en session
        set_df(DFState.ACTIVE, df_active, step_name="SeparationVariables")
        set_df(DFState.ILLUSTRATIVE, df_illustrative, step_name="SeparationVariables")

        st.write("Vous pouvez lancer la prochaine étape dans le menu à gauche : Tris croisés.")
        st.session_state["etape16_terminee"] = True





