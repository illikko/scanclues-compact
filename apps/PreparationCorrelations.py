import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from utils import preparation_process

MODE_KEY = "__NAV_MODE__"


# FONCTIONS UTILITAIRES    
# fonction pour faire des passes successives sur les corrélation masquées
def recursive_correlation_filter(df, threshold=0.9, continuous=None, discrete=None):
    df_filtered = df.copy()
    removed_vars = []

    while True:
        corr_matrix = mixed_correlation_matrix(df_filtered.copy(), continuous, discrete)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        found = False
        for col1 in upper_triangle.columns:
            for col2 in upper_triangle.columns:
                coef = upper_triangle.loc[col1, col2]
                if pd.notnull(coef) and abs(coef) >= threshold:
                    # On supprime arbitrairement col2
                    df_filtered = df_filtered.drop(columns=[col2])
                    removed_vars.append(col2)
                    found = True
                    break
            if found:
                break  # Redémarrer avec la nouvelle version du DataFrame

        if not found:
            break

    return df_filtered, removed_vars

# Fonctions de calcul des corrélations mixtes
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    
    for i in range(0,cat_num):
        cat_measures = measurements.iloc[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
        
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, (y_avg_array-y_total_avg)**2))
    denominator = np.sum((measurements-y_total_avg)**2)
    
    eta = 0.0 if denominator == 0 else np.sqrt(numerator/denominator)
    return eta

def mixed_correlation_matrix(df, continuous, discrete):
    le = LabelEncoder()
    for col in discrete:
        df[col] = le.fit_transform(df[col])
    corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
            elif col1 in continuous and col2 in continuous:
                corr_matrix.loc[col1, col2] = df[col1].corr(df[col2])
            elif col1 in discrete and col2 in discrete:
                corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
            else:
                if col1 in continuous:
                    corr_matrix.loc[col1, col2] = correlation_ratio(df[col2], df[col1])
                else:
                    corr_matrix.loc[col1, col2] = correlation_ratio(df[col1], df[col2])
    return corr_matrix.astype(float)


def run():
    # déclaration des variables
    if "etape12_terminee" not in st.session_state:
        st.session_state["etape12_terminee"] = False

    if 'correlation_data_filtered' not in st.session_state:
        st.session_state.correlation_data_filtered = []

    if 'df_ex_corr' not in st.session_state:
        st.session_state.df_ex_corr = None

    if 'to_remove_final' not in st.session_state:
        st.session_state.to_remove_final = None

    st.header("Préparation du dataset (3/4)")
    st.subheader("Détection et suppression de variables trop corrélées")

    # 1. Rechargement du dataset
    if "df_clean" in st.session_state:
        df = st.session_state.df_clean
    
        st.success("Dataset chargé depuis l'application précédente")
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())
    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")


    # 2. préparation des données:    
    continuous = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    discrete = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Calcul unique des corrélations
    with st.spinner("Calcul des corrélations..."):
        corr_matrix = mixed_correlation_matrix(df.copy(), continuous, discrete)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        correlation_data = []
        columns = upper_triangle.columns

        vars_70 = set()
        vars_80 = set()
        vars_90 = set()

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                var1 = columns[i]
                var2 = columns[j]
                coef = upper_triangle.loc[var1, var2]
                if pd.notnull(coef):
                    abs_coef = abs(coef)
                    if abs_coef >= 0.7:
                        correlation_data.append({"var1": var1, "var2": var2, "coef": coef})
                        vars_70.update([var1, var2])
                    if abs_coef >= 0.8:
                        vars_80.update([var1, var2])
                    if abs_coef >= 0.9:
                        vars_90.update([var1, var2])

        st.session_state.correlation_data = correlation_data

    # Statistiques préliminaires
    st.subheader("Statistiques de corrélation")
    st.markdown(f"""
    - Variables impliquées dans des corrélations â‰¥ 0.90 : **{len(vars_90)}**
    - Variables impliquées dans des corrélations â‰¥ 0.80 : **{len(vars_80)}**
    - Variables impliquées dans des corrélations â‰¥ 0.70 : **{len(vars_70)}**
    """)

    # Choix du seuil
    threshold = st.slider(
        label="Choisissez le seuil de corrélation",
        min_value=0.70,
        max_value=1.00,
        step=0.01,
        value=0.90,
        key="corr_threshold",
    )

    # Affichage des couples
    corr_rows = st.session_state.get("correlation_data", [])
    if not corr_rows:
        st.info("Aucune corrélation détectée.")

    filtered_data = [row for row in corr_rows if abs(row["coef"]) >= threshold]
    st.session_state["correlation_data_filtered"] = filtered_data  # garde en mémoire si dâ€™autres modules en ont besoin


    # Affichage des couples corrélés
    if not filtered_data:
        st.info("Aucune variable trop corrélée pour ce seuil.")
        df_ex_corr = df.copy()
        st.session_state.df_ex_corr = df_ex_corr
        st.session_state["etape12_terminee"] = True
        return
        
    else:
        st.subheader("Variables trop corrélées")

        target_variables = st.session_state.get("target_variables", [])
        target_set = set(target_variables) if target_variables else set()

        current_radio_keys = []

        for row in filtered_data:
            var1, var2, coef = row["var1"], row["var2"], row["coef"]
            key_radio = f"radio_corr__{var1}__{var2}"
            current_radio_keys.append(key_radio)

            options = []
            if var1 not in target_set:
                options.append(f"Supprimer {var1}")
            if var2 not in target_set:
                options.append(f"Supprimer {var2}")
            options.append("Ne rien supprimer")

            st.radio(
                f"{var1} / {var2} â€” corrélation {abs(coef):.2f}",
                options,
                key=key_radio,
            )

        st.session_state["current_radio_keys"] = current_radio_keys

    # Bouton d'application
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Appliquer les choix sélectionnés"):
            proceed = True

    if proceed:       
        selected_actions = []
        for k, v in st.session_state.items():
            if isinstance(k, str) and k.startswith("radio_corr__"):
                if isinstance(v, str) and v.startswith("Supprimer "):
                    selected_actions.append(v)

        # 2) Extraire les colonnes à supprimer
        to_remove = [s.split("Supprimer ", 1)[1].strip() for s in selected_actions]
        existing_columns = set(df.columns)
        to_remove_final = sorted({c for c in to_remove if c in existing_columns})
        missing = sorted(set(to_remove) - set(to_remove_final))
        if missing:
            st.warning(f"Colonnes non trouvées : {', '.join(missing)}")

        # 3) Appliquer (ou pas) et VALIDER dans tous les cas
        if to_remove_final:
            df_ex_corr = df.drop(columns=to_remove_final)
            action = (
                f"{len(to_remove_final)} variable(s) supprimée(s) "
                f"pour corrélation excessive : {', '.join(to_remove_final)}"
            )
        else:
            df_ex_corr = df.copy()
            action = "Aucune variable supprimée."

        st.session_state.df_ex_corr = df_ex_corr
        st.session_state.to_remove_final = to_remove_final
        st.session_state.action = action
        
    # 4) Mettre à jour l'état pour les étapes suivantes
    # MAJ des listes de variables illustratives + target
    df_ex_corr = st.session_state.df_ex_corr
    st.session_state["illustrative_variables"] = list(set(st.session_state.get("illustrative_variables", [])) & set(df_ex_corr.columns))
    st.session_state["target_variables"] = list(
        set(st.session_state.get("target_variables", [])) & set(df_ex_corr.columns)
    )
    st.write("listes des variables ex corrélation : ", df_ex_corr.columns.tolist())
    st.write("listes des illustratives : ", st.session_state["illustrative_variables"])

    st.subheader("âœ… Dataset après traitement")
    st.dataframe(df_ex_corr.head())
    csv = df_ex_corr.to_csv(index=False, sep=';', encoding='latin-1')
    st.download_button("Télécharger le dataset", csv, "df_ex_corr.csv", "text/csv")

    # 5) Mettre à jour le 'process' proprement (même s'il n'existait pas)
    preparation_process(df_ex_corr, st.session_state.action)

    st.markdown("##### État d'avancement de la préparation du dataset :")
    st.dataframe(st.session_state.process)

    # 6) Étape validée quoi qu'il arrive
    st.session_state["etape12_terminee"] = True
    st.success("Étape terminée âœ…. Vous pouvez lancer l'application suivante : Valeurs anormales.")

