import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import chi2_contingency, chi2


def run():
    st.header("Diagnostic des valeurs manquantes")
    
    # déclaration des variables
    if "etape10_terminee" not in st.session_state:
        st.session_state["etape10_terminee"] = False
    
    if "missing_df" not in st.session_state:
        st.session_state["missing_df"] = None    

    def count_missing_values(df):
        def is_nan_string(x):
            if isinstance(x, str):
                return x.strip().upper() in ['NAN', 'NA', 'N/A', 'NONE', '']
            return False

        direct_missing = df.isnull().sum()
        indirect_missing = df.applymap(is_nan_string).sum()
        total_missing = direct_missing + indirect_missing
        total_missing_percent = total_missing / df.shape[0] * 100

        return pd.DataFrame({
            'Direct Missing': direct_missing,
            'Indirect Missing': indirect_missing,
            'Total Missing': total_missing,
            'Total Missing %': total_missing_percent
        })


    # 1- Rechargement du dataset
    if "df_imputed_structural" in st.session_state:
        df = st.session_state.df_imputed_structural

    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application Préparation 1.")
        st.stop()
    
    # 2- Analyse des valeurs manquantes
    st.subheader("Analyse des valeurs manquantes")
    count_missing_values_df = count_missing_values(df)
    st.session_state.missing_df = count_missing_values_df.sort_values('Total Missing', ascending=False)
    st.dataframe(st.session_state.missing_df)

    # Nettoyage initial
    df.replace(["NULL", "NAN"], np.nan, inplace=True)
    missing_percentages = df.isnull().mean().sort_values(ascending=False) * 100
    missing_percentages = missing_percentages[missing_percentages > 0]

    st.markdown("##### Pourcentage de valeurs manquantes par variable")
    if not missing_percentages.empty:
        fig, ax = plt.subplots(figsize=(12, max(4, len(missing_percentages) / 3)))
        missing_percentages.sort_values().plot(kind='barh', ax=ax)
        for i, v in enumerate(missing_percentages.sort_values()):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
        ax.set_title("Pourcentage de données manquantes par variable")
        ax.set_xlabel("Pourcentage")
        st.session_state["fig_missing_variables"] = fig
        st.pyplot(st.session_state["fig_missing_variables"])
    else:
        st.info("Aucune valeur manquante détectée.")
        st.session_state["etape10_terminee"] = True
        return

    # 3- Taux de valeurs manquantes par variable
    st.markdown("##### variables avec valeurs manquantes")
    
    # Heatmap des données manquantes
    if len(missing_percentages) == 1 :
        st.info("Il n'y a qu'une variable avec valeurs manquantes, donc pas de dépendance à évaluer.")
        st.session_state["etape10_terminee"] = True
        return
    
    else:
        st.subheader("Cartographie des dépendances")
        fig2, ax2 = plt.subplots(figsize=(10, len(missing_percentages) * 0.4))
        sns.heatmap(df[missing_percentages.index].isnull().T, cmap='YlOrRd', cbar=False, xticklabels=False, ax=ax2)
        ax2.set_yticklabels([f'{col} ({missing_percentages[col]:.1f}%)' for col in missing_percentages.index], rotation=0)
        ax2.set_title("Visualisation des dépendances entre les variables avec données manquantes")
        st.session_state["fig_missing_percentage_heatmap"] = fig2
        st.pyplot(st.session_state["fig_missing_percentage_heatmap"])

        # Visualisations Missingno
        st.markdown("##### Heatmap de corrélation des variables avec valeurs manquantes")
        fig3 = msno.heatmap(df[missing_percentages.index])
        st.session_state["fig_missing_correlation_heatmap"] = fig3
        st.pyplot(st.session_state["fig_missing_correlation_heatmap"].figure)

        st.markdown("##### Dendrogramme de dépendance entre variables avec valeurs manquantes")
        fig_ax = msno.dendrogram(df[missing_percentages.index])
        fig4 = fig_ax.get_figure()
        st.session_state["fig_missing_correlation_dendrogram"] = fig4
        st.pyplot(st.session_state["fig_missing_correlation_dendrogram"])

        # Test de Little MCAR
        st.markdown("##### Test de Little (MCAR)")
        def little_mcar_test(data):
            missing = data.isnull()
            chi_squares = []
            for i in range(data.shape[1]):
                for j in range(i+1, data.shape[1]):
                    obs = pd.crosstab(missing.iloc[:, i], missing.iloc[:, j])
                    if obs.shape[0] > 1 and obs.shape[1] > 1:
                        chi2_stat, _, _, _ = chi2_contingency(obs)
                        chi_squares.append(chi2_stat)
            d = sum(chi_squares)
            df = len(chi_squares)
            p_value = 1 - chi2.cdf(d, df)
            return d, df, p_value

        d_stat, df_little, p_val = little_mcar_test(df[missing_percentages.index])
        little_test_result = (f"Statistique de test: **{d_stat:.2f}**, degrés de liberté: **{df_little}**, p-value: **{p_val:.4f}**")
        st.session_state.little_test_result = little_test_result
        st.write(st.session_state.little_test_result)

        if p_val < 0.05:
            st.error("Les données manquantes **ne sont pas MCAR** : des méthodes d'imputation plus sophistiquées sont nécessaires (imputation multiple, modèles MNAR, etc.).")
        else:
            st.success("Les données sont **probablement MCAR** : des méthodes d'imputation simples peuvent être envisagées.")
        
        st.success("Étape terminée. Vous pouvez lancer l'application suivante: Préparation 2.")    
        st.session_state["etape10_terminee"] = True


