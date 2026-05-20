import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2, chi2_contingency

from core.preparation_details import refresh_preparation_details_payload
from core.preparation_diagnostics import set_preparation_diagnostic


def count_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    def is_nan_string(x):
        if isinstance(x, str):
            return x.strip().upper() in ["NAN", "NA", "N/A", "NONE", ""]
        return False

    direct_missing = df.isnull().sum()
    indirect_missing = df.applymap(is_nan_string).sum()
    total_missing = direct_missing + indirect_missing
    total_missing_percent = total_missing / max(1, df.shape[0]) * 100

    return pd.DataFrame(
        {
            "Direct Missing": direct_missing,
            "Indirect Missing": indirect_missing,
            "Total Missing": total_missing,
            "Total Missing %": total_missing_percent,
        }
    )


def little_mcar_test(data: pd.DataFrame) -> tuple[float, int, float]:
    missing = data.isnull()
    chi_squares = []
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            obs = pd.crosstab(missing.iloc[:, i], missing.iloc[:, j])
            if obs.shape[0] > 1 and obs.shape[1] > 1:
                chi2_stat, _, _, _ = chi2_contingency(obs)
                chi_squares.append(chi2_stat)
    statistic = float(sum(chi_squares))
    degrees = int(len(chi_squares))
    p_value = float(1 - chi2.cdf(statistic, degrees))
    return statistic, degrees, p_value


def diagnose_missing_values(df: pd.DataFrame) -> tuple[dict, dict]:
    df_work = df.copy()
    df_work = df_work.replace(["NULL", "NAN"], np.nan)

    missing_df = count_missing_values(df_work).sort_values("Total Missing", ascending=False)
    missing_percentages = df_work.isnull().mean().sort_values(ascending=False) * 100
    missing_percentages = missing_percentages[missing_percentages > 0]
    high_missing_columns = missing_percentages[missing_percentages >= 20].index.astype(str).tolist()

    diagnostic = {
        "id": "missing_values",
        "label": "Traitement des valeurs manquantes",
        "needed": bool(not missing_percentages.empty),
        "reason": (
            f"{len(missing_percentages)} colonne(s) contiennent des valeurs manquantes"
            if not missing_percentages.empty
            else "Aucune valeur manquante détectée"
        ),
        "details": {
            "n_columns": int(len(missing_percentages)),
            "columns": missing_percentages.index.astype(str).tolist(),
            "high_missing_columns": high_missing_columns,
            "missing_percentages": {str(k): float(v) for k, v in missing_percentages.items()},
        },
        "compute_module": "Preparation2",
        "render_module": "DiagnosticMissing",
        "available": True,
    }

    artifacts = {
        "missing_df": missing_df,
        "missing_percentages": missing_percentages,
        "df_for_render": df_work,
    }
    return diagnostic, artifacts


def build_missing_artifacts(df: pd.DataFrame) -> dict:
    diagnostic, artifacts = diagnose_missing_values(df)
    set_preparation_diagnostic(diagnostic)
    st.session_state["missing_df"] = artifacts["missing_df"]
    st.session_state["missing_diagnostic"] = diagnostic
    refresh_preparation_details_payload()
    return artifacts


def render_missing_details(artifacts: dict) -> None:
    missing_df = artifacts["missing_df"]
    missing_percentages = artifacts["missing_percentages"]
    df = artifacts["df_for_render"]

    st.subheader("Analyse des valeurs manquantes")
    st.dataframe(missing_df)

    st.markdown("##### Pourcentage de valeurs manquantes par variable")
    if missing_percentages.empty:
        st.info("Aucune valeur manquante détectée.")
        st.session_state["etape10_terminee"] = True
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(missing_percentages) / 3)))
    missing_percentages.sort_values().plot(kind="barh", ax=ax)
    for i, value in enumerate(missing_percentages.sort_values()):
        ax.text(value + 0.5, i, f"{value:.1f}%", va="center")
    ax.set_title("Pourcentage de données manquantes par variable")
    ax.set_xlabel("Pourcentage")
    st.session_state["fig_missing_variables"] = fig
    st.session_state["fig_missing_percentages"] = fig
    st.pyplot(fig)

    st.markdown("##### Variables avec valeurs manquantes")
    if len(missing_percentages) == 1:
        st.info("Il n'y a qu'une variable avec valeurs manquantes, donc pas de dépendance à évaluer.")
        st.session_state["etape10_terminee"] = True
        return

    st.subheader("Cartographie des dépendances")
    fig2, ax2 = plt.subplots(figsize=(10, len(missing_percentages) * 0.4))
    sns.heatmap(df[missing_percentages.index].isnull().T, cmap="YlOrRd", cbar=False, xticklabels=False, ax=ax2)
    ax2.set_yticklabels([f"{col} ({missing_percentages[col]:.1f}%)" for col in missing_percentages.index], rotation=0)
    ax2.set_title("Visualisation des dépendances entre les variables avec données manquantes")
    st.session_state["fig_missing_percentage_heatmap"] = fig2
    st.pyplot(fig2)

    st.markdown("##### Heatmap de corrélation des variables avec valeurs manquantes")
    fig3 = msno.heatmap(df[missing_percentages.index])
    st.session_state["fig_missing_correlation_heatmap"] = fig3
    st.pyplot(fig3.figure)

    st.markdown("##### Dendrogramme de dépendance entre variables avec valeurs manquantes")
    fig_ax = msno.dendrogram(df[missing_percentages.index])
    fig4 = fig_ax.get_figure()
    st.session_state["fig_missing_correlation_dendrogram"] = fig4
    st.pyplot(fig4)

    st.markdown("##### Test de Little (MCAR)")
    statistic, degrees, p_value = little_mcar_test(df[missing_percentages.index])
    little_test_result = (
        f"Statistique de test: **{statistic:.2f}**, degrés de liberté: **{degrees}**, p-value: **{p_value:.4f}**"
    )
    st.session_state["little_test_result"] = little_test_result
    st.write(little_test_result)

    if p_value < 0.05:
        st.error(
            "Les données manquantes **ne sont pas MCAR** : des méthodes d'imputation plus sophistiquées sont nécessaires."
        )
    else:
        st.success("Les données sont **probablement MCAR** : des méthodes d'imputation simples peuvent être envisagées.")

    st.success("Étape terminée. Vous pouvez lancer l'application suivante: Préparation 2.")
    st.session_state["etape10_terminee"] = True
    refresh_preparation_details_payload()


def run():
    st.header("Diagnostic des valeurs manquantes")

    st.session_state.setdefault("etape10_terminee", False)
    st.session_state.setdefault("missing_df", None)

    df = st.session_state.get("df_imputed_structural")
    if not isinstance(df, pd.DataFrame):
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application Préparation 1.")
        st.stop()

    artifacts = build_missing_artifacts(df)
    render_missing_details(artifacts)
