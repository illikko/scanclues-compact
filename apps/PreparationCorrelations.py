import streamlit as st
import pandas as pd
import numpy as np

from core.correlations_utils import (
    correlation_matrix_nmi,
    DEFAULT_NMI_THRESHOLD,
    DEFAULT_NUM_BINS,
)
from utils import preparation_process, ensure_analysis_params

MODE_KEY = "__NAV_MODE__"


@st.cache_data(show_spinner=False)
def _cached_correlation_matrix_nmi(
    df: pd.DataFrame,
    *,
    num_bins: int,
    distinct_threshold_continuous: int,
    normalization_method: str,
    context_name: str,
):
    return correlation_matrix_nmi(
        df,
        num_bins=num_bins,
        distinct_threshold_continuous=distinct_threshold_continuous,
        normalization_method=normalization_method,
        context_name=context_name,
    )


# ============================================================
# UI Streamlit intégrée à l'application
# ============================================================
def run():
    ensure_analysis_params(st.session_state)

    st.session_state.setdefault("etape12_terminee", False)
    st.session_state.setdefault("correlation_data_filtered", [])
    st.session_state.setdefault("df_ex_corr", None)
    st.session_state.setdefault("to_remove_final", None)

    st.header("Préparation du dataset (3/4)")
    st.subheader("Détection et suppression de variables trop dépendantes (NMI)")

    if "df_clean" in st.session_state:
        df = st.session_state.df_clean
        st.success("Dataset chargé depuis l'application précédente")
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())
    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")
        return

    threshold = float(st.session_state.get("correlation_threshold_nmi", DEFAULT_NMI_THRESHOLD))
    params = {
        "num_bins": int(st.session_state.get("num_quantiles", DEFAULT_NUM_BINS)),
        "distinct_threshold_continuous": int(st.session_state.get("distinct_threshold_continuous", 5)),
        "normalization_method": st.session_state.get("nmi_normalization_method", "min"),
    }

    with st.spinner("Calcul des dépendances (NMI) ..."):
        try:
            corr_matrix, corr_info = _cached_correlation_matrix_nmi(
                df.copy(),
                num_bins=params["num_bins"],
                distinct_threshold_continuous=params["distinct_threshold_continuous"],
                normalization_method=params["normalization_method"],
                context_name="preparation_correlations_nmi",
            )
            st.session_state["correlation_info"] = corr_info
        except Exception as exc:
            st.session_state["pipeline_halt"] = {
                "module": "PreparationCorrelations",
                "cause": "error",
                "error": str(exc),
            }
            st.error(f"Erreur lors du calcul des corrélations NMI : {exc}")
            return

        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        correlation_data = []
        variables_corrigees = set()
        columns = upper_triangle.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                var1 = columns[i]
                var2 = columns[j]
                coef = upper_triangle.loc[var1, var2]
                if pd.notnull(coef) and coef >= threshold:
                    correlation_data.append({"var1": var1, "var2": var2, "coef": coef})
                    variables_corrigees.update([var1, var2])

        st.session_state.correlation_data = correlation_data
        st.session_state.correlation_data_filtered = correlation_data

    st.subheader("Statistiques de dépendance")
    st.markdown(
        f"""
    - Seuil NMI utilisé : **{threshold:.0%}**
    - Variables impliquées dans des dépendances ≥ {threshold:.0%} : **{len(variables_corrigees)}**
    - Couples détectés au-dessus du seuil : **{len(correlation_data)}**
    - Variables continues détectées : **{len(corr_info['continuous'])}**
    - Variables discrètes détectées : **{len(corr_info['discrete'])}**
    - Nombre de quantiles pour discrétisation : **{corr_info['num_bins']}**
    - Normalisation NMI : **{corr_info['normalization_method']}**
    - Couples en erreur : **{len(corr_info['errors'])}**
    """
    )

    corr_rows = st.session_state.get("correlation_data", [])
    if not corr_rows:
        st.info(f"Aucune dépendance détectée au-dessus du seuil de {threshold:.0%}.")
        df_ex_corr = df.copy()
        st.session_state.df_ex_corr = df_ex_corr
        st.session_state.action = f"Aucune variable supprimée (seuil NMI : {threshold:.0%})."
        st.session_state["etape12_terminee"] = True
        return

    st.subheader(f"Variables trop dépendantes (seuil {threshold:.0%})")

    target_variables = st.session_state.get("target_variables", [])
    target_set = set(target_variables) if target_variables else set()
    current_radio_keys = []

    for row in corr_rows:
        var1, var2, coef = row["var1"], row["var2"], row["coef"]
        key_radio = f"radio_corr__{var1}__{var2}"
        current_radio_keys.append(key_radio)

        options = []
        if var1 not in target_set:
            options.append(f"Supprimer {var1}")
        if var2 not in target_set:
            options.append(f"Supprimer {var2}")
        options.append("Ne rien supprimer")

        # En mode pipeline silencieux, st.radio ne persiste pas toujours la valeur
        # dans session_state. On force donc l'écriture pour que la suppression soit
        # appliquée automatiquement.
        choice = st.radio(
            f"{var1} / {var2} dépendance {coef:.2f}",
            options,
            key=key_radio,
        )
        st.session_state[key_radio] = choice

    st.session_state["current_radio_keys"] = current_radio_keys

    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Appliquer les choix sélectionnés"):
            proceed = True

    if proceed:
        selected_actions = []
        for k in current_radio_keys:
            v = st.session_state.get(k)
            if isinstance(v, str) and v.startswith("Supprimer "):
                selected_actions.append(v)

        to_remove = [s.split("Supprimer ", 1)[1].strip() for s in selected_actions]
        existing_columns = set(df.columns)
        to_remove_final = sorted({c for c in to_remove if c in existing_columns})
        missing = sorted(set(to_remove) - set(to_remove_final))
        if missing:
            st.warning(f"Colonnes non trouvées : {', '.join(missing)}")

        if to_remove_final:
            df_ex_corr = df.drop(columns=to_remove_final)
            action = (
                f"{len(to_remove_final)} variable(s) supprimée(s) "
                f"pour dépendance excessive (NMI, seuil {threshold:.0%}) : {', '.join(to_remove_final)}"
            )
        else:
            df_ex_corr = df.copy()
            action = f"Aucune variable supprimée (seuil NMI : {threshold:.0%})."

        st.session_state.df_ex_corr = df_ex_corr
        st.session_state.to_remove_final = to_remove_final
        st.session_state.action = action

    df_ex_corr = st.session_state.df_ex_corr
    st.session_state["illustrative_variables"] = list(set(st.session_state.get("illustrative_variables", [])) & set(df_ex_corr.columns))
    st.session_state["target_variables"] = list(
        set(st.session_state.get("target_variables", [])) & set(df_ex_corr.columns)
    )

    st.write("listes des variables ex corrélation : ", df_ex_corr.columns.tolist())
    st.write("listes des illustratives : ", st.session_state["illustrative_variables"])

    st.subheader("Dataset après traitement")
    st.dataframe(df_ex_corr.head())
    csv = df_ex_corr.to_csv(index=False, sep=';', encoding='latin-1')
    st.download_button("Télécharger le dataset", csv, "df_ex_corr.csv", "text/csv")

    preparation_process(df_ex_corr, st.session_state.action)

    st.markdown("##### Etat d'avancement de la préparation du dataset :")
    st.dataframe(st.session_state.process)

    st.session_state["etape12_terminee"] = True
    st.success("Etape terminée. Vous pouvez lancer l'application suivante : Valeurs anormales.")
