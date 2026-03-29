import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from scipy.cluster.hierarchy import dendrogram, linkage

from core.correlations_utils import (
    correlation_matrix_nmi,
    reorder_corr_matrix_by_target,
)
from utils import ensure_analysis_params

# --- Clé OpenAI
api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

MODE_KEY = "__NAV_MODE__"


# ======================
# Fonctions statistiques
# ======================
@st.cache_data(show_spinner=False, ttl=None, max_entries=10)
def compute_corr_nmi(
    df_corr: pd.DataFrame,
    *,
    num_bins: int,
    distinct_threshold_continuous: int,
    normalization_method: str,
) -> pd.DataFrame:
    """
    Fonction pure, cacheable : calcule la matrice de NMI.
    """
    corr_matrix, info = correlation_matrix_nmi(
        df_corr,
        num_bins=num_bins,
        distinct_threshold_continuous=distinct_threshold_continuous,
        normalization_method=normalization_method,
        context_name="analyse_correlations",
    )
    st.session_state["correlation_info"] = info
    st.session_state["correlation_matrix"] = corr_matrix
    return corr_matrix


def _pick_dataset():
    df_active = st.session_state.get("df_active")
    df_encoded = st.session_state.get("df_encoded")
    df_ready = st.session_state.get("df_ready")

    THRESH = 15

    def has_enough_cols(df, n=THRESH):
        return isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] >= n

    candidats = []
    if has_enough_cols(df_active):
        candidats.append(("df_active", "Variables actives, ordinales encodées"))
    if has_enough_cols(df_encoded):
        candidats.append(("df_encoded", "Toutes les variables, ordinales encodées"))
    if isinstance(df_ready, pd.DataFrame) and not df_ready.empty:
        candidats.append(("df_ready", "Toutes les variables"))

    datasets_disponibles = {}
    for key, label in candidats:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            datasets_disponibles[label] = df
    return datasets_disponibles


# ======================
# Application Streamlit
# ======================
def run():

    # drapeau d'étape
    st.session_state.setdefault("etape21_terminee", False)
    st.session_state.setdefault("dendrogram_interpretation", False)
    ensure_analysis_params(st.session_state)

    st.header("Corrélations entre variables (NMI)")

    url = "https://medium.com/@vincent.castaignet/a-comprensive-guide-for-analysing-rich-tabular-datasets-e851a222dd32"
    st.markdown(
        f"Pour une présentation des enjeux des corrélations entre variables, se référer à cet [article]({url})."
    )

    datasets_disponibles = _pick_dataset()
    if not datasets_disponibles:
        st.warning("Aucun dataset valide trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()

    # Sélection dataset
    choix = st.selectbox("Choisissez un dataset à utiliser :", list(datasets_disponibles.keys()), index=0)
    df = datasets_disponibles[choix].copy()

    # Paramètres mutualisés (utils.ensure_analysis_params)
    num_bins = int(st.session_state.get("num_quantiles", 5))
    distinct_threshold_continuous = int(st.session_state.get("distinct_threshold_continuous", 5))
    normalization_method = st.session_state.get("nmi_normalization_method", "min")

    st.caption(
        f"NMI : {num_bins} quantiles | seuil de continuité {distinct_threshold_continuous} modalités | "
        f"normalisation {normalization_method}."
    )

    df_corr = df.copy()

    # Matrice NMI calculée une fois (et stockée pour usage agentique/aval)
    corr_matrix = compute_corr_nmi(
        df_corr,
        num_bins=num_bins,
        distinct_threshold_continuous=distinct_threshold_continuous,
        normalization_method=normalization_method,
    )

    if corr_matrix.empty:
        st.warning("Matrice de corrélation vide (pas assez de variables ou de modalité).")
        return

    # Variable cible pour usage agentique futur (pas d'affichage de matrice ici)
    options = list(df_corr.columns) if len(df_corr.columns) else []
    candidates = st.session_state.get("target_variables", [])
    preferred = candidates[0] if candidates else (options[0] if options else None)
    default_index = options.index(preferred) if (preferred in options) else 0 if options else None
    if options:
        target_variable = st.selectbox(
            "Variable cible (stockée pour usage agentique éventuel) :",
            options,
            index=default_index,
        )
        st.session_state["correlation_display_target"] = target_variable
    else:
        target_variable = None

    # Construction d'un sous-ensemble pour le dendrogramme (aucune heatmap affichée)
    max_vars = st.slider(
        "Nombre maximum de variables dans le dendrogramme :",
        min_value=10,
        max_value=min(50, len(corr_matrix.columns)),
        value=min(35, len(corr_matrix.columns)),
        step=5,
    )

    corr_pairs = (
        corr_matrix.abs()
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )

    top_vars_ordered = []
    if not corr_pairs.empty:
        for (var1, var2), _ in corr_pairs.items():
            top_vars_ordered.extend([var1, var2])
            break

        # Ajout des variables les plus corrélées à celles déjà choisies
        while len(top_vars_ordered) < min(max_vars, len(corr_matrix.columns)):
            best_next = None
            best_corr = -1
            for var in corr_matrix.columns:
                if var in top_vars_ordered:
                    continue
                max_corr = max([abs(corr_matrix.loc[var, sv]) for sv in top_vars_ordered])
                if max_corr > best_corr:
                    best_corr = max_corr
                    best_next = var
            if best_next is None:
                break
            top_vars_ordered.append(best_next)

    # fallback : si aucune paire, prendre les premières colonnes
    if not top_vars_ordered:
        top_vars_ordered = list(corr_matrix.columns)[:max_vars]

    # Matrice filtrée pour le dendrogramme
    top_corr_matrix = corr_matrix.loc[top_vars_ordered, top_vars_ordered]

    # Dendrogramme des corrélations
    st.subheader("Dendrogramme des corrélations entre variables")
    fig, ax = plt.subplots(figsize=(8, max(4, len(top_vars_ordered) * 0.4)))
    linkage_matrix = linkage(top_corr_matrix.fillna(0), method="average")
    dendrogram(linkage_matrix, labels=top_corr_matrix.columns.tolist(), orientation="right", ax=ax)
    st.session_state["dendrogram"] = fig
    st.pyplot(st.session_state["dendrogram"])

    # conversion de la matrice en csv pour l'appel LLM
    labels = top_corr_matrix.columns.tolist()
    labels_df = pd.DataFrame({"id": list(range(len(labels))), "variable": labels})
    linkage_matrix_df = pd.DataFrame(
        linkage_matrix,
        columns=["idx1", "idx2", "distance", "sample_count"],
    )
    st.session_state["correlation_display_target"] = target_variable or st.session_state.get(
        "correlation_display_target"
    )

    linkage_matrix_csv = linkage_matrix_df.to_csv(index=False)
    labels_csv = labels_df.to_csv(index=False)

    # interprétation du dendrogramme par LLM
    dendrogram_interpretation = st.session_state.get("dendrogram_interpretation")
    if dendrogram_interpretation:
        st.subheader("Interprétation des corrélations")
        st.write(st.session_state.dendrogram_interpretation)

    else:
        with st.spinner("Interprétation des corrélations entre variables par LLM en cours..."):
            client = OpenAI(api_key=api_key)
            system_msg = {
                "role": "system",
                "content": """Tu es un·e data analyst senior. Réponds en français, clair et concis.
                    Un dendrogramme de variables a été généré à partir d'une matrice de corrélation (NMI).
                    Interprète les relations entre les variables, leurs corrélations 2 à 2, et la hiérarchie des regroupements.
                    N'explique pas le fonctionnement technique du dendrogramme, ne mentionne pas non plus le terme dendrogramme.
                    Fais une interprétation sémantique des relations entre les variables.
                """,
            }

            user_msg = {
                "role": "user",
                "content": (
                    "Voici la correspondance entre les identifiants et les variables (id,variable):\n"
                    f"{labels_csv}\n\n"
                    "Voici la matrice de liaison (idx1,idx2,distance,sample_count):\n"
                    f"{linkage_matrix_csv}"
                ),
            }

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[system_msg, user_msg],
                    temperature=0,
                    max_tokens=4000,
                )

                dendrogram_interpretation = response.choices[0].message.content
                st.session_state.dendrogram_interpretation = dendrogram_interpretation

            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'appel à l'API : {e}")

    st.write("… Vous pouvez lancer la prochaine étape dans le menu à gauche : Analyse factorielle.")
    st.session_state["etape21_terminee"] = True


if __name__ == "__main__":
    run()
