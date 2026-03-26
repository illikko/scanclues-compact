import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os
from openai import OpenAI

# --- Clé OpenAI
api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")


MODE_KEY = "__NAV_MODE__"

# ======================
# Fonctions statistiques
# ======================

def cramers_v(x, y):
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if df_xy.empty:
        return np.nan

    confusion_matrix = pd.crosstab(df_xy["x"], df_xy["y"])
    if confusion_matrix.empty:
        return np.nan

    r, k = confusion_matrix.shape
    if r < 2 or k < 2:
        return np.nan

    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    if n <= 1:
        return np.nan

    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan

    return float(np.sqrt(phi2corr / denom))


def correlation_ratio(categories, measurements):
    df_tmp = pd.DataFrame({
        "cat": categories,
        "val": pd.to_numeric(measurements, errors="coerce")
    }).dropna()

    if df_tmp.empty:
        return np.nan

    fcat, _ = pd.factorize(df_tmp["cat"])
    values = df_tmp["val"].to_numpy(dtype=float)

    cat_num = np.max(fcat) + 1
    if cat_num < 2:
        return np.nan

    y_avg_array = np.zeros(cat_num, dtype=float)
    n_array = np.zeros(cat_num, dtype=float)

    for i in range(cat_num):
        cat_measures = values[fcat == i]
        if len(cat_measures) == 0:
            y_avg_array[i] = np.nan
            n_array[i] = 0
        else:
            y_avg_array[i] = np.mean(cat_measures)
            n_array[i] = len(cat_measures)

    valid = n_array > 0
    if not valid.any():
        return np.nan

    y_avg_array = y_avg_array[valid]
    n_array = n_array[valid]

    y_total_avg = np.sum(y_avg_array * n_array) / np.sum(n_array)
    numerator = np.sum(n_array * (y_avg_array - y_total_avg) ** 2)
    denominator = np.sum((values - y_total_avg) ** 2)

    if denominator <= 0:
        return np.nan

    eta = np.sqrt(numerator / denominator)
    return float(eta)


def mixed_correlation_matrix(df, continuous, discrete):
    df = df.copy()

    # Normaliser les NA pandas en np.nan
    df = df.replace({pd.NA: np.nan})

    # Sécuriser les variables continues
    for col in continuous:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sécuriser les variables discrètes
    for col in discrete:
        df[col] = df[col].astype("object")

    cols = [c for c in df.columns if c in continuous or c in discrete]
    corr_matrix = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0

            elif col1 in continuous and col2 in continuous:
                s1 = pd.to_numeric(df[col1], errors="coerce")
                s2 = pd.to_numeric(df[col2], errors="coerce")
                corr_matrix.loc[col1, col2] = s1.corr(s2)

            elif col1 in discrete and col2 in discrete:
                corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

            else:
                if col1 in continuous:
                    corr_matrix.loc[col1, col2] = correlation_ratio(df[col2], df[col1])
                else:
                    corr_matrix.loc[col1, col2] = correlation_ratio(df[col1], df[col2])

    # Conversion finale robuste
    corr_matrix = corr_matrix.replace({pd.NA: np.nan})
    corr_matrix = corr_matrix.apply(pd.to_numeric, errors="coerce")
    return corr_matrix


def reorder_corr_matrix(corr_matrix, target_variable):
    cols = list(corr_matrix.columns)
    if target_variable in cols:
        cols.remove(target_variable)
        cols = [target_variable] + cols
        corr_matrix = corr_matrix[cols].loc[cols]
        target_corr = corr_matrix[target_variable].abs().sort_values(ascending=False)
        sorted_cols = [target_variable] + list(target_corr.index[1:])
        return corr_matrix.loc[sorted_cols, sorted_cols]
    return corr_matrix


# Cache du calcul lourd

def compute_corr(df_corr: pd.DataFrame, continuous: tuple, discrete: tuple) -> pd.DataFrame:
    """
    Fonction pure, cacheable : prend un sous-DF + listes immuables et renvoie la matrice de corrélation mixte.
    """
    return mixed_correlation_matrix(df_corr, list(continuous), list(discrete))


# ======================
# Application Streamlit
# ======================

def run():

    # drapeau d'étape
    st.session_state.setdefault("etape21_terminee", False)
    st.session_state.setdefault("dendrogram_interpretation", False)

    st.header("Corrélations entre variables")

    # explications sur les enjeux des diffrentes corrélations
    url = "https://medium.com/@vincent.castaignet/a-comprensive-guide-for-analysing-rich-tabular-datasets-e851a222dd32"
    st.markdown(f"Pour une présentation des enjeux des corrélations entre variables, se référer à cet [article]({url}).")

    df_active  = st.session_state.get("df_active")
    df_encoded = st.session_state.get("df_encoded")
    df_ready   = st.session_state.get("df_ready")
    
    THRESH = 15


    def has_enough_cols(df, n=THRESH):
        return isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] >= n

    # Construire la liste des candidats uniquement si DF valides
    candidats = []
    if has_enough_cols(df_active):
        candidats.append(("df_active",  "Variables actives, ordinales encodées"))
    if has_enough_cols(df_encoded):
        candidats.append(("df_encoded", "Toutes les variables, ordinales encodées"))
    # df_ready proposé même s'il a < THRESH ? à toi de choisir :
    if isinstance(df_ready, pd.DataFrame) and not df_ready.empty:
        candidats.append(("df_ready", "Toutes les variables"))

    # Filtrer ce qui existe réellement dans la session
    datasets_disponibles = {}
    for key, label in candidats:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            datasets_disponibles[label] = df

    if not datasets_disponibles:
        st.warning("Aucun dataset valide trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()

    # Sélection dataset
    choix = st.selectbox("Choisissez un dataset à utiliser :", list(datasets_disponibles.keys()), index=0)
    df = datasets_disponibles[choix].copy()

    # Détection auto des types
    continuous = df.select_dtypes(include=['number']).columns.tolist()
    discrete = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    st.subheader("Sélection manuelle des variables (optionnelle)")
    with st.expander("Ajustez la sélection des variables continues et discrètes :"):
        continuous = st.multiselect("Variables continues :", df.columns, default=continuous)
        discrete = st.multiselect("Variables discrètes :", df.columns, default=discrete)

    # Sous-DF pour la corrélation
    vars_utiles = [c for c in df.columns if c in continuous or c in discrete]
    df_corr = df[vars_utiles].copy()

    # Matrice de corrélation calculée uen fois et mise en cache
    corr_matrix = compute_corr(df_corr, tuple(continuous), tuple(discrete))

    # sélection du nombre maximum de corrélations affichées
    max_vars = st.slider("Nombre maximum de variables à afficher :", min_value=10, max_value=50, value=35, step=5)

    # -----------------------------
    # Matrice des variables les plus corrélées
    # -----------------------------
    corr_pairs = (
        corr_matrix.abs()
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )

    top_vars_set = set()
    top_vars_ordered = []

    # Démarrer avec le couple le plus corrélé
    for (var1, var2), _ in corr_pairs.items():
        top_vars_ordered.extend([var1, var2])
        top_vars_set.update([var1, var2])
        break

    # Ajout des variables les plus corrélées à celles déjà choisies
    while len(top_vars_ordered) < min(max_vars, len(corr_matrix.columns)):
        best_next = None
        best_corr = -1
        for var in corr_matrix.columns:
            if var in top_vars_set:
                continue
            max_corr = max([abs(corr_matrix.loc[var, sv]) for sv in top_vars_ordered])
            if max_corr > best_corr:
                best_corr = max_corr
                best_next = var
        if best_next is None:
            break
        top_vars_ordered.append(best_next)
        top_vars_set.add(best_next)

    # Matrice filtrée et ordonnée
    top_corr_matrix = corr_matrix.loc[top_vars_ordered, top_vars_ordered]
    st.session_state.top_corr_matrix = top_corr_matrix
    st.subheader("Matrice des variables les plus corrélées")
    st.dataframe(st.session_state.top_corr_matrix.style.background_gradient(cmap="Blues").format("{:.2f}"), use_container_width=True)

   # ---------------
    # Analyse des corrélations à partir de la variable cible
    # ---------------
    st.subheader(f"Matrice de corrélation à partir d'une variable cible")
    
    st.markdown("##### Sélection de la variable cible")

    options = list(df_corr.columns) if len(df_corr.columns) else list(df.columns)
    candidates = st.session_state.get("target_variables", [])
    preferred = candidates[0] if candidates else (options[0] if options else None)
    default_index = options.index(preferred) if (preferred in options) else 0

    target_variable = st.selectbox("Select a target variable :", options, index=default_index)

    if target_variable:
        # réordonne la matrice déjà calculée
        reordered_corr = reorder_corr_matrix(corr_matrix, target_variable)

        top_target_vars = (
            reordered_corr[target_variable]
            .abs()
            .sort_values(ascending=False)
            .head(max_vars)
            .index
        )
        selected_corr_matrix = reordered_corr.loc[top_target_vars, top_target_vars]

        st.markdown(f"##### Matrice de corrélation à partir de la variable cible : `{target_variable}`")
        styled_corr = selected_corr_matrix.style.format("{:.2f}").background_gradient(cmap="Blues")
        st.dataframe(styled_corr, use_container_width=True)

    # -------------
    # Dendrogramme des corrélations
    # -------------
    st.subheader("Dendrogramme des corrélations entre variables")
    fig, ax = plt.subplots(figsize=(8, max(4, len(top_vars_ordered) * 0.4)))
    linkage_matrix = linkage(top_corr_matrix.fillna(0), method='average')
    dendrogram(linkage_matrix, labels=top_corr_matrix.columns.tolist(), orientation='right', ax=ax)
    st.session_state["dendrogram"] = fig
    st.pyplot(st.session_state["dendrogram"])
    
    # conversion de la matrice en csv
    labels = top_corr_matrix.columns.tolist()
    labels_df = pd.DataFrame({
        "id": list(range(len(labels))),
        "variable": labels
    })
    linkage_matrix_df = pd.DataFrame(
        linkage_matrix,
        columns=["idx1", "idx2", "distance", "sample_count"]
    )
    st.subheader("Matrice de liaison utilisée pour le dendrogramme")
    st.dataframe(linkage_matrix_df)

    linkage_matrix_csv = linkage_matrix_df.to_csv(index=False)
    labels_csv = labels_df.to_csv(index=False)

    # interprétation du dendrogramme par LLM
    dendrogram_interpretation = st.session_state.get("dendrogram_interpretation")
    if dendrogram_interpretation:
        st.subheader("Interprétation des corrélations")
        st.write(st.session_state.dendrogram_interpretation)
    
    else:
        with st.spinner("Interprétation des corrélations entre variables par LLM en cours..."):
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            system_msg = {
                "role": "system",
                "content": f'''Tu es un·e data analyst senior. Réponds en français, clair et concis.
                    Un dendrogramme de variables a été généré à partir d'une matrice de corrélation.
                    Interprétez les relations entre les variables, leurs corrélations 2 à 2, et la hiérarchie des regroupements.
                    N'expliquez pas le fonctionnement technique du dendrogramme, ne mentionne pas non plus le terme dendrogramme.
                    Faites une interprétation sémantique des relations entre les variables.
                '''
            }

            user_msg = {
                "role": "user",
                "content": (
                    "Voici la correspondance entre les identifiants et les variables (id,variable):\n"
                    f"{labels_csv}\n\n"
                    "Voici la matrice de liaison (idx1,idx2,distance,sample_count):\n"
                    f"{linkage_matrix_csv}"
                )
            }
                
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[system_msg, user_msg],
                    temperature=0,
                    max_tokens=4000
                )

                dendrogram_interpretation = response.choices[0].message.content
                st.session_state.dendrogram_interpretation = dendrogram_interpretation

            except Exception as e:
                st.error(f"Une erreur est survenue lors de lâ€™appel à lâ€™API : {e}")
    

        st.write("âœ… Vous pouvez lancer la prochaine étape dans le menu à gauche : Analyse factorielle.")
        st.session_state["etape21_terminee"] = True


if __name__ == "__main__":
    run()

