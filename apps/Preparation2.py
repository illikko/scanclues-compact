import streamlit as st
import pandas as pd
import numpy as np
from utils import preparation_process


# fonctions
# Imputation hot-deck simple (non conditionnel)

def hot_deck_simple_impute(df: pd.DataFrame, cols: list[str], random_state: int = 0) -> tuple[pd.DataFrame, dict]:
    """
    Hot-deck simple (non conditionnel) :
    remplace chaque NaN par une valeur tirée aléatoirement parmi les valeurs observées de la colonne.
    Retourne df imputé + stats (nb imputé par colonne).
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()
    stats = {}

    for col in cols:
        mask = out[col].isna()
        n_missing = int(mask.sum())
        if n_missing == 0:
            stats[col] = 0
            continue

        donors = out.loc[~mask, col].dropna()
        if donors.empty:
            # Cas extrême: colonne entièrement manquante (normalement déjà supprimée par >50% si threshold_var=0.5,
            # mais si exactement 50% et les autres colonnes ont supprimé des lignes, ça peut arriver)
            stats[col] = 0
            continue

        sampled = rng.choice(donors.to_numpy(), size=n_missing, replace=True)
        out.loc[mask, col] = sampled
        stats[col] = n_missing

    return out, stats

def run():
    st.header("Préparation du dataset (2/4)")
    st.subheader("Traitement des valeurs manquantes")

    # déclaration des variables
    if "etape11_terminee" not in st.session_state:
        st.session_state["etape11_terminee"] = False

    df = None

    # rechargement du dataset
    if "df_imputed_structural" in st.session_state:
        df = st.session_state.df_imputed_structural

        st.success("Dataset chargé depuis l'application précédente.")
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())
    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")

    # 2- Nettoyage
    st.markdown("##### Caractéristiques du jeu de données")
    n_obs = df.shape[0]
    n_var = df.shape[1]
    dim = n_var / n_obs
    st.write(f"- Observations : {n_obs}")
    st.write(f"- Variables : {n_var}")
    st.write(f"- Dimensionalité : {dim:.4f}")
    st.write(f"- Duplicats : {df.duplicated().sum()} ({df.duplicated().mean():.2%})")
    
    # explications sur la prépration du jeu de données
    url = "https://medium.com/@vincent.castaignet/a-comprensive-guide-for-analysing-rich-tabular-datasets-part-2-7ddb613cc911"
    st.markdown(f"Pour une présentation des enjeux du nettoyage des jeux de données, se référer à cet [article]({url}).")

    # === Traitement des valeurs manquantes ===
    threshold_var = st.slider("Taux maximum de valeurs manquantes pour l'imputation", 0.01, 0.99, 0.5)
    threshold_var_min = st.slider("en dessous de ce seuil de valeurs manquantes l'imputation est simple", 0.01, 0.5, 0.05)
    threshold_obs = st.slider("Taux maximum de variables manquantes pour l'imputation", 0.01, 0.99, 0.5)

    # 0) Sécurité
    df = df.copy()

    # 1) Supprimer les colonnes avec trop de manquants
    col_missing_ratio = df.isna().mean()
    drop_cols = col_missing_ratio[col_missing_ratio > threshold_var].index.tolist()

    if drop_cols:
        action1 = (
            f"{len(drop_cols)} colonne(s) supprimée(s) car > {threshold_var:.0%} de valeurs manquantes :\n"
            + "- " + "\n- ".join(map(str, drop_cols))
        )
        df = df.drop(columns=drop_cols)
        st.write(action1)
        preparation_process(df, action1)
    else:
        action1 = "Aucune colonne supprimée du fait d'un excès de valeurs manquantes."
        st.write(action1)

    # 2) Supprimer les lignes avec trop de manquants
    obs_thresh = int(len(df.columns) * threshold_obs)
    drop_rows = df.index[df.isnull().sum(axis=1) > obs_thresh]

    if len(drop_rows) > 0:
        df = df.drop(index=drop_rows)
        action2 = f"{len(drop_rows)} observations supprimées car avec trop de valeurs manquantes."
        st.write(action2)
        preparation_process(df, action2)

    # 3) Identifier colonnes à traiter
    missing_pct = df.isnull().mean()

    # Colonnes avec des manquants (après suppressions)
    cols_with_na = missing_pct[missing_pct > 0].index.tolist()
    if not cols_with_na:
        action3 = "Aucune valeur manquante restante."
        st.write(action3)
        preparation_process(df, action3)
    else:
        # 3a) Imputation simple (<5%)
        low_missing_cols = missing_pct[(missing_pct > 0) & (missing_pct < threshold_var_min)].index.tolist()

        for col in low_missing_cols:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])
            else:
                # médiane robuste
                median = df[col].median(skipna=True)
                df[col] = df[col].fillna(median)

        if low_missing_cols:
            st.write("Imputation simple (5% à 50% manquants). Nb imputés par colonne :", low_missing_cols[0])

        # 3b) Colonnes à hot-deck simple (entre 5% et 50%)
        hotdeck_cols = missing_pct[(missing_pct >= threshold_var_min) & (missing_pct <= threshold_var)].index.tolist()

        if hotdeck_cols:
            df, stats_hd = hot_deck_simple_impute(df, hotdeck_cols, random_state=0)
            st.write("Hot-deck simple appliqué (5% à 50% manquants). Nb imputés par colonne :", stats_hd)

        action3 = "Valeurs manquantes traitées : imputation simple (<5%) et hot-deck simple (5% à 50%)."
        st.write(action3)
        preparation_process(df, action3)
    
    # === Groupement des modalités rares ===
    
    min_absolute = st.slider("Seuil de groupement des modalités rares (absolu)", 1, 100, 10, 1, key = "min_absolute")
    min_relative = st.selectbox("Seuil de groupement des modalités rares", [0.001, 0.005, 0.01, 0.02, 0.05], index=2, key = "min_relative")

    categorical = df.select_dtypes(include=['object', 'category']).columns
    for var in categorical:
        counts = df[var].value_counts()
        to_group = counts[counts < min_absolute].index.tolist()
        to_group2 = counts[counts < df.shape[0]*min_relative].index.tolist()
        df[var] = df[var].replace(to_group + to_group2, "autre")

    # === Suppression des colonnes avec valeurs uniques ===
    final_unique = df.nunique()
    id_like_cols = [
        col for col in df.columns
        if df[col].nunique() == df.shape[0] and df[col].dtype == 'object'
]
    
    if id_like_cols:
        message_unique = "Colonnes avec valeurs uniques :", id_like_cols
        st.write(message_unique)
        actions.append(message_unique)
        df = df.drop(columns=id_like_cols)
        st.session_state.df_clean = df
        
    # === Enregistrement du dataset nettoyé ===
    n_final_obs, n_final_vars = df.shape
    st.success(
        f"Traitement des valeurs manquantes terminé. "
        f"{n_final_obs} observations, {n_final_vars} variables."
    )
    st.dataframe(df.head())
    csv = df.to_csv(index=False, sep=';', encoding='utf-8')
    st.session_state.df_clean = df
    st.download_button("Télécharger le dataset", csv, "df_clean.csv", "text/csv")
    
    # affichage du tableau de process
    st.markdown("##### État d'avancement de la préparation du dataset :")
    st.dataframe(st.session_state.process)


    st.write("Vous pouvez lancer la prochaine étape dans le menu à gauche: Variables trop corrélées.")
    st.session_state["etape11_terminee"] = True
