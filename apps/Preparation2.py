import numpy as np
import pandas as pd
import streamlit as st

from core.df_registry import DFState, set_df
from core.preparation_details import refresh_preparation_details_payload
from utils import preparation_process


def hot_deck_simple_impute(df: pd.DataFrame, cols: list[str], random_state: int = 0) -> tuple[pd.DataFrame, dict]:
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
            stats[col] = 0
            continue

        sampled = rng.choice(donors.to_numpy(), size=n_missing, replace=True)
        out.loc[mask, col] = sampled
        stats[col] = n_missing

    return out, stats


def log_preparation_step(df: pd.DataFrame, action: str) -> None:
    if str(action).strip():
        preparation_process(df, action)


def apply_missing_value_treatment(
    df: pd.DataFrame,
    *,
    threshold_var: float,
    threshold_var_min: float,
    threshold_obs: float,
) -> tuple[pd.DataFrame, list[str], dict]:
    out = df.copy()
    actions: list[str] = []
    details: dict = {}

    col_missing_ratio = out.isna().mean()
    drop_cols = col_missing_ratio[col_missing_ratio > threshold_var].index.tolist()
    if drop_cols:
        out = out.drop(columns=drop_cols)
        action = (
            f"{len(drop_cols)} colonne(s) supprimée(s) car > {threshold_var:.0%} de valeurs manquantes :\n"
            + "- " + "\n- ".join(map(str, drop_cols))
        )
        actions.append(action)
        details["dropped_columns"] = drop_cols

    obs_thresh = int(len(out.columns) * threshold_obs)
    drop_rows = out.index[out.isnull().sum(axis=1) > obs_thresh]
    if len(drop_rows) > 0:
        out = out.drop(index=drop_rows)
        action = f"{len(drop_rows)} observations supprimées car avec trop de valeurs manquantes."
        actions.append(action)
        details["dropped_rows"] = int(len(drop_rows))

    missing_pct = out.isnull().mean()
    cols_with_na = missing_pct[missing_pct > 0].index.tolist()
    details["remaining_missing_columns"] = cols_with_na

    if cols_with_na:
        low_missing_cols = missing_pct[(missing_pct > 0) & (missing_pct < threshold_var_min)].index.tolist()
        for col in low_missing_cols:
            if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_categorical_dtype(out[col]):
                mode = out[col].mode(dropna=True)
                if not mode.empty:
                    out[col] = out[col].fillna(mode.iloc[0])
            else:
                median = out[col].median(skipna=True)
                out[col] = out[col].fillna(median)

        if low_missing_cols:
            actions.append(
                f"Imputation simple appliquée sur {len(low_missing_cols)} colonne(s) avec moins de {threshold_var_min:.0%} de manquants."
            )
            details["simple_imputation_columns"] = low_missing_cols

        hotdeck_cols = missing_pct[(missing_pct >= threshold_var_min) & (missing_pct <= threshold_var)].index.tolist()
        if hotdeck_cols:
            out, stats_hd = hot_deck_simple_impute(out, hotdeck_cols, random_state=0)
            actions.append(
                f"Hot-deck simple appliqué sur {len(hotdeck_cols)} colonne(s) entre {threshold_var_min:.0%} et {threshold_var:.0%} de manquants."
            )
            details["hotdeck_stats"] = stats_hd

    return out, actions, details


def group_rare_modalities(
    df: pd.DataFrame,
    *,
    min_absolute: int,
    min_relative: float,
) -> tuple[pd.DataFrame, list[str], dict]:
    out = df.copy()
    changed_columns: list[str] = []

    categorical = out.select_dtypes(include=["object", "category"]).columns
    for var in categorical:
        counts = out[var].value_counts()
        to_group = counts[counts < min_absolute].index.tolist()
        to_group2 = counts[counts < out.shape[0] * min_relative].index.tolist()
        replacements = list(dict.fromkeys(to_group + to_group2))
        if replacements:
            out[var] = out[var].replace(replacements, "autre")
            changed_columns.append(str(var))

    actions = []
    if changed_columns:
        actions.append(f"Regroupement des modalités rares sur {len(changed_columns)} colonne(s).")

    return out, actions, {"grouped_columns": changed_columns}


def handle_constant_columns_after_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    out = df.copy()
    final_unique = out.nunique(dropna=False)
    constant_cols = final_unique[final_unique <= 1].index.tolist()
    id_like_cols = [
        col for col in out.columns
        if out[col].nunique(dropna=False) == out.shape[0] and out[col].dtype == "object"
    ]

    dropped_columns = list(dict.fromkeys([*constant_cols, *id_like_cols]))
    actions: list[str] = []
    if dropped_columns:
        out = out.drop(columns=dropped_columns)
        actions.append(
            f"Suppression de {len(dropped_columns)} colonne(s) non informatives après nettoyage."
        )

    return out, actions, {"dropped_columns": dropped_columns}


def run_second_pass_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    return handle_constant_columns_after_cleaning(df)


def compute_preparation2(
    df: pd.DataFrame,
    *,
    threshold_var: float,
    threshold_var_min: float,
    threshold_obs: float,
    min_absolute: int,
    min_relative: float,
) -> tuple[pd.DataFrame, list[str], dict]:
    out = df.copy()
    actions: list[str] = []
    details: dict = {}

    out, missing_actions, missing_details = apply_missing_value_treatment(
        out,
        threshold_var=threshold_var,
        threshold_var_min=threshold_var_min,
        threshold_obs=threshold_obs,
    )
    actions.extend(missing_actions)
    details["missing_values"] = missing_details

    out, grouping_actions, grouping_details = group_rare_modalities(
        out,
        min_absolute=min_absolute,
        min_relative=min_relative,
    )
    actions.extend(grouping_actions)
    details["rare_modalities"] = grouping_details

    out, second_pass_actions, second_pass_details = run_second_pass_checks(out)
    actions.extend(second_pass_actions)
    details["second_pass"] = second_pass_details

    return out, actions, details


def run():
    st.header("Préparation du dataset (2/4)")
    st.subheader("Traitement des valeurs manquantes")

    st.session_state.setdefault("etape11_terminee", False)

    df = st.session_state.get("df_imputed_structural")
    if not isinstance(df, pd.DataFrame):
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()

    st.success("Dataset chargé depuis l'application précédente.")
    st.write("Aperçu du dataset :")
    st.dataframe(df.head())

    st.markdown("##### Caractéristiques du jeu de données")
    n_obs = df.shape[0]
    n_var = df.shape[1]
    dim = n_var / max(1, n_obs)
    st.write(f"- Observations : {n_obs}")
    st.write(f"- Variables : {n_var}")
    st.write(f"- Dimensionnalité : {dim:.4f}")
    st.write(f"- Duplicats : {df.duplicated().sum()} ({df.duplicated().mean():.2%})")

    url = "https://medium.com/@vincent.castaignet/a-comprensive-guide-for-analysing-rich-tabular-datasets-part-2-7ddb613cc911"
    st.markdown(f"Pour une présentation des enjeux du nettoyage des jeux de données, se référer à cet [article]({url}).")

    threshold_var = st.slider("Taux maximum de valeurs manquantes pour l'imputation", 0.01, 0.99, 0.5)
    threshold_var_min = st.slider("en dessous de ce seuil de valeurs manquantes l'imputation est simple", 0.01, 0.5, 0.05)
    threshold_obs = st.slider("Taux maximum de variables manquantes pour l'imputation", 0.01, 0.99, 0.5)
    min_absolute = st.slider("Seuil de groupement des modalités rares (absolu)", 1, 100, 10, 1, key="min_absolute")
    min_relative = st.selectbox("Seuil de groupement des modalités rares", [0.001, 0.005, 0.01, 0.02, 0.05], index=2, key="min_relative")

    df_clean, actions, details = compute_preparation2(
        df.copy(),
        threshold_var=threshold_var,
        threshold_var_min=threshold_var_min,
        threshold_obs=threshold_obs,
        min_absolute=min_absolute,
        min_relative=min_relative,
    )

    for action in actions:
        st.write(action)
        log_preparation_step(df_clean, action)

    st.session_state["preparation2_details"] = details
    st.session_state["df_clean"] = df_clean
    set_df(DFState.CLEAN, df_clean, step_name="Preparation2")
    refresh_preparation_details_payload()

    n_final_obs, n_final_vars = df_clean.shape
    st.success(
        f"Traitement des valeurs manquantes terminé. "
        f"{n_final_obs} observations, {n_final_vars} variables."
    )
    st.dataframe(df_clean.head())
    csv = df_clean.to_csv(index=False, sep=";", encoding="utf-8")
    st.download_button("Télécharger le dataset", csv, "df_clean.csv", "text/csv")

    st.markdown("##### État d'avancement de la préparation du dataset :")
    st.dataframe(st.session_state.process)

    st.write("Vous pouvez lancer la prochaine étape dans le menu à gauche: Variables trop corrélées.")
    st.session_state["etape11_terminee"] = True
