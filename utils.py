import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def _to_list_of_str(x):
    """Normalise x en liste de chaînes, sans valeurs vides."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


# construction du tableau des étapes de la préparation du dataset
def preparation_process(df, action):
    if "prep_step" not in st.session_state:
        st.session_state.prep_step = 1
    if "process" not in st.session_state or st.session_state.process is None:
        st.session_state.process = pd.DataFrame(
            columns=["Etape", "Nb observations", "Nb variables", "Traitement"]
        )

    n_obs, n_var = df.shape
    traitement_txt = " | ".join([str(action).strip()]) if str(action).strip() else "—"

    # Anti-duplica: ne rien faire si la dernière ligne est identique
    last = st.session_state.process.tail(1)
    if not last.empty:
        same_t = last["Traitement"].iloc[0] == traitement_txt
        same_obs = last["Nb observations"].iloc[0] == n_obs
        same_vars = last["Nb variables"].iloc[0] == n_var
        if same_t and same_obs and same_vars:
            return  # on n’empile pas encore la même ligne

    i = st.session_state.prep_step
    new_row = pd.DataFrame([{
        "Etape": f"préparation {i}",
        "Nb observations": n_obs,
        "Nb variables": n_var,
        "Traitement": traitement_txt,
    }])

    st.session_state.process = pd.concat([st.session_state.process, new_row], ignore_index=True)
    st.session_state.prep_step += 1

# valeurs par défaut des paramètres
def ensure_analysis_params(state):
    state.setdefault("distinct_threshold_continuous", 5)
    state.setdefault("num_quantiles", 5)
    state.setdefault("mod_freq_min", 0.90)
    state.setdefault("correlation_threshold_nmi", 0.70)
    state.setdefault("nmi_normalization_method", "min")
    state.setdefault("n_clusters_segmentation", 10)
    state.setdefault("n_clusters_target", 3)
    state.setdefault("kmodes_n_init", 2)


# Discrétisation des variables continues

def discretize_continuous_variables(
    X_in: pd.DataFrame,
    *,
    num_quantiles: int,
    mod_freq_min: float,
    distinct_threshold_continuous: int,
    context_name: str = "",
) -> tuple[pd.DataFrame, dict]:
    """
    Prépare un DataFrame X pour KModes, en reproduisant la logique que tu utilises dans tes modules :
    - détecte variables continues parmi les numériques via nb de valeurs distinctes > distinct_threshold_continuous
    - cast des numériques "discrètes" en str
    - "collapse" des continues très concentrées : mode vs reste (min..max du reste)
    - discrétisation des continues restantes via qcut
    - force toutes les colonnes en str (KModes-friendly)

    Ne stoppe pas le process sur cas anormaux : corrige / fallback et remonte des warnings/errors dans `info`.

    Retour:
      (X_prepared, info)
      info = {
        "context": str,
        "params": {...},
        "warnings": [str, ...],
        "errors": [str, ...],
        "cols": { ... },
        "stats": { col: {...}, ... }
      }
    """
    info = {
        "context": context_name,
        "params": {
            "num_quantiles": num_quantiles,
            "mod_freq_min": mod_freq_min,
            "distinct_threshold_continuous": distinct_threshold_continuous,
        },
        "warnings": [],
        "errors": [],
        "cols": {},
        "stats": {},
        "bins_by_col": {},
        "collapsed_meta": {},
    }

    # ---- garde-fous non bloquants sur les paramètres ----
    if not isinstance(num_quantiles, int):
        info["warnings"].append(f"num_quantiles n'est pas un int ({type(num_quantiles)}). Cast en int.")
        try:
            num_quantiles = int(num_quantiles)
        except Exception:
            info["warnings"].append("Impossible de caster num_quantiles. Fallback à 5.")
            num_quantiles = 5

    if num_quantiles < 2:
        info["warnings"].append(f"num_quantiles={num_quantiles} < 2 → corrigé à 2 (cas inattendu).")
        num_quantiles = 2

    if not isinstance(distinct_threshold_continuous, int):
        info["warnings"].append(
            f"distinct_threshold_continuous n'est pas un int ({type(distinct_threshold_continuous)}). Cast en int."
        )
        try:
            distinct_threshold_continuous = int(distinct_threshold_continuous)
        except Exception:
            info["warnings"].append("Impossible de caster distinct_threshold_continuous. Fallback à 5.")
            distinct_threshold_continuous = 5

    if distinct_threshold_continuous < 1:
        info["warnings"].append(
            f"distinct_threshold_continuous={distinct_threshold_continuous} < 1 → corrigé à 5 (cas inattendu)."
        )
        distinct_threshold_continuous = 5

    try:
        mod_freq_min = float(mod_freq_min)
    except Exception:
        info["warnings"].append("mod_freq_min non castable en float → fallback à 0.9.")
        mod_freq_min = 0.9

    if not (0.0 < mod_freq_min < 1.0):
        info["warnings"].append(f"mod_freq_min={mod_freq_min} hors (0,1) → corrigé à 0.9.")
        mod_freq_min = 0.9

    # Met à jour les params corrigés
    info["params"] = {
        "num_quantiles": num_quantiles,
        "mod_freq_min": mod_freq_min,
        "distinct_threshold_continuous": distinct_threshold_continuous,
    }

    # ---- validation input ----
    if X_in is None or not isinstance(X_in, pd.DataFrame):
        info["errors"].append(f"X_in invalide : attendu DataFrame, reçu {type(X_in)}.")
        return X_in, info
    if X_in.empty:
        info["errors"].append("X_in est vide : aucune variable à traiter.")
        return X_in, info

    X = X_in.copy()

    try:
        # 1) détecter continues vs discrètes (sur numériques)
        num_cols = X.select_dtypes(include=["number"])
        distinct_counts = num_cols.nunique(dropna=True)
        continuous = distinct_counts[distinct_counts > distinct_threshold_continuous].index.tolist()
        discrete = [c for c in X.columns if c not in continuous]

        info["cols"]["continuous_detected"] = continuous.copy()
        info["cols"]["discrete_detected"] = discrete.copy()

        # 2) Cast numériques discrets en str
        for var in discrete:
            # si numérique discret (int/float) -> string
            if X[var].dtype.kind in ("i", "f"):
                X[var] = X[var].astype(str)

        # 3) Collapse des continues très concentrées (mode vs reste)
        collapsed_cols = []
        for col in continuous:
            s = X[col]

            # Tu dis qu'il ne peut pas y avoir de NA.
            # On ne stoppe pas, mais on trace au cas où.
            if s.isna().any():
                info["warnings"].append(f"[{col}] NA détectés (inattendu selon ton pipeline amont).")

            # Calcul du mode et fréquence du mode
            mode_series = s.mode(dropna=True)
            if mode_series.empty:
                info["warnings"].append(f"[{col}] mode() vide → skip collapse.")
                continue

            mode_value = mode_series.iloc[0]
            vc = s.value_counts(normalize=True, dropna=True)
            mode_freq = float(vc.get(mode_value, 0.0))

            info["stats"][col] = {
                "n_unique": int(s.nunique(dropna=True)),
                "mode": str(mode_value),
                "mode_freq": mode_freq,
            }

            if mode_freq > mod_freq_min:
                X[col] = X[col].astype("string")
                non_mode = s != mode_value
                if non_mode.any():
                    min_value = s.loc[non_mode].min()
                    max_value = s.loc[non_mode].max()
                    X.loc[non_mode, col] = f"{min_value} to {max_value}"
                    other_label = f"{min_value} to {max_value}"
                else:
                    other_label = "others (none)"
                X.loc[~non_mode, col] = f"{mode_value}"
                collapsed_cols.append(col)

                vc_after = X[col].value_counts(normalize=True, dropna=False)
                info["collapsed_meta"][col] = {
                    "dominant_label": str(mode_value),
                    "dominant_freq": float(vc_after.get(str(mode_value), 0.0)),
                    "other_label": other_label,
                    "other_freq": float(vc_after.get(other_label, 0.0)),
                }

        # Supprimer les collapsed du set continu
        continuous_after = [c for c in continuous if c not in collapsed_cols]
        info["cols"]["collapsed"] = collapsed_cols
        info["cols"]["continuous_after_collapse"] = continuous_after.copy()

        # 4) Discrétisation qcut des continues restantes
        binned_cols = []
        for col in continuous_after:
            s = X[col]
            s_nonnull = s.dropna()
            uniq = int(s_nonnull.nunique())

            # qcut exige au moins 2 valeurs distinctes
            if uniq < 2:
                info["warnings"].append(f"[{col}] uniq={uniq} : qcut impossible → converti en catégorie (str).")
                X[col] = s.fillna("NA").astype(str)
                continue

            q = min(num_quantiles, max(2, uniq))

            try:
                codes, bins = pd.qcut(s, q, retbins=True, labels=False, duplicates="drop")
            except Exception as e:
                # On ne stoppe pas : on log, et fallback en string
                info["errors"].append(
                    f"[{col}] qcut a échoué (uniq={uniq}, q={q}, dtype={s.dtype}). "
                    f"Fallback en str. Détail: {repr(e)}"
                )
                X[col] = s.fillna("NA").astype(str)
                continue

            bins = [float(b) for b in bins]
            labels = [f"({round(bins[i], 2)}, {round(bins[i+1], 2)}]" for i in range(len(bins) - 1)]
            mapping = dict(enumerate(labels))

            X[col] = codes.map(mapping)
            info["bins_by_col"][col] = labels

            # Si NaN apparaissent (ex. NA dans s), on remplace
            if X[col].isna().any():
                info["warnings"].append(f"[{col}] NaN après mapping bins → remplacés par 'NA'.")
                X[col] = X[col].fillna("NA")

            X[col] = X[col].astype(str)
            binned_cols.append(col)

        info["cols"]["binned"] = binned_cols

        # 5) Final : forcer tout en str (KModes-friendly)
        # + fallback si NaN restants (ne devrait pas arriver)
        if X.isna().any().any():
            bad_cols = X.columns[X.isna().any()].tolist()
            info["warnings"].append(
                f"NaN restants après préparation (colonnes: {bad_cols}) → remplacés par 'NA'."
            )
            X = X.fillna("NA")

        X = X.astype(str)
        return X, info

    except Exception as e:
        info["errors"].append(f"Erreur inattendue dans discretize_continuous_variables: {repr(e)}")
        # On renvoie quand même quelque chose pour ne pas bloquer le process
        return X_in, info
        
