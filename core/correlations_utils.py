import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from utils import discretize_continuous_variables

# Paramètres par défaut mutualisés pour les corrélations NMI
DEFAULT_NMI_THRESHOLD = 0.70
DEFAULT_NUM_BINS = 5


def cramers_v_bias_corrected(x, y):
    """
    Version biais-corrigÃ©e du V de CramÃ©r.
    """
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if df_xy.empty:
        return np.nan

    table = pd.crosstab(df_xy["x"], df_xy["y"])
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    if n <= 1:
        return np.nan

    r, k = table.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


def compute_cramers_v_matrix(df_cat: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule une matrice de V de CramÃ©r biais-corrigÃ© sur un DataFrame catÃ©goriel.
    """
    if df_cat is None or not isinstance(df_cat, pd.DataFrame) or df_cat.empty:
        return pd.DataFrame()

    df_cat = df_cat.copy().replace({pd.NA: np.nan})
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].astype("string")

    cols = list(df_cat.columns)
    corr_matrix = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if j < i:
                continue
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
            else:
                v = cramers_v_bias_corrected(df_cat[col1], df_cat[col2])
                corr_matrix.loc[col1, col2] = v
                corr_matrix.loc[col2, col1] = v
    return corr_matrix


def correlation_matrix_v_cramer(
    df: pd.DataFrame,
    *,
    num_quantiles: int = 5,
    mod_freq_min: float = 0.9,
    distinct_threshold_continuous: int = 5,
    context_name: str = "correlation",
) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline complet : discrÃ©tisation + matrice de V de CramÃ©r.
    """
    df_disc, info = discretize_continuous_variables(
        df,
        num_quantiles=num_quantiles,
        mod_freq_min=mod_freq_min,
        distinct_threshold_continuous=distinct_threshold_continuous,
        context_name=context_name,
    )
    if isinstance(df_disc, pd.DataFrame):
        for col in df_disc.columns:
            df_disc[col] = df_disc[col].astype("string")

    corr_matrix = compute_cramers_v_matrix(df_disc)
    return corr_matrix, info


# ============================================================
# Mutual Information (NMI) – utilitaires mutualisés
# ============================================================
def infer_variable_types(df: pd.DataFrame, distinct_threshold_continuous: int = 10):
    """Retourne (continuous, discrete) en détectant les numériques à forte cardinalité."""
    continuous = []
    discrete = []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            nunique = s.dropna().nunique()
            if nunique > distinct_threshold_continuous:
                continuous.append(col)
            else:
                discrete.append(col)
        else:
            discrete.append(col)

    return continuous, discrete


def fill_missing_for_discrete(s: pd.Series) -> pd.Series:
    return s.astype("object").fillna("__MISSING__")


def fill_missing_for_continuous(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return s_num.fillna(s_num.median())
    return s_num.fillna(0.0)


def encode_discrete(s: pd.Series) -> pd.Series:
    s2 = fill_missing_for_discrete(s).astype(str)
    codes, _ = pd.factorize(s2, sort=True)
    return pd.Series(codes, index=s.index, dtype=int)


def discretize_series_quantiles(s: pd.Series, n_bins: int = DEFAULT_NUM_BINS) -> pd.Series:
    s_num = fill_missing_for_continuous(s)
    if s_num.nunique() <= 1:
        return pd.Series(np.zeros(len(s_num), dtype=int), index=s.index)

    bins = max(2, min(n_bins, int(s_num.nunique())))
    try:
        out = pd.qcut(s_num, q=bins, labels=False, duplicates="drop")
        return pd.Series(out, index=s.index).fillna(-1).astype(int)
    except Exception:
        ranks = s_num.rank(method="average")
        out = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
        return pd.Series(out, index=s.index).fillna(-1).astype(int)


def entropy_from_codes(codes: pd.Series) -> float:
    counts = pd.Series(codes).value_counts(dropna=False).to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def mutual_information_from_codes(x_codes: pd.Series, y_codes: pd.Series) -> float:
    ct = pd.crosstab(x_codes, y_codes, dropna=False)
    n = ct.to_numpy(dtype=float)
    total = n.sum()
    if total <= 0:
        return 0.0

    pxy = n / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py

    mask = (pxy > 0) & (expected > 0)
    if not np.any(mask):
        return 0.0

    return float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))


def normalized_mi_from_codes(x_codes: pd.Series, y_codes: pd.Series, method: str = "min") -> float:
    mi = mutual_information_from_codes(x_codes, y_codes)
    hx = entropy_from_codes(x_codes)
    hy = entropy_from_codes(y_codes)

    if method == "sqrt":
        denom = np.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0
    elif method == "mean":
        denom = (hx + hy) / 2.0
    else:
        denom = min(hx, hy)

    if denom <= 1e-12:
        return 0.0

    value = float(mi / denom)
    return max(0.0, min(1.0, value))


def prepare_codes_for_pair(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    continuous: set,
    num_bins: int,
):
    s1 = df[col1]
    s2 = df[col2]

    if col1 in continuous:
        x_codes = discretize_series_quantiles(s1, n_bins=num_bins)
    else:
        x_codes = encode_discrete(s1)

    if col2 in continuous:
        y_codes = discretize_series_quantiles(s2, n_bins=num_bins)
    else:
        y_codes = encode_discrete(s2)

    pair_df = pd.DataFrame({"x": x_codes, "y": y_codes}).dropna()
    return pair_df["x"].astype(int), pair_df["y"].astype(int)


def normalized_mi_pair(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    continuous: set,
    num_bins: int = DEFAULT_NUM_BINS,
    normalization_method: str = "min",
) -> float:
    x_codes, y_codes = prepare_codes_for_pair(
        df=df,
        col1=col1,
        col2=col2,
        continuous=continuous,
        num_bins=num_bins,
    )
    if len(x_codes) == 0:
        return np.nan
    return normalized_mi_from_codes(x_codes, y_codes, method=normalization_method)


def correlation_matrix_nmi(
    df: pd.DataFrame,
    num_bins: int = DEFAULT_NUM_BINS,
    distinct_threshold_continuous: int = 10,
    normalization_method: str = "min",
    context_name: str = "preparation_correlations_nmi",
):
    """Calcule la matrice NMI (symétrique) et retourne (matrice, info)."""
    continuous, discrete = infer_variable_types(df, distinct_threshold_continuous)
    continuous_set = set(continuous)

    cols = df.columns.tolist()
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    errors = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            try:
                value = normalized_mi_pair(
                    df=df,
                    col1=c1,
                    col2=c2,
                    continuous=continuous_set,
                    num_bins=num_bins,
                    normalization_method=normalization_method,
                )
            except Exception as exc:  # garde-fou non bloquant
                value = np.nan
                errors.append({"var1": c1, "var2": c2, "error": str(exc)})

            mat.loc[c1, c2] = value
            mat.loc[c2, c1] = value

    info = {
        "continuous": continuous,
        "discrete": discrete,
        "errors": errors,
        "num_bins": num_bins,
        "normalization_method": normalization_method,
        "context_name": context_name,
    }
    return mat, info


def reorder_corr_matrix_by_target(
    corr_matrix: pd.DataFrame,
    target_variable: str,
    max_vars: int | None = None,
) -> pd.DataFrame:
    """
    Réordonne la matrice en plaçant la cible en premier, puis les variables
    triées par corrélation absolue décroissante avec la cible.
    max_vars permet de limiter le nombre de variables conservées (dont la cible).
    """
    if corr_matrix is None or corr_matrix.empty or target_variable not in corr_matrix.columns:
        return corr_matrix

    cols = list(corr_matrix.columns)
    cols.remove(target_variable)
    cols = [target_variable] + cols
    mat = corr_matrix[cols].loc[cols]

    target_corr = mat[target_variable].abs().sort_values(ascending=False)
    sorted_cols = [target_variable] + list(target_corr.index[1:])

    if max_vars and max_vars > 0:
        sorted_cols = sorted_cols[: max(1, max_vars)]

    return mat.loc[sorted_cols, sorted_cols]
