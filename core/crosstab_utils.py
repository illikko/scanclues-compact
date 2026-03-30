import pandas as pd
import numpy as np

from core.correlations_utils import (
    discretize_series_quantiles,
    fill_missing_for_discrete,
)
from utils import discretize_continuous_variables


def discretize_for_display(s: pd.Series, n_bins: int):
    """
    Discrétise une série numérique en quantiles pour affichage lisible.
    Fallback: rangs en quantiles si qcut échoue, puis quantiles discrets.
    """
    s_num = fill_missing_for_discrete(s)
    try:
        cat = pd.qcut(pd.to_numeric(s_num, errors="coerce"), q=max(2, n_bins), duplicates="drop")
        return cat.astype(str).fillna("Manquant")
    except Exception:
        try:
            ranks = pd.to_numeric(s_num, errors="coerce").rank(method="average")
            cat = pd.qcut(ranks, q=max(2, n_bins), duplicates="drop")
            return cat.astype(str).fillna("Manquant")
        except Exception:
            return discretize_series_quantiles(s, n_bins=n_bins).astype(str)


def summarize_crosstab(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
    *,
    num_quantiles: int,
    mod_freq_min: float,
    distinct_threshold_continuous: int,
    top: int = 5,
    crosstab_fn=None,
    heatmap_fn=None,
    interpretation_fn=None,
):
    """
    Prépare et résume un crosstab (discrétisation incluse).
    crosstab_fn doit renvoyer (ct_count, ct_pct_row, std_res).
    heatmap_fn(ct_pct_row, std_res) -> image/bytes (optionnel).
    interpretation_fn(var_a, var_b, ct_pct_row, std_res) -> str (optionnel).
    """
    n_bins = max(3, min(12, int(num_quantiles)))
    df_subset = df[[var_a, var_b]].copy()
    df_prepared, _info = discretize_continuous_variables(
        df_subset,
        num_quantiles=num_quantiles,
        mod_freq_min=mod_freq_min,
        distinct_threshold_continuous=distinct_threshold_continuous,
        context_name="qa_crosstab",
    )

    s1 = df_prepared[var_a] if isinstance(df_prepared, pd.DataFrame) else df_subset[var_a]
    s2 = df_prepared[var_b] if isinstance(df_prepared, pd.DataFrame) else df_subset[var_b]

    if pd.api.types.is_numeric_dtype(s1):
        s1 = discretize_for_display(s1, n_bins=n_bins)
    else:
        s1 = fill_missing_for_discrete(s1).astype(str)
    if pd.api.types.is_numeric_dtype(s2):
        s2 = discretize_for_display(s2, n_bins=n_bins)
    else:
        s2 = fill_missing_for_discrete(s2).astype(str)

    ct = pd.crosstab(s1, s2)
    top_a = s1.value_counts().head(top).index
    top_b = s2.value_counts().head(top).index
    ct = ct.loc[ct.index.isin(top_a), ct.columns.isin(top_b)]
    total = ct.sum().sum()
    row_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    col_pct = (ct.div(ct.sum(axis=0), axis=1) * 100).round(1)

    assoc = ""
    try:
        chi = pd.crosstab(df[var_a], df[var_b])
        chi_val = chi.values
        chi2 = (((chi_val - chi_val.sum(axis=1)[:, None] * chi_val.sum(axis=0)[None, :] / chi_val.sum()) ** 2) /
                (chi_val.sum(axis=1)[:, None] * chi_val.sum(axis=0)[None, :] / chi_val.sum())).sum()
        n = chi_val.sum()
        phi2 = chi2 / max(n, 1)
        r, c = chi_val.shape
        from math import sqrt

        cramerv = sqrt(phi2 / max(min(r - 1, c - 1), 1))
        assoc = f"Force d'association (approx.) : Cramér V = {cramerv:.2f}"
    except Exception:
        pass

    heatmap_png = None
    ct_pct_row_full = None
    std_res = None
    if crosstab_fn:
        try:
            ct_count, ct_pct_row_full, std_res = crosstab_fn(
                df.assign(**{var_a: s1, var_b: s2}),
                var_a,
                var_b,
            )
            if heatmap_fn and ct_pct_row_full is not None and std_res is not None:
                heatmap_png = heatmap_fn(ct_pct_row_full, std_res, title=f"{var_a} vs {var_b}")
        except Exception:
            pass

    interp = None
    if interpretation_fn and isinstance(ct_pct_row_full, pd.DataFrame) and isinstance(std_res, pd.DataFrame):
        try:
            interp = interpretation_fn(var_a, var_b, ct_pct_row_full, std_res)
        except Exception:
            interp = None

    summary = f"Tableau croisé (top {top} modalités) {var_a} x {var_b} — total {int(total)} obs."
    try:
        row_md = row_pct.reset_index().rename(columns={"index": var_a}).to_markdown(index=False)
        col_md = col_pct.reset_index().rename(columns={"index": var_a}).to_markdown(index=False)
    except Exception:
        row_md = row_pct.to_string()
        col_md = col_pct.to_string()

    return {
        "summary": summary,
        "row_pct_md": row_md,
        "col_pct_md": col_md,
        "assoc": assoc,
        "row_df": row_pct.reset_index(),
        "col_df": col_pct.reset_index(),
        "vars": (var_a, var_b),
        "heatmap_png": heatmap_png,
        "interpretation": interp,
    }
