import numpy as np
import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from openai import OpenAI
from scipy.stats import chi2_contingency

from core.correlations_utils import (
    DEFAULT_NUM_BINS,
    discretize_series_quantiles,
    fill_missing_for_discrete,
)
from utils import discretize_continuous_variables


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def cramers_v(x, y):
    data = pd.crosstab(x, y)
    if data.empty or data.shape[0] < 2 or data.shape[1] < 2:
        return np.nan, np.nan, np.nan

    chi2, p, dof, expected = chi2_contingency(data)
    n = data.values.sum()
    phi2 = chi2 / n
    r, k = data.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
    r_corr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    k_corr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    if r_corr <= 1 or k_corr <= 1:
        v = 0
    else:
        v = np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))
    return chi2, p, v


def crosstab_with_std_residuals(df_cat, var_x, var_y):
    if var_x not in df_cat.columns or var_y not in df_cat.columns:
        return pd.DataFrame(), None, None

    params = {
        "num_quantiles": st.session_state.get("num_quantiles", DEFAULT_NUM_BINS),
        "mod_freq_min": st.session_state.get("mod_freq_min", 0.9),
        "distinct_threshold_continuous": st.session_state.get("distinct_threshold_continuous", 5),
    }

    df_local, _info = discretize_continuous_variables(
        df_cat[[var_x, var_y]],
        num_quantiles=params["num_quantiles"],
        mod_freq_min=params["mod_freq_min"],
        distinct_threshold_continuous=params["distinct_threshold_continuous"],
        context_name="crosstab_with_std_residuals",
    )

    if not isinstance(df_local, pd.DataFrame):
        df_local = df_cat[[var_x, var_y]].copy()

    ct_count = pd.crosstab(df_local[var_x], df_local[var_y])
    if ct_count.empty:
        return ct_count, None, None

    chi2, p, dof, expected = chi2_contingency(ct_count)
    std_res = (ct_count - expected) / np.sqrt(expected)
    ct_pct_row = pd.crosstab(df_local[var_x], df_local[var_y], normalize="index") * 100
    return ct_count, ct_pct_row, std_res


def interpret_crosstab_with_llm(var_x, var_y, ct_pct_row, std_res):
    table_pct = ct_pct_row.round(1).to_string()
    table_res = std_res.round(2).to_string()

    crosstab_interpretation = f"""Réponds en français, clair et concis
Tu es un data analyst expert (statistique + sémantique), en marketing si le jeu de données est un questionnaire.
On te donne 3 tableaux :
- un de contingence entre deux variables :
    - Variable X : {var_x}
    - Variable Y : {var_y}
- un tableau des pourcentages en ligne (% par modalité de X) :
{table_pct}
- un tableau des résidus standardisés du Khi² (positif = sur-représentation, négatif = sous-représentation) :
{table_res}

Tâche :
- Explique en 3 phrases les principaux résultats : quelles modalités de X sont particulièrement liées à quelles modalités de Y ?
- utilise l'analyse déjà faite dans {st.session_state.get("dataset_context", "")}, en particulier pour comprendre la signification des variables et le contexte global.
- N'utilise pas de termes techniques (résidus, Chi2, Khi², etc.), explique juste les relations entre modalités/variables, pas les tableaux.
- Ne fais pas d'introduction ni de résumé.
- Garde une explication concise et structurée.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": crosstab_interpretation},
            {"role": "user", "content": "réponds en fonction des instructions données dans role/system."},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        return "(Interprétation indisponible – réponse LLM vide)"
    return content


def crosstab_heatmap_png(
    frequencies,
    residuals,
    threshold: float = 2.0,
    title: str | None = None,
):
    row_labels = frequencies.index.astype(str).tolist()
    col_labels = frequencies.columns.astype(str).tolist()

    max_col_len = max((len(lbl) for lbl in col_labels), default=0)
    max_row_len = max((len(lbl) for lbl in row_labels), default=0)

    if max_col_len > max_row_len:
        frequencies = frequencies.T.copy()
        residuals = residuals.T.copy()
        row_labels, col_labels = col_labels, row_labels
        if title and "×" in title:
            left, right = title.split("×", 1)
            title = f"{right.strip()} × {left.strip()}"

    sign_map = np.zeros_like(frequencies.values, dtype=int)
    sign_map[residuals.values > threshold] = 1
    sign_map[residuals.values < -threshold] = -1

    cmap = ListedColormap(["#fcb6b6", "white", "#b6fcb6"])
    n_rows, n_cols = frequencies.shape

    fig_width = max(4, n_cols * 0.9)
    fig_height = max(3, n_rows * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(sign_map, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    for i in range(n_rows):
        for j in range(n_cols):
            val = frequencies.iat[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="black")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


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
