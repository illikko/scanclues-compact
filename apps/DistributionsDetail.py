import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def _summarize_distribution(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "Série vide."

    if pd.api.types.is_numeric_dtype(s):
        desc = s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        skew = s.skew()
        kurt = s.kurt()
        zero_pct = (s == 0).mean() * 100
        return (
            f"nb={int(desc['count'])}, moy={desc['mean']:.2f}, écart-type={desc['std']:.2f}, "
            f"min={desc['min']:.2f}, p10={desc['10%']:.2f}, médiane={desc['50%']:.2f}, "
            f"p90={desc['90%']:.2f}, max={desc['max']:.2f}, skew={skew:.2f}, "
            f"kurt={kurt:.2f}, %0={zero_pct:.1f}%"
        )

    counts = s.value_counts(normalize=True)
    gini = 1 - (counts ** 2).sum()
    top = counts.head(5).apply(lambda x: f"{x * 100:.1f}%").to_dict()
    return f"modalités={len(counts)}, Gini={gini:.2f}, top5={top}"


def run():
    generate = bool(
        st.session_state.get("generate_distribution_figures", False)
        or st.session_state.get("__QA_FORCE_DISTRIBUTIONS__", False)
    )

    st.session_state.setdefault("figs_variables_distribution_detailed", [])
    if not generate:
        st.session_state["figs_variables_distribution_detailed"] = []
        return

    df = st.session_state.get("df_ready")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.session_state["figs_variables_distribution_detailed"] = []
        return

    requested_vars = st.session_state.get("__QA_SELECTED_DISTRIBUTION_VARS__", []) or []
    variables = [str(v) for v in requested_vars if str(v) in df.columns] if requested_vars else list(df.columns)

    figures = []

    for variable in variables:
        series = df[variable].dropna()
        if series.empty:
            continue

        est_numerique = pd.api.types.is_numeric_dtype(series)
        valeurs_uniques = series.nunique()

        if est_numerique and valeurs_uniques > 10:
            if valeurs_uniques < 30:
                est_entiere = np.all(np.equal(series, series.astype(int)))
                if est_entiere:
                    val_min = int(series.min())
                    val_max = int(series.max())
                    bins = max(1, val_max - val_min + 1)
                else:
                    bins = int(valeurs_uniques)
            else:
                bins = 50

            fig, ax = plt.subplots()
            ax.hist(series, bins=bins, edgecolor="black")
            ax.set_title(f"{variable}")
            ax.set_ylabel("Occurrences")
        else:
            counts = (
                series
                .value_counts(normalize=True, dropna=True)
                .mul(100)
                .sort_values(ascending=False)
            )
            if counts.empty:
                continue

            fig, ax = plt.subplots()
            counts.plot(kind="barh", ax=ax, color="#4e79a7")
            ax.set_xlabel("Fréquence (%)")
            ax.set_title(f"{variable}")
            ax.invert_yaxis()

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        figures.append({
            "title": str(variable),
            "png": buf.getvalue(),
            "metrics_caption": _summarize_distribution(series),
        })
        plt.close(fig)

    st.session_state["figs_variables_distribution_detailed"] = figures
    st.session_state["figs_variables_distribution"] = figures
