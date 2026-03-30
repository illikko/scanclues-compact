import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def run():
    pipeline_silent = bool(st.session_state.get("__PIPELINE_SILENT__", False))
    generate = bool(st.session_state.get("generate_distribution_figures", not pipeline_silent))

    st.session_state.setdefault("figs_variables_distribution_detailed", [])
    if not generate:
        st.session_state["figs_variables_distribution_detailed"] = []
        return

    df = st.session_state.get("df_ready")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.session_state["figs_variables_distribution_detailed"] = []
        return

    figures = []

    for variable in df.columns:
        series = df[variable].dropna()
        if series.empty:
            continue

        est_numerique = pd.api.types.is_numeric_dtype(series)
        valeurs_uniques = series.nunique()

        if est_numerique and valeurs_uniques > 10:
            # Gestion des bins inspirée de DistributionVariables
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
        figures.append({"title": str(variable), "png": buf.getvalue()})
        plt.close(fig)

    st.session_state["figs_variables_distribution_detailed"] = figures
    # Compatibilité avec les usages existants
    st.session_state["figs_variables_distribution"] = figures

