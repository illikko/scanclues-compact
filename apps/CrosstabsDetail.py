import pandas as pd
import streamlit as st

from .DiagramSankey import (
    crosstab_with_std_residuals,
    interpret_crosstab_with_llm,
    crosstab_heatmap_png,
)


def run():
    pipeline_silent = bool(st.session_state.get("__PIPELINE_SILENT__", False))
    run_flag = bool(
        st.session_state.get("run_sankey_crosstabs", not pipeline_silent)
        or st.session_state.get("__QA_FORCE_CROSSTABS__", False)
        or st.session_state.get("__BRIEF_FORCE_CROSSTABS__", False)
    )

    st.session_state.setdefault("crosstabs_interpretation", [])
    st.session_state.setdefault("crosstabs_generated", False)

    if not run_flag:
        st.session_state["crosstabs_interpretation"] = []
        st.session_state["crosstabs_generated"] = False
        return

    df = st.session_state.get("df_ready")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.session_state["crosstabs_generated"] = False
        return

    targets = list(st.session_state.get("target_variables", []) or [])
    illustratives = list(st.session_state.get("illustrative_variables", []) or [])

    brief_tv = st.session_state.get("brief_target_variable")
    if brief_tv and brief_tv in targets:
        targets = [brief_tv] + [t for t in targets if t != brief_tv]

    if not targets or not illustratives:
        st.session_state["crosstabs_generated"] = False
        return

    crosstabs = []
    max_pairs = 6  # garde-fou perf
    pair_count = 0

    for var_x in targets:
        for var_y in illustratives:
            if var_x == var_y:
                continue
            if pair_count >= max_pairs:
                break
            try:
                ct_count, ct_pct_row, std_res = crosstab_with_std_residuals(df, var_x, var_y)
                base_interpretation = interpret_crosstab_with_llm(var_x, var_y, ct_pct_row, std_res)
                heatmap_png = crosstab_heatmap_png(
                    ct_pct_row,
                    std_res,
                    title=f"{var_x} vs {var_y}",
                )
                crosstabs.append({
                    "var_x": var_x,
                    "var_y": var_y,
                    "interpretation": base_interpretation,
                    "heatmap_png": heatmap_png,
                    "ct_pct_row": ct_pct_row.to_dict(),
                })
                pair_count += 1
            except Exception:
                continue

    st.session_state["crosstabs_interpretation"] = crosstabs
    st.session_state["crosstabs_generated"] = bool(crosstabs)
