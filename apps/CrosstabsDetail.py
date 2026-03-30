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

    # 1) Couples issus du Sankey (triés par V de Cramer décroissant)
    sankey_pairs = st.session_state.get("sankey_pair_results", {}) or {}
    sankey_sorted = []
    for _, item in sankey_pairs.items():
        try:
            vx, vy = str(item.get("var_x")), str(item.get("var_y"))
            v_val = float(item.get("v") or 0)
            if vx and vy:
                sankey_sorted.append((v_val, vx, vy))
        except Exception:
            continue
    sankey_sorted.sort(key=lambda t: t[0], reverse=True)

    # 2) Paires directes cibles/illustratives (niveau 1) en complément
    direct_pairs = []
    for vx in targets:
        for vy in illustratives:
            if vx == vy:
                continue
            direct_pairs.append((None, vx, vy))

    # 3) Fusion en préservant l'ordre (Sankey d'abord) et unicité
    merged = []
    seen = set()
    for entry in sankey_sorted + direct_pairs:
        _, vx, vy = entry
        key = (vx, vy)
        if key in seen:
            continue
        seen.add(key)
        merged.append((vx, vy))

    max_pairs = 20  # garde-fou perf
    crosstabs = []
    for vx, vy in merged[:max_pairs]:
        try:
            ct_count, ct_pct_row, std_res = crosstab_with_std_residuals(df, vx, vy)
            base_interpretation = interpret_crosstab_with_llm(vx, vy, ct_pct_row, std_res)
            heatmap_png = crosstab_heatmap_png(
                ct_pct_row,
                std_res,
                title=f"{vx} vs {vy}",
            )
            crosstabs.append({
                "var_x": vx,
                "var_y": vy,
                "interpretation": base_interpretation,
                "heatmap_png": heatmap_png,
                "ct_pct_row": ct_pct_row.to_dict(),
            })
        except Exception:
            continue

    st.session_state["crosstabs_interpretation"] = crosstabs
    st.session_state["crosstabs_generated"] = bool(crosstabs)
