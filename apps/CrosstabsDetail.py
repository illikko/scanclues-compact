import pandas as pd
import streamlit as st

from core.crosstab_utils import (
    cramers_v,
    crosstab_with_std_residuals,
    crosstab_heatmap_png,
    interpret_crosstab_with_llm,
)


def _unique_pairs(pairs) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen = set()
    for item in pairs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        var_x, var_y = str(item[0]), str(item[1])
        if not var_x or not var_y or var_x == var_y:
            continue
        key = (var_x, var_y)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _lookup_metrics(df_links: pd.DataFrame | None, var_x: str, var_y: str) -> dict:
    if isinstance(df_links, pd.DataFrame) and not df_links.empty:
        mask = (
            ((df_links["var_x"] == var_x) & (df_links["var_y"] == var_y))
            | ((df_links["var_x"] == var_y) & (df_links["var_y"] == var_x))
        )
        row = df_links.loc[mask].head(1)
        if not row.empty:
            rec = row.iloc[0].to_dict()
            return {
                "pair_id": rec.get("pair_id"),
                "v": rec.get("v"),
                "p": rec.get("p"),
                "chi2": rec.get("chi2"),
            }
    return {"pair_id": None, "v": None, "p": None, "chi2": None}


def _compute_metrics(df: pd.DataFrame, var_x: str, var_y: str) -> dict:
    chi2, p, v = cramers_v(df[var_x], df[var_y])
    return {"pair_id": None, "v": v, "p": p, "chi2": chi2}


def _pair_cache_key(var_x: str, var_y: str) -> str:
    left, right = sorted([str(var_x), str(var_y)])
    return f"{left}__{right}"


def _build_metrics_caption(metrics: dict) -> str:
    parts: list[str] = []
    try:
        if metrics.get("v") is not None and pd.notna(metrics.get("v")):
            parts.append(f"V de Cramer : {float(metrics['v']):.3f}")
    except Exception:
        pass
    try:
        if metrics.get("p") is not None and pd.notna(metrics.get("p")):
            parts.append(f"p-value : {float(metrics['p']):.5f}")
    except Exception:
        pass
    try:
        if metrics.get("chi2") is not None and pd.notna(metrics.get("chi2")):
            parts.append(f"Khi2 : {float(metrics['chi2']):.3f}")
    except Exception:
        pass
    return " | ".join(parts)


def _build_entry(
    df: pd.DataFrame,
    var_x: str,
    var_y: str,
    *,
    metrics: dict,
    results_store: dict,
):
    pair_id = metrics.get("pair_id")
    cache_key = _pair_cache_key(var_x, var_y)
    if cache_key in results_store:
        cached = results_store[cache_key]
        cached.setdefault("pair_id", pair_id)
        cached.setdefault("var_x", var_x)
        cached.setdefault("var_y", var_y)
        cached.setdefault("v", metrics.get("v"))
        cached.setdefault("p", metrics.get("p"))
        cached.setdefault("chi2", metrics.get("chi2"))
        cached.setdefault("metrics_caption", _build_metrics_caption(cached))
        return cached

    ct_count, ct_pct_row, std_res = crosstab_with_std_residuals(df, var_x, var_y)
    if ct_pct_row is None or std_res is None:
        return None

    entry = {
        "pair_id": pair_id,
        "var_x": var_x,
        "var_y": var_y,
        "v": metrics.get("v"),
        "p": metrics.get("p"),
        "chi2": metrics.get("chi2"),
        "metrics_caption": _build_metrics_caption(metrics),
        "ct_count": ct_count,
        "ct_pct_row": ct_pct_row,
        "std_res": std_res,
        "heatmap_png": crosstab_heatmap_png(ct_pct_row, std_res, title=f"{var_x} vs {var_y}"),
        "interpretation": interpret_crosstab_with_llm(var_x, var_y, ct_pct_row, std_res),
    }
    results_store[cache_key] = entry
    return entry


def _pairs_from_sankey() -> list[tuple[str, str]]:
    drawn_links_df = st.session_state.get("sankey_drawn_links_df")
    if not isinstance(drawn_links_df, pd.DataFrame) or drawn_links_df.empty:
        return []
    pairs = [(str(row["source_var"]), str(row["target_var"])) for _, row in drawn_links_df.iterrows()]
    return _unique_pairs(pairs)


def run():
    pipeline_silent = bool(st.session_state.get("__PIPELINE_SILENT__", False))
    run_flag = bool(
        st.session_state.get("run_sankey_crosstabs", False)
        or st.session_state.get("__QA_FORCE_CROSSTABS__", False)
        or st.session_state.get("__BRIEF_FORCE_CROSSTABS__", False)
    )

    st.session_state.setdefault("crosstabs_interpretation", [])
    st.session_state.setdefault("crosstabs_generated", False)
    st.session_state.setdefault("sankey_pair_results", {})

    if not run_flag:
        st.session_state["crosstabs_interpretation"] = []
        st.session_state["sankey_pair_results"] = {}
        st.session_state["crosstabs_generated"] = False
        return

    df = st.session_state.get("df_ready")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.session_state["crosstabs_generated"] = False
        return

    qa_mode = bool(st.session_state.get("__QA_FORCE_CROSSTABS__", False))
    requested_pairs = st.session_state.get("__QA_SELECTED_CROSSTAB_PAIRS__", []) or []
    pairs_to_run = _unique_pairs(requested_pairs) if qa_mode and requested_pairs else _pairs_from_sankey()
    if not pairs_to_run:
        st.session_state["crosstabs_interpretation"] = []
        st.session_state["sankey_pair_results"] = {}
        st.session_state["crosstabs_generated"] = False
        return

    relevant_links = st.session_state.get("sankey_drawn_links_df")
    existing_results_store = st.session_state.get("sankey_pair_results", {}) or {}
    next_results_store = {}
    crosstabs = []
    for var_x, var_y in pairs_to_run:
        if var_x not in df.columns or var_y not in df.columns:
            continue
        try:
            metrics = _lookup_metrics(relevant_links, var_x, var_y)
            if metrics.get("v") is None or metrics.get("p") is None or metrics.get("chi2") is None:
                metrics = _compute_metrics(df, var_x, var_y)
            entry = _build_entry(
                df,
                var_x,
                var_y,
                metrics=metrics,
                results_store=existing_results_store,
            )
            if entry is not None:
                cache_key = _pair_cache_key(var_x, var_y)
                next_results_store[cache_key] = entry
                crosstabs.append(entry)
        except Exception:
            continue

    st.session_state["sankey_pair_results"] = next_results_store
    st.session_state["crosstabs_interpretation"] = crosstabs
    st.session_state["crosstabs_generated"] = bool(crosstabs)
