from typing import Any

import pandas as pd
import streamlit as st

from core.preparation_diagnostics import get_preparation_diagnostics


def _df_preview(df: Any, rows: int = 50) -> list[dict[str, Any]] | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return df.head(rows).astype(object).where(pd.notnull(df.head(rows)), None).to_dict(orient="records")


def _df_columns(df: Any) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        return []
    return [str(col) for col in df.columns.tolist()]


def build_preparation_details_payload(session_state: Any) -> dict[str, Any]:
    diagnostics = get_preparation_diagnostics()
    shortened_labels_mapping = session_state.get("shortened_labels_mapping")
    df_semantic_types = session_state.get("df_semantic_types")
    missing_df = session_state.get("missing_df")
    missing_diagnostic = session_state.get("missing_diagnostic")
    candidates_df = session_state.get("candidates_df")
    columns_infos = session_state.get("columns_infos")
    df_outliers_sorted = session_state.get("df_outliers_sorted")
    preparation2_details = session_state.get("preparation2_details")
    process = session_state.get("process")

    payload = {
        "diagnostics": diagnostics,
        "label_shortening": {
            "mapping_columns": _df_columns(shortened_labels_mapping),
            "mapping_preview": _df_preview(shortened_labels_mapping),
            "semantic_types_columns": _df_columns(df_semantic_types),
            "semantic_types_preview": _df_preview(df_semantic_types),
        },
        "missing_values": {
            "diagnostic": missing_diagnostic,
            "table_columns": _df_columns(missing_df),
            "table_preview": _df_preview(missing_df),
            "little_test_result": session_state.get("little_test_result"),
        },
        "structural_missing": {
            "diagnostic": diagnostics.get("structural_missing"),
            "candidates_columns": _df_columns(candidates_df),
            "candidates_preview": _df_preview(candidates_df),
            "columns_infos_preview": _df_preview(columns_infos),
        },
        "outliers": {
            "removed": bool(session_state.get("outliers_removed", False)),
            "indices": list(session_state.get("outliers_indices", []) or []),
            "table_columns": _df_columns(df_outliers_sorted),
            "table_preview": _df_preview(df_outliers_sorted),
        },
        "preparation2": {
            "details": preparation2_details if isinstance(preparation2_details, dict) else {},
        },
        "process": {
            "columns": _df_columns(process),
            "preview": _df_preview(process),
        },
    }
    return payload


def refresh_preparation_details_payload() -> dict[str, Any]:
    payload = build_preparation_details_payload(st.session_state)
    st.session_state["preparation_details_payload"] = payload
    st.session_state["preparation_details_ready"] = True
    return payload
