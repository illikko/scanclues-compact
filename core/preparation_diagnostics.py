from __future__ import annotations

from typing import Any

import streamlit as st


PREPARATION_DIAGNOSTICS_KEY = "preparation_diagnostics"


def get_preparation_diagnostics() -> dict[str, dict[str, Any]]:
    diagnostics = st.session_state.get(PREPARATION_DIAGNOSTICS_KEY)
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        st.session_state[PREPARATION_DIAGNOSTICS_KEY] = diagnostics
    return diagnostics


def get_preparation_diagnostic(diag_id: str) -> dict[str, Any] | None:
    diagnostics = get_preparation_diagnostics()
    value = diagnostics.get(str(diag_id))
    return value if isinstance(value, dict) else None


def set_preparation_diagnostic(diagnostic: dict[str, Any]) -> dict[str, Any]:
    diag_id = str((diagnostic or {}).get("id", "")).strip()
    if not diag_id:
        raise ValueError("diagnostic['id'] is required")
    diagnostics = get_preparation_diagnostics()
    diagnostics[diag_id] = diagnostic
    st.session_state[PREPARATION_DIAGNOSTICS_KEY] = diagnostics
    return diagnostic


def clear_preparation_diagnostics() -> None:
    st.session_state[PREPARATION_DIAGNOSTICS_KEY] = {}


def remove_preparation_diagnostic(diag_id: str) -> None:
    diagnostics = get_preparation_diagnostics()
    diagnostics.pop(str(diag_id), None)
    st.session_state[PREPARATION_DIAGNOSTICS_KEY] = diagnostics
