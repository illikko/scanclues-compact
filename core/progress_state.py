"""Helpers légers pour stocker l'avancement dans st.session_state."""

from __future__ import annotations

import streamlit as st


PROGRESS_PHASE_KEY = "progress_phase"
PROGRESS_VALUE_KEY = "progress_value"
PROGRESS_LABEL_KEY = "progress_label"


def reset_progress(phase: str, label: str = "") -> None:
    st.session_state[PROGRESS_PHASE_KEY] = phase
    st.session_state[PROGRESS_VALUE_KEY] = 0
    st.session_state[PROGRESS_LABEL_KEY] = label


def set_progress(value: float | int, label: str = "", phase: str | None = None) -> None:
    if phase is not None:
        st.session_state[PROGRESS_PHASE_KEY] = phase
    st.session_state[PROGRESS_VALUE_KEY] = max(0, min(100, int(value)))
    if label:
        st.session_state[PROGRESS_LABEL_KEY] = label


def get_progress() -> tuple[str | None, int, str]:
    return (
        st.session_state.get(PROGRESS_PHASE_KEY),
        int(st.session_state.get(PROGRESS_VALUE_KEY, 0) or 0),
        str(st.session_state.get(PROGRESS_LABEL_KEY, "") or ""),
    )
