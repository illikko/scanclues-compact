from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
import streamlit as st

REGISTRY_KEY = "__DF_REGISTRY__"
ALIAS_MAP_KEY = "__DF_ALIAS_MAP__"
HISTORY_KEY = "__DF_HISTORY__"


class DFState(str, Enum):
    RAW = "RAW"
    VERBATIM_READY = "VERBATIM_READY"
    SHORT_LABELS = "SHORT_LABELS"
    MULTI_ORD_DONE = "MULTI_ORD_DONE"
    MULTI_DONE = "MULTI_DONE"
    IMPUTED_STRUCTURAL = "IMPUTED_STRUCTURAL"
    CLEAN = "CLEAN"
    EX_CORR = "EX_CORR"
    READY = "READY"
    ENCODED = "ENCODED"
    ACTIVE = "ACTIVE"
    ILLUSTRATIVE = "ILLUSTRATIVE"
    OUTLIERS = "OUTLIERS"
    SELECTED = "SELECTED"
    NEAT = "NEAT"
    SEMANTIC_TYPES = "SEMANTIC_TYPES"


DEFAULT_ALIAS_MAP = {
    "df_raw": DFState.RAW.value,
    "df_ex_verbatim": DFState.VERBATIM_READY.value,
    "df_shortlabels": DFState.SHORT_LABELS.value,
    "df_shortened_labels": DFState.SHORT_LABELS.value,
    "df_ex_ordonnees": DFState.MULTI_ORD_DONE.value,
    "df_ex_multiples": DFState.MULTI_DONE.value,
    "df_imputed_structural": DFState.IMPUTED_STRUCTURAL.value,
    "df_clean": DFState.CLEAN.value,
    "df_ex_corr": DFState.EX_CORR.value,
    "df_ready": DFState.READY.value,
    "df_encoded": DFState.ENCODED.value,
    "df_active": DFState.ACTIVE.value,
    "df_illustrative": DFState.ILLUSTRATIVE.value,
    "df_outliers_sorted": DFState.OUTLIERS.value,
    "df_selected": DFState.SELECTED.value,
    "df_neat": DFState.NEAT.value,
    "df_semantic_types": DFState.SEMANTIC_TYPES.value,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ss(session_state: Any = None):
    return st.session_state if session_state is None else session_state


def _to_state_name(state_or_alias: DFState | str, alias_map: dict[str, str]) -> str:
    if isinstance(state_or_alias, DFState):
        return state_or_alias.value
    if state_or_alias in alias_map:
        return alias_map[state_or_alias]
    return str(state_or_alias).upper()


def init_df_registry(session_state: Any = None) -> dict[str, Any]:
    ss = _ss(session_state)
    ss.setdefault(REGISTRY_KEY, {})
    ss.setdefault(ALIAS_MAP_KEY, dict(DEFAULT_ALIAS_MAP))
    ss.setdefault(HISTORY_KEY, [])

    # Hydrate from legacy aliases if registry is empty/incomplete.
    sync_registry_from_aliases(session_state=ss)
    # Keep legacy keys alive for untouched modules.
    sync_aliases_from_registry(session_state=ss)
    return ss[REGISTRY_KEY]


def register_alias(alias_key: str, state: DFState | str, session_state: Any = None) -> None:
    ss = _ss(session_state)
    init_df_registry(session_state=ss)
    alias_map = ss[ALIAS_MAP_KEY]
    alias_map[str(alias_key)] = _to_state_name(state, alias_map)


def sync_registry_from_aliases(session_state: Any = None) -> dict[str, Any]:
    ss = _ss(session_state)
    ss.setdefault(REGISTRY_KEY, {})
    ss.setdefault(ALIAS_MAP_KEY, dict(DEFAULT_ALIAS_MAP))
    reg = ss[REGISTRY_KEY]
    alias_map = ss[ALIAS_MAP_KEY]

    for alias_key, state_name in alias_map.items():
        value = ss.get(alias_key, None)
        if value is not None:
            reg[state_name] = value
    return reg


def sync_aliases_from_registry(session_state: Any = None) -> None:
    ss = _ss(session_state)
    ss.setdefault(REGISTRY_KEY, {})
    ss.setdefault(ALIAS_MAP_KEY, dict(DEFAULT_ALIAS_MAP))
    reg = ss[REGISTRY_KEY]
    alias_map = ss[ALIAS_MAP_KEY]

    for alias_key, state_name in alias_map.items():
        value = reg.get(state_name, None)
        if value is not None:
            ss[alias_key] = value


def get_df(
    state_or_alias: DFState | str,
    *,
    default: Any = None,
    required: bool = False,
    session_state: Any = None,
):
    ss = _ss(session_state)
    init_df_registry(session_state=ss)
    alias_map = ss[ALIAS_MAP_KEY]
    reg = ss[REGISTRY_KEY]
    state_name = _to_state_name(state_or_alias, alias_map)

    if state_name in reg and reg[state_name] is not None:
        return reg[state_name]

    # Fallback legacy read.
    if isinstance(state_or_alias, str) and state_or_alias in ss and ss.get(state_or_alias) is not None:
        reg[state_name] = ss[state_or_alias]
        return ss[state_or_alias]

    if required:
        raise KeyError(f"DataFrame state '{state_name}' is required but missing.")
    return default


def set_df(
    state_or_alias: DFState | str,
    df: pd.DataFrame | None,
    *,
    step_name: str | None = None,
    session_state: Any = None,
) -> pd.DataFrame | None:
    ss = _ss(session_state)
    init_df_registry(session_state=ss)
    alias_map = ss[ALIAS_MAP_KEY]
    reg = ss[REGISTRY_KEY]
    state_name = _to_state_name(state_or_alias, alias_map)
    reg[state_name] = df

    # Backward compatibility for untouched modules.
    sync_aliases_from_registry(session_state=ss)

    # Traceability.
    rows = cols = None
    if isinstance(df, pd.DataFrame):
        rows, cols = df.shape
    ss[HISTORY_KEY].append(
        {
            "ts_utc": _now_iso(),
            "state": state_name,
            "rows": rows,
            "cols": cols,
            "step_name": step_name or "",
        }
    )
    return df


def get_df_history(session_state: Any = None) -> list[dict[str, Any]]:
    ss = _ss(session_state)
    init_df_registry(session_state=ss)
    return list(ss[HISTORY_KEY])
