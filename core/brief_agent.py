from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.analysis_capabilities import (
    DISTRIB_KEYWORDS,
    RELATION_KEYWORDS,
    get_analysis_capability_catalog,
    get_module_catalog,
)


def _has_context_ready() -> bool:
    return (
        st.session_state.get("dataset_object") is not None
        and st.session_state.get("dataset_context") is not None
        and st.session_state.get("dataset_recommendations") is not None
    )


def _match_columns(df: pd.DataFrame, brief: str) -> list[str]:
    brief_low = brief.lower()
    matches: list[str] = []
    for col in df.columns:
        name = str(col)
        if name.lower() in brief_low:
            matches.append(name)
    return matches


def resolve_brief_analysis_plan(df_source: pd.DataFrame | None) -> dict[str, Any]:
    """Résout le brief utilisateur sans appel LLM et sans modifier session_state.

    Le brief n'est pas un orchestrateur concurrent : il produit uniquement un
    contexte d'analyse déterministe que PipelineRunner applique ensuite.
    """
    mode = st.session_state.get("dataset_key_questions_mode", "sb")
    brief = str(st.session_state.get("dataset_key_questions_value", "") or "")

    empty = {
        "active": False,
        "target": None,
        "illustratives": [],
        "run_sankey_crosstabs": False,
        "generate_distribution_figures": False,
        "plan": [],
        "reason": "brief inactif",
    }
    if mode != "ab" or not brief.strip():
        return empty
    if not _has_context_ready():
        return {**empty, "reason": "cadrage indisponible"}
    if not isinstance(df_source, pd.DataFrame) or df_source.empty:
        return {**empty, "reason": "dataset indisponible"}

    cols = _match_columns(df_source, brief)
    brief_low = brief.lower()
    target = cols[0] if cols else None
    illustratives = [c for c in cols[1:] if c != target]
    rel_flag = any(k in brief_low for k in RELATION_KEYWORDS) or len(cols) >= 2
    dist_flag = any(k in brief_low for k in DISTRIB_KEYWORDS)

    plan: list[dict[str, Any]] = []
    if rel_flag:
        plan.append({
            "module": "CrosstabsDetail",
            "params": {"run_sankey_crosstabs": True},
            "reason": "relation ou comparaison identifiée dans le brief",
        })
        plan.append({
            "module": "DiagramSankey",
            "params": {"target": target, "profiles": illustratives},
            "reason": "relations principales à visualiser",
        })
    if dist_flag:
        plan.append({
            "module": "DistributionsDetail",
            "params": {"generate_distribution_figures": True},
            "reason": "demande de distribution ou de répartition",
        })
    if target:
        plan.append({
            "module": "Profils_y",
            "params": {"target": target, "profiles": illustratives},
            "reason": "cible détectée explicitement dans le brief",
        })

    return {
        "active": bool(plan or target or rel_flag or dist_flag),
        "target": target,
        "illustratives": illustratives,
        "run_sankey_crosstabs": rel_flag,
        "generate_distribution_figures": dist_flag,
        "plan": plan,
        "reason": "variable cible détectée dans le brief" if target else "aucune variable explicite détectée",
    }


def apply_brief_analysis_plan(resolved: dict[str, Any]) -> None:
    """Applique le contexte de brief dans un périmètre de clés documenté."""
    if not resolved.get("active"):
        st.session_state["brief_relevance"] = False
        st.session_state["brief_reason"] = resolved.get("reason", "brief inactif")
        return

    target = resolved.get("target")
    illustratives = list(resolved.get("illustratives") or [])

    if target:
        st.session_state["brief_target_variable"] = target
        tv = list(st.session_state.get("target_variables", []) or [])
        st.session_state["target_variables"] = [target] + [t for t in tv if t != target]

    if illustratives:
        st.session_state["brief_illustrative_variables"] = illustratives
        iv = list(st.session_state.get("illustrative_variables", []) or [])
        st.session_state["illustrative_variables"] = illustratives + [
            i for i in iv if i not in illustratives and i != target
        ]

    if resolved.get("run_sankey_crosstabs"):
        st.session_state["run_sankey_crosstabs"] = True
    if resolved.get("generate_distribution_figures"):
        st.session_state["generate_distribution_figures"] = True

    plan = list(resolved.get("plan") or [])
    st.session_state["brief_analysis_plan"] = plan
    st.session_state["brief_results_synthesis"] = "\n".join(
        [f"- {p.get('module')} ({p.get('reason', '')})" for p in plan]
    )
    st.session_state["brief_relevance"] = bool(plan)
    st.session_state["brief_plan_ready"] = bool(plan)
    st.session_state["brief_reason"] = resolved.get("reason", "")


def run_brief_agent(df_source: pd.DataFrame | None) -> None:
    """Compatibilité : résolution déterministe puis application par PipelineRunner."""
    apply_brief_analysis_plan(resolve_brief_analysis_plan(df_source))


__all__ = [
    "get_analysis_capability_catalog",
    "get_module_catalog",
    "resolve_brief_analysis_plan",
    "apply_brief_analysis_plan",
    "run_brief_agent",
]
