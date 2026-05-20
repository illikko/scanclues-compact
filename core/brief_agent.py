import json
import os
from typing import Any

import pandas as pd
import streamlit as st
from openai import OpenAI

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


def _mark_plan(plan: list[dict[str, Any]]) -> None:
    st.session_state["brief_analysis_plan"] = plan
    st.session_state["brief_results_synthesis"] = "\n".join(
        [f"- {p.get('module')} ({p.get('reason', '')})" for p in plan]
    )
    st.session_state["brief_relevance"] = True
    st.session_state["brief_plan_ready"] = True


def run_brief_agent(df_source: pd.DataFrame | None) -> None:
    """
    Agent léger pour interpréter le brief et activer les modules pertinents.
    - Ne s'exécute qu'en mode 'ab' et si le cadrage est prêt.
    - Alimente st.session_state avec la cible du brief, les variables illustratives,
      les flags de performance et le plan d'analyse.
    """
    mode = st.session_state.get("dataset_key_questions_mode", "sb")
    brief = str(st.session_state.get("dataset_key_questions_value", "") or "")

    if mode != "ab" or not brief.strip():
        return
    if not _has_context_ready():
        return
    if not isinstance(df_source, pd.DataFrame) or df_source.empty:
        return

    brief_low = brief.lower()
    cols = _match_columns(df_source, brief)

    target = None
    illustratives: list[str] = []
    if cols:
        target = cols[0]
        illustratives = [c for c in cols[1:] if c != target]
        st.session_state["brief_target_variable"] = target
        st.session_state["brief_illustrative_variables"] = illustratives

        tv = st.session_state.get("target_variables", [])
        st.session_state["target_variables"] = [target] + [t for t in tv if t != target]

        iv = st.session_state.get("illustrative_variables", [])
        st.session_state["illustrative_variables"] = illustratives + [
            i for i in iv if i not in illustratives and i != target
        ]
        st.session_state["brief_reason"] = "variable cible détectée dans le brief"
    else:
        st.session_state["brief_reason"] = "aucune variable du brief trouvée dans le dataset"

    rel_flag = any(k in brief_low for k in RELATION_KEYWORDS) or len(cols) >= 2
    dist_flag = any(k in brief_low for k in DISTRIB_KEYWORDS)

    if rel_flag:
        st.session_state["run_sankey_crosstabs"] = True
    if dist_flag:
        st.session_state["generate_distribution_figures"] = True

    llm_plan: list[dict[str, Any]] = []
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        payload = {
            "brief": brief,
            "columns": [str(c) for c in df_source.columns][:120],
            "dataset_object": st.session_state.get("dataset_object"),
            "dataset_context": st.session_state.get("dataset_context"),
            "target_variables": st.session_state.get("target_variables", []),
            "illustrative_variables": st.session_state.get("illustrative_variables", []),
            "module_catalog": get_module_catalog(),
        }
        sys_prompt = """Tu es un planificateur d'analyses pour une app Streamlit.
Réponds uniquement par un JSON strict: {"plan":[{"module":str,"params":dict,"reason":str},...]}.
Règles:
- Uniquement des modules présents dans module_catalog.
- Si le brief cite une colonne, la mettre en cible prioritaire.
- Relation/impact/comparaison -> CrosstabsDetail + DiagramSankey (run_sankey_crosstabs=true).
- Distributions/répartition/histogrammes -> DistributionsDetail (generate_distribution_figures=true).
- Profils/segments -> Profils_y (target=cible).
- Pas de doublons, max 6 entrées, reason court en français."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
            max_tokens=400,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        llm_plan = parsed.get("plan", []) if isinstance(parsed, dict) else []
        st.session_state["brief_llm_plan_raw"] = parsed
    except Exception as exc:
        st.session_state["brief_llm_error"] = f"brief_agent_llm: {exc}"
        llm_plan = []

    plan: list[dict[str, Any]] = []
    plan.extend(llm_plan)

    if not plan:
        if rel_flag:
            plan.append(
                {
                    "module": "CrosstabsDetail",
                    "params": {"run_sankey_crosstabs": True},
                    "reason": "relation variables identifiée dans le brief",
                }
            )
            plan.append(
                {
                    "module": "DiagramSankey",
                    "params": {"target": target, "profiles": illustratives},
                    "reason": "vue Sankey et relations détaillées",
                }
            )
        if dist_flag:
            plan.append(
                {
                    "module": "DistributionsDetail",
                    "params": {"generate_distribution_figures": True},
                    "reason": "demande de distributions ou histogrammes",
                }
            )
        if target:
            plan.append(
                {
                    "module": "Profils_y",
                    "params": {
                        "target": target,
                        "n_clusters_target": st.session_state.get("n_clusters_target"),
                    },
                    "reason": "analyse de profils pour la cible issue du brief",
                }
            )

    if plan:
        _mark_plan(plan)
    else:
        st.session_state["brief_relevance"] = False


__all__ = ["get_analysis_capability_catalog", "run_brief_agent"]
