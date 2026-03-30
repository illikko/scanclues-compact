import re
import json
import os
from typing import List

import pandas as pd
import streamlit as st
from openai import OpenAI

# Patterns simples pour orienter l'agent sans appel LLM
RELATION_KEYWORDS = [
    "relation", "impact", "influence", "compar", "assoc", "crois", "cross", "crosstab",
    "lien", "corré", "corrélation"
]
DISTRIB_KEYWORDS = [
    "distribution", "répartition", "histogram", "densité", "fréquence", "repartition"
]

# Catalogue de modules et paramètres par défaut
MODULE_CATALOG = [
    {"module": "CrosstabsDetail", "defaults": {"run_sankey_crosstabs": True}},
    {"module": "DiagramSankey", "defaults": {}},
    {"module": "DistributionsDetail", "defaults": {"generate_distribution_figures": True}},
    {"module": "Profils_y", "defaults": {}},
    {"module": "Segmentation", "defaults": {}},
    {"module": "AnalyseCorrelations", "defaults": {}},
    {"module": "DistributionVariables", "defaults": {}},
]


def _has_context_ready() -> bool:
    """Vérifie que le cadrage LLM a déjà produit les infos minimales."""
    return (
        st.session_state.get("dataset_object") is not None
        and st.session_state.get("dataset_context") is not None
        and st.session_state.get("dataset_recommendations") is not None
    )


def _match_columns(df: pd.DataFrame, brief: str) -> List[str]:
    brief_low = brief.lower()
    matches: List[str] = []
    for col in df.columns:
        name = str(col)
        if name.lower() in brief_low:
            matches.append(name)
    return matches


def _mark_plan(plan: List[dict]) -> None:
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
    - Alimente st.session_state : brief_target_variable, brief_illustrative_variables,
      flags de perf (run_sankey_crosstabs, generate_distribution_figures) et brief_analysis_plan.
    """
    mode = st.session_state.get("dataset_key_questions_mode", "sb")
    brief = str(st.session_state.get("dataset_key_questions_value", "") or "")

    # Ne rien faire si pas de brief explicite
    if mode != "ab" or not brief.strip():
        return
    if not _has_context_ready():
        return
    if not isinstance(df_source, pd.DataFrame) or df_source.empty:
        return

    brief_low = brief.lower()
    cols = _match_columns(df_source, brief)

    target = None
    illustratives: List[str] = []
    if cols:
        target = cols[0]
        illustratives = [c for c in cols[1:] if c != target]
        st.session_state["brief_target_variable"] = target
        st.session_state["brief_illustrative_variables"] = illustratives
        # Forcer la cible du brief en tête de liste des cibles/illustratives globales
        tv = st.session_state.get("target_variables", [])
        st.session_state["target_variables"] = [target] + [t for t in tv if t != target]
        iv = st.session_state.get("illustrative_variables", [])
        st.session_state["illustrative_variables"] = illustratives + [i for i in iv if i not in illustratives and i != target]
        st.session_state["brief_reason"] = "variable cible détectée dans le brief"
    else:
        st.session_state["brief_reason"] = "aucune variable du brief trouvée dans le dataset"

    rel_flag = any(k in brief_low for k in RELATION_KEYWORDS) or len(cols) >= 2
    dist_flag = any(k in brief_low for k in DISTRIB_KEYWORDS)

    if rel_flag:
        st.session_state["run_sankey_crosstabs"] = True
    if dist_flag:
        st.session_state["generate_distribution_figures"] = True

    # --- Appel LLM pour affiner la sélection des modules/params ---
    llm_plan: List[dict] = []
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        payload = {
            "brief": brief,
            "columns": [str(c) for c in df_source.columns][:120],
            "dataset_object": st.session_state.get("dataset_object"),
            "dataset_context": st.session_state.get("dataset_context"),
            "target_variables": st.session_state.get("target_variables", []),
            "illustrative_variables": st.session_state.get("illustrative_variables", []),
            "module_catalog": MODULE_CATALOG,
        }
        sys_prompt = """Tu es un planificateur d'analyses pour une app Streamlit.
Réponds uniquement par un JSON strict: {"plan":[{"module":str,"params":dict,"reason":str},...]}.
Règles:
- Uniquement des modules présents dans module_catalog.
- Si le brief cite une colonne, la mettre en cible prioritaire.
- Relation/impact/comparaison -> CrosstabsDetail + DiagramSankey (run_sankey_crosstabs=True).
- Distributions/répartition/histogrammes -> DistributionsDetail (generate_distribution_figures=True).
- Profils/segments -> Profils_y (target=cible), Segmentation.
- Pas de doublons, max 6 entrées, reason court en français."""
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=400,
        )
        content = r.choices[0].message.content
        parsed = json.loads(content)
        llm_plan = parsed.get("plan", []) if isinstance(parsed, dict) else []
        st.session_state["brief_llm_plan_raw"] = parsed
    except Exception as e:
        st.session_state["brief_llm_error"] = f"brief_agent_llm: {e}"
        llm_plan = []

    # --- Fallback mapping si LLM insuffisant ---
    plan: List[dict] = []
    plan.extend(llm_plan)

    if not plan:
        if rel_flag:
            plan.append({
                "module": "CrosstabsDetail",
                "params": {"run_sankey_crosstabs": True},
                "reason": "relation variables identifiée dans le brief",
            })
            plan.append({
                "module": "DiagramSankey",
                "params": {"target": target, "profiles": illustratives},
                "reason": "vue graphe Sankey + crosstabs",
            })
        if dist_flag:
            plan.append({
                "module": "DistributionsDetail",
                "params": {"generate_distribution_figures": True},
                "reason": "demande de distributions/histogrammes",
            })
        if target:
            plan.append({
                "module": "Profils_y",
                "params": {"target": target, "n_clusters_target": st.session_state.get("n_clusters_target")},
                "reason": "analyse de profils pour la cible issue du brief",
            })
            plan.append({
                "module": "Segmentation",
                "params": {"n_clusters_segmentation": st.session_state.get("n_clusters_segmentation")},
                "reason": "segmentation autour de la cible du brief",
            })

    if plan:
        _mark_plan(plan)
    else:
        st.session_state["brief_relevance"] = False
