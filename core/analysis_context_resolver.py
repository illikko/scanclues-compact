from typing import Any


DEFAULT_PIPELINE_SELECTION = {
    "preparation": True,
    "details_preparation": False,
    "profilage": True,
    "analyse_descriptive": True,
    "sankey_crosstabs": False,
    "distribution_figures": False,
}


def default_pipeline_selection() -> dict[str, bool]:
    return DEFAULT_PIPELINE_SELECTION.copy()


def build_analysis_options(selection: dict[str, Any] | None) -> dict[str, Any]:
    normalized = default_pipeline_selection()
    if isinstance(selection, dict):
        for key in normalized:
            normalized[key] = bool(selection.get(key, normalized[key]))

    return {
        "pipeline_selection": normalized,
        "details_preparation_selected": bool(normalized.get("details_preparation", False)),
        "systematic_modules": [
            "Preparation1",
            "Preparation2",
            "AnalyseFactorielle",
            "AnalyseCorrelations",
            "Segmentation",
        ],
        "conditional_modules": [
            "DiagramSankey",
            "Profils_y",
            "CrosstabsDetail",
            "DistributionsDetail",
        ],
    }


def has_executable_selection(selection: dict[str, Any] | None) -> bool:
    normalized = build_analysis_options(selection)["pipeline_selection"]
    executable_keys = [
        "preparation",
        "profilage",
        "analyse_descriptive",
        "sankey_crosstabs",
        "distribution_figures",
    ]
    return any(bool(normalized.get(key, False)) for key in executable_keys)


def resolve_analysis_context(session_state: Any) -> dict[str, Any]:
    selection = {}
    if hasattr(session_state, "get"):
        selection = session_state.get("pipeline_selection", {})

    options = build_analysis_options(selection)
    return {
        "analysis_options": options,
        "target_variables": list(session_state.get("target_variables", []) or []),
        "illustrative_variables": list(session_state.get("illustrative_variables", []) or []),
        "target_modalities": dict(session_state.get("target_modalities", {}) or {}),
        "dataset_key_questions_mode": session_state.get("dataset_key_questions_mode", "sb"),
        "dataset_key_questions_value": session_state.get("dataset_key_questions_value", ""),
    }
