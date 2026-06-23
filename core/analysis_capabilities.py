import copy
from typing import Any


from core.qa_capabilities import (
    get_qa_capabilities as get_qa_functional_capability_catalog,
    get_report_artifacts as get_report_artifact_catalog,
)


RELATION_KEYWORDS = [
    "relation",
    "impact",
    "influence",
    "compar",
    "assoc",
    "crois",
    "cross",
    "crosstab",
    "lien",
    "corrÃ©",
    "corrÃ©lation",
]


DISTRIB_KEYWORDS = [
    "distribution",
    "rÃ©partition",
    "histogram",
    "densitÃ©",
    "frÃ©quence",
    "repartition",
]


ANALYSIS_CAPABILITY_CATALOG = [
    {
        "action": "analyze_relationships",
        "module": "QA",
        "category": "conditional",
        "params_schema": ["pairs", "variables", "focus_question"],
        "state_inputs": ["df_ready", "crosstabs_interpretation", "figs_variables_distribution_detailed"],
        "state_outputs": ["qa_relationship_synthesis", "crosstabs_interpretation", "figs_variables_distribution_detailed"],
        "report_visible_outputs": [],
        "qa_visible_outputs": ["qa_relationship_synthesis", "crosstabs_interpretation", "figs_variables_distribution_detailed"],
        "details_preparation_outputs": [],
        "defaults": {"run_sankey_crosstabs": True, "generate_distribution_figures": True},
        "when_to_use": "Utiliser quand la question porte sur un lien, une comparaison ou un croisement entre deux variables ou plus.",
        "example_questions": [
            "Quel lien entre <variable_a> et <variable_b> ?",
            "Croise <variable_a> et <variable_b>.",
            "Y a-t-il une relation entre <variable_a> et <variable_b> ?",
            "Compare <variable_cible> avec plusieurs variables explicatives.",
        ],
        "example_parameters": [
            {"pairs": [["<variable_a>", "<variable_b>"]], "variables": ["<variable_a>", "<variable_b>"], "focus_question": "question utilisateur reformulée"},
            {"pairs": [["<variable_a>", "<variable_cible>"], ["<variable_b>", "<variable_cible>"]], "variables": ["<variable_a>", "<variable_b>", "<variable_cible>"], "focus_question": "question utilisateur reformulée"},
        ],
    },
    {
        "action": "contextualize_segment",
        "module": "segment_context",
        "category": "conditional",
        "params_schema": ["column", "value"],
        "state_inputs": ["df_ready"],
        "state_outputs": ["qa_segment_context", "qa_segment_counts_table", "qa_segment_percent_table", "qa_segment_subdataset"],
        "report_visible_outputs": [],
        "qa_visible_outputs": ["qa_segment_context", "qa_segment_counts_table", "qa_segment_percent_table"],
        "details_preparation_outputs": [],
        "defaults": {},
        "when_to_use": "Utiliser quand la question porte sur une catÃ©gorie prÃ©cise d'observations et qu'il faut d'abord la situer dans l'Ã©chantillon global.",
        "example_questions": [
            "Que représente la modalité <modalité> de <variable> dans l'échantillon ?",
            "Combien y a-t-il d'observations dans le groupe <variable> = <modalité> ?",
            "Quelle part de l'échantillon correspond à <variable> = <modalité> ?",
        ],
        "example_parameters": [
            {"column": "<variable>", "value": "<modalité>"},
        ],
    },
    {
        "action": "run_preparation1",
        "module": "Preparation1",
        "category": "systematic",
        "params_schema": [],
        "state_inputs": ["df_raw"],
        "state_outputs": ["df_prep", "process"],
        "report_visible_outputs": ["data_preparation_synthesis"],
        "qa_visible_outputs": [],
        "details_preparation_outputs": ["preparation_details_payload"],
        "defaults": {},
    },
    {
        "action": "run_preparation2",
        "module": "Preparation2",
        "category": "systematic",
        "params_schema": [],
        "state_inputs": ["df_imputed_structural"],
        "state_outputs": ["df_clean", "preparation2_details", "process"],
        "report_visible_outputs": ["data_preparation_synthesis"],
        "qa_visible_outputs": [],
        "details_preparation_outputs": ["preparation_details_payload"],
        "defaults": {},
    },
    {
        "action": "run_factor_analysis",
        "module": "AnalyseFactorielle",
        "category": "systematic",
        "params_schema": [],
        "state_inputs": ["df_ready"],
        "state_outputs": ["interpretationACM", "dendrogram_interpretation", "latent_summary_text"],
        "report_visible_outputs": ["interpretationACM", "dendrogram_interpretation", "latent_summary_text"],
        "qa_visible_outputs": ["interpretationACM", "dendrogram_interpretation", "latent_summary_text"],
        "details_preparation_outputs": [],
        "defaults": {},
    },
    {
        "action": "run_correlations",
        "module": "AnalyseCorrelations",
        "category": "systematic",
        "params_schema": [],
        "state_inputs": ["df_ready"],
        "state_outputs": ["interpretationACM", "dendrogram_interpretation"],
        "report_visible_outputs": ["dendrogram_interpretation"],
        "qa_visible_outputs": ["dendrogram_interpretation"],
        "details_preparation_outputs": [],
        "defaults": {},
    },
    {
        "action": "run_segmentation",
        "module": "Segmentation",
        "category": "systematic",
        "params_schema": [],
        "state_inputs": ["df_ready"],
        "state_outputs": ["segmentation_profiles_text", "segmentation_profiles_table", "segmentation_detailed_profiles"],
        "report_visible_outputs": ["segmentation_profiles_text", "segmentation_profiles_table"],
        "qa_visible_outputs": ["segmentation_profiles_text"],
        "details_preparation_outputs": [],
        "defaults": {},
    },
    {
        "action": "run_crosstabs",
        "module": "CrosstabsDetail",
        "category": "conditional",
        "params_schema": ["pairs"],
        "state_inputs": ["df_ready", "sankey_drawn_links_df"],
        "state_outputs": ["crosstabs_interpretation", "sankey_pair_results"],
        "report_visible_outputs": ["crosstabs_interpretation"],
        "qa_visible_outputs": ["crosstabs_interpretation", "sankey_pair_results"],
        "details_preparation_outputs": [],
        "defaults": {"run_sankey_crosstabs": True},
    },
    {
        "action": "rerun_sankey",
        "module": "DiagramSankey",
        "category": "conditional",
        "params_schema": ["target_variable", "illustrative_variables", "candidate_variables"],
        "state_inputs": ["df_ready", "target_variables", "illustrative_variables"],
        "state_outputs": ["sankey_diagram", "sankey_interpretation_synthesis", "sankey_latents", "sankey_pair_results"],
        "report_visible_outputs": ["sankey_diagram", "sankey_interpretation_synthesis", "sankey_latents"],
        "qa_visible_outputs": ["sankey_interpretation_synthesis", "sankey_pair_results"],
        "details_preparation_outputs": [],
        "defaults": {},
    },
    {
        "action": "run_distributions",
        "module": "DistributionsDetail",
        "category": "conditional",
        "params_schema": ["variables"],
        "state_inputs": ["df_ready"],
        "state_outputs": ["figs_variables_distribution_detailed", "figs_variables_distribution"],
        "report_visible_outputs": ["figs_variables_distribution"],
        "qa_visible_outputs": ["figs_variables_distribution_detailed", "figs_variables_distribution"],
        "details_preparation_outputs": [],
        "defaults": {"generate_distribution_figures": True},
    },
    {
        "action": "run_distribution_profile",
        "module": "DistributionVariables",
        "category": "conditional",
        "params_schema": ["column", "value", "use_subdataset"],
        "state_inputs": ["df_ready"],
        "state_outputs": ["profil_dominant_analysis", "qa_segment_profile_text", "dominant_continues", "dominant_discretes"],
        "report_visible_outputs": ["profil_dominant_analysis"],
        "qa_visible_outputs": ["qa_segment_profile_text", "profil_dominant_analysis"],
        "details_preparation_outputs": [],
        "defaults": {},
        "when_to_use": "Utiliser pour dÃ©crire le profil majoritaire d'une population ou d'une sous-population quand aucune cible analytique explicite n'est demandÃ©e.",
        "example_questions": [
            "Décris le profil du groupe <variable> = <modalité>.",
            "Quel est le portrait majoritaire de la sous-population <modalité> ?",
            "À quoi ressemble le profil dominant des observations du groupe <variable> = <modalité> ?",
        ],
        "example_parameters": [
            {"column": "<variable>", "value": "<modalité>", "use_subdataset": True},
        ],
    },
    {
        "action": "rerun_profils_y",
        "module": "Profils_y",
        "category": "conditional",
        "params_schema": ["target_variable", "target_modality", "selected_columns", "num_quantiles", "n_clusters"],
        "state_inputs": ["df_ready", "target_variables", "target_modalities"],
        "state_outputs": ["profils_y_text", "profils_y_simplified", "profils_y_table"],
        "report_visible_outputs": ["profils_y_text", "profils_y_table"],
        "qa_visible_outputs": ["profils_y_text"],
        "details_preparation_outputs": [],
        "defaults": {},
    },
    {
        "action": "rerun_profils_y_for_segment",
        "module": "QA",
        "category": "conditional",
        "params_schema": ["source_column", "source_value", "derived_target"],
        "state_inputs": ["df_ready", "qa_segment_subdataset"],
        "state_outputs": ["qa_segment_profils_y_text"],
        "report_visible_outputs": [],
        "qa_visible_outputs": ["qa_segment_profils_y_text"],
        "details_preparation_outputs": [],
        "defaults": {},
        "when_to_use": "Utiliser quand la question demande ce qui distingue une sous-population du reste de l'Ã©chantillon, ou demande un niveau de dÃ©tail supÃ©rieur sur un segment.",
        "example_questions": [
            "Quels sont les différents profils au sein du groupe <variable> = <modalité> ?",
            "Détaille les sous-profils de cette sous-population.",
            "Identifie plusieurs profils distincts dans le groupe <variable> = <modalité>.",
        ],
        "example_parameters": [
            {"source_column": "<variable>", "source_value": "<modalité>", "derived_target": "__qa_target__<variable>_<modalité>"},
        ],
    },
]


def get_analysis_capability_catalog() -> list[dict[str, Any]]:
    return copy.deepcopy(ANALYSIS_CAPABILITY_CATALOG)


def get_module_catalog() -> list[dict[str, Any]]:
    return [
        {
            "module": item["module"],
            "action": item["action"],
            "category": item["category"],
            "defaults": copy.deepcopy(item.get("defaults", {})),
        }
        for item in ANALYSIS_CAPABILITY_CATALOG
    ]
