"""Catalogue fonctionnel des capacités Q&A.

Ce fichier décrit ce que le module Q&A sait faire, indépendamment du dataset courant.
Il ne doit contenir aucun nom de variable ou de modalité issu d'un dataset particulier
(ex. pas de ``income``, ``workclass``, ``race``, ``Private`` dans les descriptions
fonctionnelles elles-mêmes).

Les noms de colonnes et de modalités doivent être fournis dynamiquement par le routeur
à partir de la question utilisateur et du dataframe courant.
"""

from __future__ import annotations

import copy
from typing import Any


QA_CAPABILITIES: list[dict[str, Any]] = [
    {
        "id": "describe_group_profile",
        "label": "Décrire le profil dominant d'un groupe d'observations",
        "kind": "computed_analysis",
        "module": "DistributionVariables",
        "action": "run_distribution_profile",
        "description": (
            "Cette capacité sert à décrire le profil dominant d'une sous-population "
            "définie par une condition simple sur une variable catégorielle. Elle répond "
            "à une question du type : quel est le portrait majoritaire des observations "
            "qui appartiennent à tel groupe ? Le résultat attendu est un profil synthétique "
            "unique du groupe, construit à partir des variables disponibles dans le dataset. "
            "Elle ne doit pas chercher plusieurs profils internes au groupe : si la question "
            "demande des profils distincts ou des sous-types, il faut utiliser une capacité "
            "basée sur Profils_y."
        ),
        "question_patterns": [
            "Quel est le profil de <modalité> ?",
            "Décris le groupe <variable> = <modalité>.",
            "À quoi ressemble la sous-population <modalité> ?",
        ],
        "parameters": {
            "filter": {
                "type": "object",
                "required": True,
                "description": "Condition qui définit le groupe : {'column': <nom_colonne>, 'value': <modalité>}.",
            },
            "exclude_columns": {
                "type": "list[column_name]",
                "required": False,
                "description": "Colonnes à exclure du portrait si elles sont triviales ou déjà utilisées pour filtrer.",
            },
        },
        "execution": {
            "strategy": "single_run_on_filtered_dataframe",
            "module_function": "DistributionVariables.profil_dominant",
            "cardinality": "1 profil dominant",
        },
        "allowed_artifacts": ["qa_segment_profile_text", "dominant_continues", "dominant_discretes"],
        "excluded_artifacts": [
            "profil_dominant_analysis",
            "qa_segment_context_from_previous_question",
            "profils_y_text_from_previous_target",
            "sankey_interpretation_synthesis",
        ],
        "default_params": {},
        "cost": "medium",
    },
    {
        "id": "describe_dominant_profile_by_modality",
        "label": "Décrire le profil dominant de chaque modalité d'une variable",
        "kind": "computed_analysis",
        "module": "DistributionVariables",
        "action": "run_distribution_profile_by_modality",
        "description": (
            "Cette capacité sert à produire un profil dominant séparé pour chaque modalité "
            "d'une variable catégorielle. Elle est appropriée lorsque l'utilisateur demande "
            "explicitement le profil dominant, le portrait majoritaire ou la description synthétique "
            "de chaque groupe. Pour une variable ayant M modalités retenues, le résultat attendu "
            "contient M profils dominants, un par modalité. Elle ne convient pas aux formulations "
            "qui demandent les profils associés, les différents profils ou les sous-profils au sein "
            "de chaque modalité : dans ce cas, il faut utiliser discover_profiles_by_modality."
        ),
        "question_patterns": [
            "Quel est le profil dominant de chaque modalité de <variable> ?",
            "Compare les portraits majoritaires selon <variable>.",
            "Donne le profil dominant pour chaque groupe de <variable>.",
        ],
        "parameters": {
            "group_by": {
                "type": "column_name",
                "required": True,
                "description": "Variable catégorielle dont les modalités définissent les groupes à comparer.",
            },
            "modalities": {
                "type": "list[modality]",
                "required": False,
                "description": "Sous-ensemble éventuel de modalités à traiter. Si absent, traiter les modalités pertinentes de la variable.",
            },
            "max_modalities": {
                "type": "int",
                "required": False,
                "description": "Nombre maximal de modalités à traiter pour éviter une réponse trop longue.",
            },
            "min_group_size": {
                "type": "int",
                "required": False,
                "description": "Effectif minimal requis pour produire un portrait fiable d'une modalité.",
            },
        },
        "execution": {
            "strategy": "loop_over_modalities",
            "module_function": "DistributionVariables.profil_dominant",
            "cardinality": "nombre_modalités_retenues profils dominants",
        },
        "allowed_artifacts": ["profiles_by_modality", "dominant_continues", "dominant_discretes"],
        "excluded_artifacts": [
            "profil_dominant_analysis",
            "qa_segment_context_from_previous_question",
            "profils_y_text_from_previous_target",
            "sankey_interpretation_synthesis",
        ],
        "default_params": {"max_modalities": 8, "min_group_size": 30},
        "cost": "medium_to_high",
    },
    {
        "id": "discover_profiles_in_group",
        "label": "Identifier plusieurs profils dans un groupe d'observations",
        "kind": "computed_analysis",
        "module": "Profils_y",
        "action": "rerun_profils_y_for_segment",
        "description": (
            "Cette capacité sert à trouver plusieurs profils distincts à l'intérieur d'une "
            "sous-population définie par une condition simple. Elle répond aux questions qui "
            "demandent les différents profils, les sous-groupes, les types ou les segments internes "
            "d'un groupe. Le module Profils_y est exécuté sur la sous-population concernée, avec "
            "un nombre de profils demandé ou inféré. Le résultat attendu contient n_clusters profils "
            "pour le groupe analysé. Cette capacité est différente du profil dominant : elle cherche "
            "la diversité interne du groupe, pas seulement son portrait majoritaire."
        ),
        "question_patterns": [
            "Quels sont les différents profils parmi <modalité> ?",
            "Identifie plusieurs profils dans le groupe <variable> = <modalité>.",
            "Segmente la sous-population <modalité> en <n> profils.",
        ],
        "parameters": {
            "filter": {
                "type": "object",
                "required": True,
                "description": "Condition qui définit le groupe : {'column': <nom_colonne>, 'value': <modalité>}.",
            },
            "n_clusters": {
                "type": "int",
                "required": False,
                "description": "Nombre de profils à produire dans le groupe. Valeur par défaut si non précisée : 3.",
            },
            "continuous_binning_quantiles": {
                "type": "int",
                "allowed_values": [3, 5, 10],
                "required": False,
                "description": "Nombre de quantiles pour discrétiser les variables continues avant construction des profils.",
            },
            "selected_columns": {
                "type": "list[column_name]",
                "required": False,
                "description": "Variables à utiliser pour construire les profils. Si absent, utiliser les variables pertinentes du dataset.",
            },
        },
        "execution": {
            "strategy": "single_run_on_filtered_dataframe",
            "module_function": "Profils_y",
            "cardinality": "n_clusters profils",
        },
        "allowed_artifacts": ["qa_segment_profils_y_text", "profils_y_table"],
        "excluded_artifacts": [
            "profil_dominant_analysis",
            "qa_segment_context_from_previous_question_unless_same_filter",
            "profils_y_text_from_previous_target",
            "sankey_interpretation_synthesis",
        ],
        "default_params": {"n_clusters": 3, "continuous_binning_quantiles": 5},
        "cost": "high",
    },
    {
        "id": "discover_profiles_by_modality",
        "label": "Identifier plusieurs profils pour chaque modalité d'une variable",
        "kind": "computed_analysis",
        "module": "Profils_y",
        "action": "rerun_profils_y_by_modality",
        "description": (
            "Cette capacité sert à identifier plusieurs profils distincts au sein de chaque modalité "
            "d'une variable catégorielle. Elle doit être choisie lorsque l'utilisateur demande les "
            "profils associés aux différentes modalités d'une variable, ou les profils observés dans "
            "chaque groupe. Le module Profils_y est exécuté indépendamment pour chaque modalité retenue. "
            "Si la variable possède M modalités retenues et que n_clusters vaut K, la réponse doit "
            "contenir M × K profils. Par exemple, pour quatre modalités et trois profils demandés, "
            "la sortie attendue contient douze profils organisés par modalité. Cette capacité est plus "
            "lourde qu'un profil dominant par modalité et ne doit pas être remplacée par DistributionVariables."
        ),
        "question_patterns": [
            "Quels sont les profils associés aux différentes modalités de <variable> ?",
            "Quels profils observe-t-on dans chaque groupe de <variable> ?",
            "Pour chaque modalité de <variable>, identifie <n> profils.",
        ],
        "parameters": {
            "group_by": {
                "type": "column_name",
                "required": True,
                "description": "Variable catégorielle dont chaque modalité définit une sous-population à profiler.",
            },
            "modalities": {
                "type": "list[modality]",
                "required": False,
                "description": "Sous-ensemble éventuel de modalités à traiter. Si absent, traiter les modalités pertinentes de la variable.",
            },
            "n_clusters": {
                "type": "int",
                "required": False,
                "description": "Nombre de profils à produire pour chaque modalité. Valeur par défaut si non précisée : 3.",
            },
            "continuous_binning_quantiles": {
                "type": "int",
                "allowed_values": [3, 5, 10],
                "required": False,
                "description": "Nombre de quantiles pour discrétiser les variables continues avant construction des profils.",
            },
            "max_modalities": {
                "type": "int",
                "required": False,
                "description": "Nombre maximal de modalités à traiter pour éviter une exécution trop longue.",
            },
            "min_group_size": {
                "type": "int",
                "required": False,
                "description": "Effectif minimal requis pour exécuter Profils_y sur une modalité.",
            },
        },
        "execution": {
            "strategy": "loop_over_modalities",
            "module_function": "Profils_y",
            "cardinality": "nombre_modalités_retenues × n_clusters profils",
        },
        "allowed_artifacts": ["profiles_by_modality", "profils_y_table"],
        "excluded_artifacts": [
            "profil_dominant_analysis",
            "qa_segment_context_from_previous_question",
            "profils_y_text_from_previous_target",
            "sankey_interpretation_synthesis",
        ],
        "default_params": {
            "n_clusters": 3,
            "continuous_binning_quantiles": 5,
            "max_modalities": 8,
            "min_group_size": 50,
        },
        "cost": "very_high",
    },
    {
        "id": "rerun_target_schema",
        "label": "Recalculer le schéma de relations pour une nouvelle variable cible",
        "kind": "computed_analysis",
        "module": "DiagramSankey",
        "action": "rerun_sankey",
        "description": (
            "Cette capacité sert à recalculer le schéma des relations entre variables en remplaçant "
            "la variable cible courante par une autre colonne du dataset. Elle répond aux demandes "
            "qui expriment explicitement le souhait de refaire le schéma, le Sankey ou l'analyse des "
            "relations avec une nouvelle cible. Elle modifie le périmètre d'analyse et produit de "
            "nouveaux artefacts de schéma, de synthèse et éventuellement de paires de variables. Elle "
            "ne doit pas être utilisée pour une simple question d'interprétation sur un schéma déjà produit."
        ),
        "question_patterns": [
            "Refais le schéma avec <variable> comme cible.",
            "Prends <variable> comme nouvelle cible.",
            "Relance le Sankey pour expliquer <variable>.",
        ],
        "parameters": {
            "target_variable": {
                "type": "column_name",
                "required": True,
                "description": "Nouvelle variable cible à utiliser pour recalculer le schéma.",
            },
            "candidate_variables": {
                "type": "list[column_name]",
                "required": False,
                "description": "Variables candidates à inclure dans l'analyse, si l'utilisateur restreint le périmètre.",
            },
            "illustrative_variables": {
                "type": "list[column_name]",
                "required": False,
                "description": "Variables illustratives à conserver ou privilégier.",
            },
        },
        "execution": {
            "strategy": "single_pipeline_rerun",
            "module_function": "DiagramSankey.run",
            "cardinality": "1 schéma cible + synthèses associées",
        },
        "allowed_artifacts": ["sankey_diagram", "sankey_interpretation_synthesis", "sankey_latents", "sankey_pair_results"],
        "excluded_artifacts": ["qa_segment_context_from_previous_question", "profil_dominant_analysis"],
        "default_params": {},
        "cost": "high",
    },
    {
        "id": "explain_existing_factor_analysis",
        "label": "Expliquer l'analyse factorielle déjà produite",
        "kind": "existing_artifact",
        "module": "RapportFinal",
        "action": "read_factor_analysis",
        "description": (
            "Cette capacité sert à répondre à une question sur l'analyse factorielle déjà calculée "
            "dans le rapport. Elle ne doit pas relancer un module d'analyse. Elle doit uniquement "
            "sélectionner les artefacts factoriels disponibles, les reformuler et signaler clairement "
            "si l'analyse factorielle n'a pas été produite."
        ),
        "parameters": {},
        "execution": {"strategy": "read_existing_artifact", "cardinality": "1 explication"},
        "allowed_artifacts": ["interpretationACM", "dendrogram_interpretation"],
        "excluded_artifacts": ["profil_dominant_analysis", "qa_segment_context_from_previous_question", "profils_y_text_from_previous_target"],
        "default_params": {},
        "cost": "low",
    },
    {
        "id": "explain_existing_latents",
        "label": "Expliquer les dimensions latentes déjà identifiées",
        "kind": "existing_artifact",
        "module": "RapportFinal",
        "action": "read_latents",
        "description": (
            "Cette capacité sert à expliquer les regroupements latents de variables déjà produits "
            "dans le rapport. Elle lit les synthèses et tableaux existants sans recalculer le Sankey "
            "ni l'analyse factorielle. Elle doit rester centrée sur la signification des latents et "
            "sur les variables qui les composent."
        ),
        "parameters": {},
        "execution": {"strategy": "read_existing_artifact", "cardinality": "1 explication"},
        "allowed_artifacts": ["latent_summary_text", "sankey_latents"],
        "excluded_artifacts": ["profil_dominant_analysis", "qa_segment_context_from_previous_question", "profils_y_text_from_previous_target"],
        "default_params": {},
        "cost": "low",
    },
    {
        "id": "explain_existing_crosstab",
        "label": "Expliquer un tri croisé déjà disponible",
        "kind": "existing_artifact",
        "module": "RapportFinal",
        "action": "read_crosstab",
        "description": (
            "Cette capacité sert à répondre à une question sur un croisement entre variables lorsque "
            "le tri croisé a déjà été demandé ou produit dans le rapport. Elle doit sélectionner la "
            "paire de variables pertinente dans les artefacts disponibles et ne pas injecter de profil "
            "dominant ou de contexte de segment. Si le croisement n'existe pas encore, le routeur peut "
            "proposer une capacité de calcul dédiée, mais cette capacité-ci reste en lecture seule."
        ),
        "parameters": {
            "pairs": {
                "type": "list[tuple[column_name,column_name]]",
                "required": False,
                "description": "Paires de variables concernées par la question, si elles sont mentionnées.",
            }
        },
        "execution": {"strategy": "read_existing_artifact", "cardinality": "1 explication par paire disponible"},
        "allowed_artifacts": ["crosstabs_interpretation", "sankey_pair_results"],
        "excluded_artifacts": ["profil_dominant_analysis", "qa_segment_context_from_previous_question", "profils_y_text_from_previous_target"],
        "default_params": {},
        "cost": "low",
    },
    {
        "id": "explain_existing_distribution",
        "label": "Expliquer la distribution déjà disponible d'une variable",
        "kind": "existing_artifact",
        "module": "RapportFinal",
        "action": "read_distribution",
        "description": (
            "Cette capacité sert à répondre à une question sur la distribution d'une ou plusieurs "
            "variables lorsque les figures ou résumés de distribution ont déjà été produits. Elle "
            "doit rester centrée sur les variables demandées et ne pas élargir la réponse vers des "
            "profils globaux, des segments précédents ou une cible analytique non demandée."
        ),
        "parameters": {
            "variables": {
                "type": "list[column_name]",
                "required": True,
                "description": "Variables dont la distribution doit être expliquée.",
            }
        },
        "execution": {"strategy": "read_existing_artifact", "cardinality": "1 explication par variable disponible"},
        "allowed_artifacts": ["figs_variables_distribution", "figs_variables_distribution_detailed"],
        "excluded_artifacts": ["profil_dominant_analysis", "qa_segment_context_from_previous_question", "profils_y_text_from_previous_target"],
        "default_params": {},
        "cost": "low",
    },
]


REPORT_ARTIFACTS: list[dict[str, Any]] = [
    {
        "id": "factor_analysis",
        "label": "Analyse factorielle",
        "description": "Artefacts d'analyse factorielle déjà produits dans le rapport final.",
        "state_keys": ["interpretationACM", "dendrogram_interpretation"],
        "recompute_from_qa": False,
    },
    {
        "id": "latent_variables",
        "label": "Dimensions latentes",
        "description": "Regroupements latents de variables déjà identifiés.",
        "state_keys": ["latent_summary_text", "sankey_latents"],
        "recompute_from_qa": False,
    },
    {
        "id": "crosstabs",
        "label": "Tris croisés",
        "description": "Tris croisés déjà demandés ou produits dans le rapport.",
        "state_keys": ["crosstabs_interpretation", "sankey_pair_results"],
        "recompute_from_qa": False,
    },
    {
        "id": "variable_distributions",
        "label": "Distributions de variables",
        "description": "Distributions déjà produites pour certaines variables.",
        "state_keys": ["figs_variables_distribution", "figs_variables_distribution_detailed"],
        "recompute_from_qa": False,
    },
]


def get_qa_capabilities() -> list[dict[str, Any]]:
    """Retourne une copie du catalogue fonctionnel Q&A."""
    return copy.deepcopy(QA_CAPABILITIES)


def get_report_artifacts() -> list[dict[str, Any]]:
    """Retourne une copie du catalogue des artefacts lisibles par le Q&A."""
    return copy.deepcopy(REPORT_ARTIFACTS)


def get_qa_capability(capability_id: str) -> dict[str, Any] | None:
    """Retourne une capacité par identifiant, ou None si elle n'existe pas."""
    for capability in QA_CAPABILITIES:
        if capability.get("id") == capability_id:
            return copy.deepcopy(capability)
    return None


def get_capability_parameter_schema(capability_id: str) -> dict[str, Any]:
    """Retourne le schéma de paramètres d'une capacité."""
    capability = get_qa_capability(capability_id)
    if not capability:
        return {}
    parameters = capability.get("parameters") or {}
    return copy.deepcopy(parameters) if isinstance(parameters, dict) else {}


def capability_to_legacy_action(capability: dict[str, Any]) -> dict[str, Any]:
    """Convertit une capacité Q&A en entrée compatible avec l'ancien planner.

    Cette fonction permet de migrer progressivement depuis ANALYSIS_CAPABILITY_CATALOG
    sans exposer au planner les anciens exemples dataset-spécifiques.
    """
    params = capability.get("parameters") or {}
    return {
        "action": capability.get("action"),
        "module": capability.get("module"),
        "category": "conditional" if capability.get("kind") == "computed_analysis" else "existing_artifact",
        "params_schema": list(params.keys()) if isinstance(params, dict) else [],
        "state_inputs": ["df_ready"],
        "state_outputs": list(capability.get("allowed_artifacts") or []),
        "report_visible_outputs": [],
        "qa_visible_outputs": list(capability.get("allowed_artifacts") or []),
        "details_preparation_outputs": [],
        "defaults": copy.deepcopy(capability.get("default_params") or {}),
        "when_to_use": capability.get("description", ""),
        "example_questions": list(capability.get("question_patterns") or []),
        "example_parameters": [],
    }
