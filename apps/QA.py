import difflib
import json
import os
from typing import Any

import pandas as pd
import streamlit as st
from openai import OpenAI

from apps.CrosstabsDetail import run as run_crosstabs_detail
from apps.DiagramSankey import run as run_diagram_sankey
from apps.DistributionVariables import run as run_distribution_variables
from apps.DistributionsDetail import run as run_distributions_detail
from apps.Preparation1 import run as run_preparation1
from apps.Preparation2 import run as run_preparation2
from apps.Profils_y import run as run_profils_y
from core.brief_agent import get_analysis_capability_catalog, run_brief_agent
from core.segment_context import (
    build_segment_context_tables,
    build_segment_intro,
    resolve_segment_from_question,
)
from core.qa_memory import (
    QA_CONVERSATION_SUMMARY_KEY,
    QA_HISTORY_KEY,
    QA_LAST_FOLLOWUP_KEY,
    QA_LAST_FOLLOWUPS_KEY,
    append_qa_history,
    ensure_qa_memory,
    get_recent_qa_history,
    update_qa_conversation_summary,
)
from core.reset_state import reset_app_state
from core.qa_subset import build_subset_for_analysis, infer_subset_filters_from_question, normalize_subset_filters

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
NAV_CONTEXT_KEY = "__NAV_CONTEXT__"

PREPARATION_ACTIONS = {
    "run_preparation1": run_preparation1,
    "run_preparation2": run_preparation2,
}

PROFILE_ACTIONS = {
    "run_distribution_profile": run_distribution_variables,
}

SEGMENT_ACTIONS = {
    "contextualize_segment",
    "rerun_profils_y_for_segment",
}


def _reset_qa_history() -> None:
    state_resets = {
        QA_HISTORY_KEY: [],
        QA_CONVERSATION_SUMMARY_KEY: "",
        QA_LAST_FOLLOWUP_KEY: "",
        "qa_last_plan": None,
        "qa_last_answer": None,
        "qa_last_execution_log": [],
        "qa_last_question": "",
        "qa_last_subset_description": "",
        "qa_segment_context": None,
        "qa_segment_counts_table": None,
        "qa_segment_percent_table": None,
        "qa_segment_subdataset": None,
        "qa_segment_profile_text": "",
        "qa_segment_profils_y_text": "",
        "qa_relationship_synthesis": "",
        QA_LAST_FOLLOWUPS_KEY: [],
        "qa_last_analysis_suggestion": {},
        "qa_last_analysis_suggestions": [],
    }
    for key, default in state_resets.items():
        st.session_state[key] = default
    st.session_state.pop("qa_chat_input", None)


def to_text(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return value.to_csv(index=False)
    if isinstance(value, pd.Series):
        return value.to_csv(index=True)
    return str(value)


def _safe_json_loads(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _goto_step(step: str) -> None:
    st.session_state["__NAV_SELECTED__"] = str(step)
    st.session_state[NAV_CONTEXT_KEY] = "view"
    try:
        st.query_params["step"] = str(step)
    except Exception:
        st.experimental_set_query_params(step=str(step))
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().casefold()


def _expand_followup_reply(question: str) -> str:
    raw = str(question or "").strip()
    if not raw:
        return raw
    normalized = _normalize_text(raw)
    followups = st.session_state.get(QA_LAST_FOLLOWUPS_KEY) or []
    if not isinstance(followups, list):
        followups = []
    last_followup = str(st.session_state.get(QA_LAST_FOLLOWUP_KEY) or "").strip()
    primary_followup = str(followups[0]).strip() if followups else last_followup
    if not primary_followup:
        return raw

    yes_values = {"oui", "ok", "okay", "d'accord", "dac", "go", "vas-y", "volontiers", "yes", "y"}
    no_values = {"non", "no", "nop", "pas maintenant", "non merci"}

    if normalized in yes_values:
        return primary_followup
    if normalized in no_values:
        return f"L'utilisateur répond non à la question de relance suivante : {primary_followup}"
    return raw


def _find_cols_in_question(question: str, df: pd.DataFrame) -> list[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    qlow = question.lower()
    cols: list[str] = []
    for col in df.columns:
        name = str(col)
        if name.lower() in qlow:
            cols.append(name)
    if cols:
        return list(dict.fromkeys(cols))

    tokens = [t for t in qlow.replace("?", " ").replace(",", " ").split() if len(t) >= 4]
    candidates = [str(c) for c in df.columns]
    for token in tokens:
        match = difflib.get_close_matches(token, candidates, n=1, cutoff=0.8)
        if match:
            cols.append(match[0])
    return list(dict.fromkeys(cols))


def _resolve_column_name(df: pd.DataFrame, requested: Any) -> str | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    raw = str(requested or "").strip()
    if not raw:
        return None
    if raw in df.columns:
        return raw

    raw_norm = raw.casefold()
    by_norm = {str(col).strip().casefold(): str(col) for col in df.columns}
    if raw_norm in by_norm:
        return by_norm[raw_norm]

    candidates = [str(col) for col in df.columns]
    match = difflib.get_close_matches(raw, candidates, n=1, cutoff=0.75)
    return match[0] if match else None


def _resolve_modality_value(df: pd.DataFrame, column: str, requested: Any) -> str | None:
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return None
    raw = str(requested or "").strip()
    if not raw:
        return None

    values = df[column].dropna().astype("string").unique().tolist()
    if not values:
        return None

    if raw in values:
        return raw

    by_norm = {str(value).strip().casefold(): str(value) for value in values}
    raw_norm = raw.casefold()
    if raw_norm in by_norm:
        return by_norm[raw_norm]

    match = difflib.get_close_matches(raw, values, n=1, cutoff=0.7)
    return match[0] if match else None


def _is_explicit_reference_to_previous_segment(question: str) -> bool:
    """Vrai uniquement quand la question renvoie explicitement au segment du tour précédent.

    Une demande générique de "profil" ne doit jamais réutiliser automatiquement le
    dernier segment mémorisé, sinon un ancien contexte comme workclass = ? peut polluer
    une nouvelle question du type "quel est le profil des femmes ?".
    """
    qlow = _normalize_text(question)
    return any(
        token in qlow
        for token in [
            "ce groupe",
            "ce segment",
            "cette catégorie",
            "cette categorie",
            "cette modalité",
            "cette modalite",
            "ce sous-groupe",
            "ce sous groupe",
            "celui-ci",
            "celle-ci",
            "eux",
            "elles",
        ]
    )


def _resolve_subset_from_question(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    resolved = resolve_segment_from_question(question, df)
    if resolved:
        return resolved

    # Fallback anaphorique strict : on ne reprend le segment précédent que si
    # l'utilisateur y fait explicitement référence. Ne jamais le faire sur le
    # seul mot "profil".
    if not _is_explicit_reference_to_previous_segment(question):
        return None

    last_segment = st.session_state.get("qa_segment_context") or {}
    column = str(last_segment.get("column") or "").strip()
    value = str(last_segment.get("value") or "").strip()
    if not column or not value or not isinstance(df, pd.DataFrame) or column not in df.columns:
        return None

    mask = df[column].astype("string") == value
    subset_df = df.loc[mask].copy()
    if not subset_df.empty:
        return {
            "column": column,
            "value": value,
            "df": subset_df,
            "description": f"{column} = {value}",
        }
    return None



def _filters_from_subset_info(subset_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(subset_info, dict):
        return []
    filters = subset_info.get("filters")
    if isinstance(filters, list):
        return [item for item in filters if isinstance(item, dict)]
    column = str(subset_info.get("column") or "").strip()
    value = str(subset_info.get("value") or "").strip()
    if column and value:
        return [{"column": column, "value": value}]
    return []


def _filters_from_action(action: dict[str, Any], df_ready: pd.DataFrame) -> list[dict[str, str]]:
    """Récupère les filtres explicitement fournis par le routeur Q&A.

    Formats acceptés :
    - {"filters": [{"column": "sexe", "value": "female"}, ...]}
    - {"filter": {"column": "sexe", "value": "female"}}
    - {"source_column": "sexe", "source_value": "female"}

    Cette fonction permet aussi les croisements : sexe=female ET race=Black.
    """

    raw_filters: list[dict[str, Any]] = []
    filters = action.get("filters")
    if isinstance(filters, list):
        raw_filters.extend(item for item in filters if isinstance(item, dict))
    single_filter = action.get("filter")
    if isinstance(single_filter, dict):
        raw_filters.append(single_filter)
    source_column = str(action.get("source_column") or "").strip()
    source_value = str(action.get("source_value") or "").strip()
    if source_column and source_value:
        raw_filters.append({"column": source_column, "value": source_value})
    return normalize_subset_filters(df_ready, raw_filters)


def _resolve_subset_filters_for_action(action: dict[str, Any], question: str, df_ready: pd.DataFrame) -> list[dict[str, str]]:
    explicit_filters = _filters_from_action(action, df_ready)
    if explicit_filters:
        return explicit_filters

    inferred_filters = infer_subset_filters_from_question(df_ready, question)
    if inferred_filters:
        return inferred_filters

    subset_info = _resolve_subset_from_question(question, df_ready)
    return normalize_subset_filters(df_ready, _filters_from_subset_info(subset_info))


def _build_category_context_table(df: pd.DataFrame, column: str) -> pd.DataFrame | None:
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return None
    counts = df[column].astype("string").value_counts(dropna=False)
    if counts.empty:
        return None
    total = int(counts.sum())
    context_df = pd.DataFrame(
        {
            "Modalité": [str(idx) for idx in counts.index.tolist()],
            "Effectif": counts.astype(int).tolist(),
            "%": [round((int(val) / total) * 100, 1) if total else 0.0 for val in counts.tolist()],
        }
    )
    return context_df


def _get_crosstab_item(var_a: str, var_b: str) -> dict[str, Any] | None:
    items = st.session_state.get("crosstabs_interpretation", []) or []
    for item in items:
        xa = str(item.get("var_x"))
        ya = str(item.get("var_y"))
        if {xa, ya} == {var_a, var_b}:
            return item
    return None


def _get_distribution_item(var_name: str) -> dict[str, Any] | None:
    items = (
        st.session_state.get("figs_variables_distribution_detailed")
        or st.session_state.get("figs_variables_distribution")
        or []
    )
    for item in items:
        if str(item.get("title", "")).strip().casefold() == str(var_name).strip().casefold():
            return item
    return None


def _build_relationship_pairs_and_variables(
    raw_pairs: list[Any],
    raw_variables: list[Any],
    matched_columns: list[str],
    df_ready: pd.DataFrame,
) -> tuple[list[list[str]], list[str]]:
    pairs: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    variables: list[str] = []

    for pair in raw_pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        left = _resolve_column_name(df_ready, pair[0])
        right = _resolve_column_name(df_ready, pair[1])
        if not left or not right or left == right:
            continue
        key = tuple(sorted([left, right]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pairs.append([left, right])
        if left not in variables:
            variables.append(left)
        if right not in variables:
            variables.append(right)

    for item in raw_variables or []:
        column = _resolve_column_name(df_ready, item)
        if column and column not in variables:
            variables.append(column)

    if not pairs and len(matched_columns) >= 2:
        base_pair = [matched_columns[0], matched_columns[1]]
        pairs.append(base_pair)
        for column in base_pair:
            if column not in variables:
                variables.append(column)

    if not variables:
        variables = matched_columns[:3]

    if len(variables) >= 3 and len(pairs) < 2:
        focus = variables[0]
        for other in variables[1:3]:
            candidate = tuple(sorted([focus, other]))
            if focus != other and candidate not in seen_pairs:
                seen_pairs.add(candidate)
                pairs.append([focus, other])

    return pairs, variables[:4]


def _build_planner_capability_guide(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guide: list[dict[str, Any]] = []
    for item in catalog:
        guide.append(
            {
                "action": str(item.get("action") or "").strip(),
                "when_to_use": str(item.get("when_to_use") or "").strip(),
                "example_questions": [str(q).strip() for q in (item.get("example_questions") or []) if str(q).strip()][:4],
                "example_parameters": (item.get("example_parameters") or [])[:3],
                "params_schema": item.get("params_schema") or [],
            }
        )
    return guide


def _generate_relationship_synthesis(question: str, pairs: list[list[str]], variables: list[str]) -> str:
    crosstab_payload = []
    for pair in pairs:
        item = _get_crosstab_item(pair[0], pair[1])
        if not isinstance(item, dict):
            continue
        crosstab_payload.append(
            {
                "var_x": item.get("var_x"),
                "var_y": item.get("var_y"),
                "interpretation": str(item.get("interpretation") or "").strip(),
                "v": item.get("v"),
                "p": item.get("p"),
            }
        )

    distribution_payload = []
    for variable in variables:
        item = _get_distribution_item(variable)
        if not isinstance(item, dict):
            continue
        distribution_payload.append(
            {
                "variable": variable,
                "metrics_caption": str(item.get("metrics_caption") or "").strip(),
            }
        )

    if not crosstab_payload and not distribution_payload:
        return ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu synthétises une analyse de relations entre variables. "
                    "Réponds en français, en 3 à 6 phrases maximum, sans jargon interne. "
                    "Appuie-toi d'abord sur les tris croisés et complète par les distributions si elles apportent du contexte utile."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "pairs": pairs,
                        "variables": variables,
                        "crosstabs": crosstab_payload,
                        "distributions": distribution_payload,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ],
    )
    return str(response.choices[0].message.content or "").strip()


def summarize_sankey_pairs(results_store: Any, max_items: int = 20) -> list[dict[str, Any]]:
    if not isinstance(results_store, dict) or not results_store:
        return []

    rows = []
    for pair_id, item in results_store.items():
        interpretation = str(item.get("interpretation", "") or "").strip()
        if len(interpretation) > 500:
            interpretation = interpretation[:500] + "..."
        rows.append(
            {
                "pair_id": pair_id,
                "var_x": item.get("var_x"),
                "var_y": item.get("var_y"),
                "v": item.get("v"),
                "p": item.get("p"),
                "chi2": item.get("chi2"),
                "interpretation": interpretation,
            }
        )

    rows.sort(key=lambda row: float(row.get("v") or 0), reverse=True)
    return rows[:max_items]


def _sample_top_modalities(df: pd.DataFrame, columns: list[str], max_values: int = 5) -> dict[str, list[str]]:
    sample: dict[str, list[str]] = {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return sample
    for column in columns:
        if column not in df.columns:
            continue
        try:
            values = (
                df[column]
                .dropna()
                .astype("string")
                .value_counts(dropna=True)
                .head(max_values)
                .index.tolist()
            )
            sample[column] = [str(value) for value in values]
        except Exception:
            continue
    return sample


def _build_question_payload(question: str, df_ready: pd.DataFrame) -> dict[str, Any]:
    matched_columns = _find_cols_in_question(question, df_ready) if isinstance(df_ready, pd.DataFrame) else []
    target_variables = [str(item) for item in (st.session_state.get("target_variables", []) or [])]
    illustrative_variables = [str(item) for item in (st.session_state.get("illustrative_variables", []) or [])]
    columns_for_modalities = list(dict.fromkeys(target_variables + matched_columns[:6]))

    preview = ""
    if isinstance(df_ready, pd.DataFrame) and not df_ready.empty:
        preview = df_ready.head(12).to_csv(index=False)
        preview = preview[:20000]

    return {
        "question": question,
        "columns": [str(col) for col in df_ready.columns] if isinstance(df_ready, pd.DataFrame) else [],
        "matched_columns": matched_columns,
        "target_variables": target_variables,
        "illustrative_variables": illustrative_variables,
        "top_modalities_by_column": _sample_top_modalities(df_ready, columns_for_modalities),
        "dataset_context": st.session_state.get("dataset_context"),
        "dataset_recommendations": st.session_state.get("dataset_recommendations"),
        "global_synthesis": st.session_state.get("global_synthesis"),
        "data_preparation_synthesis": st.session_state.get("data_preparation_synthesis"),
        "sankey_interpretation_synthesis": st.session_state.get("sankey_interpretation_synthesis"),
        "profil_dominant_analysis": st.session_state.get("profil_dominant_analysis"),
        "qa_segment_profile_text": st.session_state.get("qa_segment_profile_text"),
        "qa_segment_profils_y_text": st.session_state.get("qa_segment_profils_y_text"),
        "latent_summary_text": st.session_state.get("latent_summary_text"),
        "sankey_pair_results_summary": summarize_sankey_pairs(
            st.session_state.get("sankey_pair_results", {}),
            max_items=20,
        ),
        "sankey_latents_csv": to_text(st.session_state.get("sankey_latents"))
        if isinstance(st.session_state.get("sankey_latents"), pd.DataFrame)
        else "",
        "crosstabs_interpretation": [
            {
                "var_x": item.get("var_x"),
                "var_y": item.get("var_y"),
                "interpretation": item.get("interpretation", ""),
            }
            for item in (st.session_state.get("crosstabs_interpretation") or [])
            if isinstance(item, dict)
        ],
        "data_sample_preview_as_csv": preview,
        "qa_recent_history": get_recent_qa_history(),
        "qa_conversation_summary": st.session_state.get(QA_CONVERSATION_SUMMARY_KEY, ""),
        "qa_last_analysis_suggestion": st.session_state.get(QA_LAST_FOLLOWUP_KEY, ""),
    }


def _plan_qa_actions(question: str, df_ready: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(df_ready, pd.DataFrame) or df_ready.empty:
        return {
            "can_answer_from_existing": False,
            "actions": [],
            "internal_notes": "dataset indisponible",
            "raw_answer": "",
        }

    payload = _build_question_payload(question, df_ready)
    catalog = get_analysis_capability_catalog()
    payload["capability_catalog"] = catalog
    payload["capability_guide"] = _build_planner_capability_guide(catalog)
    payload["available_actions"] = [item["action"] for item in catalog]
    payload["artefacts_available"] = {
        "global_synthesis": bool(str(st.session_state.get("global_synthesis") or "").strip()),
        "data_preparation_synthesis": bool(str(st.session_state.get("data_preparation_synthesis") or "").strip()),
        "sankey": bool(st.session_state.get("sankey_diagram")),
        "sankey_text": bool(str(st.session_state.get("sankey_interpretation_synthesis") or "").strip()),
        "profils_y": bool(str(st.session_state.get("profils_y_text") or "").strip()),
        "qa_segment_profils_y_text": bool(str(st.session_state.get("qa_segment_profils_y_text") or "").strip()),
        "profil_dominant_analysis": bool(str(st.session_state.get("profil_dominant_analysis") or "").strip()),
        "qa_segment_profile_text": bool(str(st.session_state.get("qa_segment_profile_text") or "").strip()),
        "crosstabs": bool(st.session_state.get("crosstabs_interpretation")),
        "distributions": bool(
            st.session_state.get("figs_variables_distribution_detailed")
            or st.session_state.get("figs_variables_distribution")
        ),
    }

    sys_prompt = """Tu es l'orchestrateur Q&A d'une application Streamlit d'analyse d'enquêtes.
Réponds uniquement par un JSON strict.

Objectif :
1. Dire si les artefacts existants suffisent pour répondre à la question.
2. Si non, choisir les actions nécessaires parmi available_actions.
3. Déduire si la question demande une nouvelle variable cible, une nouvelle modalité de cible,
   des tris croisés, des distributions, ou l'exécution d'un module de préparation.

Le payload contient aussi "capability_guide". Pour chaque action, tu dois utiliser :
- "when_to_use" pour comprendre l'intention métier,
- "example_questions" pour rapprocher la question utilisateur d'un cas d'usage,
- "example_parameters" pour choisir des paramètres plausibles et cohérents.

Règles :
- N'utilise que des noms de colonnes présents dans "columns".
- Ne crée aucune variable ni score inexistant.
- Compare explicitement la question utilisateur aux exemples du "capability_guide" avant de choisir une action.
- Si la question demande l'effectif, la part, le pourcentage ou la taille d'une catégorie identifiable, utilise "contextualize_segment".
- Si la question demande le profil, le portrait ou les caractéristiques d'une catégorie identifiable, utilise "run_distribution_profile" sans ajouter "contextualize_segment", sauf si la question demande aussi explicitement l'effectif ou la part du groupe.
- Si la question porte sur une nouvelle variable cible, utilise l'action "rerun_sankey".
- Si la question porte sur une nouvelle modalité de cible, utilise l'action "rerun_profils_y".
- Si la question demande plutôt un portrait / profil majoritaire ou la description d'une catégorie sans cible analytique explicite, utilise "run_distribution_profile".
- Si la question demande comment les profils se distinguent au sein d'un groupe ou demande plus de détail sur un segment, utilise "rerun_profils_y_for_segment".
- Si la question porte sur une relation, un lien, une comparaison ou un croisement entre variables, utilise en priorité "analyze_relationships".
- Si des tris croisés sont nécessaires, utilise "run_crosstabs" avec des paires de variables.
- Si des histogrammes ou distributions sont nécessaires, utilise "run_distributions".
- Si la réponse doit montrer un module de préparation, utilise "run_preparation1" ou "run_preparation2".
- Si les artefacts existants suffisent, laisse "actions" vide.
- "internal_notes" est réservé à l'app et ne sera pas affiché à l'utilisateur.

Format attendu :
{
  "can_answer_from_existing": true|false,
  "actions": [
    {
      "action": "contextualize_segment" | "analyze_relationships" | "run_crosstabs" | "run_distributions" | "run_distribution_profile" | "rerun_sankey" | "rerun_profils_y" | "rerun_profils_y_for_segment" | "run_preparation1" | "run_preparation2",
      "target_variable": "nom de colonne optionnel",
      "target_modality": "modalité optionnelle",
      "source_column": "nom de colonne optionnel",
      "source_value": "valeur optionnelle",
      "pairs": [["var_a","var_b"]],
      "variables": ["var_x"],
      "reason": "texte bref"
    }
  ],
  "internal_notes": "texte bref"
}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
    )
    raw_answer = response.choices[0].message.content or ""
    plan = _safe_json_loads(raw_answer) or {}
    plan["raw_answer"] = raw_answer
    return plan


def _sanitize_plan(plan: dict[str, Any], question: str, df_ready: pd.DataFrame) -> dict[str, Any]:
    allowed_actions = {item["action"] for item in get_analysis_capability_catalog()}
    matched_columns = _find_cols_in_question(question, df_ready)
    segment_info = _resolve_subset_from_question(question, df_ready)
    qlow = question.casefold()
    wants_category_profile = (
        any(keyword in qlow for keyword in ["profil", "portrait", "décrit", "decrit", "catégorie", "categorie", "segment", "qui sont", "ressembl"])
        and not any(keyword in qlow for keyword in ["cible", "modalité cible", "modalite cible", "top 20", "bottom 20"])
    )
    # Le contexte de segment sert uniquement à situer une catégorie en effectif / part.
    # Il ne doit pas être déclenché par une demande de profil, sinon d'anciens segments
    # mémorisés peuvent ressortir hors contexte (ex. workclass = ?).
    wants_segment_context = bool(
        segment_info
        and any(
            keyword in qlow
            for keyword in [
                "combien",
                "effectif",
                "effectifs",
                "part",
                "proportion",
                "pourcentage",
                "poids",
                "taille",
                "représent",
                "represent",
            ]
        )
    )

    wants_segment_deep_dive = bool(
        segment_info
        and any(
            keyword in qlow
            for keyword in [
                "dÃ©tail",
                "detail",
                "dÃ©taill",
                "distingu",
                "caractÃ©ris",
                "caracteris",
                "approfond",
                "plus en dÃ©tail",
                "plus en detail",
            ]
        )
    )

    sanitized: dict[str, Any] = {
        "can_answer_from_existing": bool(plan.get("can_answer_from_existing", False)),
        "internal_notes": str(plan.get("internal_notes") or "").strip(),
        "actions": [],
        "raw_answer": plan.get("raw_answer", ""),
    }

    raw_actions = plan.get("actions", [])
    if not isinstance(raw_actions, list):
        raw_actions = []

    seen_keys: set[tuple[Any, ...]] = set()

    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            continue
        action_name = str(raw_action.get("action") or "").strip()
        if action_name not in allowed_actions:
            continue

        action: dict[str, Any] = {
            "action": action_name,
            "reason": str(raw_action.get("reason") or "").strip(),
        }

        if action_name == "analyze_relationships":
            pairs, variables = _build_relationship_pairs_and_variables(
                raw_action.get("pairs", []) or [],
                raw_action.get("variables", []) or [],
                matched_columns,
                df_ready,
            )
            if pairs:
                action["pairs"] = pairs
                action["variables"] = variables
                sanitized["actions"].append(action)

        elif action_name == "run_crosstabs":
            pairs: list[list[str]] = []
            for pair in raw_action.get("pairs", []) or []:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                left = _resolve_column_name(df_ready, pair[0])
                right = _resolve_column_name(df_ready, pair[1])
                if not left or not right or left == right:
                    continue
                normalized_pair = tuple(sorted([left, right]))
                if normalized_pair in seen_keys:
                    continue
                seen_keys.add(normalized_pair)
                pairs.append([left, right])

            if not pairs and len(matched_columns) >= 2:
                fallback = [matched_columns[0], matched_columns[1]]
                key = tuple(sorted(fallback))
                if key not in seen_keys:
                    seen_keys.add(key)
                    pairs.append(fallback)

            if pairs:
                action["pairs"] = pairs
                sanitized["actions"].append(action)

        elif action_name == "run_distributions":
            variables: list[str] = []
            for item in raw_action.get("variables", []) or []:
                column = _resolve_column_name(df_ready, item)
                if not column or column in variables:
                    continue
                variables.append(column)

            if not variables:
                for column in matched_columns[:3]:
                    if column not in variables:
                        variables.append(column)

            if variables:
                action["variables"] = variables
                sanitized["actions"].append(action)

        elif action_name == "rerun_sankey":
            target_variable = _resolve_column_name(df_ready, raw_action.get("target_variable"))
            if target_variable:
                action["target_variable"] = target_variable
                sanitized["actions"].append(action)

        elif action_name == "rerun_profils_y":
            if wants_category_profile:
                continue
            target_variable = _resolve_column_name(
                df_ready,
                raw_action.get("target_variable") or (matched_columns[0] if matched_columns else None),
            )
            target_modality = _resolve_modality_value(df_ready, target_variable, raw_action.get("target_modality"))
            if target_variable and target_modality:
                action["target_variable"] = target_variable
                action["target_modality"] = target_modality
                sanitized["actions"].append(action)

        elif action_name == "rerun_profils_y_for_segment":
            if segment_info:
                action["source_column"] = str(segment_info.get("column") or "")
                action["source_value"] = str(segment_info.get("value") or "")
                sanitized["actions"].append(action)

        elif action_name in PROFILE_ACTIONS or action_name in SEGMENT_ACTIONS:
            filters = normalize_subset_filters(
                df_ready,
                (raw_action.get("filters") if isinstance(raw_action.get("filters"), list) else [])
                + ([raw_action.get("filter")] if isinstance(raw_action.get("filter"), dict) else [])
                + ([{"column": raw_action.get("source_column"), "value": raw_action.get("source_value")}]
                   if raw_action.get("source_column") and raw_action.get("source_value") else []),
            )
            if filters:
                action["filters"] = filters
                if len(filters) == 1:
                    action["source_column"] = filters[0]["column"]
                    action["source_value"] = filters[0]["value"]
            sanitized["actions"].append(action)

        elif action_name in PREPARATION_ACTIONS:
            sanitized["actions"].append(action)

    if not sanitized["actions"] and matched_columns:
        if wants_category_profile:
            sanitized["actions"].append(
                {
                    "action": "run_distribution_profile",
                    "reason": "fallback heuristique sur une demande de portrait ou profil majoritaire",
                }
            )
        elif any(keyword in qlow for keyword in ["croisé", "croise", "crosstab", "relation", "impact"]) and len(matched_columns) >= 2:
            sanitized["actions"].append(
                {
                    "action": "run_crosstabs",
                    "pairs": [[matched_columns[0], matched_columns[1]]],
                    "reason": "fallback heuristique sur les variables citées dans la question",
                }
            )
        elif any(keyword in qlow for keyword in ["distribution", "histogram", "répartition", "repartition"]):
            sanitized["actions"].append(
                {
                    "action": "run_distributions",
                    "variables": matched_columns[:3],
                    "reason": "fallback heuristique sur les variables citées dans la question",
                }
            )

    if not wants_segment_context:
        sanitized["actions"] = [
            action
            for action in sanitized["actions"]
            if action.get("action") != "contextualize_segment"
        ]

    if wants_segment_context and not any(action.get("action") == "contextualize_segment" for action in sanitized["actions"]):
        sanitized["actions"].insert(
            0,
            {
                "action": "contextualize_segment",
                "reason": "la question porte sur une catégorie identifiable à situer dans l'échantillon global",
            },
        )

    if wants_category_profile:
        sanitized["actions"] = [action for action in sanitized["actions"] if action.get("action") != "rerun_profils_y"]
        if not any(action.get("action") == "run_distribution_profile" for action in sanitized["actions"]):
            sanitized["actions"].insert(
                1 if sanitized["actions"] and sanitized["actions"][0].get("action") == "contextualize_segment" else 0,
                {
                    "action": "run_distribution_profile",
                    "reason": "demande de portrait ou de description d'une catégorie sans cible analytique explicite",
                },
            )

    if wants_segment_deep_dive:
        sanitized["actions"] = [action for action in sanitized["actions"] if action.get("action") != "rerun_profils_y"]
        if not any(action.get("action") == "rerun_profils_y_for_segment" for action in sanitized["actions"]):
            insert_at = 0
            for idx, action in enumerate(sanitized["actions"]):
                if action.get("action") in {"contextualize_segment", "run_distribution_profile"}:
                    insert_at = idx + 1
            sanitized["actions"].insert(
                insert_at,
                {
                    "action": "rerun_profils_y_for_segment",
                    "reason": "demande d'approfondissement sur un segment par comparaison au reste de l'Ã©chantillon",
                },
            )

    if any(keyword in qlow for keyword in ["croisÃ©", "croise", "crosstab", "relation", "impact", "lien", "compare"]) and len(matched_columns) >= 2:
        relation_actions = [action for action in sanitized["actions"] if action.get("action") in {"run_crosstabs", "run_distributions", "analyze_relationships"}]
        if relation_actions or not sanitized["actions"]:
            pairs, variables = _build_relationship_pairs_and_variables(
                [pair for action in relation_actions for pair in (action.get("pairs") or [])],
                [var for action in relation_actions for var in (action.get("variables") or [])],
                matched_columns,
                df_ready,
            )
            if pairs:
                sanitized["actions"] = [
                    action
                    for action in sanitized["actions"]
                    if action.get("action") not in {"run_crosstabs", "run_distributions", "analyze_relationships"}
                ]
                sanitized["actions"].append(
                    {
                        "action": "analyze_relationships",
                        "pairs": pairs,
                        "variables": variables,
                        "reason": "question de relation ou de croisement entre variables",
                    }
                )

    if sanitized["actions"]:
        sanitized["can_answer_from_existing"] = False

    return sanitized


def _set_target_variable_for_qa(target_variable: str) -> None:
    existing_targets = [str(item) for item in (st.session_state.get("target_variables", []) or []) if str(item) != target_variable]
    st.session_state["target_variables"] = [target_variable] + existing_targets
    st.session_state["brief_target_variable"] = target_variable


def _set_target_modality_for_qa(target_variable: str, target_modality: str) -> None:
    target_modalities = st.session_state.get("target_modalities", {}) or {}
    target_modalities[target_variable] = target_modality
    st.session_state["target_modalities"] = target_modalities


def _slugify_segment_token(value: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "segment"


def _build_segment_binary_target(df: pd.DataFrame, column: str, value: str) -> tuple[pd.DataFrame, str, str]:
    safe_column = _slugify_segment_token(column)
    safe_value = _slugify_segment_token(value)
    derived_target = f"__qa_target__{safe_column}_{safe_value}"
    df_segment = df.copy()
    mask = df_segment[column].astype("string").fillna("") == str(value)
    df_segment[derived_target] = mask.map({True: "segment", False: "reste"})
    return df_segment, derived_target, "segment"


def _format_segment_label(column: str, value: str) -> str:
    value_text = str(value or "").strip()
    column_text = str(column or "").strip()
    if not value_text:
        return column_text
    if not column_text:
        return value_text
    return f"{value_text} ({column_text})"


def _rewrite_segment_profils_text(text: str, column: str, value: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    lines = raw.splitlines()
    segment_label = _format_segment_label(column, value)
    objective = f"Objectif : voici comment se distinguent les profils au sein du groupe '{segment_label}'."
    if lines and lines[0].strip().lower().startswith("objectif"):
        lines[0] = objective
        return "\n".join(lines).strip()
    return f"{objective}\n\n{raw}"


def _suggestion_label(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("label") or item.get("instruction") or "").strip()
    return str(item or "").strip()


def _suggestion_instruction(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("instruction") or item.get("label") or "").strip()
    return str(item or "").strip()


def _make_analysis_suggestion(label: str, instruction: str | None = None, action_hint: str | None = None) -> dict[str, str]:
    suggestion = {
        "label": str(label or "").strip(),
        "instruction": str(instruction or label or "").strip(),
    }
    if action_hint:
        suggestion["action_hint"] = str(action_hint).strip()
    return suggestion


def _get_recent_suggested_questions(max_items: int = 10) -> list[str]:
    suggestions: list[str] = []
    for item in get_recent_qa_history(max_items=max_items):
        for suggestion in (item.get("analysis_suggestions") or item.get("followup_questions") or []):
            text = _suggestion_instruction(suggestion)
            if text and text not in suggestions:
                suggestions.append(text)
    return suggestions


def _canonical_question(text: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(text or "").strip())
    return " ".join(normalized.split())


def _build_adaptive_followups(
    question: str,
    execution_log: list[dict[str, Any]],
    llm_followups: list[Any],
) -> list[dict[str, str]]:
    previous = {_canonical_question(item): item for item in _get_recent_suggested_questions()}
    normalized_question = _canonical_question(question)
    recent_history = get_recent_qa_history(max_items=6)
    recent_actions = {
        str(action.get("action") or "").strip()
        for turn in recent_history
        for action in (turn.get("execution_log") or [])
        if isinstance(action, dict)
    }
    followups: list[dict[str, str]] = []

    for item in llm_followups:
        label = _suggestion_label(item)
        instruction = _suggestion_instruction(item)
        if not label or not instruction:
            continue
        lowered = _canonical_question(instruction)
        if lowered == normalized_question or lowered in previous:
            continue
        if instruction not in [_suggestion_instruction(x) for x in followups]:
            followups.append(_make_analysis_suggestion(label, instruction))

    for item in execution_log:
        action_name = str(item.get("action") or "").strip()
        if action_name == "run_distribution_profile":
            subset_value = str(item.get("subset_value") or "").strip()
            subset_column = str(item.get("subset_column") or "").strip()
            if subset_value and subset_column:
                suggestion = _make_analysis_suggestion(
                    f"Identifier les sous-profils de '{subset_value}'",
                    f"Identifie les profils distincts au sein du groupe '{subset_value}' pour la variable '{subset_column}'.",
                    "rerun_profils_y_for_segment",
                )
                if (
                    "rerun_profils_y_for_segment" not in recent_actions
                    and _canonical_question(suggestion["instruction"]) not in previous
                    and suggestion["instruction"] not in [_suggestion_instruction(x) for x in followups]
                ):
                    followups.append(suggestion)
        elif action_name == "rerun_profils_y_for_segment":
            subset_value = str(item.get("subset_value") or "").strip()
            subset_column = str(item.get("subset_column") or "").strip()
            if subset_value and subset_column:
                suggestion = _make_analysis_suggestion(
                    f"Comparer '{subset_value}' à une autre variable",
                    f"Compare le groupe '{subset_value}' ({subset_column}) à une autre variable de l'étude.",
                    "analyze_relationships",
                )
                if _canonical_question(suggestion["instruction"]) not in previous and suggestion["instruction"] not in [_suggestion_instruction(x) for x in followups]:
                    followups.append(suggestion)
                suggestion = _make_analysis_suggestion(
                    f"Explorer les relations autour de '{subset_column}'",
                    f"Explore les relations entre '{subset_column}' et une autre variable pour ce groupe.",
                    "analyze_relationships",
                )
                if _canonical_question(suggestion["instruction"]) not in previous and suggestion["instruction"] not in [_suggestion_instruction(x) for x in followups]:
                    followups.append(suggestion)
        elif action_name == "analyze_relationships":
            variables = item.get("available_variables") or item.get("variables") or []
            if isinstance(variables, list) and len(variables) >= 2:
                suggestion = _make_analysis_suggestion(
                    f"Approfondir la relation {variables[0]} × {variables[1]}",
                    f"Approfondis la relation entre {variables[0]} et {variables[1]} sur un segment précis.",
                    "analyze_relationships",
                )
                if _canonical_question(suggestion["instruction"]) not in previous and suggestion["instruction"] not in [_suggestion_instruction(x) for x in followups]:
                    followups.append(suggestion)

    generic_fallbacks = [
        _make_analysis_suggestion("Approfondir un segment précis", "Approfondis un autre segment précis de la population."),
        _make_analysis_suggestion("Croiser avec une autre variable", "Croise cette analyse avec une autre variable encore non explorée."),
        _make_analysis_suggestion("Détailler un profil identifié", "Produis une lecture plus détaillée d'un autre profil identifié."),
    ]
    for suggestion in generic_fallbacks:
        if _canonical_question(suggestion["instruction"]) in previous or _canonical_question(suggestion["instruction"]) == normalized_question:
            continue
        if suggestion["instruction"] not in [_suggestion_instruction(x) for x in followups]:
            followups.append(suggestion)

    return followups[:3]


def _clear_qa_force_flags() -> None:
    st.session_state["__QA_FORCE_CROSSTABS__"] = False
    st.session_state["__QA_SELECTED_CROSSTAB_PAIRS__"] = []
    st.session_state["__QA_FORCE_DISTRIBUTIONS__"] = False
    st.session_state["__QA_SELECTED_DISTRIBUTION_VARS__"] = []
    st.session_state["__QA_PROFILE_MODE__"] = False
    st.session_state["__QA_PROFILE_TARGET_VARIABLE__"] = None
    st.session_state["__QA_PROFILE_TARGET_MODALITY__"] = None


def _execute_action_plan(plan: dict[str, Any], df_ready: pd.DataFrame, question: str = "") -> list[dict[str, Any]]:
    execution_log: list[dict[str, Any]] = []
    _clear_qa_force_flags()

    action_priority = {
        "run_preparation1": 1,
        "run_preparation2": 2,
        "rerun_sankey": 3,
        "rerun_profils_y": 4,
        "rerun_profils_y_for_segment": 5,
        "contextualize_segment": 6,
        "run_distribution_profile": 7,
        "analyze_relationships": 8,
        "run_crosstabs": 9,
        "run_distributions": 10,
    }

    actions = sorted(plan.get("actions", []), key=lambda item: action_priority.get(item.get("action"), 99))

    for action in actions:
        action_name = action.get("action")
        result: dict[str, Any] = {"action": action_name, "reason": action.get("reason", "")}

        try:
            if action_name == "contextualize_segment":
                subset_info = _resolve_subset_from_question(question, df_ready)
                if subset_info:
                    subset_column = str(subset_info.get("column") or "")
                    subset_value = str(subset_info.get("value") or "")
                    subset_df = subset_info.get("df")
                    counts_df, percent_df = build_segment_context_tables(df_ready, subset_column)
                    segment_count = len(subset_df) if isinstance(subset_df, pd.DataFrame) else 0
                    total_count = len(df_ready) if isinstance(df_ready, pd.DataFrame) else 0
                    segment_share = round((segment_count / total_count) * 100, 1) if total_count else 0.0
                    intro_text = build_segment_intro(subset_column, subset_value, segment_count, segment_share, total_count)
                    st.session_state["qa_segment_context"] = {
                        "column": subset_column,
                        "value": subset_value,
                        "count": segment_count,
                        "share": segment_share,
                        "intro": intro_text,
                    }
                    st.session_state["qa_segment_counts_table"] = counts_df
                    st.session_state["qa_segment_percent_table"] = percent_df
                    st.session_state["qa_segment_subdataset"] = subset_df
                    result["status"] = "completed"
                    result["subset"] = subset_info.get("description")
                    result["subset_column"] = subset_column
                    result["subset_value"] = subset_value
                    result["intro"] = intro_text
                    result["counts_table"] = counts_df
                    result["percent_table"] = percent_df
                else:
                    result["status"] = "skipped"
                    result["error"] = "segment non résolu"

            elif action_name == "rerun_sankey":
                target_variable = action.get("target_variable")
                if target_variable:
                    _set_target_variable_for_qa(target_variable)
                st.session_state["run_sankey_crosstabs"] = True
                run_diagram_sankey()
                result["status"] = "completed"
                result["target_variable"] = target_variable
                result["produced"] = {
                    "sankey_interpretation_synthesis": bool(
                        str(st.session_state.get("sankey_interpretation_synthesis") or "").strip()
                    ),
                    "sankey_pair_results": bool(st.session_state.get("sankey_pair_results")),
                }

            elif action_name == "rerun_profils_y":
                target_variable = action.get("target_variable")
                target_modality = action.get("target_modality")
                if target_variable:
                    _set_target_variable_for_qa(target_variable)
                if target_variable and target_modality:
                    _set_target_modality_for_qa(target_variable, target_modality)
                st.session_state["__QA_PROFILE_MODE__"] = True
                st.session_state["__QA_PROFILE_TARGET_VARIABLE__"] = target_variable
                st.session_state["__QA_PROFILE_TARGET_MODALITY__"] = target_modality
                st.session_state["_selected_cols"] = None
                st.session_state["_sig_step2"] = None
                st.session_state["step3_ready"] = False
                st.session_state["step5_ready"] = False
                run_profils_y()
                result["status"] = "completed"
                result["target_variable"] = target_variable
                result["target_modality"] = target_modality
                result["produced"] = {
                    "profils_y_text": bool(str(st.session_state.get("profils_y_text") or "").strip()),
                }
                st.session_state["__QA_PROFILE_MODE__"] = False
                st.session_state["__QA_PROFILE_TARGET_VARIABLE__"] = None
                st.session_state["__QA_PROFILE_TARGET_MODALITY__"] = None

            elif action_name == "rerun_profils_y_for_segment":
                subset_info = _resolve_subset_from_question(question, df_ready)
                if not subset_info:
                    result["status"] = "skipped"
                    result["error"] = "segment non rÃ©solu"
                else:
                    source_column = str(subset_info.get("column") or "")
                    source_value = str(subset_info.get("value") or "")
                    temp_df, derived_target, derived_modality = _build_segment_binary_target(df_ready, source_column, source_value)
                    temp_df = temp_df.drop(columns=[source_column], errors="ignore")
                    original_df_ready = st.session_state.get("df_ready")
                    original_target_variables = list(st.session_state.get("target_variables", []) or [])
                    original_brief_target = st.session_state.get("brief_target_variable")
                    original_target_modalities = dict(st.session_state.get("target_modalities", {}) or {})
                    original_profils_y_text = st.session_state.get("profils_y_text")
                    original_profils_y_simplified = st.session_state.get("profils_y_simplified")
                    original_profils_y_table = st.session_state.get("profils_y_table")

                    st.session_state["qa_segment_profils_y_text"] = ""
                    st.session_state["df_ready"] = temp_df
                    _set_target_variable_for_qa(derived_target)
                    _set_target_modality_for_qa(derived_target, derived_modality)
                    st.session_state["__QA_PROFILE_MODE__"] = True
                    st.session_state["__QA_PROFILE_TARGET_VARIABLE__"] = derived_target
                    st.session_state["__QA_PROFILE_TARGET_MODALITY__"] = derived_modality
                    st.session_state["_selected_cols"] = None
                    st.session_state["_sig_step2"] = None
                    st.session_state["step3_ready"] = False
                    st.session_state["step5_ready"] = False

                    run_profils_y()

                    st.session_state["qa_segment_profils_y_text"] = _rewrite_segment_profils_text(
                        str(st.session_state.get("profils_y_text") or "").strip(),
                        source_column,
                        source_value,
                    )
                    st.session_state["df_ready"] = original_df_ready
                    st.session_state["target_variables"] = original_target_variables
                    st.session_state["brief_target_variable"] = original_brief_target
                    st.session_state["target_modalities"] = original_target_modalities
                    st.session_state["profils_y_text"] = original_profils_y_text
                    st.session_state["profils_y_simplified"] = original_profils_y_simplified
                    st.session_state["profils_y_table"] = original_profils_y_table
                    st.session_state["__QA_PROFILE_MODE__"] = False
                    st.session_state["__QA_PROFILE_TARGET_VARIABLE__"] = None
                    st.session_state["__QA_PROFILE_TARGET_MODALITY__"] = None

                    result["status"] = "completed"
                    result["subset"] = subset_info.get("description")
                    result["subset_column"] = source_column
                    result["subset_value"] = source_value
                    result["derived_target"] = derived_target
                    result["excluded_columns"] = [source_column]
                    result["produced"] = {
                        "qa_segment_profils_y_text": bool(str(st.session_state.get("qa_segment_profils_y_text") or "").strip()),
                    }

            elif action_name == "analyze_relationships":
                pairs = [tuple(pair) for pair in (action.get("pairs") or [])]
                variables = [str(item) for item in (action.get("variables") or [])]
                st.session_state["qa_relationship_synthesis"] = ""
                st.session_state["__QA_FORCE_CROSSTABS__"] = True
                st.session_state["__QA_SELECTED_CROSSTAB_PAIRS__"] = pairs
                st.session_state["run_sankey_crosstabs"] = True
                run_crosstabs_detail()
                st.session_state["__QA_FORCE_DISTRIBUTIONS__"] = True
                st.session_state["__QA_SELECTED_DISTRIBUTION_VARS__"] = variables
                st.session_state["generate_distribution_figures"] = True
                run_distributions_detail()
                available_pairs = [
                    list(pair)
                    for pair in pairs
                    if _get_crosstab_item(pair[0], pair[1]) is not None
                ]
                available_variables = [
                    variable for variable in variables if _get_distribution_item(variable) is not None
                ]
                st.session_state["qa_relationship_synthesis"] = _generate_relationship_synthesis(
                    question,
                    available_pairs,
                    available_variables,
                )
                result["status"] = "completed"
                result["pairs"] = [list(pair) for pair in pairs]
                result["variables"] = variables
                result["available_pairs"] = available_pairs
                result["available_variables"] = available_variables
                result["relationship_synthesis"] = st.session_state.get("qa_relationship_synthesis", "")
                st.session_state["__QA_FORCE_CROSSTABS__"] = False
                st.session_state["__QA_SELECTED_CROSSTAB_PAIRS__"] = []
                st.session_state["__QA_FORCE_DISTRIBUTIONS__"] = False
                st.session_state["__QA_SELECTED_DISTRIBUTION_VARS__"] = []

            elif action_name == "run_crosstabs":
                pairs = [tuple(pair) for pair in (action.get("pairs") or [])]
                st.session_state["__QA_FORCE_CROSSTABS__"] = True
                st.session_state["__QA_SELECTED_CROSSTAB_PAIRS__"] = pairs
                st.session_state["run_sankey_crosstabs"] = True
                run_crosstabs_detail()
                result["status"] = "completed"
                result["pairs"] = [list(pair) for pair in pairs]
                result["available_pairs"] = [
                    list(pair)
                    for pair in pairs
                    if _get_crosstab_item(pair[0], pair[1]) is not None
                ]
                st.session_state["__QA_FORCE_CROSSTABS__"] = False
                st.session_state["__QA_SELECTED_CROSSTAB_PAIRS__"] = []

            elif action_name == "run_distributions":
                variables = [str(item) for item in (action.get("variables") or [])]
                st.session_state["__QA_FORCE_DISTRIBUTIONS__"] = True
                st.session_state["__QA_SELECTED_DISTRIBUTION_VARS__"] = variables
                st.session_state["generate_distribution_figures"] = True
                run_distributions_detail()
                result["status"] = "completed"
                result["variables"] = variables
                result["available_variables"] = [
                    variable for variable in variables if _get_distribution_item(variable) is not None
                ]
                st.session_state["__QA_FORCE_DISTRIBUTIONS__"] = False
                st.session_state["__QA_SELECTED_DISTRIBUTION_VARS__"] = []

            elif action_name in PROFILE_ACTIONS:
                filters = _resolve_subset_filters_for_action(action, question, df_ready)
                original_df_ready = st.session_state.get("df_ready")
                original_profile_text = st.session_state.get("profil_dominant_analysis")
                original_segment_profile_text = st.session_state.get("qa_segment_profile_text")
                original_dominant_continues = st.session_state.get("dominant_continues")
                original_dominant_discretes = st.session_state.get("dominant_discretes")

                st.session_state["__QA_SILENT__"] = True
                st.session_state["__QA_PROFILE_OUTPUT_KEY__"] = "profil_dominant_analysis"

                if filters:
                    segment_df, subset_spec = build_subset_for_analysis(
                        df_ready,
                        filters,
                        exclude_filter_columns=True,
                    )
                    if segment_df.empty:
                        result["status"] = "skipped"
                        result["error"] = f"aucune observation pour le sous-groupe : {subset_spec.description}"
                        st.session_state["__QA_SILENT__"] = False
                        st.session_state.pop("__QA_SEGMENT_DF__", None)
                        st.session_state.pop("__QA_PROFILE_OUTPUT_KEY__", None)
                        execution_log.append(result)
                        continue

                    st.session_state["qa_segment_profile_text"] = ""
                    st.session_state["__QA_SEGMENT_DF__"] = segment_df
                    st.session_state["__QA_PROFILE_OUTPUT_KEY__"] = "qa_segment_profile_text"
                    st.session_state["qa_last_subset_description"] = subset_spec.description
                    st.session_state["qa_segment_subdataset"] = segment_df
                    result["subset"] = subset_spec.description
                    result["filters"] = list(subset_spec.filters)
                    result["excluded_columns"] = list(subset_spec.excluded_columns)
                    result["subset_row_count"] = subset_spec.row_count
                    result["total_row_count"] = subset_spec.total_count
                else:
                    subset_spec = None

                PROFILE_ACTIONS[action_name]()
                st.session_state["__QA_SILENT__"] = False
                st.session_state.pop("__QA_SEGMENT_DF__", None)
                st.session_state.pop("__QA_PROFILE_OUTPUT_KEY__", None)
                if original_df_ready is not None:
                    st.session_state["df_ready"] = original_df_ready
                st.session_state["dominant_continues"] = original_dominant_continues
                st.session_state["dominant_discretes"] = original_dominant_discretes
                result["status"] = "completed"
                result["produced"] = {
                    "profil_dominant_analysis": bool(str(st.session_state.get("profil_dominant_analysis") or "").strip()),
                    "qa_segment_profile_text": bool(str(st.session_state.get("qa_segment_profile_text") or "").strip()),
                }
                if filters:
                    result["profile_output_key"] = "qa_segment_profile_text"
                    if len(filters) == 1:
                        result["subset_column"] = filters[0]["column"]
                        result["subset_value"] = filters[0]["value"]
                    if not str(st.session_state.get("qa_segment_profile_text") or "").strip():
                        result["status"] = "error"
                        result["error"] = "le profil du sous-groupe n'a pas été produit"
                elif original_profile_text != st.session_state.get("profil_dominant_analysis"):
                    result["subset"] = "population complète"

            elif action_name in PREPARATION_ACTIONS:
                PREPARATION_ACTIONS[action_name]()
                result["status"] = "completed"
                result["produced"] = {
                    "data_preparation_synthesis": bool(
                        str(st.session_state.get("data_preparation_synthesis") or "").strip()
                    ),
                    "process": bool(st.session_state.get("process") is not None),
                }
            else:
                result["status"] = "skipped"
                result["error"] = "action non prise en charge"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)

        execution_log.append(result)

    _clear_qa_force_flags()
    st.session_state["qa_last_execution_log"] = execution_log
    return execution_log


def _build_final_answer_payload(question: str, execution_log: list[dict[str, Any]], df_ready: pd.DataFrame) -> dict[str, Any]:
    payload = _build_question_payload(question, df_ready)
    payload["execution_log"] = execution_log
    payload["brief_analysis_plan"] = st.session_state.get("brief_analysis_plan", [])
    payload["profils_y_text"] = st.session_state.get("profils_y_text", "")
    payload["qa_segment_context"] = st.session_state.get("qa_segment_context")
    payload["qa_segment_profile_text"] = st.session_state.get("qa_segment_profile_text", "")
    payload["qa_relationship_synthesis"] = st.session_state.get("qa_relationship_synthesis", "")
    if any(str(item.get("action") or "").strip() == "rerun_profils_y_for_segment" for item in execution_log):
        payload.pop("qa_segment_profils_y_text", None)
        payload["qa_segment_profils_y_available"] = bool(
            str(st.session_state.get("qa_segment_profils_y_text") or "").strip()
        )
    else:
        payload["qa_segment_profils_y_text"] = st.session_state.get("qa_segment_profils_y_text", "")
    payload["process_text"] = to_text(st.session_state.get("process")) if st.session_state.get("process") is not None else ""
    return payload


def _build_existing_analysis_digest(exclude_sources: set[str] | None = None) -> list[dict[str, str]]:
    digest: list[dict[str, str]] = []
    excluded = exclude_sources or set()
    for key, label in [
        ("global_synthesis", "Synthèse globale"),
        ("data_preparation_synthesis", "Préparation des données"),
        ("sankey_interpretation_synthesis", "Relations principales"),
        ("latent_summary_text", "Dimensions latentes"),
        ("qa_relationship_synthesis", "Synthèse des relations"),
        ("qa_segment_profile_text", "Profil dominant du segment"),
        ("qa_segment_profils_y_text", "Profils dÃ©taillÃ©s du segment"),
        ("profil_dominant_analysis", "Profil dominant"),
        ("profils_y_text", "Profils détaillés"),
    ]:
        if key in excluded:
            continue
        text = str(st.session_state.get(key) or "").strip()
        if text:
            digest.append({"source": key, "label": label, "text": text[:4000]})
    return digest


def _normalize_followup_questions(parsed: dict[str, Any]) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []
    raw_list = parsed.get("analysis_suggestions") or parsed.get("followup_questions")
    if isinstance(raw_list, list):
        for item in raw_list:
            label = _suggestion_label(item)
            instruction = _suggestion_instruction(item)
            if label and instruction and instruction not in [_suggestion_instruction(x) for x in suggestions]:
                suggestions.append(_make_analysis_suggestion(label, instruction, item.get("action_hint") if isinstance(item, dict) else None))
    single = parsed.get("analysis_suggestion") or parsed.get("followup_question")
    label = _suggestion_label(single)
    instruction = _suggestion_instruction(single)
    if label and instruction and instruction not in [_suggestion_instruction(x) for x in suggestions]:
        suggestions.insert(0, _make_analysis_suggestion(label, instruction))
    return suggestions[:3]


def _generate_final_answer(question: str, execution_log: list[dict[str, Any]], df_ready: pd.DataFrame) -> dict[str, str]:
    payload = _build_final_answer_payload(question, execution_log, df_ready)
    excluded_sources: set[str] = set()
    if any(str(item.get("action") or "").strip() == "rerun_profils_y_for_segment" for item in execution_log):
        excluded_sources.add("qa_segment_profils_y_text")
    payload["existing_analysis_digest"] = _build_existing_analysis_digest(excluded_sources)
    payload["used_existing_only"] = not bool(execution_log)

    if payload["used_existing_only"] and payload["existing_analysis_digest"]:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu rédiges une réponse Q&A à partir d'analyses déjà produites. "
                        'Réponds uniquement par un JSON strict {"intro":"...","answer":"...","analysis_suggestions":[{"label":"...","instruction":"...","action_hint":"..."}]} '
                        "en t'appuyant d'abord sur existing_analysis_digest. "
                        "N'invente rien au-delà des synthèses fournies. "
                        "Les analysis_suggestions ne doivent pas être formulées comme des questions, "
                        "mais comme des propositions d'analyses cliquables. "
                        "Le champ label est le texte court affiché dans l'interface. "
                        "Le champ instruction est la demande complète envoyée au routeur Q&A."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "existing_analysis_digest": payload["existing_analysis_digest"],
                            "qa_segment_context": payload.get("qa_segment_context"),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
        )
        raw_answer = response.choices[0].message.content or ""
        parsed = _safe_json_loads(raw_answer) or {}
        followup_questions = _build_adaptive_followups(
            question,
            execution_log,
            _normalize_followup_questions(parsed),
        )
        return {
            "intro": str(parsed.get("intro") or "").strip(),
            "answer": str(parsed.get("answer") or "").strip(),
            "analysis_suggestion": followup_questions[0] if followup_questions else {},
            "analysis_suggestions": followup_questions,
            "followup_question": _suggestion_instruction(followup_questions[0]) if followup_questions else "",
            "followup_questions": [_suggestion_instruction(x) for x in followup_questions],
            "raw_answer": raw_answer,
        }

    sys_prompt = """Tu rédiges la réponse finale visible par l'utilisateur.
Réponds uniquement par un JSON strict {"intro":"...","answer":"...","analysis_suggestions":[{"label":"...","instruction":"...","action_hint":"..."}]}.

Règles :
- "intro" doit contenir 1 à 3 phrases maximum pour contextualiser la réponse.
- "answer" doit répondre clairement à la question en t'appuyant sur les analyses disponibles.
- "analysis_suggestions" doit proposer 2 ou 3 analyses utiles, concrètes, cliquables et cohérentes avec l'historique Q&A.
- Chaque suggestion contient un "label" court orienté action, et une "instruction" complète envoyable au routeur Q&A.
- Ne formule pas les suggestions comme des questions ; utilise des verbes d'analyse : comparer, explorer, identifier, détailler, recalculer.
- N'écris jamais "les artefacts sont suffisants", "notes LLM", "plan JSON", "outil", "capability" ou tout jargon interne.
- Si une analyse complémentaire a été exécutée, tu peux le mentionner simplement en langage métier.
- Si l'information manque encore, dis-le franchement et précise la limite sans inventer.
- Réponds en français clair et concis.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
    )
    raw_answer = response.choices[0].message.content or ""
    parsed = _safe_json_loads(raw_answer) or {}
    followup_questions = _build_adaptive_followups(
        question,
        execution_log,
        _normalize_followup_questions(parsed),
    )
    return {
        "intro": str(parsed.get("intro") or "").strip(),
        "answer": str(parsed.get("answer") or "").strip(),
        "analysis_suggestion": followup_questions[0] if followup_questions else {},
        "analysis_suggestions": followup_questions,
        "followup_question": _suggestion_instruction(followup_questions[0]) if followup_questions else "",
        "followup_questions": [_suggestion_instruction(x) for x in followup_questions],
        "raw_answer": raw_answer,
    }

def _render_selected_outputs_legacy(execution_log: list[dict[str, Any]]) -> None:
    for item in execution_log:
        action_name = item.get("action")

        if action_name == "contextualize_segment":
            intro = str(item.get("intro") or "").strip()
            counts_table = item.get("counts_table")
            percent_table = item.get("percent_table")
            if intro:
                st.write(intro)
            if isinstance(counts_table, pd.DataFrame) and not counts_table.empty:
                st.markdown("**Effectifs par modalité**")
                st.dataframe(counts_table, use_container_width=True, hide_index=True)
            if isinstance(percent_table, pd.DataFrame) and not percent_table.empty:
                st.markdown("**Pourcentages par modalité**")
                st.dataframe(percent_table, use_container_width=True, hide_index=True)

        elif action_name == "analyze_relationships":
            synthesis = str(item.get("relationship_synthesis") or st.session_state.get("qa_relationship_synthesis") or "").strip()
            if synthesis:
                st.write(synthesis)
            for pair in item.get("available_pairs", []) or []:
                var_a, var_b = pair[0], pair[1]
                crosstab_item = _get_crosstab_item(var_a, var_b)
                if not isinstance(crosstab_item, dict):
                    continue
                st.markdown(f"**{var_a} x {var_b}**")
                ct_count = crosstab_item.get("ct_count")
                if isinstance(ct_count, pd.DataFrame) and not ct_count.empty:
                    st.dataframe(ct_count, use_container_width=True)
                interpretation = str(crosstab_item.get("interpretation") or "").strip()
                if interpretation:
                    st.write(interpretation)
                heatmap_png = crosstab_item.get("heatmap_png")
                if isinstance(heatmap_png, (bytes, bytearray, memoryview)):
                    st.image(heatmap_png)
                caption = str(crosstab_item.get("metrics_caption") or "").strip()
                if caption:
                    st.caption(caption)
            for variable in item.get("available_variables", []) or []:
                dist_item = _get_distribution_item(variable)
                if not isinstance(dist_item, dict):
                    continue
                st.markdown(f"**Distribution de {variable}**")
                png = dist_item.get("png")
                if isinstance(png, (bytes, bytearray, memoryview)):
                    st.image(png)
                caption = str(dist_item.get("metrics_caption") or "").strip()
                if caption:
                    st.caption(caption)

        elif action_name == "run_crosstabs":
            for pair in item.get("available_pairs", []) or []:
                var_a, var_b = pair[0], pair[1]
                crosstab_item = _get_crosstab_item(var_a, var_b)
                if not isinstance(crosstab_item, dict):
                    continue
                st.markdown(f"**{var_a} x {var_b}**")
                ct_count = crosstab_item.get("ct_count")
                if isinstance(ct_count, pd.DataFrame) and not ct_count.empty:
                    st.dataframe(ct_count, use_container_width=True)
                interpretation = str(crosstab_item.get("interpretation") or "").strip()
                if interpretation:
                    st.write(interpretation)
                heatmap_png = crosstab_item.get("heatmap_png")
                if isinstance(heatmap_png, (bytes, bytearray, memoryview)):
                    st.image(heatmap_png)
                caption = str(crosstab_item.get("metrics_caption") or "").strip()
                if caption:
                    st.caption(caption)

        elif action_name == "run_distributions":
            for variable in item.get("available_variables", []) or []:
                dist_item = _get_distribution_item(variable)
                if not isinstance(dist_item, dict):
                    continue
                st.markdown(f"**Distribution de {variable}**")
                png = dist_item.get("png")
                if isinstance(png, (bytes, bytearray, memoryview)):
                    st.image(png)
                caption = str(dist_item.get("metrics_caption") or "").strip()
                if caption:
                    st.caption(caption)

        elif action_name == "run_distribution_profile":
            dominant_text = str(st.session_state.get("profil_dominant_analysis") or "").strip()
            subset = str(item.get("subset") or "").strip()
            if subset:
                if subset_column and subset_value:
                    st.write(
                        f"Pour situer cette catégorie dans l'ensemble du jeu de données, voici la répartition de la variable `{subset_column}`. "
                        f"La sous-population analysée correspond à `{subset_value}`."
                    )
                else:
                    st.caption(f"Sous-population analysée : {subset}")
            if isinstance(category_context_df, pd.DataFrame) and not category_context_df.empty:
                st.dataframe(category_context_df, use_container_width=True, hide_index=True)
            if dominant_text:
                st.markdown(dominant_text)


def _render_selected_outputs_v2(execution_log: list[dict[str, Any]]) -> None:
    for item in execution_log:
        action_name = item.get("action")

        if action_name == "contextualize_segment":
            intro = str(item.get("intro") or "").strip()
            counts_table = item.get("counts_table")
            percent_table = item.get("percent_table")
            if intro:
                st.write(intro)
            if isinstance(counts_table, pd.DataFrame) and not counts_table.empty:
                st.markdown("**Effectifs par modalité**")
                st.dataframe(counts_table, use_container_width=True, hide_index=True)
            if isinstance(percent_table, pd.DataFrame) and not percent_table.empty:
                st.markdown("**Pourcentages par modalité**")
                st.dataframe(percent_table, use_container_width=True, hide_index=True)
            continue

        if action_name == "analyze_relationships":
            synthesis = str(item.get("relationship_synthesis") or st.session_state.get("qa_relationship_synthesis") or "").strip()
            if synthesis:
                st.write(synthesis)
            for pair in item.get("available_pairs", []) or []:
                var_a, var_b = pair[0], pair[1]
                crosstab_item = _get_crosstab_item(var_a, var_b)
                if not isinstance(crosstab_item, dict):
                    continue
                st.markdown(f"**{var_a} x {var_b}**")
                ct_count = crosstab_item.get("ct_count")
                if isinstance(ct_count, pd.DataFrame) and not ct_count.empty:
                    st.dataframe(ct_count, use_container_width=True)
                interpretation = str(crosstab_item.get("interpretation") or "").strip()
                if interpretation:
                    st.write(interpretation)
                heatmap_png = crosstab_item.get("heatmap_png")
                if isinstance(heatmap_png, (bytes, bytearray, memoryview)):
                    st.image(heatmap_png)
                caption = str(crosstab_item.get("metrics_caption") or "").strip()
                if caption:
                    st.caption(caption)
            for variable in item.get("available_variables", []) or []:
                dist_item = _get_distribution_item(variable)
                if not isinstance(dist_item, dict):
                    continue
                st.markdown(f"**Distribution de {variable}**")
                png = dist_item.get("png")
                if isinstance(png, (bytes, bytearray, memoryview)):
                    st.image(png)
                caption = str(dist_item.get("metrics_caption") or "").strip()
                if caption:
                    st.caption(caption)
            continue

        if action_name == "run_crosstabs":
            for pair in item.get("available_pairs", []) or []:
                var_a, var_b = pair[0], pair[1]
                crosstab_item = _get_crosstab_item(var_a, var_b)
                if not isinstance(crosstab_item, dict):
                    continue
                st.markdown(f"**{var_a} x {var_b}**")
                ct_count = crosstab_item.get("ct_count")
                if isinstance(ct_count, pd.DataFrame) and not ct_count.empty:
                    st.dataframe(ct_count, use_container_width=True)
                interpretation = str(crosstab_item.get("interpretation") or "").strip()
                if interpretation:
                    st.write(interpretation)
                heatmap_png = crosstab_item.get("heatmap_png")
                if isinstance(heatmap_png, (bytes, bytearray, memoryview)):
                    st.image(heatmap_png)
                caption = str(crosstab_item.get("metrics_caption") or "").strip()
                if caption:
                    st.caption(caption)
            continue

        if action_name == "run_distributions":
            for variable in item.get("available_variables", []) or []:
                dist_item = _get_distribution_item(variable)
                if not isinstance(dist_item, dict):
                    continue
                st.markdown(f"**Distribution de {variable}**")
                png = dist_item.get("png")
                if isinstance(png, (bytes, bytearray, memoryview)):
                    st.image(png)
                caption = str(dist_item.get("metrics_caption") or "").strip()
                if caption:
                    st.caption(caption)
            continue

        if action_name == "run_distribution_profile":
            continue

        if action_name == "rerun_profils_y_for_segment":
            detailed_text = str(st.session_state.get("qa_segment_profils_y_text") or "").strip()
            if detailed_text:
                st.markdown(detailed_text)
            continue


def _extract_used_artifacts(execution_log: list[dict[str, Any]]) -> list[str]:
    used_artifacts: list[str] = []
    if str(st.session_state.get("global_synthesis") or "").strip():
        used_artifacts.append("global_synthesis")
    if str(st.session_state.get("data_preparation_synthesis") or "").strip():
        used_artifacts.append("data_preparation_synthesis")
    if str(st.session_state.get("sankey_interpretation_synthesis") or "").strip():
        used_artifacts.append("sankey_interpretation_synthesis")
    if str(st.session_state.get("profils_y_text") or "").strip():
        used_artifacts.append("profils_y_text")
    if str(st.session_state.get("qa_segment_profils_y_text") or "").strip():
        used_artifacts.append("qa_segment_profils_y_text")
    if str(st.session_state.get("qa_segment_profile_text") or "").strip():
        used_artifacts.append("qa_segment_profile_text")
    if str(st.session_state.get("qa_relationship_synthesis") or "").strip():
        used_artifacts.append("qa_relationship_synthesis")
    if str(st.session_state.get("profil_dominant_analysis") or "").strip():
        used_artifacts.append("profil_dominant_analysis")
    if st.session_state.get("qa_segment_context"):
        used_artifacts.append("qa_segment_context")
    if st.session_state.get("crosstabs_interpretation"):
        used_artifacts.append("crosstabs_interpretation")
    if st.session_state.get("figs_variables_distribution") or st.session_state.get("figs_variables_distribution_detailed"):
        used_artifacts.append("distribution_figures")
    for item in execution_log:
        action_name = str(item.get("action") or "").strip()
        if action_name and action_name not in used_artifacts:
            used_artifacts.append(action_name)
    return used_artifacts


def _render_chat_sequence(history: list[dict[str, Any]], latest_execution_log: list[dict[str, Any]] | None = None) -> None:
    for idx, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        intro = str(item.get("intro") or "").strip()
        answer = str(item.get("answer") or "").strip()
        entry_execution_log = item.get("execution_log") or []

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            body_parts = [part for part in [intro, answer] if part]
            if body_parts:
                st.markdown("\n\n".join(body_parts))
            if isinstance(entry_execution_log, list) and entry_execution_log:
                _render_selected_outputs_v2(entry_execution_log)
            elif idx == len(history) - 1 and isinstance(latest_execution_log, list) and latest_execution_log:
                _render_selected_outputs_v2(latest_execution_log)

def _maybe_run_brief_agent(df_ready: pd.DataFrame, plan: dict[str, Any]) -> None:
    if not isinstance(df_ready, pd.DataFrame) or df_ready.empty:
        return
    if not plan.get("actions"):
        return
    try:
        run_brief_agent(df_ready)
    except Exception as exc:
        st.session_state["qa_last_brief_agent_error"] = str(exc)


def _process_qa_question(question: str, df_ready: pd.DataFrame) -> bool:
    if not question.strip():
        st.warning("Veuillez poser une question.")
        return False
    if not isinstance(df_ready, pd.DataFrame) or df_ready.empty:
        st.warning("Aucun dataset prêt n'est disponible pour répondre à la question.")
        return False

    try:
        effective_question = _expand_followup_reply(question)
        st.session_state[NAV_CONTEXT_KEY] = "action"
        st.session_state["__QA_SUBMITTED__"] = True
        st.session_state["qa_last_question"] = question

        raw_plan = _plan_qa_actions(effective_question, df_ready)
        plan = _sanitize_plan(raw_plan, effective_question, df_ready)
        st.session_state["qa_last_plan"] = plan

        _maybe_run_brief_agent(df_ready, plan)
        execution_log = _execute_action_plan(plan, df_ready, question=effective_question)

        final_answer = _generate_final_answer(effective_question, execution_log, df_ready)
        st.session_state["qa_last_answer"] = final_answer

        intro = final_answer.get("intro") or "Voici la lecture la plus utile à partir des analyses disponibles."
        answer = final_answer.get("answer") or "Je n'ai pas pu produire de réponse exploitable à partir des éléments disponibles."
        analysis_suggestions = final_answer.get("analysis_suggestions") or final_answer.get("followup_questions") or []
        if not isinstance(analysis_suggestions, list):
            analysis_suggestions = []
        analysis_suggestions = [
            _make_analysis_suggestion(_suggestion_label(x), _suggestion_instruction(x))
            for x in analysis_suggestions
            if _suggestion_label(x) and _suggestion_instruction(x)
        ][:3]
        analysis_suggestion = analysis_suggestions[0] if analysis_suggestions else {}
        followup_question = _suggestion_instruction(analysis_suggestion) if analysis_suggestion else ""
        st.session_state["qa_last_analysis_suggestion"] = analysis_suggestion
        st.session_state["qa_last_followup_question"] = followup_question
        st.session_state[QA_LAST_FOLLOWUPS_KEY] = analysis_suggestions

        append_qa_history(
            {
                "question": question,
                "effective_question": effective_question,
                "intro": intro,
                "answer": answer,
                "analysis_suggestion": analysis_suggestion,
                "analysis_suggestions": analysis_suggestions,
                "followup_question": followup_question,
                "followup_questions": [_suggestion_instruction(x) for x in analysis_suggestions],
                "execution_log": execution_log,
                "actions": plan.get("actions", []),
                "used_artifacts": _extract_used_artifacts(execution_log),
            }
        )
        st.session_state["qa_main_input"] = ""
        st.session_state["qa_followup_input"] = ""
        return True
    except Exception as exc:
        st.error(f"Erreur lors du traitement de la question : {exc}")
        return False

def run() -> None:
    ensure_qa_memory()
    update_qa_conversation_summary()
    st.session_state.setdefault(NAV_CONTEXT_KEY, "view")
    st.session_state.setdefault("qa_last_plan", None)
    st.session_state.setdefault("qa_last_answer", None)
    st.session_state.setdefault("qa_last_execution_log", [])
    st.session_state.setdefault("qa_last_question", "")

    df_ready = st.session_state.get("df_ready")

    st.subheader("Q&A")
    history = st.session_state.get(QA_HISTORY_KEY, [])
    last_followups = st.session_state.get(QA_LAST_FOLLOWUPS_KEY) or []
    last_execution_log = st.session_state.get("qa_last_execution_log", [])

    if isinstance(history, list) and history:
        _render_chat_sequence(history[-6:], latest_execution_log=last_execution_log)
        suggested_analyses = last_followups if isinstance(last_followups, list) else []
        suggested_analyses = [
            _make_analysis_suggestion(_suggestion_label(x), _suggestion_instruction(x))
            for x in suggested_analyses
            if _suggestion_label(x) and _suggestion_instruction(x)
        ][:3]
        if suggested_analyses:
            st.markdown("**Analyses proposées**")
            suggestion_cols = st.columns(len(suggested_analyses))
            for idx, suggestion in enumerate(suggested_analyses):
                label = suggestion.get("label", "")
                instruction = suggestion.get("instruction", label)
                with suggestion_cols[idx]:
                    if st.button(label, key=f"qa_analysis_suggestion_{idx}", use_container_width=True):
                        if _process_qa_question(instruction, df_ready):
                            st.rerun()

    input_col, send_col = st.columns([12, 1])
    with input_col:
        submitted_question = st.text_input(
            "",
            key="qa_chat_input",
            placeholder="Posez une question sur le jeu de données",
            label_visibility="collapsed",
        )
    with send_col:
        submitted = st.button(">", use_container_width=True)
    if submitted and submitted_question:
        if _process_qa_question(str(submitted_question).strip(), df_ready):
            st.rerun()

    reset_qa_col, _spacer_col = st.columns([3, 9])
    with reset_qa_col:
        if st.button("Réinitialiser Q&A", use_container_width=True):
            _reset_qa_history()
            st.rerun()

    st.markdown("##### Actions suivantes")
    back_col, change_col, reset_col = st.columns(3)
    with back_col:
        if st.button("Retour au rapport", use_container_width=True):
            _goto_step("3")
    with change_col:
        if st.button("Changer les objectifs", use_container_width=True):
            st.session_state["pipeline_ready_to_run"] = False
            st.session_state["pipeline_executed"] = False
            st.session_state["pipeline_status"] = None
            st.session_state["pipeline_halt"] = None
            st.session_state["final_report_ready"] = False
            st.session_state["final_export_zip_bytes"] = None
            st.session_state["etape2_terminee"] = False
            st.session_state["etape40_terminee"] = False
            st.session_state["etape41_terminee"] = False
            _goto_step("2")
    with reset_col:
        if st.button("Réinitialiser", use_container_width=True):
            reset_app_state()

    st.session_state["etape41_terminee"] = True
