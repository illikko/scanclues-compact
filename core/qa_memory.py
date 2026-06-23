from __future__ import annotations

from typing import Any

import streamlit as st


QA_HISTORY_KEY = "qa_history"
QA_CONVERSATION_SUMMARY_KEY = "qa_conversation_summary"
QA_LAST_ANALYSIS_SUGGESTION_KEY = "qa_last_analysis_suggestion"
QA_LAST_ANALYSIS_SUGGESTIONS_KEY = "qa_last_analysis_suggestions"
# Compatibilité ascendante avec le reste du module Q&A.
QA_LAST_FOLLOWUP_KEY = QA_LAST_ANALYSIS_SUGGESTION_KEY
QA_LAST_FOLLOWUPS_KEY = QA_LAST_ANALYSIS_SUGGESTIONS_KEY
QA_LEGACY_LAST_FOLLOWUP_KEY = "qa_last_followup_question"
QA_LEGACY_LAST_FOLLOWUPS_KEY = "qa_last_followup_questions"
QA_COVERED_TOPICS_KEY = "qa_covered_topics"


def _normalize_topic(value: Any) -> str:
    text = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(value or ""))
    return " ".join(text.split())


def _add_topic(topics: list[str], value: Any) -> None:
    topic = _normalize_topic(value)
    if topic and len(topic) >= 3 and topic not in topics:
        topics.append(topic)


def ensure_qa_memory() -> None:
    st.session_state.setdefault(QA_HISTORY_KEY, [])
    st.session_state.setdefault(QA_CONVERSATION_SUMMARY_KEY, "")
    st.session_state.setdefault(QA_LAST_ANALYSIS_SUGGESTION_KEY, "")
    st.session_state.setdefault(QA_LAST_ANALYSIS_SUGGESTIONS_KEY, [])
    st.session_state.setdefault(QA_LEGACY_LAST_FOLLOWUP_KEY, "")
    st.session_state.setdefault(QA_LEGACY_LAST_FOLLOWUPS_KEY, [])
    st.session_state.setdefault(QA_COVERED_TOPICS_KEY, [])


def extract_qa_topics(turn: dict[str, Any]) -> list[str]:
    """Retourne des sujets canoniques déjà couverts par un tour Q&A.

    Ces sujets servent à éviter de reproposer la même analyse sans explication.
    Ils restent volontairement simples et déterministes : pas d'appel LLM.
    """
    topics: list[str] = []
    _add_topic(topics, turn.get("effective_question") or turn.get("question"))

    for artifact in turn.get("used_artifacts") or []:
        _add_topic(topics, f"artefact {artifact}")

    for action in turn.get("actions") or []:
        if not isinstance(action, dict):
            continue
        action_name = action.get("action")
        _add_topic(topics, f"action {action_name}")
        for pair in action.get("pairs") or []:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                _add_topic(topics, f"relation {pair[0]} {pair[1]}")
        for variable in action.get("variables") or []:
            _add_topic(topics, f"variable {variable}")
        if action.get("target_variable"):
            _add_topic(topics, f"cible {action.get('target_variable')}")
        if action.get("target_variable") and action.get("target_modality"):
            _add_topic(topics, f"modalite {action.get('target_variable')} {action.get('target_modality')}")

    for item in turn.get("execution_log") or []:
        if not isinstance(item, dict):
            continue
        if item.get("subset_column") and item.get("subset_value"):
            _add_topic(topics, f"segment {item.get('subset_column')} {item.get('subset_value')}")
        for pair in item.get("available_pairs") or item.get("pairs") or []:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                _add_topic(topics, f"relation {pair[0]} {pair[1]}")
        for variable in item.get("available_variables") or item.get("variables") or []:
            _add_topic(topics, f"variable {variable}")

    return topics[:30]


def get_covered_qa_topics(max_items: int = 80) -> list[str]:
    ensure_qa_memory()
    topics = st.session_state.get(QA_COVERED_TOPICS_KEY, [])
    if not isinstance(topics, list):
        return []
    normalized: list[str] = []
    for topic in topics[-max_items:]:
        _add_topic(normalized, topic)
    return normalized


def get_recent_qa_history(max_items: int = 6) -> list[dict[str, Any]]:
    history = st.session_state.get(QA_HISTORY_KEY, [])
    if not isinstance(history, list):
        return []
    recent = history[-max_items:]
    normalized: list[dict[str, Any]] = []
    for item in recent:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "question": str(item.get("question") or "").strip(),
                "intro": str(item.get("intro") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
                "analysis_suggestion": item.get("analysis_suggestion") or item.get("followup_question") or {},
                "analysis_suggestions": item.get("analysis_suggestions") or item.get("followup_questions") or [],
                "followup_question": str(item.get("followup_question") or "").strip(),
                "followup_questions": item.get("followup_questions") or [],
                "actions": item.get("actions") or [],
                "used_artifacts": item.get("used_artifacts") or [],
                "execution_log": item.get("execution_log") or [],
                "covered_topics": item.get("covered_topics") or [],
            }
        )
    return normalized


def update_qa_conversation_summary(max_items: int = 6) -> str:
    recent = get_recent_qa_history(max_items=max_items)
    lines: list[str] = []
    for idx, item in enumerate(recent, start=1):
        question = item.get("question", "")
        answer = item.get("answer", "")
        suggestion = item.get("analysis_suggestion") or item.get("followup_question", "")
        suggestions = item.get("analysis_suggestions") or item.get("followup_questions") or []
        followup = suggestion.get("label") if isinstance(suggestion, dict) else str(suggestion or "")
        followups = suggestions
        topics = item.get("covered_topics") or []
        if answer and len(answer) > 280:
            answer = answer[:280].rstrip() + "..."
        line = f"{idx}. Q: {question}"
        if answer:
            line += f" | R: {answer}"
        if topics:
            line += " | Déjà couvert: " + ", ".join([str(x) for x in topics[:5]])
        if not followup and isinstance(followups, list) and followups:
            followup = " / ".join([str(x.get("label") if isinstance(x, dict) else x).strip() for x in followups[:2] if str(x.get("label") if isinstance(x, dict) else x).strip()])
        if followup:
            line += f" | Analyse proposée: {followup}"
        lines.append(line)
    summary = "\n".join(lines)
    st.session_state[QA_CONVERSATION_SUMMARY_KEY] = summary
    return summary


def append_qa_history(turn: dict[str, Any]) -> list[dict[str, Any]]:
    ensure_qa_memory()
    history = st.session_state.get(QA_HISTORY_KEY, [])
    if not isinstance(history, list):
        history = []

    covered_topics = extract_qa_topics(turn)
    turn = dict(turn)
    turn["covered_topics"] = covered_topics
    history.append(turn)
    st.session_state[QA_HISTORY_KEY] = history

    all_topics = get_covered_qa_topics(max_items=120)
    for topic in covered_topics:
        _add_topic(all_topics, topic)
    st.session_state[QA_COVERED_TOPICS_KEY] = all_topics[-120:]

    update_qa_conversation_summary()
    suggestions = turn.get("analysis_suggestions") or turn.get("followup_questions") or []
    suggestion = turn.get("analysis_suggestion") or (suggestions[0] if isinstance(suggestions, list) and suggestions else turn.get("followup_question") or "")
    st.session_state[QA_LAST_ANALYSIS_SUGGESTION_KEY] = suggestion
    st.session_state[QA_LAST_ANALYSIS_SUGGESTIONS_KEY] = suggestions
    # Anciennes clés conservées pour éviter de casser d'autres écrans/imports.
    st.session_state[QA_LEGACY_LAST_FOLLOWUP_KEY] = suggestion.get("instruction", suggestion.get("label", "")) if isinstance(suggestion, dict) else str(suggestion or "").strip()
    st.session_state[QA_LEGACY_LAST_FOLLOWUPS_KEY] = suggestions
    return history
