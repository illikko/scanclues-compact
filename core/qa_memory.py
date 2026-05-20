from __future__ import annotations

from typing import Any

import streamlit as st


QA_HISTORY_KEY = "qa_history"
QA_CONVERSATION_SUMMARY_KEY = "qa_conversation_summary"
QA_LAST_FOLLOWUP_KEY = "qa_last_followup_question"
QA_LAST_FOLLOWUPS_KEY = "qa_last_followup_questions"


def ensure_qa_memory() -> None:
    st.session_state.setdefault(QA_HISTORY_KEY, [])
    st.session_state.setdefault(QA_CONVERSATION_SUMMARY_KEY, "")
    st.session_state.setdefault(QA_LAST_FOLLOWUP_KEY, "")
    st.session_state.setdefault(QA_LAST_FOLLOWUPS_KEY, [])


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
                "followup_question": str(item.get("followup_question") or "").strip(),
                "followup_questions": item.get("followup_questions") or [],
                "actions": item.get("actions") or [],
                "used_artifacts": item.get("used_artifacts") or [],
                "execution_log": item.get("execution_log") or [],
            }
        )
    return normalized


def update_qa_conversation_summary(max_items: int = 6) -> str:
    recent = get_recent_qa_history(max_items=max_items)
    lines: list[str] = []
    for idx, item in enumerate(recent, start=1):
        question = item.get("question", "")
        answer = item.get("answer", "")
        followup = item.get("followup_question", "")
        followups = item.get("followup_questions") or []
        if answer and len(answer) > 280:
            answer = answer[:280].rstrip() + "..."
        line = f"{idx}. Q: {question}"
        if answer:
            line += f" | R: {answer}"
        if not followup and isinstance(followups, list) and followups:
            followup = " / ".join([str(x).strip() for x in followups[:2] if str(x).strip()])
        if followup:
            line += f" | Relance: {followup}"
        lines.append(line)
    summary = "\n".join(lines)
    st.session_state[QA_CONVERSATION_SUMMARY_KEY] = summary
    return summary


def append_qa_history(turn: dict[str, Any]) -> list[dict[str, Any]]:
    ensure_qa_memory()
    history = st.session_state.get(QA_HISTORY_KEY, [])
    if not isinstance(history, list):
        history = []
    history.append(turn)
    st.session_state[QA_HISTORY_KEY] = history
    update_qa_conversation_summary()
    st.session_state[QA_LAST_FOLLOWUP_KEY] = str(turn.get("followup_question") or "").strip()
    st.session_state[QA_LAST_FOLLOWUPS_KEY] = turn.get("followup_questions") or []
    return history
