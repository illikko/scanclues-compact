# MainApp.py
import streamlit as st

from core.df_registry import (
    init_df_registry,
    sync_aliases_from_registry,
    sync_registry_from_aliases,
)
from legal.footer import render_footer


st.set_page_config(
    page_title="Application principale",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_df_registry()
sync_registry_from_aliases()

st.markdown(
    """
<style>
section[data-testid="stSidebar"] { font-size: 0.92rem !important; }
section[data-testid="stSidebar"] .stSelectbox label { margin-bottom: 0.2rem !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0.25rem !important; }
section[data-testid="stSidebar"] .stButton > button {
  padding-top: 0.2rem !important; padding-bottom: 0.2rem !important;
  line-height: 1.1 !important; min-height: 0 !important; font-size: 0.90rem !important;
}
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div { margin-bottom: 0.25rem !important; }
div[role="radiogroup"][aria-label="Navigation"] label,
div[role="radiogroup"][aria-label="Navigation"] label * {
  font-size: 2rem !important;
  line-height: 1.2 !important;
}
div[role="radiogroup"][aria-label="Navigation"] label strong {
  font-size: 2rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

from apps import Download, DiagnosticGlobal, RapportFinal, QA
from apps.PipelineRunner import _trace_module_calls
from core.progress_state import get_progress, set_progress


ETAPES = [
    {"numero": "1", "cle_session": "etape1_terminee", "module": Download, "label": "Upload"},
    {"numero": "2", "cle_session": "etape2_terminee", "module": DiagnosticGlobal, "label": "Définition des objectifs"},
    {"numero": "3", "cle_session": "etape40_terminee", "module": RapportFinal, "label": "Rapport Final"},
    {"numero": "4", "cle_session": "etape41_terminee", "module": QA, "label": "Q&A"},
]
META = {e["numero"]: e for e in ETAPES}
OPTIONS = tuple(e["numero"] for e in ETAPES)

NAV_SELECTED_KEY = "__NAV_SELECTED__"
URL_GUARD_KEY = "__URL_GUARD__"
STATE_TRIPWIRE = "__NAV_TRIPWIRE__"
NAV_CONTEXT_KEY = "__NAV_CONTEXT__"
LAST_ACTIVE_STEP_KEY = "__LAST_ACTIVE_STEP__"
STEP_JUST_CHANGED_KEY = "__STEP_JUST_CHANGED__"

AUTO_JUMP_GUARD = "__AUTO_JUMP_GUARD__"
MODE_KEY = "__NAV_MODE__"


def _next_step_seq(num: str) -> str | None:
    i = OPTIONS.index(num)
    return OPTIONS[i + 1] if i + 1 < len(OPTIONS) else None


def _auto_next_if_completed(active_num: str, was_done_before_run: bool):
    if OPTIONS.index(active_num) >= OPTIONS.index("3"):
        return
    if st.session_state.get(MODE_KEY, "automatique") != "automatique":
        return
    if st.session_state.get(AUTO_JUMP_GUARD):
        return
    key_done = META[active_num]["cle_session"]
    is_done_now = st.session_state.get(key_done, False)
    if (not was_done_before_run) and is_done_now:
        nxt = _next_step_seq(active_num)
        if nxt is not None:
            if active_num == "1" and nxt == "2":
                set_progress(65, "Passage au diagnostic global", phase="pre_diagnostic")
            st.session_state[AUTO_JUMP_GUARD] = True
            _goto(nxt, nav_context="action")
            st.stop()


for e in ETAPES:
    st.session_state.setdefault(e["cle_session"], False)


def _get_url_step():
    try:
        v = st.query_params.get("step")
        if isinstance(v, list):
            v = v[0] if v else None
        return v
    except Exception:
        params = st.experimental_get_query_params()
        lst = params.get("step", [None])
        return lst[0] if lst else None


def _set_url_step(step: str):
    st.session_state[URL_GUARD_KEY] = step
    try:
        st.query_params["step"] = step
    except Exception:
        st.experimental_set_query_params(step=step)


if NAV_SELECTED_KEY not in st.session_state:
    from_url = _get_url_step()
    st.session_state[NAV_SELECTED_KEY] = from_url if from_url in OPTIONS else OPTIONS[0]

st.session_state.setdefault(NAV_CONTEXT_KEY, "view")
st.session_state.setdefault(LAST_ACTIVE_STEP_KEY, st.session_state[NAV_SELECTED_KEY])

st.session_state[NAV_SELECTED_KEY] = str(st.session_state[NAV_SELECTED_KEY])
if st.session_state[NAV_SELECTED_KEY] not in OPTIONS:
    st.session_state[NAV_SELECTED_KEY] = OPTIONS[0]

current_url = _get_url_step()
if current_url != st.session_state[NAV_SELECTED_KEY] and st.session_state.get(URL_GUARD_KEY) != st.session_state[NAV_SELECTED_KEY]:
    _set_url_step(st.session_state[NAV_SELECTED_KEY])
st.session_state.pop(URL_GUARD_KEY, None)


def _fmt(num: str) -> str:
    e = META[num]
    short = {
        "1": "Upload",
        "2": "Objectifs",
        "3": "Rapport",
        "4": "Q&A",
    }.get(num, e["label"])
    done = bool(st.session_state.get(e["cle_session"], False))
    suffix = " ■" if done else ""
    return f"**{num} - {short}{suffix}**\u00a0\u00a0\u00a0\u00a0"


def _goto(num: str, *, nav_context: str = "view"):
    st.session_state[NAV_SELECTED_KEY] = num
    st.session_state[NAV_CONTEXT_KEY] = nav_context
    st.session_state[STEP_JUST_CHANGED_KEY] = True
    _set_url_step(num)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


active_num = str(st.session_state.get(NAV_SELECTED_KEY, OPTIONS[0]))
if active_num not in OPTIONS:
    active_num = OPTIONS[0]
    st.session_state[NAV_SELECTED_KEY] = active_num

st.session_state[STEP_JUST_CHANGED_KEY] = bool(
    st.session_state.pop(STEP_JUST_CHANGED_KEY, False)
    or st.session_state.get(LAST_ACTIVE_STEP_KEY) != active_num
)
st.session_state[LAST_ACTIVE_STEP_KEY] = active_num

active_idx = OPTIONS.index(active_num)
st.session_state.setdefault(MODE_KEY, "automatique")

nav_choice = st.radio(
    "Navigation",
    options=OPTIONS,
    format_func=_fmt,
    index=active_idx,
    horizontal=True,
    label_visibility="collapsed",
)

if nav_choice != active_num:
    _goto(nav_choice, nav_context="view")
    st.stop()

active_num = st.session_state[NAV_SELECTED_KEY]
st.session_state.pop(AUTO_JUMP_GUARD, None)

sync_registry_from_aliases()

done_key = META[active_num]["cle_session"]
was_done_before_run = bool(st.session_state.get(done_key, False))
progress_slot = st.empty()


def _render_progress(module_label: str, fn_name: str = ""):
    phase, stored_value, stored_label = get_progress()
    if active_num == "1":
        value = max(stored_value, 10) if phase == "pre_diagnostic" else 10
        label = stored_label or module_label
        set_progress(value, label, phase="pre_diagnostic")
        progress_slot.progress(value, text=label)
    elif active_num == "2":
        value = max(stored_value, 70) if phase == "pre_diagnostic" else 70
        label = stored_label or module_label
        set_progress(value, label, phase="pre_diagnostic")
        progress_slot.progress(value, text=label)
    else:
        progress_slot.info(
            f"⚙️ Exécution pipeline en cours — Module : {module_label} — Fonction : {fn_name or '-'}"
        )


def _run_with_trace(mod, label: str):
    prev_trace = st.session_state.get("pipeline_trace_functions", False)
    st.session_state["pipeline_trace_functions"] = True
    try:
        with _trace_module_calls(mod, lambda fn: _render_progress(label, fn)):
            mod.run()
    finally:
        st.session_state["pipeline_trace_functions"] = prev_trace
        if active_num not in ("1", "2"):
            progress_slot.empty()


module_label = META[active_num]["label"]
if active_num in ("1", "2"):
    show_preprogress = True
    if active_num == "1":
        show_preprogress = Download.should_show_progress()
        set_progress(5, module_label, phase="pre_diagnostic")
    elif active_num == "2":
        show_preprogress = st.session_state.get(NAV_CONTEXT_KEY) != "view"
        set_progress(70, module_label, phase="pre_diagnostic")
    if show_preprogress:
        _render_progress(module_label)
        with st.spinner(f"Exécution des traitements en cours ({module_label})..."):
            _run_with_trace(META[active_num]["module"], module_label)
    else:
        progress_slot.empty()
        META[active_num]["module"].run()
    if active_num == "2" and st.session_state.get("etape2_terminee", False):
        set_progress(100, module_label, phase="pre_diagnostic")
    if active_num == "2":
        progress_slot.empty()
else:
    META[active_num]["module"].run()

sync_aliases_from_registry()
_auto_next_if_completed(active_num, was_done_before_run)
st.session_state[NAV_CONTEXT_KEY] = "view"
render_footer()
