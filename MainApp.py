# MainApp.py
import streamlit as st
from auth import require_invite_code
import os
from core.df_registry import (
    init_df_registry,
    sync_aliases_from_registry,
    sync_registry_from_aliases,
)


# ----------------------- Authentification -----------------------
require_invite_code()

# ----------------------- Config -----------------------
st.set_page_config(
    page_title="Application principale",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------- DF registry (phase 2) -----------------------
init_df_registry()
sync_registry_from_aliases()

# CSS: compresser la sidebar
st.markdown("""
<style>
section[data-testid="stSidebar"] { font-size: 0.92rem !important; }
section[data-testid="stSidebar"] .stSelectbox label { margin-bottom: 0.2rem !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0.25rem !important; }
section[data-testid="stSidebar"] .stButton > button {
  padding-top: 0.2rem !important; padding-bottom: 0.2rem !important;
  line-height: 1.1 !important; min-height: 0 !important; font-size: 0.90rem !important;
}
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div { margin-bottom: 0.25rem !important; }
</style>
""", unsafe_allow_html=True)


# import des modules
from apps import Preparation1, DiagnosticGlobal, RapportFinal, QA
from apps.PipelineRunner import _trace_module_calls

# ----------------------- Etapes -----------------------
ETAPES = [
    {"numero": "1", "cle_session": "etape1_terminee", "module": Preparation1, "label": "Upload"},
    {"numero": "2", "cle_session": "etape2_terminee", "module": DiagnosticGlobal, "label": "Diagnostic Global"},
    {"numero": "3", "cle_session": "etape40_terminee", "module": RapportFinal, "label": "Rapport Final"},
    {"numero": "4", "cle_session": "etape41_terminee", "module": QA, "label": "Q&A"},
]
META = {e["numero"]: e for e in ETAPES}
OPTIONS = tuple(e["numero"] for e in ETAPES)  # valeurs stables

# ----------------------- Clés réservées -----------------------
NAV_SELECTED_KEY = "__NAV_SELECTED__"   # numéro sélectionné (string)
URL_GUARD_KEY    = "__URL_GUARD__"      # garde anti ping-pong URL
STATE_TRIPWIRE   = "__NAV_TRIPWIRE__"   # (optionnel) détection reset sauvage

# ----------------------- Autochain séquentiel -----------------------
AUTO_JUMP_GUARD = "__AUTO_JUMP_GUARD__"   # évite double saut dans un même run
MODE_KEY        = "__NAV_MODE__"          # "automatique" | "manuel"

def _next_step_seq(num: str) -> str | None:
    """Retourne strictement l'étape suivante dans ETAPES (ou None si on est à la dernière)."""
    i = OPTIONS.index(num)
    return OPTIONS[i+1] if i + 1 < len(OPTIONS) else None

def _auto_next_if_completed(active_num: str, was_done_before_run: bool):
    """Enchaîne strictement vers l'étape suivante si mode = automatique et étape courante terminée."""
    # Ne jamais auto-enchaîner à partir de RapportFinal
    if OPTIONS.index(active_num) >= OPTIONS.index("3"):
        return
    if st.session_state.get(MODE_KEY, "automatique") != "automatique":
        return
    if st.session_state.get(AUTO_JUMP_GUARD):
        return
    key_done = META[active_num]["cle_session"]
    is_done_now = st.session_state.get(key_done, False)
    # Evite de re-sauter quand on revisite une étape déjà terminée.
    if (not was_done_before_run) and is_done_now:
        nxt = _next_step_seq(active_num)
        if nxt is not None:
            st.session_state[AUTO_JUMP_GUARD] = True
            _goto(nxt)
            st.stop()


# ----------------------- Init drapeaux -----------------------
for e in ETAPES:
    st.session_state.setdefault(e["cle_session"], False)

# ----------------------- URL helpers -----------------------
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

# ----------------------- Init sélection -----------------------
if NAV_SELECTED_KEY not in st.session_state:
    from_url = _get_url_step()
    st.session_state[NAV_SELECTED_KEY] = from_url if from_url in OPTIONS else OPTIONS[0]

st.session_state[NAV_SELECTED_KEY] = str(st.session_state[NAV_SELECTED_KEY])
if st.session_state[NAV_SELECTED_KEY] not in OPTIONS:
    st.session_state[NAV_SELECTED_KEY] = OPTIONS[0]

current_url = _get_url_step()
if current_url != st.session_state[NAV_SELECTED_KEY] and st.session_state.get(URL_GUARD_KEY) != st.session_state[NAV_SELECTED_KEY]:
    _set_url_step(st.session_state[NAV_SELECTED_KEY])
st.session_state.pop(URL_GUARD_KEY, None)

# ----------------------- Helpers UI -----------------------
def _fmt(num: str) -> str:
    e = META[num]
    dynamic_label = e["label"]
    if num == "3":
        if not st.session_state.get("pipeline_executed", False):
            dynamic_label = "Work in progress..."
        else:
            dynamic_label = "Rapport Final"
    chk = " [OK]" if st.session_state[e["cle_session"]] else ""
    label = f'{num} - {dynamic_label}{chk}'
    return label if len(label) <= 48 else label[:47] + "..."

def _goto(num: str):
    # Met à jour la sélection + URL, puis relance; le selectbox suit via 'index'
    st.session_state[NAV_SELECTED_KEY] = num
    _set_url_step(num)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ----------------------- UI NAV (selectbox sans key + Prev/Next) -----------------------
st.sidebar.image("logo_scanClues.png", width=300)

# 0) Récupère l'étape active depuis l'état/URL et sécurise
active_num = str(st.session_state.get(NAV_SELECTED_KEY, OPTIONS[0]))
if active_num not in OPTIONS:
    active_num = OPTIONS[0]
    st.session_state[NAV_SELECTED_KEY] = active_num

# 1) Calcul de l'index (nécessaire AVANT boutons & selectbox)
active_idx = OPTIONS.index(active_num)

# Mode unique (suppression du toggle auto/manuel)
st.session_state.setdefault(MODE_KEY, "automatique")

# Navigation par etape (suppression des fleches)
choice = st.sidebar.selectbox(
    "Etape",
    options=OPTIONS,
    format_func=_fmt,
    index=active_idx,
)

# Si l'utilisateur a choisi une autre etape, on y va et on coupe ce rendu
if choice != active_num:
    _goto(choice)
    st.stop()

# ----------------------- Exécuter le module choisi + chainage auto -----------------------
# Reprend l'étape active (au cas où)
active_num = st.session_state[NAV_SELECTED_KEY]

# reset du guard pour ce rendu
st.session_state.pop(AUTO_JUMP_GUARD, None)

# synchronise les anciennes cles df_* vers le registry avant l'etape
sync_registry_from_aliases()

# exécute la sous-app (elle met sa clé ..._terminee = True quand finie)
done_key = META[active_num]["cle_session"]
was_done_before_run = bool(st.session_state.get(done_key, False))
progress_slot = st.empty()

def _render_progress(module_label: str, fn_name: str = ""):
    progress_slot.info(f"Module en cours: {module_label}\nFonction en cours: {fn_name or '-'}")

def _run_with_trace(mod, label: str):
    prev_trace = st.session_state.get("pipeline_trace_functions", False)
    st.session_state["pipeline_trace_functions"] = True
    try:
        with _trace_module_calls(mod, lambda fn: _render_progress(label, fn)):
            mod.run()
    finally:
        st.session_state["pipeline_trace_functions"] = prev_trace
        progress_slot.empty()

module_label = META[active_num]["label"]
if active_num in ("1", "2"):
    _render_progress(module_label)
    _run_with_trace(META[active_num]["module"], module_label)
else:
    META[active_num]["module"].run()

# synchronise registry -> anciennes cles pour compatibilite inter-modules
sync_aliases_from_registry()

# si mode = automatique ET que l'étape vient d'être marquée terminée, on enchaîne
_auto_next_if_completed(active_num, was_done_before_run)
