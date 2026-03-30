import streamlit as st

from core.df_registry import init_df_registry
from apps._report import reset_report


def reset_app_state(*, trigger_rerun: bool = True, show_success: bool = True) -> None:
    """
    Remet l'application dans un état initial sans supprimer les infos d'auth.
    - Incrémente __UPLOAD_NONCE__ pour forcer les widgets dépendants du fichier uploadé.
    - Purge les artefacts de traitement et le registry de DataFrames.
    - Replace la navigation sur l'étape 1.
    """
    reset_report()
    next_upload_nonce = int(st.session_state.get("__UPLOAD_NONCE__", 0)) + 1

    keep_keys = {"__NAV_SELECTED__", "__URL_GUARD__", "__NAV_MODE__", "__NAV_TRIPWIRE__", "__AUTO_JUMP_GUARD__"}
    for k in list(st.session_state.keys()):
        lk = str(k).lower()
        if k in keep_keys:
            continue
        if "invite" in lk or "auth" in lk:
            continue
        del st.session_state[k]

    st.session_state["__UPLOAD_NONCE__"] = next_upload_nonce
    st.session_state["etape1_terminee"] = False
    st.session_state["etape2_terminee"] = False
    st.session_state["etape40_terminee"] = False
    st.session_state["etape41_terminee"] = False
    st.session_state["pipeline_ready_to_run"] = False
    st.session_state["pipeline_executed"] = False
    st.session_state["pipeline_status"] = None
    st.session_state["pipeline_halt"] = None
    st.session_state["final_report_ready"] = False
    st.session_state["final_export_zip_bytes"] = None

    init_df_registry()

    st.session_state["__NAV_SELECTED__"] = "1"
    try:
        st.query_params["step"] = "1"
    except Exception:
        st.experimental_set_query_params(step="1")

    if show_success:
        st.success("Application réinitialisée.")
    if trigger_rerun:
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
