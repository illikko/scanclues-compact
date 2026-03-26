import os, hmac
import streamlit as st

def require_invite_code(env_var: str = "INVITE_CODES"):
    raw = os.environ.get(env_var, "")
    codes = [c.strip() for c in raw.replace(",", "\n").splitlines() if c.strip()]

    # si pas de codes définis, on laisse passer (pratique en dev)
    if not codes:
        return

    if "authed" not in st.session_state:
        st.session_state.authed = False

    if not st.session_state.authed:
        code = st.text_input("Code d’accès", type="password")
        if st.button("Entrer"):
            ok = any(hmac.compare_digest(code.strip(), c) for c in codes)
            if ok:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Code invalide.")
        st.stop()