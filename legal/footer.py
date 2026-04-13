import streamlit as st
from .utils_legal import load_markdown


def render_footer():

    # état initial : rien sélectionné
    st.session_state.setdefault("legal_tab", None)

    st.markdown("""
    <style>
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.3);
    }

    /* correction : vrai ciblage des boutons Streamlit */
    div.stButton > button {
        width: 100%;
        font-size: 0.95rem !important;
    }

    /* textes juridiques plus petits */
    .legal-zone div[data-testid="stMarkdownContainer"] {
        font-size: 0.4rem;
        line-height: 1.4;
        color: #777;
    }

    .legal-zone div[data-testid="stMarkdownContainer"] ul {
        margin-top: 0.1rem;
        margin-bottom: 0.1rem;
        padding-left: 1.2rem;
    }

    .legal-zone div[data-testid="stMarkdownContainer"] li {
        margin-bottom: 0.15rem;
    }

    .legal-zone div[data-testid="stMarkdownContainer"] h3 {
        font-size: 0.95rem;
        margin-top: 0.8rem;
    }

    .legal-zone div[data-testid="stMarkdownContainer"] p {
        margin-top: 0 !important;
        margin-bottom: 0.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # boutons = "faux onglets"
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Mentions légales"):
            st.session_state.legal_tab = "mentions"

    with c2:
        if st.button("Confidentialité"):
            st.session_state.legal_tab = "confidentialite"

    with c3:
        if st.button("Cookies"):
            st.session_state.legal_tab = "cookies"

    with c4:
        if st.button("CGU"):
            st.session_state.legal_tab = "cgu"

    # affichage conditionnel
    if st.session_state.legal_tab is not None:

        if st.session_state.legal_tab == "mentions":
            st.markdown("###### Mentions légales")
            st.caption(load_markdown("legal/mentions_legales.md"))

        elif st.session_state.legal_tab == "confidentialite":
            st.markdown("###### Confidentialité / RGPD")
            st.caption(load_markdown("legal/confidentialite.md"))

        elif st.session_state.legal_tab == "cookies":
            st.markdown("###### Cookies")
            st.caption(load_markdown("legal/cookies.md"))

        elif st.session_state.legal_tab == "cgu":
            st.markdown("###### Conditions d’utilisation")
            st.caption(load_markdown("legal/cgu.md"))
