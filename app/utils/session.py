import streamlit as st


def init_session_defaults(defaults_dict: dict) -> None:
    """Call st.session_state.setdefault for each key/value in defaults_dict."""
    for key, value in defaults_dict.items():
        st.session_state.setdefault(key, value)
