from pathlib import Path

import streamlit as st

from app_inputs import render_inputs
from app_results import render_results


def run_app():
    st.set_page_config(
        page_title="Body Composition Simulator",
        page_icon="◐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    style_path = Path(__file__).with_name("style.css")
    if style_path.exists():
        st.markdown(
            f"<style>{style_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    with st.sidebar:
        config = render_inputs()
    render_results(config)
