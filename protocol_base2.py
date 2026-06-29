import streamlit as st


def render_protocol_base():
    months = st.number_input("Timeline (months)", 1, 120, 24)
    strategy = st.radio("Cycle strategy", ["Body-fat range", "Fixed duration"])
    mode = st.radio("First phase", ["Cut", "Bulk"], index=1, horizontal=True)
    return months, strategy, mode
