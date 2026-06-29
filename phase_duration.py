import streamlit as st


def render_duration_protocol():
    a, b = st.columns(2)
    first = a.number_input("First phase (weeks)", 1, 104, 20)
    gain = b.number_input("Recurring bulk (weeks)", 1, 104, 20)
    loss = st.number_input("Recurring cut (weeks)", 1, 104, 12)
    return first, gain, loss
