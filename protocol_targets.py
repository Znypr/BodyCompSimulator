import streamlit as st


def render_target_protocol(start_bf, goal_bf):
    a, b = st.columns(2)
    upper = a.number_input(
        "End bulk at body fat (%)",
        min_value=max(6.0, goal_bf + 1.0),
        max_value=35.0,
        value=float(min(25.0, max(15.0, start_bf + 3.0))),
        step=0.5,
    )
    lower = b.number_input(
        "End cut at body fat (%)",
        min_value=4.0,
        max_value=float(upper - 0.5),
        value=float(min(upper - 0.5, max(4.0, goal_bf))),
        step=0.5,
    )
    minimum = st.number_input("Minimum phase length (weeks)", 1, 16, 4)
    finish = st.toggle("Reserve time for a final cut", True)
    return upper, lower, minimum, finish
