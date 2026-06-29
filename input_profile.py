import streamlit as st
import logic


def render_profile_inputs():
    with st.expander("Starting point", expanded=True):
        a, b = st.columns(2)
        weight = a.number_input("Weight (kg)", 30.0, 300.0, 74.0, 0.1)
        body_fat = b.number_input("Body fat (%)", 3.0, 65.0, 11.0, 0.1)
        a, b = st.columns(2)
        goal_weight = a.number_input("Goal weight (kg)", 30.0, 300.0, 80.0, 0.5)
        goal_bf = b.number_input("Goal body fat (%)", 3.0, 65.0, 8.0, 0.5)
        activity = st.selectbox("Daily activity", list(logic.ACTIVITY_FACTORS), index=2)
        estimate = logic.estimate_maintenance_kcal(weight, body_fat, activity)
        use_measured = st.toggle("Use measured maintenance calories", False)
        measured = st.number_input("Maintenance (kcal/day)", 1000, 6000, int(round(estimate / 50) * 50), 25) if use_measured else None
        if not use_measured:
            st.caption(f"Estimated maintenance: {estimate:,.0f} kcal/day")
    return weight, body_fat, goal_weight, goal_bf, activity, measured
