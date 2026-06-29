import streamlit as st

from input_profile import render_profile_inputs
from input_training import render_training_inputs
from phase_duration import render_duration_protocol
from protocol_base2 import render_protocol_base
from protocol_targets import render_target_protocol


def render_inputs():
    st.header("Scenario inputs")
    st.caption("Metric units only")
    weight, body_fat, goal_weight, goal_bf, activity, measured = render_profile_inputs()
    status, quality, protein = render_training_inputs()

    with st.expander("Protocol", expanded=True):
        months, strategy, mode = render_protocol_base()
        first_weeks, bulk_weeks, cut_weeks = 12, 20, 12
        upper, lower, minimum, finish = 15.0, goal_bf, 4, False
        if strategy == "Body-fat range":
            upper, lower, minimum, finish = render_target_protocol(body_fat, goal_bf)
        else:
            first_weeks, bulk_weeks, cut_weeks = render_duration_protocol()
        a, b = st.columns(2)
        surplus = a.number_input("Bulk surplus (kcal/day)", 0, 1500, 200, 25)
        deficit = b.number_input("Cut deficit (kcal/day)", 0, 1500, 500, 25)
        hold = st.toggle("Keep calories fixed within each phase", True)

    return {
        "start_weight": weight,
        "start_bf": body_fat,
        "goal_weight": goal_weight,
        "goal_bf": goal_bf,
        "activity_level": activity,
        "measured_maintenance_kcal": measured,
        "training_status": status,
        "training_quality": quality,
        "protein_g_per_kg": protein,
        "timeline_months": months,
        "cycle_strategy": strategy,
        "start_mode": mode,
        "first_phase_weeks": first_weeks,
        "bulk_weeks": bulk_weeks,
        "cut_weeks": cut_weeks,
        "bulk_stop_body_fat_pct": upper,
        "cut_stop_body_fat_pct": lower,
        "minimum_phase_weeks": minimum,
        "finish_lean": finish,
        "surplus": surplus,
        "deficit": deficit,
        "fixed_intake": hold,
    }
