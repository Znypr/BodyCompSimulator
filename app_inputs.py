import streamlit as st

import logic
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
        upper, lower, minimum, finish = 15.0, 10.0, 4, False
        if strategy == "Body-fat range":
            upper, lower, minimum, finish = render_target_protocol(body_fat, goal_bf)
        else:
            first_weeks, bulk_weeks, cut_weeks = render_duration_protocol()
        a, b = st.columns(2)
        surplus = a.number_input("Bulk intake above maintenance (kcal/day)", 0, 1500, 200, 25)
        deficit = b.number_input(
            "Target cut energy gap (kcal/day)",
            0,
            1500,
            500,
            25,
            help="This is maintained when energy-gap mode is on. With fixed calories, it is only the day-one gap.",
        )
        maintain_gap = st.toggle(
            "Maintain selected energy gap",
            True,
            help=(
                "On: calorie intake is adjusted as expenditure changes, keeping the selected surplus or deficit effective. "
                "Off: calorie intake is held fixed after each phase begins, so the energy gap shrinks or grows as TDEE changes."
            ),
        )
        fixed = not maintain_gap
        if maintain_gap:
            st.caption(
                f"The model adjusts intake over time to keep the cut gap near {deficit} kcal/day."
            )
        else:
            st.caption(
                f"The {deficit} kcal/day cut gap applies at phase start; it will usually shrink as body weight and expenditure fall."
            )

    cut_gap_multiplier = 1.0
    calibration_summary = None
    with st.expander("Personal cut calibration", expanded=False):
        use_calibration = st.toggle(
            "Calibrate from an observed cut",
            False,
            help="Uses your observed scale trend to estimate the effective average deficit. It does not prove exact calorie intake or expenditure.",
        )
        if use_calibration:
            a, b = st.columns(2)
            observed_start = a.number_input("Observed start weight (kg)", 30.0, 300.0, 83.0, 0.1)
            observed_end = b.number_input("Observed end weight (kg)", 30.0, 300.0, 74.0, 0.1)
            a, b = st.columns(2)
            observed_weeks = a.number_input("Observed duration (weeks)", 1.0, 104.0, 11.0, 0.5)
            observed_deficit = b.number_input("Planned deficit then (kcal/day)", 100, 2000, 500, 25)
            calibration_summary = logic.infer_cut_calibration(
                observed_start,
                observed_end,
                observed_weeks,
                observed_deficit,
            )
            cut_gap_multiplier = calibration_summary["cut_gap_multiplier"]
            st.caption(
                f"Estimated effective deficit: {calibration_summary['effective_deficit_kcal']:.0f} kcal/day "
                f"({cut_gap_multiplier:.2f}× the entered deficit), including an estimated "
                f"{calibration_summary['transient_loss_kg']:.1f} kg short-term scale component."
            )

    starting_transient_state = "Neutral / maintenance"
    custom_start_transient_kg = 0.0
    with st.expander("Water and glycogen state", expanded=True):
        include_scale_transients = st.toggle(
            "Model glycogen, water and gut-content shifts",
            True,
            help="Turn this off to model only stable tissue change. Scale weight will then exclude phase-related transient shifts.",
        )
        if include_scale_transients:
            starting_transient_state = st.selectbox(
                "Starting scale state",
                logic.STARTING_TRANSIENT_STATES,
                index=0,
                help=(
                    "Choose whether the entered start weight is neutral, unusually full, or already diet-depleted. "
                    "This prevents the model from applying the initial water/glycogen drop twice when starting mid-cut."
                ),
            )
            if starting_transient_state == "Custom":
                custom_start_transient_kg = st.number_input(
                    "Starting scale offset versus neutral (kg)",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1,
                    help="Negative means already lighter from depletion; positive means fuller than neutral.",
                )

            state_help = {
                "Neutral / maintenance": (
                    "Use when starting around normal maintenance intake and typical carbohydrate, sodium and food volume. "
                    "A new cut may then include an early transient scale drop."
                ),
                "Full / high-carb": (
                    "Use after a bulk, refeed or unusually high-carbohydrate period. The model can show a larger early cut drop."
                ),
                "Already depleted / mid-cut": (
                    "Use when the early water, glycogen and gut-content drop has already happened. "
                    "The current cut will not apply that loss again."
                ),
                "Custom": (
                    "Use a measured or manually estimated offset relative to your normal neutral scale state."
                ),
            }
            st.caption(state_help[starting_transient_state])
        else:
            st.caption("Only stable fat and fat-free tissue changes will affect scale weight.")

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
        "fixed_intake": fixed,
        "cut_gap_multiplier": cut_gap_multiplier,
        "calibration_summary": calibration_summary,
        "include_scale_transients": include_scale_transients,
        "starting_transient_state": starting_transient_state,
        "custom_start_transient_kg": custom_start_transient_kg,
    }
