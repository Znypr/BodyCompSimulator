import streamlit as st

import logic
from chart_combined import combined_chart
from chart_energy import energy_chart
from chart_tissue import tissue_chart


def render_results(config):
    total_weeks = config["timeline_months"] * 365.2425 / 12.0 / 7.0
    try:
        df = logic.calculate_projection(
            start_weight=config["start_weight"],
            start_bf=config["start_bf"],
            training_quality=config["training_quality"],
            base_bulk_weeks=config["bulk_weeks"],
            base_cut_weeks=config["cut_weeks"],
            surplus=config["surplus"],
            deficit=config["deficit"],
            total_weeks=total_weeks,
            scale_coeff=1.0,
            start_mode=config["start_mode"],
            first_phase_weeks=config["first_phase_weeks"],
            training_status=config["training_status"],
            protein_g_per_kg=config["protein_g_per_kg"],
            activity_level=config["activity_level"],
            measured_maintenance_kcal=config["measured_maintenance_kcal"],
            fixed_intake=config["fixed_intake"],
            cycle_strategy=config["cycle_strategy"],
            bulk_stop_body_fat_pct=config["bulk_stop_body_fat_pct"],
            cut_stop_body_fat_pct=config["cut_stop_body_fat_pct"],
            minimum_phase_weeks=config["minimum_phase_weeks"],
            finish_lean=config["finish_lean"],
            cut_gap_multiplier=config["cut_gap_multiplier"],
            include_scale_transients=config["include_scale_transients"],
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    df["Month"] = df["Day"] / (365.2425 / 12.0)
    start, final = df.iloc[0], df.iloc[-1]
    st.title("Body Composition Simulator")
    st.caption(
        "Scale weight is separated into stable tissue and short-term glycogen, water and gut-content changes. "
        "Body-fat percentage refers to stable tissue weight."
    )

    columns = st.columns(6)
    columns[0].metric("Final scale weight", f"{final['Weight']:.1f} kg", f"{final['Weight'] - start['Weight']:+.1f} kg")
    columns[1].metric("Tissue body fat", f"{final['BodyFat']:.1f}%", f"{final['BodyFat'] - start['BodyFat']:+.1f} pp", delta_color="inverse")
    columns[2].metric("Fat mass", f"{final['FatMass']:.1f} kg", f"{final['FatMass'] - start['FatMass']:+.1f} kg", delta_color="inverse")
    columns[3].metric("Stable fat-free mass", f"{final['LeanMass']:.1f} kg", f"{final['LeanMass'] - start['LeanMass']:+.1f} kg")
    columns[4].metric("Scale transient", f"{final['ScaleTransient']:+.1f} kg")
    columns[5].metric("Duration", f"{config['timeline_months']} months")

    calibration = config.get("calibration_summary")
    if calibration:
        st.info(
            f"Your observed trend implies an effective average deficit near "
            f"{calibration['effective_deficit_kcal']:.0f} kcal/day, not necessarily the recorded "
            f"{config['deficit']} kcal/day. The model is applying a {config['cut_gap_multiplier']:.2f}× personal cut calibration."
        )

    endpoints = logic.phase_endpoints(df)
    cut_points = endpoints[endpoints["Phase"] == "Cut"]
    if not cut_points.empty:
        last = cut_points.iloc[-1]
        st.info(
            f"Latest cut endpoint: {last['Weight']:.1f} kg scale weight, "
            f"{last['BodyFat']:.1f}% tissue body fat, {last['FatMass']:.1f} kg fat and "
            f"{last['LeanMass']:.1f} kg stable fat-free mass."
        )

    if final["LeanMass"] <= start["LeanMass"] and final["BodyFat"] >= start["BodyFat"]:
        st.error("This protocol does not improve projected body composition. Adjust phase duration, surplus, deficit or training assumptions.")
    elif final["BodyFat"] > start["BodyFat"] and final["Phase"] == "Bulk":
        st.warning("The timeline ends during a bulk. Compare the latest completed cut endpoint rather than treating the final day as a lean-condition result.")

    upper = config["bulk_stop_body_fat_pct"] if config["cycle_strategy"] == "Body-fat range" else None
    lower = config["cut_stop_body_fat_pct"] if config["cycle_strategy"] == "Body-fat range" else None
    left, right = st.columns(2)
    with left:
        st.plotly_chart(combined_chart(df, "Month", config["goal_weight"], config["goal_bf"], upper, lower), width="stretch", config={"displayModeBar": False})
        st.plotly_chart(energy_chart(df, "Month"), width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(tissue_chart(df, "Month"), width="stretch", config={"displayModeBar": False})
        st.dataframe(logic.summarize_phases(df), hide_index=True, width="stretch")

    with st.expander("How to interpret the model"):
        st.markdown(
            """
- A 500 kcal/day **true average energy gap** corresponds to roughly 4–5 kg of predominantly fat tissue over 11 weeks, before metabolic adaptation.
- The first 1–3 kg of scale loss can come from glycogen, associated water and reduced gut content, especially when carbohydrate and total food intake fall.
- Losing 9 kg in 11 weeks therefore does not demonstrate that 500 kcal/day always causes 9 kg of tissue loss. It usually implies a larger effective deficit, a substantial transient component, or both.
- Stable fat-free mass is a broader compartment than skeletal muscle. The dashboard no longer counts short-term water loss as muscle loss.
            """
        )
