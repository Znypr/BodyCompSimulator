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
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    df["Month"] = df["Day"] / (365.2425 / 12.0)
    start, final = df.iloc[0], df.iloc[-1]
    st.title("Body Composition Simulator")
    st.caption("Scenario modelling for body weight, fat mass, fat-free mass and energy expenditure.")

    columns = st.columns(5)
    columns[0].metric("Final weight", f"{final['Weight']:.1f} kg", f"{final['Weight'] - start['Weight']:+.1f} kg")
    columns[1].metric("Final body fat", f"{final['BodyFat']:.1f}%", f"{final['BodyFat'] - start['BodyFat']:+.1f} pp", delta_color="inverse")
    columns[2].metric("Fat mass", f"{final['FatMass']:.1f} kg", f"{final['FatMass'] - start['FatMass']:+.1f} kg", delta_color="inverse")
    columns[3].metric("Fat-free mass", f"{final['LeanMass']:.1f} kg", f"{final['LeanMass'] - start['LeanMass']:+.1f} kg")
    columns[4].metric("Duration", f"{config['timeline_months']} months")

    endpoints = logic.phase_endpoints(df)
    cut_points = endpoints[endpoints["Phase"] == "Cut"]
    if not cut_points.empty:
        last = cut_points.iloc[-1]
        st.info(f"Latest cut checkpoint: {last['Weight']:.1f} kg, {last['BodyFat']:.1f}% body fat, {last['LeanMass']:.1f} kg fat-free mass.")

    if final["LeanMass"] <= start["LeanMass"] and final["BodyFat"] >= start["BodyFat"]:
        st.error("This protocol does not improve projected body composition. Shorten the bulk, reduce the surplus, extend the cut or use body-fat range cycling.")
    elif final["BodyFat"] > start["BodyFat"] and final["Phase"] == "Bulk":
        st.warning("The timeline ends during a bulk. Compare the latest completed cut checkpoint rather than the arbitrary final day.")

    upper = config["bulk_stop_body_fat_pct"] if config["cycle_strategy"] == "Body-fat range" else None
    lower = config["cut_stop_body_fat_pct"] if config["cycle_strategy"] == "Body-fat range" else None
    left, right = st.columns(2)
    with left:
        st.plotly_chart(combined_chart(df, "Month", config["goal_weight"], config["goal_bf"], upper, lower), width="stretch", config={"displayModeBar": False})
        st.plotly_chart(energy_chart(df, "Month"), width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(tissue_chart(df, "Month"), width="stretch", config={"displayModeBar": False})
        st.dataframe(logic.summarize_phases(df), hide_index=True, width="stretch")

    with st.expander("How to read the result"):
        st.markdown("Body fat rises during a bulk and falls during a cut. Compare completed cut endpoints. A successful cycle should retain most fat-free mass during the cut and add some during the bulk. A long bulk followed by a short cut can still finish fatter; the model no longer forces every protocol to look successful.")
