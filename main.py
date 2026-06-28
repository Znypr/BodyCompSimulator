from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import logic

st.set_page_config(
    page_title="Body Composition Simulator",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE_PATH = Path(__file__).with_name("style.css")
if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


PHASE_COLORS = {
    "Bulk": "rgba(34, 197, 94, 0.07)",
    "Cut": "rgba(239, 68, 68, 0.07)",
}


def theme_template() -> str:
    return "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"


def add_phase_backgrounds(fig: go.Figure, dataframe, x_column: str) -> None:
    phase_groups = (dataframe["Phase"] != dataframe["Phase"].shift()).cumsum()
    for _, group in dataframe.groupby(phase_groups, sort=True):
        phase = group["Phase"].iloc[0]
        fig.add_vrect(
            x0=group[x_column].iloc[0],
            x1=group[x_column].iloc[-1],
            fillcolor=PHASE_COLORS.get(phase, "rgba(148, 163, 184, 0.06)"),
            line_width=0,
            layer="below",
        )


def style_chart(fig: go.Figure, title: str, x_title: str, y_title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=19)),
        template=theme_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=54, r=34, t=64, b=48),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    return fig


def combined_chart(dataframe, x_column: str, goal_weight: float, goal_bf: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataframe[x_column],
            y=dataframe["Weight"],
            name="Weight",
            mode="lines",
            line=dict(color="#f59e0b", width=3),
            hovertemplate="%{y:.2f} kg<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe[x_column],
            y=dataframe["BodyFat"],
            name="Body fat",
            mode="lines",
            yaxis="y2",
            line=dict(color="#3b82f6", width=3, dash="dot"),
            hovertemplate="%{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=goal_weight, line_dash="dash", line_color="#f59e0b", opacity=0.6)
    fig.add_shape(
        type="line",
        xref="paper",
        yref="y2",
        x0=0,
        x1=1,
        y0=goal_bf,
        y1=goal_bf,
        line=dict(color="#3b82f6", dash="dash", width=1.5),
        opacity=0.6,
    )
    add_phase_backgrounds(fig, dataframe, x_column)
    style_chart(fig, "Weight and body-fat projection", x_column, "Weight (kg)")
    fig.update_layout(
        yaxis=dict(title="Weight (kg)"),
        yaxis2=dict(title="Body fat (%)", overlaying="y", side="right", showgrid=False),
    )
    return fig


def tissue_chart(dataframe, x_column: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataframe[x_column],
            y=dataframe["LeanMass"],
            name="Fat-free mass",
            mode="lines",
            line=dict(color="#22c55e", width=3),
            hovertemplate="%{y:.2f} kg<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe[x_column],
            y=dataframe["FatMass"],
            name="Fat mass",
            mode="lines",
            line=dict(color="#ef4444", width=3),
            hovertemplate="%{y:.2f} kg<extra></extra>",
        )
    )
    add_phase_backgrounds(fig, dataframe, x_column)
    return style_chart(
        fig,
        "Body-composition compartments",
        x_column,
        "Mass (kg)",
    )


def energy_chart(dataframe, x_column: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataframe[x_column],
            y=dataframe["Intake"],
            name="Energy intake",
            mode="lines",
            line=dict(color="#8b5cf6", width=2.5),
            hovertemplate="%{y:.0f} kcal/day<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe[x_column],
            y=dataframe["TDEE"],
            name="Estimated expenditure",
            mode="lines",
            line=dict(color="#06b6d4", width=2.5),
            hovertemplate="%{y:.0f} kcal/day<extra></extra>",
        )
    )
    add_phase_backgrounds(fig, dataframe, x_column)
    return style_chart(fig, "Energy model", x_column, "Energy (kcal/day)")


def forbes_chart(dataframe) -> go.Figure:
    max_fat = max(35.0, float(dataframe["FatMass"].max()) * 1.25)
    fat_values = [0.5 + i * (max_fat - 0.5) / 160 for i in range(161)]
    fractions = [100.0 * logic.forbes_ffm_weight_fraction(value) for value in fat_values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fat_values,
            y=fractions,
            name="Forbes baseline",
            mode="lines",
            line=dict(color="#94a3b8", width=3),
            hovertemplate="Fat mass: %{x:.1f} kg<br>FFM share: %{y:.1f}%<extra></extra>",
        )
    )
    start = dataframe.iloc[0]
    end = dataframe.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[start["FatMass"], end["FatMass"]],
            y=[
                100.0 * logic.forbes_ffm_weight_fraction(start["FatMass"]),
                100.0 * logic.forbes_ffm_weight_fraction(end["FatMass"]),
            ],
            text=["Start", "End"],
            name="Scenario",
            mode="markers+text",
            textposition="top center",
            marker=dict(size=11, color=["#3b82f6", "#22c55e"]),
            hovertemplate="%{text}: %{x:.1f} kg fat mass<extra></extra>",
        )
    )
    return style_chart(
        fig,
        "Forbes baseline partition",
        "Fat mass (kg)",
        "Fat-free share of weight change (%)",
    )


with st.sidebar:
    st.header("Scenario inputs")
    st.caption("Metric units only")

    with st.expander("Starting point", expanded=True):
        profile_left, profile_right = st.columns(2)
        start_weight = profile_left.number_input(
            "Weight (kg)", min_value=30.0, max_value=300.0, value=79.0, step=0.1
        )
        start_bf = profile_right.number_input(
            "Body fat (%)", min_value=3.0, max_value=65.0, value=18.0, step=0.1
        )
        goal_left, goal_right = st.columns(2)
        goal_weight = goal_left.number_input(
            "Goal weight (kg)", min_value=30.0, max_value=300.0, value=75.0, step=0.5
        )
        goal_bf = goal_right.number_input(
            "Goal body fat (%)", min_value=3.0, max_value=65.0, value=11.0, step=0.5
        )
        activity_level = st.selectbox(
            "Daily activity",
            list(logic.ACTIVITY_FACTORS),
            index=list(logic.ACTIVITY_FACTORS).index("Moderate"),
        )
        estimated_maintenance = logic.estimate_maintenance_kcal(
            start_weight, start_bf, activity_level
        )
        use_measured_maintenance = st.toggle(
            "Use measured maintenance calories",
            value=False,
            help="A maintenance value calibrated from several weeks of intake and weight data is usually better than an equation.",
        )
        measured_maintenance = None
        if use_measured_maintenance:
            measured_maintenance = st.number_input(
                "Maintenance (kcal/day)",
                min_value=1000,
                max_value=6000,
                value=int(round(estimated_maintenance / 50) * 50),
                step=25,
            )
        else:
            st.caption(f"Estimated maintenance: {estimated_maintenance:,.0f} kcal/day")

    with st.expander("Training and nutrition", expanded=True):
        training_status = st.selectbox(
            "Resistance-training status",
            list(logic.MAX_FFM_GAIN_FRACTION_PER_MONTH),
            index=1,
        )
        training_quality = st.slider(
            "Training quality and consistency",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="A scenario modifier, not a genetic score.",
        )
        protein_g_per_kg = st.slider(
            "Protein (g/kg/day)",
            min_value=0.6,
            max_value=3.0,
            value=2.0,
            step=0.1,
        )

    with st.expander("Protocol", expanded=True):
        mode = st.radio("First phase", ["Cut", "Bulk"], horizontal=True)
        duration_left, duration_right = st.columns(2)
        first_phase_weeks = duration_left.number_input(
            "First phase (weeks)", min_value=1, max_value=104, value=12
        )
        timeline_months = duration_right.number_input(
            "Timeline (months)", min_value=1, max_value=60, value=24
        )

        cycle_left, cycle_right = st.columns(2)
        bulk_weeks = cycle_left.number_input(
            "Recurring bulk (weeks)", min_value=1, max_value=104, value=16
        )
        cut_weeks = cycle_right.number_input(
            "Recurring cut (weeks)", min_value=1, max_value=104, value=8
        )

        energy_left, energy_right = st.columns(2)
        surplus = energy_left.number_input(
            "Bulk surplus (kcal/day)", min_value=0, max_value=1500, value=200, step=25
        )
        deficit = energy_right.number_input(
            "Cut deficit (kcal/day)", min_value=0, max_value=1500, value=500, step=25
        )

        fixed_intake = st.toggle(
            "Keep phase calories fixed",
            value=True,
            help="When enabled, the effective surplus or deficit shrinks as expenditure adapts. Disable it to model active calorie adjustments that maintain the requested energy balance.",
        )

    with st.expander("Advanced planning"):
        cycle_scale = st.number_input(
            "Cycle-duration multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="1.0 keeps recurring phase durations unchanged.",
        )
        plot_unit = st.radio("Chart timeline", ["Months", "Weeks"], horizontal=True)


total_weeks = timeline_months * 365.2425 / 12.0 / 7.0

try:
    df = logic.calculate_projection(
        start_weight=start_weight,
        start_bf=start_bf,
        training_quality=training_quality,
        base_bulk_weeks=bulk_weeks,
        base_cut_weeks=cut_weeks,
        surplus=surplus,
        deficit=deficit,
        total_weeks=total_weeks,
        scale_coeff=cycle_scale,
        start_mode=mode,
        first_phase_weeks=first_phase_weeks,
        training_status=training_status,
        protein_g_per_kg=protein_g_per_kg,
        activity_level=activity_level,
        measured_maintenance_kcal=measured_maintenance,
        fixed_intake=fixed_intake,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()


df["Month"] = df["Day"] / (365.2425 / 12.0)
x_column = "Month" if plot_unit == "Months" else "Week"

st.title("Body Composition Simulator")
st.caption(
    "Evidence-informed scenario modelling for fat mass, fat-free mass, body weight and energy expenditure. "
    "This is a planning tool, not a diagnostic or clinical prediction."
)

start = df.iloc[0]
final = df.iloc[-1]
metric_columns = st.columns(5)
metric_columns[0].metric(
    "Final weight",
    f"{final['Weight']:.1f} kg",
    f"{final['Weight'] - start['Weight']:+.1f} kg",
)
metric_columns[1].metric(
    "Final body fat",
    f"{final['BodyFat']:.1f}%",
    f"{final['BodyFat'] - start['BodyFat']:+.1f} pp",
    delta_color="inverse",
)
metric_columns[2].metric(
    "Fat mass",
    f"{final['FatMass']:.1f} kg",
    f"{final['FatMass'] - start['FatMass']:+.1f} kg",
    delta_color="inverse",
)
metric_columns[3].metric(
    "Fat-free mass",
    f"{final['LeanMass']:.1f} kg",
    f"{final['LeanMass'] - start['LeanMass']:+.1f} kg",
)
metric_columns[4].metric("Duration", f"{timeline_months} months")

high_risk = df[df["Risk"] == "High"]
moderate_risk = df[df["Risk"] == "Moderate"]
if not high_risk.empty:
    first_risk = high_risk.iloc[0]
    st.warning(
        f"High projected lean-mass-loss risk begins around week {first_risk['Week']:.1f}. "
        "This is a continuous risk estimate based on deficit severity and available fat mass, not a hard biological cutoff."
    )
elif not moderate_risk.empty:
    first_risk = moderate_risk.iloc[0]
    st.info(
        f"Moderate lean-mass-loss risk appears around week {first_risk['Week']:.1f}. "
        "Consider a smaller deficit, higher protein intake or stronger resistance-training adherence."
    )

left, right = st.columns(2)
with left:
    st.plotly_chart(combined_chart(df, x_column, goal_weight, goal_bf), width="stretch", config={"displayModeBar": False})
    st.plotly_chart(energy_chart(df, x_column), width="stretch", config={"displayModeBar": False})
with right:
    st.plotly_chart(tissue_chart(df, x_column), width="stretch", config={"displayModeBar": False})
    st.plotly_chart(forbes_chart(df), width="stretch", config={"displayModeBar": False})

with st.expander("Phase summary", expanded=False):
    st.dataframe(logic.summarize_phases(df), hide_index=True, width="stretch")

with st.expander("Model assumptions and scientific basis", expanded=False):
    st.markdown(
        """
- **Fat-free mass is not synonymous with muscle.** It also includes water, glycogen, organs and bone.
- **Forbes' relationship is used as a baseline for the composition of weight change**, not as a muscle-gain efficiency law.
- **The Alpert fat-energy-transfer estimate is treated as a graded risk signal**, not a deterministic point where muscle loss suddenly begins.
- **Protein and resistance training modify lean-mass retention**, but individual response remains highly variable.
- **Bulk lean-mass gain uses transparent training-status priors** to prevent implausible projections. These are modelling assumptions, not biological ceilings.
- A measured maintenance intake derived from repeated weight and intake data is preferable to any population equation.

Primary references: Hall et al. (2011), Chow & Hall (2008), Hall (2008), Cunningham (1980), Alpert (2005), Longland et al. (2016), Morton et al. (2018).
        """
    )
