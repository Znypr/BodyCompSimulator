"""Streamlit interface for the body-composition scenario model."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from bodycomp import SimulationConfig, simulate_sensitivity

st.set_page_config(page_title="Body Composition Scenario Lab", layout="wide")

STATUS_DEFAULTS = {
    "Beginner": 0.8,
    "Intermediate": 0.4,
    "Advanced": 0.2,
}
PHASE_COLORS = {
    "Bulk": "rgba(46, 204, 113, 0.08)",
    "Cut": "rgba(231, 76, 60, 0.08)",
    "Maintenance": "rgba(120, 120, 120, 0.07)",
}


def add_phase_backgrounds(fig: go.Figure, data, x_col: str = "Week") -> None:
    active = data[data["Phase"] != "Start"].copy()
    if active.empty:
        return

    active["Block"] = active["Phase"].ne(active["Phase"].shift()).cumsum()
    for _, block in active.groupby("Block", sort=False):
        phase = str(block["Phase"].iloc[0])
        fig.add_vrect(
            x0=float(block[x_col].iloc[0]),
            x1=float(block[x_col].iloc[-1]),
            fillcolor=PHASE_COLORS.get(phase, "rgba(120,120,120,0.05)"),
            line_width=0,
            layer="below",
        )


def style_figure(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title="Weeks",
        yaxis_title=y_title,
        hovermode="x unified",
        height=390,
        margin=dict(l=45, r=45, t=55, b=45),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    return fig


def add_band(
    fig: go.Figure,
    band,
    metric: str,
    name: str,
    yaxis: str | None = None,
) -> None:
    axis_args = {"yaxis": yaxis} if yaxis else {}
    fig.add_trace(
        go.Scatter(
            x=band["Week"],
            y=band[f"{metric}_max"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            **axis_args,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band["Week"],
            y=band[f"{metric}_min"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.16)",
            name=f"{name} sensitivity",
            hoverinfo="skip",
            **axis_args,
        )
    )


st.title("Body Composition Scenario Lab")
st.caption(
    "Evidence-informed sensitivity model — not a clinical prediction and not a muscle-mass measurement."
)

with st.sidebar:
    st.header("Scenario")
    c1, c2 = st.columns(2)
    start_weight = c1.number_input("Weight (kg)", 30.0, 300.0, 79.0, 0.1)
    start_bf = c2.number_input("Body fat (%)", 2.0, 60.0, 18.0, 0.5)

    total_weeks = st.slider("Timeline (weeks)", 4, 260, 104, 4)
    start_phase = st.radio("Start phase", ["Cut", "Bulk"], horizontal=True)

    c1, c2, c3 = st.columns(3)
    bulk_weeks = c1.number_input("Bulk", 1, 52, 12)
    cut_weeks = c2.number_input("Cut", 1, 52, 8)
    maintenance_weeks = c3.number_input("Maintain", 0, 12, 0)

    st.divider()
    st.subheader("Energy imbalance")
    c1, c2 = st.columns(2)
    surplus = c1.number_input("Bulk +kcal/day", 0, 1500, 200, 25)
    deficit = c2.number_input("Cut −kcal/day", 0, 1500, 500, 25)
    st.caption("Treat these as realized average imbalances, not guaranteed intake targets.")

    st.divider()
    st.subheader("Response assumptions")
    status = st.selectbox("Training status", list(STATUS_DEFAULTS), index=1)
    protein = st.slider("Protein (g/kg/day)", 0.8, 3.2, 1.8, 0.1)
    adherence = st.slider("Training adherence", 0.0, 1.0, 0.9, 0.05)

    with st.expander("Advanced assumptions"):
        gain_rate = st.number_input(
            "FFM gain ceiling (% BW/month)",
            0.0,
            3.0,
            STATUS_DEFAULTS[status],
            0.05,
            help="Editable scenario assumption; FFM is not identical to skeletal muscle.",
        )
        protection = st.slider(
            "FFM-loss protection vs Forbes baseline",
            0.0,
            0.95,
            0.60,
            0.05,
            help="Heuristic sensitivity parameter for resistance training, protein, recovery, and individual response.",
        )
        max_loss_rate = st.slider(
            "Cut planning ceiling (% BW/week)",
            0.25,
            1.50,
            0.75,
            0.05,
            help="Planning threshold, not a biological safety boundary.",
        )
        bf_error = st.slider(
            "BF estimate uncertainty (percentage points)", 0.0, 5.0, 2.0, 0.5
        )
        response_error = st.slider("Response sensitivity (±%)", 0, 50, 25, 5) / 100
        protection_error = st.slider(
            "Protection sensitivity (±points)", 0.0, 0.30, 0.15, 0.05
        )

config = SimulationConfig(
    start_weight_kg=start_weight,
    start_body_fat_pct=start_bf,
    total_weeks=total_weeks,
    start_phase=start_phase,
    bulk_weeks=int(bulk_weeks),
    cut_weeks=int(cut_weeks),
    maintenance_weeks=int(maintenance_weeks),
    bulk_surplus_kcal_day=float(surplus),
    cut_deficit_kcal_day=float(deficit),
    lean_gain_rate_pct_bw_month=float(gain_rate),
    protein_g_per_kg=float(protein),
    training_adherence=float(adherence),
    lean_loss_protection=float(protection),
    max_weekly_loss_pct=float(max_loss_rate),
)
base, band = simulate_sensitivity(
    config,
    body_fat_error_pp=bf_error,
    response_error_fraction=response_error,
    protection_error=protection_error,
)

start = base.iloc[0]
end = base.iloc[-1]
metrics = st.columns(4)
metrics[0].metric(
    "Final weight",
    f"{end['Weight']:.1f} kg",
    f"{end['Weight'] - start['Weight']:+.1f} kg",
)
metrics[1].metric(
    "Final body fat",
    f"{end['BodyFat']:.1f}%",
    f"{end['BodyFat'] - start['BodyFat']:+.1f} pp",
    delta_color="inverse",
)
metrics[2].metric(
    "Final FFM",
    f"{end['FatFreeMass']:.1f} kg",
    f"{end['FatFreeMass'] - start['FatFreeMass']:+.1f} kg",
)
metrics[3].metric(
    "Final fat mass",
    f"{end['FatMass']:.1f} kg",
    f"{end['FatMass'] - start['FatMass']:+.1f} kg",
    delta_color="inverse",
)

aggressive = base[base["AggressiveCut"]]
if not aggressive.empty:
    first = aggressive.iloc[0]
    st.warning(
        f"The modeled cut exceeds your {max_loss_rate:.2f}% BW/week planning ceiling "
        f"from week {first['Week']:.1f} ({first['WeeklyLossPct']:.2f}%/week). "
        "This is a planning flag, not a binary muscle-loss threshold."
    )

left, right = st.columns(2)

with left:
    combined = go.Figure()
    add_band(combined, band, "Weight", "Weight")
    combined.add_trace(
        go.Scatter(
            x=base["Week"], y=base["Weight"], name="Weight", line=dict(width=3)
        )
    )
    add_phase_backgrounds(combined, base)
    st.plotly_chart(style_figure(combined, "Body weight", "kg"), width="stretch")

    tissue = go.Figure()
    add_band(tissue, band, "FatFreeMass", "FFM")
    tissue.add_trace(
        go.Scatter(
            x=base["Week"],
            y=base["FatFreeMass"],
            name="Fat-free mass",
            line=dict(width=3),
        )
    )
    tissue.add_trace(
        go.Scatter(
            x=base["Week"],
            y=base["FatMass"],
            name="Fat mass",
            line=dict(width=3, dash="dot"),
        )
    )
    add_phase_backgrounds(tissue, base)
    st.plotly_chart(
        style_figure(tissue, "Two-compartment composition", "kg"), width="stretch"
    )

with right:
    bf_fig = go.Figure()
    add_band(bf_fig, band, "BodyFat", "Body fat")
    bf_fig.add_trace(
        go.Scatter(
            x=base["Week"],
            y=base["BodyFat"],
            name="Body fat",
            line=dict(width=3),
        )
    )
    add_phase_backgrounds(bf_fig, base)
    st.plotly_chart(style_figure(bf_fig, "Body-fat estimate", "%"), width="stretch")

    rate_fig = go.Figure()
    rate_fig.add_trace(
        go.Scatter(
            x=base["Week"],
            y=base["WeeklyLossPct"],
            name="Modeled weekly loss",
            line=dict(width=3),
        )
    )
    rate_fig.add_hline(
        y=max_loss_rate, line_dash="dash", annotation_text="Planning ceiling"
    )
    add_phase_backgrounds(rate_fig, base)
    st.plotly_chart(
        style_figure(rate_fig, "Cut-rate check", "% BW/week"), width="stretch"
    )

with st.expander("Model interpretation and limitations"):
    st.markdown(
        """
- **FFM is not skeletal muscle.** It includes water, glycogen, organs, connective tissue, and bone-related components.
- The Forbes/Hall relation is used only as a population-level starting point for partitioning weight loss. It is not used to predict anabolic efficiency in a surplus.
- The Alpert fat-energy-transfer estimate is not treated as a hard safe-deficit cutoff.
- The FFM gain ceiling and FFM-loss protection are editable scenario assumptions because no validated equation predicts individual hypertrophy or retention from body-fat percentage alone.
- Shaded areas are sensitivity envelopes created from your uncertainty settings, not statistical confidence intervals.
- Water, glycogen, sodium, menstrual-cycle effects, supplements, illness, and measurement drift are not modeled.
        """
    )
