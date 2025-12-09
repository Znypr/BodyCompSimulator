import streamlit as st
import plotly.graph_objects as go
import numpy as np
import logic

st.set_page_config(page_title="Body Comp Simulator", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #1e1e1e; }

        [data-testid="stSidebar"] { min-width: 350px; max-width: 350px; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }
        
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        
        div[data-testid="stVerticalBlock"] > div { gap: 0.3rem !important; }
        h1, h2, h3 { margin-top: 0rem !important; margin-bottom: 0.2rem !important; }
        hr { margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
        .stNumberInput, .stSlider, .stRadio { padding-bottom: 0rem !important; }
        
        /* CARD STYLING FOR CHARTS */
        /* Targets the Plotly container to give it the rounded card look */
        .js-plotly-plot .plotly, .js-plotly-plot .plot-container {
            border-radius: 16px !important; /* Rounded borders */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* Subtle shadow for depth */
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Configuration")

# 1. Stats & Goals
st.sidebar.subheader("Current & Goals")
c1, c2 = st.sidebar.columns(2)
start_weight = c1.number_input("Weight (kg)", value=80.0, step=0.1, format="%.1f")
start_bf = c2.number_input("Body Fat %", value=18.0, step=0.1, format="%.1f")

c3, c4 = st.sidebar.columns(2)
goal_weight = c3.number_input("Goal Wgt", value=85.0, step=0.5, format="%.1f")
goal_bf = c4.number_input("Goal BF%", value=14.0, step=0.5, format="%.1f")

# 2. Protocol
st.sidebar.markdown("---")
st.sidebar.subheader("Protocol")

r1_c1, r1_c2 = st.sidebar.columns(2)
mode = r1_c1.radio("Start", ["Bulk", "Cut"], index=1, horizontal=True)
first_label = "First Bulk Wks" if mode == "Bulk" else "First Cut Wks"
first_phase_weeks = r1_c2.number_input(first_label, 1, 52, 6)

r2_c1, r2_c2 = st.sidebar.columns(2)
bulk_weeks = r2_c1.number_input("Std. Bulk Wks", 1, 52, 14)
cut_weeks = r2_c2.number_input("Std. Cut Wks", 1, 52, 8)

r3_c1, r3_c2 = st.sidebar.columns(2)
surplus = r3_c1.number_input("Bulk Surplus", 0, 1500, 300)
deficit = r3_c2.number_input("Cut Deficit", 0, 1500, 300)

st.sidebar.markdown("---")
st.sidebar.subheader("Physiology")
efficiency = st.sidebar.slider("Base Training Quality", 0.1, 1.0, 1.0, help="1.0 = Genetic Limit.")
fatigue_pct = st.sidebar.slider("Bulk Staleness Decay", 0.0, 5.0, 1.5, step=0.1, format="%.1f%%")
scale_coeff = st.sidebar.number_input("Cycle Scale Multiplier", 0.8, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
r4_c1, r4_c2 = st.sidebar.columns([2, 1])
view_months = r4_c1.slider("Timeline (Months)", 1, 60, 24)
plot_unit = r4_c2.radio("Units", ["Weeks", "Months"], horizontal=True)

view_weeks_calc = view_months * 4.345 
fatigue_decimal = fatigue_pct / 100.0

# --- Logic ---
df = logic.calculate_projection(
    start_weight, start_bf, efficiency,
    bulk_weeks, cut_weeks, surplus, deficit,
    view_weeks_calc, scale_coeff, mode,
    first_phase_weeks, fatigue_decimal
)

df["Month"] = df["Week"] / 4.345

# --- REUSABLE WIDGETS ---

def apply_chart_style(fig, title, x_title, y_label):
    """Applies the visual theme: Dark card background, large text, padding"""
    fig.update_layout(
        # BIGGER TITLE (20px)
        title=dict(text=f"<b>{title}</b>", font=dict(size=20, color="#e0e0e0"), x=0.05),
        xaxis_title=x_title, 
        yaxis_title=y_label,
        template="plotly_dark",
        # Dark Card Background
        paper_bgcolor="#262626", 
        plot_bgcolor="#262626",
        # MORE PADDING (Margins)
        margin=dict(l=50, r=30, t=60, b=50), 
        height=320, 
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )
    return fig

def render_card(fig):
    """Reusable widget to render a chart card consistently"""
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- Chart Creators ---

def create_time_chart(dataframe, y_col, color_line, title, y_label, x_unit_mode, goal_val=None):
    fig = go.Figure()
    x_col = "Month" if x_unit_mode == "Months" else "Week"
    x_title = x_unit_mode

    fig.add_trace(go.Scatter(
        x=dataframe[x_col], y=dataframe[y_col], 
        mode='lines', name=title, line=dict(color=color_line, width=3)
    ))
    
    if goal_val:
        fig.add_hline(y=goal_val, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Goal")

    df['phase_change'] = df['Phase'] != df['Phase'].shift(1)
    change_indices = df.index[df['phase_change']].tolist()
    if 0 not in change_indices: change_indices.insert(0, 0)
    change_indices.append(len(df))
    
    shapes = []
    for i in range(len(change_indices) - 1):
        start, end = change_indices[i], change_indices[i+1]
        x0 = dataframe[x_col].iloc[start]
        x1 = dataframe[x_col].iloc[end-1] if end < len(df) else dataframe[x_col].iloc[-1]
        phase = df['Phase'].iloc[start]
        
        if phase == 'Bulk': color = "rgba(46, 204, 113, 0.06)"
        elif phase == 'Cut': color = "rgba(231, 76, 60, 0.06)"
        else: color = "rgba(139, 0, 0, 0.2)" 
            
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, fillcolor=color, line=dict(width=0), layer="below"))

    fig.update_layout(shapes=shapes)
    return apply_chart_style(fig, title, x_title, y_label)

def create_combined_chart(dataframe, x_unit_mode, g_weight=None, g_bf=None):
    fig = go.Figure()
    x_col = "Month" if x_unit_mode == "Months" else "Week"
    x_title = x_unit_mode

    fig.add_trace(go.Scatter(
        x=dataframe[x_col], y=dataframe["Weight"],
        name="Weight (kg)", mode='lines', line=dict(color="#FFA500", width=3)
    ))
    if g_weight:
        fig.add_hline(y=g_weight, line_dash="dash", line_color="#FFA500", opacity=0.6)

    fig.add_trace(go.Scatter(
        x=dataframe[x_col], y=dataframe["BodyFat"],
        name="Body Fat %", mode='lines', line=dict(color="#3399FF", width=3, dash='dot'), yaxis="y2"
    ))
    if g_bf:
        fig.add_shape(type="line", xref="paper", yref="y2", x0=0, x1=1, y0=g_bf, y1=g_bf, line=dict(color="#3399FF", dash="dash", width=1), opacity=0.6)

    # Backgrounds
    df['phase_change'] = df['Phase'] != df['Phase'].shift(1)
    change_indices = df.index[df['phase_change']].tolist()
    if 0 not in change_indices: change_indices.insert(0, 0)
    change_indices.append(len(df))
    shapes = []
    for i in range(len(change_indices) - 1):
        start, end = change_indices[i], change_indices[i+1]
        x0 = dataframe[x_col].iloc[start]
        x1 = dataframe[x_col].iloc[end-1] if end < len(df) else dataframe[x_col].iloc[-1]
        phase = df['Phase'].iloc[start]
        if phase == 'Bulk': color = "rgba(46, 204, 113, 0.06)"
        elif phase == 'Cut': color = "rgba(231, 76, 60, 0.06)"
        else: color = "rgba(139, 0, 0, 0.2)"
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, fillcolor=color, line=dict(width=0), layer="below"))
    
    for s in shapes: fig.add_shape(s)
    
    fig = apply_chart_style(fig, "Combined Projection", x_title, "Weight (kg)")
    fig.update_layout(
        yaxis=dict(title=dict(text="Weight (kg)", font=dict(color="#FFA500")), tickfont=dict(color="#FFA500")),
        yaxis2=dict(title=dict(text="Body Fat %", font=dict(color="#3399FF")), tickfont=dict(color="#3399FF"), overlaying="y", side="right")
    )
    return fig

# --- Dashboard Layout ---
st.title("Body Composition Dashboard")

# KPIs
k1, k2, k3, k4 = st.columns(4)
start_w = df["Weight"].iloc[0]
final_w = df["Weight"].iloc[-1]
start_bf = df["BodyFat"].iloc[0]
final_bf = df["BodyFat"].iloc[-1]
start_lm = df["LeanMass"].iloc[0]
final_lm = df["LeanMass"].iloc[-1]

k1.metric("Final Weight", f"{final_w:.1f} kg", f"{final_w - start_w:+.1f} kg")
k2.metric("Final BF%", f"{final_bf:.1f}%", f"{final_bf - start_bf:+.1f}%", delta_color="inverse")
k3.metric("Lean Mass", f"{final_lm:.1f} kg", f"{final_lm - start_lm:+.1f} kg")
k4.metric("Duration", f"{view_months} Months")

# Warning Box
unsafe_rows = df[df["Phase"] == "Unsafe"]
if not unsafe_rows.empty:
    first_crash = unsafe_rows.iloc[0]
    time_val = f"{first_crash['Month']:.1f} Months" if plot_unit == "Months" else f"{first_crash['Week']:.1f} Weeks"
    limit_cal = int(first_crash['SafeDeficitLimit'])
    st.error(f"⚠️ **MUSCLE LOSS WARNING:** At **{time_val}**, fat is too low for a {deficit} deficit. Max Safe Deficit: **{limit_cal} kcal**.")

# Charts
c_left, c_right = st.columns(2)

with c_left:
    render_card(create_time_chart(df, "Weight", "#FFA500", "Body Weight", "Weight (kg)", plot_unit, goal_weight))
    render_card(create_time_chart(df, "BodyFat", "#3399FF", "Body Fat %", "BF %", plot_unit, goal_bf))

with c_right:
    render_card(create_combined_chart(df, plot_unit, goal_weight, goal_bf))
    
    # Tissue Chart
    fig_tissue = go.Figure()
    x_col = "Month" if plot_unit == "Months" else "Week"
    fig_tissue.add_trace(go.Scatter(x=df[x_col], y=df["LeanMass"], mode='lines', name="Muscle", line=dict(color="#d640ff", width=2)))
    fig_tissue.add_trace(go.Scatter(x=df[x_col], y=df["FatMass"], mode='lines', name="Fat", line=dict(color="#ffffff", width=2, dash='dash')))
    
    # Tissue backgrounds
    df['phase_change'] = df['Phase'] != df['Phase'].shift(1)
    change_indices = df.index[df['phase_change']].tolist()
    if 0 not in change_indices: change_indices.insert(0, 0)
    change_indices.append(len(df))
    shapes = []
    for i in range(len(change_indices) - 1):
        start, end = change_indices[i], change_indices[i+1]
        x0 = df[x_col].iloc[start]
        x1 = df[x_col].iloc[end-1] if end < len(df) else df[x_col].iloc[-1]
        phase = df['Phase'].iloc[start]
        if phase == 'Bulk': color = "rgba(46, 204, 113, 0.06)"
        elif phase == 'Cut': color = "rgba(231, 76, 60, 0.06)"
        else: color = "rgba(139, 0, 0, 0.2)"
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, fillcolor=color, line=dict(width=0), layer="below"))
    
    fig_tissue.update_layout(shapes=shapes)
    render_card(apply_chart_style(fig_tissue, "Tissue Composition", plot_unit, "Mass (kg)"))