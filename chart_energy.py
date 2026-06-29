import plotly.graph_objects as go

from chart_common import add_phase_backgrounds, style_chart


def energy_chart(dataframe, x_column):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe[x_column],
        y=dataframe["Intake"],
        name="Energy intake",
        mode="lines",
        line=dict(color="#8b5cf6", width=2.5),
        hovertemplate="%{y:.0f} kcal/day<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dataframe[x_column],
        y=dataframe["TDEE"],
        name="Estimated expenditure",
        mode="lines",
        line=dict(color="#06b6d4", width=2.5),
        hovertemplate="%{y:.0f} kcal/day<extra></extra>",
    ))
    add_phase_backgrounds(fig, dataframe, x_column)
    return style_chart(fig, "Energy model", x_column, "Energy (kcal/day)")
