import plotly.graph_objects as go

from chart_common import add_phase_backgrounds, style_chart


def tissue_chart(dataframe, x_column):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe[x_column],
        y=dataframe["LeanMass"],
        name="Fat-free mass",
        mode="lines",
        line=dict(color="#22c55e", width=3),
        hovertemplate="%{y:.2f} kg<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dataframe[x_column],
        y=dataframe["FatMass"],
        name="Fat mass",
        mode="lines",
        line=dict(color="#ef4444", width=3),
        hovertemplate="%{y:.2f} kg<extra></extra>",
    ))
    add_phase_backgrounds(fig, dataframe, x_column)
    return style_chart(fig, "Body-composition compartments", x_column, "Mass (kg)")
