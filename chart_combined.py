import plotly.graph_objects as go

from chart_common import add_phase_backgrounds, style_chart


def combined_chart(dataframe, x_column, goal_weight, goal_bf, upper=None, lower=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe[x_column],
        y=dataframe["Weight"],
        name="Weight",
        mode="lines",
        line=dict(color="#f59e0b", width=3),
        hovertemplate="%{y:.2f} kg<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dataframe[x_column],
        y=dataframe["BodyFat"],
        name="Body fat",
        mode="lines",
        yaxis="y2",
        line=dict(color="#3b82f6", width=3, dash="dot"),
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=goal_weight, line_dash="dash", line_color="#f59e0b", opacity=0.45)
    for value, color in ((goal_bf, "#3b82f6"), (upper, "#22c55e"), (lower, "#ef4444")):
        if value is not None:
            fig.add_shape(
                type="line",
                xref="paper",
                yref="y2",
                x0=0,
                x1=1,
                y0=value,
                y1=value,
                line=dict(color=color, dash="dot", width=1.2),
                opacity=0.5,
            )
    add_phase_backgrounds(fig, dataframe, x_column)
    style_chart(fig, "Weight and body-fat projection", x_column, "Weight (kg)")
    fig.update_layout(
        yaxis=dict(title="Weight (kg)"),
        yaxis2=dict(title="Body fat (%)", overlaying="y", side="right", showgrid=False),
    )
    return fig
