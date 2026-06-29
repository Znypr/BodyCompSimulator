import streamlit as st

PHASE_COLORS = {
    "Bulk": "rgba(34,197,94,0.08)",
    "Cut": "rgba(239,68,68,0.08)",
    "Maintain": "rgba(59,130,246,0.07)",
}


def add_phase_backgrounds(fig, dataframe, x_column):
    groups = (dataframe["Phase"] != dataframe["Phase"].shift()).cumsum()
    for _, group in dataframe.groupby(groups, sort=True):
        phase = group["Phase"].iloc[0]
        fig.add_vrect(
            x0=group[x_column].iloc[0],
            x1=group[x_column].iloc[-1],
            fillcolor=PHASE_COLORS.get(phase, "rgba(148,163,184,0.06)"),
            line_width=0,
            layer="below",
        )


def style_chart(fig, title, x_title, y_title):
    template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=19)),
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=54, r=34, t=64, b=48),
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    return fig
