"""Yaw Optimization page — wake steering gain estimation by wind direction."""

import plotly.graph_objects as go
import streamlit as st

from src import floris_model as fm
from src.data_loader import load_floris


@st.cache_data(show_spinner=False)
def run_sweep(ws, wd_step, ti):
    """Optimize yaw for every direction in the sweep (cached)."""
    fmodel = load_floris()
    directions = list(range(0, 360, wd_step))
    return fm.yaw_sweep(fmodel, ws, directions, ti)


def render():
    st.title("⚙️ Yaw Optimization")
    st.markdown("Wake steering: FLORIS Serial-Refine optimization of yaw "
                "offsets, swept across wind directions.")

    col_ws, col_step = st.columns(2)
    ws = col_ws.slider("Wind speed (m/s)", 4.0, 15.0, 9.0, 0.5)
    wd_step = col_step.select_slider(
        "Direction step (°)", options=[30, 15, 10], value=30,
        help="Smaller steps = more directions = longer compute on first run",
    )

    n_dirs = 360 // wd_step
    with st.spinner(f"Optimizing yaw for {n_dirs} directions "
                    "(first run takes a while, then it's cached)..."):
        df = run_sweep(ws, wd_step, fm.DEFAULT_TI)

    best = df.loc[df['gain_pct'].idxmax()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Max gain", f"{best['gain_pct']:.2f}%",
                help="Best farm power improvement across the sweep")
    col2.metric("Best direction", f"{best['wind_dir']:.0f}°")
    col3.metric("Mean gain", f"{df['gain_pct'].mean():.2f}%")

    st.subheader("Farm power gain by wind direction")
    fig = go.Figure(go.Bar(
        x=df['wind_dir'], y=df['gain_pct'],
        marker_color='steelblue',
        customdata=df[['baseline_kw', 'optimized_kw']].values,
        hovertemplate=('%{x:.0f}°: +%{y:.2f}%<br>'
                       'baseline: %{customdata[0]:.0f} kW<br>'
                       'optimized: %{customdata[1]:.0f} kW<extra></extra>'),
    ))
    fig.update_layout(
        height=400,
        xaxis_title='Wind direction (°)',
        yaxis_title='Gain (%)',
        xaxis=dict(tickmode='linear', dtick=max(wd_step, 30)),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig.add_hline(y=0, line_color='black', line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Optimal yaw offsets")
    sel_wd = st.selectbox(
        "Wind direction", df['wind_dir'].tolist(),
        index=int(df['gain_pct'].idxmax()),
        format_func=lambda d: f"{d:.0f}°",
    )
    row = df[df['wind_dir'] == sel_wd].iloc[0]
    yaw_angles = row['yaw_angles']
    fig_yaw = go.Figure(go.Bar(
        x=[f'T{i + 1}' for i in range(len(yaw_angles))],
        y=yaw_angles,
        text=[f'{a:.1f}°' for a in yaw_angles],
        textposition='outside',
        marker_color='steelblue',
        hovertemplate='%{x}: %{y:.1f}°<extra></extra>',
    ))
    fig_yaw.update_layout(
        height=320, yaxis_title='Yaw offset (°)',
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig_yaw.add_hline(y=0, line_color='black', line_width=1)
    st.plotly_chart(fig_yaw, use_container_width=True)
    st.caption(f"At {sel_wd:.0f}°: baseline {row['baseline_kw']:.0f} kW → "
               f"optimized {row['optimized_kw']:.0f} kW "
               f"(+{row['gain_pct']:.2f}%)")

    with st.expander("Full sweep results"):
        st.dataframe(
            df[['wind_dir', 'baseline_kw', 'optimized_kw', 'gain_pct']]
            .round(2),
            hide_index=True,
        )
