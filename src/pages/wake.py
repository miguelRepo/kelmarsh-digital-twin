"""Wake Simulation page — interactive FLORIS wake field visualization."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import floris_model as fm
from src.data_loader import load_floris, load_static


@st.cache_data(show_spinner=False)
def compute_wake(ws, wd, ti):
    """Velocity field + per-turbine powers for one flow case (cached)."""
    fmodel = load_floris()
    x_grid, y_grid, u = fm.wake_plane(fmodel, ws, wd, ti)
    powers = fm.run_case(fmodel, ws, wd, ti)
    return x_grid, y_grid, u, powers


@st.cache_data
def get_layout():
    x, y, static = fm.farm_layout()
    return x, y, static['Title'].tolist()


def wake_field_figure(x_grid, y_grid, u, layout_x, layout_y, powers, wd):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x_grid, y=y_grid, z=u,
        colorscale='Blues', reversescale=True,  # dark = slow = wake
        colorbar=dict(title='m/s'),
        hovertemplate='x: %{x:.0f} m<br>y: %{y:.0f} m<br>'
                      'u: %{z:.2f} m/s<extra></extra>',
    ))
    # Markers sized ~to rotor scale with the label inside: the turbines stay
    # visually anchored while the wake swings around them with the wind
    fig.add_trace(go.Scatter(
        x=layout_x, y=layout_y,
        mode='markers+text',
        text=[f'T{i + 1}' for i in range(len(layout_x))],
        textposition='middle center',
        textfont=dict(color='black', size=10),
        marker=dict(size=20, color='white',
                    line=dict(color='black', width=1.5)),
        customdata=powers,
        hovertemplate='%{text}: %{customdata:.0f} kW<extra></extra>',
        showlegend=False,
    ))

    # Arrow showing where the wind blows toward (met convention: wd = from)
    wd_rad = np.deg2rad(wd)
    dx, dy = -np.sin(wd_rad), -np.cos(wd_rad)
    span = x_grid.max() - x_grid.min()
    ax_x = x_grid.min() + 0.10 * span
    ax_y = y_grid.max() - 0.10 * span
    arrow_len = 0.09 * span
    fig.add_annotation(
        x=ax_x + dx * arrow_len, y=ax_y + dy * arrow_len,
        ax=ax_x, ay=ax_y,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowwidth=2.5, arrowcolor='black',
        text='',
    )
    fig.add_annotation(
        x=ax_x, y=ax_y, xref='x', yref='y',
        text=f'wind {wd:.0f}°', showarrow=False,
        yshift=18, font=dict(size=12, color='black'),
    )

    # Fixed gridlines give the eye a static reference frame, making it
    # obvious that only the wake moves when the direction changes
    grid = dict(showgrid=True, gridcolor='rgba(128,128,128,0.35)',
                dtick=250, zeroline=False, layer='above traces')
    fig.update_yaxes(scaleanchor='x', scaleratio=1, title='Northing (m)',
                     **grid)
    fig.update_xaxes(title='Easting (m)', **grid)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render():
    st.title("🌀 Wake Simulation")
    st.markdown("Physics-based wake modelling using the FLORIS GCH model. "
                "Move the sliders to explore how wakes shift with the wind.")

    col_ws, col_wd, col_ti = st.columns(3)
    ws = col_ws.slider("Wind speed (m/s)", 4.0, 15.0, 9.0, 0.5)
    wd = col_wd.slider("Wind direction (°)", 0, 355, 225, 5)
    ti = col_ti.select_slider(
        "Turbulence intensity",
        options=[0.04, 0.06, 0.08, 0.10, 0.12], value=0.06,
    )

    with st.spinner("Running FLORIS..."):
        x_grid, y_grid, u, powers = compute_wake(ws, float(wd), ti)
    layout_x, layout_y, names = get_layout()

    farm_kw = powers.sum()
    no_wake_kw = fm.expected_power_kw(ws) * len(powers)
    wake_loss = (1 - farm_kw / no_wake_kw) * 100 if no_wake_kw > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Farm power", f"{farm_kw:,.0f} kW")
    col2.metric("No-wake power", f"{no_wake_kw:,.0f} kW")
    col3.metric("Wake loss", f"{wake_loss:.1f}%")

    st.subheader("Hub-height velocity field")
    st.plotly_chart(
        wake_field_figure(x_grid, y_grid, u, layout_x, layout_y, powers, wd),
        use_container_width=True,
    )

    st.subheader("Power per turbine")
    fig_bar = go.Figure(go.Bar(
        x=[f'T{i + 1}' for i in range(len(powers))],
        y=powers,
        text=[f'{p:.0f}' for p in powers],
        textposition='outside',
        marker_color='steelblue',
        hovertemplate='%{x}: %{y:.1f} kW<extra></extra>',
    ))
    fig_bar.update_layout(
        height=350, yaxis_title='Power (kW)',
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("Turbine specifications"):
        static = load_static()
        st.dataframe(
            static[['Title', 'Rated power (kW)', 'Hub Height (m)',
                    'Rotor Diameter (m)', 'Latitude', 'Longitude']]
        )
