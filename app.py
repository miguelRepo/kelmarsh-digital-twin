"""Kelmarsh Digital Twin — Streamlit application entry point."""

import streamlit as st

from src.pages import overview, health, wake, yaw


st.set_page_config(
    page_title="Kelmarsh Digital Twin",
    page_icon="🌬️",
    layout="wide",
)

st.sidebar.title("🌬️ Kelmarsh Digital Twin")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔧 Turbine Health", "🌀 Wake Simulation", "⚙️ Yaw Optimization"],
)

if page == "🏠 Overview":
    overview.render()
elif page == "🔧 Turbine Health":
    health.render()
elif page == "🌀 Wake Simulation":
    wake.render()
elif page == "⚙️ Yaw Optimization":
    yaw.render()
