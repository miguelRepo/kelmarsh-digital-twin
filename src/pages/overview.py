"""Overview / landing page."""

import streamlit as st


def render():
    st.title("Kelmarsh Wind Farm — Digital Twin")
    st.markdown("""
    This dashboard provides a digital twin of the **Kelmarsh wind farm**
    (Northamptonshire, UK) — a 6-turbine Senvion MM92 site operated by Cubico.

    **Capabilities:**
    - 🔧 **Turbine Health** — LSTM anomaly detection on SCADA sensor data
    - 🌀 **Wake Simulation** — Physics-based wake modelling with FLORIS
    - ⚙️ **Yaw Optimization** — AEP gain estimation via wake steering

    **Data:** 10-minute SCADA data (2020), released under CC-BY-4.0 by Cubico.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Turbines", "6 × Senvion MM92")
    col2.metric("Rated capacity", "12.3 MW")
    col3.metric("SCADA records", "52,704")
