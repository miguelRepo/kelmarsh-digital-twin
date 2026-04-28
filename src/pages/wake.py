"""Wake Simulation page — FLORIS wake field visualization."""

import streamlit as st

from src.data_loader import load_static


def render():
    st.title("🌀 Wake Simulation")
    st.markdown("Physics-based wake modelling using FLORIS GCH model.")

    static = load_static()
    st.dataframe(
        static[['Title', 'Rated power (kW)', 'Hub Height (m)',
                'Rotor Diameter (m)', 'Latitude', 'Longitude']]
    )
    st.info("Wake field visualization coming next.")
