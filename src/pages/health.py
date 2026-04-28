"""Turbine Health page — LSTM anomaly detection visualization."""

import numpy as np
import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import load_scada, load_model
from src.features import FEATURE_COLS, WINDOW_SIZE, create_sequences


@st.cache_data
def compute_anomalies(_model, _scaler, df_in):
    """Compute reconstruction errors for every window in the dataset."""
    scaled = _scaler.transform(df_in[FEATURE_COLS])
    X = create_sequences(scaled, WINDOW_SIZE)
    X_tensor = torch.FloatTensor(X)

    errors = []
    with torch.no_grad():
        batch_size = 128
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]
            output = _model(batch)
            err = torch.mean((output - batch) ** 2, dim=(1, 2)).numpy()
            errors.extend(err)

    errors = np.array(errors)
    timestamps = df_in.index[WINDOW_SIZE:][:len(errors)]
    return timestamps, errors


def render():
    st.title("🔧 Turbine Health Monitor")
    st.markdown("LSTM autoencoder anomaly detection on Turbine 1 SCADA data (2020).")

    df, scaler = load_scada()
    model, threshold = load_model()

    with st.spinner("Computing anomaly scores..."):
        timestamps, errors = compute_anomalies(model, scaler, df)

    n_anomalies = (errors > threshold).sum()
    pct_anomalies = n_anomalies / len(errors) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records analyzed", f"{len(errors):,}")
    col2.metric("Anomalies detected", f"{n_anomalies:,}")
    col3.metric("Anomaly rate", f"{pct_anomalies:.1f}%")
    col4.metric("Threshold", f"{threshold:.4f}")

    st.markdown("---")

    # Anomaly timeline
    st.subheader("Anomaly Detection Timeline")
    is_anomaly = errors > threshold

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=errors, mode='lines',
        name='Reconstruction error',
        line=dict(color='steelblue', width=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps[is_anomaly], y=errors[is_anomaly],
        mode='markers', name='Anomaly',
        marker=dict(color='red', size=4),
    ))
    fig.add_hline(
        y=threshold, line_dash='dash', line_color='red',
        annotation_text=f"Threshold: {threshold:.4f}",
    )
    fig.update_layout(
        height=400, hovermode='x unified',
        xaxis_title="Date", yaxis_title="Reconstruction error",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Power curve
    st.subheader("Power Curve")
    df_plot = df[df['power'] > 0]
    fig2 = px.scatter(
        df_plot, x='wind_speed', y='power',
        opacity=0.1, color_discrete_sequence=['steelblue'],
    )
    fig2.add_hline(
        y=2050, line_dash='dash', line_color='red',
        annotation_text="Rated power",
    )
    fig2.update_layout(
        height=400,
        xaxis_title="Wind speed (m/s)",
        yaxis_title="Power (kW)",
    )
    st.plotly_chart(fig2, use_container_width=True)
