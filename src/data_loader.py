"""Cached data and model loaders for the Streamlit app."""

import streamlit as st
import pandas as pd
import joblib
import torch

from src.model import LSTMAutoencoder
from src.features import FEATURE_COLS


@st.cache_resource
def load_model():
    """Load the trained LSTM autoencoder and anomaly threshold."""
    model = LSTMAutoencoder(n_features=len(FEATURE_COLS))
    checkpoint = torch.load('models/lstm_autoencoder_full.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['threshold']


@st.cache_data
def load_scada():
    """Load cleaned Turbine 1 SCADA data and the fitted scaler."""
    df = pd.read_csv(
        'data/processed/turbine1_clean.csv',
        index_col='timestamp',
        parse_dates=True,
    )
    scaler = joblib.load('data/processed/scaler.pkl')
    return df, scaler


@st.cache_data
def load_static():
    """Load static turbine layout data (coordinates, hub heights)."""
    return pd.read_csv('data/raw/Kelmarsh_WT_static.csv')
