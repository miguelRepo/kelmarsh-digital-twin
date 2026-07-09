"""Data ingestion pipeline — raw SCADA CSV to cleaned/scaled artifacts.

Reproduces notebooks 01 (cleaning) and 02 (scaling + windowing):
    python -m src.ingest
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.features import FEATURE_COLS, WINDOW_SIZE, create_sequences


RAW_SCADA = 'data/raw/Turbine_Data_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv'
PROCESSED_DIR = 'data/processed'

RAW_COLS = {
    'Wind speed (m/s)': 'wind_speed',
    'Wind direction (°)': 'wind_dir',
    'Power (kW)': 'power',
    'Rotor speed (RPM)': 'rotor_rpm',
    'Nacelle position (°)': 'nacelle_pos',
    'Generator bearing front temperature (°C)': 'gen_bearing_front_temp',
    'Generator bearing rear temperature (°C)': 'gen_bearing_rear_temp',
    'Gear oil temperature (°C)': 'gear_oil_temp',
    'Blade angle (pitch position) A (°)': 'pitch_angle',
}


def clean_scada(raw_path=RAW_SCADA):
    """Load raw SCADA export and return the cleaned core dataframe."""
    df = pd.read_csv(raw_path, skiprows=9, index_col=0, parse_dates=True)
    df.index.name = 'timestamp'
    df.index = pd.to_datetime(df.index)

    df_core = df[list(RAW_COLS.keys())].rename(columns=RAW_COLS)

    df_clean = df_core.dropna()
    # Keep only rows where the turbine is producing (drop idling/shutdown)
    df_clean = df_clean[df_clean['power'] > 0]
    df_clean = df_clean[df_clean['rotor_rpm'] > 1]
    return df_clean


def build_artifacts(df_clean, out_dir=PROCESSED_DIR):
    """Fit scaler, build train/test windows, and save all artifacts."""
    os.makedirs(out_dir, exist_ok=True)

    df_clean.to_csv(f'{out_dir}/turbine1_clean.csv')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_clean[FEATURE_COLS])
    joblib.dump(scaler, f'{out_dir}/scaler.pkl')

    X = create_sequences(scaled, WINDOW_SIZE)
    split = int(len(X) * 0.8)  # split by time, not randomly
    np.save(f'{out_dir}/X_train.npy', X[:split])
    np.save(f'{out_dir}/X_test.npy', X[split:])
    return X[:split].shape, X[split:].shape


if __name__ == '__main__':
    df_clean = clean_scada()
    print(f"Cleaned rows: {len(df_clean)}")
    train_shape, test_shape = build_artifacts(df_clean)
    print(f"X_train: {train_shape}  X_test: {test_shape}")
    print("Artifacts saved to data/processed/ ✅")
