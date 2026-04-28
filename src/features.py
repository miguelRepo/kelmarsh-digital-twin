"""Feature engineering helpers — sliding windows and feature selection."""

import numpy as np


FEATURE_COLS = [
    'wind_speed',
    'wind_dir',
    'power',
    'rotor_rpm',
    'nacelle_pos',
    'gen_bearing_front_temp',
    'gen_bearing_rear_temp',
    'gear_oil_temp',
    'pitch_angle',
]

WINDOW_SIZE = 30  # 30 timesteps × 10 min = 5 hours


def create_sequences(data, window_size=WINDOW_SIZE):
    """
    Convert a 2D array (timesteps × features) into overlapping windows.
    
    Returns array of shape (n_windows, window_size, n_features).
    """
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size)])
