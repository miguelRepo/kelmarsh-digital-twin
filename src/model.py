"""LSTM Autoencoder for SCADA anomaly detection."""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.
    
    Trained on normal turbine behavior. At inference time, windows that the
    model struggles to reconstruct produce high reconstruction error and
    are flagged as anomalies.
    """

    def __init__(self, n_features, hidden_size=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        decoder_out, _ = self.decoder(decoder_input)
        return self.output_layer(decoder_out)
