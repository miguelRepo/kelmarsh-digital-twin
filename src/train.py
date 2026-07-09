"""Train the LSTM autoencoder on Kelmarsh Turbine 1 windows.

Reproduces notebook 03. Saves the full checkpoint (weights, losses,
anomaly threshold) that the Streamlit app loads:
    python -m src.train
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import LSTMAutoencoder


PROCESSED_DIR = 'data/processed'
CHECKPOINT = 'models/lstm_autoencoder_full.pt'

N_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
THRESHOLD_PERCENTILE = 95


def train(n_epochs=N_EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train = np.load(f'{PROCESSED_DIR}/X_train.npy')
    X_test = np.load(f'{PROCESSED_DIR}/X_test.npy')
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, X_train_tensor),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, X_test_tensor),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = LSTMAutoencoder(n_features=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_losses.append(total / len(train_loader))

        model.eval()
        total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                total += criterion(model(X_batch), y_batch).item()
        test_losses.append(total / len(test_loader))

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{n_epochs} — "
                  f"Train: {train_losses[-1]:.6f} | Test: {test_losses[-1]:.6f}")

    # Anomaly threshold: 95th percentile of test reconstruction error
    model.eval()
    errors = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            err = torch.mean((model(X_batch) - X_batch) ** 2, dim=(1, 2))
            errors.extend(err.cpu().numpy())
    threshold = float(np.percentile(np.array(errors), THRESHOLD_PERCENTILE))
    print(f"Anomaly threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}")

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'threshold': threshold,
    }, CHECKPOINT)
    print(f"Checkpoint saved to {CHECKPOINT} ✅")


if __name__ == '__main__':
    train()
