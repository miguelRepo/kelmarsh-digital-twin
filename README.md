# Kelmarsh Digital Twin

Digital twin of the **Kelmarsh wind farm** (Northamptonshire, UK) — six
Senvion MM92 turbines, 12.3 MW. Streamlit dashboard with:

- 🔧 **Turbine Health** — LSTM autoencoder anomaly detection on SCADA data
- 🌀 **Wake Simulation** — interactive FLORIS GCH wake field by wind speed/direction
- ⚙️ **Yaw Optimization** — wake steering gain sweep across wind directions

Data: 10-minute SCADA (2020), CC-BY-4.0 by Cubico
([Zenodo record 5841834](https://zenodo.org/records/5841834)).

## Setup

```bash
# Python 3.12 environment (torch 2.5.1 needs <= 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### Data

Download from Zenodo into `data/raw/` (both are git-ignored):

```bash
mkdir -p data/raw
curl -L -o data/raw/Kelmarsh_WT_static.csv \
  'https://zenodo.org/api/records/5841834/files/Kelmarsh_WT_static.csv/content'
curl -L -o /tmp/kelmarsh_2020.zip \
  'https://zenodo.org/api/records/5841834/files/Kelmarsh_SCADA_2020_3086.zip/content'
unzip /tmp/kelmarsh_2020.zip -d data/raw/ 'Turbine_Data_Kelmarsh_1*'
```

### Build artifacts

```bash
.venv/bin/python -m src.ingest   # clean SCADA -> data/processed/
.venv/bin/python -m src.train    # train LSTM  -> models/lstm_autoencoder_full.pt
```

## Run

```bash
.venv/bin/streamlit run app.py
```

## Project layout

```
app.py                  Streamlit entry point
src/
  ingest.py             raw SCADA -> cleaned csv, scaler, train/test windows
  train.py              LSTM autoencoder training (saves checkpoint + threshold)
  model.py              LSTMAutoencoder definition
  features.py           feature list, window size, sequence builder
  floris_model.py       FLORIS model build + wake field + yaw sweep helpers
  data_loader.py        cached Streamlit loaders
  pages/                overview, health, wake, yaw
notebooks/              01 EDA, 02 features, 03 LSTM, 04 FLORIS (exploration)
data/                   raw/ (ignored), processed/, turbine_library/
```
