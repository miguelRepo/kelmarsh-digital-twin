"""Portable FLORIS model loader for the Kelmarsh wind farm.

The turbine library path cannot be baked into the YAML config as an
absolute path (it would break on any other machine), and a relative
path only works when FLORIS runs from the repo root. This loader
resolves the path at runtime relative to the repository, so the model
works from the Streamlit app, notebooks, or scripts alike.
"""

from pathlib import Path

import yaml
from floris import FlorisModel

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "data" / "kelmarsh_floris_configured.yaml"
TURBINE_LIBRARY_PATH = REPO_ROOT / "data" / "turbine_library"


def load_floris_model(config_path: str | Path = CONFIG_PATH) -> FlorisModel:
    """Load the configured Kelmarsh FLORIS model with a resolved turbine library path."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["farm"]["turbine_library_path"] = str(TURBINE_LIBRARY_PATH)
    return FlorisModel(config)


def compute_wake_loss(fmodel: FlorisModel) -> tuple[float, float, float]:
    """Run the model with and without wakes for the currently set conditions.

    Returns (farm_power_kw, no_wake_power_kw, wake_loss_pct). Using
    ``run_no_wake`` keeps the baseline consistent with FLORIS's
    rotor-averaged wind speeds, instead of comparing against a raw
    power-curve lookup.
    """
    fmodel.run_no_wake()
    no_wake_kw = fmodel.get_farm_power().sum() / 1000
    fmodel.run()
    farm_kw = fmodel.get_farm_power().sum() / 1000
    wake_loss_pct = (1 - farm_kw / no_wake_kw) * 100
    return farm_kw, no_wake_kw, wake_loss_pct
