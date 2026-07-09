"""FLORIS wake model construction and computation for the Kelmarsh farm.

Builds the model in code from the base GCH config + static turbine CSV,
resolving the turbine library path at runtime (the checked-in
kelmarsh_floris_configured.yaml carries a stale absolute path).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from floris import FlorisModel
from pyproj import Proj

BASE_CONFIG = 'data/kelmarsh_floris.yaml'
TURBINE_LIBRARY = 'data/turbine_library'
TURBINE_YAML = f'{TURBINE_LIBRARY}/senvion_mm92.yaml'
STATIC_CSV = 'data/raw/Kelmarsh_WT_static.csv'

HUB_HEIGHT = 78.5
ROTOR_DIAMETER = 92.0
DEFAULT_TI = 0.06


def farm_layout():
    """Project turbine lat/lon to local coordinates (UTM 30N, metres)."""
    static = pd.read_csv(STATIC_CSV)
    proj = Proj(proj='utm', zone=30, ellps='WGS84')
    easting, northing = proj(static['Longitude'].values, static['Latitude'].values)
    x = easting - easting.min()
    y = northing - northing.min()
    return x, y, static


def build_model():
    """Instantiate the FLORIS GCH model with the real Kelmarsh layout."""
    with open(BASE_CONFIG) as f:
        config = yaml.safe_load(f)

    x, y, _ = farm_layout()
    config['farm']['layout_x'] = x.tolist()
    config['farm']['layout_y'] = y.tolist()
    config['farm']['turbine_type'] = ['senvion_mm92'] * len(x)
    config['farm']['turbine_library_path'] = str(Path(TURBINE_LIBRARY).resolve())
    config['flow_field']['wind_speeds'] = [9.0]
    config['flow_field']['wind_directions'] = [225.0]
    config['flow_field']['turbulence_intensities'] = [DEFAULT_TI]
    return FlorisModel(config)


def run_case(fmodel, wind_speed, wind_direction, ti=DEFAULT_TI):
    """Run one flow case and return per-turbine powers in kW."""
    fm = fmodel.copy()
    fm.reset_operation()  # clear any yaw offsets from previous runs
    fm.set(
        wind_speeds=[float(wind_speed)],
        wind_directions=[float(wind_direction)],
        turbulence_intensities=[float(ti)],
    )
    fm.run()
    return fm.get_turbine_powers()[0] / 1000.0


def domain_bounds(x, y, margin=6 * ROTOR_DIAMETER):
    """Fixed plot window in global coordinates, independent of wind direction.

    FLORIS auto-computes cut-plane bounds in the wind-aligned frame, so the
    visible domain shifts and the wake gets clipped as the direction changes.
    A fixed global window with a uniform margin keeps every direction
    rendered consistently.
    """
    x, y = np.asarray(x), np.asarray(y)
    return (
        (float(x.min() - margin), float(x.max() + margin)),
        (float(y.min() - margin), float(y.max() + margin)),
    )


def wake_plane(fmodel, wind_speed, wind_direction, ti=DEFAULT_TI,
               resolution=(220, 180)):
    """Compute the hub-height velocity field on a direction-independent grid.

    FLORIS applies the cut-plane bounds in the wind-aligned (rotated) frame
    and returns a rotated lattice of points, so for non-axis-aligned
    directions the field cannot be treated as a regular grid — that is what
    made the wake render wrong when the direction changed. We request a
    rotation-proof square around the farm and resample onto a fixed global
    grid with scipy.

    Returns (x_grid, y_grid, u) where u has shape (len(y_grid), len(x_grid)).
    """
    from scipy.interpolate import griddata

    fm = fmodel.copy()
    fm.reset_operation()
    fm.set(
        wind_speeds=[float(wind_speed)],
        wind_directions=[float(wind_direction)],
        turbulence_intensities=[float(ti)],
    )
    (x0, x1), (y0, y1) = domain_bounds(fm.layout_x, fm.layout_y)

    # Square request that covers the target window under any rotation
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_diag = 1.1 * np.hypot(x1 - x0, y1 - y0) / 2
    plane = fm.calculate_horizontal_plane(
        height=HUB_HEIGHT,
        x_resolution=resolution[0],
        y_resolution=resolution[1],
        x_bounds=(cx - half_diag, cx + half_diag),
        y_bounds=(cy - half_diag, cy + half_diag),
    )

    x_grid = np.linspace(x0, x1, resolution[0])
    y_grid = np.linspace(y0, y1, resolution[1])
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([plane.df.x1.values, plane.df.x2.values])
    u = griddata(points, plane.df.u.values, (mesh_x, mesh_y), method='linear')
    if np.isnan(u).any():  # corners outside the solved lattice
        u_near = griddata(points, plane.df.u.values, (mesh_x, mesh_y),
                          method='nearest')
        u = np.where(np.isnan(u), u_near, u)
    return x_grid, y_grid, u


def expected_power_kw(wind_speed):
    """Waked-free power of a single MM92 from the empirical curve (kW)."""
    with open(TURBINE_YAML) as f:
        table = yaml.safe_load(f)['power_thrust_table']
    return float(np.interp(wind_speed, table['wind_speed'], table['power']))


def yaw_sweep(fmodel, wind_speed, wind_directions, ti=DEFAULT_TI):
    """Optimize yaw offsets per direction; returns a tidy results dataframe.

    Each direction runs on a fresh copy with operation reset, so optimized
    yaw angles never leak into the next baseline.
    """
    from floris.optimization.yaw_optimization.yaw_optimizer_sr import (
        YawOptimizationSR,
    )

    results = []
    for wd in wind_directions:
        fm = fmodel.copy()
        fm.reset_operation()
        fm.set(
            wind_speeds=[float(wind_speed)],
            wind_directions=[float(wd)],
            turbulence_intensities=[float(ti)],
        )
        fm.run()
        baseline_kw = fm.get_farm_power()[0] / 1000.0

        df_opt = YawOptimizationSR(fm).optimize()
        optimized_kw = df_opt['farm_power_opt'].values[0] / 1000.0
        yaw_angles = np.asarray(df_opt['yaw_angles_opt'].values[0]).ravel()

        results.append({
            'wind_dir': float(wd),
            'baseline_kw': baseline_kw,
            'optimized_kw': optimized_kw,
            'gain_pct': (optimized_kw / baseline_kw - 1) * 100.0,
            'yaw_angles': yaw_angles.round(1).tolist(),
        })
    return pd.DataFrame(results)
