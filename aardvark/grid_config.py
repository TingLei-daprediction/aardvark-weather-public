"""
Central grid configuration for Aardvark.

Holds the *free* grid parameters -- the era5_x/era5_y coordinate file names, the other
grid/domain-dependent data file names, and the inner ViT grid size (int_x, int_y) -- in one
place, loaded from a YAML file with built-in defaults.

The data/target grid (Grid A: nlon, nlat, extent) is NOT configured here; it is derived from
the era5_x/era5_y files named below. This module only decides *which files* to read and the
inner-grid size.

File names are explicit (no resolution token). `{era5_mode}`, `{freq}`, `{year}` placeholders
in the data-file templates are substituted at runtime from the actual run parameters.

Usage:
    from grid_config import load_grid_config, set_active_config, loader_grid_x_path
    cfg = load_grid_config(path)      # path=None -> defaults
    set_active_config(cfg)            # makes path helpers below use this config
    ... loader_grid_x_path(data_path) ...

`set_active_config` must be called early in each (DDP) process; if it is never called the
helpers fall back to DEFAULT_GRID_CONFIG, preserving the original hard-wired file names.
"""

import os

try:
    import yaml
except ImportError:  # pyyaml is in environment.yml; only needed to read a YAML file
    yaml = None


DEFAULT_GRID_CONFIG = {
    "grid_files": {
        "loader_x": "era5/era5_x_1.npy",
        "loader_y": "era5/era5_y_1.npy",
        "model_x": "grid_lon_lat/era5_x_1.npy",
        "model_y": "grid_lon_lat/era5_y_1.npy",
    },
    # Other grid/domain-dependent data files. {era5_mode}, {freq}, {year}, {month} are
    # substituted at runtime. Defaults reproduce the original names. The "era5" directory /
    # file-name token is purely conventional -- override these templates (e.g. in a regional
    # YAML) to rename it; the loader has no other hard-wired "era5" paths on the target side.
    "data_files": {
        "elev_vars": "era5/elev_vars_1.npy",
        "norm_mean": "norm_factors/mean_{era5_mode}_1.npy",
        "norm_std": "norm_factors/std_{era5_mode}_1.npy",
        "era5_memmap": "era5/era5_{era5_mode}_1_{freq}_{year}.memmap",
        # Per-month files used by the sub-daily (monthly) rtma_surface path.
        "era5_month": "era5/era5_{era5_mode}_1_{freq}_{year}-{month:02d}.memmap",
        "background_month": "era5/background_{era5_mode}_1_{year}-{month:02d}.memmap",
        "lat_weights": "lat_weights/weights_lat_1.npy",
    },
    "int_x": 256,
    "int_y": 128,
}

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_config.yaml")

_ACTIVE = None


def load_grid_config(path=None):
    """Return a grid-config dict, merging a YAML file (if given) over the defaults.

    path=None returns the defaults. A relative/absolute YAML path overrides any keys it sets;
    unspecified keys keep their default values.
    """
    cfg = {
        "grid_files": dict(DEFAULT_GRID_CONFIG["grid_files"]),
        "data_files": dict(DEFAULT_GRID_CONFIG["data_files"]),
        "int_x": DEFAULT_GRID_CONFIG["int_x"],
        "int_y": DEFAULT_GRID_CONFIG["int_y"],
    }
    if path:
        if yaml is None:
            raise ImportError("pyyaml is required to read a grid config YAML file")
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for key, value in user.items():
            if key in ("grid_files", "data_files") and isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def set_active_config(cfg):
    """Install `cfg` as the process-wide active grid config used by the path helpers."""
    global _ACTIVE
    _ACTIVE = cfg


def get_active_config():
    """Return the active config, falling back to defaults if none was installed."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_grid_config(None)
    return _ACTIVE


def _grid_path(root, key):
    # os.path.join handles roots with or without a trailing separator.
    name = get_active_config()["grid_files"][key]
    return os.path.join(root, name)


def loader_grid_x_path(data_path):
    return _grid_path(data_path, "loader_x")


def loader_grid_y_path(data_path):
    return _grid_path(data_path, "loader_y")


def model_grid_x_path(model_data_path):
    return _grid_path(model_data_path, "model_x")


def model_grid_y_path(model_data_path):
    return _grid_path(model_data_path, "model_y")


# --- Other grid/domain-dependent data files (templates in cfg["data_files"]) -------------

def _data_file(root, key, **kw):
    # os.path.join handles roots with or without a trailing separator.
    name = get_active_config()["data_files"][key].format(**kw)
    return os.path.join(root, name)


def elev_vars_path(data_path):
    return _data_file(data_path, "elev_vars")


def norm_mean_path(root, era5_mode):
    return _data_file(root, "norm_mean", era5_mode=era5_mode)


def norm_std_path(root, era5_mode):
    return _data_file(root, "norm_std", era5_mode=era5_mode)


def era5_memmap_path(data_path, era5_mode, freq, year):
    return _data_file(data_path, "era5_memmap", era5_mode=era5_mode, freq=freq, year=year)


def era5_month_path(data_path, era5_mode, freq, year, month):
    return _data_file(
        data_path, "era5_month", era5_mode=era5_mode, freq=freq, year=year, month=month
    )


def background_month_path(data_path, era5_mode, year, month):
    return _data_file(data_path, "background_month", era5_mode=era5_mode, year=year, month=month)


def lat_weights_path(aux_data_path):
    return _data_file(aux_data_path, "lat_weights")


def assert_grid_files_consistent(model_data_path, data_path):
    """Assert the model's and dataset's grid coordinate files describe the same grid.

    The model reads its grid from ``model_data_path`` and the loaders from ``data_path``
    (two copies of the same grid under different roots). This compares both **shapes and
    actual coordinate values** -- so same-shape/different-extent grids (e.g. global vs
    Oklahoma) fail, not just size mismatches. Raises ValueError on any mismatch.
    """
    import numpy as np

    m_lon = np.load(model_grid_x_path(model_data_path))
    m_lat = np.load(model_grid_y_path(model_data_path))
    d_lon = np.load(loader_grid_x_path(data_path))
    d_lat = np.load(loader_grid_y_path(data_path))
    if (
        m_lon.shape != d_lon.shape
        or m_lat.shape != d_lat.shape
        or not np.allclose(m_lon, d_lon)
        or not np.allclose(m_lat, d_lat)
    ):
        raise ValueError(
            f"[grid] model grid files (under {model_data_path}) and dataset grid files "
            f"(under {data_path}) describe different grids: lon shapes {m_lon.shape} vs "
            f"{d_lon.shape}, lat {m_lat.shape} vs {d_lat.shape}; both shapes and coordinate "
            "values must match."
        )
