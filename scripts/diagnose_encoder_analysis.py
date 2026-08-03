#!/usr/bin/env python3
"""Diagnose one RTMA-OK Aardvark encoder analysis.

This standalone, read-only tool compares one saved encoder prediction with its
URMA truth, the same day's 00 UTC background, and the surface observations that
were available for the selected output variable.  It also separates grid-point
errors into regions near and far from valid observations.

No training dataset is constructed.  File paths and time indexing are obtained
from Aardvark's existing ``grid_config`` and ``loader_utils_new`` helpers.

Example
-------
python scripts/diagnose_encoder_analysis.py \
  --run_dir /path/to/OK-infer-20220115T0600 \
  --analysis_time 2022-01-15T06:00 \
  --data_root /path/to/dr-av-rtma_ok_data \
  --grid_config aardvark/grid_config_ok.yaml \
  --rank 0 --sample_index 0 --channel 0 --obs_radius_km 25

Output channel order for ``rtma_ok_sfc`` is:
  0=t2m/tas, 1=q2m/sh, 2=sp/psl, 3=u10/u, 4=v10/v.

The observations define the near/far masks.  RMSE and bias are calculated
against gridded URMA truth, not directly against station observations.
"""

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
AARDVARK_DIR = REPO_ROOT / "aardvark"
if str(AARDVARK_DIR) not in sys.path:
    sys.path.insert(0, str(AARDVARK_DIR))

from grid_config import (  # noqa: E402
    background_month_path,
    era5_month_path,
    load_grid_config,
    loader_grid_x_path,
    loader_grid_y_path,
    norm_mean_path,
    norm_std_path,
    set_active_config,
)
from loader_utils_new import days_in_month, month_frame, parse_time_freq  # noqa: E402


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    from metpy.plots import USCOUNTIES

    HAS_METPY_COUNTIES = True
except ImportError:
    HAS_METPY_COUNTIES = False


CHANNELS = (
    {
        "analysis": "t2m",
        "observation": "tas",
        "description": "2 m temperature",
        "units": "K",
        "physical_range": (150.0, 350.0),
        "obs_range": (150.0, 350.0),
    },
    {
        "analysis": "q2m",
        "observation": "sh",
        "description": "2 m specific humidity",
        "units": "kg/kg",
        "physical_range": (-0.001, 0.06),
        "obs_range": (-0.001, 0.06),
    },
    {
        "analysis": "sp",
        "observation": "psl",
        "description": "surface pressure",
        "units": "Pa",
        "physical_range": (70000.0, 110000.0),
        "obs_range": (70000.0, 115000.0),
    },
    {
        "analysis": "u10",
        "observation": "u",
        "description": "10 m zonal wind",
        "units": "m/s",
        "physical_range": (-150.0, 150.0),
        "obs_range": (-150.0, 150.0),
    },
    {
        "analysis": "v10",
        "observation": "v",
        "description": "10 m meridional wind",
        "units": "m/s",
        "physical_range": (-150.0, 150.0),
        "obs_range": (-150.0, 150.0),
    },
)

EARTH_RADIUS_KM = 6371.0088


def parse_analysis_time(value: str) -> datetime:
    """Parse an ISO-like UTC timestamp without silently changing its timezone."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid analysis time {value!r}; use YYYY-MM-DDTHH:MM"
        ) from exc
    if result.utcoffset() is not None and result.utcoffset().total_seconds() != 0:
        raise argparse.ArgumentTypeError("--analysis_time must be UTC")
    return result.replace(tzinfo=None)


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return re.sub(r"_+", "_", value).strip("_").lower()


def load_saved_pair(
    run_dir: Path,
    rank: str,
    pred_file: Optional[str],
    target_file: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Path, Path]:
    """Load physical-unit prediction and target arrays for one inference run."""
    if bool(pred_file) != bool(target_file):
        raise ValueError("--pred_file and --target_file must be supplied together")

    if pred_file:
        pred_path = Path(pred_file)
        target_path = Path(target_file)
        if not pred_path.is_absolute():
            pred_path = run_dir / pred_path
        if not target_path.is_absolute():
            target_path = run_dir / target_path
    else:
        pred_path = run_dir / f"unnorm_preds_{rank}.npy"
        target_path = run_dir / f"unnorm_targets_{rank}.npy"

    for label, path in (("prediction", pred_path), ("target", target_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    return np.load(pred_path), np.load(target_path), pred_path, target_path


def align_channels_last(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Accept saved channels-last arrays and a limited legacy channels-first layout."""
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "expected 4-D prediction/target arrays, got "
            f"pred={pred.shape}, target={target.shape}"
        )
    if pred.shape == target.shape and pred.shape[-1] == len(CHANNELS):
        return pred, target
    if pred.shape[1] == len(CHANNELS):
        pred = np.moveaxis(pred, 1, -1)
    if target.shape[1] == len(CHANNELS):
        target = np.moveaxis(target, 1, -1)
    if pred.shape != target.shape or pred.shape[-1] != len(CHANNELS):
        raise ValueError(
            "prediction/target shape mismatch after channel alignment: "
            f"pred={pred.shape}, target={target.shape}; expected five output channels"
        )
    return pred, target


def require_memmap_size(path: Path, shape: Tuple[int, ...], label: str) -> None:
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype("float32").itemsize
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{label} file {path} has {actual} bytes; expected {expected} for shape {shape}"
        )


def load_target_witness(
    path: Path,
    shape: Tuple[int, int, int, int],
    frame: int,
    channel: int,
) -> np.ndarray:
    require_memmap_size(path, shape, "monthly truth")
    data = np.memmap(path, dtype="float32", mode="r", shape=shape)
    return np.asarray(data[frame, channel, :, :]).T.copy()


def load_background(
    path: Path,
    shape: Tuple[int, int, int, int],
    day_index: int,
    channel: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, Optional[Path]]:
    """Read normalized 00 UTC background and return one physical-unit lat/lon field."""
    require_memmap_size(path, shape, "monthly background")
    data = np.memmap(path, dtype="float32", mode="r", shape=shape)
    normalized = np.asarray(data[day_index, channel, :, :])
    physical = (normalized * std[channel] + mean[channel]).T.copy()

    # normalize_background.py normally leaves this sibling in place.  When available it is
    # an independent value/orientation witness for the denormalization above.
    raw_path = path.with_name(path.name.replace("background_", "background_raw_", 1))
    if raw_path.is_file():
        require_memmap_size(raw_path, shape, "raw monthly background")
        raw = np.memmap(raw_path, dtype="float32", mode="r", shape=shape)
        raw_field = np.asarray(raw[day_index, channel, :, :]).T
        if not np.allclose(physical, raw_field, rtol=2.0e-6, atol=1.0e-5, equal_nan=True):
            delta = np.abs(physical - raw_field)
            raise ValueError(
                "denormalized background does not match its raw-background witness: "
                f"max_abs_diff={np.nanmax(delta):.6g}, path={raw_path}"
            )
        return physical, raw_path
    return physical, None


def load_observations(
    data_root: Path,
    variable: str,
    freq_tag: str,
    year: int,
    month: int,
    frame: int,
    frames_expected: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    obs_dir = data_root / "hadisd_processed"
    month_tag = f"{year}-{month:02d}"
    lon_path = obs_dir / f"{variable}_lon_train-{month_tag}.npy"
    lat_path = obs_dir / f"{variable}_lat_train-{month_tag}.npy"
    alt_path = obs_dir / f"{variable}_alt_train-{month_tag}.npy"
    value_path = obs_dir / f"{variable}_vals_{freq_tag}_{month_tag}.memmap"

    for label, path in (("longitude", lon_path), ("latitude", lat_path), ("altitude", alt_path)):
        if not path.is_file():
            raise FileNotFoundError(f"observation {label} file not found: {path}")

    lon = np.asarray(np.load(lon_path)).reshape(-1)
    lat = np.asarray(np.load(lat_path)).reshape(-1)
    alt = np.asarray(np.load(alt_path)).reshape(-1)
    if lon.size == 0 or lon.shape != lat.shape or lon.shape != alt.shape:
        raise ValueError(
            f"observation coordinate mismatch for {variable}: "
            f"lon={lon.shape}, lat={lat.shape}, alt={alt.shape}"
        )

    shape = (frames_expected, lon.size)
    require_memmap_size(value_path, shape, f"{variable} observations")
    values_file = np.memmap(value_path, dtype="float32", mode="r", shape=shape)
    values = np.asarray(values_file[frame, :]).copy()
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)
    if not np.any(valid):
        raise ValueError(
            f"no valid {variable} observations at frame {frame} in {value_path}"
        )
    return lon[valid], lat[valid], values[valid], value_path


def warn_if_implausible(name: str, values: np.ndarray, limits: Tuple[float, float]) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values")
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin < limits[0] or vmax > limits[1]:
        print(
            f"[WARN] {name} range [{vmin:.6g}, {vmax:.6g}] is outside broad expected "
            f"range [{limits[0]:.6g}, {limits[1]:.6g}]; check units and normalization",
            flush=True,
        )


def nearest_observation_distance_km(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Calculate exact spherical nearest-observation distance in bounded-memory chunks."""
    if chunk_size <= 0:
        raise ValueError("--distance_chunk_size must be positive")
    lon2d, lat2d = np.meshgrid(grid_lon, grid_lat, indexing="xy")
    point_lon = np.deg2rad(lon2d.reshape(-1).astype(np.float64))
    point_lat = np.deg2rad(lat2d.reshape(-1).astype(np.float64))
    station_lon = np.deg2rad(obs_lon.astype(np.float64))[None, :]
    station_lat = np.deg2rad(obs_lat.astype(np.float64))[None, :]
    cos_station_lat = np.cos(station_lat)
    result = np.empty(point_lon.size, dtype=np.float64)

    for start in range(0, point_lon.size, chunk_size):
        stop = min(start + chunk_size, point_lon.size)
        lon = point_lon[start:stop, None]
        lat = point_lat[start:stop, None]
        dlon = station_lon - lon
        dlat = station_lat - lat
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat) * cos_station_lat * np.sin(dlon / 2.0) ** 2
        )
        central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        result[start:stop] = EARTH_RADIUS_KM * np.min(central_angle, axis=1)

    return result.reshape(lat2d.shape)


def calculate_metrics(
    prediction: np.ndarray,
    background: np.ndarray,
    truth: np.ndarray,
    region_masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    output = {}
    grid_total = int(truth.size)
    for region, region_mask in region_masks.items():
        pred_mask = region_mask & np.isfinite(prediction) & np.isfinite(truth)
        bg_mask = region_mask & np.isfinite(background) & np.isfinite(truth)
        if not np.any(pred_mask) or not np.any(bg_mask):
            raise ValueError(f"region {region!r} has no finite points for metric calculation")
        pred_error = prediction[pred_mask] - truth[pred_mask]
        bg_error = background[bg_mask] - truth[bg_mask]
        pred_mse = float(np.mean(pred_error ** 2))
        bg_mse = float(np.mean(bg_error ** 2))
        pred_rmse = math.sqrt(pred_mse)
        bg_rmse = math.sqrt(bg_mse)
        output[region] = {
            "grid_points": int(np.count_nonzero(region_mask)),
            "grid_fraction": float(np.count_nonzero(region_mask) / grid_total),
            "prediction_valid_points": int(np.count_nonzero(pred_mask)),
            "background_valid_points": int(np.count_nonzero(bg_mask)),
            "prediction_rmse": pred_rmse,
            "background_rmse": bg_rmse,
            "prediction_bias": float(np.mean(pred_error)),
            "background_bias": float(np.mean(bg_error)),
            "rmse_improvement": bg_rmse - pred_rmse,
            "mse_skill_score": float(1.0 - pred_mse / bg_mse) if bg_mse > 0.0 else np.nan,
        }
    return output


def add_map_features(ax) -> None:
    if not HAS_CARTOPY:
        return
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
    ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor="black")
    if HAS_METPY_COUNTIES:
        ax.add_feature(USCOUNTIES.with_scale("20m"), linewidth=0.25, edgecolor="gray", alpha=0.6)


def finite_limits(*fields: np.ndarray) -> Tuple[float, float]:
    finite = [field[np.isfinite(field)] for field in fields if np.any(np.isfinite(field))]
    if not finite:
        return -1.0, 1.0
    merged = np.concatenate(finite)
    low = float(np.min(merged))
    high = float(np.max(merged))
    if low == high:
        delta = max(abs(low) * 0.01, 1.0e-6)
        return low - delta, high + delta
    return low, high


def symmetric_limit(*fields: np.ndarray) -> float:
    finite = [np.abs(field[np.isfinite(field)]) for field in fields if np.any(np.isfinite(field))]
    if not finite:
        return 1.0
    result = float(np.max(np.concatenate(finite)))
    return result if result > 0.0 else 1.0


def plot_diagnostics(
    out_path: Path,
    analysis_time: datetime,
    info: Dict[str, object],
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    background: np.ndarray,
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_values: np.ndarray,
    distance: np.ndarray,
    radius_km: float,
    metrics: Dict[str, Dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_lon = np.where(grid_lon > 180.0, grid_lon - 360.0, grid_lon)
    plot_obs_lon = np.where(obs_lon > 180.0, obs_lon - 360.0, obs_lon)
    extent = (
        float(np.min(plot_lon)),
        float(np.max(plot_lon)),
        float(np.min(grid_lat)),
        float(np.max(grid_lat)),
    )
    prediction_error = prediction - truth
    background_error = background - truth
    increment = prediction - background

    if HAS_CARTOPY:
        projection = ccrs.PlateCarree()
        subplot_kw = {"projection": projection}
    else:
        projection = None
        subplot_kw = {}
        print("[WARN] Cartopy is unavailable; drawing array axes without map boundaries")

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 12), constrained_layout=True, subplot_kw=subplot_kw
    )
    image_kwargs = {"origin": "lower", "extent": extent, "aspect": "auto"}
    field_min, field_max = finite_limits(truth, prediction, background)
    error_limit = symmetric_limit(prediction_error, background_error)
    increment_limit = symmetric_limit(increment)

    field_panels = (
        (axes[0, 0], truth, "URMA truth"),
        (axes[0, 1], prediction, "Aardvark analysis"),
        (axes[0, 2], background, "00 UTC background persistence"),
    )
    for ax, field, title in field_panels:
        image = ax.imshow(field, cmap="viridis", vmin=field_min, vmax=field_max, **image_kwargs)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=str(info["units"]))

    for ax, field, title in (
        (axes[1, 0], prediction_error, "Analysis - truth"),
        (axes[1, 1], background_error, "Background - truth"),
    ):
        image = ax.imshow(field, cmap="RdBu_r", vmin=-error_limit, vmax=error_limit, **image_kwargs)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=str(info["units"]))

    image = axes[1, 2].imshow(
        increment, cmap="RdBu_r", vmin=-increment_limit, vmax=increment_limit, **image_kwargs
    )
    axes[1, 2].set_title("Analysis - background")
    fig.colorbar(image, ax=axes[1, 2], fraction=0.046, pad=0.04, label=str(info["units"]))

    obs_min, obs_max = finite_limits(obs_values)
    scatter_kwargs = {
        "c": obs_values,
        "cmap": "viridis",
        "vmin": obs_min,
        "vmax": obs_max,
        "s": 18,
        "edgecolors": "black",
        "linewidths": 0.25,
    }
    if projection is not None:
        scatter_kwargs["transform"] = projection
    scatter = axes[2, 0].scatter(plot_obs_lon, obs_lat, **scatter_kwargs)
    obs_title = f"Valid {info['observation']} observations (n={obs_values.size})"
    if info["analysis"] == "sp":
        obs_title += "\nPSL values shown; locations only define SP mask"
    axes[2, 0].set_title(obs_title)
    axes[2, 0].set_xlim(extent[0], extent[1])
    axes[2, 0].set_ylim(extent[2], extent[3])
    fig.colorbar(scatter, ax=axes[2, 0], fraction=0.046, pad=0.04, label=str(info["units"]))

    image = axes[2, 1].imshow(distance, cmap="magma", **image_kwargs)
    axes[2, 1].set_title(f"Distance to nearest valid observation\nnear <= {radius_km:g} km")
    contour_kwargs = {}
    if projection is not None:
        contour_kwargs["transform"] = projection
    axes[2, 1].contour(
        plot_lon,
        grid_lat,
        distance,
        levels=[radius_km],
        colors="cyan",
        linewidths=1.0,
        **contour_kwargs,
    )
    fig.colorbar(image, ax=axes[2, 1], fraction=0.046, pad=0.04, label="km")

    axes[2, 2].set_axis_off()
    metric_lines = [
        "Region       Model RMSE   BG RMSE   Model bias   BG bias   Skill",
    ]
    for region in ("full", "near", "far"):
        row = metrics[region]
        metric_lines.append(
            f"{region:<8} {row['prediction_rmse']:>11.4g} {row['background_rmse']:>9.4g} "
            f"{row['prediction_bias']:>11.4g} {row['background_bias']:>9.4g} "
            f"{row['mse_skill_score']:>7.3f}"
        )
    metric_lines.extend(
        [
            "",
            f"Valid observations: {obs_values.size}",
            f"Near-grid fraction: {metrics['near']['grid_fraction']:.3f}",
            f"Far-grid fraction:  {metrics['far']['grid_fraction']:.3f}",
            "Skill = 1 - MSE(model)/MSE(background)",
        ]
    )
    axes[2, 2].text(
        0.0,
        1.0,
        "\n".join(metric_lines),
        transform=axes[2, 2].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    for row in axes:
        for ax in row:
            if ax is axes[2, 2]:
                continue
            if HAS_CARTOPY:
                add_map_features(ax)
                ax.set_extent(extent, crs=projection)
            else:
                ax.set_xlabel("longitude")
                ax.set_ylabel("latitude")

    fig.suptitle(
        f"RTMA-OK encoder diagnostic: {info['analysis']} ({info['description']})\n"
        f"{analysis_time.isoformat(timespec='minutes')} UTC; observation radius={radius_km:g} km",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_metrics_csv(
    path: Path,
    analysis_time: datetime,
    channel: int,
    info: Dict[str, object],
    radius_km: float,
    valid_observations: int,
    metrics: Dict[str, Dict[str, float]],
    pred_path: Path,
    target_path: Path,
    background_path_value: Path,
    observation_path: Path,
) -> None:
    fieldnames = [
        "analysis_time_utc",
        "channel",
        "analysis_variable",
        "observation_variable",
        "units",
        "obs_radius_km",
        "valid_observations",
        "region",
        "grid_points",
        "grid_fraction",
        "prediction_valid_points",
        "background_valid_points",
        "prediction_rmse",
        "background_rmse",
        "prediction_bias",
        "background_bias",
        "rmse_improvement",
        "mse_skill_score",
        "prediction_file",
        "target_file",
        "background_file",
        "observation_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for region in ("full", "near", "far"):
            row = dict(metrics[region])
            row.update(
                {
                    "analysis_time_utc": analysis_time.isoformat(timespec="minutes") + "Z",
                    "channel": channel,
                    "analysis_variable": info["analysis"],
                    "observation_variable": info["observation"],
                    "units": info["units"],
                    "obs_radius_km": radius_km,
                    "valid_observations": valid_observations,
                    "region": region,
                    "prediction_file": str(pred_path),
                    "target_file": str(target_path),
                    "background_file": str(background_path_value),
                    "observation_file": str(observation_path),
                }
            )
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="Directory with unnorm inference arrays")
    parser.add_argument("--analysis_time", required=True, type=parse_analysis_time)
    parser.add_argument("--data_root", required=True, help="RTMA-OK data root")
    parser.add_argument(
        "--grid_config",
        default=str(AARDVARK_DIR / "grid_config_ok.yaml"),
        help="Aardvark grid/path YAML used by the inference run",
    )
    parser.add_argument("--era5_mode", default="rtma_ok_sfc")
    parser.add_argument("--time_freq", default="1H")
    parser.add_argument("--rank", default="0")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--channel", type=int, required=True, choices=range(len(CHANNELS)))
    parser.add_argument("--obs_radius_km", type=float, required=True)
    parser.add_argument("--pred_file", help="Override prediction file path")
    parser.add_argument("--target_file", help="Override target file path")
    parser.add_argument("--out", help="Output PNG; default is generated under run_dir")
    parser.add_argument("--csv_out", help="Output CSV; default matches the generated PNG")
    parser.add_argument("--distance_chunk_size", type=int, default=1024)
    parser.add_argument("--target_rtol", type=float, default=2.0e-6)
    parser.add_argument("--target_atol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.obs_radius_km < 0.0:
        raise ValueError("--obs_radius_km must be nonnegative")

    run_dir = Path(args.run_dir).resolve()
    data_root = Path(args.data_root).resolve()
    grid_config_path = Path(args.grid_config).resolve()
    config = load_grid_config(str(grid_config_path))
    set_active_config(config)

    step_minutes, frames_per_day, freq_tag = parse_time_freq(args.time_freq)
    (year, month), frame = month_frame(args.analysis_time, step_minutes)
    days = days_in_month(year, month)
    frames_expected = days * frames_per_day
    if frame < 0 or frame >= frames_expected:
        raise IndexError(f"monthly frame {frame} outside [0, {frames_expected - 1}]")

    pred, target, pred_path, target_path = load_saved_pair(
        run_dir, args.rank, args.pred_file, args.target_file
    )
    pred, target = align_channels_last(pred, target)
    if pred.shape[0] != 1:
        raise ValueError(
            f"single-time diagnostic requires exactly one saved sample, found {pred.shape[0]}; "
            "the saved arrays do not contain timestamps"
        )
    if args.sample_index != 0:
        raise ValueError("single-time inference contains only --sample_index 0")

    lon_path = Path(loader_grid_x_path(str(data_root)))
    lat_path = Path(loader_grid_y_path(str(data_root)))
    if not lon_path.is_file() or not lat_path.is_file():
        raise FileNotFoundError(f"grid coordinate files not found: {lon_path}, {lat_path}")
    grid_lon = np.asarray(np.load(lon_path)).reshape(-1)
    grid_lat = np.asarray(np.load(lat_path)).reshape(-1)
    if grid_lon.size == 0 or grid_lat.size == 0:
        raise ValueError("grid longitude/latitude arrays must be nonempty")
    if np.any(np.diff(grid_lon) <= 0.0) or np.any(np.diff(grid_lat) <= 0.0):
        raise ValueError("grid longitude and latitude axes must be strictly ascending")
    nlon, nlat = grid_lon.size, grid_lat.size
    expected_saved_shape = (1, nlat, nlon, len(CHANNELS))
    if pred.shape != expected_saved_shape:
        raise ValueError(
            f"saved arrays have shape {pred.shape}; expected {expected_saved_shape} from grid files"
        )

    mean_path = Path(norm_mean_path(str(data_root), args.era5_mode))
    std_path = Path(norm_std_path(str(data_root), args.era5_mode))
    mean = np.asarray(np.load(mean_path)).reshape(-1)
    std = np.asarray(np.load(std_path)).reshape(-1)
    if mean.shape != (len(CHANNELS),) or std.shape != (len(CHANNELS),):
        raise ValueError(
            f"target norm factors must each contain {len(CHANNELS)} scalars: "
            f"mean={mean.shape}, std={std.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("target normalization factors must be finite with strictly positive std")

    channel_info = CHANNELS[args.channel]
    prediction = np.asarray(pred[0, :, :, args.channel])
    saved_truth = np.asarray(target[0, :, :, args.channel])

    target_month = Path(
        era5_month_path(str(data_root), args.era5_mode, freq_tag, year, month)
    )
    monthly_shape = (frames_expected, len(CHANNELS), nlon, nlat)
    raw_truth = load_target_witness(target_month, monthly_shape, frame, args.channel)
    if not np.allclose(
        saved_truth,
        raw_truth,
        rtol=args.target_rtol,
        atol=args.target_atol,
        equal_nan=True,
    ):
        delta = np.abs(saved_truth - raw_truth)
        raise ValueError(
            "saved inference truth does not match the monthly target at --analysis_time: "
            f"max_abs_diff={np.nanmax(delta):.6g}, mean_abs_diff={np.nanmean(delta):.6g}; "
            "check timestamp, sample, channel, and grid configuration"
        )

    bg_path = Path(background_month_path(str(data_root), args.era5_mode, year, month))
    background_shape = (days, len(CHANNELS), nlon, nlat)
    background, raw_background_path = load_background(
        bg_path,
        background_shape,
        args.analysis_time.day - 1,
        args.channel,
        mean,
        std,
    )

    obs_lon, obs_lat, obs_values, obs_path = load_observations(
        data_root,
        str(channel_info["observation"]),
        freq_tag,
        year,
        month,
        frame,
        frames_expected,
    )

    warn_if_implausible("saved truth", saved_truth, channel_info["physical_range"])
    warn_if_implausible("prediction", prediction, channel_info["physical_range"])
    warn_if_implausible("physical background", background, channel_info["physical_range"])
    warn_if_implausible(
        f"{channel_info['observation']} observations", obs_values, channel_info["obs_range"]
    )

    distance = nearest_observation_distance_km(
        grid_lon,
        grid_lat,
        obs_lon,
        obs_lat,
        args.distance_chunk_size,
    )
    near = distance <= args.obs_radius_km
    far = ~near
    if not np.any(near):
        raise ValueError(
            f"no grid points are within {args.obs_radius_km:g} km of a valid observation"
        )
    if not np.any(far):
        raise ValueError(
            f"all grid points are within {args.obs_radius_km:g} km of a valid observation; "
            "choose a smaller radius to calculate a far-observation metric"
        )
    metrics = calculate_metrics(
        prediction,
        background,
        saved_truth,
        {
            "full": np.ones(saved_truth.shape, dtype=bool),
            "near": near,
            "far": far,
        },
    )

    time_tag = args.analysis_time.strftime("%Y%m%dT%H%M")
    variable_tag = sanitize_filename(str(channel_info["analysis"]))
    radius_tag = sanitize_filename(f"{args.obs_radius_km:g}km")
    default_stem = f"diagnostic_{time_tag}_{variable_tag}_radius{radius_tag}"
    png_path = Path(args.out) if args.out else run_dir / f"{default_stem}.png"
    csv_path = Path(args.csv_out) if args.csv_out else run_dir / f"{default_stem}.csv"

    plot_diagnostics(
        png_path,
        args.analysis_time,
        channel_info,
        grid_lon,
        grid_lat,
        saved_truth,
        prediction,
        background,
        obs_lon,
        obs_lat,
        obs_values,
        distance,
        args.obs_radius_km,
        metrics,
    )
    write_metrics_csv(
        csv_path,
        args.analysis_time,
        args.channel,
        channel_info,
        args.obs_radius_km,
        obs_values.size,
        metrics,
        pred_path,
        target_path,
        bg_path,
        obs_path,
    )

    print(f"Timestamp witness passed: {target_month}")
    if raw_background_path is not None:
        print(f"Raw-background witness passed: {raw_background_path}")
    else:
        print("[INFO] raw background sibling not found; orientation follows the documented file contract")
    print(f"Valid {channel_info['observation']} observations: {obs_values.size}")
    for region in ("full", "near", "far"):
        row = metrics[region]
        print(
            f"{region}: model_rmse={row['prediction_rmse']:.6g} "
            f"background_rmse={row['background_rmse']:.6g} "
            f"model_bias={row['prediction_bias']:.6g} "
            f"background_bias={row['background_bias']:.6g} "
            f"skill={row['mse_skill_score']:.6g}"
        )
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
