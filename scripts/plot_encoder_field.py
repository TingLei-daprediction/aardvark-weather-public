#!/usr/bin/env python3
"""
Plot one saved Aardvark encoder prediction field against its ERA5 truth with
state and county borders, using Oklahoma 1D grid coordinate files (urma_x_ok.npy and urma_y_ok.npy).

Examples:
  python scripts/plot_encoder_field.py --run_dir /path/to/encoder --sample_index 0 --channel 0
  python scripts/plot_encoder_field.py --channel 0 --grid_dir /custom/path/to/grid_lon_lat
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Optional dependencies for mapping
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


DEFAULT_GRID_DIR = "/scratch3/NCEPDEV/fv3-cam/Annette.Gibbs/aardvark_OK/aardvark-weather-public/data/grid_lon_lat"

PRESSURE_LEVELS_4U = [850, 700, 500, 200]
PRESSURE_VARS_4U = [
    ("z", "geopotential"),
    ("t", "temperature"),
    ("r", "relative humidity"),
    ("u", "u wind"),
    ("v", "v wind"),
    ("q", "specific humidity"),
]
SURFACE_VARS = [
    ("t2m", "2m temperature"),
    ("d2m", "2m dewpoint temperature"),
    ("u10", "10m u wind"),
    ("v10", "10m v wind"),
    ("msl", "mean sea-level pressure"),
    ("sp", "surface pressure"),
]
OK_SFC_VARS = [
    ("t2m", "2m temperature [K]"),
    ("q2m", "2m specific humidity [kg/kg]"),
    ("sp", "surface pressure [Pa]"),
    ("u10", "10m u wind [m/s]"),
    ("v10", "10m v wind [m/s]"),
]


def infer_channel_info(channel: int, n_channels: int) -> Tuple[str, str]:
    """Returns (short_code, full_description) for a given channel index."""
    if n_channels == len(OK_SFC_VARS):
        short_name, long_name = OK_SFC_VARS[channel]
        return short_name, f"{short_name} ({long_name})"

    pressure_count = len(PRESSURE_VARS_4U) * len(PRESSURE_LEVELS_4U)
    if n_channels in (24, 30) and channel < pressure_count:
        var_index = channel // len(PRESSURE_LEVELS_4U)
        level_index = channel % len(PRESSURE_LEVELS_4U)
        short_name, long_name = PRESSURE_VARS_4U[var_index]
        level = PRESSURE_LEVELS_4U[level_index]
        code = f"{short_name}{level}"
        desc = f"{short_name}{level} ({long_name} {level} hPa)"
        return code, desc

    if n_channels == 30 and pressure_count <= channel < pressure_count + len(SURFACE_VARS):
        short_name, long_name = SURFACE_VARS[channel - pressure_count]
        return short_name, f"{short_name} ({long_name})"

    return f"chan{channel}", f"channel {channel}"


def sanitize_filename(name: str) -> str:
    """Sanitizes string into a safe filename tag."""
    name = re.sub(r"[^\w\-]", "_", name)
    return re.sub(r"_+", "_", name).strip("_").lower()


def load_pair(run_dir: Path, rank: Optional[str], pred_file: Optional[str], target_file: Optional[str]):
    if pred_file or target_file:
        if not pred_file or not target_file:
            raise ValueError("--pred_file and --target_file must be provided together")
        pred_path = Path(pred_file)
        target_path = Path(target_file)
        if not pred_path.is_absolute():
            pred_path = run_dir / pred_path
        if not target_path.is_absolute():
            target_path = run_dir / target_path
        return np.load(pred_path), np.load(target_path), pred_path, target_path

    candidates = []
    if rank is not None:
        candidates.append((f"unnorm_preds_{rank}.npy", f"unnorm_targets_{rank}.npy"))
    candidates.append(("unnorm_preds.npy", "unnorm_targets.npy"))
    for pred_path in sorted(run_dir.glob("unnorm_preds_*.npy")):
        suffix = pred_path.name.replace("unnorm_preds_", "")
        candidates.append((pred_path.name, f"unnorm_targets_{suffix}"))
    candidates.append(("preds_eval.npy", "y_target_eval.npy"))

    for pred_name, target_name in candidates:
        pred_path = run_dir / pred_name
        target_path = run_dir / target_name
        if pred_path.exists() and target_path.exists():
            if pred_name == "preds_eval.npy":
                print(
                    "[WARN] only the NORMALIZED eval arrays (preds_eval.npy) were found; "
                    "plotted values are z-scores, not physical units."
                )
            return np.load(pred_path), np.load(target_path), pred_path, target_path

    raise FileNotFoundError(
        f"No prediction/target pair found in {run_dir}. "
        "Expected unnorm_preds*.npy + unnorm_targets*.npy or preds_eval.npy + y_target_eval.npy."
    )


def load_grid_coordinates(grid_dir: Path) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads 1D longitude (urma_x_ok.npy) and latitude (urma_y_ok.npy) coordinate arrays.
    Converts 0-360° longitude scale to -180..180° for Cartopy compatibility.
    """
    x_file = grid_dir / "urma_x_ok.npy"
    y_file = grid_dir / "urma_y_ok.npy"

    if not (x_file.exists() and y_file.exists()):
        print(f"[WARN] Grid files {x_file.name} / {y_file.name} not found in {grid_dir}.")
        return None, None, None

    lon_coords = np.load(x_file)
    lat_coords = np.load(y_file)

    # Convert 0-360° easting to -180..180° range if needed
    if np.any(lon_coords > 180):
        lon_coords = np.where(lon_coords > 180, lon_coords - 360, lon_coords)

    lon_min, lon_max = float(lon_coords.min()), float(lon_coords.max())
    lat_min, lat_max = float(lat_coords.min()), float(lat_coords.max())

    extent = (lon_min, lon_max, lat_min, lat_max)
    print(f"Loaded Grid Coordinates from {grid_dir}:")
    print(f"  Longitude extent: [{lon_min:.4f}, {lon_max:.4f}]")
    print(f"  Latitude extent:  [{lat_min:.4f}, {lat_max:.4f}]")

    return extent, lon_coords, lat_coords


def align_to_channels_last(pred: np.ndarray, target: np.ndarray):
    if pred.shape == target.shape:
        return pred, target

    if pred.ndim != target.ndim:
        raise ValueError(f"ndim mismatch: pred {pred.shape}, target {target.shape}")

    if pred.ndim == 4 and pred.shape[0] == target.shape[0]:
        if pred.shape[1] == target.shape[-1] and pred.shape[2:] == target.shape[1:-1]:
            pred = np.moveaxis(pred, 1, -1)
        if target.shape[1] == pred.shape[-1] and target.shape[2:] == pred.shape[1:-1]:
            target = np.moveaxis(target, 1, -1)

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch after alignment: pred {pred.shape}, target {target.shape}")
    return pred, target


def symmetric_limit(*arrays):
    finite = []
    for arr in arrays:
        vals = arr[np.isfinite(arr)]
        if vals.size:
            finite.append(vals)
    if not finite:
        return 1.0
    return float(np.max(np.abs(np.concatenate(finite))))


def add_geospatial_features(ax):
    """Adds coastlines, state borders, and county borders to a GeoAxes plot."""
    if not HAS_CARTOPY:
        return

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor="black")

    if HAS_METPY_COUNTIES:
        ax.add_feature(USCOUNTIES.with_scale("20m"), linewidth=0.3, edgecolor="gray", alpha=0.6)
    else:
        counties = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_2_counties",
            scale="10m",
            facecolor="none",
        )
        ax.add_feature(counties, linewidth=0.3, edgecolor="gray", alpha=0.6)


def main():
    parser = argparse.ArgumentParser(
        description="Plot prediction, ERA5 truth, and prediction-minus-truth using OK grid coordinate files."
    )
    parser.add_argument(
        "--run_dir",
        default=".",
        help="Directory containing unnorm_preds*.npy and unnorm_targets*.npy",
    )
    parser.add_argument(
        "--grid_dir",
        default=DEFAULT_GRID_DIR,
        help="Directory containing urma_x_ok.npy and urma_y_ok.npy coordinate files",
    )
    parser.add_argument("--pred_file", help="Prediction .npy path or filename under --run_dir")
    parser.add_argument("--target_file", help="Truth .npy path or filename under --run_dir")
    parser.add_argument(
        "--rank",
        help="Rank suffix to load, e.g. 0 for unnorm_preds_0.npy and unnorm_targets_0.npy",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index along the saved batch dimension. This is not a datetime.",
    )
    parser.add_argument("--channel", type=int, required=True, help="ERA5 output channel index")
    parser.add_argument(
        "--out",
        help="Output PNG path. If omitted, generated automatically based on field variable name.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title prefix, for example a variable name like t850",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=20.0,
        help="Figure width in inches (default: 20).",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=6.5,
        help="Figure height in inches (default: 6.5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output resolution in dots per inch (default: 180).",
    )
    args = parser.parse_args()

    if args.fig_width <= 0 or args.fig_height <= 0:
        raise ValueError("--fig_width and --fig_height must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    run_dir = Path(args.run_dir)
    grid_dir = Path(args.grid_dir)

    pred, target, pred_path, target_path = load_pair(
        run_dir, args.rank, args.pred_file, args.target_file
    )
    pred, target = align_to_channels_last(pred, target)

    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays [sample, y, x, channel], got {pred.shape}")
    if not (0 <= args.sample_index < pred.shape[0]):
        raise ValueError(f"--sample_index must be in [0, {pred.shape[0] - 1}]")
    if not (0 <= args.channel < pred.shape[-1]):
        raise ValueError(f"--channel must be in [0, {pred.shape[-1] - 1}]")

    # Load 1D grid coordinates
    grid_extent, lons, lats = load_grid_coordinates(grid_dir)

    truth = target[args.sample_index, ..., args.channel]
    forecast = pred[args.sample_index, ..., args.channel]
    diff = forecast - truth
    rmse = float(np.sqrt(np.nanmean(diff**2)))
    bias = float(np.nanmean(diff))

    finite_fields = np.concatenate(
        [
            truth[np.isfinite(truth)].reshape(-1),
            forecast[np.isfinite(forecast)].reshape(-1),
        ]
    )
    if finite_fields.size:
        field_min = float(np.min(finite_fields))
        field_max = float(np.max(finite_fields))
    else:
        field_min = -1.0
        field_max = 1.0
    diff_lim = symmetric_limit(diff)

    short_code, channel_desc = infer_channel_info(args.channel, pred.shape[-1])
    channel_name = args.title or channel_desc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        subplot_kwargs = {"projection": proj}
    else:
        subplot_kwargs = {}
        print("[WARN] Cartopy not installed; plotting standard grid axes without map borders.")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(args.fig_width, args.fig_height),
        constrained_layout=True,
        subplot_kw=subplot_kwargs,
    )

    imshow_kwargs = {"origin": "lower"}
    if grid_extent:
        imshow_kwargs["extent"] = grid_extent

    im0 = axes[0].imshow(truth, cmap="viridis", vmin=field_min, vmax=field_max, **imshow_kwargs)
    axes[0].set_title("Truth")

    im1 = axes[1].imshow(forecast, cmap="viridis", vmin=field_min, vmax=field_max, **imshow_kwargs)
    axes[1].set_title("Prediction")

    im2 = axes[2].imshow(diff, cmap="bwr", vmin=-diff_lim, vmax=diff_lim, **imshow_kwargs)
    axes[2].set_title(f"Prediction - Truth\nRMSE={rmse:.6g}, Bias={bias:.6g}")

    for ax in axes:
        if HAS_CARTOPY:
            add_geospatial_features(ax)
            if grid_extent:
                ax.set_extent(grid_extent, crs=proj)
        else:
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{channel_name}; sample={args.sample_index}, channel={args.channel}\n"
        f"pred={pred_path.name}, truth={target_path.name}",
        fontsize=10,
    )

    # Output filename generation
    if args.out:
        out_path = Path(args.out)
    else:
        var_tag = sanitize_filename(short_code)
        out_path = run_dir / f"sample_{args.sample_index}_{var_tag}_ch{args.channel}.png"

    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"pred_shape={pred.shape} target_shape={target.shape}")
    print(f"rmse={rmse:.6g} bias={bias:.6g}")


if __name__ == "__main__":
    main()
