#!/usr/bin/env python3
"""
Plot one saved Aardvark encoder prediction field against its ERA5 truth.

Examples:
  python scripts/plot_encoder_field.py --run_dir /path/to/encoder --sample_index 0 --channel 0
  python scripts/plot_encoder_field.py --run_dir /path/to/encoder --sample_index 4 --channel 12 --rank 0
  python scripts/plot_encoder_field.py --pred_file unnorm_preds.npy --target_file unnorm_targets.npy --channel 3
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


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


def infer_channel_name(channel: int, n_channels: int) -> str:
    pressure_count = len(PRESSURE_VARS_4U) * len(PRESSURE_LEVELS_4U)
    if n_channels in (24, 30) and channel < pressure_count:
        var_index = channel // len(PRESSURE_LEVELS_4U)
        level_index = channel % len(PRESSURE_LEVELS_4U)
        short_name, long_name = PRESSURE_VARS_4U[var_index]
        level = PRESSURE_LEVELS_4U[level_index]
        return f"{short_name}{level} ({long_name} {level} hPa)"

    if n_channels == 30 and pressure_count <= channel < pressure_count + len(SURFACE_VARS):
        short_name, long_name = SURFACE_VARS[channel - pressure_count]
        return f"{short_name} ({long_name})"

    return f"channel {channel}"


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
    candidates.extend(
        [
            ("unnorm_preds.npy", "unnorm_targets.npy"),
            ("preds_eval.npy", "y_target_eval.npy"),
        ]
    )

    for pred_name, target_name in candidates:
        pred_path = run_dir / pred_name
        target_path = run_dir / target_name
        if pred_path.exists() and target_path.exists():
            return np.load(pred_path), np.load(target_path), pred_path, target_path

    ranked_preds = sorted(run_dir.glob("unnorm_preds_*.npy"))
    for pred_path in ranked_preds:
        suffix = pred_path.name.replace("unnorm_preds_", "")
        target_path = run_dir / f"unnorm_targets_{suffix}"
        if target_path.exists():
            return np.load(pred_path), np.load(target_path), pred_path, target_path

    raise FileNotFoundError(
        f"No prediction/target pair found in {run_dir}. "
        "Expected unnorm_preds*.npy + unnorm_targets*.npy or preds_eval.npy + y_target_eval.npy."
    )


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


def main():
    parser = argparse.ArgumentParser(
        description="Plot prediction, ERA5 truth, and prediction-minus-truth for one saved encoder field."
    )
    parser.add_argument(
        "--run_dir",
        default=".",
        help="Directory containing unnorm_preds*.npy and unnorm_targets*.npy",
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
        help="Output PNG path. Default: encoder_field_sample_<sample>_channel_<channel>.png in --run_dir",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title prefix, for example a variable name like t850",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
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

    truth = target[args.sample_index, ..., args.channel]
    forecast = pred[args.sample_index, ..., args.channel]
    diff = forecast - truth

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
    channel_name = args.title or infer_channel_name(args.channel, pred.shape[-1])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)

    im0 = axes[0].imshow(truth, origin="lower", cmap="viridis", vmin=field_min, vmax=field_max)
    axes[0].set_title("Truth")
    im1 = axes[1].imshow(forecast, origin="lower", cmap="viridis", vmin=field_min, vmax=field_max)
    axes[1].set_title("Prediction")
    im2 = axes[2].imshow(diff, origin="lower", cmap="bwr", vmin=-diff_lim, vmax=diff_lim)
    axes[2].set_title("Prediction - Truth")

    for ax in axes:
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

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = run_dir / f"encoder_field_sample_{args.sample_index}_channel_{args.channel}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    rmse = float(np.sqrt(np.nanmean(diff**2)))
    bias = float(np.nanmean(diff))
    print(f"Wrote {out_path}")
    print(f"pred_shape={pred.shape} target_shape={target.shape}")
    print(f"rmse={rmse:.6g} bias={bias:.6g}")


if __name__ == "__main__":
    main()
