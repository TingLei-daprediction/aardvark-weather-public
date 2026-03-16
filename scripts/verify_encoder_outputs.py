#!/usr/bin/env python3
"""
Compare saved encoder prediction and truth arrays from a training run.

Primary use:
  python scripts/verify_encoder_outputs.py --run_dir /path/to/encoder/output

Examples:
  python scripts/verify_encoder_outputs.py --run_dir /path/to/run
  python scripts/verify_encoder_outputs.py --run_dir /path/to/run --plot_channel 5
  python scripts/verify_encoder_outputs.py --run_dir /path/to/run --sample_index 7 --plot_channel 3
  python scripts/verify_encoder_outputs.py --run_dir /path/to/run --no_plots

The script prefers unnormalized arrays:
  - unnorm_preds.npy
  - unnorm_targets.npy

If those are absent, it can fall back to:
  - preds_eval.npy
  - y_target_eval.npy

Outputs:
  - metrics_summary.txt
  - metrics_by_channel.csv
  - optional PNG plots
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_pair(run_dir: Path):
    unnorm_pred = run_dir / "unnorm_preds.npy"
    unnorm_target = run_dir / "unnorm_targets.npy"
    eval_pred = run_dir / "preds_eval.npy"
    eval_target = run_dir / "y_target_eval.npy"

    ranked_unnorm_preds = sorted(run_dir.glob("unnorm_preds_*.npy"))
    ranked_unnorm_targets = sorted(run_dir.glob("unnorm_targets_*.npy"))

    if unnorm_pred.exists() and unnorm_target.exists():
        return (
            np.load(unnorm_pred),
            np.load(unnorm_target),
            "unnorm_preds.npy",
            "unnorm_targets.npy",
        )

    if eval_pred.exists() and eval_target.exists():
        return (
            np.load(eval_pred),
            np.load(eval_target),
            "preds_eval.npy",
            "y_target_eval.npy",
        )

    if ranked_unnorm_preds and ranked_unnorm_targets:
        pred_map = {p.name.replace("unnorm_preds_", ""): p for p in ranked_unnorm_preds}
        target_map = {p.name.replace("unnorm_targets_", ""): p for p in ranked_unnorm_targets}
        common_keys = sorted(set(pred_map) & set(target_map))
        if common_keys:
            key = common_keys[0]
            pred_path = pred_map[key]
            target_path = target_map[key]
            return (
                np.load(pred_path),
                np.load(target_path),
                pred_path.name,
                target_path.name,
            )

    raise FileNotFoundError(
        "Could not find prediction/target pairs in run_dir. "
        "Expected unnorm_preds.npy + unnorm_targets.npy "
        "or rank-specific unnorm_preds_<rank>.npy + unnorm_targets_<rank>.npy "
        "or preds_eval.npy + y_target_eval.npy."
    )


def align_last_dim(pred: np.ndarray, target: np.ndarray):
    if pred.shape == target.shape:
        return pred, target

    if pred.ndim != target.ndim:
        raise ValueError(
            f"Prediction/target ndim mismatch: pred {pred.shape}, target {target.shape}"
        )

    # Common case in this repo: arrays may be [B, C, H, W] or [B, H, W, C].
    # If one side has channels in dim=1 and the other in dim=-1, permute.
    if pred.shape[1:] == target.shape[:-1] + (target.shape[-1],):
        pass

    if pred.shape[0] == target.shape[0]:
        if pred.shape[1] == target.shape[-1] and pred.shape[2:] == target.shape[1:-1]:
            pred = np.moveaxis(pred, 1, -1)
            if pred.shape == target.shape:
                return pred, target
        if target.shape[1] == pred.shape[-1] and target.shape[2:] == pred.shape[1:-1]:
            target = np.moveaxis(target, 1, -1)
            if pred.shape == target.shape:
                return pred, target

    raise ValueError(f"Prediction/target shape mismatch: pred {pred.shape}, target {target.shape}")


def compute_channel_metrics(pred: np.ndarray, target: np.ndarray):
    n_channels = pred.shape[-1]
    rows = []
    for ch in range(n_channels):
        p = pred[..., ch]
        t = target[..., ch]
        mask = np.isfinite(p) & np.isfinite(t)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "channel": ch,
                    "count": 0,
                    "bias": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "corr": np.nan,
                }
            )
            continue

        p1 = p[mask].reshape(-1)
        t1 = t[mask].reshape(-1)
        diff = p1 - t1
        bias = float(np.mean(diff))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        if p1.size > 1 and np.std(p1) > 0 and np.std(t1) > 0:
            corr = float(np.corrcoef(p1, t1)[0, 1])
        else:
            corr = np.nan

        rows.append(
            {
                "channel": ch,
                "count": count,
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
                "corr": corr,
            }
        )
    return rows


def write_metrics(run_dir: Path, rows, pred_name: str, target_name: str, pred_shape, target_shape):
    summary_path = run_dir / "metrics_summary.txt"
    csv_path = run_dir / "metrics_by_channel.csv"

    rmse_vals = np.array([r["rmse"] for r in rows], dtype=float)
    mae_vals = np.array([r["mae"] for r in rows], dtype=float)
    bias_vals = np.array([r["bias"] for r in rows], dtype=float)
    corr_vals = np.array([r["corr"] for r in rows], dtype=float)

    with summary_path.open("w", encoding="ascii") as f:
        f.write(f"pred_file: {pred_name}\n")
        f.write(f"target_file: {target_name}\n")
        f.write(f"pred_shape: {pred_shape}\n")
        f.write(f"target_shape: {target_shape}\n")
        f.write(f"n_channels: {len(rows)}\n")
        f.write(f"mean_rmse: {np.nanmean(rmse_vals):.6f}\n")
        f.write(f"mean_mae: {np.nanmean(mae_vals):.6f}\n")
        f.write(f"mean_bias: {np.nanmean(bias_vals):.6f}\n")
        f.write(f"mean_corr: {np.nanmean(corr_vals):.6f}\n")

    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(
            f, fieldnames=["channel", "count", "bias", "mae", "rmse", "corr"]
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    run_dir: Path,
    rows,
    pred: np.ndarray,
    target: np.ndarray,
    sample_index: int,
    channel: int,
):
    import matplotlib.pyplot as plt

    channels = np.array([r["channel"] for r in rows], dtype=int)
    rmse = np.array([r["rmse"] for r in rows], dtype=float)
    bias = np.array([r["bias"] for r in rows], dtype=float)
    corr = np.array([r["corr"] for r in rows], dtype=float)

    plt.figure(figsize=(9, 4.5))
    plt.plot(channels, rmse, marker="o", label="rmse")
    plt.plot(channels, np.abs(bias), marker="s", label="abs_bias")
    plt.plot(channels, corr, marker="^", label="corr")
    plt.xlabel("Channel")
    plt.ylabel("Metric")
    plt.title("Encoder Verification by Channel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "metrics_by_channel.png", dpi=150)
    plt.close()

    p = pred[sample_index, ..., channel]
    t = target[sample_index, ..., channel]
    d = p - t
    vmax = np.nanmax(np.abs([np.nanmin(t), np.nanmax(t), np.nanmin(p), np.nanmax(p)]))
    dmax = np.nanmax(np.abs(d))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(t, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Truth sample {sample_index} ch{channel}")
    im1 = axes[1].imshow(p, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Prediction sample {sample_index} ch{channel}")
    im2 = axes[2].imshow(d, origin="lower", cmap="bwr", vmin=-dmax, vmax=dmax)
    axes[2].set_title(f"Diff sample {sample_index} ch{channel}")
    for ax in axes:
        ax.set_xlabel("Lon index")
        ax.set_ylabel("Lat index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        run_dir / f"sample_{sample_index}_maps_channel_{channel}.png",
        dpi=150,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare encoder prediction and truth arrays saved in a run directory."
    )
    parser.add_argument("--run_dir", required=True, help="Encoder output directory")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help=(
            "Index along the saved batch dimension to plot. "
            "This is the sample index within the saved prediction batch, not a datetime."
        ),
    )
    parser.add_argument(
        "--plot_channel",
        type=int,
        default=0,
        help="Output channel index to use for the sample truth/pred/diff map",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip matplotlib plots and only write metrics text/csv",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    pred, target, pred_name, target_name = load_pair(run_dir)
    pred, target = align_last_dim(pred, target)

    if pred.ndim != 4:
        raise ValueError(
            f"Expected 4D arrays after alignment, got pred {pred.shape}, target {target.shape}"
        )

    if not (0 <= args.sample_index < pred.shape[0]):
        raise ValueError(f"--sample_index must be in [0, {pred.shape[0] - 1}]")

    if not (0 <= args.plot_channel < pred.shape[-1]):
        raise ValueError(f"--plot_channel must be in [0, {pred.shape[-1] - 1}]")

    rows = compute_channel_metrics(pred, target)
    write_metrics(run_dir, rows, pred_name, target_name, pred.shape, target.shape)

    if not args.no_plots:
        make_plots(
            run_dir,
            rows,
            pred,
            target,
            args.sample_index,
            args.plot_channel,
        )

    print(f"Wrote verification outputs to {run_dir}")


if __name__ == "__main__":
    main()
