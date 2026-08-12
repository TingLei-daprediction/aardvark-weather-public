"""Plot encoder truth/prediction as ANOMALIES -- each field minus its own domain mean.

Companion to plot_encoder_field.py. Same inputs, same grid handling, same channel naming;
the only difference is that both fields are centred before plotting:

    truth_anom = truth - mean(truth)
    pred_anom  = pred  - mean(pred)

Because each field is centred on itself, the difference panel has a mean of exactly zero, so
its RMSE is the CENTERED RMSE -- the pattern error with the bias removed:

    crmse = sqrt(rmse^2 - bias^2)

That is the decomposition that separates "the forecast is uniformly offset" from "the forecast
has the wrong spatial structure". A large bias with a small crmse means the model has the right
pattern at the wrong level (seasonal offset, normalization mismatch); a large crmse means the
structure itself is wrong.

The bias is NOT hidden -- it is removed from the field and reported in the titles, since it is
usually the more interesting number of the two.

Colour scales differ from plot_encoder_field.py on purpose: anomalies are diverging data
centred on zero, so both field panels use a symmetric diverging colormap on a SHARED scale,
making truth and prediction directly comparable.

Output PNG carries an "_anomaly" suffix so it never collides with plot_encoder_field.py:
    sample_<i>_<var>_ch<n>_anomaly.png

Usage:
    python scripts/plot_encoder_field_anomaly.py --run_dir <run> --rank 0 --channel 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the sibling script's helpers rather than copying them, so grid handling, channel
# naming and array alignment cannot drift between the two tools.
import plot_encoder_field as pef
from plot_encoder_field import (
    DEFAULT_GRID_DIR,
    HAS_CARTOPY,
    add_geospatial_features,
    align_to_channels_last,
    infer_channel_info,
    load_grid_coordinates,
    load_pair,
    sanitize_filename,
    symmetric_limit,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--channel", type=int, required=True, help="Output channel index")
    parser.add_argument("--out", help="Output PNG path. Default adds an _anomaly suffix.")
    parser.add_argument("--title", default="", help="Optional title prefix, e.g. a variable name")
    parser.add_argument("--cmap", default="RdBu_r", help="Diverging colormap for the anomalies")
    parser.add_argument("--fig_width", type=float, default=16.0)
    parser.add_argument("--fig_height", type=float, default=3.6)
    parser.add_argument("--dpi", type=int, default=180)
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

    grid_extent, _lons, _lats = load_grid_coordinates(grid_dir)

    truth = target[args.sample_index, ..., args.channel]
    forecast = pred[args.sample_index, ..., args.channel]

    # Full-field statistics, computed BEFORE centring so the removed offset can be reported.
    raw_diff = forecast - truth
    rmse = float(np.sqrt(np.nanmean(raw_diff**2)))
    bias = float(np.nanmean(raw_diff))
    # Share of the mean squared error explained by the offset alone. Near 100% means the
    # forecast has the right pattern at the wrong level; near 0% means the error is structural.
    bias_share = 100.0 * (bias**2) / (rmse**2) if rmse > 0 else float("nan")

    truth_mean = float(np.nanmean(truth))
    pred_mean = float(np.nanmean(forecast))

    truth_anom = truth - truth_mean
    forecast_anom = forecast - pred_mean

    # Zero by construction; recomputed rather than assumed so a NaN mask mismatch shows up.
    diff = forecast_anom - truth_anom
    crmse = float(np.sqrt(np.nanmean(diff**2)))
    residual_bias = float(np.nanmean(diff))

    anom_lim = symmetric_limit(
        np.concatenate(
            [
                truth_anom[np.isfinite(truth_anom)].reshape(-1),
                forecast_anom[np.isfinite(forecast_anom)].reshape(-1),
            ]
        )
    )
    diff_lim = symmetric_limit(diff)

    short_code, channel_desc = infer_channel_info(args.channel, pred.shape[-1])
    channel_name = args.title or channel_desc

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_cartopy = HAS_CARTOPY and grid_extent is not None
    if use_cartopy:
        proj = pef.ccrs.PlateCarree()
        subplot_kwargs = {"projection": proj}
    else:
        proj = None
        subplot_kwargs = {}
        if not HAS_CARTOPY:
            print("[WARN] Cartopy not installed; plotting index axes without map borders.")
        else:
            print(
                "[WARN] No grid coordinates loaded; plotting index axes without map borders. "
                "Pass --grid_dir to get a georeferenced plot."
            )

    fig, axes = plt.subplots(
        1, 3, figsize=(args.fig_width, args.fig_height),
        constrained_layout=True, subplot_kw=subplot_kwargs,
    )

    imshow_kwargs = {"origin": "lower"}
    if grid_extent:
        imshow_kwargs["extent"] = grid_extent

    im0 = axes[0].imshow(
        truth_anom, cmap=args.cmap, vmin=-anom_lim, vmax=anom_lim, **imshow_kwargs
    )
    axes[0].set_title(f"Truth anomaly\nmean removed = {truth_mean:.6g}")

    im1 = axes[1].imshow(
        forecast_anom, cmap=args.cmap, vmin=-anom_lim, vmax=anom_lim, **imshow_kwargs
    )
    axes[1].set_title(f"Prediction anomaly\nmean removed = {pred_mean:.6g}")

    im2 = axes[2].imshow(diff, cmap="bwr", vmin=-diff_lim, vmax=diff_lim, **imshow_kwargs)
    axes[2].set_title(
        f"Prediction - Truth (anomalies)\n"
        f"Centered RMSE={crmse:.6g}\n"
        f"bias REMOVED = {bias:.6g} ({bias_share:.1f}% of MSE)\n"
        f"residual bias = {residual_bias:.2g} (should be ~0)"
    )

    for ax in axes:
        if use_cartopy:
            add_geospatial_features(ax)
            ax.set_extent(grid_extent, crs=proj)
        else:
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{channel_name}; sample={args.sample_index}, channel={args.channel} "
        f"-- ANOMALY (each field minus its own mean)\n"
        f"full RMSE={rmse:.6g}, bias={bias:.6g} ({bias_share:.1f}% of MSE), "
        f"centered RMSE={crmse:.6g}\n"
        f"pred={pred_path.name}, truth={target_path.name}",
        fontsize=9,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        var_tag = sanitize_filename(short_code)
        out_path = run_dir / f"sample_{args.sample_index}_{var_tag}_ch{args.channel}_anomaly.png"

    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"truth_mean={truth_mean:.6g} pred_mean={pred_mean:.6g}")
    print(
        f"full rmse={rmse:.6g} bias={bias:.6g} ({bias_share:.1f}% of MSE) "
        f"centered rmse={crmse:.6g}"
    )


if __name__ == "__main__":
    main()
