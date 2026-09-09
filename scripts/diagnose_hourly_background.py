"""Evaluate hour-matched RTMA-OK backgrounds before starting model training.

This CPU-only, streaming diagnostic compares the normalized hourly-background memmaps with the
raw hourly analysis targets. It reports, for every UTC hour and channel:

* GES[H] - analysis[H]: bias and RMSE (the background baseline the model must beat)
* GES[H] - analysis[H-1]: bias and RMSE (a persistence/contamination diagnostic)

Backgrounds are converted back to physical units with the target mean/std, matching the loader's
normalization contract. Full monthly files are size-checked before use and only one frame pair is
materialized at a time.
"""

import argparse
import calendar
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aardvark"))

from grid_config import (  # noqa: E402
    background_hourly_month_path,
    era5_month_path,
    load_grid_config,
    loader_grid_x_path,
    loader_grid_y_path,
    norm_mean_path,
    norm_std_path,
    set_active_config,
)

DEFAULT_CHANNEL_NAMES = ("tas", "sh", "psl", "u", "v")


def month_range(start, end):
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    if (y1, m1) < (y0, m0):
        raise ValueError(f"--end {end} is before --start {start}")
    months = []
    year, month = y0, m0
    while (year, month) <= (y1, m1):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def require_size(path, shape, label):
    expected = int(np.prod(shape)) * np.dtype("float32").itemsize
    actual = path.stat().st_size if path.is_file() else None
    if actual != expected:
        raise ValueError(
            f"{label} {path}: size={actual}, expected={expected} bytes for shape={shape}"
        )


def update(sum_values, sum_squares, counts, hour, differences):
    finite = np.isfinite(differences)
    safe = np.where(finite, differences, 0.0).astype(np.float64, copy=False)
    sum_values[hour] += safe.sum(axis=(1, 2), dtype=np.float64)
    sum_squares[hour] += np.square(safe).sum(axis=(1, 2), dtype=np.float64)
    counts[hour] += finite.sum(axis=(1, 2), dtype=np.int64)


def finalize(sum_values, sum_squares, counts):
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = sum_values / counts
        rmse = np.sqrt(sum_squares / counts)
    return bias, rmse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--start", required=True, help="First complete month, YYYY-MM")
    parser.add_argument("--end", required=True, help="Last complete month, YYYY-MM")
    parser.add_argument("--era5_mode", default="rtma_ok_sfc")
    parser.add_argument(
        "--grid_config",
        default=str(REPO / "aardvark" / "grid_config_ok.yaml"),
        help="Grid/path YAML, or 'default' for the built-in global path templates",
    )
    parser.add_argument(
        "--sample_every_days",
        type=int,
        default=2,
        help="Use every Nth day within each month (default 2, about 365 days over two years)",
    )
    parser.add_argument(
        "--csv_out",
        default="hourly_background_diagnostics.csv",
        help="Output table path",
    )
    parser.add_argument(
        "--leakage_threshold",
        type=float,
        default=1.0e-4,
        help="Fail any channel-hour with RMSE/target_std below this value",
    )
    parser.add_argument(
        "--leakage_ratio_threshold",
        type=float,
        default=1.0e-2,
        help="Fail any channel-hour whose current/previous-analysis RMSE ratio is below this",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_every_days < 1:
        raise ValueError("--sample_every_days must be positive")
    if args.leakage_threshold < 0:
        raise ValueError("--leakage_threshold must be nonnegative")
    if args.leakage_ratio_threshold < 0:
        raise ValueError("--leakage_ratio_threshold must be nonnegative")

    data_root = Path(args.data_root).resolve()
    config_path = (
        None
        if args.grid_config.lower() == "default"
        else str(Path(args.grid_config).resolve())
    )
    set_active_config(load_grid_config(config_path))
    nlon = np.asarray(np.load(loader_grid_x_path(str(data_root)))).size
    nlat = np.asarray(np.load(loader_grid_y_path(str(data_root)))).size
    mean = np.asarray(np.load(norm_mean_path(str(data_root), args.era5_mode))).reshape(-1)
    std = np.asarray(np.load(norm_std_path(str(data_root), args.era5_mode))).reshape(-1)
    if mean.shape != std.shape or not np.all(np.isfinite(mean)):
        raise ValueError(f"invalid target norm shapes/values: mean={mean.shape}, std={std.shape}")
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("target standard deviations must be finite and positive")

    channels = mean.size
    channel_names = (
        DEFAULT_CHANNEL_NAMES
        if channels == len(DEFAULT_CHANNEL_NAMES)
        else tuple(f"ch{i}" for i in range(channels))
    )
    shape_tail = (channels, nlon, nlat)
    current_sum = np.zeros((24, channels), dtype=np.float64)
    current_sumsq = np.zeros_like(current_sum)
    current_count = np.zeros((24, channels), dtype=np.int64)
    previous_sum = np.zeros_like(current_sum)
    previous_sumsq = np.zeros_like(current_sum)
    previous_count = np.zeros_like(current_count)
    sampled_days = 0

    for year, month in month_range(args.start, args.end):
        days = calendar.monthrange(year, month)[1]
        frames = days * 24
        target_path = Path(
            era5_month_path(str(data_root), args.era5_mode, "1h", year, month)
        )
        background_path = Path(
            background_hourly_month_path(str(data_root), args.era5_mode, year, month)
        )
        shape = (frames,) + shape_tail
        require_size(target_path, shape, "analysis")
        require_size(background_path, shape, "hourly background")
        target = np.memmap(target_path, dtype="float32", mode="r", shape=shape)
        background = np.memmap(background_path, dtype="float32", mode="r", shape=shape)

        for day in range(1, days + 1, args.sample_every_days):
            sampled_days += 1
            for hour in range(24):
                frame = (day - 1) * 24 + hour
                analysis = np.asarray(target[frame], dtype=np.float32)
                normalized_background = np.asarray(background[frame], dtype=np.float32)
                physical_background = (
                    normalized_background * std[:, None, None] + mean[:, None, None]
                )
                update(
                    current_sum,
                    current_sumsq,
                    current_count,
                    hour,
                    physical_background - analysis,
                )
                if frame > 0:
                    previous_analysis = np.asarray(target[frame - 1], dtype=np.float32)
                    update(
                        previous_sum,
                        previous_sumsq,
                        previous_count,
                        hour,
                        physical_background - previous_analysis,
                    )
        del target, background
        print(f"[OK] {year}-{month:02d}: sampled every {args.sample_every_days} day(s)")

    if np.any(current_count == 0):
        missing = np.argwhere(current_count == 0)
        raise ValueError(f"diagnostic has empty current-analysis channel-hour bins: {missing}")
    if np.any(previous_count == 0):
        missing = np.argwhere(previous_count == 0)
        raise ValueError(f"diagnostic has empty previous-analysis channel-hour bins: {missing}")

    current_bias, current_rmse = finalize(current_sum, current_sumsq, current_count)
    previous_bias, previous_rmse = finalize(previous_sum, previous_sumsq, previous_count)
    normalized_rmse = current_rmse / std[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        current_previous_ratio = current_rmse / previous_rmse

    csv_path = Path(args.csv_out).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "utc_hour",
                "channel",
                "ges_minus_analysis_bias",
                "ges_minus_analysis_rmse",
                "rmse_over_target_std",
                "ges_minus_previous_analysis_bias",
                "ges_minus_previous_analysis_rmse",
                "current_to_previous_rmse_ratio",
                "current_count",
                "previous_count",
            ]
        )
        for hour in range(24):
            for channel, name in enumerate(channel_names):
                writer.writerow(
                    [
                        hour,
                        name,
                        current_bias[hour, channel],
                        current_rmse[hour, channel],
                        normalized_rmse[hour, channel],
                        previous_bias[hour, channel],
                        previous_rmse[hour, channel],
                        current_previous_ratio[hour, channel],
                        current_count[hour, channel],
                        previous_count[hour, channel],
                    ]
                )

    print(f"\nSampled days: {sampled_days}")
    print(f"Wrote: {csv_path}")
    print(
        "\nUTC  variable   bias(GES-anl)  RMSE(GES-anl)  RMSE/std  "
        "RMSE(GES-anl[-1])  ratio"
    )
    for hour in range(24):
        for channel, name in enumerate(channel_names):
            print(
                f"{hour:02d}   {name:<8} {current_bias[hour, channel]:>13.6g} "
                f"{current_rmse[hour, channel]:>14.6g} "
                f"{normalized_rmse[hour, channel]:>9.6g} "
                f"{previous_rmse[hour, channel]:>18.6g} "
                f"{current_previous_ratio[hour, channel]:>8.4g}"
            )

    suspicious = np.argwhere(
        (normalized_rmse < args.leakage_threshold)
        | (current_previous_ratio < args.leakage_ratio_threshold)
    )
    if suspicious.size:
        bins = ", ".join(
            f"{hour:02d}Z/{channel_names[channel]}"
            for hour, channel in suspicious
        )
        raise SystemExit(
            "LEAKAGE GATE FAILED for channel-hour bin(s): "
            f"{bins}; thresholds: RMSE/target_std<{args.leakage_threshold:g} or "
            f"RMSE(current)/RMSE(previous)<{args.leakage_ratio_threshold:g}"
        )
    print("Leakage gate passed: all channel-hour bins contain data and exceed both thresholds.")


if __name__ == "__main__":
    main()
