"""Compute RTMA-OK target mean/std from existing monthly hourly target memmaps."""

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aardvark"))

from grid_config import (  # noqa: E402
    era5_month_path,
    load_grid_config,
    loader_grid_x_path,
    loader_grid_y_path,
    norm_mean_path,
    norm_std_path,
    set_active_config,
)
from month_manifest import read_month_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--months_file", required=True, help="Training months only")
    parser.add_argument("--era5_mode", default="rtma_ok_sfc")
    parser.add_argument(
        "--grid_config",
        default=str(REPO / "aardvark" / "grid_config_ok.yaml"),
        help="Grid/path YAML, or 'default' for the built-in global path templates",
    )
    parser.add_argument("--chunk_frames", type=int, default=24)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace loader-facing target norm files after all inputs validate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_frames < 1:
        raise ValueError("--chunk_frames must be positive")

    data_root = Path(args.data_root).resolve()
    months = read_month_manifest(args.months_file, "--months_file")
    config_path = (
        None
        if args.grid_config.lower() == "default"
        else str(Path(args.grid_config).resolve())
    )
    set_active_config(load_grid_config(config_path))
    mean_path = Path(norm_mean_path(str(data_root), args.era5_mode))
    std_path = Path(norm_std_path(str(data_root), args.era5_mode))
    if (mean_path.exists() or std_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"target norms already exist: {mean_path}, {std_path}; rerun with --overwrite "
            "after confirming the training-only month manifest"
        )
    nlon = np.asarray(np.load(loader_grid_x_path(str(data_root)))).size
    nlat = np.asarray(np.load(loader_grid_y_path(str(data_root)))).size

    channels = None
    total = None
    total_square = None
    count = 0
    monthly_inputs = []
    for year, month in months:
        frames = calendar.monthrange(year, month)[1] * 24
        path = Path(era5_month_path(str(data_root), args.era5_mode, "1h", year, month))
        if not path.is_file():
            raise FileNotFoundError(f"target month not found: {path}")
        frame_values = frames * nlon * nlat
        if path.stat().st_size % (frame_values * 4) != 0:
            raise ValueError(
                f"{path}: size does not represent {frames} complete float32 frames on "
                f"the {nlon}x{nlat} grid"
            )
        file_channels = path.stat().st_size // (frame_values * 4)
        if channels is None:
            channels = file_channels
            total = np.zeros(channels, dtype=np.float64)
            total_square = np.zeros(channels, dtype=np.float64)
        elif file_channels != channels:
            raise ValueError(f"{path}: channels={file_channels}, expected {channels}")
        monthly_inputs.append((path, frames))

    for path, frames in monthly_inputs:
        target = np.memmap(
            path,
            dtype="float32",
            mode="r",
            shape=(frames, channels, nlon, nlat),
        )
        for start in range(0, frames, args.chunk_frames):
            chunk = np.asarray(target[start : start + args.chunk_frames])
            finite = np.isfinite(chunk)
            if not np.all(finite):
                raise ValueError(f"{path}: target contains non-finite values")
            total += chunk.sum(axis=(0, 2, 3), dtype=np.float64)
            total_square += np.square(chunk, dtype=np.float64).sum(axis=(0, 2, 3))
            count += chunk.shape[0] * nlon * nlat
        del target
        print(f"[OK] included {path.name} ({frames} frames)")

    mean = total / count
    variance = total_square / count - np.square(mean)
    std = np.sqrt(np.clip(variance, 0.0, None)) + 1.0e-8
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"wrote {mean_path}")
    print(f"wrote {std_path}")
    print(f"samples per channel: {count}; months: {len(monthly_inputs)}")


if __name__ == "__main__":
    main()
