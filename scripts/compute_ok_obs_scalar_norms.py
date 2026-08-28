#!/usr/bin/env python3
"""Compute station-independent RTMA-OK observation normalization factors.

Month-specific station networks cannot use the original per-station mean/std arrays: station
index ``i`` may represent a different location (or not exist) in another month. This utility
scans the selected TRAINING months and writes one scalar mean and standard deviation per
variable, each stored as a one-element ``.npy`` array expected by the dynamic monthly loader.

Use training months only; including validation/test months would leak their distribution into
training preprocessing.

Example
-------
python scripts/compute_ok_obs_scalar_norms.py \
  --data_root /path/to/dr-av-rtma_ok_data \
  --months_file configs/rtma_train_months.txt \
  --freq_tag 1h

Outputs
-------
<data_root>/norm_factors/mean_hadisd_{tas,sh,psl,u,v}.npy
<data_root>/norm_factors/std_hadisd_{tas,sh,psl,u,v}.npy
"""

import argparse
import calendar
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aardvark"))
from month_manifest import parse_month as parse_manifest_month
from month_manifest import read_month_manifest


VARIABLES = ("tas", "sh", "psl", "u", "v")
FRAMES_PER_DAY = {"1h": 24, "15min": 96}


def save_native_norm(path, array, dry_run=False):
    """Write a real norm file rather than following a legacy output symlink.

    ``numpy.save`` follows an existing symbolic link.  Some staged datasets have links such
    as ``mean_hadisd_sh.npy -> mean_surface_q.npy``; saving through that path would overwrite
    the legacy ``surface`` file and leave the loader-facing name as a symlink.  Remove only
    the link itself before writing so this utility always creates the documented
    ``*_hadisd_*.npy`` output while leaving the former link target untouched.
    """
    if path.is_symlink():
        link_target = os.readlink(str(path))
        if dry_run:
            print(f"  [dry-run] would replace symlink {path} -> {link_target}")
        else:
            print(f"  replacing symlink {path} -> {link_target}")
            path.unlink()

    if dry_run:
        print(f"  [dry-run] would write regular file {path}")
    else:
        np.save(path, array)
        print(f"  wrote regular file {path}")


def parse_month(value):
    try:
        return parse_manifest_month(value, source="--months")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    month_group = parser.add_mutually_exclusive_group(required=True)
    month_group.add_argument(
        "--months",
        nargs="+",
        type=parse_month,
        help="training months only, for example 2022-01 2022-02",
    )
    month_group.add_argument(
        "--months_file",
        help="Text file listing training months, one exact YYYY-MM per line; blank lines "
        "and lines beginning with # are ignored.",
    )
    parser.add_argument("--freq_tag", default="1h", choices=sorted(FRAMES_PER_DAY))
    parser.add_argument("--vars", nargs="+", default=list(VARIABLES), choices=VARIABLES)
    parser.add_argument(
        "--out_dir",
        help="output norm_factors directory (default: <data_root>/norm_factors)",
    )
    parser.add_argument("--chunk_frames", type=int, default=96)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.months_file:
        try:
            args.months = read_month_manifest(args.months_file, "--months_file")
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
    return args


def main():
    args = parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("--chunk_frames must be positive")

    data_root = Path(args.data_root)
    obs_dir = data_root / "hadisd_processed"
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "norm_factors"
    frames_per_day = FRAMES_PER_DAY[args.freq_tag]
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for variable in args.vars:
        total = 0.0
        total_square = 0.0
        count = 0
        print(f"\n{variable}")

        for year, month in args.months:
            tag = f"{year}-{month:02d}"
            lon_path = obs_dir / f"{variable}_lon_train-{tag}.npy"
            value_path = obs_dir / f"{variable}_vals_{args.freq_tag}_{tag}.memmap"
            if not lon_path.is_file():
                raise FileNotFoundError(f"coordinate file not found: {lon_path}")
            stations = np.asarray(np.load(lon_path)).reshape(-1).size
            if stations == 0:
                raise ValueError(f"{lon_path} contains no stations")

            frames = calendar.monthrange(year, month)[1] * frames_per_day
            expected_bytes = frames * stations * np.dtype("float32").itemsize
            if not value_path.is_file():
                raise FileNotFoundError(f"observation file not found: {value_path}")
            if value_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"{value_path}: size {value_path.stat().st_size} != {expected_bytes} "
                    f"for shape ({frames}, {stations})"
                )

            values = np.memmap(
                value_path, dtype="float32", mode="r", shape=(frames, stations)
            )
            month_count = 0
            for start in range(0, frames, args.chunk_frames):
                chunk = np.asarray(values[start : start + args.chunk_frames], dtype=np.float64)
                finite_values = chunk[np.isfinite(chunk)]
                if finite_values.size:
                    total += float(np.sum(finite_values, dtype=np.float64))
                    total_square += float(
                        np.sum(np.square(finite_values), dtype=np.float64)
                    )
                    count += int(finite_values.size)
                    month_count += int(finite_values.size)
            print(f"  {tag}: stations={stations}, finite_values={month_count}")

        if count == 0:
            raise ValueError(f"no finite training values found for {variable}")
        mean = total / count
        variance = max(total_square / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"invalid scalar norms for {variable}: mean={mean}, std={std}")

        mean_array = np.asarray([mean], dtype=np.float32)
        std_array = np.asarray([std], dtype=np.float32)
        mean_path = out_dir / f"mean_hadisd_{variable}.npy"
        std_path = out_dir / f"std_hadisd_{variable}.npy"
        print(f"  combined: count={count}, mean={mean:.8g}, std={std:.8g}")
        save_native_norm(mean_path, mean_array, dry_run=args.dry_run)
        save_native_norm(std_path, std_array, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
