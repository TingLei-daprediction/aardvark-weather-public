"""
Build climatology_data.mmap from ERA5 memmaps.

Output shape: (4, 366, channels, 240, 121) by default.
We compute daily means from 6-hourly data and replicate across the first dim.
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build climatology_data.mmap from ERA5 memmaps")
    p.add_argument("--data_dir", required=True, help="Base data_path containing era5 memmaps")
    p.add_argument("--era5_mode", default="4u_sfc", help="ERA5 mode in file names (default: 4u_sfc)")
    p.add_argument("--res", type=int, default=1, help="Resolution suffix (default: 1)")
    p.add_argument("--years", nargs="+", type=int, required=True, help="Years to include")
    p.add_argument("--output_name", default="climatology_data.mmap", help="Output file name")
    p.add_argument("--time_freq", default="1D", help="6H or 1D (daily 00 UTC)")
    p.add_argument(
        "--fill_nan",
        type=float,
        default=None,
        help="If set, replace NaN/Inf with this value before averaging.",
    )
    return p.parse_args()


def is_leap(year):
    return year % 4 == 0


def main():
    args = parse_args()
    era5_dir = Path(args.data_dir) / "era5"
    freq_tag = "6" if args.time_freq == "6H" else "1d"
    step_factor = 4 if args.time_freq == "6H" else 1

    # Load one file to get channel/shape
    sample_year = args.years[0]
    sample_path = era5_dir / f"era5_{args.era5_mode}_{args.res}_{freq_tag}_{sample_year}.memmap"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    # infer shape
    d = (366 if is_leap(sample_year) else 365) * step_factor
    # channels, x, y are inferred from file size by loading with known dims
    # We use the standard 1-degree grid size from the code
    x = 240
    y = 121

    # Read channel count from file size
    n_floats = sample_path.stat().st_size // 4
    channels = n_floats // (d * x * y)
    if n_floats % (d * x * y) != 0:
        raise ValueError("Sample memmap size is not divisible by (d*x*y); check inputs.")

    climatology = np.memmap(
        era5_dir / args.output_name,
        dtype="float32",
        mode="w+",
        shape=(4, 366, channels, x, y),
    )
    sum_days = np.zeros((4, 366, channels, x, y), dtype=np.float64)
    count_days = np.zeros((4, 366), dtype=np.int64)

    for year in args.years:
        days = 366 if is_leap(year) else 365
        d = days * step_factor
        path = era5_dir / f"era5_{args.era5_mode}_{args.res}_{freq_tag}_{year}.memmap"
        if not path.exists():
            print(f"[WARN] Missing {path}, skipping.")
            continue
        data = np.memmap(path, dtype="float32", mode="r", shape=(d, channels, x, y))
        for day in range(days):
            for slot in range(step_factor):
                idx = day * step_factor + slot
                frame = data[idx, ...]
                if args.fill_nan is not None:
                    frame = np.nan_to_num(
                        frame,
                        nan=args.fill_nan,
                        posinf=args.fill_nan,
                        neginf=args.fill_nan,
                    )
                sum_days[slot, day, ...] += frame
                count_days[slot, day] += 1

    for slot in range(4):
        for day in range(366):
            if count_days[slot, day] > 0:
                mean_day = sum_days[slot, day] / count_days[slot, day]
            else:
                mean_day = np.zeros((channels, x, y), dtype=np.float32)
            climatology[slot, day, ...] = mean_day.astype("float32")

    del climatology
    print(f"Wrote {era5_dir / args.output_name}")


if __name__ == "__main__":
    main()
