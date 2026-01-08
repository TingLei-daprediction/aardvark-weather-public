"""
Compute mean/std for ERA5 memmaps already written to data_path/era5.

Expected memmap layout:
  era5/era5_<mode>_<res>_{6|1d}_<year>.memmap with shape (time, channels, lon, lat)

Outputs:
  <output_dir>/norm_factors/mean_<mode>_<res>.npy
  <output_dir>/norm_factors/std_<mode>_<res>.npy
"""

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compute mean/std from ERA5 memmaps")
    p.add_argument("--data_dir", required=True, help="Root containing era5/ memmaps")
    p.add_argument("--output_dir", required=True, help="Root for norm_factors outputs")
    p.add_argument("--era5_mode", default="4u_sfc")
    p.add_argument("--res", type=int, default=1)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--grid_dir", required=True, help="Dir with era5_x_<res>.npy and era5_y_<res>.npy")
    p.add_argument("--time_freq", default="6H", help="6H or 1D")
    p.add_argument("--chunk", type=int, default=64, help="Time chunk size")
    return p.parse_args()


def year_steps(year, time_freq):
    factor = 4 if time_freq == "6H" else 1
    return (366 if year % 4 == 0 else 365) * factor


def infer_channels(memmap_path, t, lon, lat):
    nbytes = os.path.getsize(memmap_path)
    denom = t * lon * lat * 4
    if nbytes % denom != 0:
        raise ValueError(f"File size not divisible by expected frame size: {memmap_path}")
    return nbytes // denom


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / "norm_factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    lon = np.load(Path(args.grid_dir) / f"era5_x_{args.res}.npy")
    lat = np.load(Path(args.grid_dir) / f"era5_y_{args.res}.npy")
    lon_len = lon.shape[0]
    lat_len = lat.shape[0]

    sum_channels = None
    sumsq_channels = None
    count = 0

    for year in args.years:
        t = year_steps(year, args.time_freq)
        freq_tag = "6" if args.time_freq == "6H" else "1d"
        memmap_path = data_dir / "era5" / f"era5_{args.era5_mode}_{args.res}_{freq_tag}_{year}.memmap"
        if not memmap_path.exists():
            print(f"[WARN] Missing memmap: {memmap_path}, skipping.")
            continue

        c = infer_channels(memmap_path, t, lon_len, lat_len)
        mmap = np.memmap(memmap_path, mode="r", dtype="float32", shape=(t, c, lon_len, lat_len))

        if sum_channels is None:
            sum_channels = np.zeros((c,), dtype=np.float64)
            sumsq_channels = np.zeros((c,), dtype=np.float64)

        for i in range(0, t, args.chunk):
            j = min(i + args.chunk, t)
            chunk = mmap[i:j]  # (chunk, c, lon, lat)
            sum_channels += chunk.sum(axis=(0, 2, 3))
            sumsq_channels += np.square(chunk, dtype=np.float64).sum(axis=(0, 2, 3))
            count += chunk.shape[0] * lon_len * lat_len

    if count == 0:
        raise RuntimeError("No data processed; check paths/years.")

    mean = sum_channels / count
    var = sumsq_channels / count - np.square(mean)
    std = np.sqrt(np.clip(var, 0, None)) + 1e-8

    mean_path = out_dir / f"mean_{args.era5_mode}_{args.res}.npy"
    std_path = out_dir / f"std_{args.era5_mode}_{args.res}.npy"
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"[INFO] Saved norms: {mean_path}, {std_path}")


if __name__ == "__main__":
    main()
