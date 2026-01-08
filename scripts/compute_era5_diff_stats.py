"""
Compute ERA5 diff-mode stats from existing memmaps.

Outputs (under output_dir):
  - era5_spatial_means.npy        (1, channels, lon, lat)
  - era5_avdiff_means.npy         (channels,)
  - era5_avdiff_stds.npy          (channels,)

These are used when --diff 1 in the assimilation loader.
"""

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compute ERA5 diff-mode stats")
    p.add_argument("--data_dir", required=True, help="Root containing era5/ memmaps")
    p.add_argument("--output_dir", required=True, help="Dir to write stats (aux_data_path)")
    p.add_argument("--era5_mode", default="4u_sfc")
    p.add_argument("--res", type=int, default=1)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--grid_dir", required=True, help="Dir with era5_x_<res>.npy and era5_y_<res>.npy")
    p.add_argument("--chunk", type=int, default=64, help="Time chunk size")
    return p.parse_args()


def year_steps(year):
    return (366 if year % 4 == 0 else 365) * 4


def infer_channels(memmap_path, t, lon, lat):
    nbytes = os.path.getsize(memmap_path)
    denom = t * lon * lat * 4
    if nbytes % denom != 0:
        raise ValueError(f"File size not divisible by expected frame size: {memmap_path}")
    return nbytes // denom


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lon = np.load(Path(args.grid_dir) / f"era5_x_{args.res}.npy")
    lat = np.load(Path(args.grid_dir) / f"era5_y_{args.res}.npy")
    lon_len = lon.shape[0]
    lat_len = lat.shape[0]

    sum_fields = None
    count_t = 0

    for year in args.years:
        t = year_steps(year)
        memmap_path = data_dir / "era5" / f"era5_{args.era5_mode}_{args.res}_6_{year}.memmap"
        if not memmap_path.exists():
            print(f"[WARN] Missing memmap: {memmap_path}, skipping.")
            continue

        c = infer_channels(memmap_path, t, lon_len, lat_len)
        mmap = np.memmap(memmap_path, mode="r", dtype="float32", shape=(t, c, lon_len, lat_len))

        if sum_fields is None:
            sum_fields = np.zeros((c, lon_len, lat_len), dtype=np.float64)

        for i in range(0, t, args.chunk):
            j = min(i + args.chunk, t)
            chunk = mmap[i:j]
            sum_fields += chunk.sum(axis=0)
            count_t += chunk.shape[0]

    if count_t == 0:
        raise RuntimeError("No data processed; check paths/years.")

    spatial_means = sum_fields / count_t
    np.save(out_dir / "era5_spatial_means.npy", spatial_means[np.newaxis, ...])

    sum_channels = np.zeros(spatial_means.shape[0], dtype=np.float64)
    sumsq_channels = np.zeros(spatial_means.shape[0], dtype=np.float64)
    count = 0

    for year in args.years:
        t = year_steps(year)
        memmap_path = data_dir / "era5" / f"era5_{args.era5_mode}_{args.res}_6_{year}.memmap"
        if not memmap_path.exists():
            continue

        c = spatial_means.shape[0]
        mmap = np.memmap(memmap_path, mode="r", dtype="float32", shape=(t, c, lon_len, lat_len))

        for i in range(0, t, args.chunk):
            j = min(i + args.chunk, t)
            chunk = mmap[i:j]
            diff = chunk - spatial_means
            sum_channels += diff.sum(axis=(0, 2, 3))
            sumsq_channels += np.square(diff, dtype=np.float64).sum(axis=(0, 2, 3))
            count += diff.shape[0] * lon_len * lat_len

    mean = sum_channels / count
    var = sumsq_channels / count - np.square(mean)
    std = np.sqrt(np.clip(var, 0, None)) + 1e-8

    np.save(out_dir / "era5_avdiff_means.npy", mean)
    np.save(out_dir / "era5_avdiff_stds.npy", std)
    print("[INFO] Saved era5_spatial_means.npy, era5_avdiff_means.npy, era5_avdiff_stds.npy")


if __name__ == "__main__":
    main()
