#!/usr/bin/env python3
"""
Check ERA5 truth memmaps and norm files for NaN/Inf values.

Purpose:
- Verify the generated ERA5 truth memmaps do not contain NaNs/Inf.
- Identify which channels (if any) contain NaNs across all years.
- Report NaNs/Inf in mean/std norm files used for training.

Edit the paths and parameters in the CONFIG section before running.
"""

import glob
import os

import numpy as np


# CONFIG: update these paths for your environment.
DATA_DIR = "/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/era5"
NORM_DIR = "/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/norm_factors"
ERA5_MODE = "4u_sfc"
RES = 1
TIME_TAG = "1d"  # "1d" for daily, "6" for 6-hourly filenames
CHANNELS = 30
N_LON = 240
N_LAT = 121


def main():
    # Check norm files for NaNs/Inf.
    mean_path = os.path.join(NORM_DIR, f"mean_{ERA5_MODE}_{RES}.npy")
    std_path = os.path.join(NORM_DIR, f"std_{ERA5_MODE}_{RES}.npy")
    mean = np.load(mean_path)
    std = np.load(std_path)
    print(f"mean shape: {mean.shape} std shape: {std.shape}")
    print(f"mean nan: {np.isnan(mean).sum()} std nan: {np.isnan(std).sum()}")
    print(
        f"std zeros: {np.sum(std == 0)} std min: {np.nanmin(std)} std max: {np.nanmax(std)}"
    )

    # Scan all memmap files for NaNs/Inf per channel.
    pattern = os.path.join(DATA_DIR, f"era5_{ERA5_MODE}_{RES}_{TIME_TAG}_*.memmap")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No memmaps found for pattern: {pattern}")

    nan_counts = np.zeros(CHANNELS, dtype=np.int64)
    inf_counts = np.zeros(CHANNELS, dtype=np.int64)
    total_counts = np.zeros(CHANNELS, dtype=np.int64)

    for path in files:
        nbytes = os.path.getsize(path)
        frame = CHANNELS * N_LON * N_LAT * 4
        if nbytes % frame != 0:
            raise ValueError(f"File size not divisible by frame size: {path}")
        n_days = nbytes // frame

        arr = np.memmap(
            path, dtype="float32", mode="r", shape=(n_days, CHANNELS, N_LON, N_LAT)
        )
        nan_counts += np.isnan(arr).sum(axis=(0, 2, 3))
        inf_counts += np.isinf(arr).sum(axis=(0, 2, 3))
        total_counts += n_days * N_LON * N_LAT

    bad = np.where((nan_counts > 0) | (inf_counts > 0))[0]
    print("channels with NaN/Inf:", bad.tolist())
    for c in bad:
        print(
            f"ch {c}: nan={nan_counts[c]} inf={inf_counts[c]} total={total_counts[c]}"
        )


if __name__ == "__main__":
    main()
