"""
Prepare ERA5 single-level fields on the target Aardvark grid and write memmaps.

Inputs:
  - Monthly ERA5 NetCDF files (e.g., era5_single_1p5deg_YYYY_MM.nc) under an input directory.
  - Target grid files: data/grid_lon_lat/era5_x_1.npy and era5_y_1.npy (lon/lat in degrees).

Outputs (per year):
  - <output_dir>/era5/era5_4u_1_6_<year>.memmap  (or other mode name via --era5_mode)
  - <output_dir>/norm_factors/mean_<mode>_1.npy and std_<mode>_1.npy (per-channel mean/std)

Notes:
  - This script assumes 6-hourly data and uses the variables listed via --variables in the given order.
  - Reindexes each monthly file to the target grid (nearest) after normalizing lon to 0–360 and flipping lat to 90→-90.
  - Channels are stacked as given in --variables; ensure this matches what your training expects.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args():
    p = argparse.ArgumentParser(description="Prepare ERA5 memmaps on target grid")
    p.add_argument("--input_dir", required=True, help="Dir with monthly ERA5 NetCDF files")
    p.add_argument("--output_dir", required=True, help="Base output dir for memmaps/norms")
    p.add_argument(
        "--era5_mode",
        default="4u",
        help="Mode name for output files (e.g., sfc, 4u, 13u)",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        required=True,
        help="Variable names in desired channel order (must match NetCDF names)",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Years to process (e.g., 2007 2008 ...)",
    )
    p.add_argument(
        "--pattern",
        default="era5_*_{year}_*.nc",
        help="Glob pattern for files in input_dir; use {year} placeholder (default: era5_*_{year}_*.nc)",
    )
    p.add_argument(
        "--grid_dir",
        default="../data/grid_lon_lat",
        help="Directory containing era5_x_1.npy and era5_y_1.npy",
    )
    return p.parse_args()


def ensure_dirs(base_out):
    memmap_dir = Path(base_out) / "era5"
    norms_dir = Path(base_out) / "norm_factors"
    memmap_dir.mkdir(parents=True, exist_ok=True)
    norms_dir.mkdir(parents=True, exist_ok=True)
    return memmap_dir, norms_dir


def load_target_grid(grid_dir):
    lon_tgt = np.load(Path(grid_dir) / "era5_x_1.npy")
    lat_tgt = np.load(Path(grid_dir) / "era5_y_1.npy")
    return lon_tgt, lat_tgt


def normalize_and_reindex(ds, lon_tgt, lat_tgt):
    # Normalize lon to 0–360 and sort
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    time_name = "time" if "time" in ds.coords else "valid_time"
    ds = ds.assign_coords({lon_name: ((ds[lon_name] + 360) % 360)}).sortby(lon_name)
    # Flip lat to 90 -> -90 if needed
    if ds[lat_name][0] < ds[lat_name][-1]:
        ds = ds.reindex({lat_name: list(reversed(ds[lat_name]))})
    # Reindex to target grid (nearest neighbor, no fill)
    ds = ds.reindex({lon_name: lon_tgt, lat_name: lat_tgt}, method="nearest")
    return ds, time_name, lon_name, lat_name


def write_memmap(year, arr, memmap_path):
    mmap = np.memmap(memmap_path, mode="w+", dtype="float32", shape=arr.shape)
    mmap[:] = arr
    del mmap


def main():
    args = parse_args()
    memmap_dir, norms_dir = ensure_dirs(args.output_dir)
    lon_tgt, lat_tgt = load_target_grid(args.grid_dir)

    name_map = {
        "2m_temperature": "t2m",
        "2m_dewpoint_temperature": "d2m",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "mean_sea_level_pressure": "msl",
        "surface_pressure": "sp",
        "geopotential": "z",
        "temperature": "t",
        "specific_humidity": "q",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
    }

    # Accumulate for mean/std across all years
    sum_channels = None
    sumsq_channels = None
    count = 0

    for year in args.years:
        glob_pat = args.pattern.format(year=year)
        files = sorted(glob.glob(os.path.join(args.input_dir, glob_pat)))
        if not files:
            print(f"[WARN] No files for year {year}, skipping.")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords")
        ds, time_name, lon_name, lat_name = normalize_and_reindex(ds, lon_tgt, lat_tgt)

        # Stack variables in the given order, flattening levels (if present) into channels
        channel_arrays = []
        for v in args.variables:
            v_in = name_map.get(v, v)
            if v_in not in ds.data_vars:
                raise ValueError(
                    f"Variable {v} (mapped to {v_in}) not found in dataset for year {year}"
                )
            da = ds[v_in]
            if "pressure_level" in da.dims:
                da = da.transpose(time_name, "pressure_level", lat_name, lon_name)
                arr_v = da.values.astype("float32")  # (time, level, lat, lon)
                # move to (time, level, lon, lat) and treat each level as a channel
                arr_v = np.transpose(arr_v, (0, 1, 3, 2))
                # reshape to (time, channels, lon, lat)
                arr_v = arr_v.reshape(arr_v.shape[0], arr_v.shape[1], arr_v.shape[2], arr_v.shape[3])
            else:
                da = da.transpose(time_name, lat_name, lon_name)
                arr_v = da.values.astype("float32")  # (time, lat, lon)
                arr_v = np.transpose(arr_v, (0, 2, 1))  # (time, lon, lat)
                arr_v = arr_v[:, np.newaxis, ...]      # add channel dim
            channel_arrays.append(arr_v)

        # Concatenate all channels: resulting shape (time, channels, lon, lat)
        arr = np.concatenate(channel_arrays, axis=1)

        memmap_path = memmap_dir / f"era5_{args.era5_mode}_1_6_{year}.memmap"
        print(f"[INFO] Writing {memmap_path} with shape {arr.shape}")
        write_memmap(year, arr, memmap_path)

        # Update running sums for mean/std (channel-wise)
        if sum_channels is None:
            sum_channels = np.zeros(arr.shape[1:], dtype=np.float64)
            sumsq_channels = np.zeros(arr.shape[1:], dtype=np.float64)
        sum_channels += arr.sum(axis=0)
        sumsq_channels += np.square(arr, dtype=np.float64).sum(axis=0)
        count += arr.shape[0]

    if count == 0:
        print("[WARN] No data processed; exiting without norms.")
        return

    # Compute mean/std per channel (over time, lon, lat)
    mean = sum_channels / count
    var = sumsq_channels / count - np.square(mean)
    std = np.sqrt(np.clip(var, 0, None)) + 1e-8

    mean_path = norms_dir / f"mean_{args.era5_mode}_1.npy"
    std_path = norms_dir / f"std_{args.era5_mode}_1.npy"
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"[INFO] Saved norms: {mean_path}, {std_path}")


if __name__ == "__main__":
    main()
