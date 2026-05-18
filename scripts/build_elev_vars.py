"""
Build static elevation-related fields for Aardvark.

Standard (approximate) definition used here:
  1) Orography in meters (ERA5 geopotential / g)
  2) Land-sea mask
  3) sin(latitude)
  4) cos(latitude)

Output: era5/elev_vars_1.npy with shape (4, lat, lon)
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


G = 9.80665


def parse_args():
    p = argparse.ArgumentParser(description="Build elev_vars_1.npy from ERA5 static fields")
    p.add_argument("--input_file", required=True, help="NetCDF file with geopotential and land_sea_mask")
    p.add_argument("--output_dir", required=True, help="Base output dir (data_path)")
    p.add_argument("--grid_dir", default="data/grid_lon_lat", help="Dir with era5_x_1.npy and era5_y_1.npy")
    p.add_argument("--geopotential_var", default="geopotential", help="Variable name for geopotential")
    p.add_argument("--lsm_var", default="land_sea_mask", help="Variable name for land-sea mask")
    return p.parse_args()


def normalize_and_reindex(ds, lon_tgt, lat_tgt):
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    ds = ds.assign_coords({lon_name: ((ds[lon_name] + 360) % 360)}).sortby(lon_name)
    if ds[lat_name][0] < ds[lat_name][-1]:
        ds = ds.reindex({lat_name: list(reversed(ds[lat_name]))})
    ds = ds.reindex({lon_name: lon_tgt, lat_name: lat_tgt}, method="nearest")
    return ds, lon_name, lat_name


def main():
    args = parse_args()
    grid_dir = Path(args.grid_dir)
    lon_tgt = np.load(grid_dir / "era5_x_1.npy")
    lat_tgt = np.load(grid_dir / "era5_y_1.npy")

    ds = xr.open_dataset(args.input_file)
    ds, lon_name, lat_name = normalize_and_reindex(ds, lon_tgt, lat_tgt)

    # Allow common ERA5 short names (z, lsm) as fallbacks
    if args.geopotential_var not in ds.data_vars:
        if "z" in ds.data_vars:
            args.geopotential_var = "z"
        else:
            raise ValueError(f"Missing geopotential var: {args.geopotential_var}")
    if args.lsm_var not in ds.data_vars:
        if "lsm" in ds.data_vars:
            args.lsm_var = "lsm"
        else:
            raise ValueError(f"Missing land-sea mask var: {args.lsm_var}")

    z = ds[args.geopotential_var]
    lsm = ds[args.lsm_var]
    # Drop any non-lat/lon singleton dims (e.g., valid_time, number)
    for dim in list(z.dims):
        if dim not in (lat_name, lon_name):
            z = z.isel({dim: 0})
    for dim in list(lsm.dims):
        if dim not in (lat_name, lon_name):
            lsm = lsm.isel({dim: 0})

    orog = (z.transpose(lat_name, lon_name).values.astype("float32")) / G
    lsm = lsm.transpose(lat_name, lon_name).values.astype("float32")

    lat_rad = np.deg2rad(lat_tgt).astype("float32")
    sin_lat = np.sin(lat_rad)[:, None] * np.ones((1, lon_tgt.shape[0]), dtype="float32")
    cos_lat = np.cos(lat_rad)[:, None] * np.ones((1, lon_tgt.shape[0]), dtype="float32")

    elev = np.stack([orog, lsm, sin_lat, cos_lat], axis=0)

    out_dir = Path(args.output_dir) / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "elev_vars_1.npy", elev)
    print(f"Wrote {out_dir / 'elev_vars_1.npy'} with shape {elev.shape}")


if __name__ == "__main__":
    main()
