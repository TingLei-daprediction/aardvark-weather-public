"""
Inspect ERA5 NetCDF variables in a given input directory and year.
"""

import argparse
import glob
import os
from pathlib import Path

import xarray as xr


def parse_args():
    p = argparse.ArgumentParser(description="Inspect ERA5 NetCDF variables")
    p.add_argument("--input_dir", required=True, help="Dir with ERA5 NetCDF files")
    p.add_argument("--year", type=int, required=True, help="Year to inspect")
    p.add_argument(
        "--pattern",
        default="era5_*_{year}_*.nc",
        help="Glob pattern; use {year} placeholder",
    )
    return p.parse_args()


def main():
    args = parse_args()
    glob_pat = args.pattern.format(year=args.year)
    files = sorted(glob.glob(os.path.join(args.input_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"No files found for {glob_pat} in {args.input_dir}")

    ds = xr.open_mfdataset(files, combine="by_coords")
    print("Files:")
    for f in files[:5]:
        print(f"  {f}")
    if len(files) > 5:
        print(f"  ... ({len(files)} files total)")
    print("Data variables:")
    for v in sorted(ds.data_vars):
        print(f"  {v} dims={ds[v].dims}")


if __name__ == "__main__":
    main()
