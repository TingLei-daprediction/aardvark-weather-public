"""
Inspect original observation NetCDFs and report time ranges and key dims.

Uses the same --input_dir interface as convert_obs_training.py.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args():
    p = argparse.ArgumentParser(description="Check observation NetCDF properties")
    p.add_argument("--input_dir", required=True, help="Dir containing *_data_v1.nc files")
    return p.parse_args()


def time_summary(ds):
    time_name = None
    for candidate in ("time", "valid_time"):
        if candidate in ds.coords:
            time_name = candidate
            break
    if time_name is None:
        return "time: N/A"
    t = ds[time_name].values
    if t.size == 0:
        return "time: empty"
    t0 = np.datetime_as_string(t[0], unit="s")
    t1 = np.datetime_as_string(t[-1], unit="s")
    if t.size >= 2:
        step = t[1] - t[0]
        return f"time: {t0} -> {t1}, step={step}"
    return f"time: {t0} -> {t1}, step=N/A"


def report(path):
    if not path.exists():
        print(f"[MISSING] {path.name}")
        return
    ds = xr.open_dataset(path)
    dims = ", ".join([f"{k}={v}" for k, v in ds.dims.items()])
    print(f"[OK] {path.name}")
    print(f"  dims: {dims}")
    print(f"  {time_summary(ds)}")


def main():
    args = parse_args()
    inp = Path(args.input_dir)
    files = [
        "amsua_data_v1.nc",
        "amsub_data_v1.nc",
        "ascat_data_v1.nc",
        "hirs_data_v1.nc",
        "gridsat_data_v1.nc",
        "iasi_data_2007_2013_v1.nc",
        "iasi_data_2014_2019_v1.nc",
        "igra_data_v1.nc",
        "icoads_data_v1.nc",
        "hadisd_data_v1.nc",
    ]
    for name in files:
        report(inp / name)


if __name__ == "__main__":
    main()
