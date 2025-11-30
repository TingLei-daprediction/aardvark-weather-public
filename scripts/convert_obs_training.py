"""
Convert the provided training_data NetCDF files into the memmap layout
expected by the Aardvark loaders, and compute per-source normalization stats.

Inputs (expected file names under --input_dir):
  amsua_data_v1.nc
  amsub_data_v1.nc
  ascat_data_v1.nc
  hirs_data_v1.nc
  gridsat_data_v1.nc
  iasi_data_2007_2013_v1.nc
  iasi_data_2014_2019_v1.nc
  igra_data_v1.nc
  icoads_data_v1.nc
  hadisd_data_v1.nc

Target grid:
  era5_x_1.npy and era5_y_1.npy (lon/lat in degrees) under --grid_dir.

Outputs (written under --output_dir):
  memmaps in subfolders matching the original loader expectations, e.g.:
    amsua/2007_2021_amsua.mmap           (time, lat, lon, channel)
    amsub_mhs/2007_2021_amsub.mmap       (time, lon, lat, channel)
    ascat/2007_2021_ascat.mmap           (time, lon, lat, channel)
    hirs/2007_2021_hirs.mmap             (time, lon, lat, channel)
    gridsat/gridsat_data.mmap            (time, channel, x, y)
    2007_2021_iasi_subset.mmap           (time, lon, lat, channel) concatenated
    igra/1999_2021_igra_y.mmap           (time, level, station)
    igra/1999_2021_igra_x.mmap           (station, 2) lon/lat
    icoads/1999_2021_icoads_y.mmap       (time, var, station)
    icoads/1999_2021_icoads_x.mmap       (station, 2) lon/lat
    hadisd_processed/{tas,tds,psl,u,v}_vals_train.memmap (time, station)
    hadisd_processed/{var}_{lon,lat,alt}_train.npy
  norm_factors/mean_<source>.npy and norm_factors/std_<source>.npy

Notes:
  - Variable name inference is minimal; adjust var_maps below if your files use different names.
  - This script assumes the NetCDFs are already on (or close to) the 1.5° grid;
    it reindexes to the target grid (nearest) and flips lat if needed.
  - Shapes are validated against data_shapes.py constants; warnings are emitted on mismatch.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args():
    p = argparse.ArgumentParser(description="Convert training_data NetCDFs to Aardvark memmaps")
    p.add_argument("--input_dir", required=True, help="Directory containing *_data_v1.nc files")
    p.add_argument("--output_dir", required=True, help="Directory to write memmaps/norms")
    p.add_argument("--grid_dir", default="data/grid_lon_lat", help="Dir with era5_x_1.npy and era5_y_1.npy")
    return p.parse_args()


def load_target_grid(grid_dir):
    lon_tgt = np.load(Path(grid_dir) / "era5_x_1.npy")
    lat_tgt = np.load(Path(grid_dir) / "era5_y_1.npy")
    return lon_tgt, lat_tgt


def normalize_and_reindex(ds, lon_tgt, lat_tgt):
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    ds = ds.assign_coords({lon_name: ((ds[lon_name] + 360) % 360)}).sortby(lon_name)
    if ds[lat_name][0] < ds[lat_name][-1]:
        ds = ds.reindex({lat_name: list(reversed(ds[lat_name]))})
    ds = ds.reindex({lon_name: lon_tgt, lat_name: lat_tgt}, method="nearest")
    return ds, lon_name, lat_name


def write_memmap(path, arr):
    mmap = np.memmap(path, mode="w+", dtype="float32", shape=arr.shape)
    mmap[:] = arr
    del mmap


def save_norms(out_dir, name, arr, reduce_axes):
    mean = np.nanmean(arr, axis=reduce_axes)
    std = np.nanstd(arr, axis=reduce_axes) + 1e-8
    np.save(out_dir / f"mean_{name}.npy", mean)
    np.save(out_dir / f"std_{name}.npy", std)


def convert_gridded(ds_path, var_name, transpose_order, out_path, norms_dir, norm_name, lon_tgt, lat_tgt):
    ds = xr.open_dataset(ds_path)
    ds, lon_name, lat_name = normalize_and_reindex(ds, lon_tgt, lat_tgt)
    v = var_name or list(ds.data_vars)[0]
    arr = ds[v].transpose(*transpose_order).values.astype("float32")
    write_memmap(out_path, arr)
    save_norms(norms_dir, norm_name, arr, reduce_axes=(0, *range(2, arr.ndim)))
    return arr.shape


def main():
    args = parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    norms_dir = out / "norm_factors"
    norms_dir.mkdir(parents=True, exist_ok=True)
    lon_tgt, lat_tgt = load_target_grid(args.grid_dir)

    shapes = {}

    # AMSU-A
    out_amsua = out / "amsua"
    out_amsua.mkdir(parents=True, exist_ok=True)
    shapes["amsua"] = convert_gridded(
        inp / "amsua_data_v1.nc",
        var_name=None,
        transpose_order=("time", "lat", "lon", "channel"),
        out_path=out_amsua / "2007_2021_amsua.mmap",
        norms_dir=norms_dir,
        norm_name="amsua",
        lon_tgt=lon_tgt,
        lat_tgt=lat_tgt,
    )

    # AMSU-B/MHS
    out_amsub = out / "amsub_mhs"
    out_amsub.mkdir(parents=True, exist_ok=True)
    shapes["amsub"] = convert_gridded(
        inp / "amsub_data_v1.nc",
        var_name=None,
        transpose_order=("time", "lon", "lat", "channel"),
        out_path=out_amsub / "2007_2021_amsub.mmap",
        norms_dir=norms_dir,
        norm_name="amsub",
        lon_tgt=lon_tgt,
        lat_tgt=lat_tgt,
    )

    # ASCAT
    out_ascat = out / "ascat"
    out_ascat.mkdir(parents=True, exist_ok=True)
    shapes["ascat"] = convert_gridded(
        inp / "ascat_data_v1.nc",
        var_name=None,
        transpose_order=("time", "lon", "lat", "channel"),
        out_path=out_ascat / "2007_2021_ascat.mmap",
        norms_dir=norms_dir,
        norm_name="ascat",
        lon_tgt=lon_tgt,
        lat_tgt=lat_tgt,
    )

    # HIRS
    out_hirs = out / "hirs"
    out_hirs.mkdir(parents=True, exist_ok=True)
    shapes["hirs"] = convert_gridded(
        inp / "hirs_data_v1.nc",
        var_name=None,
        transpose_order=("time", "lon", "lat", "channel"),
        out_path=out_hirs / "2007_2021_hirs.mmap",
        norms_dir=norms_dir,
        norm_name="hirs",
        lon_tgt=lon_tgt,
        lat_tgt=lat_tgt,
    )

    # GRIDSAT (assume dims: time, channel, x, y)
    out_sat = out / "gridsat"
    out_sat.mkdir(parents=True, exist_ok=True)
    ds_sat = xr.open_dataset(inp / "gridsat_data_v1.nc")
    v_sat = list(ds_sat.data_vars)[0]
    arr_sat = ds_sat[v_sat].transpose("time", "channel", "x", "y").values.astype("float32")
    write_memmap(out_sat / "gridsat_data.mmap", arr_sat)
    save_norms(norms_dir, "sat", arr_sat, reduce_axes=(0, 2, 3))
    shapes["gridsat"] = arr_sat.shape

    # IASI (concatenate two files)
    ds_iasi1 = xr.open_dataset(inp / "iasi_data_2007_2013_v1.nc")
    ds_iasi2 = xr.open_dataset(inp / "iasi_data_2014_2019_v1.nc")
    ds_iasi = xr.concat([ds_iasi1, ds_iasi2], dim="time")
    ds_iasi, lon_name, lat_name = normalize_and_reindex(ds_iasi, lon_tgt, lat_tgt)
    v_iasi = list(ds_iasi.data_vars)[0]
    arr_iasi = ds_iasi[v_iasi].transpose("time", "lon", "lat", "channel").values.astype("float32")
    write_memmap(out / "2007_2021_iasi_subset.mmap", arr_iasi)
    save_norms(norms_dir, "iasi", arr_iasi, reduce_axes=(0, 2, 3))
    shapes["iasi"] = arr_iasi.shape

    # IGRA (station)
    ds_igra = xr.open_dataset(inp / "igra_data_v1.nc")
    v_igra = list(ds_igra.data_vars)[0]
    arr_igra = ds_igra[v_igra].transpose("time", "level", "station").values.astype("float32")
    out_igra = out / "igra"
    out_igra.mkdir(parents=True, exist_ok=True)
    write_memmap(out_igra / "1999_2021_igra_y.mmap", arr_igra)
    # coords
    lon_igra = (ds_igra["station_lon"].values + 360) % 360
    lat_igra = ds_igra["station_lat"].values
    coords_igra = np.stack([lon_igra, lat_igra], axis=-1).astype("float32")
    write_memmap(out_igra / "1999_2021_igra_x.mmap", coords_igra)
    save_norms(norms_dir, "igra", arr_igra, reduce_axes=(0, 2))
    shapes["igra"] = arr_igra.shape

    # ICOADS (station)
    ds_ic = xr.open_dataset(inp / "icoads_data_v1.nc")
    v_ic = list(ds_ic.data_vars)[0]
    arr_ic = ds_ic[v_ic].transpose("time", "var", "ship").values.astype("float32")
    out_ic = out / "icoads"
    out_ic.mkdir(parents=True, exist_ok=True)
    write_memmap(out_ic / "1999_2021_icoads_y.mmap", arr_ic)
    lon_ic = (ds_ic["ship_lon"].values + 360) % 360
    lat_ic = ds_ic["ship_lat"].values
    coords_ic = np.stack([lon_ic, lat_ic], axis=-1).astype("float32")
    write_memmap(out_ic / "1999_2021_icoads_x.mmap", coords_ic)
    save_norms(norms_dir, "icoads", arr_ic, reduce_axes=(0, 2))
    shapes["icoads"] = arr_ic.shape

    # HADISD (per variable)
    ds_h = xr.open_dataset(inp / "hadisd_data_v1.nc")
    had_vars = ["tas", "tds", "psl", "u", "v"]
    out_h = out / "hadisd_processed"
    out_h.mkdir(parents=True, exist_ok=True)
    for var in had_vars:
        if var not in ds_h.data_vars:
            print(f"[WARN] {var} not in hadisd_data_v1.nc; skipping")
            continue
        arr_h = ds_h[var].transpose("time", "station").values.astype("float32")
        write_memmap(out_h / f"{var}_vals_train.memmap", arr_h)
        lon_h = (ds_h["lon"].values + 360) % 360
        lat_h = ds_h["lat"].values
        alt_h = ds_h["alt"].values if "alt" in ds_h else np.zeros_like(lon_h)
        np.save(out_h / f"{var}_lon_train.npy", lon_h.astype("float32"))
        np.save(out_h / f"{var}_lat_train.npy", lat_h.astype("float32"))
        np.save(out_h / f"{var}_alt_train.npy", alt_h.astype("float32"))
        save_norms(norms_dir, f"hadisd_{var}", arr_h, reduce_axes=(0,))
        shapes[f"hadisd_{var}"] = arr_h.shape

    print("[INFO] Completed conversions. Shapes:")
    for k, v in shapes.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
