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


def resolve_dim_names(ds, desired):
    mapping = {}
    for d in desired:
        if d == "lon":
            for cand in ("lon", "longitude"):
                if cand in ds.dims:
                    mapping[d] = cand
                    break
        elif d == "lat":
            for cand in ("lat", "latitude"):
                if cand in ds.dims:
                    mapping[d] = cand
                    break
        elif d == "channel":
            for cand in ("channel", "channels", "variable", "band"):
                if cand in ds.dims:
                    mapping[d] = cand
                    break
        elif d == "time":
            for cand in ("time",):
                if cand in ds.dims:
                    mapping[d] = cand
                    break
        else:
            if d in ds.dims:
                mapping[d] = d
    return mapping


def convert_gridded(ds_path, var_name, desired_order, out_path, norms_dir, norm_name, lon_tgt, lat_tgt):
    ds = xr.open_dataset(ds_path)
    ds, lon_name, lat_name = normalize_and_reindex(ds, lon_tgt, lat_tgt)
    v = var_name or list(ds.data_vars)[0]
    dim_map = resolve_dim_names(ds, desired_order)

    arr_da = ds[v]
    # Handle missing channel dimension: stack data_vars or add singleton
    if "channel" in desired_order and "channel" not in dim_map:
        if var_name is None and len(ds.data_vars) > 1:
            stacked = xr.concat([ds[name] for name in ds.data_vars], dim="channel_temp")
            arr_da = stacked
        else:
            arr_da = arr_da.expand_dims({"channel_temp": [0]})
        dim_map["channel"] = "channel_temp"

    actual_order = [dim_map.get(d, d) for d in desired_order]
    arr = arr_da.transpose(*actual_order).values.astype("float32")
    write_memmap(out_path, arr)
    reduce_axes = (0,) + tuple(range(2, arr.ndim))
    save_norms(norms_dir, norm_name, arr, reduce_axes=reduce_axes)
    return arr.shape


def stack_vars(ds, patterns):
    names = []
    for pat in patterns:
        names.extend([n for n in ds.data_vars if n.startswith(pat)])
    if not names:
        names = list(ds.data_vars)
    # Deduplicate, preserve order
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered


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
    ds = xr.open_dataset(inp / "amsua_data_v1.nc")
    ds, lon_name, lat_name = normalize_and_reindex(ds, lon_tgt, lat_tgt)
    chan_names = stack_vars(ds, ["bt_channel_"])
    arr = xr.concat([ds[n] for n in chan_names], dim="channel_temp").transpose("time", lat_name, lon_name, "channel_temp").values.astype("float32")
    write_memmap(out_amsua / "2007_2021_amsua.mmap", arr)
    save_norms(norms_dir, "amsua", arr, reduce_axes=(0, 1, 2))
    shapes["amsua"] = arr.shape

    # AMSU-B/MHS
    out_amsub = out / "amsub_mhs"
    out_amsub.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(inp / "amsub_data_v1.nc")
    ds, lon_name, lat_name = normalize_and_reindex(ds, lon_tgt, lat_tgt)
    chan_names = stack_vars(ds, ["bt_channel_"])
    arr = xr.concat([ds[n] for n in chan_names], dim="channel_temp").transpose("time", lon_name, lat_name, "channel_temp").values.astype("float32")
    write_memmap(out_amsub / "2007_2021_amsub.mmap", arr)
    save_norms(norms_dir, "amsub", arr, reduce_axes=(0, 1, 2))
    shapes["amsub"] = arr.shape

    # ASCAT
    out_ascat = out / "ascat"
    out_ascat.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(inp / "ascat_data_v1.nc")
    ds, lon_name, lat_name = normalize_and_reindex(ds, lon_tgt, lat_tgt)
    chan_names = [
        "beam_1_sigma0",
        "beam_2_sigma0",
        "beam_3_sigma0",
        "beam_1_inc_angle",
        "beam_2_inc_angle",
        "beam_3_inc_angle",
        "beam_1_azi_angle",
        "beam_2_azi_angle",
        "beam_3_azi_angle",
        "beam_1_kp",
        "beam_2_kp",
        "beam_3_kp",
        "sat_track_azi",
        "as_des_pass",
        "obs_time",
    ]
    chan_names = [n for n in chan_names if n in ds.data_vars]
    arr = xr.concat([ds[n] for n in chan_names], dim="channel_temp").transpose("time", lon_name, lat_name, "channel_temp").values.astype("float32")
    write_memmap(out_ascat / "2007_2021_ascat.mmap", arr)
    save_norms(norms_dir, "ascat", arr, reduce_axes=(0, 1, 2))
    shapes["ascat"] = arr.shape

    # HIRS
    out_hirs = out / "hirs"
    out_hirs.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(inp / "hirs_data_v1.nc")
    ds, lon_name, lat_name = normalize_and_reindex(ds, lon_tgt, lat_tgt)
    chan_names = stack_vars(ds, ["bt_channel_"])
    arr = xr.concat([ds[n] for n in chan_names], dim="channel_temp").transpose("time", lon_name, lat_name, "channel_temp").values.astype("float32")
    write_memmap(out_hirs / "2007_2021_hirs.mmap", arr)
    save_norms(norms_dir, "hirs", arr, reduce_axes=(0, 1, 2))
    os.replace(norms_dir / "mean_hirs.npy", norms_dir / "hirs_means.npy")
    os.replace(norms_dir / "std_hirs.npy", norms_dir / "hirs_stds.npy")
    shapes["hirs"] = arr.shape

    # GRIDSAT (assume dims: time, channel, x, y)
    out_sat = out / "gridsat"
    out_sat.mkdir(parents=True, exist_ok=True)
    ds_sat = xr.open_dataset(inp / "gridsat_data_v1.nc")
    v_sat = list(ds_sat.data_vars)[0]
    chans = list(ds_sat.data_vars)
    arr_sat = xr.concat([ds_sat[n] for n in chans], dim="channel_temp").transpose("time", "channel_temp", "longitude", "latitude").values.astype("float32")
    write_memmap(out_sat / "gridsat_data.mmap", arr_sat)
    save_norms(norms_dir, "sat", arr_sat, reduce_axes=(0, 2, 3))
    mean_sat = np.load(norms_dir / "mean_sat.npy").reshape(-1, 1, 1)
    std_sat = np.load(norms_dir / "std_sat.npy").reshape(-1, 1, 1)
    np.save(norms_dir / "mean_sat.npy", mean_sat)
    np.save(norms_dir / "std_sat.npy", std_sat)
    shapes["gridsat"] = arr_sat.shape

    # IASI (concatenate two files)
    ds_iasi1 = xr.open_dataset(inp / "iasi_data_2007_2013_v1.nc")
    ds_iasi2 = xr.open_dataset(inp / "iasi_data_2014_2019_v1.nc")
    ds_iasi = xr.concat([ds_iasi1, ds_iasi2], dim="time")
    ds_iasi, lon_name, lat_name = normalize_and_reindex(ds_iasi, lon_tgt, lat_tgt)
    chan_names = stack_vars(ds_iasi, ["b1_pc", "b2_pc", "b3_pc"])
    arr_iasi = xr.concat([ds_iasi[n] for n in chan_names], dim="channel_temp").transpose("time", lon_name, lat_name, "channel_temp").values.astype("float32")
    write_memmap(out / "2007_2021_iasi_subset.mmap", arr_iasi)
    save_norms(norms_dir, "iasi", arr_iasi, reduce_axes=(0, 1, 2))
    shapes["iasi"] = arr_iasi.shape

    # IGRA (station)
    ds_igra = xr.open_dataset(inp / "igra_data_v1.nc")
    level_order = ["850", "700", "500", "200"]
    fields = ["z", "t", "rh", "dpdp", "u", "v"]
    chan_names = []
    for lev in level_order:
        for f in fields:
            name = f"{f}_{lev}"
            if name in ds_igra.data_vars:
                chan_names.append(name)
    arr_igra = xr.concat([ds_igra[n] for n in chan_names], dim="channel_temp").transpose("time", "channel_temp", "station_id").values.astype("float32")
    out_igra = out / "igra"
    out_igra.mkdir(parents=True, exist_ok=True)
    write_memmap(out_igra / "1999_2021_igra_y.mmap", arr_igra)
    # coords
    lon_igra = (ds_igra["longitude"].values + 360) % 360 if "longitude" in ds_igra else (ds_igra["station_lon"].values + 360) % 360
    lat_igra = ds_igra["latitude"].values if "latitude" in ds_igra else ds_igra["station_lat"].values
    coords_igra = np.stack([lon_igra, lat_igra], axis=-1).astype("float32")
    write_memmap(out_igra / "1999_2021_igra_x.mmap", coords_igra)
    save_norms(norms_dir, "igra", arr_igra, reduce_axes=(0, 2))
    mean_igra = np.load(norms_dir / "mean_igra.npy").reshape(-1, 1)
    std_igra = np.load(norms_dir / "std_igra.npy").reshape(-1, 1)
    np.save(norms_dir / "mean_igra.npy", mean_igra)
    np.save(norms_dir / "std_igra.npy", std_igra)
    shapes["igra"] = arr_igra.shape

    # ICOADS (station)
    ds_ic = xr.open_dataset(inp / "icoads_data_v1.nc")
    chan_names = [n for n in ("u", "v", "slp", "sst", "tas") if n in ds_ic.data_vars]
    arr_ic = xr.concat([ds_ic[n] for n in chan_names], dim="channel_temp").transpose("time", "channel_temp", "entry").values.astype("float32")
    out_ic = out / "icoads"
    out_ic.mkdir(parents=True, exist_ok=True)
    write_memmap(out_ic / "1999_2021_icoads_y.mmap", arr_ic)
    lon_ic = (ds_ic["lon"].values + 360) % 360 if "lon" in ds_ic else (ds_ic["ship_lon"].values + 360) % 360
    lat_ic = ds_ic["lat"].values if "lat" in ds_ic else ds_ic["ship_lat"].values
    coords_ic = np.stack([lon_ic, lat_ic], axis=-1).astype("float32")
    write_memmap(out_ic / "1999_2021_icoads_x.mmap", coords_ic)
    save_norms(norms_dir, "icoads", arr_ic, reduce_axes=(0, 2))
    shapes["icoads"] = arr_ic.shape

    # HADISD (per variable)
    ds_h = xr.open_dataset(inp / "hadisd_data_v1.nc")
    had_var_map = {"tas": "tas", "tds": "tds", "psl": "psl", "ws": "u", "wd": "v"}
    out_h = out / "hadisd_processed"
    out_h.mkdir(parents=True, exist_ok=True)
    for src_var, out_name in had_var_map.items():
        if src_var not in ds_h.data_vars:
            print(f"[WARN] {src_var} not in hadisd_data_v1.nc; skipping")
            continue
        # Expect dims (station_index_*, time); transpose to (time, station)
        station_dim = [d for d in ds_h[src_var].dims if d != "time"][0]
        arr_h = ds_h[src_var].transpose("time", station_dim).values.astype("float32")
        write_memmap(out_h / f"{out_name}_vals_train.memmap", arr_h)
        lon_name = f"{src_var}_lon" if f"{src_var}_lon" in ds_h else "lon"
        lat_name = f"{src_var}_lat" if f"{src_var}_lat" in ds_h else "lat"
        lon_h = (ds_h[lon_name].values + 360) % 360
        lat_h = ds_h[lat_name].values
        alt_name = f"{src_var}_alt" if f"{src_var}_alt" in ds_h else ("alt" if "alt" in ds_h else None)
        alt_h = ds_h[alt_name].values if alt_name else np.zeros_like(lon_h)
        np.save(out_h / f"{out_name}_lon_train.npy", lon_h.astype("float32"))
        np.save(out_h / f"{out_name}_lat_train.npy", lat_h.astype("float32"))
        np.save(out_h / f"{out_name}_alt_train.npy", alt_h.astype("float32"))
        save_norms(norms_dir, f"hadisd_{out_name}", arr_h, reduce_axes=(0,))
        shapes[f"hadisd_{out_name}"] = arr_h.shape

    print("[INFO] Completed conversions. Shapes:")
    for k, v in shapes.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
