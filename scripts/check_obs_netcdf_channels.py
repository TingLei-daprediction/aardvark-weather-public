"""
Report channel counts for observation NetCDFs used by convert_obs_training.py.

This inspects the source NetCDFs directly (not memmaps) and reports the
variable counts used to build channel stacks.
"""

import argparse
from pathlib import Path

import xarray as xr


def count_prefix_vars(ds, prefixes):
    return [n for n in ds.data_vars if n.startswith(prefixes)]


def report(name, path, prefixes=None, explicit_list=None):
    if not path.exists():
        print(f"[MISSING] {name}: {path}")
        return
    ds = xr.open_dataset(path)
    if explicit_list is not None:
        vars_used = [n for n in explicit_list if n in ds.data_vars]
        print(f"[OK] {name}: {len(vars_used)} channels (explicit list)")
        print(f"  vars: {vars_used}")
        return
    if prefixes is None:
        print(f"[OK] {name}: {len(ds.data_vars)} data_vars (no channel stacking)")
        print(f"  vars: {list(ds.data_vars)[:10]}{' ...' if len(ds.data_vars) > 10 else ''}")
        return
    vars_used = count_prefix_vars(ds, prefixes)
    print(f"[OK] {name}: {len(vars_used)} channels (prefixes={prefixes})")
    print(f"  vars: {vars_used[:10]}{' ...' if len(vars_used) > 10 else ''}")


def main():
    p = argparse.ArgumentParser(description="Check obs NetCDF channel counts")
    p.add_argument("--input_dir", default="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/aardvark-weather/training_data/"
                   , help="Dir containing *_data_v1.nc files")
    args = p.parse_args()
    inp = Path(args.input_dir)

    report("AMSU-A", inp / "amsua_data_v1.nc", prefixes=("bt_channel_",))
    report("AMSU-B/MHS", inp / "amsub_data_v1.nc", prefixes=("bt_channel_",))
    report(
        "ASCAT",
        inp / "ascat_data_v1.nc",
        explicit_list=[
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
        ],
    )
    report("HIRS", inp / "hirs_data_v1.nc", prefixes=("bt_channel_",))
    report("GRIDSAT", inp / "gridsat_data_v1.nc", prefixes=None)
    report(
        "IASI 2007-2013",
        inp / "iasi_data_2007_2013_v1.nc",
        prefixes=("b1_pc", "b2_pc", "b3_pc"),
    )
    report(
        "IASI 2014-2019",
        inp / "iasi_data_2014_2019_v1.nc",
        prefixes=("b1_pc", "b2_pc", "b3_pc"),
    )
    report("IGRA", inp / "igra_data_v1.nc", prefixes=None)
    report("ICOADS", inp / "icoads_data_v1.nc", prefixes=None)
    report("HADISD", inp / "hadisd_data_v1.nc", prefixes=None)


if __name__ == "__main__":
    main()
