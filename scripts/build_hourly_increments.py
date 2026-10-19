"""Build verified hourly increments from existing raw analysis/background months."""

import argparse
import calendar
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aardvark"))
from grid_config import (
    background_input_path,
    era5_month_path,
    increment_month_path,
    load_grid_config,
    set_active_config,
    loader_grid_x_path,
    loader_grid_y_path,
)
from increment_data import CHANNEL_ORDER, build_increment
from month_manifest import read_month_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--months_file")
    p.add_argument("--months", nargs="+")
    p.add_argument("--era5_mode", default="rtma_ok_sfc")
    p.add_argument("--grid_config", default=str(REPO / "aardvark/grid_config_ok.yaml"))
    p.add_argument("--background_mode", choices=["hourly"], default="hourly")
    p.add_argument("--channel_order", nargs="+", default=CHANNEL_ORDER)
    p.add_argument("--allow_legacy_inputs", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--chunk_frames", type=int, default=24)
    args = p.parse_args()
    if bool(args.months_file) == bool(args.months):
        p.error("provide exactly one of --months_file or --months")
    months = (
        read_month_manifest(args.months_file, "--months_file")
        if args.months_file
        else [tuple(map(int, m.split("-"))) for m in args.months]
    )
    set_active_config(
        load_grid_config(None if args.grid_config == "default" else args.grid_config)
    )
    lon = np.load(loader_grid_x_path(args.data_root))
    lat = np.load(loader_grid_y_path(args.data_root))
    if np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0):
        raise ValueError("increment grid axes must ascend from SW origin")
    for year, month in months:
        shape = (
            calendar.monthrange(year, month)[1] * 24,
            len(args.channel_order),
            lon.size,
            lat.size,
        )
        output = increment_month_path(args.data_root, args.era5_mode, "1h", year, month)
        build_increment(
            era5_month_path(args.data_root, args.era5_mode, "1h", year, month),
            background_input_path(
                args.data_root, args.era5_mode, year, month, "hourly", "raw"
            ),
            output,
            shape,
            year,
            month,
            args.channel_order,
            args.allow_legacy_inputs,
            args.overwrite,
            args.chunk_frames,
        )
        print("[OK] " + output)


if __name__ == "__main__":
    main()
