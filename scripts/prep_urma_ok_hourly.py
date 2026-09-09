"""
Build hourly RTMA-OK targets and daily-00z or hourly backgrounds from URMA grib2 files.

Inputs: URMA 2.5 km files ALREADY interpolated to the target OK lat-lon grid (no
interpolation is done here), organized one directory per day:

  <input_dir>/<YYYYMMDD>/urma2p5.t<HH>z.2dvaranl_OK.grb2   (hourly analysis -> TARGET)
  <input_dir>/<YYYYMMDD>/urma2p5.t00z.2dvarges_OK.grb2     (00z first guess -> BACKGROUND)
  <input_dir>/<YYYYMMDD>/urma2p5.t<HH>z.2dvarges_OK.grb2   (hourly-mode BACKGROUND)

Each grib2 file carries the 5 surface fields on a regular lat-lon grid scanned WE:SN
(lon fastest, south->north), e.g. 331 x 171 for the OK box.

Outputs (the loader's per-month contract for time_freq=1H, see docs/plan_rtma_15min.md):

  <output_dir>/era5/era5_rtma_ok_sfc_1_1h_<YYYY>-<MM>.memmap
      (days_in_month * 24, 5, nlon, nlat) float32, RAW values
      (WeatherDataset._load_era5_monthly normalizes the target at runtime)

  <output_dir>/era5/background_raw_rtma_ok_sfc_1_<YYYY>-<MM>.memmap
      (days_in_month, 5, nlon, nlat) float32, RAW values -- one 00z frame per day.
      NOT read by the loader directly: run scripts/normalize_background.py afterwards to
      produce background_rtma_ok_sfc_1_<YYYY>-<MM>.memmap (normalized with target mean/std).
      The background name carries NO freq tag: one 00z frame/day is cadence-independent.

  <output_dir>/era5/background_raw_hourly_rtma_ok_sfc_1_<YYYY>-<MM>.memmap
      (days_in_month * 24, 5, nlon, nlat) float32, RAW hour-matched first guesses.
      This is written only with ``--background_mode hourly`` and normalized separately to
      ``background_hourly_rtma_ok_sfc_1_<YYYY>-<MM>.memmap``.

  <output_dir>/norm_factors/mean_rtma_ok_sfc_1.npy, std_rtma_ok_sfc_1.npy   (--write_norms)
      Per-channel mean/std over all processed TARGET frames. Only pass --write_norms when
      the processed months are the intended TRAINING period (norms must not include val data
      if you want a clean split, and must exist before normalize_background.py can run).

Channel order (identical to the obs order in loader.py hadisd_vars for rtma_surface):
  0 tas  TMP  2 m above ground      [K]
  1 sh   SPFH 2 m above ground      [kg/kg]
  2 psl  PRES surface               [Pa]
  3 u    UGRD 10 m above ground     [m/s]
  4 v    VGRD 10 m above ground     [m/s]

Grid axes: by default (--grid_source grib) the 1-D lon/lat axes are DERIVED from the first
grib file's metadata; era5_x_<tag>.npy / era5_y_<tag>.npy are then written to --grid_dir if
absent (training reads them too -- copy BOTH to model_data_path/grid_lon_lat/), or, if present,
cross-checked (abort on mismatch, never silently contradicted). --grid_source files requires
pre-built axes (build_grid_lonlat.py). Either way every subsequent grib message is still
cross-checked against the axes, so all files are guaranteed to share one grid.
Orientation: lon ascending on the 0-360 branch, lat ascending south->north, arrays
(.., nlon, nlat).

Whole months only: the loader hard-asserts frames == days_in_month * 24, so a partial month
is useless. Every (day, hour) file is checked to exist BEFORE any writing starts; missing
files are all listed at once. --fill_missing persist copies the previous available frame
instead (use only for isolated archive gaps -- every fill is logged).

Usage (Jan-Jun 2022, write norms from these months). Default --grid_source grib needs ONLY the
grib archive; the grid dimensions come from the first grib file and era5_x/y_ok.npy are written
as a side product (no interpolation anywhere -- values are copied as-is):
  python scripts/prep_urma_ok_hourly.py \
      --input_dir /path/urma_ok --output_dir /path/aardvark-data-ok \
      --start 2022-01 --end 2022-06 --write_norms
With pre-built axes (build_grid_lonlat.py) instead:
  python scripts/prep_urma_ok_hourly.py ... --grid_source files --grid_dir /path/grids --grid_tag ok
Then normalize the background (needs the norm factors from --write_norms):
  python scripts/normalize_background.py --data_dir /path/aardvark-data-ok \
      --era5_mode rtma_ok_sfc --months 2022-01 2022-02 2022-03 2022-04 2022-05 2022-06
Other options: --fill_missing persist (fill isolated gaps from the previous frame; default is
abort listing all missing files), --skip_time_check, --anl_pattern / --ges_pattern.
Use --background_only --grid_source files to add backgrounds without rewriting existing targets.
"""

import argparse
import calendar
import os
from pathlib import Path

import numpy as np
import eccodes as ec

ERA5_MODE = "rtma_ok_sfc"
FREQ_TAG = "1h"
FRAMES_PER_DAY = 24

# (name, discipline, parameterCategory, parameterNumber, typeOfLevel, level) -- GRIB2-native
# identification, deliberately not shortName (table-dependent). Order = channel order.
VAR_SPECS = [
    ("tas", 0, 0, 0, "heightAboveGround", 2),  # TMP 2 m
    ("sh", 0, 1, 0, "heightAboveGround", 2),  # SPFH 2 m
    ("psl", 0, 3, 0, "surface", 0),  # PRES surface
    ("u", 0, 2, 2, "heightAboveGround", 10),  # UGRD 10 m
    ("v", 0, 2, 3, "heightAboveGround", 10),  # VGRD 10 m
]
CHANNELS = len(VAR_SPECS)


def parse_args():
    p = argparse.ArgumentParser(
        description="URMA grib2 -> per-month hourly target + selectable background memmaps (RTMA OK)"
    )
    p.add_argument("--input_dir", required=True, help="Root dir containing <YYYYMMDD>/ day dirs")
    p.add_argument("--output_dir", required=True, help="Base data_path (era5/ and norm_factors/ under it)")
    p.add_argument("--start", required=True, help="First month, YYYY-MM")
    p.add_argument("--end", required=True, help="Last month (inclusive), YYYY-MM")
    p.add_argument(
        "--grid_source",
        choices=["grib", "files"],
        default="grib",
        help="grib (default): derive the axes from the first grib file and write "
        "era5_x/y_<tag>.npy if absent (cross-check if present); files: require existing axes",
    )
    p.add_argument(
        "--grid_dir",
        default=None,
        help="Dir for era5_x_<tag>.npy / era5_y_<tag>.npy (default: <output_dir>/era5)",
    )
    p.add_argument("--grid_tag", default="ok", help="Grid file tag (<name_root>_x_<tag>.npy)")
    p.add_argument(
        "--name_root",
        default="era5",
        help="Naming token for the output subdir, target prefix and grid-axis prefix "
        "(default 'era5' for historical consistency; e.g. 'urma' writes "
        "urma/urma_rtma_ok_sfc_1_1h_....memmap). MUST match the era5_month/background_month/"
        "grid_files templates in the training grid_config YAML.",
    )
    p.add_argument(
        "--anl_pattern",
        default="urma2p5.t{hour:02d}z.2dvaranl_OK.grb2",
        help="Analysis filename pattern within a day dir ({hour} placeholder)",
    )
    p.add_argument(
        "--ges_pattern",
        default=None,
        help="First-guess filename within a day dir. Defaults to t00z in daily_00z mode "
        "and t{hour:02d}z in hourly mode.",
    )
    p.add_argument(
        "--background_mode",
        default="daily_00z",
        choices=["daily_00z", "hourly"],
        help="Write the existing one-frame-per-day 00 UTC background or a distinct "
        "24-frames-per-day hour-matched background product.",
    )
    p.add_argument(
        "--fill_missing",
        choices=["error", "persist"],
        default="error",
        help="error: abort if any file is missing (default); persist: reuse the previous frame",
    )
    p.add_argument(
        "--write_norms",
        action="store_true",
        help="Write norm_factors/mean|std_rtma_ok_sfc_1.npy from the processed target frames "
        "(only when processing the training period)",
    )
    p.add_argument(
        "--background_only",
        action="store_true",
        help="Write only the selected background product. Requires existing grid files and "
        "does not rewrite targets or target norm factors.",
    )
    p.add_argument(
        "--skip_time_check",
        action="store_true",
        help="Do not assert grib validity time == expected frame time",
    )
    return p.parse_args()


def month_range(start, end):
    """Inclusive list of (year, month) from 'YYYY-MM' strings."""
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    if (y1, m1) < (y0, m0):
        raise SystemExit(f"--end {end} is before --start {start}")
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def axes_from_grib(path):
    """Derive ascending 1-D lon/lat axes (float64 degrees) from a grib file's grid metadata."""
    with open(path, "rb") as f:
        gid = ec.codes_grib_new_from_file(f)
        if gid is None:
            raise SystemExit(f"{path}: no grib messages")
        try:
            grid_type = ec.codes_get(gid, "gridType")
            if grid_type != "regular_ll":
                raise SystemExit(
                    f"{path}: gridType {grid_type!r} != regular_ll; the axes can only be "
                    "derived from a regular lat-lon grid"
                )
            if ec.codes_get(gid, "iScansNegatively") or not ec.codes_get(gid, "jScansPositively"):
                raise SystemExit(f"{path}: scanning mode is not WE:SN")
            ni = ec.codes_get(gid, "Ni")
            nj = ec.codes_get(gid, "Nj")
            lon0 = ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees") % 360
            lon1 = ec.codes_get(gid, "longitudeOfLastGridPointInDegrees") % 360
            lat0 = ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
            lat1 = ec.codes_get(gid, "latitudeOfLastGridPointInDegrees")
        finally:
            ec.codes_release(gid)
    if lon1 <= lon0 or lat1 <= lat0:
        raise SystemExit(
            f"{path}: grid corners not ascending (lon {lon0}..{lon1}, lat {lat0}..{lat1})"
        )
    lon = np.linspace(lon0, lon1, ni, dtype=np.float64)
    lat = np.linspace(lat0, lat1, nj, dtype=np.float64)
    return lon, lat


def first_existing_anl(input_dir, months, anl_pattern):
    """First analysis grib file present in the requested range (grid template source)."""
    for year, month in months:
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            ddir = Path(input_dir) / f"{year:04d}{month:02d}{day:02d}"
            for hour in range(FRAMES_PER_DAY):
                fp = ddir / anl_pattern.format(hour=hour)
                if fp.exists():
                    return fp
    raise SystemExit("No analysis grib files found in the requested date range")


def check_grid(gid, lon, lat, path, tol=1e-4):
    """Assert one grib message's grid == the target axes (shape, corners, scan mode)."""
    ni = ec.codes_get(gid, "Ni")
    nj = ec.codes_get(gid, "Nj")
    if (ni, nj) != (lon.size, lat.size):
        raise SystemExit(
            f"{path}: grid {ni}x{nj} != era5_x/era5_y {lon.size}x{lat.size}; wrong grid files "
            "or the grib was not interpolated to the target grid"
        )
    lon0 = ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees") % 360
    lat0 = ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
    lon1 = ec.codes_get(gid, "longitudeOfLastGridPointInDegrees") % 360
    lat1 = ec.codes_get(gid, "latitudeOfLastGridPointInDegrees")
    if ec.codes_get(gid, "iScansNegatively") or not ec.codes_get(gid, "jScansPositively"):
        raise SystemExit(f"{path}: scanning mode is not WE:SN; this script assumes WE:SN")
    for got, want, name in [
        (lon0, lon[0], "lon first"),
        (lon1, lon[-1], "lon last"),
        (lat0, lat[0], "lat first"),
        (lat1, lat[-1], "lat last"),
    ]:
        if abs(got - want) > tol:
            raise SystemExit(f"{path}: {name} {got} != grid axis {want} (tol {tol})")


def read_frame(path, lon, lat, expected_ymdh, skip_time_check):
    """Read one grib2 file -> (CHANNELS, nlon, nlat) float32 in the decided channel order."""
    found = {}
    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                key = (
                    ec.codes_get(gid, "discipline"),
                    ec.codes_get(gid, "parameterCategory"),
                    ec.codes_get(gid, "parameterNumber"),
                    ec.codes_get(gid, "typeOfLevel"),
                    ec.codes_get(gid, "level"),
                )
                for name, d, c, n, tol_, lev in VAR_SPECS:
                    if key == (d, c, n, tol_, lev) and name not in found:
                        check_grid(gid, lon, lat, path)
                        if not skip_time_check:
                            vdate = ec.codes_get(gid, "validityDate")
                            vtime = ec.codes_get(gid, "validityTime")
                            got = f"{vdate:08d}{vtime // 100:02d}"
                            if got != expected_ymdh:
                                raise SystemExit(
                                    f"{path}: {name} validity {got} != expected {expected_ymdh}; "
                                    "frame would be misaligned (--skip_time_check to override)"
                                )
                        vals = ec.codes_get_values(gid).astype(np.float32)
                        # WE:SN scan: reshape (nlat, nlon) lat-ascending, transpose -> (nlon, nlat)
                        found[name] = vals.reshape(lat.size, lon.size).T
            finally:
                ec.codes_release(gid)
    missing = [s[0] for s in VAR_SPECS if s[0] not in found]
    if missing:
        raise SystemExit(f"{path}: variables not found in grib: {missing}")
    frame = np.stack([found[s[0]] for s in VAR_SPECS], axis=0)
    if not np.all(np.isfinite(frame)):
        raise SystemExit(
            f"{path}: {np.sum(~np.isfinite(frame))} non-finite values; the loader/model "
            "cannot handle NaN in target/background"
        )
    return frame


def prescan(
    input_dir,
    months,
    anl_pattern,
    ges_pattern,
    background_mode,
    fill_missing,
    background_only=False,
):
    """Verify every expected file exists before writing anything; list ALL gaps at once."""
    missing = []
    for year, month in months:
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            ddir = Path(input_dir) / f"{year:04d}{month:02d}{day:02d}"
            if not background_only:
                for hour in range(FRAMES_PER_DAY):
                    fp = ddir / anl_pattern.format(hour=hour)
                    if not fp.exists():
                        missing.append(str(fp))
            background_hours = range(FRAMES_PER_DAY) if background_mode == "hourly" else (0,)
            for hour in background_hours:
                gp = ddir / ges_pattern.format(hour=hour)
                if not gp.exists():
                    missing.append(str(gp))
    if missing:
        print(f"[{'WARN' if fill_missing == 'persist' else 'ERROR'}] {len(missing)} missing files:")
        for m in missing:
            print("  ", m)
        if fill_missing == "error":
            raise SystemExit(
                "Aborting: the loader requires complete months. Fix the archive or rerun "
                "with --fill_missing persist (fills from the previous frame)."
            )
    return set(missing)


def main():
    args = parse_args()
    if args.ges_pattern is None:
        args.ges_pattern = (
            "urma2p5.t{hour:02d}z.2dvarges_OK.grb2"
            if args.background_mode == "hourly"
            else "urma2p5.t00z.2dvarges_OK.grb2"
        )
    if args.background_mode == "hourly" and "{hour" not in args.ges_pattern:
        raise SystemExit("hourly background mode requires {hour} in --ges_pattern")
    if args.background_only and args.grid_source != "files":
        raise SystemExit("--background_only requires --grid_source files")
    if args.background_only and args.write_norms:
        raise SystemExit("--background_only cannot be combined with --write_norms")
    months = month_range(args.start, args.end)
    grid_dir = Path(args.grid_dir or os.path.join(args.output_dir, args.name_root))
    x_path = grid_dir / f"{args.name_root}_x_{args.grid_tag}.npy"
    y_path = grid_dir / f"{args.name_root}_y_{args.grid_tag}.npy"
    if args.grid_source == "grib":
        src = first_existing_anl(args.input_dir, months, args.anl_pattern)
        lon, lat = axes_from_grib(src)
        print(f"Grid axes derived from {src}")
        if x_path.exists() and y_path.exists():
            # Existing grid files are what training reads -- never contradict them silently.
            lon_f, lat_f = np.load(x_path), np.load(y_path)
            if (
                lon_f.shape != lon.shape
                or lat_f.shape != lat.shape
                or not np.allclose(lon_f, lon, atol=1e-4)
                or not np.allclose(lat_f, lat, atol=1e-4)
            ):
                raise SystemExit(
                    f"Axes derived from {src} do not match existing {x_path} / {y_path}; "
                    "delete/rebuild the grid files or run with --grid_source files"
                )
        else:
            grid_dir.mkdir(parents=True, exist_ok=True)
            np.save(x_path, lon)
            np.save(y_path, lat)
            print(
                f"Wrote {x_path} and {y_path} -- copy BOTH to model_data_path/grid_lon_lat/ "
                "for training (assert_grid_files_consistent reads the two copies)"
            )
    else:
        lon, lat = np.load(x_path), np.load(y_path)
        if not (np.all(np.diff(lon) > 0) and np.all(np.diff(lat) > 0)):
            raise SystemExit("Grid axes must be ascending (build_grid_lonlat.py convention).")
    nlon, nlat = lon.size, lat.size
    era5_dir = Path(args.output_dir) / args.name_root
    era5_dir.mkdir(parents=True, exist_ok=True)
    print(f"Grid: nlon={nlon} lon {lon[0]:.4f}..{lon[-1]:.4f}, nlat={nlat} lat {lat[0]:.4f}..{lat[-1]:.4f}")

    missing = prescan(
        args.input_dir,
        months,
        args.anl_pattern,
        args.ges_pattern,
        args.background_mode,
        args.fill_missing,
        args.background_only,
    )

    # Running per-channel sums over TARGET frames only (float64, same as prep_era5_truth.py)
    sum_c = np.zeros(CHANNELS, dtype=np.float64)
    sumsq_c = np.zeros(CHANNELS, dtype=np.float64)
    count_c = 0

    for year, month in months:
        days = calendar.monthrange(year, month)[1]
        tgt_path = era5_dir / f"{args.name_root}_{ERA5_MODE}_1_{FREQ_TAG}_{year}-{month:02d}.memmap"
        bg_stem = "background_raw_hourly" if args.background_mode == "hourly" else "background_raw"
        bg_path = era5_dir / f"{bg_stem}_{ERA5_MODE}_1_{year}-{month:02d}.memmap"
        background_frames = days * (24 if args.background_mode == "hourly" else 1)
        tgt = None
        if not args.background_only:
            tgt = np.memmap(
                tgt_path,
                dtype="float32",
                mode="w+",
                shape=(days * FRAMES_PER_DAY, CHANNELS, nlon, nlat),
            )
        bg = np.memmap(
            bg_path,
            dtype="float32",
            mode="w+",
            shape=(background_frames, CHANNELS, nlon, nlat),
        )

        prev_frame = None
        prev_background = None
        for day in range(1, days + 1):
            ddir = Path(args.input_dir) / f"{year:04d}{month:02d}{day:02d}"
            for hour in range(FRAMES_PER_DAY):
                idx = (day - 1) * FRAMES_PER_DAY + hour
                if not args.background_only:
                    fp = ddir / args.anl_pattern.format(hour=hour)
                    if str(fp) in missing:
                        if prev_frame is None:
                            raise SystemExit(
                                f"{fp}: missing with no previous frame to persist from"
                            )
                        print(f"[FILL] {fp} missing -> persisted previous frame into row {idx}")
                        frame = prev_frame
                    else:
                        expected = f"{year:04d}{month:02d}{day:02d}{hour:02d}"
                        frame = read_frame(
                            fp, lon, lat, expected, args.skip_time_check
                        )
                    tgt[idx] = frame
                    prev_frame = frame
                    sum_c += frame.sum(axis=(1, 2), dtype=np.float64)
                    sumsq_c += np.square(frame, dtype=np.float64).sum(axis=(1, 2))
                    count_c += nlon * nlat

                if args.background_mode == "hourly":
                    gp = ddir / args.ges_pattern.format(hour=hour)
                    if str(gp) in missing:
                        if prev_background is None:
                            raise SystemExit(
                                f"{gp}: missing with no previous background to persist from"
                            )
                        print(f"[FILL] {gp} missing -> persisted previous background into row {idx}")
                        background_frame = prev_background
                    else:
                        expected = f"{year:04d}{month:02d}{day:02d}{hour:02d}"
                        background_frame = read_frame(
                            gp, lon, lat, expected, args.skip_time_check
                        )
                    bg[idx] = background_frame
                    prev_background = background_frame

            if args.background_mode == "daily_00z":
                gp = ddir / args.ges_pattern.format(hour=0)
                if str(gp) in missing:
                    if args.background_only:
                        raise SystemExit(
                            f"{gp}: --background_only cannot fill from an analysis; "
                            "provide a complete background archive"
                        )
                    print(f"[FILL] {gp} missing -> persisted 00z ANALYSIS of the day as background")
                    bg[day - 1] = tgt[(day - 1) * FRAMES_PER_DAY]
                else:
                    bg[day - 1] = read_frame(
                        gp,
                        lon,
                        lat,
                        f"{year:04d}{month:02d}{day:02d}00",
                        args.skip_time_check,
                    )

        if tgt is not None:
            tgt.flush()
        bg.flush()
        del bg
        if tgt is not None:
            del tgt
            print(
                f"[OK] {year}-{month:02d}: {tgt_path.name} "
                f"({days * FRAMES_PER_DAY}, {CHANNELS}, {nlon}, {nlat}), "
                f"{bg_path.name} ({background_frames}, {CHANNELS}, {nlon}, {nlat})"
            )
        else:
            print(
                f"[OK] {year}-{month:02d}: {bg_path.name} "
                f"({background_frames}, {CHANNELS}, {nlon}, {nlat})"
            )

    if args.write_norms:
        norms_dir = Path(args.output_dir) / "norm_factors"
        norms_dir.mkdir(parents=True, exist_ok=True)
        mean = sum_c / count_c
        var = sumsq_c / count_c - np.square(mean)
        std = np.sqrt(np.clip(var, 0, None)) + 1e-8
        np.save(norms_dir / f"mean_{ERA5_MODE}_1.npy", mean)
        np.save(norms_dir / f"std_{ERA5_MODE}_1.npy", std)
        print(f"[OK] norms over {count_c // (nlon * nlat)} target frames:")
        for (name, *_), m, s in zip(VAR_SPECS, mean, std):
            print(f"     {name:4s} mean={m:.6g} std={s:.6g}")

    print(
        "\nNext: python scripts/normalize_background.py --data_dir",
        args.output_dir,
        "--era5_mode",
        ERA5_MODE,
        "--background_mode",
        args.background_mode,
        "--months",
        " ".join(f"{y}-{m:02d}" for y, m in months),
    )


if __name__ == "__main__":
    main()
