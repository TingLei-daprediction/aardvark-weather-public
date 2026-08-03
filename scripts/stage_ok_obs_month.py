"""Stage one month of colleague-processed OK surface obs into the loader layout.

Input dir (per month) uses the colleague's names:
    {t,q,ps,u,v}_{lon,lat,alt}_train.npy   {t,q,ps,u,v}_vals_train.memmap

Output dir is <data_path>/hadisd_processed/ with the names the loader builds
(aardvark/loader.py load_hadisd + _load_obs_monthly):
    {tas,sh,psl,u,v}_{lon,lat,alt}_train-<YYYY>-<MM>.npy  (one set per month)
    {tas,sh,psl,u,v}_vals_1h_<YYYY>-<MM>.memmap   (one per month)
    ../norm_factors/{mean,std}_hadisd_{var}_train-<YYYY>-<MM>.npy

Checks performed (hard errors unless noted):
  - all 4 files present per variable
  - lon/lat/alt are 1-D and share one station count
  - vals file size == days_in_month * frames_per_day * n_stations * 4 (float32)
  - each month's lon/lat/alt arrays match that month's values station dimension
  - unit sanity per variable (warning only): tas in K, sh in kg/kg, psl in Pa

Usage (once per month directory):
    python stage_ok_obs_month.py --src_dir /path/to/202201_dir \
        --year_month 2022-01 --out_dir <data_root>/hadisd_processed
    python stage_ok_obs_month.py --src_dir /path/to/202202_dir \
        --year_month 2022-02 --out_dir <data_root>/hadisd_processed

The script computes one mean/std value per station from this month's finite values. Monthly
coordinates, values, means, and standard deviations therefore share one station ordering.
Stations with no finite values receive inert mean=0/std=1 factors (their values remain NaN and
are masked); positive but degenerate standard deviations are floored to 1e-8.
"""

import argparse
import calendar
import os
import shutil
import sys

import numpy as np

# colleague's token -> loader token (loader.py: hadisd_vars for rtma_surface)
VAR_MAP = {"t": "tas", "q": "sh", "ps": "psl", "u": "u", "v": "v"}

FRAMES_PER_DAY = {"1h": 24, "15min": 96}

# (lo, hi, hint) — plausible physical range per LOADER variable; outside -> unit warning
UNIT_RANGES = {
    "tas": (150.0, 350.0, "expected Kelvin; values look like Celsius?"),
    "sh": (0.0, 0.06, "expected kg/kg; values look like g/kg (divide by 1000)?"),
    "psl": (55000.0, 110000.0, "expected Pa; values look like hPa (multiply by 100)?"),
    "u": (-80.0, 80.0, "expected m/s u-component"),
    "v": (-80.0, 80.0, "expected m/s v-component"),
}

errors = []
warnings = []


def err(msg):
    errors.append(msg)
    print("  [ERROR] " + msg)


def warn(msg):
    warnings.append(msg)
    print("  [WARN]  " + msg)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src_dir", required=True, help="colleague's per-month obs dir")
    p.add_argument("--year_month", required=True, help="e.g. 2022-02")
    p.add_argument("--out_dir", required=True, help="<data_path>/hadisd_processed")
    p.add_argument(
        "--norm_dir",
        help="monthly norm output directory (default: sibling norm_factors directory)",
    )
    p.add_argument("--freq_tag", default="1h", choices=sorted(FRAMES_PER_DAY))
    p.add_argument("--print_counts", action="store_true",
                   help="print loader-variable=station-count records and exit")
    p.add_argument("--max_stations", action="append", default=[], metavar="VAR=N",
                   help="pad VAR to cross-month station count N; repeat per variable")
    p.add_argument(
        "--dry_run", action="store_true", help="check everything, write nothing"
    )

    args = p.parse_args()

    max_stations = {}
    for item in args.max_stations:
        try:
            var, count = item.split("=", 1)
            count = int(count)
        except (ValueError, TypeError):
            p.error(f"invalid --max_stations {item!r}; expected VAR=N")
        if var not in VAR_MAP.values() or count <= 0:
            p.error(f"invalid --max_stations {item!r}")
        max_stations[var] = count

    if args.print_counts:
        for src_var, tgt_var in VAR_MAP.items():
            lon_path = os.path.join(args.src_dir, f"{src_var}_lon_train.npy")
            if not os.path.isfile(lon_path):
                p.error(f"missing coordinate file {lon_path}")
            lon = np.asarray(np.load(lon_path))
            if lon.ndim != 1 or lon.size == 0:
                p.error(f"invalid coordinate array {lon_path}: shape={lon.shape}")
            print(f"{tgt_var}={lon.size}")
        return

    year, month = map(int, args.year_month.split("-"))
    frames = calendar.monthrange(year, month)[1] * FRAMES_PER_DAY[args.freq_tag]
    print(
        f"Staging {args.year_month} ({frames} frames at {args.freq_tag}) "
        f"from {args.src_dir}"
    )
    norm_dir = args.norm_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out_dir)), "norm_factors"
    )
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)

    plans = []  # (src, dst) copies to perform after all checks pass
    generated = []  # (array, dst) padded coordinate/norm .npy files
    generated_memmaps = []  # source, frames, real/stored stations, destination
    for src_var, tgt_var in VAR_MAP.items():
        print(f"\n{src_var} -> {tgt_var}")

        paths = {
            k: os.path.join(args.src_dir, f"{src_var}_{k}_train.npy")
            for k in ("lon", "lat", "alt")
        }
        paths["vals"] = os.path.join(args.src_dir, f"{src_var}_vals_train.memmap")
        missing = [k for k, f in paths.items() if not os.path.isfile(f)]
        if missing:
            err(f"missing files: {', '.join(sorted(missing))}")
            continue

        coords = {k: np.load(paths[k]) for k in ("lon", "lat", "alt")}
        shapes = {k: a.shape for k, a in coords.items()}
        if len({s for s in shapes.values()}) != 1 or coords["lon"].ndim != 1:
            err(f"coord shape mismatch: {shapes}")
            continue
        n_stations = coords["lon"].shape[0]
        padded_stations = max_stations.get(tgt_var, n_stations)
        if padded_stations < n_stations:
            err(f"{tgt_var}: maximum {padded_stations} is smaller than "
                f"real station count {n_stations}")
            continue
        print(f"  stations: real={n_stations}, stored={padded_stations}, "
              f"padding={padded_stations - n_stations}")

        nbytes = os.path.getsize(paths["vals"])
        expected = frames * n_stations * 4
        if nbytes != expected:
            got = nbytes / (n_stations * 4)
            err(
                f"vals size {nbytes} != {expected} "
                f"({frames} frames x {n_stations} stations x 4B); file has "
                f"{got:g} frames"
            )
            continue

        vals = np.memmap(
            paths["vals"], dtype="float32", mode="r", shape=(frames, n_stations)
        )
        finite = np.isfinite(vals)
        n_nan = vals.size - finite.sum()
        vmin, vmax = (
            (np.nanmin(vals), np.nanmax(vals)) if finite.any() else (np.nan, np.nan)
        )
        print(
            f"  range: [{vmin:.4g}, {vmax:.4g}]   "
            f"NaN: {n_nan} ({100.0 * n_nan / vals.size:.2f}%)"
        )
        lo, hi, hint = UNIT_RANGES[tgt_var]
        if finite.any() and not (lo <= vmin and vmax <= hi):
            warn(f"range outside [{lo:g}, {hi:g}] -- {hint}")

        # Preserve the original HadISD contract: one mean/std per station, reduced over
        # time. These are generated from the same monthly values matrix and therefore
        # cannot drift from its station ordering.
        values64 = np.asarray(vals, dtype=np.float64)
        finite_count = np.sum(np.isfinite(values64), axis=0)
        station_mean = np.zeros(n_stations, dtype=np.float64)
        np.divide(
            np.nansum(values64, axis=0),
            finite_count,
            out=station_mean,
            where=finite_count > 0,
        )
        centered = np.where(
            np.isfinite(values64), values64 - station_mean[np.newaxis, :], 0.0
        )
        station_var = np.zeros(n_stations, dtype=np.float64)
        np.divide(
            np.sum(centered * centered, axis=0),
            finite_count,
            out=station_var,
            where=finite_count > 0,
        )
        station_std = np.sqrt(np.maximum(station_var, 0.0))
        empty = finite_count == 0
        degenerate = (~empty) & (station_std < 1.0e-8)
        station_mean[empty] = 0.0
        station_std[empty] = 1.0
        station_std[degenerate] = 1.0e-8
        if np.any(empty):
            warn(
                f"{int(np.sum(empty))} stations have no finite {tgt_var} values; "
                "writing mean=0/std=1 (observations remain NaN-masked)"
            )
        if np.any(degenerate):
            warn(
                f"{int(np.sum(degenerate))} stations have degenerate {tgt_var} std; "
                "flooring std to 1e-8"
            )
        padded_mean = np.zeros(padded_stations, dtype="float32")
        padded_std = np.ones(padded_stations, dtype="float32")
        padded_mean[:n_stations] = station_mean.astype("float32")
        padded_std[:n_stations] = station_std.astype("float32")
        generated.extend(
            [
                (
                    padded_mean,
                    os.path.join(
                        norm_dir,
                        f"mean_hadisd_{tgt_var}_train-{year}-{month:02d}.npy",
                    ),
                ),
                (
                    padded_std,
                    os.path.join(
                        norm_dir,
                        f"std_hadisd_{tgt_var}_train-{year}-{month:02d}.npy",
                    ),
                ),
            ]
        )

        # Keep the real monthly order and append NaN padding to the common size.
        for k in ("lon", "lat", "alt"):
            dst = os.path.join(
                args.out_dir, f"{tgt_var}_{k}_train-{year}-{month:02d}.npy"
            )
            padded_coord = np.full(padded_stations, np.nan, dtype="float32")
            padded_coord[:n_stations] = np.asarray(coords[k], dtype="float32")
            generated.append((padded_coord, dst))
        generated_memmaps.append(
            (
                vals, frames, n_stations, padded_stations,
                os.path.join(
                    args.out_dir,
                    f"{tgt_var}_vals_{args.freq_tag}_{year}-{month:02d}.memmap",
                ),
            )
        )

    print()
    if errors:
        print(f"FAILED: {len(errors)} error(s); nothing written.")
        sys.exit(1)
    for src, dst in plans:
        if args.dry_run:
            print(f"[dry-run] {src} -> {dst}")
        else:
            shutil.copyfile(src, dst)
            print(f"copied {src} -> {dst}")
    for array, dst in generated:
        if args.dry_run:
            print(f"[dry-run] monthly norm {array.shape} -> {dst}")
        else:
            np.save(dst, array)
            print(f"wrote padded array {array.shape} -> {dst}")
    for source, frames, real_stations, padded_stations, dst in generated_memmaps:
        if args.dry_run:
            print(f"[dry-run] padded memmap ({frames}, {padded_stations}) -> {dst}")
        else:
            output = np.memmap(dst, dtype="float32", mode="w+",
                               shape=(frames, padded_stations))
            output[:] = np.nan
            output[:, :real_stations] = source
            output.flush()
            del output
            print(f"wrote padded memmap ({frames}, {padded_stations}) -> {dst}")
    print(
        f"\nOK: {args.year_month} staged"
        + (f" with {len(warnings)} warning(s) -- review above" if warnings else "")
        + (" (dry run, nothing written)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
