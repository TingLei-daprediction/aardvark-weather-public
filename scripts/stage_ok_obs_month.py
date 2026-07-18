"""Stage one month of colleague-processed OK surface obs into the loader layout.

Input dir (per month) uses the colleague's names:
    {t,q,ps,u,v}_{lon,lat,alt}_train.npy   {t,q,ps,u,v}_vals_train.memmap

Output dir is <data_path>/hadisd_processed/ with the names the loader builds
(aardvark/loader.py load_hadisd + _load_obs_monthly):
    {tas,sh,psl,u,v}_{lon,lat,alt}_train.npy      (static, shared by all months)
    {tas,sh,psl,u,v}_vals_1h_<YYYY>-<MM>.memmap   (one per month)

Checks performed (hard errors unless noted):
  - all 4 files present per variable
  - lon/lat/alt are 1-D and share one station count
  - vals file size == days_in_month * frames_per_day * n_stations * 4 (float32)
  - coords identical to any already-staged copy in out_dir (station list must
    be static across months)
  - unit sanity per variable (warning only): tas in K, sh in kg/kg, psl in Pa

Usage (once per month directory):
    python stage_ok_obs_month.py --src_dir /path/to/202201_dir \
        --year_month 2022-01 --out_dir <data_root>/hadisd_processed
    python stage_ok_obs_month.py --src_dir /path/to/202202_dir \
        --year_month 2022-02 --out_dir <data_root>/hadisd_processed

Norm factors (norm_factors/mean_hadisd_{var}.npy, std_hadisd_{var}.npy) are a
separate deliverable and are not handled here.
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
    p.add_argument("--freq_tag", default="1h", choices=sorted(FRAMES_PER_DAY))
    p.add_argument(
        "--dry_run", action="store_true", help="check everything, write nothing"
    )
    args = p.parse_args()

    year, month = map(int, args.year_month.split("-"))
    frames = calendar.monthrange(year, month)[1] * FRAMES_PER_DAY[args.freq_tag]
    print(
        f"Staging {args.year_month} ({frames} frames at {args.freq_tag}) "
        f"from {args.src_dir}"
    )
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    plans = []  # (src, dst) copies to perform after all checks pass
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
        print(f"  stations: {n_stations}")

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

        # Coords must be identical across months (loader loads ONE static set).
        for k in ("lon", "lat", "alt"):
            dst = os.path.join(args.out_dir, f"{tgt_var}_{k}_train.npy")
            if os.path.isfile(dst):
                prev = np.load(dst)
                if prev.shape != coords[k].shape or not np.array_equal(
                    prev, coords[k], equal_nan=True
                ):
                    err(
                        f"{tgt_var}_{k}: differs from already-staged {dst}; "
                        "station list must be identical across months"
                    )
            else:
                plans.append((paths[k], dst))
        plans.append(
            (
                paths["vals"],
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
    print(
        f"\nOK: {args.year_month} staged"
        + (f" with {len(warnings)} warning(s) -- review above" if warnings else "")
        + (" (dry run, nothing written)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
