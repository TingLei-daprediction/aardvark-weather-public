"""Preflight check for an RTMA-OK encoder training script.

Reads a training sbatch script (shell variables + train_module.py flags), then verifies
that every file the monthly rtma_surface loader will open actually exists with a
consistent shape/size -- BEFORE burning a SLURM allocation to find out.

Path construction is delegated to the repo's own grid_config / loader_utils_new modules,
so the names checked here are byte-identical to what the loader builds.

Checks:
  - date flags parse as real calendar dates (catches e.g. 2022-02-31)
  - grid axes under data_path and model_data_path exist and match (values, not just shape)
  - elev_vars file exists with shape (4, nlat, nlon)
  - target norm factors exist, finite, std > 0
  - per month (union of train+val ranges): target memmap and 00z background memmap sizes
    match days_in_month * frames * channels * nlon * nlat * 4 bytes
  - obs (tas, sh, psl, u, v): static lon/lat/alt coords consistent; per-month vals memmap
    sized frames x n_stations x 4; per-station norm factors exist with shape (n_stations,)
    (warns on std <= 0 / non-finite -- see degenerate-station guard discussion)
  - lat_weights only if the script uses --loss lw_rmse

Usage:
    python check_ok_run_files.py [--train_script ../training/new-tl-train-encoder-rtma_ok_sfc.sh]
"""

import argparse
import calendar
import os
import re
import sys
from datetime import date

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "aardvark"))

from grid_config import (  # noqa: E402
    load_grid_config,
    set_active_config,
    loader_grid_x_path,
    loader_grid_y_path,
    model_grid_x_path,
    model_grid_y_path,
    elev_vars_path,
    norm_mean_path,
    norm_std_path,
    era5_month_path,
    background_month_path,
    lat_weights_path,
    assert_grid_files_consistent,
)
from loader_utils_new import parse_time_freq, days_in_month  # noqa: E402

OBS_VARS = ["tas", "sh", "psl", "u", "v"]

n_ok = 0
errors = []
warnings = []


def ok(msg):
    global n_ok
    n_ok += 1
    print(f"  [ok]    {msg}")


def err(msg):
    errors.append(msg)
    print(f"  [ERROR] {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  [WARN]  {msg}")


def parse_train_script(path):
    """Extract shell variables and --flag values from the sbatch script."""
    text = open(path).read().replace("\\\n", " ")
    varmap = {}
    for name, raw in re.findall(r'^\s*(\w+)=("[^"]*"|\S+)\s*$', text, re.M):
        varmap[name] = raw.strip('"')

    def expand(value, depth=0):
        if depth > 10:
            return value
        out = re.sub(
            r"\$\{(\w+)\}|\$(\w+)",
            lambda m: varmap.get(m.group(1) or m.group(2), m.group(0)),
            value,
        )
        return expand(out, depth + 1) if "$" in out and out != value else out

    varmap = {k: expand(v) for k, v in varmap.items()}
    flags = {
        name: expand(raw.strip('"'))
        for name, raw in re.findall(r'--(\w+)\s+("[^"]*"|[^-\s]\S*)', text)
    }
    return varmap, flags


def parse_date(flags, key):
    raw = flags.get(key)
    if raw is None:
        err(f"--{key} not found in training script")
        return None
    try:
        y, m, d = map(int, raw.split("-"))
        return date(y, m, d)  # raises ValueError for e.g. Feb 31
    except ValueError:
        err(f"--{key} {raw!r} is not a valid calendar date")
        return None


def month_range(d0, d1):
    months = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def check_memmap(path, expected_frames, per_frame_bytes, what):
    """Verify existence and that size == frames * per_frame; return True if usable."""
    if not os.path.isfile(path):
        err(f"{what}: MISSING {path}")
        return False
    nbytes = os.path.getsize(path)
    if nbytes != expected_frames * per_frame_bytes:
        err(
            f"{what}: {path} has {nbytes / per_frame_bytes:g} frames, "
            f"expected {expected_frames}"
        )
        return False
    ok(f"{what}: {os.path.basename(path)} ({expected_frames} frames)")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--train_script",
        default=os.path.join(REPO, "training", "new-tl-train-encoder-rtma_ok_sfc.sh"),
    )
    args = p.parse_args()

    print(f"Training script: {args.train_script}")
    varmap, flags = parse_train_script(args.train_script)

    data_path = flags.get("data_path", "")
    aux_path = flags.get("aux_data_path", "")
    model_path = flags.get("model_data_path", "")
    era5_mode = flags.get("era5_mode", "rtma_ok_sfc")
    time_freq = flags.get("time_freq", "1H")
    for name, val in [
        ("data_path", data_path),
        ("aux_data_path", aux_path),
        ("model_data_path", model_path),
    ]:
        print(f"  --{name} = {val or '<UNSET>'}")
        if not val or "$" in val:
            err(f"--{name} is unset or contains an unexpanded shell variable")
    if flags.get("obs_set") != "rtma_surface":
        warn(f"--obs_set is {flags.get('obs_set')!r}; this checker assumes rtma_surface")

    # Same grid config the run will install. --grid_config is relative to rundir.
    gc = flags.get("grid_config", "")
    rundir = varmap.get("rundir", os.path.dirname(os.path.abspath(args.train_script)))
    gc_path = gc if os.path.isabs(gc) else os.path.normpath(os.path.join(rundir, gc))
    if not os.path.isfile(gc_path):
        alt = os.path.join(REPO, "aardvark", os.path.basename(gc))
        if os.path.isfile(alt):
            gc_path = alt
        else:
            err(f"grid config not found: {gc_path}")
            print(f"\nFAILED: {len(errors)} error(s)")
            sys.exit(1)
    print(f"  grid config = {gc_path}")
    set_active_config(load_grid_config(gc_path))
    _, frames_per_day, freq_tag = parse_time_freq(time_freq)

    if errors:
        print(f"\nFAILED early: {len(errors)} error(s) in script parsing")
        sys.exit(1)

    print("\n== Dates ==")
    d = {k: parse_date(flags, f"assim_{k}") for k in
         ("train_start_date", "train_end_date", "val_start_date", "val_end_date")}
    months = []
    if all(d.values()):
        for a, b in [(d["train_start_date"], d["train_end_date"]),
                     (d["val_start_date"], d["val_end_date"])]:
            if a > b:
                err(f"date range reversed: {a} > {b}")
            months += [m for m in month_range(a, b) if m not in months]
        ok(f"months required: {', '.join(f'{y}-{m:02d}' for y, m in months)}")

    print("\n== Grid axes ==")
    nlon = nlat = None
    try:
        lon = np.load(loader_grid_x_path(data_path))
        lat = np.load(loader_grid_y_path(data_path))
        nlon, nlat = lon.shape[0], lat.shape[0]
        ok(f"loader axes: nlon={nlon}, nlat={nlat}, "
           f"lon [{lon.min():.3f}, {lon.max():.3f}], lat [{lat.min():.3f}, {lat.max():.3f}]")
    except FileNotFoundError as e:
        err(f"loader grid axes: {e}")
    try:
        assert_grid_files_consistent(model_path, data_path)
        ok("model grid axes exist and match loader axes (values compared)")
    except (FileNotFoundError, ValueError) as e:
        err(f"model/loader grid consistency: {e}")

    if nlon is None:
        print(f"\nFAILED: cannot continue without grid axes ({len(errors)} error(s))")
        sys.exit(1)
    grid_bytes = nlon * nlat * 4

    print("\n== Static / norm files (targets) ==")
    ep = elev_vars_path(data_path)
    if os.path.isfile(ep):
        elev = np.load(ep)
        if elev.shape == (4, nlat, nlon):
            ok(f"elev_vars {elev.shape}")
        else:
            err(f"elev_vars shape {elev.shape} != (4, {nlat}, {nlon}) [(4, nlat, nlon)]")
    else:
        err(f"elev_vars MISSING: {ep}")

    channels = None
    for name, path in [("target mean", norm_mean_path(aux_path, era5_mode)),
                       ("target std", norm_std_path(aux_path, era5_mode))]:
        if not os.path.isfile(path):
            err(f"{name} MISSING: {path}")
            continue
        arr = np.load(path)
        bad = (~np.isfinite(arr)).sum() or ("std" in name and (arr <= 0).sum())
        if bad:
            err(f"{name} {path}: non-finite or non-positive entries")
        else:
            ok(f"{name}: shape {arr.shape}")
        channels = arr.shape[0]

    if flags.get("loss") == "lw_rmse":
        lw = lat_weights_path(aux_path)
        (ok if os.path.isfile(lw) else err)(f"lat_weights ({lw})")

    print("\n== Per-month target + background memmaps ==")
    ch = channels or 5
    for y, m in months:
        frames = days_in_month(y, m) * frames_per_day
        check_memmap(era5_month_path(data_path, era5_mode, freq_tag, y, m),
                     frames, ch * grid_bytes, f"target {y}-{m:02d}")
        check_memmap(background_month_path(data_path, era5_mode, y, m),
                     days_in_month(y, m), ch * grid_bytes, f"background {y}-{m:02d}")

    print("\n== Surface observations ==")
    for var in OBS_VARS:
        coords = {}
        for k in ("lon", "lat", "alt"):
            path = os.path.join(data_path, "hadisd_processed", f"{var}_{k}_train.npy")
            if os.path.isfile(path):
                coords[k] = np.load(path)
            else:
                err(f"obs {var}: MISSING {path}")
        if len(coords) < 3 or len({a.shape for a in coords.values()}) != 1:
            if len(coords) == 3:
                err(f"obs {var}: coord shapes differ "
                    f"{ {k: a.shape for k, a in coords.items()} }")
            continue
        n = coords["lon"].shape[0]
        ok(f"obs {var}: {n} stations")
        for y, m in months:
            path = os.path.join(data_path, "hadisd_processed",
                                f"{var}_vals_{freq_tag}_{y}-{m:02d}.memmap")
            check_memmap(path, days_in_month(y, m) * frames_per_day, n * 4,
                         f"obs {var} {y}-{m:02d}")
        for stat in ("mean", "std"):
            path = os.path.join(aux_path, "norm_factors", f"{stat}_hadisd_{var}.npy")
            if not os.path.isfile(path):
                err(f"obs {var} {stat} MISSING: {path}")
                continue
            arr = np.load(path)
            if arr.shape != (n,):
                err(f"obs {var} {stat}: shape {arr.shape} != ({n},)")
                continue
            n_nonfinite = (~np.isfinite(arr)).sum()
            note = f", {n_nonfinite} non-finite (masked stations)" if n_nonfinite else ""
            ok(f"obs {var} {stat}: shape {arr.shape}{note}")
            if stat == "std":
                n_bad = int((arr[np.isfinite(arr)] <= 1e-6).sum())
                if n_bad:
                    warn(f"obs {var} std: {n_bad} station(s) <= 1e-6 -- apply the "
                         "degenerate-station guard (std -> median station std)")

    print("\n" + "=" * 60)
    print(f"{n_ok} checks passed, {len(warnings)} warning(s), {len(errors)} error(s)")
    if errors:
        print("NOT READY -- fix the errors above before submitting.")
        sys.exit(1)
    print("READY: all files the loader will open are present and consistent."
          + (" Review warnings above." if warnings else ""))


if __name__ == "__main__":
    main()
