"""
Build-time normalization of per-month daily-00z or hourly RTMA-OK BACKGROUND files.

The encoder's "climatology" input slot is fed unnormalized by the loader, so the background must
be normalized at BUILD time. It is normalized with the SAME target mean/std as the rtma_ok_sfc
target fields -- NOT the HadISD observation normalization.

  normalized = (raw_background - target_mean) / target_std        (per channel)

Layout (same channels/order as the target):
  raw in:  <data_dir>/era5/<raw_name>_<era5_mode>_1_<YYYY>-<MM>.memmap
  norm:    <data_dir>/norm_factors/mean_<era5_mode>_1.npy, std_<era5_mode>_1.npy   (C,)
  out:     <data_dir>/era5/<out_name>_<era5_mode>_1_<YYYY>-<MM>.memmap

daily_00z uses ``days`` frames and the existing background names. hourly uses ``days * 24``
frames and distinct background_hourly names.

The loader reads the outputs as-is and hard-asserts the mode-dependent frame count and target
channel count.

Also writes the RAW background's own per-channel stats (diagnostic; the loader does not read
them -- normalization uses the TARGET mean/std by design):
  <data_dir>/norm_factors/mean_background_<era5_mode>_1.npy, std_background_<era5_mode>_1.npy
Comparing them with mean/std_<era5_mode>_1.npy shows the background/analysis offset (bias).

Example:
  python scripts/normalize_background.py --data_dir /path/aardvark-data-ok \
      --era5_mode rtma_ok_sfc --months 2019-01 2019-02 2019-03
"""

import argparse
import calendar
from pathlib import Path

import numpy as np


def days_in_month(year, month):
    return calendar.monthrange(year, month)[1]


def parse_months(years, months):
    if months:
        out = []
        for m in months:
            y, mm = m.split("-")
            out.append((int(y), int(mm)))
        return out
    return [(y, mm) for y in years for mm in range(1, 13)]


def parse_args():
    p = argparse.ArgumentParser(
        description="Normalize per-month RTMA background with target mean/std (build time)."
    )
    p.add_argument("--data_dir", required=True, help="Base data_path (has era5/ and norm_factors/)")
    p.add_argument("--era5_mode", default="rtma_ok_sfc")
    p.add_argument("--years", nargs="*", type=int, default=[], help="Years (all 12 months each)")
    p.add_argument("--months", nargs="*", default=[], help="Explicit YYYY-MM list (overrides --years)")
    p.add_argument(
        "--background_mode",
        default="daily_00z",
        choices=["daily_00z", "hourly"],
        help="Background cadence; hourly selects distinct *_hourly input/output names.",
    )
    p.add_argument("--raw_name", default=None, help="Override the mode-specific raw input stem")
    p.add_argument("--out_name", default=None, help="Override the mode-specific output stem")
    p.add_argument(
        "--chunk_frames",
        type=int,
        default=24,
        help="Number of time frames normalized at once (default: 24)",
    )
    p.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip missing raw months. Default is to fail before writing any output.",
    )
    p.add_argument(
        "--subdir",
        default="era5",
        help="Subdir under data_dir holding the background memmaps (the naming token; must "
        "match the background_month template in the training grid_config YAML)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.chunk_frames < 1:
        raise ValueError("--chunk_frames must be positive")
    if args.raw_name is None:
        args.raw_name = (
            "background_raw_hourly"
            if args.background_mode == "hourly"
            else "background_raw"
        )
    if args.out_name is None:
        args.out_name = (
            "background_hourly" if args.background_mode == "hourly" else "background"
        )
    base = Path(args.data_dir)
    era5_dir = base / args.subdir
    nf = base / "norm_factors"

    mean = np.load(nf / f"mean_{args.era5_mode}_1.npy").astype(np.float32).reshape(-1)
    std = np.load(nf / f"std_{args.era5_mode}_1.npy").astype(np.float32).reshape(-1)
    C = mean.shape[0]
    if std.shape[0] != C:
        raise ValueError(f"mean/std channel mismatch: {C} vs {std.shape[0]}")
    if np.any(std == 0):
        raise ValueError("std has zero entries; cannot normalize the background")

    months = parse_months(args.years, args.months)
    if not months:
        raise SystemExit("Provide --months YYYY-MM ... or --years ...")

    raw_inputs = []
    missing_inputs = []
    for year, month in months:
        tag = f"{year}-{month:02d}"
        raw_path = era5_dir / f"{args.raw_name}_{args.era5_mode}_1_{tag}.memmap"
        if raw_path.is_file():
            raw_inputs.append((year, month, raw_path))
        else:
            missing_inputs.append(raw_path)
    if missing_inputs and not args.allow_missing:
        formatted = "\n  ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(
            "missing raw background month(s); no outputs were written:\n  " + formatted
        )
    for path in missing_inputs:
        print(f"[WARN] missing {path}, skipping because --allow_missing was supplied")
    if not raw_inputs:
        raise FileNotFoundError("no raw background months are available to normalize")

    # Raw-background stats (diagnostic; float64 running sums, same scheme as the target norms)
    bg_sum = np.zeros(C, dtype=np.float64)
    bg_sumsq = np.zeros(C, dtype=np.float64)
    bg_count = 0

    for year, month, raw_path in raw_inputs:
        tag = f"{year}-{month:02d}"
        out_path = era5_dir / f"{args.out_name}_{args.era5_mode}_1_{tag}.memmap"
        days = days_in_month(year, month)
        frames = days * (24 if args.background_mode == "hourly" else 1)
        nbytes = raw_path.stat().st_size
        denom = frames * C * 4
        if nbytes % denom != 0:
            raise ValueError(
                f"{raw_path}: size {nbytes} not divisible by frames*C*4={denom} "
                f"(frames={frames}, C={C}, background_mode={args.background_mode})"
            )
        spatial = nbytes // denom  # nlon*nlat, kept flat (per-channel norm needs no split)
        raw = np.memmap(raw_path, dtype="float32", mode="r", shape=(frames, C, spatial))
        out = np.memmap(out_path, dtype="float32", mode="w+", shape=(frames, C, spatial))
        for start in range(0, frames, args.chunk_frames):
            end = min(start + args.chunk_frames, frames)
            raw_chunk = np.asarray(raw[start:end], dtype=np.float32)
            out[start:end] = (raw_chunk - mean[None, :, None]) / std[None, :, None]
            bg_sum += raw_chunk.sum(axis=(0, 2), dtype=np.float64)
            bg_sumsq += np.square(raw_chunk, dtype=np.float64).sum(axis=(0, 2))
            bg_count += raw_chunk.shape[0] * spatial
        out.flush()
        del raw, out
        print(f"wrote {out_path}  (frames={frames}, C={C}, spatial={spatial})")

    if bg_count:
        bg_mean = bg_sum / bg_count
        bg_std = np.sqrt(np.clip(bg_sumsq / bg_count - np.square(bg_mean), 0, None)) + 1e-8
        bg_mean_path = nf / f"mean_{args.raw_name.replace('_raw', '')}_{args.era5_mode}_1.npy"
        bg_std_path = nf / f"std_{args.raw_name.replace('_raw', '')}_{args.era5_mode}_1.npy"
        np.save(bg_mean_path, bg_mean)
        np.save(bg_std_path, bg_std)
        print(f"wrote {bg_mean_path} and {bg_std_path} (RAW background stats, diagnostic):")
        for c in range(C):
            print(
                f"  ch{c}: bg mean={bg_mean[c]:.6g} std={bg_std[c]:.6g}  |  "
                f"target mean={mean[c]:.6g} std={std[c]:.6g}"
            )


if __name__ == "__main__":
    main()
