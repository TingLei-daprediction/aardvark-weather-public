"""Diagnose stations whose normalization std is degenerate (~0 or non-finite).

Background: obs are normalized per station as (x - mean[i]) / std[i] with no epsilon
(loader.py norm_hadisd), and the SetConv encoder masks NaN -- but NOT inf
(set_convs.py). So a degenerate std is harmless exactly when every value it divides
is NaN or equals the mean (both yield NaN -> masked), and dangerous when a varying
value meets std ~ 0 (finite/0 = inf passes the mask into the encoder).

For every obs variable, this script finds the stations flagged by
check_ok_run_files.py (std <= eps or non-finite) and, per staged month file, reports
the station's valid (non-NaN) sample count and its distinct reported values -- the
two facts that fully determine what the division above will produce. Verdicts:

  PADDING   -- no valid obs in any month checked. All values NaN; NaN propagates
               through the division regardless of std and is masked. Harmless.
  SINGLE-OB -- exactly 1 valid ob in the stats window. That value equals the mean,
               so 0/0 -> NaN -> masked. Harmless in-window.
  STUCK     -- many valid samples but one distinct value (stuck sensor / QC
               flattening). Every value equals the mean -> 0/0 -> NaN -> masked.
               Numerically harmless, but a data-quality defect worth excluding
               at the source.
  INF-RISK  -- valid, VARYING values in some month while std ~ 0: (x - mean)/0 =
               +/-inf enters the encoder. Regenerate the norm factors with the
               degenerate-std guard (std -> median station std) before running.
               The script exits 1 so wrappers can gate on it.

Also noted inline: a station with values but a NaN stored mean normalizes to NaN
(masked) -- there the NaN mean, not the std, is doing the protecting.

Scope: only stations whose std is ALREADY degenerate are inspected, and only for
the months passed via --months. A station can be PADDING for the checked months
and still wake up later; the permanent fix is the std guard in the norm-factor
generation script -- this diagnostic only certifies the files you have today.

Usage:
    python diagnose_degenerate_stations.py \
        --data_root /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data \
        --months 2022-01 2022-02
(--aux_root defaults to --data_root; --months should list every month you intend
to train/validate on -- missing month files are reported and skipped.)
"""

import argparse
import os

import numpy as np

OBS_VARS = ["tas", "sh", "psl", "u", "v"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_root", required=True)
    p.add_argument("--aux_root", default=None, help="root holding norm_factors/ (default: data_root)")
    p.add_argument("--months", nargs="+", required=True, help="e.g. 2022-01 2022-02")
    p.add_argument("--vars", nargs="+", default=OBS_VARS, choices=OBS_VARS)
    p.add_argument("--freq_tag", default="1h")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--show_values", type=int, default=5, help="distinct values to print")
    args = p.parse_args()
    aux = args.aux_root or args.data_root
    obs_dir = os.path.join(args.data_root, "hadisd_processed")

    any_risk = False
    for var in args.vars:
        std_path = os.path.join(aux, "norm_factors", f"std_hadisd_{var}.npy")
        mean_path = os.path.join(aux, "norm_factors", f"mean_hadisd_{var}.npy")
        if not os.path.isfile(std_path):
            print(f"\n{var}: no std file at {std_path} -- skipped")
            continue
        std = np.load(std_path)
        mean = np.load(mean_path) if os.path.isfile(mean_path) else np.full_like(std, np.nan)
        flagged = np.where(~np.isfinite(std) | (std <= args.eps))[0]
        print(f"\n===== {var}: {std.shape[0]} stations, {flagged.size} degenerate =====")
        if flagged.size == 0:
            continue

        # Open each requested month's values file once; column i = station i.
        month_vals = {}
        for ym in args.months:
            path = os.path.join(obs_dir, f"{var}_vals_{args.freq_tag}_{ym}.memmap")
            if os.path.isfile(path):
                month_vals[ym] = np.memmap(path, dtype="float32", mode="r").reshape(
                    -1, std.shape[0]
                )
            else:
                print(f"  (no file for {ym}: {path} -- skipped)")

        lon = lat = None
        lon_path = os.path.join(obs_dir, f"{var}_lon_train.npy")
        if os.path.isfile(lon_path):
            lon = np.load(lon_path)
            lat = np.load(os.path.join(obs_dir, f"{var}_lat_train.npy"))

        for i in flagged:
            loc = f" @({lon[i]:.3f}E,{lat[i]:.3f}N)" if lon is not None else ""
            print(f"  station {i}{loc}: mean={mean[i]:.6g}, std={std[i]:.6g}")
            total_valid = 0
            for ym, vals in month_vals.items():
                col = np.asarray(vals[:, i])
                finite = np.isfinite(col)
                nv = int(finite.sum())
                total_valid += nv
                uniq = np.unique(col[finite])
                shown = ", ".join(f"{u:.6g}" for u in uniq[: args.show_values])
                more = f" (+{uniq.size - args.show_values} more)" if uniq.size > args.show_values else ""
                print(f"    {ym}: valid={nv}/{col.shape[0]}, distinct values: [{shown}]{more}")
                if nv > 0 and np.isfinite(std[i]) and std[i] <= args.eps and uniq.size > 1:
                    any_risk = True
                    print(f"    --> INF-RISK in {ym}: varying values will be divided by std~0")
                elif nv > 0 and not np.isfinite(mean[i]):
                    print(f"    (values in {ym} but mean is NaN -> normalized to NaN, masked)")
            if total_valid == 0:
                print("    verdict: PADDING (no valid obs anywhere checked; NaN-masked, harmless)")
            elif total_valid == 1:
                print("    verdict: SINGLE-OB (0/0 -> NaN in stats window, harmless in-window)")
            else:
                uniq_all = np.unique(
                    np.concatenate(
                        [np.asarray(v[:, i])[np.isfinite(v[:, i])] for v in month_vals.values()]
                    )
                )
                if uniq_all.size == 1:
                    print("    verdict: STUCK (constant sensor; in-window harmless, consider excluding)")
                else:
                    print("    verdict: INF-RISK (valid, varying obs meet std~0 -- fix before running)")

    print("\n" + "=" * 60)
    if any_risk:
        print("RESULT: INF-RISK found -- regenerate norm factors with the degenerate-std guard")
        raise SystemExit(1)
    print("RESULT: no inf risk in the months checked (padding/single-ob/stuck only).")


if __name__ == "__main__":
    main()
