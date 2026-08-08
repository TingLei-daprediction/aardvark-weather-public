"""One-command plot summary for an RTMA OK encoder run.

Takes the Slurm log of a training run and produces, in the run's output
directory:

  * epoch_losses.png                      via plot_epoch_losses.py
  * sample_<i>_<var>_ch<n>.png  (x5)      via plot_encoder_field.py

The run's output directory is read out of the log. Not every training script
echoes it -- the warm-start scripts print "New output directory:", the infer
scripts print "Output dir:", and test-val-regression prints "Output directory:",
but the plain new-tl-train-encoder-*.sh scripts only assign the variable without
printing it. For those, pass --run_dir explicitly.

Usage
-----
    python scripts/plot_run_summary.py training/new-aardvark-...-warmstart.12345.err

    python scripts/plot_run_summary.py <log> --run_dir /path/to/OK-output-.../

Notes
-----
plot_epoch_losses.py prefers the "[INFO] epoch N/M train_loss=.. val_loss=.."
summary lines, so pass the stream those land in. trainer.py prints them to
stdout, which under these sbatch scripts is the .out file; tqdm progress goes to
stderr. If one file yields nothing, try the other.

Channels default to 0-4, the five rtma_ok_sfc surface targets. plot_encoder_field
names each PNG from the channel index automatically, so no variable list is
hardcoded here.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Ordered by preference; the last match in the log wins, so a log containing
# several runs resolves to the most recent one.
RUN_DIR_PATTERNS = [
    re.compile(r"New output directory:\s*(\S+)"),
    re.compile(r"Output directory:\s*(\S+)"),
    re.compile(r"Output dir:\s*(\S+)"),
    re.compile(r"--output_dir\s+\"?(\S+?)\"?\s"),
]


def extract_run_dir(log_path):
    text = log_path.read_text(errors="ignore")
    for pattern in RUN_DIR_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip().strip('"').strip("'")
    return None


def run(cmd):
    """Run a child tool, streaming its output. Returns True on success."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_file", help="Slurm .out/.err log from the training run")
    p.add_argument(
        "--run_dir",
        help="Run output directory. Required when the log does not echo it.",
    )
    p.add_argument(
        "--channels",
        default="0,1,2,3,4",
        help="Comma-separated output channel indices (default: the 5 sfc targets)",
    )
    p.add_argument("--rank", default="0", help="Rank suffix of unnorm_preds_<rank>.npy")
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument(
        "--loss_out",
        default="epoch_losses.png",
        help="Filename for the loss plot, written into the run directory",
    )
    p.add_argument("--grid_dir", help="Passed through to plot_encoder_field.py")
    args = p.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"[ERROR] log file not found: {log_path}", file=sys.stderr)
        return 2

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        found = extract_run_dir(log_path)
        if not found:
            print(
                f"[ERROR] could not find an output directory in {log_path}.\n"
                "        The plain new-tl-train-encoder-*.sh scripts do not echo it.\n"
                "        Re-run with --run_dir /path/to/output/",
                file=sys.stderr,
            )
            return 2
        run_dir = Path(found)
        print(f"[INFO] run directory from log: {run_dir}")

    if not run_dir.is_dir():
        print(f"[ERROR] not a directory: {run_dir}", file=sys.stderr)
        return 2

    try:
        channels = [int(c) for c in args.channels.split(",") if c.strip() != ""]
    except ValueError:
        print(f"[ERROR] --channels must be comma-separated integers", file=sys.stderr)
        return 2
    if not channels:
        print("[ERROR] --channels is empty", file=sys.stderr)
        return 2

    produced = []
    failed = []

    # Step 1: loss curve. plot_epoch_losses.py joins --out onto the log's parent
    # directory, and pathlib lets an absolute right-hand operand win, so passing
    # an absolute path lands the PNG in the run directory instead.
    loss_png = (run_dir / args.loss_out).resolve()
    ok = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "plot_epoch_losses.py"),
            "--err_file",
            str(log_path),
            "--out",
            str(loss_png),
        ]
    )
    (produced if ok else failed).append(str(loss_png))

    # Step 2: one field plot per channel. plot_encoder_field.py derives the
    # filename from the channel index, so --out is deliberately omitted.
    for channel in channels:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "plot_encoder_field.py"),
            "--run_dir",
            str(run_dir),
            "--rank",
            args.rank,
            "--sample_index",
            str(args.sample_index),
            "--channel",
            str(channel),
        ]
        if args.grid_dir:
            cmd += ["--grid_dir", args.grid_dir]
        ok = run(cmd)
        (produced if ok else failed).append(f"channel {channel}")

    print("\n" + "=" * 60)
    print(f"run directory: {run_dir}")
    pngs = sorted(str(p.name) for p in run_dir.glob("*.png"))
    if pngs:
        print(f"PNG files now in the run directory ({len(pngs)}):")
        for name in pngs:
            print(f"  {name}")
    else:
        print("No PNG files found in the run directory.")

    if failed:
        print(f"\n[FAIL] {len(failed)} step(s) failed:")
        for item in failed:
            print(f"  - {item}")
        return 1

    print(f"\n[OK] {len(produced)} plots generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
