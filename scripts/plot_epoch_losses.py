#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def load_series(patterns, root):
    series = []
    for pat in patterns:
        for path in sorted(root.glob(pat)):
            arr = np.load(path)
            if arr.ndim == 0:
                arr = np.array([arr])
            series.append(arr)
    return series


def truncate_to_complete(series_list):
    if not series_list:
        return np.array([])
    min_len = min(len(s) for s in series_list)
    if min_len == 0:
        return np.array([])
    stacked = np.stack([s[:min_len] for s in series_list], axis=0)
    return stacked.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean train/val loss per epoch (complete epochs only)."
    )
    parser.add_argument(
        "--run_dir",
        help="Output run directory containing losses_*.npy and train_losses_*.npy",
    )
    parser.add_argument(
        "--err_file",
        help="Slurm stderr log to parse tqdm loss lines from (dd.err).",
    )
    parser.add_argument(
        "--plot_batches",
        action="store_true",
        help="When using --err_file, also plot per-batch loss points.",
    )
    parser.add_argument(
        "--out",
        default="epoch_losses.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

    if args.err_file:
        err_path = Path(args.err_file)
        if not err_path.exists():
            raise FileNotFoundError(err_path)
        import re

        # Parse tqdm lines like "365/365 ... loss=0.67".
        # Use the "cur/total" fields to group batch losses into epochs and
        # compute an epoch-mean loss per full pass through the data.
        pattern = re.compile(r"\\b(\\d+)/(\\d+).*loss=([0-9.+-eE]+)")
        batch_losses = []
        epoch_means = []
        cur_epoch = []
        for line in err_path.read_text(errors="ignore").splitlines():
            match = pattern.search(line)
            if match:
                try:
                    cur = int(match.group(1))
                    total = int(match.group(2))
                    loss = float(match.group(3))
                    batch_losses.append(loss)
                    cur_epoch.append(loss)
                    if cur == total:
                        # End of an epoch: average all batch losses in this epoch.
                        epoch_means.append(float(np.mean(cur_epoch)))
                        cur_epoch = []
                except ValueError:
                    pass
        if not batch_losses:
            raise RuntimeError("No loss values found in err file.")
        train_mean = np.array(epoch_means)
        val_mean = np.array([])
        batch_series = np.array(batch_losses)
        out_path = err_path.parent / args.out
    else:
        if not args.run_dir:
            raise RuntimeError("Provide --run_dir or --err_file.")
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)

        train_series = load_series(["train_losses_*.npy"], run_dir)
        val_series = load_series(["losses_*.npy"], run_dir)

        train_mean = truncate_to_complete(train_series)
        val_mean = truncate_to_complete(val_series)

        if train_mean.size == 0 and val_mean.size == 0:
            raise RuntimeError("No loss arrays found in run_dir.")
        out_path = run_dir / args.out

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    if args.err_file:
        if args.plot_batches and "batch_series" in locals():
            plt.plot(batch_series, label="batch_loss", alpha=0.4)
        if train_mean.size:
            plt.plot(train_mean, label="epoch_mean_loss", linewidth=2)
    else:
        if train_mean.size:
            plt.plot(train_mean, label="train_loss")
    if val_mean.size:
        plt.plot(val_mean, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mean Loss per Epoch (complete epochs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
