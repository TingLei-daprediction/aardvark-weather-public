"""Compare two one-epoch warm-start runs of the RTMA OK encoder.

Companion to ``training/test-val-regression-rtma_ok_sfc-1GPU.sh``. Given a
reference run directory and a candidate run directory, checks that the metrics a
code change must not move are unchanged, and reports the size of any drift.

Intended sequence
-----------------
1. On UNCHANGED code, run the shell script twice, into ``ref/`` and ``noise/``::

       python scripts/check_val_regression.py ref/ noise/

   This measures the run-to-run noise floor (non-deterministic cuDNN backward
   kernels) and validates the harness. Whatever tolerance passes here is the
   tolerance to use in step 3. If this step fails outright, determinism is not
   pinned and no later comparison means anything -- the fix would be to add
   seeding to train_module.py before relying on this test.

2. Apply the code change.

3. Run the shell script again into ``after/`` and compare against the same
   reference::

       python scripts/check_val_regression.py ref/ after/ --rtol <from step 1>

Which files are compared, and why
---------------------------------
Compared by value -- these are the assertions:

  losses_0.npy         validation loss; the number that gates checkpoint
                       selection, and the primary assertion
  train_losses_0.npy   training loss; the training path is untouched
  rmse_0.npy           unnormalized per-variable RMSE; changing this was
                       explicitly deferred, so movement means something broke
  preds_train.npy      last TRAINING batch; the training sampler is unchanged
  y_target_train.npy   likewise

Compared by shape only -- these are expected to change:

  preds_eval.npy       these hold the last VALIDATION batch, and switching the
  y_target_eval.npy    validation sampler to shuffle=False changes which sample
  unnorm_preds_0.npy   is last. Their contents will differ for a reason that is
  unnorm_targets_0.npy not a regression; only the shape must hold.
"""

import argparse
import os
import sys

import numpy as np

VALUE_FILES = [
    "losses_0.npy",
    "train_losses_0.npy",
    "rmse_0.npy",
    "preds_train.npy",
    "y_target_train.npy",
]

SHAPE_ONLY_FILES = [
    "preds_eval.npy",
    "y_target_eval.npy",
    "unnorm_preds_0.npy",
    "unnorm_targets_0.npy",
]


def load(run_dir, name):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    return np.load(path)


def describe_delta(a, b):
    """Max absolute and relative difference, ignoring NaNs present in both."""
    a = np.atleast_1d(a.astype(np.float64))
    b = np.atleast_1d(b.astype(np.float64))

    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0

    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, diff / denom, 0.0)
    rel[both_nan] = 0.0
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    return max_abs, max_rel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_dir", help="Run directory captured on known-good code")
    p.add_argument("candidate_dir", help="Run directory to check")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    args = p.parse_args()

    for d in (args.reference_dir, args.candidate_dir):
        if not os.path.isdir(d):
            print(f"[ERROR] not a directory: {d}", file=sys.stderr)
            return 2

    failures = []
    missing = []
    print(f"reference: {args.reference_dir}")
    print(f"candidate: {args.candidate_dir}")
    print(f"tolerance: rtol={args.rtol} atol={args.atol}\n")

    print("must not move:")
    for name in VALUE_FILES:
        ref = load(args.reference_dir, name)
        new = load(args.candidate_dir, name)
        if ref is None or new is None:
            which = "reference" if ref is None else "candidate"
            missing.append(f"{name} (absent from {which})")
            continue

        if ref.shape != new.shape:
            failures.append(f"{name}: shape {ref.shape} -> {new.shape}")
            print(f"  [FAIL] {name:<22} shape {ref.shape} -> {new.shape}")
            continue

        ok = np.allclose(
            ref.astype(np.float64),
            new.astype(np.float64),
            rtol=args.rtol,
            atol=args.atol,
            equal_nan=True,
        )
        max_abs, max_rel = describe_delta(ref, new)
        print(
            f"  [{'OK  ' if ok else 'FAIL'}] {name:<22} "
            f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        )
        if not ok:
            failures.append(f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

    print("\nshape only (contents expected to differ):")
    for name in SHAPE_ONLY_FILES:
        ref = load(args.reference_dir, name)
        new = load(args.candidate_dir, name)
        if ref is None or new is None:
            which = "reference" if ref is None else "candidate"
            missing.append(f"{name} (absent from {which})")
            continue
        if ref.shape != new.shape:
            failures.append(f"{name}: shape {ref.shape} -> {new.shape}")
            print(f"  [FAIL] {name:<22} shape {ref.shape} -> {new.shape}")
        else:
            max_abs, _ = describe_delta(ref, new)
            print(f"  [OK  ] {name:<22} shape {ref.shape} max_abs={max_abs:.3e}")

    if missing:
        print("\nmissing files:")
        for line in missing:
            print(f"  - {line}")

    if failures:
        print("\n[FAIL] metrics that must not move have changed:")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("\n[OK] all compared metrics match within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
