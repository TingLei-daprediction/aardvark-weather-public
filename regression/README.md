# Compact validation regression reference

Commit the approved real-run summary as `regression/rtma_ok_sfc_reference.json`.
No reference is supplied yet: it must come from a successful run on the cluster,
not synthetic data. Keep large arrays and checkpoints outside Git.

Each single-GPU regression job now runs `scripts/summarize_val_regression.py`
after training and writes `summary.json` inside its output directory. All nine
expected `.npy` files must exist. Existing summaries are never overwritten.

The JSON text preserves all validation-loss, training-loss and RMSE values.
For each large training prediction/target array it stores shape, nonfinite
counts and a hash of nonfinite locations, up to 256 evenly spaced values, and
min/max/mean/standard deviation/RMS for up to 64 contiguous flattened blocks.
Validation prediction/target files are checked by shape only because changing
validation shuffle order changes the final saved batch. Statistics use float64.
The summary size is bounded for prediction arrays; metric vectors are retained
in full. Provenance records the code commit and checkpoint path but is not
compared, since code commits necessarily change during regression testing.

This is a lossy statistical regression check: localized changes or rearrangements
can escape detection. Keep raw outputs when diagnosing a failure or when full
elementwise assurance is required; `scripts/check_val_regression.py` remains
available. Summary tolerances must be calibrated independently of raw-array
tolerances. NaN/Inf locations in value-compared files must match exactly.

## Establish a reference

For the raw-background change, use the committed A/B worktree procedure in
[the regression handoff](../docs/regression_test_handoff.md#capture-and-calibrate-the-first-baseline).
Run the candidate unit suite in the pinned cluster environment first, capture
the approved baseline from commit A (`23b19c8`, old loader), then compare both
candidate modes from commit B. Pin `REGRESSION_TRAINING_DIR` and
`REGRESSION_CODE_COMMIT` as shown there; do not capture this baseline from the
feature working tree. The general commands below assume the intended checkout
has already been selected.

Adapt the existing cluster/environment/data paths in the training scripts.
Choose an immutable checkpoint and use it for every run (the default otherwise
selects the latest checkpoint, which could change). From the repository root:

```bash
export REGRESSION_CHECKPOINT=/absolute/path/to/epoch_123
cd training
sbatch test-install-val-regression.sh /absolute/path/to/regression-runs
```

Wait for successful completion. This runs unchanged code twice, generates two
summaries and sweeps relative tolerances at `atol=1e-12`. Inspect the report and
choose a tolerance with a margin above observed variation. Repeat as needed to
establish stability; a single pair of runs cannot prove determinism.

After approving the baseline, from the repository root:

```bash
cp /absolute/path/to/regression-runs/ref/summary.json regression/rtma_ok_sfc_reference.json
```

Alongside the reference, record the chosen `rtol`/`atol`, data version, exact
checkpoint identity, GPU/software environment, and any script/config changes in
this README. Record a checkpoint checksum if its path is mutable. The recorded
commit does not describe uncommitted changes. Review and commit the JSON and
documentation together; never automatically replace an approved reference.

## Check a code change

With the same data, environment and checkpoint, submit a fresh run:

```bash
cd training
sbatch test-val-regression-rtma_ok_sfc-1GPU.sh /absolute/path/to/after
# After the job completes, from the repository root:
cd ..
python scripts/summarize_val_regression.py compare regression/rtma_ok_sfc_reference.json /absolute/path/to/after/summary.json --rtol <approved-rtol> --atol <approved-atol>
```

Exit codes: 0 means pass, 1 means mismatched results, 2 means invalid/missing
inputs. Numeric comparisons use `abs(candidate-reference) <= atol +
rtol*abs(reference)`. Shapes and nonfinite masks/counts compare exactly.

Existing completed runs can also be post-processed without rerunning training:

```bash
python scripts/summarize_val_regression.py create /path/to/run /path/to/summary.json --commit <run-code-commit> --checkpoint /path/to/checkpoint
```
