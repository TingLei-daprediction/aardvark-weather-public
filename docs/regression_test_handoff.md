# First regression test: handoff

Saved September 27, 2026. The user plans to return to this task later.

## Current status

The initial inspection identified the setup steps below. During the subsequent
raw-background implementation, the parent-shell conda activation, fixed seed,
and stale-comment fixes were applied locally. No cluster jobs were submitted and
no baseline was captured. Use the pre-change loader with only these harness fixes
to establish the baseline before merging the loader changes.

The repository already contains:

- `regression/README.md`: baseline and comparison procedure.
- `training/test-install-val-regression.sh`: runs unchanged code twice into
  `ref/` and `noise/`, then sweeps comparison tolerances.
- `training/test-val-regression-rtma_ok_sfc-1GPU.sh`: one-epoch, single-GPU,
  warm-start RTMA Oklahoma surface encoder run over January 1–2, 2022.
- `scripts/summarize_val_regression.py`: creates and compares compact JSON summaries.
- `scripts/check_val_regression.py`: compares raw output arrays.

There is no approved `regression/rtma_ok_sfc_reference.json` yet.

Dependency for `docs/plan_raw_background.md`: complete the setup fixes below and
approve a baseline using the existing loader before merging raw-background loader
changes. The harness currently uses default `daily_00z`; it does not establish
hourly end-to-end coverage. After implementation, compare both default normalized
and opt-in raw candidate runs against the same approved daily baseline. Preserve
the exact normalization factors and their checkpoint provenance for these runs.

## Setup work to do first

Steps 2 and 3 are now implemented locally; validate the actual cluster environment
and inputs before submission. Default harness invocation omits the new input flags
so it can also run against the pre-change loader.

1. Verify cluster paths, Slurm account/partition/QoS, and the Linux/CUDA
   `aardvark-env` in both shell scripts. Required external inputs are a compatible
   encoder checkpoint, RTMA/URMA data, observation data, normalization files,
   and grid/model auxiliary files. Monthly memmaps must contain the expected
   full month even though the test selects only two days.
2. Completed in commit A: the install script activates `aardvark-env` in its own
   shell, so its final Python comparison uses the intended environment.
3. Completed in commit A: the single-run command passes `--seed 42`, and both
   scripts' stale seeding comments are corrected. PyTorch, CUDA, NumPy, and Python
   are seeded; GPU kernels still require repeat-run tolerance calibration.
4. Pin an immutable checkpoint with `REGRESSION_CHECKPOINT`; do not rely on
   automatic selection of the latest checkpoint. Keep data, grid configuration,
   software, and GPU type fixed. Record the code revision and any local changes.

## Capture and calibrate the first baseline

### 1. Use separate committed checkouts

The review split is:

- **Commit A: `23b19c80d0163a3507ef964b344cd87446305318`** contains only
  `training/test-install-val-regression.sh` and
  `training/test-val-regression-rtma_ok_sfc-1GPU.sh`. The loader is the old loader.
- **Formatting: `958e56f15dc22ffd23cfdb2e419f12074ddacf86`** contains only Black
  formatting of nine existing Python files. Their ASTs were verified unchanged.
- **Commit B:** the following `feat: support raw background input with normalization provenance`
  commit contains implementation, tests, and documentation. Use its full hash as
  `candidate_commit` below, not the formatting commit or a dirty checkout.

On the cluster, after transferring these commits, from a checkout at commit B:

```bash
candidate_commit=$(git rev-parse HEAD)
# Confirm this is the feature commit before continuing:
git show -s --format='%H %s' "$candidate_commit"
baseline_commit=23b19c80d0163a3507ef964b344cd87446305318
baseline_dir=/absolute/path/to/aardvark-baseline-A
candidate_dir=/absolute/path/to/aardvark-candidate-B
runs=/absolute/path/to/regression-runs/first

git worktree add --detach "$baseline_dir" "$baseline_commit"
git worktree add --detach "$candidate_dir" "$candidate_commit"
```

Both regression scripts accept `REGRESSION_TRAINING_DIR`; without this override,
explicit hardcoded cluster paths would select the original checkout. They also
check `REGRESSION_CODE_COMMIT` and reject tracked modifications when it is set.
Use these two settings for every submission below. Choose new worktree and output
paths. If cluster path/account changes require script edits, create a new
harness-only revision on A, then carry the same edits to B and record both revised
hashes. Do not disable the clean-checkout check or change loaders in A.

### 2. Run the candidate unit suite in the pinned environment first

This check is **pending**. The prior Windows/Python 3.12 tests do not satisfy it.
The suite lives on B, so run it there before submitting either regression job;
A does not contain the new tests and discovering zero tests on A is not a pass.

```bash
source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
cd "$candidate_dir"
python -c 'import sys, numpy, torch; print(sys.version); print(numpy.__version__, torch.__version__); assert sys.version_info[:2] == (3, 8); assert numpy.__version__ == "1.24.3"; assert torch.__version__.split("+")[0] == "2.3.0"'
python -m unittest discover -s tests -v
```

Retain the test output and environment versions. All tests must pass. Resolve any
failure before running the regressions, and record a new B hash if code changes.

### 3. Capture the reference from A

```bash
export REGRESSION_CHECKPOINT=/absolute/path/to/immutable/epoch_123
export REGRESSION_TRAINING_DIR="$baseline_dir/training"
export REGRESSION_CODE_COMMIT="$baseline_commit"
export BACKGROUND_INPUT=normalized
unset BACKGROUND_NORM_MANIFEST
cd "$baseline_dir/training"
sbatch --export=ALL test-install-val-regression.sh "$runs"
```

Wait for successful completion. Inspect `ref/summary.json`, `noise/summary.json`,
and the job logs; verify their provenance commit equals A. The harness sweeps
`rtol` from `1e-12` to `1e-2` at `atol=1e-12`. Choose tolerances with margin above
observed variation and repeat calibration as needed. Large differences require
investigation. Record checkpoint checksum, exact norm files/hashes, data version,
GPU/software versions, seed, and approval in an external run record. Keep the
approved summary outside the clean checkouts until comparisons finish:

```bash
cp "$runs/ref/summary.json" "$runs/approved-reference.json"
```

**Do not proceed to candidate GPU comparisons until the A baseline is approved.**
No baseline or approval is supplied by this repository yet.

## Compare normalized and raw candidates from B

Keep the same checkpoint, data, statistics, seed, environment, and GPU type.
For raw use of a legacy checkpoint, first review the original norm provenance and
record a checkpoint-specific manifest as described in `docs/background_input.md`.
That manifest must represent the factors used for the A reference.

```bash
export REGRESSION_TRAINING_DIR="$candidate_dir/training"
export REGRESSION_CODE_COMMIT="$candidate_commit"
cd "$candidate_dir/training"
export BACKGROUND_INPUT=normalized
unset BACKGROUND_NORM_MANIFEST
sbatch --export=ALL test-val-regression-rtma_ok_sfc-1GPU.sh "$runs/after-normalized"

# Submit after the first candidate job finishes (both use the same master port).
export BACKGROUND_INPUT=raw
export BACKGROUND_NORM_MANIFEST=/absolute/path/to/epoch_123.normalization.json
sbatch --export=ALL test-val-regression-rtma_ok_sfc-1GPU.sh "$runs/after-raw"
```

After both jobs complete, from B, compare **both against the same A reference**:

```bash
cd "$candidate_dir"
python scripts/summarize_val_regression.py compare \
  "$runs/approved-reference.json" "$runs/after-normalized/summary.json" \
  --rtol <approved-rtol> --atol <approved-atol>
python scripts/summarize_val_regression.py compare \
  "$runs/approved-reference.json" "$runs/after-raw/summary.json" \
  --rtol <approved-rtol> --atol <approved-atol>
```

Exit codes: 0 = pass, 1 = mismatch, 2 = invalid/missing inputs. Verify candidate
summary provenance is B. Once validation is complete, copy the approved A JSON to
`regression/rtma_ok_sfc_reference.json` in the development checkout and record the
approved tolerances and provenance in `regression/README.md`. Commit those together
as a follow-up; do not modify either pinned checkout during queued/running jobs.
Keep raw arrays and checkpoints outside Git. Never automatically replace a baseline.

## Coverage and limits

All nine expected output arrays must exist. Summaries preserve full validation
loss, training loss, and per-variable RMSE vectors. Training predictions and
targets are checked through shapes, nonfinite counts/location hashes, sampled
values, and block statistics. The four validation prediction/target arrays are
checked only by shape because validation sampler changes can change the saved
last batch.

This is a compact statistical check of one training/validation configuration.
It does not check every prediction value or multi-GPU behavior. Retain raw
outputs for diagnosis; raw-array tolerances need separate calibration.

## Resume point

Start by reviewing this note, `regression/README.md`, and `docs/background_input.md`.
The harness adjustments are committed as A. Next run the B unit suite in the
pinned cluster environment, verify the actual inputs, and capture the reference
from the separate A worktree. Then compare the candidate's normalized and raw modes against that approved
reference. No cluster execution has occurred in this session.
