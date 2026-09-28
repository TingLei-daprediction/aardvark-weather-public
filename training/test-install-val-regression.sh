#!/bin/bash
# First-install smoke test for the validation-path regression harness.
#
# Runs the whole "step 1" procedure in a single allocation:
#
#   1. one-epoch warm-start run  -> <base_dir>/ref
#   2. the SAME run again        -> <base_dir>/noise
#   3. compare the two, then sweep tolerances to find the noise floor
#
# Both runs execute identical, UNMODIFIED code, so any difference between them
# is pure numerical noise -- non-deterministic cuDNN backward kernels, plus
# anything else unseeded in the training path. That number is the resolution
# limit of the whole test: a later before/after comparison can only detect
# changes larger than it.
#
# Run this BEFORE applying any code change. It answers two questions at once:
# does the harness work at all, and what tolerance should gate the real
# comparison later.
#
# The two runs reuse training/test-val-regression-rtma_ok_sfc-1GPU.sh rather
# than repeating its flags, so the reference and the later "after" run cannot
# drift apart. Executing that script with bash (instead of sbatch) simply runs
# it; its own #SBATCH lines are inert comments.
#
# Usage:
#   sbatch test-install-val-regression.sh <base_dir>
#SBATCH -A fv3-cam
#SBATCH -J av-ok-sfc-valreg-install
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 01:00:00
#SBATCH -o new-aardvark-valreg-install.%j.out
#SBATCH -e new-aardvark-valreg-install.%j.err

set -euo pipefail
source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH

base_dir="${1:?usage: $0 <base_dir>}"
base_dir="${base_dir%/}"

rundir="${REGRESSION_TRAINING_DIR:-/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/}"
cd "$rundir"

# Pin execution to the intended clean checkout, including when sbatch copies this script.
if [[ -n "${REGRESSION_CODE_COMMIT:-}" ]]; then
  expected_commit=$(git rev-parse "${REGRESSION_CODE_COMMIT}^{commit}")
  [[ "$(git rev-parse HEAD)" == "$expected_commit" ]] || {
    echo "ERROR: checkout does not match REGRESSION_CODE_COMMIT=$expected_commit" >&2
    exit 1
  }
  git diff --quiet HEAD -- || {
    echo "ERROR: tracked files differ from the pinned regression commit" >&2
    exit 1
  }
fi


single_run="./test-val-regression-rtma_ok_sfc-1GPU.sh"
compare="../scripts/summarize_val_regression.py"

for f in "$single_run" "$compare"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

ref_dir="${base_dir}/ref"
noise_dir="${base_dir}/noise"

# Refuse to reuse a populated directory: stale arrays from an earlier or
# partially-failed run would silently corrupt the comparison.
for d in "$ref_dir" "$noise_dir"; do
  if [[ -d "$d" && -n "$(ls -A "$d" 2>/dev/null)" ]]; then
    echo "ERROR: $d exists and is not empty. Remove it or pick a new base_dir." >&2
    exit 1
  fi
done

# train_module.py uses os.mkdir, which needs the parent to exist already.
mkdir -p "$ref_dir" "$noise_dir"

echo "============================================================"
echo "base_dir:  $base_dir"
echo "reference: $ref_dir"
echo "noise:     $noise_dir"
echo "============================================================"

echo
echo ">>> [1/3] reference run"
bash "$single_run" "$ref_dir"

echo
echo ">>> [2/3] repeat run (identical code, identical flags)"
bash "$single_run" "$noise_dir"

echo
echo ">>> [3/3] comparison"
echo

set +e
python "$compare" compare "$ref_dir/summary.json" "$noise_dir/summary.json"
compare_status=$?
set -e

echo
echo "------------------------------------------------------------"
echo "noise floor: smallest relative tolerance at which the two"
echo "identical runs still agree"
echo "------------------------------------------------------------"

noise_floor=""
for tol in 1e-12 1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2; do
  if python "$compare" compare "$ref_dir/summary.json" "$noise_dir/summary.json" \
       --rtol "$tol" --atol 1e-12 >/dev/null 2>&1; then
    noise_floor="$tol"
    break
  fi
done

echo
if [[ -z "$noise_floor" ]]; then
  echo "RESULT: FAILED -- the two identical runs disagree even at rtol=1e-2."
  echo
  echo "The fixed seed did not bound run-to-run variation sufficiently."
  echo "Inspect the per-file deltas and GPU determinism before accepting a baseline."
  exit 1
fi

echo "RESULT: PASSED -- noise floor is rtol=${noise_floor}"
if [[ "$compare_status" -ne 0 ]]; then
  echo "(the default rtol=1e-6 comparison above did NOT pass; use the value below)"
fi
echo
echo "Keep ${ref_dir}/summary.json as the reference (see regression/README.md)."
echo "After applying the code change:"
echo
echo "  sbatch ${single_run} ${base_dir}/after"
echo "  python ${compare} compare ${ref_dir}/summary.json ${base_dir}/after/summary.json --rtol <tolerance> --atol 1e-12"
echo
echo "Choose a tolerance comfortably above ${noise_floor} -- roughly 10x is a"
echo "starting margin. Confirm the chosen tolerance with repeat runs; this is"
echo "a compact statistical check, not an exhaustive elementwise comparison."
