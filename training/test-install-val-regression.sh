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
#clt   sbatch test-install-val-regression.sh <base_dir>
#clt hardwired base_dir now
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

#clt base_dir="${1:?usage: $0 <base_dir>}"
#clt base_dir="${base_dir%/}"
base_dir=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-basedir

rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd "$rundir"

single_run="./test-val-regression-rtma_ok_sfc-1GPU.sh"
compare="../scripts/check_val_regression.py"

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
python "$compare" "$ref_dir" "$noise_dir"
compare_status=$?
set -e

echo
echo "------------------------------------------------------------"
echo "noise floor: smallest relative tolerance at which the two"
echo "identical runs still agree"
echo "------------------------------------------------------------"

noise_floor=""
for tol in 1e-12 1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2; do
  if python "$compare" "$ref_dir" "$noise_dir" \
       --rtol "$tol" --atol 1e-12 >/dev/null 2>&1; then
    noise_floor="$tol"
    break
  fi
done

echo
if [[ -z "$noise_floor" ]]; then
  echo "RESULT: FAILED -- the two identical runs disagree even at rtol=1e-2."
  echo
  echo "Something substantial in the training path is unseeded (dropout is the"
  echo "usual cause; train_module.py sets no torch.manual_seed and no cuDNN"
  echo "determinism flags). This harness cannot gate anything until that is"
  echo "fixed. Re-read the per-file deltas printed above to see which array"
  echo "moved, then add seeding before capturing a reference."
  exit 1
fi

echo "RESULT: PASSED -- noise floor is rtol=${noise_floor}"
if [[ "$compare_status" -ne 0 ]]; then
  echo "(the default rtol=1e-6 comparison above did NOT pass; use the value below)"
fi
echo
echo "Keep ${ref_dir} as the reference. After applying the code change:"
echo
echo "  sbatch ${single_run} ${base_dir}/after"
echo "  python ${compare} ${ref_dir} ${base_dir}/after --rtol <tolerance>"
echo
echo "Choose a tolerance comfortably above ${noise_floor} -- roughly 10x is a"
echo "reasonable margin. Anything larger than that is a real regression, not"
echo "numerical noise."
