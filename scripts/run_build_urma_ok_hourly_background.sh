#!/bin/bash
# Submit the complete RTMA-OK hourly-background build and quality-control chain.
# Run this from a login node; this script submits jobs and exits.
#
# Default (recommended): recompute 2022-only target/static-observation norms, re-normalize the
# daily control backgrounds, build all 24 hourly raw months, normalize them with the target
# factors, and run separate 2022/2023 leakage diagnostics.
#
# If the 2022-only norms have already been created and verified, skip that first job with:
#   RECOMPUTE_TRAINING_NORMS=0 bash scripts/run_build_urma_ok_hourly_background.sh

set -euo pipefail

repo="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
recompute_training_norms="${RECOMPUTE_TRAINING_NORMS:-1}"
norms_script="${repo}/scripts/run_compute_rtma_ok_2022_norms.sbatch"
prep_script="${repo}/scripts/run_prep_urma_ok_hourly_background.sbatch"
normalize_script="${repo}/scripts/run_normalize_urma_ok_hourly_background.sbatch"
diagnose_script="${repo}/scripts/run_diagnose_urma_ok_hourly_background.sbatch"

for required in "$norms_script" "$prep_script" "$normalize_script" "$diagnose_script"; do
  [[ -f "$required" ]] || { echo "ERROR: required script not found: $required"; exit 1; }
done

case "$recompute_training_norms" in
  0|1) ;;
  *) echo "ERROR: RECOMPUTE_TRAINING_NORMS must be 0 or 1"; exit 1 ;;
esac

if [[ "$recompute_training_norms" == 1 ]]; then
  norms_submit=$(sbatch --parsable "$norms_script")
  norms_job=${norms_submit%%;*}
  echo "Submitted 2022-only target/observation norms: $norms_job"
  prep_submit=$(sbatch --parsable --dependency="afterok:${norms_job}" "$prep_script")
else
  echo "Skipping norm recomputation; existing target/static-observation norms will be used."
  prep_submit=$(sbatch --parsable "$prep_script")
fi

prep_job=${prep_submit%%;*}
normalize_submit=$(sbatch --parsable --dependency="afterok:${prep_job}" "$normalize_script")
normalize_job=${normalize_submit%%;*}
diagnose_submit=$(sbatch --parsable --dependency="afterok:${normalize_job}" "$diagnose_script")
diagnose_job=${diagnose_submit%%;*}

echo "Submitted 24-month hourly raw-background array: $prep_job"
echo "Submitted hourly normalization job:             $normalize_job"
echo "Submitted per-year leakage/quality gate:        $diagnose_job"
echo "Training may start only after job $diagnose_job finishes successfully."
