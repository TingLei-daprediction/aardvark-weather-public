#!/bin/bash
# Stage the colleague-processed OK surface obs months into the loader layout.
# Usage:  ./run_stage_ok_obs.sh            # stage 2022-01 and 2022-02
#         ./run_stage_ok_obs.sh --dry_run  # checks only, write nothing
#
# Station membership/order may change. First find each variable's largest station
# count over all months, then pad every monthly coordinate/value/norm product to it.
set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#clt src_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-annette/aardvark_OK/OK_data"
src_root="/scratch3/NCEPDEV/gpu-emc-ai/Annette.Gibbs/aardvark_OK/OK_data"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data"
out_dir="${data_root}/hadisd_processed"

# Months whose files should be staged.
#clt for 2022 and 2023
months=({2022..2023}-{01..12})
#clt months=(2022-01 2022-02  2022-03)

declare -A max_stations=()
for ym in "${months[@]}"; do
  src_dir="${src_root}/${ym//-/}/surface_processed"
  while IFS='=' read -r var count; do
    if [[ -z "${max_stations[$var]:-}" || "$count" -gt "${max_stations[$var]}" ]]; then
      max_stations[$var]="$count"
    fi
  done < <(python "${script_dir}/stage_ok_obs_month.py" \
    --src_dir "$src_dir" --year_month "$ym" --out_dir "$out_dir" --print_counts)
done

pad_args=()
for var in tas sh psl u v; do
  if [[ -z "${max_stations[$var]:-}" ]]; then
    echo "ERROR: no maximum station count for ${var}" >&2
    exit 1
  fi
  echo "maximum stations for ${var}: ${max_stations[$var]}"
  pad_args+=(--max_stations "${var}=${max_stations[$var]}")
done

for ym in "${months[@]}"; do
#jfor ym in 2022-01 2022-02; do
  src_dir="${src_root}/${ym//-/}/surface_processed"
  echo "==================== ${ym} ===================="
  python "${script_dir}/stage_ok_obs_month.py" \
    --src_dir "$src_dir" \
    --year_month "$ym" \
    --out_dir "$out_dir" \
    --norm_dir "${data_root}/norm_factors" \
    "${pad_args[@]}" \
    "$@"
done

# stage_ok_obs_month.py also writes station-dependent monthly mean/std vectors derived
# from the same month's values matrix, so coordinates, values, and norms share one order.
# All are padded to the same cross-month dimension for that variable.
# Use --obs_norm_mode monthly in train_module.py to activate them. Existing static/global
# runs continue to use the untagged mean_hadisd_<var>.npy/std_hadisd_<var>.npy files.
