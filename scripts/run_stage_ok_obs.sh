#!/bin/bash
# Stage the colleague-processed OK surface obs months into the loader layout.
# Usage:  ./run_stage_ok_obs.sh            # stage 2022-01 and 2022-02
#         ./run_stage_ok_obs.sh --dry_run  # checks only, write nothing
#
# Each month receives its own lon/lat/alt coordinate files. Station membership,
# ordering, and count may therefore change between months.
set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-annette/aardvark_OK/OK_data"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data"
out_dir="${data_root}/hadisd_processed"

# Months whose files should be staged.
months=(2022-01 2022-02)

for ym in "${months[@]}"; do
  src_dir="${src_root}/${ym//-/}/surface_processed"
  echo "==================== ${ym} ===================="
  python "${script_dir}/stage_ok_obs_month.py" \
    --src_dir "$src_dir" \
    --year_month "$ym" \
    --out_dir "$out_dir" \
    --norm_dir "${data_root}/norm_factors" \
    "$@"
done

# stage_ok_obs_month.py also writes station-dependent monthly mean/std vectors derived
# from the same month's values matrix, so coordinates, values, and norms share one order.
# Use --obs_norm_mode monthly in train_module.py to activate them. Existing static/global
# runs continue to use the untagged mean_hadisd_<var>.npy/std_hadisd_<var>.npy files.