#!/bin/bash
# Stage the colleague-processed OK surface obs months into the loader layout.
# Usage:  ./run_stage_ok_obs.sh            # stage 2022-01 and 2022-02
#         ./run_stage_ok_obs.sh --dry_run  # checks only, write nothing
set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-annette/aardvark_OK/OK_data"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data"
out_dir="${data_root}/hadisd_processed"

#jfor ym in 2022-01 2022-02; do
for ym in 2022-01 ; do
  src_dir="${src_root}/${ym//-/}/surface_processed"
  echo "==================== ${ym} ===================="
  python "${script_dir}/stage_ok_obs_month.py" \
    --src_dir "$src_dir" \
    --year_month "$ym" \
    --out_dir "$out_dir" \
    "$@"
done
