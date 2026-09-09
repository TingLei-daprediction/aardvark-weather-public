#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J prep-urma-ok-hourly-bg
#SBATCH -p u1-compute
#SBATCH -q batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-23%4
#SBATCH --open-mode=truncate
#SBATCH -t 04:00:00
#SBATCH -o prep-urma-ok-hourly-bg.%A_%a.out
#SBATCH -e prep-urma-ok-hourly-bg.%A_%a.err

# Build hour-matched URMA first-guess backgrounds without changing the existing daily-00z
# background, target, or target-normalization files.
#
# Each array task writes one raw YYYY-MM product:
#   urma/background_raw_hourly_rtma_ok_sfc_1_YYYY-MM.memmap
# Submit run_normalize_urma_ok_hourly_background.sbatch with an afterok dependency on this array
# to create normalized files and aggregate statistics without array-task write races.

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

repo="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
# Archive layout:
#   <input_dir>/<YYYYMMDD>/urma2p5.t<HH>z.2dvarges_OK.grb2
input_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-raw-2p5"
data_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data"

name_root="urma"

cd "$repo"

month=$(python - "$SLURM_ARRAY_TASK_ID" <<'EOF'
import sys
offset = int(sys.argv[1])
year = 2022 + offset // 12
month = offset % 12 + 1
print(f"{year}-{month:02d}")
EOF
)
echo "Array task $SLURM_ARRAY_TASK_ID: $month"

test -f "$data_dir/$name_root/${name_root}_x_ok.npy"
test -f "$data_dir/$name_root/${name_root}_y_ok.npy"
test -f "$data_dir/norm_factors/mean_rtma_ok_sfc_1.npy"
test -f "$data_dir/norm_factors/std_rtma_ok_sfc_1.npy"

# Step 1: decode one GES for every valid hour. --background_only prevents target rewrites.
python scripts/prep_urma_ok_hourly.py \
  --input_dir "$input_dir" \
  --output_dir "$data_dir" \
  --start "$month" --end "$month" \
  --name_root "$name_root" \
  --grid_source files \
  --background_only \
  --background_mode hourly

echo "Raw hourly background complete for $month. Existing daily files were not changed."
