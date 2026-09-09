#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J diagnose-ok-hourly-bg
#SBATCH -p u1-compute
#SBATCH -q batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --open-mode=truncate
#SBATCH -t 02:00:00
#SBATCH -o diagnose-ok-hourly-bg.out
#SBATCH -e diagnose-ok-hourly-bg.err

# Mandatory CPU-only quality/leakage gate before hourly-background training.

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

repo="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data"
output_dir="${data_root}/diagnostics/hourly-background"

cd "$repo"
mkdir -p "$output_dir"

# Report training and validation years separately. Every second day gives about 183 days per
# report; set this to 1 for the full archive after the initial gate passes.
python scripts/diagnose_hourly_background.py \
  --data_root "$data_root" \
  --start 2022-01 \
  --end 2022-12 \
  --sample_every_days 2 \
  --grid_config aardvark/grid_config_ok_reduced_grid_B.yaml \
  --csv_out "$output_dir/hourly_background_2022_every_2_days.csv"

python scripts/diagnose_hourly_background.py \
  --data_root "$data_root" \
  --start 2023-01 \
  --end 2023-12 \
  --sample_every_days 2 \
  --grid_config aardvark/grid_config_ok_reduced_grid_B.yaml \
  --csv_out "$output_dir/hourly_background_2023_every_2_days.csv"

echo "Hourly-background quality/leakage gate passed."
