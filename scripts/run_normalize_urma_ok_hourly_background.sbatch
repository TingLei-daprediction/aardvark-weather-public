#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J norm-urma-ok-hourly-bg
#SBATCH -p u1-compute
#SBATCH -q batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --open-mode=truncate
#SBATCH -t 04:00:00
#SBATCH -o normalize-urma-ok-hourly-bg.out
#SBATCH -e normalize-urma-ok-hourly-bg.err

# Run after all 24 raw-background array tasks succeed. Missing months are fatal and checked
# before any normalized output is written.

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

repo="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
data_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data"

cd "$repo"

python scripts/normalize_background.py \
  --data_dir "$data_dir" \
  --era5_mode rtma_ok_sfc \
  --subdir urma \
  --background_mode hourly \
  --chunk_frames 24 \
  --years 2022 2023

echo "All 24 hourly background months normalized; aggregate raw statistics written."
