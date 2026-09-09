#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J norms-ok-train2022
#SBATCH -p u1-compute
#SBATCH -q batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --open-mode=truncate
#SBATCH -t 02:00:00
#SBATCH -o norms-ok-train2022.out
#SBATCH -e norms-ok-train2022.err

# Recompute every loader-facing normalization factor from the strict 2022 training split.
# Run this before normalizing either daily or hourly background files for the A/B experiment.

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

repo="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data"
months_file="${repo}/training/rtma_ok_train_months_2022.txt"

cd "$repo"

# Refuse before changing target norms if any daily raw background needed by the A/B control arm
# is absent. This prevents old normalized files from silently surviving a norm-factor change.
for year in 2022 2023; do
  for month in $(seq -w 1 12); do
    raw="$data_root/urma/background_raw_rtma_ok_sfc_1_${year}-${month}.memmap"
    [[ -f "$raw" ]] || { echo "ERROR: missing daily raw background: $raw"; exit 1; }
  done
done

python scripts/compute_rtma_ok_target_norms.py \
  --data_root "$data_root" \
  --months_file "$months_file" \
  --grid_config aardvark/grid_config_ok_reduced_grid_B.yaml \
  --overwrite

python scripts/compute_ok_obs_scalar_norms.py \
  --data_root "$data_root" \
  --months_file "$months_file" \
  --freq_tag 1h

python scripts/normalize_background.py \
  --data_dir "$data_root" \
  --era5_mode rtma_ok_sfc \
  --subdir urma \
  --background_mode daily_00z \
  --years 2022 2023

echo "2022-only target and static-observation norms are ready."
echo "Daily backgrounds were re-normalized with the 2022 target factors."
echo "The hourly preparation job will use the same factors."
