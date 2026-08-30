#!/bin/bash
# Two-GPU RTMA-OK encoder training using 12 non-contiguous months from 2022-2023.
#
# Training months are read from rtma_ok_train_months_2022_2023_alternating.txt:
#   2022: January, March, May, July, September, November
#   2023: February, April, June, August, October, December
#
# January 2023 is reserved for validation and is not present in the training manifest.
# The outer training date range intentionally spans both complete years; the manifest is
# the authoritative filter that determines which monthly files and samples are loaded.
#
# Based on the existing static-normalization, reduced-Grid-B RTMA-OK training setup.
# This version uses batch 2.
# --batch_size is per GPU, so two DDP ranks give an effective global batch of 4.
#
# Prerequisite for --obs_norm_mode static: compute scalar observation norms from the same
# training-month manifest with scripts/compute_ok_obs_scalar_norms.py. Target/background
# normalization factors must likewise be consistent with the intended training period.
#SBATCH -A fv3-cam
#SBATCH -J av-12mon-sel-ok-bs2-gB336
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 6:00:00
#SBATCH -o new-12mon-selected-static-bs2-gB336.%j.out
#SBATCH -e new-12mon-selected-static-bs2-gB336.%j.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd "$rundir"

output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-12mon-selected-static-gridB336-bsize2/"
weights_dir="checking-point"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data/"
aux_data_root="${data_root}"
model_data_dir="${data_root}/model_data_dir"
train_months_file="${rundir}/rtma_ok_train_months_2022_2023_alternating.txt"

for v in data_root aux_data_root model_data_dir train_months_file; do
  [[ -n "${!v}" ]] || { echo "ERROR: $v is not set; edit this script first."; exit 1; }
done
[[ -f "$train_months_file" ]] || {
  echo "ERROR: training-month manifest not found: $train_months_file"
  exit 1
}

python ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  --weights_dir "$weights_dir" \
  --master_port 12362 \
  --decoder vit_assimilation \
  --loss rmse \
  --diff 0 \
  --obs_set rtma_surface \
  --era5_mode rtma_ok_sfc \
  --in_channels 24 \
  --int_channels 24 \
  --mode assimilation \
  --lr 5e-5 \
  --batch_size 2 \
  --start_ind 0 \
  --end_ind 5 \
  --epoch 900 \
  --cmd_init_ls 2e-4 \
  --data_path "$data_root" \
  --aux_data_path "$aux_data_root" \
  --model_data_path "$model_data_dir" \
  --assim_train_start_date "2022-01-01 00:00" \
  --assim_train_end_date "2023-12-31 23:00" \
  --assim_train_months_file "$train_months_file" \
  --assim_val_start_date "2023-01-01 00:00" \
  --assim_val_end_date "2023-01-31 23:00" \
  --time_freq 1H \
  --obs_norm_mode static \
  --grid_config ../aardvark/grid_config_ok_reduced_grid_B.yaml
