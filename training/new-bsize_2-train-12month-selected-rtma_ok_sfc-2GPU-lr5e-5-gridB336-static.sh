#!/bin/bash
# Two-GPU RTMA-OK encoder daily-00z/hourly-background A/B experiment.
#
# Training: all of 2022. Validation: all of 2023. The year holdout is strict: no validation
# timestamp appears in training. Both A/B arms use this same script, data, optimizer settings,
# and number of steps; only BACKGROUND_MODE and the output directory differ.
# Submit fresh arms with:
#   sbatch --export=ALL,BACKGROUND_MODE=daily_00z <this-script>
#   sbatch --export=ALL,BACKGROUND_MODE=hourly <this-script>
#
# Based on the existing static-normalization, reduced-Grid-B RTMA-OK training setup.
# This version uses batch 2.
# --batch_size is per GPU, so two DDP ranks give an effective global batch of 4.
#
# REQUIRED: recompute static observation and target normalization factors from 2022 training
# data only. Do not include any 2023 validation samples. Both arms must use identical factors.
#SBATCH -A fv3-cam
#SBATCH -J av-bg-ab-ok-bs2-gB336
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 6:00:00
#SBATCH -o new-bg-ab-static-bs2-gB336.%j.out
#SBATCH -e new-bg-ab-static-bs2-gB336.%j.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd "$rundir"

background_mode="${BACKGROUND_MODE:-daily_00z}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
case "$background_mode" in
  daily_00z|hourly) ;;
  *) echo "ERROR: BACKGROUND_MODE must be daily_00z or hourly"; exit 1 ;;
esac

output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-train2022-val2023-${background_mode}-static-gridB336-bsize2/"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data/"
aux_data_root="${data_root}"
model_data_dir="${data_root}/model_data_dir"
train_months_file="${rundir}/rtma_ok_train_months_2022.txt"
val_months_file="${rundir}/rtma_ok_val_months_2023.txt"

for v in data_root aux_data_root model_data_dir train_months_file val_months_file; do
  [[ -n "${!v}" ]] || { echo "ERROR: $v is not set; edit this script first."; exit 1; }
done
[[ -f "$train_months_file" ]] || {
  echo "ERROR: training-month manifest not found: $train_months_file"
  exit 1
}
[[ -f "$val_months_file" ]] || {
  echo "ERROR: validation-month manifest not found: $val_months_file"
  exit 1
}

echo "A/B background mode: $background_mode"
resume_args=(--resume_training 0)
if [[ -n "$resume_checkpoint" ]]; then
  [[ -d "$output_dir" ]] || {
    echo "ERROR: resume output directory does not exist: $output_dir"
    exit 1
  }
  [[ -f "$resume_checkpoint" ]] || {
    echo "ERROR: resume checkpoint does not exist: $resume_checkpoint"
    exit 1
  }
  case "$resume_checkpoint" in
    "$output_dir"*) ;;
    *) echo "ERROR: resume checkpoint must be inside $output_dir"; exit 1 ;;
  esac
  resume_args=(--weights_dir "$resume_checkpoint" --resume_training 1)
  echo "Resuming $background_mode from: $resume_checkpoint"
else
  echo "Training: 2022; validation: 2023; fresh output: $output_dir"
  [[ ! -e "$output_dir" ]] || {
    echo "ERROR: fresh-initialization output already exists: $output_dir"
    echo "Set RESUME_CHECKPOINT to its checkpoint_last file or choose a new output path."
    exit 1
  }
fi

python ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  "${resume_args[@]}" \
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
  --seed 202209 \
  --cmd_init_ls 2e-4 \
  --data_path "$data_root" \
  --aux_data_path "$aux_data_root" \
  --model_data_path "$model_data_dir" \
  --assim_train_start_date "2022-01-01 00:00" \
  --assim_train_end_date "2022-12-31 23:00" \
  --assim_train_months_file "$train_months_file" \
  --assim_val_start_date "2023-01-01 00:00" \
  --assim_val_end_date "2023-12-31 23:00" \
  --assim_val_months_file "$val_months_file" \
  --assim_val_stride 7 \
  --time_freq 1H \
  --obs_norm_mode static \
  --background_mode "$background_mode" \
  --grid_config ../aardvark/grid_config_ok_reduced_grid_B.yaml
