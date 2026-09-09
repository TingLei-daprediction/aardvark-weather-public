#!/bin/bash
# RTMA-OK encoder training with hour-matched backgrounds.
#
# Fit period:       2022-01-01 00:00 through 2022-12-31 23:00
# Holdout period:   2023-01-01 00:00 through 2023-12-31 23:00
# Validation sample: every seventh timestamp (all 24 UTC hours; 1,252 samples)
#
# This is derived from the three-month Grid-B/static-normalization run. It uses a distinct
# output directory and never loads the older three-month checkpoint. A model trained with the
# daily 00 UTC background is not scientifically valid for this hourly-background experiment.
#SBATCH -A fv3-cam
#SBATCH -J av-2022-ok-hourly-bg
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 6:00:00
#SBATCH -o train-2022-ok-hourly-bg-gB336.%j.out
#SBATCH -e train-2022-ok-hourly-bg-gB336.%j.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

repo="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
rundir="${repo}/training"
script_path="${rundir}/new-bsize_2-train-2022-rtma_ok_sfc-2GPU-lr5e-5-gridB336-static-hourly-background.sh"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data/"
aux_data_root="${data_root}"
model_data_dir="${data_root}/model_data_dir"
train_months_file="${rundir}/rtma_ok_train_months_2022.txt"
val_months_file="${rundir}/rtma_ok_val_months_2023.txt"
output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-train2022-val2023-hourly-static-gridB336-bsize2/"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
background_mode="hourly"

cd "$rundir"

for required in "$script_path" "$train_months_file" "$val_months_file"; do
  [[ -f "$required" ]] || { echo "ERROR: required file not found: $required"; exit 1; }
done

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
  echo "Resuming hourly-background training from $resume_checkpoint"
else
  [[ ! -e "$output_dir" ]] || {
    echo "ERROR: fresh-run output already exists: $output_dir"
    echo "Set RESUME_CHECKPOINT to $output_dir/checkpoint_last to resume."
    exit 1
  }
  echo "Starting fresh hourly-background training."
fi

# Fail before requesting model training if any 2022/2023 target, hourly background,
# observation, grid, or normalization file is absent or has an incompatible size.
python ../scripts/check_ok_run_files.py --train_script "$script_path"

python ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  "${resume_args[@]}" \
  --master_port 12363 \
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

