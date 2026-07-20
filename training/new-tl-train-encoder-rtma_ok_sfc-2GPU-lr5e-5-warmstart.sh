#!/bin/bash
# Warm-start another 300 epochs from the latest best checkpoint written by the
# completed 2-GPU lr=5e-5 run. The current Aardvark trainer restores model
# weights only: optimizer state and displayed epoch numbering restart from zero.
# Outputs go to a new directory so the original run is preserved.
#SBATCH -A fv3-cam
#SBATCH -J av-ok-sfc-2gpu-warm
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 24:00:00
#SBATCH -o new-aardvark-gpu-train-ok-2gpu-warmstart.%j.out
#SBATCH -e new-aardvark-gpu-train-ok-2gpu-warmstart.%j.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd "$rundir"

source_output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-2gpu-lr5e-5/"
output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-2gpu-lr5e-5-warmstart/"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data/"
aux_data_root="${data_root}"
model_data_dir="${data_root}/model_data_dir"

# Checkpoints are written only when validation loss improves. Therefore, the
# numerically latest epoch_* file is also the best checkpoint from the run.
checkpoint="$(find "$source_output_dir" -maxdepth 1 -type f -name 'epoch_*' | sort -V | tail -n 1)"
if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
  echo "ERROR: no epoch_* checkpoint found under $source_output_dir" >&2
  exit 1
fi
echo "Warm-start checkpoint: $checkpoint"
echo "New output directory: $output_dir"

python ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  --weights_dir "$checkpoint" \
  --master_port 12351 \
  --decoder vit_assimilation \
  --loss rmse \
  --diff 0 \
  --obs_set rtma_surface \
  --era5_mode rtma_ok_sfc \
  --in_channels 24 \
  --int_channels 24 \
  --mode assimilation \
  --lr 5e-5 \
  --batch_size 1 \
  --start_ind 0 \
  --end_ind 5 \
  --epoch 300 \
  --cmd_init_ls 2e-4 \
  --data_path "$data_root" \
  --aux_data_path "$aux_data_root" \
  --model_data_path "$model_data_dir" \
  --assim_train_start_date 2022-01-01 \
  --assim_train_end_date 2022-01-31 \
  --assim_val_start_date 2022-01-01 \
  --assim_val_end_date 2022-01-31 \
  --time_freq 1H \
  --grid_config ../aardvark/grid_config_ok.yaml
