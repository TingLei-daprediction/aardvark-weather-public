#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J av-ok-sfc
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 8:00:00
#SBATCH -o new-aardvark-gpu-train-ok.%j.out
#SBATCH -e new-aardvark-gpu-train-ok.%j.err



source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
set -euo pipefail
rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd $rundir



# TODO: fill in before submitting.
# data_root       = --data_path: root holding urma/ (targets, backgrounds, grid axes)
#                   and hadisd_processed/ (surface obs)
# aux_data_root   = --aux_data_path: root holding norm_factors/ (same dir as data_root
#                   in our setup)
# model_data_dir  = --model_data_path: root holding grid_lon_lat/urma_x_ok.npy and
#                   urma_y_ok.npy (copied from data_root/urma/ after preprocessing)

output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output/"
weights_dir="checking-point"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data/"
aux_data_root="${data_root}"
model_data_dir="${data_root}/model_data_dir"


for v in data_root aux_data_root model_data_dir; do
  [[ -n "${!v}" ]] || { echo "ERROR: $v is not set; edit this script first."; exit 1; }
done
python ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  --weights_dir "$weights_dir" \
  --master_port 12348 \
  --decoder vit_assimilation \
  --loss rmse \
  --diff 0 \
  --obs_set rtma_surface \
  --era5_mode rtma_ok_sfc \
  --in_channels 24 \
  --int_channels 24 \
  --mode assimilation \
  --lr 5e-4 \
  --batch_size 2 \
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
