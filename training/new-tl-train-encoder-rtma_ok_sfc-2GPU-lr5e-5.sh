#!/bin/bash
# 2-GPU single-node RTMA-OK encoder run with CONSTANT lr=5e-5 (10x lower than the 5e-4
# baseline in new-tl-train-encoder-rtma_ok_sfc-2GPU.sh; the assimilation branch has no LR
# schedule, so --lr is the constant rate for the whole run). Separate output_dir and
# master_port so it can run alongside the baseline. --batch_size is PER PROCESS, so it
# stays 1; DDP averages gradients across the 2 ranks (mp.spawn world_size auto-detects
# 2 GPUs). NOTE: must stay single-node -- ddp_setup hardcodes MASTER_ADDR=localhost
# (train_module.py), so multi-node ranks cannot rendezvous through this launch path.
#SBATCH -A fv3-cam
#SBATCH -J av-ok-sfc-2gpu-lr5e-5
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 24:00:00
#SBATCH -o new-aardvark-gpu-train-ok-2gpu-lr5e-5.%j.out
#SBATCH -e new-aardvark-gpu-train-ok-2gpu-lr5e-5.%j.err



source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
# The submitting shell's environment leaks into the job (sbatch --export=ALL default);
# a loaded spack-stack module sets PYTHONPATH to a python3.11 numpy that shadows the
# conda env's own packages. Clear it so only aardvark-env is visible.
unset PYTHONPATH
set -euo pipefail
rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd $rundir

# data_root       = --data_path: root holding urma/ (targets, backgrounds, grid axes)
#                   and hadisd_processed/ (surface obs)
# aux_data_root   = --aux_data_path: root holding norm_factors/ (same dir as data_root
#                   in our setup)
# model_data_dir  = --model_data_path: root holding grid_lon_lat/urma_x_ok.npy and
#                   urma_y_ok.npy (copied from data_root/urma/ after preprocessing)

output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-2gpu-lr5e-5/"
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
  --master_port 12350 \
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
