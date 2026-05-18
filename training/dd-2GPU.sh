#!/bin/bash
# 2-GPU multi-node training run (1 GPU per node, cleaner logs per task)
#SBATCH -A fv3-cam
#SBATCH -J av-da-2gpu
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 0:30:00
#SBATCH -o dd-2gpu-%t.out
#SBATCH -e dd-2gpu-%t.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
set -euo pipefail
rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd $rundir
output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-output/encoder/"
weights_dir="for-checking-point"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/"
model_data_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/data/"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12348

torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=1 \
  --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  --weights_dir "$weights_dir" \
  --master_port "$MASTER_PORT" \
  --decoder vit_assimilation \
  --loss rmse \
  --diff 0 \
  --era5_mode 4u_sfc \
  --in_channels 235 \
  --out_channels 30 \
  --int_channels 24 \
  --mode assimilation \
  --lr 5e-4 \
  --batch_size 6 \
  --start_ind 0 \
  --end_ind 30 \
  --epoch 100 \
  --weight_per_variable 1 \
  --data_path "$data_root" \
  --aux_data_path "$data_root" \
  --model_data_path "$model_data_dir" \
  --assim_train_start_date 2007-01-02 \
  --assim_train_end_date 2008-12-31 \
  --assim_val_start_date 2019-01-01 \
  --assim_val_end_date 2019-12-31 \
  --time_freq 1D
