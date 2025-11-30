#!/bin/bash
#SBATCH --account nesccmgmt
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # one launcher task per node
#SBATCH --cpus-per-task=192

set -euo pipefail
source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env

cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-weather-public/training
MASTER_PORT=12348  # change if port conflicts
python3 ../aardvark/train_module.py \
  --output_dir /scratch3/NCEPDEV/fv3-cam/Ting.Lei/runs/encoder/ \
  --master_port ${MASTER_PORT} \
  --decoder vit_assimilation \
  --loss lw_rmse \
  --diff 0 \
  --in_channels 277 \
  --out_channels 24 \
  --int_channels 24 \
  --mode assimilation \
  --lr 5e-4 \
  --batch_size 1 \
  --start_ind 0 \
  --end_ind 24 \
  --epoch 1 \
  --weight_per_variable 1

