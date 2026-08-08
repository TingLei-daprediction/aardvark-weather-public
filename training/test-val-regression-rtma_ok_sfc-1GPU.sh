#!/bin/bash
# Minimal validation-path regression reference for the RTMA OK encoder.
#
# Warm-starts from an existing checkpoint and runs exactly ONE epoch over a
# two-day window. Warm start is what makes the run reproducible: train_module.py
# seeds nothing (no torch.manual_seed, no cuDNN determinism flags), so a cold
# run randomises the model weights and two runs of identical code disagree.
# Loading a checkpoint pins the weights, and the training DistributedSampler is
# seeded via set_epoch(), so the only remaining run-to-run variation comes from
# non-deterministic cuDNN backward kernels.
#
# One epoch rather than zero: losses_*.npy is written only inside the epoch
# loop, so --epoch 0 would record no loss at all and leave nothing to compare.
#
# SINGLE GPU is deliberate. On more than one GPU the planned change replaces the
# rank-local validation mean with a globally reduced one, so losses_*.npy is
# EXPECTED to change there. The 1-GPU path skips the collective entirely
# (world_size > 1 guard), which makes it a strict "results must not move" gate.
#
# Usage:
#   sbatch test-val-regression-rtma_ok_sfc-1GPU.sh <output_dir>
#
# Run it TWICE into two different directories on UNCHANGED code first. The
# difference between those two runs is the numerical noise floor, and it sets
# the tolerance for the real before/after comparison. If run A and run B differ
# by more than roundoff, determinism is not pinned and nothing downstream is
# meaningful -- see scripts/check_val_regression.py.
#SBATCH -A fv3-cam
#SBATCH -J av-ok-sfc-valreg
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 00:30:00
#SBATCH -o new-aardvark-valreg.%j.out
#SBATCH -e new-aardvark-valreg.%j.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

output_dir="${1:?usage: $0 <output_dir>}"
[[ "$output_dir" == */ ]] || output_dir="${output_dir}/"

rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd "$rundir"

source_output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-2gpu-lr5e-5/"
data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data/"
aux_data_root="${data_root}"
model_data_dir="${data_root}/model_data_dir"

# Checkpoints are written only when validation loss improves, so the
# numerically latest epoch_* file is also the best checkpoint from that run.
checkpoint="$(find "$source_output_dir" -maxdepth 1 -type f -name 'epoch_*' | sort -V | tail -n 1)"
if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
  echo "ERROR: no epoch_* checkpoint found under $source_output_dir" >&2
  exit 1
fi
echo "Warm-start checkpoint: $checkpoint"
echo "Output directory:      $output_dir"

# Flags are copied verbatim from
# new-tl-train-encoder-rtma_ok_sfc-2GPU-lr5e-5-warmstart.sh except for:
#   --epoch 1                     one epoch, so the run is short and comparable
#   --assim_*_date 01-01..01-02   two days instead of the full month
#   --master_port 12451           distinct port, so this can run alongside training
python ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  --weights_dir "$checkpoint" \
  --master_port 12451 \
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
  --epoch 1 \
  --cmd_init_ls 2e-4 \
  --data_path "$data_root" \
  --aux_data_path "$aux_data_root" \
  --model_data_path "$model_data_dir" \
  --assim_train_start_date 2022-01-01 \
  --assim_train_end_date 2022-01-02 \
  --assim_val_start_date 2022-01-01 \
  --assim_val_end_date 2022-01-02 \
  --time_freq 1H \
  --grid_config ../aardvark/grid_config_ok.yaml

echo "Done. Compare with:"
echo "  python ../scripts/check_val_regression.py <reference_dir> ${output_dir}"
