#!/bin/bash
# Warm-start another 300 epochs from the latest best checkpoint written by the reduced-Grid-B
# run (new-tl-train-encoder-rtma_ok_sfc-2GPU-lr5e-5-gridB336.sh, Grid B = 336 x 174).
# Model weights only are restored: optimizer (Adam) moments, displayed epoch numbering, and
# best-loss history all restart from zero -- this is a chained warm start, not a true resume.
# Outputs go to a new directory so the source run is preserved.
#
# CRITICAL -- --grid_config MUST stay grid_config_ok_reduced_grid_B.yaml.
#   The checkpoint's positional embedding is sized to this run's token grid:
#     336 x 174, patch 3  ->  112 x 58 = 6496 tokens  ->  pos_embed [1, 6496, 512]
#   Pointing this at grid_config_ok.yaml (384 x 192 -> 8192 tokens) makes load_state_dict raise
#   a size-mismatch error on pos_embed. That is a genuine safety net, not a bug: strict=False
#   only ignores MISSING and UNEXPECTED keys; a shape mismatch on a key present in both always
#   raises. So a mismatched grid_config fails loudly at startup rather than training silently
#   from a half-loaded model. Do not "fix" such an error by adding strict handling -- fix the
#   config.
#
# A/B FAIRNESS NOTE. The 384 x 192 baseline has been advanced by chained warm starts
# (OK-output-2gpu-lr5e-5 -> -warmstart -> -warmstart2, ~600 epochs total). For the Grid-B
# comparison to mean anything, both arms need the SAME chain depth and the same total optimizer
# steps. Compare loss vs. cumulative optimizer step, not vs. the per-run epoch counter, which
# resets to zero on every warm start.
#
# Rationale for the reduced grid: docs/note_grid_b_sizing.md
# Port map for the OK runs: 12350 baseline, 12351 warmstart, 12352 warmstart2,
#                           12360 gridB336, 12361 this run.
#SBATCH -A fv3-cam
#SBATCH -J av-ok-sfc-2gpu-gB336-warm
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 24:00:00
#SBATCH -o new-aardvark-gpu-train-ok-2gpu-gB336-warmstart.%j.out
#SBATCH -e new-aardvark-gpu-train-ok-2gpu-gB336-warmstart.%j.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd "$rundir"

source_output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-2gpu-lr5e-5-gridB336/"
output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-2gpu-lr5e-5-gridB336-warmstart/"
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
echo "Grid B: 336 x 174 (reduced) -- grid_config must match the source run"

python ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  --weights_dir "$checkpoint" \
  --master_port 12361 \
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
  --grid_config ../aardvark/grid_config_ok_reduced_grid_B.yaml
