#!/bin/bash
# GRID B A/B TEST -- reduced inner ViT grid: 336 x 174 instead of 384 x 192.
#
# Identical to new-tl-train-encoder-rtma_ok_sfc-2GPU-lr5e-5.sh in EVERY training argument.
# The ONLY functional change is --grid_config, which points at
# aardvark/grid_config_ok_reduced_grid_B.yaml (int_x 336, int_y 174 instead of 384/192).
# Job name, output_dir, master_port and log names are changed only so this can run alongside
# the baseline without collision. Port map for the OK runs: 12350 baseline, 12351 warmstart,
# 12352 warmstart2, 12360 this run, 12361 its warmstart.
#
# Rationale and full analysis: docs/note_grid_b_sizing.md
#   Grid A (OK) is 331 x 171. 336 = 112*3 and 174 = 58*3, so 336 x 174 is the smallest box that
#   is even, >= Grid A, and exactly divisible by the assimilation ViT patch_size=3.
#   Expected: 6496 tokens instead of 8192 (-21%); quadratic attention terms ~0.63x (-37%);
#   linear terms (QKV/MLP/norm/patch-decode) ~0.79x (-21%), so TOTAL wall time falls by less
#   than 37%. A->B nearest-neighbour resample ratio improves from 1.160x to 1.015x.
#
# THIS IS AN EXPERIMENT, NOT A KNOWN WIN. The larger grid gives the transformer more latent
# tokens and therefore more capacity, so the smaller grid trades latent resolution for
# geometric alignment and cost. Compare skill, not just speed.
#
# ---------------------------------------------------------------------------------------------
# EXPERIMENT PROTOCOL -- read before running
#
# 1. FRESH START ON BOTH SIDES. Do NOT warm-start this from a 384 x 192 checkpoint: the token
#    grid changes (8192 -> 6496), so pos_embed is [1,8192,512] in the checkpoint vs [1,6496,512]
#    in this model. load_state_dict raises a size-mismatch error on that key even with
#    strict=False (strict only governs missing/unexpected keys; shape mismatches always raise).
#    The baseline must also be re-run from scratch for the comparison to be valid.
#
# 2. DO NOT change the interpolation mode in this test. models.py:516/:532 use
#    nn.functional.interpolate with no mode argument (i.e. mode="nearest"). Making that explicit
#    is worth doing -- as a SEPARATE commit. Changing it here would confound the Grid-B result.
#
# 3. NO SEED CONTROL EXISTS. There is no --seed argument and no torch.manual_seed anywhere in
#    aardvark/. Run-to-run variance is therefore uncontrolled, and a single baseline-vs-variant
#    pair may not be conclusive for small skill differences. Treat large cost differences as
#    reliable and small skill differences as provisional; repeat if the skill call is close.
#
# 4. MEASURE: wall time per batch and per epoch; peak GPU memory; training loss vs OPTIMIZER
#    STEP (not vs epoch); RMSE and bias per variable; fine-scale spatial structure, especially
#    over terrain.
# ---------------------------------------------------------------------------------------------
#
# 2-GPU single-node RTMA-OK encoder run with CONSTANT lr=5e-5 (the assimilation branch has no LR
# schedule, so --lr is the constant rate for the whole run). --batch_size is PER PROCESS, so it
# stays 1; DDP averages gradients across the 2 ranks (mp.spawn world_size auto-detects 2 GPUs).
# NOTE: must stay single-node -- ddp_setup hardcodes MASTER_ADDR=localhost (train_module.py),
# so multi-node ranks cannot rendezvous through this launch path.
#SBATCH -A fv3-cam
#SBATCH -J av-3month-ok-sfc-2gpu-lr5e-5-gB336
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 6:00:00
#SBATCH -o new-bsize_2-3mon-ob_mode_static-gpu-train-ok-2gpu-lr5e-5-gB336.%j.out
#SBATCH -e new-bsize_2-3mon-ob_mode_static-gpu-train-ok-2gpu-lr5e-5-gB336.%j.err



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

# Separate output_dir from the 384x192 baseline so the two runs' checkpoints and logs do not
# overwrite each other.
output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-3mon-static-gridB336/"
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
  --master_port 12360 \
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
  --assim_train_start_date 2022-01-01 \
  --assim_train_end_date 2022-03-30 \
  --assim_val_start_date 2022-01-01 \
  --assim_val_end_date 2022-01-31 \
  --time_freq 1H \
  --obs_norm_mode static \
  --grid_config ../aardvark/grid_config_ok_reduced_grid_B.yaml
