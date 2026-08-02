#!/bin/bash
# Single-sample encoder INFERENCE at a specified analysis time, reusing train_module.py
# UNCHANGED (clone of new-tl-train-encoder-rtma_ok_sfc-2GPU-lr5e-5.sh with 4 argument
# changes). How it works: trainer.train() runs one eval pass over the val_loader BEFORE
# the training loop (trainer.py eval_epoch), so --epoch 0 = eval-only. The trainer loads
# --weights_dir (a checkpoint FILE here, strict=True) and eval_epoch dumps physical-unit
#   unnorm_preds_0.npy / unnorm_targets_0.npy   shape (1, nlat, nlon, 5)
# into --output_dir, readable directly by scripts/plot_encoder_field.py.
#
# Date-window quirk: WeatherDataset.__len__ = len(dates) - 2, so the val window must span
# infer_time .. infer_time+2h (3 hourly stamps -> dataset length 1 -> exactly ONE sample,
# = infer_time). The +2h stamps are never evaluated but their month must be staged, so
# infer_time must be <= 21:00 UTC on the last staged day of a month (else +2h crosses
# into an un-staged month and the loader fails at init).
#
# ONE time per run: with a longer window eval_epoch saves only the LAST val batch, and
# the val sampler shuffles -- do not widen the window expecting multi-time output.
#SBATCH -A fv3-cam
#SBATCH -J av-ok-sfc-infer
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 00:30:00
#SBATCH -o new-aardvark-gpu-infer-ok.%j.out
#SBATCH -e new-aardvark-gpu-infer-ok.%j.err

# -------- inference inputs: EDIT THESE TWO --------
# Analysis time T (UTC, on the 1H cadence, minute must be :00)
infer_time="2022-01-15T06:00"
# Trained checkpoint FILE (epoch_N), e.g. from the lr5e-5 run
checkpoint="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-output-2gpu-lr5e-5/epoch_250"
# --------------------------------------------------

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
# The submitting shell's environment leaks into the job (sbatch --export=ALL default);
# a loaded spack-stack module sets PYTHONPATH to a python3.11 numpy that shadows the
# conda env's own packages. Clear it so only aardvark-env is visible.
unset PYTHONPATH
set -euo pipefail
rundir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/training/"
cd $rundir

data_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data/"
aux_data_root="${data_root}"
model_data_dir="${data_root}/model_data_dir"

# One output dir per inference time; unnorm_preds_0.npy etc. land here.
time_token=${infer_time//[-:]/}          # e.g. 20220115T0600
output_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data/dr-rtma/OK-infer-${time_token}/"

# The trainer only WARNS and skips on a bad weights path (running random weights);
# fail here instead so a typo can never produce plausible-looking garbage.
[[ -f "$checkpoint" ]] || { echo "ERROR: checkpoint file not found: $checkpoint"; exit 1; }
for v in data_root aux_data_root model_data_dir; do
  [[ -n "${!v}" ]] || { echo "ERROR: $v is not set; edit this script first."; exit 1; }
done
mkdir -p "$output_dir"

# Window end = T + 2 hourly steps (see header). Train window is unused for learning at
# --epoch 0 but the loader is still constructed, so keep it identically tiny.
end_window=$(date -u -d "${infer_time/T/ } +2 hours" +"%Y-%m-%dT%H:%M")
echo "Inference time: ${infer_time}  (val window ${infer_time}..${end_window})"
echo "Checkpoint:     ${checkpoint}"
echo "Output dir:     ${output_dir}"

python ../aardvark/train_module.py \
  --output_dir "$output_dir" \
  --weights_dir "$checkpoint" \
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
  --batch_size 1 \
  --start_ind 0 \
  --end_ind 5 \
  --epoch 0 \
  --cmd_init_ls 2e-4 \
  --data_path "$data_root" \
  --aux_data_path "$aux_data_root" \
  --model_data_path "$model_data_dir" \
  --assim_train_start_date "$infer_time" \
  --assim_train_end_date "$end_window" \
  --assim_val_start_date "$infer_time" \
  --assim_val_end_date "$end_window" \
  --time_freq 1H \
  --grid_config ../aardvark/grid_config_ok.yaml

echo "Done. Outputs in ${output_dir}:"
ls -l "${output_dir}"unnorm_preds_0.npy "${output_dir}"unnorm_targets_0.npy
echo "Plot a channel (0=t2m 1=q2m 2=sp 3=u10 4=v10) with:"
echo "  python ../scripts/plot_encoder_field.py --run_dir ${output_dir} --rank 0 --sample_index 0 --channel 0"
