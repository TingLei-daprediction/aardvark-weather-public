#!/bin/bash
# Run the same single-time RTMA-OK inference sequentially for every checkpoint below.
#
# Edit the checkpoints=(...) block in USER SETTINGS, then submit this file directly:
#   sbatch dev-mul_checkpoint-infer.sh
#
# Optional submission-time settings:
#   sbatch --export=ALL,INFER_TIME=2023-04-10T23:00 dev-mul_checkpoint-infer.sh
#   sbatch --export=ALL,BACKGROUND_MODE=hourly dev-mul_checkpoint-infer.sh
#
# Every checkpoint is evaluated with identical inference settings. Outputs are isolated under
# a job-specific batch directory and labeled by list position, training-run directory, and
# checkpoint filename. inference_runs.tsv records the exact checkpoint-to-output mapping.
#SBATCH -A gpu-emc-ai
#SBATCH -J av-ok-multi-ckpt-infer
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --open-mode=truncate
#SBATCH -t 02:00:00
#SBATCH -o aardvark-multi-checkpoint-infer.%j.out
#SBATCH -e aardvark-multi-checkpoint-infer.%j.err

source /scratch3/NCEPDEV/fv3-cam/Annette.Gibbs/miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail

repo="/scratch3/NCEPDEV/gpu-emc-ai/Annette.Gibbs/aardvark_OK/aardvark-weather-public"
rundir="${repo}/training"
data_root="/scratch3/NCEPDEV/gpu-emc-ai/Annette.Gibbs/aardvark_OK/rtma_ok_data_2022_2023/"
aux_data_root="${data_root}"
model_data_dir="${data_root}/model_data_dir"
inference_root="/scratch3/NCEPDEV/gpu-emc-ai/Annette.Gibbs/aardvark_OK/rtma_ok_output/multi-checkpoint-inference"

# -----------------------------------------------------------------------------
# USER SETTINGS
# Add or remove full checkpoint paths here. Keep one quoted checkpoint per line.
# All checkpoints in one submission must use the same model/grid, background mode, and
# observation-normalization mode.
# -----------------------------------------------------------------------------
checkpoints=(
  "/scratch3/NCEPDEV/gpu-emc-ai/Annette.Gibbs/aardvark_OK/rtma_ok_output/OK-output-12mon-static-gridB336-warmstart5/epoch_114"
  # "/scratch3/.../another-training-run/epoch_250"
  # "/scratch3/.../another-training-run/epoch_899"
)

infer_time="${INFER_TIME:-2023-04-10T23:00}"
background_mode="${BACKGROUND_MODE:-daily_00z}"
obs_norm_mode="${OBS_NORM_MODE:-static}"
# -----------------------------------------------------------------------------
# END USER SETTINGS
# -----------------------------------------------------------------------------

case "$background_mode" in
  daily_00z|hourly) ;;
  *) echo "ERROR: BACKGROUND_MODE must be daily_00z or hourly"; exit 1 ;;
esac
case "$obs_norm_mode" in
  static|monthly) ;;
  *) echo "ERROR: OBS_NORM_MODE must be static or monthly"; exit 1 ;;
esac

[[ -d "$rundir" ]] || { echo "ERROR: training directory not found: $rundir"; exit 1; }
[[ -d "$data_root" ]] || { echo "ERROR: data root not found: $data_root"; exit 1; }
[[ -d "$model_data_dir" ]] || {
  echo "ERROR: model-data directory not found: $model_data_dir"
  exit 1
}

# Validate and canonicalize the complete block before creating output or running inference. A
# typo in a later entry therefore cannot leave a misleading, partially completed comparison.
checkpoint_count=${#checkpoints[@]}
(( checkpoint_count > 0 )) || {
  echo "ERROR: the checkpoints=(...) block is empty"
  exit 1
}
for ((index = 0; index < checkpoint_count; index++)); do
  checkpoint=${checkpoints[$index]}
  [[ -f "$checkpoint" ]] || {
    echo "ERROR: checkpoint does not exist: $checkpoint"
    exit 1
  }
  checkpoints[$index]=$(readlink -f "$checkpoint")
done

time_token=${infer_time//[-:]/}
job_token=${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)_$$}
batch_root="${inference_root}/${time_token}/${background_mode}/job_${job_token}"
summary_file="${batch_root}/inference_runs.tsv"
mkdir -p "$batch_root"
printf 'index\tlabel\tcheckpoint\toutput_dir\tstatus\n' > "$summary_file"

echo "Multi-checkpoint RTMA-OK inference"
echo "  inference time:     $infer_time"
echo "  checkpoint count:   $checkpoint_count"
echo "  background mode:    $background_mode"
echo "  observation norms:  $obs_norm_mode"
echo "  batch output root:  $batch_root"

cd "$rundir"

for ((index = 0; index < checkpoint_count; index++)); do
  checkpoint=${checkpoints[$index]}
  run_label=$(basename "$(dirname "$checkpoint")")
  checkpoint_label=$(basename "$checkpoint")
  run_label=${run_label//[^[:alnum:]_.-]/_}
  checkpoint_label=${checkpoint_label//[^[:alnum:]_.-]/_}
  printf -v label '%03d__%s__%s' "$((index + 1))" "$run_label" "$checkpoint_label"
  output_dir="${batch_root}/${label}/"

  # The job-specific root prevents cross-job collisions. Within this job, refuse an unexpected
  # existing directory rather than overwriting or mixing products.
  [[ ! -e "$output_dir" ]] || {
    echo "ERROR: output directory already exists: $output_dir"
    exit 1
  }
  mkdir -p "$output_dir"

  {
    echo "label=$label"
    echo "checkpoint=$checkpoint"
    echo "inference_time=$infer_time"
    echo "background_mode=$background_mode"
    echo "obs_norm_mode=$obs_norm_mode"
    echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
    echo "data_root=$data_root"
    echo "grid_config=${repo}/aardvark/grid_config_ok_reduced_grid_B.yaml"
  } > "${output_dir}/inference_metadata.txt"

  echo
  echo "[$((index + 1))/$checkpoint_count] $label"
  echo "  checkpoint: $checkpoint"
  echo "  output:     $output_dir"

  # --epoch 0 performs one validation pass without optimization. Start=end creates exactly one
  # sample; eval_epoch writes unnorm_preds_0.npy and unnorm_targets_0.npy.
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
    --assim_train_end_date "$infer_time" \
    --assim_val_start_date "$infer_time" \
    --assim_val_end_date "$infer_time" \
    --time_freq 1H \
    --obs_norm_mode "$obs_norm_mode" \
    --background_mode "$background_mode" \
    --grid_config ../aardvark/grid_config_ok_reduced_grid_B.yaml \
    2>&1 | tee "${output_dir}/inference.log"

  prediction="${output_dir}/unnorm_preds_0.npy"
  target="${output_dir}/unnorm_targets_0.npy"
  [[ -f "$prediction" && -f "$target" ]] || {
    echo "ERROR: inference completed without expected outputs in $output_dir"
    exit 1
  }
  printf '%d\t%s\t%s\t%s\tcomplete\n' \
    "$((index + 1))" "$label" "$checkpoint" "$output_dir" >> "$summary_file"
done

echo
echo "Completed $checkpoint_count checkpoint inference run(s)."
echo "Checkpoint/output index: $summary_file"
echo "Batch output root:       $batch_root"
