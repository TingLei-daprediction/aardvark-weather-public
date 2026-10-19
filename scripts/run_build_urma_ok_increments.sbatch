#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J prep-urma-ok-increments
#SBATCH -p u1-compute
#SBATCH -q batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -o prep-urma-ok-increments.%j.out
#SBATCH -e prep-urma-ok-increments.%j.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
unset PYTHONPATH
set -euo pipefail
repo="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
data_root="${DATA_ROOT:-/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data}"
aux_root="${AUX_DATA_ROOT:-$data_root}"
train_months="${INCREMENT_TRAIN_MONTHS_FILE:-${repo}/training/rtma_ok_train_months_2022.txt}"
val_months="${INCREMENT_VAL_MONTHS_FILE:-${repo}/training/rtma_ok_val_months_2023.txt}"
grid_config="${GRID_CONFIG:-${repo}/aardvark/grid_config_ok_reduced_grid_B.yaml}"
cd "$repo"
legacy_args=()
case "${ALLOW_LEGACY_INCREMENT_INPUTS:-0}" in
  0) ;;
  1) legacy_args=(--allow_legacy_inputs) ;;
  *) echo "ERROR: ALLOW_LEGACY_INCREMENT_INPUTS must be 0 or 1" >&2; exit 1 ;;
esac
echo "Target mode: increment; background mode: hourly; training statistics: $train_months"
# Combined unique months avoid regenerating overlapping train/validation products.
months=$(python - "$train_months" "$val_months" <<'PY'
import sys
sys.path.insert(0, "aardvark")
from month_manifest import read_month_manifest, format_months
print(" ".join(format_months(sorted(set(read_month_manifest(sys.argv[1])) | set(read_month_manifest(sys.argv[2]))))))
PY
)
python scripts/build_hourly_increments.py \
  --data_root "$data_root" --grid_config "$grid_config" \
  --months $months --background_mode hourly --overwrite "${legacy_args[@]}"
python scripts/compute_increment_norms.py \
  --data_root "$data_root" --aux_data_path "$aux_root" --grid_config "$grid_config" \
  --months_file "$train_months" --overwrite
