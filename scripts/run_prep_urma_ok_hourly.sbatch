#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J prep-urma-ok
#SBATCH -p u1-compute
#SBATCH -q batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --open-mode=truncate
#SBATCH -t 04:00:00
#SBATCH -o prep-urma-ok.out
#SBATCH -e prep-urma-ok.err

# Preprocess URMA OK grib2 -> hourly target + daily 00z background memmaps (rtma_ok_sfc, 1H).
# CPU-only job (grib decode + memmap writes; no GPU). Partition/qos: u1-compute/batch on Ursa;
# adjust if your allocation differs (the GPU training job uses u1-h100/gpuwf, see training/dd.sh).
#
# Steps:
#   1. prep_urma_ok_hourly.py  -- targets + background_raw + norm factors + era5_x/y_ok.npy
#                                 (grid derived from the first grib; no interpolation)
#   2. normalize_background.py -- background_raw -> background (target mean/std, build time)

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
set -euo pipefail

repo="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
# URMA grib archive: <input_dir>/<YYYYMMDD>/urma2p5.t<HH>z.2dvar{anl,ges}_OK.grb2  -- EDIT ME
#cltorg input_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-raw-2p5"
input_dir="/scratch3/NCEPDEV/gpu-emc-ai/Annette.Gibbs/aardvark_OK/OK_data/urma2p5_OK_grb2"
# Output data_path for the OK runs (era5/ and norm_factors/ created under it)
data_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-rtma-data/dr-av-rtma_ok_data"

start="2022-01"
end="2023-12"

cd "$repo"

# Month list for normalize_background.py (inclusive start..end)
months=$(python - "$start" "$end" <<'EOF'
import sys
y0, m0 = map(int, sys.argv[1].split("-")); y1, m1 = map(int, sys.argv[2].split("-"))
out = []
while (y0, m0) <= (y1, m1):
    out.append(f"{y0}-{m0:02d}")
    m0 += 1
    if m0 == 13:
        y0, m0 = y0 + 1, 1
print(" ".join(out))
EOF
)
echo "Months: $months"

# Naming token for the processed files: <data_dir>/urma/urma_rtma_ok_sfc_1_1h_....memmap etc.
# MUST match the era5_month/background_month/grid_files templates in grid_config_ok.yaml.
name_root="urma"

# 1. Targets + raw background + norms (+ grid axes as side product).
#    --write_norms: ONLY because start..end here is the intended TRAINING period.
python scripts/prep_urma_ok_hourly.py \
  --input_dir "$input_dir" \
  --output_dir "$data_dir" \
  --start "$start" --end "$end" \
  --name_root "$name_root" \
  --write_norms

# 2. Normalize the 00z background with the target mean/std written in step 1.
python scripts/normalize_background.py \
  --data_dir "$data_dir" \
  --era5_mode rtma_ok_sfc \
  --subdir "$name_root" \
  --months $months

echo "Done. Before TRAINING (not part of preprocessing):"
echo "  - copy $data_dir/$name_root/${name_root}_x_ok.npy + ${name_root}_y_ok.npy to <model_data_path>/grid_lon_lat/"
echo "  - build elev_vars_ok.npy (scripts/build_elev_vars.py --tag ok --name_root $name_root)"
echo "  - build obs files hadisd_processed/{var}_vals_1h_YYYY-MM.memmap + station coords + hadisd norms"
