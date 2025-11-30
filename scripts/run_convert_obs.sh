#!/usr/bin/env bash
# Convert observational NetCDF training data to the memmap layout Aardvark expects.
# Edit the paths below to your locations before running.

set -euo pipefail

INPUT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/aardvark-weather/training_data"
OUTPUT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data"
GRID_DIR="data/grid_lon_lat"

python scripts/convert_obs_training.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --grid_dir "${GRID_DIR}"

echo "Done. Memmaps and norm_factors are under ${OUTPUT_DIR}."
