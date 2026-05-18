#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J convert_obs 
#SBATCH -p u1-service          # or any partition with outbound HTTPS allowed
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -o convert_obs-dl.%j.out
#SBATCH -e convert_obs-dl.%j.err


#!/usr/bin/env bash
# Convert observational NetCDF training data to the memmap layout Aardvark expects.
# Edit the paths below to your locations before running.

set -euo pipefail
set -euo pipefail
source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env


INPUT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/aardvark-weather/training_data"
OUTPUT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/emc-aardvark-data"
GRID_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/data/grid_lon_lat/"

python scripts/convert_obs_training.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --grid_dir "${GRID_DIR}"

echo "Done. Memmaps and norm_factors are under ${OUTPUT_DIR}."
