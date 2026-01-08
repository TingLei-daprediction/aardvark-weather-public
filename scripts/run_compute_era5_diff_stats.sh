#!/usr/bin/env bash
# Compute diff-mode ERA5 stats (era5_spatial_means.npy, era5_avdiff_means.npy, era5_avdiff_stds.npy).
# Edit the paths/years below, then run this script.

set -euo pipefail

SCRIPT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/scripts"
DATA_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/tlei-aardvark-data"
AUX_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/tlei-aardvark-data"
GRID_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/data/grid_lon_lat"
YEARS=(2007 2008)
ERA5_MODE="4u_sfc"
RES=1

python "${SCRIPT_DIR}/compute_era5_diff_stats.py" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${AUX_DIR}" \
  --grid_dir "${GRID_DIR}" \
  --era5_mode "${ERA5_MODE}" \
  --res "${RES}" \
  --years "${YEARS[@]}"

python "${SCRIPT_DIR}/compute_era5_norms.py" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${AUX_DIR}" \
  --grid_dir "${GRID_DIR}" \
  --era5_mode "${ERA5_MODE}" \
  --res "${RES}" \
  --years "${YEARS[@]}"
