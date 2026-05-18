#!/usr/bin/env bash
# Inspect ERA5 NetCDF variables for a given year.

set -euo pipefail

INPUT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-weatherbench/era5_1p5deg/pressure_level"
YEAR=2007
PATTERN="era5_*_{year}_*.nc"

python check_era5_vars.py \
  --input_dir "${INPUT_DIR}" \
  --year "${YEAR}" \
  --pattern "${PATTERN}"
