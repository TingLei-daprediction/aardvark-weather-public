#!/usr/bin/env bash
# Prepare ERA5 memmaps and norms on the target grid.
# Edit paths and variable list to match what you downloaded and what your training expects.

set -euo pipefail

INPUT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-weatherbench/era5_1p5deg/single_level"
OUTPUT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data"
GRID_DIR="data/grid_lon_lat"
ERA5_MODE="sfc"  # change to 4u/13u if you later prepare those modes
YEARS=(2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019)

# Variables in the channel order you want to store
VARS=(
  2m_temperature
  2m_dewpoint_temperature
  10m_u_component_of_wind
  10m_v_component_of_wind
  mean_sea_level_pressure
  surface_pressure
)

python scripts/prep_era5_truth.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --era5_mode "${ERA5_MODE}" \
  --variables "${VARS[@]}" \
  --years "${YEARS[@]}" \
  --grid_dir "${GRID_DIR}"

echo "Done. ERA5 memmaps are under ${OUTPUT_DIR}/era5 and norms under ${OUTPUT_DIR}/norm_factors."
