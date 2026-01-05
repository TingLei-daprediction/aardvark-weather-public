#!/usr/bin/env bash
# Prepare ERA5 memmaps and norms on the target grid.
# Edit paths and variable list to match what you downloaded and what your training expects.

root_dir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public"
set -euo pipefail

scriptdir="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/scripts"
OUTPUT_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/tlei-aardvark-data"
GRID_DIR="${root_dir}/data/grid_lon_lat"
YEARS=(2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019)
TIME_FREQ="6H"  # use "1D" for daily 00 UTC outputs

# --- Surface/single-level fields (example) ---
INPUT_DIR_SFC="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-weatherbench/era5_1p5deg/single_level"
ERA5_MODE_SFC="sfc"
VARS_SFC=(
  2m_temperature
  2m_dewpoint_temperature
  10m_u_component_of_wind
  10m_v_component_of_wind
  mean_sea_level_pressure
  surface_pressure
)

# --- Pressure-level fields (example) ---
# Requires downloading reanalysis-era5-pressure-levels with these variables/levels.
INPUT_DIR_PL="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-weatherbench/era5_1p5deg/pressure_level"
ERA5_MODE_PL="4u"
VARS_PL=(
  geopotential
  temperature
  specific_humidity
  u_component_of_wind
  v_component_of_wind
)

# Run surface
python ${scriptdir}/prep_era5_truth.py \
  --input_dir "${INPUT_DIR_SFC}" \
  --output_dir "${OUTPUT_DIR}" \
  --era5_mode "${ERA5_MODE_SFC}" \
  --variables "${VARS_SFC[@]}" \
  --years "${YEARS[@]}" \
  --grid_dir "${GRID_DIR}" \
  --time_freq "${TIME_FREQ}"

# Run pressure-level
python ${scriptdir}/prep_era5_truth.py \
  --input_dir "${INPUT_DIR_PL}" \
  --output_dir "${OUTPUT_DIR}" \
  --era5_mode "${ERA5_MODE_PL}" \
  --variables "${VARS_PL[@]}" \
  --years "${YEARS[@]}" \
  --grid_dir "${GRID_DIR}" \
  --time_freq "${TIME_FREQ}"

echo "Done. ERA5 memmaps are under ${OUTPUT_DIR}/era5 and norms under ${OUTPUT_DIR}/norm_factors."
