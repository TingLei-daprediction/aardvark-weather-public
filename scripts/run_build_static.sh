#!/usr/bin/env bash
# Build static fields (elev_vars_1.npy) and climatology_data.mmap.
# Edit paths/years/vars as needed.

set -euo pipefail
ROOT_DIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/
DATA_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/tlei-aardvark-data"
GRID_DIR="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/data/grid_lon_lat"

# 1) Build elev_vars_1.npy from a file containing geopotential + land_sea_mask
STATIC_FILE="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-weatherbench/era5_1p5deg/static/era5_static_1p5deg.nc"

python ${ROOT_DIR}/scripts/build_elev_vars.py \
  --input_file "${STATIC_FILE}" \
  --output_dir "${DATA_DIR}" \
  --grid_dir "${GRID_DIR}" \
  --geopotential_var geopotential \
  --lsm_var land_sea_mask

# 2) Build climatology_data.mmap from ERA5 4u memmaps
python scripts/build_climatology.py \
  --data_dir /scratch3/NCEPDEV/fv3-cam/Ting.Lei/aardvark-data \
  --era5_mode 4u_sfc \
  --res 1 \
  --years 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 \
  --time_freq 1D \
  --fill_nan 0.0

echo "Done. Static fields and climatology written under ${DATA_DIR}/era5."
