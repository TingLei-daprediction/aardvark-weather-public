#!/usr/bin/env bash
# Check observation memmaps and inferred shapes.
# Edit DATA_PATH below, then run this script.

set -euo pipefail

DATA_PATH="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/aardvark-weather/training_data"
python check_obs_netcdfs.py --input_dir $DATA_PATH 

