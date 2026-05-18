#!/usr/bin/env bash
# Check observation memmaps and inferred shapes.
# Edit DATA_PATH below, then run this script.

set -euo pipefail

DATA_PATH="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-aardvark/aardvark-weather-public/dr-data/aardvark-weather/training_data"

python "$(dirname "$0")/check_obs_memmaps.py" --data_path "${DATA_PATH}"
