#!/usr/bin/env bash
# Check observation memmaps and inferred shapes.
# Edit DATA_PATH below, then run this script.

set -euo pipefail

DATA_PATH="/path/to/data_path"

python "$(dirname "$0")/check_obs_memmaps.py" --data_path "${DATA_PATH}"
