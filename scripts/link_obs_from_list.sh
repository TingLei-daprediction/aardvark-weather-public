#!/usr/bin/env bash
# Link observation files from a processed obs dir into the data_path layout.
# Edit the paths below, then run:
#   ./scripts/link_obs_from_list.sh

set -euo pipefail

PROCESSED_OBS_DIR="/path/to/processed_obs"
DATA_PATH="/path/to/data_path"
LIST_FILE="/path/to/av-obs-list.txt"

if [[ -z "${PROCESSED_OBS_DIR}" || -z "${DATA_PATH}" || -z "${LIST_FILE}" ]]; then
  echo "ERROR: edit PROCESSED_OBS_DIR, DATA_PATH, and LIST_FILE in this script." >&2
  exit 1
fi

if [[ ! -f "${LIST_FILE}" ]]; then
  echo "ERROR: LIST_FILE does not exist: ${LIST_FILE}" >&2
  exit 1
fi

while IFS= read -r raw; do
  line="${raw## }"
  [[ -z "${line}" ]] && continue

  # Normalize leading ./
  rel="${line#./}"

  src="${PROCESSED_OBS_DIR%/}/${rel}"
  dst="${DATA_PATH%/}/${rel}"

  dst_dir="$(dirname "${dst}")"
  mkdir -p "${dst_dir}"

  if [[ ! -e "${src}" ]]; then
    echo "WARNING: source missing, skipping: ${src}" >&2
    continue
  fi

  if [[ -e "${dst}" || -L "${dst}" ]]; then
    echo "INFO: target exists, skipping: ${dst}" >&2
    continue
  fi

  ln -s "${src}" "${dst}"
  echo "Linked: ${dst} -> ${src}"
 done < "${LIST_FILE}"
