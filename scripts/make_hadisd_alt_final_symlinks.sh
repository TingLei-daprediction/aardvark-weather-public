#!/usr/bin/env bash
# Create symlinks for HadISD altitude files to the *_alt_*_final.npy names expected by loaders.
# Usage: DATA_PATH=/path/to/data ./scripts/make_hadisd_alt_final_symlinks.sh

set -euo pipefail

DATA_PATH="${DATA_PATH:-}"
if [[ -z "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH is not set. Example: DATA_PATH=/path/to/data ./scripts/make_hadisd_alt_final_symlinks.sh" >&2
  exit 1
fi

HADISD_DIR="${DATA_PATH%/}/hadisd_processed"

vars=(tas tds u v psl)

for var in "${vars[@]}"; do
  src="${HADISD_DIR}/${var}_alt_train.npy"
  dst="${HADISD_DIR}/${var}_alt_train_final.npy"

  if [[ ! -f "${src}" ]]; then
    echo "WARNING: missing source file: ${src}" >&2
    continue
  fi

  if [[ -L "${dst}" || -f "${dst}" ]]; then
    echo "INFO: target already exists, skipping: ${dst}" >&2
    continue
  fi

  ln -s "${src}" "${dst}"
  echo "Linked: ${dst} -> ${src}"
 done
