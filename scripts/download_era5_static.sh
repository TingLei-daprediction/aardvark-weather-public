#!/usr/bin/env bash
# Download ERA5 static fields needed for elev_vars_1.npy (geopotential + land-sea mask).
# Requires ~/.cdsapirc configured for the CDS API.

#SBATCH -A fv3-cam
#SBATCH -J era5-static
#SBATCH -p u1-service
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH -o era5-static.%j.out
#SBATCH -e era5-static.%j.err

set -euo pipefail

OUTDIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-weatherbench/era5_1p5deg/static
mkdir -p "$OUTDIR"
cd "$OUTDIR"

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env

python - <<'PY'
import cdsapi, os

c = cdsapi.Client()
target = "era5_static_1p5deg.nc"
if os.path.exists(target):
    print(f"Skip {target}, exists")
else:
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask"],
            "year": "2019",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "grid": [1.5, 1.5],
            "format": "netcdf",
        },
        target,
    )
PY

echo "Done. Static ERA5 file at ${OUTDIR}/era5_static_1p5deg.nc"
