#!/bin/bash
#SBATCH -A fv3-cam 
#SBATCH -J era5-dl
#SBATCH -p u1-service          # or any partition with outbound HTTPS allowed
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o era5-dl.%j.out
#SBATCH -e era5-dl.%j.err

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
set -euo pipefail

# Directory to write downloads
OUTDIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-weatherbench/era5_1p5deg
mkdir -p "$OUTDIR"
cd "$OUTDIR"

# Python download using cdsapi
python - <<'PY'
import cdsapi, os

c = cdsapi.Client()

years = range(2007, 2020)
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
times = [f"{h:02d}:00" for h in (0, 6, 12, 18)]

variables = [
    "2m_temperature", "2m_dewpoint_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "surface_pressure",
]

for y in years:
    for m in months:
        target = f"single_level/era5_single_1p5deg_{y}_{m}.nc"
        if os.path.exists(target):
            print(f"Skip {target}, exists")
            continue
        print(f"Request {y}-{m}")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": variables,
                "year": str(y),
                "month": m,
                "day": days,
                "time": times,
                "grid": [1.5, 1.5],
                "format": "netcdf",
            },
            target,
        )
PY

