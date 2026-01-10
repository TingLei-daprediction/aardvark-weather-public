#!/usr/bin/env bash
# Download missing ERA5 pressure-level fields (relative_humidity, vertical_velocity).
# Requires a valid ~/.cdsapirc with the new API format (url: https://cds.climate.copernicus.eu/api, key: <APIKEY>).

#SBATCH -A fv3-cam
#SBATCH -J era5-pl-missing
#SBATCH -p u1-service          # or a partition with outbound HTTPS
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o era5-pl-missing.%j.out
#SBATCH -e era5-pl-missing.%j.err

set -euo pipefail

OUTDIR=/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-weatherbench/era5_1p5deg/pressure_level
mkdir -p "$OUTDIR"
cd "$OUTDIR"

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate aardvark-env
python - <<'PY'
import cdsapi, os
c = cdsapi.Client()

years = range(2007, 2020)
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
times = [f"{h:02d}:00" for h in (0, 6, 12, 18)]
levels = ["1000","925","850","700","600","500","400","300","250","200","150","100","70","50","30","20","10","7","5","3","2","1"]
variables = [
    "relative_humidity",
    "vertical_velocity",
]

os.makedirs("pressure_level", exist_ok=True)
for year in years:
    for month in months:
        for var in variables:
            target = f"pressure_level/era5_pl_1p5deg_{var}_{year}_{month}.nc"
            if os.path.exists(target):
                print(f"Skip {target}, exists")
                continue
            print(f"Request {var} {year}-{month}")
            c.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": var,
                    "year": str(year),
                    "month": month,
                    "day": days,
                    "time": times,
                    "pressure_level": levels,
                    "grid": [1.5, 1.5],
                    "format": "netcdf",
                },
                target,
            )
PY

echo "Done. Files in ${OUTDIR}/pressure_level"
