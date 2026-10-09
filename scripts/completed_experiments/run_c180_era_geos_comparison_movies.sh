#!/usr/bin/env bash
# Generate side-by-side ERA5-vs-GEOS C180 comparison movies with shared
# color ranges for the ERA and GEOS panels.

set -euo pipefail

cd /home/cfranken/code/gitHub/AtmosTransportModel

ERA_DIR=${ERA_DIR:-/temp1/c180_era5_geosgrid_cfl85_3d}
GEOS_DIR=${GEOS_DIR:-/temp1/c180_geosit_native_3d}
OUT_DIR=${OUT_DIR:-/tmp/tm5_smoke/viz_era_geos_c180_comparison_movies}
RUNS=${RUNS:-advonly_ppm,advdiff_ppm,fullphysics_ppm}
TRACERS=${TRACERS:-co2_natural,co2_fossil}
SPECS=${SPECS:-column_mean}
FPS=${FPS:-4}
RESOLUTION=${RESOLUTION:-360x181}

export ERA_DIR GEOS_DIR OUT_DIR RUNS TRACERS SPECS FPS RESOLUTION
mkdir -p "$OUT_DIR"

julia --project=. scripts/visualization/compare_c180_era_geos_movies.jl

find "$OUT_DIR" -maxdepth 1 -type f -name '*.mp4' \
  -printf '%f %s bytes\n' | sort
