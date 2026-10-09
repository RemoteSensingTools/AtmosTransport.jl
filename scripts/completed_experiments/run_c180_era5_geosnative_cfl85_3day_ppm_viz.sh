#!/usr/bin/env bash
# Generate PNG grids and column-mean MP4s for the ERA5-on-GEOS-native
# C180/L85 3-day PPM reruns.
#
# Outputs:
#   /tmp/tm5_smoke/viz_era5_geosnative_c180_cfl85_3d_ppm
#   /tmp/tm5_smoke/viz_era5_geosnative_c180_cfl85_3d_ppm_summary.txt
#
# Expected inputs:
#   /temp1/c180_era5_geosgrid_cfl85_3d/advonly_ppm.nc
#   /temp1/c180_era5_geosgrid_cfl85_3d/advdiff_ppm.nc
#   /temp1/c180_era5_geosgrid_cfl85_3d/fullphysics_ppm.nc

set -euo pipefail

cd /home/cfranken/code/gitHub/AtmosTransportModel

INPUT_DIR=${INPUT_DIR:-/temp1/c180_era5_geosgrid_cfl85_3d}
VIZ_DIR=${VIZ_DIR:-/tmp/tm5_smoke/viz_era5_geosnative_c180_cfl85_3d_ppm}
SUMMARY=${SUMMARY:-/tmp/tm5_smoke/viz_era5_geosnative_c180_cfl85_3d_ppm_summary.txt}
mkdir -p "$VIZ_DIR"
: > "$SUMMARY"

# L85 levels are top-to-bottom in the snapshot files. `surface_slice` handles
# the bottom level; the explicit levels below are quick tropospheric diagnostics.
declare -A LEVELS=(
  [surface]="--transform surface_slice"
  [mid_trop]="--transform level_slice --level 65"
  [upper_trop]="--transform level_slice --level 40"
  [column_mean]="--transform column_mean"
)

for cfg in advonly_ppm advdiff_ppm fullphysics_ppm; do
  input="${INPUT_DIR}/${cfg}.nc"
  if [[ ! -f "$input" ]]; then
    printf '[%s] %s not found; skipping %s\n' "$(date -Is)" "$input" "$cfg" | tee -a "$SUMMARY"
    continue
  fi

  for tracer in co2_natural co2_fossil; do
    for slice in surface mid_trop upper_trop column_mean; do
      out="$VIZ_DIR/${cfg}_${tracer}_${slice}.png"
      args=${LEVELS[$slice]}
      printf '[%s] %s / %s / %s -> %s\n' "$(date -Is)" "$cfg" "$tracer" "$slice" "$out" | tee -a "$SUMMARY"
      julia --project=. scripts/visualization/atmos_viz.jl \
        --input "$input" --tracer "$tracer" --kind grid $args --ppm \
        --out "$out" >> "$SUMMARY" 2>&1
    done

    out_mp4="$VIZ_DIR/${cfg}_${tracer}_column_mean.mp4"
    printf '[%s] %s / %s / movie -> %s\n' "$(date -Is)" "$cfg" "$tracer" "$out_mp4" | tee -a "$SUMMARY"
    julia --project=. scripts/visualization/atmos_viz.jl \
      --input "$input" --tracer "$tracer" --kind movie --transform column_mean \
      --ppm --fps 4 --out "$out_mp4" >> "$SUMMARY" 2>&1
  done
done

printf '[%s] complete; outputs in %s\n' "$(date -Is)" "$VIZ_DIR" | tee -a "$SUMMARY"
find "$VIZ_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.mp4' \) \
  -printf '%f %s bytes\n' | sort | tee -a "$SUMMARY"
