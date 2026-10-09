#!/usr/bin/env bash
# Generate PNG grids and column-mean MP4s for the C180/L137 3-day PPM reruns.
#
# Outputs:
#   /tmp/tm5_smoke/viz_3d_ppm
#   /tmp/tm5_smoke/viz_3d_ppm_summary.txt
#
# Expected inputs:
#   /temp1/c180_full137_3d/advonly_ppm.nc
#   /temp1/c180_full137_3d/advdiff_ppm.nc
#   /temp1/c180_full137_3d/fullphysics_ppm.nc

set -euo pipefail

cd /home/cfranken/code/gitHub/AtmosTransportModel

VIZ_DIR=${VIZ_DIR:-/tmp/tm5_smoke/viz_3d_ppm}
SUMMARY=${SUMMARY:-/tmp/tm5_smoke/viz_3d_ppm_summary.txt}
mkdir -p "$VIZ_DIR"
: > "$SUMMARY"

# L137 levels: surface = 137; roughly 500 hPa = level 100; roughly 200 hPa = level 70.
declare -A LEVELS=(
  [surface]="--transform surface_slice"
  [mid_trop]="--transform level_slice --level 100"
  [upper_trop]="--transform level_slice --level 70"
  [column_mean]="--transform column_mean"
)

for cfg in advonly_ppm advdiff_ppm fullphysics_ppm; do
  input="/temp1/c180_full137_3d/${cfg}.nc"
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
