#!/usr/bin/env bash
# Build ERA5 binaries on the GEOS-IT native C180 horizontal grid, using the
# ml137_cfl85 upper-layer merge and carrying TM5 + raw PBL surface payloads.
#
# Defaults build the Dec 2-4 2021 campaign at 8 substeps/hour.
# Override with:
#   START_DAY=02 END_DAY=04 STEPS_PER_WINDOW=12 THREADS=16 ./scripts/...
#
# Set LL_DIR/CS_DIR to build into fresh output directories. When LL_DIR differs
# from the TOML's [output].directory, this script writes a derived TOML in
# LOG_DIR so the preprocessor and the regrid step use the same LL source.
# Set FORCE_LL=1 to rebuild LL files even if they already exist.

set -euo pipefail

cd /home/cfranken/code/gitHub/AtmosTransportModel

THREADS=${THREADS:-16}
START_DAY=${START_DAY:-02}
END_DAY=${END_DAY:-04}
STEPS_PER_WINDOW=${STEPS_PER_WINDOW:-8}
FLOAT_TYPE=${FLOAT_TYPE:-Float32}

LL_CFG=${LL_CFG:-config/preprocessing/era5_ll720x361_cfl85_dec2021_f32_tm5_surface.toml}
LL_DIR=${LL_DIR:-/home/cfranken/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_cfl85_dec2021_f32_tm5_surface}
CS_DIR=${CS_DIR:-/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps${STEPS_PER_WINDOW}}
LOG_DIR=${LOG_DIR:-${CS_DIR}/logs}
FORCE_LL=${FORCE_LL:-0}

mkdir -p "$LL_DIR" "$CS_DIR" "$LOG_DIR"

LL_CFG_EFFECTIVE="$LL_CFG"
ll_cfg_abs="$(realpath "$LL_CFG")"
ll_cfg_output="$(
  awk '
    /^\[output\]/ { in_output=1; next }
    /^\[/ { in_output=0 }
    in_output && /^[[:space:]]*directory[[:space:]]*=/ {
      line=$0
      sub(/^[^"]*"/, "", line)
      sub(/".*$/, "", line)
      print line
      exit
    }
  ' "$LL_CFG"
)"
if [[ -n "$ll_cfg_output" ]]; then
  ll_cfg_output="${ll_cfg_output/#\~/$HOME}"
fi

if [[ -n "$ll_cfg_output" && "$(realpath -m "$ll_cfg_output")" != "$(realpath -m "$LL_DIR")" ]]; then
  LL_CFG_EFFECTIVE="${LOG_DIR}/$(basename "$LL_CFG" .toml).effective.toml"
  awk -v ll_dir="$LL_DIR" '
    /^\[output\]/ { in_output=1; print; next }
    /^\[/ { in_output=0 }
    in_output && /^[[:space:]]*directory[[:space:]]*=/ {
      print "directory  = \"" ll_dir "\""
      next
    }
    { print }
  ' "$LL_CFG" > "$LL_CFG_EFFECTIVE"
  echo "[$(date -Is)] wrote effective LL config ${LL_CFG_EFFECTIVE}"
fi
echo "[$(date -Is)] LL config: ${LL_CFG_EFFECTIVE} (source ${ll_cfg_abs})"
echo "[$(date -Is)] LL dir: ${LL_DIR}"
echo "[$(date -Is)] CS dir: ${CS_DIR}"

for day in $(seq -w "$START_DAY" "$END_DAY"); do
  date_iso="2021-12-${day}"
  ymd="202112${day}"
  ll_bin="${LL_DIR}/era5_transport_${ymd}_merged1000Pa_float32.bin"
  cs_bin="${CS_DIR}/era5_transport_${ymd}_merged1000Pa_float32.bin"

  if [[ "$FORCE_LL" == "1" || ! -f "$ll_bin" ]]; then
    echo "[$(date -Is)] building LL cfl85 source ${date_iso}"
    julia -t"$THREADS" --project=. scripts/preprocessing/preprocess_transport_binary.jl \
      "$LL_CFG_EFFECTIVE" --day "$date_iso" \
      >"${LOG_DIR}/preprocess_ll_${ymd}.log" 2>&1
  else
    echo "[$(date -Is)] LL source exists, skipping ${ll_bin}"
  fi

  echo "[$(date -Is)] regridding ${ymd} to GEOS-native C180 (${STEPS_PER_WINDOW} steps/window)"
  tmp_bin="${cs_bin}.tmp"
  rm -f "$tmp_bin"
  julia -t"$THREADS" --project=. scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \
    --input "$ll_bin" \
    --output "$tmp_bin" \
    --Nc 180 \
    --float-type "$FLOAT_TYPE" \
    --mass-basis dry \
    --convention geos_native \
    --definition gmao_equal_distance \
    --steps-per-window "$STEPS_PER_WINDOW" \
    >"${LOG_DIR}/regrid_geosnative_${ymd}_steps${STEPS_PER_WINDOW}.log" 2>&1
  mv "$tmp_bin" "$cs_bin"
done

echo "[$(date -Is)] complete: ${CS_DIR}"
find "$CS_DIR" -maxdepth 1 -type f -name '*.bin' -printf '%f %s bytes\n' | sort
