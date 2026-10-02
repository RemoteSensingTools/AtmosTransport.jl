#!/usr/bin/env bash
# Generate one year of ERA5 N320 → C90 L66 v4 binaries as independent day jobs.
# The driver waits for each ARCO day plus its next-day endpoint, so it can run
# concurrently with a chronological raw-data download. Successful days receive
# a `.validated` sentinel after both the runtime-reader and continuity checks.

set -uo pipefail

YEAR="${1:-2021}"
MAX_JOBS="${2:-4}"
THREADS_PER_JOB="${3:-16}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="config/preprocessing/era5_n320_arco_diffusion_to_c90.toml"
DATA_ROOT="${ATMOS_DATA_ROOT:-$HOME/data/AtmosTransport}"
RAW="$DATA_ROOT/met/era5/N320/hourly/raw"
OUTPUT="$DATA_ROOT/met/era5/n320_to_c90/transport_binary_v4_l66_f32_no_convection"
LOGS="$OUTPUT/_logs"
SURFACE_VARS=(surface_pressure boundary_layer_height friction_velocity \
              surface_sensible_heat_flux surface_latent_heat_flux 2m_temperature)

mkdir -p "$OUTPUT" "$LOGS"
cd "$REPO" || exit 1

core_path() {
    printf '%s/ml_an_native_core/era5_core_%s.grib' "$RAW" "$1"
}

surface_ready() {
    local ymd="$1" variable path
    for variable in "${SURFACE_VARS[@]}"; do
        path="$RAW/sfc_an_native/arco/$ymd/$variable.nc"
        [ -f "$path" ] && [ "$(stat -c %s "$path" 2>/dev/null)" -gt 10000000 ] || return 1
    done
}

inputs_ready() {
    local ymd="$1" next_ymd="$2"
    [ -s "$(core_path "$ymd")" ] && [ -s "$(core_path "$next_ymd")" ] && \
        surface_ready "$ymd"
}

run_day() {
    local iso="$1" ymd="$2"
    local binary="$OUTPUT/era5_n320_transport_${ymd}_float32.bin"
    local log="$LOGS/${ymd}.log"
    local sentinel="$binary.validated"

    if [ -s "$binary" ] && [ -f "$sentinel" ]; then
        echo "[year] skip $iso (validated)"
        return 0
    fi

    echo "[year] start $iso"
    if julia -t"$THREADS_PER_JOB" --project=. \
            scripts/preprocessing/preprocess_transport_binary.jl "$CONFIG" --day "$iso" \
            >"$log" 2>&1 && \
       julia --project=. scripts/diagnostics/inspect_transport_binary.jl "$binary" \
            >>"$log" 2>&1 && \
       julia --project=. scripts/validation/verify_cs_binary_continuity.jl \
            --binary "$binary" --threshold 5e-6 >>"$log" 2>&1; then
        touch "$sentinel"
        echo "[year] done $iso"
        return 0
    fi

    echo "[year] FAILED $iso (see $log)"
    return 1
}

start="${YEAR}-01-01"
end="${YEAR}-12-31"
current_epoch="$(date -u -d "$start" +%s)"
end_epoch="$(date -u -d "$end" +%s)"

echo "[year] ERA5 C90 $YEAR: max_jobs=$MAX_JOBS threads/job=$THREADS_PER_JOB"
while [ "$current_epoch" -le "$end_epoch" ]; do
    iso="$(date -u -d "@$current_epoch" +%Y-%m-%d)"
    ymd="$(date -u -d "@$current_epoch" +%Y%m%d)"
    next_epoch=$((current_epoch + 86400))
    next_ymd="$(date -u -d "@$next_epoch" +%Y%m%d)"

    while ! inputs_ready "$ymd" "$next_ymd"; do
        echo "[year] wait $iso (current/next core or surface incomplete)"
        sleep 60
    done
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n || true
    done
    run_day "$iso" "$ymd" &
    current_epoch="$next_epoch"
done

while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    wait -n || true
done

validated="$(find "$OUTPUT" -maxdepth 1 -type f -name "era5_n320_transport_${YEAR}*_float32.bin.validated" | wc -l)"
echo "[year] complete: $validated validated day(s) for $YEAR"
[ "$validated" -eq "$(date -u -d "$end" +%j)" ]
