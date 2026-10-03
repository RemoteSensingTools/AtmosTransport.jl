#!/usr/bin/env bash
set -euo pipefail

# EXPERIMENTAL C90 -> C30 nested operator restriction.
#
# This is a throughput/resume wrapper around coarsen_cs_transport_binary.jl.
# Every binary is stamped testing-only in its JSON header. This driver does
# not create `.validated` sentinels: direct-C30 and tracer validation remain.

if (( $# < 1 || $# > 3 )); then
    echo "Usage: $0 YEAR [JOBS=4] [JULIA_THREADS=4]" >&2
    exit 2
fi

year=$1
jobs=${2:-4}
julia_threads=${3:-4}

[[ $year =~ ^[0-9]{4}$ ]] || { echo "YEAR must be four digits" >&2; exit 2; }
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be positive" >&2; exit 2; }
[[ $julia_threads =~ ^[1-9][0-9]*$ ]] || {
    echo "JULIA_THREADS must be positive" >&2
    exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
project_dir=${ATMOS_CS_COARSEN_PROJECT:-/tmp/AtmosTransportModel-era-c90}
julia_bin=${ATMOS_CS_COARSEN_JULIA:-/home/cfranken/.juliaup/bin/julia}
source_dir=${ATMOS_CS_COARSEN_SOURCE_DIR:-/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_no_convection}
output_dir=${ATMOS_CS_COARSEN_OUTPUT_DIR:-/home/cfranken/data/AtmosTransport/met/era5/c90_to_c30/transport_binary_v4_l66_f32_no_convection_experimental}
log_dir="$output_dir/_logs/$year"

[[ -f $project_dir/Project.toml ]] || {
    echo "Pinned format-v4 Julia project is missing: $project_dir" >&2
    exit 1
}
[[ -x $julia_bin ]] || {
    echo "Julia executable is missing: $julia_bin" >&2
    exit 1
}
mkdir -p "$output_dir" "$log_dir"

dates=()
day="${year}-01-01"
while [[ ${day:0:4} == "$year" ]]; do
    dates+=("${day//-/}")
    day=$(date -u -d "$day + 1 day" +%F)
done

missing=0
for ymd in "${dates[@]}"; do
    source="$source_dir/era5_n320_transport_${ymd}_float32.bin"
    if [[ ! -s $source || ! -e $source.validated ]]; then
        echo "Missing validated C90 source: $source" >&2
        missing=$((missing + 1))
    fi
done
(( missing == 0 )) || {
    echo "Refusing partial-year conversion: $missing validated source day(s) missing" >&2
    exit 1
}

export repo_dir project_dir julia_bin source_dir output_dir log_dir julia_threads
coarsen_one_day() {
    local ymd=$1
    local source="$source_dir/era5_n320_transport_${ymd}_float32.bin"
    local output="$output_dir/era5_c30_transport_${ymd}_float32.bin"
    local log="$log_dir/${ymd}.log"
    if [[ -s $output ]]; then
        echo "SKIP $ymd (complete output exists)"
        return 0
    fi
    echo "START $ymd"
    if "$julia_bin" -t "$julia_threads" --project="$project_dir" \
        "$repo_dir/scripts/preprocessing/coarsen_cs_transport_binary.jl" \
        "$source" "$output" --target-nc 30 >"$log" 2>&1; then
        echo "DONE  $ymd"
    else
        status=$?
        echo "FAIL  $ymd (status $status; see $log)" >&2
        return "$status"
    fi
}
export -f coarsen_one_day

printf '%s\n' "${dates[@]}" |
    xargs -r -P "$jobs" -n 1 bash -c 'coarsen_one_day "$1"' _

expected=${#dates[@]}
complete=$(find "$output_dir" -maxdepth 1 -type f \
    -name "era5_c30_transport_${year}[0-9][0-9][0-9][0-9]_float32.bin" -size +0c |
    wc -l)
if (( complete != expected )); then
    echo "Year $year incomplete: $complete/$expected outputs" >&2
    exit 1
fi
echo "EXPERIMENTAL C30 year $year complete: $complete/$expected files"
