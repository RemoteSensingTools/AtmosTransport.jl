#!/usr/bin/env bash
set -euo pipefail

year=${1:?usage: run_c90_tm5_attach_year.sh YEAR [JOBS=2] [THREADS=8]}
jobs=${2:-2}
threads=${3:-8}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
julia_bin=${ATMOS_C90_CONV_JULIA:-/home/cfranken/.juliaup/bin/julia}
project=${ATMOS_C90_CONV_PROJECT:-$repo}
source_dir=${ATMOS_C90_CONV_SOURCE:-/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_no_convection}
conv_root=${ATMOS_C90_CONVECTION_ROOT:-/home/cfranken/data/AtmosTransport/met/era5/1.0x1.0/raw/convection}
# v3: convection levels mapped surface-first -> top-first (the earlier
# ..._tm5_convection_1deg_3hour set stored the profiles upside down).
output_dir=${ATMOS_C90_CONV_OUTPUT:-/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v3}
cache_dir=${ATMOS_C90_CONV_CACHE:-$HOME/.cache/AtmosTransport/tm5_attach_c90}
mkdir -p "$cache_dir"
log_dir="$output_dir/_logs/$year"
mkdir -p "$output_dir" "$log_dir"

expected=$(date -u -d "$year-12-31" +%j)
available_transport=$(find "$source_dir" -maxdepth 1 -type f \
    -name "era5_n320_transport_${year}[0-9][0-9][0-9][0-9]_float32.bin" -size +0c | wc -l)
available_convection=$(find "$conv_root/$year" -maxdepth 2 -type f \
    -name "convec_${year}[0-9][0-9][0-9][0-9]_00p03.nc" -size +0c | wc -l)
(( available_transport == expected )) || {
    echo "C90 source year incomplete: $available_transport/$expected" >&2
    exit 1
}
(( available_convection == expected )) || {
    echo "Convection source year incomplete: $available_convection/$expected" >&2
    exit 1
}

export repo julia_bin project source_dir conv_root output_dir log_dir threads cache_dir
attach_one() {
    local ymd=$1
    local year=${ymd:0:4}
    local month=${ymd:4:2}
    local src="$source_dir/era5_n320_transport_${ymd}_float32.bin"
    local nc="$conv_root/$year/$month/convec_${ymd}_00p03.nc"
    local dst="$output_dir/era5_n320_transport_${ymd}_float32.bin"
    local marker="$dst.validated"
    local log="$log_dir/$ymd.log"
    [[ -s $dst && -e $marker ]] && { echo "SKIP $ymd"; return 0; }
    [[ -s $src ]] || { echo "MISSING transport $src" >&2; return 1; }
    [[ -s $nc ]] || { echo "MISSING convection $nc" >&2; return 1; }
    echo "START $ymd"
    "$julia_bin" -t "$threads" --project="$project" \
        "$repo/scripts/preprocessing/attach_catrine_tm5_convection_cs.jl" \
        "$src" "$nc" "$dst" --force --cache-dir "$cache_dir" >"$log" 2>&1 || {
        # xargs workers do not inherit `set -e`: fail explicitly, and leave no
        # partial output or marker that a rerun would mistake for success.
        rm -f "$dst" "$marker"
        echo "FAILED $ymd (see $log)" >&2
        return 1
    }
    touch "$marker"
    echo "DONE  $ymd"
}
export -f attach_one

find "$source_dir" -maxdepth 1 -type f \
    -name "era5_n320_transport_${year}[0-9][0-9][0-9][0-9]_float32.bin" \
    -printf '%f\n' |
    sed -E 's/.*_([0-9]{8})_float32\.bin/\1/' |
    sort |
    xargs -r -P "$jobs" -n 1 bash -c 'attach_one "$1"' _

complete=$(find "$output_dir" -maxdepth 1 -type f \
    -name "era5_n320_transport_${year}[0-9][0-9][0-9][0-9]_float32.bin.validated" | wc -l)
(( complete == expected )) || {
    echo "C90 convection attachment incomplete: $complete/$expected" >&2
    exit 1
}
echo "C90 L66 plus 1-degree/3-hour TM5 convection ready: $complete/$expected for $year"
