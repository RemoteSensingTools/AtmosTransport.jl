#!/usr/bin/env bash
set -euo pipefail

year=${1:-2021}
jobs=${2:-4}
threads=${3:-4}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
julia_bin=${ATMOS_CATRINE_JULIA:-/home/cfranken/.juliaup/bin/julia}
project=${ATMOS_CATRINE_PROJECT:-$repo}
source_dir=${ATMOS_CATRINE_C30_SOURCE:-/home/cfranken/data/AtmosTransport/met/era5/c90_to_c30/transport_binary_v4_l66_f32_no_convection_experimental}
conv_root=${ATMOS_CATRINE_CONVECTION_ROOT:-/home/cfranken/data/AtmosTransport/met/era5/1.0x1.0/raw/convection}
# v3: convection levels mapped surface-first -> top-first; the earlier
# era5_c30_*_fullphysics_experimental sets store the profiles upside down.
output_dir=${ATMOS_CATRINE_C30_FULLPHYS:-/temp2/catrine-runs/met/era5_c30_2021_fullphysics_experimental_v3}
log_dir="$output_dir/_logs"
mkdir -p "$output_dir" "$log_dir"

expected=$(date -u -d "$year-12-31" +%j)
available=$(find "$source_dir" -maxdepth 1 -type f \
    -name "era5_c30_transport_${year}[0-9][0-9][0-9][0-9]_float32.bin" -size +0c | wc -l)
(( available == expected )) || {
    echo "C30 source year is incomplete: $available/$expected files" >&2
    exit 1
}

export repo julia_bin project source_dir conv_root output_dir log_dir threads year
attach_one() {
    local ymd=$1
    local month=${ymd:4:2}
    local src="$source_dir/era5_c30_transport_${ymd}_float32.bin"
    local nc="$conv_root/$year/$month/convec_${ymd}_00p03.nc"
    local dst="$output_dir/era5_c30_transport_${ymd}_float32.bin"
    local log="$log_dir/$ymd.log"
    [[ -s $dst ]] && { echo "SKIP $ymd"; return 0; }
    [[ -s $nc ]] || { echo "MISSING convection $nc" >&2; return 1; }
    echo "START $ymd"
    "$julia_bin" -t "$threads" --project="$project" \
        "$repo/scripts/preprocessing/attach_catrine_tm5_convection_cs.jl" \
        "$src" "$nc" "$dst" >"$log" 2>&1 || {
        # xargs workers do not inherit `set -e`: fail explicitly, and remove a
        # partial output that the `-s $dst` skip check would otherwise accept.
        rm -f "$dst"
        echo "FAILED $ymd (see $log)" >&2
        return 1
    }
    echo "DONE  $ymd"
}
export -f attach_one

find "$source_dir" -maxdepth 1 -type f \
    -name "era5_c30_transport_${year}[0-9][0-9][0-9][0-9]_float32.bin" \
    -printf '%f\n' | sed -E 's/.*_([0-9]{8})_float32\.bin/\1/' | sort |
    xargs -r -P "$jobs" -n 1 bash -c 'attach_one "$1"' _

complete=$(find "$output_dir" -maxdepth 1 -type f \
    -name "era5_c30_transport_${year}[0-9][0-9][0-9][0-9]_float32.bin" -size +0c | wc -l)
(( complete == expected )) || { echo "Attachment incomplete: $complete/$expected" >&2; exit 1; }
echo "EXPERIMENTAL full-physics C30 forcing ready: $complete/$expected"
