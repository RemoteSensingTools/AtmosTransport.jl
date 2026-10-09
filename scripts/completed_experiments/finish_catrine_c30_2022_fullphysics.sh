#!/usr/bin/env bash
set -euo pipefail

# Wait for the resumable 2022 C90->C30 restriction service, attach the matching
# TM5 convection/PBL fields, and assemble the Dec-2021-through-Dec-2022 input
# directory used by the full CATRINE tracer configuration.

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
coarsen_unit=${ATMOS_CATRINE_COARSEN_UNIT:-atmos-catrine-c30-coarsen-2022.service}
c30_dir=${ATMOS_CATRINE_C30_SOURCE:-/home/cfranken/data/AtmosTransport/met/era5/c90_to_c30/transport_binary_v4_l66_f32_no_convection_experimental}
# v3 folders: convection with corrected level order (the unsuffixed sets are upside down).
fullphys_2022=${ATMOS_CATRINE_C30_FULLPHYS:-/temp2/catrine-runs/met/era5_c30_2022_fullphysics_experimental_v3}
combined=${ATMOS_CATRINE_C30_COMBINED:-/temp2/catrine-runs/met/era5_c30_dec2021_through_2022_fullphysics_experimental_v3}
fullphys_2021=${ATMOS_CATRINE_C30_2021_FULLPHYS:-/temp2/catrine-runs/met/era5_c30_2021_fullphysics_experimental_v3}

while true; do
    complete=$(find "$c30_dir" -maxdepth 1 -type f \
        -name 'era5_c30_transport_2022????_float32.bin' -size +0c | wc -l)
    (( complete == 365 )) && break
    if ! systemctl --user is-active --quiet "$coarsen_unit"; then
        echo "C90->C30 service stopped with only $complete/365 files" >&2
        exit 1
    fi
    echo "Waiting for C90->C30 restriction: $complete/365 files"
    sleep 30
done

ATMOS_CATRINE_C30_SOURCE="$c30_dir" \
ATMOS_CATRINE_C30_FULLPHYS="$fullphys_2022" \
    "$repo/scripts/preprocessing/run_catrine_tm5_attach_year.sh" 2022 4 4

mkdir -p "$combined"
for src in "$fullphys_2021"/era5_c30_transport_202112??_float32.bin \
           "$fullphys_2022"/era5_c30_transport_2022????_float32.bin; do
    [[ -s $src ]] || continue
    ln -sfn "$src" "$combined/$(basename -- "$src")"
done

ready=$(find -L "$combined" -maxdepth 1 -type f \
    -name 'era5_c30_transport_20??????_float32.bin' -size +0c | wc -l)
(( ready == 396 )) || {
    echo "Combined Dec-2021-through-2022 forcing is incomplete: $ready/396" >&2
    exit 1
}
echo "CATRINE C30 Dec-2021-through-2022 full-physics forcing ready: $ready/396 files"
