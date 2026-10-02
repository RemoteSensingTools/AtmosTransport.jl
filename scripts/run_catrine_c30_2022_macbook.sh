#!/usr/bin/env bash
# Run the portable CATRINE C30 2022 bundle from any location on a MacBook.
set -euo pipefail

bundle_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ATMOSTRANSPORT_DATA_ROOT="$bundle_root"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$(sysctl -n hw.ncpu)}"

mkdir -p "$bundle_root/output"
julia --project="$bundle_root/runtime/AtmosTransportModel-era-c90" \
  "$bundle_root/runtime/AtmosTransportModel-era-c90/scripts/run_transport.jl" \
  "$bundle_root/config/catrine_c30_dec2021_2022_fullphysics_column_macbook.toml" \
  2>&1 | tee "$bundle_root/output/run.log"
