#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project=${ATMOS_CATRINE_PROJECT:-/tmp/AtmosTransportModel-era-c90}
julia_bin=${ATMOS_CATRINE_JULIA:-/home/cfranken/.juliaup/bin/julia}
gpu=${ATMOS_CATRINE_GPU:-1}
config=${1:-$repo/config/runs/catrine_c30_2021_fullphysics_experimental.toml}
log=${ATMOS_CATRINE_LOG:-/temp2/catrine-runs/output/catrine_c30_2021_fullphysics_experimental/run.log}
telemetry=${ATMOS_CATRINE_GPU_TELEMETRY:-/temp2/catrine-runs/output/catrine_c30_2021_fullphysics_experimental/gpu_telemetry.csv}
stage_unit=${ATMOS_CATRINE_STAGE_UNIT:-atmos-catrine-c30-fullphysics-stage-2021.service}

if systemctl --user is-active --quiet "$stage_unit"; then
    echo "NVMe staging is already running in $stage_unit"
    while systemctl --user is-active --quiet "$stage_unit"; do
        ready=$(find /temp2/catrine-runs/met/era5_c30_2021_fullphysics_experimental \
            -maxdepth 1 -type f -name 'era5_c30_transport_2021*_float32.bin' -size +0c | wc -l)
        echo "Waiting for NVMe staging: $ready/365 files ready"
        sleep 15
    done
fi
"$repo/scripts/preprocessing/run_catrine_tm5_attach_year.sh" 2021 4 4
mkdir -p "$(dirname -- "$log")"

echo "GPU $gpu status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader -i "$gpu"
echo "Running: $config"
echo "Log:     $log"
echo "GPU CSV: $telemetry"

export CUDA_VISIBLE_DEVICES=$gpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ATMOSTR_TIMERS=${ATMOSTR_TIMERS:-1}
mkdir -p "$(dirname -- "$telemetry")"
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current \
    --format=csv -lms 500 -i "$gpu" >"$telemetry" &
telemetry_pid=$!
cleanup_telemetry() {
    if kill -0 "$telemetry_pid" 2>/dev/null; then
        kill "$telemetry_pid"
        wait "$telemetry_pid" 2>/dev/null || true
    fi
}
trap cleanup_telemetry EXIT INT TERM
/usr/bin/time -f $'\nTotal Julia elapsed: %E\nUser CPU: %U s  System CPU: %S s\nMax RSS: %M KiB' \
    "$julia_bin" -t8 --project="$project" \
    "$project/scripts/run_transport.jl" "$config" 2>&1 | tee "$log"
