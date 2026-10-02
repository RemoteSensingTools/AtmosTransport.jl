#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project=${ATMOS_CATRINE_PROJECT:-/tmp/AtmosTransportModel-era-c90}
julia_bin=${ATMOS_CATRINE_JULIA:-/home/cfranken/.juliaup/bin/julia}
gpu=${ATMOS_CATRINE_GPU:-1}
config=${1:-$repo/config/runs/catrine_c30_2021H1_fullphysics_column_experimental.toml}
input_dir=/temp2/catrine-runs/met/era5_c30_2021_fullphysics_experimental
output_dir=/temp2/catrine-runs/output/catrine_c30_2021H1_fullphysics_column
log=${ATMOS_CATRINE_LOG:-$output_dir/run.log}
telemetry=${ATMOS_CATRINE_GPU_TELEMETRY:-$output_dir/gpu_telemetry.csv}

missing=0
day=2021-01-01
while [[ $day != 2021-07-01 ]]; do
    ymd=${day//-/}
    path=$input_dir/era5_c30_transport_${ymd}_float32.bin
    if [[ ! -s $path ]]; then
        printf 'Missing input: %s\n' "$path" >&2
        missing=$((missing + 1))
    fi
    day=$(date -I -d "$day + 1 day")
done
if (( missing > 0 )); then
    printf 'Cannot start: %d January--June input files are missing.\n' "$missing" >&2
    exit 1
fi

mkdir -p "$output_dir"
if find "$output_dir" -maxdepth 1 -type f -name 'catrine_c30_2021H1_column_*.nc' -print -quit | grep -q .; then
    if [[ ${ATMOS_CATRINE_ALLOW_OVERWRITE:-0} != 1 ]]; then
        echo "Output files already exist in $output_dir" >&2
        echo "Move them aside, or set ATMOS_CATRINE_ALLOW_OVERWRITE=1 to replace them." >&2
        exit 1
    fi
fi

echo "All 181 January--June C30 full-physics inputs are ready."
if systemctl --user is-active --quiet atmos-catrine-c30-fullphysics-stage-2021.service; then
    echo "Note: July--December staging is still active; this does not block the run,"
    echo "but it may add a little CPU/NVMe contention to a runtime measurement."
fi
echo "GPU $gpu status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader -i "$gpu"
echo "Running: $config"
echo "Output:  $output_dir"
echo "Log:     $log"
echo "GPU CSV: $telemetry"

export CUDA_VISIBLE_DEVICES=$gpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ATMOSTR_TIMERS=${ATMOSTR_TIMERS:-1}
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
