#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project=${ATMOS_CATRINE_PROJECT:-/tmp/AtmosTransportModel-era-c90}
julia_bin=${ATMOS_CATRINE_JULIA:-/home/cfranken/.juliaup/bin/julia}
gpu=${ATMOS_CATRINE_GPU:-1}
config=${1:-$repo/config/runs/catrine_c30_2021_fullphysics_experimental.toml}
profile_dir=${ATMOS_CATRINE_PROFILE_DIR:-/temp2/catrine-runs/profiles/catrine_c30_fullphysics_5day}
report="$profile_dir/catrine_c30_fullphysics_5day"

mkdir -p "$profile_dir"
mkdir -p "$profile_dir/tmp"
export TMPDIR="$profile_dir/tmp"
export CUDA_VISIBLE_DEVICES=$gpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ATMOSTR_TIMERS=1
export ATMOSTR_NVTX=0

/usr/local/cuda/bin/nsys profile \
    --trace=cuda \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="$report" \
    "$julia_bin" -t8 --project="$project" -e '
using CUDA
using AtmosTransport, TOML
cfg = TOML.parsefile(ARGS[1])
cfg["input"]["end_date"] = "2021-01-05"
cfg["output"]["snapshot_file"] = "/temp2/catrine-runs/output/catrine_c30_2021_fullphysics_nsys5/catrine_c30_fullphys_{YYYYMMDD}.atmsnap"
AtmosTransport.run_driven_simulation(cfg)
' "$config"

/usr/local/cuda/bin/nsys stats \
    --report cuda_gpu_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    --format csv \
    --output "$profile_dir/stats" \
    --force-overwrite=true \
    "$report.nsys-rep"

echo "Nsight report: $report.nsys-rep"
echo "CSV summaries: $profile_dir/stats_*.csv"
