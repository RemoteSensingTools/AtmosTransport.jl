#!/usr/bin/env bash
# Run the 23-model TRENDYv14 S3 ensemble as four GPU-memory-safe C90 jobs.
# Two batches run concurrently (one per GPU), followed by the remaining pair.

set -uo pipefail

REPO_DIR=/home/cfranken/code/gitHub/AtmosTransportModel
RUNTIME_DIR=/tmp/AtmosTransportModel-era-c90
OUTPUT_ROOT=/home/cfranken/data/AtmosTransport/output/trendy_v14_s3_gpp_ter_c90_2021_advdiff
JOB_DIR="$OUTPUT_ROOT/_job"

mkdir -p "$JOB_DIR"

terminate_children() {
    local children
    children=$(jobs -pr)
    if [[ -n "$children" ]]; then
        kill -TERM $children 2>/dev/null || true
    fi
}
trap terminate_children TERM INT

run_batch() {
    local batch=$1
    local gpu=$2
    local tag
    local config
    local log
    tag=$(printf 'batch%02d' "$batch")
    config="$REPO_DIR/config/runs/trendy_v14_s3_gpp_ter_c90_2021_advdiff_${tag}.toml"
    log="$JOB_DIR/${tag}.log"

    if find "$OUTPUT_ROOT/$tag" -maxdepth 1 -type f -name '*.nc' -print -quit 2>/dev/null | grep -q .; then
        echo "[$(date --iso-8601=seconds)] Refusing to overwrite existing $tag output" >&2
        return 20
    fi

    echo "[$(date --iso-8601=seconds)] Starting $tag on physical GPU $gpu"
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$gpu" JULIA_NUM_THREADS=2 \
        julia --project="$RUNTIME_DIR" \
        "$RUNTIME_DIR/scripts/run_transport.jl" "$config" \
        > "$log" 2>&1
    local status=$?
    echo "$status" > "$JOB_DIR/${tag}.exit_status"
    echo "[$(date --iso-8601=seconds)] Finished $tag with status $status"
    return "$status"
}

run_wave() {
    local batch_a=$1
    local gpu_a=$2
    local batch_b=$3
    local gpu_b=$4
    local pid_a pid_b status_a status_b

    run_batch "$batch_a" "$gpu_a" &
    pid_a=$!
    echo "$pid_a" > "$JOB_DIR/$(printf 'batch%02d' "$batch_a").pid"
    run_batch "$batch_b" "$gpu_b" &
    pid_b=$!
    echo "$pid_b" > "$JOB_DIR/$(printf 'batch%02d' "$batch_b").pid"

    wait "$pid_a"; status_a=$?
    wait "$pid_b"; status_b=$?
    if (( status_a != 0 || status_b != 0 )); then
        echo "Wave failed: batch $batch_a status=$status_a; batch $batch_b status=$status_b" >&2
        return 1
    fi
}

echo "[$(date --iso-8601=seconds)] TRENDY 2021 four-batch campaign started"
run_wave 1 0 2 1 || exit 1
run_wave 3 0 4 1 || exit 1
date --iso-8601=seconds > "$JOB_DIR/COMPLETE"
echo "[$(date --iso-8601=seconds)] TRENDY 2021 four-batch campaign complete"
