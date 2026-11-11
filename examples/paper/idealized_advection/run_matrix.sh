#!/usr/bin/env bash
# Driver of the idealized advection tests (GMD paper, Sect. 4.1).
#
#   IDEALIZED_ADVECTION_WORKDIR=<large scratch dir> \
#       examples/paper/idealized_advection/run_matrix.sh [generate|configs|run|analyze|plot|all]
#
# Stages:
#   generate  transport binaries, initial conditions and analytic references for every
#             grid × flow of cases.toml (generate.jl)
#   configs   one run config per grid × flow × scheme × precision (make_configs.jl)
#   run       every config through scripts/run_transport.jl; skips runs with a finished log
#   analyze   error norms, extrema, mass drift, FP32 − FP64 → results/*.csv (analyze.py)
#   plot      figures/*.pdf and *.png (plot_figures.py)
#
# Environment: IDEALIZED_ADVECTION_WORKDIR (required), JOBS (parallel runs, default 4),
# THREADS (Julia threads per run, default 8), JULIA (default: julia), FLOWS (generate only these
# flows, e.g. "B C"; default all), REGENERATE (1: regenerate cases whose case.toml exists; by
# default the generate stage writes only missing cases), IDEALIZED_ADVECTION_GPU_PRECISIONS
# (configs stage: precisions that run on a CUDA GPU, e.g. Float32; default none), GPUS (run
# stage: GPU indices to use, default "0 1"; one run at a time per GPU, each under the lock
# $GPU_LOCK_DIR/gpu<N>.lock, so that other jobs sharing the GPUs wait).
#
# A run is skipped only if its log is complete and its stored fingerprint (hash of config.toml and
# of the case's case.toml, which generate.jl rewrites last) matches the current one; the run stage
# exits non-zero if any run failed.
# Run from the repository root.
set -euo pipefail

STAGE="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${IDEALIZED_ADVECTION_WORKDIR:?set IDEALIZED_ADVECTION_WORKDIR to a scratch directory}"
JOBS="${JOBS:-4}"
THREADS="${THREADS:-8}"
JULIA="${JULIA:-julia}"
mkdir -p "$WORK/logs"
cd "$REPO"

grid_flows() {   # "<grid> <flow> <courant> <tilt> <windows per day> <tag>" lines from cases.toml
    python3 - "$HERE/cases.toml" <<'EOF'
import sys
try:
    import tomllib
except ImportError:          # Python < 3.11
    import tomli as tomllib
cases = tomllib.load(open(sys.argv[1], "rb"))
for g in cases["grid"]:
    for f in g["flows"]:
        print(g["label"], f, g.get("courant", 0.5), g.get("tilt", 45.0), g.get("windows_per_day", 24),
              g.get("tag", "") or "-")
EOF
}

stage_generate() {
    grid_flows | while read -r grid flow courant tilt windows tag; do
        [[ -n "${FLOWS:-}" && " $FLOWS " != *" $flow "* ]] && continue
        local name="${grid}_${flow}" extra=()
        if [[ "$tag" != "-" ]]; then name="${name}_${tag}"; extra=(--tag "$tag"); fi
        if [[ -f "$WORK/inputs/$name/case.toml" && "${REGENERATE:-0}" != 1 ]]; then
            echo "exists: $name"
            continue
        fi
        echo "generate $name (courant $courant, tilt $tilt, $windows windows per day)"
        "$JULIA" --project=. -t "$THREADS" "$HERE/generate.jl" --grid "$grid" --flow "$flow" \
            --courant "$courant" --tilt "$tilt" --windows-per-day "$windows" "${extra[@]}" \
            --workdir "$WORK" \
            > "$WORK/logs/generate_${name}.log" 2>&1 \
            || { echo "generate $name FAILED (see $WORK/logs/generate_${name}.log)"; return 1; }
        tail -n 3 "$WORK/logs/generate_${name}.log"
    done
}

stage_configs() {
    "$JULIA" --project=. "$HERE/make_configs.jl" --workdir "$WORK"
}

fingerprint() {   # $1 = run directory
    local case_file
    case_file="$(sed -n 's/^case_file = "\(.*\)"$/\1/p' "$1/run.toml")"
    { sha256sum < "$1/config.toml"; sha256sum < "$case_file"; stat -c %Y "$case_file"; } | sha256sum | cut -c1-16
}

run_one() {   # $1 = config path, $2 = GPU index (empty: CPU)
    local cfg="$1" gpu="${2:-}" dir log fp
    dir="$(dirname "$cfg")"
    log="$dir/log.txt"
    fp="$(fingerprint "$dir")"
    if [[ -f "$log" ]] && grep -q "Forward run wall" "$log" && [[ "$(cat "$dir/fingerprint" 2>/dev/null)" == "$fp" ]]; then
        echo "done already: $(basename "$dir")"
        return 0
    fi
    rm -f "$dir/fingerprint" "$dir/output.nc"
    local cmd=("$JULIA" --project=. -t "$THREADS" scripts/run_transport.jl "$cfg")
    if [[ -n "$gpu" ]]; then
        cmd=(flock "$GPU_LOCK_DIR/gpu$gpu.lock" env CUDA_VISIBLE_DEVICES="$gpu" "$JULIA" --project=. -t 2
             scripts/run_transport.jl "$cfg")
    fi
    if "${cmd[@]}" > "$log" 2>&1; then
        echo "$fp" > "$dir/fingerprint"
        echo "finished${gpu:+ on GPU $gpu}: $(basename "$dir")"
    else
        echo "FAILED: $(basename "$dir") (see $log)"
        return 1
    fi
}
export -f run_one fingerprint
export JULIA THREADS

device_of() {   # $1 = config path -> cpu | gpu
    sed -n 's/^device = "\(.*\)"$/\1/p' "$(dirname "$1")/run.toml" | grep . || echo cpu
}

gpu_worker() {   # $1 = GPU index, $2 = list of configs; returns non-zero if a run failed
    local status=0 cfg
    while read -r cfg; do
        run_one "$cfg" "$1" || status=1
    done < "$2"
    return $status
}

stage_run() {
    local cpu_list gpu_list status=0
    cpu_list="$(mktemp)"; gpu_list="$(mktemp)"
    while read -r cfg; do
        [[ -z "$cfg" ]] && continue
        if [[ "$(device_of "$cfg")" == gpu ]]; then echo "$cfg" >> "$gpu_list"; else echo "$cfg" >> "$cpu_list"; fi
    done < "$WORK/runs/runs.txt"
    local pids=()
    if [[ -s "$gpu_list" ]]; then
        : "${GPU_LOCK_DIR:?set GPU_LOCK_DIR to the directory of the gpu<N>.lock files}"
        read -r -a gpus <<< "${GPUS:-0 1}"
        local n=${#gpus[@]} i
        for i in "${!gpus[@]}"; do
            awk -v n="$n" -v i="$i" '(NR - 1) % n == i' "$gpu_list" > "$gpu_list.$i"
            gpu_worker "${gpus[$i]}" "$gpu_list.$i" & pids+=($!)
        done
    fi
    xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} < "$cpu_list" || status=1
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    rm -f "$cpu_list" "$gpu_list" "$gpu_list".*
    if [[ $status != 0 ]]; then
        echo "run stage: at least one run failed" >&2
        return 1
    fi
}

stage_analyze() {
    python3 "$HERE/analyze.py" --workdir "$WORK" --results "$HERE/results"
    python3 "$HERE/panel_edge_split.py" --workdir "$WORK" --results "$HERE/results"
    python3 "$HERE/uniform_tracer_stats.py" --workdir "$WORK" --results "$HERE/results"
}

stage_plot() {
    python3 "$HERE/plot_figures.py" --workdir "$WORK" --results "$HERE/results" --figures "$HERE/figures"
}

case "$STAGE" in
    generate) stage_generate ;;
    configs)  stage_configs ;;
    run)      stage_run ;;
    analyze)  stage_analyze ;;
    plot)     stage_plot ;;
    all)      stage_generate; stage_configs; stage_run; stage_analyze; stage_plot ;;
    *) echo "unknown stage '$STAGE'"; exit 1 ;;
esac
