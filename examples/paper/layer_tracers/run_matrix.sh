#!/usr/bin/env bash
# Driver of the process experiments with layered tracers (GMD paper, Sect. 6.2).
#
#   LAYER_TRACERS_WORKDIR=<large scratch dir> ERA5_BINARIES=<dir> MERRA2_BINARIES=<dir> \
#       examples/paper/layer_tracers/run_matrix.sh [crossover|configs|run|reduce|figures|all]
#
# Stages:
#   crossover  MERRA-2 binaries of July 2022 that also carry the ERA5-based TM5 convection fields
#              (scripts/preprocessing/attach_catrine_tm5_convection_cs.jl, pressure-overlap mode);
#              needs ERA5_CONVECTION_ROOT and ERA5_ROOT; writes $WORK/binaries_merra2_era5conv
#   configs    run configurations of all cases (make_configs.py): the two main cases with every
#              process combination and four months, the two solver swaps (C and APC, four
#              months), and the field swap (C and APC, July) if its binaries exist
#   run        every configuration through scripts/run_transport.jl on the GPUs of GPUS, one run
#              at a time per GPU; skips runs with a finished log
#   reduce     percentile files (sigma_spread.py) and column maps (column_lift.py)
#   figures    figures/*.pdf and *.png (paper_figures.py), results/*.csv (plot_spread.py)
#   all        configs, run, reduce, figures
#
# Environment: LAYER_TRACERS_WORKDIR (required); ERA5_BINARIES, MERRA2_BINARIES (configs: folders
# of the daily C90 transport binaries); MERRA2_ERA5CONV_BINARIES (configs: binaries of the field
# swap, default $WORK/binaries_merra2_era5conv); ERA5_CONVECTION_ROOT (crossover: folder with
# <YYYY>/<MM>/convec_<YYYYMMDD>_00p03.nc), ERA5_ROOT (crossover: ERA5 N320 native GRIB root, for
# the humidity); PRECISIONS (default "Float32"); GPUS (GPU indices, default "0 1"); GPU_LOCK_DIR
# (run: directory of the gpu<N>.lock files, so that other jobs sharing the GPUs wait); THREADS
# (Julia threads per run, default 8); JOBS (parallel reductions and crossover builds, default 4);
# JULIA (default: julia).
# The run stage exits non-zero if any run failed. Run from the repository root.
set -euo pipefail

STAGE="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${LAYER_TRACERS_WORKDIR:?set LAYER_TRACERS_WORKDIR to a scratch directory}"
PRECISIONS="${PRECISIONS:-Float32}"
THREADS="${THREADS:-8}"
JOBS="${JOBS:-4}"
JULIA="${JULIA:-julia}"
SWAP_BINARIES="${MERRA2_ERA5CONV_BINARIES:-$WORK/binaries_merra2_era5conv}"
mkdir -p "$WORK/logs" "$WORK/configs" "$WORK/runs"
cd "$REPO"

stage_crossover() {
    : "${MERRA2_BINARIES:?set MERRA2_BINARIES}" "${ERA5_CONVECTION_ROOT:?set ERA5_CONVECTION_ROOT}" "${ERA5_ROOT:?set ERA5_ROOT}"
    mkdir -p "$SWAP_BINARIES" "$WORK/regrid_cache"
    attach_day() {
        local ymd="$1" out="$SWAP_BINARIES/merra2_transport_${1}_float32.bin"
        [[ -s "$out" ]] && { echo "exists: $ymd"; return 0; }
        rm -f "$out.partial"                       # an interrupted build leaves only this file
        "$JULIA" --project=. -t "$THREADS" scripts/preprocessing/attach_catrine_tm5_convection_cs.jl \
            "$MERRA2_BINARIES/merra2_transport_${ymd}_float32.bin" \
            "$ERA5_CONVECTION_ROOT/${ymd:0:4}/${ymd:4:2}/convec_${ymd}_00p03.nc" "$out.partial" \
            --vertical-mapping pressure_overlap --era5-root "$ERA5_ROOT" --cache-dir "$WORK/regrid_cache" \
            > "$WORK/logs/attach_$ymd.log" 2>&1 < /dev/null && mv "$out.partial" "$out" && echo "attached: $ymd" \
            || { echo "FAILED: $ymd"; return 1; }
    }
    export -f attach_day
    export JULIA THREADS SWAP_BINARIES MERRA2_BINARIES ERA5_CONVECTION_ROOT ERA5_ROOT WORK
    seq -f "202207%02g" 1 31 | xargs -P "$JOBS" -I{} bash -c 'attach_day {}'
}

stage_configs() {
    : "${ERA5_BINARIES:?set ERA5_BINARIES}" "${MERRA2_BINARIES:?set MERRA2_BINARIES}"
    local common=(--resolution c90 --configs "$WORK/configs" --output "$WORK/runs" --precisions $PRECISIONS)
    python3 "$HERE/make_configs.py" --met era5 --binaries "$ERA5_BINARIES" "${common[@]}"
    python3 "$HERE/make_configs.py" --met merra2 --binaries "$MERRA2_BINARIES" "${common[@]}"
    python3 "$HERE/make_configs.py" --met era5 --binaries "$ERA5_BINARIES" "${common[@]}" \
        --arms C APC --convection ifs --name era5ifs
    python3 "$HERE/make_configs.py" --met merra2 --binaries "$MERRA2_BINARIES" "${common[@]}" \
        --arms C APC --convection cmfmc_matrix --name merra2tm5matrix
    local swap_days
    swap_days="$(find "$SWAP_BINARIES" -maxdepth 1 -name 'merra2_transport_202207[0-3][0-9]_float32.bin' -size +0c 2>/dev/null | wc -l)"
    if [[ "$swap_days" == 31 ]]; then
        python3 "$HERE/make_configs.py" --met merra2 --binaries "$SWAP_BINARIES" "${common[@]}" \
            --months 2022-07 --arms C APC --convection tm5 --name merra2era5conv
    else
        echo "field-swap binaries in $SWAP_BINARIES: $swap_days of 31 days; case merra2era5conv is not configured (stage crossover builds them)"
    fi
}

run_one() {   # $1 = config path, $2 = GPU index
    local cfg="$1" gpu="$2" dir
    dir="$WORK/runs/$(basename "$cfg" .toml)"
    mkdir -p "$dir"
    if grep -q "Forward run wall" "$dir/log.txt" 2>/dev/null; then
        echo "done already: $(basename "$dir")"
        return 0
    fi
    if flock "$GPU_LOCK_DIR/gpu$gpu.lock" env CUDA_VISIBLE_DEVICES="$gpu" "$JULIA" --project=. -t "$THREADS" \
            scripts/run_transport.jl "$cfg" > "$dir/log.txt" 2>&1 < /dev/null; then
        echo "finished on GPU $gpu: $(basename "$dir")"
    else
        echo "FAILED: $(basename "$dir") (see $dir/log.txt)"
        return 1
    fi
}

gpu_worker() {   # $1 = GPU index, $2 = list of configs; returns non-zero if a run failed
    local status=0 cfg
    while read -r cfg; do
        run_one "$cfg" "$1" || status=1
    done < "$2"
    return $status
}

stage_run() {
    : "${GPU_LOCK_DIR:?set GPU_LOCK_DIR to the directory of the gpu<N>.lock files}"
    local list status=0 pids=() i
    list="$(mktemp)"
    # July first, and the runs with all processes or convection alone before the others
    ls "$WORK"/configs/*.toml | awk -F_ '{m = $(NF-2); a = $(NF-1); print (m == "202207" ? 0 : 1) (a == "APC" || a == "C" ? 0 : 1), $0}' \
        | sort -s -k1,1 | cut -d' ' -f2- > "$list"
    read -r -a gpus <<< "${GPUS:-0 1}"
    for i in "${!gpus[@]}"; do
        awk -v n="${#gpus[@]}" -v i="$i" '(NR - 1) % n == i' "$list" > "$list.$i"
        gpu_worker "${gpus[$i]}" "$list.$i" & pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    local cfg unfinished=0
    while read -r cfg; do
        grep -q "Forward run wall" "$WORK/runs/$(basename "$cfg" .toml)/log.txt" 2>/dev/null || unfinished=$((unfinished + 1))
    done < "$list"
    rm -f "$list" "$list".*
    if [[ $status != 0 || $unfinished != 0 ]]; then
        echo "run stage: $unfinished of the configured runs have no finished log" >&2
        return 1
    fi
}

stage_reduce() {
    python3 "$HERE/sigma_spread.py" --runs "$WORK/runs" --out "$WORK/spread" --jobs "$JOBS"
    python3 "$HERE/column_lift.py" --runs "$WORK/runs" --out "$WORK/maps" --jobs "$JOBS" \
        --names {era5,merra2}_c90_2022{01,04,07,10}_C_f32
}

stage_figures() {
    python3 "$HERE/paper_figures.py" --spread "$WORK/spread" --maps "$WORK/maps" --figures "$HERE/figures"
    local cases=(era5 merra2 merra2era5conv era5ifs merra2tm5matrix) arm stem
    if [[ ! -f "$WORK/spread/merra2era5conv_c90_202207_C_f32.npz" ]]; then
        echo "case merra2era5conv is missing: results/*.csv are left as they are (they hold all five cases)"
        return 0
    fi
    mkdir -p "$WORK/figures" "$HERE/results"
    for arm in C APC; do
        python3 "$HERE/plot_spread.py" --spread "$WORK/spread" --figures "$WORK/figures" --month 202207 --arm "$arm" \
            --cases "${cases[@]}" > "$WORK/logs/spread_table_$arm.log"
        stem="spread_202207_${arm}_$(IFS=-; echo "${cases[*]}")"
        cp "$WORK/figures/$stem.csv" "$HERE/results/spread_202207_$arm.csv"
    done
}

case "$STAGE" in
    crossover) stage_crossover ;;
    configs)   stage_configs ;;
    run)       stage_run ;;
    reduce)    stage_reduce ;;
    figures)   stage_figures ;;
    all)       stage_configs; stage_run; stage_reduce; stage_figures ;;
    *) echo "unknown stage '$STAGE'"; exit 1 ;;
esac
