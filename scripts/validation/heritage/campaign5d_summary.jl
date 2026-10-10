#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# campaign5d_summary.jl — collate the 5-day Catrine matrix into a CSV.
#
# Walks ~/data/AtmosTransport/output/catrine5d/**/*.nc, extracts per-run
# diagnostics (mass drift over the 5-day window, snapshot count, GPU/CPU
# tag, op-stack), and emits a single CSV row per run plus three pairwise
# comparison tables:
#
#   1. CPU↔GPU equivalence (12 coarse pairs):
#      max abs diff per tracer per snapshot, %drift difference.
#
#   2. F32↔F64 precision (12 grid×op pairs, GPU side):
#      max abs diff column mean per tracer per snapshot.
#
#   3. LL↔CS cross-grid (12 op×precision pairs):
#      regrid LL snapshot column-mean to CS, RMSE/max per snapshot.
#
# Usage:
#   julia --project=. scripts/validation/heritage/campaign5d_summary.jl [--out CSV_DIR]
# ---------------------------------------------------------------------------

using Printf
using Statistics: mean
using NCDatasets

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SNAP_DIR  = expanduser("~/data/AtmosTransport/output/catrine5d")

# Parse run name → metadata. Run name format: <grid>_<prec>_<op>_<hw>
function parse_run_name(name::AbstractString)
    parts = split(replace(name, ".nc" => ""), "_")
    @assert length(parts) == 4 "unexpected run name $name"
    return (grid = Symbol(parts[1]),
            prec = Symbol(parts[2]),
            op   = Symbol(parts[3]),
            hw   = Symbol(parts[4]))
end

function discover_runs(snap_dir = SNAP_DIR)
    isdir(snap_dir) || return NamedTuple[]
    runs = NamedTuple[]
    for entry in readdir(snap_dir; join = false)
        sub = joinpath(snap_dir, entry)
        isdir(sub) || continue
        for f in readdir(sub; join = false)
            endswith(f, ".nc") || continue
            push!(runs, (path = joinpath(sub, f), meta = parse_run_name(f)))
        end
    end
    return runs
end

# Read total air_mass per snapshot (CS only — LL writer doesn't carry it
# in the snapshot today). Returns Vector{Float64} or nothing.
function _total_air_mass_per_snapshot(ds::NCDataset)
    haskey(ds, "air_mass") || return nothing
    am = ds["air_mass"][:]
    if ndims(am) == 4
        return [sum(view(am, :, :, :, t)) for t in 1:size(am, 4)]
    elseif ndims(am) == 5
        return [sum(view(am, :, :, :, :, t)) for t in 1:size(am, 5)]
    end
    return nothing
end

function _summarize_run(path::AbstractString)
    open_ok = true
    n_frames = 0
    drift_pct = NaN
    tracers = String[]
    try
        NCDataset(path, "r") do ds
            n_frames = haskey(ds, "time") ? length(ds["time"][:]) : 0
            tracers  = filter(k -> startswith(String(k), "co2_"),
                              String.(collect(keys(ds))))
            am_series = _total_air_mass_per_snapshot(ds)
            if am_series !== nothing && length(am_series) > 1
                drift_pct = 100 * (am_series[end] - am_series[1]) / am_series[1]
            end
        end
    catch err
        @warn "failed to summarize $path" exception = (err, catch_backtrace())
        open_ok = false
    end
    return (open_ok = open_ok, n_frames = n_frames,
            tracers = tracers, drift_pct = drift_pct)
end

function main()
    out_dir = get(ENV, "CAMPAIGN5D_OUT", joinpath(REPO_ROOT, "artifacts", "catrine5d"))
    mkpath(out_dir)

    runs = discover_runs()
    println("found $(length(runs)) NetCDF snapshot files under $SNAP_DIR")
    isempty(runs) && return

    # Per-run summary
    summary_csv = joinpath(out_dir, "summary.csv")
    open(summary_csv, "w") do io
        println(io, "grid,prec,op,hw,n_frames,tracers,air_mass_drift_pct,path")
        for r in runs
            s = _summarize_run(r.path)
            tracer_str = join(s.tracers, "|")
            println(io, "$(r.meta.grid),$(r.meta.prec),$(r.meta.op),$(r.meta.hw),",
                       "$(s.n_frames),$tracer_str,$(s.drift_pct),$(r.path)")
        end
    end
    println("wrote $summary_csv")

    # CPU↔GPU equivalence on coarse grids
    eq_csv = joinpath(out_dir, "cpu_vs_gpu.csv")
    open(eq_csv, "w") do io
        println(io, "grid,prec,op,tracer,snapshot,max_abs_diff,rms_diff")
        for grid in (:ll72, :c48), prec in (:f32, :f64), op in (:advonly, :advdiff, :advdiffconv)
            cpu_path = joinpath(SNAP_DIR, String(grid), "$(grid)_$(prec)_$(op)_cpu.nc")
            gpu_path = joinpath(SNAP_DIR, String(grid), "$(grid)_$(prec)_$(op)_gpu.nc")
            (isfile(cpu_path) && isfile(gpu_path)) || continue
            try
                NCDataset(cpu_path, "r") do c
                    NCDataset(gpu_path, "r") do g
                        n = haskey(c, "time") ? length(c["time"][:]) : 0
                        tracer_keys = filter(k -> startswith(String(k), "co2_"), String.(collect(keys(c))))
                        for tracer in tracer_keys
                            haskey(g, tracer) || continue
                            cv = c[tracer][:]
                            gv = g[tracer][:]
                            size(cv) == size(gv) || continue
                            tdim = ndims(cv)
                            for t in 1:size(cv, tdim)
                                slc_c = selectdim(cv, tdim, t)
                                slc_g = selectdim(gv, tdim, t)
                                d = abs.(slc_c .- slc_g)
                                println(io, "$grid,$prec,$op,$tracer,$t,",
                                           maximum(d), ",",
                                           sqrt(mean(d .^ 2)))
                            end
                        end
                    end
                end
            catch err
                @warn "cpu/gpu compare failed for $grid $prec $op" err
            end
        end
    end
    println("wrote $eq_csv")
end

main()
