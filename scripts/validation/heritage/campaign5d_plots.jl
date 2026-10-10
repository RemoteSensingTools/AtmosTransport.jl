#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# campaign5d_plots.jl — visualize the 5-day Catrine campaign deltas using the
# project Visualization API (`SnapshotDataset` + `fieldview` + `as_raster`
# + `mapplot!`).  CS panels are conservatively regridded to a 360×181 LL
# raster via the same `SnapshotRegridCache` the docs Literate tutorials use,
# so the same code path serves runtime visualization.
#
# Outputs under `artifacts/catrine5d/plots/`:
#
#   1. `<tracer>_overview.png` — per-tracer column-mean at the final
#      snapshot, one panel per (grid, op-stack), F64 GPU only.  Gives a
#      "what the dynamics did" overview across LL 72×37, LL 144×73, CS
#      C48, CS C180.
#
#   2. `<tracer>_f32_minus_f64.png` — column-mean abs diff between F32
#      and F64 GPU runs at final snapshot, one panel per (grid,
#      op-stack).  Quantifies precision sensitivity.
#
#   3. `<tracer>_cpu_minus_gpu.png` — column-mean abs diff between CPU
#      and GPU runs at final snapshot for the coarse grids
#      (LL 72×37 + CS C48), one panel per (op-stack, prec).  The
#      equivalence check.
#
# Skips runs whose NetCDF has any NaN frames (e.g. the C180 binaries
# from the original campaign — only valid after the steps=12 fix).
#
# Usage: julia --project=. scripts/validation/heritage/campaign5d_plots.jl
# ---------------------------------------------------------------------------

using Printf
using CairoMakie
using AtmosTransport
using AtmosTransport.Visualization: SnapshotDataset, SnapshotRegridCache,
                                     open_snapshot, fieldview, as_raster,
                                     frame_indices
using AtmosTransport.Visualization: mapplot!

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SNAP_DIR  = expanduser("~/data/AtmosTransport/output/catrine5d")
const OUT_DIR   = joinpath(REPO_ROOT, "artifacts", "catrine5d", "plots")
const RES       = (360, 181)        # LL raster resolution for CS unfolding

const GRIDS = (:ll72, :ll144, :c48, :c180)
const OPS   = (:advonly, :advdiff, :advdiffconv)
const PRECS = (:f32, :f64)

mkpath(OUT_DIR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

snapshot_path(grid, prec, op, hw) =
    joinpath(SNAP_DIR, String(grid), "$(grid)_$(prec)_$(op)_$(hw).nc")

function _final_raster(path::AbstractString, tracer::AbstractString,
                       cache::SnapshotRegridCache)
    isfile(path) || return nothing
    snap = open_snapshot(path)
    indices = frame_indices(snap, :all)
    isempty(indices) && return nothing
    field = fieldview(snap, tracer; transform = :column_mean, time = indices[end])
    raster = as_raster(field; resolution = RES, cache = cache)
    all(isfinite, raster.values) || return nothing
    return raster
end

function _heatmap!(ax, raster; colormap = :viridis, colorrange = nothing)
    cr = colorrange === nothing ? extrema(raster.values) : colorrange
    Makie.heatmap!(ax, raster.lons, raster.lats, raster.values;
                   colormap = colormap, colorrange = cr)
end

# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

function fig_tracer_overview(tracer::AbstractString)
    cache = SnapshotRegridCache()
    fig = Figure(size = (1700, 1100))
    Label(fig[0, 1:length(OPS)],
          "$tracer column-mean at day-4 (96 h), F64 GPU";
          fontsize = 18, halign = :center)
    rasters = Vector{Any}(undef, length(GRIDS) * length(OPS))
    idx = 1
    for grid in GRIDS, op in OPS
        rasters[idx] = _final_raster(snapshot_path(grid, :f64, op, :gpu),
                                     tracer, cache)
        idx += 1
    end
    finite = filter(!isnothing, rasters)
    cr = isempty(finite) ? (0.0, 1.0) :
         (minimum(r -> minimum(r.values), finite),
          maximum(r -> maximum(r.values), finite))
    idx = 1
    for (gi, grid) in enumerate(GRIDS), (oi, op) in enumerate(OPS)
        ax = Axis(fig[gi, oi]; title = "$grid · $op")
        if rasters[idx] === nothing
            text!(ax, 0.5, 0.5; text = "n/a", align = (:center, :center))
        else
            _heatmap!(ax, rasters[idx]; colorrange = cr)
        end
        idx += 1
    end
    out_path = joinpath(OUT_DIR, "$(tracer)_overview.png")
    save(out_path, fig)
    println("wrote $out_path")
end

function fig_precision_diff(tracer::AbstractString)
    cache = SnapshotRegridCache()
    fig = Figure(size = (1700, 1100))
    Label(fig[0, 1:length(OPS)],
          "$tracer column-mean |F32 − F64| at day-4, GPU";
          fontsize = 18, halign = :center)
    for (gi, grid) in enumerate(GRIDS), (oi, op) in enumerate(OPS)
        f32 = _final_raster(snapshot_path(grid, :f32, op, :gpu), tracer, cache)
        f64 = _final_raster(snapshot_path(grid, :f64, op, :gpu), tracer, cache)
        ax = Axis(fig[gi, oi]; title = "$grid · $op")
        if f32 === nothing || f64 === nothing
            text!(ax, 0.5, 0.5; text = "n/a", align = (:center, :center))
            continue
        end
        diff = abs.(f32.values .- f64.values)
        Makie.heatmap!(ax, f32.lons, f32.lats, diff; colormap = :magma)
        Label(fig[gi, oi, Top()],
              @sprintf("max=%.2e", maximum(diff)); padding = (0, 0, 5, 0))
    end
    out_path = joinpath(OUT_DIR, "$(tracer)_f32_minus_f64.png")
    save(out_path, fig)
    println("wrote $out_path")
end

function fig_cpu_gpu_diff(tracer::AbstractString)
    cache = SnapshotRegridCache()
    coarse = (:ll72, :c48)
    fig = Figure(size = (1700, 900))
    Label(fig[0, 1:length(OPS)],
          "$tracer column-mean |CPU − GPU| at day-4 (coarse grids only)";
          fontsize = 18, halign = :center)
    row = 1
    for prec in PRECS, grid in coarse
        for (oi, op) in enumerate(OPS)
            cpu = _final_raster(snapshot_path(grid, prec, op, :cpu), tracer, cache)
            gpu = _final_raster(snapshot_path(grid, prec, op, :gpu), tracer, cache)
            ax = Axis(fig[row, oi]; title = "$grid · $prec · $op")
            if cpu === nothing || gpu === nothing
                text!(ax, 0.5, 0.5; text = "n/a", align = (:center, :center))
                continue
            end
            diff = abs.(cpu.values .- gpu.values)
            Makie.heatmap!(ax, cpu.lons, cpu.lats, diff; colormap = :magma)
            Label(fig[row, oi, Top()],
                  @sprintf("max=%.2e", maximum(diff)); padding = (0, 0, 5, 0))
        end
        row += 1
    end
    out_path = joinpath(OUT_DIR, "$(tracer)_cpu_minus_gpu.png")
    save(out_path, fig)
    println("wrote $out_path")
end

# ---------------------------------------------------------------------------

for tracer in ("co2_natural", "co2_fossil")
    fig_tracer_overview(tracer)
    fig_precision_diff(tracer)
    fig_cpu_gpu_diff(tracer)
end

println("\nWrote plots under $OUT_DIR")
