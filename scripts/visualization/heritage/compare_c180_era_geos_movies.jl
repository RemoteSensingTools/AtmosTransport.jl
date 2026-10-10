#!/usr/bin/env julia
# Side-by-side C180 ERA5-vs-GEOS movie generator with shared color ranges.
#
# Defaults compare the new ERA5-on-GEOS-native C180/L85 campaign against
# GEOS-IT native C180/L72. Each output movie has three columns:
#   ERA5 | GEOS-IT | GEOS-IT - ERA5
#
# Environment:
#   ERA_DIR     default /temp1/c180_era5_geosgrid_cfl85_3d
#   GEOS_DIR    default /temp1/c180_geosit_native_3d
#   OUT_DIR     default /tmp/tm5_smoke/viz_era_geos_c180_comparison_movies
#   RUNS        comma-separated run names, default advonly_ppm,advdiff_ppm,fullphysics_ppm
#   TRACERS     comma-separated tracer names, default co2_natural,co2_fossil
#   SPECS       comma-separated specs: column_mean,surface,mid_trop,upper_trop
#   FPS         default 4
#   RESOLUTION  default 360x181
#   TRIM        robust color trim quantiles, default 0.01,0.99

using CairoMakie
using Printf

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Visualization:
    SnapshotDataset, SnapshotRegridCache, RasterField,
    open_snapshot, snapshot_times, fieldview, frame_indices, as_raster,
    robust_colorrange

const DEFAULT_ERA_DIR = "/temp1/c180_era5_geosgrid_cfl85_3d"
const DEFAULT_GEOS_DIR = "/temp1/c180_geosit_native_3d"
const DEFAULT_OUT_DIR = "/tmp/tm5_smoke/viz_era_geos_c180_comparison_movies"

struct MovieSpec
    name::String
    transform::Symbol
    era_level::Union{Nothing, Int}
    geos_level::Union{Nothing, Int}
end

function _split_env(name::AbstractString, default::AbstractString)
    return [String(strip(x)) for x in split(get(ENV, name, default), ",") if !isempty(strip(x))]
end

function _parse_resolution(raw::AbstractString)
    m = match(r"^(\d+)x(\d+)$", lowercase(strip(raw)))
    m === nothing && error("RESOLUTION must look like 360x181, got $(raw)")
    return (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

function _parse_trim(raw::AbstractString)
    parts = split(raw, ",")
    length(parts) == 2 || error("TRIM must look like 0.01,0.99, got $(raw)")
    lo, hi = parse.(Float64, strip.(parts))
    0.0 <= lo < hi <= 1.0 || error("TRIM quantiles must satisfy 0 <= lo < hi <= 1")
    return (lo, hi)
end

function _parse_specs(raw_specs)
    out = MovieSpec[]
    for raw in raw_specs
        name = lowercase(strip(raw))
        if name == "column_mean"
            push!(out, MovieSpec("column_mean", :column_mean, nothing, nothing))
        elseif name == "surface"
            push!(out, MovieSpec("surface", :surface_slice, nothing, nothing))
        elseif name == "mid_trop"
            # Fast tropospheric visual diagnostics matching the per-source
            # single-run scripts: L85 ERA level 65, L72 GEOS level 55.
            push!(out, MovieSpec("mid_trop", :level_slice, 65, 55))
        elseif name == "upper_trop"
            # Fast upper-tropospheric visual diagnostics matching the
            # per-source single-run scripts: L85 ERA level 40, L72 GEOS level 35.
            push!(out, MovieSpec("upper_trop", :level_slice, 40, 35))
        else
            error("Unknown SPECS entry $(raw). Supported: column_mean,surface,mid_trop,upper_trop")
        end
    end
    isempty(out) && error("No specs requested")
    return out
end

function _snapshot_path(root::AbstractString, run::AbstractString)
    path = joinpath(root, run * ".nc")
    isfile(path) || error("Snapshot file not found: $(path)")
    return path
end

function _matched_indices(reference_times, other_times)
    return [argmin(abs.(other_times .- t)) for t in reference_times]
end

function _load_rasters(snapshot::SnapshotDataset, tracer::AbstractString,
                       spec::MovieSpec, indices;
                       level::Union{Nothing, Int},
                       resolution::Tuple{Int, Int},
                       cache::SnapshotRegridCache)
    rasters = RasterField[]
    for idx in indices
        field = fieldview(snapshot, tracer;
                          transform = spec.transform,
                          time = idx,
                          level = level,
                          unit = :ppm)
        push!(rasters, as_raster(field; resolution = resolution, cache = cache))
    end
    return rasters
end

function _shared_range(era_rasters, geos_rasters, tracer::AbstractString, trim)
    lo, hi = robust_colorrange(vcat(era_rasters, geos_rasters); trim = trim)
    if occursin("fossil", tracer)
        # Fossil CO2 should be nonnegative; tiny negative values are numerical
        # noise and would waste color resolution if allowed to drive the lower
        # bound.
        lo = min(0.0, lo)
    end
    if !isfinite(lo) || !isfinite(hi) || lo == hi
        lo, hi = 0.0, 1.0
    end
    return (Float32(lo), Float32(hi))
end

function _difference_range(era_rasters, geos_rasters; trim)
    diffs = Matrix{Float64}[]
    for idx in eachindex(era_rasters, geos_rasters)
        push!(diffs, geos_rasters[idx].values .- era_rasters[idx].values)
    end
    lo, hi = robust_colorrange(diffs; trim = trim)
    maxabs = max(abs(lo), abs(hi))
    (!isfinite(maxabs) || maxabs == 0.0) && (maxabs = 1.0)
    return (Float32(-maxabs), Float32(maxabs))
end

function _heatmap_axis!(fig, slot, raster::RasterField, obs; title, colorrange, colormap)
    ax = Axis(fig[slot...];
              aspect = DataAspect(),
              title = title,
              xlabel = "longitude",
              ylabel = "latitude")
    ax.xticksvisible[] = false
    ax.yticksvisible[] = false
    ax.xticklabelsvisible[] = false
    ax.yticklabelsvisible[] = false
    hm = heatmap!(ax, raster.lons, raster.lats, obs;
                  colormap = colormap,
                  colorrange = colorrange)
    return ax, hm
end

function _nice_time_label(hours)
    @sprintf("t = %.0f h", Float64(hours))
end

function _write_movie(era_path::String, geos_path::String, out_path::String;
                      run::String,
                      tracer::String,
                      spec::MovieSpec,
                      fps::Int,
                      resolution::Tuple{Int, Int},
                      trim)
    era = open_snapshot(era_path)
    geos = open_snapshot(geos_path)
    era_indices = frame_indices(era, :all)
    geos_indices = _matched_indices(snapshot_times(era), snapshot_times(geos))
    length(era_indices) == length(geos_indices) ||
        error("Internal time-match mismatch for $(run) $(tracer) $(spec.name)")

    cache = SnapshotRegridCache()
    @info "Loading rasters" run tracer spec=spec.name
    era_rasters = _load_rasters(era, tracer, spec, era_indices;
                                level = spec.era_level,
                                resolution = resolution,
                                cache = cache)
    geos_rasters = _load_rasters(geos, tracer, spec, geos_indices;
                                 level = spec.geos_level,
                                 resolution = resolution,
                                 cache = cache)

    shared_cr = _shared_range(era_rasters, geos_rasters, tracer, trim)
    diff_cr = _difference_range(era_rasters, geos_rasters; trim = trim)
    @info "Color ranges" run tracer spec=spec.name shared=shared_cr diff=diff_cr

    mkpath(dirname(out_path))
    fig = Figure(size = (1500, 520), fontsize = 12)
    era_obs = Observable(era_rasters[1].values)
    geos_obs = Observable(geos_rasters[1].values)
    diff_obs = Observable(geos_rasters[1].values .- era_rasters[1].values)

    transform_label = replace(spec.name, "_" => " ")
    ax_era, hm_era = _heatmap_axis!(fig, (1, 1), era_rasters[1], era_obs;
                                    title = "ERA5 GEOS-native C180 - $(transform_label)",
                                    colorrange = shared_cr,
                                    colormap = :viridis)
    ax_geos, hm_geos = _heatmap_axis!(fig, (1, 2), geos_rasters[1], geos_obs;
                                      title = "GEOS-IT native C180 - $(transform_label)",
                                      colorrange = shared_cr,
                                      colormap = :viridis)
    ax_diff, hm_diff = _heatmap_axis!(fig, (1, 3), era_rasters[1], diff_obs;
                                      title = "GEOS-IT - ERA5",
                                      colorrange = diff_cr,
                                      colormap = :RdBu)
    Colorbar(fig[1, 4], hm_geos; label = "$(tracer) [ppm]", width = 16)
    Colorbar(fig[1, 5], hm_diff; label = "difference [ppm]", width = 16)

    title_obs = Observable("$(run) $(tracer) $(transform_label), " *
                           _nice_time_label(era_rasters[1].time))
    Label(fig[0, 1:5], title_obs; fontsize = 18, font = :bold)

    @info "Writing movie" out=out_path frames=length(era_rasters)
    record(fig, out_path, eachindex(era_rasters); framerate = fps) do frame
        er = era_rasters[frame]
        gr = geos_rasters[frame]
        era_obs[] = er.values
        geos_obs[] = gr.values
        diff_obs[] = gr.values .- er.values
        label = "$(run) $(tracer) $(transform_label), " * _nice_time_label(er.time)
        title_obs[] = label
        ax_era.title[] = "ERA5 GEOS-native C180 - $(transform_label)"
        ax_geos.title[] = "GEOS-IT native C180 - $(transform_label)"
        ax_diff.title[] = "GEOS-IT - ERA5"
    end
    @info "Saved" out=out_path
    return out_path
end

function main()
    era_dir = get(ENV, "ERA_DIR", DEFAULT_ERA_DIR)
    geos_dir = get(ENV, "GEOS_DIR", DEFAULT_GEOS_DIR)
    out_dir = get(ENV, "OUT_DIR", DEFAULT_OUT_DIR)
    runs = _split_env("RUNS", "advonly_ppm,advdiff_ppm,fullphysics_ppm")
    tracers = _split_env("TRACERS", "co2_natural,co2_fossil")
    specs = _parse_specs(_split_env("SPECS", "column_mean"))
    fps = parse(Int, get(ENV, "FPS", "4"))
    resolution = _parse_resolution(get(ENV, "RESOLUTION", "360x181"))
    trim = _parse_trim(get(ENV, "TRIM", "0.01,0.99"))

    mkpath(out_dir)
    inventory = String[]
    for run in runs, tracer in tracers, spec in specs
        era_path = _snapshot_path(era_dir, run)
        geos_path = _snapshot_path(geos_dir, run)
        out_path = joinpath(out_dir, "$(run)_$(tracer)_$(spec.name)_era_geos_compare.mp4")
        push!(inventory, _write_movie(era_path, geos_path, out_path;
                                      run, tracer, spec, fps, resolution, trim))
    end

    open(joinpath(out_dir, "README.txt"), "w") do io
        println(io, "C180 ERA5-vs-GEOS comparison movies")
        println(io, "ERA_DIR: ", era_dir)
        println(io, "GEOS_DIR: ", geos_dir)
        println(io, "Resolution: ", resolution[1], "x", resolution[2])
        println(io, "Shared color range per movie: robust trim ", trim)
        println(io, "Panels: ERA5 | GEOS-IT | GEOS-IT - ERA5")
        println(io, "Files:")
        for path in inventory
            println(io, "  ", basename(path))
        end
    end
    println("Wrote $(length(inventory)) movie(s) to $(out_dir)")
end

main()
