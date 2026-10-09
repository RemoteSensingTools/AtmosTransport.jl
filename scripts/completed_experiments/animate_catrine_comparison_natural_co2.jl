#!/usr/bin/env julia
# ===========================================================================
# 4-panel comparison animation: GEOS-Chem vs AtmosTransport (CATRINE D7.1)
# Natural CO2 = total CO2 - fossil CO2
#
# Layout:
#   Row 1: Surface natural CO2 VMR       (GEOS-Chem | AtmosTransport)
#   Row 2: natural CO2 VMR at ~750hPa    (GEOS-Chem | AtmosTransport)
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_catrine_comparison_natural_co2.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const GC_DIR = get(ENV, "GC_DIR",
    joinpath(homedir(), "data", "AtmosTransport", "catrine-geoschem-runs"))
const AT_DIR   = get(ENV, "AT_DIR", "/temp2/catrine-runs/output")
const OUT_GIF  = get(ENV, "OUT_GIF", "catrine_comparison_natural_co2.gif")
const FPS      = parse(Int, get(ENV, "FPS", "4"))

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1)
const DATE_END   = DateTime(2021, 12, 14, 21, 0, 0)

# ---------------------------------------------------------------------------
# GEOS-Chem loader — compute natural CO2 = CO2 - fossil CO2
# ---------------------------------------------------------------------------
function load_geoschem_natural(dir::String, rmap::CSRegridMap, levs::Vector{Int};
                                date_start::DateTime, date_end::DateTime)
    all_files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"),
                             readdir(dir)))
    files = String[]
    times = DateTime[]
    for f in all_files
        m = match(r"(\d{8})_(\d{4})z", f)
        m === nothing && continue
        dt = DateTime(m[1] * m[2], dateformat"yyyymmddHHMM")
        if date_start <= dt <= date_end
            push!(files, f)
            push!(times, dt)
        end
    end

    @info "GEOS-Chem: $(length(files)) snapshots ($(times[1]) → $(times[end]))"

    nt  = length(files)
    nl  = length(levs)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    fields = [zeros(Float32, rmap.nlon, rmap.nlat, nt) for _ in 1:nl]

    for (ti, fname) in enumerate(files)
        NCDataset(joinpath(dir, fname), "r") do ds
            co2_data = ds["SpeciesConcVV_CO2"]
            fco2_data = ds["SpeciesConcVV_FossilCO2"]
            for (li, lev) in enumerate(levs)
                co2_cs = Float64.(co2_data[:, :, :, lev, 1]) .* 1e6
                fco2_cs = Float64.(fco2_data[:, :, :, lev, 1]) .* 1e6
                nat_cs = co2_cs .- fco2_cs
                regrid_cs!(buf, nat_cs, rmap)
                fields[li][:, :, ti] .= buf
            end
        end
    end

    return (; times, fields)
end

# ---------------------------------------------------------------------------
# AtmosTransport loader — compute natural CO2 = co2_3d - fossil_co2_3d
# ---------------------------------------------------------------------------
function load_atmostr_natural(at_dir::String, rmap::CSRegridMap,
                               target_times::Vector{DateTime}, levs::Vector{Int})
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, "catrine_geosit"),
                               readdir(at_dir)))

    nt  = length(target_times)
    nl  = length(levs)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    fields = [zeros(Float32, rmap.nlon, rmap.nlat, nt) for _ in 1:nl]
    filled = falses(nt)   # which target frames actually received AtmosTransport data

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            haskey(ds, "co2_3d") || return
            haskey(ds, "fossil_co2_3d") || return
            at_times = ds["time"][:]
            co2 = ds["co2_3d"]
            fco2 = ds["fossil_co2_3d"]
            for (ti, tgt) in enumerate(target_times)
                diffs = [abs(Dates.value(at_t - tgt)) for at_t in at_times]
                best_idx = argmin(diffs)
                diffs[best_idx] / 60_000 > 30 && continue
                for (li, lev) in enumerate(levs)
                    co2_cs = Float32.(co2[:, :, :, lev, best_idx]) .* 1f6
                    fco2_cs = Float32.(fco2[:, :, :, lev, best_idx]) .* 1f6
                    nat_cs = co2_cs .- fco2_cs
                    regrid_cs!(buf, nat_cs, rmap)
                    fields[li][:, :, ti] .= buf
                end
                filled[ti] = true
            end
        end
    end

    # Time-axis assert: every target frame must have a matching AtmosTransport
    # snapshot. Unfilled frames stay all-zero and render as blank panels (the
    # Dec-11–14 "missing last days" bug); fail loudly with the gap rather than
    # silently animating blanks. Regenerate the AT NetCDF to cover these times.
    if !all(filled)
        miss = target_times[.!filled]
        error("AtmosTransport coverage gap: $(count(!, filled))/$nt target frames have " *
              "no snapshot within 30 min — frames would render blank. Missing times: " *
              "$(first(miss)) … $(last(miss)) ($(length(miss)) frames). " *
              "Regenerate the AtmosTransport run to cover the full GEOS-Chem time axis.")
    end

    @info "AtmosTransport: matched all $nt timesteps"
    return (; fields)
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(gc, at, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    sfc_lo, sfc_hi = 400f0, 430f0
    hpa_lo, hpa_hi = 408f0, 420f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(1600, 900), fontsize=12)

    ax_sfc_gc = GeoAxis(fig[1, 1]; dest="+proj=robin",
        title="GEOS-Chem — Surface natural CO2")
    ax_sfc_at = GeoAxis(fig[1, 2]; dest="+proj=robin",
        title="AtmosTransport — Surface natural CO2")
    ax_hpa_gc = GeoAxis(fig[2, 1]; dest="+proj=robin",
        title="GEOS-Chem — natural CO2 at ~750 hPa")
    ax_hpa_at = GeoAxis(fig[2, 2]; dest="+proj=robin",
        title="AtmosTransport — natural CO2 at ~750 hPa")

    z_sfc_gc = Observable(gc.fields[1][:, :, 1]')
    z_sfc_at = Observable(at.fields[1][:, :, 1]')
    z_hpa_gc = Observable(gc.fields[2][:, :, 1]')
    z_hpa_at = Observable(at.fields[2][:, :, 1]')

    sf1 = surface!(ax_sfc_gc, lon2d, lat2d, z_sfc_gc;
        shading=NoShading, colormap=:viridis, colorrange=(sfc_lo, sfc_hi))
    surface!(ax_sfc_at, lon2d, lat2d, z_sfc_at;
        shading=NoShading, colormap=:viridis, colorrange=(sfc_lo, sfc_hi))

    sf2 = surface!(ax_hpa_gc, lon2d, lat2d, z_hpa_gc;
        shading=NoShading, colormap=:viridis, colorrange=(hpa_lo, hpa_hi))
    surface!(ax_hpa_at, lon2d, lat2d, z_hpa_at;
        shading=NoShading, colormap=:viridis, colorrange=(hpa_lo, hpa_hi))

    for ax in [ax_sfc_gc, ax_sfc_at, ax_hpa_gc, ax_hpa_at]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 3], sf1;
        label="Surface natural CO2 [ppm]", width=16,
        ticks=range(sfc_lo, sfc_hi, length=7) .|> (x -> round(x, digits=0)))
    Colorbar(fig[2, 3], sf2;
        label="natural CO2 at ~750 hPa [ppm]", width=16,
        ticks=range(hpa_lo, hpa_hi, length=7) .|> (x -> round(x, digits=0)))

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
                           " UTC — Natural CO2 (total - fossil)")
    Label(fig[0, 1:3], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        z_sfc_gc[] = gc.fields[1][:, :, frame_num]'
        z_sfc_at[] = at.fields[1][:, :, frame_num]'
        z_hpa_gc[] = gc.fields[2][:, :, frame_num]'
        z_hpa_at[] = at.fields[2][:, :, frame_num]'

        title_obs[] = Dates.format(times[frame_num], "yyyy-mm-dd HH:MM") *
                      " UTC — Natural CO2 (total - fossil)"
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    isdir(GC_DIR) || error("GEOS-Chem directory not found: $GC_DIR")
    isdir(AT_DIR) || error("AtmosTransport output not found: $AT_DIR")

    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    @info "Loading CS coordinates from $(gc_files[1])"
    cs_lons, cs_lats = load_cs_coordinates(joinpath(GC_DIR, gc_files[1]))

    @info "Building CS -> lat-lon regridding map (1 deg)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    levs = [LEV_SURFACE, LEV_750HPA]

    @info "Loading GEOS-Chem natural CO2..."
    gc = load_geoschem_natural(GC_DIR, rmap, levs;
        date_start=DATE_START, date_end=DATE_END)

    @info "Loading AtmosTransport natural CO2..."
    at = load_atmostr_natural(AT_DIR, rmap, gc.times, levs)

    make_animation(gc, at, gc.times, rmap)
end

main()
