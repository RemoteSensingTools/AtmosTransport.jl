#!/usr/bin/env julia
# ===========================================================================
# 6-panel comparison: GEOS-Chem vs Strang vs Vertical Remap — CO2
#
# Layout:
#   Row 1: Surface CO2 VMR        (GEOS-Chem | Strang | VRemap)
#   Row 2: CO2 VMR at ~750 hPa    (GEOS-Chem | Strang | VRemap)
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_vremap_vs_strang_vs_geoschem.jl
#
# Environment variables:
#   AT_DIR       — AT output directory (default: /temp2/catrine-runs/output)
#   GC_DIR       — GEOS-Chem output directory
#   STRANG_PAT   — filename pattern for Strang output (default: "catrine_geosit_c180")
#   VREMAP_PAT   — filename pattern for VRemap output (default: "test_vremap_week")
#   OUT_GIF      — output gif path
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const GC_DIR     = get(ENV, "GC_DIR",
    joinpath(homedir(), "data", "AtmosTransport", "catrine-geoschem-runs"))
const AT_DIR     = get(ENV, "AT_DIR", "/temp2/catrine-runs/output")
const STRANG_PAT = get(ENV, "STRANG_PAT", "catrine_geosit_c180")
const VREMAP_PAT = get(ENV, "VREMAP_PAT", "test_vremap_week")
const OUT_GIF    = get(ENV, "OUT_GIF", "vremap_vs_strang_vs_geoschem_co2.gif")
const FPS        = parse(Int, get(ENV, "FPS", "4"))

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1)
const DATE_END   = DateTime(2021, 12, 7, 21, 0, 0)

# ---------------------------------------------------------------------------
# AtmosTransport loader (generic: takes a filename pattern)
# ---------------------------------------------------------------------------
function load_at_data(at_dir::String, pattern::String, rmap::CSRegridMap,
                      target_times::Vector{DateTime}, levs::Vector{Int};
                      label::String="")
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, pattern),
                               readdir(at_dir)))

    lbl = isempty(label) ? pattern : label
    @info "$lbl: found $(length(daily_files)) daily files"

    nt  = length(target_times)
    nl  = length(levs)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    fields = [zeros(Float32, rmap.nlon, rmap.nlat, nt) for _ in 1:nl]
    matched = 0

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            haskey(ds, "co2_3d") || return
            at_times = ds["time"][:]
            co2 = ds["co2_3d"]
            for (ti, tgt) in enumerate(target_times)
                diffs = [abs(Dates.value(at_t - tgt)) for at_t in at_times]
                best_idx = argmin(diffs)
                diffs[best_idx] / 60_000 > 30 && continue
                matched += 1
                for (li, lev) in enumerate(levs)
                    data_cs = Float32.(co2[:, :, :, lev, best_idx]) .* 1f6
                    regrid_cs!(buf, data_cs, rmap)
                    fields[li][:, :, ti] .= buf
                end
            end
        end
    end

    @info "$lbl: matched $matched / $nt timesteps"
    return (; fields)
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(gc, strang, vremap, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    sfc_lo, sfc_hi = 400f0, 435f0
    hpa_lo, hpa_hi = 408f0, 422f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(2100, 900), fontsize=12)

    ax_sfc_gc = GeoAxis(fig[1, 1]; dest="+proj=robin",
        title="GEOS-Chem — Surface CO2")
    ax_sfc_st = GeoAxis(fig[1, 2]; dest="+proj=robin",
        title="Strang (PPM-7) — Surface CO2")
    ax_sfc_vr = GeoAxis(fig[1, 3]; dest="+proj=robin",
        title="VRemap (Lin-Rood) — Surface CO2")

    ax_hpa_gc = GeoAxis(fig[2, 1]; dest="+proj=robin",
        title="GEOS-Chem — CO2 at ~750 hPa")
    ax_hpa_st = GeoAxis(fig[2, 2]; dest="+proj=robin",
        title="Strang (PPM-7) — CO2 at ~750 hPa")
    ax_hpa_vr = GeoAxis(fig[2, 3]; dest="+proj=robin",
        title="VRemap (Lin-Rood) — CO2 at ~750 hPa")

    z_sfc_gc = Observable(gc.fields[1][:, :, 1]')
    z_sfc_st = Observable(strang.fields[1][:, :, 1]')
    z_sfc_vr = Observable(vremap.fields[1][:, :, 1]')
    z_hpa_gc = Observable(gc.fields[2][:, :, 1]')
    z_hpa_st = Observable(strang.fields[2][:, :, 1]')
    z_hpa_vr = Observable(vremap.fields[2][:, :, 1]')

    sf1 = surface!(ax_sfc_gc, lon2d, lat2d, z_sfc_gc;
        shading=NoShading, colormap=:viridis, colorrange=(sfc_lo, sfc_hi))
    surface!(ax_sfc_st, lon2d, lat2d, z_sfc_st;
        shading=NoShading, colormap=:viridis, colorrange=(sfc_lo, sfc_hi))
    surface!(ax_sfc_vr, lon2d, lat2d, z_sfc_vr;
        shading=NoShading, colormap=:viridis, colorrange=(sfc_lo, sfc_hi))

    sf2 = surface!(ax_hpa_gc, lon2d, lat2d, z_hpa_gc;
        shading=NoShading, colormap=:viridis, colorrange=(hpa_lo, hpa_hi))
    surface!(ax_hpa_st, lon2d, lat2d, z_hpa_st;
        shading=NoShading, colormap=:viridis, colorrange=(hpa_lo, hpa_hi))
    surface!(ax_hpa_vr, lon2d, lat2d, z_hpa_vr;
        shading=NoShading, colormap=:viridis, colorrange=(hpa_lo, hpa_hi))

    for ax in [ax_sfc_gc, ax_sfc_st, ax_sfc_vr, ax_hpa_gc, ax_hpa_st, ax_hpa_vr]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 4], sf1;
        label="Surface CO2 [ppm]", width=16,
        ticks=range(sfc_lo, sfc_hi, length=8) .|> (x -> round(x, digits=0)))
    Colorbar(fig[2, 4], sf2;
        label="CO2 at ~750 hPa [ppm]", width=16,
        ticks=range(hpa_lo, hpa_hi, length=8) .|> (x -> round(x, digits=0)))

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
                           " UTC — CO2: GEOS-Chem vs Strang vs VRemap")
    Label(fig[0, 1:4], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        z_sfc_gc[] = gc.fields[1][:, :, frame_num]'
        z_sfc_st[] = strang.fields[1][:, :, frame_num]'
        z_sfc_vr[] = vremap.fields[1][:, :, frame_num]'
        z_hpa_gc[] = gc.fields[2][:, :, frame_num]'
        z_hpa_st[] = strang.fields[2][:, :, frame_num]'
        z_hpa_vr[] = vremap.fields[2][:, :, frame_num]'

        title_obs[] = Dates.format(times[frame_num], "yyyy-mm-dd HH:MM") *
                      " UTC — CO2: GEOS-Chem vs Strang vs VRemap"
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

    @info "Loading GEOS-Chem CO2..."
    gc = load_geoschem_nc(GC_DIR, rmap,
        "SpeciesConcVV_CO2", levs;
        date_start=DATE_START, date_end=DATE_END,
        scale=1e6)

    @info "Loading Strang (PPM-7) CO2 (pattern=$STRANG_PAT)..."
    strang = load_at_data(AT_DIR, STRANG_PAT, rmap, gc.times, levs;
                           label="Strang")

    @info "Loading VRemap (Lin-Rood) CO2 (pattern=$VREMAP_PAT)..."
    vremap = load_at_data(AT_DIR, VREMAP_PAT, rmap, gc.times, levs;
                           label="VRemap")

    make_animation(gc, strang, vremap, gc.times, rmap)
end

main()
