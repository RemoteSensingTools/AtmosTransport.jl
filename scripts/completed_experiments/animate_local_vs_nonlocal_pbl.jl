#!/usr/bin/env julia
# ===========================================================================
# 4-panel comparison: Local K-diffusion vs Non-Local PBL (Holtslag-Boville)
#
# Layout:
#   Row 1: Surface fossil CO2 VMR      (Local PBL | Non-Local PBL)
#   Row 2: fossil CO2 VMR at ~750 hPa  (Local PBL | Non-Local PBL)
#
# Both datasets are GEOS-IT C180 cubed-sphere, converted to NetCDF.
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_local_vs_nonlocal_pbl.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates
using Printf

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const AT_DIR   = get(ENV, "AT_DIR", "/temp2/catrine-runs/output")
const OUT_GIF  = get(ENV, "OUT_GIF", "catrine_local_vs_nonlocal_pbl.gif")
const FPS      = parse(Int, get(ENV, "FPS", "4"))

const LOCAL_PATTERN    = "catrine_geosit_c180_2021"
const NONLOCAL_PATTERN = "catrine_geosit_c180_nonlocal_2021"

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1)
const DATE_END   = DateTime(2021, 12, 7, 21, 0, 0)

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(local_data, nonlocal_data, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    sfc_max = 20f0
    hpa_max = 5f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(1600, 900), fontsize=12)

    ax_sfc_lo = GeoAxis(fig[1, 1]; dest="+proj=robin",
        title="Local K-diffusion — Surface fossil CO2")
    ax_sfc_nl = GeoAxis(fig[1, 2]; dest="+proj=robin",
        title="Non-Local PBL — Surface fossil CO2")
    ax_hpa_lo = GeoAxis(fig[2, 1]; dest="+proj=robin",
        title="Local K-diffusion — ~750 hPa")
    ax_hpa_nl = GeoAxis(fig[2, 2]; dest="+proj=robin",
        title="Non-Local PBL — ~750 hPa")

    z_sfc_lo = Observable(local_data.fields[1][:, :, 1]')
    z_sfc_nl = Observable(nonlocal_data.fields[1][:, :, 1]')
    z_hpa_lo = Observable(local_data.fields[2][:, :, 1]')
    z_hpa_nl = Observable(nonlocal_data.fields[2][:, :, 1]')

    sf1 = surface!(ax_sfc_lo, lon2d, lat2d, z_sfc_lo;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)
    surface!(ax_sfc_nl, lon2d, lat2d, z_sfc_nl;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)

    sf2 = surface!(ax_hpa_lo, lon2d, lat2d, z_hpa_lo;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)
    surface!(ax_hpa_nl, lon2d, lat2d, z_hpa_nl;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)

    for ax in [ax_sfc_lo, ax_sfc_nl, ax_hpa_lo, ax_hpa_nl]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 3], sf1; label="Surface fossil CO2 [ppm]", width=16,
        ticks=range(0, sfc_max, length=6) .|> (x -> round(x, digits=1)))
    Colorbar(fig[2, 3], sf2; label="fossil CO2 at ~750 hPa [ppm]", width=16,
        ticks=range(0, hpa_max, length=6) .|> (x -> round(x, digits=2)))

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
                           " UTC — Local vs Non-Local PBL Diffusion")
    Label(fig[0, 1:3], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        z_sfc_lo[] = local_data.fields[1][:, :, frame_num]'
        z_sfc_nl[] = nonlocal_data.fields[1][:, :, frame_num]'
        z_hpa_lo[] = local_data.fields[2][:, :, frame_num]'
        z_hpa_nl[] = nonlocal_data.fields[2][:, :, frame_num]'

        title_obs[] = Dates.format(times[frame_num], "yyyy-mm-dd HH:MM") *
                      " UTC — Local vs Non-Local PBL Diffusion"
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    isdir(AT_DIR) || error("Output directory not found: $AT_DIR")

    # Load CS coordinates from one of the NC files
    ref_file = first(sort(filter(f -> endswith(f, ".nc") && contains(f, "catrine_geosit"),
                                  readdir(AT_DIR))))
    @info "Loading CS coordinates from $ref_file"
    cs_lons, cs_lats = load_cs_coordinates(joinpath(AT_DIR, ref_file))

    @info "Building CS -> lat-lon regridding map (1 deg)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    levs = [LEV_SURFACE, LEV_750HPA]

    @info "Loading local PBL data..."
    local_data = load_cs_daily_nc(AT_DIR, LOCAL_PATTERN, rmap,
        "fossil_co2_3d", levs;
        date_start=DATE_START, date_end=DATE_END,
        exclude_pattern="nonlocal", label="Local PBL")

    @info "Loading non-local PBL data..."
    nonlocal_data = load_cs_daily_nc(AT_DIR, NONLOCAL_PATTERN, rmap,
        "fossil_co2_3d", levs;
        date_start=DATE_START, date_end=DATE_END,
        label="Non-Local PBL")

    # Use common times
    common_times = sort(collect(intersect(Set(local_data.times), Set(nonlocal_data.times))))
    @info "Common timesteps: $(length(common_times))"

    lo_idx = [findfirst(==(t), local_data.times) for t in common_times]
    nl_idx = [findfirst(==(t), nonlocal_data.times) for t in common_times]

    lo = (; times=common_times,
           fields=[local_data.fields[li][:, :, lo_idx] for li in 1:length(levs)])
    nl = (; times=common_times,
           fields=[nonlocal_data.fields[li][:, :, nl_idx] for li in 1:length(levs)])

    make_animation(lo, nl, common_times, rmap)
end

main()
