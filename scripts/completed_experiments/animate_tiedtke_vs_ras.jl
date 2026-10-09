#!/usr/bin/env julia
# ===========================================================================
# 4-panel comparison animation: Tiedtke vs RAS convection (CATRINE D7.1)
#
# Layout:
#   Row 1: Surface fossil CO2 VMR    (Tiedtke | RAS)
#   Row 2: fossil CO2 VMR at ~750hPa (Tiedtke | RAS)
#
# Both datasets are GEOS-IT C180 cubed-sphere (native CS NetCDF output).
# Data is regridded to 1 deg lat-lon for clean global map visualization.
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_tiedtke_vs_ras.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const AT_DIR   = get(ENV, "AT_DIR", "/temp2/catrine-runs/output")
const OUT_GIF  = get(ENV, "OUT_GIF", "catrine_tiedtke_vs_ras.gif")
const FPS      = parse(Int, get(ENV, "FPS", "4"))

const TIEDTKE_PATTERN = "catrine_geosit_c180_2021"
const RAS_PATTERN     = "catrine_geosit_c180_ras_2021"

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1)
const DATE_END   = DateTime(2021, 12, 7, 21, 0, 0)

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(tiedtke, ras, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    sfc_max = 20f0
    hpa_max = 5f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(1600, 900), fontsize=12)

    ax_sfc_td = GeoAxis(fig[1, 1]; dest="+proj=robin",
        title="Tiedtke — Surface fossil CO2")
    ax_sfc_rs = GeoAxis(fig[1, 2]; dest="+proj=robin",
        title="RAS — Surface fossil CO2")
    ax_hpa_td = GeoAxis(fig[2, 1]; dest="+proj=robin",
        title="Tiedtke — fossil CO2 at ~750 hPa")
    ax_hpa_rs = GeoAxis(fig[2, 2]; dest="+proj=robin",
        title="RAS — fossil CO2 at ~750 hPa")

    z_sfc_td = Observable(tiedtke.fields[1][:, :, 1]')
    z_sfc_rs = Observable(ras.fields[1][:, :, 1]')
    z_hpa_td = Observable(tiedtke.fields[2][:, :, 1]')
    z_hpa_rs = Observable(ras.fields[2][:, :, 1]')

    sf1 = surface!(ax_sfc_td, lon2d, lat2d, z_sfc_td;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)
    surface!(ax_sfc_rs, lon2d, lat2d, z_sfc_rs;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)

    sf2 = surface!(ax_hpa_td, lon2d, lat2d, z_hpa_td;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)
    surface!(ax_hpa_rs, lon2d, lat2d, z_hpa_rs;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)

    for ax in [ax_sfc_td, ax_sfc_rs, ax_hpa_td, ax_hpa_rs]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 3], sf1;
        label="Surface fossil CO2 [ppm]", width=16,
        ticks=range(0, sfc_max, length=6) .|> (x -> round(x, digits=1)))
    Colorbar(fig[2, 3], sf2;
        label="fossil CO2 at ~750 hPa [ppm]", width=16,
        ticks=range(0, hpa_max, length=6) .|> (x -> round(x, digits=2)))

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
                           " UTC — Tiedtke vs RAS Convection")
    Label(fig[0, 1:3], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        z_sfc_td[] = tiedtke.fields[1][:, :, frame_num]'
        z_sfc_rs[] = ras.fields[1][:, :, frame_num]'
        z_hpa_td[] = tiedtke.fields[2][:, :, frame_num]'
        z_hpa_rs[] = ras.fields[2][:, :, frame_num]'

        title_obs[] = Dates.format(times[frame_num], "yyyy-mm-dd HH:MM") *
                      " UTC — Tiedtke vs RAS Convection"
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    isdir(AT_DIR) || error("Output directory not found: $AT_DIR")

    ref_file = first(sort(filter(f -> endswith(f, ".nc") && contains(f, "catrine_geosit"),
                                  readdir(AT_DIR))))
    @info "Loading CS coordinates from $ref_file"
    cs_lons, cs_lats = load_cs_coordinates(joinpath(AT_DIR, ref_file))

    @info "Building CS -> lat-lon regridding map (1 deg)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    levs = [LEV_SURFACE, LEV_750HPA]

    @info "Loading Tiedtke data..."
    tiedtke = load_cs_daily_nc(AT_DIR, TIEDTKE_PATTERN, rmap,
        "fossil_co2_3d", levs;
        date_start=DATE_START, date_end=DATE_END,
        exclude_pattern="ras", label="Tiedtke")

    @info "Loading RAS data..."
    ras = load_cs_daily_nc(AT_DIR, RAS_PATTERN, rmap,
        "fossil_co2_3d", levs;
        date_start=DATE_START, date_end=DATE_END,
        label="RAS")

    common_times = sort(collect(intersect(Set(tiedtke.times), Set(ras.times))))
    @info "Common timesteps: $(length(common_times))"

    td_idx = [findfirst(==(t), tiedtke.times) for t in common_times]
    rs_idx = [findfirst(==(t), ras.times) for t in common_times]

    td = (; times=common_times,
           fields=[tiedtke.fields[li][:, :, td_idx] for li in 1:length(levs)])
    rs = (; times=common_times,
           fields=[ras.fields[li][:, :, rs_idx] for li in 1:length(levs)])

    make_animation(td, rs, common_times, rmap)
end

main()
