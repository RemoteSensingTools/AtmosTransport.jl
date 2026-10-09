#!/usr/bin/env julia
# ===========================================================================
# 4-panel comparison animation: ERA5 vs GEOS-IT C180
#
# Layout:
#   Row 1: Surface CO₂          (ERA5 | GEOS-IT)
#   Row 2: CO₂ at ~800 hPa      (ERA5 | GEOS-IT)
#
# Each row has its own colorbar with a fixed range based on the 90th
# percentile of a late timestep. Timesteps are matched by time value.
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_4panel_comparison.jl
# ===========================================================================

using NCDatasets
using CairoMakie
using GeoMakie
using Dates
using Printf

const ERA5_FILE   = get(ENV, "ERA5_OUTPUT",
    joinpath(homedir(), "data", "output", "era5_june2023.nc"))
const GEOSFP_FILE = get(ENV, "GEOSFP_OUTPUT",
    joinpath(homedir(), "data", "output", "geosit_c180_june2023_05deg.nc"))
const OUT_GIF     = get(ENV, "OUT_GIF", "comparison_4panel_era5_geosit.gif")
const FPS         = parse(Int, get(ENV, "FPS", "4"))

function load_output(filepath::String, varnames::Vector{String})
    ds = NCDataset(filepath, "r")
    lon_raw = Float64.(ds["lon"][:])
    lat_raw = Float64.(ds["lat"][:])

    # Read time — handle both DateTime and numeric
    time_raw = ds["time"][:]
    t0 = time_raw[1]
    time_days = if eltype(time_raw) <: DateTime
        Float64.(Dates.value.(time_raw .- Ref(t0))) ./ 86_400_000.0
    else
        Float64.(time_raw .- t0) ./ 86400.0
    end

    fields = Dict{String, Array{Float32,3}}()
    for v in varnames
        if haskey(ds, v)
            fields[v] = Float32.(ds[v][:, :, :])
        end
    end
    close(ds)

    # Shift lon 0..360 → -180..180 and sort ascending
    lon_shifted = [l > 180.0 ? l - 360.0 : l for l in lon_raw]
    lon_order   = sortperm(lon_shifted)
    lon = lon_shifted[lon_order]

    # Ensure lat is ascending (S→N)
    lat_order = lat_raw[1] > lat_raw[end] ?
        collect(length(lat_raw):-1:1) : collect(1:length(lat_raw))
    lat = lat_raw[lat_order]

    # Reorder data to match sorted coordinates
    for (k, v) in fields
        fields[k] = v[lon_order, lat_order, :]
    end

    return (; lon, lat, time_days, fields)
end

"""Find nearest index in `times` for each value in `target_times`."""
function match_times(target_times, times)
    [argmin(abs.(times .- t)) for t in target_times]
end

"""Compute the p-th percentile of non-NaN values."""
function percentile(data, p)
    valid = sort(filter(isfinite, vec(Float64.(data))))
    isempty(valid) && return 0.0
    idx = clamp(round(Int, p / 100.0 * length(valid)), 1, length(valid))
    return valid[idx]
end

function make_animation(era5, geosfp; fps=FPS)
    # Use ERA5 timesteps as reference (fewer = 6-hourly)
    era5_nt = length(era5.time_days)
    geos_match = match_times(era5.time_days, geosfp.time_days)

    # Frame selection: cap at ~120 frames
    frame_step = max(1, era5_nt ÷ 120)
    idx = 1:frame_step:era5_nt
    nframes = length(idx)
    @info "Animating $nframes frames (of $era5_nt total) at $fps fps"

    # --- Colorranges from 90th percentile of a late timestep ---
    ti_late = max(1, era5_nt - 2)  # near-final ERA5 timestep
    gi_late = geos_match[ti_late]

    # Surface: take max p90 across both models
    sfc_p90 = max(
        percentile(era5.fields["co2_surface"][:, :, ti_late], 90),
        percentile(geosfp.fields["co2_surface"][:, :, gi_late], 90))
    sfc_max = Float32(ceil(sfc_p90))
    sfc_max = max(sfc_max, 1f0)

    # 800 hPa: take max p90 across both models
    hpa_p90 = max(
        percentile(era5.fields["co2_800hPa"][:, :, ti_late], 90),
        percentile(geosfp.fields["co2_800hPa"][:, :, gi_late], 90))
    hpa_max = Float32(ceil(hpa_p90 * 10) / 10)  # round up to 0.1
    hpa_max = max(hpa_max, 0.5f0)

    @info "Colorranges: surface 0–$(sfc_max) ppm, 800hPa 0–$(hpa_max) ppm"

    # 2D coordinate meshes
    function make_mesh(lon, lat)
        nlon, nlat = length(lon), length(lat)
        lon_2d = Float32[lon[i] for _j in 1:nlat, i in 1:nlon]
        lat_2d = Float32[lat[j] for  j in 1:nlat, _i in 1:nlon]
        return lon_2d, lat_2d
    end

    era5_lon2d, era5_lat2d = make_mesh(era5.lon, era5.lat)
    geos_lon2d, geos_lat2d = make_mesh(geosfp.lon, geosfp.lat)

    # --- Figure: 2 rows × 2 columns + colorbars ---
    fig = Figure(size=(1600, 900), fontsize=12)

    # Row 1: Surface
    ax_sfc_e = GeoAxis(fig[1, 1]; dest="+proj=robin", title="ERA5 — Surface CO₂")
    ax_sfc_g = GeoAxis(fig[1, 2]; dest="+proj=robin", title="GEOS-IT C180 — Surface CO₂")

    # Row 2: 800 hPa
    ax_hpa_e = GeoAxis(fig[2, 1]; dest="+proj=robin", title="ERA5 — CO₂ at ~800 hPa")
    ax_hpa_g = GeoAxis(fig[2, 2]; dest="+proj=robin", title="GEOS-IT C180 — CO₂ at ~800 hPa")

    # Initial slices
    ti0_e = idx[1]
    ti0_g = geos_match[ti0_e]

    z_sfc_e = Observable(era5.fields["co2_surface"][:, :, ti0_e]')
    z_sfc_g = Observable(geosfp.fields["co2_surface"][:, :, ti0_g]')
    z_hpa_e = Observable(era5.fields["co2_800hPa"][:, :, ti0_e]')
    z_hpa_g = Observable(geosfp.fields["co2_800hPa"][:, :, ti0_g]')

    # Surface plots
    sf_sfc_e = surface!(ax_sfc_e, era5_lon2d, era5_lat2d, z_sfc_e;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=sqrt)
    sf_sfc_g = surface!(ax_sfc_g, geos_lon2d, geos_lat2d, z_sfc_g;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=sqrt)

    # 800 hPa plots
    sf_hpa_e = surface!(ax_hpa_e, era5_lon2d, era5_lat2d, z_hpa_e;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=sqrt)
    sf_hpa_g = surface!(ax_hpa_g, geos_lon2d, geos_lat2d, z_hpa_g;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=sqrt)

    # Coastlines
    for ax in [ax_sfc_e, ax_sfc_g, ax_hpa_e, ax_hpa_g]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    # Colorbars (one per row, in column 3)
    Colorbar(fig[1, 3], sf_sfc_g;
        label="Surface CO₂ [ppm]", width=16,
        ticks=range(0, sfc_max, length=6) .|> (x -> round(x, digits=1)))
    Colorbar(fig[2, 3], sf_hpa_g;
        label="CO₂ at ~800 hPa [ppm]", width=16,
        ticks=range(0, hpa_max, length=6) .|> (x -> round(x, digits=2)))

    # Supertitle
    title_obs = Observable("CO₂ Enhancement — Day 0.0")
    Label(fig[0, 1:3], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        ti_e = idx[frame_num]
        ti_g = geos_match[ti_e]

        z_sfc_e[] = era5.fields["co2_surface"][:, :, ti_e]'
        z_sfc_g[] = geosfp.fields["co2_surface"][:, :, ti_g]'
        z_hpa_e[] = era5.fields["co2_800hPa"][:, :, ti_e]'
        z_hpa_g[] = geosfp.fields["co2_800hPa"][:, :, ti_g]'

        days = era5.time_days[ti_e]
        title_obs[] = @sprintf("CO₂ Enhancement — Day %.1f", days)
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

function main()
    isfile(ERA5_FILE)   || error("ERA5 output not found: $ERA5_FILE")
    isfile(GEOSFP_FILE) || error("GEOS-IT output not found: $GEOSFP_FILE")

    varnames = ["co2_surface", "co2_800hPa"]

    @info "Loading ERA5 output: $ERA5_FILE"
    era5 = load_output(ERA5_FILE, varnames)

    @info "Loading GEOS-IT output: $GEOSFP_FILE"
    geosfp = load_output(GEOSFP_FILE, varnames)

    make_animation(era5, geosfp)
end

main()
