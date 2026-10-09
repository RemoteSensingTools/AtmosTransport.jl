#!/usr/bin/env julia
# ===========================================================================
# Two-panel comparison animation: ERA5 | GEOS-IT C180
#
# Reads output NetCDF files from both runs and produces a side-by-side
# column-mean CO2 animation with a shared colorbar and Robinson projection.
#
# Timesteps are matched by time value (nearest-neighbor) so the two runs
# don't need the same output frequency.
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_comparison.jl
#
# Environment:
#   ERA5_OUTPUT     — path to ERA5 output NetCDF
#   GEOSFP_OUTPUT   — path to GEOS-IT CS output NetCDF (regridded to latlon)
#   OUT_GIF         — output GIF path (default: comparison_era5_geosit.gif)
#   FPS             — frames per second (default: 4)
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
const OUT_GIF     = get(ENV, "OUT_GIF", "comparison_era5_geosit.gif")
const FPS         = parse(Int, get(ENV, "FPS", "4"))

function load_output(filepath::String)
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

    # Try co2_column_mean first, fall back to co2
    varname = haskey(ds, "co2_column_mean") ? "co2_column_mean" : "co2"
    col = Float32.(ds[varname][:, :, :])
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
    col = col[lon_order, lat_order, :]

    return (; lon, lat, time_days, col)
end

"""Find nearest index in `times` for each value in `target_times`."""
function match_times(target_times, times)
    [argmin(abs.(times .- t)) for t in target_times]
end

function make_animation(era5, geosfp; fps=FPS, vmin=0f0, vmax=7f0)
    # Use ERA5 timesteps as the reference (fewer steps = 6-hourly)
    # and find matching GEOS-IT frames by time
    era5_nt   = size(era5.col, 3)
    geosfp_nt = size(geosfp.col, 3)

    # Match GEOS-IT to ERA5 times (ERA5 is sparser)
    geos_match = match_times(era5.time_days, geosfp.time_days)

    @info "Matching timesteps: $(era5_nt) ERA5 frames → $(era5_nt) paired frames " *
          "(GEOS-IT has $(geosfp_nt) total)"

    # Frame selection: cap at ~120 frames for manageable GIF size
    frame_step = max(1, era5_nt ÷ 120)
    idx = 1:frame_step:era5_nt
    nframes = length(idx)
    @info "Animating $nframes frames (of $era5_nt total, step=$frame_step) at $fps fps"

    # 2D coordinate meshes for surface! (GeoMakie expects (nlat, nlon))
    function make_mesh(lon, lat)
        nlon, nlat = length(lon), length(lat)
        lon_2d = Float32[lon[i] for _j in 1:nlat, i in 1:nlon]
        lat_2d = Float32[lat[j] for  j in 1:nlat, _i in 1:nlon]
        return lon_2d, lat_2d
    end

    era5_lon2d, era5_lat2d = make_mesh(era5.lon, era5.lat)
    geos_lon2d, geos_lat2d = make_mesh(geosfp.lon, geosfp.lat)

    # Figure
    fig = Figure(size=(1600, 500), fontsize=13)

    ax1 = GeoAxis(fig[1, 1];
        dest  = "+proj=robin",
        title = "ERA5 (~1° lat-lon)")
    ax2 = GeoAxis(fig[1, 2];
        dest  = "+proj=robin",
        title = "GEOS-IT C180 (~0.5° cubed-sphere)")

    # Observables for efficient animation updates
    z1_obs = Observable(era5.col[:, :, idx[1]]')
    z2_obs = Observable(geosfp.col[:, :, geos_match[idx[1]]]')

    sf1 = surface!(ax1, era5_lon2d, era5_lat2d, z1_obs;
        shading=NoShading, colormap=:YlOrRd,
        colorrange=(vmin, vmax), colorscale=sqrt)
    sf2 = surface!(ax2, geos_lon2d, geos_lat2d, z2_obs;
        shading=NoShading, colormap=:YlOrRd,
        colorrange=(vmin, vmax), colorscale=sqrt)

    # Coastlines
    lines!(ax1, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    lines!(ax2, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)

    # Shared colorbar
    Colorbar(fig[1, 3], sf2;
        label = "Column-mean CO₂ enhancement [ppm]", width = 18,
        ticks = range(vmin, vmax, length=8) .|> (x -> round(x, digits=1)))

    # Supertitle
    title_obs = Observable("Column-mean CO₂ — Day 0.0")
    Label(fig[0, 1:3], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        ti_era5  = idx[frame_num]
        ti_geos  = geos_match[ti_era5]
        z1_obs[] = era5.col[:, :, ti_era5]'
        z2_obs[] = geosfp.col[:, :, ti_geos]'

        days = era5.time_days[ti_era5]
        title_obs[] = @sprintf("Column-mean CO₂ — Day %.1f", days)
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

function main()
    isfile(ERA5_FILE) || error("ERA5 output not found: $ERA5_FILE")
    isfile(GEOSFP_FILE) || error("GEOS-IT output not found: $GEOSFP_FILE")

    @info "Loading ERA5 output: $ERA5_FILE"
    era5 = load_output(ERA5_FILE)

    @info "Loading GEOS-IT output: $GEOSFP_FILE"
    geosfp = load_output(GEOSFP_FILE)

    make_animation(era5, geosfp)
end

main()
