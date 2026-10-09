#!/usr/bin/env julia
# ===========================================================================
# Flat IC comparison: n_sub loop+remap vs single-step GCHP transport
#
# Layout:
#   Row 1: Surface CO2 deviation from 400 ppm (n_sub | 1-step | difference)
#   Row 2: ~500 hPa CO2 deviation from 400 ppm (n_sub | 1-step | difference)
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_gchp_flat_nsub.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates
using Statistics

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const AT_DIR     = get(ENV, "AT_DIR", "/temp1/catrine/output")
const NSUB_PAT   = get(ENV, "NSUB_PATTERN", "catrine_gchp_flat_2")       # n_sub loop+remap
const ONESTEP_PAT = get(ENV, "ONESTEP_PATTERN", "catrine_gchp_flat_current_best_2")  # single step
const OUT_GIF    = get(ENV, "OUT_GIF", "/temp1/catrine/output/gchp_flat_nsub_vs_1step.gif")
const FPS        = parse(Int, get(ENV, "FPS", "4"))
const COORD_FILE = joinpath(homedir(), "code", "gitHub", "AtmosTransportModel",
                            "data", "grids", "cs_c180_gridspec.nc")

# CS level indices (k=1 = surface in our output convention after regrid)
const LEV_SURFACE = 1
const LEV_500HPA  = 26   # ~500 hPa level in 72-level GEOS-IT

const INIT_PPM = 400f0   # initial CO2 in ppm

# ---------------------------------------------------------------------------
# Loader: read AT output files matching pattern, regrid to lat-lon
# ---------------------------------------------------------------------------
function load_at_data(at_dir, pattern, rmap, levs)
    daily_files = sort(filter(f -> endswith(f, ".nc") && startswith(f, pattern),
                               readdir(at_dir)))
    @info "Pattern '$pattern': $(length(daily_files)) files"

    all_times = DateTime[]
    all_fields = [Float32[] for _ in levs]

    buf = zeros(Float32, rmap.nlon, rmap.nlat)

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            haskey(ds, "co2_3d") || return
            at_times = ds["time"][:]
            co2 = ds["co2_3d"]
            for ti in 1:length(at_times)
                push!(all_times, at_times[ti])
                for (li, lev) in enumerate(levs)
                    data_cs = Float32.(co2[:, :, :, lev, ti]) .* 1f6  # → ppm
                    regrid_cs!(buf, data_cs, rmap)
                    append!(all_fields[li], vec(buf))
                end
            end
        end
    end

    nt = length(all_times)
    fields = [reshape(all_fields[li], rmap.nlon, rmap.nlat, nt) for li in eachindex(levs)]
    @info "  Loaded $nt timesteps"
    return (; times=all_times, fields)
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(nsub, onestep, rmap; fps=FPS)
    # Find common timesteps
    common_times = intersect(nsub.times, onestep.times)
    sort!(common_times)
    nframes = length(common_times)
    @info "Common timesteps: $nframes"

    nsub_idx = [findfirst(==(t), nsub.times) for t in common_times]
    one_idx  = [findfirst(==(t), onestep.times) for t in common_times]

    # Color ranges (deviation from 400 ppm)
    dev_max_sfc  = 0.15f0     # surface: ±0.15 ppm
    dev_max_500  = 0.10f0     # 500 hPa: ±0.1 ppm
    diff_max_sfc = 0.15f0     # difference: ±0.15 ppm
    diff_max_500 = 0.10f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(2000, 900), fontsize=13)

    # Row 1: Surface
    ax_sfc_nsub = GeoAxis(fig[1, 1]; dest="+proj=robin", title="n_sub loop — Surface CO2 Δ")
    ax_sfc_one  = GeoAxis(fig[1, 2]; dest="+proj=robin", title="Single step — Surface CO2 Δ")
    ax_sfc_diff = GeoAxis(fig[1, 3]; dest="+proj=robin", title="n_sub − 1step")

    # Row 2: 500 hPa
    ax_500_nsub = GeoAxis(fig[2, 1]; dest="+proj=robin", title="n_sub loop — ~500 hPa CO2 Δ")
    ax_500_one  = GeoAxis(fig[2, 2]; dest="+proj=robin", title="Single step — ~500 hPa CO2 Δ")
    ax_500_diff = GeoAxis(fig[2, 3]; dest="+proj=robin", title="n_sub − 1step")

    # Initial observables (deviation from 400 ppm)
    z_sfc_nsub = Observable(zeros(Float32, rmap.nlat, rmap.nlon))
    z_sfc_one  = Observable(zeros(Float32, rmap.nlat, rmap.nlon))
    z_sfc_diff = Observable(zeros(Float32, rmap.nlat, rmap.nlon))
    z_500_nsub = Observable(zeros(Float32, rmap.nlat, rmap.nlon))
    z_500_one  = Observable(zeros(Float32, rmap.nlat, rmap.nlon))
    z_500_diff = Observable(zeros(Float32, rmap.nlat, rmap.nlon))

    # Surface plots
    sf1 = surface!(ax_sfc_nsub, lon2d, lat2d, z_sfc_nsub;
        shading=NoShading, colormap=:RdBu, colorrange=(-dev_max_sfc, dev_max_sfc))
    surface!(ax_sfc_one, lon2d, lat2d, z_sfc_one;
        shading=NoShading, colormap=:RdBu, colorrange=(-dev_max_sfc, dev_max_sfc))
    sf1d = surface!(ax_sfc_diff, lon2d, lat2d, z_sfc_diff;
        shading=NoShading, colormap=:RdBu, colorrange=(-diff_max_sfc, diff_max_sfc))

    # 500 hPa plots
    sf2 = surface!(ax_500_nsub, lon2d, lat2d, z_500_nsub;
        shading=NoShading, colormap=:RdBu, colorrange=(-dev_max_500, dev_max_500))
    surface!(ax_500_one, lon2d, lat2d, z_500_one;
        shading=NoShading, colormap=:RdBu, colorrange=(-dev_max_500, dev_max_500))
    sf2d = surface!(ax_500_diff, lon2d, lat2d, z_500_diff;
        shading=NoShading, colormap=:RdBu, colorrange=(-diff_max_500, diff_max_500))

    # Coastlines
    for ax in [ax_sfc_nsub, ax_sfc_one, ax_sfc_diff, ax_500_nsub, ax_500_one, ax_500_diff]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    # Colorbars
    Colorbar(fig[1, 4], sf1; label="CO2 Δ [ppm]", width=14)
    Colorbar(fig[1, 5], sf1d; label="Diff [ppm]", width=14)
    Colorbar(fig[2, 4], sf2; label="CO2 Δ [ppm]", width=14)
    Colorbar(fig[2, 5], sf2d; label="Diff [ppm]", width=14)

    title_obs = Observable("")
    Label(fig[0, 1:5], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    Makie.record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        ni = nsub_idx[frame_num]
        oi = one_idx[frame_num]

        sfc_n = nsub.fields[1][:, :, ni] .- INIT_PPM
        sfc_o = onestep.fields[1][:, :, oi] .- INIT_PPM
        z_sfc_nsub[] = sfc_n'
        z_sfc_one[]  = sfc_o'
        z_sfc_diff[] = (sfc_n .- sfc_o)'

        h500_n = nsub.fields[2][:, :, ni] .- INIT_PPM
        h500_o = onestep.fields[2][:, :, oi] .- INIT_PPM
        z_500_nsub[] = h500_n'
        z_500_one[]  = h500_o'
        z_500_diff[] = (h500_n .- h500_o)'

        # Stats for title
        sfc_std_n = round(std(sfc_n), sigdigits=3)
        sfc_std_o = round(std(sfc_o), sigdigits=3)
        title_obs[] = Dates.format(common_times[frame_num], "yyyy-mm-dd HH:MM") *
            " UTC — Flat IC 400 ppm | sfc std: n_sub=$(sfc_std_n), 1step=$(sfc_std_o) ppm"
    end

    @info "Saved: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    # Load CS coordinates from gridspec
    cs_lons, cs_lats = NCDataset(COORD_FILE, "r") do ds
        Float64.(ds["lons"][:, :, :]), Float64.(ds["lats"][:, :, :])
    end

    @info "Building CS -> lat-lon map (1 deg)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    levs = [LEV_SURFACE, LEV_500HPA]

    @info "Loading n_sub loop+remap data..."
    nsub_data = load_at_data(AT_DIR, NSUB_PAT, rmap, levs)

    @info "Loading single-step data..."
    onestep_data = load_at_data(AT_DIR, ONESTEP_PAT, rmap, levs)

    make_animation(nsub_data, onestep_data, rmap)
end

main()
