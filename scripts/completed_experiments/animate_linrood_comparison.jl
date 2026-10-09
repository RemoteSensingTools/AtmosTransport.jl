#!/usr/bin/env julia
# ===========================================================================
# 2-way comparison animation: Strang vs Lin-Rood (full physics)
#
# Layout (2 rows × 2 columns):
#   Row 1: Surface fossil CO2    (Strang | Lin-Rood)
#   Row 2: fossil CO2 at ~750hPa (Strang | Lin-Rood)
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_linrood_comparison.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const STRANG_DIR  = get(ENV, "STRANG_DIR", "/temp2/catrine-runs/output")
const LINROOD_DIR = get(ENV, "LINROOD_DIR", "/temp1/catrine/output")
const OUT_GIF     = get(ENV, "OUT_GIF", "/temp1/catrine/output/linrood_comparison_21d.gif")
const FPS         = parse(Int, get(ENV, "FPS", "4"))

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1)
const DATE_END   = DateTime(2021, 12, 22, 21, 0, 0)

# File patterns for each version
const STRANG_PATTERN  = "catrine_geosit_c180_2021"
const LINROOD_PATTERN = "catrine_linrood_fullphys_21d_2021"

# ---------------------------------------------------------------------------
# Generic loader for AtmosTransport CS binary → NC output
# ---------------------------------------------------------------------------
function load_at_data(at_dir, pattern, rmap, var_name, levs;
                      date_start, date_end, label="",
                      exclude_pattern::String="")
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, pattern), readdir(at_dir)))
    if !isempty(exclude_pattern)
        daily_files = filter(f -> !contains(f, exclude_pattern), daily_files)
    end

    nl   = length(levs)
    nlon = rmap.nlon
    nlat = rmap.nlat
    buf  = zeros(Float32, nlon, nlat)

    all_times  = DateTime[]
    all_fields = [Vector{Matrix{Float32}}() for _ in 1:nl]

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            file_times = ds["time"][:]
            data_var = ds[var_name]
            for (ti, dt) in enumerate(file_times)
                date_start <= dt <= date_end || continue
                push!(all_times, dt)
                for (li, lev) in enumerate(levs)
                    data_cs = Float32.(data_var[:, :, :, lev, ti]) .* 1f6  # mol/mol → ppm
                    regrid_cs!(buf, data_cs, rmap)
                    push!(all_fields[li], copy(buf))
                end
            end
        end
    end

    perm = sortperm(all_times)
    all_times = all_times[perm]
    for li in 1:nl
        all_fields[li] = all_fields[li][perm]
    end

    nt = length(all_times)
    lbl = isempty(label) ? pattern : label
    @info "$lbl: $nt snapshots ($(all_times[1]) → $(all_times[end]))"

    fields = [zeros(Float32, nlon, nlat, nt) for _ in 1:nl]
    for li in 1:nl, ti in 1:nt
        fields[li][:, :, ti] .= all_fields[li][ti]
    end

    return (; times=all_times, fields)
end

# ---------------------------------------------------------------------------
# Find common timesteps across datasets
# ---------------------------------------------------------------------------
function intersect_times(datasets...)
    common = Set(datasets[1].times)
    for ds in datasets[2:end]
        intersect!(common, Set(ds.times))
    end
    sorted = sort(collect(common))

    results = map(datasets) do ds
        idx = [findfirst(==(t), ds.times) for t in sorted]
        fields = [ds.fields[l][:, :, idx] for l in eachindex(ds.fields)]
        (; times=sorted, fields)
    end

    return sorted, results...
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(strang, linrood, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    sfc_max = 20f0
    hpa_max = 5f0
    diff_max = 2f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(1600, 1000), fontsize=12)

    # Row 1: Surface — Strang vs Lin-Rood
    ax11 = GeoAxis(fig[1, 1]; dest="+proj=robin", title="Strang — Surface fossil CO₂")
    ax12 = GeoAxis(fig[1, 2]; dest="+proj=robin", title="Lin-Rood — Surface fossil CO₂")
    ax13 = GeoAxis(fig[1, 3]; dest="+proj=robin", title="Difference (LR − Strang) — Surface")

    # Row 2: 750 hPa — Strang vs Lin-Rood
    ax21 = GeoAxis(fig[2, 1]; dest="+proj=robin", title="Strang — fossil CO₂ ~750 hPa")
    ax22 = GeoAxis(fig[2, 2]; dest="+proj=robin", title="Lin-Rood — fossil CO₂ ~750 hPa")
    ax23 = GeoAxis(fig[2, 3]; dest="+proj=robin", title="Difference (LR − Strang) — ~750 hPa")

    # Observables
    z_s_strang  = Observable(strang.fields[1][:, :, 1]')
    z_s_linrood = Observable(linrood.fields[1][:, :, 1]')
    z_s_diff    = Observable((linrood.fields[1][:, :, 1] .- strang.fields[1][:, :, 1])')
    z_h_strang  = Observable(strang.fields[2][:, :, 1]')
    z_h_linrood = Observable(linrood.fields[2][:, :, 1]')
    z_h_diff    = Observable((linrood.fields[2][:, :, 1] .- strang.fields[2][:, :, 1])')

    # Surface plots (row 1)
    sf1 = surface!(ax11, lon2d, lat2d, z_s_strang;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)
    surface!(ax12, lon2d, lat2d, z_s_linrood;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)
    sf_diff1 = surface!(ax13, lon2d, lat2d, z_s_diff;
        shading=NoShading, colormap=:RdBu, colorrange=(-diff_max, diff_max))

    # 750 hPa plots (row 2)
    sf2 = surface!(ax21, lon2d, lat2d, z_h_strang;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)
    surface!(ax22, lon2d, lat2d, z_h_linrood;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)
    sf_diff2 = surface!(ax23, lon2d, lat2d, z_h_diff;
        shading=NoShading, colormap=:RdBu, colorrange=(-diff_max, diff_max))

    # Coastlines
    for ax in [ax11, ax12, ax13, ax21, ax22, ax23]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    # Colorbars
    Colorbar(fig[1, 4], sf1;
        label="Surface fossil CO₂ [ppm]", width=16,
        ticks=range(0, sfc_max, length=6) .|> (x -> round(x, digits=1)))
    Colorbar(fig[2, 4], sf2;
        label="fossil CO₂ ~750 hPa [ppm]", width=16,
        ticks=range(0, hpa_max, length=6) .|> (x -> round(x, digits=2)))

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
                           " UTC — Strang vs Lin-Rood (full physics)")
    Label(fig[0, 1:4], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        z_s_strang[]  = strang.fields[1][:, :, frame_num]'
        z_s_linrood[] = linrood.fields[1][:, :, frame_num]'
        z_s_diff[]    = (linrood.fields[1][:, :, frame_num] .- strang.fields[1][:, :, frame_num])'
        z_h_strang[]  = strang.fields[2][:, :, frame_num]'
        z_h_linrood[] = linrood.fields[2][:, :, frame_num]'
        z_h_diff[]    = (linrood.fields[2][:, :, frame_num] .- strang.fields[2][:, :, frame_num])'

        title_obs[] = Dates.format(times[frame_num], "yyyy-mm-dd HH:MM") *
                      " UTC — Strang vs Lin-Rood (full physics)"
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    # Load CS coordinates from first Strang output file
    first_nc = joinpath(STRANG_DIR, sort(filter(f -> endswith(f, ".nc") && contains(f, STRANG_PATTERN) &&
                        !contains(f, "linrood"), readdir(STRANG_DIR)))[1])
    @info "Loading CS coordinates from $first_nc"
    cs_lons, cs_lats = NCDataset(first_nc, "r") do ds
        Float64.(ds["lons"][:, :, :]), Float64.(ds["lats"][:, :, :])
    end

    @info "Building CS → lat-lon regridding map (1°)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    levs = [LEV_SURFACE, LEV_750HPA]
    var = "fossil_co2_3d"

    @info "Loading Strang data from $STRANG_DIR..."
    strang = load_at_data(STRANG_DIR, STRANG_PATTERN, rmap, var, levs;
                          date_start=DATE_START, date_end=DATE_END, label="Strang",
                          exclude_pattern="linrood")

    @info "Loading Lin-Rood data from $LINROOD_DIR..."
    linrood = load_at_data(LINROOD_DIR, LINROOD_PATTERN, rmap, var, levs;
                           date_start=DATE_START, date_end=DATE_END, label="Lin-Rood")

    # Align to common timesteps
    times, strang, linrood = intersect_times(strang, linrood)
    @info "Common timesteps: $(length(times))"

    make_animation(strang, linrood, times, rmap)
end

main()
