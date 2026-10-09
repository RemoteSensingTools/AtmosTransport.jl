#!/usr/bin/env julia
# ===========================================================================
# Parameterized comparison animation: Lin-Rood variant vs GEOS-Chem (CO2)
#
# Layout (3 rows × 2 columns):
#   Row 1: Surface CO2          (GEOS-Chem | AtmosTransport variant)
#   Row 2: CO2 at ~750hPa       (GEOS-Chem | AtmosTransport variant)
#   Row 3: Column water (GC)    | Surface CO2 difference (AT − GC)
#
# Environment variables:
#   AT_PATTERN  — file pattern to match AT output (e.g. "linrood_advonly")
#   AT_LABEL    — display label (e.g. "LR advonly")
#   OUT_GIF     — output filename
#   AT_DIR      — AT output directory
#   GC_DIR      — GEOS-Chem directory
#
# Usage:
#   AT_PATTERN="linrood_advonly" AT_LABEL="LR advonly" OUT_GIF="lr_advonly_vs_gc.gif" \
#     julia --project=. scripts/completed_experiments/animate_linrood_vs_geoschem.jl
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
const AT_DIR     = get(ENV, "AT_DIR", "/temp2/catrine-runs/output")
const AT_PATTERN = get(ENV, "AT_PATTERN", "linrood_advonly")
const AT_LABEL   = get(ENV, "AT_LABEL", AT_PATTERN)
const OUT_GIF    = get(ENV, "OUT_GIF", "$(AT_PATTERN)_vs_geoschem_co2.gif")
const FPS        = parse(Int, get(ENV, "FPS", "4"))

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1)
const DATE_END   = DateTime(2021, 12, 21, 9, 0, 0)  # GC data ends Dec 21

const G_ACCEL = 9.80665f0  # m/s² — convert Pa → kg/m²

# ---------------------------------------------------------------------------
# Load column water from GC met fields (PSC2WET - PSC2DRY) / g
# ---------------------------------------------------------------------------
function load_gc_column_water(gc_dir, rmap; date_start, date_end)
    all_files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"),
                             readdir(gc_dir)))
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

    nt  = length(files)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    tcwv = zeros(Float32, rmap.nlon, rmap.nlat, nt)

    for (ti, fname) in enumerate(files)
        NCDataset(joinpath(gc_dir, fname), "r") do ds
            ps_wet = Float64.(ds["Met_PSC2WET"][:, :, :, 1])  # hPa
            ps_dry = Float64.(ds["Met_PSC2DRY"][:, :, :, 1])  # hPa
            cw = Float32.((ps_wet .- ps_dry) .* (100.0 / G_ACCEL))  # hPa→Pa→kg/m²
            regrid_cs!(buf, cw, rmap)
            tcwv[:, :, ti] .= buf
        end
    end

    @info "GC column water: $nt snapshots"
    return (; times, fields=[tcwv])
end

# ---------------------------------------------------------------------------
# Find common timesteps
# ---------------------------------------------------------------------------
function align_times(datasets...)
    common = Set(datasets[1].times)
    for ds in datasets[2:end]
        intersect!(common, Set(ds.times))
    end
    sorted = sort(collect(common))

    return sorted, map(datasets) do ds
        idx = [findfirst(==(t), ds.times) for t in sorted]
        (; times=sorted, fields=[ds.fields[l][:, :, idx] for l in eachindex(ds.fields)])
    end...
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(gc, at, cw, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    # CO2 color ranges — wide enough to show surface buildup from emissions
    sfc_range = (380f0, 550f0)
    hpa_range = (390f0, 450f0)
    diff_max  = 3f0
    cw_range  = (0f0, 60f0)  # kg/m² total column water vapor

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(1200, 1000), fontsize=11)

    # Row 1: Surface CO2
    ax_sfc_gc = GeoAxis(fig[1, 1]; dest="+proj=robin",
        title="GEOS-Chem — Surface CO2")
    ax_sfc_at = GeoAxis(fig[1, 2]; dest="+proj=robin",
        title="$AT_LABEL — Surface CO2")

    # Row 2: 750 hPa CO2
    ax_hpa_gc = GeoAxis(fig[2, 1]; dest="+proj=robin",
        title="GEOS-Chem — CO2 at ~750 hPa")
    ax_hpa_at = GeoAxis(fig[2, 2]; dest="+proj=robin",
        title="$AT_LABEL — CO2 at ~750 hPa")

    # Row 3: Column water + CO2 difference
    ax_cw   = GeoAxis(fig[3, 1]; dest="+proj=robin",
        title="Total Column Water Vapor (GEOS-IT)")
    ax_diff = GeoAxis(fig[3, 2]; dest="+proj=robin",
        title="Surface CO2 Difference ($AT_LABEL − GC)")

    # Observables
    z_sfc_gc = Observable(gc.fields[1][:, :, 1]')
    z_sfc_at = Observable(at.fields[1][:, :, 1]')
    z_hpa_gc = Observable(gc.fields[2][:, :, 1]')
    z_hpa_at = Observable(at.fields[2][:, :, 1]')
    z_cw     = Observable(cw.fields[1][:, :, 1]')
    z_diff   = Observable((at.fields[1][:, :, 1] .- gc.fields[1][:, :, 1])')

    # Surface CO2
    sf1 = surface!(ax_sfc_gc, lon2d, lat2d, z_sfc_gc;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=sfc_range)
    surface!(ax_sfc_at, lon2d, lat2d, z_sfc_at;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=sfc_range)

    # 750 hPa CO2
    sf2 = surface!(ax_hpa_gc, lon2d, lat2d, z_hpa_gc;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=hpa_range)
    surface!(ax_hpa_at, lon2d, lat2d, z_hpa_at;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=hpa_range)

    # Column water
    sf_cw = surface!(ax_cw, lon2d, lat2d, z_cw;
        shading=NoShading, colormap=:Blues, colorrange=cw_range)

    # CO2 difference
    sf_diff = surface!(ax_diff, lon2d, lat2d, z_diff;
        shading=NoShading, colormap=:RdBu, colorrange=(-diff_max, diff_max))

    for ax in [ax_sfc_gc, ax_sfc_at, ax_hpa_gc, ax_hpa_at, ax_cw, ax_diff]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    # Colorbars
    Colorbar(fig[1, 3], sf1;
        label="Surface CO2 [ppm]", width=16,
        ticks=range(sfc_range..., length=6) .|> (x -> round(x, digits=1)))
    Colorbar(fig[2, 3], sf2;
        label="CO2 at ~750 hPa [ppm]", width=16,
        ticks=range(hpa_range..., length=6) .|> (x -> round(x, digits=1)))
    Colorbar(fig[3, 3], sf_cw;
        label="Column water [kg/m²]", width=16,
        ticks=range(cw_range..., length=5) .|> (x -> round(x, digits=0)))

    # Small colorbar for difference (inside row 3 right panel)
    Colorbar(fig[3, 4], sf_diff;
        label="ΔCO2 [ppm]", width=12,
        ticks=range(-diff_max, diff_max, length=5) .|> (x -> round(x, digits=1)))

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
                           " UTC — $AT_LABEL vs GEOS-Chem (CO2)")
    Label(fig[0, 1:3], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        z_sfc_gc[] = gc.fields[1][:, :, frame_num]'
        z_sfc_at[] = at.fields[1][:, :, frame_num]'
        z_hpa_gc[] = gc.fields[2][:, :, frame_num]'
        z_hpa_at[] = at.fields[2][:, :, frame_num]'
        z_cw[]     = cw.fields[1][:, :, frame_num]'
        z_diff[]   = (at.fields[1][:, :, frame_num] .- gc.fields[1][:, :, frame_num])'

        title_obs[] = Dates.format(times[frame_num], "yyyy-mm-dd HH:MM") *
                      " UTC — $AT_LABEL vs GEOS-Chem (CO2)"
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    isdir(GC_DIR) || error("GEOS-Chem directory not found: $GC_DIR")
    isdir(AT_DIR) || error("AtmosTransport output not found: $AT_DIR")

    # Load CS coordinates from first GC file
    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    @info "Loading CS coordinates from $(gc_files[1])"
    cs_lons, cs_lats = load_cs_coordinates(joinpath(GC_DIR, gc_files[1]))

    @info "Building CS → lat-lon regridding map (1°)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    levs = [LEV_SURFACE, LEV_750HPA]

    @info "Loading GEOS-Chem CO2 data..."
    gc = load_geoschem_nc(GC_DIR, rmap,
        "SpeciesConcVV_CO2", levs;
        date_start=DATE_START, date_end=DATE_END,
        scale=1e6)

    @info "Loading GEOS-Chem column water..."
    cw = load_gc_column_water(GC_DIR, rmap;
        date_start=DATE_START, date_end=DATE_END)

    @info "Loading AtmosTransport ($AT_LABEL) CO2 data..."
    at = load_cs_daily_nc(AT_DIR, AT_PATTERN, rmap,
        "co2_3d", levs;
        date_start=DATE_START, date_end=DATE_END,
        scale=1e6, label=AT_LABEL)

    # Align to common timesteps
    times, gc, cw, at = align_times(gc, cw, at)
    @info "Common timesteps: $(length(times))"

    make_animation(gc, at, cw, times, rmap)
end

main()
