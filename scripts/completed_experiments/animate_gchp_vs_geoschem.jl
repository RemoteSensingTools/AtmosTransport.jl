#!/usr/bin/env julia
# ===========================================================================
# 6-panel comparison animation: GEOS-Chem vs AtmosTransport GCHP path
# Total CO2 — GCHP 1:1 port validation
#
# Layout:
#   Row 1: Surface CO2   (GEOS-Chem | AtmosTransport GCHP | Difference AT-GC)
#   Row 2: ~750 hPa CO2  (GEOS-Chem | AtmosTransport GCHP | Difference AT-GC)
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_gchp_vs_geoschem.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates
using Statistics

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const GC_DIR = get(ENV, "GC_DIR",
    joinpath(homedir(), "data", "AtmosTransport", "catrine-geoschem-runs"))
const AT_DIR   = get(ENV, "AT_DIR", "/temp1/catrine/output")
const AT_PATTERN = get(ENV, "AT_PATTERN", "catrine_gchp_8d")
const OUT_GIF  = get(ENV, "OUT_GIF", "/temp1/catrine/output/gchp_vs_geoschem_co2_8d.gif")
const FPS      = parse(Int, get(ENV, "FPS", "4"))

# CS level indices (k=1 = surface, k=72 = TOA for GEOS-IT)
const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1, 3)
const DATE_END   = DateTime(2021, 12, 8, 21)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
function load_atmostr(at_dir, rmap, target_times, levs)
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, AT_PATTERN),
                               readdir(at_dir)))
    nt  = length(target_times)
    nl  = length(levs)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    fields = [zeros(Float32, rmap.nlon, rmap.nlat, nt) for _ in 1:nl]

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            haskey(ds, "co2_3d") || return
            at_times = ds["time"][:]
            co2 = ds["co2_3d"]
            for (ti, tgt) in enumerate(target_times)
                diffs = [abs(Dates.value(at_t - tgt)) for at_t in at_times]
                best_idx = argmin(diffs)
                diffs[best_idx] / 60_000 > 90 && continue
                for (li, lev) in enumerate(levs)
                    data_cs = Float32.(co2[:, :, :, lev, best_idx]) .* 1f6
                    regrid_cs!(buf, data_cs, rmap)
                    fields[li][:, :, ti] .= buf
                end
            end
        end
    end
    @info "AtmosTransport GCHP: loaded for $nt timesteps"
    return (; fields)
end

function load_geoschem(gc_dir, rmap, levs; date_start, date_end)
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

    @info "GEOS-Chem: $(length(files)) snapshots"
    nt  = length(files)
    nl  = length(levs)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    fields = [zeros(Float32, rmap.nlon, rmap.nlat, nt) for _ in 1:nl]

    for (ti, fname) in enumerate(files)
        NCDataset(joinpath(gc_dir, fname), "r") do ds
            data = ds["SpeciesConcVV_CO2"]
            for (li, lev) in enumerate(levs)
                data_cs = Float64.(data[:, :, :, lev, 1]) .* 1e6
                regrid_cs!(buf, data_cs, rmap)
                fields[li][:, :, ti] .= buf
            end
        end
    end
    return (; times, fields)
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(gc, at, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    sfc_lo, sfc_hi = 380f0, 550f0
    hpa_lo, hpa_hi = 390f0, 450f0
    diff_max = 30f0

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(2000, 900), fontsize=13)

    ax_sfc_gc  = GeoAxis(fig[1, 1]; dest="+proj=robin", title="GEOS-Chem — Surface CO2")
    ax_sfc_at  = GeoAxis(fig[1, 2]; dest="+proj=robin", title="AtmosTransport GCHP — Surface CO2")
    ax_sfc_dif = GeoAxis(fig[1, 3]; dest="+proj=robin", title="Difference (GCHP - GC)")

    ax_hpa_gc  = GeoAxis(fig[2, 1]; dest="+proj=robin", title="GEOS-Chem — CO2 ~750 hPa")
    ax_hpa_at  = GeoAxis(fig[2, 2]; dest="+proj=robin", title="AtmosTransport GCHP — CO2 ~750 hPa")
    ax_hpa_dif = GeoAxis(fig[2, 3]; dest="+proj=robin", title="Difference (GCHP - GC)")

    z_sfc_gc  = Observable(gc.fields[1][:, :, 1]')
    z_sfc_at  = Observable(at.fields[1][:, :, 1]')
    z_sfc_dif = Observable((at.fields[1][:, :, 1] .- gc.fields[1][:, :, 1])')
    z_hpa_gc  = Observable(gc.fields[2][:, :, 1]')
    z_hpa_at  = Observable(at.fields[2][:, :, 1]')
    z_hpa_dif = Observable((at.fields[2][:, :, 1] .- gc.fields[2][:, :, 1])')

    sf1 = surface!(ax_sfc_gc, lon2d, lat2d, z_sfc_gc;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=(sfc_lo, sfc_hi))
    surface!(ax_sfc_at, lon2d, lat2d, z_sfc_at;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=(sfc_lo, sfc_hi))
    sf1d = surface!(ax_sfc_dif, lon2d, lat2d, z_sfc_dif;
        shading=NoShading, colormap=:RdBu, colorrange=(-diff_max, diff_max))

    sf2 = surface!(ax_hpa_gc, lon2d, lat2d, z_hpa_gc;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=(hpa_lo, hpa_hi))
    surface!(ax_hpa_at, lon2d, lat2d, z_hpa_at;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=(hpa_lo, hpa_hi))
    sf2d = surface!(ax_hpa_dif, lon2d, lat2d, z_hpa_dif;
        shading=NoShading, colormap=:RdBu, colorrange=(-diff_max, diff_max))

    for ax in [ax_sfc_gc, ax_sfc_at, ax_sfc_dif, ax_hpa_gc, ax_hpa_at, ax_hpa_dif]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 4], sf1; label="Surface CO2 [ppm]", width=14)
    Colorbar(fig[1, 5], sf1d; label="Diff [ppm]", width=14)
    Colorbar(fig[2, 4], sf2; label="CO2 ~750 hPa [ppm]", width=14)
    Colorbar(fig[2, 5], sf2d; label="Diff [ppm]", width=14)

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
        " UTC — CATRINE CO2: AtmosTransport (GCHP 1:1) vs GEOS-Chem")
    Label(fig[0, 1:5], title_obs; fontsize=18, font=:bold)

    @info "Writing $nframes frames to $OUT_GIF"

    Makie.record(fig, OUT_GIF, 1:nframes; framerate=fps) do frame_num
        z_sfc_gc[]  = gc.fields[1][:, :, frame_num]'
        z_sfc_at[]  = at.fields[1][:, :, frame_num]'
        z_sfc_dif[] = (at.fields[1][:, :, frame_num] .- gc.fields[1][:, :, frame_num])'
        z_hpa_gc[]  = gc.fields[2][:, :, frame_num]'
        z_hpa_at[]  = at.fields[2][:, :, frame_num]'
        z_hpa_dif[] = (at.fields[2][:, :, frame_num] .- gc.fields[2][:, :, frame_num])'

        title_obs[] = Dates.format(times[frame_num], "yyyy-mm-dd HH:MM") *
            " UTC — CATRINE CO2: AtmosTransport (GCHP 1:1) vs GEOS-Chem"
    end

    @info "Saved: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    @info "Loading CS coordinates..."
    cs_lons, cs_lats = load_cs_coordinates(joinpath(GC_DIR, gc_files[1]))

    @info "Building CS -> lat-lon map (1 deg)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    levs = [LEV_SURFACE, LEV_750HPA]

    @info "Loading GEOS-Chem..."
    gc = load_geoschem(GC_DIR, rmap, levs; date_start=DATE_START, date_end=DATE_END)

    @info "Loading AtmosTransport GCHP..."
    at = load_atmostr(AT_DIR, rmap, gc.times, levs)

    make_animation(gc, at, gc.times, rmap)
end

main()
