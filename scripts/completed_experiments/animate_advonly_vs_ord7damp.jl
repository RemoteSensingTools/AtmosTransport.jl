#!/usr/bin/env julia
# ===========================================================================
# Advection-only vs Full-physics (ORD=7+damp) vs GEOS-Chem: CO2 comparison
#
# Layout:
#   Row 1: Surface CO2 VMR (GC | ORD7+damp | Adv-only)
#   Row 2: 750 hPa CO2 VMR (GC | ORD7+damp | Adv-only)
#   Row 3: Mass tracking (total mass | mass change from initial)
#   Footer: per-frame stats
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_advonly_vs_ord7damp.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using NCDatasets
using Dates
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const GC_DIR      = expanduser("~/data/AtmosTransport/catrine-geoschem-runs")
const AT_DIR      = "/temp2/catrine-runs/output"
const FULL_PATTERN = "catrine_geosit_c180_ord7damp"
const ADV_PATTERN  = "catrine_geosit_c180_advonly"
const OUT_GIF     = "catrine_advonly_vs_full_co2.gif"
const FPS         = 8

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1, 3, 0, 0)
const DATE_END   = DateTime(2021, 12, 31, 21, 0, 0)

# CO2 colorscale (ppm)
const SFC_RANGE = (390f0, 450f0)
const HPA_RANGE = (408f0, 428f0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function frame_stats(gc_frame, at_frame)
    g, a = vec(gc_frame), vec(at_frame)
    mask = (g .> 350f0) .& (g .< 500f0) .& (a .> 350f0) .& (a .< 500f0)
    n = count(mask)
    n < 10 && return (; r2=NaN, rmse=NaN)
    gm, am = g[mask], a[mask]
    r2 = cor(gm, am)^2
    rmse = sqrt(mean((am .- gm).^2))
    return (; r2, rmse)
end

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

function load_gc(gc_dir, rmap, area)
    all_files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"),
                             readdir(gc_dir)))
    files, times = String[], DateTime[]
    for f in all_files
        m = match(r"(\d{8})_(\d{4})z", f)
        m === nothing && continue
        dt = DateTime(m[1] * m[2], dateformat"yyyymmddHHMM")
        DATE_START <= dt <= DATE_END && (push!(files, f); push!(times, dt))
    end

    nt = length(files)
    @info "GEOS-Chem: $nt snapshots ($(times[1]) -> $(times[end]))"

    nlon, nlat = rmap.nlon, rmap.nlat
    buf = zeros(Float32, nlon, nlat)

    co2_sfc = zeros(Float32, nlon, nlat, nt)
    co2_750 = zeros(Float32, nlon, nlat, nt)
    global_mass = zeros(Float64, nt)

    for (ti, fname) in enumerate(files)
        NCDataset(joinpath(gc_dir, fname), "r") do ds
            d = Float64.(ds["SpeciesConcVV_CO2"][:, :, :, LEV_SURFACE, 1]) .* 1e6
            regrid_cs!(buf, d, rmap); co2_sfc[:, :, ti] .= buf

            d = Float64.(ds["SpeciesConcVV_CO2"][:, :, :, LEV_750HPA, 1]) .* 1e6
            regrid_cs!(buf, d, rmap); co2_750[:, :, ti] .= buf

            cm = Float64.(ds["ColumnMass_CO2"][:, :, :, 1])
            global_mass[ti] = sum(cm .* area)
        end
    end

    dm = global_mass .- global_mass[1]
    @info @sprintf("  GC CO2 mass: init=%.1f Gt  final=%.1f Gt  DM=%.2f Gt",
        global_mass[1]/1e12, global_mass[end]/1e12, dm[end]/1e12)

    return (; times, co2_sfc, co2_750, global_mass)
end

function load_at(at_dir, pattern, rmap, area)
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, pattern), readdir(at_dir)))

    nlon, nlat = rmap.nlon, rmap.nlat
    buf = zeros(Float32, nlon, nlat)

    all_times  = DateTime[]
    sfc_list   = Matrix{Float32}[]
    hpa_list   = Matrix{Float32}[]
    mass_list  = Float64[]

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            file_times = ds["time"][:]
            co2_3d = ds["co2_3d"]
            co2_cm = ds["co2_column_mass"]

            for (ti, dt) in enumerate(file_times)
                DATE_START <= dt <= DATE_END || continue
                push!(all_times, dt)

                d = Float32.(co2_3d[:, :, :, LEV_SURFACE, ti]) .* 1f6
                clamp!(d, 0f0, Inf32)
                regrid_cs!(buf, d, rmap); push!(sfc_list, copy(buf))

                d = Float32.(co2_3d[:, :, :, LEV_750HPA, ti]) .* 1f6
                clamp!(d, 0f0, Inf32)
                regrid_cs!(buf, d, rmap); push!(hpa_list, copy(buf))

                cm = Float64.(co2_cm[:, :, :, ti])
                push!(mass_list, sum(cm .* area))
            end
        end
    end

    perm = sortperm(all_times)
    times = all_times[perm]
    nt = length(times)

    co2_sfc = zeros(Float32, nlon, nlat, nt)
    co2_750 = zeros(Float32, nlon, nlat, nt)
    for ti in 1:nt
        co2_sfc[:, :, ti] .= sfc_list[perm[ti]]
        co2_750[:, :, ti] .= hpa_list[perm[ti]]
    end

    global_mass = mass_list[perm]
    dm = global_mass .- global_mass[1]

    @info "$pattern: $nt snapshots ($(times[1]) -> $(times[end]))"
    @info @sprintf("  AT CO2 mass: init=%.1f Gt  final=%.1f Gt  DM=%.2f Gt",
        global_mass[1]/1e12, global_mass[end]/1e12, dm[end]/1e12)

    return (; times, co2_sfc, co2_750, global_mass)
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

function make_animation(gc, full, adv, rmap)
    # Build time index: AT full -> GC
    gc_idx = zeros(Int, length(full.times))
    for (ai, at_t) in enumerate(full.times)
        for (gi, gc_t) in enumerate(gc.times)
            abs(Dates.value(at_t - gc_t)) < 30 * 60_000 && (gc_idx[ai] = gi; break)
        end
    end

    # Build time index: AT full -> AT adv (should be 1:1)
    adv_idx = zeros(Int, length(full.times))
    for (fi, ft) in enumerate(full.times)
        for (ai, at) in enumerate(adv.times)
            abs(Dates.value(ft - at)) < 30 * 60_000 && (adv_idx[fi] = ai; break)
        end
    end

    nframes = length(full.times)
    @info "Animation: $nframes frames at $FPS fps"

    lon2d, lat2d = lon_lat_meshes(rmap)

    fig = Figure(size=(2100, 1300), fontsize=12)

    # Row 1: Surface maps (3 columns)
    ax_sfc_gc   = GeoAxis(fig[1, 1]; dest="+proj=robin", title="GEOS-Chem  Surface CO\u2082")
    ax_sfc_full = GeoAxis(fig[1, 2]; dest="+proj=robin", title="Full Physics (ORD7+damp)  Surface CO\u2082")
    ax_sfc_adv  = GeoAxis(fig[1, 3]; dest="+proj=robin", title="Advection Only  Surface CO\u2082")

    # Row 2: 750 hPa maps
    ax_hpa_gc   = GeoAxis(fig[2, 1]; dest="+proj=robin", title="GEOS-Chem  ~750 hPa CO\u2082")
    ax_hpa_full = GeoAxis(fig[2, 2]; dest="+proj=robin", title="Full Physics  ~750 hPa CO\u2082")
    ax_hpa_adv  = GeoAxis(fig[2, 3]; dest="+proj=robin", title="Advection Only  ~750 hPa CO\u2082")

    # Row 3: Mass tracking
    ax_mass = Axis(fig[3, 1:2]; ylabel="CO\u2082 Mass (Gt)", title="Total Atmospheric CO\u2082 Mass")
    ax_dm   = Axis(fig[3, 3]; ylabel="\u0394M (Gt)", title="Mass Change from Initial")

    # Map observables
    z_sfc_gc   = Observable(gc.co2_sfc[:, :, 1]')
    z_sfc_full = Observable(full.co2_sfc[:, :, 1]')
    z_sfc_adv  = Observable(adv.co2_sfc[:, :, 1]')
    z_hpa_gc   = Observable(gc.co2_750[:, :, 1]')
    z_hpa_full = Observable(full.co2_750[:, :, 1]')
    z_hpa_adv  = Observable(adv.co2_750[:, :, 1]')

    sf1 = surface!(ax_sfc_gc, lon2d, lat2d, z_sfc_gc;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=SFC_RANGE)
    surface!(ax_sfc_full, lon2d, lat2d, z_sfc_full;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=SFC_RANGE)
    surface!(ax_sfc_adv, lon2d, lat2d, z_sfc_adv;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=SFC_RANGE)

    sf2 = surface!(ax_hpa_gc, lon2d, lat2d, z_hpa_gc;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=HPA_RANGE)
    surface!(ax_hpa_full, lon2d, lat2d, z_hpa_full;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=HPA_RANGE)
    surface!(ax_hpa_adv, lon2d, lat2d, z_hpa_adv;
        shading=NoShading, colormap=Reverse(:RdYlBu), colorrange=HPA_RANGE)

    for ax in [ax_sfc_gc, ax_sfc_full, ax_sfc_adv, ax_hpa_gc, ax_hpa_full, ax_hpa_adv]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 4], sf1; label="Surface CO\u2082 [ppm]", width=14)
    Colorbar(fig[2, 4], sf2; label="750 hPa CO\u2082 [ppm]", width=14)

    # Timeseries
    t0 = full.times[1]
    full_days = [(t - t0).value / (24*3600*1000) for t in full.times]
    adv_days  = [(t - t0).value / (24*3600*1000) for t in adv.times]
    gc_days   = [(t - t0).value / (24*3600*1000) for t in gc.times]

    lines!(ax_mass, gc_days, gc.global_mass ./ 1e12;
        color=:steelblue, linewidth=2, label="GEOS-Chem")
    lines!(ax_mass, full_days, full.global_mass ./ 1e12;
        color=:firebrick, linewidth=2, label="Full Physics")
    lines!(ax_mass, adv_days, adv.global_mass ./ 1e12;
        color=:forestgreen, linewidth=2, label="Adv Only")
    axislegend(ax_mass; position=:lt, labelsize=10)

    gc_dm   = (gc.global_mass .- gc.global_mass[1]) ./ 1e12
    full_dm = (full.global_mass .- full.global_mass[1]) ./ 1e12
    adv_dm  = (adv.global_mass .- adv.global_mass[1]) ./ 1e12
    lines!(ax_dm, gc_days, gc_dm; color=:steelblue, linewidth=2, label="GC")
    lines!(ax_dm, full_days, full_dm; color=:firebrick, linewidth=2, label="Full")
    lines!(ax_dm, adv_days, adv_dm; color=:forestgreen, linewidth=2, label="Adv")
    hlines!(ax_dm, 0.0; color=:black, linewidth=0.5, linestyle=:dash)
    axislegend(ax_dm; position=:lt, labelsize=10)

    vline_day = Observable(full_days[1])
    vlines!(ax_mass, vline_day; color=(:gray50, 0.7), linewidth=1.5)
    vlines!(ax_dm,   vline_day; color=(:gray50, 0.7), linewidth=1.5)

    title_obs = Observable("CO\u2082 — 2021-12-01 03:00 UTC")
    Label(fig[0, 1:4], title_obs; fontsize=18, font=:bold)

    stats_obs = Observable("")
    Label(fig[4, 1:4], stats_obs; fontsize=13, halign=:center, color=:gray30)

    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=FPS) do frame
        z_sfc_full[] = full.co2_sfc[:, :, frame]'
        z_hpa_full[] = full.co2_750[:, :, frame]'

        ai = adv_idx[frame]
        if ai > 0
            z_sfc_adv[] = adv.co2_sfc[:, :, ai]'
            z_hpa_adv[] = adv.co2_750[:, :, ai]'
        end

        gi = gc_idx[frame]
        if gi > 0
            z_sfc_gc[] = gc.co2_sfc[:, :, gi]'
            z_hpa_gc[] = gc.co2_750[:, :, gi]'
        end

        vline_day[] = full_days[frame]

        dt_str = Dates.format(full.times[frame], "yyyy-mm-dd HH:MM")
        title_obs[] = "CO\u2082 — $dt_str UTC"

        full_mass_Gt = full.global_mass[frame] / 1e12
        adv_mass_Gt = ai > 0 ? adv.global_mass[ai] / 1e12 : NaN

        if gi > 0
            s_full = frame_stats(gc.co2_sfc[:, :, gi], full.co2_sfc[:, :, frame])
            s_adv = ai > 0 ? frame_stats(gc.co2_sfc[:, :, gi], adv.co2_sfc[:, :, ai]) : (; r2=NaN, rmse=NaN)
            gc_mass_Gt = gc.global_mass[gi] / 1e12
            stats_obs[] = @sprintf(
                "Full vs GC: R\u00b2=%.3f RMSE=%.1f  |  Adv vs GC: R\u00b2=%.3f RMSE=%.1f  |  Mass: GC=%.1f  Full=%.1f  Adv=%.1f Gt",
                s_full.r2, s_full.rmse, s_adv.r2, s_adv.rmse, gc_mass_Gt, full_mass_Gt, adv_mass_Gt)
        else
            stats_obs[] = @sprintf(
                "GC data ends Dec 14  |  Mass: Full=%.1f Gt  Adv=%.1f Gt",
                full_mass_Gt, adv_mass_Gt)
        end
    end

    @info "Saved: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    isdir(GC_DIR) || error("GEOS-Chem directory not found: $GC_DIR")
    isdir(AT_DIR) || error("AtmosTransport output not found: $AT_DIR")

    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    @info "Loading CS coordinates + areas from $(gc_files[1])"
    gc_path1 = joinpath(GC_DIR, gc_files[1])
    cs_lons, cs_lats = load_cs_coordinates(gc_path1)
    area = NCDataset(gc_path1, "r") do ds; Float64.(ds["Met_AREAM2"][:, :, :, 1]); end

    @info "Building CS -> lat-lon regridding map (1 deg)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    @info "Loading GEOS-Chem CO2..."
    gc = load_gc(GC_DIR, rmap, area)

    @info "Loading Full Physics (ORD7+damp) CO2..."
    full = load_at(AT_DIR, FULL_PATTERN, rmap, area)

    @info "Loading Advection-Only CO2..."
    adv = load_at(AT_DIR, ADV_PATTERN, rmap, area)

    @info "Creating animation..."
    make_animation(gc, full, adv, rmap)
end

main()
