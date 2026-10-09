#!/usr/bin/env julia
# ===========================================================================
# CATRINE ORD=7+damping vs GEOS-Chem comparison animation
#
# Layout:
#   Row 1: Surface fossil CO2 VMR (GEOS-Chem | AtmosTransport)
#   Row 2: 750 hPa fossil CO2 VMR (GEOS-Chem | AtmosTransport)
#   Row 3: Mass balance timeseries (total mass + mass error)
#   Footer: per-frame R², RMSE, mass conservation stats
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_catrine_ord7damp.jl
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
const GC_DIR     = expanduser("~/data/AtmosTransport/catrine-geoschem-runs")
const AT_DIR     = "/temp2/catrine-runs/output"
const AT_PATTERN = "catrine_geosit_c180_ord7damp"
const OUT_GIF    = "catrine_ord7damp_vs_geoschem.gif"
const FPS        = 8

const LEV_SURFACE = 1
const LEV_750HPA  = 15

const DATE_START = DateTime(2021, 12, 1, 3, 0, 0)
const DATE_END   = DateTime(2021, 12, 31, 21, 0, 0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Trapezoidal cumulative integration of rates [kg/s] over DateTime vector."""
function cumulative_emissions(rates::Vector{Float64}, times::Vector{DateTime})
    nt = length(times)
    cum = zeros(Float64, nt)
    for t in 2:nt
        dt_s = Dates.value(times[t] - times[t-1]) / 1000.0
        cum[t] = cum[t-1] + 0.5 * (rates[t-1] + rates[t]) * dt_s
    end
    return cum
end

"""Correlation stats between two regridded frames (where GC > threshold)."""
function frame_stats(gc_frame, at_frame; threshold=0.01f0)
    gc_flat = vec(gc_frame)
    at_flat = vec(at_frame)
    mask = gc_flat .> threshold
    n = count(mask)
    n < 10 && return (; r2=NaN, rmse=NaN)
    g, a = gc_flat[mask], at_flat[mask]
    r2 = cor(g, a)^2
    rmse = sqrt(mean((a .- g).^2))
    return (; r2, rmse)
end

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

"""Load GC snapshots: regridded concentration maps + native mass balance fields."""
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
    @info "GEOS-Chem: $nt snapshots ($(times[1]) → $(times[end]))"

    nlon, nlat = rmap.nlon, rmap.nlat
    buf = zeros(Float32, nlon, nlat)

    fco2_sfc = zeros(Float32, nlon, nlat, nt)
    fco2_750 = zeros(Float32, nlon, nlat, nt)
    global_mass     = zeros(Float64, nt)
    emission_rate   = zeros(Float64, nt)

    for (ti, fname) in enumerate(files)
        NCDataset(joinpath(gc_dir, fname), "r") do ds
            d = Float64.(ds["SpeciesConcVV_FossilCO2"][:, :, :, LEV_SURFACE, 1]) .* 1e6
            regrid_cs!(buf, d, rmap); fco2_sfc[:, :, ti] .= buf

            d = Float64.(ds["SpeciesConcVV_FossilCO2"][:, :, :, LEV_750HPA, 1]) .* 1e6
            regrid_cs!(buf, d, rmap); fco2_750[:, :, ti] .= buf

            # Column mass × area = total tracer mass (GC convention)
            cm = Float64.(ds["ColumnMass_FossilCO2"][:, :, :, 1])
            global_mass[ti] = sum(cm .* area)

            emis = Float64.(ds["Emis_FossilCO2_Total"][:, :, :, 1])
            emission_rate[ti] = sum(emis .* area)
        end
    end

    cum_emis = cumulative_emissions(emission_rate, times)
    mass_error = (global_mass .- global_mass[1]) .- cum_emis

    @info @sprintf("  GC: init=%.2f Tg  final=%.2f Tg  cum_emis=%.2f Tg  error=%.3f Tg",
        global_mass[1]/1e9, global_mass[end]/1e9, cum_emis[end]/1e9, mass_error[end]/1e9)

    return (; times, fco2_sfc, fco2_750, global_mass, cum_emis, mass_error)
end

"""Load AT daily NC files: regridded maps + native mass balance fields."""
function load_at(at_dir, pattern, rmap, area)
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, pattern), readdir(at_dir)))

    nlon, nlat = rmap.nlon, rmap.nlat
    buf = zeros(Float32, nlon, nlat)

    all_times  = DateTime[]
    sfc_list   = Matrix{Float32}[]
    hpa_list   = Matrix{Float32}[]
    mass_list  = Float64[]
    emis_list  = Float64[]

    for fname in daily_files
        NCDataset(joinpath(at_dir, fname), "r") do ds
            file_times = ds["time"][:]
            fco2_3d = ds["fossil_co2_3d"]
            fco2_cm = ds["fco2_column_mass"]
            has_emis = haskey(ds, "fco2_emission")

            for (ti, dt) in enumerate(file_times)
                DATE_START <= dt <= DATE_END || continue
                push!(all_times, dt)

                d = Float32.(fco2_3d[:, :, :, LEV_SURFACE, ti]) .* 1f6
                regrid_cs!(buf, d, rmap); push!(sfc_list, copy(buf))

                d = Float32.(fco2_3d[:, :, :, LEV_750HPA, ti]) .* 1f6
                regrid_cs!(buf, d, rmap); push!(hpa_list, copy(buf))

                cm = Float64.(fco2_cm[:, :, :, ti])
                push!(mass_list, sum(cm .* area))

                if has_emis
                    em = Float64.(ds["fco2_emission"][:, :, :, ti])
                    push!(emis_list, sum(em .* area))
                else
                    push!(emis_list, 0.0)
                end
            end
        end
    end

    perm = sortperm(all_times)
    times = all_times[perm]
    nt = length(times)

    fco2_sfc = zeros(Float32, nlon, nlat, nt)
    fco2_750 = zeros(Float32, nlon, nlat, nt)
    for ti in 1:nt
        fco2_sfc[:, :, ti] .= sfc_list[perm[ti]]
        fco2_750[:, :, ti] .= hpa_list[perm[ti]]
    end

    global_mass   = mass_list[perm]
    emission_rate = emis_list[perm]
    cum_emis      = cumulative_emissions(emission_rate, times)
    mass_error    = (global_mass .- global_mass[1]) .- cum_emis

    @info "AtmosTransport: $nt snapshots ($(times[1]) → $(times[end]))"
    @info @sprintf("  AT: init=%.2f Tg  final=%.2f Tg  cum_emis=%.2f Tg  error=%.3f Tg",
        global_mass[1]/1e9, global_mass[end]/1e9, cum_emis[end]/1e9, mass_error[end]/1e9)

    return (; times, fco2_sfc, fco2_750, global_mass, cum_emis, mass_error)
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

function make_animation(gc, at, rmap)
    # Build AT→GC time index (0 = no GC data for this frame)
    gc_idx = zeros(Int, length(at.times))
    for (ai, at_t) in enumerate(at.times)
        for (gi, gc_t) in enumerate(gc.times)
            abs(Dates.value(at_t - gc_t)) < 30 * 60_000 && (gc_idx[ai] = gi; break)
        end
    end

    nframes = length(at.times)
    n_matched = count(gc_idx .> 0)
    @info "Animation: $nframes frames ($n_matched with GC match)"

    sfc_max, hpa_max = 30f0, 10f0
    lon2d, lat2d = lon_lat_meshes(rmap)

    # --- Figure layout ---
    fig = Figure(size=(1600, 1300), fontsize=12)

    # Row 1: Surface maps
    ax_sfc_gc = GeoAxis(fig[1, 1]; dest="+proj=robin",
        title="GEOS-Chem  Surface fossil CO\u2082")
    ax_sfc_at = GeoAxis(fig[1, 2]; dest="+proj=robin",
        title="AtmosTransport (ORD=7 + damp)  Surface fossil CO\u2082")

    # Row 2: 750 hPa maps
    ax_hpa_gc = GeoAxis(fig[2, 1]; dest="+proj=robin",
        title="GEOS-Chem  ~750 hPa fossil CO\u2082")
    ax_hpa_at = GeoAxis(fig[2, 2]; dest="+proj=robin",
        title="AtmosTransport  ~750 hPa fossil CO\u2082")

    # Row 3: Mass balance timeseries
    ax_mass = Axis(fig[3, 1]; ylabel="Fossil CO\u2082 Mass (Tg)",
        title="Total Atmospheric Mass")
    ax_err  = Axis(fig[3, 2]; ylabel="Mass Error (Tg)",
        title="Mass Balance Error = \u0394M \u2212 \u03a3 Emissions")

    # --- Map observables ---
    z_sfc_gc = Observable(gc.fco2_sfc[:, :, 1]')
    z_sfc_at = Observable(at.fco2_sfc[:, :, 1]')
    z_hpa_gc = Observable(gc.fco2_750[:, :, 1]')
    z_hpa_at = Observable(at.fco2_750[:, :, 1]')

    sf1 = surface!(ax_sfc_gc, lon2d, lat2d, z_sfc_gc;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)
    surface!(ax_sfc_at, lon2d, lat2d, z_sfc_at;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, sfc_max), colorscale=safe_sqrt)
    sf2 = surface!(ax_hpa_gc, lon2d, lat2d, z_hpa_gc;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)
    surface!(ax_hpa_at, lon2d, lat2d, z_hpa_at;
        shading=NoShading, colormap=:YlOrRd, colorrange=(0f0, hpa_max), colorscale=safe_sqrt)

    for ax in [ax_sfc_gc, ax_sfc_at, ax_hpa_gc, ax_hpa_at]
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 3], sf1; label="Surface [ppm]", width=14)
    Colorbar(fig[2, 3], sf2; label="750 hPa [ppm]", width=14)

    # --- Timeseries (pre-drawn, vertical line moves) ---
    t0 = at.times[1]
    at_days = [(t - t0).value / (24*3600*1000) for t in at.times]
    gc_days = [(t - t0).value / (24*3600*1000) for t in gc.times]

    # Mass panel: solid = actual mass, dashed = initial + cumulative emissions
    lines!(ax_mass, gc_days, gc.global_mass ./ 1e9;
        color=:steelblue, linewidth=2, label="GC mass")
    lines!(ax_mass, at_days, at.global_mass ./ 1e9;
        color=:firebrick, linewidth=2, label="AT mass")
    lines!(ax_mass, gc_days, gc.cum_emis ./ 1e9 .+ gc.global_mass[1] / 1e9;
        color=:steelblue, linewidth=1, linestyle=:dash, label="GC M\u2080+\u03a3E")
    lines!(ax_mass, at_days, at.cum_emis ./ 1e9 .+ at.global_mass[1] / 1e9;
        color=:firebrick, linewidth=1, linestyle=:dash, label="AT M\u2080+\u03a3E")
    axislegend(ax_mass; position=:lt, labelsize=10, nbanks=2)

    # Error panel
    lines!(ax_err, gc_days, gc.mass_error ./ 1e9;
        color=:steelblue, linewidth=2, label="GEOS-Chem")
    lines!(ax_err, at_days, at.mass_error ./ 1e9;
        color=:firebrick, linewidth=2, label="AtmosTransport")
    hlines!(ax_err, 0.0; color=:black, linewidth=0.5, linestyle=:dash)
    axislegend(ax_err; position=:lt, labelsize=10)

    # Moving vertical line
    vline_day = Observable(at_days[1])
    vlines!(ax_mass, vline_day; color=(:gray50, 0.7), linewidth=1.5)
    vlines!(ax_err,  vline_day; color=(:gray50, 0.7), linewidth=1.5)

    # Supertitle + stats footer
    title_obs = Observable("Fossil CO\u2082 Enhancement — 2021-12-01 03:00 UTC")
    Label(fig[0, 1:3], title_obs; fontsize=18, font=:bold)

    stats_obs = Observable("")
    Label(fig[4, 1:3], stats_obs; fontsize=13, halign=:center, color=:gray30)

    # --- Record animation ---
    @info "Writing $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=FPS) do frame
        # AT maps (always available)
        z_sfc_at[] = at.fco2_sfc[:, :, frame]'
        z_hpa_at[] = at.fco2_750[:, :, frame]'

        # GC maps (freeze on last available when past GC data range)
        gi = gc_idx[frame]
        if gi > 0
            z_sfc_gc[] = gc.fco2_sfc[:, :, gi]'
            z_hpa_gc[] = gc.fco2_750[:, :, gi]'
        end

        vline_day[] = at_days[frame]

        dt_str = Dates.format(at.times[frame], "yyyy-mm-dd HH:MM")
        title_obs[] = "Fossil CO\u2082 Enhancement — $dt_str UTC"

        # Stats annotation
        at_err_pct = at.global_mass[frame] > 0 ?
            100 * at.mass_error[frame] / at.global_mass[frame] : 0.0

        if gi > 0
            s_sfc = frame_stats(gc.fco2_sfc[:, :, gi], at.fco2_sfc[:, :, frame])
            s_hpa = frame_stats(gc.fco2_750[:, :, gi], at.fco2_750[:, :, frame])
            gc_err_pct = gc.global_mass[gi] > 0 ?
                100 * gc.mass_error[gi] / gc.global_mass[gi] : 0.0
            stats_obs[] = @sprintf(
                "Surface: R\u00b2=%.3f  RMSE=%.2f ppm  |  750hPa: R\u00b2=%.3f  RMSE=%.2f ppm  |  Mass error: AT=%.3f%%  GC=%.3f%%",
                s_sfc.r2, s_sfc.rmse, s_hpa.r2, s_hpa.rmse, at_err_pct, gc_err_pct)
        else
            stats_obs[] = @sprintf(
                "GC data ends Dec 14  |  AT mass: %.1f Tg  |  AT mass error: %.3f%%",
                at.global_mass[frame] / 1e9, at_err_pct)
        end
    end

    @info "Saved animation: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    isdir(GC_DIR) || error("GEOS-Chem directory not found: $GC_DIR")
    isdir(AT_DIR) || error("AtmosTransport output not found: $AT_DIR")

    # Load cell areas from first GC file (same C180 grid as AT)
    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    @info "Loading CS coordinates + areas from $(gc_files[1])"
    gc_path1 = joinpath(GC_DIR, gc_files[1])
    cs_lons, cs_lats = load_cs_coordinates(gc_path1)
    area = NCDataset(gc_path1, "r") do ds
        Float64.(ds["Met_AREAM2"][:, :, :, 1])
    end

    @info "Building CS -> lat-lon regridding map (1 deg)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    @info "Loading GEOS-Chem data..."
    gc = load_gc(GC_DIR, rmap, area)

    @info "Loading AtmosTransport data..."
    at = load_at(AT_DIR, AT_PATTERN, rmap, area)

    @info "Creating animation..."
    make_animation(gc, at, rmap)
end

main()
