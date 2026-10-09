#!/usr/bin/env julia
# ===========================================================================
# 4-species surface comparison: AtmosTransport (hybrid PE) vs GEOS-Chem
#
# Layout (4 rows × 2 columns):
#   Row 1: Surface CO2          (GEOS-Chem | AtmosTransport)
#   Row 2: Surface Fossil CO2   (GEOS-Chem | AtmosTransport)
#   Row 3: Surface SF6          (GEOS-Chem | AtmosTransport)
#   Row 4: Surface Rn222        (GEOS-Chem | AtmosTransport)
#
# Usage:
#   julia --project=. scripts/completed_experiments/animate_hybrid_pe_vs_geoschem.jl
# ===========================================================================

using CairoMakie
using GeoMakie
using Dates
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "visualization", "cs_regrid_utils.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const GC_DIR = get(ENV, "GC_DIR",
    joinpath(homedir(), "data", "AtmosTransport", "catrine-geoschem-runs"))
const AT_DIR = get(ENV, "AT_DIR", "/temp2/catrine-runs/output")
const AT_PATTERN = get(ENV, "AT_PATTERN", "linrood_fullphys_7d")
const AT_LABEL = get(ENV, "AT_LABEL", "LR+HybridPE")
const OUT_GIF = get(ENV, "OUT_GIF", "hybrid_pe_vs_geoschem_4species.gif")
const FPS = parse(Int, get(ENV, "FPS", "4"))

const DATE_START = DateTime(2021, 12, 1)
const DATE_END   = DateTime(2021, 12, 8, 21, 0, 0)

# Level indices — both GC and our NC files are surface-first (k=1 = surface)
# (binary output is TOA-first, but the binary→NC converter flips to surface-first)
const GC_LEV_SURFACE = 1       # GC k=1 = surface (~1000 hPa)
const AT_LEV_SURFACE = 1       # Our NC k=1 = surface (after binary→NC flip)

# ---------------------------------------------------------------------------
# Load AT daily NC files with level mapping
# ---------------------------------------------------------------------------
function load_at_daily_nc(dir, pattern, rmap, var_name, levs;
                           date_start, date_end, scale=1.0, label="AT")
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, pattern), readdir(dir)))

    nlon, nlat = rmap.nlon, rmap.nlat
    buf = zeros(Float32, nlon, nlat)
    nl = length(levs)

    all_times = DateTime[]
    all_fields = [Vector{Matrix{Float32}}() for _ in 1:nl]

    for fname in daily_files
        NCDataset(joinpath(dir, fname), "r") do ds
            file_times = ds["time"][:]
            data_var = ds[var_name]
            for (ti, dt) in enumerate(file_times)
                date_start <= dt <= date_end || continue
                push!(all_times, dt)
                for (li, lev) in enumerate(levs)
                    data_cs = Float32.(data_var[:, :, :, lev, ti]) .* Float32(scale)
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
    @info "$label $var_name: $nt snapshots"
    fields = [zeros(Float32, nlon, nlat, nt) for _ in 1:nl]
    for li in 1:nl, ti in 1:nt
        fields[li][:, :, ti] .= all_fields[li][ti]
    end
    return (; times=all_times, fields)
end

# ---------------------------------------------------------------------------
# Compute per-frame statistics
# ---------------------------------------------------------------------------
function frame_stats(gc_field, at_field)
    gc_v = vec(gc_field)
    at_v = vec(at_field)
    valid = .!isnan.(gc_v) .& .!isnan.(at_v) .& (gc_v .!= 0)
    if count(valid) < 100
        return (r=NaN, bias=NaN, rmse=NaN)
    end
    g = gc_v[valid]
    a = at_v[valid]
    r = length(g) > 2 ? cor(a, g) : NaN
    bias = mean(a) - mean(g)
    rmse = sqrt(mean((a .- g).^2))
    return (; r, bias, rmse)
end

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
function make_animation(species_data, times, rmap; fps=FPS)
    nframes = length(times)
    @info "Animating $nframes frames at $fps fps"

    lon2d, lat2d = lon_lat_meshes(rmap)

    # Species config: (label, unit, gc_fields, at_fields, colormap, colorrange, diff_range)
    specs = [
        ("CO2", "ppm", species_data[:co2_gc], species_data[:co2_at],
         Reverse(:RdYlBu), (395f0, 430f0), (-15f0, 15f0)),
        ("Fossil CO2", "ppm", species_data[:fco2_gc], species_data[:fco2_at],
         :YlOrRd, (0f0, 2f0), (-0.5f0, 0.5f0)),
        ("SF6", "ppt", species_data[:sf6_gc], species_data[:sf6_at],
         :viridis, (9f0, 13f0), (-2f0, 2f0)),
        ("Rn222", "mBq/m³ STP", species_data[:rn_gc], species_data[:rn_at],
         :inferno, (0f0, 200f0), (-100f0, 100f0)),
    ]

    nrows = length(specs)
    fig = Figure(size=(1400, 280 * nrows + 60), fontsize=11)

    # Observables and axes for each species
    obs_gc = Observable[]
    obs_at = Observable[]
    obs_diff = Observable[]

    for (row, (label, unit, gc_f, at_f, cmap, crange, drange)) in enumerate(specs)
        ax_gc = GeoAxis(fig[row, 1]; dest="+proj=robin",
            title="GEOS-Chem — $label")
        ax_at = GeoAxis(fig[row, 2]; dest="+proj=robin",
            title="$AT_LABEL — $label")
        ax_diff = GeoAxis(fig[row, 3]; dest="+proj=robin",
            title="Δ$label ($AT_LABEL − GC)")

        z_gc = Observable(gc_f[:, :, 1]')
        z_at = Observable(at_f[:, :, 1]')
        z_d  = Observable((at_f[:, :, 1] .- gc_f[:, :, 1])')

        push!(obs_gc, z_gc)
        push!(obs_at, z_at)
        push!(obs_diff, z_d)

        sf_gc = surface!(ax_gc, lon2d, lat2d, z_gc;
            shading=NoShading, colormap=cmap, colorrange=crange)
        surface!(ax_at, lon2d, lat2d, z_at;
            shading=NoShading, colormap=cmap, colorrange=crange)
        sf_diff = surface!(ax_diff, lon2d, lat2d, z_d;
            shading=NoShading, colormap=:RdBu, colorrange=drange)

        for ax in [ax_gc, ax_at, ax_diff]
            lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.5)
        end

        Colorbar(fig[row, 4], sf_gc; label="$label [$unit]", width=14)
        Colorbar(fig[row, 5], sf_diff; label="Δ [$unit]", width=14)
    end

    # Stats label
    stats_obs = Observable("r=... bias=...")
    Label(fig[nrows+1, 1:3], stats_obs; fontsize=10, halign=:left,
          font=:regular, color=:gray40)

    title_obs = Observable(Dates.format(times[1], "yyyy-mm-dd HH:MM") *
                           " UTC — $AT_LABEL vs GEOS-Chem")
    Label(fig[0, 1:5], title_obs; fontsize=16, font=:bold)

    @info "Rendering $nframes frames to $OUT_GIF"

    record(fig, OUT_GIF, 1:nframes; framerate=fps) do fi
        for (row, (label, _, gc_f, at_f, _, _, _)) in enumerate(specs)
            obs_gc[row][]   = gc_f[:, :, fi]'
            obs_at[row][]   = at_f[:, :, fi]'
            obs_diff[row][] = (at_f[:, :, fi] .- gc_f[:, :, fi])'
        end

        # Compute stats for fossil CO2 (best transport test)
        s = frame_stats(species_data[:fco2_gc][:, :, fi],
                        species_data[:fco2_at][:, :, fi])
        stats_obs[] = @sprintf("Fossil CO2 sfc: r=%.3f  bias=%+.3f ppm  RMSE=%.3f ppm",
                               s.r, s.bias, s.rmse)

        title_obs[] = Dates.format(times[fi], "yyyy-mm-dd HH:MM") *
                      " UTC — $AT_LABEL vs GEOS-Chem"
    end

    @info "Saved: $OUT_GIF ($nframes frames)"
end

# ---------------------------------------------------------------------------
# Rn222 unit conversion: mol/mol → mBq/m³ at STP
# ---------------------------------------------------------------------------
# Rn222 half-life = 3.8235 days → λ = ln(2)/t½
# Activity = λ × N_atoms; at STP: n_air = P/(RT) = 101325/(8.314×273.15) ≈ 44.615 mol/m³
# mBq/m³ = VMR × n_air × N_A × λ × 1e3
const RN_LAMBDA = log(2) / (3.8235 * 86400)  # s⁻¹
const N_AIR_STP = 101325.0 / (8.314 * 273.15)  # mol/m³
const N_A = 6.02214076e23
const RN_SCALE = Float32(N_AIR_STP * N_A * RN_LAMBDA * 1e3)  # VMR → mBq/m³

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    isdir(GC_DIR) || error("GEOS-Chem directory not found: $GC_DIR")
    isdir(AT_DIR) || error("AT output not found: $AT_DIR")

    # Load CS coordinates from first GC file
    gc_files = sort(filter(f -> endswith(f, ".nc4"), readdir(GC_DIR)))
    @info "Loading CS coordinates from $(gc_files[1])"
    cs_lons, cs_lats = load_cs_coordinates(joinpath(GC_DIR, gc_files[1]))

    @info "Building CS → lat-lon regridding map (1°)..."
    rmap = build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

    # --- Load GEOS-Chem data (surface = level 1) ---
    @info "Loading GEOS-Chem data..."
    gc_co2 = load_geoschem_nc(GC_DIR, rmap, "SpeciesConcVV_CO2",
        [GC_LEV_SURFACE]; date_start=DATE_START, date_end=DATE_END, scale=1e6)
    gc_fco2 = load_geoschem_nc(GC_DIR, rmap, "SpeciesConcVV_FossilCO2",
        [GC_LEV_SURFACE]; date_start=DATE_START, date_end=DATE_END, scale=1e6)
    gc_sf6 = load_geoschem_nc(GC_DIR, rmap, "SpeciesConcVV_SF6",
        [GC_LEV_SURFACE]; date_start=DATE_START, date_end=DATE_END, scale=1e12)
    gc_rn = load_geoschem_nc(GC_DIR, rmap, "SpeciesConcVV_Rn222",
        [GC_LEV_SURFACE]; date_start=DATE_START, date_end=DATE_END, scale=Float64(RN_SCALE))

    # --- Load AtmosTransport data (surface = level 72, TOA-first) ---
    @info "Loading AtmosTransport ($AT_LABEL) data..."
    at_co2 = load_at_daily_nc(AT_DIR, AT_PATTERN, rmap, "co2_3d",
        [AT_LEV_SURFACE]; date_start=DATE_START, date_end=DATE_END,
        scale=1e6, label=AT_LABEL)
    at_fco2 = load_at_daily_nc(AT_DIR, AT_PATTERN, rmap, "fossil_co2_3d",
        [AT_LEV_SURFACE]; date_start=DATE_START, date_end=DATE_END,
        scale=1e6, label=AT_LABEL)
    at_sf6 = load_at_daily_nc(AT_DIR, AT_PATTERN, rmap, "sf6_3d",
        [AT_LEV_SURFACE]; date_start=DATE_START, date_end=DATE_END,
        scale=1e12, label=AT_LABEL)
    at_rn = load_at_daily_nc(AT_DIR, AT_PATTERN, rmap, "rn222_3d",
        [AT_LEV_SURFACE]; date_start=DATE_START, date_end=DATE_END,
        scale=Float64(RN_SCALE), label=AT_LABEL)

    # --- Align to common timesteps ---
    # Collect all unique times across all datasets
    all_datasets = [gc_co2, gc_fco2, gc_sf6, gc_rn, at_co2, at_fco2, at_sf6, at_rn]
    common = Set(all_datasets[1].times)
    for ds in all_datasets[2:end]
        intersect!(common, Set(ds.times))
    end
    times = sort(collect(common))
    @info "Common timesteps: $(length(times))"

    # Extract aligned fields
    function extract_aligned(ds)
        idx = [findfirst(==(t), ds.times) for t in times]
        ds.fields[1][:, :, idx]
    end

    species_data = (
        co2_gc  = extract_aligned(gc_co2),
        co2_at  = extract_aligned(at_co2),
        fco2_gc = extract_aligned(gc_fco2),
        fco2_at = extract_aligned(at_fco2),
        sf6_gc  = extract_aligned(gc_sf6),
        sf6_at  = extract_aligned(at_sf6),
        rn_gc   = extract_aligned(gc_rn),
        rn_at   = extract_aligned(at_rn),
    )

    make_animation(species_data, times, rmap)
end

main()
