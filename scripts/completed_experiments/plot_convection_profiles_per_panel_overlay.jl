#!/usr/bin/env julia
# Overlay-plot: per-panel mass-weighted mean vertical profile of each
# tracer at selected timestamps, overplotting three convection schemes
# (TM5 post-closure, CMFMC matrix, CMFMC post-fix GCHP-faithful) with
# distinct line styles on a shared pressure y-axis. One PNG per tracer.
#
# Usage:
#   julia --project=. scripts/completed_experiments/plot_convection_profiles_per_panel_overlay.jl
#
# Outputs to <repo>/plots/convection_per_panel_overlay/.

using CairoMakie
using NCDatasets
using Printf
using Statistics
using Dates

const TRACERS = ("co2_lowest_layer", "co2_p08", "co2_p06", "co2_p04")

# Fewer timesteps than the per-scheme plot because we now stack 3 schemes
# per panel (3 × Nt lines per subplot — keep readable).
const SNAPSHOT_HOURS = (0.0, 12.0, 24.0, 72.0)

# Each entry: (label, path, linestyle).  Order = draw order; later
# schemes paint over earlier ones at line crossings.
const SCHEMES = (
    (label = "cmfmc_post (GCHP-faithful, leaky)",
     path  = "/temp1/convection_compare_4tracer_postfix/convonly_cmfmc_geosit_c180_3day.nc",
     style = :dot),
    (label = "cmfmc_matrix (GEOS rates → TM5 LU)",
     path  = "/temp1/convection_compare_4tracer/convonly_cmfmc_matrix_geosit_c180_3day.nc",
     style = :dash),
    (label = "tm5_postclosure (TM5 ERA5 + closure)",
     path  = "/temp1/convection_compare_4tracer/convonly_tm5_postclosure_era5_c180_3day.nc",
     style = :solid),
)

const G_ACCEL = 9.80665  # m/s² — for air_mass_per_area → pressure thickness.

const OUT_DIR = joinpath(dirname(@__DIR__), "..", "plots",
                          "convection_per_panel_overlay") |> normpath

# Mass-weighted panel-mean profile at one time slice (panel-flat (Nij, Nz)).
function panel_mean_profile(tracer_slice::AbstractMatrix, air_mass_slice::AbstractMatrix)
    Nij, Nz = size(tracer_slice)
    out = zeros(Float64, Nz)
    @inbounds for k in 1:Nz
        num = 0.0
        den = 0.0
        @inbounds for ij in 1:Nij
            tv = tracer_slice[ij, k]
            mv = air_mass_slice[ij, k]
            (tv === missing || mv === missing) && continue
            num += Float64(tv) * Float64(mv)
            den += Float64(mv)
        end
        out[k] = den > 0 ? num / den : 0.0
    end
    return out
end

# Panel-mean layer-thickness in Pa from air_mass_per_area at t=0.
# Returns p_mid[k] (Pa), one entry per layer, with k=1=TOA, k=Nz=surface.
function panel_mean_pressure_mid(air_mass_per_area_panel_t0::AbstractMatrix)
    Nij, Nz = size(air_mass_per_area_panel_t0)
    # Average per layer to get a single panel-mean column thickness.
    Δp = zeros(Float64, Nz)
    @inbounds for k in 1:Nz
        s = 0.0
        n = 0
        @inbounds for ij in 1:Nij
            v = air_mass_per_area_panel_t0[ij, k]
            v === missing && continue
            s += Float64(v) * G_ACCEL
            n += 1
        end
        Δp[k] = n > 0 ? s / n : 0.0
    end
    # Interface pressures: integrate top-down.
    p_iface = zeros(Float64, Nz + 1)
    p_iface[1] = 0.0
    @inbounds for k in 1:Nz
        p_iface[k + 1] = p_iface[k] + Δp[k]
    end
    p_mid = (p_iface[1:Nz] .+ p_iface[2:Nz+1]) ./ 2
    return p_mid
end

function nearest_time_idx(time_hours::Vector{<:Real}, target_h::Real)
    return argmin(abs.(time_hours .- target_h))
end

# Time-axis handling — DateTime or numeric hours. Mirrors the helper in
# `plot_convection_profiles_per_panel_percentiles.jl`; some output
# writers store numeric time rather than DateTime, and the earlier
# `Dates.Millisecond(t - first(raw_times))` path failed on numeric input.
function _time_hours(time_var)
    raw = collect(time_var)
    if eltype(raw) <: Dates.AbstractDateTime
        t0 = first(raw)
        return [Float64(Dates.value(Dates.Millisecond(t - t0))) / 3_600_000
                for t in raw]
    else
        t0 = Float64(first(raw))
        return Float64.(raw) .- t0
    end
end

function load_one(path::AbstractString)
    NCDataset(path, "r") do ds
        time_hours = _time_hours(ds["time"])
        Nz   = ds.dim["lev"]
        npan = ds.dim["nf"]
        Nx   = ds.dim["Xdim"]
        Ny   = ds.dim["Ydim"]
        snap_idx   = [nearest_time_idx(time_hours, t) for t in SNAPSHOT_HOURS]
        snap_hours = time_hours[snap_idx]
        air = ds["air_mass"]                # (Nx, Ny, nf, lev, time)
        ampa = ds["air_mass_per_area"]      # (Nx, Ny, nf, lev, time)
        # Per-panel pressure axis (t=0 panel-mean column).
        p_mid = Dict{Int, Vector{Float64}}()
        for p in 1:npan
            ampa_slice = reshape(ampa[:, :, p, :, 1], Nx * Ny, Nz)
            p_mid[p] = panel_mean_pressure_mid(ampa_slice)
        end
        # Tracer profiles per (tracer, panel, snapshot).
        results = Dict{Tuple{String, Int, Int}, Vector{Float64}}()
        for tracer in TRACERS
            haskey(ds, tracer) || continue
            arr = ds[tracer]
            for (ti, t_idx) in enumerate(snap_idx)
                for p in 1:npan
                    tslice = reshape(arr[:, :, p, :, t_idx], Nx * Ny, Nz)
                    mslice = reshape(air[:, :, p, :, t_idx], Nx * Ny, Nz)
                    results[(tracer, p, ti)] = panel_mean_profile(tslice, mslice)
                end
            end
        end
        return (; time_hours, snap_idx, snap_hours, Nz, npan, p_mid, results)
    end
end

function plot_one(tracer::AbstractString, loaded::AbstractVector, snap_hours, out_path)
    fig = Figure(size = (1380, 820))
    Label(fig[0, 1:3],
          "$(tracer): mass-weighted panel-mean VMR — solid=TM5 post-closure, " *
          "dashed=CMFMC matrix, dotted=CMFMC post (GCHP-faithful)";
          fontsize = 16, halign = :center)
    # Time-step color ramp (4 timesteps); apply to all three schemes.
    cmap = cgrad(:viridis, length(snap_hours); categorical = true)
    # Common x-range across the figure: pull from the union of all
    # scheme/panel/timestep profile maxima for this tracer.
    max_vmr = 0.0
    for scheme_idx in eachindex(loaded), p in 1:loaded[scheme_idx].npan,
        ti in eachindex(snap_hours)
        prof = get(loaded[scheme_idx].results, (tracer, p, ti), nothing)
        prof === nothing && continue
        max_vmr = max(max_vmr, maximum(prof))
    end
    xhi = max_vmr * 1.5
    xlo = max(xhi * 1e-6, 1e-30)

    # Common y-range (pressure): use the deepest (TM5 ERA5) column's
    # surface pressure so all three schemes fit. Min y = TOA (some
    # small Pa), max y = max surface pressure across panels & schemes.
    ymax = 0.0
    for L in loaded, p in 1:L.npan
        ymax = max(ymax, last(L.p_mid[p]))
    end
    # Convert to hPa for display.
    yhi_hpa = ymax / 100.0
    ylo_hpa = 0.1   # ~0.1 hPa near TOA

    for p in 1:loaded[1].npan
        row = (p - 1) ÷ 3 + 1
        col = (p - 1) % 3 + 1
        # Drop `yreversed = true` (its behavior across CairoMakie
        # versions was inconsistent for this user's render) and instead
        # rely on explicit reversed `ylims!(ax, yhi, ylo)` below.
        # Surface (high pressure) renders at the bottom either way.
        ax = Axis(fig[row, col];
                  title  = "panel $p",
                  xlabel = "VMR  (log)",
                  ylabel = (col == 1 ? "pressure (hPa)" : ""),
                  xscale = log10,
                  xtickformat = vs -> [@sprintf("%.0e", v) for v in vs])
        ax.xticklabelrotation = π / 6
        ax.xticklabelsize = 9
        for L_idx in eachindex(loaded)
            L = loaded[L_idx]
            scheme_style = SCHEMES[L_idx].style
            # Compensate for line-style visibility — Makie's `:dot` is
            # near-invisible at default thickness; bump it. Solid and
            # dashed render fine at standard width.
            lw = scheme_style === :dot   ? 2.6 :
                 scheme_style === :dash  ? 2.0 : 1.8
            p_axis = L.p_mid[p] ./ 100.0   # Pa → hPa
            for (ti, _hour) in enumerate(snap_hours)
                prof = get(L.results, (tracer, p, ti), nothing)
                prof === nothing && continue
                prof_safe = [max(v, xlo) for v in prof]
                lines!(ax, prof_safe, p_axis;
                       color = cmap[ti], linewidth = lw,
                       linestyle = scheme_style)
            end
        end
        xlims!(ax, xlo, xhi)
        # Explicit reversed ylims (high first) — combined with
        # `yreversed = true` above, the axis is unambiguously flipped:
        # surface pressure (~1000 hPa) at the bottom, TOA (~0.1 hPa) at
        # the top.
        ylims!(ax, yhi_hpa, ylo_hpa)
    end

    # Custom legend: line styles for schemes; colors for timesteps.
    legends = Tuple{LineElement, String}[]
    for L_idx in eachindex(loaded)
        lw = SCHEMES[L_idx].style === :dot   ? 2.6 :
             SCHEMES[L_idx].style === :dash  ? 2.0 : 1.8
        push!(legends,
              (LineElement(linestyle = SCHEMES[L_idx].style,
                            linewidth = lw, color = :black),
               SCHEMES[L_idx].label))
    end
    for (ti, hour) in enumerate(snap_hours)
        push!(legends,
              (LineElement(linestyle = :solid, linewidth = 2.0, color = cmap[ti]),
               @sprintf("t = %4.1f h", hour)))
    end
    Legend(fig[1:2, 4], [l for (l, _) in legends], [s for (_, s) in legends];
           framevisible = true, labelsize = 10, padding = (4, 4, 4, 4))

    # PDF preserves vector fidelity (zoomable in VSCode preview); also
    # mirror to a PNG sidecar for quick scroll-through.
    save(out_path, fig)
    save(replace(out_path, ".pdf" => ".png"), fig; px_per_unit = 1.5)
    return out_path
end

function main()
    isdir(OUT_DIR) || mkpath(OUT_DIR)
    loaded = NamedTuple[]
    for scheme in SCHEMES
        if !isfile(scheme.path)
            @warn "skip (not found): $(scheme.path)"
            continue
        end
        println("loading $(scheme.label) ...")
        push!(loaded, load_one(scheme.path))
    end
    isempty(loaded) && error("No scheme NetCDFs found.")
    for tracer in TRACERS
        out = joinpath(OUT_DIR, "overlay__$(tracer).pdf")
        plot_one(tracer, loaded, SNAPSHOT_HOURS, out)
        println("  wrote $out")
    end
end

main()
