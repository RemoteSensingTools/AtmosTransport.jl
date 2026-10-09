#!/usr/bin/env julia
# Per-panel mass-weighted mean vertical profile of each tracer at selected
# timestamps, one PNG per (scheme, tracer). Reads the convection-comparison
# NetCDFs in /temp1/convection_compare_4tracer.
#
# Usage:
#   julia --project=. scripts/completed_experiments/plot_convection_profiles_per_panel.jl
#
# Outputs to /temp1/convection_compare_4tracer/plots_per_panel/.

using CairoMakie
using NCDatasets
using Printf
using Statistics
using Dates

const TRACERS = ("co2_lowest_layer", "co2_p08", "co2_p06", "co2_p04")

const SNAPSHOT_HOURS = (0.0, 6.0, 12.0, 24.0, 48.0, 72.0)

const SCHEMES = (
    (label = "cmfmc_post",       path = "/temp1/convection_compare_4tracer_postfix/convonly_cmfmc_geosit_c180_3day.nc"),
    (label = "cmfmc_matrix",     path = "/temp1/convection_compare_4tracer/convonly_cmfmc_matrix_geosit_c180_3day.nc"),
    (label = "tm5_pre",          path = "/temp1/convection_compare_4tracer_postfix/convonly_tm5_era5_c180_3day.nc"),
    (label = "tm5_postclosure",  path = "/temp1/convection_compare_4tracer/convonly_tm5_postclosure_era5_c180_3day.nc"),
)

# Write into the repo's untracked `plots/` folder so VSCode can preview
# the PNGs without a `/temp1` round-trip. `plots/` is in `.gitignore`.
const OUT_DIR = joinpath(dirname(@__DIR__), "..", "plots", "convection_per_panel") |>
                normpath

# Mass-weighted panel-mean profile at one time slice:
#   profile[k] = Σ_xy (tracer[x,y,panel,k,t] * air_mass[x,y,panel,k,t])
#                / Σ_xy air_mass[x,y,panel,k,t]
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
            t = Float64(tv)
            m = Float64(mv)
            num += t * m
            den += m
        end
        out[k] = den > 0 ? num / den : 0.0
    end
    return out
end

# Find the time index in `time_vec` (hours since start) closest to `target_h`.
function nearest_time_idx(time_hours::Vector{<:Real}, target_h::Real)
    return argmin(abs.(time_hours .- target_h))
end

function load_one(path::AbstractString)
    NCDataset(path, "r") do ds
        time_var = ds["time"]
        # `time` is encoded as DateTime in the NetCDF — convert to
        # hours-since-first-snapshot for our snapshot selection.
        raw_times = collect(time_var)
        time_hours = [Float64(Dates.value(Dates.Millisecond(t - first(raw_times)))) / 3_600_000
                      for t in raw_times]
        Nz   = ds.dim["lev"]
        npan = ds.dim["nf"]
        Nx   = ds.dim["Xdim"]
        Ny   = ds.dim["Ydim"]
        Nt   = ds.dim["time"]
        # Pre-pull air_mass into memory (one snapshot per requested
        # timestep). Tracer arrays are 4.6 GB each, so read lazily.
        snap_idx = [nearest_time_idx(time_hours, t) for t in SNAPSHOT_HOURS]
        snap_hours = time_hours[snap_idx]
        air = ds["air_mass"]  # (Nx, Ny, nf, lev, time)
        # Build flat (Nx*Ny, Nz) views per (panel, time) via reshape on
        # the device-loaded slice. For each scheme this is ~6 panels × 6
        # snapshots × 180*180*64 Float32 ≈ 80 MB per tracer — fine.
        results = Dict{Tuple{String, Int, Int}, Vector{Float64}}()
        for tracer in TRACERS
            haskey(ds, tracer) || continue
            arr = ds[tracer]   # (Nx, Ny, nf, lev, time)
            for (ti, t_idx) in enumerate(snap_idx)
                for p in 1:npan
                    tracer_slice = reshape(arr[:, :, p, :, t_idx],
                                            Nx * Ny, Nz)
                    air_slice    = reshape(air[:, :, p, :, t_idx],
                                            Nx * Ny, Nz)
                    results[(tracer, p, ti)] = panel_mean_profile(tracer_slice,
                                                                   air_slice)
                end
            end
        end
        return (; time_hours, snap_idx, snap_hours, Nz, npan,
                  results)
    end
end

# ─── Plot routine: one figure per (scheme, tracer), 2x3 panel layout ───
function plot_one(scheme_label, tracer, snap_hours, Nz, npan, results, out_path)
    fig = Figure(size = (1280, 720))
    Label(fig[0, 1:3],
          "$(scheme_label) — $(tracer): mass-weighted panel-mean VMR";
          fontsize = 18, halign = :center)

    # Time color ramp: 0h = light, 72h = dark.
    cmap = cgrad(:viridis, length(snap_hours); categorical = true)

    # Establish a common x-range across all panels of this figure so
    # subplots are visually comparable. Use the largest finite VMR across
    # snapshots & panels for the upper bound; clip the lower to 6 orders
    # of magnitude below (log scale handles the rest).
    max_vmr = 0.0
    for p in 1:npan, ti in eachindex(snap_hours)
        prof = get(results, (tracer, p, ti), nothing)
        prof === nothing && continue
        max_vmr = max(max_vmr, maximum(prof))
    end
    xhi = max_vmr * 1.5
    xlo = max(xhi * 1e-6, 1e-30)

    for p in 1:npan
        row = (p - 1) ÷ 3 + 1
        col = (p - 1) % 3 + 1
        ax = Axis(fig[row, col];
                  title  = "panel $p",
                  xlabel = "VMR  (log)",
                  ylabel = (col == 1 ? "level k (k=1 TOA, k=$Nz surface)" : ""),
                  yreversed = true,
                  xscale = log10,
                  xtickformat = vs -> [@sprintf("%.0e", v) for v in vs])
        ax.xticklabelrotation = π / 6
        ax.xticklabelsize = 9
        for (ti, hour) in enumerate(snap_hours)
            prof = get(results, (tracer, p, ti), nothing)
            prof === nothing && continue
            ks = 1:Nz
            # Clip zeros / negatives to a tiny positive for log plotting
            prof_safe = [max(v, xlo) for v in prof]
            lines!(ax, prof_safe, collect(ks);
                   color = cmap[ti], linewidth = 1.8,
                   label = @sprintf("t = %4.1f h", hour))
        end
        xlims!(ax, xlo, xhi)
        if p == 1
            axislegend(ax; position = :rb, framevisible = true,
                       labelsize = 9, padding = (4, 4, 4, 4))
        end
    end

    save(out_path, fig; px_per_unit = 1.5)
    return out_path
end

function main()
    isdir(OUT_DIR) || mkpath(OUT_DIR)
    for scheme in SCHEMES
        isfile(scheme.path) || (println("skip (not found): $(scheme.path)"); continue)
        println("loading $(scheme.label) ...")
        data = load_one(scheme.path)
        for tracer in TRACERS
            out = joinpath(OUT_DIR, "$(scheme.label)__$(tracer).png")
            plot_one(scheme.label, tracer, data.snap_hours, data.Nz,
                     data.npan, data.results, out)
            println("  wrote $out")
        end
    end
    return nothing
end

main()
