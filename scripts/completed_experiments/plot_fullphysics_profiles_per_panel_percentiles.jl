#!/usr/bin/env julia
# Per-panel 10/50/90 percentile vertical profiles at t = 24 h for the
# full-physics experiment: PPM advection + diffusion + convection on
# GEOS-IT and ERA5 binaries. Mirror of the advection / convection /
# diffusion percentile scripts; same conventions.

using CairoMakie
using NCDatasets
using Printf
using Statistics
using Dates

const TRACERS = ("co2_lowest_layer", "co2_p08", "co2_p06", "co2_p04")
const TARGET_HOUR = 24.0

const SCHEMES = (
    (label = "Full physics · GEOS-IT C180/L72 (PPM + VDIFF + CMFMC-matrix)",
     path  = "/temp1/fullphysics_compare_4tracer/fullphysics_geosit_c180_3day.nc",
     style = :dot),
    (label = "Full physics · ERA5 C180/L85 (PPM + Beljaars-Viterbo + TM5)",
     path  = "/temp1/fullphysics_compare_4tracer/fullphysics_era5_c180_3day.nc",
     style = :solid),
)

const PERCENTILES = (0.10, 0.50, 0.90)
const PCT_LABELS  = ("10th", "median (50th)", "90th")

const G_ACCEL = 9.80665
const OUT_DIR = joinpath(dirname(@__DIR__), "..", "plots",
                          "fullphysics_per_panel_percentiles") |> normpath

function panel_mean_pressure_mid(air_mass_per_area_panel_t0::AbstractMatrix)
    Nij, Nz = size(air_mass_per_area_panel_t0)
    Δp = zeros(Float64, Nz)
    @inbounds for k in 1:Nz
        s = 0.0; n = 0
        @inbounds for ij in 1:Nij
            v = air_mass_per_area_panel_t0[ij, k]
            v === missing && continue
            s += Float64(v) * G_ACCEL
            n += 1
        end
        Δp[k] = n > 0 ? s / n : 0.0
    end
    p_iface = zeros(Float64, Nz + 1)
    @inbounds for k in 1:Nz
        p_iface[k + 1] = p_iface[k] + Δp[k]
    end
    return (p_iface[1:Nz] .+ p_iface[2:Nz + 1]) ./ 2
end

function panel_percentile_profile(tracer_slice::AbstractMatrix, percentiles)
    Nij, Nz = size(tracer_slice)
    out = zeros(Float64, length(percentiles), Nz)
    buf = Vector{Float64}(undef, Nij)
    @inbounds for k in 1:Nz
        n = 0
        @inbounds for ij in 1:Nij
            v = tracer_slice[ij, k]
            v === missing && continue
            n += 1
            buf[n] = Float64(v)
        end
        if n == 0
            for q in eachindex(percentiles); out[q, k] = 0.0; end
            continue
        end
        view_n = @view buf[1:n]
        sort!(view_n)
        for (q, p) in pairs(percentiles)
            idx = clamp(p * (n - 1) + 1, 1.0, Float64(n))
            lo  = floor(Int, idx)
            hi  = ceil(Int, idx)
            frac = idx - lo
            out[q, k] = view_n[lo] * (1 - frac) + view_n[hi] * frac
        end
    end
    return out
end

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

nearest_time_idx(time_hours::Vector{<:Real}, target_h::Real) =
    argmin(abs.(time_hours .- target_h))

function load_one(path::AbstractString)
    NCDataset(path, "r") do ds
        time_hours = _time_hours(ds["time"])
        Nz   = ds.dim["lev"]
        npan = ds.dim["nf"]
        Nx   = ds.dim["Xdim"]
        Ny   = ds.dim["Ydim"]
        t_idx = nearest_time_idx(time_hours, TARGET_HOUR)
        actual_h = time_hours[t_idx]
        ampa = ds["air_mass_per_area"]
        p_mid = Dict{Int, Vector{Float64}}()
        for p in 1:npan
            ampa_slice = reshape(ampa[:, :, p, :, 1], Nx * Ny, Nz)
            p_mid[p] = panel_mean_pressure_mid(ampa_slice)
        end
        percentiles = Dict{Tuple{String, Int}, Matrix{Float64}}()
        for tracer in TRACERS
            haskey(ds, tracer) || continue
            arr = ds[tracer]
            for p in 1:npan
                tslice = reshape(arr[:, :, p, :, t_idx], Nx * Ny, Nz)
                percentiles[(tracer, p)] = panel_percentile_profile(tslice, PERCENTILES)
            end
        end
        return (; actual_h, Nz, npan, p_mid, percentiles)
    end
end

function plot_one(tracer::AbstractString, loaded::AbstractVector, out_path)
    fig = Figure(size = (1380, 820))
    actual_h = loaded[1].actual_h
    header = string(tracer, " — per-cell VMR percentiles at t ≈ ",
                    @sprintf("%.1f", actual_h), " h ",
                    "(full physics: PPM + diffusion + convection; ",
                    "solid=ERA5 C180/L85, dotted=GEOS-IT C180/L72)")
    Label(fig[0, 1:3], header; fontsize = 14, halign = :center)

    cmap = cgrad(:plasma, length(PERCENTILES); categorical = true)

    max_vmr = 0.0
    for L in loaded, p in 1:L.npan
        mat = get(L.percentiles, (tracer, p), nothing)
        mat === nothing && continue
        max_vmr = max(max_vmr, maximum(mat))
    end
    xhi = max_vmr * 1.5
    xlo = max(xhi * 1e-6, 1e-30)
    ymax = 0.0
    for L in loaded, p in 1:L.npan
        ymax = max(ymax, last(L.p_mid[p]))
    end
    yhi_hpa = ymax / 100.0
    ylo_hpa = 0.1

    for p in 1:loaded[1].npan
        row = (p - 1) ÷ 3 + 1
        col = (p - 1) % 3 + 1
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
            scheme_style = L.scheme.style
            lw = scheme_style === :dot   ? 2.6 :
                 scheme_style === :dash  ? 2.0 : 1.8
            p_axis = L.p_mid[p] ./ 100.0
            mat = get(L.percentiles, (tracer, p), nothing)
            mat === nothing && continue
            for q in eachindex(PERCENTILES)
                prof = mat[q, :]
                prof_safe = [max(v, xlo) for v in prof]
                lines!(ax, prof_safe, p_axis;
                       color = cmap[q], linewidth = lw,
                       linestyle = scheme_style)
            end
        end
        xlims!(ax, xlo, xhi)
        ylims!(ax, yhi_hpa, ylo_hpa)
    end

    legends = Tuple{LineElement, String}[]
    for L_idx in eachindex(loaded)
        lw = loaded[L_idx].scheme.style === :dot   ? 2.6 :
             loaded[L_idx].scheme.style === :dash  ? 2.0 : 1.8
        push!(legends,
              (LineElement(linestyle = loaded[L_idx].scheme.style, linewidth = lw, color = :black),
               loaded[L_idx].scheme.label))
    end
    for q in eachindex(PERCENTILES)
        push!(legends,
              (LineElement(linestyle = :solid, linewidth = 2.0, color = cmap[q]),
               PCT_LABELS[q]))
    end
    Legend(fig[1:2, 4], [l for (l, _) in legends], [s for (_, s) in legends];
           framevisible = true, labelsize = 10, padding = (4, 4, 4, 4))

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
        push!(loaded, (; load_one(scheme.path)..., scheme = scheme))
    end
    isempty(loaded) && error("No scheme NetCDFs found.")
    for tracer in TRACERS
        out = joinpath(OUT_DIR, "percentiles_t24h__$(tracer).pdf")
        plot_one(tracer, loaded, out)
        println("  wrote $out")
    end
end

main()
