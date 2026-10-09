#!/usr/bin/env julia
# ===========================================================================
# Diagnose vertical redistribution of a BL-enhanced tracer across the
# 5° / 2.5° / 1° advection-only resolution ladder.
#
# Reads the three snapshot NetCDFs produced by the advresln configs,
# partitions tracer mass into BL (below 850 hPa), free troposphere
# (850-500 hPa), and upper troposphere (above 500 hPa) using each
# snapshot's stored hybrid pressure half-levels, normalizes by the total
# tracer mass perturbation (bl_enhanced excess at t=0), and prints the
# three time series per resolution.
#
# Usage:
#   julia --project=. scripts/completed_experiments/advresln_vertical_redistribution.jl
# ===========================================================================

using NCDatasets
using Statistics
using Printf

const SNAPSHOT_FILES = [
    ("5°  (72×37)",  expanduser("~/data/AtmosTransport/output/advresln/ll72x37_advonly.nc")),
    ("2.5° (144×73)", expanduser("~/data/AtmosTransport/output/advresln/ll144x73_advonly.nc")),
    ("1°  (360×181)", expanduser("~/data/AtmosTransport/output/advresln/ll360x181_advonly.nc")),
]

const P_BL_FT_BOUNDARY_HPA = 850.0
const P_FT_UT_BOUNDARY_HPA = 500.0
const TRACER_NAME = "co2_bl"
const BACKGROUND = 4.0e-4
const ENHANCEMENT = 1.0e-4

# Layer pressure midpoints from the binary's stored hybrid coefficients
# and per-cell ps. Returns p_mid[lon, lat, lev] in Pa.
function layer_pressure_midpoints(ds, ps_var_name::String)
    A = ds["A"][:]   # interface coefficients (Pa); length Nz+1
    B = ds["B"][:]   # interface coefficients (-);  length Nz+1
    ps = Array(ds[ps_var_name][:, :, 1])  # use t=0 ps; (lon, lat)
    Nlon, Nlat = size(ps)
    Nz = length(A) - 1
    p_mid = Array{Float64}(undef, Nlon, Nlat, Nz)
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        p_top = A[k]   + B[k]   * ps[i, j]
        p_bot = A[k+1] + B[k+1] * ps[i, j]
        p_mid[i, j, k] = 0.5 * (p_top + p_bot)
    end
    return p_mid
end

function partition_mass_fractions(snapshot_path::String)
    isfile(snapshot_path) || error("snapshot not found: $snapshot_path")
    ds = NCDataset(snapshot_path, "r")
    try
        # Detect ps variable name (could be `ps`, `surface_pressure`, ...)
        ps_var = "ps" in keys(ds) ? "ps" :
                 "surface_pressure" in keys(ds) ? "surface_pressure" :
                 error("could not find ps in $snapshot_path; vars: $(keys(ds))")

        col_mass_var = "$(TRACER_NAME)_column_mass_per_area"
        haskey(ds, col_mass_var) ||
            error("missing $col_mass_var in $snapshot_path; vars: $(keys(ds))")

        # Time series of layered tracer mass per area: (lon, lat, lev, time)
        air_per_area = ds["air_mass_per_area"][:, :, :, :]
        tracer_mr    = ds[TRACER_NAME][:, :, :, :]   # mixing ratio (lev still TOA→surf)
        time_hours   = ds["time"][:]

        Nlon, Nlat, Nz, Nt = size(tracer_mr)
        p_mid_t0 = layer_pressure_midpoints(ds, ps_var)  # use t=0 reference pressures

        # Tracer mass per area per layer = tracer_mr × air_per_area (kg/m²)
        # Subtract background to get the excess perturbation (BL-enhanced - background)
        # so that fractions sum to 1.0 over the perturbation only.
        bl_mass = zeros(Float64, Nt)
        ft_mass = zeros(Float64, Nt)
        ut_mass = zeros(Float64, Nt)
        @inbounds for t in 1:Nt, k in 1:Nz, j in 1:Nlat, i in 1:Nlon
            excess_mr = Float64(tracer_mr[i, j, k, t]) - BACKGROUND
            mass = excess_mr * Float64(air_per_area[i, j, k, t])
            p_hpa = p_mid_t0[i, j, k] / 100.0
            if p_hpa >= P_BL_FT_BOUNDARY_HPA
                bl_mass[t] += mass
            elseif p_hpa >= P_FT_UT_BOUNDARY_HPA
                ft_mass[t] += mass
            else
                ut_mass[t] += mass
            end
        end

        total = bl_mass[1] + ft_mass[1] + ut_mass[1]   # initial perturbation budget
        return time_hours, bl_mass ./ total, ft_mass ./ total, ut_mass ./ total
    finally
        close(ds)
    end
end

function main()
    println("Advection-only vertical redistribution of BL excess")
    println("BL: p ≥ $(P_BL_FT_BOUNDARY_HPA) hPa, FT: $(P_FT_UT_BOUNDARY_HPA)-$(P_BL_FT_BOUNDARY_HPA) hPa, UT: p < $(P_FT_UT_BOUNDARY_HPA) hPa")
    println("Fractions normalized by initial perturbation total mass.")
    println()
    for (label, path) in SNAPSHOT_FILES
        println("=== $label ===")
        if !isfile(path)
            println("  (snapshot not on disk yet: $path)")
            println()
            continue
        end
        t, fbl, fft, fut = partition_mass_fractions(path)
        @printf("  %-6s  %-12s  %-12s  %-12s\n", "t(h)", "BL frac", "FT frac", "UT frac")
        for i in eachindex(t)
            @printf("  %-6.1f  %-12.6f  %-12.6f  %-12.6f\n",
                    Float64(t[i]), fbl[i], fft[i], fut[i])
        end
        println()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
