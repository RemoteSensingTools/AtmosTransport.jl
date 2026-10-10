#!/usr/bin/env julia
# ===========================================================================
# Test 1 (gating) — Panel-convention lat/lon validation against raw GEOS-IT.
#
# Compares per-cell longitudes and latitudes computed by
# `CubedSphereMesh(Nc=180, convention=GEOSNativePanelConvention())` against
# the `lons`, `lats` (and `corner_lons`, `corner_lats` if present) arrays
# stored inside an actual GEOS-IT C180 NetCDF file.
#
# BOE for acceptance: lats/lons are computed analytically from the GMAO
# equal-distance gnomonic edge law + GEOS `cell_center2` center law + a -10°
# z-rotation. Native files store center coordinates as F32. Away from polar
# singularities this gives O(1e-5°) noise; at the polar panel center,
# longitude is ill-conditioned, so the default longitude tolerance is looser
# than the latitude tolerance. Both are far below any panel rotation error
# (O(1°) disagreement at minimum).
#
# On failure: per-panel pattern hints at the failure mode:
#   - All panels uniform offset → -10° rotation amount/sign
#   - Panel 3 only → north-pole 90° rotation
#   - Panels 4 + 5 only → Y-axis flip
#   - Random panel → panel ordering map
# ===========================================================================

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention,
    panel_cell_center_lonlat, panel_cell_corner_lonlat

using NCDatasets

const USAGE = """
Usage: julia --project=. scripts/validation/heritage/validate_geos_native_lonlat.jl \\
           --geosit-file <GEOSIT.YYYYMMDD.A3dyn.C180.nc> \\
           [--Nc 180] [--lon-threshold 5e-3] [--lat-threshold 5e-4] \\
           [--report <out.csv>]
"""

function parse_args(argv::Vector{String})
    args = Dict{String, String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--geosit-file" || a == "--Nc" || a == "--lon-threshold" ||
           a == "--lat-threshold" || a == "--report"
            i + 1 <= length(argv) || error("Missing value for $a")
            args[a] = argv[i + 1]
            i += 2
        elseif a == "-h" || a == "--help"
            println(USAGE); exit(0)
        else
            error("Unknown argument: $a\n$USAGE")
        end
    end
    haskey(args, "--geosit-file") || error("Missing --geosit-file\n$USAGE")
    return args
end

# Difference modulo 360° (signed shortest-arc distance).
@inline lon_diff(a, b) = ((a - b + 540) % 360) - 180

function main(argv = ARGS)
    args = parse_args(Vector{String}(argv))
    geosit_path = args["--geosit-file"]
    Nc = parse(Int, get(args, "--Nc", "180"))
    lon_threshold = parse(Float64, get(args, "--lon-threshold", "5e-3"))
    lat_threshold = parse(Float64, get(args, "--lat-threshold", "5e-4"))
    report_path = get(args, "--report", "")

    @info "Test 1: GEOS-native panel lat/lon validation"
        @info "  GEOS-IT file: $geosit_path"
        @info "  Nc = $Nc"
        @info "  thresholds: |Δlon| < $lon_threshold °, |Δlat| < $lat_threshold °"

    mesh = CubedSphereMesh(; Nc = Nc, Hp = 0, FT = Float64,
                          convention = GEOSNativePanelConvention())

    NCDataset(geosit_path, "r") do ds
        @info "  GEOS-IT dims: $(Tuple(d => length(ds.dim[d]) for d in keys(ds.dim)))"
        haskey(ds, "lons") || error("GEOS-IT file lacks `lons` variable")
        haskey(ds, "lats") || error("GEOS-IT file lacks `lats` variable")

        lons_var = ds["lons"]
        lats_var = ds["lats"]
        sz_lon = size(lons_var)
        sz_lat = size(lats_var)
        @info "  lons size = $sz_lon (dtype $(eltype(lons_var)))   units = $(get(lons_var.attrib, "units", "(none)"))"
        @info "  lats size = $sz_lat (dtype $(eltype(lats_var)))   units = $(get(lats_var.attrib, "units", "(none)"))"

        # Expect (Xdim, Ydim, nf) = (Nc, Nc, 6)
        sz_lon == (Nc, Nc, 6) ||
            error("lons size $sz_lon ≠ ($Nc, $Nc, 6); abort")
        sz_lat == (Nc, Nc, 6) ||
            error("lats size $sz_lat ≠ ($Nc, $Nc, 6); abort")

        lons_geos = Float64.(Array(lons_var[:, :, :]))
        lats_geos = Float64.(Array(lats_var[:, :, :]))

        # ----- Per-panel comparison -----
        per_panel = Tuple{Int, Float64, Float64, Float64, Float64,
                           Tuple{Int, Int}, Tuple{Int, Int}}[]
        @info ""
        println("panel | max|Δlon| (°) | max|Δlat| (°) | mean|Δlon|     | mean|Δlat|     | worst lon (i,j) | worst lat (i,j)")
        println("------+---------------+---------------+----------------+----------------+----------------+----------------")
        max_lon_overall = 0.0
        max_lat_overall = 0.0
        for p in 1:6
            lon_ours, lat_ours = panel_cell_center_lonlat(mesh, p)
            # Wrap our lon to [-180, 180) to match GEOS-IT convention if needed
            # (we'll use signed shortest-arc difference so range doesn't matter).

            dlon = [lon_diff(lon_ours[i, j], lons_geos[i, j, p]) for i in 1:Nc, j in 1:Nc]
            dlat = [lat_ours[i, j] - lats_geos[i, j, p] for i in 1:Nc, j in 1:Nc]

            adlon = abs.(dlon)
            adlat = abs.(dlat)
            max_lon = maximum(adlon)
            max_lat = maximum(adlat)
            mean_lon = mean(adlon)
            mean_lat = mean(adlat)
            worst_lon_idx = Tuple(argmax(adlon))
            worst_lat_idx = Tuple(argmax(adlat))

            push!(per_panel, (p, max_lon, max_lat, mean_lon, mean_lat,
                              worst_lon_idx, worst_lat_idx))
            max_lon_overall = max(max_lon_overall, max_lon)
            max_lat_overall = max(max_lat_overall, max_lat)

            @printf("  %d   | %13.3e | %13.3e | %14.3e | %14.3e | (%3d,%3d)      | (%3d,%3d)\n",
                    p, max_lon, max_lat, mean_lon, mean_lat,
                    worst_lon_idx..., worst_lat_idx...)
        end

        # ----- Corners (if present) -----
        if haskey(ds, "corner_lons") && haskey(ds, "corner_lats")
            @info ""
            @info "Corner check (corner_lons, corner_lats present):"
            corn_lon_var = ds["corner_lons"]
            corn_lat_var = ds["corner_lats"]
            sz_cl = size(corn_lon_var)
            @info "  corner_lons size = $sz_cl"
            if sz_cl == (Nc + 1, Nc + 1, 6)
                clon = Float64.(Array(corn_lon_var[:, :, :]))
                clat = Float64.(Array(corn_lat_var[:, :, :]))
                println("panel | corner max|Δlon| | corner max|Δlat|")
                for p in 1:6
                    lon_ours, lat_ours = panel_cell_corner_lonlat(mesh, p)
                    @assert size(lon_ours) == (Nc + 1, Nc + 1)
                    dlon = [lon_diff(lon_ours[i, j], clon[i, j, p])
                            for i in 1:(Nc + 1), j in 1:(Nc + 1)]
                    dlat = [lat_ours[i, j] - clat[i, j, p]
                            for i in 1:(Nc + 1), j in 1:(Nc + 1)]
                    @printf("  %d   | %15.3e | %15.3e\n",
                            p, maximum(abs.(dlon)), maximum(abs.(dlat)))
                end
            else
                @info "  corner array shape unexpected: $sz_cl, skipping"
            end
        else
            @info ""
            @info "(No corner_lons/corner_lats variables in this NetCDF; centers only.)"
        end

        # ----- CSV report -----
        if !isempty(report_path)
            mkpath(dirname(report_path))
            open(report_path, "w") do io
                println(io, "panel,max_abs_dlon_deg,max_abs_dlat_deg,mean_abs_dlon_deg,mean_abs_dlat_deg,worst_lon_i,worst_lon_j,worst_lat_i,worst_lat_j")
                for (p, ml, mt, am, at, wli, wlt) in per_panel
                    @printf(io, "%d,%.6e,%.6e,%.6e,%.6e,%d,%d,%d,%d\n",
                            p, ml, mt, am, at, wli..., wlt...)
                end
            end
            @info "  Report written: $report_path"
        end

        # ----- Verdict -----
        @info ""
        @info @sprintf("Overall max |Δlon| = %.4e °  (threshold %.1e °)",
                       max_lon_overall, lon_threshold)
        @info @sprintf("Overall max |Δlat| = %.4e °  (threshold %.1e °)",
                       max_lat_overall, lat_threshold)
        if max_lon_overall < lon_threshold && max_lat_overall < lat_threshold
            @info "PASS — panel convention matches GEOS-IT NetCDF."
            return 0
        else
            @warn "FAIL — convention disagreement exceeds threshold."
            failing_panels = [t[1] for t in per_panel
                              if t[2] >= lon_threshold || t[3] >= lat_threshold]
            @warn "  Failing panels: $failing_panels"
            return 1
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
