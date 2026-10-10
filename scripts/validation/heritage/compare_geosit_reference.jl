#!/usr/bin/env julia

"""
Compare an ERA5-on-GEOS-native cubed-sphere transport binary against GEOS-IT
native C180 reference files.

The products do not need to be numerically identical: ERA5 and GEOS-IT are
different meteorological analyses. This validator is instead designed to catch
preprocessor mistakes that produce the wrong geometry, panel orientation, sign,
or flux scale:

  * exact binary/header/grid metadata checks;
  * surface-pressure correlation and bias against GEOS-IT CTM_A1 `PS`;
  * column mass-weighted vector wind comparison from face fluxes.

For the vector comparison, both datasets are converted to panel-local
cell-center winds and then rotated back to geographic east/north. The generated
binary uses its balanced face fluxes (`am`, `bm`) and dry layer masses. GEOS-IT
uses `CTM_A1` `MFXC`, `MFYC`, and `DELP`. GEOS archives `MFXC/MFYC` accumulated
over the FV3 dynamics step (`mass_flux_dt`, 450 s for GEOS-IT C180), so the
conversion to binary substep mass is `MFXC / mass_flux_dt * dt_factor / g`.
The script uses the same `geos_native_to_face_flux!` and
`recover_cs_cell_center_winds!` helpers used by the preprocessing path.
"""

using Dates
using LinearAlgebra
using Printf
using Statistics
using NCDatasets

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO_ROOT, "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.MetDrivers:
    TransportBinaryReader, load_window!, load_grid
using .AtmosTransport.Preprocessing:
    geos_native_to_face_flux!, recover_cs_cell_center_winds!,
    rotate_panel_to_geographic!

const GRAVITY = Float32(9.80665)
const DEFAULT_GEOS_ROOT =
    "/home/cfranken/data/AtmosTransport/met/geosit/C180/raw_catrine"

mutable struct CorrStats
    n::Int64
    sx::Float64
    sy::Float64
    sx2::Float64
    sy2::Float64
    sxy::Float64
end

CorrStats() = CorrStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)

@inline function push_corr!(s::CorrStats, x::Real, y::Real)
    xf = Float64(x)
    yf = Float64(y)
    isfinite(xf) && isfinite(yf) || return s
    s.n += 1
    s.sx += xf
    s.sy += yf
    s.sx2 += xf * xf
    s.sy2 += yf * yf
    s.sxy += xf * yf
    return s
end

function corr(s::CorrStats)
    s.n > 2 || return NaN
    n = Float64(s.n)
    vx = s.sx2 - s.sx * s.sx / n
    vy = s.sy2 - s.sy * s.sy / n
    denom = sqrt(max(vx, 0.0) * max(vy, 0.0))
    denom > 0 || return NaN
    return (s.sxy - s.sx * s.sy / n) / denom
end

function parse_args(args)
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            if key in ("help", "strict")
                push!(flags, key)
                i += 1
            else
                i == length(args) && error("Missing value for --$key")
                opts[key] = args[i + 1]
                i += 2
            end
        else
            error("Unexpected positional argument: $a")
        end
    end
    return opts, flags
end

function usage()
    println("""
    Usage:
      julia --project=. scripts/validation/heritage/compare_geosit_reference.jl \\
        --binary PATH --date YYYYMMDD [options]

    Options:
      --geos-root DIR          GEOS-IT daily root [$DEFAULT_GEOS_ROOT]
      --windows SPEC           Windows to compare, e.g. 1:24 or 1,2,3 [1:24]
      --speed-threshold MS     Ignore vector angles below this speed [2.0]
      --geos-mass-flux-dt S    FV3 dynamics step for MFXC/MFYC [450.0]
      --opposite-threshold F   Red-flag fraction for angle > 90 deg [0.02]
      --angle95-threshold DEG  Red-flag 95th-percentile vector angle [80]
      --speed-ratio-min X      Red-flag median generated/GEOS speed ratio [0.25]
      --speed-ratio-max X      Red-flag median generated/GEOS speed ratio [4.0]
      --ps-corr-threshold R    Red-flag surface-pressure correlation [0.95]
      --strict                 Exit nonzero if red-flag thresholds are exceeded
    """)
end

function parse_windows(spec::AbstractString, nwindow::Int)
    s = strip(spec)
    if occursin(':', s)
        parts = split(s, ':')
        length(parts) == 2 || error("Window range must be A:B, got $spec")
        a = parse(Int, parts[1])
        b = parse(Int, parts[2])
        return collect(a:b)
    end
    return [parse(Int, strip(x)) for x in split(s, ',') if !isempty(strip(x))]
end

function geos_ctm_path(root::AbstractString, date::AbstractString)
    path = joinpath(root, date, "GEOSIT.$date.CTM_A1.C180.nc")
    isfile(path) || error("Missing GEOS-IT CTM_A1 reference file: $path")
    return path
end

@inline function fill_dp_from_mass!(dp, m, areas, g::Float32)
    for p in 1:6
        dpp = dp[p]
        mp = m[p]
        @inbounds for k in axes(mp, 3), j in axes(mp, 2), i in axes(mp, 1)
            dpp[i, j, k] = mp[i, j, k] * g / Float32(areas[i, j])
        end
    end
    return dp
end

function panel_tuple_from_nc(v, w::Int, Nz::Int)
    return ntuple(p -> Array{Float32}(v[:, :, p, 1:Nz, w]), 6)
end

function panel_matrix_tuple_from_nc(v, w::Int)
    return ntuple(p -> Array{Float32}(v[:, :, p, w]), 6)
end

function weighted_column_mean!(u_col, v_col, u, v, weights)
    for p in 1:6
        uc = u_col[p]
        vc = v_col[p]
        up = u[p]
        vp = v[p]
        wp = weights[p]
        fill!(uc, 0.0f0)
        fill!(vc, 0.0f0)
        @inbounds for j in axes(uc, 2), i in axes(uc, 1)
            sw = 0.0f0
            su = 0.0f0
            sv = 0.0f0
            for k in axes(up, 3)
                wgt = max(wp[i, j, k], 0.0f0)
                sw += wgt
                su += up[i, j, k] * wgt
                sv += vp[i, j, k] * wgt
            end
            if sw > 0
                uc[i, j] = su / sw
                vc[i, j] = sv / sw
            end
        end
    end
    return nothing
end

function quantile_or_nan(x, q)
    isempty(x) && return NaN
    return quantile(x, q)
end

function main()
    opts, flags = parse_args(ARGS)
    if "help" in flags || !haskey(opts, "binary") || !haskey(opts, "date")
        usage()
        return "help" in flags ? 0 : 2
    end

    bin_path = expanduser(opts["binary"])
    date = opts["date"]
    root = expanduser(get(opts, "geos-root", DEFAULT_GEOS_ROOT))
    speed_threshold = parse(Float64, get(opts, "speed-threshold", "2.0"))
    geos_mass_flux_dt = parse(Float64, get(opts, "geos-mass-flux-dt", "450.0"))
    opposite_threshold = parse(Float64, get(opts, "opposite-threshold", "0.02"))
    angle95_threshold = parse(Float64, get(opts, "angle95-threshold", "80"))
    speed_ratio_min = parse(Float64, get(opts, "speed-ratio-min", "0.25"))
    speed_ratio_max = parse(Float64, get(opts, "speed-ratio-max", "4.0"))
    ps_corr_threshold = parse(Float64, get(opts, "ps-corr-threshold", "0.95"))
    strict = "strict" in flags

    reader = TransportBinaryReader(bin_path; FT=Float32)
    h = reader.header
    windows = parse_windows(get(opts, "windows", "1:$(h.nwindow)"), h.nwindow)
    all(1 .<= windows .<= h.nwindow) ||
        error("Requested windows $(windows) outside binary range 1:$(h.nwindow)")

    grid = load_grid(reader; FT=Float64, arch=CPU())
    mesh = grid.horizontal
    tangent_basis = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), 6)
    Nc = h.geometry.Nc
    Nz_gen = h.nlevel

    @assert h.geometry.panel_convention === :geos_native
    @assert h.geometry.definition === :gmao_equal_distance
    @assert h.geometry.coordinate_law === :gmao_equal_distance_gnomonic
    @assert h.geometry.center_law === :four_corner_normalized
    @assert h.geometry.longitude_offset_deg == -10.0

    geos_path = geos_ctm_path(root, date)
    ds = NCDataset(geos_path)
    try
        size(ds["MFXC"])[1:3] == (Nc, Nc, 6) ||
            error("GEOS MFXC grid does not match binary C$Nc")
        Nz_geos = size(ds["MFXC"], 4)

        # Exact grid-coordinate sanity: the point that exposed the previous
        # equiangular mismatch must match the binary mesh definition.
        lon, lat = panel_cell_center_lonlat(mesh, 1)
        geos_lon = Float64(ds["lons"][90, 140, 1])
        geos_lat = Float64(ds["lats"][90, 140, 1])
        @printf("Grid check p1(i=90,j=140): binary lon/lat %.9f %.9f, GEOS %.9f %.9f\n",
                lon[90, 140], lat[90, 140], geos_lon, geos_lat)

        gen_dp = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_gen), 6)
        gen_u_loc = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_gen), 6)
        gen_v_loc = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_gen), 6)
        gen_u_geo = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_gen), 6)
        gen_v_geo = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_gen), 6)
        gen_u_col = ntuple(_ -> zeros(Float32, Nc, Nc), 6)
        gen_v_col = ntuple(_ -> zeros(Float32, Nc, Nc), 6)

        geos_am = ntuple(_ -> Array{Float32}(undef, Nc + 1, Nc, Nz_geos), 6)
        geos_bm = ntuple(_ -> Array{Float32}(undef, Nc, Nc + 1, Nz_geos), 6)
        geos_u_loc = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_geos), 6)
        geos_v_loc = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_geos), 6)
        geos_u_geo = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_geos), 6)
        geos_v_geo = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz_geos), 6)
        geos_u_col = ntuple(_ -> zeros(Float32, Nc, Nc), 6)
        geos_v_col = ntuple(_ -> zeros(Float32, Nc, Nc), 6)

        u_corr = CorrStats()
        v_corr = CorrStats()
        speed_corr = CorrStats()
        ps_corr = CorrStats()
        ps_bias_sum = 0.0
        ps_abs_sum = 0.0
        ps_n = 0
        angles = Float32[]
        speed_ratios = Float32[]
        opposite = 0
        compared_vec = 0

        @printf("Comparing %d window(s), generated Nz=%d, GEOS Nz=%d, speed gate %.2f m/s\n",
                length(windows), Nz_gen, Nz_geos, speed_threshold)
        if length(unique(h.steps_per_window_by_window[windows])) == 1
            steps = h.steps_per_window_by_window[first(windows)]
            dt_factor = Float32(AtmosTransport.MetDrivers.flux_application_seconds(
                h.dt_met_seconds, steps, AtmosTransport.MetDrivers.flux_kind(reader)))
            geos_flux_scale = Float32(Float64(dt_factor) / geos_mass_flux_dt) / GRAVITY
            @printf("GEOS MFXC/MFYC scale: dt_factor %.1f s / mass_flux_dt %.1f s / g = %.6g\n",
                    Float64(dt_factor), geos_mass_flux_dt, geos_flux_scale)
        else
            @printf("GEOS MFXC/MFYC scale varies by window via steps_per_window_by_window.\n")
        end

        for w in windows
            steps = h.steps_per_window_by_window[w]
            dt_factor = Float32(AtmosTransport.MetDrivers.flux_application_seconds(
                h.dt_met_seconds, steps, AtmosTransport.MetDrivers.flux_kind(reader)))
            geos_flux_scale = Float32(Float64(dt_factor) / geos_mass_flux_dt) / GRAVITY
            win = load_window!(reader, w)
            fill_dp_from_mass!(gen_dp, win.m, mesh.cell_areas, GRAVITY)
            recover_cs_cell_center_winds!(gen_u_loc, gen_v_loc, win.am, win.bm,
                                          gen_dp, mesh.Δx, mesh.Δy,
                                          GRAVITY, dt_factor, Nc, Nz_gen)
            rotate_panel_to_geographic!(gen_u_geo, gen_v_geo, gen_u_loc, gen_v_loc,
                                        tangent_basis, Nc, Nz_gen)
            weighted_column_mean!(gen_u_col, gen_v_col, gen_u_geo, gen_v_geo, win.m)

            geos_mfx = panel_tuple_from_nc(ds["MFXC"], w, Nz_geos)
            geos_mfy = panel_tuple_from_nc(ds["MFYC"], w, Nz_geos)
            geos_dp = panel_tuple_from_nc(ds["DELP"], w, Nz_geos)
            geos_native_to_face_flux!(geos_am, geos_bm, geos_mfx, geos_mfy,
                                      mesh.connectivity, Nc, Nz_geos,
                                      geos_flux_scale)
            recover_cs_cell_center_winds!(geos_u_loc, geos_v_loc, geos_am, geos_bm,
                                          geos_dp, mesh.Δx, mesh.Δy,
                                          GRAVITY, dt_factor, Nc, Nz_geos)
            rotate_panel_to_geographic!(geos_u_geo, geos_v_geo, geos_u_loc, geos_v_loc,
                                        tangent_basis, Nc, Nz_geos)
            weighted_column_mean!(geos_u_col, geos_v_col, geos_u_geo, geos_v_geo,
                                  geos_dp)

            geos_ps = panel_matrix_tuple_from_nc(ds["PS"], w)
            for p in 1:6
                gps = geos_ps[p]
                bps = win.ps[p]
                @inbounds for j in 1:Nc, i in 1:Nc
                    # GEOS-IT CTM_A1 PS declares hPa.
                    ref_ps = 100.0 * Float64(gps[i, j])
                    got_ps = Float64(bps[i, j])
                    push_corr!(ps_corr, got_ps, ref_ps)
                    ps_bias_sum += got_ps - ref_ps
                    ps_abs_sum += abs(got_ps - ref_ps)
                    ps_n += 1
                end
            end

            for p in 1:6
                gu = gen_u_col[p]
                gv = gen_v_col[p]
                ru = geos_u_col[p]
                rv = geos_v_col[p]
                @inbounds for j in 1:Nc, i in 1:Nc
                    ug = Float64(gu[i, j])
                    vg = Float64(gv[i, j])
                    ur = Float64(ru[i, j])
                    vr = Float64(rv[i, j])
                    sg = hypot(ug, vg)
                    sr = hypot(ur, vr)
                    push_corr!(u_corr, ug, ur)
                    push_corr!(v_corr, vg, vr)
                    push_corr!(speed_corr, sg, sr)
                    if sg >= speed_threshold && sr >= speed_threshold
                        c = clamp((ug * ur + vg * vr) / (sg * sr), -1.0, 1.0)
                        angle = acosd(c)
                        push!(angles, Float32(angle))
                        push!(speed_ratios, Float32(sg / sr))
                        compared_vec += 1
                        angle > 90 && (opposite += 1)
                    end
                end
            end

            @printf("  window %2d done\n", w)
        end

        ps_r = corr(ps_corr)
        u_r = corr(u_corr)
        v_r = corr(v_corr)
        speed_r = corr(speed_corr)
        opp_frac = compared_vec > 0 ? opposite / compared_vec : NaN
        angle50 = quantile_or_nan(angles, 0.50)
        angle95 = quantile_or_nan(angles, 0.95)
        angle99 = quantile_or_nan(angles, 0.99)
        ratio50 = quantile_or_nan(speed_ratios, 0.50)
        ratio95 = quantile_or_nan(speed_ratios, 0.95)

        println()
        @printf("Surface pressure: corr=%.5f  mean_bias=%+.2f Pa  mean_abs=%.2f Pa  n=%d\n",
                ps_r, ps_bias_sum / max(ps_n, 1), ps_abs_sum / max(ps_n, 1), ps_n)
        @printf("Column vector:    corr(u)=%.5f  corr(v)=%.5f  corr(speed)=%.5f\n",
                u_r, v_r, speed_r)
        @printf("Direction gate:   n=%d  opposite_frac=%.4f  angle p50/p95/p99=%.1f/%.1f/%.1f deg\n",
                compared_vec, opp_frac, angle50, angle95, angle99)
        @printf("Speed ratio:      generated/GEOS p50/p95=%.3f/%.3f\n",
                ratio50, ratio95)

        red_flags = String[]
        abs(geos_lon - lon[90, 140]) > 5e-3 &&
            push!(red_flags, "grid longitude mismatch at p1(90,140)")
        abs(geos_lat - lat[90, 140]) > 5e-4 &&
            push!(red_flags, "grid latitude mismatch at p1(90,140)")
        ps_r < ps_corr_threshold &&
            push!(red_flags, @sprintf("surface-pressure correlation %.4f < %.4f",
                                      ps_r, ps_corr_threshold))
        opp_frac > opposite_threshold &&
            push!(red_flags, @sprintf("opposite-direction fraction %.4f > %.4f",
                                      opp_frac, opposite_threshold))
        angle95 > angle95_threshold &&
            push!(red_flags, @sprintf("angle p95 %.1f > %.1f deg",
                                      angle95, angle95_threshold))
        ratio50 < speed_ratio_min &&
            push!(red_flags, @sprintf("median speed ratio %.3f < %.3f",
                                      ratio50, speed_ratio_min))
        ratio50 > speed_ratio_max &&
            push!(red_flags, @sprintf("median speed ratio %.3f > %.3f",
                                      ratio50, speed_ratio_max))

        if isempty(red_flags)
            println("Validation status: PASS (no orientation/scale red flags)")
            return 0
        else
            println("Validation status: RED FLAGS")
            foreach(msg -> println("  - ", msg), red_flags)
            return strict ? 1 : 0
        end
    finally
        close(ds)
        # Do not close the mmap-backed binary reader explicitly here. The script
        # exits immediately, and closing the stream while slices are still live
        # has triggered libhdf5/mmap shutdown crashes on this workstation.
    end
end

code = main()
flush(stdout)
flush(stderr)

# NCDatasets/HDF5 plus mmap-backed readers can crash during process shutdown on
# this workstation after all validation output has already been produced. This
# script is read-only, so skip Julia finalizers and let the OS reclaim handles.
ccall(:_exit, Cvoid, (Cint,), code)
