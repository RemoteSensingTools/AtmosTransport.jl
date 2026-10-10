#!/usr/bin/env julia

"""
Write a NetCDF diagnostics file comparing an ERA5-on-GEOS-native CS binary
against GEOS-IT native C180 fields on the exact GEOS L72 vertical grid.

This script is deliberately stricter than `compare_geosit_reference.jl`: it
requires the generated binary to use the GEOS L72 interface coefficients. The
older `ml137_tropo34` files are useful for horizontal orientation and
column-integrated checks, but they cannot support layer-by-layer wind
agreement diagnostics.

The output NetCDF contains:

  * grid coordinates (`lons`, `lats`, `cell_area`) on the native CS panels;
  * all-level summary statistics `(lev,time)`;
  * map fields on selected exact GEOS levels `(Xdim,Ydim,nf,map_lev,time)`;
  * surface-pressure bias maps and summary statistics.
"""

using Dates
using LinearAlgebra
using Printf
using Statistics
using TOML
using NCDatasets

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO_ROOT, "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Architectures: CPU
using .AtmosTransport.Grids:
    n_levels, panel_cell_center_lonlat, panel_cell_local_tangent_basis
using .AtmosTransport.MetDrivers:
    TransportBinaryReader, load_window!, load_grid
using .AtmosTransport.Preprocessing:
    load_hybrid_coefficients,
    geos_native_to_face_flux!, recover_cs_cell_center_winds!,
    rotate_panel_to_geographic!

const GRAVITY = Float32(9.80665)
const DEFAULT_GEOS_ROOT =
    "/home/cfranken/data/AtmosTransport/met/geosit/C180/raw_catrine"
const GEOS_L72_COEFFS = joinpath(REPO_ROOT, "config", "geos_L72_coefficients.toml")

mutable struct CorrStats
    n::Int64
    sx::Float64
    sy::Float64
    sx2::Float64
    sy2::Float64
    sxy::Float64
end

CorrStats() = CorrStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)

@inline function reset!(s::CorrStats)
    s.n = 0
    s.sx = 0.0
    s.sy = 0.0
    s.sx2 = 0.0
    s.sy2 = 0.0
    s.sxy = 0.0
    return s
end

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
            if key == "help"
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
      julia --project=. scripts/validation/heritage/write_geosit_agreement_netcdf.jl \\
        --binary PATH --date YYYYMMDD --output OUT.nc [options]

    Options:
      --geos-root DIR          GEOS-IT daily root [$DEFAULT_GEOS_ROOT]
      --windows SPEC           Windows to compare, e.g. 1:24 or 1,2,3 [1:24]
      --map-levels SPEC        Exact GEOS levels to map, e.g. 1,12,24,36,48,60,72
      --speed-threshold MS     Ignore vector angles below this speed [2.0]
      --geos-mass-flux-dt S    FV3 dynamics step for MFXC/MFYC [450.0]
    """)
end

function parse_index_spec(spec::AbstractString, nmax::Int)
    s = strip(spec)
    isempty(s) && return Int[]
    if occursin(':', s)
        parts = split(s, ':')
        length(parts) in (2, 3) || error("Range must be A:B or A:S:B, got $spec")
        vals = length(parts) == 2 ?
            collect(parse(Int, parts[1]):parse(Int, parts[2])) :
            collect(parse(Int, parts[1]):parse(Int, parts[2]):parse(Int, parts[3]))
    else
        vals = [parse(Int, strip(x)) for x in split(s, ',') if !isempty(strip(x))]
    end
    all(1 .<= vals .<= nmax) || error("Index spec $spec outside 1:$nmax")
    return unique(vals)
end

function geos_ctm_path(root::AbstractString, date::AbstractString)
    path = joinpath(root, date, "GEOSIT.$date.CTM_A1.C180.nc")
    isfile(path) || error("Missing GEOS-IT CTM_A1 reference file: $path")
    return path
end

function default_output_path(bin_path::AbstractString, date::AbstractString)
    stem = splitext(basename(bin_path))[1]
    return joinpath(dirname(bin_path), "$(stem)_GEOSIT_$(date)_agreement.nc")
end

function panel_tuple_from_nc(v, w::Int, Nz::Int; reverse_levels::Bool=false)
    lev_idx = reverse_levels ? (Nz:-1:1) : (1:Nz)
    return ntuple(p -> Array{Float32}(v[:, :, p, lev_idx, w]), 6)
end

function panel_matrix_tuple_from_nc(v, w::Int)
    return ntuple(p -> Array{Float32}(v[:, :, p, w]), 6)
end

@inline function quantile_or_nan(x, q)
    isempty(x) && return NaN32
    return Float32(quantile(x, q))
end

function fill_dp_from_mass!(dp, m, areas, g::Float32)
    for p in 1:6
        dpp = dp[p]
        mp = m[p]
        @inbounds for k in axes(mp, 3), j in axes(mp, 2), i in axes(mp, 1)
            dpp[i, j, k] = mp[i, j, k] * g / Float32(areas[i, j])
        end
    end
    return dp
end

function assert_geos_l72_header!(h)
    geos_vc = load_hybrid_coefficients(GEOS_L72_COEFFS)
    Nz = n_levels(geos_vc)
    h.nlevel == Nz ||
        error("Binary has $(h.nlevel) levels, but GEOS-IT layer diagnostics require GEOS L72. " *
              "Regenerate with config/preprocessing/era5_geosit_c180_l72_transport_binary_f32.toml")
    length(h.A_ifc) == Nz + 1 && length(h.B_ifc) == Nz + 1 ||
        error("Binary A_ifc/B_ifc lengths do not match GEOS L72")
    max_A = maximum(abs.(h.A_ifc .- Float64.(geos_vc.A)))
    max_B = maximum(abs.(h.B_ifc .- Float64.(geos_vc.B)))
    max_A < 1e-6 && max_B < 1e-12 ||
        error(@sprintf("Binary vertical coefficients are not GEOS L72 (max |A| %.3g Pa, max |B| %.3g)",
                       max_A, max_B))
    return nothing
end

function geos_levels_are_surface_to_top(ds, w::Int, Nz::Int)
    delp = ds["DELP"]
    mean_first = mean(Float64.(delp[:, :, 1, 1, w]))
    mean_last = mean(Float64.(delp[:, :, 1, Nz, w]))
    return mean_first > mean_last
end

function define_geometry!(ds, mesh, Nc::Int, Nz::Int, windows, map_levels)
    defDim(ds, "Xdim", Nc)
    defDim(ds, "Ydim", Nc)
    defDim(ds, "nf", 6)
    defDim(ds, "lev", Nz)
    defDim(ds, "ilev", Nz + 1)
    defDim(ds, "time", length(windows))
    defDim(ds, "map_lev", length(map_levels))

    defVar(ds, "Xdim", Int32, ("Xdim",),
           attrib = Dict("long_name" => "cubed-sphere panel X index"))[:] = Int32.(1:Nc)
    defVar(ds, "Ydim", Int32, ("Ydim",),
           attrib = Dict("long_name" => "cubed-sphere panel Y index"))[:] = Int32.(1:Nc)
    defVar(ds, "nf", Int32, ("nf",),
           attrib = Dict("long_name" => "cubed-sphere panel"))[:] = Int32.(1:6)
    defVar(ds, "lev", Int32, ("lev",),
           attrib = Dict("long_name" => "GEOS L72 model layer index, top to surface"))[:] =
        Int32.(1:Nz)
    defVar(ds, "ilev", Int32, ("ilev",),
           attrib = Dict("long_name" => "GEOS L72 interface index, top to surface"))[:] =
        Int32.(1:(Nz + 1))
    defVar(ds, "map_lev", Int32, ("map_lev",),
           attrib = Dict("long_name" => "GEOS L72 levels included in map variables"))[:] =
        Int32.(map_levels)
    defVar(ds, "time", Int32, ("time",),
           attrib = Dict("long_name" => "1-based GEOS-IT CTM_A1 time/window index"))[:] =
        Int32.(windows)

    lons = Array{Float64}(undef, Nc, Nc, 6)
    lats = similar(lons)
    area = similar(lons)
    for p in 1:6
        lonp, latp = panel_cell_center_lonlat(mesh, p)
        lons[:, :, p] = lonp
        lats[:, :, p] = latp
        @inbounds for j in 1:Nc, i in 1:Nc
            area[i, j, p] = Float64(mesh.cell_areas[i, j])
        end
    end
    defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "degrees_east",
                         "standard_name" => "longitude",
                         "long_name" => "cell-center longitude"))[:, :, :] = lons
    defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "degrees_north",
                         "standard_name" => "latitude",
                         "long_name" => "cell-center latitude"))[:, :, :] = lats
    defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "m2",
                         "standard_name" => "cell_area",
                         "long_name" => "horizontal cell area"))[:, :, :] = area
    return nothing
end

function defmap(ds, name, long_name, units)
    return defVar(ds, name, Float32, ("Xdim", "Ydim", "nf", "map_lev", "time");
                  attrib = Dict("long_name" => long_name,
                                "units" => units,
                                "_FillValue" => NaN32,
                                "coordinates" => "lons lats"),
                  deflatelevel = 1, shuffle = true)
end

function defsummary(ds, name, long_name, units; T=Float32)
    return defVar(ds, name, T, ("lev", "time");
                  attrib = Dict("long_name" => long_name,
                                "units" => units,
                                "_FillValue" => T <: AbstractFloat ? T(NaN) : typemin(T)),
                  deflatelevel = 1, shuffle = true)
end

function main()
    opts, flags = parse_args(ARGS)
    if "help" in flags || !haskey(opts, "binary") || !haskey(opts, "date")
        usage()
        return "help" in flags ? 0 : 2
    end

    bin_path = expanduser(opts["binary"])
    isfile(bin_path) || error("Binary not found: $bin_path")
    date = opts["date"]
    root = expanduser(get(opts, "geos-root", DEFAULT_GEOS_ROOT))
    output = expanduser(get(opts, "output", default_output_path(bin_path, date)))
    speed_threshold = parse(Float64, get(opts, "speed-threshold", "2.0"))
    geos_mass_flux_dt = parse(Float64, get(opts, "geos-mass-flux-dt", "450.0"))

    reader = TransportBinaryReader(bin_path; FT=Float32)
    h = reader.header
    assert_geos_l72_header!(h)
    windows = parse_index_spec(get(opts, "windows", "1:$(h.nwindow)"), h.nwindow)
    map_levels = parse_index_spec(get(opts, "map-levels", "1,12,24,36,48,60,72"), h.nlevel)

    h.geometry.panel_convention === :geos_native ||
        error("Expected GEOS-native panel convention, got $(h.geometry.panel_convention)")
    h.geometry.definition === :gmao_equal_distance ||
        error("Expected GMAO CS definition, got $(h.geometry.definition)")
    h.geometry.coordinate_law === :gmao_equal_distance_gnomonic ||
        error("Expected GMAO equal-distance coordinate law, got $(h.geometry.coordinate_law)")
    h.geometry.center_law === :four_corner_normalized ||
        error("Expected four-corner normalized centers, got $(h.geometry.center_law)")

    geos_path = geos_ctm_path(root, date)
    mkpath(dirname(output))

    grid = load_grid(reader; FT=Float64, arch=CPU())
    mesh = grid.horizontal
    tangent_basis = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), 6)
    Nc = h.geometry.Nc
    Nz = h.nlevel
    map_index = Dict(k => i for (i, k) in enumerate(map_levels))

    @info @sprintf("Writing GEOS-IT agreement NetCDF: C%d L%d, %d window(s), %d map level(s)",
                   Nc, Nz, length(windows), length(map_levels))
    @info "  Binary: $(bin_path)"
    @info "  GEOS:   $(geos_path)"
    @info "  Output: $(output)"

    NCDataset(geos_path) do gds
        mfxc_shape = size(gds["MFXC"])
        mfxc_shape[1:4] == (Nc, Nc, 6, Nz) ||
            error("GEOS MFXC shape $(mfxc_shape) does not match C$Nc L$Nz")
        reverse_geos_levels = geos_levels_are_surface_to_top(gds, first(windows), Nz)
        reverse_geos_levels &&
            @info "  GEOS lev order is surface-to-top; reversing GEOS 3D fields to top-to-surface"

        NCDataset(output, "c") do ds
            ds.attrib["Conventions"] = "CF-1.8"
            ds.attrib["title"] = "ERA5 preprocessor agreement against GEOS-IT on GEOS-native C180 L72"
            ds.attrib["source_binary"] = bin_path
            ds.attrib["geos_reference"] = geos_path
            ds.attrib["date"] = date
            ds.attrib["mass_basis"] = String(h.mass_basis)
            ds.attrib["vertical_grid"] = "GEOS L72"
            ds.attrib["horizontal_grid"] = "GEOS-native GMAO C$(Nc)"
            ds.attrib["speed_threshold_ms"] = speed_threshold
            ds.attrib["geos_mass_flux_dt_seconds"] = geos_mass_flux_dt
            ds.attrib["geos_levels_reversed_to_match_binary"] = Int32(reverse_geos_levels ? 1 : 0)
            ds.attrib["history"] = "written by scripts/validation/heritage/write_geosit_agreement_netcdf.jl"

            define_geometry!(ds, mesh, Nc, Nz, windows, map_levels)
            defVar(ds, "A_ifc", Float64, ("ilev",),
                   attrib = Dict("units" => "Pa",
                                 "long_name" => "GEOS L72 hybrid A coefficient"))[:] = h.A_ifc
            defVar(ds, "B_ifc", Float64, ("ilev",),
                   attrib = Dict("units" => "1",
                                 "long_name" => "GEOS L72 hybrid B coefficient"))[:] = h.B_ifc

            v_ps_bias = defVar(ds, "ps_bias_pa", Float32, ("Xdim", "Ydim", "nf", "time");
                               attrib = Dict("long_name" => "generated PS minus GEOS-IT PS",
                                             "units" => "Pa",
                                             "_FillValue" => NaN32,
                                             "coordinates" => "lons lats"),
                               deflatelevel = 1, shuffle = true)
            v_dir = defmap(ds, "direction_error_deg",
                           "signed horizontal wind direction error, generated minus GEOS-IT", "degree")
            v_absdir = defmap(ds, "abs_direction_error_deg",
                              "absolute horizontal wind direction error", "degree")
            v_speed_bias = defmap(ds, "speed_bias_ms",
                                  "horizontal wind speed bias, generated minus GEOS-IT", "m s-1")
            v_log_speed_ratio = defmap(ds, "log_speed_ratio",
                                       "log generated over GEOS-IT horizontal wind speed", "1")
            v_parallel = defmap(ds, "parallel_diff_ms",
                                "wind-vector difference projected onto GEOS-IT wind direction", "m s-1")
            v_perp = defmap(ds, "perpendicular_diff_ms",
                            "wind-vector difference projected 90 degrees left of GEOS-IT wind", "m s-1")

            # Summary arrays are written after all windows are processed.
            ps_corr = fill(NaN32, length(windows))
            ps_mean_bias = fill(NaN32, length(windows))
            ps_mean_abs = fill(NaN32, length(windows))
            corr_u = fill(NaN32, Nz, length(windows))
            corr_v = fill(NaN32, Nz, length(windows))
            corr_speed = fill(NaN32, Nz, length(windows))
            angle_p50 = fill(NaN32, Nz, length(windows))
            angle_p95 = fill(NaN32, Nz, length(windows))
            speed_ratio_p50 = fill(NaN32, Nz, length(windows))
            speed_ratio_p95 = fill(NaN32, Nz, length(windows))
            opposite_fraction = fill(NaN32, Nz, length(windows))
            mean_speed_bias = fill(NaN32, Nz, length(windows))
            mean_parallel_diff = fill(NaN32, Nz, length(windows))
            mean_perp_diff = fill(NaN32, Nz, length(windows))
            rms_vector_diff = fill(NaN32, Nz, length(windows))
            valid_vector_count = zeros(Int32, Nz, length(windows))

            gen_dp = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)
            gen_u_loc = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)
            gen_v_loc = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)
            gen_u_geo = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)
            gen_v_geo = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)

            geos_am = ntuple(_ -> Array{Float32}(undef, Nc + 1, Nc, Nz), 6)
            geos_bm = ntuple(_ -> Array{Float32}(undef, Nc, Nc + 1, Nz), 6)
            geos_u_loc = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)
            geos_v_loc = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)
            geos_u_geo = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)
            geos_v_geo = ntuple(_ -> Array{Float32}(undef, Nc, Nc, Nz), 6)

            ps_bias_buf = Array{Float32}(undef, Nc, Nc, 6)
            dir_buf = similar(ps_bias_buf)
            absdir_buf = similar(ps_bias_buf)
            speed_bias_buf = similar(ps_bias_buf)
            log_speed_ratio_buf = similar(ps_bias_buf)
            parallel_buf = similar(ps_bias_buf)
            perp_buf = similar(ps_bias_buf)
            u_corr = CorrStats()
            v_corr = CorrStats()
            s_corr = CorrStats()
            p_corr = CorrStats()

            for (tw, w) in enumerate(windows)
                @info @sprintf("  Window %d/%d (GEOS time index %d)", tw, length(windows), w)
                steps = h.steps_per_window_by_window[w]
                dt_factor = Float32(AtmosTransport.MetDrivers.flux_application_seconds(
                h.dt_met_seconds, steps, AtmosTransport.MetDrivers.flux_kind(reader)))
                geos_flux_scale = Float32(Float64(dt_factor) / geos_mass_flux_dt) / GRAVITY
                win = load_window!(reader, w)
                fill_dp_from_mass!(gen_dp, win.m, mesh.cell_areas, GRAVITY)
                recover_cs_cell_center_winds!(gen_u_loc, gen_v_loc, win.am, win.bm,
                                              gen_dp, mesh.Δx, mesh.Δy,
                                              GRAVITY, dt_factor, Nc, Nz)
                rotate_panel_to_geographic!(gen_u_geo, gen_v_geo, gen_u_loc, gen_v_loc,
                                            tangent_basis, Nc, Nz)

                geos_mfx = panel_tuple_from_nc(gds["MFXC"], w, Nz;
                                                reverse_levels=reverse_geos_levels)
                geos_mfy = panel_tuple_from_nc(gds["MFYC"], w, Nz;
                                                reverse_levels=reverse_geos_levels)
                geos_dp = panel_tuple_from_nc(gds["DELP"], w, Nz;
                                               reverse_levels=reverse_geos_levels)
                geos_native_to_face_flux!(geos_am, geos_bm, geos_mfx, geos_mfy,
                                          mesh.connectivity, Nc, Nz,
                                          geos_flux_scale)
                recover_cs_cell_center_winds!(geos_u_loc, geos_v_loc, geos_am, geos_bm,
                                              geos_dp, mesh.Δx, mesh.Δy,
                                              GRAVITY, dt_factor, Nc, Nz)
                rotate_panel_to_geographic!(geos_u_geo, geos_v_geo, geos_u_loc, geos_v_loc,
                                            tangent_basis, Nc, Nz)

                reset!(p_corr)
                ps_sum = 0.0
                ps_abs_sum = 0.0
                ps_n = 0
                geos_ps = panel_matrix_tuple_from_nc(gds["PS"], w)
                for p in 1:6
                    gps = geos_ps[p]
                    bps = win.ps[p]
                    @inbounds for j in 1:Nc, i in 1:Nc
                        ref_ps = 100.0f0 * gps[i, j]
                        got_ps = bps[i, j]
                        bias = got_ps - ref_ps
                        ps_bias_buf[i, j, p] = bias
                        push_corr!(p_corr, got_ps, ref_ps)
                        ps_sum += Float64(bias)
                        ps_abs_sum += abs(Float64(bias))
                        ps_n += 1
                    end
                end
                v_ps_bias[:, :, :, tw] = ps_bias_buf
                ps_corr[tw] = Float32(corr(p_corr))
                ps_mean_bias[tw] = Float32(ps_sum / max(ps_n, 1))
                ps_mean_abs[tw] = Float32(ps_abs_sum / max(ps_n, 1))

                for k in 1:Nz
                    reset!(u_corr)
                    reset!(v_corr)
                    reset!(s_corr)
                    angles = Float32[]
                    ratios = Float32[]
                    nvalid = 0
                    nopposite = 0
                    sum_speed = 0.0
                    sum_parallel = 0.0
                    sum_perp = 0.0
                    sum_vec2 = 0.0
                    map_slot = get(map_index, k, 0)

                    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
                        ug = Float64(gen_u_geo[p][i, j, k])
                        vg = Float64(gen_v_geo[p][i, j, k])
                        ur = Float64(geos_u_geo[p][i, j, k])
                        vr = Float64(geos_v_geo[p][i, j, k])
                        sg = hypot(ug, vg)
                        sr = hypot(ur, vr)
                        push_corr!(u_corr, ug, ur)
                        push_corr!(v_corr, vg, vr)
                        push_corr!(s_corr, sg, sr)

                        signed_angle = NaN
                        parallel = NaN
                        perp = NaN
                        ratio = NaN
                        if sg >= speed_threshold && sr >= speed_threshold
                            dotv = ug * ur + vg * vr
                            crossv = ur * vg - vr * ug
                            signed_angle = atan(crossv, dotv) * 180.0 / pi
                            ratio = sg / sr
                            dug = ug - ur
                            dvg = vg - vr
                            parallel = (dug * ur + dvg * vr) / sr
                            perp = (dug * (-vr) + dvg * ur) / sr
                            push!(angles, Float32(abs(signed_angle)))
                            push!(ratios, Float32(ratio))
                            nvalid += 1
                            abs(signed_angle) > 90 && (nopposite += 1)
                            sum_parallel += parallel
                            sum_perp += perp
                        end
                        sum_speed += sg - sr
                        sum_vec2 += (ug - ur)^2 + (vg - vr)^2

                        if map_slot > 0
                            dir_buf[i, j, p] = Float32(signed_angle)
                            absdir_buf[i, j, p] = Float32(abs(signed_angle))
                            speed_bias_buf[i, j, p] = Float32(sg - sr)
                            log_speed_ratio_buf[i, j, p] =
                                sr > 0 && sg > 0 ? Float32(log(sg / sr)) : NaN32
                            parallel_buf[i, j, p] = Float32(parallel)
                            perp_buf[i, j, p] = Float32(perp)
                        end
                    end

                    ncells = 6 * Nc * Nc
                    corr_u[k, tw] = Float32(corr(u_corr))
                    corr_v[k, tw] = Float32(corr(v_corr))
                    corr_speed[k, tw] = Float32(corr(s_corr))
                    angle_p50[k, tw] = quantile_or_nan(angles, 0.50)
                    angle_p95[k, tw] = quantile_or_nan(angles, 0.95)
                    speed_ratio_p50[k, tw] = quantile_or_nan(ratios, 0.50)
                    speed_ratio_p95[k, tw] = quantile_or_nan(ratios, 0.95)
                    opposite_fraction[k, tw] = nvalid > 0 ? Float32(nopposite / nvalid) : NaN32
                    mean_speed_bias[k, tw] = Float32(sum_speed / ncells)
                    mean_parallel_diff[k, tw] = nvalid > 0 ? Float32(sum_parallel / nvalid) : NaN32
                    mean_perp_diff[k, tw] = nvalid > 0 ? Float32(sum_perp / nvalid) : NaN32
                    rms_vector_diff[k, tw] = Float32(sqrt(sum_vec2 / ncells))
                    valid_vector_count[k, tw] = Int32(nvalid)

                    if map_slot > 0
                        v_dir[:, :, :, map_slot, tw] = dir_buf
                        v_absdir[:, :, :, map_slot, tw] = absdir_buf
                        v_speed_bias[:, :, :, map_slot, tw] = speed_bias_buf
                        v_log_speed_ratio[:, :, :, map_slot, tw] = log_speed_ratio_buf
                        v_parallel[:, :, :, map_slot, tw] = parallel_buf
                        v_perp[:, :, :, map_slot, tw] = perp_buf
                    end
                end
            end

            defVar(ds, "ps_corr", Float32, ("time",),
                   attrib = Dict("long_name" => "surface-pressure correlation",
                                 "units" => "1"))[:] = ps_corr
            defVar(ds, "ps_mean_bias_pa", Float32, ("time",),
                   attrib = Dict("long_name" => "mean generated minus GEOS-IT surface pressure",
                                 "units" => "Pa"))[:] = ps_mean_bias
            defVar(ds, "ps_mean_abs_bias_pa", Float32, ("time",),
                   attrib = Dict("long_name" => "mean absolute surface-pressure bias",
                                 "units" => "Pa"))[:] = ps_mean_abs

            defsummary(ds, "corr_u", "correlation of eastward wind component", "1")[:, :] = corr_u
            defsummary(ds, "corr_v", "correlation of northward wind component", "1")[:, :] = corr_v
            defsummary(ds, "corr_speed", "correlation of horizontal wind speed", "1")[:, :] = corr_speed
            defsummary(ds, "angle_p50_deg", "median absolute wind direction error", "degree")[:, :] = angle_p50
            defsummary(ds, "angle_p95_deg", "95th percentile absolute wind direction error", "degree")[:, :] = angle_p95
            defsummary(ds, "speed_ratio_p50", "median generated over GEOS-IT speed ratio", "1")[:, :] = speed_ratio_p50
            defsummary(ds, "speed_ratio_p95", "95th percentile generated over GEOS-IT speed ratio", "1")[:, :] = speed_ratio_p95
            defsummary(ds, "opposite_fraction", "fraction of valid vectors with direction error over 90 degrees", "1")[:, :] = opposite_fraction
            defsummary(ds, "mean_speed_bias_ms", "mean generated minus GEOS-IT speed", "m s-1")[:, :] = mean_speed_bias
            defsummary(ds, "mean_parallel_diff_ms", "mean vector difference parallel to GEOS-IT wind", "m s-1")[:, :] = mean_parallel_diff
            defsummary(ds, "mean_perpendicular_diff_ms", "mean vector difference perpendicular to GEOS-IT wind", "m s-1")[:, :] = mean_perp_diff
            defsummary(ds, "rms_vector_diff_ms", "root-mean-square horizontal vector wind difference", "m s-1")[:, :] = rms_vector_diff
            defsummary(ds, "valid_vector_count", "number of vectors passing the speed threshold", "1"; T=Int32)[:, :] = valid_vector_count
        end
    end

    @info "Done: $(output)"
    return 0
end

code = main()
flush(stdout)
flush(stderr)
ccall(:_exit, Cvoid, (Cint,), code)
