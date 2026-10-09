#!/usr/bin/env julia

using Dates
using Printf
using Statistics
using GRIB
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: panel_cell_center_lonlat, panel_cell_local_tangent_basis
using .AtmosTransport.Preprocessing: read_spectral_coeffs!, vod2uv!, spectral_to_grid!

const GRAV = 9.80665

const ERA5_BIN = get(ENV, "ERA5_BIN",
    "/home/cfranken/data/AtmosTransport/met/era5/cs_c180/transport_binary_v2_full137_dec2021_f32_tm5_surface_pblrename_steps24/era5_transport_20211202_merged1000Pa_float32.bin")
const GEOS_BIN = get(ENV, "GEOS_BIN",
    "/temp1/c180_geosit_native_v4_dec2021_f32/geos_transport_20211202_float32.bin")
const GEOS_RAW = get(ENV, "GEOS_RAW",
    "/home/cfranken/data/AtmosTransport/met/geosit/C180/raw_catrine/20211202")
const ERA5_SPECTRAL = get(ENV, "ERA5_SPECTRAL",
    "/home/cfranken/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly")
const OUT_DIR = get(ENV, "OUT_DIR", "/temp1/c180_speed_diagnostics")

const WINDOW_INDEX = 2          # 2021-12-02 01:00-02:00-ish; GEOS A3dyn record 1 is 01:30.
const ERA5_HOUR = 1
const TARGET_HPA = [850.0, 500.0, 250.0]
const SITES = [
    (; name = "equatorial_pacific", lon = -150.0, lat = 0.0),
    (; name = "north_pacific_jet", lon = 160.0, lat = 45.0),
    (; name = "north_atlantic", lon = -50.0, lat = 45.0),
    (; name = "sahara_trades", lon = 0.0, lat = 20.0),
]

_lon180(lon) = mod(lon + 180.0, 360.0) - 180.0
_londiff(a, b) = abs(_lon180(a - b))

function _gc_distance2(lon1, lat1, lon2, lat2)
    dlon = deg2rad(_londiff(lon1, lon2))
    dlat = deg2rad(lat1 - lat2)
    latm = deg2rad((lat1 + lat2) / 2)
    return (dlon * cos(latm))^2 + dlat^2
end

function nearest_cs_cell(mesh, lon, lat)
    best = (dist = Inf, panel = 0, i = 0, j = 0, lon = NaN, lat = NaN)
    for p in 1:6
        lons, lats = panel_cell_center_lonlat(mesh, p)
        for j in 1:mesh.Nc, i in 1:mesh.Nc
            d = _gc_distance2(lon, lat, Float64(lons[i, j]), Float64(lats[i, j]))
            if d < best.dist
                best = (dist = d, panel = p, i = i, j = j,
                        lon = Float64(lons[i, j]), lat = Float64(lats[i, j]))
            end
        end
    end
    return best
end

function interior_window_arrays(window, mesh, panel)
    Hp, Nc = mesh.Hp, mesh.Nc
    ir = (Hp + 1):(Hp + Nc)
    m = @view window.air_mass[panel][ir, ir, :]
    am = @view window.fluxes.am[panel][(Hp + 1):(Hp + Nc + 1), ir, :]
    bm = @view window.fluxes.bm[panel][ir, (Hp + 1):(Hp + Nc + 1), :]
    return m, am, bm
end

function pressure_midpoints_hpa(m_panel, mesh, i, j)
    Nz = size(m_panel, 3)
    mids = Vector{Float64}(undef, Nz)
    ptop = 0.0
    area = Float64(mesh.cell_areas[i, j])
    for k in 1:Nz
        dp = Float64(m_panel[i, j, k]) * GRAV / area
        mids[k] = (ptop + 0.5 * dp) / 100.0
        ptop += dp
    end
    return mids
end

function face_normal_to_geographic(up, vp, basis, panel, i, j)
    x_east, x_north, y_east, y_north = basis[panel]
    xe = Float64(x_east[i, j])
    xn = Float64(x_north[i, j])
    ye = Float64(y_east[i, j])
    yn = Float64(y_north[i, j])
    c = clamp(xe * ye + xn * yn, -1.0, 1.0)
    denom = max(1.0 - c * c, eps(Float64))
    s = sqrt(denom)
    nx_east = (xe - c * ye) / s
    nx_north = (xn - c * yn) / s
    ny_east = (ye - c * xe) / s
    ny_north = (yn - c * xn) / s
    ax = (up + c * vp) / denom
    ay = (vp + c * up) / denom
    return (;
        u_east = ax * nx_east + ay * ny_east,
        v_north = ax * nx_north + ay * ny_north,
    )
end

function wind_at_cell(window, mesh, basis, dt_factor, panel, i, j, k)
    m, am, bm = interior_window_arrays(window, mesh, panel)
    area = Float64(mesh.cell_areas[i, j])
    dp = Float64(m[i, j, k]) * GRAV / area
    xsec_x = Float64(mesh.Δy[i, j]) * dp / GRAV * dt_factor
    xsec_y = Float64(mesh.Δx[i, j]) * dp / GRAV * dt_factor
    up = xsec_x > 0 ? 0.5 * (Float64(am[i, j, k]) + Float64(am[i + 1, j, k])) / xsec_x : 0.0
    vp = xsec_y > 0 ? 0.5 * (Float64(bm[i, j, k]) + Float64(bm[i, j + 1, k])) / xsec_y : 0.0
    geo = face_normal_to_geographic(up, vp, basis, panel, i, j)
    return (; u_panel = up, v_panel = vp, u_east = geo.u_east,
            v_north = geo.v_north, speed = hypot(geo.u_east, geo.v_north))
end

function wind_stats_at_pressures(window, mesh, basis, dt_factor)
    vals = [Float64[] for _ in TARGET_HPA]
    for p in 1:6
        m, _, _ = interior_window_arrays(window, mesh, p)
        Nz = size(m, 3)
        for j in 1:mesh.Nc, i in 1:mesh.Nc
            ptop = 0.0
            bestdiff = fill(Inf, length(TARGET_HPA))
            bestspeed = zeros(Float64, length(TARGET_HPA))
            area = Float64(mesh.cell_areas[i, j])
            for k in 1:Nz
                dp = Float64(m[i, j, k]) * GRAV / area
                ph = (ptop + 0.5 * dp) / 100.0
                w = wind_at_cell(window, mesh, basis, dt_factor, p, i, j, k).speed
                for t in eachindex(TARGET_HPA)
                    d = abs(ph - TARGET_HPA[t])
                    if d < bestdiff[t]
                        bestdiff[t] = d
                        bestspeed[t] = w
                    end
                end
                ptop += dp
            end
            for t in eachindex(TARGET_HPA)
                push!(vals[t], bestspeed[t])
            end
        end
    end
    return vals
end

function load_binary_window(path)
    driver = TransportBinaryDriver(path; FT = Float32, arch = CPU(), Hp = 3,
                                        validate_replay = false)
    window = load_transport_window(driver, WINDOW_INDEX)
    mesh = driver_grid(driver).horizontal
    steps = steps_per_window(driver, WINDOW_INDEX)
    # per-substep storage: flux [kg per palindrome application] over
    # dt/(2*steps) seconds. full-window storage: flux [kg per window]
    # over dt seconds — same wind either way.
    dt_factor = AtmosTransport.MetDrivers.flux_kind(driver) === :full_window_mass_amount ?
        Float64(window_dt(driver)) :
        Float64(window_dt(driver)) / (2.0 * Float64(steps))
    basis = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), 6)
    return (; driver, window, mesh, dt_factor, basis,
            dt = Float64(window_dt(driver)), steps = steps,
            Nz = size(window.air_mass[1], 3))
end

function geos_orientation()
    ctm = joinpath(GEOS_RAW, "GEOSIT.20211202.CTM_A1.C180.nc")
    NCDataset(ctm) do ds
        delp = ds["DELP"]
        Nz = size(delp, 4)
        topish = mean(skipmissing(delp[:, :, :, 1, 1]))
        botish = mean(skipmissing(delp[:, :, :, Nz, 1]))
        return topish > botish ? :bottom_up : :top_down
    end
end

function geos_actual_at(site_rows)
    out = Dict{Tuple{String, String, Float64}, NamedTuple}()
    orient = geos_orientation()
    a3 = joinpath(GEOS_RAW, "GEOSIT.20211202.A3dyn.C180.nc")
    ctm = joinpath(GEOS_RAW, "GEOSIT.20211202.CTM_A1.C180.nc")
    NCDataset(a3) do ads
        NCDataset(ctm) do cds
            for row in site_rows
                row.source == "GEOS" || continue
                raw_k = orient === :bottom_up ? row.Nz - row.k + 1 : row.k
                p, i, j = row.panel, row.i, row.j
                u = Float64(ads["U"][i, j, p, raw_k, 1])
                v = Float64(ads["V"][i, j, p, raw_k, 1])
                cx = Float64(cds["CX"][i, j, p, raw_k, WINDOW_INDEX])
                cy = Float64(cds["CY"][i, j, p, raw_k, WINDOW_INDEX])
                delp = Float64(cds["DELP"][i, j, p, raw_k, WINDOW_INDEX])
                mfxc = Float64(cds["MFXC"][i, j, p, raw_k, WINDOW_INDEX])
                mfyc = Float64(cds["MFYC"][i, j, p, raw_k, WINDOW_INDEX])
                out[("GEOS", row.site, row.target_hpa)] = (actual_source = "GEOS A3dyn U/V",
                    actual_u = u, actual_v = v, actual_speed = hypot(u, v),
                    raw_cx = cx, raw_cy = cy, raw_delp = delp, raw_mfxc = mfxc,
                    raw_mfyc = mfyc)
            end
        end
    end
    return out
end

function read_era5_selected_spectral(levels; hour = ERA5_HOUR, T_target = 359)
    vo_d_path = joinpath(ERA5_SPECTRAL, "era5_spectral_20211202_vo_d.gb")
    f = GRIB.GribFile(vo_d_path)
    msg1 = first(f)
    T_file = Int(msg1["J"])
    GRIB.destroy(f)
    T = min(T_file, T_target)

    want = Set(Int.(levels))
    vo = Dict{Int, Matrix{ComplexF64}}()
    d = Dict{Int, Matrix{ComplexF64}}()
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    vals_buf = Float64[]

    f = GRIB.GribFile(vo_d_path)
    try
        for msg in f
            div(Int(msg["dataTime"]), 100) == hour || continue
            lev = Int(msg["level"])
            lev in want || continue
            name = String(msg["shortName"])
            (name == "vo" || name == "d") || continue
            read_spectral_coeffs!(spec_buf, msg, vals_buf)
            coeffs = copy(@view spec_buf[1:T + 1, 1:T + 1])
            if name == "vo"
                vo[lev] = coeffs
            else
                d[lev] = coeffs
            end
            length(vo) == length(want) && length(d) == length(want) && break
        end
    finally
        GRIB.destroy(f)
    end
    return (; vo, d, T)
end

function era5_actual_at(site_rows)
    rows = filter(r -> r.source == "ERA5", site_rows)
    isempty(rows) && return Dict{Tuple{String, String, Float64}, NamedTuple}()
    levels = sort(unique(r.k for r in rows))
    spec = read_era5_selected_spectral(levels)
    out = Dict{Tuple{String, String, Float64}, NamedTuple}()
    Nlon = 720
    P = zeros(Float64, spec.T + 1, spec.T + 1)
    fft_buf = zeros(ComplexF64, Nlon)
    field = zeros(Float64, Nlon, 1)
    u_spec = zeros(ComplexF64, spec.T + 1, spec.T + 1)
    v_spec = zeros(ComplexF64, spec.T + 1, spec.T + 1)

    for row in rows
        if !haskey(spec.vo, row.k) || !haskey(spec.d, row.k)
            out[("ERA5", row.site, row.target_hpa)] = (actual_source = "ERA5 spectral unavailable",
                actual_u = NaN, actual_v = NaN, actual_speed = NaN)
            continue
        end
        vod2uv!(u_spec, v_spec, spec.vo[row.k], spec.d[row.k], spec.T)
        lon = row.lon
        lat = row.lat
        spectral_to_grid!(field, u_spec, spec.T, [lat], Nlon, P, fft_buf;
                          lon_shift_rad = deg2rad(lon))
        u = field[1, 1] / max(cosd(lat), 1e-6)
        spectral_to_grid!(field, v_spec, spec.T, [lat], Nlon, P, fft_buf;
                          lon_shift_rad = deg2rad(lon))
        # `vod2uv!` returns ECMWF pseudo-wind coefficients for both components:
        # U*cos(phi) and V*cos(phi). Convert both to physical wind here.
        v = field[1, 1] / max(cosd(lat), 1e-6)
        out[("ERA5", row.site, row.target_hpa)] = (actual_source = "ERA5 spectral VO/D T359",
            actual_u = u, actual_v = v, actual_speed = hypot(u, v))
    end
    return out
end

function write_global_stats(path, data)
    open(path, "w") do io
        println(io, "source,target_hpa,mean,median,p95,p99,max")
        for (source, vals_by_target) in data
            for (target, vals) in zip(TARGET_HPA, vals_by_target)
                qs = quantile(vals, [0.5, 0.95, 0.99])
                @printf(io, "%s,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        source, target, mean(vals), qs[1], qs[2], qs[3], maximum(vals))
            end
        end
    end
end

_ntget(nt, name::Symbol, default) = name in propertynames(nt) ? getproperty(nt, name) : default

function write_site_rows(path, rows, actuals)
    open(path, "w") do io
        println(io, join(("source", "site", "target_hpa", "requested_lon", "requested_lat",
                          "cell_lon", "cell_lat", "panel", "i", "j", "k", "pmid_hpa",
                          "inferred_u", "inferred_v", "inferred_speed",
                          "actual_source", "actual_u", "actual_v", "actual_speed",
                          "speed_ratio_inferred_to_actual", "raw_cx", "raw_cy",
                          "raw_delp", "raw_mfxc", "raw_mfyc"), ","))
        for row in rows
            actual = get(actuals, (row.source, row.site, row.target_hpa),
                         (actual_source = "", actual_u = NaN, actual_v = NaN,
                          actual_speed = NaN))
            ratio = isfinite(actual.actual_speed) && actual.actual_speed > 0 ?
                    row.speed / actual.actual_speed : NaN
            @printf(io, "%s,%s,%.0f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%.3f,%.6f,%.6f,%.6f,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    row.source, row.site, row.target_hpa, row.requested_lon, row.requested_lat,
                    row.lon, row.lat, row.panel, row.i, row.j, row.k, row.pmid_hpa,
                    row.u_east, row.v_north, row.speed, actual.actual_source,
                    actual.actual_u, actual.actual_v, actual.actual_speed, ratio,
                    _ntget(actual, :raw_cx, NaN), _ntget(actual, :raw_cy, NaN),
                    _ntget(actual, :raw_delp, NaN), _ntget(actual, :raw_mfxc, NaN),
                    _ntget(actual, :raw_mfyc, NaN))
        end
    end
end

function site_rows_for(source, loaded)
    rows = NamedTuple[]
    for site in SITES
        cell = nearest_cs_cell(loaded.mesh, site.lon, site.lat)
        m, _, _ = interior_window_arrays(loaded.window, loaded.mesh, cell.panel)
        pmids = pressure_midpoints_hpa(m, loaded.mesh, cell.i, cell.j)
        for target in TARGET_HPA
            k = argmin(abs.(pmids .- target))
            w = wind_at_cell(loaded.window, loaded.mesh, loaded.basis, loaded.dt_factor,
                             cell.panel, cell.i, cell.j, k)
            push!(rows, (; source, site = site.name, requested_lon = site.lon,
                         requested_lat = site.lat, lon = cell.lon, lat = cell.lat,
                         panel = cell.panel, i = cell.i, j = cell.j, k,
                         Nz = loaded.Nz, target_hpa = target, pmid_hpa = pmids[k],
                         u_east = w.u_east, v_north = w.v_north, speed = w.speed))
        end
    end
    return rows
end

function main()
    mkpath(OUT_DIR)
    println("Loading ERA5 binary window...")
    era = load_binary_window(ERA5_BIN)
    println("Loading GEOS-IT binary window...")
    geo = load_binary_window(GEOS_BIN)

    println("Computing global speed distributions...")
    global_stats = [
        "ERA5" => wind_stats_at_pressures(era.window, era.mesh, era.basis, era.dt_factor),
        "GEOS" => wind_stats_at_pressures(geo.window, geo.mesh, geo.basis, geo.dt_factor),
    ]
    write_global_stats(joinpath(OUT_DIR, "binary_wind_global_stats.csv"), global_stats)

    rows = vcat(site_rows_for("ERA5", era), site_rows_for("GEOS", geo))
    println("Reading GEOS raw U/V comparison...")
    actuals = Dict{Tuple{String, String, Float64}, NamedTuple}()
    merge!(actuals, geos_actual_at(rows))
    println("Reading ERA5 spectral U/V comparison...")
    try
        merge!(actuals, era5_actual_at(rows))
    catch err
        @warn "ERA5 spectral comparison failed; binary-inferred comparison still written" exception=(err, catch_backtrace())
    end
    write_site_rows(joinpath(OUT_DIR, "binary_wind_key_locations.csv"), rows, actuals)

    open(joinpath(OUT_DIR, "README.txt"), "w") do io
        println(io, "C180 binary wind-speed diagnostic")
        println(io, "Generated: ", Dates.now())
        println(io, "Window index: ", WINDOW_INDEX)
        println(io, "ERA5 binary: ", ERA5_BIN)
        println(io, "GEOS binary: ", GEOS_BIN)
        println(io, "Formula: u ~= 2 * half_step_flux * cell_area / (air_mass * face_length * dt_sub)")
        println(io, "CSV files:")
        println(io, "  binary_wind_global_stats.csv")
        println(io, "  binary_wind_key_locations.csv")
    end

    close(era.driver)
    close(geo.driver)
    println("Wrote ", OUT_DIR)
end

main()
