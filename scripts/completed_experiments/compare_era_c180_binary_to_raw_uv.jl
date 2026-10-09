#!/usr/bin/env julia

using NCDatasets
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: panel_cell_center_lonlat, panel_cell_local_tangent_basis

const GRAV = 9.80665

Base.@kwdef struct Config
    era_cs_bin::String = get(ENV, "ERA_CS_BIN",
        "/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211204_merged1000Pa_float32.bin")
    raw_uv_nc::String = get(ENV, "RAW_UV_NC",
        "/temp1/era5_cds_global_winds_20211204/era5_pressure_level_uv_global_20211204.nc")
    out_dir::String = get(ENV, "OUT_DIR", "/temp1/c180_era_raw_uv_compare")
    window::Int = parse(Int, get(ENV, "WINDOW", "13"))
    time_index::Int = parse(Int, get(ENV, "TIME_INDEX", "13"))
    region::NTuple{4, Float64} = parse_region(get(ENV, "REGION", "10,40,-40,-20"))
    target_hpa::Vector{Float64} = parse_list(get(ENV, "TARGET_HPA", "950,925,900,875,850,825,800,775,750,700,650,600,550,500"))
    min_raw_speed::Float64 = parse(Float64, get(ENV, "MIN_RAW_SPEED", "1.0"))
    max_pmid_error_hpa::Float64 = parse(Float64, get(ENV, "MAX_PMID_ERROR_HPA", "20.0"))
end

parse_region(s::AbstractString) = Tuple(parse.(Float64, split(s, ",")))::NTuple{4, Float64}
parse_list(s::AbstractString) = parse.(Float64, split(s, ","))

lon180(lon) = mod(lon + 180.0, 360.0) - 180.0
lon360(lon) = mod(lon, 360.0)
bearing_deg(u, v) = mod(rad2deg(atan(u, v)), 360.0)
angular_diff(a, b) = mod(a - b + 180.0, 360.0) - 180.0

function nearest_bracket(v::AbstractVector{<:Real}, x::Real; periodic::Bool=false)
    n = length(v)
    if periodic
        dx = Float64(v[2] - v[1])
        xw = lon360(x)
        x0 = lon360(Float64(v[1]))
        pos = mod((xw - x0) / dx, n)
        i0 = floor(Int, pos) + 1
        i1 = i0 == n ? 1 : i0 + 1
        f = pos - floor(pos)
        return i0, i1, f
    end

    asc = Float64(v[end]) > Float64(v[1])
    if asc
        x <= v[1] && return 1, 1, 0.0
        x >= v[end] && return n, n, 0.0
        i1 = searchsortedfirst(v, x)
        i0 = i1 - 1
        f = (x - v[i0]) / (v[i1] - v[i0])
        return i0, i1, Float64(f)
    else
        x >= v[1] && return 1, 1, 0.0
        x <= v[end] && return n, n, 0.0
        vr = reverse(v)
        ir1 = searchsortedfirst(vr, x)
        ir0 = ir1 - 1
        i0 = n - ir0 + 1
        i1 = n - ir1 + 1
        f = (x - v[i0]) / (v[i1] - v[i0])
        return i0, i1, Float64(f)
    end
end

function interp_raw(var::Array{FT, 3}, lons, lats, levels, lon, lat, pressure_hpa) where FT
    li0, li1, lf = nearest_bracket(lons, lon; periodic=true)
    aj0, aj1, af = nearest_bracket(lats, lat)
    pk0, pk1, pf = nearest_bracket(levels, pressure_hpa)

    value_at(k) = begin
        v00 = Float64(var[li0, aj0, k])
        v10 = Float64(var[li1, aj0, k])
        v01 = Float64(var[li0, aj1, k])
        v11 = Float64(var[li1, aj1, k])
        v0 = muladd(lf, v10 - v00, v00)
        v1 = muladd(lf, v11 - v01, v01)
        muladd(af, v1 - v0, v0)
    end

    v0 = value_at(pk0)
    v1 = value_at(pk1)
    return muladd(pf, v1 - v0, v0)
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
    return (; u = ax * nx_east + ay * ny_east,
            v = ax * nx_north + ay * ny_north)
end

function cell_wind(raw, mesh, basis, dt_factor, p, i, j, k)
    area = Float64(mesh.cell_areas[i, j])
    dp = Float64(raw.m[p][i, j, k]) * GRAV / area
    xsec_x = Float64(mesh.Δy[i, j]) * dp / GRAV * dt_factor
    xsec_y = Float64(mesh.Δx[i, j]) * dp / GRAV * dt_factor
    up = xsec_x > 0 ? 0.5 * (Float64(raw.am[p][i, j, k]) + Float64(raw.am[p][i + 1, j, k])) / xsec_x : 0.0
    vp = xsec_y > 0 ? 0.5 * (Float64(raw.bm[p][i, j, k]) + Float64(raw.bm[p][i, j + 1, k])) / xsec_y : 0.0
    geo = face_normal_to_geographic(up, vp, basis, p, i, j)
    return (; u = geo.u, v = geo.v, speed = hypot(geo.u, geo.v), bearing = bearing_deg(geo.u, geo.v))
end

function pressure_midpoints_hpa(raw, mesh, p, i, j)
    Nz = size(raw.m[p], 3)
    area = Float64(mesh.cell_areas[i, j])
    mids = Vector{Float64}(undef, Nz)
    ptop = 0.0
    for k in 1:Nz
        dp = Float64(raw.m[p][i, j, k]) * GRAV / area
        mids[k] = (ptop + 0.5 * dp) / 100.0
        ptop += dp
    end
    return mids
end

function in_region(lon, lat, region)
    lon_min, lon_max, lat_min, lat_max = region
    λ = lon180(lon)
    return lon_min <= λ <= lon_max && lat_min <= lat <= lat_max
end

function load_raw_uv(path, time_index)
    NCDataset(path, "r") do ds
        lons = Float64.(ds["longitude"][:])
        lats = Float64.(ds["latitude"][:])
        levels = Float64.(ds["pressure_level"][:])
        u = Array{Float32, 3}(ds["u"][:, :, :, time_index])
        v = Array{Float32, 3}(ds["v"][:, :, :, time_index])
        return (; lons, lats, levels, u, v)
    end
end

function summarize(rows, cfg::Config)
    open(joinpath(cfg.out_dir, "summary.csv"), "w") do io
        println(io, "target_hpa,n,mean_binary_speed,mean_raw_speed,mean_speed_bias,rmse_speed,mean_abs_bearing_error,p95_abs_bearing_error,mean_u_bias,mean_v_bias,corr_u,corr_v")
        for target in cfg.target_hpa
            group = [r for r in rows if r.target_hpa == target && abs(r.pmid_hpa - target) <= cfg.max_pmid_error_hpa]
            strong = [r for r in group if r.raw_speed >= cfg.min_raw_speed]
            isempty(group) && continue
            du = [r.binary_u - r.raw_u for r in group]
            dv = [r.binary_v - r.raw_v for r in group]
            ds = [r.binary_speed - r.raw_speed for r in group]
            be = [abs(angular_diff(r.binary_bearing, r.raw_bearing)) for r in strong]
            bu = [r.binary_u for r in group]
            ru = [r.raw_u for r in group]
            bv = [r.binary_v for r in group]
            rv = [r.raw_v for r in group]
            corr_u = length(group) > 1 ? cor(bu, ru) : NaN
            corr_v = length(group) > 1 ? cor(bv, rv) : NaN
            @printf(io, "%.1f,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
                    target, length(group), mean(r.binary_speed for r in group),
                    mean(r.raw_speed for r in group), mean(ds),
                    sqrt(mean(abs2, ds)),
                    isempty(be) ? NaN : mean(be),
                    isempty(be) ? NaN : quantile(be, 0.95),
                    mean(du), mean(dv), corr_u, corr_v)
        end
    end
end

function write_rows(path, rows)
    open(path, "w") do io
        println(io, "target_hpa,panel,i,j,k,lon,lat,ps_hpa,pmid_hpa,binary_u,binary_v,binary_speed,binary_bearing,raw_u,raw_v,raw_speed,raw_bearing,speed_bias,bearing_error")
        for r in rows
            @printf(io, "%.1f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
                    r.target_hpa, r.panel, r.i, r.j, r.k, r.lon, r.lat,
                    r.ps_hpa, r.pmid_hpa, r.binary_u, r.binary_v,
                    r.binary_speed, r.binary_bearing, r.raw_u, r.raw_v,
                    r.raw_speed, r.raw_bearing, r.binary_speed - r.raw_speed,
                    angular_diff(r.binary_bearing, r.raw_bearing))
        end
    end
end

function main()
    cfg = Config()
    mkpath(cfg.out_dir)

    raw_uv = load_raw_uv(cfg.raw_uv_nc, cfg.time_index)
    reader = TransportBinaryReader(cfg.era_cs_bin; FT=Float32)
    try
        raw = load_window!(reader, cfg.window)
        mesh = load_grid(reader; FT=Float32, arch=CPU(), Hp=0).horizontal
        basis = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), 6)
        lonlat = ntuple(p -> panel_cell_center_lonlat(mesh, p), 6)
        steps = reader.header.steps_per_window_by_window[cfg.window]
        dt_factor = AtmosTransport.MetDrivers.flux_application_seconds(
            reader.header.dt_met_seconds, steps,
            AtmosTransport.MetDrivers.flux_kind(reader))

        rows = NamedTuple[]
        for p in 1:6
            lons, lats = lonlat[p]
            for j in 1:mesh.Nc, i in 1:mesh.Nc
                lon = Float64(lons[i, j])
                lat = Float64(lats[i, j])
                in_region(lon, lat, cfg.region) || continue
                pmids = pressure_midpoints_hpa(raw, mesh, p, i, j)
                ps_hpa = Float64(raw.ps[p][i, j]) / 100.0
                for target in cfg.target_hpa
                    k = argmin(abs.(pmids .- target))
                    abs(pmids[k] - target) <= cfg.max_pmid_error_hpa || continue
                    bw = cell_wind(raw, mesh, basis, dt_factor, p, i, j, k)
                    ru = interp_raw(raw_uv.u, raw_uv.lons, raw_uv.lats, raw_uv.levels, lon, lat, pmids[k])
                    rv = interp_raw(raw_uv.v, raw_uv.lons, raw_uv.lats, raw_uv.levels, lon, lat, pmids[k])
                    rs = hypot(ru, rv)
                    push!(rows, (; target_hpa = target, panel = p, i, j, k,
                                  lon = lon180(lon), lat, ps_hpa, pmid_hpa = pmids[k],
                                  binary_u = bw.u, binary_v = bw.v,
                                  binary_speed = bw.speed, binary_bearing = bw.bearing,
                                  raw_u = ru, raw_v = rv, raw_speed = rs,
                                  raw_bearing = bearing_deg(ru, rv)))
                end
            end
        end

        rows_path = joinpath(cfg.out_dir, "samples.csv")
        write_rows(rows_path, rows)
        summarize(rows, cfg)
        open(joinpath(cfg.out_dir, "README.txt"), "w") do io
            println(io, "ERA C180 binary vs raw ERA pressure-level U/V")
            println(io, "Binary: ", cfg.era_cs_bin)
            println(io, "Raw U/V: ", cfg.raw_uv_nc)
            println(io, "Window: ", cfg.window, "  raw time index: ", cfg.time_index)
            println(io, "Region lon/lat: ", cfg.region)
            println(io, "Samples CSV: ", rows_path)
            println(io, "Summary CSV: ", joinpath(cfg.out_dir, "summary.csv"))
            println(io, "Each raw ERA U/V value is bilinearly interpolated in lon/lat and linearly interpolated in pressure to the binary layer midpoint pressure.")
        end

        println("Wrote ", cfg.out_dir)
        for line in eachline(joinpath(cfg.out_dir, "summary.csv"))
            println(line)
        end
    finally
        close(reader)
    end
end

main()
