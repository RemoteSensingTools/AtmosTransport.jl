#!/usr/bin/env julia
# NOTE(flux_kind): this script reads LL/ERA binaries, which are always
# flux_kind=substep_mass_amount (full_window_mass_amount is written only
# by the GEOS-CS preprocessor, and the LL runtime rejects it). The
# unconditional dt/(2*steps) normalization here is therefore correct.


module CompareSpectralNativeVsCDSLL

using Dates
using NCDatasets
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))

using .AtmosTransport.Grids: cell_area
using .AtmosTransport.MetDrivers: TransportBinaryReader, load_grid, load_window!
using .AtmosTransport.Preprocessing: build_target_geometry, load_ab_coefficients,
    read_day_spectral, recover_ll_cell_center_winds!, spectral_to_grid!,
    target_spectral_truncation, vod2uv!

const GRAV = 9.80665

Base.@kwdef struct Config
    out_dir::String = get(ENV, "OUT_DIR", "/temp1/c180_era_wind_preprocessing_investigation")
    spectral_cache_dir::String = get(ENV, "SPECTRAL_CACHE_DIR", "/temp1/atmos_transport/preproc_profile/spectral_cache")
    spectral_dir::String = expanduser(get(ENV, "SPECTRAL_DIR", "~/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly"))
    coeff_path::String = get(ENV, "COEFF_PATH", "config/era5_L137_coefficients.toml")
    cds_uv::String = get(ENV, "CDS_UV", "/temp1/era5_cds_global_winds_20211204/era5_pressure_level_uv_global_20211204.nc")
    ll_bin::String = get(ENV, "LL_BIN", "/home/cfranken/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_cfl85_dec2021_f32_tm5_surface/era5_transport_20211204_merged1000Pa_float32.bin")
    date::Date = Date(get(ENV, "DATE", "2021-12-04"))
    hour::Int = parse(Int, get(ENV, "HOUR_UTC", "12"))
    cds_time_index::Int = parse(Int, get(ENV, "CDS_TIME_INDEX", "13"))
    ll_window::Int = parse(Int, get(ENV, "LL_WINDOW", "13"))
    region::NTuple{4, Float64} = parse_region(get(ENV, "REGION", "10,40,-40,-20"))
    pressures::Vector{Float64} = parse_float_list(get(ENV, "PRESSURES_HPA", "950,925,900,875,850,825,800,775,750,700,650,600,550,500"))
end

ProfileRow = NamedTuple{(:source, :pressure_hpa, :native_level, :u_east_ms, :v_north_ms,
                         :speed_ms, :bearing_deg, :sample_cells),
                        Tuple{String, Float64, Int, Float64, Float64, Float64, Float64, Int}}

lon180(lon) = mod(Float64(lon) + 180.0, 360.0) - 180.0
bearing_deg(u, v) = mod(rad2deg(atan(u, v)), 360.0)
speed_ms(u, v) = hypot(u, v)
angular_diff(a, b) = mod(a - b + 180.0, 360.0) - 180.0

function parse_float_list(s::AbstractString)
    return [parse(Float64, strip(part)) for part in split(s, ",") if !isempty(strip(part))]
end

function parse_region(s::AbstractString)
    vals = parse_float_list(s)
    length(vals) == 4 || error("REGION must be lon_min,lon_max,lat_min,lat_max; got `$s`")
    return (vals[1], vals[2], vals[3], vals[4])
end

function region_indices(lons, lats, region)
    lon_min, lon_max, lat_min, lat_max = region
    ii = findall(lon -> lon_min <= lon180(lon) <= lon_max, lons)
    jj = findall(lat -> lat_min <= Float64(lat) <= lat_max, lats)
    isempty(ii) && error("empty longitude selection for region $region")
    isempty(jj) && error("empty latitude selection for region $region")
    return ii, jj
end

function area_mean_uv(u, v, weights, lon_idxs, lat_local_idxs)
    unum = 0.0
    vnum = 0.0
    den = 0.0
    n = 0
    for jl in lat_local_idxs, i in lon_idxs
        uu = Float64(u[i, jl])
        vv = Float64(v[i, jl])
        isfinite(uu) && isfinite(vv) || continue
        w = Float64(weights[i, jl])
        unum += uu * w
        vnum += vv * w
        den += w
        n += 1
    end
    uu = unum / den
    vv = vnum / den
    return uu, vv, n
end

function cds_rows(cfg::Config)
    lon_min, lon_max, lat_min, lat_max = cfg.region
    NCDataset(cfg.cds_uv, "r") do ds
        lons = Float64.(ds["longitude"][:])
        lats = Float64.(ds["latitude"][:])
        pressures = Float64.(ds["pressure_level"][:])
        lon_idxs = findall(lon -> lon_min <= lon180(lon) <= lon_max, lons)
        lat_idxs = findall(lat -> lat_min <= lat <= lat_max, lats)
        rows = ProfileRow[]
        for (k, pressure_hpa) in enumerate(pressures)
            unum = 0.0
            vnum = 0.0
            den = 0.0
            n = 0
            for j in lat_idxs, i in lon_idxs
                uu = Float64(ds["u"][i, j, k, cfg.cds_time_index])
                vv = Float64(ds["v"][i, j, k, cfg.cds_time_index])
                isfinite(uu) && isfinite(vv) || continue
                w = cosd(lats[j])
                unum += uu * w
                vnum += vv * w
                den += w
                n += 1
            end
            uu = unum / den
            vv = vnum / den
            push!(rows, (source="CDS_RAW_ERA5", pressure_hpa=pressure_hpa, native_level=0,
                         u_east_ms=uu, v_north_ms=vv, speed_ms=speed_ms(uu, vv),
                         bearing_deg=bearing_deg(uu, vv), sample_cells=n))
        end
        return rows
    end
end

function ll_binary_rows(cfg::Config)
    reader = TransportBinaryReader(cfg.ll_bin; FT=Float32)
    try
        grid = load_grid(reader; FT=Float32)
        mesh = grid.horizontal
        m, ps, fluxes = load_window!(reader, cfg.ll_window)
        nx, ny, nz = size(m)
        u = Array{Float32}(undef, nx, ny, nz)
        v = Array{Float32}(undef, nx, ny, nz)
        steps = reader.header.steps_per_window_by_window[cfg.ll_window]
        dt_factor = Float32(reader.header.dt_met_seconds / (2.0 * steps))
        recover_ll_cell_center_winds!(u, v, fluxes.am, fluxes.bm, ps,
            Float32.(reader.header.A_ifc), Float32.(reader.header.B_ifc),
            Float32.(mesh.φᶜ), Float32(mesh.radius * deg2rad(mesh.Δφ)),
            Float32(deg2rad(mesh.Δλ)), Float32(mesh.radius), Float32(GRAV),
            dt_factor)

        lon_idxs, lat_idxs = region_indices(mesh.λᶜ, mesh.φᶜ, cfg.region)
        area = [Float64(cell_area(mesh, i, j)) for i in 1:mesh.Nx, j in 1:mesh.Ny]
        rows = ProfileRow[]
        for k in 1:nz
            pnum = 0.0
            unum = 0.0
            vnum = 0.0
            den = 0.0
            n = 0
            for j in lat_idxs, i in lon_idxs
                p0 = Float64(reader.header.A_ifc[k] + reader.header.B_ifc[k] * ps[i, j])
                p1 = Float64(reader.header.A_ifc[k + 1] + reader.header.B_ifc[k + 1] * ps[i, j])
                w = area[i, j]
                pnum += 0.5 * (p0 + p1) / 100.0 * w
                unum += Float64(u[i, j, k]) * w
                vnum += Float64(v[i, j, k]) * w
                den += w
                n += 1
            end
            uu = unum / den
            vv = vnum / den
            push!(rows, (source="ERA_LL_BINARY", pressure_hpa=pnum / den, native_level=k,
                         u_east_ms=uu, v_north_ms=vv, speed_ms=speed_ms(uu, vv),
                         bearing_deg=bearing_deg(uu, vv), sample_cells=n))
        end
        return rows
    finally
        close(reader)
    end
end

function interpolate_uv(rows, target_hpa)
    g = sort([r for r in rows if isfinite(r.pressure_hpa)], by = r -> r.pressure_hpa)
    isempty(g) && return nothing
    target_hpa < first(g).pressure_hpa && return nothing
    target_hpa > last(g).pressure_hpa && return nothing
    for i in 1:(length(g) - 1)
        p0 = g[i].pressure_hpa
        p1 = g[i + 1].pressure_hpa
        if p0 <= target_hpa <= p1
            f = (target_hpa - p0) / (p1 - p0)
            u = g[i].u_east_ms + f * (g[i + 1].u_east_ms - g[i].u_east_ms)
            v = g[i].v_north_ms + f * (g[i + 1].v_north_ms - g[i].v_north_ms)
            return (; u, v, speed=speed_ms(u, v), bearing=bearing_deg(u, v))
        end
    end
    return nothing
end

function exact_or_interp(rows, target_hpa)
    idx = findfirst(r -> isapprox(r.pressure_hpa, target_hpa; atol=1e-6), rows)
    idx !== nothing && return (u=rows[idx].u_east_ms, v=rows[idx].v_north_ms,
                               speed=rows[idx].speed_ms, bearing=rows[idx].bearing_deg)
    return interpolate_uv(rows, target_hpa)
end

function spectral_native_rows(cfg::Config)
    grid = build_target_geometry(Val(:latlon), Dict("nlon" => 720, "nlat" => 361), Float64)
    lon_idxs, lat_global_idxs = region_indices(grid.lons, grid.lats, cfg.region)
    lats_subset = grid.lats[lat_global_idxs]
    lat_local_idxs = collect(eachindex(lats_subset))
    weights = [Float64(grid.area[i, lat_global_idxs[jl]]) for i in 1:length(grid.lons), jl in eachindex(lats_subset)]

    date_str = Dates.format(cfg.date, "yyyymmdd")
    vo_d_path = joinpath(cfg.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(cfg.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")
    T_target = target_spectral_truncation(grid)
    spec = read_day_spectral(vo_d_path, lnsp_path; T_target, cache_dir=cfg.spectral_cache_dir)
    haskey(spec.lnsp_all, cfg.hour) || error("hour $(cfg.hour) not found in spectral file; hours=$(sort(collect(keys(spec.lnsp_all))))")
    T = spec.T

    field = zeros(Float64, length(grid.lons), length(lats_subset))
    P = zeros(Float64, T + 1, T + 1)
    fft = zeros(ComplexF64, length(grid.lons))
    sp_shift = deg2rad(grid.lons[1])

    spectral_to_grid!(field, spec.lnsp_all[cfg.hour], T, lats_subset, length(grid.lons), P, fft;
                      lon_shift_rad=sp_shift)
    sp = exp.(field)
    ab = load_ab_coefficients(cfg.coeff_path, 1:137)

    mean_p_by_level = Float64[]
    for k in 1:137
        pnum = 0.0
        den = 0.0
        for jl in lat_local_idxs, i in lon_idxs
            p0 = ab.a_ifc[k] + ab.b_ifc[k] * sp[i, jl]
            p1 = ab.a_ifc[k + 1] + ab.b_ifc[k + 1] * sp[i, jl]
            w = weights[i, jl]
            pnum += 0.5 * (p0 + p1) / 100.0 * w
            den += w
        end
        push!(mean_p_by_level, pnum / den)
    end

    selected_levels = sort(unique([argmin(abs.(mean_p_by_level .- p)) for p in cfg.pressures]))
    variants = (
        ("SPECTRAL_NATIVE_CURRENT", false, sp_shift, 1.0),
        ("SPECTRAL_NATIVE_CONJ_COEFF", true, sp_shift, 1.0),
        ("SPECTRAL_NATIVE_NEG_SHIFT", false, -sp_shift, 1.0),
        ("SPECTRAL_NATIVE_SHIFT_180", false, sp_shift + pi, 1.0),
        ("SPECTRAL_NATIVE_V_SIGN_FLIP", false, sp_shift, -1.0),
    )

    u_spec = zeros(ComplexF64, T + 1, T + 1)
    v_spec = zeros(ComplexF64, T + 1, T + 1)
    u_cos = similar(field)
    v_cos = similar(field)
    rows = ProfileRow[]

    for k in selected_levels
        vo = @view spec.vo_by_hour[cfg.hour][:, :, k]
        dd = @view spec.d_by_hour[cfg.hour][:, :, k]
        for (name, use_conj, shift, vfac) in variants
            if use_conj
                vod2uv!(u_spec, v_spec, conj.(vo), conj.(dd), T)
            else
                vod2uv!(u_spec, v_spec, vo, dd, T)
            end
            spectral_to_grid!(u_cos, u_spec, T, lats_subset, length(grid.lons), P, fft;
                              lon_shift_rad=shift)
            spectral_to_grid!(v_cos, v_spec, T, lats_subset, length(grid.lons), P, fft;
                              lon_shift_rad=shift)
            for jl in eachindex(lats_subset)
                c = cosd(lats_subset[jl])
                @views u_cos[:, jl] ./= c
                @views v_cos[:, jl] ./= c
                @views v_cos[:, jl] .*= vfac
            end
            uu, vv, n = area_mean_uv(u_cos, v_cos, weights, lon_idxs, lat_local_idxs)
            push!(rows, (source=name, pressure_hpa=mean_p_by_level[k], native_level=k,
                         u_east_ms=uu, v_north_ms=vv, speed_ms=speed_ms(uu, vv),
                         bearing_deg=bearing_deg(uu, vv), sample_cells=n))
        end
    end

    return rows, mean_p_by_level
end

function write_profile_csv(path, rows)
    open(path, "w") do io
        println(io, "source,pressure_hpa,native_level,u_east_ms,v_north_ms,speed_ms,bearing_deg,sample_cells")
        for r in rows
            @printf(io, "%s,%.12g,%d,%.12g,%.12g,%.12g,%.12g,%d\n",
                    r.source, r.pressure_hpa, r.native_level, r.u_east_ms, r.v_north_ms,
                    r.speed_ms, r.bearing_deg, r.sample_cells)
        end
    end
    return path
end

function write_standard_csv(path, sources, pressures)
    open(path, "w") do io
        names = ["CDS_RAW_ERA5", "ERA_LL_BINARY", "SPECTRAL_NATIVE_CURRENT",
                 "SPECTRAL_NATIVE_CONJ_COEFF", "SPECTRAL_NATIVE_NEG_SHIFT",
                 "SPECTRAL_NATIVE_SHIFT_180", "SPECTRAL_NATIVE_V_SIGN_FLIP"]
        println(io, "pressure_hpa," * join([n * "_speed_ms," * n * "_bearing_deg" for n in names], ",") *
                    ",ll_minus_cds_bearing_deg,spectral_current_minus_cds_bearing_deg")
        for p in pressures
            vals = [exact_or_interp(sources[n], p) for n in names]
            print(io, @sprintf("%.1f", p))
            for v in vals
                if v === nothing
                    print(io, ",,")
                else
                    print(io, @sprintf(",%.8g,%.8g", v.speed, v.bearing))
                end
            end
            if vals[1] === nothing || vals[2] === nothing || vals[3] === nothing
                print(io, ",,")
            else
                print(io, @sprintf(",%.8g,%.8g",
                    angular_diff(vals[2].bearing, vals[1].bearing),
                    angular_diff(vals[3].bearing, vals[1].bearing)))
            end
            println(io)
        end
    end
    return path
end

function main(; kwargs...)
    cfg = Config(; kwargs...)
    mkpath(cfg.out_dir)
    cds = cds_rows(cfg)
    ll = ll_binary_rows(cfg)
    spectral, mean_p_by_level = spectral_native_rows(cfg)
    sources = Dict("CDS_RAW_ERA5" => cds, "ERA_LL_BINARY" => ll)
    for name in unique(r.source for r in spectral)
        sources[name] = [r for r in spectral if r.source == name]
    end

    profile_csv = write_profile_csv(joinpath(cfg.out_dir, "south_africa_box_spectral_native_cds_ll_profiles.csv"),
                                    vcat(cds, ll, spectral))
    standard_csv = write_standard_csv(joinpath(cfg.out_dir, "south_africa_box_spectral_native_cds_ll_standard_pressures.csv"),
                                      sources, cfg.pressures)
    level_csv = joinpath(cfg.out_dir, "south_africa_box_spectral_native_level_pressures.csv")
    open(level_csv, "w") do io
        println(io, "native_level,region_mean_pressure_hpa")
        for (k, p) in enumerate(mean_p_by_level)
            @printf(io, "%d,%.12g\n", k, p)
        end
    end

    println("Wrote:")
    println("  ", profile_csv)
    println("  ", standard_csv)
    println("  ", level_csv)
    println("Key pressures:")
    for p in (900.0, 850.0, 800.0, 750.0, 700.0)
        rv = exact_or_interp(cds, p)
        lv = interpolate_uv(ll, p)
        sv = interpolate_uv(sources["SPECTRAL_NATIVE_CURRENT"], p)
        rv === nothing || lv === nothing || sv === nothing && continue
        @printf("%4.0f hPa  CDS %5.2f@%6.1f  spectral %5.2f@%6.1f ddir %+6.1f  LL %5.2f@%6.1f ddir %+6.1f\n",
                p, rv.speed, rv.bearing, sv.speed, sv.bearing,
                angular_diff(sv.bearing, rv.bearing), lv.speed, lv.bearing,
                angular_diff(lv.bearing, rv.bearing))
    end
    return (; profile_csv, standard_csv, level_csv)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CompareSpectralNativeVsCDSLL.main()
end
