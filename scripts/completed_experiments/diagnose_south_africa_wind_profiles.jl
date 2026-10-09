#!/usr/bin/env julia

module DiagnoseSouthAfricaWindProfiles

using CairoMakie
using NCDatasets
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))

using .AtmosTransport.Grids: CubedSphereMesh, panel_cell_local_tangent_basis
using .AtmosTransport.MetDrivers: TransportBinaryReader, load_window!, mesh_definition,
    flux_application_seconds, flux_kind

const GRAV = 9.80665
const R_EARTH_M = 6.371229e6
const DX_FIELD = Symbol(Char(0x0394), "x")
const DY_FIELD = Symbol(Char(0x0394), "y")

Base.@kwdef struct Config
    run::String = get(ENV, "RUN", "advdiff_ppm")
    hour::Float64 = parse(Float64, get(ENV, "HOUR", "60"))
    profile_window::Int = parse(Int, get(ENV, "PROFILE_WINDOW", "13"))
    top_fraction::Float64 = parse(Float64, get(ENV, "TOP_FRACTION", "0.20"))
    region::NTuple{4, Float64} = parse_region(get(ENV, "REGION", "10,40,-40,-20"))
    era_nc::String = get(ENV, "ERA_NC", "/temp1/c180_era5_geosgrid_cfl85_3d/$(get(ENV, "RUN", "advdiff_ppm")).nc")
    geos_nc::String = get(ENV, "GEOS_NC", "/temp1/c180_geosit_native_3d/$(get(ENV, "RUN", "advdiff_ppm")).nc")
    era_bin::String = get(ENV, "ERA_BIN", "/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211204_merged1000Pa_float32.bin")
    geos_bin::String = get(ENV, "GEOS_BIN", "/temp1/c180_geosit_native_v4_dec2021_f32/geos_transport_20211204_float32.bin")
    out_dir::String = get(ENV, "OUT_DIR", "/temp1/c180_south_africa_plume_direction")
    standard_pressures::Vector{Float64} = parse_float_list(get(ENV, "STANDARD_PRESSURES", "950,925,900,875,850,825,800,775,750,700,650,600,550,500"))
end

struct SourceSpec
    name::String
    nc_path::String
    bin_path::String
end

struct MaskSpec
    name::String
    mask::BitArray{3}
end

lon180(lon) = mod(lon + 180.0, 360.0) - 180.0
bearing_deg(u_east, v_north) = mod(rad2deg(atan(u_east, v_north)), 360.0)

function parse_region(s::AbstractString)
    vals = parse_float_list(s)
    length(vals) == 4 || error("REGION must be lon_min,lon_max,lat_min,lat_max; got `$s`")
    return (vals[1], vals[2], vals[3], vals[4])
end

function parse_float_list(s::AbstractString)
    return [parse(Float64, strip(part)) for part in split(s, ",") if !isempty(strip(part))]
end

function time_values(ds)
    return Float64.(collect(ds["time"].var[:]))
end

function time_index(ds, hour)
    times = time_values(ds)
    idx = findfirst(t -> isapprox(t, hour; atol=1e-6), times)
    idx === nothing && error("hour $(hour) not found in $(times)")
    return idx
end

function load_snapshot(path::AbstractString, hour::Float64)
    NCDataset(path, "r") do ds
        tidx = time_index(ds, hour)
        return (;
            lon = Float64.(ds["lons"][:, :, :]),
            lat = Float64.(ds["lats"][:, :, :]),
            area = Float64.(ds["cell_area"][:, :, :]),
            column = Float64.(ds["co2_fossil_column_mean"][:, :, :, tidx]) .* 1e6,
            tidx,
        )
    end
end

function region_mask(lon, lat, region)
    lon_min, lon_max, lat_min, lat_max = region
    lonm = lon180.(lon)
    return BitArray((lonm .>= lon_min) .& (lonm .<= lon_max) .&
                    (lat .>= lat_min) .& (lat .<= lat_max))
end

function top_column_mask(column, base_mask; fraction)
    vals = sort([v for v in column[base_mask] if isfinite(v) && v > 0])
    isempty(vals) && return falses(size(column))
    qidx = clamp(floor(Int, (1.0 - fraction) * length(vals)), 1, length(vals))
    threshold = vals[qidx]
    return BitArray(base_mask .& (column .>= threshold))
end

function build_masks(era, geos, cfg::Config)
    box = region_mask(era.lon, era.lat, cfg.region)
    era_top = top_column_mask(era.column, box; fraction=cfg.top_fraction)
    geos_top = top_column_mask(geos.column, box; fraction=cfg.top_fraction)
    suffix = "t$(round(Int, cfg.hour))"
    return MaskSpec[
        MaskSpec("region_box", box),
        MaskSpec("union_top20_column_$(suffix)", BitArray(era_top .| geos_top)),
        MaskSpec("era_top20_column_$(suffix)", era_top),
        MaskSpec("geos_top20_column_$(suffix)", geos_top),
    ]
end

function source_mesh(reader)
    return CubedSphereMesh(; FT=Float64, Nc=reader.header.geometry.Nc, Hp=0,
                           radius=R_EARTH_M, definition=mesh_definition(reader))
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

function wind_at_cell_level(window, mesh, basis, dt_factor, panel, i, j, k)
    mass = Float64(window.m[panel][i, j, k])
    area = Float64(mesh.cell_areas[i, j])
    dx = getfield(mesh, DX_FIELD)
    dy = getfield(mesh, DY_FIELD)
    xsec_x = Float64(dy[i, j]) * mass / area * dt_factor
    xsec_y = Float64(dx[i, j]) * mass / area * dt_factor
    up = xsec_x > 0 ? 0.5 * (Float64(window.am[panel][i, j, k]) +
                              Float64(window.am[panel][i + 1, j, k])) / xsec_x : 0.0
    vp = xsec_y > 0 ? 0.5 * (Float64(window.bm[panel][i, j, k]) +
                              Float64(window.bm[panel][i, j + 1, k])) / xsec_y : 0.0
    return face_normal_to_geographic(up, vp, basis, panel, i, j)
end

function pressure_midpoint_hpa(window, mesh, panel, i, j, k)
    area = Float64(mesh.cell_areas[i, j])
    ptop = 0.0
    for kk in 1:k
        dp = Float64(window.m[panel][i, j, kk]) * GRAV / area / 100.0
        kk == k && return ptop + 0.5 * dp
        ptop += dp
    end
    return NaN
end

function profile_rows(source::SourceSpec, masks::Vector{MaskSpec}, area, cfg::Config)
    reader = TransportBinaryReader(source.bin_path; FT=Float32)
    try
        mesh = source_mesh(reader)
        window = load_window!(reader, cfg.profile_window)
        basis = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), 6)
        steps = reader.header.steps_per_window_by_window[cfg.profile_window]
        dt_factor = flux_application_seconds(reader.header.dt_met_seconds, steps,
                                              flux_kind(reader))
        rows = NamedTuple[]
        for mask_spec in masks
            idxs = findall(mask_spec.mask)
            for k in 1:reader.header.nlevel
                pressure_num = 0.0
                u_num = 0.0
                v_num = 0.0
                den = 0.0
                n_used = 0
                for idx in idxs
                    i, j, panel = Tuple(idx)
                    w = Float64(area[idx])
                    w > 0 || continue
                    pres = pressure_midpoint_hpa(window, mesh, panel, i, j, k)
                    wind = wind_at_cell_level(window, mesh, basis, dt_factor, panel, i, j, k)
                    pressure_num += pres * w
                    u_num += wind.u * w
                    v_num += wind.v * w
                    den += w
                    n_used += 1
                end
                u = den > 0 ? u_num / den : NaN
                v = den > 0 ? v_num / den : NaN
                push!(rows, (;
                    source = source.name,
                    mask = mask_spec.name,
                    averaging = "area_mean",
                    k,
                    pressure_hpa = den > 0 ? pressure_num / den : NaN,
                    u_east_ms = u,
                    v_north_ms = v,
                    speed_ms = hypot(u, v),
                    bearing_deg = bearing_deg(u, v),
                    total_weight = den,
                    sample_cells = n_used,
                    profile_window = cfg.profile_window,
                    binary_path = source.bin_path,
                    snapshot_path = source.nc_path,
                ))
            end
        end
        return rows
    finally
        close(reader)
    end
end

function write_profile_csv(path, rows)
    open(path, "w") do io
        println(io, "source,mask,averaging,k,pressure_hpa,u_east_ms,v_north_ms,speed_ms,bearing_deg,total_weight,sample_cells,profile_window,binary_path,snapshot_path")
        for r in rows
            @printf(io, "%s,%s,%s,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d,%s,%s\n",
                    r.source, r.mask, r.averaging, r.k, r.pressure_hpa,
                    r.u_east_ms, r.v_north_ms, r.speed_ms, r.bearing_deg,
                    r.total_weight, r.sample_cells, r.profile_window,
                    r.binary_path, r.snapshot_path)
        end
    end
    return path
end

function row_groups(rows, mask_name, source_name)
    g = [r for r in rows if r.mask == mask_name && r.source == source_name &&
                         isfinite(r.pressure_hpa) && isfinite(r.u_east_ms) &&
                         isfinite(r.v_north_ms)]
    sort!(g; by = r -> r.pressure_hpa)
    return g
end

function interpolate_uv(rows, target_hpa)
    isempty(rows) && return nothing
    target_hpa < first(rows).pressure_hpa && return nothing
    target_hpa > last(rows).pressure_hpa && return nothing
    for i in 1:(length(rows) - 1)
        p0 = rows[i].pressure_hpa
        p1 = rows[i + 1].pressure_hpa
        if p0 <= target_hpa <= p1
            f = (target_hpa - p0) / (p1 - p0)
            u = rows[i].u_east_ms + f * (rows[i + 1].u_east_ms - rows[i].u_east_ms)
            v = rows[i].v_north_ms + f * (rows[i + 1].v_north_ms - rows[i].v_north_ms)
            return (; u, v, speed = hypot(u, v), bearing = bearing_deg(u, v))
        end
    end
    return nothing
end

function angular_diff(a, b)
    return mod(a - b + 180.0, 360.0) - 180.0
end

function write_standard_pressure_csv(path, rows, masks, pressures)
    open(path, "w") do io
        println(io, "mask,pressure_hpa,era_speed_ms,era_bearing_deg,geos_speed_ms,geos_bearing_deg,bearing_geos_minus_era_deg,speed_geos_minus_era_ms")
        for mask_spec in masks
            era = row_groups(rows, mask_spec.name, "ERA")
            geos = row_groups(rows, mask_spec.name, "GEOS")
            for p in pressures
                ev = interpolate_uv(era, p)
                gv = interpolate_uv(geos, p)
                if ev === nothing || gv === nothing
                    @printf(io, "%s,%.1f,,,,,,\n", mask_spec.name, p)
                else
                    @printf(io, "%s,%.1f,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
                            mask_spec.name, p, ev.speed, ev.bearing, gv.speed,
                            gv.bearing, angular_diff(gv.bearing, ev.bearing),
                            gv.speed - ev.speed)
                end
            end
        end
    end
    return path
end

function write_mask_plot(path, rows, mask_name)
    fig = Figure(size = (1120, 540), fontsize = 13)
    ax_speed = Axis(fig[1, 1], xlabel = "speed [m s^-1]",
                    ylabel = "pressure [hPa]",
                    title = "$(mask_name): regional wind speed")
    ax_dir = Axis(fig[1, 2], xlabel = "bearing [deg clockwise from north]",
                  ylabel = "pressure [hPa]",
                  title = "$(mask_name): regional wind direction")
    colors = Dict("ERA" => :dodgerblue3, "GEOS" => :firebrick3)
    labels = Dict("ERA" => "ERA5-derived Met", "GEOS" => "GEOS-IT Met")
    for source in ("ERA", "GEOS")
        g = row_groups(rows, mask_name, source)
        isempty(g) && continue
        p = [r.pressure_hpa for r in g]
        speed = [r.speed_ms for r in g]
        bearing = [r.bearing_deg for r in g]
        lines!(ax_speed, speed, p; linewidth = 2.5, color = colors[source], label = labels[source])
        scatter!(ax_speed, speed, p; markersize = 4, color = colors[source])
        lines!(ax_dir, bearing, p; linewidth = 2.5, color = colors[source], label = labels[source])
        scatter!(ax_dir, bearing, p; markersize = 4, color = colors[source])
    end
    ylims!(ax_speed, 1000, 0)
    ylims!(ax_dir, 1000, 0)
    xlims!(ax_dir, 0, 360)
    ax_dir.xticks = 0:45:360
    axislegend(ax_speed, position = :rt)
    axislegend(ax_dir, position = :rt)
    Label(fig[2, 1:2], "Vector mean over selected C180 cells; area weighted, not tracer/emission weighted.",
          fontsize = 12)
    save(path, fig, px_per_unit = 2)
    return path
end

function write_plots(out_dir, rows, masks, cfg::Config)
    paths = String[]
    suffix = "t$(round(Int, cfg.hour))"
    for mask_spec in masks
        path = joinpath(out_dir, "south_africa_$(suffix)_$(mask_spec.name)_unweighted_wind_profile.png")
        write_mask_plot(path, rows, mask_spec.name)
        push!(paths, path)
    end
    return paths
end

function write_readme(path, cfg::Config, masks, outputs)
    open(path, "w") do io
        lon_min, lon_max, lat_min, lat_max = cfg.region
        println(io, "South Africa unweighted regional wind profile diagnostic")
        println(io, "Run: ", cfg.run)
        println(io, "Hour: ", cfg.hour)
        println(io, "Profile window: ", cfg.profile_window)
        println(io, "Region: lon ", lon_min, " to ", lon_max,
                ", lat ", lat_min, " to ", lat_max)
        println(io, "Top-column fraction: ", cfg.top_fraction)
        println(io, "ERA_NC: ", cfg.era_nc)
        println(io, "GEOS_NC: ", cfg.geos_nc)
        println(io, "ERA_BIN: ", cfg.era_bin)
        println(io, "GEOS_BIN: ", cfg.geos_bin)
        println(io, "Masks:")
        for m in masks
            println(io, "  ", m.name, ": ", count(m.mask), " cells")
        end
        println(io, "Outputs:")
        for output in outputs
            println(io, "  ", output)
        end
    end
    return path
end

function main(; kwargs...)
    cfg = Config(; kwargs...)
    mkpath(cfg.out_dir)
    era = load_snapshot(cfg.era_nc, cfg.hour)
    geos = load_snapshot(cfg.geos_nc, cfg.hour)
    masks = build_masks(era, geos, cfg)
    sources = (
        SourceSpec("ERA", cfg.era_nc, cfg.era_bin),
        SourceSpec("GEOS", cfg.geos_nc, cfg.geos_bin),
    )
    rows = NamedTuple[]
    for source in sources
        append!(rows, profile_rows(source, masks, era.area, cfg))
    end

    suffix = "t$(round(Int, cfg.hour))"
    profile_csv = write_profile_csv(joinpath(cfg.out_dir, "south_africa_$(suffix)_unweighted_regional_wind_profiles.csv"), rows)
    standard_csv = write_standard_pressure_csv(joinpath(cfg.out_dir, "south_africa_$(suffix)_unweighted_wind_profile_standard_pressures.csv"),
                                               rows, masks, cfg.standard_pressures)
    plot_paths = write_plots(cfg.out_dir, rows, masks, cfg)
    readme = write_readme(joinpath(cfg.out_dir, "south_africa_$(suffix)_unweighted_wind_profiles_README.txt"),
                          cfg, masks, vcat([profile_csv, standard_csv], plot_paths))

    println("Wrote:")
    println("  ", profile_csv)
    println("  ", standard_csv)
    for path in plot_paths
        println("  ", path)
    end
    println("  ", readme)
    for mask_spec in masks
        era_rows = row_groups(rows, mask_spec.name, "ERA")
        geos_rows = row_groups(rows, mask_spec.name, "GEOS")
        @printf("%s: ERA %d levels %.3f-%.3f hPa, GEOS %d levels %.3f-%.3f hPa, cells %d\n",
                mask_spec.name, length(era_rows), minimum(r.pressure_hpa for r in era_rows),
                maximum(r.pressure_hpa for r in era_rows), length(geos_rows),
                minimum(r.pressure_hpa for r in geos_rows),
                maximum(r.pressure_hpa for r in geos_rows), count(mask_spec.mask))
    end
    return (; profile_csv, standard_csv, plot_paths, readme)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    DiagnoseSouthAfricaWindProfiles.main()
end
