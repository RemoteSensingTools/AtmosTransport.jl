#!/usr/bin/env julia
# NOTE(flux_kind): this script reads LL/ERA binaries, which are always
# flux_kind=substep_mass_amount (full_window_mass_amount is written only
# by the GEOS-CS preprocessor, and the LL runtime rejects it). The
# unconditional dt/(2*steps) normalization here is therefore correct.


module IsolateERALLToCSWindError

using CairoMakie
using NCDatasets
using Printf

include(joinpath(@__DIR__, "diagnose_south_africa_wind_profiles.jl"))

const D = DiagnoseSouthAfricaWindProfiles
const AT = D.AtmosTransport

using .D.AtmosTransport.Grids: cell_area
using .D.AtmosTransport.MetDrivers: TransportBinaryReader, load_grid, load_window!
using .D.AtmosTransport.Preprocessing: recover_ll_cell_center_winds!

const GRAV = 9.80665

Base.@kwdef struct Config
    out_dir::String = get(ENV, "OUT_DIR", "/temp1/c180_era_wind_preprocessing_investigation")
    cds_uv::String = get(ENV, "CDS_UV", "/temp1/era5_cds_global_winds_20211204/era5_pressure_level_uv_global_20211204.nc")
    ll_bin::String = get(ENV, "LL_BIN", "/home/cfranken/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_cfl85_dec2021_f32_tm5_surface/era5_transport_20211204_merged1000Pa_float32.bin")
    era_nc::String = get(ENV, "ERA_NC", "/temp1/c180_era5_geosgrid_cfl85_3d/advdiff_ppm.nc")
    geos_nc::String = get(ENV, "GEOS_NC", "/temp1/c180_geosit_native_3d/advdiff_ppm.nc")
    era_cs_bin::String = get(ENV, "ERA_CS_BIN", "/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211204_merged1000Pa_float32.bin")
    geos_bin::String = get(ENV, "GEOS_BIN", "/temp1/c180_geosit_native_v4_dec2021_f32/geos_transport_20211204_float32.bin")
    hour::Float64 = parse(Float64, get(ENV, "HOUR", "60"))
    profile_window::Int = parse(Int, get(ENV, "PROFILE_WINDOW", "13"))
    cds_time_index::Int = parse(Int, get(ENV, "CDS_TIME_INDEX", "13")) # 2021-12-04 12 UTC for hourly day file
    region::NTuple{4, Float64} = D.parse_region(get(ENV, "REGION", "10,40,-40,-20"))
    pressures::Vector{Float64} = D.parse_float_list(get(ENV, "STANDARD_PRESSURES", "950,925,900,875,850,825,800,775,750,700,650,600,550,500"))
end

ProfileRow = NamedTuple{(:source, :mask, :pressure_hpa, :u_east_ms, :v_north_ms, :speed_ms, :bearing_deg, :sample_cells),
                         Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int}}

lon180(lon) = mod(lon + 180.0, 360.0) - 180.0
bearing_deg(u, v) = mod(rad2deg(atan(u, v)), 360.0)
speed_ms(u, v) = hypot(u, v)
angular_diff(a, b) = mod(a - b + 180.0, 360.0) - 180.0

function ll_region_mask(mesh, region)
    lon_min, lon_max, lat_min, lat_max = region
    mask = falses(mesh.Nx, mesh.Ny)
    for j in 1:mesh.Ny, i in 1:mesh.Nx
        lon = lon180(Float64(mesh.λᶜ[i]))
        lat = Float64(mesh.φᶜ[j])
        mask[i, j] = lon_min <= lon <= lon_max && lat_min <= lat <= lat_max
    end
    return mask
end

function ll_area_weights(mesh)
    area = Array{Float64}(undef, mesh.Nx, mesh.Ny)
    for j in 1:mesh.Ny, i in 1:mesh.Nx
        area[i, j] = Float64(cell_area(mesh, i, j))
    end
    return area
end

function ll_binary_profile(path::AbstractString, cfg::Config)
    reader = TransportBinaryReader(path; FT=Float32)
    try
        grid = load_grid(reader; FT=Float32)
        mesh = grid.horizontal
        m, ps, fluxes = load_window!(reader, cfg.profile_window)
        nx, ny, nz = size(m)
        u = Array{Float32}(undef, nx, ny, nz)
        v = Array{Float32}(undef, nx, ny, nz)
        steps = reader.header.steps_per_window_by_window[cfg.profile_window]
        dt_factor = Float32(reader.header.dt_met_seconds / (2.0 * steps))
        recover_ll_cell_center_winds!(u, v, fluxes.am, fluxes.bm, ps,
            Float32.(reader.header.A_ifc), Float32.(reader.header.B_ifc),
            Float32.(mesh.φᶜ), Float32(mesh.radius * deg2rad(mesh.Δφ)),
            Float32(deg2rad(mesh.Δλ)), Float32(mesh.radius), Float32(GRAV),
            dt_factor)

        mask = ll_region_mask(mesh, cfg.region)
        area = ll_area_weights(mesh)
        idxs = findall(mask)
        rows = ProfileRow[]
        for k in 1:nz
            pnum = 0.0
            unum = 0.0
            vnum = 0.0
            den = 0.0
            n_used = 0
            for idx in idxs
                i, j = Tuple(idx)
                p0 = Float64(reader.header.A_ifc[k] + reader.header.B_ifc[k] * ps[i, j])
                p1 = Float64(reader.header.A_ifc[k + 1] + reader.header.B_ifc[k + 1] * ps[i, j])
                pressure_hpa = 0.5 * (p0 + p1) / 100.0
                w = area[i, j]
                pnum += pressure_hpa * w
                unum += Float64(u[i, j, k]) * w
                vnum += Float64(v[i, j, k]) * w
                den += w
                n_used += 1
            end
            uu = unum / den
            vv = vnum / den
            push!(rows, (source="ERA_LL_BINARY", mask="region_box",
                         pressure_hpa=pnum / den, u_east_ms=uu, v_north_ms=vv,
                         speed_ms=speed_ms(uu, vv), bearing_deg=bearing_deg(uu, vv),
                         sample_cells=n_used))
        end
        return rows
    finally
        close(reader)
    end
end

function cds_region_profile(path::AbstractString, cfg::Config)
    lon_min, lon_max, lat_min, lat_max = cfg.region
    NCDataset(path, "r") do ds
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
            n_used = 0
            for j in lat_idxs, i in lon_idxs
                uu = Float64(ds["u"][i, j, k, cfg.cds_time_index])
                vv = Float64(ds["v"][i, j, k, cfg.cds_time_index])
                isfinite(uu) && isfinite(vv) || continue
                w = cosd(lats[j])
                unum += uu * w
                vnum += vv * w
                den += w
                n_used += 1
            end
            uu = unum / den
            vv = vnum / den
            push!(rows, (source="CDS_RAW_ERA5", mask="region_box",
                         pressure_hpa, u_east_ms=uu, v_north_ms=vv,
                         speed_ms=speed_ms(uu, vv), bearing_deg=bearing_deg(uu, vv),
                         sample_cells=n_used))
        end
        return rows
    end
end

function run_cs_profile(cfg::Config)
    cs_out = joinpath(cfg.out_dir, "cs_profile")
    result = D.main(; out_dir=cs_out, hour=cfg.hour, profile_window=cfg.profile_window,
                    era_nc=cfg.era_nc, geos_nc=cfg.geos_nc, era_bin=cfg.era_cs_bin,
                    geos_bin=cfg.geos_bin)
    return read_cs_profile(result.profile_csv)
end

function read_cs_profile(path::AbstractString)
    rows = ProfileRow[]
    for (n, line) in enumerate(eachline(path))
        n == 1 && continue
        p = split(chomp(line), ",")
        length(p) >= 11 || continue
        p[1] == "ERA" && p[2] == "region_box" || continue
        pressure_hpa = parse(Float64, p[5])
        uu = parse(Float64, p[6])
        vv = parse(Float64, p[7])
        push!(rows, (source="ERA_C180_BINARY", mask="region_box",
                     pressure_hpa, u_east_ms=uu, v_north_ms=vv,
                     speed_ms=parse(Float64, p[8]), bearing_deg=parse(Float64, p[9]),
                     sample_cells=parse(Int, p[11])))
    end
    return rows
end

function interpolate_uv(rows, target_hpa)
    g = sort([r for r in rows if isfinite(r.pressure_hpa) && isfinite(r.u_east_ms) && isfinite(r.v_north_ms)],
             by = r -> r.pressure_hpa)
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
    exact = findfirst(r -> isapprox(r.pressure_hpa, target_hpa; atol=1e-6), rows)
    exact !== nothing && return (u=rows[exact].u_east_ms, v=rows[exact].v_north_ms,
                                speed=rows[exact].speed_ms, bearing=rows[exact].bearing_deg)
    return interpolate_uv(rows, target_hpa)
end

function write_profile_csv(path, rows)
    open(path, "w") do io
        println(io, "source,mask,pressure_hpa,u_east_ms,v_north_ms,speed_ms,bearing_deg,sample_cells")
        for r in rows
            @printf(io, "%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%d\n",
                    r.source, r.mask, r.pressure_hpa, r.u_east_ms, r.v_north_ms,
                    r.speed_ms, r.bearing_deg, r.sample_cells)
        end
    end
    return path
end

function write_standard_csv(path, sources, cfg::Config)
    open(path, "w") do io
        println(io, "pressure_hpa,cds_speed_ms,cds_bearing_deg,ll_speed_ms,ll_bearing_deg,c180_speed_ms,c180_bearing_deg,ll_minus_cds_bearing_deg,c180_minus_cds_bearing_deg,ll_minus_cds_speed_ms,c180_minus_cds_speed_ms")
        cds = sources["CDS_RAW_ERA5"]
        ll = sources["ERA_LL_BINARY"]
        cs = sources["ERA_C180_BINARY"]
        for p in cfg.pressures
            rv = exact_or_interp(cds, p)
            lv = interpolate_uv(ll, p)
            cv = interpolate_uv(cs, p)
            if rv === nothing || lv === nothing || cv === nothing
                @printf(io, "%.1f,,,,,,,,,,\n", p)
            else
                @printf(io, "%.1f,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
                        p, rv.speed, rv.bearing, lv.speed, lv.bearing, cv.speed, cv.bearing,
                        angular_diff(lv.bearing, rv.bearing), angular_diff(cv.bearing, rv.bearing),
                        lv.speed - rv.speed, cv.speed - rv.speed)
            end
        end
    end
    return path
end

function write_plot(path, sources)
    fig = Figure(size=(1200, 560), fontsize=13)
    ax1 = Axis(fig[1, 1], xlabel="speed [m s^-1]", ylabel="pressure [hPa]",
               title="South Africa box wind speed")
    ax2 = Axis(fig[1, 2], xlabel="bearing [deg clockwise from north]", ylabel="pressure [hPa]",
               title="South Africa box wind direction")
    styles = (
        ("CDS_RAW_ERA5", "raw CDS ERA5", :black),
        ("ERA_LL_BINARY", "ERA LL binary", :forestgreen),
        ("ERA_C180_BINARY", "ERA C180 binary", :dodgerblue3),
    )
    for (key, label, color) in styles
        rows = sort(sources[key], by = r -> r.pressure_hpa)
        p = [r.pressure_hpa for r in rows]
        speed = [r.speed_ms for r in rows]
        bearing = [r.bearing_deg for r in rows]
        lines!(ax1, speed, p; label, color, linewidth=2.5)
        scatter!(ax1, speed, p; color, markersize=4)
        lines!(ax2, bearing, p; label, color, linewidth=2.5)
        scatter!(ax2, bearing, p; color, markersize=4)
    end
    ylims!(ax1, 1000, 0)
    ylims!(ax2, 1000, 0)
    xlims!(ax2, 0, 360)
    ax2.xticks = 0:45:360
    axislegend(ax1, position=:rt)
    axislegend(ax2, position=:rt)
    Label(fig[2, 1:2],
          "All profiles are area/vector means over lon 10-40E, lat 40-20S; t=60 / 2021-12-04 12 UTC.",
          fontsize=12)
    save(path, fig, px_per_unit=2)
    return path
end

function write_memo(path, cfg::Config, standard_csv::AbstractString)
    open(path, "w") do io
        println(io, "# ERA5 LL-to-C180 Wind Error Isolation")
        println(io)
        println(io, "## Question")
        println(io, "Raw CDS ERA5 pressure-level U/V agrees better with GEOS-IT than with the ERA-derived C180 binary near the South Africa fossil plume. Determine whether the error is already present in the ERA LL source binary or introduced by LL-to-CS regridding/rotation.")
        println(io)
        println(io, "## Inputs")
        println(io, "- Raw CDS U/V: `", cfg.cds_uv, "`")
        println(io, "- ERA LL source binary: `", cfg.ll_bin, "`")
        println(io, "- ERA C180 binary: `", cfg.era_cs_bin, "`")
        println(io, "- ERA C180 snapshot: `", cfg.era_nc, "`")
        println(io, "- GEOS C180 snapshot for plume masks only: `", cfg.geos_nc, "`")
        println(io, "- Region: lon 10-40E, lat 40-20S; profile window ", cfg.profile_window, "; CDS time index ", cfg.cds_time_index, " (2021-12-04 12 UTC).")
        println(io)
        println(io, "## First Isolation Result")
        println(io, "See `", standard_csv, "`. If the LL binary matches CDS and C180 does not, focus on `src/Preprocessing/transport_binary/cubed_sphere_regrid.jl` and `src/Preprocessing/cs_*.jl`. If LL already differs from CDS, focus on spectral/LL preprocessing and flux construction before CS regridding.")
        println(io)
        println(io, "## Code Paths To Audit")
        println(io, "- LL wind recovery used by regridding: `recover_ll_cell_center_winds!` in `src/Preprocessing/cs_transport_helpers.jl`.")
        println(io, "- LL-to-CS conversion: `regrid_ll_binary_to_cs` in `src/Preprocessing/transport_binary/cubed_sphere_regrid.jl`.")
        println(io, "- CS flux reconstruction and rotation: `rotate_winds_to_panel_local!` and `recover_cs_cell_center_winds!` in `src/Preprocessing/cs_wind_rotation.jl`, `reconstruct_cs_fluxes!` in `src/Preprocessing/cs_flux_reconstruction.jl`.")
        println(io, "- ERA LL binary generation: `scripts/preprocessing/preprocess_transport_binary.jl` and `src/Preprocessing/transport_binary/latlon_workspaces.jl`.")
        println(io)
        println(io, "## Specific Hypotheses")
        println(io, "- Wrong pressure thickness or dry/moist pressure used when recovering LL winds from mass fluxes.")
        println(io, "- Latitude orientation or meridional face indexing issue in LL `bm` recovery.")
        println(io, "- Sign/rotation error when converting east/north winds to panel-local C180 face normals.")
        println(io, "- Area-normalized vs extensive-field regridding mistake for U/V, mass fluxes, or pressure thickness.")
        println(io, "- Vertical merge/remap issue near 900-800 hPa that changes the sampled layer relative to CDS.")
        println(io)
        println(io, "## Suggested F90 Comparison")
        println(io, "Compare this Julia LL binary's reconstructed U/V against the legacy/F90 preprocessing output before CS regridding at the same timestamp, same LL cells, and same pressure levels. The most useful fields are reconstructed cell-center `u`, `v`, model-layer midpoint pressure, `am`, `bm`, `ps`, and `dp` for the South Africa box.")
    end
    return path
end

function main(; kwargs...)
    cfg = Config(; kwargs...)
    mkpath(cfg.out_dir)

    cds_rows = cds_region_profile(cfg.cds_uv, cfg)
    ll_rows = ll_binary_profile(cfg.ll_bin, cfg)
    cs_rows = run_cs_profile(cfg)
    sources = Dict(
        "CDS_RAW_ERA5" => cds_rows,
        "ERA_LL_BINARY" => ll_rows,
        "ERA_C180_BINARY" => cs_rows,
    )
    profile_csv = write_profile_csv(joinpath(cfg.out_dir, "south_africa_box_cds_ll_c180_profiles.csv"),
                                    vcat(cds_rows, ll_rows, cs_rows))
    standard_csv = write_standard_csv(joinpath(cfg.out_dir, "south_africa_box_cds_ll_c180_standard_pressures.csv"),
                                      sources, cfg)
    plot_path = write_plot(joinpath(cfg.out_dir, "south_africa_box_cds_ll_c180_profiles.png"), sources)
    memo_path = write_memo(joinpath(cfg.out_dir, "claude_era_wind_preprocessing_investigation_memo.md"),
                           cfg, standard_csv)

    println("Wrote:")
    for path in (profile_csv, standard_csv, plot_path, memo_path)
        println("  ", path)
    end
    println("Key pressures:")
    for p in (900.0, 850.0, 800.0, 750.0, 700.0)
        rv = exact_or_interp(cds_rows, p)
        lv = interpolate_uv(ll_rows, p)
        cv = interpolate_uv(cs_rows, p)
        rv === nothing || lv === nothing || cv === nothing && continue
        @printf("%4.0f hPa  CDS %5.2f@%6.1f  LL %5.2f@%6.1f ddir %+6.1f  C180 %5.2f@%6.1f ddir %+6.1f\n",
                p, rv.speed, rv.bearing, lv.speed, lv.bearing,
                angular_diff(lv.bearing, rv.bearing), cv.speed, cv.bearing,
                angular_diff(cv.bearing, rv.bearing))
    end
    return (; profile_csv, standard_csv, plot_path, memo_path)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    IsolateERALLToCSWindError.main()
end
