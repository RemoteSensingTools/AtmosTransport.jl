#!/usr/bin/env julia

using Dates
using Printf
using Statistics
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Grids: CubedSphereMesh, LatLonMesh,
    GnomonicPanelConvention, GEOSNativePanelConvention
using .AtmosTransport.Regridding: build_regridder, apply_regridder!

const R_EARTH_M = 6.371229e6
const GRAV = 9.80665

const ERA_DIR = get(ENV, "ERA_DIR", "/temp1/c180_full137_3d")
const GEOS_DIR = get(ENV, "GEOS_DIR", "/temp1/c180_geosit_native_3d")
const OUT_DIR = get(ENV, "OUT_DIR", "/temp1/c180_era_geos_ic_divergence")
const LL_NX = parse(Int, get(ENV, "LL_NX", "180"))
const LL_NY = parse(Int, get(ENV, "LL_NY", "90"))

const RUNS = ("advonly_ppm", "advdiff_ppm", "fullphysics_ppm")
const COLUMN_VARS = ("co2_natural_column_mean", "co2_fossil_column_mean")

function _snapshot_path(root, run)
    path = joinpath(root, run * ".nc")
    isfile(path) || error("Snapshot file not found: $path")
    return path
end

function _source_mesh(ds)
    conv = lowercase(String(get(ds.attrib, "panel_convention", "gnomonic")))
    convention = if conv in ("geos_native", "geosnative", "geos-native")
        GEOSNativePanelConvention()
    elseif conv in ("gnomonic", "gnomic")
        GnomonicPanelConvention()
    else
        error("Unsupported panel_convention=$(conv)")
    end
    return CubedSphereMesh(; FT=Float64, Nc=Int(ds.attrib["Nc"]),
                           convention, radius=R_EARTH_M)
end

function _ll_mesh()
    return LatLonMesh(; FT=Float64, Nx=LL_NX, Ny=LL_NY,
                      longitude=(-180.0, 180.0), latitude=(-90.0, 90.0),
                      radius=R_EARTH_M)
end

function _latlon_cell_areas(mesh::LatLonMesh)
    areas = Matrix{Float64}(undef, mesh.Nx, mesh.Ny)
    dlon = deg2rad(mesh.Δλ)
    @inbounds for j in 1:mesh.Ny
        strip = mesh.radius^2 * dlon *
                (sind(Float64(mesh.φᶠ[j + 1])) - sind(Float64(mesh.φᶠ[j])))
        for i in 1:mesh.Nx
            areas[i, j] = strip
        end
    end
    return areas
end

function _weighted_mean(x, w)
    return sum(x .* w) / sum(w)
end

function _weighted_std(x, w, mean)
    return sqrt(sum(w .* (x .- mean).^2) / sum(w))
end

function _weighted_corr(x, y, w)
    mx = _weighted_mean(x, w)
    my = _weighted_mean(y, w)
    sx = _weighted_std(x, w, mx)
    sy = _weighted_std(y, w, my)
    (sx == 0 || sy == 0) && return NaN
    return sum(w .* (x .- mx) .* (y .- my)) / sum(w) / (sx * sy)
end

function _weighted_quantile(x, w, q)
    order = sortperm(x)
    total = sum(w)
    threshold = q * total
    acc = 0.0
    for idx in order
        acc += w[idx]
        acc >= threshold && return x[idx]
    end
    return x[order[end]]
end

function _stats(x, w)
    mean = _weighted_mean(x, w)
    return (
        mean = mean,
        std = _weighted_std(x, w, mean),
        min = minimum(x),
        p01 = _weighted_quantile(x, w, 0.01),
        p05 = _weighted_quantile(x, w, 0.05),
        p50 = _weighted_quantile(x, w, 0.50),
        p95 = _weighted_quantile(x, w, 0.95),
        p99 = _weighted_quantile(x, w, 0.99),
        max = maximum(x),
    )
end

function write_metadata(path)
    open(path, "w") do io
        println(io, "source,run,path,Nc,Nz,panel_convention,cs_definition,time_count,start_time,end_time")
        for run in RUNS, (source, root) in (("ERA5", ERA_DIR), ("GEOS", GEOS_DIR))
            f = _snapshot_path(root, run)
            NCDataset(f) do ds
                times = ds["time"][:]
                @printf(io, "%s,%s,%s,%d,%d,%s,%s,%d,%s,%s\n",
                    source, run, f,
                    Int(ds.attrib["Nc"]), size(ds["air_mass"], 4),
                    String(get(ds.attrib, "panel_convention", "")),
                    String(get(ds.attrib, "cs_definition", "")),
                    length(times), string(times[1]), string(times[end]))
            end
        end
    end
end

function write_direct_column_stats(path)
    open(path, "w") do io
        println(io, "source,run,var,time_index,time,mean_ppm,std_ppm,min_ppm,p01_ppm,p05_ppm,p50_ppm,p95_ppm,p99_ppm,max_ppm")
        for run in RUNS, (source, root) in (("ERA5", ERA_DIR), ("GEOS", GEOS_DIR))
            f = _snapshot_path(root, run)
            NCDataset(f) do ds
                area = vec(Float64.(ds["cell_area"][:, :, :]))
                times = ds["time"][:]
                for var in COLUMN_VARS
                    for t in eachindex(times)
                        field = vec(Float64.(ds[var][:, :, :, t])) .* 1e6
                        s = _stats(field, area)
                        @printf(io, "%s,%s,%s,%d,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                            source, run, var, t, string(times[t]),
                            s.mean, s.std, s.min, s.p01, s.p05, s.p50,
                            s.p95, s.p99, s.max)
                    end
                end
            end
        end
    end
end

function _regrid_column_fields(path)
    ll = _ll_mesh()
    ll_area = vec(_latlon_cell_areas(ll))
    ds = NCDataset(path)
    mesh = _source_mesh(ds)
    regridder = build_regridder(mesh, ll; normalize=false)
    times = ds["time"][:]
    out = Dict{String, Matrix{Float64}}()
    for var in COLUMN_VARS
        arr = Matrix{Float64}(undef, LL_NX * LL_NY, length(times))
        tmp = zeros(Float64, LL_NX * LL_NY)
        for t in eachindex(times)
            src = vec(Float64.(ds[var][:, :, :, t])) .* 1e6
            fill!(tmp, 0.0)
            apply_regridder!(tmp, regridder, src)
            arr[:, t] = tmp
        end
        out[var] = arr
    end
    close(ds)
    return (; times, fields=out, area=ll_area)
end

function write_common_grid_diffs(path)
    open(path, "w") do io
        println(io, "run,var,time_index,time,era_mean_ppm,geos_mean_ppm,bias_geos_minus_era_ppm,mae_ppm,rmse_ppm,maxabs_ppm,corr")
        for run in RUNS
            println("Regridding common-grid fields for $run")
            era = _regrid_column_fields(_snapshot_path(ERA_DIR, run))
            geo = _regrid_column_fields(_snapshot_path(GEOS_DIR, run))
            length(era.times) == length(geo.times) || error("time length mismatch for $run")
            for var in COLUMN_VARS, t in eachindex(era.times)
                e = era.fields[var][:, t]
                g = geo.fields[var][:, t]
                w = era.area
                d = g .- e
                era_mean = _weighted_mean(e, w)
                geos_mean = _weighted_mean(g, w)
                mae = _weighted_mean(abs.(d), w)
                rmse = sqrt(_weighted_mean(d.^2, w))
                corr = _weighted_corr(e, g, w)
                @printf(io, "%s,%s,%d,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    run, var, t, string(era.times[t]), era_mean, geos_mean,
                    geos_mean - era_mean, mae, rmse, maximum(abs.(d)), corr)
            end
        end
    end
end

function write_common_grid_increment_diffs(path)
    open(path, "w") do io
        println(io, "run,var,time_index,time,era_increment_mean_ppm,geos_increment_mean_ppm,increment_bias_geos_minus_era_ppm,increment_mae_ppm,increment_rmse_ppm,increment_maxabs_ppm,increment_corr")
        for run in RUNS
            println("Regridding common-grid increments for $run")
            era = _regrid_column_fields(_snapshot_path(ERA_DIR, run))
            geo = _regrid_column_fields(_snapshot_path(GEOS_DIR, run))
            length(era.times) == length(geo.times) || error("time length mismatch for $run")
            for var in COLUMN_VARS
                e0 = era.fields[var][:, 1]
                g0 = geo.fields[var][:, 1]
                for t in eachindex(era.times)
                    e = era.fields[var][:, t] .- e0
                    g = geo.fields[var][:, t] .- g0
                    w = era.area
                    d = g .- e
                    era_mean = _weighted_mean(e, w)
                    geos_mean = _weighted_mean(g, w)
                    mae = _weighted_mean(abs.(d), w)
                    rmse = sqrt(_weighted_mean(d.^2, w))
                    corr = _weighted_corr(e, g, w)
                    @printf(io, "%s,%s,%d,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        run, var, t, string(era.times[t]), era_mean, geos_mean,
                        geos_mean - era_mean, mae, rmse, maximum(abs.(d)), corr)
                end
            end
        end
    end
end

function pressure_bin_profile(path, source, run, time_index)
    bins = collect(0.0:100.0:1000.0)
    nb = length(bins) - 1
    num = zeros(Float64, nb)
    den = zeros(Float64, nb)
    NCDataset(path) do ds
        area = Float64.(ds["cell_area"][:, :, :])
        mass = Float64.(ds["air_mass"][:, :, :, :, time_index])
        q = Float64.(ds["co2_natural"][:, :, :, :, time_index]) .* 1e6
        nx, ny, nf, nz = size(mass)
        @inbounds for p in 1:nf, j in 1:ny, i in 1:nx
            ptop = 0.0
            a = area[i, j, p]
            for k in 1:nz
                m = mass[i, j, p, k]
                dp_hpa = m * GRAV / a / 100.0
                pmid = ptop + 0.5 * dp_hpa
                b = clamp(fld(Int(floor(pmid)), 100) + 1, 1, nb)
                num[b] += q[i, j, p, k] * m
                den[b] += m
                ptop += dp_hpa
            end
        end
    end
    rows = NamedTuple[]
    for b in 1:nb
        push!(rows, (; source, run, time_index,
                     p_low_hpa=bins[b], p_high_hpa=bins[b+1],
                     mean_ppm=den[b] > 0 ? num[b] / den[b] : NaN,
                     air_mass=den[b]))
    end
    return rows
end

function write_vertical_profiles(path)
    open(path, "w") do io
        println(io, "source,run,time_index,p_low_hpa,p_high_hpa,mean_ppm,air_mass")
        for run in ("advonly_ppm", "fullphysics_ppm")
            for (source, root) in (("ERA5", ERA_DIR), ("GEOS", GEOS_DIR))
                f = _snapshot_path(root, run)
                NCDataset(f) do ds
                    nt = size(ds["air_mass"], 5)
                    for t in (1, nt)
                        for row in pressure_bin_profile(f, source, run, t)
                            @printf(io, "%s,%s,%d,%.0f,%.0f,%.9g,%.9g\n",
                                row.source, row.run, row.time_index,
                                row.p_low_hpa, row.p_high_hpa,
                                row.mean_ppm, row.air_mass)
                        end
                    end
                end
            end
        end
    end
end

function main()
    mkpath(OUT_DIR)
    write_metadata(joinpath(OUT_DIR, "metadata.csv"))
    write_direct_column_stats(joinpath(OUT_DIR, "direct_column_stats.csv"))
    write_common_grid_diffs(joinpath(OUT_DIR, "common_grid_column_diffs.csv"))
    write_common_grid_increment_diffs(joinpath(OUT_DIR, "common_grid_column_increment_diffs.csv"))
    write_vertical_profiles(joinpath(OUT_DIR, "pressure_bin_profiles.csv"))
    open(joinpath(OUT_DIR, "README.txt"), "w") do io
        println(io, "ERA5-vs-GEOS C180 IC and divergence diagnostics")
        println(io, "Generated: ", Dates.now())
        println(io, "ERA_DIR: ", ERA_DIR)
        println(io, "GEOS_DIR: ", GEOS_DIR)
        println(io, "Common grid: ", LL_NX, " x ", LL_NY)
        println(io, "Outputs:")
        println(io, "  metadata.csv")
        println(io, "  direct_column_stats.csv")
        println(io, "  common_grid_column_diffs.csv")
        println(io, "  common_grid_column_increment_diffs.csv")
        println(io, "  pressure_bin_profiles.csv")
    end
    println("Wrote ", OUT_DIR)
end

main()
