#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Surface-flux ingestion (issue #59):
#
#   1. a time-varying source whose files do not cover the run is an error
#      (it used to hold one slice silently), unless the source opts in with
#      `hold_outside_coverage = true`;
#   2. lat-lon and reduced-Gaussian targets regrid conservatively by default,
#      and the conservative path regrids the per-cell rate F × A (file areas,
#      else source-mesh areas) as an extensive field, so every target total is
#      the file's own total to rounding, also for net-negative and area-less
#      inventories (LMDZ-, EDGAR-SF6- and Zhang-Rn-like fields); a file area
#      that is not m² of the same cells is refused;
#   3. the explicit bilinear path renormalises emission and uptake separately
#      (no sign flip, no silent skip) or refuses.
# ---------------------------------------------------------------------------

using Test
using Dates
using Logging
using Printf: @sprintf
import NCDatasets: NCDataset, defDim, defVar

using AtmosTransport
using AtmosTransport: LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, HybridSigmaPressure,
                      AtmosGrid, CPU
using AtmosTransport.Models.InitialConditionIO: build_surface_flux_source
using AtmosTransport.Operators: integrated_emission
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS,
                                 DAYS_PER_JULIAN_YEAR, EARTH_RADIUS

const ICIO = AtmosTransport.Models.InitialConditionIO
const RUNNER = AtmosTransport.Models.DrivenRunner

quietly(f) = with_logger(f, SimpleLogger(devnull, Logging.Warn))
storage_scale(species) = DRY_AIR_MOLAR_MASS / getproperty(SPECIES_MOLAR_MASS, species)
total_rate(src::AtmosTransport.SurfaceFluxSource) =
    src.cell_mass_rate isa Tuple ? sum(sum, src.cell_mass_rate) : sum(src.cell_mass_rate)

const CS_GRID = AtmosGrid(CubedSphereMesh(; Nc = 8, Hp = 3, FT = Float64),
                          HybridSigmaPressure(Float64[0, 100, 300], Float64[0, 0, 1]), CPU();
                          FT = Float64)

# ---------------------------------------------------------------------------
# 1. Time coverage of time-varying sources
# ---------------------------------------------------------------------------

# A CAMS/LMDZ-like month: 3-hourly interval-start stamps, no bounds, a uniform
# flux 1e-9 cos(2πh/24) kgC m⁻² s⁻¹ whose every whole day integrates to zero, so
# any emission over whole days comes from a held slice.
function lmdz_like_month(path, year, month)
    n = 8 * Dates.daysinmonth(year, month)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", 36); defDim(ds, "lat", 18); defDim(ds, "time", n)
        defVar(ds, "lon", Float64, ("lon",))[:] = collect(5.0:10:355)
        defVar(ds, "lat", Float64, ("lat",))[:] = collect(-85.0:10:85)
        defVar(ds, "time", Float64, ("time",), attrib = Dict(
            "units" => @sprintf("hours since %04d-%02d-01 00:00:00", year, month)))[:] = 3.0 .* (0:n-1)
        defVar(ds, "flux_apos", Float64, ("lon", "lat", "time"),
               attrib = Dict("units" => "kgC m-2 s-1"))[:, :, :] =
            [1e-9 * cospi(2 * 3.0 * (k - 1) / 24) for i in 1:36, j in 1:18, k in 1:n]
    end
    return path
end

# A cs_native series of `stamps` (hours since 2022-01-01), with or without
# `time_bnds` ending at `bounds_end`.
function cs_native_series(path, stamps; bounds_end = nothing)
    Nc = CS_GRID.horizontal.Nc
    NCDataset(path, "c") do ds
        for (name, n) in (("Xdim", Nc), ("Ydim", Nc), ("nf", 6), ("time", length(stamps)))
            defDim(ds, name, n)
        end
        time = defVar(ds, "time", Float64, ("time",),
                      attrib = Dict("units" => "hours since 2022-01-01 00:00:00"))
        time[:] = stamps
        if bounds_end !== nothing
            time.attrib["bounds"] = "time_bnds"
            defDim(ds, "nv", 2)
            defVar(ds, "time_bnds", Float64, ("nv", "time"))[:, :] =
                permutedims(hcat(stamps, vcat(stamps[2:end], bounds_end)))
        end
        defVar(ds, "F", Float32, ("Xdim", "Ydim", "nf", "time"),
               attrib = Dict("units" => "kg CO2 m-2 s-1"))[:, :, :, :] =
            fill(1f-9, Nc, Nc, 6, length(stamps))
    end
    return path
end

kgc(storage) = storage / storage_scale(:co2) * CARBON_MOLAR_MASS / SPECIES_MOLAR_MASS.co2

# An LMDZ-like lat-lon file with the stamps `values` in `units`, no bounds.
function ll_stamped(path, values, units)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", 36); defDim(ds, "lat", 18); defDim(ds, "time", length(values))
        defVar(ds, "lon", Float64, ("lon",))[:] = collect(5.0:10:355)
        defVar(ds, "lat", Float64, ("lat",))[:] = collect(-85.0:10:85)
        defVar(ds, "time", Float64, ("time",), attrib = Dict("units" => units))[:] = values
        defVar(ds, "flux_apos", Float64, ("lon", "lat", "time"),
               attrib = Dict("units" => "kgC m-2 s-1"))[:, :, :] = fill(1e-9, 36, 18, length(values))
    end
    return path
end

@testset "time-varying sources must cover the run" begin
    mktempdir() do dir
        dec = lmdz_like_month(joinpath(dir, "z_202112.nc"), 2021, 12)
        jan = lmdz_like_month(joinpath(dir, "z_202201.nc"), 2022, 1)
        feb = lmdz_like_month(joinpath(dir, "z_202202.nc"), 2022, 2)
        options(extra) = (String(k) => v for (k, v) in extra)
        lmdz(files; extra...) = Dict{String, Any}("kind" => "lmdz_co2", "files" => files,
                                                  "time_varying" => true, options(extra)...)
        build(cfg, start, days) = quietly() do
            build_surface_flux_source(CS_GRID, :co2_natural, cfg, Float64;
                                      reference_time = start,
                                      run_span = RUNNER._run_time_span(24 * days))
        end
        message(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end

        # Dec 25 → Jan 9 with the December file only: Jan 1–9 has no slice.
        msg = message(() -> build(lmdz([dec]), DateTime(2021, 12, 25), 15))
        @test occursin("co2_natural", msg)
        @test occursin("no time slice covers 2022-01-01T00:00:00 to 2022-01-09T00:00:00", msg)
        @test occursin("z_202112.nc", msg) && occursin("hold_outside_coverage", msg)
        # The control: December and January cover it; no emission over whole days.
        src = build(lmdz([dec, jan]), DateTime(2021, 12, 25), 15)
        @test abs(kgc(integrated_emission(src, 7 * 86400.0, 15 * 86400.0, 900.0))) < 1.0
        # Jan 1 → Dec 31 2021 with the December file (the retired C30 2021 configs).
        msg = message(() -> build(lmdz([dec]), DateTime(2021, 1, 1), 365))
        @test occursin("no time slice covers 2021-01-01T00:00:00 to 2021-12-01T00:00:00", msg)
        # A gap between files: December and February, a run through both.
        msg = message(() -> build(lmdz([dec, feb]), DateTime(2021, 12, 1), 70))
        @test occursin("no time slice covers 2022-01-01T00:00:00 to 2022-02-01T00:00:00", msg)
        # One stamp without bounds that is not a month start has no known
        # interval: the loader names the source and both remedies.
        single = joinpath(dir, "z_single.nc")
        NCDataset(single, "c") do ds
            defDim(ds, "lon", 36); defDim(ds, "lat", 18); defDim(ds, "time", 1)
            defVar(ds, "lon", Float64, ("lon",))[:] = collect(5.0:10:355)
            defVar(ds, "lat", Float64, ("lat",))[:] = collect(-85.0:10:85)
            defVar(ds, "time", Float64, ("time",),
                   attrib = Dict("units" => "hours since 2021-12-01 00:00:00"))[:] = [3.0]
            defVar(ds, "flux_apos", Float64, ("lon", "lat", "time"),
                   attrib = Dict("units" => "kgC m-2 s-1"))[:, :, :] = fill(1e-9, 36, 18, 1)
        end
        msg = message(() -> build(lmdz([single]), DateTime(2021, 12, 1), 1))
        @test occursin("co2_natural", msg) && occursin("z_single.nc", msg) &&
              occursin("time coverage of its files is unknown", msg) &&
              occursin("bounds", msg) && occursin("hold_outside_coverage", msg)
        @test build(lmdz([single]; hold_outside_coverage = true), DateTime(2021, 12, 1), 1) isa
              AtmosTransport.TimeVaryingSurfaceFluxSource
        # A run that ends exactly where the file ends is covered (the runner's
        # 1 s clock slack is not asked of the file).
        @test build(lmdz([dec]), DateTime(2021, 12, 1), 31) isa AtmosTransport.TimeVaryingSurfaceFluxSource
        # The opt-in holds the last December slice (Dec 31 21:00) through
        # January 1–9, as before: 1e-9 cos(2π 21/24) kgC m⁻² s⁻¹ over the sphere.
        held = build(lmdz([dec]; hold_outside_coverage = true), DateTime(2021, 12, 25), 15)
        @test held.times[end] == 165 * 3600.0
        @test kgc(integrated_emission(held, 7 * 86400.0, 15 * 86400.0, 900.0)) ≈
              1e-9 * cospi(2 * 21 / 24) * 4π * EARTH_RADIUS^2 * 8 * 86400 rtol = 1e-9
        @test_throws ArgumentError build(lmdz([dec]; hold_outside_coverage = "yes"),
                                         DateTime(2021, 12, 25), 15)

        # GridFED without a time coordinate covers its year, month by month.
        gridfed = joinpath(dir, "gridfed_2022.nc")
        NCDataset(gridfed, "c") do ds
            defDim(ds, "longitude", 4); defDim(ds, "latitude", 2); defDim(ds, "time", 12)
            defVar(ds, "longitude", Float64, ("longitude",))[:] = [45.0, 135.0, 225.0, 315.0]
            defVar(ds, "latitude", Float64, ("latitude",))[:] = [-45.0, 45.0]
            defVar(ds, "TOTAL", Float64, ("longitude", "latitude", "time"),
                   attrib = Dict("units" => "kgCO2/month/m2"))[:, :, :] = ones(4, 2, 12)
        end
        gcfg = Dict{String, Any}("kind" => "gridfed_fossil_co2", "file" => gridfed,
                                 "year" => 2022, "time_varying" => true, "species" => "co2")
        gbuild(start, days) = quietly() do
            build_surface_flux_source(CS_GRID, :fossil, gcfg, Float64; reference_time = start,
                                      run_span = RUNNER._run_time_span(24 * days))
        end
        @test gbuild(DateTime(2022, 12, 1), 31) isa AtmosTransport.TimeVaryingSurfaceFluxSource
        @test occursin("2023-01-01T00:00:00 to 2023-01-02T00:00:00",
                       message(() -> gbuild(DateTime(2022, 12, 1), 32)))

        # cs_native: `time_bnds` set the coverage; without bounds, the spacing of
        # the stamps (here daily, so the last slice ends at 72 h).
        bounded = cs_native_series(joinpath(dir, "bounded.nc"), [0.0, 24.0, 48.0]; bounds_end = 168.0)
        stamped = cs_native_series(joinpath(dir, "stamped.nc"), [0.0, 24.0, 48.0])
        native(path; extra...) = Dict{String, Any}("kind" => "cs_native", "file" => path,
                                                   "variable" => "F", "time_varying" => true,
                                                   "species" => "co2", options(extra)...)
        @test build(native(bounded), DateTime(2022, 1, 1), 7) isa AtmosTransport.TimeVaryingSurfaceFluxSource
        @test occursin("2022-01-08T00:00:00 to 2022-01-09T00:00:00",
                       message(() -> build(native(bounded), DateTime(2022, 1, 1), 8)))
        @test build(native(stamped), DateTime(2022, 1, 1), 3) isa AtmosTransport.TimeVaryingSurfaceFluxSource
        msg = message(() -> build(native(stamped), DateTime(2022, 1, 1), 4))
        @test occursin("cs_native", msg) &&
              occursin("no time slice covers 2022-01-04T00:00:00 to 2022-01-05T00:00:00", msg)
        msg = message(() -> build(native(stamped), DateTime(2021, 12, 31), 2))
        @test occursin("no time slice covers 2021-12-31T00:00:00 to 2022-01-01T00:00:00", msg)
        @test build(native(stamped; hold_outside_coverage = true), DateTime(2021, 12, 31), 5) isa
              AtmosTransport.TimeVaryingSurfaceFluxSource
        one_stamp = cs_native_series(joinpath(dir, "one_stamp.nc"), [3.0])
        msg = message(() -> build(native(one_stamp), DateTime(2022, 1, 1), 1))
        @test occursin("kind = cs_native", msg) && occursin("one_stamp.nc", msg) &&
              occursin("time coverage of its files is unknown", msg) &&
              occursin("hold_outside_coverage", msg)
        # A single month-start stamp is no longer read as its month (review A1).
        month_start = cs_native_series(joinpath(dir, "month_start.nc"), [0.0])
        @test occursin("single time stamp",
                       message(() -> build(native(month_start), DateTime(2022, 1, 1), 1)))
        # A gap inside one cs_native file (daily stamps, 2022-01-03 missing).
        gappy = cs_native_series(joinpath(dir, "gappy.nc"), [0.0, 24.0, 72.0])
        @test build(native(gappy), DateTime(2022, 1, 1), 2) isa AtmosTransport.TimeVaryingSurfaceFluxSource
        @test occursin("no time slice covers 2022-01-03T00:00:00 to 2022-01-04T00:00:00",
                       message(() -> build(native(gappy), DateTime(2022, 1, 1), 4)))
    end
end

@testset "time coverage comes from the stamps of all files together (review)" begin
    mktempdir() do dir
        lmdz(files; extra...) = Dict{String, Any}("kind" => "lmdz_co2", "files" => files,
                                                  "time_varying" => true,
                                                  (String(k) => v for (k, v) in extra)...)
        build(cfg, start, days) = quietly() do
            build_surface_flux_source(CS_GRID, :co2_natural, cfg, Float64;
                                      reference_time = start,
                                      run_span = RUNNER._run_time_span(24 * days))
        end
        message(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
        isbuilt(x) = x isa AtmosTransport.TimeVaryingSurfaceFluxSource

        # Daily files with one 00Z stamp each are a daily series (A1, A2) …
        daily = [ll_stamped(joinpath(dir, @sprintf("daily_202201%02d.nc", day)), [0.0],
                            @sprintf("days since 2022-01-%02d", day)) for day in 1:3]
        @test isbuilt(build(lmdz(daily), DateTime(2022, 1, 1), 3))
        @test isbuilt(build(lmdz(reverse(daily)), DateTime(2022, 1, 1), 3))
        @test occursin("no time slice covers 2022-01-04T00:00:00 to 2022-01-31T00:00:00",
                       message(() -> build(lmdz(daily), DateTime(2022, 1, 1), 30)))
        # … and the day-1 file alone does not cover a 31-day run (no silent hold).
        msg = message(() -> build(lmdz(daily[1:1]), DateTime(2022, 1, 1), 31))
        @test occursin("co2_natural", msg) && occursin("single time stamp", msg) &&
              occursin("time coverage of its files is unknown", msg) &&
              occursin("hold_outside_coverage", msg)
        @test isbuilt(build(lmdz(daily[1:1]; hold_outside_coverage = true), DateTime(2022, 1, 1), 31))
        # An annual file with one 1 January stamp is not January only (A9).
        annual = ll_stamped(joinpath(dir, "annual_2022.nc"), [0.0], "days since 2022-01-01")
        @test occursin("single time stamp", message(() -> build(lmdz([annual]), DateTime(2022, 1, 1), 31)))
        # Monthly files with one stamp each: a monthly series; the last month
        # is a calendar month.
        monthly = [ll_stamped(joinpath(dir, @sprintf("monthly_2022%02d.nc", m)), [0.0],
                              @sprintf("days since 2022-%02d-01", m)) for m in 1:2]
        @test isbuilt(build(lmdz(monthly), DateTime(2022, 1, 1), 59))
        @test occursin("no time slice covers 2022-03-01T00:00:00 to 2022-03-02T00:00:00",
                       message(() -> build(lmdz(monthly), DateTime(2022, 1, 1), 60)))
        # A gap inside one file (Jan 1, Mar 1; A3): January holds its month only.
        jan_mar = ll_stamped(joinpath(dir, "jan_mar.nc"), [0.0, 59.0], "days since 2022-01-01")
        @test isbuilt(build(lmdz([jan_mar]), DateTime(2022, 1, 1), 31))
        @test occursin("no time slice covers 2022-02-01T00:00:00 to 2022-03-01T00:00:00",
                       message(() -> build(lmdz([jan_mar]), DateTime(2022, 1, 1), 90)))
        # Mid-month stamps without bounds (A4): refused, saying so.
        mid = ll_stamped(joinpath(dir, "mid_month.nc"),
                         [14.0, 45, 73, 104, 134, 165, 195, 226, 257, 287, 318, 348],
                         "days since 2022-01-01")
        msg = message(() -> build(lmdz([mid]), DateTime(2022, 2, 1), 28))
        @test occursin("mid-month", msg) && occursin("bounds", msg) && occursin("mid_month.nc", msg)
    end
end

# ---------------------------------------------------------------------------
# 2. Static inventories: conservative by default, totals = the file's own
# ---------------------------------------------------------------------------

const NX_SRC, NY_SRC = 72, 36                       # 5° source grid
const LON_SRC = collect(2.5:5.0:357.5)
const LAT_SRC = collect(-87.5:5.0:87.5)

# Latitude-circle cell areas of the 5° source grid on a sphere of radius R.
band_areas(R) = [R^2 * deg2rad(5.0) * (sind(LAT_SRC[j] + 2.5) - sind(LAT_SRC[j] - 2.5))
                 for i in 1:NX_SRC, j in 1:NY_SRC]

# Net-negative CO₂ flux [kgC m⁻² s⁻¹]: NH uptake, a weak positive background
# and two point sources, net about -20 % of the gross.
lmdz_field(t) = [-1e-8 * exp(-((LAT_SRC[j] - 50) / 12)^2) * (1 + 0.3 * sind(LON_SRC[i]) + 0.1t) +
                 2e-10 * cosd(LAT_SRC[j]) +
                 ((i, j) == (3, 28) ? 4e-7 : 0.0) + ((i, j) == (50, 26) ? 2e-7 : 0.0)
                 for i in 1:NX_SRC, j in 1:NY_SRC]

# Three inventories modelled on the shipped ones; each returns (cfg, species,
# expected total in kg species s⁻¹, computed here from the file values).
function lmdz_like_static(dir)
    path = joinpath(dir, "lmdz_like.nc")
    # Areas on another sphere (the file's radius convention), as CAMS's `area`.
    A = band_areas(6.3781e6)
    F = cat(lmdz_field(0), lmdz_field(1); dims = 3)
    NCDataset(path, "c") do ds
        defDim(ds, "longitude", NX_SRC); defDim(ds, "latitude", NY_SRC); defDim(ds, "time", 2)
        defVar(ds, "longitude", Float64, ("longitude",))[:] = LON_SRC
        defVar(ds, "latitude", Float64, ("latitude",))[:] = LAT_SRC
        defVar(ds, "time", Float64, ("time",), attrib = Dict("units" => "hours since 2022-06-01"))[:] = [0.0, 3.0]
        defVar(ds, "area", Float64, ("longitude", "latitude"), attrib = Dict("units" => "m2"))[:, :] = A
        defVar(ds, "flux_apos", Float64, ("longitude", "latitude", "time"),
               attrib = Dict("units" => "kgC m-2 s-1"))[:, :, :] = F
    end
    mean_flux = (zeros(NX_SRC, NY_SRC) .+ F[:, :, 1] .+ F[:, :, 2]) ./ 2   # the loader's time mean
    expected = sum(mean_flux .* A) * SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS
    return Dict{String, Any}("kind" => "lmdz_co2", "file" => path), :co2, expected
end

function edgar_like(dir)
    path = joinpath(dir, "edgar_like.nc")
    T = [(mod(7i + 3j, 41) == 0 ? 2.0 + sind(10i) : 0.0) + 1e-3 * cosd(LAT_SRC[j])
         for i in 1:NX_SRC, j in 1:NY_SRC]                             # tonnes per cell and year
    NCDataset(path, "c") do ds
        defDim(ds, "lon", NX_SRC); defDim(ds, "lat", NY_SRC)
        defVar(ds, "lon", Float64, ("lon",))[:] = LON_SRC
        defVar(ds, "lat", Float64, ("lat",))[:] = LAT_SRC
        defVar(ds, "emissions", Float64, ("lon", "lat"), attrib = Dict("units" => "Tonnes"))[:, :] = T
    end
    expected = sum(T) * 1e3 / (DAYS_PER_JULIAN_YEAR * 86400)
    return Dict{String, Any}("kind" => "edgar_sf6", "file" => path, "variable" => "emissions"),
           :sf6, expected
end

function zhang_like(dir)
    path = joinpath(dir, "zhang_like.nc")
    F = [abs(LAT_SRC[j]) < 60 ? 1e-19 * (1 + cosd(LAT_SRC[j]) + 0.5 * sind(LON_SRC[i])) : 0.0
         for i in 1:NX_SRC, j in 1:NY_SRC]                             # kg m⁻² s⁻¹
    NCDataset(path, "c") do ds
        defDim(ds, "lon", NX_SRC); defDim(ds, "lat", NY_SRC)
        defVar(ds, "lon", Float64, ("lon",))[:] = LON_SRC
        defVar(ds, "lat", Float64, ("lat",))[:] = LAT_SRC
        defVar(ds, "rnemis", Float64, ("lon", "lat"), attrib = Dict("units" => "kg/m2/s"))[:, :] = F
    end
    expected = sum(F .* band_areas(EARTH_RADIUS))      # no area variable: source-mesh areas
    return Dict{String, Any}("kind" => "zhang_rn222", "file" => path, "variable" => "rnemis"),
           :rn222, expected
end

const VERTICAL = HybridSigmaPressure(Float64[0, 50000, 0], Float64[1, 0.5, 0])
const LL_GRID = AtmosGrid(LatLonMesh(; FT = Float64, Nx = 48, Ny = 25), VERTICAL, CPU(); FT = Float64)
const RG_GRID = AtmosGrid(ReducedGaussianMesh([-67.5, -22.5, 22.5, 67.5], [12, 24, 24, 12];
                                              FT = Float64), VERTICAL, CPU(); FT = Float64)

# The bilinear sample of the source at every target cell centre (flat order).
function bilinear_samples(grid, cfg, species)
    field = ICIO._load_file_surface_flux_field(cfg, Float64; species)
    sample(lon, lat) = ICIO._sample_bilinear_scalar(field.raw, field.lon, field.lat, lon, lat)
    mesh = grid.horizontal
    mesh isa LatLonMesh && return vec([sample(mesh.λᶜ[i], mesh.φᶜ[j])
                                       for i in 1:length(mesh.λᶜ), j in 1:length(mesh.φᶜ)])
    out = zeros(AtmosTransport.Grids.ncells(mesh))
    for j in 1:AtmosTransport.Grids.nrings(mesh)
        lons = AtmosTransport.Grids.ring_longitudes(mesh, j)
        for i in eachindex(lons)
            out[AtmosTransport.Grids.cell_index(mesh, i, j)] = sample(lons[i], mesh.latitudes[j])
        end
    end
    return out
end

@testset "static inventories keep the file's total on every topology" begin
    @test ICIO._regridding_method(Dict{String, Any}()) isa ICIO.ConservativeFluxRegridding
    @test ICIO._regridding_method(Dict("regridding" => "bilinear")) isa ICIO.BilinearFluxRegridding
    mktempdir() do dir
        withenv("ATMOSTR_REGRID_CACHE_DIR" => mkpath(joinpath(dir, "cache"))) do
            for (label, make) in (("LMDZ-like, net negative, file areas", lmdz_like_static),
                                  ("EDGAR-SF6-like, tonnes per cell, no areas", edgar_like),
                                  ("Zhang-Rn-like, no areas", zhang_like))
                cfg, species, expected = make(dir)
                @testset "$label" begin
                    for (topology, grid) in (("LL", LL_GRID), ("RG", RG_GRID), ("CS", CS_GRID))
                        src = quietly(() -> build_surface_flux_source(grid, species, cfg, Float64))
                        @test total_rate(src) / storage_scale(species) ≈ expected rtol = 1e-12
                    end
                    # Explicit bilinear (LL, RG): the same total, and every cell keeps
                    # the sign of its bilinear sample.
                    for grid in (LL_GRID, RG_GRID)
                        bcfg = merge(cfg, Dict{String, Any}("regridding" => "bilinear"))
                        src = quietly(() -> build_surface_flux_source(grid, species, bcfg, Float64))
                        rate = vec(src.cell_mass_rate)
                        @test sum(rate) / storage_scale(species) ≈ expected rtol = 1e-12
                        @test sign.(rate) == sign.(bilinear_samples(grid, cfg, species))
                    end
                end
            end
            # The file's area sets the total, so it must be m² of the same cells:
            # km² units, or m² values that are not the cells' areas, are refused.
            for (units, factor, needle) in (("km2", 1e-6, "units"), ("m2", 1e-6, "differ"),
                                            ("m2", 0.5, "differ"))
                path = joinpath(dir, "bad_area_$(units)_$(factor).nc")
                NCDataset(path, "c") do ds
                    defDim(ds, "longitude", NX_SRC); defDim(ds, "latitude", NY_SRC)
                    defVar(ds, "longitude", Float64, ("longitude",))[:] = LON_SRC
                    defVar(ds, "latitude", Float64, ("latitude",))[:] = LAT_SRC
                    defVar(ds, "area", Float64, ("longitude", "latitude"),
                           attrib = Dict("units" => units))[:, :] = factor .* band_areas(EARTH_RADIUS)
                    defVar(ds, "flux", Float64, ("longitude", "latitude"),
                           attrib = Dict("units" => "kg m-2 s-1"))[:, :] = fill(1e-10, NX_SRC, NY_SRC)
                end
                cfg = Dict{String, Any}("kind" => "file", "file" => path, "variable" => "flux",
                                        "species" => "co2")
                err = try quietly(() -> build_surface_flux_source(LL_GRID, :co2, cfg, Float64)); nothing
                      catch e; e end
                @test err isa ArgumentError && occursin(needle, err.msg) && occursin("area", err.msg)
            end
            # Masked (zero) polar cells pass the norm test; with a nonzero flux there
            # the emission would be dropped, so they are refused (review C4a). Zero
            # areas where the flux is zero are accepted.
            for (flux_in_polar_rows, refused) in ((1e-10, true), (0.0, false))
                path = joinpath(dir, "zero_polar_area_$(refused).nc")
                A = band_areas(EARTH_RADIUS); A[1:6, 1] .= 0; A[1:6, end] .= 0
                F = fill(1e-10, NX_SRC, NY_SRC); F[1:6, [1, NY_SRC]] .= flux_in_polar_rows
                NCDataset(path, "c") do ds
                    defDim(ds, "longitude", NX_SRC); defDim(ds, "latitude", NY_SRC)
                    defVar(ds, "longitude", Float64, ("longitude",))[:] = LON_SRC
                    defVar(ds, "latitude", Float64, ("latitude",))[:] = LAT_SRC
                    defVar(ds, "area", Float64, ("longitude", "latitude"),
                           attrib = Dict("units" => "m2"))[:, :] = A
                    defVar(ds, "flux", Float64, ("longitude", "latitude"),
                           attrib = Dict("units" => "kg m-2 s-1"))[:, :] = F
                end
                cfg = Dict{String, Any}("kind" => "file", "file" => path, "variable" => "flux",
                                        "species" => "co2")
                err = try quietly(() -> build_surface_flux_source(LL_GRID, :co2, cfg, Float64)); nothing
                      catch e; e end
                if refused
                    @test err isa ArgumentError && occursin("zero, negative or not finite", err.msg) &&
                          occursin("12 cell(s)", err.msg)
                else
                    @test err === nothing
                end
            end
            # A regional (latitude-truncated) target: a partly covered source row
            # puts its whole rate into the covered part (documented behaviour of
            # ConservativeFluxRegridding), so the total is that of every source
            # row the target touches, here the half-covered rows at ±77.5° too.
            cfg, species, _ = edgar_like(dir)       # nonzero in every row
            regional = AtmosGrid(LatLonMesh(; FT = Float64, Nx = 36, Ny = 31, latitude = (-77.5, 77.5)),
                                 VERTICAL, CPU(); FT = Float64)
            field = ICIO._load_file_surface_flux_field(cfg, Float64; species)
            touched = [abs(LAT_SRC[j]) < 80 for i in 1:NX_SRC, j in 1:NY_SRC]
            src = quietly(() -> build_surface_flux_source(regional, species, cfg, Float64))
            @test total_rate(src) / storage_scale(species) ≈
                  sum((field.raw .* field.cell_area)[touched]) rtol = 1e-12
        end
    end
end

# ---------------------------------------------------------------------------
# 3. Bilinear renormalisation: emission and uptake separately, or refuse
# ---------------------------------------------------------------------------

@testset "bilinear renormalisation never flips signs or skips" begin
    # Source: emission 10 kg/s, uptake -12 kg/s (net -2). A bilinear sample
    # whose net is +0.5 kg/s: renormalising the net would multiply every cell
    # by -4 and turn the uptake into a source.
    source = ICIO.FileSurfaceFluxField{Float64}([10.0 -12.0], [0.0], [0.0, 1.0],
                                                ones(1, 2), -2.0)
    rate = [11.0, -10.5]
    ICIO._renormalize_surface_flux_rate!(rate, source)
    @test rate ≈ [10.0, -12.0] rtol = 1e-15
    # Both factors are logged; a factor more than 10 % from 1 (here the uptake,
    # 12/10.5) is a warning (review finding 6).
    logged = (:info, r"bilinear renormalisation factors: emission 0\.909091, uptake 1\.14286")
    warned = (:warn, r"misses the uptake total by a factor 1\.14")
    @test_logs logged warned match_mode = :any ICIO._renormalize_surface_flux_rate!([11.0, -10.5], source)
    # Float32 rates are renormalised in Float64 and rounded once.
    rate32 = Float32[11.0, -10.5, 0.0]
    ICIO._renormalize_surface_flux_rate!(rate32, source)
    @test rate32 == Float32[10.0, -12.0, 0.0]
    # A part the sample misses entirely cannot be renormalised: refuse.
    err = try ICIO._renormalize_surface_flux_rate!([11.0, 0.0], source; label = "flux of x"); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("flux of x", err.msg) &&
          occursin("uptake", err.msg) && occursin("conservative", err.msg)
    # A one-signed source is scaled as a whole (the former behaviour).
    positive = ICIO.FileSurfaceFluxField{Float64}([3.0 1.0], [0.0], [0.0, 1.0], ones(1, 2), 4.0)
    rate = [1.0, 1.0, 0.0]
    ICIO._renormalize_surface_flux_rate!(rate, positive)
    @test rate == [2.0, 2.0, 0.0]
end
