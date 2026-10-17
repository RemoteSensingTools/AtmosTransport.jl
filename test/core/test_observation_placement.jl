# Vertical placement of observation requests (issue #56): records that give an
# altitude above sea level without their ground elevation (ObsPack
# aircraft-insitu and AirCore rows) are placed at their height above the true
# ground (the record's elevation, else a DEM), measured up from the model
# surface; else by their pressure. Records that cannot be placed fail the build
# or are skipped and counted, never sampled silently in the lowest layer, and
# the model surface height is never used as the true ground.
using Test, AtmosTransport, NCDatasets, Dates, Logging
using AtmosTransport.Output: observation_output_spec, build_observation_sampler, begin_observation_day!,
                             observe_window_boundary!, finish_observations!, ObservationTerrain,
                             VerticalPlacement, VERTICAL_PLACEMENT_CODES, resolve_placement,
                             placement_layer, AboveGroundPlacement, AbsoluteAltitudeAbove,
                             SoundingRequest, SiteRequest, CellLocation, ModelSurfaceHeight,
                             ErrorOnUnplaced, SkipUnplaced, LowestLayerWithoutHeight,
                             UnplacedWithoutHeight, missing_height_rule
using AtmosTransport.GriddedIO: write_model_surface_height
const O = AtmosTransport.Output
const DR = AtmosTransport.Models.DrivenRunner
include(joinpath(@__DIR__, "..", "fixtures", "observation_fixtures.jl"))
using .ObservationTestFixtures

const CODES = VERTICAL_PLACEMENT_CODES
const LON, LAT = -100.0, 10.0          # inside lat-lon cell (1, 2) of the 4×3 mesh

# 10-level lat-lon model (p_top = 100 Pa, Δσ = 0.1, ps = 1000 hPa); co2 falls from
# 420 ppm in the surface layer (k = 10) to 400 ppm in the top layer. With the
# default 280 K the interface heights are 0, 864, 1830, 2925, 4189, 5684,
# 7514, 9873, 13197, 18877 m, so 500/3000/8000 m sit in layers 10/7/4.
function model10()
    mesh = LatLonMesh(; FT = Float64, Nx = 4, Ny = 3)
    A = vcat(100.0, zeros(10))
    B = collect(0.0:0.1:1.0)
    grid = AtmosGrid(mesh, HybridSigmaPressure(A, B), CPU(); FT = Float64)
    air = Array{Float64}(undef, 4, 3, 10)
    co2 = similar(air)
    for k in 1:10, j in 1:3, i in 1:4
        air[i, j, k] = ((A[k + 1] + B[k + 1] * 1e5) - (A[k] + B[k] * 1e5)) * cell_area(mesh, i, j) / GRAVITY
        co2[i, j, k] = air[i, j, k] * (400e-6 + 20e-6 * (k - 1) / 9)
    end
    return (; state = CellState(DryBasis, air; co2), grid, mesh)
end

# An ObsPack file in the layout of the OCO-2 MIP daily files: optional
# variables are absent when `nothing`; NaN entries are written as the fill value.
function write_pack(path; altitude = nothing, elevation = nothing, intake = nothing, pressure = nothing,
                    pressure_units = "hPa", lon = LON, lat = LAT)
    n = length(something(altitude, intake, pressure, elevation))
    NCDataset(path, "c") do ds
        defDim(ds, "obs", n)
        defVar(ds, "time", Float64, ("obs",); attrib = Dict("units" => "seconds since 1970-01-01T00:00:00Z"))[:] =
            fill(unix(ORIGIN + Minute(30)), n)
        defVar(ds, "latitude", Float32, ("obs",))[:] = Float32.(lat isa AbstractVector ? lat : fill(lat, n))
        defVar(ds, "longitude", Float32, ("obs",))[:] = Float32.(lon isa AbstractVector ? lon : fill(lon, n))
        for (name, values) in (("altitude", altitude), ("elevation", elevation), ("intake_height", intake),
                               ("pressure", pressure))
            values === nothing && continue
            attrib = Dict{String, Any}("_FillValue" => -1f34)
            name == "pressure" && (attrib["units"] = pressure_units)
            defVar(ds, name, Float32, ("obs",); attrib)[:] = [isnan(x) ? -1f34 : Float32(x) for x in values]
        end
        defVar(ds, "obspack_id", String, ("obs",))[:] = ["$(first(splitext(basename(path))))~$(k)" for k in 1:n]
    end
    return path
end

obspack(path; mode = "soundings") = Dict{String, Any}("kind" => "obspack", "mode" => mode, "path" => path)

# The per-grid model surface height of `model10`, `zs` at the observation cell.
function surface_file(dir, zs; name = "zs_$(round(Int, zs)).nc")
    mesh = LatLonMesh(; FT = Float64, Nx = 4, Ny = 3)
    field = zeros(4, 3)
    field[1, 2] = zs
    return write_model_surface_height(joinpath(dir, name), mesh, field)
end

# A regional DEM around the observation point with a constant elevation.
function dem_file(dir, z; name = "dem_$(round(Int, z)).nc")
    path = joinpath(dir, name)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", 21); defDim(ds, "lat", 21)
        defVar(ds, "lon", Float64, ("lon",))[:] = collect(-110.0:1.0:-90.0)
        defVar(ds, "lat", Float64, ("lat",))[:] = collect(20.0:-1.0:0.0)
        defVar(ds, "elevation", Float32, ("lon", "lat"); attrib = Dict("units" => "m"))[:, :] = fill(Float32(z), 21, 21)
    end
    return path
end

# Build the sampler the way the runner does (terrain through GriddedIO), sample
# two window ends, and return the soundings and sites file paths.
function sample(dir, sources; name = "obs", keys...)
    cfg = Dict{String, Any}("path" => joinpath(dir, "snap.nc"), "hours" => [0.0], "split" => "single",
        "observations" => Dict{String, Any}("path" => joinpath(dir, "$(name).nc"), "sources" => sources,
                                            (String(k) => v for (k, v) in keys)...))
    spec = observation_output_spec(cfg)
    m = model10()
    terrain = DR._observation_terrain(spec.placement, m.mesh)
    s = quiet() do
        build_observation_sampler(spec, m.state, m.grid; origin = ORIGIN, window_seconds = (0.0, 86400.0),
                                  halo_width = 0, terrain)
    end
    begin_observation_day!(s, "20211202", 1)
    observe_window_boundary!(s, m.state, 0.0; next_window_seconds = 3600.0)
    observe_window_boundary!(s, m.state, 3600.0; next_window_seconds = 3600.0)
    quiet(() -> finish_observations!(s))
    close(s)
    return joinpath(dir, "$(name)_soundings.nc"), joinpath(dir, "$(name)_sites.nc")
end

function soundings(path)
    NCDataset(path, "r") do ds
        (id = ds["id"][:], level = Int.(ds["intake_level"][:]), method = Int8.(ds["vertical_placement"][:]),
         height = Float64.(ds["placement_height_agl"][:]), zs = Float64.(ds["model_surface_height"][:]),
         ppm = 1e6 .* Float64.(ds["co2_intake"][:]), pressure = Float64.(ds["pressure"][:]),
         unplaced = ds.attrib["n_unplaced_soundings"], p_half = Float64.(ds["p_half_dry"][:, :]))
    end
end

const ALTS = [500.0, 3000.0, 8000.0]
# Levels and values of records with an intake height (the issue's control column).
const CONTROL_LEVELS = [10, 7, 4]
const CONTROL_PPM = [420.0, 400 + 20 * 6 / 9, 400 + 20 * 3 / 9]

failure(f) = try
    f(); nothing
catch e
    e
end

@testset "altitude-only records are never sampled silently in the lowest layer" begin
    mktempdir() do dir
        alt_only = write_pack(joinpath(dir, "co2_air_aircraft-insitu_1.nc"); altitude = ALTS)
        control = write_pack(joinpath(dir, "co2_ctl_aircraft-pfp_1.nc"); altitude = ALTS, intake = ALTS,
                             elevation = zeros(3))
        # Without a ground elevation the build fails and names the records (on
        # main they were sampled in layer 10, the lowest, without a message).
        err = failure(() -> sample(dir, Any[obspack(alt_only), obspack(control)]))
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("3 point events", msg) && occursin("dem_file", msg)
        @test occursin("co2_air_aircraft-insitu_1~1", msg) && occursin("altitude 500.0 m", msg)
        # The model surface height is not a ground: it does not place them either.
        err = failure(() -> sample(dir, Any[obspack(alt_only)]; name = "zs",
                                   surface_height_file = surface_file(dir, 0.0)))
        @test err isa ArgumentError && occursin("3 point events", sprint(showerror, err))
        # `unplaced_records = "skip"` drops them and records the count.
        soundings_path, _ = sample(dir, Any[obspack(alt_only), obspack(control)]; name = "skip",
                                   unplaced_records = "skip")
        r = soundings(soundings_path)
        @test r.unplaced == 3
        @test all(startswith("co2_ctl"), r.id)
        @test r.level == CONTROL_LEVELS
        @test r.ppm ≈ CONTROL_PPM
        @test all(==(CODES.record_height), r.method)
        # An ObsPack record without any vertical coordinate is not a surface sample.
        bare = write_pack(joinpath(dir, "co2_bare_aircraft-insitu_1.nc"); elevation = [100.0])
        err = failure(() -> sample(dir, Any[obspack(bare)]; name = "bare"))
        @test err isa ArgumentError && occursin("1 point events", sprint(showerror, err))
    end
end

@testset "altitude-only records: height above the DEM ground, from the model surface" begin
    mktempdir() do dir
        alt_only = write_pack(joinpath(dir, "co2_air_aircraft-insitu_1.nc"); altitude = ALTS)
        flat, _ = sample(dir, Any[obspack(alt_only)]; name = "flat", dem_file = dem_file(dir, 0.0),
                         dem_variable = "elevation")
        r = soundings(flat)
        @test r.level == CONTROL_LEVELS            # was [10, 10, 10] before the fix
        @test r.ppm ≈ CONTROL_PPM
        @test all(==(CODES.altitude_minus_dem), r.method)
        @test r.height == ALTS && all(isnan, r.zs)
        @test r.unplaced == 0

        # True ground 1500 m: 3000 and 8000 m asl are 1500 and 6500 m above it,
        # exactly where records with those intake heights are sampled, whatever
        # the model ground (here 2000 m) is.
        high = write_pack(joinpath(dir, "co2_hi_aircraft-insitu_1.nc"); altitude = [3000.0, 8000.0])
        control = write_pack(joinpath(dir, "co2_ctl_tower-insitu_1.nc"); intake = [1500.0, 6500.0],
                             elevation = [1500.0, 1500.0], altitude = [3000.0, 8000.0])
        elevated, _ = sample(dir, Any[obspack(high), obspack(control)]; name = "elevated",
                             dem_file = dem_file(dir, 1500.0), dem_variable = "elevation",
                             surface_height_file = surface_file(dir, 2000.0))
        e = soundings(elevated)
        hi = findall(startswith("co2_hi"), e.id)
        ctl = findall(startswith("co2_ctl"), e.id)
        @test e.level[hi] == e.level[ctl] == [9, 5]
        @test e.ppm[hi] == e.ppm[ctl]
        @test e.method[hi] == fill(CODES.altitude_minus_dem, 2)
        @test e.height[hi] == [1500.0, 6500.0] && all(==(2000.0), e.zs)
    end
end

@testset "per-row fill of intake_height and elevation (OCO-2 MIP layout)" begin
    mktempdir() do dir
        # tower (intake 100 m), aircraft without intake/elevation, aircraft-pfp with
        # elevation but no intake, AirCore without intake/elevation.
        pack = write_pack(joinpath(dir, "co2_mix_1.nc");
                          intake = [100.0, NaN, NaN, NaN], elevation = [300.0, NaN, 1000.0, NaN],
                          altitude = [400.0, 3000.0, 4000.0, 20000.0])
        err = failure(() -> sample(dir, Any[obspack(pack)]))
        @test err isa ArgumentError && occursin("2 point events", sprint(showerror, err))
        path, _ = sample(dir, Any[obspack(pack)]; name = "mix", dem_file = dem_file(dir, 0.0),
                         dem_variable = "elevation")
        r = soundings(path)
        @test r.method == [CODES.record_height, CODES.altitude_minus_dem, CODES.record_height,
                           CODES.altitude_minus_dem]
        @test r.height == [100.0, 3000.0, 3000.0, 20000.0]
        @test r.level == [10, 7, 7, 1]               # row 3 was the lowest layer before (file-level fallback)
    end
end

@testset "intake_height records are unchanged by the terrain" begin
    mktempdir() do dir
        control = write_pack(joinpath(dir, "co2_ctl_aircraft-pfp_1.nc"); altitude = ALTS, intake = ALTS,
                             elevation = zeros(3))
        plain, _ = sample(dir, Any[obspack(control)]; name = "plain")
        terrain, _ = sample(dir, Any[obspack(control)]; name = "terrain",
                            surface_height_file = surface_file(dir, 2000.0), dem_file = dem_file(dir, 3000.0),
                            dem_variable = "elevation")
        a, b = soundings(plain), soundings(terrain)
        @test a.level == b.level == CONTROL_LEVELS
        @test a.ppm == b.ppm
        @test a.method == b.method == fill(CODES.record_height, 3)
        @test a.height == b.height == ALTS
        @test all(isnan, a.zs) && all(==(2000.0), b.zs)
    end
end

@testset "mountain site: true ground above the model ground" begin
    mktempdir() do dir
        # A Mauna Loa-like record (ground 3397 m, intake 40 m) where the model ground
        # is 2000 m: it is sampled 40 m above the model surface, in the lowest layer.
        station = write_pack(joinpath(dir, "co2_mlo_surface-flask_1.nc"); intake = [40.0], elevation = [3397.0],
                             altitude = [3437.0])
        # The same height given only as an altitude: a DEM supplies the true ground.
        alt_only = write_pack(joinpath(dir, "co2_mla_aircraft-insitu_1.nc"); altitude = [3437.0])
        zs = surface_file(dir, 2000.0)
        dem = dem_file(dir, 3397.0)
        path, _ = sample(dir, Any[obspack(station), obspack(alt_only)]; name = "dem",
                         surface_height_file = zs, dem_file = dem, dem_variable = "elevation")
        r = soundings(path)
        @test r.method == [CODES.record_height, CODES.altitude_minus_dem]
        @test r.height ≈ [40.0, 40.0]
        @test r.level == [10, 10] && all(==(2000.0), r.zs)
        # Without the DEM the altitude-only record is not placed 1437 m above the
        # model ground (layer 9); it is skipped and counted.
        nodem, _ = sample(dir, Any[obspack(station), obspack(alt_only)]; name = "nodem", surface_height_file = zs,
                          unplaced_records = "skip")
        n = soundings(nodem)
        @test n.method == [CODES.record_height] && n.level == [10] && n.unplaced == 1
        # With the absolute-altitude option every record is placed by its altitude.
        absolute, _ = sample(dir, Any[obspack(station)]; name = "absolute", surface_height_file = zs,
                             absolute_altitude_above_agl_m = 0.0)
        a = soundings(absolute)
        @test a.method == [CODES.absolute_altitude] && a.height ≈ [1437.0] && a.level == [9]
    end
end

@testset "pressure places records without a usable height" begin
    mktempdir() do dir
        # Row 1 pressure only; row 2 altitude without ground and a pressure; row 3
        # altitude without ground and no pressure (unplaced). 650 hPa is in layer 7.
        pack = write_pack(joinpath(dir, "co2_prs_aircraft-insitu_1.nc"); altitude = [NaN, 3000.0, 3000.0],
                          pressure = [650.0, 650.0, NaN])
        path, _ = sample(dir, Any[obspack(pack)]; name = "prs", unplaced_records = "skip")
        r = soundings(path)
        @test r.method == [CODES.pressure, CODES.pressure] && r.unplaced == 1
        @test r.pressure ≈ [65000.0, 65000.0] && all(isnan, r.height)
        @test r.level == [7, 7]
        @test all(r.p_half[7, k] <= 65000.0 < r.p_half[8, k] for k in 1:2)
        # Pressure units: case and blanks do not matter; an unknown or missing unit fails.
        pa = write_pack(joinpath(dir, "co2_pa_aircraft-insitu_1.nc"); pressure = [65000.0], pressure_units = " Pa ")
        @test soundings(first(sample(dir, Any[obspack(pa)]; name = "pa"))).level == [7]
        for (k, units) in enumerate(("psi", ""))
            bad = write_pack(joinpath(dir, "co2_bad$(k)_aircraft-insitu_1.nc"); pressure = [650.0],
                             pressure_units = units)
            err = failure(() -> sample(dir, Any[obspack(bad)]; name = "bad$(k)"))
            @test err isa ArgumentError && occursin("variable pressure has units", sprint(showerror, err))
        end
    end
end

@testset "absolute-altitude option: pressure first, then altitude, above a height" begin
    mktempdir() do dir
        pack = write_pack(joinpath(dir, "co2_iag_aircraft-insitu_1.nc"); altitude = [500.0, 3000.0, 8000.0],
                          pressure = [950.0, 650.0, NaN])
        path, _ = sample(dir, Any[obspack(pack)]; name = "abs", surface_height_file = surface_file(dir, 0.0),
                         dem_file = dem_file(dir, 0.0), dem_variable = "elevation",
                         absolute_altitude_above_agl_m = 1000.0)
        r = soundings(path)
        @test r.method == [CODES.altitude_minus_dem, CODES.pressure, CODES.absolute_altitude]
        @test r.pressure[1:2] ≈ [95000.0, 65000.0] && isnan(r.pressure[3])
        @test isnan(r.height[2]) && r.height[[1, 3]] == [500.0, 8000.0]
        @test r.level == [10, 7, 4]
        k = r.level[2]
        @test r.p_half[k, 2] <= 65000.0 < r.p_half[k + 1, 2]
    end
end

@testset "station series and tables with altitudes" begin
    mktempdir() do dir
        site = write_pack(joinpath(dir, "co2_twr_tower-insitu_1.nc"); altitude = [3000.0, 3000.0])
        _, sites_path = sample(dir, Any[obspack(site; mode = "sites")]; name = "sites",
                               dem_file = dem_file(dir, 0.0), dem_variable = "elevation")
        NCDataset(sites_path, "r") do ds
            @test ds["vertical_placement"][:] == [CODES.altitude_minus_dem]
            @test ds["altitude"][:] == [3000.0] && isnan(ds["intake_height"][1])
            @test all(==(7), ds["intake_level"][1, :])
            @test ds.attrib["n_unplaced_sites"] == 0
        end
        err = failure(() -> sample(dir, Any[obspack(site; mode = "sites")]; name = "sites2"))
        @test err isa ArgumentError && occursin("1 station series", sprint(showerror, err))

        csv = joinpath(dir, "points.csv")
        write(csv, "id,time,lat,lon,altitude\na,2021-12-02T00:30:00,$(LAT),$(LON),8000\n")
        @test_throws ArgumentError sample(dir, Any[table_source(csv)]; name = "t1",
                                          surface_height_file = surface_file(dir, 0.0))
        path, _ = sample(dir, Any[table_source(csv)]; name = "t2", dem_file = dem_file(dir, 0.0),
                         dem_variable = "elevation")
        r = soundings(path)
        @test r.method == [CODES.altitude_minus_dem] && r.level == [4]
        # A table without height columns samples the lowest layer, recorded as such.
        surface = joinpath(dir, "surface.csv")
        write(surface, "id,time,lat,lon\nb,2021-12-02T00:30:00,$(LAT),$(LON)\n")
        lowest, _ = sample(dir, Any[table_source(surface)]; name = "t3")
        l = soundings(lowest)
        @test l.method == [CODES.lowest_layer] && l.level == [10] && isnan(l.height[1])
    end
end

@testset "the sampler keeps the terrain provenance, not the DEM" begin
    mktempdir() do dir
        dem = dem_file(dir, 0.0)
        zs = surface_file(dir, 0.0)
        cfg = Dict{String, Any}("path" => joinpath(dir, "snap.nc"), "hours" => [0.0], "split" => "single",
            "observations" => Dict{String, Any}("path" => joinpath(dir, "keep.nc"),
                "sources" => Any[obspack(write_pack(joinpath(dir, "co2_k_aircraft-insitu_1.nc"); altitude = ALTS))],
                "dem_file" => dem, "dem_variable" => "elevation", "surface_height_file" => zs))
        spec = observation_output_spec(cfg)
        m = model10()
        s = quiet(() -> build_observation_sampler(spec, m.state, m.grid; origin = ORIGIN,
                                                  window_seconds = (0.0, 86400.0), halo_width = 0,
                                                  terrain = DR._observation_terrain(spec.placement, m.mesh)))
        held = [getfield(s, f) for f in fieldnames(typeof(s))]
        @test !any(x -> x isa ObservationTerrain || x isa O.AbstractPointElevation ||
                        x isa O.AbstractModelSurfaceHeight, held)
        @test s.terrain_sources == (surface_height_file = zs, dem_file = dem)
        @test [p.method for p in s.sounding_placements] == fill(CODES.altitude_minus_dem, 3)
        close(s)
    end
end

@testset "review fixes: fill pressures, below-ground records, moving platforms in site mode" begin
    mktempdir() do dir
        # Undeclared pressure fill (-999.9 hPa) on a row without height: unplaced,
        # never sampled in the top layer; a valid pressure still places its row.
        pack = write_pack(joinpath(dir, "co2_fill_aircraft-insitu_1.nc"); pressure = [-999.9, 0.0, 650.0])
        reqs = @test_logs (:warn, r"2 pressure values are zero or negative") match_mode = :any begin
            first(O.read_observation_requests(O.ObsPackSource(pack, O.SoundingMode(), O.SiteCodeGrouping()), 1,
                                              ORIGIN, [Date(ORIGIN)]))
        end
        @test isnan(reqs[1].pressure_pa) && isnan(reqs[2].pressure_pa) && reqs[3].pressure_pa ≈ 65000.0
        path, _ = sample(dir, Any[obspack(pack)]; name = "pfill", unplaced_records = "skip")
        r = soundings(path)
        @test r.unplaced == 2 && r.method == [CODES.pressure] && r.level == [7]
        err = failure(() -> sample(dir, Any[obspack(pack)]; name = "pfill2"))
        @test err isa ArgumentError && occursin("2 point events", sprint(showerror, err))

        # The JMA aircraft-flask rows of the OCO-2 MIP v5.0.1 ObsPack give
        # elevation = intake_height = -1000 (undeclared fill) at 5200 m altitude:
        # placed by altitude - DEM (layer 6), not in the lowest layer by the fill.
        aoa = write_pack(joinpath(dir, "co2_aoa_aircraft-flask_19_allvalid.nc"); altitude = [5200.0, 600.0],
                         elevation = [-1000.0, 0.0], intake = [-1000.0, NaN])
        reqs = @test_logs (:warn, r"1 elevation values are below -450.0 m") (:warn, r"1 intake_height values are negative") begin
            first(O.read_observation_requests(O.ObsPackSource(aoa, O.SoundingMode(), O.SiteCodeGrouping()), 1,
                                              ORIGIN, [Date(ORIGIN)]))
        end
        @test isnan(reqs[1].elevation_m) && isnan(reqs[1].intake_height_m) && reqs[2].intake_height_m == 600.0
        path, _ = sample(dir, Any[obspack(aoa)]; name = "aoa", dem_file = dem_file(dir, 0.0), dem_variable = "elevation")
        r = soundings(path)
        @test r.method == [CODES.altitude_minus_dem, CODES.record_height] && r.height == [5200.0, 600.0]
        @test r.level == [6, 10]
        err = failure(() -> sample(dir, Any[obspack(aoa)]; name = "aoa2"))
        @test err isa ArgumentError && occursin("1 point events", sprint(showerror, err))

        # A record 400 m below its stated ground (altitude 100, elevation 500) is
        # sampled in the lowest layer, and reported.
        below = write_pack(joinpath(dir, "co2_low_aircraft-insitu_1.nc"); altitude = [100.0, 3000.0],
                           elevation = [500.0, 0.0])
        spec = observation_output_spec(Dict{String, Any}("observations" => Dict{String, Any}(
            "path" => joinpath(dir, "below.nc"), "sources" => Any[obspack(below)])))
        m = model10()
        s = @test_logs (:warn, r"1 requests are more than 100.0 m below the ground") match_mode = :any begin
            build_observation_sampler(spec, m.state, m.grid; origin = ORIGIN, window_seconds = (0.0, 86400.0),
                                      halo_width = 0)
        end
        @test [p.height_agl_m for p in s.sounding_placements] == [-400.0, 3000.0]
        close(s)
        @test O._warn_below_ground(SoundingRequest[], VerticalPlacement[], SiteRequest[], VerticalPlacement[]) == 0

        # Site mode: a tower whose levels are given as altitudes yields one site
        # per level without a warning; a moving aircraft is reported.
        tower = write_pack(joinpath(dir, "co2_twr_tower-insitu_1.nc"); altitude = [3010.0, 3100.0, 3010.0],
                           lon = [LON, LON + 0.01, LON], lat = LAT)
        sites = @test_logs min_level = Logging.Warn begin
            last(O.read_observation_requests(O.ObsPackSource(tower, O.SiteMode(), O.SiteCodeGrouping()), 1,
                                             ORIGIN, [Date(ORIGIN)]))
        end
        @test [x.altitude_m for x in sites] == [3010.0, 3100.0]
        plane = write_pack(joinpath(dir, "co2_air_aircraft-insitu_1.nc"); altitude = [500.0, 3000.0, 8000.0],
                           lon = [LON, LON + 3.0, LON + 6.0], lat = [LAT, LAT + 2.0, LAT + 4.0])
        sites = @test_logs (:warn, r"moving platform") begin
            last(O.read_observation_requests(O.ObsPackSource(plane, O.SiteMode(), O.SiteCodeGrouping()), 1,
                                             ORIGIN, [Date(ORIGIN)]))
        end
        @test length(sites) == 3
    end
end

# A DEM stand-in for the placement rules.
struct ConstantDEM <: O.AbstractPointElevation
    z::Float64
end
O.point_elevation(d::ConstantDEM, lon::Real, lat::Real) = d.z

@testset "placement rules, layers and configuration" begin
    cell = CellLocation(1, 1, 2, 5, 5, -135.0, 0.0, 1.0)
    field = zeros(4, 3)
    field[1, 2] = 250.0
    zs = ModelSurfaceHeight(field, "zs.nc")
    terrain = ObservationTerrain(; model_surface = zs, dem = ConstantDEM(100.0))
    request(; intake = NaN, altitude = NaN, elevation = NaN, pressure = NaN) =
        SoundingRequest("x", 0.0, LON, LAT, 1, elevation, intake, altitude, pressure)
    above = AboveGroundPlacement()
    lowest = LowestLayerWithoutHeight()
    unplaced = UnplacedWithoutHeight()
    place(r, t = terrain; option = above, rule = lowest) = resolve_placement(option, rule, r, cell, t)
    p = place(request(intake = 30.0, altitude = 1000.0))
    @test p.method == CODES.record_height && p.height_agl_m == 30.0 && p.model_surface_m == 250.0
    p = place(request(altitude = 1000.0))
    @test p.method == CODES.altitude_minus_dem && p.height_agl_m == 900.0 && p.model_surface_m == 250.0
    # The model surface is not a ground: an altitude without elevation or DEM is unplaced.
    @test place(request(altitude = 1000.0), ObservationTerrain(; model_surface = zs)) === nothing
    @test place(request(altitude = 1000.0), ObservationTerrain()) === nothing
    # A pressure places what a height cannot, but never overrides one.
    p = place(request(pressure = 50000.0), ObservationTerrain())
    @test p.method == CODES.pressure && p.pressure_pa == 50000.0 && isnan(p.height_agl_m)
    p = place(request(altitude = 1000.0, pressure = 50000.0), ObservationTerrain(; model_surface = zs))
    @test p.method == CODES.pressure
    @test place(request(altitude = 1000.0, pressure = 50000.0)).method == CODES.altitude_minus_dem
    # No vertical coordinate: the source decides.
    p = place(request(), ObservationTerrain())
    @test p.method == CODES.lowest_layer && isnan(p.model_surface_m)
    @test place(request(elevation = 10.0); rule = unplaced) === nothing
    @test missing_height_rule(O.ObsPackSource("x.nc", O.SoundingMode(), O.SiteCodeGrouping())) === unplaced
    @test missing_height_rule(O.TableSource("x.csv", O.SoundingMode(), O.AutoTableFormat())) === lowest
    @test missing_height_rule(O.OCO2LiteSource("x.nc", 0)) === lowest
    # A station whose rows give only an altitude, with its elevation from the file.
    p = place(SiteRequest("a", LON, LAT, 3397.0, NaN, 1, O.EveryWindow(), 3437.0))
    @test p.method == CODES.record_height && p.height_agl_m == 40.0
    site = SiteRequest("s", LON, LAT, 3397.0, 40.0, 1)
    p = place(site; option = AbsoluteAltitudeAbove(0.0))
    @test p.method == CODES.absolute_altitude && p.height_agl_m == 3437.0 - 250.0
    @test place(site; option = AbsoluteAltitudeAbove(100.0)).method == CODES.record_height
    @test place(request(intake = 5000.0, pressure = 50000.0); option = AbsoluteAltitudeAbove(1000.0)).method ==
          CODES.pressure
    # An intake height without elevation or altitude takes its ground from the DEM
    # (100 m): absolute altitude 5100 m. Without a DEM it keeps its height above ground.
    p = place(request(intake = 5000.0); option = AbsoluteAltitudeAbove(1000.0))
    @test p.method == CODES.absolute_altitude && p.height_agl_m == 5100.0 - 250.0
    p = place(request(intake = 5000.0, elevation = 300.0); option = AbsoluteAltitudeAbove(1000.0))
    @test p.method == CODES.absolute_altitude && p.height_agl_m == 5300.0 - 250.0
    p = place(request(intake = 5000.0), ObservationTerrain(; model_surface = zs); option = AbsoluteAltitudeAbove(1000.0))
    @test p.method == CODES.record_height && p.height_agl_m == 5000.0
    @test place(request(); option = AbsoluteAltitudeAbove(0.0)).method == CODES.lowest_layer
    @test O.record_height_above_ground(NaN, 500.0, 100.0) == 400.0
    @test O.record_height_above_ground(10.0, 500.0, 100.0) == 10.0
    @test isnan(O.record_height_above_ground(NaN, 500.0, NaN))

    z_half = [1000.0, 500.0, 100.0, 0.0]
    p_half = [100.0, 30000.0, 70000.0, 100000.0]
    @test O.pressure_layer_index(p_half, 50.0) == 1
    @test O.pressure_layer_index(p_half, 30000.0) == 2
    @test O.pressure_layer_index(p_half, 69999.0) == 2
    @test O.pressure_layer_index(p_half, 70000.0) == 3
    @test O.pressure_layer_index(p_half, 101000.0) == 3
    @test_throws ArgumentError O.pressure_layer_index(p_half, NaN)
    # A pressure at or below zero (an undeclared -999.9 hPa fill) is not the column top.
    @test_throws ArgumentError O.pressure_layer_index(p_half, 0.0)
    @test_throws ArgumentError O.pressure_layer_index(p_half, -99990.0)
    @test !O.valid_pressure(0.0) && !O.valid_pressure(-99990.0) && !O.valid_pressure(Inf) && O.valid_pressure(1.0)
    for bad in (-99990.0, 0.0, -Inf)
        @test place(request(pressure = bad), ObservationTerrain(); rule = unplaced) === nothing
        @test place(request(pressure = bad), ObservationTerrain()).method == CODES.lowest_layer
        @test place(request(intake = 5000.0, pressure = bad); option = AbsoluteAltitudeAbove(1000.0)).method ==
              CODES.absolute_altitude
    end
    # Placements far below the ground are flagged, not silently put in the lowest layer.
    @test O.below_ground(place(request(altitude = -100.0)))                     # 200 m below the DEM ground
    @test !O.below_ground(place(request(altitude = 50.0)))                      # 50 m below: tolerance
    @test !O.below_ground(VerticalPlacement(CODES.absolute_altitude, -500.0, NaN, 250.0))
    @test placement_layer(VerticalPlacement(CODES.lowest_layer, NaN, NaN, NaN), z_half, p_half) == 3
    @test placement_layer(VerticalPlacement(CODES.pressure, NaN, 50000.0, NaN), z_half, p_half) == 2
    @test placement_layer(VerticalPlacement(CODES.altitude_minus_dem, 600.0, NaN, NaN), z_half, p_half) == 1
    @test_throws ArgumentError O.intake_layer_index(z_half, NaN)
    # Every code is described in the file attribute.
    @test all(k -> occursin("$(CODES[k]) = $(k)", O.vertical_placement_codes_description()), keys(CODES))

    mktempdir() do dir
        base = Dict{String, Any}("path" => joinpath(dir, "obs.nc"),
                                 "sources" => Any[table_source(joinpath(dir, "x.csv"))])
        spec(extra...) = observation_output_spec(Dict{String, Any}("observations" => merge(base, Dict(extra...))))
        @test spec().placement.unplaced isa ErrorOnUnplaced
        @test spec().placement.altitude_placement isa AboveGroundPlacement
        @test spec("unplaced_records" => "skip").placement.unplaced isa SkipUnplaced
        @test_throws ArgumentError spec("unplaced_records" => "lowest")
        @test_throws ArgumentError spec("dem_variable" => "z")
        @test_throws ArgumentError spec("absolute_altitude_above_agl_m" => 3000.0)
        zs_path = surface_file(dir, 0.0)
        s = spec("surface_height_file" => zs_path, "absolute_altitude_above_agl_m" => 3000)
        @test s.placement.altitude_placement == AbsoluteAltitudeAbove(3000.0)
        @test_throws ArgumentError spec("surface_height_file" => zs_path, "absolute_altitude_above_agl_m" => -1.0)
        @test_throws ArgumentError spec("surface_height_file" => "")
        # A spec naming a dataset the caller did not load, or a layout of another grid.
        m = model10()
        csv = joinpath(dir, "x.csv")
        write(csv, "id,time,lat,lon\na,2021-12-02T00:30:00,$(LAT),$(LON)\n")
        @test_throws ArgumentError quiet(() -> build_observation_sampler(spec("surface_height_file" => zs_path),
            m.state, m.grid; origin = ORIGIN, window_seconds = (0.0, 86400.0), halo_width = 0))
        @test_throws DimensionMismatch quiet(() -> build_observation_sampler(spec(), m.state, m.grid;
            origin = ORIGIN, window_seconds = (0.0, 86400.0), halo_width = 0,
            terrain = ObservationTerrain(; model_surface = ModelSurfaceHeight(zeros(2, 2), "small.nc"))))
        @test_throws ArgumentError DR._check_observation_placement_files(
            spec("surface_height_file" => joinpath(dir, "absent.nc")))
        @test DR._check_observation_placement_files(spec("surface_height_file" => zs_path)) === nothing
    end
end
