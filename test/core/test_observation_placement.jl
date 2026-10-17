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
                             UnplacedWithoutHeight, missing_height_rule, VerticalPlacementSpec
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

# `model10` with water vapour: the same total pressures (interfaces exactly
# A + B·1000 hPa), so the dry air mass of layer k is (1 − q_k) Δp_k A / g, with
# q = 0.02, 0.01, 0.005 in layers 10, 9, 8 and 0.001 above. The dry interfaces
# then lie 219.9 Pa (bottom of layer 9) and 119.9 Pa (bottom of layer 8) above
# the total ones. `basis = MoistBasis` stores the moist (total) air mass.
const MOIST_Q = [fill(0.001, 7); 0.005; 0.01; 0.02]
function moist_model10(; basis = DryBasis)
    m = model10()
    air = copy(m.state.air_mass)
    basis === DryBasis && (air .*= reshape(1 .- MOIST_Q, 1, 1, 10))
    co2 = air .* reshape(400e-6 .+ 20e-6 .* (0:9) ./ 9, 1, 1, 10)
    q = repeat(reshape(MOIST_Q, 1, 1, 10), 4, 3, 1)
    return (; state = CellState(basis, air; co2), grid = m.grid, mesh = m.mesh, q)
end

# An ObsPack file in the layout of the OCO-2 MIP daily files: optional
# variables are absent when `nothing`; NaN entries are written as the fill value.
function write_pack(path; altitude = nothing, elevation = nothing, intake = nothing, pressure = nothing,
                    pressure_units = "hPa", lon = LON, lat = LAT, time = ORIGIN + Minute(30))
    n = length(something(altitude, intake, pressure, elevation))
    NCDataset(path, "c") do ds
        defDim(ds, "obs", n)
        defVar(ds, "time", Float64, ("obs",); attrib = Dict("units" => "seconds since 1970-01-01T00:00:00Z"))[:] =
            unix.(time isa AbstractVector ? time : fill(time, n))
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
# two window ends, and return the soundings and sites file paths. `humidity`
# (one field for both window ends) is passed when given, and the build is told
# the run has humidity; a pair gives the humidity of each window end.
function sample(dir, sources; name = "obs", m = model10(), humidity = nothing, keys...)
    q0, q1 = humidity isa Tuple ? humidity : (humidity, humidity)
    cfg = Dict{String, Any}("path" => joinpath(dir, "snap.nc"), "hours" => [0.0], "split" => "single",
        "observations" => Dict{String, Any}("path" => joinpath(dir, "$(name).nc"), "sources" => sources,
                                            (String(k) => v for (k, v) in keys)...))
    spec = observation_output_spec(cfg)
    terrain = DR._observation_terrain(spec.placement, m.mesh)
    s = quiet() do
        build_observation_sampler(spec, m.state, m.grid; origin = ORIGIN, window_seconds = (0.0, 86400.0),
                                  halo_width = 0, terrain, has_humidity = humidity !== nothing)
    end
    begin_observation_day!(s, "20211202", 1)
    observe_window_boundary!(s, m.state, 0.0; next_window_seconds = 3600.0, humidity = q0)
    observe_window_boundary!(s, m.state, 3600.0; next_window_seconds = 3600.0, humidity = q1)
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
        path, _ = sample(dir, Any[obspack(pack)]; name = "prs", unplaced_records = "skip", pressure_basis = "dry")
        r = soundings(path)
        @test r.method == [CODES.pressure, CODES.pressure] && r.unplaced == 1
        @test r.pressure ≈ [65000.0, 65000.0] && all(isnan, r.height)
        @test r.level == [7, 7]
        @test all(r.p_half[7, k] <= 65000.0 < r.p_half[8, k] for k in 1:2)
        # Pressure units: case and blanks do not matter; an unknown or missing unit fails.
        pa = write_pack(joinpath(dir, "co2_pa_aircraft-insitu_1.nc"); pressure = [65000.0], pressure_units = " Pa ")
        @test soundings(first(sample(dir, Any[obspack(pa)]; name = "pa", pressure_basis = "dry"))).level == [7]
        for (k, units) in enumerate(("psi", ""))
            bad = write_pack(joinpath(dir, "co2_bad$(k)_aircraft-insitu_1.nc"); pressure = [650.0],
                             pressure_units = units)
            err = failure(() -> sample(dir, Any[obspack(bad)]; name = "bad$(k)", pressure_basis = "dry"))
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
                         absolute_altitude_above_agl_m = 1000.0, pressure_basis = "dry")
        r = soundings(path)
        @test r.method == [CODES.altitude_minus_dem, CODES.pressure, CODES.absolute_altitude]
        @test r.pressure[1:2] ≈ [95000.0, 65000.0] && isnan(r.pressure[3])
        @test isnan(r.height[2]) && r.height[[1, 3]] == [500.0, 8000.0]
        @test r.level == [10, 7, 4]
        k = r.level[2]
        @test r.p_half[k, 2] <= 65000.0 < r.p_half[k + 1, 2]
    end
end

# Every variable of two observation files, raw (fill values as stored).
file_variables(path) = NCDataset(ds -> Dict(k => Array(ds[k].var) for k in keys(ds)), path, "r")

@testset "pressure placement against total interface pressures (OF-116, issue #20)" begin
    mktempdir() do dir
        # 899 and 799 hPa lie between the dry and the total bottom interfaces of
        # layers 9 and 8 (89780.1 / 90000 and 79880.1 / 80000 Pa); 650 hPa is
        # mid-layer 7 for both. The intake-height rows are the control.
        pack = write_pack(joinpath(dir, "co2_prs_aircraft-insitu_1.nc"); pressure = [899.0, 799.0, 650.0, NaN, NaN, NaN],
                          intake = [NaN, NaN, NaN, ALTS...])
        m = moist_model10()
        # The analytic column: total interfaces from the dry air mass and q are A + B·1000 hPa.
        column = m.state.air_mass[1, 2, :]
        p_total = O.total_interface_pressures!(zeros(11), column, MOIST_Q, cell_area(m.mesh, 1, 2), GRAVITY, 100.0)
        @test p_total ≈ vcat(100.0, 1e4 .* (1:10)) rtol = 1e-14
        p_dry = O.interface_pressures!(zeros(11), column, cell_area(m.mesh, 1, 2), GRAVITY, 100.0)
        @test p_dry[9:10] ≈ [80000 - 119.9, 90000 - 219.9] rtol = 1e-12

        total_path, _ = sample(dir, Any[obspack(pack)]; name = "total", m, humidity = m.q)
        r = soundings(total_path)
        @test r.method == [fill(CODES.pressure, 3); fill(CODES.record_height, 3)]
        @test r.level[1:3] == [9, 8, 7]                       # by total pressure; [10, 9, 7] before
        @test r.ppm[1:3] ≈ 400 .+ 20 .* [8, 7, 6] ./ 9
        # The file keeps the dry interfaces: they would put the first two rows one layer lower.
        @test r.p_half[10, 1] < 89900 < 90000 && r.p_half[9, 2] < 79900 < 80000
        dry_path, _ = sample(dir, Any[obspack(pack)]; name = "dry", m, pressure_basis = "dry")
        d = soundings(dry_path)
        @test d.level[1:3] == [10, 9, 7]                      # the dry-interface rule (before issue #20)
        # Height placements and every other output do not depend on the pressure basis.
        @test d.level[4:6] == r.level[4:6] && d.ppm[4:6] == r.ppm[4:6]
        @test d.p_half == r.p_half && isequal(d.height, r.height) && d.method == r.method
        NCDataset(total_path, "r") do ds
            @test ds.attrib["pressure_basis"] == "total"
            @test ds.attrib["vertical_coordinates"] == "height_above_ground pressure"
        end
        NCDataset(dry_path, "r") do ds
            @test ds.attrib["pressure_basis"] == "dry"
        end
        # The default order written out is the same configuration: identical rows.
        explicit_path, _ = sample(dir, Any[obspack(pack)]; name = "explicit", m, humidity = m.q,
                                  vertical_coordinates = ["height_above_ground", "pressure"], pressure_basis = "total")
        @test isequal(file_variables(explicit_path), file_variables(total_path))

        # A record with a height and a pressure: placed by whichever coordinate comes first.
        both = write_pack(joinpath(dir, "co2_both_aircraft-pfp_1.nc"); intake = [500.0], pressure = [899.0])
        h_first = soundings(first(sample(dir, Any[obspack(both)]; name = "hfirst", m, humidity = m.q)))
        @test h_first.method == [CODES.record_height] && h_first.level == [10]
        p_first = soundings(first(sample(dir, Any[obspack(both)]; name = "pfirst", m, humidity = m.q,
                                         vertical_coordinates = ["pressure", "height_above_ground"])))
        @test p_first.method == [CODES.pressure] && p_first.level == [9]
        # Pressure only: the intake-height rows are unplaced, never put in the lowest layer.
        err = failure(() -> sample(dir, Any[obspack(pack)]; name = "ponly", m, humidity = m.q,
                                   vertical_coordinates = ["pressure"]))
        @test err isa ArgumentError
        @test occursin("3 point events", sprint(showerror, err)) &&
              occursin("vertical_coordinates = [\"pressure\"]", sprint(showerror, err))
        # Absolute altitude aloft re-places by the total pressure (899 hPa: layer 9, not the dry layer 10).
        aloft = write_pack(joinpath(dir, "co2_iag_aircraft-insitu_1.nc"); intake = [500.0, 3000.0],
                           elevation = [0.0, 0.0], pressure = [NaN, 899.0])
        abs_rows = soundings(first(sample(dir, Any[obspack(aloft)]; name = "abstotal", m, humidity = m.q,
                                          surface_height_file = surface_file(dir, 0.0),
                                          absolute_altitude_above_agl_m = 1000.0)))
        @test abs_rows.method == [CODES.record_height, CODES.pressure] && abs_rows.level[2] == 9
        # Height only: the pressure rows are unplaced, and no humidity is needed.
        h_only = soundings(first(sample(dir, Any[obspack(pack)]; name = "honly", m,
                                        vertical_coordinates = ["height_above_ground"], unplaced_records = "skip")))
        @test h_only.unplaced == 3 && h_only.method == fill(CODES.record_height, 3) && h_only.level == r.level[4:6]
    end
end

@testset "pressure basis: humidity requirements, moist-basis runs, time blending" begin
    mktempdir() do dir
        pack = write_pack(joinpath(dir, "co2_p_aircraft-insitu_1.nc"); pressure = [899.0])
        heights = write_pack(joinpath(dir, "co2_h_aircraft-pfp_1.nc"); intake = ALTS)
        m = moist_model10()
        # Without humidity the default total pressures cannot be built: the error names the opt-in.
        err = failure(() -> sample(dir, Any[obspack(pack)]; name = "nohumidity", m))
        msg = sprint(showerror, err)
        @test err isa ArgumentError && occursin("1 requests are placed by pressure", msg) &&
              occursin("[output.observations].pressure_basis = \"dry\"", msg)
        # Without pressure-placed records no humidity is needed.
        @test soundings(first(sample(dir, Any[obspack(heights)]; name = "heights", m))).method ==
              fill(CODES.record_height, 3)
        # A sampler that needs the humidity refuses a window end without it, or of the wrong layout.
        spec = observation_output_spec(Dict{String, Any}("observations" => Dict{String, Any}(
            "path" => joinpath(dir, "strict.nc"), "sources" => Any[obspack(pack)])))
        s = quiet(() -> build_observation_sampler(spec, m.state, m.grid; origin = ORIGIN,
                                                  window_seconds = (0.0, 86400.0), halo_width = 0, has_humidity = true))
        @test s.needs_humidity && s.pressure_basis isa O.TotalFromHumidity
        begin_observation_day!(s, "20211202", 1)
        @test_throws ArgumentError observe_window_boundary!(s, m.state, 0.0; next_window_seconds = 3600.0)
        @test_throws DimensionMismatch observe_window_boundary!(s, m.state, 0.0; next_window_seconds = 3600.0,
                                                                humidity = m.q[:, :, 1:9])
        close(s)

        # A moist-basis run: the air mass is the total mass, its interfaces are total pressures without
        # humidity, and the dry opt-in is refused.
        moist = moist_model10(; basis = MoistBasis)
        moist_path, _ = sample(dir, Any[obspack(pack)]; name = "moist", m = moist)
        @test NCDataset(ds -> Int.(ds["intake_level"][:]), moist_path, "r") == [9]
        err = failure(() -> sample(dir, Any[obspack(pack)]; name = "moistdry", m = moist, pressure_basis = "dry"))
        @test err isa ArgumentError && occursin("needs a dry-basis run", sprint(showerror, err))

        # Linear time interpolation blends the humidity like the air mass: q0 at t = 0, q0/2 at 1 h, so
        # 0.75 q0 at 0:30. a < b < c are the total bottom interfaces of layer 9 for q0/2, 0.75 q0 and q0;
        # (a+b)/2 is in layer 9 and (b+c)/2 in layer 10 only with the blended humidity.
        area = cell_area(m.mesh, 1, 2)
        column = m.state.air_mass[1, 2, :]
        bottom9(q) = O.total_interface_pressures!(zeros(11), column, q, area, GRAVITY, 100.0)[10]
        a, b, c = bottom9(MOIST_Q ./ 2), bottom9(0.75 .* MOIST_Q), bottom9(MOIST_Q)
        @test a < b < c
        pressures = [(a + b) / 2, (b + c) / 2] ./ 100
        times = [ORIGIN + Minute(30), ORIGIN + Minute(30), ORIGIN + Hour(1), ORIGIN + Hour(1)]
        timed = write_pack(joinpath(dir, "co2_t_aircraft-insitu_1.nc"); pressure = [pressures; pressures], time = times)
        linear = soundings(first(sample(dir, Any[obspack(timed)]; name = "linear", m, humidity = (m.q, m.q ./ 2))))
        # 0:30 blended; 1:00 is the final window end, emitted from its own sample (q0/2).
        @test linear.level == [9, 10, 10, 10]
        nearest = soundings(first(sample(dir, Any[obspack(timed)]; name = "nearest", m, humidity = (m.q, m.q ./ 2),
                                         time_interpolation = "nearest_window")))
        @test nearest.level == [10, 10, 10, 10]               # 0:30 rounds to the 1 h window end (q0/2)
    end
end

@testset "cubed sphere: total pressures from VDIFF humidity panels" begin
    mktempdir() do dir
        # Two layers, total interfaces 100, 50000, 100000 Pa; q = 0.001 and 0.02. The dry interface
        # between the layers is 100 + 0.999·49900 = 49950.1 Pa, so 499.8 hPa is in layer 1 by total
        # pressure and in layer 2 by dry pressure.
        mesh = CubedSphereMesh(; FT = Float64, Nc = 4, Hp = 2)
        grid = AtmosGrid(mesh, HybridSigmaPressure([100.0, 0.0, 0.0], [0.0, 0.5, 1.0]), CPU(); FT = Float64)
        q, dp = [0.001, 0.02], [49900.0, 50000.0]
        air = ntuple(6) do _
            a = zeros(8, 8, 2)
            for k in 1:2, j in 1:4, i in 1:4
                a[i + 2, j + 2, k] = (1 - q[k]) * dp[k] * cell_area(mesh, i, j) / GRAVITY
            end
            a
        end
        state = CubedSphereState(DryBasis, air; co2 = map(m -> m .* 400e-6, air), halo_width = 2)
        pack = write_pack(joinpath(dir, "co2_cs_aircraft-insitu_1.nc"); pressure = [499.8], lon = 100.0, lat = 0.0)
        interior = ntuple(_ -> repeat(reshape(q, 1, 1, 2), 4, 4, 1), 6)       # GCHP VDIFF layout
        padded = ntuple(_ -> repeat(reshape(q, 1, 1, 2), 8, 8, 1), 6)         # air-mass layout
        function level(name, humidity; keys...)
            spec = observation_output_spec(Dict{String, Any}("observations" => Dict{String, Any}(
                "path" => joinpath(dir, "$(name).nc"), "sources" => Any[obspack(pack)],
                (String(k) => v for (k, v) in keys)...)))
            s = quiet(() -> build_observation_sampler(spec, state, grid; origin = ORIGIN,
                window_seconds = (0.0, 86400.0), halo_width = 2, has_humidity = humidity !== nothing))
            begin_observation_day!(s, "20211202", 1)
            observe_window_boundary!(s, state, 0.0; next_window_seconds = 3600.0, humidity)
            observe_window_boundary!(s, state, 3600.0; next_window_seconds = 3600.0, humidity)
            quiet(() -> finish_observations!(s)); close(s)
            return NCDataset(ds -> Int.(ds["intake_level"][:]), joinpath(dir, "$(name)_soundings.nc"), "r")
        end
        @test level("interior", interior) == [1]
        @test level("padded", padded) == [1]
        @test level("dry", nothing; pressure_basis = "dry") == [2]
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
        path, _ = sample(dir, Any[obspack(pack)]; name = "pfill", unplaced_records = "skip", pressure_basis = "dry")
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
    place(r, t = terrain; option = above, rule = lowest, coordinates = O.DEFAULT_VERTICAL_COORDINATES) =
        resolve_placement(VerticalPlacementSpec(; altitude_placement = option, coordinates), rule, r, cell, t)
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

    # Coordinate order (issue #20): the first configured coordinate that places a record wins.
    H, P = O.HeightAboveGroundCoordinate(), O.PressureCoordinate()
    both = request(intake = 30.0, pressure = 50000.0)
    for (coordinates, method) in (((H, P), CODES.record_height), ((P, H), CODES.pressure),
                                  ((P,), CODES.pressure), ((H,), CODES.record_height))
        @test place(both; coordinates).method == method
    end
    @test place(both; coordinates = (P, H)).pressure_pa == 50000.0
    height_only = request(altitude = 1000.0)                  # 900 m above the 100 m DEM ground
    @test place(height_only; coordinates = (P, H)).method == CODES.altitude_minus_dem
    @test place(height_only; coordinates = (P,)) === nothing   # a coordinate the run does not use
    pressure_only = request(pressure = 50000.0)
    @test place(pressure_only; coordinates = (H, P)).method == CODES.pressure
    @test place(pressure_only; coordinates = (H,)) === nothing
    @test place(request(intake = 30.0); coordinates = (P,)) === nothing
    for coordinates in ((H, P), (P, H), (P,), (H,))
        # Neither: the source decides (column and surface products go to the lowest layer).
        @test place(request(elevation = 10.0); coordinates).method == CODES.lowest_layer
        @test place(request(elevation = 10.0); coordinates, rule = unplaced) === nothing
    end
    # Aloft, absolute-altitude placement prefers the pressure only when pressure is a coordinate.
    aloft = request(intake = 5000.0, elevation = 300.0, pressure = 50000.0)
    @test place(aloft; option = AbsoluteAltitudeAbove(1000.0), coordinates = (H, P)).method == CODES.pressure
    p = place(aloft; option = AbsoluteAltitudeAbove(1000.0), coordinates = (H,))
    @test p.method == CODES.absolute_altitude && p.height_agl_m == 5300.0 - 250.0

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
