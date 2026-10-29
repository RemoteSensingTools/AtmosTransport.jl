#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# `bind_to_mesh`: schema-v2 observation records -> `CSObservation`s of the CS
# 4D-Var path.
#
# Coverage:
#   * Time. Step `k` covers `[t_start + (k-1)dt, t_start + k dt)`; records
#     without a window sample that step, records with `time_window_s > 0` the
#     overlap-weighted steps of their window (weights sum to 1, truncated and
#     renormalised at the run's ends); out-of-range policies.
#   * Geography. Float64 coordinates bind to the cell `Observations.locate` gives,
#     also within Float32 rounding of a cell edge (OF-84); seams, wraps.
#   * Vertical. Point records become `CSPlacedPointObjective`s placed by the
#     forward sampler's rule (`resolve_placement`): intake height, altitude
#     minus elevation, altitude minus a DEM, pressure; the coordinate order;
#     unplaced records (error / skip). Column records become column means.
#   * Flags, model sigma, record ids, tracer filter, argument validation.
# ---------------------------------------------------------------------------

using Test
using Dates: DateTime, Second, year, month, day, hour, minute, second
using Random: MersenneTwister
using Logging: NullLogger, with_logger

import AtmosTransport
const AT = AtmosTransport
const O = AT.Observations

const NC_TEST = 6
const DT_TEST = 3600.0                 # 1 hour per step
const T0_TEST = DateTime(2024, 6, 15, 0, 0, 0)
const ORIGIN_TEST = "2024-06-15 00:00:00"

_mesh() = AT.CubedSphereMesh(; Nc = NC_TEST, FT = Float64)

function _dc_at(offset::Real)
    t = T0_TEST + Second(round(Int, offset))
    return (year(t), month(t), day(t), hour(t), minute(t), second(t))
end

# A column record (bound to a column mean) unless `kind = "point"`.
_record(; id, dc, lat, lon, kind = "column", value = 420.0, value_sigma = 0.5, tracer = "CO2",
        instrument = "TCCON", kwargs...) =
    AT.CSObservationRecord(; id, date_components = dc, lat, lon, kind, value, value_sigma,
                           instrument_type = instrument, tracer, kwargs...)

_set(records) = AT.CSObservationSet(records; time_origin = ORIGIN_TEST)
_bind(records; kwargs...) = AT.bind_to_mesh(_set(records), _mesh(), T0_TEST, DT_TEST; nsteps = 24, kwargs...)
_cell(o) = (o.objective.panel, o.objective.i, o.objective.j)
_steps(o) = o.sampling.steps

# ---------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------

@testset "bind_to_mesh — time mapping (left-closed half-open)" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 1)
    lat0, lon0 = lats[3, 4], lons[3, 4]
    offsets = (0, DT_TEST, 2DT_TEST, 3DT_TEST - 1, DT_TEST / 2)
    obs = _bind([_record(id = k, dc = _dc_at(t), lat = lat0, lon = lon0) for (k, t) in enumerate(offsets)])
    @test [o.step for o in obs] == [1, 2, 3, 3, 1]
    @test all(o -> o.sampling == AT.CSTimeSampling(o.step), obs)
end

@testset "bind_to_mesh — window weights (Δt/ΣΔt over overlapping steps)" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 2)
    at(t, window; kwargs...) = only(_bind([_record(id = 1, dc = _dc_at(t), lat = lats[2, 2], lon = lons[2, 2],
                                                  time_window_s = window)]; kwargs...)).sampling
    # Step k covers [k − 1, k) h. A 3 h window centred on 05:30 covers [4, 7) h:
    # steps 5, 6 and 7 whole, weights 1/3 each.
    s = at(5.5DT_TEST, 3DT_TEST)
    @test s.steps == [5, 6, 7] && s.weights ≈ fill(1 / 3, 3) && sum(s.weights) ≈ 1
    s = at(5DT_TEST, 2DT_TEST)                      # [4, 6) h: steps 5 and 6 whole
    @test s.steps == [5, 6] && s.weights == [0.5, 0.5]
    s = at(5.25DT_TEST, DT_TEST)                    # [4.75, 5.75) h: 1/4 of step 5, 3/4 of step 6
    @test s.steps == [5, 6] && s.weights ≈ [0.25, 0.75] && sum(s.weights) ≈ 1
    # A window inside one step is the instantaneous sample of that step.
    @test at(5.5DT_TEST, 600) == AT.CSTimeSampling(6)
    @test at(5.5DT_TEST, 0) == AT.CSTimeSampling(6)
    # Truncated at the run start and end, renormalised over the part inside.
    s = at(0.5DT_TEST, 4DT_TEST)                    # [−1.5, 2.5) h → steps 1, 2, 3: 1, 1, 0.5 h
    @test s.steps == [1, 2, 3] && s.weights ≈ [0.4, 0.4, 0.2] && sum(s.weights) ≈ 1
    s = at(23.5DT_TEST, 4DT_TEST)                   # [21.5, 25.5) h → steps 22, 23, 24: 0.5, 1, 1 h
    @test s.steps == [22, 23, 24] && s.weights ≈ [0.2, 0.4, 0.4] && sum(s.weights) ≈ 1
    @test at(23.5DT_TEST, 100DT_TEST).steps == 1:24
    @test sum(at(23.5DT_TEST, 100DT_TEST).weights) ≈ 1
    # A window of any length is clipped to the run (at most nsteps steps).
    @test at(12DT_TEST, 1e12).steps == 1:24
    # Windows need the run length.
    err = try
        AT.bind_to_mesh(_set([_record(id = 1, dc = _dc_at(23.5DT_TEST), lat = lats[2, 2], lon = lons[2, 2],
                                      time_window_s = 4DT_TEST)]), _mesh(), T0_TEST, DT_TEST)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("needs `nsteps`", err.msg)
    # Out-of-range window records: the record time decides.
    @test_throws ArgumentError at(-DT_TEST, 4DT_TEST)
    @test (@test_logs (:info, r"1 records outside the run are skipped") _bind([_record(id = 1,
        dc = _dc_at(-DT_TEST), lat = lats[2, 2], lon = lons[2, 2], time_window_s = 4DT_TEST)];
        out_of_range_policy = "skip")) == AT.CSObservation[]
end

@testset "bind_to_mesh — ClampOutOfRange keeps the window's part inside the run" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 2)
    rec(id, t, window) = _record(; id, dc = _dc_at(t), lat = lats[2, 2], lon = lons[2, 2], time_window_s = window)
    records = [rec(1, -0.5DT_TEST, 5DT_TEST),     # [−3, 2) h: steps 1 and 2 of the run, renormalised
               rec(2, 25DT_TEST, 6DT_TEST),       # [22, 28) h: steps 23 and 24
               rec(3, -5DT_TEST, 2DT_TEST),       # [−6, −4) h misses the run: the first step
               rec(4, 30DT_TEST, 0),              # no window: the last step
               rec(5, 3.5DT_TEST, 0)]             # inside the run
    obs = @test_logs (:info, r"4 records outside the run are clamped") _bind(records; out_of_range_policy = "clamp")
    @test [o.sampling for o in obs] == [AT.CSTimeSampling([1, 2], [0.5, 0.5]), AT.CSTimeSampling([23, 24], [0.5, 0.5]),
                                        AT.CSTimeSampling(1), AT.CSTimeSampling(24), AT.CSTimeSampling(4)]
    # The observation is complete after its last sampled step.
    o = only(_bind([_record(id = 1, dc = _dc_at(5.5DT_TEST), lat = lats[2, 2], lon = lons[2, 2],
                            time_window_s = 3DT_TEST)]))
    @test o.step == 7
end

@testset "CSTimeSampling validation" begin
    @test AT.CSTimeSampling(3) == AT.CSTimeSampling([3], [1.0])
    @test_throws ArgumentError AT.CSTimeSampling(Int[], Float64[])
    @test_throws DimensionMismatch AT.CSTimeSampling([1, 2], [1.0])
    @test_throws ArgumentError AT.CSTimeSampling([0], [1.0])
    @test_throws ArgumentError AT.CSTimeSampling([2, 1], [0.5, 0.5])
    @test_throws ArgumentError AT.CSTimeSampling([1, 1], [0.5, 0.5])
    @test_throws ArgumentError AT.CSTimeSampling([1, 2], [0.5, 0.6])
    @test_throws ArgumentError AT.CSTimeSampling([1, 2], [1.0, 0.0])
    @test_throws ArgumentError AT.CSObservation(0, AT.CSColumnMeanObjective(1, 1, 1), 1.0, 1.0)
    obs = AT.CSObservation(AT.CSTimeSampling([2, 4], [0.25, 0.75]), AT.CSColumnMeanObjective(1, 1, 1),
                           1.0, 0.5; id = 7)
    @test obs.step == 4 && obs.id == 7
    @test AT.CSObservation(2, AT.CSColumnMeanObjective(1, 1, 1), 1.0, 0.5).id == 0
end

@testset "bind_to_mesh — out-of-range policies" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 1)
    lat0, lon0 = lats[2, 2], lons[2, 2]
    records = [_record(id = 10, dc = _dc_at(-DT_TEST / 2), lat = lat0, lon = lon0),
               _record(id = 11, dc = _dc_at(0), lat = lat0, lon = lon0),
               _record(id = 12, dc = _dc_at(5DT_TEST + 1), lat = lat0, lon = lon0)]
    @test_throws ArgumentError _bind(records; nsteps = 5)
    obs_skip = _bind(records; nsteps = 5, out_of_range_policy = AT.SkipOutOfRange())
    @test length(obs_skip) == 1 && obs_skip[1].step == 1 && obs_skip[1].id == 11
    @test obs_skip[1].objective isa AT.CSColumnMeanObjective
    @test [o.step for o in _bind(records; nsteps = 5, out_of_range_policy = AT.ClampOutOfRange())] == [1, 1, 5]
end

# An observation at exactly `t_start + nsteps*dt` maps to step `nsteps + 1`.
@testset "bind_to_mesh — terminal boundary t_start + nsteps*dt" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 1)
    records = [_record(id = 21, dc = _dc_at(5DT_TEST), lat = lats[2, 2], lon = lons[2, 2])]
    @test_throws ArgumentError _bind(records; nsteps = 5)
    @test isempty(_bind(records; nsteps = 5, out_of_range_policy = "skip"))
    @test only(_bind(records; nsteps = 5, out_of_range_policy = "clamp")).step == 5
end

@testset "bind_to_mesh — argument validation" begin
    set = _set([_record(id = 1, dc = _dc_at(0), lat = 0.0, lon = 0.0)])
    mesh = _mesh()
    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, -1.0)
    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, 0.0)
    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 0)
    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; out_of_range_policy = "bogus")
    for removed in (:reject, :skip, :clamp)
        error = try
            AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; out_of_range_policy = removed)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("Symbol policies are not supported", sprint(showerror, error))
    end
    # A model surface height of the wrong grid, or files named without loaded terrain.
    wrong = O.ModelSurfaceHeight(ntuple(_ -> zeros(3, 3), 6), "wrong.nc")
    @test_throws DimensionMismatch AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                                                   terrain = O.ObservationTerrain(; model_surface = wrong))
    err = try
        AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; placement = O.VerticalPlacementSpec(; dem_file = "dem.nc"))
    catch e
        e
    end
    @test err isa ArgumentError && occursin("[observations.placement].dem_file", err.msg)
end

@testset "bind_to_mesh — tracer filter, flags, model sigma, ids" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 1)
    lat0, lon0 = lats[1, 1], lons[1, 1]
    records = [_record(id = 1, dc = _dc_at(0), lat = lat0, lon = lon0, tracer = "CO2", value = 420.0),
               _record(id = 2, dc = _dc_at(DT_TEST), lat = lat0, lon = lon0, tracer = "CH4", value = 1900.0),
               _record(id = 3, dc = _dc_at(2DT_TEST), lat = lat0, lon = lon0, tracer = "CO2", value = 421.0,
                       model_sigma = 1.2),
               _record(id = 4, dc = _dc_at(2DT_TEST), lat = lat0, lon = lon0, tracer = "CO2", flag = 1)]
    co2 = with_logger(NullLogger()) do
        _bind(records; tracer_filter = "CO2")
    end
    @test [o.value for o in co2] == [420.0, 421.0]
    @test [o.id for o in co2] == [1, 3]
    @test co2[1].sigma === 0.5                       # exactly value_sigma without a model error
    @test co2[2].sigma ≈ sqrt(0.5^2 + 1.2^2)
    @test (@test_logs (:info, r"1 records with a nonzero flag") _bind(records[4:4])) == AT.CSObservation[]
    ch4 = _bind(records; tracer_filter = "CH4")
    @test length(ch4) == 1 && ch4[1].value == 1900.0
end

# ---------------------------------------------------------------------------
# Geographic mapping
# ---------------------------------------------------------------------------

@testset "bind_to_mesh — geographic round trip at cell centres" begin
    targets = [(1, 2, 3), (2, 4, 5), (3, 1, 1), (4, 3, 6), (5, 5, 2), (6, 6, 6)]
    records = map(enumerate(targets)) do (k, (p, i, j))
        lons, lats = AT.panel_cell_center_lonlat(_mesh(), p)
        _record(id = k, dc = _dc_at((k - 1) * DT_TEST), lat = lats[i, j], lon = lons[i, j])
    end
    @test [_cell(o) for o in _bind(records)] == targets
end

@testset "bind_to_mesh — sub-cell perturbations stay in same cell" begin
    p, i, j = 3, 4, 4
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), p)
    deltas = ((0.0, 0.0), (0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3))
    records = [_record(id = k, dc = _dc_at((k - 1) * DT_TEST), lat = lats[i, j] + d[1], lon = lons[i, j] + d[2])
               for (k, d) in enumerate(deltas)]
    @test all(o -> _cell(o) == (p, i, j), _bind(records))
end

@testset "bind_to_mesh — same cell as the forward sampler's locator" begin
    locator = O.cell_locator(_mesh())
    rng = MersenneTwister(4242)
    # Uniform points on the sphere: many lie near cell edges, corners and seams.
    points = [(rad2deg(asin(2rand(rng) - 1)), 360rand(rng) - 180) for _ in 1:2000]
    obs = _bind([_record(id = k, dc = _dc_at(0), lat = lat, lon = lon) for (k, (lat, lon)) in enumerate(points)])
    cells = [O.locate(locator, lon, lat) for (lat, lon) in points]
    @test [_cell(o) for o in obs] == [(c.panel, c.i, c.j) for c in cells]
    lat, lon = points[1]
    wrapped(l) = _cell(only(_bind([_record(id = 1, dc = _dc_at(0), lat = lat, lon = l)])))
    @test wrapped(lon) == wrapped(lon + 360) == wrapped(lon - 360)
    # Longitudes next to the 0/360 and 180 meridians: the longitude reaches
    # `locate` as given (reducing −1e-14 with mod(lon, 360) gives 360.0 and the
    # neighbouring cell).
    boundary = [(0.0, -1e-14), (0.0, 360 - 1e-13), (0.0, 180 + 1e-13), (0.0, 180 - 1e-13), (0.0, -180 + 1e-13),
                (10.0, -1e-14), (-35.0, 360 - 1e-13), (0.0, 0.0), (0.0, 360.0)]
    obs = _bind([_record(id = k, dc = _dc_at(0), lat = lat, lon = lon) for (k, (lat, lon)) in enumerate(boundary)])
    @test [_cell(o) for o in obs] == [(c = O.locate(locator, lon, lat); (c.panel, c.i, c.j)) for (lat, lon) in boundary]
    reduced = O.locate(locator, mod(-1e-14, 360.0), 0.0)
    @test (reduced.panel, reduced.i, reduced.j) != _cell(obs[1])      # the case the reduction got wrong
end

@testset "bind_to_mesh — record ids are unique" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 1)
    r(id) = _record(; id, dc = _dc_at(0), lat = lats[1, 1], lon = lons[1, 1])
    err = try AT.CSObservationSet([r(1), r(2), r(1)]) catch e e end
    @test err isa ArgumentError && occursin("observation id 1 appears in rows 1 and 3", err.msg)
    set = _set([r(1), r(2)])
    push!(set.records, r(2))                       # a set changed after construction
    @test_throws ArgumentError AT.bind_to_mesh(set, _mesh(), T0_TEST, DT_TEST; nsteps = 24)
end

# Points within Float32 rounding of a cell edge: the Float64 record binds to the
# cell that contains it, where the Float32-rounded point of a v1 file lies in the
# neighbouring cell (OF-84).
@testset "bind_to_mesh — Float64 point next to a cell edge binds as locate does" begin
    mesh = _mesh()
    locator = O.cell_locator(mesh)
    cell64(lat, lon) = (c = O.locate(locator, lon, lat); (c.panel, c.i, c.j))
    lons, lats = AT.panel_cell_center_lonlat(mesh, 2)
    found = Tuple{Float64, Float64}[]
    for (a, b) in (((3, 3), (4, 3)), ((2, 2), (2, 3)), ((5, 1), (5, 2)))
        lat_a, lon_a = Float64(lats[a...]), Float64(lons[a...])
        lat_b, lon_b = Float64(lats[b...]), Float64(lons[b...])
        point(f) = (lat_a + f * (lat_b - lat_a), lon_a + f * (lon_b - lon_a))
        lo, hi = 0.0, 1.0                        # bisect the edge between the two centres
        for _ in 1:200
            mid = (lo + hi) / 2
            cell64(point(mid)...) == cell64(point(0.0)...) ? (lo = mid) : (hi = mid)
        end
        for δ in (1e-9, 3e-9, 1e-8, 3e-8, 1e-7), f in (lo - δ, hi + δ)
            lat, lon = point(f)
            if cell64(lat, lon) != cell64(Float64(Float32(lat)), Float64(Float32(lon)))
                push!(found, (lat, lon))
                break
            end
        end
    end
    @test length(found) >= 2
    for (lat, lon) in found
        o = only(_bind([_record(id = 1, dc = _dc_at(0), lat = lat, lon = lon, kind = "point",
                                intake_height_m = 10.0)]))
        @test _cell(o) == cell64(lat, lon)
        @test _cell(o) != cell64(Float64(Float32(lat)), Float64(Float32(lon)))
    end
end

# ---------------------------------------------------------------------------
# Vertical placement of point records (the forward sampler's rule)
# ---------------------------------------------------------------------------

# A DEM that gives one elevation everywhere.
const DEM_1500 = AT.GriddedIO.LatLonPointElevation(collect(0.0:10.0:350.0), collect(-90.0:10.0:90.0),
                                                   fill(1500.0f0, 36, 19), true, "constant-dem")

@testset "bind_to_mesh — point records are placed by the forward rule" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 4)
    lat0, lon0 = lats[3, 3], lons[3, 3]
    point(id; kwargs...) = _record(; id, dc = _dc_at(1800), lat = lat0, lon = lon0, kind = "point",
                                   instrument = "ICOS", kwargs...)
    records = [point(1; intake_height_m = 10.0),                               # surface intake
               point(2; altitude_m = 5000.0, elevation_m = 120.0),             # aircraft with ground elevation
               point(3; pressure_pa = 50000.0),                                # pressure only
               point(4; intake_height_m = 30.0, pressure_pa = 99000.0),        # both: height first
               point(5; altitude_m = 5000.0)]                                  # altitude, DEM ground
    obs = _bind(records; terrain = O.ObservationTerrain(; dem = DEM_1500))
    codes = O.VERTICAL_PLACEMENT_CODES
    @test all(o -> o.objective isa AT.CSPlacedPointObjective, obs)
    @test all(o -> _cell(o) == (4, 3, 3), obs)
    placements = [o.objective.placement for o in obs]
    @test [p.method for p in placements] ==
          [codes.record_height, codes.record_height, codes.pressure, codes.record_height, codes.altitude_minus_dem]
    @test placements[1].height_agl_m == 10.0
    @test placements[2].height_agl_m == 4880.0
    @test placements[3].pressure_pa == 50000.0 && isnan(placements[3].height_agl_m)
    @test placements[4].height_agl_m == 30.0
    @test placements[5].height_agl_m == 3500.0
    @test all(o -> o.objective.pressure_basis === O.TotalFromHumidity(), obs)
    @test obs[1].objective.area == AT.cell_area(_mesh(), 3, 3)
    # Each placement is `resolve_placement` of the record as a forward request.
    locator = O.cell_locator(_mesh())
    cell = O.locate(locator, lon0, lat0)
    for (o, r) in zip(obs, records)
        request = O.SoundingRequest(string(r.id), 1800.0, r.lon, r.lat, 1, r.elevation_m, r.intake_height_m,
                                    r.altitude_m, r.pressure_pa)
        expected = O.resolve_placement(O.VerticalPlacementSpec(), O.UnplacedWithoutHeight(), request, cell,
                                       O.ObservationTerrain(; dem = DEM_1500))
        @test o.objective.placement == expected
    end
    # Pressure first; dry pressure basis; pressure-only coordinates.
    pressure_first = O.VerticalPlacementSpec(; coordinates = (O.PressureCoordinate(), O.HeightAboveGroundCoordinate()),
                                             pressure_basis = O.DryInterfaces())
    o4 = only(_bind(records[4:4]; placement = pressure_first))
    @test o4.objective.placement.method == codes.pressure && o4.objective.placement.pressure_pa == 99000.0
    @test o4.objective.pressure_basis === O.DryInterfaces()
    # A column record is the column mean of its cell, whatever the placement.
    @test only(_bind([_record(id = 9, dc = _dc_at(0), lat = lat0, lon = lon0)])).objective ==
          AT.CSColumnMeanObjective(4, 3, 3)
end

@testset "bind_to_mesh — unplaced point records: error or skip and count" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 1)
    point(id; kwargs...) = _record(; id, dc = _dc_at(0), lat = lats[2, 2], lon = lons[2, 2], kind = "point",
                                   kwargs...)
    placed = point(1; intake_height_m = 10.0)
    no_height = point(2)                                  # no vertical coordinate at all
    altitude_only = point(3; altitude_m = 2800.0)         # no elevation, no DEM
    pressure_only = point(4; pressure_pa = 70000.0)
    failure(f) = try f(); nothing catch e e end
    err = failure(() -> _bind([placed, no_height, altitude_only]))
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("bind_to_mesh cannot place every request", msg)
    @test occursin("2 point events", msg) && occursin("[observations.placement].unplaced_records", msg)
    skip = O.VerticalPlacementSpec(; unplaced = O.SkipUnplaced())
    kept = @test_logs (:warn, r"bind_to_mesh skips requests it cannot place") match_mode = :any begin
        _bind([placed, no_height, altitude_only]; placement = skip)
    end
    @test [o.id for o in kept] == [1]
    # The DEM places the altitude-only record; a pressure-only record is
    # unplaced when the run leaves pressure out of the coordinates.
    @test length(_bind([placed, altitude_only]; terrain = O.ObservationTerrain(; dem = DEM_1500))) == 2
    height_only = O.VerticalPlacementSpec(; coordinates = (O.HeightAboveGroundCoordinate(),))
    @test failure(() -> _bind([pressure_only]; placement = height_only)) isa ArgumentError
    @test only(_bind([pressure_only])).objective.placement.method == O.VERTICAL_PLACEMENT_CODES.pressure
end

@testset "bind_to_mesh — below-ground records are kept and reported" begin
    lons, lats = AT.panel_cell_center_lonlat(_mesh(), 1)
    deep = _record(id = 1, dc = _dc_at(0), lat = lats[2, 2], lon = lons[2, 2], kind = "point",
                   altitude_m = 100.0, elevation_m = 900.0)
    o = @test_logs (:warn, r"more than 100.0 m below the") only(_bind([deep]))
    @test o.objective.placement.height_agl_m == -800.0
end

# ---------------------------------------------------------------------------
# Hand-built observations and file round trips
# ---------------------------------------------------------------------------

@testset "bind_to_mesh — bit-exact equivalence to hand-built CSObservation" begin
    targets = [(step = 1, panel = 1, i = 2, j = 3, value = 410.1, sigma = 0.5),
               (step = 3, panel = 4, i = 5, j = 1, value = 411.2, sigma = 0.7),
               (step = 7, panel = 6, i = 1, j = 6, value = 415.0, sigma = 1.0)]
    literal = [AT.CSObservation(t.step, AT.CSColumnMeanObjective(t.panel, t.i, t.j), t.value, t.sigma)
               for t in targets]
    records = map(enumerate(targets)) do (k, t)
        lons, lats = AT.panel_cell_center_lonlat(_mesh(), t.panel)
        _record(id = k, dc = _dc_at((t.step - 0.5) * DT_TEST), lat = lats[t.i, t.j], lon = lons[t.i, t.j],
                value = t.value, value_sigma = t.sigma)
    end
    bound = _bind(records)
    @test length(bound) == length(literal)
    for (a, b) in zip(literal, bound)
        @test a.step == b.step && a.sampling == b.sampling
        @test a.objective == b.objective
        @test a.value === b.value && a.sigma === b.sigma
    end
end

@testset "bind_to_mesh — survives write_observations / read_observations" begin
    targets = [(1, 2, 2), (2, 3, 3), (5, 4, 5)]
    records = map(enumerate(targets)) do (k, (p, i, j))
        lons, lats = AT.panel_cell_center_lonlat(_mesh(), p)
        _record(id = k, dc = _dc_at((k - 1) * DT_TEST), lat = lats[i, j], lon = lons[i, j],
                kind = isodd(k) ? "point" : "column", intake_height_m = isodd(k) ? 25.0 : NaN,
                time_window_s = k * 1800, value = 420.0 + k, value_sigma = 0.1k)
    end
    mktempdir() do dir
        path = joinpath(dir, "obs_v2.nc")
        AT.write_observations(path, _set(records))
        a = _bind(records)
        b = _bind(AT.read_observations(path).records)
        @test length(a) == length(b) == length(targets)
        for (oa, ob) in zip(a, b)
            @test oa.sampling == ob.sampling && oa.objective == ob.objective
            @test oa.value === ob.value && oa.sigma === ob.sigma && oa.id == ob.id
        end
    end
end
