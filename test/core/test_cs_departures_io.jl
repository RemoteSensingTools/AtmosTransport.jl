#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Departures IO, schema v2 (`schemas/cs_departures_v2.toml`).
#
# Coverage:
#   * `CSDepartureRecord`: dtype coercion, finite and range checks, the
#     placement code / level conventions of point and column records, the
#     time-sampling invariants, and the sign convention with the model error
#     (`normalized_departure = departure / √(value_sigma² + model_sigma²)`),
#     through the keyword and the positional constructors.
#   * `build_departure_set`: joins observations to records by id and rejects
#     hand-built observations, unknown or duplicate ids, a moved cell, a wrong
#     time sampling, value or sigma, a foreign stencil, a kind mismatch,
#     non-finite simulated values and clamped records.
#   * End to end: placed point and column observations through
#     `cs_surface_flux_4dvar` -> `build_departure_set(set, result, ...)` ->
#     `write_departures` / `read_departures`, bit for bit.
#   * The v2 on-disk layout (Float64 coordinates, the ragged `sample`
#     dimension) and the reader's rejection of v1 files and violations.
# ---------------------------------------------------------------------------

using Test
using Dates: DateTime, Second, year, month, day, hour, minute, second
using NCDatasets: NCDataset, defDim, defVar, dimnames

import AtmosTransport
const AT = AtmosTransport
const O = AT.Output

include(joinpath(@__DIR__, "..", "helpers", "cs_observation_columns.jl"))

const NC_TEST = 6
const DT_TEST = 3600.0
const T0_TEST_STR = "2024-06-15 00:00:00"
const T0_TEST = DateTime(2024, 6, 15, 0, 0, 0)
const CODES = O.VERTICAL_PLACEMENT_CODES

_mesh() = AT.CubedSphereMesh(; Nc = NC_TEST, FT = Float64)

_dc_at(offset::Real) = (t = T0_TEST + Second(round(Int, offset));
                        (year(t), month(t), day(t), hour(t), minute(t), second(t)))

# The stencil `_observe_cs_forward` gives a column-mean observation.
_column_stencil(o) = AT.CSObservationStencil(o.objective.panel, o.objective.i, o.objective.j,
                                             o.sampling.steps, o.sampling.weights,
                                             zeros(Int, length(o.sampling.steps)),
                                             AT.COLUMN_MEAN_PLACEMENT_CODE, NaN, NaN)

# Column records at the centres of `targets` (id, panel, i, j, observed, sigma,
# hour_offset[, window, model_sigma]), bound to the 24-step run, with
# simulated values `observed + sim_offset` and their stencils.
function _aligned_inputs(targets; tracer = "CO2", instrument = "TCCON", sim_offset = 0.5)
    mesh = _mesh()
    records = map(targets) do t
        lons, lats = AT.panel_cell_center_lonlat(mesh, t.panel)
        AT.CSObservationRecord(id = t.id, date_components = _dc_at(t.hour_offset * DT_TEST + 1800),
                               lat = lats[t.i, t.j], lon = lons[t.i, t.j], kind = "column",
                               time_window_s = get(t, :window, 0.0), model_sigma = get(t, :model_sigma, 0.0),
                               value = t.observed, value_sigma = t.sigma,
                               instrument_type = instrument, tracer = tracer)
    end
    set = AT.CSObservationSet(records; time_origin = T0_TEST_STR)
    obs = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 24)
    sims = [t.observed + sim_offset for t in targets]
    return (; set, observations = obs, simulated = sims, stencils = _column_stencil.(obs), mesh, sim_offset)
end

_build(inputs; observations = inputs.observations, simulated = inputs.simulated,
       stencils = inputs.stencils, nsteps = 24, kwargs...) =
    AT.build_departure_set(inputs.set, observations, simulated, stencils, inputs.mesh,
                           T0_TEST_STR, DT_TEST, nsteps; kwargs...)

_errmsg(f) = try f(); "" catch e e isa ArgumentError ? e.msg : rethrow() end

const THREE = [(id = 1, panel = 1, i = 2, j = 3, observed = 410.0, sigma = 0.5, hour_offset = 0),
               (id = 2, panel = 4, i = 5, j = 1, observed = 411.0, sigma = 0.7, hour_offset = 2,
                window = 3DT_TEST, model_sigma = 0.3),
               (id = 3, panel = 6, i = 1, j = 6, observed = 415.0, sigma = 1.0, hour_offset = 5)]

# ---------------------------------------------------------------------------
# CSDepartureRecord
# ---------------------------------------------------------------------------

_dep(; kwargs...) = AT.CSDepartureRecord(; id = 1, tracer = "T", instrument_type = "I",
    date_components = (2024, 1, 1, 0, 0, 0), lat = 0.0, lon = 0.0, kind = "point", panel = 1, i = 1, j = 1,
    vertical_placement = CODES.record_height, placement_height_agl_m = 10.0,
    steps = [1], weights = [1.0], levels = [5], observed_value = 1.0, simulated_value = 1.0,
    departure = 0.0, value_sigma = 1.0, normalized_departure = 0.0, kwargs...)

@testset "CSDepartureRecord keyword constructor" begin
    r = _dep(id = 7, lat = 34.123456789012, lon = -118.0, panel = 3, i = 4, j = 6, steps = [5, 6],
             weights = [0.25, 0.75], levels = [9, 10], observed_value = 420.0, simulated_value = 420.5,
             departure = 0.5, normalized_departure = 0.5)
    @test r.id === Int64(7)
    @test r.lat === 34.123456789012 && r.lon === -118.0
    @test r.kind === AT.PointObservationKind() && r.time_window_s === 0.0
    @test r.panel === Int8(3) && r.i === Int32(4) && r.j === Int32(6)
    @test r.vertical_placement === CODES.record_height
    @test r.placement_height_agl_m === 10.0 && isnan(r.placement_pressure_pa)
    @test r.steps == Int64[5, 6] && r.weights == [0.25, 0.75] && r.levels == Int32[9, 10]
    @test r.model_sigma === 0.0
    column = _dep(kind = "column", vertical_placement = AT.COLUMN_MEAN_PLACEMENT_CODE,
                  placement_height_agl_m = NaN, levels = [0])
    @test column.kind === AT.ColumnObservationKind()

    for bad in ((simulated_value = NaN,), (simulated_value = Inf,), (departure = NaN,),
                (normalized_departure = Inf,), (observed_value = NaN,), (value_sigma = -1.0,),
                (value_sigma = Inf,), (model_sigma = -1.0,), (lat = NaN,), (lat = 91.0,), (lon = Inf,),
                (panel = 0,), (panel = 7,), (i = 0,), (j = -1,), (time_window_s = -1.0,),
                (steps = Int[], weights = Float64[], levels = Int[]), (steps = [0],), (steps = [2, 1],
                weights = [0.5, 0.5], levels = [1, 1]), (weights = [0.9],), (levels = [0],),
                (levels = [1, 2],), (vertical_placement = AT.COLUMN_MEAN_PLACEMENT_CODE,),
                (vertical_placement = CODES.lowest_layer,), (vertical_placement = CODES.pressure,),
                (placement_height_agl_m = NaN,), (placement_height_agl_m = Inf,))
        @test_throws ArgumentError _dep(; bad...)
    end
    # Column records: column-mean code and levels 0.
    @test_throws ArgumentError _dep(kind = "column", placement_height_agl_m = NaN, levels = [0])
    @test_throws ArgumentError _dep(kind = "column", vertical_placement = AT.COLUMN_MEAN_PLACEMENT_CODE,
                                    placement_height_agl_m = NaN, levels = [3])
    @test _dep(vertical_placement = CODES.pressure, placement_height_agl_m = NaN,
               placement_pressure_pa = 50000.0).placement_pressure_pa == 50000.0
end

@testset "CSDepartureRecord — sign convention with the model error" begin
    ctor(; departure, normalized_departure, model_sigma = 0.0) =
        _dep(observed_value = 10.0, simulated_value = 11.0, value_sigma = 2.0; departure,
             normalized_departure, model_sigma)
    @test ctor(departure = 1.0, normalized_departure = 0.5) isa AT.CSDepartureRecord
    @test_throws ArgumentError ctor(departure = -1.0, normalized_departure = -0.5)
    @test_throws ArgumentError ctor(departure = 1.0, normalized_departure = 0.3)
    @test_throws ArgumentError ctor(departure = 2.0, normalized_departure = 1.0)
    @test ctor(departure = 1.0 + 1e-13, normalized_departure = 0.5 + 5e-14) isa AT.CSDepartureRecord
    # σ = √(2² + 1.5²) = 2.5
    @test ctor(departure = 1.0, normalized_departure = 0.4, model_sigma = 1.5) isa AT.CSDepartureRecord
    @test_throws ArgumentError ctor(departure = 1.0, normalized_departure = 0.5, model_sigma = 1.5)
end

@testset "CSDepartureRecord — positional constructor goes through validation" begin
    args(; departure = 1.0, simulated = 11.0, panel = Int8(1)) =
        (Int64(1), "CO2", "TCCON", (Int16(2024), Int16(1), Int16(1), Int16(0), Int16(0), Int16(0)),
         0.0, 0.0, AT.PointObservationKind(), 0.0, panel, Int32(1), Int32(1), CODES.record_height, 10.0, NaN,
         Int64[1], [1.0], Int32[3], 10.0, simulated, departure, 2.0, 0.0, departure / 2)
    @test AT.CSDepartureRecord(args()...) isa AT.CSDepartureRecord
    @test_throws ArgumentError AT.CSDepartureRecord(args(departure = -1.0)...)
    @test_throws ArgumentError AT.CSDepartureRecord(args(simulated = NaN, departure = NaN)...)
    @test_throws ArgumentError AT.CSDepartureRecord(args(panel = Int8(7))...)
end

# ---------------------------------------------------------------------------
# build_departure_set
# ---------------------------------------------------------------------------

@testset "build_departure_set — join by id, sign convention, metadata" begin
    inputs = _aligned_inputs(THREE)
    @test inputs.observations[2].sampling.steps == [2, 3, 4]       # 02:30 ± 1.5 h
    dep = _build(inputs)
    @test dep isa AT.CSDepartureSet && length(dep) == 3
    @test dep.mesh_Nc == NC_TEST
    @test dep.mesh_panel_convention == "GnomonicPanelConvention"
    @test dep.mesh_cs_definition_tag == "equiangular_gnomonic"
    @test dep.t_start == T0_TEST_STR && dep.dt_seconds == DT_TEST && dep.nsteps == 24
    @test dep.run_id === nothing && dep.iteration === nothing
    for (k, r) in enumerate(dep.records)
        rec = inputs.set.records[k]
        @test r.id == rec.id && r.lat === rec.lat && r.lon === rec.lon
        @test r.observed_value == rec.value && r.simulated_value == inputs.simulated[k]
        @test r.departure ≈ inputs.sim_offset atol = 1e-12
        @test r.normalized_departure ≈ inputs.sim_offset / hypot(rec.value_sigma, rec.model_sigma) atol = 1e-12
        @test r.vertical_placement == AT.COLUMN_MEAN_PLACEMENT_CODE && all(iszero, r.levels)
        @test r.steps == inputs.observations[k].sampling.steps
    end
    # Rows follow the observations; the record order of the set does not matter.
    order = [3, 1, 2]
    reordered = _build(inputs; observations = inputs.observations[order], simulated = inputs.simulated[order],
                       stencils = inputs.stencils[order])
    @test [r.id for r in reordered] == [3, 1, 2]
    shuffled_set = AT.CSObservationSet(inputs.set.records[[2, 3, 1]]; time_origin = T0_TEST_STR)
    @test [r.id for r in AT.build_departure_set(shuffled_set, inputs.observations, inputs.simulated,
                                                inputs.stencils, inputs.mesh, T0_TEST_STR, DT_TEST, 24)] == [1, 2, 3]
    # Skipped records need no pre-filtering: observations name their records.
    @test length(_build(inputs; observations = inputs.observations[2:3], simulated = inputs.simulated[2:3],
                        stencils = inputs.stencils[2:3])) == 2
end

@testset "build_departure_set — consistency gatekeeper" begin
    inputs = _aligned_inputs(THREE)
    obs, sims, st = inputs.observations, inputs.simulated, inputs.stencils
    replace_obs(k, o) = (v = copy(obs); v[k] = o; v)
    # Lengths, non-finite simulated values, t_start, nsteps.
    @test occursin("length mismatch", _errmsg(() -> _build(inputs; observations = obs[1:2])))
    @test occursin("length mismatch", _errmsg(() -> _build(inputs; stencils = st[1:2])))
    for bad in (NaN, Inf)
        @test occursin("not finite", _errmsg(() -> _build(inputs; simulated = [sims[1], bad, sims[3]])))
    end
    @test_throws ArgumentError AT.build_departure_set(inputs.set, obs, sims, st, inputs.mesh, "not a date",
                                                      DT_TEST, 24)
    @test_throws ArgumentError _build(inputs; nsteps = 0)
    # Hand-built observations (id 0) and unknown ids have no record.
    o = obs[1]
    hand = AT.CSObservation(o.sampling, o.objective, o.value, o.sigma)
    @test occursin("not a record", _errmsg(() -> _build(inputs; observations = replace_obs(1, hand))))
    foreign = AT.CSObservation(o.sampling, o.objective, o.value, o.sigma; id = 999)
    @test occursin("not a record", _errmsg(() -> _build(inputs; observations = replace_obs(1, foreign))))
    # Duplicate ids in the set.
    dup = AT.CSObservationSet([inputs.set.records; inputs.set.records[1]]; time_origin = T0_TEST_STR)
    @test occursin("more than once", _errmsg(() -> AT.build_departure_set(dup, obs, sims, st, inputs.mesh,
                                                                          T0_TEST_STR, DT_TEST, 24)))
    # A moved cell, another time sampling, value or sigma.
    p, i, j = o.objective.panel, o.objective.i, o.objective.j
    moved = AT.CSObservation(o.sampling, AT.CSColumnMeanObjective(p, mod1(i + 1, NC_TEST), j), o.value, o.sigma;
                             id = o.id)
    @test occursin("is in cell", _errmsg(() -> _build(inputs; observations = replace_obs(1, moved))))
    later = AT.CSObservation(o.step + 1, o.objective, o.value, o.sigma; id = o.id)
    @test occursin("maps to", _errmsg(() -> _build(inputs; observations = replace_obs(1, later),
                                                    stencils = [_column_stencil(later); st[2:3]])))
    for (value, sigma, what) in ((999.0, o.sigma, "value"), (o.value, 99.0, "sigma"))
        changed = AT.CSObservation(o.sampling, o.objective, value, sigma; id = o.id)
        @test occursin(what, _errmsg(() -> _build(inputs; observations = replace_obs(1, changed))))
    end
    # The window record's σ includes its model error.
    o2 = obs[2]
    @test o2.sigma ≈ hypot(0.7, 0.3)
    value_only = AT.CSObservation(o2.sampling, o2.objective, o2.value, 0.7; id = o2.id)
    @test occursin("sigma", _errmsg(() -> _build(inputs; observations = replace_obs(2, value_only))))
    # A stencil of another observation, and a kind mismatch.
    @test occursin("stencil", _errmsg(() -> _build(inputs; stencils = st[[2, 1, 3]])))
    placed = AT.CSObservation(o.sampling, AT.CSPlacedPointObjective(p, i, j, 1.0e12,
                              O.VerticalPlacement(CODES.record_height, 10.0, NaN, NaN), O.TotalFromHumidity()),
                              o.value, o.sigma; id = o.id)
    @test occursin("column record bound to", _errmsg(() -> _build(inputs; observations = replace_obs(1, placed))))
    # A record moved into the run by ClampOutOfRange.
    late = AT.CSObservationRecord(id = 50, date_components = _dc_at(30DT_TEST), lat = inputs.set.records[1].lat,
                                  lon = inputs.set.records[1].lon, kind = "column", value = 1.0, value_sigma = 1.0,
                                  instrument_type = "I", tracer = "CO2")
    late_set = AT.CSObservationSet([late]; time_origin = T0_TEST_STR)
    clamped = AT.bind_to_mesh(late_set, inputs.mesh, T0_TEST, DT_TEST; nsteps = 24, out_of_range_policy = "clamp")
    @test occursin("ClampOutOfRange", _errmsg(() -> AT.build_departure_set(late_set, clamped, [1.0],
        _column_stencil.(clamped), inputs.mesh, T0_TEST_STR, DT_TEST, 24)))
end

@testset "CSDepartureSet — per-record bounds vs run metadata" begin
    _set(rec; mesh_Nc = NC_TEST, nsteps = 24) = AT.CSDepartureSet([rec]; mesh_Nc, nsteps,
        mesh_panel_convention = "GnomonicPanelConvention", mesh_cs_definition_tag = "equiangular_gnomonic",
        t_start = T0_TEST_STR, dt_seconds = DT_TEST)
    @test _set(_dep()) isa AT.CSDepartureSet
    @test _set(_dep(steps = [23, 24], weights = [0.5, 0.5], levels = [1, 1], i = NC_TEST, j = NC_TEST)) isa
          AT.CSDepartureSet
    @test_throws ArgumentError _set(_dep(steps = [24, 25], weights = [0.5, 0.5], levels = [1, 1]))
    @test_throws ArgumentError _set(_dep(i = 999))
    @test_throws ArgumentError _set(_dep(j = NC_TEST + 1))
end

# ---------------------------------------------------------------------------
# End to end: placed point and column observations through the 4D-Var
# ---------------------------------------------------------------------------

@testset "4D-Var result -> departures -> NetCDF, bit for bit" begin
    P = hydrostatic_cs_problem(; nsteps = 3)
    t0, dt = T0_TEST, 600.0
    lons, lats = AT.panel_cell_center_lonlat(P.mesh, 2)
    at(s) = (t = t0 + Second(round(Int, s)); (year(t), month(t), day(t), hour(t), minute(t), second(t)))
    rec(id, kind; kwargs...) = AT.CSObservationRecord(; id, kind, date_components = at(900), lat = lats[2, 3],
        lon = lons[2, 3], value = 405e-6, value_sigma = 1e-6, instrument_type = "X", tracer = "CO2", kwargs...)
    records = [rec(42, "point"; intake_height_m = 10.0), rec(7, "point"; altitude_m = 5000.0, elevation_m = 0.0),
               rec(9, "point"; pressure_pa = 50000.0, time_window_s = 1200.0, model_sigma = 0.5e-6),
               rec(3, "column"; time_window_s = 1500.0)]
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")
    obs = AT.bind_to_mesh(set, P.mesh, t0, dt; nsteps = P.nsteps)
    control = AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:all, 1:P.nsteps), ntuple(_ -> fill(1e3, P.Nc, P.Nc), 6))
    result = AT.cs_surface_flux_4dvar(P.rm, P.m, P.am, P.bm, P.cm, P.mesh, obs, [control];
                                      scheme = AT.UpwindScheme(), dt, column_met = column_met(P))
    dep = AT.build_departure_set(set, result, P.mesh, "2024-06-15 00:00:00", dt, P.nsteps;
                                 run_id = "run-A", iteration = 3)
    @test [r.id for r in dep] == [42, 7, 9, 3]
    @test [r.vertical_placement for r in dep] ==
          [CODES.record_height, CODES.record_height, CODES.pressure, AT.COLUMN_MEAN_PLACEMENT_CODE]
    @test [r.levels for r in dep] == [s.levels for s in result.stencils]
    @test dep[1].levels != dep[2].levels                          # surface intake vs 5 km aircraft
    @test dep[3].steps == [1, 2, 3] && sum(dep[3].weights) ≈ 1
    @test dep[3].normalized_departure ≈ result.residuals[3] / hypot(1e-6, 0.5e-6) rtol = 1e-12
    @test [r.simulated_value for r in dep] == result.simulated
    mktempdir() do dir
        path = joinpath(dir, "dep_v2.nc")
        AT.write_departures(path, dep)
        back = AT.read_departures(path)
        @test (back.run_id, back.iteration, back.nsteps, back.dt_seconds) == ("run-A", 3, P.nsteps, dt)
        for (a, b) in zip(dep, back), field in fieldnames(AT.CSDepartureRecord)
            x, y = getfield(a, field), getfield(b, field)
            @test x == y || (x isa Float64 && isnan(x) && isnan(y))
        end
    end
end

# ---------------------------------------------------------------------------
# Round trips, layout and reader validation
# ---------------------------------------------------------------------------

@testset "write_departures / read_departures: rows, ids and optional attributes" begin
    inputs = _aligned_inputs([(id = 99, panel = 1, i = 1, j = 1, observed = 410.0, sigma = 0.5, hour_offset = 0),
                              (id = 1, panel = 3, i = 2, j = 4, observed = 411.0, sigma = 0.6, hour_offset = 1,
                               window = 7200.0),
                              (id = 42, panel = 5, i = 3, j = 2, observed = 412.0, sigma = 0.7, hour_offset = 2),
                              (id = 7, panel = 6, i = 6, j = 6, observed = 413.0, sigma = 0.8, hour_offset = 3)])
    mktempdir() do dir
        with_attrs = joinpath(dir, "ordered.nc")
        AT.write_departures(with_attrs, _build(inputs; run_id = "run-B", iteration = 7))
        back = AT.read_departures(with_attrs)
        @test [r.id for r in back] == [99, 1, 42, 7]
        @test back[2].steps == [1, 2, 3] && back[2].weights == [0.25, 0.5, 0.25]
        @test back.run_id == "run-B" && back.iteration == 7
        plain = joinpath(dir, "plain.nc")
        AT.write_departures(plain, _build(inputs))
        NCDataset(plain, "r") do ds
            @test !haskey(ds.attrib, "run_id") && !haskey(ds.attrib, "iteration")
        end
        @test AT.read_departures(plain).run_id === nothing
    end
end

@testset "v2 NetCDF on-disk layout matches the schema" begin
    inputs = _aligned_inputs(THREE)
    dep = _build(inputs; run_id = "run-B", iteration = 7)
    mktempdir() do dir
        path = joinpath(dir, "layout.nc")
        AT.write_departures(path, dep)
        NCDataset(path, "r") do ds
            @test ds.attrib["cs_departures_schema"] == "v2"
            @test ds.attrib["mesh_Nc"] == NC_TEST
            @test ds.attrib["departure_sign_convention"] == "simulated_minus_observed"
            @test occursin("-1 = column_mean", ds.attrib["vertical_placement_codes"])
            @test occursin("2 = pressure", ds.attrib["vertical_placement_codes"])
            @test ds.dim["obs"] == 3 && ds.dim["date_component"] == 6 && ds.dim["sample"] == 5
            types = Dict("id" => Int64, "date_components" => Int16, "lat" => Float64, "lon" => Float64,
                         "time_window_s" => Float64, "panel" => Int8, "i" => Int32, "j" => Int32,
                         "vertical_placement" => Int8, "placement_height_agl_m" => Float64,
                         "placement_pressure_pa" => Float64, "sample_count" => Int32, "sample_step" => Int64,
                         "sample_weight" => Float64, "sample_level" => Int32, "observed_value" => Float64,
                         "simulated_value" => Float64, "departure" => Float64, "value_sigma" => Float64,
                         "model_sigma" => Float64, "normalized_departure" => Float64)
            for (name, T) in types
                @test eltype(ds[name].var) === T
            end
            @test ds["kind"][:] == ["column", "column", "column"]
            @test ds["sample_count"][:] == [1, 3, 1]
            @test ds["sample_step"][:] == [1, 2, 3, 4, 6]
        end
    end
end

@testset "read_departures rejects v1 files and schema violations" begin
    inputs = _aligned_inputs(THREE)
    mktempdir() do dir
        good = joinpath(dir, "good.nc")
        AT.write_departures(good, _build(inputs))
        @test AT.read_departures(good) isa AT.CSDepartureSet
        mutated(name, f!) = (path = joinpath(dir, "$(name).nc"); cp(good, path); NCDataset(f!, path, "a"); path)
        @test occursin("schema v1", _errmsg(() -> AT.read_departures(
            mutated("v1", ds -> ds.attrib["cs_departures_schema"] = "v1"))))
        @test_throws ArgumentError AT.read_departures(mutated("v3", ds -> ds.attrib["cs_departures_schema"] = "v3"))
        @test_throws ArgumentError AT.read_departures(mutated("no_schema", ds -> delete!(ds.attrib, "cs_departures_schema")))
        @test_throws ArgumentError AT.read_departures(
            mutated("sign", ds -> ds.attrib["departure_sign_convention"] = "observed_minus_simulated"))
        for attr in ("mesh_Nc", "mesh_panel_convention", "mesh_cs_definition_tag", "t_start", "dt_seconds",
                     "nsteps", "departure_sign_convention")
            @test_throws ArgumentError AT.read_departures(mutated("no_$(attr)", ds -> delete!(ds.attrib, attr)))
        end
        @test_throws ArgumentError AT.read_departures(joinpath(dir, "missing.nc"))
        # Sample counts that do not tile the sample dimension, or a row breaking an invariant.
        @test occursin("inconsistent sample arrays", _errmsg(() -> AT.read_departures(
            mutated("counts", ds -> ds["sample_count"][:] = Int32[1, 2, 1]))))
        @test_throws ArgumentError AT.read_departures(mutated("weights", ds -> ds["sample_weight"][2] = 0.9))
        @test_throws ArgumentError AT.read_departures(mutated("departure", ds -> ds["departure"][1] = -5.0))
        # Missing variables: rewrite the file without one.
        function without(var)
            path = joinpath(dir, "no_var_$(var).nc")
            NCDataset(good, "r") do src
                NCDataset(path, "c") do ds
                    for (d, n) in src.dim
                        defDim(ds, d, n)
                    end
                    for (k, v) in src.attrib
                        ds.attrib[k] = v
                    end
                    for name in keys(src)
                        name == var && continue
                        data = Array(src[name])
                        v = defVar(ds, name, eltype(src[name].var), dimnames(src[name]))
                        v[ntuple(_ -> Colon(), ndims(data))...] = data
                    end
                end
            end
            return path
        end
        for var in ("lat", "kind", "vertical_placement", "sample_count", "sample_step", "sample_weight",
                    "sample_level", "model_sigma", "normalized_departure")
            @test occursin("missing required variable", _errmsg(() -> AT.read_departures(without(var))))
        end
    end
end
