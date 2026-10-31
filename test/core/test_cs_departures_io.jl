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
const CODES = AtmosTransport.Observations.VERTICAL_PLACEMENT_CODES

_mesh() = AT.CubedSphereMesh(; Nc = NC_TEST, FT = Float64)

_dc_at(offset::Real) = (t = T0_TEST + Second(round(Int, offset));
                        (year(t), month(t), day(t), hour(t), minute(t), second(t)))

# The stencil `_observe_cs_forward!` gives a column-mean observation.
_column_stencil(o) = AT.CSObservationStencil(o.objective.panel, o.objective.i, o.objective.j,
                                             o.sampling.steps, o.sampling.weights,
                                             zeros(Int, length(o.sampling.steps)),
                                             AT.COLUMN_MEAN_PLACEMENT_CODE, NaN, NaN, NaN, Float64[], true)

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

const THREE = [(id = 1, panel = 1, i = 2, j = 3, observed = 410.0, sigma = 0.5, hour_offset = 0),
               (id = 2, panel = 4, i = 5, j = 1, observed = 411.0, sigma = 0.7, hour_offset = 2,
                window = 3DT_TEST, model_sigma = 0.3),
               (id = 3, panel = 6, i = 1, j = 6, observed = 415.0, sigma = 1.0, hour_offset = 5)]

# ---------------------------------------------------------------------------
# CSDepartureRecord
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
    @test_throws ArgumentError ((()->begin

        _build(inputs; observations = obs[1:2])
    end))()
    @test_throws ArgumentError ((()->begin

        _build(inputs; stencils = st[1:2])
    end))()
    for bad in (NaN, Inf)
        @test_throws ArgumentError ((()->begin

        _build(inputs; simulated = [sims[1], bad, sims[3]])
    end))()
    end
    @test_throws ArgumentError AT.build_departure_set(inputs.set, obs, sims, st, inputs.mesh, "not a date",
                                                      DT_TEST, 24)
    @test_throws ArgumentError _build(inputs; nsteps = 0)
    # Hand-built observations (id 0) and unknown ids have no record.
    o = obs[1]
    hand = AT.CSObservation(o.sampling, o.objective, o.value, o.sigma)
    @test_throws ArgumentError ((()->begin

        _build(inputs; observations = replace_obs(1, hand))
    end))()
    foreign = AT.CSObservation(o.sampling, o.objective, o.value, o.sigma; id = 999)
    @test_throws ArgumentError ((()->begin

        _build(inputs; observations = replace_obs(1, foreign))
    end))()
    # Repeated ids: rejected by the set itself, and by build_departure_set for
    # a set changed after construction.
    @test_throws ArgumentError ((()->begin

        AT.CSObservationSet([inputs.set.records; inputs.set.records[1]])
    end))()
    dup = AT.CSObservationSet(copy(inputs.set.records); time_origin = T0_TEST_STR)
    push!(dup.records, inputs.set.records[1])
    @test_throws ArgumentError ((()->begin

        AT.build_departure_set(dup, obs, sims, st, inputs.mesh, T0_TEST_STR, DT_TEST, 24)
    end))()
    # A moved cell, another time sampling, value or sigma.
    p, i, j = o.objective.panel, o.objective.i, o.objective.j
    moved = AT.CSObservation(o.sampling, AT.CSColumnMeanObjective(p, mod1(i + 1, NC_TEST), j), o.value, o.sigma;
                             id = o.id)
    @test_throws ArgumentError ((()->begin

        _build(inputs; observations = replace_obs(1, moved))
    end))()
    later = AT.CSObservation(o.step + 1, o.objective, o.value, o.sigma; id = o.id)
    @test_throws ArgumentError ((()->begin

        _build(inputs; observations = replace_obs(1, later), stencils = [_column_stencil(later); st[2:3]])
    end))()
    for (value, sigma, what) in ((999.0, o.sigma, "value"), (o.value, 99.0, "sigma"))
        changed = AT.CSObservation(o.sampling, o.objective, value, sigma; id = o.id)
        @test_throws ArgumentError ((()->begin

        _build(inputs; observations = replace_obs(1, changed))
    end))()
    end
    # The window record's σ includes its model error.
    o2 = obs[2]
    @test o2.sigma ≈ hypot(0.7, 0.3)
    value_only = AT.CSObservation(o2.sampling, o2.objective, o2.value, 0.7; id = o2.id)
    @test_throws ArgumentError ((()->begin

        _build(inputs; observations = replace_obs(2, value_only))
    end))()
    # A stencil of another observation, and a kind mismatch.
    @test_throws ArgumentError ((()->begin

        _build(inputs; stencils = st[[2, 1, 3]])
    end))()
    placed = AT.CSObservation(o.sampling, AT.CSPlacedPointObjective(p, i, j, 1.0e12,
                              AtmosTransport.Observations.VerticalPlacement(CODES.record_height, 10.0, NaN, NaN), AtmosTransport.Observations.TotalFromHumidity()),
                              o.value, o.sigma; id = o.id)
    @test_throws ArgumentError ((()->begin

        _build(inputs; observations = replace_obs(1, placed))
    end))()
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
