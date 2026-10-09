#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.D2 — `bind_to_mesh` bridge from on-disk `CSObservationSet`
# to `Vector{CSObservation{CSColumnMeanObjective, Float64}}`.
#
# Coverage:
#   * Time mapping. Half-open interval `[t_start + (k-1)*dt, t_start + k*dt)`:
#     obs at `t_start` -> step 1, at `t_start + dt` -> step 2, etc.
#   * Geographic mapping. (lat, lon) of a cell center round-trips to
#     its own (panel, i, j); perturbations within the cell stay there.
#   * Argument validation. dt > 0, nsteps >= 1, valid policy symbol.
#   * Tracer filter keeps only the matching tracer.
#   * Out-of-range policy. `:reject` throws, `:skip` drops, `:clamp`
#     snaps to the nearest valid step.
#   * Bit-exact 4D-Var equivalence. A hand-built `Vector{CSObservation}`
#     and a round-tripped `CSObservationSet -> bind_to_mesh` produce
#     identical (step, panel, i, j, value, sigma) tuples.
# ---------------------------------------------------------------------------

using Test
using Dates: DateTime, Second

import AtmosTransport
const AT = AtmosTransport

const FT_TEST = Float64
const NC_TEST = 6
const DT_TEST = 3600.0                 # 1 hour per step
const T0_TEST = DateTime(2024, 6, 15, 0, 0, 0)

_mesh() = AT.CubedSphereMesh(; Nc = NC_TEST, FT = FT_TEST)

function _record(; id, dc, lat, lon, alt = 0.0f0,
                  value = 420.0, value_sigma = 0.5,
                  instrument = "TCCON", tracer = "CO2")
    return AT.CSObservationRecord(id = id,
                                  date_components = dc,
                                  lat = Float32(lat), lon = Float32(lon),
                                  alt = Float32(alt),
                                  value = Float64(value),
                                  value_sigma = Float64(value_sigma),
                                  instrument_type = instrument,
                                  tracer = tracer)
end

function _dc_at(offset::Real)
    t = T0_TEST + Second(round(Int, offset))
    return (Int16(year(t)), Int16(month(t)), Int16(day(t)),
            Int16(hour(t)), Int16(minute(t)), Int16(second(t)))
end

# Dates exports we actually need; keep them local to avoid polluting Main.
using Dates: year, month, day, hour, minute, second

# ---------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------

@testset "bind_to_mesh — time mapping (left-closed half-open)" begin
    mesh = _mesh()
    # Pick a single cell-center lat/lon for unambiguous geographic mapping.
    lons, lats = AT.panel_cell_center_lonlat(mesh, 1)
    lat0, lon0 = lats[3, 4], lons[3, 4]

    # Boundaries: 0 -> step 1, dt -> step 2, 2dt -> step 3, etc.
    boundary_records = [
        _record(id = 1, dc = _dc_at(0),          lat = lat0, lon = lon0),
        _record(id = 2, dc = _dc_at(DT_TEST),    lat = lat0, lon = lon0),
        _record(id = 3, dc = _dc_at(2*DT_TEST),  lat = lat0, lon = lon0),
        _record(id = 4, dc = _dc_at(3*DT_TEST - 1), lat = lat0, lon = lon0),
        _record(id = 5, dc = _dc_at(DT_TEST/2),  lat = lat0, lon = lon0),
    ]
    set = AT.CSObservationSet(boundary_records;
                              time_origin = "2024-06-15 00:00:00")

    obs = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 24)
    @test [o.step for o in obs] == [1, 2, 3, 3, 1]
end

@testset "bind_to_mesh — out-of-range policies" begin
    mesh = _mesh()
    lons, lats = AT.panel_cell_center_lonlat(mesh, 1)
    lat0, lon0 = lats[2, 2], lons[2, 2]

    # Records: before t_start, at t_start, past nsteps*dt.
    records = [
        _record(id = 10, dc = _dc_at(-DT_TEST/2),     lat = lat0, lon = lon0),
        _record(id = 11, dc = _dc_at(0),              lat = lat0, lon = lon0),
        _record(id = 12, dc = _dc_at(5*DT_TEST + 1),  lat = lat0, lon = lon0),
    ]
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")

    # :reject default
    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                                               nsteps = 5)

    # :skip drops the two out-of-range records, keeps the middle one.
    obs_skip = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                               nsteps = 5,
                               out_of_range_policy = :skip)
    @test length(obs_skip) == 1
    @test obs_skip[1].step == 1
    @test obs_skip[1].objective isa AT.CSColumnMeanObjective

    # :clamp snaps to [1, nsteps].
    obs_clamp = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                                nsteps = 5,
                                out_of_range_policy = :clamp)
    @test [o.step for o in obs_clamp] == [1, 1, 5]
end

# Explicit terminal-boundary case: an observation at exactly
# `t_start + nsteps*dt` maps to step `nsteps + 1` and must be treated as
# out-of-range under each policy. The previous policy testset only
# exercised `5*DT + 1` (off by 1 second past the boundary); this pins
# the bit-exact boundary the ship note claims.
@testset "bind_to_mesh — terminal boundary t_start + nsteps*dt" begin
    mesh = _mesh()
    lons, lats = AT.panel_cell_center_lonlat(mesh, 1)
    lat0, lon0 = lats[2, 2], lons[2, 2]
    nsteps = 5
    terminal = nsteps * DT_TEST       # exactly one step past the last valid

    records = [ _record(id = 21, dc = _dc_at(terminal),
                        lat = lat0, lon = lon0) ]
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")

    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                                               nsteps = nsteps)

    obs_skip = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                               nsteps = nsteps,
                               out_of_range_policy = :skip)
    @test isempty(obs_skip)

    obs_clamp = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                                nsteps = nsteps,
                                out_of_range_policy = :clamp)
    @test length(obs_clamp) == 1
    @test obs_clamp[1].step == nsteps
end

@testset "bind_to_mesh — argument validation" begin
    mesh = _mesh()
    set = AT.CSObservationSet([_record(id = 1, dc = _dc_at(0),
                                       lat = 0.0, lon = 0.0)];
                              time_origin = "2024-06-15 00:00:00")

    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, -1.0)
    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, 0.0)
    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                                               nsteps = 0)
    @test_throws ArgumentError AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                                               out_of_range_policy = :bogus)
end

# ---------------------------------------------------------------------------
# Fail-fast on non-finite / out-of-range coordinates and payloads.
#
# `CSObservationRecord`'s keyword constructor (and therefore every
# record loaded from disk) is already finite-checked. These tests
# bypass the keyword validator by going through the positional inner
# constructor, simulating a record built by hand without going through
# the public surface. `bind_to_mesh` must still refuse to bind them.
# ---------------------------------------------------------------------------

_dc_const() = (Int16(2024), Int16(6), Int16(15),
                Int16(0),    Int16(0), Int16(0))

_bad_record(; lat = 0.0, lon = 0.0, alt = 0.0,
            value = 420.0, sigma = 0.5) =
    AT.CSObservationRecord(Int64(1), _dc_const(),
                            Float32(lat), Float32(lon), Float32(alt),
                            Float64(value), Float64(sigma),
                            "TCCON", "CO2")

@testset "bind_to_mesh — rejects non-finite / out-of-range fields" begin
    mesh = _mesh()
    function _bind(record)
        set = AT.CSObservationSet([record];
                                  time_origin = "2024-06-15 00:00:00")
        return AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 24)
    end

    @test_throws ArgumentError _bind(_bad_record(lat = NaN))
    @test_throws ArgumentError _bind(_bad_record(lat = Inf))
    @test_throws ArgumentError _bind(_bad_record(lat = 90.5))     # > 90
    @test_throws ArgumentError _bind(_bad_record(lat = -91.0))    # < -90
    @test_throws ArgumentError _bind(_bad_record(lon = NaN))
    @test_throws ArgumentError _bind(_bad_record(lon = Inf))
    @test_throws ArgumentError _bind(_bad_record(value = NaN))
    @test_throws ArgumentError _bind(_bad_record(value = Inf))
    @test_throws ArgumentError _bind(_bad_record(sigma = NaN))
    @test_throws ArgumentError _bind(_bad_record(sigma = Inf))    # not finite

    # Sanity: a record with all-finite fields still binds successfully.
    obs = _bind(_bad_record())
    @test length(obs) == 1
end

# ---------------------------------------------------------------------------
# Tracer filter
# ---------------------------------------------------------------------------

@testset "bind_to_mesh — tracer_filter" begin
    mesh = _mesh()
    lons, lats = AT.panel_cell_center_lonlat(mesh, 1)
    lat0, lon0 = lats[1, 1], lons[1, 1]

    records = [
        _record(id = 1, dc = _dc_at(0),       lat = lat0, lon = lon0,
                tracer = "CO2", value = 420.0),
        _record(id = 2, dc = _dc_at(DT_TEST), lat = lat0, lon = lon0,
                tracer = "CH4", value = 1900.0),
        _record(id = 3, dc = _dc_at(2*DT_TEST), lat = lat0, lon = lon0,
                tracer = "CO2", value = 421.0),
    ]
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")

    co2 = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                          nsteps = 24, tracer_filter = "CO2")
    @test length(co2) == 2
    @test [o.value for o in co2] == [420.0, 421.0]

    ch4 = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST;
                          nsteps = 24, tracer_filter = "CH4")
    @test length(ch4) == 1
    @test ch4[1].value == 1900.0
end

# ---------------------------------------------------------------------------
# Geographic mapping: cell-center round-trip + sub-cell stability
# ---------------------------------------------------------------------------

@testset "bind_to_mesh — geographic round-trip at cell centers" begin
    mesh = _mesh()
    # Pick a representative cell per panel; build one observation each.
    targets = [(1, 2, 3), (2, 4, 5), (3, 1, 1),
               (4, 3, 6), (5, 5, 2), (6, 6, 6)]

    records = AT.CSObservationRecord[]
    for (k, (p, i, j)) in enumerate(targets)
        lons, lats = AT.panel_cell_center_lonlat(mesh, p)
        push!(records, _record(id = k,
                               dc = _dc_at((k - 1) * DT_TEST),
                               lat = lats[i, j], lon = lons[i, j]))
    end
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")

    obs = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 24)
    @test length(obs) == length(targets)
    for (o, (p, i, j)) in zip(obs, targets)
        @test o.objective.panel == p
        @test o.objective.i == i
        @test o.objective.j == j
    end
end

@testset "bind_to_mesh — sub-cell perturbations stay in same cell" begin
    mesh = _mesh()
    p, i, j = 3, 4, 4
    lons, lats = AT.panel_cell_center_lonlat(mesh, p)
    lat0, lon0 = Float64(lats[i, j]), Float64(lons[i, j])

    # Build a "neighbour ring" of small lat/lon offsets. Cell width at
    # NC_TEST = 6 is ~15 degrees, so a 0.5-degree offset stays well inside.
    deltas = (
        ( 0.0,  0.0),
        ( 0.3,  0.0),
        (-0.3,  0.0),
        ( 0.0,  0.3),
        ( 0.0, -0.3),
    )
    records = [ _record(id = k,
                        dc = _dc_at((k - 1) * DT_TEST),
                        lat = lat0 + d[1], lon = lon0 + d[2])
                for (k, d) in enumerate(deltas) ]
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")
    obs = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 24)

    @test length(obs) == length(deltas)
    for o in obs
        @test o.objective.panel == p
        @test o.objective.i == i
        @test o.objective.j == j
    end
end

# ---------------------------------------------------------------------------
# 4D-Var equivalence: hand-built CSObservation vs bind_to_mesh
# ---------------------------------------------------------------------------

@testset "bind_to_mesh — bit-exact equivalence to hand-built CSObservation" begin
    mesh = _mesh()

    # Hand-built CSObservation vector. Each entry pins a specific
    # (step, panel, i, j, value, sigma).
    targets = [
        (step = 1, panel = 1, i = 2, j = 3, value = 410.1, sigma = 0.5),
        (step = 3, panel = 4, i = 5, j = 1, value = 411.2, sigma = 0.7),
        (step = 7, panel = 6, i = 1, j = 6, value = 415.0, sigma = 1.0),
    ]
    literal = [ AT.CSObservation(t.step,
                                 AT.CSColumnMeanObjective(t.panel, t.i, t.j),
                                 t.value, t.sigma)
                for t in targets ]

    # Build an equivalent CSObservationSet. For each target take the
    # cell-center (lat, lon) and the mid-step time `t_start + (k - 0.5)*dt`.
    records = AT.CSObservationRecord[]
    for (k, t) in enumerate(targets)
        lons, lats = AT.panel_cell_center_lonlat(mesh, t.panel)
        push!(records, _record(id = k,
                               dc = _dc_at((t.step - 0.5) * DT_TEST),
                               lat = lats[t.i, t.j], lon = lons[t.i, t.j],
                               value = t.value, value_sigma = t.sigma))
    end
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")
    bound = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 24)

    @test length(bound) == length(literal)
    for (a, b) in zip(literal, bound)
        @test a.step == b.step
        @test a.objective.panel == b.objective.panel
        @test a.objective.i == b.objective.i
        @test a.objective.j == b.objective.j
        @test a.value === b.value
        @test a.sigma === b.sigma
    end
end

# ---------------------------------------------------------------------------
# NetCDF round-trip + bind_to_mesh stays bit-exact
# ---------------------------------------------------------------------------

@testset "bind_to_mesh — survives write_observations / read_observations" begin
    mesh = _mesh()
    targets = [(1, 2, 2), (2, 3, 3), (5, 4, 5)]
    records = AT.CSObservationRecord[]
    for (k, (p, i, j)) in enumerate(targets)
        lons, lats = AT.panel_cell_center_lonlat(mesh, p)
        push!(records, _record(id = k,
                               dc = _dc_at((k - 1) * DT_TEST),
                               lat = lats[i, j], lon = lons[i, j],
                               value = 420.0 + k, value_sigma = 0.1 * k))
    end
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")

    mktempdir() do dir
        path = joinpath(dir, "obs_v1.nc")
        AT.write_observations(path, set)
        recovered = AT.read_observations(path)

        a = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 24)
        b = AT.bind_to_mesh(recovered, mesh, T0_TEST, DT_TEST; nsteps = 24)
        @test length(a) == length(b) == length(targets)
        for (oa, ob) in zip(a, b)
            @test oa.step == ob.step
            @test oa.objective.panel == ob.objective.panel
            @test oa.objective.i == ob.objective.i
            @test oa.objective.j == ob.objective.j
            @test oa.value === ob.value
            @test oa.sigma === ob.sigma
        end
    end
end

# ---------------------------------------------------------------------------
# Binding uses the containing cell, as the forward observation sampler does
# ---------------------------------------------------------------------------

using Random: MersenneTwister

@testset "bind_to_mesh — same cell as the forward sampler's locator" begin
    mesh = _mesh()
    locator = AT.Output.cell_locator(mesh)
    rng = MersenneTwister(4242)
    # Uniform points on the sphere: many lie near cell edges, corners and panel
    # seams, where the nearest cell centre is not the containing cell.
    points = [(Float32(rad2deg(asin(2rand(rng) - 1))), Float32(360rand(rng) - 180)) for _ in 1:2000]
    records = [_record(id = k, dc = _dc_at(0), lat = lat, lon = lon) for (k, (lat, lon)) in enumerate(points)]
    obs = AT.bind_to_mesh(AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00"),
                          mesh, T0_TEST, DT_TEST; nsteps = 24)
    cells = [AT.Output.locate(locator, Float64(lon), Float64(lat)) for (lat, lon) in points]
    @test [(o.objective.panel, o.objective.i, o.objective.j) for o in obs] ==
          [(c.panel, c.i, c.j) for c in cells]

    # Longitudes of any wrap bind to the same cell.
    lat, lon = points[1]
    wrapped(l) = only(AT.bind_to_mesh(AT.CSObservationSet([_record(id = 1, dc = _dc_at(0), lat = lat, lon = l)];
                                                          time_origin = "2024-06-15 00:00:00"),
                                      mesh, T0_TEST, DT_TEST; nsteps = 24)).objective
    @test wrapped(lon) == wrapped(lon + 720) == wrapped(lon - 1080)

    # The departure set's alignment check uses the same locator: it accepts the
    # bound observations and rejects one moved to a neighbouring cell.
    set = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")
    simulated = fill(420.0, length(obs))
    @test length(AT.build_departure_set(set, obs, simulated, mesh, "2024-06-15T00:00:00", DT_TEST, 24)) == length(obs)
    o = obs[1]
    moved = AT.CSObservation(o.step, AT.CSColumnMeanObjective(o.objective.panel, mod1(o.objective.i + 1, NC_TEST),
                                                               o.objective.j), o.value, o.sigma)
    @test_throws ArgumentError AT.build_departure_set(set, [moved; obs[2:end]], simulated, mesh,
                                                      "2024-06-15T00:00:00", DT_TEST, 24)
end
