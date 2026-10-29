#!/usr/bin/env julia
#
# Issue #20, part 3: point observations placed in the model column, and time
# windows, in the 4D-Var.
#
# A point record is bound to a `CSPlacedPointObjective` with the forward
# observation sampler's placement rule; at every sampled state its layer is
# chosen from the replayed air mass of that state and the column meteorology
# (`CSObservationColumnMet`) with `Observations.sample_column_layer!`, the sampler's
# own function. The observation is then the mixing ratio of that layer,
# `h = r_k / max(m_k, ε)`, and its functional has the weight `1/max(m_k, ε)`
# there. A record with a time window samples several steps,
# `H = Σ_s ω_s h(r_s)` with Δt/ΣΔt weights, and its functional spreads over
# those steps.
#
# Checked here:
#   * parity with the forward sampler (`build_observation_sampler`,
#     nearest-window mode): same layer and same sampled value;
#   * a 10 m surface intake and a 5 km aircraft sample in one cell and step
#     read different layers and give different simulated values;
#   * pressure placement against total vs dry interface pressures;
#   * layers follow the replayed air mass of each sampled step;
#   * a window observation's simulated value and gradient are the Δt-weighted
#     sums of its single-step observations';
#   * the single sweep equals the per-observation footprint reference
#     (rtol 1e-12, Float64) for placed and window observations;
#   * central finite differences of the cost match the gradient.

using Test
using Dates: DateTime, Second, year, month, day, hour, minute, second, datetime2unix
using NCDatasets: NCDataset, defDim, defVar
using Logging: NullLogger, with_logger

import AtmosTransport
const AT = AtmosTransport
const ADJ = AT.Adjoints
const O = AT.Observations
const Adv = AT.Operators.Advection

include(joinpath(@__DIR__, "..", "helpers", "cs_observation_columns.jl"))

const T0 = DateTime(2024, 6, 15)
const DT = 600.0
const CODES = O.VERTICAL_PLACEMENT_CODES

_dc(s) = (t = T0 + Second(round(Int, s)); (year(t), month(t), day(t), hour(t), minute(t), second(t)))

function _record(id; lat, lon, t = 450.0, kind = "point", value = 405e-6, value_sigma = 1e-6, kwargs...)
    return AT.CSObservationRecord(; id, date_components = _dc(t), lat, lon, kind, value, value_sigma,
                                  instrument_type = "test", tracer = "CO2", kwargs...)
end

_bind(P, records; kwargs...) =
    AT.bind_to_mesh(AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00"), P.mesh, T0, DT;
                    nsteps = P.nsteps, kwargs...)

# Surface-rate controls: one window over every step, about 1e-6 of a surface
# layer's air mass per second.
function _controls(P; scale = 1.0)
    value(phase) = ntuple(p -> [P.FT(scale * 5e8 * (1 + 0.3sin(phase + i + 2j + p))) for i in 1:P.Nc, j in 1:P.Nc], 6)
    return [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:all, 1:P.nsteps), value(0.0)),
            AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:late, 2:P.nsteps; normalize = true), value(1.3))]
end

_fourdvar(P, observations, controls; scheme = AT.UpwindScheme(), met = column_met(P), kwargs...) =
    AT.cs_surface_flux_4dvar(P.rm, P.m, P.am, P.bm, P.cm, P.mesh, observations, controls;
                             scheme, dt = DT, column_met = met, kwargs...)

_relerr(g, ref) = maximum(maximum(abs.(g[c][p] .- ref[c][p])) for c in eachindex(ref), p in 1:6) /
                  maximum(maximum(abs.(ref[c][p])) for c in eachindex(ref), p in 1:6)

# The centre of cell (2, 3) of panel 2, as Float32-representable degrees (the
# ObsPack reader stores Float32 coordinates; both paths then see one point).
function _site(P; panel = 2, i = 2, j = 3)
    lons, lats = AT.panel_cell_center_lonlat(P.mesh, panel)
    return (lat = Float64(Float32(lats[i, j])), lon = Float64(Float32(lons[i, j])))
end

# A constant 1500 m DEM.
const DEM_1500 = AT.GriddedIO.LatLonPointElevation(collect(0.0:10.0:350.0), collect(-90.0:10.0:90.0),
                                                   fill(1500.0f0, 36, 19), true, "constant-dem")

# ---------------------------------------------------------------------------
# Parity with the forward observation sampler
# ---------------------------------------------------------------------------

# An ObsPack file (the layout of the OCO-2 MIP daily files) with the records'
# vertical fields; NaN entries are written as the fill value.
function _write_obspack(path, records)
    n = length(records)
    NCDataset(path, "c") do ds
        defDim(ds, "obs", n)
        defVar(ds, "time", Float64, ("obs",); attrib = Dict("units" => "seconds since 1970-01-01T00:00:00Z"))[:] =
            [datetime2unix(DateTime(Int.(r.date_components)...)) for r in records]
        defVar(ds, "latitude", Float32, ("obs",))[:] = Float32[r.lat for r in records]
        defVar(ds, "longitude", Float32, ("obs",))[:] = Float32[r.lon for r in records]
        for (name, field) in (("altitude", :altitude_m), ("elevation", :elevation_m),
                              ("intake_height", :intake_height_m), ("pressure", :pressure_pa))
            attrib = Dict{String, Any}("_FillValue" => -1f34)
            name == "pressure" && (attrib["units"] = "Pa")
            defVar(ds, name, Float32, ("obs",); attrib)[:] =
                [isnan(getfield(r, field)) ? -1f34 : Float32(getfield(r, field)) for r in records]
        end
        defVar(ds, "obspack_id", String, ("obs",))[:] = ["rec~$(r.id)" for r in records]
    end
    return path
end

@testset "parity with the forward sampler (nearest window): layer and value" begin
    # No transport: the state after step 1 is the initial state, which the
    # sampler then samples at the 600 s window end (records at 450 s).
    P = hydrostatic_cs_problem(; flux_fraction = 0.0, nsteps = 2)
    site = _site(P)
    records = [_record(1; site..., intake_height_m = 10.0),                       # surface intake
               _record(2; site..., altitude_m = 5000.0, elevation_m = 120.0),     # aircraft
               _record(3; site..., altitude_m = 3000.0),                          # altitude, DEM ground
               _record(4; site..., pressure_pa = 50000.0),                        # pressure only
               _record(5; site..., pressure_pa = 96900.0),                        # near the surface
               _record(6; site..., intake_height_m = 300.0, pressure_pa = 20000.0),
               # The surface layer's top is 251.8 m on the total pressures with the virtual
               # temperature and 246.6 m by the dry rule (dry interfaces and T, before OF-117).
               _record(7; site..., intake_height_m = 249.0)]
    terrain = O.ObservationTerrain(; dem = DEM_1500)
    Hp = P.mesh.Hp
    A = vcat(P.p_top, zeros(P.Nz))
    grid = AT.AtmosGrid(P.mesh, AT.HybridSigmaPressure(A, collect(range(0.0, 1.0; length = P.Nz + 1))),
                        AT.CPU(); FT = Float64)
    state = AT.CubedSphereState(AT.DryBasis, P.m; co2 = P.rm, halo_width = Hp)
    # Both paths on one pressure basis: the default total pressures with the
    # virtual temperature (humidity passed to both), and the dry rule (none).
    function parity(dir, basis)
        total = basis == "total"
        pack = _write_obspack(joinpath(dir, "co2_test_aircraft-insitu_1.nc"), records)
        spec = O.observation_output_spec(Dict{String, Any}("observations" => Dict{String, Any}(
            "path" => joinpath(dir, "obs.nc"), "time_interpolation" => "nearest_window", "tracers" => ["co2"],
            "pressure_basis" => basis,
            "sources" => Any[Dict{String, Any}("kind" => "obspack", "mode" => "soundings", "path" => pack)])))
        sampler = with_logger(NullLogger()) do
            O.build_observation_sampler(spec, state, grid; origin = T0, window_seconds = (0.0, 86400.0),
                                        halo_width = Hp, terrain, has_humidity = total)
        end
        O.begin_observation_day!(sampler, "20240615", 1)
        for t in (0.0, DT)
            O.observe_window_boundary!(sampler, state, t; next_window_seconds = DT,
                                       humidity = total ? P.qpanels : nothing)
        end
        with_logger(NullLogger()) do
            O.finish_observations!(sampler)
        end
        close(sampler)
        forward = NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            Dict(id => (level = Int(l), value = Float64(v), method = Int8(c))
                 for (id, l, v, c) in zip(ds["id"][:], ds["intake_level"][:], ds["co2_intake"][:],
                                          ds["vertical_placement"][:]))
        end
        @test length(forward) == length(records)

        placement = O.VerticalPlacementSpec(; pressure_basis = total ? O.TotalFromHumidity() : O.DryInterfaces())
        obs = _bind(P, records; terrain, placement)
        zero_controls = [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:all, 1:P.nsteps),
                                                 ntuple(_ -> zeros(P.Nc, P.Nc), 6))]
        met = column_met(P; gravity = AT.gravity(grid), humidity = total ? P.qpanels : nothing)
        result = _fourdvar(P, obs, zero_controls; met)
        for (k, r) in enumerate(records)
            f = forward["rec~$(r.id)"]
            @test result.stencils[k].steps == [1]
            @test result.stencils[k].levels == [f.level]
            @test result.stencils[k].placement_method == f.method
            # The sounding files store Float32 values; the Float64 value both
            # paths compute is the layer's tracer / air mass of the sampled state.
            @test Float32(result.simulated[k]) == Float32(f.value)
            p, i, j = only(unique((o.objective.panel, o.objective.i, o.objective.j) for o in obs))
            @test result.simulated[k] == P.rm[p][Hp + i, Hp + j, f.level] / P.m[p][Hp + i, Hp + j, f.level]
        end
        @test [s.placement_method for s in result.stencils] ==
              [CODES.record_height, CODES.record_height, CODES.altitude_minus_dem, CODES.pressure,
               CODES.pressure, CODES.record_height, CODES.record_height]
        return [only(s.levels) for s in result.stencils]
    end
    levels = mktempdir(dir -> parity(dir, "total"))
    @test levels[1] == P.Nz && levels[2] < levels[3] < P.Nz && levels[5] == P.Nz - 1
    @test levels[7] == P.Nz
    # The dry rule: 96900 Pa and the 249 m intake one layer up.
    dry = mktempdir(dir -> parity(dir, "dry"))
    @test dry[5] == P.Nz && dry[7] == P.Nz - 1
end

# ---------------------------------------------------------------------------
# Vertical binding changes the simulated values
# ---------------------------------------------------------------------------

@testset "a 10 m intake and a 5 km aircraft sample read different layers" begin
    P = hydrostatic_cs_problem()
    site = _site(P)
    obs = _bind(P, [_record(1; site..., intake_height_m = 10.0),
                    _record(2; site..., altitude_m = 5000.0, elevation_m = 0.0)])
    result = _fourdvar(P, obs, _controls(P))
    surface, aircraft = result.stencils
    @test surface.levels == [P.Nz]
    # 5 km at 280 K (scale height 8.2 km) is near 540 hPa: layer 5 (450-600 hPa).
    @test aircraft.levels == [5]
    @test result.simulated[1] != result.simulated[2]
    # Surface emissions raise the surface intake far above the aircraft sample.
    @test result.simulated[1] - result.simulated[2] > 1e-6
    @test result.gradients[1][2] != zero(result.gradients[1][2])
end

@testset "pressure placement: total vs dry interface pressures" begin
    # 96900 Pa lies between the dry (96842.8 Pa) and the total (97000 Pa)
    # interface above the surface layer: layer 9 by total pressure, 10 by dry.
    P = hydrostatic_cs_problem(; flux_fraction = 0.0)
    site = _site(P)
    record = [_record(1; site..., pressure_pa = 96900.0)]
    total = _fourdvar(P, _bind(P, record), _controls(P))
    dry_obs = _bind(P, record; placement = O.VerticalPlacementSpec(; pressure_basis = O.DryInterfaces()))
    dry = _fourdvar(P, dry_obs, _controls(P))
    @test total.stencils[1].levels == [P.Nz - 1]
    @test dry.stencils[1].levels == [P.Nz]
    @test total.simulated[1] != dry.simulated[1]
    # Total pressures need the humidity on a dry basis; the dry basis does not.
    err = try _fourdvar(P, _bind(P, record), _controls(P); met = column_met(P; humidity = nothing)) catch e e end
    @test err isa ArgumentError && occursin("pressure_basis = \"dry\"", sprint(showerror, err))
    @test _fourdvar(P, dry_obs, _controls(P); met = column_met(P; humidity = nothing)).simulated == dry.simulated
    # The humidity is panels (or per-step panels), not a temperature type.
    err = try column_met(P; humidity = O.ConstantLayerTemperature(280.0)) catch e e end
    @test err isa ArgumentError && occursin("humidity must be nothing", err.msg)
    # Placed observations need the column meteorology.
    err = try _fourdvar(P, dry_obs, _controls(P); met = nothing) catch e e end
    @test err isa ArgumentError && occursin("column_met", sprint(showerror, err))
    # A placed objective has no final-time footprint.
    @test_throws ArgumentError AT.cs_surface_emission_footprint(P.rm, P.m, P.am, P.bm, P.cm, P.mesh,
                                                                dry_obs[1].objective; scheme = AT.UpwindScheme(),
                                                                dt = DT)
end

@testset "layers follow the replayed air mass of each sampled step" begin
    P = hydrostatic_cs_problem(; nsteps = 4, flux_fraction = 0.03)
    p, i, j = 2, 2, 3
    Hp = P.mesh.Hp
    # Advance the air mass independently, with the production model, and
    # record the total interface pressures of the column after every step.
    Nz = size(P.m[1], 3)
    grid = AT.AtmosGrid(P.mesh, AT.HybridSigmaPressure(zeros(Nz + 1), collect(range(0.0, 1.0, Nz + 1))),
                        AT.CPU(); FT = Float64)
    state = AT.CubedSphereState(AT.DryBasis, P.mesh, ntuple(q -> copy(P.m[q]), 6);
                                CO2 = ntuple(q -> copy(P.rm[q]), 6))
    model = AT.TransportModel(state, AT.allocate_face_fluxes(P.mesh, Nz; basis = AT.DryBasis),
                              grid, AT.UpwindScheme())
    m = model.state.air_mass
    area = AT.cell_area(P.mesh, i, j)
    columns = Vector{Vector{Float64}}()
    for s in 1:P.nsteps
        foreach(copyto!, model.fluxes.am, P.am[s])
        foreach(copyto!, model.fluxes.bm, P.bm[s])
        foreach(copyto!, model.fluxes.cm, P.cm[s])
        AT.step!(model, DT)
        push!(columns, m[p][Hp + i, Hp + j, :])
    end
    total(col) = O.total_interface_pressures!(zeros(P.Nz + 1), col, COLUMN_Q, area, COLUMN_GRAVITY, P.p_top)
    # A pressure between the extremes of one interface over the steps changes layer.
    interface = [total(col)[8] for col in columns]
    @test maximum(interface) - minimum(interface) > 10
    pressure = (maximum(interface) + minimum(interface)) / 2
    site = _site(P; panel = p, i, j)
    obs = _bind(P, [_record(1; site..., t = 1200.0, time_window_s = 2400.0, pressure_pa = pressure),
                    _record(2; site..., t = 1200.0, time_window_s = 2400.0, intake_height_m = 2500.0)])
    result = _fourdvar(P, obs, _controls(P))
    @test result.stencils[1].steps == 1:4
    expected = [pressure >= total(col)[8] ? 8 : 7 for col in columns]
    @test result.stencils[1].levels == expected
    @test length(unique(expected)) == 2
    # The height placement uses the hypsometric heights of the same columns (on the total
    # pressures with the virtual temperature, from the column's humidity).
    heights = map(columns) do col
        column = O.PlacementColumn(P.Nz)
        column.humidity .= COLUMN_Q
        O.sample_column_layer!(column, zeros(P.Nz + 1), zeros(P.Nz + 1), obs[2].objective.placement, col, area,
                               COLUMN_GRAVITY, P.p_top, O.ConstantLayerTemperature(280.0), O.TotalFromHumidity(),
                               AT.DryBasis())
    end
    @test result.stencils[2].levels == heights
end

# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------

@testset "a window observation is the Δt-weighted sum of its single-step observations" begin
    P = hydrostatic_cs_problem(; nsteps = 4)
    site = _site(P)
    records = [_record(1; site..., t = 1200.0, time_window_s = 2100.0, pressure_pa = 62000.0, value = 380e-6),
               _record(2; site..., t = 1200.0, time_window_s = 2100.0, kind = "column", value = 380e-6)]
    for scheme in (AT.UpwindScheme(), AT.PPMScheme(AT.MonotoneLimiter()))
        for record in records
            window = _bind(P, [record])
            sampling = only(window).sampling
            @test sampling.steps == [1, 2, 3, 4] && sum(sampling.weights) ≈ 1
            @test sampling.weights ≈ [450, 600, 600, 450] ./ 2100
            controls = _controls(P)
            w = _fourdvar(P, window, controls; scheme)
            o = only(window)
            singles = [AT.CSObservation(s, o.objective, o.value, o.sigma) for s in sampling.steps]
            parts = [_fourdvar(P, [single], controls; scheme) for single in singles]
            # H = Σ ω_s h_s, and ∇h = Σ ω_s ∇h_s (each gradient divided by its weighted departure).
            @test only(w.simulated) ≈ sum(ω * only(r.simulated) for (ω, r) in zip(sampling.weights, parts)) rtol = 1e-14
            unit(r) = [ntuple(q -> g[q] ./ (only(r.residuals) / only(r.observations).sigma^2), 6) for g in r.gradients]
            expected = [ntuple(q -> sum(ω .* unit(r)[c][q] for (ω, r) in zip(sampling.weights, parts)), 6)
                        for c in eachindex(controls)]
            @test _relerr(unit(w), expected) <= 1e-12
            @test [only(r.stencils).levels for r in parts] == [[l] for l in only(w.stencils).levels]
            # The placed sample changes layer within its window (the air mass evolves).
            o.objective isa AT.CSPlacedPointObjective && @test length(unique(only(w.stencils).levels)) > 1
        end
    end
end

# ---------------------------------------------------------------------------
# Single sweep vs the per-observation reference
# ---------------------------------------------------------------------------

# Per observation and sampled step: the forward value and the footprint of a
# run truncated to that step, for the level the stencil records; the
# observation's departure scales the time-weighted sum.
function _reference(P, observations, stencils, controls; scheme, physics)
    FT = P.FT
    rates = ADJ._surface_rates_from_controls(controls, P.nsteps, P.mesh, P.rm[1])
    gradients = [ntuple(_ -> zeros(FT, P.Nc, P.Nc), 6) for _ in controls]
    simulated = FT[]
    for (obs, stencil) in zip(observations, stencils)
        parts = map(zip(stencil.steps, stencil.weights, stencil.levels)) do (s, ω, level)
            objective = level == 0 ? AT.CSColumnMeanObjective(stencil.panel, stencil.i, stencil.j) :
                                     AT.CSLayerMeanObjective(stencil.panel, stencil.i, stencil.j, level)
            truncated = (P.rm, P.m, P.am[1:s], P.bm[1:s], P.cm[1:s], P.mesh, objective)
            value = AT.run_cs_footprint_forward(truncated...; scheme, dt = DT, emission_rates = rates[1:s],
                                                physics(P)...)
            footprint = AT.cs_surface_emission_footprint(truncated...; scheme, dt = DT,
                                                         base_emission_rates = rates[1:s], physics(P)...)
            (; s, ω, value, footprint)
        end
        H = sum(part.ω * part.value for part in parts)
        push!(simulated, H)
        scale = (H - obs.value) / obs.sigma^2
        for part in parts, (c, control) in enumerate(controls)
            for (idx, step) in enumerate(control.window.steps)
                step <= part.s || continue
                weight = FT(scale) * FT(part.ω) * FT(control.window.weights[idx])
                for q in 1:6
                    gradients[c][q] .+= weight .* part.footprint.footprints[step][q]
                end
            end
        end
    end
    return (; simulated, gradients)
end

# TM5 entrainment/detrainment on layers `ks` (kg m⁻² s⁻¹): with layers of
# 300-1000 kg m⁻², tens of percent of a layer per 600 s step.
_layers(P, ks, v) = ntuple(_ -> (e = zeros(P.FT, P.Nc, P.Nc, P.Nz); e[:, :, ks] .= P.FT(v); e), 6)

function _mixed_observations(P)
    a, b = _site(P), _site(P; panel = 5, i = 3, j = 1)
    records = [_record(1; a..., t = 100.0, intake_height_m = 10.0, value = 420e-6),
               _record(2; a..., t = 1000.0, altitude_m = 5200.0, elevation_m = 300.0, value = 401e-6),
               _record(3; b..., t = 700.0, pressure_pa = 70000.0, time_window_s = 1500.0, value = 399e-6),
               _record(4; b..., t = 1500.0, kind = "column", time_window_s = 900.0, value = 404e-6,
                       model_sigma = 0.5e-6),
               _record(5; a..., t = 1700.0, intake_height_m = 10.0, time_window_s = 600.0, value = 430e-6)]
    return _bind(P, records)
end

@testset "single sweep == per-observation footprints for placed and window observations" begin
    P = hydrostatic_cs_problem()
    observations = _mixed_observations(P)
    controls = _controls(P)
    tm5(P) = (; convection_op = AT.TM5Convection(),
                convection_forcing = AT.ConvectionForcing(nothing, nothing,
                    (; entu = _layers(P, 5:8, 0.4), detu = _layers(P, 5:8, 0.28),
                       entd = _layers(P, 6:8, 0.12), detd = _layers(P, 6:8, 0.06))),
                convection_workspace = AT.TM5Workspace(P.m; tile_columns = P.Nc^2,
                    cell_metrics = ntuple(_ -> [AT.cell_area(P.mesh, i, j) for i in 1:P.Nc, j in 1:P.Nc], 6)))
    @testset "$(nameof(typeof(scheme))) $(label)" for (scheme, label, physics) in (
            (AT.UpwindScheme(), "no physics", P -> (;)),
            (AT.PPMScheme(AT.MonotoneLimiter()), "no physics", P -> (;)),
            (AT.LinRoodPPMScheme(), "no physics", P -> (;)),
            (AT.UpwindScheme(), "TM5 convection", tm5))
        full = _fourdvar(P, observations, controls; scheme, physics(P)...)
        ref = _reference(P, observations, full.stencils, controls; scheme, physics)
        @test full.simulated ≈ ref.simulated rtol = 1e-14
        @test _relerr(full.gradients, ref.gradients) <= 1e-12
        stride = _fourdvar(P, observations, controls; scheme, checkpoint = AT.StrideCheckpoint(2), physics(P)...)
        @test stride.simulated == full.simulated
        @test _relerr(stride.gradients, full.gradients) <= 1e-12
    end
end

# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------

@testset "central finite differences of the cost match the gradient" begin
    P = hydrostatic_cs_problem()
    observations = _mixed_observations(P)
    controls = _controls(P)
    direction = _controls(P; scale = 0.37)
    shifted(ε) = [AT.CSSurfaceFluxControl(c.window, ntuple(q -> c.value[q] .+ ε .* d.value[q], 6))
                  for (c, d) in zip(controls, direction)]
    for (scheme, tol) in ((AT.UpwindScheme(), 1e-8), (AT.PPMScheme(AT.MonotoneLimiter()), 1e-6))
        base = _fourdvar(P, observations, controls; scheme)
        directional = sum(sum(sum(g[q] .* d.value[q]) for q in 1:6) for (g, d) in zip(base.gradients, direction))
        ε = 1e-2
        fd = (_fourdvar(P, observations, shifted(ε); scheme).cost -
              _fourdvar(P, observations, shifted(-ε); scheme).cost) / 2ε
        @test abs(fd - directional) <= tol * abs(directional)
        @test base.observation_cost > 1                           # well above rounding
    end
end
