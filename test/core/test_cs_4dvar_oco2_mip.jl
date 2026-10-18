#!/usr/bin/env julia
#
# OCO-2 level retrievals in the 4D-Var and the forward sampler (issue #20
# follow-up, decision D30): hat-projected averaging-kernel columns, TM5's
# 19-layer mode, forward-only co-sampling of records that are not assimilated,
# cell averaging, and the forward sampler's averaging-kernel columns.
#
# Checked here:
#   * the single sweep equals the per-observation reference for level columns
#     (weights from an independent construction of the hat projection), with
#     windows and TM5-mode records (rtol 1e-12, Float64); central finite
#     differences of the cost;
#   * co-sampling: flagged records are simulated and reported with their flag
#     but change neither the cost nor the reverse pass's forcing and tape
#     length (bit for bit; the gradient bit for bit with one thread), and are
#     not screened; departures carry them;
#   * cell averaging: the averaged observation's value, error and gradient are
#     the 1/σ²-weighted combinations of its members'; the error formula with
#     and without correlation; the model-error inflation alternative;
#   * forward sampler (nearest window) parity: the averaging-kernel column of a
#     sounding read from a MIP file equals the 4D-Var's simulated value of the
#     same record converted to schema v2; without the option the sounding file
#     is unchanged.

using Test
using Dates: DateTime, Second, year, month, day, hour, minute, second
using LinearAlgebra: dot
using NCDatasets: NCDataset
using Logging: NullLogger, with_logger

import AtmosTransport
const AT = AtmosTransport
const ADJ = AT.Adjoints
const O = AT.Output
const Adv = AT.Operators.Advection

include(joinpath(@__DIR__, "..", "helpers", "cs_observation_columns.jl"))
include(joinpath(@__DIR__, "..", "helpers", "oco2_mip_fixture.jl"))

const T0 = DateTime(2024, 6, 15)
const DT = 600.0

_dc(s) = (t = T0 + Second(round(Int, s)); (year(t), month(t), day(t), hour(t), minute(t), second(t)))

# An OCO-2 level retrieval at surface pressure `ps` (Pa), surface first, in mol/mol.
function _level(ps; tilt = 0.0, sigma = Float64[], mapping = O.HatLevelProjection(), prior_column = NaN)
    r = oco2_fixture_retrieval(ps / 100; tilt)
    column = O.RetrievalColumn(; pressure_pa = reverse(OCO2_SIGMA .* ps), pressure_weight = reverse(r.pw),
                               averaging_kernel = reverse(r.ak), prior_profile = reverse(r.prior) .* 1e-6,
                               profile_sigma = sigma, mapping = O.HatLevelProjection(), prior_column)
    return O.oco2_level_form(mapping, column)
end

function _record(id; lat, lon, t = 450.0, value = 405e-6, value_sigma = 0.6e-6, kind = "column", kwargs...)
    return AT.CSObservationRecord(; id, date_components = _dc(t), lat, lon, kind, value, value_sigma,
                                  instrument_type = "OCO2_OG", tracer = "CO2", kwargs...)
end

_set(records) = AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00")
_bind(P, records; kwargs...) = AT.bind_to_mesh(_set(records), P.mesh, T0, DT; nsteps = P.nsteps, kwargs...)

function _controls(P; scale = 1.0)
    value(phase) = ntuple(p -> [P.FT(scale * 5e8 * (1 + 0.3sin(phase + i + 2j + p))) for i in 1:P.Nc, j in 1:P.Nc], 6)
    return [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:all, 1:P.nsteps), value(0.0)),
            AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:late, 2:P.nsteps; normalize = true), value(1.3))]
end

_fourdvar(P, observations, controls; scheme = AT.UpwindScheme(), met = column_met(P), kwargs...) =
    AT.cs_surface_flux_4dvar(P.rm, P.m, P.am, P.bm, P.cm, P.mesh, observations, controls;
                             scheme, dt = DT, column_met = met, kwargs...)

# A cell centre as Float32-representable degrees (MIP files store Float32
# coordinates; the sampler and the 4D-Var then see one point).
function _site(P; panel = 2, i = 2, j = 3)
    lons, lats = AT.panel_cell_center_lonlat(P.mesh, panel)
    return (lat = Float64(Float32(lats[i, j])), lon = Float64(Float32(lons[i, j])))
end

_relerr(g, ref) = maximum(maximum(abs.(g[c][p] .- ref[c][p])) for c in eachindex(ref), p in 1:6) /
                  maximum(maximum(abs.(ref[c][p])) for c in eachindex(ref), p in 1:6)

# ---------------------------------------------------------------------------
# Single sweep vs the per-observation reference
# ---------------------------------------------------------------------------

function _replayed_states(P, controls; scheme)
    rates = ADJ._surface_rates_from_controls(controls, P.nsteps, P.mesh, P.rm[1])
    m = ntuple(q -> copy(P.m[q]), 6)
    rm = ntuple(q -> copy(P.rm[q]), 6)
    Adv.fill_panel_halos!(m, P.mesh; dir = 0)
    Adv.fill_panel_halos!(rm, P.mesh; dir = 0)
    ws = Adv.CSAdvectionWorkspace(P.mesh, rm[1]; column_scratch = ADJ.needs_column_scratch(scheme))
    states = map(1:P.nsteps) do s
        ADJ._replay_cs_step!(rm, m, P.am[s], P.bm[s], P.cm[s], P.mesh, scheme, ws,
                             τ -> ADJ._add_surface_rates!(rm, rates[s], τ, P.mesh);
                             dt = DT, flux_scale = 1.0, cfl_limit = AT.Parameters.DEFAULT_CFL_LIMIT,
                             diffusion_op = AT.NoDiffusion(), diffusion_workspace = nothing,
                             diffusion_meteo = nothing, convection_op = AT.NoConvection(),
                             convection_forcing = nothing, convection_workspace = nothing)
        (m = ntuple(q -> copy(m[q]), 6), rm = ntuple(q -> copy(rm[q]), 6))
    end
    return states, rates
end

# The weights of an averaging-kernel column on the tracer mass of one sampled
# column: the hat projection from the independent construction (a level
# column) or TM5's rebin (a layer column), on the total interfaces.
function _column_weights(o::AT.CSRetrievalColumnObjective, m, P)
    p_half = O.total_interface_pressures!(zeros(P.Nz + 1), m, COLUMN_Q, o.area, COLUMN_GRAVITY, P.p_top)
    rc = o.retrieval
    proj = rc.mapping isa O.HatLevelProjection ? hat_projection_reference(p_half, Float64.(m), rc.pressure_pa).P :
                                                  O.retrieval_rebin_matrix(p_half, rc.pressure_pa)
    H = vec((rc.pressure_weight .* rc.averaging_kernel)' * proj)
    return H ./ max.(Float64.(m), eps(P.FT)), O.retrieval_prior_offset(rc), proj
end

function _reference(P, observations, stencils, controls; scheme)
    FT = P.FT
    states, rates = _replayed_states(P, controls; scheme)
    Hp = P.mesh.Hp
    gradients = [ntuple(_ -> zeros(FT, P.Nc, P.Nc), 6) for _ in controls]
    simulated = FT[]
    profiles = Vector{Float64}[]
    for (obs, stencil) in zip(observations, stencils)
        o = obs.objective
        parts = map(zip(stencil.steps, stencil.weights)) do (s, ω)
            m = states[s].m[o.panel][Hp + o.i, Hp + o.j, :]
            rm = states[s].rm[o.panel][Hp + o.i, Hp + o.j, :]
            w, offset, profile = if o isa AT.CSRetrievalColumnObjective
                w, c, proj = _column_weights(o, m, P)
                (FT.(w), FT(c), proj * (Float64.(rm) ./ Float64.(m)))
            else
                (fill(one(FT) / sum(m), P.Nz), zero(FT), Float64[])
            end
            seed = ntuple(q -> zeros(FT, size(P.rm[q])), 6)
            seed[o.panel][Hp + o.i, Hp + o.j, :] .= w
            footprint = AT.cs_surface_emission_footprint_from_seed(seed, P.m, P.am[1:s], P.bm[1:s], P.cm[1:s],
                                                                   P.mesh; scheme, dt = DT, base_panels_rm0 = P.rm,
                                                                   base_emission_rates = rates[1:s])
            (; s, ω, value = offset + sum(w .* rm), footprint, profile)
        end
        H = sum(part.ω * part.value for part in parts)
        push!(simulated, H)
        push!(profiles, o isa AT.CSRetrievalColumnObjective ? sum(part.ω .* part.profile for part in parts) : Float64[])
        obs.assimilate || continue
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
    return (; simulated, gradients, profiles)
end

# Level columns (instantaneous, windowed, model surface above and below the
# retrieval's, with a profile uncertainty), a TM5-mode record and a column mean.
function _level_observations(P; kwargs...)
    a, b = _site(P), _site(P; panel = 5, i = 3, j = 1)
    records = [_record(1; a..., t = 100.0, value = 402e-6, retrieval = _level(97000.0; prior_column = 404.2e-6)),
               _record(2; a..., t = 1000.0, time_window_s = 1500.0, value = 399e-6,
                       retrieval = _level(101500.0; tilt = 0.3, sigma = fill(1e-6, 20))),
               _record(3; b..., t = 700.0, time_window_s = 900.0, value = 404e-6, retrieval = _level(99500.0; tilt = -0.2)),
               _record(4; b..., t = 1500.0, value = 405e-6, retrieval = _level(98500.0; mapping = O.LayerRebin())),
               _record(5; a..., t = 1700.0, value = 403e-6)]
    return _bind(P, records; kwargs...)
end

@testset "single sweep == per-observation reference for level columns (hat projection)" begin
    P = hydrostatic_cs_problem()
    observations = _level_observations(P)
    @test observations[1].objective.retrieval.mapping isa O.HatLevelProjection
    @test observations[4].objective.retrieval.mapping isa O.LayerRebin
    controls = _controls(P)
    @testset "$(nameof(typeof(scheme)))" for scheme in (AT.UpwindScheme(), AT.PPMScheme(AT.MonotoneLimiter()))
        full = _fourdvar(P, observations, controls; scheme)
        ref = _reference(P, observations, full.stencils, controls; scheme)
        @test full.simulated ≈ ref.simulated rtol = 1e-14
        @test _relerr(full.gradients, ref.gradients) <= 1e-12
        for idx in 1:4
            @test full.stencils[idx].retrieval_profile ≈ ref.profiles[idx] rtol = 1e-13
        end
        @test length(full.stencils[1].retrieval_profile) == 20 && length(full.stencils[4].retrieval_profile) == 19
    end
end

@testset "central finite differences of the cost match the gradient (level columns)" begin
    P = hydrostatic_cs_problem()
    observations = _level_observations(P)
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
        @test base.observation_cost > 1
    end
end

# ---------------------------------------------------------------------------
# Forward-only co-sampling
# ---------------------------------------------------------------------------

@testset "co-sampling: flagged records are simulated, not assimilated (bit for bit)" begin
    P = hydrostatic_cs_problem()
    a, b = _site(P), _site(P; panel = 5, i = 3, j = 1)
    ps = COLUMN_P_HALF[end]
    assimilated = [_record(1; a..., t = 100.0, value = 402e-6, retrieval = _level(97000.0)),
                   _record(2; b..., t = 700.0, value = 404e-6, retrieval = _level(99500.0)),
                   _record(3; a..., t = 650.0, kind = "point", intake_height_m = 10.0, value = 430e-6)]
    flagged = [_record(11; b..., t = 300.0, value = 401e-6, flag = 1, retrieval = _level(98000.0)),
               _record(12; a..., t = 650.0, kind = "point", intake_height_m = 10.0, value = 425e-6, flag = 3),
               _record(13; a..., t = 1700.0, value = 406e-6, flag = 2),             # after the last assimilated step
               # far from the model surface pressure: not screened, not an error
               _record(14; a..., t = 500.0, value = 405e-6, flag = 1, retrieval = _level(ps - 30000.0))]
    records = [assimilated[1], flagged[1], assimilated[2], flagged[2], flagged[3], assimilated[3], flagged[4]]
    controls = _controls(P)
    limit = (; max_surface_pressure_mismatch_pa = 15000.0)
    dropped = with_logger(NullLogger()) do
        _bind(P, records; limit...)
    end
    cosampled = with_logger(NullLogger()) do
        _bind(P, records; flagged_records = "cosample", limit...)
    end
    @test length(dropped) == 3 && length(cosampled) == 7
    @test [o.assimilate for o in cosampled] == [true, false, true, false, false, true, false]
    @test AT.bind_to_mesh(_set(records), P.mesh, T0, DT; nsteps = P.nsteps, flagged_records = AT.CosampleFlaggedRecords(),
                          limit...)[2].id == 11
    @test_throws ArgumentError _bind(P, records; flagged_records = "keep")
    rates = ADJ._surface_rates_from_controls(controls, P.nsteps, P.mesh, P.rm[1])
    # What the reverse pass receives: the step forcing of the assimilated
    # observations and the length of the taped run.
    function reverse_input(observations, scheme)
        f = ADJ._observe_cs_forward(P.rm, P.m, P.am, P.bm, P.cm, P.mesh, observations; scheme, dt = DT,
                                    emission_rates = rates, column_met = column_met(P), build_functionals = true)
        active = findall(s -> s.assimilated, f.stencils)
        scales = [(f.simulated[i] - observations[i].value) / observations[i].sigma^2 for i in active]
        forcing = ADJ._assemble_step_forcing(f.functionals[active], scales, P.mesh, P.Nz, P.rm[1])
        return forcing, maximum(observations[i].step for i in active)
    end
    for scheme in (AT.UpwindScheme(), AT.PPMScheme(AT.MonotoneLimiter()))
        r_drop = _fourdvar(P, dropped, controls; scheme)
        r_co = _fourdvar(P, cosampled, controls; scheme)
        # The cost is that without the flagged records, and the reverse pass
        # receives the same forcing and tape length, bit for bit.
        @test r_co.cost == r_drop.cost && r_co.observation_cost == r_drop.observation_cost
        (f_drop, n_drop), (f_co, n_co) = reverse_input(dropped, scheme), reverse_input(cosampled, scheme)
        @test n_co == n_drop == 2                                            # record 13 (step 3) does not extend the tape
        @test all(getfield(f_co, name) == getfield(f_drop, name) for name in fieldnames(typeof(f_co)))
        # The gradient: bit for bit single-threaded, to rounding with threads
        # (the atomics of the threaded adjoint; `gradients_match`).
        @test gradients_match(r_co.gradients, r_drop.gradients)
        @test r_co.simulated[[1, 3, 6]] == r_drop.simulated
        @test [s.assimilated for s in r_co.stencils] == [o.assimilate for o in cosampled]
        # A co-sample's simulated value is the one the record has when it is assimilated.
        alone = _fourdvar(P, _bind(P, [_record(11; b..., t = 300.0, value = 401e-6, retrieval = _level(98000.0)),
                                       _record(13; a..., t = 1700.0, value = 406e-6)]), controls; scheme)
        @test r_co.simulated[[2, 5]] == alone.simulated
        @test abs(r_co.stencils[7].surface_pressure_mismatch_pa) > 15000       # beyond the limit, not screened
    end
    # Departures report the co-samples with their flag.
    result = _fourdvar(P, cosampled, controls)
    deps = AT.build_departure_set(_set(records), result, P.mesh, "2024-06-15 00:00:00", DT, P.nsteps)
    @test [d.flag for d in deps.records] == [0, 1, 0, 3, 2, 0, 1]
    @test [d.assimilated for d in deps.records] == [true, false, true, false, false, true, false]
    mktempdir() do dir
        path = AT.write_departures(joinpath(dir, "deps.nc"), deps)
        NCDataset(path) do ds
            @test ds["flag"][:] == Int8[0, 1, 0, 3, 2, 0, 1]
            @test ds["assimilated"][:] == Int8[1, 0, 1, 0, 0, 1, 0]
            @test !haskey(ds, "cell_average_id")
        end
        back = AT.read_departures(path)
        for (x, y) in zip(deps.records, back.records)
            @test all(f -> isequal(getfield(x, f), getfield(y, f)), fieldnames(AT.CSDepartureRecord))
        end
    end
    # A flagged record is never assimilated.
    kw = Dict(f => getfield(deps[2], f) for f in fieldnames(AT.CSDepartureRecord))
    @test_throws ArgumentError AT.CSDepartureRecord(; kw..., assimilated = true)
end

# ---------------------------------------------------------------------------
# Cell averaging
# ---------------------------------------------------------------------------

@testset "cell averaging: 1/σ² weights, error formula, gradient = weighted members" begin
    P = hydrostatic_cs_problem()
    a, b = _site(P), _site(P; panel = 5, i = 3, j = 1)
    group = [_record(21; a..., t = 100.0, value = 402e-6, value_sigma = 0.5e-6, model_sigma = 0.3e-6,
                     retrieval = _level(97000.0)),
             _record(22; a..., t = 110.0, value = 403e-6, value_sigma = 0.8e-6, model_sigma = 0.2e-6,
                     retrieval = _level(97500.0; tilt = 0.1, sigma = fill(1e-6, 20))),
             _record(23; a..., t = 125.0, value = 401e-6, value_sigma = 0.6e-6, model_sigma = 0.25e-6)]
    others = [_record(24; a..., t = 900.0, value = 404e-6, retrieval = _level(97000.0)),     # another step
              _record(25; b..., t = 105.0, value = 404e-6, retrieval = _level(99000.0)),     # another cell
              _record(26; a..., t = 300.0, value = 404e-6, retrieval = _level(97000.0))]     # beyond the tolerance
    records = vcat(group[1:2], others[1:2], group[3:3], others[3:3])
    obs = with_logger(NullLogger()) do
        _bind(P, records; cell_averaging = AT.CellAverage(; time_tolerance_s = 60.0))
    end
    @test length(obs) == 7
    @test [o.assimilate for o in obs] == [false, false, true, true, false, true, true]
    avg = obs[end]
    @test avg.objective isa AT.CSCellAverageObjective && avg.objective.ids == [21, 22, 23] && avg.id == 21
    σ = [0.5e-6, 0.8e-6, 0.6e-6]
    w = (1 ./ σ .^ 2) ./ sum(1 ./ σ .^ 2)
    @test avg.objective.weights ≈ w rtol = 1e-14
    @test avg.value ≈ dot(w, [402e-6, 403e-6, 401e-6]) rtol = 1e-14
    profile_var = O.retrieval_model_error_variance(group[2].retrieval)
    σ_mod2 = w[1] * 0.3e-6^2 + w[2] * (0.2e-6^2 + profile_var) + w[3] * 0.25e-6^2
    @test avg.sigma^2 ≈ 1 / sum(1 ./ σ .^ 2) + σ_mod2 rtol = 1e-12             # ρ = 0: Σw²σ²/(Σw)² = 1/Σ(1/σ²)
    correlated = with_logger(NullLogger()) do
        _bind(P, records; cell_averaging = AT.CellAverage(; correlation = 0.5))[end]
    end
    ρ = [i == j ? 1.0 : 0.5 for i in 1:3, j in 1:3]
    @test correlated.sigma^2 ≈ sum(w[i] * w[j] * ρ[i, j] * σ[i] * σ[j] for i in 1:3, j in 1:3) + σ_mod2 rtol = 1e-12
    @test_throws ArgumentError _bind(P, records; cell_averaging = AT.CellAverage(),
                                     max_surface_pressure_mismatch_pa = 5000.0)
    @test_throws ArgumentError AT.CellAverage(; correlation = 1.5)
    # The averaged observation's simulated value and gradient are the weighted
    # combinations of the members'.
    controls = _controls(P)
    only_group = with_logger(NullLogger()) do
        _bind(P, group; cell_averaging = AT.CellAverage())
    end
    r = _fourdvar(P, only_group, controls)
    H = r.simulated[1:3]
    @test r.simulated[4] ≈ dot(w, H) rtol = 1e-14
    @test r.stencils[4].placement_method == AT.CELL_AVERAGE_PLACEMENT_CODE && r.stencils[4].assimilated
    scale = (r.simulated[4] - only_group[4].value) / only_group[4].sigma^2
    # Each member alone with σ = 1 and the value that makes its weighted
    # departure w_i (H̄ − ȳ)/σ̄²: their gradients sum to the average's.
    members = [AT.CSObservation(o.sampling, o.objective, H[i] - w[i] * scale, 1.0; id = o.id)
               for (i, o) in enumerate(only_group[1:3])]
    r_members = _fourdvar(P, members, controls)
    @test _relerr(r.gradients, r_members.gradients) <= 1e-12
    @test r.observation_cost ≈ 0.5 * ((r.simulated[4] - only_group[4].value) / only_group[4].sigma)^2 rtol = 1e-14
    # Departures: the members are rows (co-samples of the average), and so is
    # the average, numbered 1, with a row id above every record id (26 + 1).
    result = _fourdvar(P, obs, controls)
    deps = AT.build_departure_set(_set(records), result, P.mesh, "2024-06-15 00:00:00", DT, P.nsteps)
    @test [d.id for d in deps.records] == [21, 22, 24, 25, 23, 26, 27]
    @test [d.cell_average_id for d in deps.records] == [1, 1, 0, 0, 1, 0, 1]
    @test [d.assimilated for d in deps.records] == [false, false, true, true, false, true, true]
    row = deps.records[end]
    @test row.vertical_placement == AT.CELL_AVERAGE_PLACEMENT_CODE && row.cell_average_correlation == 0
    @test row.observed_value == obs[end].value && row.simulated_value == result.simulated[end]
    @test row.value_sigma^2 ≈ 1 / sum(1 ./ σ .^ 2) rtol = 1e-12
    @test row.model_sigma^2 ≈ σ_mod2 rtol = 1e-12
    # The assimilated rows are the terms of the cost.
    assimilated_rows(d) = [r for r in d.records if r.assimilated]
    @test sum(r.normalized_departure^2 for r in assimilated_rows(deps)) ≈ 2 * result.observation_cost rtol = 1e-12
    plain = with_logger(NullLogger()) do
        _bind(P, records)
    end
    r_plain = _fourdvar(P, plain, controls)
    d_plain = AT.build_departure_set(_set(records), r_plain, P.mesh, "2024-06-15 00:00:00", DT, P.nsteps)
    @test sum(r.normalized_departure^2 for r in assimilated_rows(d_plain)) ≈ 2 * r_plain.observation_cost rtol = 1e-12
    mktempdir() do dir
        back = AT.read_departures(AT.write_departures(joinpath(dir, "avg.nc"), deps))
        @test [d.cell_average_id for d in back.records] == [1, 1, 0, 0, 1, 0, 1]
        for (x, y) in zip(deps.records, back.records)
            @test all(f -> isequal(getfield(x, f), getfield(y, f)), fieldnames(AT.CSDepartureRecord))
        end
    end
    # Record ids of any sign (0 and negative included) keep the departure ids
    # unique: the average's row id is above every record id of the set.
    signed = [_record(0; a..., t = 100.0, value = 402e-6, value_sigma = 0.5e-6, retrieval = _level(97000.0)),
              _record(-21; a..., t = 110.0, value = 403e-6, value_sigma = 0.8e-6),
              _record(-22; b..., t = 105.0, value = 404e-6)]
    signed_obs = with_logger(NullLogger()) do
        _bind(P, signed; cell_averaging = AT.CellAverage())
    end
    signed_deps = AT.build_departure_set(_set(signed), _fourdvar(P, signed_obs, controls), P.mesh,
                                         "2024-06-15 00:00:00", DT, P.nsteps)
    @test [d.id for d in signed_deps.records] == [0, -21, -22, 1]
    @test [d.cell_average_id for d in signed_deps.records] == [1, 1, 0, 1]
    # Departure ids must be unique, and every member row names an average row.
    twin = deps.records[1]
    kw = Dict(f => getfield(twin, f) for f in fieldnames(AT.CSDepartureRecord))
    @test_throws ArgumentError AT.CSDepartureSet([deps.records[3], AT.CSDepartureRecord(; kw...,
                                                     id = deps.records[3].id)]; mesh_Nc = deps.mesh_Nc,
                                                 mesh_panel_convention = deps.mesh_panel_convention,
                                                 mesh_cs_definition_tag = deps.mesh_cs_definition_tag,
                                                 t_start = deps.t_start, dt_seconds = deps.dt_seconds,
                                                 nsteps = deps.nsteps)
    @test_throws ArgumentError AT.CSDepartureSet(deps.records[1:6]; mesh_Nc = deps.mesh_Nc,
                                                 mesh_panel_convention = deps.mesh_panel_convention,
                                                 mesh_cs_definition_tag = deps.mesh_cs_definition_tag,
                                                 t_start = deps.t_start, dt_seconds = deps.dt_seconds,
                                                 nsteps = deps.nsteps)
    # A hand-built average is checked against its records as bind_to_mesh
    # builds it: here of two flagged (not assimilable) records.
    flagged = [_record(31; a..., t = 100.0, value = 402e-6, flag = 1),
               _record(32; a..., t = 110.0, value = 403e-6, flag = 1)]
    co = with_logger(NullLogger()) do
        _bind(P, flagged; flagged_records = "cosample")
    end
    w2 = [0.5, 0.5]
    fake = AT.CSObservation(co[1].sampling,
                            AT.CSCellAverageObjective(co[1].objective.panel, co[1].objective.i, co[1].objective.j,
                                                      [o.objective for o in co], w2, [31, 32];
                                                      measurement_sigma = sqrt(0.5) * 1e-6, model_sigma = 0.0,
                                                      correlation = 0.0),
                            402.5e-6, sqrt(0.5) * 1e-6; id = 999)
    r_fake = _fourdvar(P, [co; fake], controls)
    err = try
        AT.build_departure_set(_set(flagged), r_fake, P.mesh, "2024-06-15 00:00:00", DT, P.nsteps)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("flag 1", err.msg)
    # ... and its members must be co-sampled observations of the run.
    err = try
        AT.build_departure_set(_set(group), _fourdvar(P, only_group[4:4], controls), P.mesh,
                               "2024-06-15 00:00:00", DT, P.nsteps)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("has no observation", err.msg)
    # ... sampled by the same operator as each member's co-sample (here a dry
    # pressure basis where the co-sample uses total pressures) ...
    avg4 = only_group[4].objective
    m1 = only_group[1].objective
    dry = AT.CSRetrievalColumnObjective(m1.panel, m1.i, m1.j, m1.area, group[1].retrieval, O.DryInterfaces())
    swapped = AT.CSObservation(only_group[4].sampling,
                               AT.CSCellAverageObjective(avg4.panel, avg4.i, avg4.j, [dry; avg4.members[2:end]],
                                                         avg4.weights, avg4.ids;
                                                         measurement_sigma = avg4.measurement_sigma,
                                                         model_sigma = avg4.model_sigma,
                                                         correlation = avg4.correlation),
                               only_group[4].value, only_group[4].sigma; id = only_group[4].id)
    err = try
        AT.build_departure_set(_set(group), _fourdvar(P, [only_group[1:3]; swapped], controls), P.mesh,
                               "2024-06-15 00:00:00", DT, P.nsteps)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("different observation operator", err.msg)
    # ... and of one tracer (bind_to_mesh groups by tracer).
    mixed = [_record(41; a..., t = 100.0, value = 402e-6, value_sigma = 0.5e-6, retrieval = _level(97000.0)),
             _record(42; a..., t = 110.0, value = 1.9e-6, value_sigma = 0.5e-8, tracer = "CH4",
                     retrieval = _level(97000.0))]
    bound = with_logger(NullLogger()) do
        _bind(P, mixed)
    end
    cos = [AT.CSObservation(o.sampling, o.objective, o.value, o.sigma; id = o.id, assimilate = false) for o in bound]
    wm = (1 ./ [0.5e-6, 0.5e-8] .^ 2) ./ sum(1 ./ [0.5e-6, 0.5e-8] .^ 2)
    σm = sqrt(1 / sum(1 ./ [0.5e-6, 0.5e-8] .^ 2))
    σmod = sqrt(sum(wm[i] * (mixed[i].model_sigma^2 + O.retrieval_model_error_variance(mixed[i].retrieval))
                    for i in 1:2))
    mixed_avg = AT.CSObservation(cos[1].sampling,
                                 AT.CSCellAverageObjective(m1.panel, m1.i, m1.j, [o.objective for o in cos], wm,
                                                           [41, 42]; measurement_sigma = σm, model_sigma = σmod,
                                                           correlation = 0.0),
                                 dot(wm, [402e-6, 1.9e-6]), hypot(σm, σmod); id = 41)
    err = try
        AT.build_departure_set(_set(mixed), _fourdvar(P, [cos; mixed_avg], controls), P.mesh,
                               "2024-06-15 00:00:00", DT, P.nsteps)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("mix tracers", err.msg)
    # A cell average cannot hold an averaging-kernel column with a surface-pressure limit.
    limited = AT.CSRetrievalColumnObjective(only_group[1].objective.panel, only_group[1].objective.i,
                                            only_group[1].objective.j, only_group[1].objective.area,
                                            group[1].retrieval, O.TotalFromHumidity();
                                            max_surface_pressure_mismatch_pa = 5000.0)
    @test_throws ArgumentError AT.CSCellAverageObjective(limited.panel, limited.i, limited.j, [limited], [1.0], [21])
    # The alternative: keep the records, inflate their model error by k.
    inflated = AT.inflate_cell_model_error(_set(records), P.mesh, T0, DT; nsteps = P.nsteps)
    @test [r.model_sigma for r in inflated.records] ≈
          [sqrt(3) * 0.3e-6, sqrt(3) * 0.2e-6, 0.0, 0.0, sqrt(3) * 0.25e-6, 0.0] rtol = 1e-14
    @test all(x.retrieval == y.retrieval && x.value == y.value for (x, y) in zip(inflated.records, records))
end

# ---------------------------------------------------------------------------
# Forward sampler: averaging-kernel columns
# ---------------------------------------------------------------------------

function _sample_file(P, path, dir, label; averaging_kernel = nothing)
    Hp = P.mesh.Hp
    A = vcat(P.p_top, zeros(P.Nz))
    grid = AT.AtmosGrid(P.mesh, AT.HybridSigmaPressure(A, collect(range(0.0, 1.0; length = P.Nz + 1))), AT.CPU();
                        FT = Float64)
    state = AT.CubedSphereState(AT.DryBasis, P.m; co2 = P.rm, halo_width = Hp)
    source = Dict{String, Any}("kind" => "oco2_lite", "path" => path, "quality_filter" => "none")
    averaging_kernel === nothing || merge!(source, averaging_kernel)
    spec = O.observation_output_spec(Dict{String, Any}("observations" => Dict{String, Any}(
        "path" => joinpath(dir, "$(label).nc"), "time_interpolation" => "nearest_window", "tracers" => ["co2"],
        "sources" => Any[source])))
    sampler = with_logger(NullLogger()) do
        O.build_observation_sampler(spec, state, grid; origin = T0, window_seconds = (0.0, 86400.0),
                                    halo_width = Hp, has_humidity = true)
    end
    O.begin_observation_day!(sampler, "20240615", 1)
    for t in (0.0, DT)
        O.observe_window_boundary!(sampler, state, t; next_window_seconds = DT, humidity = P.qpanels)
    end
    with_logger(NullLogger()) do
        O.finish_observations!(sampler)
    end
    close(sampler)
    return joinpath(dir, "$(label)_soundings.nc"), grid
end

@testset "forward sampler: averaging-kernel columns == the 4D-Var's (nearest window)" begin
    # No transport: the state after step 1 is the initial state, which the
    # sampler samples at the 600 s window end (records at 450 s).
    P = hydrostatic_cs_problem(; flux_fraction = 0.0, nsteps = 2)
    site = _site(P)
    mktempdir() do dir
        recs = [(; id = 2024061500073016 + 10k, time = T0 + Second(450), site..., xco2 = 404.0 + k, unc = 0.6,
                 model_error = 0.3, psurf, altitude = 100.0, data_type = 6, assimilate_flag = 1,
                 oco2_fixture_retrieval(psurf; tilt = 0.1k)...)
                for (k, psurf) in enumerate((985.0, 1004.0, 930.0))]
        mip = write_oco2_mip_fixture(joinpath(dir, "OCO2_b11.2_10sec_GOOD_r2.nc4"), recs)
        plain, _ = _sample_file(P, mip, dir, "plain")
        hat, grid = _sample_file(P, mip, dir, "hat"; averaging_kernel = Dict("averaging_kernel_tracers" => ["co2"]))
        tm5, _ = _sample_file(P, mip, dir, "tm5"; averaging_kernel = Dict("averaging_kernel_tracers" => ["co2"],
                                                                          "averaging_kernel_mapping" => "tm5"))
        read_ak(path) = NCDataset(path) do ds
            Dict(id => v for (id, v) in zip(ds["id"][:], ds["co2_ak_column"][:]))
        end
        zero_controls = [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:all, 1:P.nsteps),
                                                 ntuple(_ -> zeros(P.Nc, P.Nc), 6))]
        met = column_met(P; gravity = AT.gravity(grid))
        for (path, mapping) in ((hat, O.HatLevelProjection()), (tm5, O.LayerRebin()))
            set = AT.read_oco2_mip_observations(mip; mapping)
            obs = AT.bind_to_mesh(set, P.mesh, T0, DT; nsteps = P.nsteps)
            result = _fourdvar(P, obs, zero_controls; met)
            sampled = read_ak(path)
            @test length(sampled) == 3
            for (k, o) in enumerate(obs)
                @test result.stencils[k].steps == [1]
                @test sampled[string(o.id)] ≈ result.simulated[k] rtol = 1e-13
            end
            # The kernel matters: not the air-mass-weighted column mean.
            means = NCDataset(path) do ds
                Dict(id => Float64(v) for (id, v) in zip(ds["id"][:], ds["co2_column_mean"][:]))
            end
            @test all(abs(sampled[string(o.id)] - means[string(o.id)]) > 1e-8 for o in obs)
        end
        @test read_ak(hat) != read_ak(tm5)
        # Without the option the file is unchanged: no averaging-kernel
        # variable and the same values as with it.
        NCDataset(plain) do p
            NCDataset(hat) do h
                @test !haskey(p, "co2_ak_column") && !haskey(p.attrib, "averaging_kernel_columns")
                @test sort(setdiff(collect(keys(h)), collect(keys(p)))) == ["co2_ak_column"]
                for name in keys(p)
                    @test isequal(Array(p[name]), Array(h[name]))
                end
            end
        end
        # Configuration checks.
        @test_throws ArgumentError _sample_file(P, mip, dir, "bad";
                                                averaging_kernel = Dict("averaging_kernel_tracers" => ["ch4"]))
        @test_throws ArgumentError O.observation_source_from_cfg(
            Dict{String, Any}("kind" => "oco2_lite", "path" => mip, "averaging_kernel_mapping" => "tm5"), "src")
        @test_throws ArgumentError O.observation_source_from_cfg(
            Dict{String, Any}("kind" => "oco2_lite", "path" => mip, "averaging_kernel_tracers" => ["co2"],
                              "averaging_kernel_mapping" => "midpoint"), "src")
    end
end
