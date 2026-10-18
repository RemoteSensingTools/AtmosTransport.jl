#!/usr/bin/env julia
#
# The cubed-sphere adjoint replays the production step plan (owner decision
# D31): `DrivenSimulation` executes its `StepPlan`, the adjoint records and
# transposes the same plan (`CSStepSchedule(sim)`).
#
# * Parity matrix: for every scheme family the adjoint supports, both physics
#   cadences, TM5 / CMFMC / CMFMC-matrix convection, both diffusion couplings,
#   diffusion without surface sources and both air-mass reset modes, the
#   adjoint's forward replay and its tape recording reproduce the production
#   run BIT FOR BIT, and so do the replays of the Stride and Revolve
#   checkpoint drivers (restarts inside met windows).
# * OF-114: with `physics_cadence = :window` (the default) production runs
#   convection once per met window; the replay matches it, and the old
#   per-step composition does not.
# * OF-115: diffusion without a surface-flux operator is one `V(dt)` solve in
#   production; the replay without a surface-flux operator matches it, and the
#   old coupled center `V(dt/2) S V(dt/2)` with `S = 0` does not.
# * Guard: a plan with an operator the tape cannot transpose is an error that
#   names it.
# * Gradients: footprints and the initial-condition sensitivity against
#   central finite differences of the production DrivenSimulation.
#
# Bit-for-bit assertions use `==` in one thread; with several threads the
# halo and Lin-Rood adjoint kernels accumulate with atomics, so the guard
# `exact` falls back to a 1e-13 relative bound.

using Test
import AtmosTransport
const AT = AtmosTransport
const ADJ = AT.Adjoints
const SP = AT.Operators.StepPlans

const NC = 4
const NZ = 5
const S = 4e15          # kg per cell and layer (≈ 1500 kg m⁻² on C4 cells)
const WINDOW_DT = 3600.0

_close(a::AbstractArray, b::AbstractArray) = isapprox(a, b; rtol = 1e-13)
_close(a::Tuple, b::Tuple) = all(map(_close, a, b))
exact(a, b) = Threads.nthreads() == 1 ? a == b : _close(a, b)

# ---------------------------------------------------------------------------
# A mock cubed-sphere met driver: three windows with different air mass, mass
# fluxes, window mass change, convection forcing and (optionally) surface
# pressure, so the layer thickness of the diffusion changes per window, and
# either flux storage convention.
# ---------------------------------------------------------------------------

struct _PlanDriver{FT, G, W} <: AT.AbstractMetDriver
    grid::G
    windows::Vector{W}
    dt::FT
    steps::Vector{Int}
    contract::Bool
    flux_kind::Symbol
end
AT.total_windows(d::_PlanDriver) = length(d.windows)
AT.window_dt(d::_PlanDriver) = d.dt
AT.steps_per_window(d::_PlanDriver) = d.steps[1]
AT.MetDrivers.steps_per_window_schedule(d::_PlanDriver) = d.steps
AT.load_transport_window(d::_PlanDriver, w::Int) = d.windows[w]
AT.driver_grid(d::_PlanDriver) = d.grid
AT.air_mass_basis(::_PlanDriver) = :dry
AT.MetDrivers.flux_interpolation_mode(::_PlanDriver) = :constant
AT.MetDrivers.supports_native_vertical_flux(::_PlanDriver) = true
AT.MetDrivers.uses_binary_substep_contract(d::_PlanDriver) = d.contract
AT.MetDrivers.flux_kind(d::_PlanDriver) = d.flux_kind
AT.supports_convection(::_PlanDriver) = true

const GRID = AT.AtmosGrid(AT.CubedSphereMesh(Nc = NC, Hp = 3, FT = Float64),
                          AT.HybridSigmaPressure([0.0, 100, 300, 600, 1000, 2000],
                                                 [0.0, 0, 0.1, 0.3, 0.7, 1]),
                          AT.CPU(); FT = Float64)
const MESH = GRID.horizontal
const HP = MESH.Hp
const NP = NC + 2HP

function _tm5_forcing(w)
    mk(lo, hi, v) = ntuple(p -> (e = zeros(NC, NC, NZ); e[:, :, lo:min(hi, NZ - 1)] .= v * (1 + 0.1w); e), 6)
    return AT.ConvectionForcing(nothing, nothing,
        (; entu = mk(2, 4, 0.06), detu = mk(2, 4, 0.04), entd = mk(3, 4, 0.02), detd = mk(3, 4, 0.01)))
end

function _cmfmc_forcing(w)
    cmfmc = ntuple(_ -> (c = zeros(NC, NC, NZ + 1);
                         c[:, :, 2] .= 0.012 * (1 + 0.1w); c[:, :, 3] .= 0.020; c[:, :, 4] .= 0.015; c), 6)
    dtrain = ntuple(_ -> (d = zeros(NC, NC, NZ); d[:, :, 2] .= 0.006; d[:, :, 3] .= 0.005; d), 6)
    return AT.ConvectionForcing(cmfmc, dtrain, nothing)
end

function _window(w, conv; vary_ps = false)
    m = ntuple(p -> [S * (2.0 + 0.25k + 0.01p + 0.02w + 0.03sin(0.7i + 0.3j + p + w))
                     for i in 1:NP, j in 1:NP, k in 1:NZ], 6)
    dm = ntuple(p -> [S * 0.002cos(0.4i + 0.2j + 0.3k + p + w) for i in 1:NP, j in 1:NP, k in 1:NZ], 6)
    am = ntuple(p -> (a = zeros(NP + 1, NP, NZ);
        for k in 1:NZ, j in HP+1:HP+NC, i in HP+1:HP+NC+1
            a[i, j, k] = S * 0.05sin(0.2w + 0.3p + 0.7i + 0.4j + 0.2k)
        end; a), 6)
    bm = ntuple(p -> (b = zeros(NP, NP + 1, NZ);
        for k in 1:NZ, j in HP+1:HP+NC+1, i in HP+1:HP+NC
            b[i, j, k] = S * 0.04cos(0.3w + 0.5p + 0.4i + 0.6j + 0.1k)
        end; b), 6)
    cm = ntuple(p -> (c = zeros(NP, NP, NZ + 1);
        for k in 2:NZ, j in HP+1:HP+NC, i in HP+1:HP+NC
            c[i, j, k] = -S * 0.03(1 + 0.3sin(i + j + k + p + w))
        end; c), 6)
    forcing = conv === :tm5 ? _tm5_forcing(w) :
              conv in (:cmfmc, :cmfmc_matrix) ? _cmfmc_forcing(w) : nothing
    ps = vary_ps ? 95_000.0 - 3000.0w : 95_000.0
    return AT.TransportWindow(m, ntuple(_ -> fill(ps, NC, NC), 6),
                              AT.CubedSphereFaceFluxState{AT.DryBasis}(am, bm, cm);
                              deltas = AT.MetDrivers.CubedSphereFluxDeltas(dm), convection = forcing)
end

_driver(; conv = :none, contract = true, steps = [2, 3, 2], vary_ps = false,
        flux_kind = :substep_mass_amount) =
    (ws = [_window(w, conv; vary_ps) for w in eachindex(steps)];
     _PlanDriver{Float64, typeof(GRID), eltype(ws)}(GRID, ws, WINDOW_DT, steps, contract, flux_kind))

const RATES = ntuple(p -> [S * 1e-6 * (1 + 0.3sin(i + 2j + p)) for i in 1:NC, j in 1:NC], 6)

_convection_op(conv) = conv === :tm5 ? AT.TM5Convection() :
                       conv === :cmfmc ? AT.CMFMCConvection() :
                       conv === :cmfmc_matrix ? AT.CMFMCMatrixConvection() : AT.NoConvection()

# Kz [m² s⁻¹] = kz / k on level k. The hydrostatic layers of the mock grid are
# kilometres thick: `kz = 30` diffuses weakly, `kz = 3e4` strongly.
_diffusion_op(d, kz = 30.0) = d === :none ? AT.NoDiffusion() : AT.ImplicitVerticalDiffusion(;
    kz_field = AT.CubedSphereField(ntuple(_ -> AT.ProfileKzField([kz / k for k in 1:NZ]), 6)),
    surface_flux_coupling = d === :split ? AT.SplitSurfaceFluxCoupling() : AT.DiffusiveSurfaceFluxBoundary())

_initial_rm(air) = ntuple(p -> [air[p][i, j, k] * (0.08 + 0.01sin(0.27i + 0.13j + 0.19k + 0.07p))
                                for i in 1:NP, j in 1:NP, k in 1:NZ], 6)

# A configuration of the matrix.
Base.@kwdef struct Case
    scheme = AT.UpwindScheme()
    conv::Symbol = :none
    diffusion::Symbol = :none
    emissions::Bool = true
    cadence = nothing
    contract::Bool = true
    reset::Symbol = :preserve_tracer_mass
    steps::Vector{Int} = [2, 3, 2]
    kz::Float64 = 30.0
    vary_ps::Bool = false
    flux_kind::Symbol = :substep_mass_amount
end

_label(c::Case) = "$(nameof(typeof(c.scheme))) $(something(c.cadence, :window)) " *
    "contract=$(c.contract) conv=$(c.conv) diffusion=$(c.diffusion) emissions=$(c.emissions) " *
    "reset=$(c.reset)" * (c.vary_ps ? " per-window ps" : "") *
    (c.flux_kind === :substep_mass_amount ? "" : " $(c.flux_kind)")

# The production simulation of a case (not yet run) and its initial state.
function _simulation(c::Case; rate_scale = 1.0, rm_perturbation = nothing)
    driver = _driver(; conv = c.conv, contract = c.contract, steps = c.steps,
                     vary_ps = c.vary_ps, flux_kind = c.flux_kind)
    air = ntuple(p -> copy(driver.windows[1].air_mass[p]), 6)
    rm = _initial_rm(air)
    rm_perturbation === nothing || (rm = ntuple(p -> rm[p] .+ rm_perturbation[p], 6))
    state = AT.CubedSphereState(AT.DryBasis, MESH, air; CO2 = rm)
    model = AT.TransportModel(state, AT.allocate_face_fluxes(MESH, NZ; basis = AT.DryBasis),
                              GRID, c.scheme; diffusion = _diffusion_op(c.diffusion, c.kz),
                              convection = _convection_op(c.conv))
    sources = c.emissions ? (AT.SurfaceFluxSource(:CO2, ntuple(p -> rate_scale .* RATES[p], 6)),) : ()
    return AT.DrivenSimulation(model, driver; surface_sources = sources,
                               physics_cadence = c.cadence, air_mass_reset_mode = c.reset)
end

_rm(sim) = AT.get_tracer(sim.model.state, :CO2)
_interior(panels) = ntuple(p -> panels[p][HP+1:HP+NC, HP+1:HP+NC, :], 6)

# The adjoint inputs of a case, captured from a second, identical production
# simulation (`cs_adjoint_inputs`): the face fluxes of every step as production
# refreshes them, the convection forcing and diffusion layer thickness of each
# met window, the schedule, and fresh workspaces.
function _adjoint_inputs(c::Case; kw...)
    x = AT.cs_adjoint_inputs(_simulation(c; kw...); tracer = :CO2)
    return (; rm0 = x.panels_rm0, m0 = x.panels_m0, am = x.panels_am_steps,
            bm = x.panels_bm_steps, cm = x.panels_cm_steps, kw = x.kwargs)
end

# The keywords of the footprint entry points (base emissions, no `surface_flux`).
_footprint_kwargs(inp) = (; Base.structdiff(inp.kw, NamedTuple{(:surface_flux, :emission_rates)})...,
                          base_emission_rates = inp.kw.emission_rates)

_replay(inp; kw...) = first(ADJ._cs_replay_for_call(inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH;
                                                     inp.kw..., kw...))

# Final (rm, m) interiors of production, the forward replay, the recording
# replay, and Stride / Revolve-style restarts from a snapshot inside a window.
function _parity(c::Case)
    sim = _simulation(c)
    inp = _adjoint_inputs(c)
    AT.run!(sim)
    production = (_interior(_rm(sim)), _interior(sim.model.state.air_mass))
    n = sum(c.steps)

    forward = _replay(inp)
    ADJ._replay_steps!(forward, 1:n)
    replayed = (_interior(ADJ._replay_rm(forward)), _interior(ADJ._replay_m(forward)))

    recorded = _replay(inp)
    ops = ADJ._record_cs_tape!(recorded, 1:n, AT.DeviceCSTapeStorage())
    taped = (_interior(ADJ._replay_rm(recorded)), _interior(ADJ._replay_m(recorded)))

    # A checkpoint inside the second met window (after its first step), as the
    # Stride and Revolve drivers take them, and a restart from it.
    restart = _replay(inp)
    k = c.steps[1] + 1
    ADJ._replay_steps!(restart, 1:k)
    snap = ADJ._replay_snapshot(restart)
    ADJ._replay_steps!(restart, (k + 1):n)          # disturb the state ...
    ADJ._replay_restore!(restart, snap)             # ... and restore the checkpoint
    ADJ._replay_steps!(restart, (k + 1):n)
    restarted = (_interior(ADJ._replay_rm(restart)), _interior(ADJ._replay_m(restart)))
    return (; production, replayed, taped, restarted, ops)
end

const SCHEMES = (AT.UpwindScheme(), AT.SlopesScheme(AT.NoLimiter()),
                 AT.PPMScheme(AT.MonotoneLimiter()), AT.LinRoodPPMScheme())

# Every scheme family under each operator/cadence pattern.
const PATTERNS = (
    (cadence = nothing,  conv = :tm5,   diffusion = :split,    emissions = true,  reset = :preserve_tracer_mass),
    (cadence = :substep, conv = :cmfmc, diffusion = :boundary, emissions = true,  reset = :preserve_tracer_mass),
    (cadence = nothing,  conv = :cmfmc, diffusion = :split,    emissions = false, reset = :preserve_tracer_mass),
    (cadence = :substep, conv = :tm5,   diffusion = :none,     emissions = true,  reset = :preserve_vmr),
)

const EXTRA_CASES = (
    Case(conv = :cmfmc_matrix, diffusion = :boundary),                       # CMFMC-matrix, :window
    Case(conv = :tm5, diffusion = :split, contract = false),                 # no window contract
    Case(scheme = AT.PPMScheme(AT.CW84Limiter()), conv = :tm5, diffusion = :boundary, reset = :preserve_vmr),
    Case(scheme = AT.LinRoodPPMScheme(5; vertical = AT.UpwindScheme()), conv = :tm5, diffusion = :boundary),
    Case(scheme = AT.PPMScheme(AT.NoLimiter()), conv = :cmfmc, diffusion = :none, reset = :none),
    # Production per-step inputs that change with the window: the diffusion
    # layer thickness (per-window surface pressure) and the full-window flux
    # storage scaled by 1/(2 steps) at each window.
    Case(conv = :tm5, diffusion = :split, vary_ps = true, flux_kind = :full_window_mass_amount),
    Case(scheme = AT.LinRoodPPMScheme(), conv = :cmfmc, diffusion = :boundary, vary_ps = true,
         cadence = :substep),
)

@testset "adjoint replay == production, bit for bit (D31 parity matrix)" begin
    cases = [[Case(; scheme, p...) for scheme in SCHEMES for p in PATTERNS]; collect(EXTRA_CASES)]
    @testset "$(_label(c))" for c in cases
        r = _parity(c)
        @test exact(r.replayed[1], r.production[1]) && exact(r.replayed[2], r.production[2])
        @test exact(r.taped[1], r.production[1]) && exact(r.taped[2], r.production[2])
        @test exact(r.restarted[1], r.production[1]) && exact(r.restarted[2], r.production[2])
        # The tape holds one end-of-step marker per step and, at the plan's
        # cadence, the convection records.
        n = sum(c.steps)
        @test count(op -> op isa AT.Tape._CSStepEndRecord, r.ops) == n
        nconv = c.conv === :none ? 0 :
            (something(c.cadence, :window) === :window && c.contract ? length(c.steps) : n)
        @test count(op -> op isa AT.Tape._CSConvectionRecord, r.ops) == nconv
        @test count(op -> op isa AT.Tape._CSMidpointRecord, r.ops) == (c.emissions ? n : 0)
    end
end

@testset "Full, Stride and Revolve footprints agree across met windows" begin
    for c in (Case(conv = :tm5, diffusion = :split),
              Case(scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :cmfmc, diffusion = :boundary,
                   reset = :preserve_vmr),
              Case(scheme = AT.LinRoodPPMScheme(), conv = :tm5, diffusion = :split, cadence = :substep))
        inp = _adjoint_inputs(c)
        objective = AT.CSLayerMeanObjective(2, 2, 3, NZ)
        footprint(checkpoint) = AT.cs_surface_emission_footprint(
            inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH, objective;
            _footprint_kwargs(inp)..., checkpoint)
        full = footprint(AT.FullCheckpoint())
        for checkpoint in (AT.StrideCheckpoint(2), AT.StrideCheckpoint(3), AT.RevolveCheckpoint())
            other = footprint(checkpoint)
            @test all(exact(other.footprints[s][p], full.footprints[s][p])
                      for s in eachindex(full.footprints), p in 1:6)
            @test all(exact(other.initial_adjoint[p], full.initial_adjoint[p]) for p in 1:6)
        end
    end
end

# ---------------------------------------------------------------------------
# OF-114: physics_cadence = :window
# ---------------------------------------------------------------------------

@testset "OF-114: :window cadence convection once per met window" begin
    c = Case(conv = :tm5, diffusion = :split)
    sim = _simulation(c)
    @test sim.plan.cadence isa SP.WindowPhysics
    @test sim.plan.step == (SP.TransportStage(),)
    inp = _adjoint_inputs(c)
    AT.run!(sim)
    production = _interior(_rm(sim))
    replay = _replay(inp)
    ADJ._replay_steps!(replay, 1:sum(c.steps))
    @test exact(_interior(ADJ._replay_rm(replay)), production)
    # The per-step composition the adjoint applied before D31 (convection every
    # step) differs from the production run by far more than rounding.
    substep = ADJ.CSStepSchedule(; physics_cadence = :substep, binary_window_contract = true,
                                 air_mass_reset = :preserve_tracer_mass, steps_per_window = c.steps,
                                 window_dt = WINDOW_DT,
                                 window_start_air_mass = ADJ.CSStepSchedule(sim).window_start_air_mass,
                                 window_end_air_mass = ADJ.CSStepSchedule(sim).window_end_air_mass)
    old = _replay(inp; schedule = substep)
    ADJ._replay_steps!(old, 1:sum(c.steps))
    rel = maximum(maximum(abs.(_interior(ADJ._replay_rm(old))[p] .- production[p]) ./
                          abs.(production[p])) for p in 1:6)
    @test rel > 1e-6
end

# ---------------------------------------------------------------------------
# OF-115: diffusion without surface sources
# ---------------------------------------------------------------------------

@testset "OF-115: no-emission diffusion is V(dt)" begin
    c = Case(diffusion = :split, emissions = false, kz = 3e4)
    sim = _simulation(c)
    inp = _adjoint_inputs(c)
    AT.run!(sim)
    production = _interior(_rm(sim))
    replay = _replay(inp)
    ADJ._replay_steps!(replay, 1:sum(c.steps))
    @test exact(_interior(ADJ._replay_rm(replay)), production)
    # The coupled center V(dt/2) S V(dt/2) with S = 0 is not production's V(dt).
    coupled = _replay(inp; surface_flux = true)
    ADJ._replay_steps!(coupled, 1:sum(c.steps))
    rel = maximum(maximum(abs.(_interior(ADJ._replay_rm(coupled))[p] .- production[p]) ./
                          abs.(production[p])) for p in 1:6)
    @test rel > 1e-3
end

# ---------------------------------------------------------------------------
# Guard: every operator of the plan has an adjoint, or setup fails naming it
# ---------------------------------------------------------------------------

struct _NoAdjointConvection <: AT.AbstractConvection end
struct _NoAdjointStage <: SP.AbstractStepStage end

@testset "adjoint setup guard" begin
    supported = _simulation(Case(conv = :tm5, diffusion = :split))
    @test AT.require_adjoint_support(supported) === supported
    @test ADJ.adjoint_support(AT.TM5Convection()) === nothing

    message(f) = try f(); "" catch err; err isa ArgumentError ? err.msg : rethrow() end
    decay = AT.DrivenSimulation(supported.model, supported.driver;
                                chemistry = AT.ExponentialDecay(; Rn222 = 330_350.4))
    @test occursin("ExponentialDecay", message(() -> AT.require_adjoint_support(decay)))

    ops(; kw...) = SP.StepOperators(; state = supported.model.state, fluxes = supported.model.fluxes,
                                    grid = GRID, advection = AT.UpwindScheme(),
                                    advection_workspace = nothing, kw...)
    plan = SP.StepPlan()
    @test occursin("use_collab_lu", message(() -> AT.require_adjoint_support(
        plan, ops(convection = AT.TM5Convection(; use_collab_lu = true)))))
    @test occursin("clamp", message(() -> AT.require_adjoint_support(
        plan, ops(convection = AT.CMFMCConvection(; clamp = true)))))
    @test occursin("_NoAdjointConvection", message(() -> AT.require_adjoint_support(
        plan, ops(convection = _NoAdjointConvection()))))
    @test occursin("SlopesScheme", message(() -> AT.require_adjoint_support(
        plan, ops(advection = AT.SlopesScheme(AT.MonotoneLimiter())))))
    # A new stage without an adjoint record makes every plan containing it fail.
    extended = SP.StepPlan(plan.cadence, plan.window_advance, plan.window_start,
                           (plan.step..., _NoAdjointStage()), plan.window_end)
    @test occursin("_NoAdjointStage", message(() -> AT.require_adjoint_support(extended, ops())))

    # The entry points check the plan before recording anything.
    c = Case(conv = :tm5)
    inp = _adjoint_inputs(c)
    collab = AT.TM5Convection(; use_collab_lu = true)
    @test_throws ArgumentError AT.cs_surface_emission_footprint(
        inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH, AT.CSLayerMeanObjective(1, 1, 1, NZ);
        scheme = c.scheme, schedule = inp.kw.schedule, convection_op = collab,
        convection_forcing = inp.kw.convection_forcing, convection_workspace = inp.kw.convection_workspace)
end

# ---------------------------------------------------------------------------
# Gradients against finite differences of the production simulation
# ---------------------------------------------------------------------------

@testset "gradient == finite differences of production" begin
    objective = AT.CSLayerMeanObjective(3, 2, 2, NZ)
    J(sim) = (AT.run!(sim); ADJ.evaluate_objective(objective, _rm(sim), sim.model.state.air_mass, MESH))
    for (c, h, tol) in ((Case(conv = :tm5, diffusion = :split), 0.5, 1e-9),
                        (Case(conv = :cmfmc, diffusion = :boundary, cadence = :substep), 0.5, 1e-9),
                        (Case(scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :tm5,
                              diffusion = :split, reset = :preserve_vmr), 1e-4, 1e-7),
                        (Case(scheme = AT.LinRoodPPMScheme(), conv = :cmfmc, diffusion = :split), 1e-4, 1e-6))
        inp = _adjoint_inputs(c)
        fp = AT.cs_surface_emission_footprint(inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH, objective;
                                              _footprint_kwargs(inp)...)
        adjoint = sum(sum(fp.footprints[s][p] .* RATES[p]) for s in eachindex(fp.footprints), p in 1:6)
        fd = (J(_simulation(c; rate_scale = 1 + h)) - J(_simulation(c; rate_scale = 1 - h))) / 2h
        @test abs(adjoint - fd) <= tol * abs(fd)
        @info "footprint vs production FD" case = _label(c) rel = abs(adjoint - fd) / abs(fd) maxlog = 4
    end

    # Initial-condition sensitivity of a configuration without surface
    # sources (production center V(dt)).
    c = Case(conv = :tm5, diffusion = :split, emissions = false)
    inp = _adjoint_inputs(c)
    seed = ntuple(p -> zeros(NP, NP, NZ), 6)
    seed[3][HP + 2, HP + 2, NZ] = 1 / inp.m0[3][HP + 2, HP + 2, NZ]   # a rough dJ/drm_final
    kw = Base.structdiff(inp.kw, NamedTuple{(:surface_flux, :emission_rates)})
    result = AT.cs_surface_emission_footprint_from_seed(seed, inp.m0, inp.am, inp.bm, inp.cm, MESH;
                                                        kw..., surface_flux = false)
    @test isempty(result.footprints)
    δ = ntuple(p -> [(HP < i <= HP + NC && HP < j <= HP + NC) ? 1e-3 * inp.rm0[p][i, j, k] * cos(i + j + k + p) : 0.0
                     for i in 1:NP, j in 1:NP, k in 1:NZ], 6)
    Jseed(sim) = (AT.run!(sim); sum(sum(seed[p] .* _rm(sim)[p]) for p in 1:6))
    fd = (Jseed(_simulation(c; rm_perturbation = δ)) -
          Jseed(_simulation(c; rm_perturbation = ntuple(p -> -δ[p], 6)))) / 2
    adjoint = sum(sum(result.initial_adjoint[p] .* δ[p]) for p in 1:6)
    @test abs(adjoint - fd) <= 1e-9 * abs(fd)
    @info "initial-condition sensitivity vs production FD" rel = abs(adjoint - fd) / abs(fd)
end
