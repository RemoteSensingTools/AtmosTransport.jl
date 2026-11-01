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
#   run BIT FOR BIT, and so do the replays of the Stride checkpoint driver
#   (restarts inside met windows).
# * Capture: `cs_adjoint_inputs` copies the face fluxes each time production
#   rewrites them (once per met window, every step when interpolated), bit for
#   bit production's; shared and per-step copies give the same footprints.
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

include(joinpath(@__DIR__, "..", "helpers", "cs_adjoint_step_plan.jl"))

import Adapt
import KernelAbstractions

# A distinct storage backend exercised entirely on the CPU. Workspaces built
# for state on it must hold no host `Array` (e.g. the mesh's cell areas): a GPU
# kernel would receive such an array as a non-isbits argument.
struct _CaptureArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
_CaptureArray(a::_CaptureArray) = copy(a)
Base.size(a::_CaptureArray) = size(a.data)
Base.getindex(a::_CaptureArray{T, N}, i::Vararg{Int, N}) where {T, N} = a.data[i...]
Base.setindex!(a::_CaptureArray{T, N}, v, i::Vararg{Int, N}) where {T, N} = (a.data[i...] = v)
Base.similar(a::_CaptureArray, ::Type{T}, dims::Dims) where T = _CaptureArray(similar(a.data, T, dims))
Adapt.adapt_storage(::Type{_CaptureArray}, a::Array) = _CaptureArray(copy(a))
KernelAbstractions.get_backend(::_CaptureArray) = KernelAbstractions.CPU()
AT.Architectures._array_adapter_for(::_CaptureArray) = _CaptureArray

# Every array a workspace or forcing holds, through fields, tuples and Refs.
_storage_leaves(x::AbstractArray) = Any[x]
_storage_leaves(x::Union{Tuple, NamedTuple}) = reduce(vcat, map(_storage_leaves, Tuple(x)); init = Any[])
_storage_leaves(x::Base.RefValue) = _storage_leaves(x[])
_storage_leaves(x) = isstructtype(typeof(x)) ?
    reduce(vcat, [_storage_leaves(getfield(x, f)) for f in fieldnames(typeof(x))]; init = Any[]) : Any[]
_cell_metrics(ws::AT.Operators.Convection.CMFMCMatrixWorkspace) = ws.tm5_workspace.cell_metrics
_cell_metrics(ws) = ws.cell_metrics

@testset "convection workspaces follow the model backend: $conv" for conv in (:cmfmc, :tm5, :cmfmc_matrix)
    host = _simulation(Case(; conv, emissions = false, steps = [2]))
    model = Adapt.adapt(_CaptureArray, host.model)
    # The public operator swap builds its workspace from the model's state.
    swapped = AT.with_convection(model, model.convection).workspace.convection_ws
    sim = AT.DrivenSimulation(model, host.driver; prefetch_windows = false)
    inputs = AT.cs_adjoint_inputs(sim)
    ws = inputs.kwargs.convection_workspace
    @test ws !== model.workspace.convection_ws
    for w in (ws, swapped)
        @test all(a -> a isa _CaptureArray, _storage_leaves(w))
        @test all(_cell_metrics(w)[p] == MESH.cell_areas for p in 1:6)
    end
    @test all(a -> a isa _CaptureArray, _storage_leaves(inputs.kwargs.convection_forcing[1]))
    ws isa AT.Operators.Convection.CMFMCWorkspace && @test !ws.cache_valid[]
end

# Final (rm, m) interiors of production, the forward replay, the recording
# replay, and Stride-style restarts from a snapshot inside a window.
function _parity(c::Case)
    sim = _simulation(c)
    inp = _adjoint_inputs(c)
    AT.run!(sim)
    production = (_interior(_rm(sim)), _interior(sim.model.state.air_mass))
    n = sum(c.steps)

    forward = _replay(inp)
    ADJ.Footprint._replay_steps!(forward, 1:n)
    replayed = (_interior(ADJ.Footprint._replay_rm(forward)), _interior(ADJ.Footprint._replay_m(forward)))

    recorded = _replay(inp)
    ops = ADJ.Footprint._record_cs_tape!(recorded, 1:n, AT.StateTape(), begin
        runtime = ADJ.Footprint._TapeRuntime(AT.AdjointTape())
        ADJ.Footprint._prepare_tape!(runtime, recorded, n)
        runtime.pool
    end)
    taped = (_interior(ADJ.Footprint._replay_rm(recorded)), _interior(ADJ.Footprint._replay_m(recorded)))

    # A checkpoint inside the second met window (after its first step), as the
    # Stride driver takes them, and a restart from it.
    restart = _replay(inp)
    k = c.steps[1] + 1
    ADJ.Footprint._replay_steps!(restart, 1:k)
    snap = ADJ.Footprint._replay_snapshot(restart)
    ADJ.Footprint._replay_steps!(restart, (k + 1):n)          # disturb the state ...
    ADJ.Footprint._replay_restore!(restart, snap)             # ... and restore the checkpoint
    ADJ.Footprint._replay_steps!(restart, (k + 1):n)
    restarted = (_interior(ADJ.Footprint._replay_rm(restart)), _interior(ADJ.Footprint._replay_m(restart)))
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
    Case(conv = :tm5, diffusion = :dkg),                                    # conservative Dkg ledger
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

# ---------------------------------------------------------------------------
# Capture: one copy of the face fluxes each time production rewrites them
# ---------------------------------------------------------------------------

_flux_sets(inp) = [(inp.am[s], inp.bm[s], inp.cm[s]) for s in eachindex(inp.am)]
_unshared(v) = [map(copy, x) for x in v]

@testset "cs_adjoint_inputs shares each met window's fluxes" begin
    for c in (Case(scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :cmfmc, diffusion = :split,
                   cadence = :substep, contract = false, vary_ps = true,
                   flux_kind = :full_window_mass_amount),
              Case(conv = :tm5, diffusion = :boundary, reset = :preserve_vmr))
        inp = _adjoint_inputs(c)
        # The fluxes each production step read (nothing writes them between
        # the step and the next refresh).
        sim = _simulation(c)
        production = Any[]
        while sim.iteration < sim.final_iteration
            AT.step!(sim)
            push!(production, map(f -> map(copy, f), (sim.model.fluxes.am, sim.model.fluxes.bm,
                                                      sim.model.fluxes.cm)))
        end
        sets = _flux_sets(inp)
        @test all(isequal(sets[s], production[s]) for s in eachindex(sets))
        # The steps of a window reference one copy; every window has its own.
        window_first = cumsum([1; c.steps[1:end-1]])
        @test all((sets[s] === sets[s - 1]) == !(s in window_first) for s in 2:length(sets))
        @test length(unique(objectid, sets)) == length(c.steps)

        # Shared and per-step copies give the same footprints, also with a
        # `flux_scale` (the replay scales each shared set once).
        objective = AT.CSLayerMeanObjective(2, 2, 3, NZ)
        for flux_scale in (1.0, 0.5)
            replay = _replay(inp; flux_scale)
            @test length(unique(objectid, replay.fluxes)) == length(c.steps)
            @test all(isequal(replay.fluxes[s].am, map(a -> flux_scale .* a, inp.am[s]))
                      for s in eachindex(inp.am))
            # Without a scale the replay references the captured arrays.
            @test all((replay.fluxes[s].am === inp.am[s]) == isone(flux_scale)
                      for s in eachindex(inp.am))
            footprint(am, bm, cm) = AT.cs_surface_emission_footprint(
                inp.rm0, inp.m0, am, bm, cm, MESH, objective; _footprint_kwargs(inp)..., flux_scale)
            shared = footprint(inp.am, inp.bm, inp.cm)
            copied = footprint(_unshared(inp.am), _unshared(inp.bm), _unshared(inp.cm))
            @test all(exact(shared.footprints[s][p], copied.footprints[s][p])
                      for s in eachindex(copied.footprints), p in 1:6)
            @test all(exact(shared.initial_adjoint[p], copied.initial_adjoint[p]) for p in 1:6)
        end
    end

    # Fluxes interpolated within the window are rewritten, so copied, every step.
    c = Case(conv = :tm5)
    inp = _adjoint_inputs(c; interpolate_fluxes_within_window = true)
    @test length(unique(objectid, _flux_sets(inp))) == sum(c.steps)
    @test all(isequal.(_flux_sets(inp), _flux_sets(_adjoint_inputs(c))))
end

@testset "Full and Stride footprints agree across met windows" begin
    for c in (Case(conv = :tm5, diffusion = :split),
              Case(scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :cmfmc, diffusion = :boundary,
                   reset = :preserve_vmr),
              Case(scheme = AT.LinRoodPPMScheme(), conv = :tm5, diffusion = :split, cadence = :substep),
              Case(scheme = AT.PPMScheme(AT.CW84Limiter()), conv = :tm5, diffusion = :none))
        inp = _adjoint_inputs(c)
        objective = AT.CSLayerMeanObjective(2, 2, 3, NZ)
        footprint(checkpoint) = AT.cs_surface_emission_footprint(
            inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH, objective;
            _footprint_kwargs(inp)..., tape = AT.AdjointTape(; checkpoint))
        full = footprint(AT.FullCheckpoint())
        for checkpoint in (AT.StrideCheckpoint(1), AT.StrideCheckpoint(2), AT.StrideCheckpoint(3))
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
    ADJ.Footprint._replay_steps!(replay, 1:sum(c.steps))
    @test exact(_interior(ADJ.Footprint._replay_rm(replay)), production)
    # The per-step composition the adjoint applied before D31 (convection every
    # step) differs from the production run by far more than rounding.
    substep = ADJ.CSStepSchedule(; physics_cadence = :substep, binary_window_contract = true,
                                 air_mass_reset = :preserve_tracer_mass, steps_per_window = c.steps,
                                 window_dt = WINDOW_DT,
                                 window_start_air_mass = ADJ.CSStepSchedule(sim).window_start_air_mass,
                                 window_end_air_mass = ADJ.CSStepSchedule(sim).window_end_air_mass)
    old = _replay(inp; schedule = substep)
    ADJ.Footprint._replay_steps!(old, 1:sum(c.steps))
    rel = maximum(maximum(abs.(_interior(ADJ.Footprint._replay_rm(old))[p] .- production[p]) ./
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
    ADJ.Footprint._replay_steps!(replay, 1:sum(c.steps))
    @test exact(_interior(ADJ.Footprint._replay_rm(replay)), production)
    # The coupled center V(dt/2) S V(dt/2) with S = 0 is not production's V(dt).
    coupled = _replay(inp; surface_flux = true)
    ADJ.Footprint._replay_steps!(coupled, 1:sum(c.steps))
    rel = maximum(maximum(abs.(_interior(ADJ.Footprint._replay_rm(coupled))[p] .- production[p]) ./
                          abs.(production[p])) for p in 1:6)
    @test rel > 1e-3
end

# ---------------------------------------------------------------------------
# Guard: every operator of the plan has an adjoint, or setup fails naming it
# ---------------------------------------------------------------------------

struct _NoAdjointConvection <: AT.AbstractConvection end
struct _NoAdjointStage <: SP.AbstractStepStage end

@testset "adjoint setup guard" begin
    style = AT.Models.CubedSphereRuntimeRecipeStyle()
    context = AT.Models.InversionAdvectionContext()
    scheme_of(section) = AT.Models.materialize(AT.Models.advection_spec(section, context), style)
    for section in (Dict("scheme" => "upwind"),
                    Dict("scheme" => "ppm"),
                    Dict("scheme" => "ppm", "limiter" => "cw84"),
                    Dict("scheme" => "ppm", "limiter" => "none"),
                    Dict("scheme" => "linrood", "vertical" => "upwind"),
                    Dict("scheme" => "linrood", "ppm_order" => 7,
                         "vertical" => "upwind"),
                    Dict("scheme" => "linrood"),
                    Dict("scheme" => "linrood", "vertical" => "fv3_kord8"),
                    Dict("scheme" => "linrood", "ppm_order" => 7,
                         "vertical" => "fv3_kord8_signed"))
        scheme = scheme_of(section)
        @test AT.require_adjoint_support(scheme) === scheme
    end
    for section in (Dict("scheme" => "slopes"),
                    Dict("scheme" => "ppm", "vertical" => "fv3_kord8"),
                    Dict("scheme" => "ppm", "vertical" => "fv3_kord8_signed"),
                    Dict("scheme" => "none"))
        @test_throws ArgumentError AT.require_adjoint_support(scheme_of(section))
    end
    # Schemes outside the runtime vocabulary follow the same rule.
    @test AT.require_adjoint_support(AT.SlopesScheme(AT.NoLimiter())) == AT.SlopesScheme(AT.NoLimiter())
    @test_throws ArgumentError AT.require_adjoint_support(AT.PPMScheme(AT.Operators.PositivityLimiter()))

    supported = _simulation(Case(conv = :tm5, diffusion = :split))
    @test AT.require_adjoint_support(supported) === supported
    @test ADJ.adjoint_support(AT.TM5Convection()) === nothing

    message(f) = try f(); "" catch err; err isa ArgumentError ? err.msg : rethrow() end
    decay = AT.DrivenSimulation(supported.model, supported.driver;
                                chemistry = AT.ExponentialDecay(; Rn222 = 330_350.4))
    @test occursin("ExponentialDecay", message(() -> AT.require_adjoint_support(decay)))
    @test occursin("ExponentialDecay", message(() -> ADJ.CSStepSchedule(decay)))
    @test occursin("ExponentialDecay", message(() -> AT.cs_adjoint_inputs(decay)))

    ops(; kw...) = SP.StepOperators(; state = supported.model.state, fluxes = supported.model.fluxes,
                                    grid = GRID, advection = AT.UpwindScheme(),
                                    advection_workspace = nothing, kw...)
    plan = SP.StepPlan()
    for convection in (AT.CMFMCConvection(clamp = true),
                       AT.TM5Convection(use_collab_lu = true), AT.TM5Convection(lmax_conv = 3),
                       AT.TM5Convection(n_merge = 2), AT.CMFMCMatrixConvection(use_collab_lu = true))
        @test_throws ArgumentError AT.require_adjoint_support(plan, ops(; convection))
    end
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

    # Even a manually constructed schedule retains chemistry for the replay
    # guard, rather than substituting NoChemistry at operator construction.
    inp_decay = _adjoint_inputs(Case())
    schedule_decay = ADJ.CSStepSchedule(; window_dt = WINDOW_DT,
        chemistry = AT.ExponentialDecay(; CO2 = 330_350.4))
    @test occursin("ExponentialDecay", message(() -> AT.cs_surface_emission_footprint(
        inp_decay.rm0, inp_decay.m0, inp_decay.am, inp_decay.bm, inp_decay.cm, MESH,
        AT.CSLayerMeanObjective(1, 1, 1, NZ); scheme = AT.UpwindScheme(), schedule = schedule_decay)))

    # The entry points check the plan before recording anything.
    c = Case(conv = :tm5)
    inp = _adjoint_inputs(c)
    collab = AT.TM5Convection(; use_collab_lu = true)
    @test_throws ArgumentError AT.cs_surface_emission_footprint(
        inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH, AT.CSLayerMeanObjective(1, 1, 1, NZ);
        scheme = c.scheme, schedule = inp.kw.schedule, convection_op = collab,
        convection_forcing = inp.kw.convection_forcing, convection_workspace = inp.kw.convection_workspace)
end

@testset "capture rejects continued persistent operator state" begin
    message(f) = try f(); "" catch err; err isa ArgumentError ? err.msg : rethrow() end
    for (label, c, contaminate!) in (
        ("Kahan compensation", Case(), sim -> fill!(sim.model.emissions.flux_map.sources[1].compensation[2], 1.0)),
        ("diffusion carry", Case(diffusion = :split), sim -> fill!(sim.model.workspace.diffusion_ws.ledger_carry[2], 1.0)),
        ("Dkg carry", Case(diffusion = :dkg), sim -> fill!(sim.model.workspace.diffusion_ws.ledger_carry[2], 1.0)),
        ("TM5 carry", Case(conv = :tm5), sim ->
            (sim.model.workspace.convection_ws.ledger_carry = ntuple(_ -> ones(NC, NC, 1), 6))),
        ("CMFMC carry", Case(conv = :cmfmc), sim ->
            (sim.model.workspace.convection_ws.ledger_carry[] = ntuple(_ -> ones(NC, NC, 1), 6))),
        ("CMFMC CFL cache", Case(conv = :cmfmc), sim ->
            (sim.model.workspace.convection_ws.cache_valid[] = true)),
        ("CMFMC matrix carry", Case(conv = :cmfmc_matrix), sim ->
            (sim.model.workspace.convection_ws.tm5_workspace.ledger_carry = ntuple(_ -> ones(NC, NC, 1), 6))),
    )
        @testset "$label" begin
            fresh = _simulation(c)
            @test AT.cs_adjoint_inputs(fresh).kwargs.schedule isa ADJ.CSStepSchedule
            continued = _simulation(c)
            contaminate!(continued)
            @test continued.iteration == 0
            @test occursin("persistent operator state", message(() -> AT.cs_adjoint_inputs(continued)))
            @test continued.iteration == 0 # reject before consuming the simulation
        end
    end
end

# ---------------------------------------------------------------------------
# Gradients against finite differences of the production simulation
# ---------------------------------------------------------------------------

@testset "gradient == finite differences of production" begin
    objective = AT.CSLayerMeanObjective(3, 2, 2, NZ)
    J(sim) = (AT.run!(sim); ADJ.evaluate_objective(objective, _rm(sim), sim.model.state.air_mass, MESH))
    for (c, h, tol) in ((Case(conv = :tm5, diffusion = :split), 0.5, 1e-9),
                        (Case(conv = :tm5, diffusion = :boundary), 0.5, 1e-9),
                        (Case(scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :cmfmc,
                              diffusion = :boundary), 1e-4, 1e-7),
                        (Case(scheme = AT.LinRoodPPMScheme(), conv = :tm5,
                              diffusion = :boundary), 1e-4, 1e-7),
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
