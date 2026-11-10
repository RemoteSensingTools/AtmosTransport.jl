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
#   bit production's; shared and per-step copies give the same footprints;
#   the diffusion layer thickness is one copy per met window.
# * Reset targets: with continuous windows a window's start target is the
#   previous window's end target (one array), otherwise a copy; shared and
#   separate targets give the same footprints bit for bit.
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
    Case(diffusion=:tm5_calc_kzz),
    Case(diffusion=:tm5_calc_kzz_boundary),
    Case(diffusion=:tm5_calc_kzz,reset=:none),
    Case(diffusion=:tm5_calc_kzz_boundary,reset=:none),
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
    # Continuous windows: each window's start reset target is the previous
    # window's end target, one shared array (both reset modes).
    Case(conv = :tm5, diffusion = :split, vary_ps = true, continuous = true),
    Case(scheme = AT.LinRoodPPMScheme(), conv = :cmfmc, diffusion = :boundary, reset = :preserve_vmr,
         continuous = true),
    Case(scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :tm5, diffusion = :boundary,
         reset = :preserve_vmr, cadence = :substep, continuous = true),
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

# The met of every window of a replay, and the flux state each step reads.
_window_mets(replay) = [ADJ.Footprint.acquire_window!(replay.source, w).met
                        for w in 1:ADJ.Footprint.window_count(replay.source)]
_step_flux_states(replay) = [ADJ.Footprint._fluxes_at(_window_mets(replay)[replay.steps.window[s]], s)
                             for s in 1:ADJ.Footprint._replay_nsteps(replay)]

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
        production_dz = Any[]
        while sim.iteration < sim.final_iteration
            AT.step!(sim)
            push!(production, map(f -> map(copy, f), (sim.model.fluxes.am, sim.model.fluxes.bm,
                                                      sim.model.fluxes.cm)))
            dws = sim.model.workspace.diffusion_ws
            dws === nothing || push!(production_dz, map(copy, dws.layer_thickness))
        end
        sets = _flux_sets(inp)
        @test all(isequal(sets[s], production[s]) for s in eachindex(sets))
        # The layer thickness production's diffusion read at each step: one
        # copy per met window, shared by the window's steps.
        if c.diffusion !== :none
            dz = inp.kw.diffusion_layer_thickness
            @test all(isequal(dz[s], production_dz[s]) for s in eachindex(dz))
            @test length(unique(objectid, dz)) == length(c.steps)
            @test all((dz[s] === dz[s - 1]) == !(s in cumsum([1; c.steps[1:end-1]])) for s in 2:length(dz))
        end
        # The steps of a window reference one copy; every window has its own.
        window_first = cumsum([1; c.steps[1:end-1]])
        @test all((sets[s] === sets[s - 1]) == !(s in window_first) for s in 2:length(sets))
        @test length(unique(objectid, sets)) == length(c.steps)

        # Shared and per-step copies give the same footprints, also with a
        # `flux_scale` (the replay scales each shared set once).
        objective = AT.CSLayerMeanObjective(2, 2, 3, NZ)
        for flux_scale in (1.0, 0.5)
            replay = _replay(inp; flux_scale)
            fluxes = _step_flux_states(replay)
            # One flux state per met window, held by the window's met.
            @test all(met -> met.fluxes isa AT.CubedSphereFaceFluxState, _window_mets(replay))
            @test length(unique(objectid, fluxes)) == length(c.steps)
            @test all(isequal(fluxes[s].am, map(a -> flux_scale .* a, inp.am[s]))
                      for s in eachindex(inp.am))
            # Without a scale the replay references the captured arrays.
            @test all((fluxes[s].am === inp.am[s]) == isone(flux_scale)
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

    # Fluxes interpolated within the window are rewritten, so copied, every step;
    # the window's met holds one flux state per step.
    c = Case(conv = :tm5)
    inp = _adjoint_inputs(c; interpolate_fluxes_within_window = true)
    @test length(unique(objectid, _flux_sets(inp))) == sum(c.steps)
    @test all(isequal.(_flux_sets(inp), _flux_sets(_adjoint_inputs(c))))
    replay = _replay(inp)
    @test [length(met.fluxes) for met in _window_mets(replay)] == c.steps
    @test all(_step_flux_states(replay)[s].am === inp.am[s] for s in eachindex(inp.am))
end

@testset "the replay reads its met through a resident window source" begin
    FP = ADJ.Footprint
    c = Case(conv = :cmfmc, diffusion = :split, reset = :preserve_vmr, steps = [3, 2, 2])
    inp = _adjoint_inputs(c)
    replay = _replay(inp)
    source = replay.source
    @test source isa FP.ResidentCSWindowSource
    @test FP.window_count(source) == length(c.steps)
    first_steps = cumsum([1; c.steps[1:end-1]])
    for (w, met) in enumerate(_window_mets(replay))
        s = first_steps[w]
        @test met.window == w && met.steps == s:(s + c.steps[w] - 1)
        @test met.convection_forcing === inp.kw.convection_forcing[s]
        @test met.diffusion_op === inp.kw.diffusion_op[s]
        @test met.layer_thickness === inp.kw.diffusion_layer_thickness[s]
        @test met.start_air_mass === inp.kw.schedule.window_start_air_mass[w]
        @test met.end_air_mass === inp.kw.schedule.window_end_air_mass[w]
    end
    @test_throws ArgumentError FP.acquire_window!(source, length(c.steps) + 1)

    # An untaped run holds only its current window; a recording run retains
    # every window it reads until the driver releases them.
    held(r) = [(h.lease.met.window, h.retained) for h in r.leases]
    FP._replay_steps!(replay, 1:sum(c.steps))
    @test held(replay) == [(length(c.steps), false)]
    FP._release_replay_windows!(replay)
    @test isempty(replay.leases)
    recorded = _replay(inp)
    FP._replay_steps!(recorded, 1:(first_steps[2] - 1))
    counter = FP._CSCountingRecorder(AT.StateTape(), recorded.advection)
    FP._replay_steps!(recorded, first_steps[2]:sum(c.steps); recorder = counter)
    @test held(recorded) == [(w, true) for w in 2:length(c.steps)]
    FP._release_replay_windows!(recorded)
    @test isempty(recorded.leases)

    # Window inputs change only at a window's first step: one forcing, one
    # diffusion operator and one layer thickness per window.
    for (key, msg) in ((:convection_forcing, "convection_forcing changes at step 3 inside met window 1"),
                       (:diffusion_op, "diffusion_op changes at step 3 inside met window 1"),
                       (:diffusion_layer_thickness, "diffusion_layer_thickness changes at step 3 inside met window 1"))
        v = getproperty(inp.kw, key)
        bad = copy(v)
        bad[3] = v[4]                            # the last step of window 1 takes window 2's
        err = try
            _replay(inp; NamedTuple{(key,)}((bad,))...); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin(msg, err.msg)
    end

    # Each window is validated when the replay is built: reset targets have
    # six panels of the initial air mass's shapes, the layer thickness the
    # replay workspace's.
    sched = inp.kw.schedule
    with_start(targets) = AT.CSStepSchedule(sched.plan; steps_per_window = sched.steps_per_window,
                                            window_dt = sched.window_dt, start_time = sched.start_time,
                                            binary_substeps = sched.binary_substeps,
                                            chemistry = sched.chemistry,
                                            window_start_air_mass = targets,
                                            window_end_air_mass = sched.window_end_air_mass)
    shrink(p) = p[1:end-1, ntuple(_ -> Colon(), ndims(p) - 1)...]
    five = Any[sched.window_start_air_mass...]               # a vector that also takes five panels
    five[2] = five[2][1:5]
    @test_throws "require six panels" _replay(inp; schedule = with_start(five))
    misshaped = copy(sched.window_start_air_mass)
    misshaped[2] = (shrink(misshaped[2][1]), misshaped[2][2:6]...)
    @test_throws "must match the initial air-mass panel shapes" _replay(inp; schedule = with_start(misshaped))
    dz = copy(inp.kw.diffusion_layer_thickness)
    small = map(shrink, dz[1])
    dz[1:c.steps[1]] .= Ref(small)
    @test_throws "diffusion_layer_thickness at step 1" _replay(inp; diffusion_layer_thickness = dz)
end

@testset "Full and Stride footprints agree across met windows" begin
    for c in (Case(diffusion=:tm5_calc_kzz),
              Case(diffusion=:tm5_calc_kzz_boundary,reset=:none),
              Case(conv = :tm5, diffusion = :split),
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
# Reset targets: a window's start target is the previous window's end target
# (one array) exactly when the two are equal bit for bit
# ---------------------------------------------------------------------------

@testset "reset targets shared only when equal bit for bit" begin
    RU = AT.Models.Runner
    a = ntuple(p -> [0.5p + 0.25i + j + k for i in 1:NP, j in 1:NP, k in 1:NZ], 6)
    @test RU._bitwise_equal(a, map(copy, a))
    b = map(copy, a); b[4][2, 3, 1] = nextfloat(b[4][2, 3, 1])
    @test !RU._bitwise_equal(a, b)
    z = map(x -> zero(x), a); nz = map(x -> fill!(similar(x), -0.0), a)
    @test !RU._bitwise_equal(z, nz)                     # -0.0 and 0.0 differ in their bits
    @test !RU._bitwise_equal(a, map(x -> Float32.(x), a))

    objective = AT.CSLayerMeanObjective(2, 2, 3, NZ)
    for c in (Case(conv = :tm5, diffusion = :split, continuous = true),
              Case(scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :cmfmc, diffusion = :boundary,
                   reset = :preserve_vmr, continuous = true),
              Case(conv = :tm5, diffusion = :split))
        inp = _adjoint_inputs(c)
        starts, ends = inp.kw.schedule.window_start_air_mass, inp.kw.schedule.window_end_air_mass
        nw = length(c.steps)
        @test length(starts) == length(ends) == nw
        @test length(unique(objectid, ends)) == nw
        @test all((starts[w] === ends[w - 1]) == c.continuous for w in 2:nw)
        @test all(starts[1] !== e for e in ends)
        c.continuous && @test all(isequal(starts[w], ends[w - 1]) for w in 2:nw)
        # Shared targets give the footprints of separate copies, bit for bit
        # (Full and Stride; the resets refresh target halos in place). Each
        # comparison starts from a fresh capture and copies taken before any
        # forward run.
        separate(targets) = [map(copy, t) for t in targets]
        function footprint(inp, starts, ends, checkpoint)
            s = inp.kw.schedule
            schedule = ADJ.CSStepSchedule(s.plan; steps_per_window = c.steps, window_dt = WINDOW_DT,
                start_time = s.start_time, binary_substeps = s.binary_substeps,
                chemistry = s.chemistry, window_start_air_mass = starts, window_end_air_mass = ends)
            return AT.cs_surface_emission_footprint(inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH,
                objective; _footprint_kwargs(inp)..., schedule, tape = AT.AdjointTape(; checkpoint))
        end
        for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2))
            fresh = _adjoint_inputs(c)
            s0, e0 = fresh.kw.schedule.window_start_air_mass, fresh.kw.schedule.window_end_air_mass
            s1, e1 = separate(s0), separate(e0)
            shared = footprint(fresh, s0, e0, checkpoint)
            copied = footprint(fresh, s1, e1, checkpoint)
            @test all(exact(shared.footprints[s][p], copied.footprints[s][p])
                      for s in eachindex(copied.footprints), p in 1:6)
            @test all(exact(shared.initial_adjoint[p], copied.initial_adjoint[p]) for p in 1:6)
        end
    end
end

# ---------------------------------------------------------------------------
# Diffusion inputs: the layer thickness a capture keeps, and the reverse
# scratch the replay owns (never read before it is written)
# ---------------------------------------------------------------------------

@testset "Dkg diffusion captures no layer thickness" begin
    inp = _adjoint_inputs(Case(conv = :tm5, diffusion = :dkg))
    # Dkg carries its exchange: no layer thickness is captured.
    @test inp.kw.diffusion_layer_thickness === nothing
end

@testset "reverse diffusion never reads stale scratch: $d" for d in (:split, :tm5_calc_kzz, :dkg)
    FP = ADJ.Footprint
    c = Case(; conv = :tm5, diffusion = d)
    inp = _adjoint_inputs(c)
    starts = [FP._ObjectiveStart(AT.CSLayerMeanObjective(2, 2, 3, NZ))]
    for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2))
        function footprint(poison::Bool)
            r, dt = FP._cs_replay_for_call(inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH;
                                           _footprint_kwargs(inp)...)
            @test r.diffusion_scratch isa NTuple{6, AbstractArray}
            poison && foreach(a -> fill!(a, NaN), r.diffusion_scratch)
            return FP._with_tape_runtime(AT.AdjointTape(; checkpoint)) do runtime
                only(FP._collect_footprints(r, starts, dt, runtime))
            end
        end
        clean, poisoned = footprint(false), footprint(true)
        @test all(p -> all(isfinite, poisoned.initial_adjoint[p]), 1:6)
        @test all(exact(poisoned.footprints[s][p], clean.footprints[s][p])
                  for s in eachindex(clean.footprints), p in 1:6)
        @test all(exact(poisoned.initial_adjoint[p], clean.initial_adjoint[p]) for p in 1:6)
    end
end

@testset "TM5 frozen window D equals production" begin
    for d in (:tm5_calc_kzz,:tm5_calc_kzz_boundary),reset in (:preserve_tracer_mass,:none)
        c=Case(;diffusion=d,reset)
        sim=_simulation(c)
        inp=_adjoint_inputs(c)
        ops=inp.kw.diffusion_op
        for step in eachindex(ops)
            AT.step!(sim)
            field=ops[step].kz_field
            @test isequal(field.dkg,sim.model.diffusion.kz_field.dkg)
            @test field.scratch === nothing && field.metrics === nothing
            # C4 arrays have appreciable fixed object headers. The strict
            # 10% bound at C8 is checked in test_tm5_pbl_cs.jl.
            @test Base.summarysize(field) <= sum(sizeof,field.dkg)+1024
        end
        first_steps=cumsum([1;c.steps[1:end-1]])
        @test all((ops[s]===ops[s-1])==!(s in first_steps) for s in 2:length(ops))
        # D carries its exchange: no layer thickness is captured.
        @test inp.kw.diffusion_layer_thickness === nothing
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
    for (c, h, tol) in ((Case(diffusion=:tm5_calc_kzz),0.1,1e-10),
                        (Case(diffusion=:tm5_calc_kzz_boundary,reset=:none),0.1,1e-10),
                        (Case(conv = :tm5, diffusion = :split), 0.5, 1e-9),
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
