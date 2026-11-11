#!/usr/bin/env julia
#
# Streamed met windows (`HostCSWindowSource`): every window in host memory, a
# few device slots refilled as the replay moves. The adjoint must not see the
# difference:
#
# * Exactness: footprints, the Jacobian, the 4D-Var cost, simulated values and
#   gradient, and a two-iteration optimization (one source reused by every
#   evaluation) equal the resident inputs' bit for bit (one thread), with two
#   and three device slots and `MetWindowCheckpoint` plain, split, with device
#   snapshots and AutoTape-constrained. The cases have variable 9/10-step met
#   windows, both reset modes, CFL subcycles and the binary contract,
#   split-sweep and Lin–Rood schemes, direct per-step fluxes (one flux state
#   per step in a window) and production captures (`cs_adjoint_inputs(sim;
#   residency = :host)`, one shared state per window). Observations span
#   windows and segments; a truncated set ends inside a window, and a
#   surface-pressure screen shortens the reverse pass into an early segment.
# * Sizing: a Lin–Rood tape sized from window 1 and the longest later window
#   equals the count over every step.
# * Lifetime: a lease whose slot is refilled is stale (`StaleWindowSlotError`),
#   slots with leases are never refilled, `close` frees them; a failure
#   inside an acquisition, in the forward run or in the reverse pass, leaves
#   no window leased.
# * Guards: Full, Stride (with and without a resident tail), `AirMassTape` and
#   an unconstrained `AutoTape` are an `ArgumentError` before anything runs
#   (no slot is filled); a met source excludes the per-step met arguments, a
#   schedule with reset targets and a mismatched step schedule; the host
#   budget is checked before the run.

using Test
using Logging: NullLogger, with_logger
import AtmosTransport
const AT = AtmosTransport
const F = AT.Adjoints.Footprint
const T = AT.Tape
include(joinpath(@__DIR__, "..", "helpers", "cs_adjoint_tape_harness.jl"))
const H = CSAdjointTapeHarness

# Bit-for-bit in one thread (NaN-aware, -0.0 ≠ 0.0); CPU kernels scatter with
# atomics otherwise.
same(a::Number, b::Number) = Threads.nthreads() == 1 ? isequal(a, b) : isapprox(a, b; rtol = 1e-10)
same(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}) =
    Threads.nthreads() == 1 ? isequal(a, b) : isapprox(a, b; rtol = 1e-10)
same(a::Union{Tuple, AbstractArray}, b::Union{Tuple, AbstractArray}) =
    axes(a) == axes(b) && all(same(x, y) for (x, y) in zip(a, b))
finite(x::AbstractArray{<:Number}) = all(isfinite, x)
finite(x) = all(finite, x)

# The schedule of a problem without reset targets: the source carries them.
_targetless(s) = AT.CSStepSchedule(s.plan; steps_per_window = s.steps_per_window, window_dt = s.window_dt,
                                   start_time = s.start_time, binary_substeps = s.binary_substeps,
                                   chemistry = s.chemistry)

# A host source with the windows of a problem's per-step inputs, each window's
# met copied to the host as the production capture does (`_host_window_met`).
function _host_source(P; device_slots = 2)
    replay, _ = F._cs_replay_for_call(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh; base_emission_rates = P.rates,
                                      H._common(P)...)
    hosts = map(replay.source.windows) do met
        F._host_window_met(met.window, met.steps, P.am[met.steps], P.bm[met.steps], P.cm[met.steps];
                           convection_forcing = met.convection_forcing, diffusion_op = met.diffusion_op,
                           layer_thickness = met.layer_thickness,
                           start_air_mass = F._host_copy(met.start_air_mass),
                           end_air_mass = F._host_copy(met.end_air_mass))
    end
    return AT.HostCSWindowSource(hosts; device_like = first(P.m0), device_slots)
end

# Resident and streamed inputs of a case: the per-step arguments and keywords,
# and `(positional fluxes, keywords)` of a host source with `device_slots`.
function _inputs(c::H.Case; device_slots)
    P = H.problem(c)
    resident = (; am = P.am, bm = P.bm, cm = P.cm, kw = (; H._common(P)..., base_emission_rates = P.rates))
    streamed_kw(source, kw) = (; scheme = c.scheme, schedule = _targetless(P.schedule),
                               Base.structdiff(kw, NamedTuple{(:diffusion_op, :diffusion_layer_thickness,
                                                               :convection_forcing)})...,
                               met_source = source, base_emission_rates = P.rates)
    if c.inputs isa H.CapturedInputs
        x = AT.cs_adjoint_inputs(H._captured_simulation(c, c.inputs); tracer = :CO2, residency = :host,
                                 device_slots)
        @test x.panels_am_steps === nothing && x.kwargs.schedule.window_start_air_mass === nothing
        @test isequal(x.panels_m0, P.m0) && isequal(x.panels_rm0, P.rm0)
        source = x.kwargs.met_source
        kw = (; Base.structdiff(x.kwargs, NamedTuple{(:surface_flux, :emission_rates)})...,
              base_emission_rates = x.kwargs.emission_rates)
        streamed = (; kw = kw)
    else
        source = _host_source(P; device_slots)
        streamed = (; kw = streamed_kw(source, H.physics_kwargs(P)))
    end
    @test source isa AT.HostCSWindowSource && length(source.slots) == device_slots
    @test F.window_steps(source) == c.steps
    return P, resident, streamed, source
end
# The keywords of a call with fresh operator workspaces (they carry scratch).
_fresh(P, inp) = (; inp.kw..., Base.structdiff(H.physics_kwargs(P),
                                               NamedTuple{(:diffusion_op, :diffusion_layer_thickness,
                                                           :convection_forcing)})...)
_fresh_resident(P, inp) = (; inp.kw..., H.physics_kwargs(P)...)

# Observations that span met windows and segments (time windows across a
# window boundary), one inside window 2, and a co-sample at the last step;
# the truncated set ends inside window 2.
function _spanning_observations(P)
    b = cumsum(P.case.steps)
    n, Nz = P.nsteps, P.case.Nz
    full = [AT.CSObservation(AT.CSTimeSampling([b[1], b[1] + 1], [0.4, 0.6]), AT.CSColumnMeanObjective(2, 3, 2), 0.02, 0.01),
            AT.CSObservation(b[1] - 1, AT.CSLayerMeanObjective(5, 1, 4, Nz), -0.01, 0.02),
            AT.CSObservation(AT.CSTimeSampling([b[1] + 1, b[2], b[2] + 1], [0.2, 0.5, 0.3]),
                             AT.CSColumnMeanObjective(6, 4, 4), 0.03, 0.01),
            AT.CSObservation(b[2] - 3, AT.CSLayerMeanObjective(3, 2, 2, Nz), 0.01, 0.01),
            AT.CSObservation(n - 1, AT.CSColumnMeanObjective(4, 1, 3), 0.0, 0.02),
            AT.CSObservation(n, AT.CSColumnMeanObjective(3, 2, 2), 0.0, 0.01; assimilate = false)]
    truncated = [full[1], full[2],
                 AT.CSObservation(b[1] + cld(P.case.steps[2], 2), AT.CSColumnMeanObjective(6, 4, 4), 0.03, 0.01),
                 full[end]]
    return full, truncated
end

# Averaging-kernel columns early in the run and a column the surface-pressure
# screen leaves out at the end of window 2: the tape records through that
# step, the reverse pass starts three steps into window 2.
function _screened_observations(P)
    b = cumsum(P.case.steps)
    return [AT.CSObservation(b[1] - 2, H._column(3, 2, 2, 99000.0), 0.03, 0.01),
            AT.CSObservation(AT.CSTimeSampling([b[1], b[1] + 1], [0.5, 0.5]),
                             H._column(5, 4, 3, 100000.0; hat = true, n = 5, tag = 160), 0.02, 0.01),
            AT.CSObservation(b[1] + 3, H._placed(2, 3, 1; height = 800.0), 0.02, 0.01),
            AT.CSObservation(b[2] - 1, H._column(2, 2, 2, 200000.0; limit = 1000.0, tag = 170), 0.02, 0.01),
            AT.CSObservation(P.nsteps, AT.CSColumnMeanObjective(3, 2, 2), 0.0, 0.01; assimilate = false)]
end

const STREAMED_TAPES = (
    "metwindow" => () -> AT.AdjointTape(; checkpoint = AT.MetWindowCheckpoint()),
    "metwindow split 4, device snapshots" =>
        () -> AT.AdjointTape(; checkpoint = AT.MetWindowCheckpoint(; split_steps = 4,
                                                                     snapshots = AT.DeviceCSTapeStorage())),
    "AutoTape metwindow split 3" => () -> AT.AutoTape(; checkpoint = AT.MetWindowCheckpoint(; split_steps = 3)),
)

const W = [9, 10, 9]
const CASES = [
    H.Case(name = "upwind_cmfmcwin_split_vmr", scheme = AT.UpwindScheme(), conv = :cmfmc, diffusion = :split,
           reset = :preserve_vmr, steps = W, operators = true),
    H.Case(name = "monotone_tm5sub_boundary_mass_cfl", scheme = AT.PPMScheme(AT.MonotoneLimiter()), conv = :tm5,
           diffusion = :boundary, cadence = :substep, contract = false, reset = :preserve_tracer_mass,
           flow = 0.6, steps = W),
    H.Case(name = "linrood5_tm5win_boundary_vmr", scheme = AT.LinRoodPPMScheme(5), conv = :tm5,
           diffusion = :boundary, reset = :preserve_vmr, steps = W),
    H.Case(name = "captured_monotone_cmfmcsub_split_mass_cfl", scheme = AT.PPMScheme(AT.MonotoneLimiter()),
           conv = :cmfmc, diffusion = :split, cadence = :substep, contract = false,
           reset = :preserve_tracer_mass, flow = 2.4, steps = W, inputs = H.CapturedInputs()),
    H.Case(name = "captured_ppm_tm5win_boundary_vmr", scheme = AT.PPMScheme(AT.NoLimiter()), conv = :tm5,
           diffusion = :boundary, reset = :preserve_vmr, steps = W, inputs = H.CapturedInputs()),
]

@testset "streamed windows give the resident results bit for bit: $(c.name)" for c in CASES
    with_logger(NullLogger()) do
        objective = AT.CSLayerMeanObjective(2, 2, 3, c.Nz)
        full, truncated = nothing, nothing
        for device_slots in (2, 3)
            P, res, str, source = _inputs(c; device_slots)
            full, truncated = _spanning_observations(P)
            controls = H._controls(P)
            reference = AT.cs_surface_emission_footprint(P.rm0, P.m0, res.am, res.bm, res.cm, P.mesh, objective;
                                                         _fresh_resident(P, res)..., tape = AT.AdjointTape())
            @test finite(reference.footprints)
            jacobian_windows = [AT.CSSurfaceFluxWindow(:all, 1:P.nsteps),
                                AT.CSSurfaceFluxWindow(:head, 1:3; normalize = true)]
            jacobian = AT.cs_surface_flux_jacobian(P.rm0, P.m0, res.am, res.bm, res.cm, P.mesh,
                                                   H._objectives(P), jacobian_windows;
                                                   _fresh_resident(P, res)..., tape = AT.AdjointTape())
            fourdvar = [AT.cs_surface_flux_4dvar(P.rm0, P.m0, res.am, res.bm, res.cm, P.mesh, obs, controls;
                                                 Base.structdiff(_fresh_resident(P, res),
                                                                 NamedTuple{(:base_emission_rates,)})...,
                                                 tape = AT.AdjointTape())
                        for obs in (full, truncated)]
            @test all(r -> finite(r.gradients) && isfinite(r.cost), fourdvar)
            for (name, tape) in STREAMED_TAPES
                @testset "$device_slots slots, $name" begin
                    r = AT.cs_surface_emission_footprint(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, objective;
                                                         _fresh(P, str)..., tape = tape())
                    @test same(r.footprints, reference.footprints)
                    @test same(r.initial_adjoint, reference.initial_adjoint)
                    for (k, obs) in enumerate((full, truncated))
                        kw = Base.structdiff(_fresh(P, str), NamedTuple{(:base_emission_rates,)})
                        s = AT.cs_surface_flux_4dvar(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, obs, controls;
                                                     kw..., tape = tape())
                        @test same(s.cost, fourdvar[k].cost)
                        @test same(s.simulated, fourdvar[k].simulated)
                        @test same(s.gradients, fourdvar[k].gradients)
                    end
                    @test all(slot -> slot.refcount == 0, source.slots)
                end
            end
            j = AT.cs_surface_flux_jacobian(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, H._objectives(P),
                                            jacobian_windows; _fresh(P, str)...,
                                            tape = AT.AdjointTape(; checkpoint = AT.MetWindowCheckpoint(; split_steps = 4)))
            @test same([x.footprints for x in j.per_step_results], [x.footprints for x in jacobian.per_step_results])
            @test same(j.footprints, jacobian.footprints)
            # With fewer slots than windows, windows were refilled.
            fills = sum(slot -> slot.generation, source.slots)
            @test device_slots < length(c.steps) ? fills > length(c.steps) : fills == length(c.steps)
            close(source)
        end

        # A cost-only evaluation and an optimization whose evaluations reuse one source.
        P, res, str, source = _inputs(c; device_slots = 2)
        controls = H._controls(P)
        reskw = Base.structdiff(_fresh_resident(P, res), NamedTuple{(:base_emission_rates,)})
        strkw() = Base.structdiff(_fresh(P, str), NamedTuple{(:base_emission_rates,)})
        tape = AT.AdjointTape(; checkpoint = AT.MetWindowCheckpoint())
        cost = AT.cs_surface_flux_4dvar(P.rm0, P.m0, res.am, res.bm, res.cm, P.mesh, full, controls; reskw...,
                                        evaluation = AT.CSCostOnly())
        streamed_cost = AT.cs_surface_flux_4dvar(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, full, controls;
                                                 strkw()..., tape, evaluation = AT.CSCostOnly())
        @test same(streamed_cost.cost, cost.cost) && same(streamed_cost.simulated, cost.simulated)
        probe = AT.cs_surface_flux_4dvar(P.rm0, P.m0, res.am, res.bm, res.cm, P.mesh, full, controls; reskw...)
        step = H._norm([x.value for x in controls]) / (10 * H._norm(probe.gradients))
        optimizer() = AT.CSGradientDescent(; iterations = 2, initial_step = step, min_step = step, line_search = false)
        a = AT.cs_surface_flux_4dvar_optimize(P.rm0, P.m0, res.am, res.bm, res.cm, P.mesh, full, controls;
                                              reskw..., optimizer = optimizer(), tape = AT.AdjointTape())
        b = AT.cs_surface_flux_4dvar_optimize(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, full, controls;
                                              strkw()..., optimizer = optimizer(), tape)
        @test length(a.cost_history) == 3
        @test same(b.cost_history, a.cost_history)
        @test same([x.value for x in b.controls], [x.value for x in a.controls])
        @test same(b.last.gradients, a.last.gradients)

        # The surface-pressure screen shortens the reverse pass into an early
        # segment of window 2 (direct inputs: the columns' area is sized for them).
        if c.operators
            observations, column_met = _screened_observations(P), H._column_met(P)
            r = AT.cs_surface_flux_4dvar(P.rm0, P.m0, res.am, res.bm, res.cm, P.mesh, observations,
                                         H._background_controls(P); reskw..., column_met, tape = AT.AdjointTape())
            @test [st.assimilated for st in r.stencils] == [true, true, true, false, false]
            for (_, t) in STREAMED_TAPES
                s = AT.cs_surface_flux_4dvar(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, observations,
                                             H._background_controls(P); strkw()..., column_met, tape = t())
                @test same(s.cost, r.cost) && same(s.simulated, r.simulated) && same(s.gradients, r.gradients)
            end
        end
    end
end

@testset "Lin–Rood sizing from one window equals the count over every step" begin
    for steps in ([9, 10, 9], [2, 3, 2], [3, 1, 4]), contract in (true, false),
        (cadence, reset) in ((:window, :preserve_vmr), (:substep, :preserve_tracer_mass), (:window, :none))
        c = H.Case(; name = "linrood_sizing", scheme = AT.LinRoodPPMScheme(5), conv = :cmfmc, diffusion = :split,
                   cadence, contract, reset, steps)
        P = H.problem(c)
        replay, _ = F._cs_replay_for_call(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh; base_emission_rates = P.rates,
                                          H._common(P)...)
        for n in (P.nsteps, P.nsteps - 2)
            runtime = F._TapeRuntime(AT.AdjointTape())
            counted = F._every_step_tape_size!(runtime, replay, n)
            classes = F._class_tape_size(replay, n)
            if steps == [3, 1, 4]
                @test classes === nothing            # the one-step window is a class the counted windows lack
                continue
            end
            entries(layout) = sort([string((T._layout_key(e), length(e) > 3 ? e[4] : 0)) for e in layout])
            @test [entries(l) for l in classes.layouts] == [entries(l) for l in counted.layouts]
            @test classes.pending == counted.pending
            @test classes.boundary_flush == counted.boundary_flush
        end
    end
end

# A test-only source that fails at its `fail_at`-th acquisition and otherwise
# hands out the windows of `inner`.
mutable struct _FailingSource{S} <: F.AbstractStreamedCSWindowSource
    inner    :: S
    fail_at  :: Int
    acquired :: Int
end
F.window_count(s::_FailingSource) = F.window_count(s.inner)
F.window_steps(s::_FailingSource) = F.window_steps(s.inner)
F.window_layout(s::_FailingSource) = F.window_layout(s.inner)
function F.acquire_window!(s::_FailingSource, w::Int)
    (s.acquired += 1) == s.fail_at && error("injected failure acquiring window $w")
    return F.acquire_window!(s.inner, w)
end
F.release_window!(s::_FailingSource, lease) = F.release_window!(s.inner, lease)

@testset "leases: stale slots, full slots, close and failures inside an acquisition" begin
    c = H.Case(name = "lifetime", scheme = AT.UpwindScheme(), conv = :cmfmc, diffusion = :split,
               reset = :preserve_tracer_mass, steps = [3, 2, 4])
    P = H.problem(c)
    source = _host_source(P; device_slots = 2)
    lease = F.acquire_window!(source, 1)
    @test F.require_current(lease) === lease
    F.release_window!(source, lease)
    for w in (2, 3)                                  # two refills: window 1's slot is the least recently used
        F.release_window!(source, F.acquire_window!(source, w))
    end
    err = try F.require_current(lease); nothing catch e; e end
    @test err isa AT.StaleWindowSlotError && occursin("now holds window 3", sprint(showerror, err))
    @test_throws ArgumentError F.release_window!(source, lease)
    held = [F.acquire_window!(source, w) for w in (1, 2)]
    @test_throws "all 2 device slots" F.acquire_window!(source, 3)
    @test F.acquire_window!(source, 2).slot === held[2].slot          # a held window is shared, not refilled
    F.release_window!(source, held[2])
    @test_throws "leased windows" close(source)
    foreach(l -> F.release_window!(source, l), held)
    @test all(slot -> slot.refcount == 0, source.slots)
    close(source)
    @test_throws AT.StaleWindowSlotError F.require_current(held[1])
    @test_throws "closed" F.acquire_window!(source, 1)
    @test_throws "closed" F.window_layout(source)
    @test_throws ArgumentError F.acquire_window!(_host_source(P), 4)

    # A failure in the forward run (2nd acquisition) or in the reverse pass
    # (the first re-recorded segment after the forward run's 3 windows)
    # leaves no window leased.
    observations, _ = _spanning_observations(P)
    kw = (; scheme = c.scheme, schedule = _targetless(P.schedule),
          Base.structdiff(H.physics_kwargs(P), NamedTuple{(:diffusion_op, :diffusion_layer_thickness,
                                                          :convection_forcing)})...)
    for fail_at in (2, 5)
        inner = _host_source(P)
        failing = _FailingSource(inner, fail_at, 0)
        @test_throws "injected failure" with_logger(NullLogger()) do
            AT.cs_surface_flux_4dvar(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, observations, H._controls(P);
                                     kw..., met_source = failing,
                                     tape = AT.AdjointTape(; checkpoint = AT.MetWindowCheckpoint()))
        end
        @test failing.acquired == fail_at
        @test all(slot -> slot.refcount == 0, inner.slots)
    end
end

@testset "guards: tapes, keywords, schedules, layout and host budget" begin
    c = H.Case(name = "guards", scheme = AT.UpwindScheme(), conv = :tm5, diffusion = :split,
               reset = :preserve_vmr, steps = [3, 2, 4])
    P = H.problem(c)
    objective = AT.CSLayerMeanObjective(2, 2, 3, c.Nz)
    observations, _ = _spanning_observations(P)
    physics = Base.structdiff(H.physics_kwargs(P), NamedTuple{(:diffusion_op, :diffusion_layer_thickness,
                                                              :convection_forcing)})
    kw = (; scheme = c.scheme, schedule = _targetless(P.schedule), physics...)
    rejected = [AT.AdjointTape(), AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(2)),
                AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(2; resident_tail = 3)),
                AT.AdjointTape(; content = AT.AirMassTape()), AT.AutoTape(),
                AT.AutoTape(; checkpoint = AT.StrideSearch()), AT.AutoTape(; checkpoint = AT.FullCheckpoint()),
                AT.AutoTape(; content = AT.AirMassTape())]
    for tape in rejected
        source = _host_source(P)
        footprint = try
            AT.cs_surface_emission_footprint(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, objective;
                                             kw..., met_source = source, base_emission_rates = P.rates, tape)
        catch e
            e
        end
        @test footprint isa ArgumentError && occursin("streamed met source", footprint.msg)
        for evaluation in (AT.CSCostAndGradient(), AT.CSCostOnly())
            @test_throws "streamed met source" AT.cs_surface_flux_4dvar(
                P.rm0, P.m0, nothing, nothing, nothing, P.mesh, observations, H._controls(P);
                kw..., met_source = source, tape, evaluation)
        end
        # Rejected before anything ran: no slot was ever filled.
        @test all(slot -> slot.generation == 0, source.slots)
    end
    @test occursin("MetWindowCheckpoint", sprint(showerror, try
        AT.cs_surface_flux_4dvar(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, observations, H._controls(P);
                                 kw..., met_source = _host_source(P))
    catch e; e end))

    # A met source excludes the per-step met and a schedule with reset targets.
    source = _host_source(P)
    forward(; am = nothing, extra...) = AT.run_cs_footprint_forward(P.rm0, P.m0, am, am, am, P.mesh, objective;
                                                                  kw..., met_source = source, extra...)
    @test isfinite(forward())
    @test_throws "not both" forward(; am = P.am)
    for (key, value) in ((:diffusion_op, H.physics_kwargs(P).diffusion_op),
                         (:diffusion_layer_thickness, H.physics_kwargs(P).diffusion_layer_thickness),
                         (:convection_forcing, H.physics_kwargs(P).convection_forcing), (:flux_scale, 0.5))
        @test_throws "given with `met_source`" forward(; NamedTuple{(key,)}((value,))...)
    end
    @test_throws "carries the air-mass reset targets" forward(; schedule = P.schedule)
    @test_throws "no met" AT.run_cs_footprint_forward(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, objective; kw...)
    s = P.schedule
    shifted = AT.CSStepSchedule(s.plan; steps_per_window = [2, 3, 4], window_dt = s.window_dt,
                                binary_substeps = s.binary_substeps)
    @test_throws "puts step 3 in window 1" forward(; schedule = shifted)
    # A plan that resets needs targets from the schedule or the source.
    @test_throws "resets the air mass" AT.run_cs_footprint_forward(P.rm0, P.m0, P.am, P.bm, P.cm, P.mesh, objective;
                                                                   H._common(P)..., schedule = _targetless(s))
    # A convection operator needs the windows' forcing.
    mets = [h.met for h in source.windows]
    unforced = AT.HostCSWindowSource([F.CSWindowMet(m.window, m.steps, m.fluxes, nothing, m.diffusion_op,
                                                    m.layer_thickness, m.start_air_mass, m.end_air_mass)
                                      for m in mets]; device_like = first(P.m0))
    @test_throws "carry none" AT.run_cs_footprint_forward(P.rm0, P.m0, nothing, nothing, nothing, P.mesh, objective;
                                                          kw..., met_source = unforced)

    # Source construction: numbering, layout, host arrays and budget.
    @test_throws "numbered" AT.HostCSWindowSource(mets[[2, 1, 3]]; device_like = first(P.m0))
    small = F.CSWindowMet(2, mets[2].steps, mets[2].fluxes, mets[2].convection_forcing, mets[2].diffusion_op,
                          map(a -> a[1:end-1, :, :], mets[2].layer_thickness), mets[2].start_air_mass,
                          mets[2].end_air_mass)
    @test_throws "differs in layout" AT.HostCSWindowSource([mets[1], small, mets[3]]; device_like = first(P.m0))
    @test_throws "host memory" AT.HostCSWindowSource(mets; device_like = first(P.m0), host_budget_bytes = 1)
    @test_throws "device_slots" AT.HostCSWindowSource(mets; device_like = first(P.m0), device_slots = 0)
    @test AT.HostCSWindowSource(mets; device_like = first(P.m0)).host_bytes == source.host_bytes

    # The capture: residency keywords, and the budget before the run.
    @test_throws "residency" AT.cs_adjoint_inputs(H._captured_simulation(H.CAPTURED_CASES[2], H.CAPTURED_CASES[2].inputs);
                                                  residency = :disk)
    @test_throws "configure residency = :host" AT.cs_adjoint_inputs(
        H._captured_simulation(H.CAPTURED_CASES[2], H.CAPTURED_CASES[2].inputs); device_slots = 3)
    @test_throws "host memory" AT.cs_adjoint_inputs(
        H._captured_simulation(H.CAPTURED_CASES[2], H.CAPTURED_CASES[2].inputs); residency = :host,
        host_budget_bytes = 4 << 10)
    # A budget above the reset targets but below all windows fails during the
    # capture, from the projection after the first windows.
    captured() = H._captured_simulation(H.CAPTURED_CASES[2], H.CAPTURED_CASES[2].inputs)
    total = AT.cs_adjoint_inputs(captured(); residency = :host).kwargs.met_source.host_bytes
    @test_throws "projected" AT.cs_adjoint_inputs(captured(); residency = :host, host_budget_bytes = total - 1)
    @test AT.cs_adjoint_inputs(captured(); residency = :host, host_budget_bytes = total).kwargs.met_source.host_bytes == total
end
