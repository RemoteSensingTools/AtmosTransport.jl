#!/usr/bin/env julia
#
# Issue #20: the 4D-Var gradient in one reverse sweep.
#
# `cs_surface_flux_4dvar` evaluates `J_obs = ½ Σᵢ ((Hᵢ(x) − yᵢ)/σᵢ)²` and its
# gradient with one forward run, which samples the observations and records
# the tape through the last assimilated observation step, and one reverse
# pass. Each observation is a fixed sparse linear functional of the
# tracer mass after its step (weights from the replayed air mass), and its
# weighted departure (Hᵢ − yᵢ)/σᵢ² times those weights is added to the adjoint
# state at that step's end-of-step marker (`_CSStepEndRecord`, after
# convection), as TM5-4DVAR adds its forcing before the adjoint of the operator
# after which the forward run sampled.
#
# The reference below is the previous algorithm, rebuilt here from public
# calls: one `cs_surface_emission_footprint` per observation, truncated to its
# step, scaled by its weighted departure. In exact arithmetic the two agree
# (the reverse pass is linear in the adjoint state for a fixed tape), so they
# must agree to rounding: rtol 1e-12 in Float64 and 1e-5 in Float32.

using Test
using Logging: NullLogger, with_logger
import AtmosTransport
const AT = AtmosTransport
const ADJ = AT.Adjoints
const TapeMod = AT.Tape
const Adv = AT.Operators.Advection

# ---------------------------------------------------------------------------
# Synthetic problem: smooth air and tracer masses, one set of face mass fluxes
# per step (Courant numbers of a few percent). Cell areas of the convection
# workspaces are 1 m², so the air mass per area is the cell mass (~2.5 kg).
# ---------------------------------------------------------------------------

function _sweep_problem(; FT = Float64, Nc = 4, Nz = 5, nsteps = 4)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    Hp = mesh.Hp
    N = Nc + 2Hp
    m = ntuple(p -> [FT(2.0 + 0.25k + 0.01p + 0.03sin(0.7i + 0.3j + p))
                     for i in 1:N, j in 1:N, k in 1:Nz], 6)
    rm = ntuple(p -> [m[p][i, j, k] * FT(0.08 + 0.01sin(0.27i + 0.13j + 0.19k + 0.07p))
                      for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(m, mesh; dir = 0)
    Adv.fill_panel_halos!(rm, mesh; dir = 0)
    am = [ntuple(p -> begin
              a = zeros(FT, N + 1, N, Nz)
              for k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc+1
                  a[i, j, k] = FT(0.12sin(0.2s + 0.3p + 0.7i + 0.4j + 0.2k))
              end
              a
          end, 6) for s in 1:nsteps]
    bm = [ntuple(p -> begin
              b = zeros(FT, N, N + 1, Nz)
              for k in 1:Nz, j in Hp+1:Hp+Nc+1, i in Hp+1:Hp+Nc
                  b[i, j, k] = FT(0.10cos(0.3s + 0.5p + 0.4i + 0.6j + 0.1k))
              end
              b
          end, 6) for s in 1:nsteps]
    cm = [ntuple(p -> begin
              c = zeros(FT, N, N, Nz + 1)
              for k in 2:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
                  c[i, j, k] = FT(-0.06(1 + 0.3sin(i + j + k + p + s)))
              end
              c
          end, 6) for s in 1:nsteps]
    return (; mesh, m, rm, am, bm, cm, FT, Nc, Nz, nsteps)
end

const DT = 600.0

# Physics keywords, built fresh for every call (workspaces carry scratch).
function _physics(kind::Symbol, P)
    FT, Nc, Nz = P.FT, P.Nc, P.Nz
    kind === :none && return (;)
    if kind === :split || kind === :boundary
        coupling = kind === :split ? AT.SplitSurfaceFluxCoupling() :
                                     AT.DiffusiveSurfaceFluxBoundary()
        op = AT.ImplicitVerticalDiffusion(;
            kz_field = AT.CubedSphereField(ntuple(_ -> AT.PreComputedKzField(repeat(reshape(FT[30 / k for k in 1:Nz], 1, 1, Nz), Nc, Nc, 1)), 6)),
            surface_flux_coupling = coupling)
        ws = AT.DiffusionWorkspace(P.m, P.mesh.Hp, 0)
        for p in 1:6, k in 1:Nz
            ws.layer_thickness[p][:, :, k] .= FT(40 + 15k + 3p)
        end
        return (; diffusion_op = op, diffusion_workspace = ws)
    end
    metrics = ntuple(_ -> ones(FT, Nc, Nc), 6)
    if kind === :tm5
        # About a quarter of a layer entrained per 600 s step.
        mk(lo, hi, v) = ntuple(_ -> (e = zeros(FT, Nc, Nc, Nz);
                                     e[:, :, lo:min(hi, Nz - 1)] .= FT(v); e), 6)
        forcing = AT.ConvectionForcing(nothing, nothing,
            (; entu = mk(2, 4, 1e-3), detu = mk(2, 4, 7e-4),
               entd = mk(3, 4, 3e-4), detd = mk(3, 4, 1.5e-4)))
        return (; convection_op = AT.TM5Convection(), convection_forcing = forcing,
                  convection_workspace = AT.TM5Workspace(P.m; tile_columns = Nc * Nc,
                                                         cell_metrics = metrics))
    end
    kind === :cmfmc || error("unknown physics $kind")
    # About a tenth of a layer moved per step.
    cmfmc = ntuple(_ -> (c = zeros(FT, Nc, Nc, Nz + 1);
                         c[:, :, 2] .= FT(2e-4); c[:, :, 3] .= FT(3e-4); c[:, :, 4] .= FT(2.5e-4); c), 6)
    dtrain = ntuple(_ -> (d = zeros(FT, Nc, Nc, Nz);
                          d[:, :, 2] .= FT(1e-4); d[:, :, 3] .= FT(8e-5); d), 6)
    return (; convection_op = AT.CMFMCConvection(),
              convection_forcing = AT.ConvectionForcing(cmfmc, dtrain, nothing),
              convection_workspace = AT.CMFMCWorkspace(P.m; cell_metrics = metrics))
end

# Controls: two overlapping windows (an average over steps 1-2, and steps
# 2..nsteps with explicit weights) of surface rates of about 1 % of the surface
# tracer mass per step.
function _controls(P)
    FT, Nc = P.FT, P.Nc
    value(phase) = ntuple(p -> [FT(1e-5 * (1 + 0.3sin(phase + i + 2j + p))) for i in 1:Nc, j in 1:Nc], 6)
    late = 2:P.nsteps
    return [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:early, 1:2; normalize = true), value(0.0)),
            AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:late, late;
                                                           weights = [1.0 / s for s in late]),
                                    value(1.3))]
end

# Observations at step 1, at the last step, two in one column at one step
# (a column mean and a layer of the same column), two of one cell at one step,
# and observations at different steps.
function _observations(P; last = P.nsteps)
    FT, Nz = P.FT, P.Nz
    obs(step, objective, value, sigma) = AT.CSObservation(step, objective, FT(value), FT(sigma))
    return [obs(1, AT.CSLayerMeanObjective(1, 2, 3, Nz), 0.083, 0.002),
            obs(last, AT.CSColumnMeanObjective(4, 1, 2), 0.079, 0.001),
            obs(last, AT.CSLayerMeanObjective(4, 1, 2, Nz - 1), 0.081, 0.002),
            obs(2, AT.CSLayerMeanObjective(6, 4, 4, Nz), 0.090, 0.003),
            obs(3, AT.CSLayerMeanObjective(2, 3, 3, Nz), 0.085, 0.002),
            obs(3, AT.CSLayerMeanObjective(2, 3, 3, Nz), 0.075, 0.004),
            obs(3, AT.CSColumnMeanObjective(5, 4, 1), 0.080, 0.001)]
end

_fourdvar(P, observations, controls; scheme, physics = :none, kw...) =
    AT.cs_surface_flux_4dvar(P.rm, P.m, P.am, P.bm, P.cm, P.mesh, observations, controls;
                             scheme, dt = DT, _physics(physics, P)..., kw...)

# The previous algorithm, from public calls: per observation, the simulated
# value and the footprint of a run truncated to its step, scaled by its
# weighted departure; window steps after the observation are skipped.
function _reference_4dvar(P, observations, controls; scheme, physics = :none)
    FT = P.FT
    rates = ADJ.Inversion.Solver._surface_rates_from_controls(controls, P.nsteps, P.mesh, P.rm[1])
    gradients = [ntuple(_ -> zeros(FT, P.Nc, P.Nc), 6) for _ in controls]
    simulated = FT[]
    for obs in observations
        s = obs.step
        truncated = (P.rm, P.m, P.am[1:s], P.bm[1:s], P.cm[1:s], P.mesh, obs.objective)
        value = AT.run_cs_footprint_forward(truncated...; scheme, dt = DT,
                                            emission_rates = rates[1:s],
                                            _physics(physics, P)...)
        push!(simulated, value)
        residual = value - obs.value
        scale = residual / (obs.sigma * obs.sigma)
        footprint = AT.cs_surface_emission_footprint(truncated...; scheme, dt = DT,
                                                     base_emission_rates = rates[1:s],
                                                     _physics(physics, P)...)
        for (c, control) in enumerate(controls)
            for (idx, step) in enumerate(control.window.steps)
                step <= s || continue
                weight = FT(scale) * FT(control.window.weights[idx])
                for p in 1:6
                    gradients[c][p] .+= weight .* footprint.footprints[step][p]
                end
            end
        end
    end
    return (; simulated, gradients)
end

# Largest difference relative to the largest reference entry, over all
# controls and panels.
_relerr(g, ref) = maximum(maximum(abs.(g[c][p] .- ref[c][p])) for c in eachindex(ref), p in 1:6) /
                  maximum(maximum(abs.(ref[c][p])) for c in eachindex(ref), p in 1:6)

_label(s) = s isa AT.UpwindScheme ? "upwind" : s isa AT.SlopesScheme ? "slopes" :
            s isa AT.LinRoodPPMScheme ? (s.vertical isa AT.UpwindScheme ? "Lin-Rood" : "Lin-Rood FV3") :
            "PPM $(nameof(typeof(s.limiter)))"

const SCHEMES = (AT.UpwindScheme(), AT.SlopesScheme(AT.NoLimiter()), AT.PPMScheme(AT.NoLimiter()),
                 AT.PPMScheme(AT.MonotoneLimiter()), AT.LinRoodPPMScheme())
const PHYSICS = (:none, :split, :boundary, :tm5, :cmfmc)
const P64 = _sweep_problem()

# ---------------------------------------------------------------------------
# Gradient equality with the per-observation reference
# ---------------------------------------------------------------------------

@testset "single-sweep 4D-Var gradient == per-observation footprints (Float64)" begin
    observations = _observations(P64)
    controls = _controls(P64)
    cases = [(s, p) for s in SCHEMES for p in PHYSICS]
    fv3 = AT.PPMScheme(; vertical = Adv.FV3ScalarProfile())
    append!(cases, [(s, p) for s in (AT.PPMScheme(AT.CW84Limiter()),
                                   AT.LinRoodPPMScheme(5; vertical = fv3)) for p in (:none, :boundary)])
    @testset "$(_label(scheme)), physics $physics" for (scheme, physics) in cases
        ref = _reference_4dvar(P64, observations, controls; scheme, physics)
        full = _fourdvar(P64, observations, controls; scheme, physics)
        # The forward simulated values are untouched by the new gradient path.
        @test full.simulated == ref.simulated
        @test full.observation_cost ≈
              sum(0.5 * ((ref.simulated[i] - o.value) / o.sigma)^2 for (i, o) in enumerate(observations)) rtol = 1e-14
        @test _relerr(full.gradients, ref.gradients) <= 1e-12
        # The schedules replay the same tape: Stride matches Full.
        for K in (1, 2, 3)
            result = _fourdvar(P64, observations, controls; scheme, physics,
                               tape = AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(K)))
            @test result.simulated == full.simulated
            @test _relerr(result.gradients, full.gradients) <= 1e-12
        end
    end
end

@testset "observations ending before the run: later window steps get zero gradient" begin
    # The last observation is at step 3 of 4: the taped run stops there, and
    # the `:late` window's step 4 contributes nothing (the old per-observation
    # `ignore_future` semantics).
    observations = _observations(P64; last = 3)
    controls = _controls(P64)
    for scheme in (AT.UpwindScheme(), AT.PPMScheme(AT.MonotoneLimiter()))
        ref = _reference_4dvar(P64, observations, controls; scheme, physics = :tm5)
        for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2), AT.StrideCheckpoint(3))
            result = _fourdvar(P64, observations, controls; scheme, physics = :tm5,
                               tape = AT.AdjointTape(; checkpoint))
            @test _relerr(result.gradients, ref.gradients) <= 1e-12
        end
    end
end

@testset "single-sweep 4D-Var gradient == per-observation footprints (Float32)" begin
    P32 = _sweep_problem(; FT = Float32)
    observations = _observations(P32)
    controls = _controls(P32)
    @testset "$(_label(scheme)), physics $physics" for
            scheme in (AT.PPMScheme(AT.NoLimiter()), AT.PPMScheme(AT.MonotoneLimiter())),
            physics in (:none, :split)
        ref = _reference_4dvar(P32, observations, controls; scheme, physics)
        result = _fourdvar(P32, observations, controls; scheme, physics)
        @test eltype(result.gradients[1][1]) == Float32
        @test _relerr(result.gradients, ref.gradients) <= 1e-5
    end
end

# ---------------------------------------------------------------------------
# One tape and one reverse pass per gradient, whatever the observation count
# ---------------------------------------------------------------------------

# A tape storage that is its own pool and counts staged states, reads of
# staged states, and the opens, rewinds and releases of its pool (with the
# `generation` and `released` fields of a pool).
mutable struct _CountingTapeStorage <: TapeMod.AbstractCSTapeStorage
    stages::Int
    reads::Int
    opens::Int
    rewinds::Int
    releases::Int
    generation::Int
    released::Bool
end
_CountingTapeStorage() = _CountingTapeStorage(0, 0, 0, 0, 0, 1, false)
struct _CountingSlot{P}
    storage::_CountingTapeStorage
    panels::P
end
AT.Adjoints.Footprint._tape_storage_tier(::_CountingTapeStorage) = :device
TapeMod._open_tape_pool(s::_CountingTapeStorage, _nbytes::Integer) = (s.opens += 1; s)
TapeMod._holds_tape_bytes(::_CountingTapeStorage, _nbytes::Integer) = true
TapeMod._rewind_tape_pool!(s::_CountingTapeStorage) = (s.rewinds += 1; s.generation += 1; s)
TapeMod._release_tape_pool!(s::_CountingTapeStorage) = (s.releases += 1; s.released = true; nothing)
TapeMod._allocate_tape_slot(s::_CountingTapeStorage, panels::NTuple{6}) =
    _CountingSlot(s, ntuple(p -> similar(panels[p]), 6))
function TapeMod.stage_panels!(slot::_CountingSlot, src::NTuple{6})
    slot.storage.stages += 1
    foreach(p -> copyto!(slot.panels[p], src[p]), 1:6)
    return slot
end
function TapeMod._tape_panels(slot::_CountingSlot, ::TapeMod.AbstractTapeReadRole)
    slot.storage.reads += 1
    return slot.panels
end
_counting_tape(storage; kw...) = AT.AdjointTape(; storage, kw...)



@testset "single-sweep control and tape storage validation" begin
    scheme = AT.UpwindScheme()
    controls = _controls(P64)
    observations = _observations(P64; last = 3)
    # Rejections come before the forward run and create no scratch directory.
    mktempdir() do root
        dir = joinpath(root, "tape")
        duplicate = [controls[1], controls[1]]          # duplicate control names
        mmap = AT.AdjointTape(; storage = AT.MmapCSTapeStorage(dir))
        @test_throws ArgumentError _fourdvar(P64, observations, duplicate; scheme, tape = mmap)
        @test_throws TypeError _fourdvar(P64, observations, controls; scheme, tape = :mmap)
        # Lin-Rood tapes are device-only.
        @test_throws ArgumentError _fourdvar(P64, observations, controls;
                                             scheme = AT.LinRoodPPMScheme(), tape = mmap)
        @test_throws ArgumentError _fourdvar(P64, observations, controls;
                                             scheme = AT.LinRoodPPMScheme(),
                                             tape = AT.AdjointTape(; storage = AT.PinnedHostCSTapeStorage()))
        @test !ispath(dir)
    end
    # One pool per call: Full records into it once; Stride rewinds it for each
    # window and stages every step once, as Full does. The call releases it.
    full = _CountingTapeStorage()
    reference = _fourdvar(P64, observations, controls; scheme, tape = _counting_tape(full))
    @test (full.opens, full.rewinds, full.releases) == (1, 0, 1)
    for K in (2, 3)
        storage = _CountingTapeStorage()
        result = _fourdvar(P64, observations, controls; scheme,
                           tape = _counting_tape(storage; checkpoint = AT.StrideCheckpoint(K)))
        @test storage.stages == full.stages
        @test (storage.opens, storage.rewinds, storage.releases) == (1, cld(3, K), 1)
        @test _relerr(result.gradients, reference.gradients) <= 1e-13
    end
end

# ---------------------------------------------------------------------------
# One forward run per evaluation: the observation forward records the tape
# ---------------------------------------------------------------------------

# Diffusion meteorology whose clock counts the steps the replay runs: the replay
# reads `current_time` once at the start of each step, and without a diffusion
# operator nothing else reads it.
mutable struct _StepCountingClock
    reads::Int
end
AT.MetDrivers.current_time(c::_StepCountingClock) = (c.reads += 1; 0.0)

function _replayed_steps(f)
    clock = _StepCountingClock(0)
    result = f(clock)
    return clock.reads, result
end

@testset "one forward run per evaluation; CSCostOnly runs it without a tape" begin
    scheme = AT.PPMScheme(AT.MonotoneLimiter())
    controls = _controls(P64)
    # Assimilated observations through step 3 and a co-sample at step 4: the
    # forward runs 4 steps, the tape and the reverse pass 3.
    observations = [_observations(P64; last = 3);
                    AT.CSObservation(4, AT.CSColumnMeanObjective(2, 2, 3), 0.08, 0.001; assimilate = false)]
    evaluate(clock; kw...) = _fourdvar(P64, observations, controls; scheme, diffusion_meteo = clock, kw...)
    n, footprint = _replayed_steps(clock -> AT.cs_surface_emission_footprint(
        P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh, AT.CSColumnMeanObjective(2, 2, 3);
        scheme, dt = DT, diffusion_meteo = clock))
    @test n == 2 * P64.nsteps                   # one cached CFL size count, then recording
    n, full = _replayed_steps(evaluate)
    @test n == 3 + 4                            # cached CFL size count, then one recording forward
    storage = _CountingTapeStorage()
    _fourdvar(P64, observations, controls; scheme, tape = _counting_tape(storage))
    est = AT.cs_tape_byte_estimate(P64.m, P64.am[1:3], P64.bm[1:3], P64.cm[1:3], P64.mesh, scheme)
    @test storage.stages == est.state_records    # the co-sample's step is not taped
    n, stride = _replayed_steps(clock -> evaluate(clock; tape = AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(2))))
    @test n == 3 + 4 + 3                        # size count, forward checkpoints, head re-recording
    mktempdir() do root
        mmap_tape(dir) = AT.AdjointTape(; storage = AT.MmapCSTapeStorage(dir))
        n, mmap = _replayed_steps(clock -> evaluate(clock; tape = mmap_tape(joinpath(root, "full"))))
        @test n == 3 + 4                        # the counting run sizes the file, then one forward run
        dir = joinpath(root, "cost")
        n, cost = _replayed_steps(clock -> evaluate(clock; tape = mmap_tape(dir),
                                                    evaluation = AT.CSCostOnly()))
        @test n == 4                            # the forward run alone: no tape, no reverse pass
        @test !ispath(dir)                      # and no tape storage
        @test cost.cost == full.cost && cost.observation_cost == full.observation_cost
        @test cost.simulated == full.simulated && cost.residuals == full.residuals
        @test [s.assimilated for s in cost.stencils] == [s.assimilated for s in full.stencils]
        @test isempty(cost.gradients) && isempty(cost.gradient_by_name)
        for result in (stride, mmap)
            @test result.simulated == full.simulated
            # Bit for bit on one thread (CPU atomics otherwise).
            @test Threads.nthreads() == 1 ? result.gradients == full.gradients :
                                            _relerr(result.gradients, full.gradients) <= 1e-13
        end
    end
    # The optimizers need the gradient at every point.
    @test_throws ArgumentError AT.cs_surface_flux_4dvar_optimize(
        P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh, observations, controls;
        scheme, dt = DT, evaluation = AT.CSCostOnly())
    # Background terms and the preconditioned space: the same costs without the
    # gradient.
    background = [AT.CSSurfaceFluxControl(c.window, c.value; background = map(zero, c.value), sigma = 2e-5)
                  for c in controls]
    @test _fourdvar(P64, observations, background; scheme, evaluation = AT.CSCostOnly()).cost ==
          _fourdvar(P64, observations, background; scheme).cost
    prec = AT.CSSurfaceFluxPreconditioner(AT.DiagonalCSCovariance(ntuple(_ -> fill(2e-5, P64.Nc, P64.Nc), 6)),
                                          ntuple(_ -> zeros(P64.Nc, P64.Nc), 6), AT.LinearOptimType())
    chi = [AT.CSSurfaceFluxControl(c.window, map(v -> v ./ 2e-5, c.value)) for c in controls]
    chi_cost = _fourdvar(P64, observations, chi; scheme, preconditioner = prec, evaluation = AT.CSCostOnly())
    chi_full = _fourdvar(P64, observations, chi; scheme, preconditioner = prec)
    @test chi_cost.cost == chi_full.cost && isempty(chi_cost.gradients)
end

@testset "an optimization records every evaluation into one tape pool" begin
    scheme = AT.PPMScheme(AT.MonotoneLimiter())
    observations, controls = _observations(P64), _controls(P64)
    optimizer = AT.CSGradientDescent(; iterations = 2, initial_step = 1e3, min_step = 1e-3)
    optimize(clock; tape) = AT.cs_surface_flux_4dvar_optimize(
        P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh, observations, controls;
        scheme, dt = DT, diffusion_meteo = clock, optimizer, tape)
    storage = _CountingTapeStorage()
    n, counted = _replayed_steps(clock -> optimize(clock; tape = _counting_tape(storage)))
    evaluations = n ÷ P64.nsteps - 1            # one size count, then one forward run per evaluation
    @test evaluations >= 3 && n == (evaluations + 1) * P64.nsteps
    # Opened once, rewound by every later evaluation, released at the end.
    @test (storage.opens, storage.rewinds, storage.releases) == (1, evaluations - 1, 1)
    mktempdir() do root
        dir = joinpath(root, "tape")
        n_mmap, mmap = _replayed_steps(clock -> optimize(clock; tape = AT.AdjointTape(; storage = AT.MmapCSTapeStorage(dir))))
        # One file for the optimization: its counting run (every step) comes once.
        @test n_mmap == n                           # both tiers cache the same CFL size count
        @test isempty(readdir(dir))
        @test mmap.cost_history == counted.cost_history
        @test Threads.nthreads() == 1 ? mmap.last.gradients == counted.last.gradients :
                                        _relerr(mmap.last.gradients, counted.last.gradients) <= 1e-13
    end
    stride = optimize(nothing; tape = AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(3)))
    @test Threads.nthreads() == 1 ? stride.cost_history == counted.cost_history :
                                    isapprox(stride.cost_history, counted.cost_history; rtol = 1e-13)
end

# ---------------------------------------------------------------------------
# AirMassTape: one recording per call or optimization, reused after an
# identity check
# ---------------------------------------------------------------------------

@testset "AirMassTape needs a scheme whose reverse reads the air mass only" begin
    air = AT.AdjointTape(; content = AT.AirMassTape())
    @test_throws ArgumentError AT.AdjointTape(; content = AT.AirMassTape(), checkpoint = AT.StrideCheckpoint(2))
    for scheme in (AT.UpwindScheme(), AT.SlopesScheme(AT.NoLimiter()), AT.PPMScheme(AT.NoLimiter()))
        @test ADJ.tape_dependence(scheme) === ADJ.AirMassDependence()
    end
    observations, controls = _observations(P64), _controls(P64)
    for scheme in (AT.PPMScheme(AT.MonotoneLimiter()), AT.PPMScheme(AT.CW84Limiter()), AT.LinRoodPPMScheme())
        @test ADJ.tape_dependence(scheme) === ADJ.TracerDependence()
        # Rejected before the forward run.
        n, err = _replayed_steps(clock -> try
            _fourdvar(P64, observations, controls; scheme, tape = air, diffusion_meteo = clock)
        catch e
            e
        end)
        @test err isa ArgumentError && occursin("AirMassTape", err.msg) && n == 0
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh, AT.CSColumnMeanObjective(2, 2, 3);
            scheme, dt = DT, tape = air)
    end
end

@testset "AirMassTape: the first evaluation's tape serves the optimization ($physics)" for
        physics in (:none, :tm5, :split)
    scheme = AT.UpwindScheme()
    observations, controls = _observations(P64), _controls(P64)
    kw = _physics(physics, P64)
    # Fixed-step gradient descent: two steps, three evaluations, each moving
    # the controls by about a tenth of their size.
    probe = _fourdvar(P64, observations, controls; scheme, physics)
    norm2(tuples) = sqrt(sum(sum(abs2, a) for t in tuples for a in t))
    step = norm2([c.value for c in controls]) / (10 * norm2(probe.gradients))
    optimizer = AT.CSGradientDescent(; iterations = 2, initial_step = step, min_step = step, line_search = false)
    # Without diffusion only the replay reads the clock, once per step.
    clocked(clock) = physics === :none ? (; diffusion_meteo = clock) : (;)
    optimize(tape, clock) = AT.cs_surface_flux_4dvar_optimize(
        P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh, observations, controls;
        scheme, dt = DT, optimizer, tape, kw..., clocked(clock)...)
    fresh_storage, air_storage = _CountingTapeStorage(), _CountingTapeStorage()
    _, fresh = _replayed_steps(clock -> optimize(_counting_tape(fresh_storage), clock))
    n, reused = _replayed_steps(clock -> optimize(_counting_tape(air_storage; content = AT.AirMassTape()), clock))
    @test length(fresh.cost_history) == length(reused.cost_history) == 3
    physics === :none && @test n == 4 * P64.nsteps                # one forward run per evaluation
    # Recorded by the first evaluation only, never rewound; the later ones walk it.
    @test fresh_storage.stages == 3 * air_storage.stages
    @test (air_storage.opens, air_storage.rewinds, air_storage.releases) == (1, 0, 1)
    @test air_storage.reads == fresh_storage.reads
    # The same gradients and costs, bit for bit on one thread.
    if Threads.nthreads() == 1
        @test reused.cost_history == fresh.cost_history
        @test reused.last.gradients == fresh.last.gradients
        @test [c.value for c in reused.controls] == [c.value for c in fresh.controls]
    else
        @test reused.cost_history ≈ fresh.cost_history rtol = 1e-13
    end
end

@testset "AirMassTape: a forward run that does not fit the recording is an error" begin
    Footprint, Solver = ADJ.Footprint, ADJ.Inversion.Solver
    scheme = AT.UpwindScheme()
    observations, controls = _observations(P64), _controls(P64)
    # Met arrays of this test only (one of them is changed in place below).
    P = (; P64..., am = [map(copy, a) for a in P64.am])
    evaluate(runtime; Q = P, obs = observations, kw...) = Solver._cs_surface_flux_4dvar(
        runtime, Q.rm, Q.m, Q.am, Q.bm, Q.cm, Q.mesh, obs, controls; scheme = get(kw, :scheme, scheme),
        dt = get(kw, :dt, DT))
    message(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
    Footprint._with_tape_runtime(AT.AdjointTape(; content = AT.AirMassTape())) do runtime
        first = evaluate(runtime)
        held = runtime.held
        @test held isa Footprint._HeldAirMassTape && held.identity.steps == P.nsteps
        again = evaluate(runtime)                     # reused: the same tape
        @test runtime.held === held
        @test Threads.nthreads() == 1 ? again.gradients == first.gradients :
                                        _relerr(again.gradients, first.gradients) <= 1e-13
        # Each part of the identity, checked before the forward run.
        @test occursin("number of taped steps", message(() -> evaluate(runtime; obs = _observations(P; last = 3))))
        @test occursin("time windows", message(() -> evaluate(runtime; dt = 2DT)))
        @test occursin("scheme", message(() -> evaluate(runtime; scheme = AT.SlopesScheme(AT.NoLimiter()))))
        P.am[2][3][5, 5, 2] += 1e-3                    # the met changed in place
        @test occursin("met inputs", message(() -> evaluate(runtime)))
        P.am[2][3][5, 5, 2] -= 1e-3
        id = held.identity
        changed(; grid = id.grid, steps = id.steps, window = id.window, scheme = id.scheme, met = id.met) =
            Footprint._TapeIdentity(met, window, scheme, grid, steps)
        @test Footprint._held_air_mass_tape(runtime, changed()) === held.ops
        for (what, identity) in (("grid", changed(; grid = (id.grid[1] + 1, Base.tail(id.grid)...))),
                                 ("number of taped steps", changed(; steps = id.steps - 1)),
                                 ("time windows", changed(; window = (id.window[1:end-1]..., 2id.window[end]))),
                                 ("scheme", changed(; scheme = (AT.PPMScheme(AT.NoLimiter()), Base.tail(id.scheme)...))),
                                 ("met inputs", changed(; met = [id.met; zero(UInt64)])))
            @test occursin(what, message(() -> Footprint._held_air_mass_tape(runtime, identity)))
        end
        # A rewound pool makes the held tape stale.
        TapeMod._rewind_tape_pool!(runtime.pool)
        @test occursin("stale", message(() -> evaluate(runtime)))
    end
end

@testset "the reverse pass stops at the last assimilated step" begin
    # An averaging-kernel column at step 4 that the surface-pressure screen
    # rejects (retrieval surface at 2000 hPa, model about 1000 hPa): the
    # forward run tapes step 4 too, but the walk starts at step 3.
    O = AT.Observations
    scheme, controls, met = AT.UpwindScheme(), _controls(P64), AT.CSObservationColumnMet(; p_top = 1000.0)
    column = O.RetrievalColumn(; pressure_pa = [2e5, 1e5, 0.0], pressure_weight = [0.5, 0.5],
                               averaging_kernel = [1.0, 1.0], prior_profile = [0.08, 0.08])
    screened = AT.CSObservation(4, AT.CSRetrievalColumnObjective(3, 2, 2, 1.36e-3, column, O.DryInterfaces();
        max_surface_pressure_mismatch_pa = 1000.0, mismatch_policy = O.SkipUnplaced()), 0.08, 0.001)
    counted(observations) = (storage = _CountingTapeStorage();
        r = _fourdvar(P64, observations, controls; scheme, tape = _counting_tape(storage), column_met = met);
        (storage, r))
    kept, r_kept = counted(_observations(P64; last = 3))
    taped, r_taped = with_logger(NullLogger()) do
        counted([_observations(P64; last = 3); screened])
    end
    @test !r_taped.stencils[end].assimilated
    states(n) = AT.cs_tape_byte_estimate(P64.m, P64.am[1:n], P64.bm[1:n], P64.cm[1:n], P64.mesh, scheme).state_records
    @test kept.stages == states(3) && taped.stages == states(4)    # step 4 is taped ...
    @test taped.reads == kept.reads == states(3)                    # ... but not walked
    @test Threads.nthreads() == 1 ? r_taped.gradients == r_kept.gradients :
                                    _relerr(r_taped.gradients, r_kept.gradients) <= 1e-13
end

# ---------------------------------------------------------------------------
# Jacobian: one tape, one walk per objective
# ---------------------------------------------------------------------------

# The halo-adjoint and Lin-Rood face kernels accumulate with atomics, so a
# multithreaded CPU reverse pass reproduces itself only to rounding (two
# identical `cs_surface_emission_footprint` calls differ by ~1e-16 relative
# at `-t 4`). Single-threaded, as the test runner runs, it is bit for bit.
const _BITWISE = Threads.nthreads() == 1

function _footprint_difference(a, b)
    nsteps = length(b.footprints)
    diff = maximum(maximum(abs.(a.footprints[s][p] .- b.footprints[s][p])) for s in 1:nsteps, p in 1:6)
    return diff / maximum(maximum(abs.(b.footprints[s][p])) for s in 1:nsteps, p in 1:6)
end

_same_footprints(a, b) = _BITWISE ? _footprint_difference(a, b) == 0 :
                                    _footprint_difference(a, b) <= 1e-13

@testset "Jacobian reuses one tape and matches per-objective footprints bit for bit" begin
    objectives = [AT.CSLayerMeanObjective(1, 2, 3, P64.Nz), AT.CSColumnMeanObjective(4, 1, 2),
                  AT.CSLayerMeanObjective(4, 1, 2, P64.Nz - 1)]
    windows = [AT.CSSurfaceFluxWindow(:all, 1:P64.nsteps), AT.CSSurfaceFluxWindow(:mean12, 1:2; normalize = true)]
    rates = ADJ.Inversion.Solver._surface_rates_from_controls(_controls(P64), P64.nsteps, P64.mesh, P64.rm[1])
    @testset "$(_label(scheme)), physics $physics, $(nameof(typeof(tape.storage))), $(nameof(typeof(tape.checkpoint)))" for
            (scheme, physics, tape) in (
            (AT.UpwindScheme(), :tm5, AT.AdjointTape()),
            (AT.UpwindScheme(), :split, AT.AdjointTape(; storage = AT.PinnedHostCSTapeStorage())),
            (AT.UpwindScheme(), :boundary, AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(3))),
            (AT.PPMScheme(AT.MonotoneLimiter()), :cmfmc, AT.AdjointTape()),
            (AT.PPMScheme(AT.MonotoneLimiter()), :boundary,
             AT.AdjointTape(; storage = AT.MmapCSTapeStorage(mktempdir()))),
            (AT.PPMScheme(AT.MonotoneLimiter()), :split,
             AT.AdjointTape(; storage = AT.PinnedHostCSTapeStorage(), checkpoint = AT.StrideCheckpoint(2))),
            (AT.LinRoodPPMScheme(), :tm5, AT.AdjointTape()),
            (AT.LinRoodPPMScheme(), :cmfmc, AT.AdjointTape(; checkpoint = AT.StrideCheckpoint(2))))
        base = scheme isa AT.UpwindScheme ? nothing : rates
        jac = AT.cs_surface_flux_jacobian(P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh,
                                          objectives, windows; scheme, dt = DT,
                                          base_emission_rates = base, tape,
                                          _physics(physics, P64)...)
        for (o, objective) in enumerate(objectives)
            single = AT.cs_surface_emission_footprint(
                P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh, objective;
                scheme, dt = DT, base_emission_rates = base, tape,
                _physics(physics, P64)...)
            @test jac.per_step_results[o].objective == objective
            @test _same_footprints(jac.per_step_results[o], single)
        end
    end
    @testset "one recording for all objectives ($(nameof(typeof(checkpoint))))" for
            checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(3))
        scheme = AT.UpwindScheme()
        est = AT.cs_tape_byte_estimate(P64.m, P64.am, P64.bm, P64.cm, P64.mesh, scheme)
        storage = _CountingTapeStorage()
        AT.cs_surface_flux_jacobian(P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh,
                                    objectives, windows; scheme, dt = DT,
                                    tape = _counting_tape(storage; checkpoint))
        # Stride records each window once and walks it for every objective.
        @test storage.stages == est.state_records
        @test storage.reads == length(objectives) * est.state_records
    end
end

# ---------------------------------------------------------------------------
# Observations as linear functionals, and the step forcing
# ---------------------------------------------------------------------------

@testset "objectives as linear functionals; step-forcing injection" begin
    P = P64
    mesh, Hp, Nz = P.mesh, P.mesh.Hp, P.Nz
    @testset "functional value == evaluate_objective" begin
        for objective in (AT.CSLayerMeanObjective(3, 1, 4, 2), AT.CSColumnMeanObjective(5, 2, 2))
            f = ADJ._linear_functional(objective, 2, P.m, mesh)
            @test f.steps == fill(2, length(f.weight))
            @test f.offset == 0
            forcing = ADJ._assemble_step_forcing([f], [1.0], mesh, Nz, P.m[1])
            seed = ntuple(p -> zeros(size(P.rm[p])), 6)
            ADJ._inject_step_forcing!(seed, forcing, 2, mesh)
            @test sum(sum(seed[p] .* P.rm[p]) for p in 1:6) ≈
                  AT.evaluate_objective(objective, P.rm, P.m, mesh) rtol = 1e-14
        end
        @test_throws ArgumentError ADJ._linear_functional(AT.CSSeedObjective(), 1, P.m, mesh)
    end

    @testset "injection: steps, panels, repeated cells" begin
        f1 = ADJ.CSLinearFunctional{Float64}(2, [3, 3, 1], [1, 1, 4], [2, 2, 4], [Nz, Nz, 1],
                                             [0.5, 0.25, 2.0], 0.0)
        f2 = ADJ.CSLinearFunctional{Float64}(4, [6], [2], [3], [2], [1.0], 0.0)
        f3 = ADJ.CSLinearFunctional{Float64}(2, [3], [1], [2], [Nz], [4.0], 0.0)
        forcing = ADJ._assemble_step_forcing([f1, f2, f3], [2.0, -1.0, 0.5], mesh, Nz, P.m[1])
        lambda = ntuple(p -> zeros(size(P.m[p])), 6)
        ADJ._inject_step_forcing!(lambda, forcing, 2, mesh)
        # Two entries of f1 and one of f3 land in one cell: 2(0.5 + 0.25) + 0.5·4.
        @test lambda[3][Hp + 1, Hp + 2, Nz] == 3.5
        @test lambda[1][Hp + 4, Hp + 4, 1] == 4.0
        @test sum(sum(abs, l) for l in lambda) == 7.5
        ADJ._inject_step_forcing!(lambda, forcing, 3, mesh)     # no entries
        @test sum(sum(abs, l) for l in lambda) == 7.5
        ADJ._inject_step_forcing!(lambda, forcing, 4, mesh)
        @test lambda[6][Hp + 2, Hp + 3, 2] == -1.0
        ADJ._inject_step_forcing!(lambda, ADJ.NoStepForcing(), 2, mesh)
        @test sum(sum(abs, l) for l in lambda) == 8.5
        # The forcing must lie within the taped run.
        @test_throws ArgumentError ADJ._validate_step_forcing(forcing, 3, Nz)
        @test_throws DimensionMismatch ADJ._validate_step_forcing(forcing, 4, Nz + 1)
        @test ADJ._validate_step_forcing(forcing, 4, Nz) === nothing
        bad = ADJ.CSLinearFunctional{Float64}(1, [7], [1], [1], [1], [1.0], 0.0)
        @test_throws ArgumentError ADJ._assemble_step_forcing([bad], [1.0], mesh, Nz, P.m[1])
    end
end
