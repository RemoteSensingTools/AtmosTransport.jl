#!/usr/bin/env julia
#
# Issue #20: the 4D-Var gradient in one reverse sweep.
#
# `cs_surface_flux_4dvar` evaluates `J_obs = ½ Σᵢ ((Hᵢ(x) − yᵢ)/σᵢ)²` and its
# gradient with one taped forward run to the last observation step and one
# reverse pass. Each observation is a fixed sparse linear functional of the
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
            kz_field = AT.CubedSphereField(ntuple(_ -> AT.ProfileKzField(FT[30 / k for k in 1:Nz]), 6)),
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
    rates = ADJ._surface_rates_from_controls(controls, P.nsteps, P.mesh, P.rm[1])
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
    @testset "$(_label(scheme)), physics $physics" for scheme in SCHEMES, physics in PHYSICS
        ref = _reference_4dvar(P64, observations, controls; scheme, physics)
        full = _fourdvar(P64, observations, controls; scheme, physics)
        # The forward simulated values are untouched by the new gradient path.
        @test full.simulated == ref.simulated
        @test full.observation_cost ≈
              sum(0.5 * ((ref.simulated[i] - o.value) / o.sigma)^2 for (i, o) in enumerate(observations)) rtol = 1e-14
        @test _relerr(full.gradients, ref.gradients) <= 1e-12
        # The schedules replay the same tape: Stride and Revolve match Full.
        for checkpoint in (AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
            result = _fourdvar(P64, observations, controls; scheme, physics, checkpoint)
            @test result.simulated == full.simulated
            @test _relerr(result.gradients, full.gradients) <= 1e-12
        end
    end
end

@testset "remaining scheme families: CW84 PPM, Lin-Rood with the FV3 vertical" begin
    fv3 = AT.PPMScheme(; vertical = Adv.FV3ScalarProfile())
    observations = _observations(P64)
    controls = _controls(P64)
    @testset "$(_label(scheme)), physics $physics" for
            scheme in (AT.PPMScheme(AT.CW84Limiter()), AT.LinRoodPPMScheme(5; vertical = fv3)),
            physics in (:none, :boundary)
        ref = _reference_4dvar(P64, observations, controls; scheme, physics)
        full = _fourdvar(P64, observations, controls; scheme, physics)
        @test _relerr(full.gradients, ref.gradients) <= 1e-12
        stride = _fourdvar(P64, observations, controls; scheme, physics,
                           checkpoint = AT.StrideCheckpoint(2))
        @test _relerr(stride.gradients, full.gradients) <= 1e-12
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
        for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
            result = _fourdvar(P64, observations, controls; scheme, physics = :tm5, checkpoint)
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

# Device tape storage that counts staged states and reads of staged states.
mutable struct _CountingTapeStorage <: TapeMod.AbstractCSTapeStorage
    stages::Int
    reads::Int
end
_CountingTapeStorage() = _CountingTapeStorage(0, 0)
TapeMod._allocate_tape_slot(s::_CountingTapeStorage, panels::NTuple{6}) =
    TapeMod.CSTapeSlot(s, ntuple(p -> similar(panels[p]), 6))
function TapeMod.stage_panels!(slot::TapeMod.CSTapeSlot{_CountingTapeStorage}, src::NTuple{6})
    slot.storage.stages += 1
    foreach(p -> copyto!(slot.panels[p], src[p]), 1:6)
    return slot
end
function TapeMod._tape_panels(slot::TapeMod.CSTapeSlot{_CountingTapeStorage},
                              ::TapeMod.AbstractTapeReadRole)
    slot.storage.reads += 1
    return slot.panels
end

@testset "one taped run and one reverse pass per gradient" begin
    scheme = AT.UpwindScheme()
    controls = _controls(P64)
    all_obs = _observations(P64)
    # The linear mass tape stages one air-mass state per sweep, and the reverse
    # pass reads each once: a recording is `state_records` stages and a reverse
    # pass as many reads, for a tape to the last observation step.
    est = AT.cs_tape_byte_estimate(P64.m, P64.am, P64.bm, P64.cm, P64.mesh, scheme)
    @test est.state_records == est.sweep_records
    for observations in (all_obs[2:2], all_obs)   # one and seven observations
        storage = _CountingTapeStorage()
        result = _fourdvar(P64, observations, controls; scheme, tape_storage = storage)
        @test isfinite(result.cost)
        @test storage.stages == est.state_records
        @test storage.reads == est.state_records
    end
end

@testset "checkpoint and tape_path: one schedule's windows on disk" begin
    scheme = AT.UpwindScheme()
    controls = _controls(P64)
    observations = _observations(P64; last = 3)          # tape to step 3
    est = AT.cs_tape_byte_estimate(P64.m, P64.am[1:3], P64.bm[1:3], P64.cm[1:3],
                                   P64.mesh, scheme)
    reference = _fourdvar(P64, observations, controls; scheme)
    records_bytes(dir) = filesize(joinpath(dir, "records.bin"))
    for (checkpoint, subdirs) in ((AT.FullCheckpoint(), String[]),
                                  (AT.StrideCheckpoint(2), ["window_00001", "window_00002"]),
                                  (AT.RevolveCheckpoint(), ["step_00001", "step_00002", "step_00003"]))
        mktempdir() do dir
            result = _fourdvar(P64, observations, controls; scheme, checkpoint,
                               tape_storage = :mmap, tape_path = dir)
            @test _relerr(result.gradients, reference.gradients) <= 1e-12
            if isempty(subdirs)
                @test sort(readdir(dir)) == ["manifest.toml", "records.bin"]
                @test records_bytes(dir) == est.state_bytes
            else
                @test sort(readdir(dir)) == subdirs
                @test all(isfile(joinpath(dir, d, "manifest.toml")) for d in subdirs)
                # The windows together hold exactly one tape to step 3.
                @test sum(records_bytes(joinpath(dir, d)) for d in subdirs) == est.state_bytes
            end
        end
    end
    # Rejections come before the forward run and create no `tape_path`.
    mktempdir() do root
        dir = joinpath(root, "tape")
        duplicate = [controls[1], controls[1]]          # duplicate control names
        @test_throws ArgumentError _fourdvar(P64, observations, duplicate; scheme,
                                             tape_storage = :mmap, tape_path = dir)
        @test_throws ArgumentError _fourdvar(P64, observations, controls; scheme,
                                             tape_storage = :device, tape_path = dir)
        # Lin-Rood tapes are device-only.
        @test_throws ArgumentError _fourdvar(P64, observations, controls;
                                             scheme = AT.LinRoodPPMScheme(),
                                             tape_storage = :mmap, tape_path = dir)
        @test_throws ArgumentError _fourdvar(P64, observations, controls;
                                             scheme = AT.LinRoodPPMScheme(),
                                             tape_storage = :pinned_host)
        @test !ispath(dir)
    end
    # Stride and Revolve build one storage per window.
    for checkpoint in (AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
        storage = _CountingTapeStorage()
        @test_throws ArgumentError _fourdvar(P64, observations, controls; scheme,
                                             checkpoint, tape_storage = storage)
        @test storage.stages == 0
    end
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
    rates = ADJ._surface_rates_from_controls(_controls(P64), P64.nsteps, P64.mesh, P64.rm[1])
    @testset "$(_label(scheme)), physics $physics, $storage" for (scheme, physics, storage) in (
            (AT.UpwindScheme(), :tm5, :device),
            (AT.UpwindScheme(), :split, :pinned_host),
            (AT.PPMScheme(AT.MonotoneLimiter()), :cmfmc, :device),
            (AT.PPMScheme(AT.MonotoneLimiter()), :boundary, :mmap),
            (AT.LinRoodPPMScheme(), :tm5, :device))
        base = scheme isa AT.UpwindScheme ? nothing : rates
        jac = AT.cs_surface_flux_jacobian(P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh,
                                          objectives, windows; scheme, dt = DT,
                                          base_emission_rates = base, tape_storage = storage,
                                          _physics(physics, P64)...)
        for (o, objective) in enumerate(objectives)
            single = AT.cs_surface_emission_footprint(
                P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh, objective;
                scheme, dt = DT, base_emission_rates = base, tape_storage = storage,
                _physics(physics, P64)...)
            @test jac.per_step_results[o].objective == objective
            @test _same_footprints(jac.per_step_results[o], single)
        end
    end
    @testset "one recording for all objectives" begin
        scheme = AT.UpwindScheme()
        est = AT.cs_tape_byte_estimate(P64.m, P64.am, P64.bm, P64.cm, P64.mesh, scheme)
        storage = _CountingTapeStorage()
        AT.cs_surface_flux_jacobian(P64.rm, P64.m, P64.am, P64.bm, P64.cm, P64.mesh,
                                    objectives, windows; scheme, dt = DT,
                                    tape_storage = storage)
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
            @test f.step == 2
            @test f.offset == 0
            @test ADJ._evaluate_functional(f, P.rm, mesh) ≈
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
