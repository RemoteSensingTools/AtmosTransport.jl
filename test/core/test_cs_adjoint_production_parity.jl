#!/usr/bin/env julia
#
# The cubed-sphere adjoint differentiates the production step.
#
# * Issue #57: host-side tape storage (`:pinned_host`, and `:mmap` on a GPU)
#   reads each staged state through a device-side cache. A limited-PPM sweep
#   record hands its reverse kernel both the air mass and the tracer mass, so
#   the two reads must land in different buffers (one cache per read role).
# * Issue #60: the forward replays (`run_cs_footprint_forward`, the 4D-Var
#   simulated values) and the tape recorders must compose a step as production
#   `strang_split!` plus the `TransportModel.step!` convection block do: the
#   diffusion operator's surface-flux coupling at the palindrome center, and
#   the step-start halo refresh that follows the previous step's convection.
#
# Since the adjoint replays the production step plan (D31) with the
# production operators and their persistent state (the convection ledger
# carry, the surface source's Kahan compensation), the replay equals
# `TransportModel.step!` bit for bit (`==` in one thread; with several threads
# a 1e-13 bound, as the halo and Lin-Rood adjoint kernels use atomics). The
# adjoint's default schedule is one step per met window, so `_production`
# invalidates the CMFMC CFL cache before each step, as `DrivenSimulation`
# does when it installs a window's convection forcing.
# test_cs_adjoint_step_plan_parity.jl covers driven runs with several steps
# per window and both physics cadences.

using Test
using Random
import AtmosTransport
const AT = AtmosTransport
const ADJ = AT.Adjoints
const TapeMod = AT.Tape
const Adv = AT.Operators.Advection

const NC = 4
const NZ = 5
const NSTEPS = 6

# Synthetic cubed-sphere problem: smooth air mass and tracer mass with filled
# halos, and one set of face mass fluxes per step. Masses and mass fluxes are
# multiples of `mass_scale` [kg]; the production comparisons use a scale at
# which the air mass per area is atmospheric (about 2000 kg m⁻² per layer on
# the C4 cells), because the convection operators combine it with forcings in
# kg m⁻² s⁻¹. Transport, diffusion and mixing ratios do not depend on it.
function _parity_problem(; Nc = NC, Nz = NZ, nsteps = NSTEPS, FT = Float64,
                         mass_scale = 1.0)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    Hp = mesh.Hp
    N = Nc + 2Hp
    S = mass_scale
    m = ntuple(p -> [FT(S * (2.0 + 0.25k + 0.01p + 0.03sin(0.7i + 0.3j + p)))
                     for i in 1:N, j in 1:N, k in 1:Nz], 6)
    rm = ntuple(p -> [m[p][i, j, k] * FT(0.08 + 0.01sin(0.27i + 0.13j + 0.19k + 0.07p))
                      for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(m, mesh; dir = 0)
    Adv.fill_panel_halos!(rm, mesh; dir = 0)
    am = [ntuple(p -> begin
              a = zeros(FT, N + 1, N, Nz)
              for k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc+1
                  a[i, j, k] = FT(S * 0.05sin(0.2s + 0.3p + 0.7i + 0.4j + 0.2k))
              end
              a
          end, 6) for s in 1:nsteps]
    bm = [ntuple(p -> begin
              b = zeros(FT, N, N + 1, Nz)
              for k in 1:Nz, j in Hp+1:Hp+Nc+1, i in Hp+1:Hp+Nc
                  b[i, j, k] = FT(S * 0.04cos(0.3s + 0.5p + 0.4i + 0.6j + 0.1k))
              end
              b
          end, 6) for s in 1:nsteps]
    cm = [ntuple(p -> begin
              c = zeros(FT, N, N, Nz + 1)
              for k in 2:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
                  c[i, j, k] = FT(-S * 0.03(1 + 0.3sin(i + j + k + p + s)))
              end
              c
          end, 6) for s in 1:nsteps]
    return (; mesh, m, rm, am, bm, cm)
end

const MASS_SCALE = 4e15
const PB = _parity_problem(; mass_scale = MASS_SCALE)
const MESH = PB.mesh
const HP = MESH.Hp
const GRID = AT.AtmosGrid(MESH, AT.HybridSigmaPressure(zeros(NZ + 1),
                                                      collect(range(0.0, 1.0, NZ + 1))),
                          AT.CPU(); FT = Float64)
# Surface emission rates [kg s⁻¹ per cell]: about 1 % of the surface-layer
# tracer mass per 1800 s step.
const RATES = ntuple(p -> [MASS_SCALE * 1e-6 * (1 + 0.3sin(i + 2j + p))
                           for i in 1:NC, j in 1:NC], 6)

_layer_thickness(k, p) = 40.0 + 15k + 3p

function _fill_layer_thickness!(ws)
    for p in 1:6, k in 1:NZ
        ws.layer_thickness[p][:, :, k] .= _layer_thickness(k, p)
    end
    return ws
end

_diffusion_workspace() = _fill_layer_thickness!(AT.DiffusionWorkspace(PB.m, HP, 0))

_kz_diffusion(coupling) = AT.ImplicitVerticalDiffusion(;
    kz_field = AT.CubedSphereField(ntuple(_ -> AT.PreComputedKzField(repeat(reshape(Float64[30.0 / k for k in 1:NZ], 1, 1, NZ), NC, NC, 1)), 6)),
    surface_flux_coupling = coupling)

const COUPLINGS = (AT.SplitSurfaceFluxCoupling(), AT.DiffusiveSurfaceFluxBoundary())

# TM5 entrainment and detrainment rates [kg m⁻² s⁻¹] (ten times the forcing of
# the issue #60 report): an 1800 s step entrains about a quarter of a layer's
# air mass into the updraft.
function _tm5_forcing()
    mk(lo, hi, v) = ntuple(p -> (e = zeros(NC, NC, NZ); e[:, :, lo:min(hi, NZ - 1)] .= v; e), 6)
    return AT.ConvectionForcing(nothing, nothing,
        (; entu = mk(2, 4, 0.3), detu = mk(2, 4, 0.2),
           entd = mk(3, 4, 0.1), detd = mk(3, 4, 0.05)))
end

# CMFMC updraft mass flux and detrainment [kg m⁻² s⁻¹]: an 1800 s step moves
# about a tenth of a layer's air mass.
function _cmfmc_forcing()
    cmfmc = ntuple(_ -> (c = zeros(NC, NC, NZ + 1);
                         c[:, :, 2] .= 0.06; c[:, :, 3] .= 0.10; c[:, :, 4] .= 0.075; c), 6)
    dtrain = ntuple(_ -> (d = zeros(NC, NC, NZ); d[:, :, 2] .= 0.03; d[:, :, 3] .= 0.025; d), 6)
    return AT.ConvectionForcing(cmfmc, dtrain, nothing)
end

const CONVECTIONS = (
    none  = (op = AT.NoConvection(), forcing = nothing),
    tm5   = (op = AT.TM5Convection(), forcing = _tm5_forcing()),
    cmfmc = (op = AT.CMFMCConvection(), forcing = _cmfmc_forcing()),
)

# Production `TransportModel.step!` run. It always carries a surface-flux
# operator (zero rates for `rate_scale = 0`): a footprint is the derivative
# with respect to the emissions, so the adjoint's forward model is a run with
# one.
function _production(scheme, conv::Symbol; diffusion = AT.NoDiffusion(), nsteps = NSTEPS,
                     dt = 1800.0, rate_scale = 1.0)
    state = AT.CubedSphereState(AT.DryBasis, MESH, ntuple(p -> copy(PB.m[p]), 6);
                                CO2 = ntuple(p -> copy(PB.rm[p]), 6))
    fluxes = AT.allocate_face_fluxes(MESH, NZ; FT = Float64, basis = AT.DryBasis)
    kw = Dict{Symbol, Any}(:diffusion => diffusion,
        :emissions => AT.SurfaceFluxOperator(AT.SurfaceFluxSource(:CO2,
            ntuple(p -> rate_scale .* RATES[p], 6))))
    if conv !== :none
        kw[:convection] = CONVECTIONS[conv].op
        kw[:convection_forcing] = CONVECTIONS[conv].forcing
    end
    model = AT.TransportModel(state, fluxes, GRID, scheme; kw...)
    diffusion isa AT.NoDiffusion || _fill_layer_thickness!(model.workspace.diffusion_ws)
    for s in 1:nsteps
        for p in 1:6
            copyto!(model.fluxes.am[p], PB.am[s][p])
            copyto!(model.fluxes.bm[p], PB.bm[s][p])
            copyto!(model.fluxes.cm[p], PB.cm[s][p])
        end
        conv !== :none && AT.invalidate_cmfmc_cache!(model.workspace.convection_ws)
        AT.step!(model, dt)
    end
    return model
end

# The replay's convection workspaces use production's cell areas [m²].
const CELL_METRICS = (
    tm5 = _production(AT.UpwindScheme(), :tm5; nsteps = 0).workspace.convection_ws.cell_metrics,
    cmfmc = _production(AT.UpwindScheme(), :cmfmc; nsteps = 0).workspace.convection_ws.cell_metrics,
)

_convection_workspace(::Val{:none}) = nothing
_convection_workspace(::Val{:tm5}) =
    AT.TM5Workspace(PB.m; tile_columns = NC * NC, cell_metrics = CELL_METRICS.tm5)
_convection_workspace(::Val{:cmfmc}) = AT.CMFMCWorkspace(PB.m; cell_metrics = CELL_METRICS.cmfmc)

_convection_kwargs(conv::Symbol) = conv === :none ? (;) :
    (; convection_op = CONVECTIONS[conv].op, convection_forcing = CONVECTIONS[conv].forcing,
       convection_workspace = _convection_workspace(Val(conv)))

_diffusion_kwargs(diffusion) = diffusion isa AT.NoDiffusion ? (;) :
    (; diffusion_op = diffusion, diffusion_workspace = _diffusion_workspace())

const STATE_OBJECTIVES = [AT.CSLayerMeanObjective(p, i, j, k)
                          for p in 1:6 for k in 1:NZ for j in 1:NC for i in 1:NC]

# Final dry VMR of every interior cell: production state, and the 4D-Var
# simulated values of the adjoint replay (`_run_cs_observations_forward`).
_production_vmr(model) = [ADJ.evaluate_objective(obj, AT.get_tracer(model.state, :CO2),
                                                 model.state.air_mass, MESH)
                          for obj in STATE_OBJECTIVES]

function _replay_vmr(scheme, conv::Symbol; diffusion = AT.NoDiffusion(), nsteps = NSTEPS,
                     dt = 1800.0)
    observations = [AT.CSObservation(nsteps, obj, 0.0, 1.0) for obj in STATE_OBJECTIVES]
    return ADJ._run_cs_observations_forward(
        PB.rm, PB.m, PB.am[1:nsteps], PB.bm[1:nsteps], PB.cm[1:nsteps], MESH, observations;
        scheme, dt, emission_rates = fill(RATES, nsteps),
        _diffusion_kwargs(diffusion)..., _convection_kwargs(conv)...)
end

_maxrel(a, b) = maximum(abs.(a .- b) ./ abs.(b))
exact(a, b) = Threads.nthreads() == 1 ? a == b : _maxrel(a, b) <= 1e-13

_footprint_maxrel(a, b) =
    maximum(maximum(abs.(a.footprints[s][p] .- b.footprints[s][p]))
            for s in eachindex(a.footprints), p in 1:6) /
    maximum(maximum(abs.(b.footprints[s][p])) for s in eachindex(b.footprints), p in 1:6)

_dot_footprint(result, rates) =
    sum(sum(result.footprints[s][p] .* rates[p]) for s in eachindex(result.footprints), p in 1:6)

_scheme_label(scheme) = scheme isa AT.UpwindScheme ? "Upwind" :
    scheme isa AT.LinRoodPPMScheme ? "LinRood" : "PPM $(nameof(typeof(scheme.limiter)))"

# ---------------------------------------------------------------------------
# #60: forward replay == production step
# ---------------------------------------------------------------------------

@testset "adjoint forward replay equals the production step (#60)" begin
    # The forcings are strong enough to matter: parity is not trivial.
    @testset "forcing controls" begin
        scheme = AT.UpwindScheme()
        split = _kz_diffusion(AT.SplitSurfaceFluxCoupling())
        reference = _production_vmr(_production(scheme, :none; diffusion = split))
        @test _maxrel(_production_vmr(_production(scheme, :tm5; diffusion = split)), reference) > 1e-3
        @test _maxrel(_production_vmr(_production(scheme, :cmfmc; diffusion = split)), reference) > 1e-3
        boundary = _kz_diffusion(AT.DiffusiveSurfaceFluxBoundary())
        @test _maxrel(_production_vmr(_production(scheme, :none; diffusion = boundary)), reference) > 1e-4
    end

    schemes = (AT.UpwindScheme(), AT.PPMScheme(AT.MonotoneLimiter()))
    @testset "$(_scheme_label(scheme)), $(nameof(typeof(coupling))), convection $conv" for
            scheme in schemes, coupling in COUPLINGS, conv in (:none, :tm5, :cmfmc)
        diffusion = _kz_diffusion(coupling)
        replay = _replay_vmr(scheme, conv; diffusion)
        @test exact(replay, _production_vmr(_production(scheme, conv; diffusion)))
    end

    # Lin-Rood's horizontal operator refills the halos it reads, so its replay
    # needs no step-start fill; it must still honour the coupling.
    @testset "LinRood, $(nameof(typeof(coupling))), convection $conv" for
            (coupling, conv) in ((AT.DiffusiveSurfaceFluxBoundary(), :tm5),
                                 (AT.SplitSurfaceFluxCoupling(), :cmfmc))
        scheme = AT.LinRoodPPMScheme()
        diffusion = _kz_diffusion(coupling)
        @test exact(_replay_vmr(scheme, conv; diffusion),
                    _production_vmr(_production(scheme, conv; diffusion)))
    end

    # `run_cs_footprint_forward` (the finite-difference reference) composes the
    # step through the same helper.
    @testset "run_cs_footprint_forward, $(nameof(typeof(coupling)))" for coupling in COUPLINGS
        diffusion = _kz_diffusion(coupling)
        scheme = AT.PPMScheme(AT.MonotoneLimiter())
        objective = AT.CSLayerMeanObjective(3, 2, 2, NZ)
        replay = AT.run_cs_footprint_forward(
            PB.rm, PB.m, PB.am, PB.bm, PB.cm, MESH, objective;
            scheme, dt = 1800.0, emission_rates = fill(RATES, NSTEPS),
            _diffusion_kwargs(diffusion)..., _convection_kwargs(:tm5)...)
        model = _production(scheme, :tm5; diffusion)
        prod = ADJ.evaluate_objective(objective, AT.get_tracer(model.state, :CO2),
                                      model.state.air_mass, MESH)
        @test exact([replay], [prod])
    end
end

# ---------------------------------------------------------------------------
# #60: footprint gradient == finite differences of the production model
# ---------------------------------------------------------------------------

@testset "footprint matches finite differences of production (#60)" begin
    objective = AT.CSLayerMeanObjective(3, 2, 2, NZ)
    prod_objective(scheme, conv, diffusion, c) = begin
        model = _production(scheme, conv; diffusion, rate_scale = c)
        ADJ.evaluate_objective(objective, AT.get_tracer(model.state, :CO2),
                               model.state.air_mass, MESH)
    end
    cases = (
        (AT.UpwindScheme(), AT.DiffusiveSurfaceFluxBoundary(), :tm5),
        (AT.UpwindScheme(), AT.SplitSurfaceFluxCoupling(), :cmfmc),
        (AT.PPMScheme(AT.MonotoneLimiter()), AT.DiffusiveSurfaceFluxBoundary(), :cmfmc),
        (AT.PPMScheme(AT.MonotoneLimiter()), AT.SplitSurfaceFluxCoupling(), :tm5),
        (AT.LinRoodPPMScheme(), AT.DiffusiveSurfaceFluxBoundary(), :tm5),
    )
    @testset "$(_scheme_label(scheme)), $(nameof(typeof(coupling))), convection $conv" for
            (scheme, coupling, conv) in cases
        diffusion = _kz_diffusion(coupling)
        linear = scheme isa AT.UpwindScheme
        footprint = AT.cs_surface_emission_footprint(
            PB.rm, PB.m, PB.am, PB.bm, PB.cm, MESH, objective;
            scheme, dt = 1800.0,
            base_emission_rates = linear ? nothing : fill(RATES, NSTEPS),
            _diffusion_kwargs(diffusion)..., _convection_kwargs(conv)...)
        adjoint = _dot_footprint(footprint, RATES)
        # The objective is affine in the emission scale for linear transport.
        # Limited schemes are piecewise smooth: their central difference must
        # not straddle a limiter branch switch (h = 1e-3 does for the PPM +
        # TM5 case, giving 9e-4 against the replay's own differences too).
        h = linear ? 0.5 : 1e-4
        fd = (prod_objective(scheme, conv, diffusion, 1 + h) -
              prod_objective(scheme, conv, diffusion, 1 - h)) / 2h
        @test abs(adjoint - fd) <= (linear ? 1e-9 : 1e-7) * abs(fd)
    end
end

# ---------------------------------------------------------------------------
# #60: checkpoint schedules replay the recording exactly with convection
# ---------------------------------------------------------------------------

@testset "limited-PPM checkpoint schedules match FullCheckpoint with convection (#60)" begin
    rng = MersenneTwister(3)
    base = [ntuple(_ -> MASS_SCALE * 1e-6 .* rand(rng, NC, NC), 6) for _ in 1:NSTEPS]
    objective = AT.CSLayerMeanObjective(2, 1, 1, NZ)
    @testset "$(nameof(typeof(limiter))), convection $conv" for
            limiter in (AT.MonotoneLimiter(), AT.CW84Limiter()), conv in (:tm5, :cmfmc)
        footprint(checkpoint) = AT.cs_surface_emission_footprint(
            PB.rm, PB.m, PB.am, PB.bm, PB.cm, MESH, objective;
            scheme = AT.PPMScheme(limiter), dt = 1800.0, base_emission_rates = base,
            checkpoint, _convection_kwargs(conv)...)
        full = footprint(AT.FullCheckpoint())
        for checkpoint in (AT.StrideCheckpoint(1), AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
            @test _footprint_maxrel(footprint(checkpoint), full) <= 1e-12
        end
    end
    # The other tape families and the diffusive boundary with diffusion.
    @testset "$(_scheme_label(scheme)), $(nameof(typeof(coupling))), convection $conv" for
            (scheme, coupling, conv) in (
                (AT.UpwindScheme(), AT.SplitSurfaceFluxCoupling(), :cmfmc),
                (AT.PPMScheme(AT.MonotoneLimiter()), AT.DiffusiveSurfaceFluxBoundary(), :tm5),
                (AT.LinRoodPPMScheme(), AT.DiffusiveSurfaceFluxBoundary(), :tm5))
        diffusion = _kz_diffusion(coupling)
        linear = scheme isa AT.UpwindScheme
        footprint(checkpoint) = AT.cs_surface_emission_footprint(
            PB.rm, PB.m, PB.am, PB.bm, PB.cm, MESH, objective;
            scheme, dt = 1800.0, base_emission_rates = linear ? nothing : base,
            checkpoint, _diffusion_kwargs(diffusion)..., _convection_kwargs(conv)...)
        full = footprint(AT.FullCheckpoint())
        for checkpoint in (AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
            @test _footprint_maxrel(footprint(checkpoint), full) <= 1e-12
        end
    end
end

# ---------------------------------------------------------------------------
# Tape layout: the recorders write the production step, and the byte
# estimate counts what they write
# ---------------------------------------------------------------------------

@testset "tape records the production step; cs_tape_byte_estimate counts it" begin
    nsteps = 3
    am, bm, cm = PB.am[1:nsteps], PB.bm[1:nsteps], PB.cm[1:nsteps]
    @testset "$(_scheme_label(scheme)), $(nameof(typeof(coupling))), convection $conv" for
            scheme in (AT.UpwindScheme(), AT.PPMScheme(AT.MonotoneLimiter()), AT.LinRoodPPMScheme()),
            coupling in COUPLINGS, conv in (:none, :tm5)
        diffusion = _kz_diffusion(coupling)
        convection = _convection_kwargs(conv)
        ops, _ = ADJ._record_cs_adjoint_tape(
            PB.rm, PB.m, am, bm, cm, MESH, scheme;
            cfl_limit = ADJ.DEFAULT_CFL_LIMIT, flux_scale = 1.0, dt = 1800.0,
            _diffusion_kwargs(diffusion)..., convection...)
        est = AT.cs_tape_byte_estimate(PB.m, am, bm, cm, MESH, scheme;
                                       diffusion_op = diffusion,
                                       convection_op = get(convection, :convection_op,
                                                           AT.NoConvection()))
        count_of(T) = count(op -> op isa T, ops)
        @test count_of(TapeMod._CSHaloRecord) == est.halo_records
        @test count_of(TapeMod._CSMidpointRecord) == est.midpoint_records == nsteps
        @test count_of(TapeMod._CSConvectionRecord) == est.convection_records
        # Split center: two half solves; boundary coupling: one full solve.
        solves = coupling isa AT.SplitSurfaceFluxCoupling ? 2 : 1
        @test count_of(TapeMod._CSDiffusionRecord) == est.diffusion_records == solves * nsteps
        # Every step closes with its end-of-step marker (issue #20), after
        # convection: the state the observation sampler sees.
        ends = findall(op -> op isa TapeMod._CSStepEndRecord, ops)
        @test length(ends) == est.step_records == nsteps
        @test [ops[i].step for i in ends] == 1:nsteps
        @test last(ops) isa TapeMod._CSStepEndRecord
        convections = findall(op -> op isa TapeMod._CSConvectionRecord, ops)
        @test all(ops[i + 1] isa TapeMod._CSStepEndRecord for i in convections)
        if !(scheme isa AT.LinRoodPPMScheme)
            @test count_of(TapeMod._CSSweepRecord) == est.sweep_records
            # Every split-sweep step opens with the step-start halo refresh,
            # which therefore sits between a step's end marker (after its
            # convection) and the next step's first X sweep.
            @test first(ops) isa TapeMod._CSHaloRecord && first(ops).dir == 1
            @test all(i == lastindex(ops) ||
                      (ops[i + 1] isa TapeMod._CSHaloRecord && ops[i + 1].dir == 1)
                      for i in ends)
        end
        mids = findall(op -> op isa TapeMod._CSMidpointRecord, ops)
        if coupling isa AT.DiffusiveSurfaceFluxBoundary
            # S(dt) → V(dt): the midpoint record precedes the single solve.
            @test all(ops[i + 1] isa TapeMod._CSDiffusionRecord && ops[i + 1].dt == 1800.0
                      for i in mids)
        else
            # V(dt/2) → S(dt) → V(dt/2).
            @test all(ops[i - 1] isa TapeMod._CSDiffusionRecord && ops[i - 1].dt == 900.0 &&
                      ops[i + 1] isa TapeMod._CSDiffusionRecord && ops[i + 1].dt == 900.0
                      for i in mids)
        end
    end
end

# ---------------------------------------------------------------------------
# #57: host tape storage keeps the air-mass and tracer-mass reads apart
# ---------------------------------------------------------------------------

# The issue #57 report's problem (unit masses; no convection or diffusion).
const PB57 = _parity_problem()

# An mmap tape that reads through device-side caches, as the CUDA extension
# sets it up for `CuArray` panels: the generic `_mmap_prepare_for_panels!`
# (the CPU method returns mmap views instead).
function _mmap_with_read_cache()
    storage = TapeMod.MmapCSTapeStorage()
    invoke(TapeMod._mmap_prepare_for_panels!,
           Tuple{TapeMod.MmapCSTapeStorage, NTuple{6}}, storage, PB57.m)
    return storage
end

@testset "host tape storage gives distinct air-mass and tracer-mass reads (#57)" begin
    @testset "slot reads ($(nameof(typeof(storage))))" for
            storage in (TapeMod.PinnedHostCSTapeStorage(), _mmap_with_read_cache())
        m_slot = TapeMod._stage_panels(storage, PB57.m)
        rm_slot = TapeMod._stage_panels(storage, PB57.rm)
        # Both operands of one limited-PPM sweep record, read as the reverse
        # loop reads them.
        m_read = TapeMod._tape_panels(m_slot, TapeMod.AirMassRead())
        rm_read = TapeMod._tape_panels(rm_slot, TapeMod.TracerMassRead())
        @test all(m_read[p] !== rm_read[p] for p in 1:6)
        @test all(m_read[p] == PB57.m[p] for p in 1:6)
        @test all(rm_read[p] == PB57.rm[p] for p in 1:6)
        # A read stays valid until the next read with the same role.
        TapeMod._tape_panels(rm_slot, TapeMod.TracerMassRead())
        @test all(m_read[p] == PB57.m[p] for p in 1:6)
        TapeMod.finalize_tape!(storage)
    end

    objective = AT.CSLayerMeanObjective(2, 2, 3, NZ)
    nsteps = 4
    am, bm, cm = PB57.am[1:nsteps], PB57.bm[1:nsteps], PB57.cm[1:nsteps]
    @testset "footprints, PPM $(nameof(typeof(limiter)))" for
            limiter in (AT.MonotoneLimiter(), AT.CW84Limiter())
        scheme = AT.PPMScheme(limiter)
        footprint(storage; kw...) = AT.cs_surface_emission_footprint(
            PB57.rm, PB57.m, am, bm, cm, MESH, objective;
            scheme, dt = 30.0, tape_storage = storage, kw...)
        for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
            reference = footprint(:device; checkpoint)
            for storage in (:pinned_host, :mmap)
                @test _footprint_maxrel(footprint(storage; checkpoint), reference) <= 1e-12
            end
        end
        # Stride and Revolve build their own storage per window, so the
        # device-cache read path of a GPU mmap tape is exercised with a
        # pre-built storage under FullCheckpoint.
        mmap_storage = _mmap_with_read_cache()
        @test _footprint_maxrel(footprint(mmap_storage), footprint(:device)) <= 1e-12
        TapeMod.finalize_tape!(mmap_storage)

        seed = ntuple(p -> (a = zeros(size(PB57.rm[p])); a[HP + 2, HP + 2, NZ] = 1.0; a), 6)
        from_seed(storage; kw...) = AT.cs_surface_emission_footprint_from_seed(
            seed, PB57.m, am, bm, cm, MESH;
            scheme, dt = 30.0, base_panels_rm0 = PB57.rm, tape_storage = storage, kw...)
        for checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
            @test _footprint_maxrel(from_seed(:pinned_host; checkpoint),
                                    from_seed(:device; checkpoint)) <= 1e-12
        end
    end

    @testset "4D-Var gradient, PPM MonotoneLimiter" begin
        scheme = AT.PPMScheme(AT.MonotoneLimiter())
        rng = MersenneTwister(5)
        control = AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:all, 1:nsteps; normalize = true),
            ntuple(_ -> 0.02 .* rand(rng, NC, NC), 6))
        observations = [AT.CSObservation(nsteps, objective, 0.09, 0.01),
                        AT.CSObservation(3, AT.CSColumnMeanObjective(4, 1, 2), 0.07, 0.01)]
        direction = ntuple(_ -> randn(rng, NC, NC), 6)
        shifted(h) = AT.CSSurfaceFluxControl(control.window,
            ntuple(p -> control.value[p] .+ h .* direction[p], 6);
            background = control.background, sigma = control.sigma)
        fourdvar(c; storage = :device) = AT.cs_surface_flux_4dvar(
            PB57.rm, PB57.m, am, bm, cm, MESH, observations, [c];
            scheme, dt = 30.0, tape_storage = storage)
        gd(result) = sum(sum(result.gradients[1][p] .* direction[p]) for p in 1:6)
        h = 1e-7
        fd = (fourdvar(shifted(h)).cost - fourdvar(shifted(-h)).cost) / 2h
        reference = gd(fourdvar(control))
        @test abs(reference - fd) <= 1e-8 * abs(fd)
        mmap_storage = _mmap_with_read_cache()
        for storage in (:pinned_host, mmap_storage)
            @test abs(gd(fourdvar(control; storage)) - reference) <= 1e-12 * abs(reference)
        end
        TapeMod.finalize_tape!(mmap_storage)
    end
end
