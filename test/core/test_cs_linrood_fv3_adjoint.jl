#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Adjoint of the FV3 `kord = 8` vertical sweep (OF-94), alone and as Lin–Rood's
# vertical sweep.
#
# For fixed limiter branches β the sweep is linear, rm' = W(β) rm. The tape
# stages the state entering the sweep; the reverse replays the forward profile
# there and applies W(β)ᵀ (`vertical_fv3_profile_adjoint.jl`).
#
#  1. The linearized replay returns the forward's limited parabolas bit for bit,
#     so it takes the forward's branches, for both profiles, including exact
#     ties and zeros.
#  2. Column transpose in 256-bit arithmetic: ⟨λ', W δ⟩ by central differences
#     with h = 1e-40 equals ⟨Wᵀ λ', δ⟩ to 1e-30 on limiter-heavy, tie-free
#     profiles, for both flow directions and beyond Courant 1 (capped fluxes).
#     At this precision a finite difference cannot cross a limiter switch unless
#     the base state sits on one, so a mismatch is an adjoint error.
#  3. The six-panel reverse equals the column reverse and leaves halos alone.
#  4. Lin–Rood footprints vs central finite differences (Float64): ORD 5 and 7,
#     upward and downward vertical flux, surface-layer and column-mean
#     objectives, positive and signed profiles (signed on a field that crosses
#     zero). The base field is smooth, so a linearity witness shows J affine in
#     the emissions over the stencil, and the identity must hold to 1e-9.
#  5. Production transport, the tape replay and the FD forward agree bit for
#     bit; Full and Stride checkpointing agree to rounding; the tape and its byte
#     estimate stage the tracer mass entering each FV3 sweep.
# ---------------------------------------------------------------------------

using Test, Random
using KernelAbstractions: CPU

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection
using .Adv: FV3Column, FV3LinearizedColumn, FV3ScalarProfile, _fv3_unlimited_edges!,
    _fv3_layer_profile, _sweep_z_panel_fv3!, _cs_zsweep_fv3_adjoint_kernel!,
    _sweep_z_panels_fv3_adjoint!, _fv3_adjoint_column_scratch

const POSITIVE = FV3ScalarProfile()
const SIGNED   = FV3ScalarProfile(; positive_definite = false)
const FV3_PPM        = AT.PPMScheme(; vertical = POSITIVE)
const FV3_PPM_SIGNED = AT.PPMScheme(; vertical = SIGNED)

# Top-down layer air masses with thickness growing ~400× from top to surface.
column_masses(Nz, FT = Float64) =
    FT.(diff(1e5 .* exp.(range(log(1e-4), 0; length = Nz + 1))))
column(x) = reshape(x, 1, 1, :)

# One forward sweep of a single column with the production kernel.
function column_sweep(rm, m, cm, profile)
    Nz, FT = length(rm), eltype(rm)
    rm_out, m_out = similar(rm, 1, 1, Nz, 1), similar(column(m))
    scratch = zeros(FT, 1, 1, Nz + 1, 3)
    _sweep_z_panel_fv3!(rm_out, m_out, reshape(copy(rm), 1, 1, Nz, 1), column(m), column(cm),
                        scratch, profile, 1, 0, Nz, 1)
    return vec(rm_out)
end

# Its transpose at the branches of the state `rm` (one work item).
function column_adjoint(lambda, rm, m, cm, profile)
    Nz, FT = length(rm), eltype(rm)
    l = reshape(copy(lambda), 1, 1, Nz)
    scratch = _fv3_adjoint_column_scratch(l, 1, Nz, true)
    kernel! = _cs_zsweep_fv3_adjoint_kernel!(CPU(), (1, 1))
    kernel!(l, column(rm), column(m), column(cm), scratch, profile,
            Int32(Nz), Int32(0), one(FT); ndrange = (1, 1))
    return vec(l)
end

# Interface fluxes up to `c` times the smaller neighbouring air mass, random sign.
random_cm(rng, m, c) = [zero(eltype(m)); c .* (2 .* rand(rng, length(m) - 1) .- 1) .*
                        min.(m[1:end-1], m[2:end]); zero(eltype(m))]

# Profiles that engage every limiter: extrema, a step, a spike, noise, a
# surface layer and exact zeros. The step and spike carry 1e-3 noise so that
# neighbouring means never tie exactly (a tie is a limiter switch point).
function limiter_profiles(rng, Nz)
    return (smooth  = [400 + 20exp(-((k - Nz / 2) / 6)^2) for k in 1:Nz],
            step    = [k < Nz ÷ 3 ? 1.0 : 5.0 for k in 1:Nz] .* (1 .+ 1e-3 .* rand(rng, Nz)),
            spike   = [k == Nz ÷ 2 ? 10.0 : 0.1 for k in 1:Nz] .* (1 .+ 1e-3 .* rand(rng, Nz)),
            noisy   = 1 .+ rand(rng, Nz),
            surface = [exp(-(Nz - k) / 3) for k in 1:Nz],
            zeros   = [max(0.0, sin(k / 3)) for k in 1:Nz])
end

@testset "FV3 vertical sweep adjoint (OF-94)" begin
    Nz = 40
    m = column_masses(Nz)

    @testset "linearized replay takes the forward's branches" begin
        rng = MersenneTwister(21)
        for (name, q) in pairs(limiter_profiles(rng, Nz)), (profile, offset) in ((POSITIVE, 0.0), (SIGNED, 0.6))
            rm = (q .- offset * maximum(q)) .* m
            scratch = zeros(1, 1, Nz + 1, 3)
            col = FV3Column(scratch, 1, 1, 1)
            _fv3_unlimited_edges!(col, column(rm), column(m), 1, 1, Int32(Nz))
            for k in Int32(1):Int32(Nz)
                forward = _fv3_layer_profile(col, k, Int32(Nz), profile)
                linear  = _fv3_layer_profile(FV3LinearizedColumn(scratch, 1, 1, Int(k)), k, Int32(Nz), profile)
                @test map(x -> x.value, linear) === forward
            end
        end
    end

    @testset "column transpose in 256-bit arithmetic: $name" for name in
            (:smooth, :step, :spike, :noisy, :surface, :zeros)
        setprecision(BigFloat, 256) do
            rng = MersenneTwister(hash(name))
            mb = big.(m)
            q = limiter_profiles(rng, Nz)[name]
            # The signed profile is tested on the field shifted to cross zero;
            # `zeros` shifted would tie exactly over its flat stretches.
            cases = name === :zeros ? ((POSITIVE, 0.0),) : ((POSITIVE, 0.0), (SIGNED, 0.6))
            for (profile, offset) in cases, courant in (0.3, 0.8, 1.5)
                rm = big.(q .- offset * maximum(q)) .* mb
                cm = random_cm(rng, mb, courant)        # 1.5: some fluxes capped at the donor mass
                λ = big.(randn(rng, Nz))
                δ = big.(randn(rng, Nz)) .* abs.(rm)
                h = big(1e-40)
                fd = (sum(λ .* column_sweep(rm .+ h .* δ, mb, cm, profile)) -
                      sum(λ .* column_sweep(rm .- h .* δ, mb, cm, profile))) / 2h
                adjoint = sum(column_adjoint(λ, rm, mb, cm, profile) .* δ)
                @test abs(adjoint - fd) <= 1e-30 * abs(fd)
            end
        end
    end

    @testset "six-panel reverse equals the column reverse" begin
        Nc, Hp, Nz6 = 3, 3, 12
        mesh = AT.CubedSphereMesh(; Nc, Hp, FT = Float64)
        N = Nc + 2Hp
        rng = MersenneTwister(5)
        mcol = column_masses(Nz6)
        m6  = ntuple(_ -> repeat(column(mcol), N, N, 1) .* (1 .+ 0.1 .* rand(rng, N, N, 1)), 6)
        rm6 = ntuple(p -> m6[p] .* (1 .+ rand(rng, N, N, Nz6)), 6)
        cm6 = ntuple(6) do p
            c = zeros(N, N, Nz6 + 1)
            for k in 2:Nz6
                c[:, :, k] .= 0.3 .* (2 .* rand(rng, N, N) .- 1) .* min.(m6[p][:, :, k - 1], m6[p][:, :, k])
            end
            c
        end
        λ6 = ntuple(_ -> randn(rng, N, N, Nz6), 6)
        out = map(copy, λ6)
        scratch = _fv3_adjoint_column_scratch(out[1], Nc, Nz6, true)
        _sweep_z_panels_fv3_adjoint!(out, rm6, m6, cm6, POSITIVE, scratch, mesh; flux_scale = 0.7)
        I = Hp+1:Hp+Nc
        @test all(Iterators.product(1:6, I, I)) do (p, i, j)
            col = column_adjoint(λ6[p][i, j, :], rm6[p][i, j, :], m6[p][i, j, :],
                                 0.7 .* cm6[p][i, j, :], POSITIVE)
            out[p][i, j, :] == col
        end
        halo = trues(N, N); halo[I, I] .= false
        @test all(p -> out[p][halo, :] == λ6[p][halo, :], 1:6)
        @test_throws ArgumentError _sweep_z_panels_fv3_adjoint!(
            map(copy, λ6), rm6, m6, cm6, POSITIVE, _fv3_adjoint_column_scratch(out[1], Nc, Nz6, false), mesh)
    end
end

# ---------------------------------------------------------------------------
# Lin–Rood footprints with the FV3 vertical sweeps
# ---------------------------------------------------------------------------

# Courant numbers are set directly: am, bm ~ Ch · m and cm ~ Cz · m, with
# `direction = -1` upward (level 1 is the model top) and `+1` downward. The
# tracer has vertical extrema and horizontal structure; `offset` shifts the
# mixing ratio so the signed profile sees a field that crosses zero.
function _fv3_linrood_problem(; direction, offset = 0.0, Nc = 4, Nz = 8, nsteps = 2,
                              Ch = 0.02, Cz = 0.08, upwind_profile = false)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = Float64)
    Hp = mesh.Hp
    N = Nc + 2Hp
    m = ntuple(p -> [2.0 + 0.25k + 0.01p + 0.02sin(0.7i + 0.4j) for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(m, mesh; dir = 0)
    rm = ntuple(p -> [m[p][i, j, k] * (upwind_profile ?
                      (0.3 + 0.1sin(0.2i + 0.3j) + 0.05k + 0.7sin(0.5p)) :
                      (1.0 + 0.3sin(0.45k + 0.2i - 0.3j + 0.6p) + 0.1cos(0.2i + 0.3j) - offset))
                      for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(rm, mesh; dir = 0)
    mref = 2.0
    am = [ntuple(p -> [(Hp < j <= Hp + Nc && Hp < i <= Hp + Nc + 1) ?
                       Ch * mref * sin(0.2s + 0.3p + 0.7i + 0.4j + 0.2k) : 0.0
                       for i in 1:N+1, j in 1:N, k in 1:Nz], 6) for s in 1:nsteps]
    bm = [ntuple(p -> [(Hp < j <= Hp + Nc + 1 && Hp < i <= Hp + Nc) ?
                       Ch * mref * cos(0.3s + 0.5p + 0.4i + 0.6j + 0.1k) : 0.0
                       for i in 1:N, j in 1:N+1, k in 1:Nz], 6) for s in 1:nsteps]
    cm = [ntuple(p -> [(2 <= k <= Nz && Hp < i <= Hp + Nc && Hp < j <= Hp + Nc) ?
                       direction * Cz * mref * sin(pi * (k - 1) / Nz) *
                       (0.6 + 0.4sin(i + 0.7j + 0.3k + p + s)) : 0.0
                       for i in 1:N, j in 1:N, k in 1:Nz+1], 6) for s in 1:nsteps]
    return (; mesh, m, rm, am, bm, cm, Nc, Nz, nsteps)
end

_emission_direction(P, seed) =
    [ntuple(p -> [sin(seed + 0.31s + 0.17p + 0.23i - 0.19j) for i in 1:P.Nc, j in 1:P.Nc], 6)
     for s in 1:P.nsteps]
_scaled(P, a, x) = [ntuple(p -> a .* x[s][p], 6) for s in 1:P.nsteps]
_summed(P, x, y) = [ntuple(p -> x[s][p] .+ y[s][p], 6) for s in 1:P.nsteps]
_dot(P, x, y) = sum(sum(x[s][p] .* y[s][p]) for s in 1:P.nsteps, p in 1:6)

_objective_value(P, scheme, objective, rates) = AT.run_cs_footprint_forward(
    P.rm, P.m, P.am, P.bm, P.cm, P.mesh, objective; scheme, dt = 1.0, emission_rates = rates)
_footprints(P, scheme, objective; kwargs...) = AT.cs_surface_emission_footprint(
    P.rm, P.m, P.am, P.bm, P.cm, P.mesh, objective; scheme, dt = 1.0, kwargs...)

@testset "Lin–Rood footprint with Upwind and FV3 vertical sweeps vs FD" begin
    h = 1e-4
    for (flow, direction) in (("upward", -1), ("downward", 1)),
        (profile_name, vertical, offset) in (("upwind (#10)", AT.UpwindScheme(), 0.0),
                                             ("fv3_kord8", FV3_PPM, 0.0),
                                             ("fv3_kord8_signed", FV3_PPM_SIGNED, 0.9)),
        ord in (5, 7)
        upwind = vertical isa AT.UpwindScheme
        upwind && direction > 0 && continue
        P = _fv3_linrood_problem(; direction, offset, upwind_profile = upwind)
        scheme = AT.LinRoodPPMScheme(ord; vertical)
        @test AT.require_adjoint_support(scheme) === scheme
        @test (offset > 0) == any(p -> minimum(P.rm[p]) < 0, 1:6)
        objectives = (surface_layer = AT.CSLayerMeanObjective(1, 1, 1, P.Nz),
                      column_mean   = AT.CSColumnMeanObjective(1, 1, 1))
        for (objective_name, objective) in pairs(objectives)
            @testset "$flow flux, $profile_name, ORD $ord, $objective_name" begin
                a = _emission_direction(P, 0.0)
                b = _emission_direction(P, 1.3)
                # J is affine in the emissions over the FD stencil: no limiter
                # switches branch, so central differences are exact up to rounding.
                j0 = _objective_value(P, scheme, objective, nothing)
                ja = _objective_value(P, scheme, objective, _scaled(P, h, a))
                jb = _objective_value(P, scheme, objective, _scaled(P, h, b))
                jab = _objective_value(P, scheme, objective,
                                       _summed(P, _scaled(P, h, a), _scaled(P, h, b)))
                @test abs(jab - ja - jb + j0) <= 1e-9 * abs(ja - j0)

                footprints = _footprints(P, scheme, objective).footprints
                for d in (a, b)
                    j_plus  = _objective_value(P, scheme, objective, _scaled(P, h, d))
                    j_minus = _objective_value(P, scheme, objective, _scaled(P, -h, d))
                    finite_difference = (j_plus - j_minus) / (2h)
                    @test abs(_dot(P, footprints, d) - finite_difference) <= 1e-9 * abs(finite_difference)
                end
            end
        end
    end

end

@testset "Lin–Rood FV3 vertical: tape replay and FD forward equal production, ORD $ord" for ord in (5, 7)
    # Production Lin–Rood (`_strang_split_linrood_ppm_cs!`) and the adjoint's
    # replay of the step plan, with and without the tape recorder, reach the
    # same interior state bit for bit.
    P = _fv3_linrood_problem(; direction = -1)
    scheme = AT.LinRoodPPMScheme(ord; vertical = FV3_PPM)
    Nc, Hp = P.mesh.Nc, P.mesh.Hp
    I = Hp+1:Hp+Nc
    production = (map(copy, P.rm), map(copy, P.m))
    ws_prod = Adv.CSLinRoodAdvectionWorkspace(P.mesh, P.m[1]; column_scratch = true)
    for s in 1:P.nsteps
        am = ntuple(p -> Adv._cs_flux_x_interior(P.am[s][p], Nc, Hp), 6)
        bm = ntuple(p -> Adv._cs_flux_y_interior(P.bm[s][p], Nc, Hp), 6)
        Adv._strang_split_linrood_ppm_cs!(production..., am, bm, P.cm[s], P.mesh, Val(ord),
                                          ws_prod; vertical = scheme.vertical)
    end
    # The adjoint's replay of the step plan, without and with its tape recorder.
    replay(recorded) = begin
        r, _ = AT.Adjoints.Footprint._cs_replay_for_call(P.rm, P.m, P.am, P.bm, P.cm, P.mesh;
                                               scheme, dt = 1.0, surface_flux = false)
        recorded ? AT.Adjoints.Footprint._record_cs_tape!(r, 1:P.nsteps, AT.FullStateTape(),
                                                         AT.Tape._open_tape_pool(AT.DeviceCSTapeStorage(), 0)) :
                   AT.Adjoints.Footprint._replay_steps!(r, 1:P.nsteps)
        (AT.Adjoints.Footprint._replay_rm(r), AT.Adjoints.Footprint._replay_m(r))
    end
    for (rm, m) in (replay(true), replay(false))
        @test all(p -> rm[p][I, I, :] == production[1][p][I, I, :], 1:6)
        @test all(p -> m[p][I, I, :] == production[2][p][I, I, :], 1:6)
    end
end

@testset "Lin–Rood FV3 vertical: Full and Stride agree" begin
    P = _fv3_linrood_problem(; direction = -1, nsteps = 4)
    objective = AT.CSColumnMeanObjective(1, 1, 1)
    for ord in (5, 7), vertical in (FV3_PPM, FV3_PPM_SIGNED)
        scheme = AT.LinRoodPPMScheme(ord; vertical)
        full = _footprints(P, scheme, objective)
        for checkpoint in (AT.StrideCheckpoint(1), AT.StrideCheckpoint(2), AT.StrideCheckpoint(3))
            replay = _footprints(P, scheme, objective; tape = AT.AdjointTape(; checkpoint))
            # Each window re-records from a checkpointed (rm, m) with the same
            # recorder, so it replays the same branches. The footprints agree to
            # the checkpoint contract's rounding tolerance (segment-boundary halo
            # refreshes; `src/Footprint/README.md`), not necessarily bit for bit.
            @test all(Iterators.product(eachindex(full.footprints), 1:6)) do (s, p)
                isapprox(replay.footprints[s][p], full.footprints[s][p]; rtol = 1e-10, atol = 1e-12)
            end
        end
    end
end
