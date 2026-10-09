#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 25 — LinRood CS adjoint, kernel-level transposition tests.
#
# Verifies the discrete transpose of each forward LinRood kernel by the
# adjoint identity
#     ⟨y, F·x⟩ = ⟨Fᵀ·y, x⟩
# evaluated on random inputs. The identity must hold to floating-point
# precision because each forward kernel is linear in the differentiated
# inputs once velocities (`am`, `bm`) are fixed.
#
# Commit 1 covers `_linrood_update_kernel!` (the averaged-flux update).
# Commits 2 and 3 will add `_pre_advect_*_kernel!` and the four face
# kernels to this file as they ship.
# ---------------------------------------------------------------------------

using Test
using Random
using KernelAbstractions: get_backend, synchronize

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# Sum of element-wise products over the interior region `[Hp+1 : Hp+Nc,
# Hp+1 : Hp+Nc, :]` of two haloed cubed-sphere arrays.
function _inner_interior(a::AbstractArray{FT, 3}, b::AbstractArray{FT, 3},
                          Nc::Int, Hp::Int) where {FT}
    s = zero(FT)
    @inbounds for k in axes(a, 3)
        for j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
            s += a[i, j, k] * b[i, j, k]
        end
    end
    return s
end

# Sum of element-wise products over the full (already non-haloed) face
# array.
function _inner_full(a::AbstractArray{FT}, b::AbstractArray{FT}) where {FT}
    s = zero(FT)
    @inbounds for I in eachindex(a)
        s += a[I] * b[I]
    end
    return s
end

# Build a tiny CS panel layout for one face: random rm, m, am, bm,
# fx_in, fx_out, fy_in, fy_out arrays sized exactly the way
# `_linrood_update_kernel!` expects.
function _linrood_update_panel_inputs(; Nc=4, Hp=3, Nz=3, FT=Float64, seed=1)
    rng = MersenneTwister(seed)
    N = Nc + 2Hp
    rm = randn(rng, FT, N, N, Nz)
    m  = FT(2) .+ rand(rng, FT, N, N, Nz)  # strictly positive
    am = FT(0.01) .* randn(rng, FT, Nc + 1, Nc, Nz)
    bm = FT(0.01) .* randn(rng, FT, Nc, Nc + 1, Nz)
    fx_in  = randn(rng, FT, Nc + 1, Nc, Nz)
    fx_out = randn(rng, FT, Nc + 1, Nc, Nz)
    fy_in  = randn(rng, FT, Nc, Nc + 1, Nz)
    fy_out = randn(rng, FT, Nc, Nc + 1, Nz)
    return (; rm, m, am, bm, fx_in, fx_out, fy_in, fy_out)
end

# Run the forward `_linrood_update_kernel!` directly on the inputs and
# return the (rm_new, m_new) interior result, written into freshly
# allocated haloed arrays so the inner-product helpers can be used
# without adjustment.
function _run_linrood_update_forward(inputs, mesh::AT.CubedSphereMesh{FT}) where {FT}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(inputs.rm, 3)
    N = Nc + 2Hp
    rm_new = zeros(FT, N, N, Nz)
    m_new  = zeros(FT, N, N, Nz)
    backend = get_backend(rm_new)
    k! = Adv._linrood_update_kernel!(backend, 256)
    k!(rm_new, m_new,
       inputs.rm, inputs.m, inputs.am, inputs.bm,
       inputs.fx_in, inputs.fx_out, inputs.fy_in, inputs.fy_out, Hp;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return (rm_new, m_new)
end

@testset "Plan 25 Commit 1 — LinRood update kernel adjoint" begin
    @testset "transposition identity ($(FT))" for FT in (Float64, Float32)
        Nc = 4; Hp = 3; Nz = 3
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        inputs = _linrood_update_panel_inputs(; Nc, Hp, Nz, FT, seed=42)

        # Forward
        rm_new, m_new = _run_linrood_update_forward(inputs, mesh)

        # `_linrood_update_kernel!` is affine in `(rm, m, fx_*, fy_*)`
        # for fixed velocities: `m_new = m + (am_w - am_e) + (bm_s - bm_n)`
        # carries a constant velocity-divergence offset that depends
        # only on the meteo tape. The adjoint identity holds for the
        # **linear** part of the map only, so we subtract the
        # zero-tracer-state baseline before forming the LHS inner
        # product.
        zero_inputs = (
            rm     = zero(inputs.rm),
            m      = zero(inputs.m),
            am     = inputs.am,
            bm     = inputs.bm,
            fx_in  = zero(inputs.fx_in),
            fx_out = zero(inputs.fx_out),
            fy_in  = zero(inputs.fy_in),
            fy_out = zero(inputs.fy_out),
        )
        rm_new_zero, m_new_zero = _run_linrood_update_forward(zero_inputs, mesh)
        rm_new_linear = rm_new .- rm_new_zero
        m_new_linear  = m_new  .- m_new_zero

        # Random adjoint seed on the outputs. Halo cells of the seed are
        # irrelevant — the adjoint kernel only reads interior `[ii, jj]`
        # — but we keep them zero so the LHS inner product is restricted
        # to the interior region by construction.
        rng = MersenneTwister(43)
        lambda_rm_new = zeros(FT, N, N, Nz)
        lambda_m_new  = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
            lambda_rm_new[i, j, k] = randn(rng, FT)
            lambda_m_new[i, j, k]  = randn(rng, FT)
        end

        # Adjoint outputs (initialised to zero — the kernel only
        # accumulates).
        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        lambda_fx_in  = zeros(FT, Nc + 1, Nc, Nz)
        lambda_fx_out = zeros(FT, Nc + 1, Nc, Nz)
        lambda_fy_in  = zeros(FT, Nc, Nc + 1, Nz)
        lambda_fy_out = zeros(FT, Nc, Nc + 1, Nz)

        Adv.apply_linrood_update_adjoint!(
            lambda_rm, lambda_m,
            lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
            lambda_rm_new, lambda_m_new,
            inputs.am, inputs.bm, mesh,
        )

        # LHS: ⟨y, F·x⟩ — sum over interior outputs only, with the
        # affine constant subtracted (see comment above).
        lhs = _inner_interior(lambda_rm_new, rm_new_linear, Nc, Hp) +
              _inner_interior(lambda_m_new,  m_new_linear,  Nc, Hp)

        # RHS: ⟨Fᵀ·y, x⟩ — sum the six adjoint inputs against the
        # forward inputs. Cell-centred terms restricted to interior;
        # face terms span the full face array.
        rhs = _inner_interior(lambda_rm, inputs.rm, Nc, Hp) +
              _inner_interior(lambda_m,  inputs.m,  Nc, Hp) +
              _inner_full(lambda_fx_in,  inputs.fx_in)  +
              _inner_full(lambda_fx_out, inputs.fx_out) +
              _inner_full(lambda_fy_in,  inputs.fy_in)  +
              _inner_full(lambda_fy_out, inputs.fy_out)

        tol = FT === Float64 ? 1e-12 : 1f-5
        @test isapprox(lhs, rhs; atol=tol, rtol=tol)
    end

    @testset "adjoint zeros on zero seed (update kernel)" begin
        # Sanity check: if the seed is zero, no adjoint accumulators
        # should be touched.
        FT = Float64
        Nc = 4; Hp = 3; Nz = 3
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        inputs = _linrood_update_panel_inputs(; Nc, Hp, Nz, FT, seed=7)

        lambda_rm_new = zeros(FT, N, N, Nz)
        lambda_m_new  = zeros(FT, N, N, Nz)

        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        lambda_fx_in  = zeros(FT, Nc + 1, Nc, Nz)
        lambda_fx_out = zeros(FT, Nc + 1, Nc, Nz)
        lambda_fy_in  = zeros(FT, Nc, Nc + 1, Nz)
        lambda_fy_out = zeros(FT, Nc, Nc + 1, Nz)

        Adv.apply_linrood_update_adjoint!(
            lambda_rm, lambda_m,
            lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
            lambda_rm_new, lambda_m_new,
            inputs.am, inputs.bm, mesh,
        )

        @test all(iszero, lambda_rm)
        @test all(iszero, lambda_m)
        @test all(iszero, lambda_fx_in)
        @test all(iszero, lambda_fx_out)
        @test all(iszero, lambda_fy_in)
        @test all(iszero, lambda_fy_out)
    end
end

# ---------------------------------------------------------------------------
# Commit 2 — pre-advect kernel adjoints
#
# The pre-advect kernels have a different shape from the update kernel:
# they compute `q = _safe_mixing_ratio(rm + bm·fy_face_div, m + bm_div)`,
# which is smooth in `(rm, m, fy_face)` above the `100·eps` threshold
# and exactly zero below. Treat the operator as the LINEAR-ABOVE-
# THRESHOLD map; the transposition identity holds for the linear part.
# ---------------------------------------------------------------------------

# Run one panel of the forward `_pre_advect_y_kernel!` and return the
# q_i array as written into a haloed buffer.
function _run_pre_advect_y_forward(rm, m, bm, fy_face, mesh::AT.CubedSphereMesh{FT}) where {FT}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm, 3)
    N = Nc + 2Hp
    q_i = zeros(FT, N, N, Nz)
    backend = get_backend(q_i)
    k! = Adv._pre_advect_y_kernel!(backend, 256)
    k!(q_i, rm, m, bm, fy_face, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return q_i
end

function _run_pre_advect_x_forward(rm, m, am, fx_face, mesh::AT.CubedSphereMesh{FT}) where {FT}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm, 3)
    N = Nc + 2Hp
    q_j = zeros(FT, N, N, Nz)
    backend = get_backend(q_j)
    k! = Adv._pre_advect_x_kernel!(backend, 256)
    k!(q_j, rm, m, am, fx_face, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return q_j
end

# Build typical pre-advect inputs: rm of arbitrary sign, m strictly
# positive and well above the safe-mixing-ratio threshold, small mass
# fluxes so that m_new = m + (bm_s - bm_n) stays positive.
function _pre_advect_y_inputs(; Nc=4, Hp=3, Nz=3, FT=Float64, seed=11)
    rng = MersenneTwister(seed)
    N = Nc + 2Hp
    rm = randn(rng, FT, N, N, Nz)
    m  = FT(2) .+ rand(rng, FT, N, N, Nz)
    bm = FT(0.01) .* randn(rng, FT, Nc, Nc + 1, Nz)
    fy_face = randn(rng, FT, Nc, Nc + 1, Nz)
    return (; rm, m, bm, fy_face)
end

function _pre_advect_x_inputs(; Nc=4, Hp=3, Nz=3, FT=Float64, seed=12)
    rng = MersenneTwister(seed)
    N = Nc + 2Hp
    rm = randn(rng, FT, N, N, Nz)
    m  = FT(2) .+ rand(rng, FT, N, N, Nz)
    am = FT(0.01) .* randn(rng, FT, Nc + 1, Nc, Nz)
    fx_face = randn(rng, FT, Nc + 1, Nc, Nz)
    return (; rm, m, am, fx_face)
end

# Compute the analytical Jacobian-vector product dq = (∂F/∂x)·dx for
# `_pre_advect_y_kernel!` at state `(rm, m, bm, fy_face)` and perturbation
# `(drm, dm, dfy)`. Used to test the adjoint via the linearization
# identity ⟨lambda_q, dq⟩ = ⟨(∂F/∂x)ᵀ·lambda_q, dx⟩.
function _pre_advect_y_jvp(rm, m, bm, fy_face, drm, dm, dfy,
                            mesh::AT.CubedSphereMesh{FT}) where {FT}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm, 3)
    N = Nc + 2Hp
    dq = zeros(FT, N, N, Nz)
    thresh = FT(100) * eps(FT)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        ii = Hp + i; jj = Hp + j
        bm_s = bm[i, j,     k]
        bm_n = bm[i, j + 1, k]
        m_new = m[ii, jj, k] + bm_s - bm_n
        if m_new > thresh
            rm_new = rm[ii, jj, k] +
                     bm_s * fy_face[i, j, k] - bm_n * fy_face[i, j + 1, k]
            inv_m_new = one(FT) / m_new
            q = rm_new * inv_m_new
            drm_new = drm[ii, jj, k] +
                      bm_s * dfy[i, j, k] - bm_n * dfy[i, j + 1, k]
            dm_new = dm[ii, jj, k]   # bm is fixed
            dq[ii, jj, k] = (drm_new - q * dm_new) * inv_m_new
        end
    end
    return dq
end

function _pre_advect_x_jvp(rm, m, am, fx_face, drm, dm, dfx,
                            mesh::AT.CubedSphereMesh{FT}) where {FT}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm, 3)
    N = Nc + 2Hp
    dq = zeros(FT, N, N, Nz)
    thresh = FT(100) * eps(FT)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        ii = Hp + i; jj = Hp + j
        am_w = am[i,     j, k]
        am_e = am[i + 1, j, k]
        m_new = m[ii, jj, k] + am_w - am_e
        if m_new > thresh
            rm_new = rm[ii, jj, k] +
                     am_w * fx_face[i, j, k] - am_e * fx_face[i + 1, j, k]
            inv_m_new = one(FT) / m_new
            q = rm_new * inv_m_new
            drm_new = drm[ii, jj, k] +
                      am_w * dfx[i, j, k] - am_e * dfx[i + 1, j, k]
            dm_new = dm[ii, jj, k]
            dq[ii, jj, k] = (drm_new - q * dm_new) * inv_m_new
        end
    end
    return dq
end

@testset "Plan 25 Commit 2 — pre-advect kernel adjoints" begin
    # The forward `_pre_advect_y_kernel!` is rational in `m` (via the
    # 1/m_new factor inside `_safe_mixing_ratio`), so the strict-linear
    # transposition identity ⟨y, F·x⟩ = ⟨Fᵀ·y, x⟩ does NOT apply to
    # absolute states. The adjoint must instead transpose the
    # Frechet derivative dF/dx at the current state. We verify
    #     ⟨lambda_q, dF/dx · dx⟩ = ⟨(dF/dx)ᵀ · lambda_q, dx⟩
    # for any random perturbation `dx`, with the analytical JVP
    # computed by `_pre_advect_{x,y}_jvp` above.

    @testset "pre_advect_y JVP/VJP identity ($(FT))" for FT in (Float64, Float32)
        Nc = 4; Hp = 3; Nz = 3
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        in_ = _pre_advect_y_inputs(; Nc, Hp, Nz, FT, seed=21)

        # VJP from the adjoint kernel.
        rng = MersenneTwister(22)
        lambda_q = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
            lambda_q[i, j, k] = randn(rng, FT)
        end

        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        lambda_fy_face = zeros(FT, Nc, Nc + 1, Nz)

        Adv.apply_pre_advect_y_adjoint!(
            lambda_rm, lambda_m, lambda_fy_face,
            lambda_q, in_.rm, in_.m, in_.bm, in_.fy_face, mesh,
        )

        # JVP from the analytical Frechet derivative against a random
        # perturbation `dx = (drm, dm, dfy)`.
        drm = randn(rng, FT, N, N, Nz)
        dm  = randn(rng, FT, N, N, Nz)
        dfy = randn(rng, FT, Nc, Nc + 1, Nz)
        dq  = _pre_advect_y_jvp(in_.rm, in_.m, in_.bm, in_.fy_face,
                                 drm, dm, dfy, mesh)

        lhs = _inner_interior(lambda_q, dq, Nc, Hp)
        rhs = _inner_interior(lambda_rm, drm, Nc, Hp) +
              _inner_interior(lambda_m,  dm,  Nc, Hp) +
              _inner_full(lambda_fy_face, dfy)

        tol = FT === Float64 ? 1e-12 : 1f-5
        @test isapprox(lhs, rhs; atol=tol, rtol=tol)
    end

    @testset "pre_advect_x JVP/VJP identity ($(FT))" for FT in (Float64, Float32)
        Nc = 4; Hp = 3; Nz = 3
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        in_ = _pre_advect_x_inputs(; Nc, Hp, Nz, FT, seed=31)

        rng = MersenneTwister(32)
        lambda_q = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
            lambda_q[i, j, k] = randn(rng, FT)
        end

        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        lambda_fx_face = zeros(FT, Nc + 1, Nc, Nz)

        Adv.apply_pre_advect_x_adjoint!(
            lambda_rm, lambda_m, lambda_fx_face,
            lambda_q, in_.rm, in_.m, in_.am, in_.fx_face, mesh,
        )

        drm = randn(rng, FT, N, N, Nz)
        dm  = randn(rng, FT, N, N, Nz)
        dfx = randn(rng, FT, Nc + 1, Nc, Nz)
        dq  = _pre_advect_x_jvp(in_.rm, in_.m, in_.am, in_.fx_face,
                                 drm, dm, dfx, mesh)

        lhs = _inner_interior(lambda_q, dq, Nc, Hp)
        rhs = _inner_interior(lambda_rm, drm, Nc, Hp) +
              _inner_interior(lambda_m,  dm,  Nc, Hp) +
              _inner_full(lambda_fx_face, dfx)

        tol = FT === Float64 ? 1e-12 : 1f-5
        @test isapprox(lhs, rhs; atol=tol, rtol=tol)
    end

    @testset "small-mass column zeroes the gradient (pre-advect-y)" begin
        # Confirm the `m_new <= 100·eps(FT)` branch returns zero
        # gradient (matching the forward `_safe_mixing_ratio` zero
        # output). Construct a column with m below threshold and bm
        # such that m_new stays below threshold; the adjoint must NOT
        # write into any of (lambda_rm, lambda_m, lambda_fy_face) for
        # that cell.
        FT = Float64
        Nc = 2; Hp = 3; Nz = 1
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rm = zeros(FT, N, N, Nz)
        m  = zeros(FT, N, N, Nz)  # zero air mass everywhere
        bm = zeros(FT, Nc, Nc + 1, Nz)
        fy_face = randn(MersenneTwister(99), FT, Nc, Nc + 1, Nz)

        # Seed adjoint with non-zero values to be sure we'd notice a
        # spurious write.
        lambda_q_i = ones(FT, N, N, Nz)
        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        lambda_fy_face = zeros(FT, Nc, Nc + 1, Nz)

        Adv.apply_pre_advect_y_adjoint!(
            lambda_rm, lambda_m, lambda_fy_face,
            lambda_q_i, rm, m, bm, fy_face, mesh,
        )

        @test all(iszero, lambda_rm)
        @test all(iszero, lambda_m)
        @test all(iszero, lambda_fy_face)
    end
end

# ===========================================================================
# Commit 3 — `_ppm_x/y_face_from_q_kernel!` adjoints (ORD=5)
#
# The forward face kernels are piecewise-smooth, branch-rich
# compositions (Huynh constraint clamp; apply_monotonicity flatten;
# parabolic ppm_face_value with α-form donor mass denominator). We
# verify the kernel-level adjoint via the JVP/VJP identity, with the
# Jacobian-vector product approximated by CENTERED FINITE DIFFERENCES.
# This is the standard adjoint verification protocol for branchy
# rational maps: pick a state, pick a random perturbation, run the
# adjoint, and check ⟨lambda, FD-JVP⟩ ≈ ⟨adjoint, perturbation⟩.
# ===========================================================================

# Centered FD JVP for the X-direction `_ppm_x_face_from_q_kernel!`.
# Returns a (Nc+1, Nc, Nz) face array of dF·δq.
function _ppm_x_face_from_q_fd_jvp(q, am, m, dq, mesh::AT.CubedSphereMesh{FT};
                                    eps_fd, ord::Val{ORD}=Val(5)) where {FT, ORD}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(q, 3)
    fx_plus  = zeros(FT, Nc + 1, Nc, Nz)
    fx_minus = zeros(FT, Nc + 1, Nc, Nz)
    backend = get_backend(q)
    k! = Adv._ppm_x_face_from_q_kernel!(backend, 256)
    q_plus  = q .+ FT(eps_fd) .* dq
    q_minus = q .- FT(eps_fd) .* dq
    k!(fx_plus,  q_plus,  am, m, Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    k!(fx_minus, q_minus, am, m, Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    synchronize(backend)
    return (fx_plus .- fx_minus) ./ (FT(2) * FT(eps_fd))
end

function _ppm_y_face_from_q_fd_jvp(q, bm, m, dq, mesh::AT.CubedSphereMesh{FT};
                                    eps_fd, ord::Val{ORD}=Val(5)) where {FT, ORD}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(q, 3)
    fy_plus  = zeros(FT, Nc, Nc + 1, Nz)
    fy_minus = zeros(FT, Nc, Nc + 1, Nz)
    backend = get_backend(q)
    k! = Adv._ppm_y_face_from_q_kernel!(backend, 256)
    q_plus  = q .+ FT(eps_fd) .* dq
    q_minus = q .- FT(eps_fd) .* dq
    k!(fy_plus,  q_plus,  bm, m, Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
    k!(fy_minus, q_minus, bm, m, Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
    synchronize(backend)
    return (fy_plus .- fy_minus) ./ (FT(2) * FT(eps_fd))
end

@testset "Plan 25 Commit 3 — PPM `_from_q` face kernel adjoints (ORD=5)" begin
    @testset "X face_from_q VJP vs FD JVP" begin
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(101)
        # Smooth field so the limiter/monotonicity branches are
        # mostly inactive — keeps the FD JVP a clean linearization.
        q = FT.([sin(0.13i + 0.21j + 0.07k) for i in 1:N, j in 1:N, k in 1:Nz])
        am = FT(0.02) .* randn(rng, FT, Nc + 1, Nc, Nz)
        m  = FT(2) .+ rand(rng, FT, N, N, Nz)

        # Adjoint seed on the face array.
        lambda_fx_face = randn(rng, FT, Nc + 1, Nc, Nz)
        lambda_q = zeros(FT, N, N, Nz)
        Adv.apply_ppm_x_face_from_q_adjoint!(
            lambda_q, lambda_fx_face, q, am, m, mesh, Val(5),
        )

        # Random q-perturbation; only stencil-reachable interior + halo
        # cells participate so we exercise both interior face writes
        # and halo cell contributions.
        dq = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_x_face_from_q_fd_jvp(q, am, m, dq, mesh; eps_fd=1e-6)

        lhs = _inner_full(lambda_fx_face, fd_jvp)
        # Sum lambda_q against dq over the FULL haloed array (the
        # adjoint may write into halo cells via the stencil; the
        # FD JVP also reads halo cells of q).
        rhs = sum(lambda_q .* dq)

        @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-6)
    end

    @testset "Y face_from_q VJP vs FD JVP" begin
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(202)
        q = FT.([sin(0.11i - 0.17j + 0.05k) for i in 1:N, j in 1:N, k in 1:Nz])
        bm = FT(0.02) .* randn(rng, FT, Nc, Nc + 1, Nz)
        m  = FT(2) .+ rand(rng, FT, N, N, Nz)

        lambda_fy_face = randn(rng, FT, Nc, Nc + 1, Nz)
        lambda_q = zeros(FT, N, N, Nz)
        Adv.apply_ppm_y_face_from_q_adjoint!(
            lambda_q, lambda_fy_face, q, bm, m, mesh, Val(5),
        )

        dq = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_y_face_from_q_fd_jvp(q, bm, m, dq, mesh; eps_fd=1e-6)

        lhs = _inner_full(lambda_fy_face, fd_jvp)
        rhs = sum(lambda_q .* dq)

        @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-6)
    end

    @testset "X face_from_q ORD=7 VJP vs FD JVP (panel-edge boundary correction)" begin
        # ORD=7 differs from ORD=5 only at panel-edge faces
        # (`face_idx ∈ {1, Nc+1}`) where the forward path overrides the
        # PPM edges with the linear discontinuous extrapolation. The
        # interior is bit-equal to ORD=5; this test exercises BOTH
        # interior and panel-edge faces simultaneously through the same
        # JVP-vs-VJP identity used for ORD=5.
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(7101)
        q = FT.([sin(0.13i + 0.21j + 0.07k) for i in 1:N, j in 1:N, k in 1:Nz])
        am = FT(0.02) .* randn(rng, FT, Nc + 1, Nc, Nz)
        m  = FT(2) .+ rand(rng, FT, N, N, Nz)

        lambda_fx_face = randn(rng, FT, Nc + 1, Nc, Nz)
        lambda_q = zeros(FT, N, N, Nz)
        Adv.apply_ppm_x_face_from_q_adjoint!(
            lambda_q, lambda_fx_face, q, am, m, mesh, Val(7),
        )

        dq = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_x_face_from_q_fd_jvp(q, am, m, dq, mesh; eps_fd=1e-6, ord=Val(7))

        lhs = _inner_full(lambda_fx_face, fd_jvp)
        rhs = sum(lambda_q .* dq)
        @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-6)
    end

    @testset "Y face_from_q ORD=7 VJP vs FD JVP (panel-edge boundary correction)" begin
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(7202)
        q = FT.([sin(0.11i - 0.17j + 0.05k) for i in 1:N, j in 1:N, k in 1:Nz])
        bm = FT(0.02) .* randn(rng, FT, Nc, Nc + 1, Nz)
        m  = FT(2) .+ rand(rng, FT, N, N, Nz)

        lambda_fy_face = randn(rng, FT, Nc, Nc + 1, Nz)
        lambda_q = zeros(FT, N, N, Nz)
        Adv.apply_ppm_y_face_from_q_adjoint!(
            lambda_q, lambda_fy_face, q, bm, m, mesh, Val(7),
        )

        dq = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_y_face_from_q_fd_jvp(q, bm, m, dq, mesh; eps_fd=1e-6, ord=Val(7))

        lhs = _inner_full(lambda_fy_face, fd_jvp)
        rhs = sum(lambda_q .* dq)
        @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-6)
    end

    @testset "ORD=7 interior is bit-equal to ORD=5 (no panel-edge cells in scope)" begin
        # Build a config where the stencil never touches face_idx ∈ {1, Nc+1}.
        # We do this by zeroing lambda_fx_face/lambda_fy_face at those
        # columns / rows: ∂L/∂q is then a pure sum over interior faces,
        # which the ORD=7 adjoint must produce bit-equal to ORD=5.
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp
        rng = MersenneTwister(7303)
        q = FT.([sin(0.13i + 0.21j + 0.07k) for i in 1:N, j in 1:N, k in 1:Nz])
        am = FT(0.02) .* randn(rng, FT, Nc + 1, Nc, Nz)
        m  = FT(2) .+ rand(rng, FT, N, N, Nz)
        lambda_fx_face = randn(rng, FT, Nc + 1, Nc, Nz)
        # Zero the two panel-edge face columns.
        lambda_fx_face[1, :, :] .= 0
        lambda_fx_face[Nc + 1, :, :] .= 0
        lambda_q_ord5 = zeros(FT, N, N, Nz)
        lambda_q_ord7 = zeros(FT, N, N, Nz)
        Adv.apply_ppm_x_face_from_q_adjoint!(
            lambda_q_ord5, lambda_fx_face, q, am, m, mesh, Val(5))
        Adv.apply_ppm_x_face_from_q_adjoint!(
            lambda_q_ord7, lambda_fx_face, q, am, m, mesh, Val(7))
        @test lambda_q_ord5 == lambda_q_ord7
    end

    @testset "rejects unsupported ORD" begin
        FT = Float64
        Nc = 2; Hp = 3; Nz = 1
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp
        q  = zeros(FT, N, N, Nz)
        am = zeros(FT, Nc + 1, Nc, Nz)
        m  = ones(FT, N, N, Nz)
        lambda_fx_face = zeros(FT, Nc + 1, Nc, Nz)
        lambda_q = zeros(FT, N, N, Nz)
        @test_throws ArgumentError Adv.apply_ppm_x_face_from_q_adjoint!(
            lambda_q, lambda_fx_face, q, am, m, mesh, Val(4))
        @test_throws ArgumentError Adv.apply_ppm_y_face_from_q_adjoint!(
            lambda_q, lambda_fx_face, q, am, m, mesh, Val(6))
    end
end

# ===========================================================================
# Commit 3b — rm-input `_ppm_x/y_face_kernel!` adjoints (ORD=5)
#
# Like the `_from_q` variants, but fold `_safe_mixing_ratio` into the
# d6-AD chain and add the donor-cell α-denominator contribution. Tests
# verify the JVP/VJP identity against centered finite differences with
# perturbations on BOTH rm and m.
# ===========================================================================

function _ppm_x_face_fd_jvp(rm, m, am, drm, dm, mesh::AT.CubedSphereMesh{FT};
                             eps_fd, ord::Val{ORD}=Val(5)) where {FT, ORD}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm, 3)
    fx_plus  = zeros(FT, Nc + 1, Nc, Nz)
    fx_minus = zeros(FT, Nc + 1, Nc, Nz)
    backend = get_backend(rm)
    k! = Adv._ppm_x_face_kernel!(backend, 256)
    rm_plus  = rm .+ FT(eps_fd) .* drm
    rm_minus = rm .- FT(eps_fd) .* drm
    m_plus   = m  .+ FT(eps_fd) .* dm
    m_minus  = m  .- FT(eps_fd) .* dm
    k!(fx_plus,  rm_plus,  m_plus,  am, Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    k!(fx_minus, rm_minus, m_minus, am, Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    synchronize(backend)
    return (fx_plus .- fx_minus) ./ (FT(2) * FT(eps_fd))
end

function _ppm_y_face_fd_jvp(rm, m, bm, drm, dm, mesh::AT.CubedSphereMesh{FT};
                             eps_fd, ord::Val{ORD}=Val(5)) where {FT, ORD}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm, 3)
    fy_plus  = zeros(FT, Nc, Nc + 1, Nz)
    fy_minus = zeros(FT, Nc, Nc + 1, Nz)
    backend = get_backend(rm)
    k! = Adv._ppm_y_face_kernel!(backend, 256)
    rm_plus  = rm .+ FT(eps_fd) .* drm
    rm_minus = rm .- FT(eps_fd) .* drm
    m_plus   = m  .+ FT(eps_fd) .* dm
    m_minus  = m  .- FT(eps_fd) .* dm
    k!(fy_plus,  rm_plus,  m_plus,  bm, Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
    k!(fy_minus, rm_minus, m_minus, bm, Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
    synchronize(backend)
    return (fy_plus .- fy_minus) ./ (FT(2) * FT(eps_fd))
end

@testset "Plan 25 Commit 3b — PPM rm-input face kernel adjoints (ORD=5)" begin
    @testset "X face (rm-input) VJP vs FD JVP" begin
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(301)
        # Smooth fields, positive m well above threshold.
        rm = FT.([sin(0.13i + 0.21j + 0.07k) for i in 1:N, j in 1:N, k in 1:Nz])
        m  = FT(3) .+ rand(rng, FT, N, N, Nz)
        am = FT(0.02) .* randn(rng, FT, Nc + 1, Nc, Nz)

        lambda_fx_face = randn(rng, FT, Nc + 1, Nc, Nz)
        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        Adv.apply_ppm_x_face_adjoint!(
            lambda_rm, lambda_m, lambda_fx_face, rm, m, am, mesh, Val(5),
        )

        drm = randn(rng, FT, N, N, Nz)
        dm  = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_x_face_fd_jvp(rm, m, am, drm, dm, mesh; eps_fd=1e-6)

        lhs = _inner_full(lambda_fx_face, fd_jvp)
        rhs = sum(lambda_rm .* drm) + sum(lambda_m .* dm)

        @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-6)
    end

    @testset "X face (rm-input) VJP vs FD JVP, Courant number > 1 ($FT, ORD=$ORD)" for
            (FT, ORD, eps_fd, tol) in ((Float64, 5, 1e-6, 1e-6), (Float64, 7, 1e-6, 1e-6))
        # Half the faces carry more than the donor mass (|F| > m): the forward
        # clamps the Courant fraction to ±1, so the face value no longer depends
        # on α and the adjoint must not differentiate through it.
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(302)
        rm = FT.([sin(0.13i + 0.21j + 0.07k) for i in 1:N, j in 1:N, k in 1:Nz])
        m  = FT(3) .+ rand(rng, FT, N, N, Nz)                       # 3 ≤ m < 4
        am = [isodd(i + j + k) ? FT(12) * sign(randn(rng, FT)) : FT(0.02) * randn(rng, FT)
              for i in 1:Nc + 1, j in 1:Nc, k in 1:Nz]               # |α| ≥ 3 or ≪ 1, never near the clamp

        lambda_fx_face = randn(rng, FT, Nc + 1, Nc, Nz)
        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        Adv.apply_ppm_x_face_adjoint!(
            lambda_rm, lambda_m, lambda_fx_face, rm, m, am, mesh, Val(ORD),
        )

        drm = randn(rng, FT, N, N, Nz)
        dm  = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_x_face_fd_jvp(rm, m, am, drm, dm, mesh; eps_fd, ord=Val(ORD))

        lhs = _inner_full(lambda_fx_face, fd_jvp)
        rhs = sum(lambda_rm .* drm) + sum(lambda_m .* dm)

        @test isapprox(lhs, rhs; atol=tol, rtol=tol)
    end

    @testset "X face (rm-input) adjoint, Courant number > 1: Float32 agrees with Float64" begin
        # Finite differences in Float32 cross limiter kinks; compare with the Float64 adjoint instead.
        Nc = 4; Hp = 3; Nz = 2
        N = Nc + 2Hp
        rng = MersenneTwister(303)
        rm = [sin(0.13i + 0.21j + 0.07k) for i in 1:N, j in 1:N, k in 1:Nz]
        m  = 3 .+ rand(rng, N, N, Nz)
        am = [isodd(i + j + k) ? 12 * sign(randn(rng)) : 0.02 * randn(rng) for i in 1:Nc + 1, j in 1:Nc, k in 1:Nz]
        seed = randn(rng, Nc + 1, Nc, Nz)
        adjoint(FT) = begin
            mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
            lambda_rm, lambda_m = zeros(FT, N, N, Nz), zeros(FT, N, N, Nz)
            Adv.apply_ppm_x_face_adjoint!(lambda_rm, lambda_m, FT.(seed), FT.(rm), FT.(m), FT.(am), mesh, Val(5))
            (lambda_rm, lambda_m)
        end
        (rm32, m32), (rm64, m64) = adjoint(Float32), adjoint(Float64)
        @test isapprox(Float64.(rm32), rm64; rtol = 1e-5)
        @test isapprox(Float64.(m32), m64; rtol = 1e-5)
    end

    @testset "Y face (rm-input) VJP vs FD JVP" begin
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(401)
        rm = FT.([sin(0.11i - 0.17j + 0.05k) for i in 1:N, j in 1:N, k in 1:Nz])
        m  = FT(3) .+ rand(rng, FT, N, N, Nz)
        bm = FT(0.02) .* randn(rng, FT, Nc, Nc + 1, Nz)

        lambda_fy_face = randn(rng, FT, Nc, Nc + 1, Nz)
        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        Adv.apply_ppm_y_face_adjoint!(
            lambda_rm, lambda_m, lambda_fy_face, rm, m, bm, mesh, Val(5),
        )

        drm = randn(rng, FT, N, N, Nz)
        dm  = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_y_face_fd_jvp(rm, m, bm, drm, dm, mesh; eps_fd=1e-6)

        lhs = _inner_full(lambda_fy_face, fd_jvp)
        rhs = sum(lambda_rm .* drm) + sum(lambda_m .* dm)

        @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-6)
    end

    @testset "X face (rm-input) ORD=7 VJP vs FD JVP" begin
        # ORD=7 adds the linear discontinuous-edge correction at panel-
        # edge faces. The donor-state lo/hi helpers carry the corrected
        # `(bl, br, b0)` into the α-contribution so the FD JVP and the
        # adjoint VJP agree at both interior and panel-edge faces.
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(7301)
        rm = FT.([sin(0.13i + 0.21j + 0.07k) for i in 1:N, j in 1:N, k in 1:Nz])
        m  = FT(3) .+ rand(rng, FT, N, N, Nz)
        am = FT(0.02) .* randn(rng, FT, Nc + 1, Nc, Nz)

        lambda_fx_face = randn(rng, FT, Nc + 1, Nc, Nz)
        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        Adv.apply_ppm_x_face_adjoint!(
            lambda_rm, lambda_m, lambda_fx_face, rm, m, am, mesh, Val(7),
        )

        drm = randn(rng, FT, N, N, Nz)
        dm  = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_x_face_fd_jvp(rm, m, am, drm, dm, mesh;
                                      eps_fd=1e-6, ord=Val(7))

        lhs = _inner_full(lambda_fx_face, fd_jvp)
        rhs = sum(lambda_rm .* drm) + sum(lambda_m .* dm)
        @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-6)
    end

    @testset "Y face (rm-input) ORD=7 VJP vs FD JVP" begin
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        rng = MersenneTwister(7401)
        rm = FT.([sin(0.11i - 0.17j + 0.05k) for i in 1:N, j in 1:N, k in 1:Nz])
        m  = FT(3) .+ rand(rng, FT, N, N, Nz)
        bm = FT(0.02) .* randn(rng, FT, Nc, Nc + 1, Nz)

        lambda_fy_face = randn(rng, FT, Nc, Nc + 1, Nz)
        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        Adv.apply_ppm_y_face_adjoint!(
            lambda_rm, lambda_m, lambda_fy_face, rm, m, bm, mesh, Val(7),
        )

        drm = randn(rng, FT, N, N, Nz)
        dm  = randn(rng, FT, N, N, Nz)
        fd_jvp = _ppm_y_face_fd_jvp(rm, m, bm, drm, dm, mesh;
                                      eps_fd=1e-6, ord=Val(7))

        lhs = _inner_full(lambda_fy_face, fd_jvp)
        rhs = sum(lambda_rm .* drm) + sum(lambda_m .* dm)
        @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-6)
    end

    @testset "rm-input ORD=7 interior bit-equals ORD=5" begin
        FT = Float64
        Nc = 4; Hp = 3; Nz = 2
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp
        rng = MersenneTwister(7501)
        rm = FT.([sin(0.13i + 0.21j + 0.07k) for i in 1:N, j in 1:N, k in 1:Nz])
        m  = FT(3) .+ rand(rng, FT, N, N, Nz)
        am = FT(0.02) .* randn(rng, FT, Nc + 1, Nc, Nz)
        lambda_fx_face = randn(rng, FT, Nc + 1, Nc, Nz)
        lambda_fx_face[1, :, :] .= 0
        lambda_fx_face[Nc + 1, :, :] .= 0
        lambda_rm_5 = zeros(FT, N, N, Nz); lambda_m_5 = zeros(FT, N, N, Nz)
        lambda_rm_7 = zeros(FT, N, N, Nz); lambda_m_7 = zeros(FT, N, N, Nz)
        Adv.apply_ppm_x_face_adjoint!(
            lambda_rm_5, lambda_m_5, lambda_fx_face, rm, m, am, mesh, Val(5))
        Adv.apply_ppm_x_face_adjoint!(
            lambda_rm_7, lambda_m_7, lambda_fx_face, rm, m, am, mesh, Val(7))
        @test lambda_rm_5 == lambda_rm_7
        @test lambda_m_5  == lambda_m_7
    end
end

# ===========================================================================
# Commit 4 — single-panel, zero-halo LinRood horizontal adjoint composition
# ===========================================================================

# Run one panel of the forward `fv_tp_2d_cs!` chain by hand, with all
# cross-panel halo / corner copies skipped (halos held at the user-
# supplied values for the duration). Captures the intermediate q_buf
# states needed by the reverse pass. Mirrors `fv_tp_2d_cs!` (linrood_horizontal.jl).
function _linrood_single_panel_forward(rm0, m0, am, bm,
                                        mesh::AT.CubedSphereMesh{FT}) where {FT}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm0, 3)
    N = Nc + 2Hp

    rm = copy(rm0)
    m  = copy(m0)
    backend = get_backend(rm)

    init_k!    = Adv._init_q_buf_kernel!(backend, 256)
    y_face_k!  = Adv._ppm_y_face_kernel!(backend, 256)
    x_face_k!  = Adv._ppm_x_face_kernel!(backend, 256)
    xq_face_k! = Adv._ppm_x_face_from_q_kernel!(backend, 256)
    yq_face_k! = Adv._ppm_y_face_from_q_kernel!(backend, 256)
    pre_y_k!   = Adv._pre_advect_y_kernel!(backend, 256)
    pre_x_k!   = Adv._pre_advect_x_kernel!(backend, 256)
    update_k!  = Adv._linrood_update_kernel!(backend, 256)

    fy_in  = zeros(FT, Nc, Nc + 1, Nz)
    fy_out = zeros(FT, Nc, Nc + 1, Nz)
    fx_in  = zeros(FT, Nc + 1, Nc, Nz)
    fx_out = zeros(FT, Nc + 1, Nc, Nz)
    q_buf  = zeros(FT, N, N, Nz)

    # Phase 1
    init_k!(q_buf, rm, m; ndrange=(N, N, Nz))
    synchronize(backend)
    y_face_k!(fy_in, rm, m, bm, Hp, Nc, Val(5);
              ndrange=(Nc, Nc + 1, Nz))
    pre_y_k!(q_buf, rm, m, bm, fy_in, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    q_buf_phase2 = copy(q_buf)

    # Phase 2
    xq_face_k!(fx_out, q_buf_phase2, am, m, Hp, Nc, Val(5);
               ndrange=(Nc + 1, Nc, Nz))
    x_face_k!(fx_in, rm, m, am, Hp, Nc, Val(5);
              ndrange=(Nc + 1, Nc, Nz))
    synchronize(backend)
    init_k!(q_buf, rm, m; ndrange=(N, N, Nz))
    synchronize(backend)
    pre_x_k!(q_buf, rm, m, am, fx_in, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    q_buf_phase3 = copy(q_buf)

    # Phase 3
    yq_face_k!(fy_out, q_buf_phase3, bm, m, Hp, Nc, Val(5);
               ndrange=(Nc, Nc + 1, Nz))
    rm_new_buf = zeros(FT, N, N, Nz)
    m_new_buf  = zeros(FT, N, N, Nz)
    update_k!(rm_new_buf, m_new_buf, rm, m, am, bm,
              fx_in, fx_out, fy_in, fy_out, Hp;
              ndrange=(Nc, Nc, Nz))
    synchronize(backend)

    return (rm_new=rm_new_buf, m_new=m_new_buf,
            q_buf_phase2=q_buf_phase2, q_buf_phase3=q_buf_phase3,
            fx_in=fx_in, fx_out=fx_out, fy_in=fy_in, fy_out=fy_out)
end

@testset "Plan 25 Commit 4 — single-panel horizontal adjoint composition" begin
    FT = Float64
    Nc = 4; Hp = 3; Nz = 2
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
    N = Nc + 2Hp

    # Single panel with zero halos so cross-panel halo / corner
    # adjoints don't contribute. Interior fields are smooth.
    rng = MersenneTwister(501)
    rm0 = zeros(FT, N, N, Nz)
    m0  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        rm0[i, j, k] = FT(0.5) * sin(0.13i + 0.21j + 0.07k)
        m0[i, j, k]  = FT(3) + FT(0.1) * sin(0.09i - 0.11j)
    end
    am = FT(0.005) .* randn(rng, FT, Nc + 1, Nc, Nz)
    bm = FT(0.005) .* randn(rng, FT, Nc, Nc + 1, Nz)

    # Forward
    out = _linrood_single_panel_forward(rm0, m0, am, bm, mesh)

    # Adjoint seeds: random on interior only.
    lambda_rm_new = zeros(FT, N, N, Nz)
    lambda_m_new  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        lambda_rm_new[i, j, k] = randn(rng, FT)
        lambda_m_new[i, j, k]  = randn(rng, FT)
    end

    # Adjoint
    lambda_rm = zeros(FT, N, N, Nz)
    lambda_m  = zeros(FT, N, N, Nz)
    Adv.apply_linrood_horizontal_adjoint_single_panel!(
        lambda_rm, lambda_m,
        lambda_rm_new, lambda_m_new,
        rm0, m0, am, bm,
        out.q_buf_phase2, out.q_buf_phase3,
        out.fx_in, out.fx_out, out.fy_in,
        mesh, Val(5),
    )

    # FD JVP through the full forward. Restrict perturbations to the
    # INTERIOR cells — perturbing halo cells (where m0 = 0) would
    # push m_perturbed slightly above the `_safe_mixing_ratio` zero
    # threshold and produce explosive 1/m values in the FD numerator.
    # The adjoint also writes zero to halo lambda cells in this
    # zero-halo configuration, so the transposition identity is
    # restricted to the interior.
    drm = zeros(FT, N, N, Nz)
    dm  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        drm[i, j, k] = randn(rng, FT)
        dm[i, j, k]  = randn(rng, FT)
    end
    eps_fd = 1e-6
    out_plus  = _linrood_single_panel_forward(rm0 .+ eps_fd .* drm,
                                               m0  .+ eps_fd .* dm,
                                               am, bm, mesh)
    out_minus = _linrood_single_panel_forward(rm0 .- eps_fd .* drm,
                                               m0  .- eps_fd .* dm,
                                               am, bm, mesh)
    fd_drm_new = (out_plus.rm_new .- out_minus.rm_new) ./ (2eps_fd)
    fd_dm_new  = (out_plus.m_new  .- out_minus.m_new)  ./ (2eps_fd)

    lhs = sum(lambda_rm_new .* fd_drm_new) + sum(lambda_m_new .* fd_dm_new)
    rhs = sum(lambda_rm .* drm) + sum(lambda_m .* dm)

    @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-5)
end

# ===========================================================================
# Commit 5 — multi-substep replay test
# ===========================================================================

@testset "Plan 25 Commit 5 — multi-substep LinRood horizontal adjoint" begin
    FT = Float64
    Nc = 4; Hp = 3; Nz = 2
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
    N = Nc + 2Hp
    nsteps = 3

    rng = MersenneTwister(601)
    rm0 = zeros(FT, N, N, Nz)
    m0  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        rm0[i, j, k] = FT(0.5) * sin(0.13i + 0.21j + 0.07k)
        m0[i, j, k]  = FT(3) + FT(0.1) * sin(0.09i - 0.11j)
    end
    am_steps = [FT(0.005) .* randn(rng, FT, Nc + 1, Nc, Nz) for _ in 1:nsteps]
    bm_steps = [FT(0.005) .* randn(rng, FT, Nc, Nc + 1, Nz) for _ in 1:nsteps]

    # Forward sequence
    function _forward_sequence(rm0, m0)
        rm = copy(rm0); m = copy(m0)
        tape = Vector{Any}(undef, nsteps)
        for t in 1:nsteps
            entry, rm_next, m_next = Adv.record_linrood_substep!(
                rm, m, am_steps[t], bm_steps[t], mesh)
            tape[t] = entry
            rm = rm_next
            m  = m_next
        end
        return (tape=tape, rm_final=rm, m_final=m)
    end

    out = _forward_sequence(rm0, m0)
    tape_vec = Vector{typeof(out.tape[1])}(out.tape)

    # Adjoint seeds on the final state (interior only).
    lambda_rm_final = zeros(FT, N, N, Nz)
    lambda_m_final  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        lambda_rm_final[i, j, k] = randn(rng, FT)
        lambda_m_final[i, j, k]  = randn(rng, FT)
    end

    lambda_rm0 = zeros(FT, N, N, Nz)
    lambda_m0  = zeros(FT, N, N, Nz)
    Adv.apply_linrood_multi_substep_adjoint!(
        lambda_rm0, lambda_m0,
        lambda_rm_final, lambda_m_final,
        tape_vec, am_steps, bm_steps, mesh,
    )

    # FD JVP over the full nsteps sequence. Interior perturbations
    # only (same reason as Commit 4).
    drm = zeros(FT, N, N, Nz)
    dm  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        drm[i, j, k] = randn(rng, FT)
        dm[i, j, k]  = randn(rng, FT)
    end
    eps_fd = 1e-6
    out_plus  = _forward_sequence(rm0 .+ eps_fd .* drm, m0 .+ eps_fd .* dm)
    out_minus = _forward_sequence(rm0 .- eps_fd .* drm, m0 .- eps_fd .* dm)
    fd_drm_final = (out_plus.rm_final .- out_minus.rm_final) ./ (2eps_fd)
    fd_dm_final  = (out_plus.m_final  .- out_minus.m_final)  ./ (2eps_fd)

    lhs = sum(lambda_rm_final .* fd_drm_final) +
          sum(lambda_m_final  .* fd_dm_final)
    rhs = sum(lambda_rm0 .* drm) + sum(lambda_m0 .* dm)

    @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-4)
end

@testset "Plan 25 Commit 5 (ORD=7) — multi-substep LinRood adjoint, no Val(7) kwarg" begin
    # Codex review M1 2026-05-15: before binding ORD into
    # `LinRoodHorizontalTapeEntry`, a tape recorded at ORD=7 silently
    # reversed with ORD=5 when the caller did not pass an explicit
    # `Val(7)` to `apply_linrood_multi_substep_adjoint!` (the kwarg
    # defaulted to Val(5)). Codex reproduced this with a smooth
    # one-step probe: default reverse error ~1.34e-4, explicit
    # Val(7) error ~6.9e-10. After the binding, the reverse pass
    # reads ORD from the tape's element type and the kwarg is gone,
    # so the foot-gun is dispatch-impossible. This test calls
    # `apply_linrood_multi_substep_adjoint!` WITHOUT specifying ORD
    # and verifies the FD/VJP identity passes — i.e., the ORD=7
    # reverse path is the one being exercised.
    FT = Float64
    Nc = 4; Hp = 3; Nz = 2
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
    N = Nc + 2Hp
    nsteps = 3

    rng = MersenneTwister(701)
    rm0 = zeros(FT, N, N, Nz)
    m0  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        rm0[i, j, k] = FT(0.5) * sin(0.13i + 0.21j + 0.07k)
        m0[i, j, k]  = FT(3) + FT(0.1) * sin(0.09i - 0.11j)
    end
    am_steps = [FT(0.005) .* randn(rng, FT, Nc + 1, Nc, Nz) for _ in 1:nsteps]
    bm_steps = [FT(0.005) .* randn(rng, FT, Nc, Nc + 1, Nz) for _ in 1:nsteps]

    # Forward sequence with ord=Val(7) — tape entries are typed
    # LinRoodHorizontalTapeEntry{…, 7}.
    function _forward_sequence_ord7(rm0, m0)
        rm = copy(rm0); m = copy(m0)
        tape = Vector{Any}(undef, nsteps)
        for t in 1:nsteps
            entry, rm_next, m_next = Adv.record_linrood_substep!(
                rm, m, am_steps[t], bm_steps[t], mesh; ord=Val(7))
            tape[t] = entry
            rm = rm_next
            m  = m_next
        end
        return (tape=tape, rm_final=rm, m_final=m)
    end

    out = _forward_sequence_ord7(rm0, m0)
    tape_vec = Vector{typeof(out.tape[1])}(out.tape)

    lambda_rm_final = zeros(FT, N, N, Nz)
    lambda_m_final  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        lambda_rm_final[i, j, k] = randn(rng, FT)
        lambda_m_final[i, j, k]  = randn(rng, FT)
    end

    lambda_rm0 = zeros(FT, N, N, Nz)
    lambda_m0  = zeros(FT, N, N, Nz)
    # CRITICAL: do NOT pass an explicit Val(ORD). ORD must come from
    # the tape's element type via dispatch.
    Adv.apply_linrood_multi_substep_adjoint!(
        lambda_rm0, lambda_m0,
        lambda_rm_final, lambda_m_final,
        tape_vec, am_steps, bm_steps, mesh,
    )

    drm = zeros(FT, N, N, Nz)
    dm  = zeros(FT, N, N, Nz)
    @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        drm[i, j, k] = randn(rng, FT)
        dm[i, j, k]  = randn(rng, FT)
    end
    eps_fd = 1e-6
    out_plus  = _forward_sequence_ord7(rm0 .+ eps_fd .* drm, m0 .+ eps_fd .* dm)
    out_minus = _forward_sequence_ord7(rm0 .- eps_fd .* drm, m0 .- eps_fd .* dm)
    fd_drm_final = (out_plus.rm_final .- out_minus.rm_final) ./ (2eps_fd)
    fd_dm_final  = (out_plus.m_final  .- out_minus.m_final)  ./ (2eps_fd)

    lhs = sum(lambda_rm_final .* fd_drm_final) +
          sum(lambda_m_final  .* fd_dm_final)
    rhs = sum(lambda_rm0 .* drm) + sum(lambda_m0 .* dm)

    @test isapprox(lhs, rhs; atol=1e-7, rtol=1e-4)
end

