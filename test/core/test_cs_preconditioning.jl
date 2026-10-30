#!/usr/bin/env julia
# Forward preconditioner values, Float32 covariance stability, and adjoint
# identities against central differences of the production forward map.

using Test

import AtmosTransport
const AT = AtmosTransport

const FT_TEST = Float64
const NC_SMALL = 4

# Deterministic pseudo-random tuple from sin/cos seeding.
_random_tuple(Nc, seed) = ntuple(p ->
    [FT_TEST(sin(0.7 * seed * (i + 13 * j + 41 * p)))
     for i in 1:Nc, j in 1:Nc], 6)

_unit_sigma(Nc) = ntuple(_ -> ones(FT_TEST, Nc, Nc), 6)
_constant_sigma(Nc, σ) = ntuple(_ -> fill(FT_TEST(σ), Nc, Nc), 6)
_constant_background(Nc, b) = ntuple(_ -> fill(FT_TEST(b), Nc, Nc), 6)
_zeros_tuple(Nc) = ntuple(_ -> zeros(FT_TEST, Nc, Nc), 6)

_diag_cov(Nc, σ) = AT.DiagonalCSCovariance(_constant_sigma(Nc, σ))
_gauss_cov(Nc, σ, L) = AT.IsotropicGaussianCSCovariance(_constant_sigma(Nc, σ), L)

_panel_inner(a, b) = sum(p -> sum(a[p] .* b[p]), 1:6)

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

@testset "CSSurfaceFluxPreconditioner — constructor validation" begin
    cov = _diag_cov(NC_SMALL, 0.5)
    bg = _constant_background(NC_SMALL, 1.0)

    @test AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LinearOptimType()) isa
          AT.CSSurfaceFluxPreconditioner
    @test AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LogNormalOptimType()) isa
          AT.CSSurfaceFluxPreconditioner

    # Shape mismatch with covariance's Nc.
    bad_shape = ntuple(_ -> ones(FT_TEST, NC_SMALL + 1, NC_SMALL + 1), 6)
    @test_throws DimensionMismatch AT.CSSurfaceFluxPreconditioner(
        cov, bad_shape, AT.LinearOptimType())

    # Non-finite background.
    nan_bg = ntuple(_ -> fill(FT_TEST(NaN), NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.CSSurfaceFluxPreconditioner(
        cov, nan_bg, AT.LinearOptimType())

    # Non-positive background under LogNormal — rejected.
    neg_bg = ntuple(p -> p == 3 ?
                            fill(FT_TEST(-1.0), NC_SMALL, NC_SMALL) :
                            ones(FT_TEST, NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.CSSurfaceFluxPreconditioner(
        cov, neg_bg, AT.LogNormalOptimType())
    zero_bg = ntuple(_ -> zeros(FT_TEST, NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.CSSurfaceFluxPreconditioner(
        cov, zero_bg, AT.LogNormalOptimType())

    # Non-positive background under Linear is OK (linear allows
    # any sign).
    @test AT.CSSurfaceFluxPreconditioner(cov, neg_bg, AT.LinearOptimType()) isa
          AT.CSSurfaceFluxPreconditioner
end

@testset "Gaussian forward map remains finite at operational Float32 sizes" begin
    for (Nc, correlation_length) in ((48, 2.0f0), (48, 3.0f0), (180, 2.0f0))
        sigma = ntuple(_ -> fill(0.5f0, Nc, Nc), 6)
        covariance = AT.IsotropicGaussianCSCovariance(sigma, correlation_length)
        @test all(isfinite, covariance.L_transfer_sqrt)
        @test minimum(covariance.L_transfer_sqrt) > 0.0f0
    end

    Nc = 48
    sigma = ntuple(_ -> fill(0.5f0, Nc, Nc), 6)
    covariance = AT.IsotropicGaussianCSCovariance(sigma, 2.0f0)
    x = ntuple(p -> Float32[
        sin(0.031f0 * i + 0.047f0 * j + 0.13f0 * p)
        for i in 1:Nc, j in 1:Nc
    ], 6)
    y = ntuple(_ -> zeros(Float32, Nc, Nc), 6)
    AT.apply_B_half!(y, covariance, x)
    @test all(panel -> all(isfinite, panel), y)

    # Diagonal of B from the production square root and transpose.
    spike = ntuple(_ -> zeros(Float32, Nc, Nc), 6)
    spike[1][1, 1] = 1.0f0
    AT.apply_B_half_adjoint!(y, covariance, spike)
    @test sum(sum(abs2, panel) for panel in y) ≈ 0.25f0 rtol = 2f-5
end

# ---------------------------------------------------------------------------
# Forward T(χ): correctness
# ---------------------------------------------------------------------------

@testset "apply_preconditioner! — Linear/Diagonal literal" begin
    σ_scalar = FT_TEST(0.5)
    cov = _diag_cov(NC_SMALL, σ_scalar)
    bg_value = FT_TEST(420.0)
    bg = _constant_background(NC_SMALL, bg_value)
    prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LinearOptimType())

    chi = _random_tuple(NC_SMALL, 3)
    x = _zeros_tuple(NC_SMALL)
    AT.apply_preconditioner!(x, prec, chi)

    # x = x_b + σ ⊙ χ — literal check.
    for p in 1:6
        @test x[p] ≈ bg_value .+ σ_scalar .* chi[p] atol = 1e-12 rtol = 1e-12
    end
end

@testset "apply_preconditioner! — LogNormal/Diagonal literal + positivity" begin
    σ_scalar = FT_TEST(0.1)
    cov = _diag_cov(NC_SMALL, σ_scalar)
    bg_value = FT_TEST(420.0)
    bg = _constant_background(NC_SMALL, bg_value)
    prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LogNormalOptimType())

    # χ = 0 → x = x_b.
    chi0 = _zeros_tuple(NC_SMALL)
    x0 = _zeros_tuple(NC_SMALL)
    AT.apply_preconditioner!(x0, prec, chi0)
    for p in 1:6
        @test x0[p] ≈ bg[p] atol = 1e-12 rtol = 1e-12
    end

    # Arbitrary χ (including negative) → x = x_b ⊙ exp(σ ⊙ χ), always > 0.
    chi = _random_tuple(NC_SMALL, 4)
    x = _zeros_tuple(NC_SMALL)
    AT.apply_preconditioner!(x, prec, chi)
    for p in 1:6
        @test x[p] ≈ bg_value .* exp.(σ_scalar .* chi[p]) atol = 1e-10 rtol = 1e-10
        @test all(x[p] .> 0)
    end
end

# ---------------------------------------------------------------------------
# Adjoint identity — ⟨T'(χ) δχ, g_phys⟩ = ⟨δχ, T'(χ)^T g_phys⟩
# ---------------------------------------------------------------------------

@testset "Adjoint identity — Linear (Diagonal + Gaussian)" begin
    for cov in (_diag_cov(NC_SMALL, 0.5),
                _gauss_cov(NC_SMALL, 0.5, 1.5))
        bg = _constant_background(NC_SMALL, 420.0)
        prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LinearOptimType())

        chi = _random_tuple(NC_SMALL, 9)
        delta_chi = _random_tuple(NC_SMALL, 10)
        g_phys = _random_tuple(NC_SMALL, 11)

        # Base point x = T(χ) — Linear's tangent doesn't actually use x.
        x = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(x, prec, chi)

        h = 1e-4
        plus, minus = _zeros_tuple(NC_SMALL), _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(plus, prec, ntuple(p -> chi[p] .+ h .* delta_chi[p], 6))
        AT.apply_preconditioner!(minus, prec, ntuple(p -> chi[p] .- h .* delta_chi[p], 6))
        delta_x = ntuple(p -> (plus[p] .- minus[p]) ./ (2h), 6)

        g_chi = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner_adjoint!(g_chi, prec, x, g_phys)

        lhs = _panel_inner(delta_x, g_phys)
        rhs = _panel_inner(delta_chi, g_chi)
        @test isapprox(lhs, rhs; atol = 1e-8, rtol = 1e-8)
    end
end

@testset "Adjoint identity — LogNormal (Diagonal + Gaussian)" begin
    for cov in (_diag_cov(NC_SMALL, 0.1),
                _gauss_cov(NC_SMALL, 0.1, 1.5))
        bg = _constant_background(NC_SMALL, 1.0)
        prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LogNormalOptimType())

        chi = _random_tuple(NC_SMALL, 12)
        delta_chi = _random_tuple(NC_SMALL, 13)
        g_phys = _random_tuple(NC_SMALL, 14)

        # Base point x = T(χ) — LogNormal's tangent uses x.
        x = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(x, prec, chi)

        h = 1e-4
        plus, minus = _zeros_tuple(NC_SMALL), _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(plus, prec, ntuple(p -> chi[p] .+ h .* delta_chi[p], 6))
        AT.apply_preconditioner!(minus, prec, ntuple(p -> chi[p] .- h .* delta_chi[p], 6))
        delta_x = ntuple(p -> (plus[p] .- minus[p]) ./ (2h), 6)

        g_chi = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner_adjoint!(g_chi, prec, x, g_phys)

        lhs = _panel_inner(delta_x, g_phys)
        rhs = _panel_inner(delta_chi, g_chi)
        @test isapprox(lhs, rhs; atol = 1e-8, rtol = 1e-8)
    end
end
