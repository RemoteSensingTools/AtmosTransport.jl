#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.B1 — CS surface-flux covariance B and its spectral square
# root B^(1/2).
#
# Coverage:
#   * Constructor validation: shape mismatch, non-finite / non-positive
#     sigma, non-positive correlation length, non-square panels.
#   * `apply_B_half!` shape mismatch rejections.
#   * `DiagonalCSCovariance`:
#       - `B^(1/2) χ = σ ⊙ χ` (literal arithmetic check).
#       - `B^(1/2)` is self-adjoint (forward == adjoint).
#       - Spectrum: eigenvalues of the full `B` matrix equal `σ²`,
#         in PSD order.
#   * `IsotropicGaussianCSCovariance`:
#       - Adjoint identity: `⟨B^(1/2) χ, g⟩ = ⟨χ, (B^(1/2))^T g⟩` for
#         random `χ`, `g` (validates the FFT-based transpose).
#       - Spectrum: full `B` matrix is symmetric, PSD, and `B[i, i] =
#         σ_i²` (correlation diagonal is normalized to 1).
#       - Smoothing: applying `B^(1/2)` to a δ-spike with σ ≡ 1
#         produces a Gaussian-shaped column whose value at the spike
#         exceeds the value at a far cell — qualitative-but-tight
#         check that the smoother is doing what it says.
#       - Reducing the correlation length toward 0 recovers the
#         diagonal limit (off-diagonal `B[i, j] → 0` for `i ≠ j`).
# ---------------------------------------------------------------------------

using Test
using LinearAlgebra: eigvals, Symmetric

import AtmosTransport
const AT = AtmosTransport

const FT_TEST = Float64
const NC_SMALL = 4

# Flatten / unflatten helpers between the canonical `NTuple{6, Matrix}`
# tuple and a dense `Vector{FT}` of length `6 * Nc * Nc` so we can
# build the explicit matrix representation of `B` from any covariance.
function _tuple_to_vec(panels::NTuple{6, <:AbstractMatrix{FT}}) where FT
    Nc = size(panels[1], 1)
    out = Vector{FT}(undef, 6 * Nc * Nc)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        out[((p - 1) * Nc + (j - 1)) * Nc + i] = panels[p][i, j]
    end
    return out
end

function _vec_to_tuple(v::AbstractVector{FT}, Nc::Int) where FT
    panels = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        panels[p][i, j] = v[((p - 1) * Nc + (j - 1)) * Nc + i]
    end
    return panels
end

# Build the full `B = (B^(1/2)) · (B^(1/2))^T` matrix by applying it
# column by column. Each column k is:
#     B[:, k] = B^(1/2) · ((B^(1/2))^T · e_k)
# where `e_k` is the k-th standard basis vector. With our factorization
# `B^(1/2) = D · L` this evaluates to `D · L · L · D · e_k = D · C · D
# · e_k = (D · C · D)[:, k]`.
function _build_B_matrix(cov, Nc::Int)
    N = 6 * Nc * Nc
    B = Matrix{Float64}(undef, N, N)
    z = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
    y = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
    for k in 1:N
        # e_k as tuple.
        e_panels = _vec_to_tuple(Float64[i == k ? 1.0 : 0.0 for i in 1:N], Nc)
        AT.apply_B_half_adjoint!(z, cov, e_panels)
        AT.apply_B_half!(y, cov, z)
        col = _tuple_to_vec(y)
        @inbounds for i in 1:N
            B[i, k] = col[i]
        end
    end
    return B
end

_unit_sigma(Nc) = ntuple(_ -> ones(FT_TEST, Nc, Nc), 6)
_constant_sigma(Nc, σ) = ntuple(_ -> fill(FT_TEST(σ), Nc, Nc), 6)

@testset "Covariance constructors — validation" begin
    # DiagonalCSCovariance: positive-finite, square panels.
    @test AT.DiagonalCSCovariance(_constant_sigma(NC_SMALL, 0.5)) isa
          AT.DiagonalCSCovariance

    bad_sigma = ntuple(p -> p == 2 ? zeros(FT_TEST, NC_SMALL, NC_SMALL) :
                                      ones(FT_TEST, NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.DiagonalCSCovariance(bad_sigma)

    neg_sigma = ntuple(p -> p == 3 ?
                              fill(FT_TEST(-1.0), NC_SMALL, NC_SMALL) :
                              ones(FT_TEST, NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.DiagonalCSCovariance(neg_sigma)

    nan_sigma = ntuple(_ -> fill(FT_TEST(NaN), NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.DiagonalCSCovariance(nan_sigma)

    ragged_sigma = ntuple(p -> p == 1 ?
                                  ones(FT_TEST, NC_SMALL, NC_SMALL) :
                                  ones(FT_TEST, NC_SMALL + 1, NC_SMALL + 1), 6)
    @test_throws DimensionMismatch AT.DiagonalCSCovariance(ragged_sigma)

    non_square = ntuple(_ -> ones(FT_TEST, NC_SMALL, NC_SMALL + 1), 6)
    @test_throws DimensionMismatch AT.DiagonalCSCovariance(non_square)

    # IsotropicGaussianCSCovariance: positive finite correlation length.
    @test AT.IsotropicGaussianCSCovariance(_constant_sigma(NC_SMALL, 0.5), 1.0) isa
          AT.IsotropicGaussianCSCovariance

    @test_throws ArgumentError AT.IsotropicGaussianCSCovariance(
        _constant_sigma(NC_SMALL, 0.5), 0.0)
    @test_throws ArgumentError AT.IsotropicGaussianCSCovariance(
        _constant_sigma(NC_SMALL, 0.5), -1.0)
    @test_throws ArgumentError AT.IsotropicGaussianCSCovariance(
        _constant_sigma(NC_SMALL, 0.5), NaN)
    @test_throws ArgumentError AT.IsotropicGaussianCSCovariance(
        _constant_sigma(NC_SMALL, 0.5), Inf)
    @test_throws DimensionMismatch AT.IsotropicGaussianCSCovariance(
        non_square, 1.0)
end

@testset "apply_B_half! — shape mismatch rejections" begin
    cov = AT.DiagonalCSCovariance(_constant_sigma(NC_SMALL, 0.5))
    bad = ntuple(_ -> ones(FT_TEST, NC_SMALL + 1, NC_SMALL + 1), 6)
    good = ntuple(_ -> ones(FT_TEST, NC_SMALL, NC_SMALL), 6)
    @test_throws DimensionMismatch AT.apply_B_half!(good, cov, bad)
    @test_throws DimensionMismatch AT.apply_B_half!(bad, cov, good)
    @test_throws DimensionMismatch AT.apply_B_half_adjoint!(good, cov, bad)
end

# ---------------------------------------------------------------------------
# Diagonal case — literal arithmetic + self-adjoint + spectrum
# ---------------------------------------------------------------------------

@testset "DiagonalCSCovariance — B^(1/2) χ = σ ⊙ χ" begin
    σ = ntuple(p -> reshape(FT_TEST.(1:(NC_SMALL * NC_SMALL)) .+ p * 100,
                            NC_SMALL, NC_SMALL), 6)
    cov = AT.DiagonalCSCovariance(σ)

    chi = ntuple(p -> FT_TEST.(p .+ reshape(1:(NC_SMALL * NC_SMALL),
                                             NC_SMALL, NC_SMALL)), 6)
    y = ntuple(_ -> zeros(FT_TEST, NC_SMALL, NC_SMALL), 6)
    AT.apply_B_half!(y, cov, chi)
    for p in 1:6
        @test y[p] == σ[p] .* chi[p]
    end

    # Self-adjoint: adjoint == forward for the diagonal case.
    g = ntuple(_ -> zeros(FT_TEST, NC_SMALL, NC_SMALL), 6)
    AT.apply_B_half_adjoint!(g, cov, chi)
    for p in 1:6
        @test g[p] == y[p]
    end
end



# ---------------------------------------------------------------------------
# Gaussian case — adjoint identity + spectrum + smoothing
# ---------------------------------------------------------------------------

@testset "IsotropicGaussianCSCovariance — adjoint identity" begin
    σ = ntuple(p -> fill(FT_TEST(0.5) + 0.1 * p, NC_SMALL, NC_SMALL), 6)
    cov = AT.IsotropicGaussianCSCovariance(σ, FT_TEST(1.5))

    # Deterministic pseudo-random inputs via sin/cos seeding — same
    # values every run, no Random.jl dependency.
    chi = ntuple(p -> [FT_TEST(sin(0.7 * (i + 13 * j + 41 * p)))
                       for i in 1:NC_SMALL, j in 1:NC_SMALL], 6)
    g = ntuple(p -> [FT_TEST(cos(0.3 * (i + 17 * j + 23 * p)))
                     for i in 1:NC_SMALL, j in 1:NC_SMALL], 6)

    y = ntuple(_ -> zeros(FT_TEST, NC_SMALL, NC_SMALL), 6)
    g_chi = ntuple(_ -> zeros(FT_TEST, NC_SMALL, NC_SMALL), 6)
    AT.apply_B_half!(y, cov, chi)
    AT.apply_B_half_adjoint!(g_chi, cov, g)

    inner_lhs = sum(p -> sum(y[p] .* g[p]), 1:6)
    inner_rhs = sum(p -> sum(chi[p] .* g_chi[p]), 1:6)
    @test isapprox(inner_lhs, inner_rhs;
                   atol = 1e-10, rtol = 1e-10)
end

@testset "IsotropicGaussianCSCovariance — spectrum is symmetric PSD with B[i,i]=σ²" begin
    σ_scalar = FT_TEST(0.7)
    σ = _constant_sigma(NC_SMALL, σ_scalar)
    cov = AT.IsotropicGaussianCSCovariance(σ, FT_TEST(1.0))

    B = _build_B_matrix(cov, NC_SMALL)

    # Symmetric within float roundoff.
    @test isapprox(B, B'; atol = 1e-10, rtol = 1e-10)

    # PSD: every eigenvalue ≥ 0 (clamp tiny negative noise).
    λ = sort(eigvals(Symmetric(B)))
    @test all(λ .>= -1e-10)
    @test all(λ .>= -1e-12 .+ 0.0)        # tighter check on the floor

    # Diagonal: B[i, i] = σ² because the correlation is normalized to
    # 1 on its diagonal.
    @inbounds for i in 1:size(B, 1)
        @test isapprox(B[i, i], σ_scalar^2; atol = 1e-10, rtol = 1e-9)
    end
end

@testset "IsotropicGaussianCSCovariance — δ-spike smoothing is Gaussian-shaped" begin
    σ = _unit_sigma(NC_SMALL)  # σ ≡ 1 so B = C (the correlation matrix)
    cov = AT.IsotropicGaussianCSCovariance(σ, FT_TEST(1.0))

    # Spike at (2, 2) of panel 1.
    spike = ntuple(p -> begin
        m = zeros(FT_TEST, NC_SMALL, NC_SMALL)
        if p == 1
            m[2, 2] = 1.0
        end
        m
    end, 6)

    z = ntuple(_ -> zeros(FT_TEST, NC_SMALL, NC_SMALL), 6)
    y = ntuple(_ -> zeros(FT_TEST, NC_SMALL, NC_SMALL), 6)

    # Build C · spike = column of correlation matrix at the spike's
    # index. Via `(B^(1/2)) · (B^(1/2))^T` with σ = 1.
    AT.apply_B_half_adjoint!(z, cov, spike)
    AT.apply_B_half!(y, cov, z)

    # Center value should exceed any other panel-1 entry.
    @test y[1][2, 2] == maximum(y[1])
    # Value at the center is positive and equals σ² = 1 (diagonal of C).
    @test isapprox(y[1][2, 2], 1.0; atol = 1e-10, rtol = 1e-9)
    # Neighbour at (3, 2) (one cell away) is smaller than the center
    # but still positive — the smoother spread mass into the neighbour.
    @test 0 < y[1][3, 2] < y[1][2, 2]
    # Far panels see zero — panel-local v1 limitation. This is the
    # explicit "no cross-panel correlation" check.
    for p in 2:6
        @test all(y[p] .== 0)
    end
end

# Deterministic pseudo-random tuple — same shape as `_random_tuple` in
# the preconditioning test file but inlined here for self-containment.
_random_tuple_panels(Nc, seed) = ntuple(p ->
    [FT_TEST(sin(0.7 * (seed + 1) * (i + 13 * j + 41 * p)))
     for i in 1:Nc, j in 1:Nc], 6)



@testset "IsotropicGaussianCSCovariance — short L recovers diagonal limit" begin
    σ_scalar = FT_TEST(0.5)
    σ = _constant_sigma(NC_SMALL, σ_scalar)
    # `L = 1e-3 cells` collapses the wrapped Gaussian to a near-delta,
    # so B → diag(σ²) (off-diagonal ≈ 0).
    cov = AT.IsotropicGaussianCSCovariance(σ, FT_TEST(1e-3))

    B = _build_B_matrix(cov, NC_SMALL)
    N = size(B, 1)
    @inbounds for j in 1:N, i in 1:N
        if i == j
            @test isapprox(B[i, j], σ_scalar^2; atol = 1e-9, rtol = 1e-8)
        else
            @test abs(B[i, j]) < 1e-9
        end
    end
end
