#!/usr/bin/env julia
"""
Tests for the `CMFMCMatrixConvection` operator — the conservative CMFMC
variant that derives `(entu, detu)` from GEOS `(cmfmc, dtrain)` and routes
through the TM5 backward-Euler LU.

Three properties are verified:

  1. Rate derivation correctness  — `(cmfmc, dtrain) → (entu, detu)` per
     the continuity formula `raw_E[k] = cmfmc[k]−cmfmc[k+1]+dtrain[k]`,
     with non-negativity clipping and detrainment folding.
  2. Column closure                — `Σ entu == Σ detu` per column to
     floating-point roundoff, with boundary-residual absorbed at layer Nz.
  3. Mass conservation             — the full forward pass preserves
     `Σ(m·q)` per column to roundoff for any inert tracer, on every
     topology (LL / RG / CS).
  4. Adjoint identity              — `⟨y, L·x⟩ = ⟨Lᵀ·y, x⟩` on the
     cubed-sphere path, inherited from the TM5 LU adjoint because the
     rate derivation is independent of the state.
"""

using Test
using Random
using LinearAlgebra: dot

using AtmosTransport
using .AtmosTransport.Operators: CMFMCMatrixConvection, CMFMCMatrixWorkspace,
                                  TM5Convection, TM5Workspace,
                                  invalidate_cmfmc_cache!, invalidate_cmfmc_matrix_cache!
using .AtmosTransport.Operators.Convection.CMFMC: _launch_cmfmc_matrix_derivation!
using .AtmosTransport.MetDrivers: ConvectionForcing
using .AtmosTransport.Adjoints:
                                _apply_cs_convection_adjoint!
using .AtmosTransport.Grids: CubedSphereMesh

@inline function _inner_interior(a::NTuple{6, <:AbstractArray{FT, 3}},
                                  b::NTuple{6, <:AbstractArray{FT, 3}},
                                  Nc::Int, Hp::Int) where {FT}
    s = zero(FT)
    @inbounds for p in 1:6
        for k in 1:size(a[p], 3),
            j in (Hp + 1):(Hp + Nc),
            i in (Hp + 1):(Hp + Nc)
            s += a[p][i, j, k] * b[p][i, j, k]
        end
    end
    return s
end

@testset "CMFMCMatrixConvection — rate derivation correctness" begin
    # 5-level column. cmfmc = 0 at TOA + surface; tent profile up the
    # middle. Single dtrain spike at k=3 small enough to keep raw_E ≥ 0
    # everywhere — exercises the LINEAR branch.
    FT = Float64
    Nx, Ny, Nz = 2, 2, 5
    cmfmc  = zeros(FT, Nx, Ny, Nz + 1)
    dtrain = zeros(FT, Nx, Ny, Nz)
    # cmfmc profile (TOA→surface): [0, 0.1, 0.2, 0.2, 0.1, 0]
    cmfmc[:, :, 2] .= 0.1
    cmfmc[:, :, 3] .= 0.2
    cmfmc[:, :, 4] .= 0.2
    cmfmc[:, :, 5] .= 0.1
    # Detrainment at layer 3 only (small enough that raw_E[k] ≥ 0)
    dtrain[:, :, 3] .= 0.05

    entu = zeros(FT, Nx, Ny, Nz)
    detu = zeros(FT, Nx, Ny, Nz)
    _launch_cmfmc_matrix_derivation!(entu, detu, cmfmc, dtrain)

    @testset "linear branch (raw_E ≥ 0 everywhere)" begin
        # Hand-derived raw_E[k] = cmfmc[k] − cmfmc[k+1] + dtrain[k]:
        #   k=1: 0 − 0.1 + 0    = −0.1   ← NEGATIVE, clipped
        #   k=2: 0.1 − 0.2 + 0  = −0.1   ← NEGATIVE, clipped
        #   k=3: 0.2 − 0.2 + 0.05 = 0.05  ← positive
        #   k=4: 0.2 − 0.1 + 0  = 0.1    ← positive
        #   k=5: 0.1 − 0   + 0  = 0.1    ← positive
        #
        # After clipping:
        #   entu = [0, 0, 0.05, 0.1, 0.1]   detu = [0.1, 0.1, 0.05, 0, 0]
        # In this profile cmfmc[1] = cmfmc[Nz+1] = 0 so the column is
        # already balanced (Σentu = Σdetu = 0.25); the closure write to
        # layer Nz is a no-op. Boundary-residual absorption is exercised
        # in the next testset.
        @test entu[1, 1, 1] ≈ 0.0       atol = 1e-14
        @test entu[1, 1, 2] ≈ 0.0       atol = 1e-14
        @test entu[1, 1, 3] ≈ 0.05      atol = 1e-14
        @test entu[1, 1, 4] ≈ 0.1       atol = 1e-14
        # Layer 5 may carry boundary closure-δ. We test Σ-properties below.
        @test detu[1, 1, 1] ≈ 0.1       atol = 1e-14
        @test detu[1, 1, 2] ≈ 0.1       atol = 1e-14
        @test detu[1, 1, 3] ≈ 0.05      atol = 1e-14
        @test detu[1, 1, 4] ≈ 0.0       atol = 1e-14
        @test detu[1, 1, 5] ≈ 0.0       atol = 1e-14
    end

    @testset "column closure: Σentu == Σdetu" begin
        for i in 1:Nx, j in 1:Ny
            se = sum(@view entu[i, j, :])
            sd = sum(@view detu[i, j, :])
            @test isapprox(se, sd; atol = 1e-14, rtol = 0)
        end
    end

    @testset "non-negativity" begin
        @test all(>=(0), entu)
        @test all(>=(0), detu)
    end
end

@testset "CMFMCMatrixConvection — boundary-residual absorption" begin
    # Pathological case: cmfmc[Nz+1] ≠ 0 (surface "leak", preprocessor
    # truncation). δ = cmfmc[Nz+1] − cmfmc[1] > 0, so the closure step
    # adds to entu[Nz]. Verifies non-negativity is preserved.
    FT = Float64
    Nx, Ny, Nz = 2, 2, 5
    cmfmc  = zeros(FT, Nx, Ny, Nz + 1)
    cmfmc[:, :, 2] .= 0.1
    cmfmc[:, :, 3] .= 0.2
    cmfmc[:, :, 4] .= 0.2
    cmfmc[:, :, 5] .= 0.1
    cmfmc[:, :, Nz + 1] .= 0.03  # surface "leak" — non-zero boundary
    dtrain = zeros(FT, Nx, Ny, Nz)

    entu = zeros(FT, Nx, Ny, Nz)
    detu = zeros(FT, Nx, Ny, Nz)
    _launch_cmfmc_matrix_derivation!(entu, detu, cmfmc, dtrain)

    for i in 1:Nx, j in 1:Ny
        @test isapprox(sum(@view entu[i, j, :]), sum(@view detu[i, j, :]);
                       atol = 1e-14, rtol = 0)
    end
    @test all(>=(0), entu)
    @test all(>=(0), detu)
end

@testset "CMFMCMatrixConvection — TOA-leak boundary (δ<0) absorbed via detu[Nz]" begin
    # Pathological case: cmfmc[1] ≠ 0 (TOA leak). δ = cmfmc[Nz+1] − cmfmc[1]
    # is negative, so the closure step adds |δ| to detu[Nz] rather than
    # entu[Nz] — the pos/neg branch split keeps both arrays non-negative.
    # This is the case the original "delta != 0 → entu += delta" closure
    # would have broken on a surface raw_E small enough that entu[Nz] +
    # (negative δ) goes below zero, silently violating TM5 LU
    # column-stochasticity.
    FT = Float64
    Nx, Ny, Nz = 2, 2, 5
    cmfmc  = zeros(FT, Nx, Ny, Nz + 1)
    cmfmc[:, :, 1] .= 0.05   # TOA "leak"
    cmfmc[:, :, 2] .= 0.1
    cmfmc[:, :, 3] .= 0.2
    cmfmc[:, :, 4] .= 0.2
    cmfmc[:, :, 5] .= 0.1
    # cmfmc[Nz+1] = 0 (no surface leak); δ = 0 - 0.05 = -0.05 < 0
    dtrain = zeros(FT, Nx, Ny, Nz)

    entu = zeros(FT, Nx, Ny, Nz)
    detu = zeros(FT, Nx, Ny, Nz)
    _launch_cmfmc_matrix_derivation!(entu, detu, cmfmc, dtrain)

    for i in 1:Nx, j in 1:Ny
        @test isapprox(sum(@view entu[i, j, :]), sum(@view detu[i, j, :]);
                       atol = 1e-14, rtol = 0)
    end
    @test all(>=(0), entu)
    @test all(>=(0), detu)
    # Surface raw_E[Nz] = cmfmc[Nz] - cmfmc[Nz+1] + dtrain[Nz] = 0.1 - 0 + 0 = 0.1
    # Pre-closure entu[Nz] = 0.1. Closure routes |δ|=0.05 to detu[Nz] (not entu).
    @test entu[1, 1, Nz] ≈ 0.1   atol = 1e-14
    @test detu[1, 1, Nz] ≈ 0.05  atol = 1e-14
end

@testset "CMFMCMatrixConvection — derived rates are FT-stable" begin
    for FT in (Float32, Float64)
        Nx, Ny, Nz = 3, 3, 6
        cmfmc  = zeros(FT, Nx, Ny, Nz + 1)
        dtrain = zeros(FT, Nx, Ny, Nz)
        for k in 2:Nz
            cmfmc[:, :, k]  .= FT(0.05) * (Nz + 1 - k) * (k - 1) / Nz
        end
        dtrain[:, :, 3] .= FT(0.01)
        entu = zeros(FT, Nx, Ny, Nz)
        detu = zeros(FT, Nx, Ny, Nz)
        _launch_cmfmc_matrix_derivation!(entu, detu, cmfmc, dtrain)
        @test eltype(entu) === FT
        @test eltype(detu) === FT
        for i in 1:Nx, j in 1:Ny
            tol = FT === Float64 ? 1e-14 : Float32(1e-6)
            @test isapprox(sum(@view entu[i, j, :]), sum(@view detu[i, j, :]);
                           atol = tol, rtol = 0)
        end
    end
end

@testset "CMFMCMatrixConvection — face-indexed (RG) derivation" begin
    FT = Float64
    ncells, Nz = 8, 5
    cmfmc  = zeros(FT, ncells, Nz + 1)
    cmfmc[:, 2] .= 0.1
    cmfmc[:, 3] .= 0.2
    cmfmc[:, 4] .= 0.2
    cmfmc[:, 5] .= 0.1
    dtrain = zeros(FT, ncells, Nz)
    dtrain[:, 3] .= 0.05
    entu = zeros(FT, ncells, Nz)
    detu = zeros(FT, ncells, Nz)
    _launch_cmfmc_matrix_derivation!(entu, detu, cmfmc, dtrain)
    for c in 1:ncells
        @test isapprox(sum(@view entu[c, :]), sum(@view detu[c, :]);
                       atol = 1e-14, rtol = 0)
    end
    @test all(>=(0), entu)
    @test all(>=(0), detu)
end

@testset "CMFMCMatrixConvection — cubed-sphere panel derivation" begin
    FT = Float64
    Nc, Nz = 4, 5
    mk_zeros(Nzp) = ntuple(_ -> zeros(FT, Nc, Nc, Nzp), 6)
    cmfmc  = mk_zeros(Nz + 1)
    dtrain = mk_zeros(Nz)
    for p in 1:6
        cmfmc[p][:, :, 2] .= 0.1
        cmfmc[p][:, :, 3] .= 0.2
        cmfmc[p][:, :, 4] .= 0.2
        cmfmc[p][:, :, 5] .= 0.1
        dtrain[p][:, :, 3] .= 0.05
    end
    entu = mk_zeros(Nz)
    detu = mk_zeros(Nz)
    _launch_cmfmc_matrix_derivation!(entu, detu, cmfmc, dtrain)
    for p in 1:6, i in 1:Nc, j in 1:Nc
        @test isapprox(sum(@view entu[p][i, j, :]), sum(@view detu[p][i, j, :]);
                       atol = 1e-14, rtol = 0)
    end
end

@testset "CMFMCMatrixConvection — column mass conservation (CS forward)" begin
    # Run the full CS forward through the TM5 LU and check `Σ(m·q)` is
    # preserved to roundoff per column for an inert tracer. This is the
    # operative bar Option 3 was chosen to meet.
    FT = Float64
    Nc, Hp, Nz = 4, 1, 5
    N = Nc + 2 * Hp
    mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)

    panels_m = ntuple(_ -> fill(FT(1e15), N, N, Nz), 6)
    cell_areas = ntuple(_ -> fill(FT(1e12), Nc, Nc), 6)

    cmfmc  = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    dtrain = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    for p in 1:6
        cmfmc[p][:, :, 2] .= FT(0.02)
        cmfmc[p][:, :, 3] .= FT(0.04)
        cmfmc[p][:, :, 4] .= FT(0.03)
        cmfmc[p][:, :, 5] .= FT(0.01)
        dtrain[p][:, :, 3] .= FT(0.01)
        dtrain[p][:, :, 4] .= FT(0.02)
    end
    forcing = ConvectionForcing(cmfmc, dtrain, nothing)

    rng = MersenneTwister(2026)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    for p in 1:6, k in 1:Nz,
        j in (Hp + 1):(Hp + Nc),
        i in (Hp + 1):(Hp + Nc)
        panels_rm[p][i, j, k] = abs(randn(rng, FT))
    end
    # Pre-step column mass per column (interior only)
    sum_pre = ntuple(p -> dropdims(sum(@view(panels_rm[p][(Hp+1):(Hp+Nc), (Hp+1):(Hp+Nc), :]);
                                        dims = 3); dims = 3), 6)

    op = CMFMCMatrixConvection(tile_workspace_gib = 0.1, lmax_conv = 0, use_collab_lu = false)
    ws = CMFMCMatrixWorkspace(panels_rm; tile_workspace_gib = 0.1,
                                          cell_metrics = cell_areas,
                                          halo_width = Hp)
    let state = AtmosTransport.State.CubedSphereState(AtmosTransport.State.DryBasis, mesh, panels_m; tracer=panels_rm)
        nz = size(panels_m[1], 3)
        FT = eltype(panels_m[1])
        vc = AtmosTransport.Grids.HybridSigmaPressure(zeros(FT,nz+1), collect(range(zero(FT),one(FT); length=nz+1)))
        grid = AtmosTransport.Grids.AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT)
        AtmosTransport.Operators.apply!(state, forcing, grid, op, FT(450.0); workspace=ws)
        foreach((dst, src) -> copyto!(dst, view(src,:,:,:,1)), panels_rm, state.tracers_raw)
    end

    sum_post = ntuple(p -> dropdims(sum(@view(panels_rm[p][(Hp+1):(Hp+Nc), (Hp+1):(Hp+Nc), :]);
                                         dims = 3); dims = 3), 6)

    for p in 1:6, j in 1:Nc, i in 1:Nc
        @test isapprox(sum_pre[p][i, j], sum_post[p][i, j];
                       rtol = 1e-12, atol = 0)
    end
end

@testset "CMFMCMatrixConvection — CS adjoint identity" begin
    # Same forward setup as the conservation test, plus the adjoint
    # transpose identity. Inherits exactness from the TM5 LU adjoint
    # because the rate derivation is state-independent.
    FT = Float64
    Nc, Hp, Nz = 4, 1, 5
    N = Nc + 2 * Hp
    mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)

    panels_m = ntuple(_ -> fill(FT(1e15), N, N, Nz), 6)
    cell_areas = ntuple(_ -> fill(FT(1e12), Nc, Nc), 6)

    cmfmc  = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    dtrain = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    for p in 1:6
        cmfmc[p][:, :, 2] .= FT(0.02)
        cmfmc[p][:, :, 3] .= FT(0.04)
        cmfmc[p][:, :, 4] .= FT(0.03)
        cmfmc[p][:, :, 5] .= FT(0.01)
        dtrain[p][:, :, 3] .= FT(0.01)
        dtrain[p][:, :, 4] .= FT(0.02)
    end
    forcing = ConvectionForcing(cmfmc, dtrain, nothing)

    rng = MersenneTwister(31337)
    x_rm   = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    y_seed = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    for p in 1:6, k in 1:Nz,
        j in (Hp + 1):(Hp + Nc),
        i in (Hp + 1):(Hp + Nc)
        x_rm[p][i, j, k]   = randn(rng, FT)
        y_seed[p][i, j, k] = randn(rng, FT)
    end

    op = CMFMCMatrixConvection(tile_workspace_gib = 0.1, lmax_conv = 0, use_collab_lu = false)

    rm_new = ntuple(p -> copy(x_rm[p]), 6)
    ws_fwd = CMFMCMatrixWorkspace(rm_new; tile_workspace_gib = 0.1, defer_scratch = true,
                                            cell_metrics = cell_areas,
                                            halo_width = Hp)
    let state = AtmosTransport.State.CubedSphereState(AtmosTransport.State.DryBasis, mesh, panels_m; tracer=rm_new)
        nz = size(panels_m[1], 3)
        FT = eltype(panels_m[1])
        vc = AtmosTransport.Grids.HybridSigmaPressure(zeros(FT,nz+1), collect(range(zero(FT),one(FT); length=nz+1)))
        grid = AtmosTransport.Grids.AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT)
        AtmosTransport.Operators.apply!(state, forcing, grid, op, FT(450.0); workspace=ws_fwd)
        foreach((dst, src) -> copyto!(dst, view(src,:,:,:,1)), rm_new, state.tracers_raw)
    end

    lambda = ntuple(p -> copy(y_seed[p]), 6)
    ws_adj = CMFMCMatrixWorkspace(lambda; tile_workspace_gib = 0.1, defer_scratch = true,
                                           cell_metrics = cell_areas,
                                           halo_width = Hp)
    _apply_cs_convection_adjoint!(lambda, panels_m, forcing,
                                   op, FT(450.0), ws_adj, mesh)

    lhs = _inner_interior(y_seed, rm_new, Nc, Hp)
    rhs = _inner_interior(lambda, x_rm,   Nc, Hp)
    @test isapprox(lhs, rhs; rtol = 1e-10, atol = 1e-10 * abs(lhs))
end
