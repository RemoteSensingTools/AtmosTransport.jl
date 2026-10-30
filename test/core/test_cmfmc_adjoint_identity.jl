#!/usr/bin/env julia
"""
Adjoint-identity verification for the CMFMC convection operator.

Confirms `⟨y, L·x⟩ = ⟨Lᵀ·y, x⟩` to machine precision (Float64) and
single-precision tolerance (Float32). The forward `L` is the
production CMFMC column update (`_cmfmc_cs_column!`, one panel per launch in
`_cmfmc_cs_panels_column_kernel!`), which the adjoint path replays; its transpose is `_cmfmc_cs_panel_column_single_adjoint_kernel!`
in `src/Adjoints/ConvectionAdjoint.jl` (GG1 surface-up cloud-base scan, CC1
kg/m² well-mix with cloud-base closure, C3 entrn≥0 guard). A second testset
pins the replay to the production `apply_convection!` bit for bit.
"""

using Test
using Random
using LinearAlgebra: dot

using AtmosTransport
using .AtmosTransport.Operators: CMFMCWorkspace, CMFMCConvection,
                                  invalidate_cmfmc_cache!, apply_convection!
using .AtmosTransport.MetDrivers: ConvectionForcing
using .AtmosTransport.Adjoints:
                                _apply_cs_convection_adjoint!
using .AtmosTransport.Grids: CubedSphereMesh

# Inner product over the INTERIOR `[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]`
# of a halo-padded panel — the kernel only touches interior cells.
@inline function _inner_interior(a::NTuple{6, <:AbstractArray{FT, 3}},
                                  b::NTuple{6, <:AbstractArray{FT, 3}},
                                  Nc::Int, Hp::Int) where FT
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

# Build a deterministic CMFMC / DTRAIN forcing whose updraft budget
# closes (entrn ≥ 0 at every layer) so that the C3 guard never
# zero-coefficients the qc update — exercising the LINEAR branch of
# every kernel statement.
function _cmfmc_adjoint_test_forcing(::Type{FT}, Nc::Int, Nz::Int) where FT
    cmfmc = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    dtrain = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    # Cloud base at k = Nz - 1 (one layer below surface). cmfmc tent:
    # rises from cloud base to a peak, falls back to zero at cloud top.
    # In our TOA-first orientation:
    #   cmfmc[Nz]   = 0.02 (cloud-base inflow)
    #   cmfmc[Nz-1] = 0.02
    #   cmfmc[Nz-2] = 0.01 (some detrainment)
    #   cmfmc[Nz-3] = 0    (cloud top)
    # with dtrain to balance.
    for p in 1:6, j in 1:Nc, i in 1:Nc
        cmfmc[p][i, j, Nz]     = FT(0.02)
        cmfmc[p][i, j, Nz - 1] = FT(0.02)
        cmfmc[p][i, j, Nz - 2] = FT(0.01)
        # cmfmc[Nz-3] stays zero. Detrainment matches cmfmc step:
        dtrain[p][i, j, Nz - 2] = FT(0.01)  # half the updraft escapes here
        dtrain[p][i, j, Nz - 3] = FT(0.01)  # the rest at cloud top
    end
    return cmfmc, dtrain
end

@testset "CMFMC adjoint identity" begin
    for FT in (Float64, Float32)
        @testset "transposition $(FT)" begin
            Nc, Hp, Nz = 4, 1, 5
            N = Nc + 2 * Hp
            mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)

            # Physically reasonable air mass (CS panel cell area ~ 1e12 m²
            # for Nc=4 globally, so kg-per-cell ~ 1e15 kg). Use a flat
            # column thickness so bmass = m / cell_area is uniform.
            panels_m = ntuple(_ -> fill(FT(1e15), N, N, Nz), 6)
            cell_areas = ntuple(_ -> fill(FT(1e12), Nc, Nc), 6)

            cmfmc, dtrain = _cmfmc_adjoint_test_forcing(FT, Nc, Nz)
            forcing = ConvectionForcing(cmfmc, dtrain, nothing)

            # Random initial rm and adjoint seed restricted to the
            # interior region — the kernel doesn't touch halos.
            rng = MersenneTwister(31337)
            x_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
            y_seed = ntuple(_ -> zeros(FT, N, N, Nz), 6)
            for p in 1:6, k in 1:Nz,
                j in (Hp + 1):(Hp + Nc),
                i in (Hp + 1):(Hp + Nc)
                x_rm[p][i, j, k]   = randn(rng, FT)
                y_seed[p][i, j, k] = randn(rng, FT)
            end

            # Single-substep dt: choose small enough that CFL cache picks
            # n_sub = 1 (max(cmfmc·dt/bmass) well below 0.5).
            dt = FT(60.0)

            # Forward: copy x → rm_new, apply L.
            rm_new = ntuple(p -> copy(x_rm[p]), 6)
            ws_fwd = CMFMCWorkspace(rm_new; cell_metrics = cell_areas)
            let state = AtmosTransport.State.CubedSphereState(AtmosTransport.State.DryBasis, mesh, panels_m; tracer=rm_new)
                nz = size(panels_m[1], 3)
                FT = eltype(panels_m[1])
                vc = AtmosTransport.Grids.HybridSigmaPressure(zeros(FT,nz+1), collect(range(zero(FT),one(FT); length=nz+1)))
                grid = AtmosTransport.Grids.AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT)
                AtmosTransport.Operators.apply!(state, forcing, grid, CMFMCConvection(), dt; workspace=ws_fwd)
                foreach((dst, src) -> copyto!(dst, view(src,:,:,:,1)), rm_new, state.tracers_raw)
            end
            @test ws_fwd.cached_n_sub[] == 1   # ensure single-substep regime

            # Adjoint: copy y → λ, apply Lᵀ.
            lambda = ntuple(p -> copy(y_seed[p]), 6)
            ws_adj = CMFMCWorkspace(lambda; cell_metrics = cell_areas)
            _apply_cs_convection_adjoint!(lambda, panels_m, forcing,
                                           CMFMCConvection(), dt, ws_adj, mesh)

            # Inner products on the interior — halos stay zero on both
            # sides so they don't contribute, but the explicit loop
            # makes the assertion unambiguous.
            lhs = _inner_interior(y_seed, rm_new, Nc, Hp)
            rhs = _inner_interior(lambda, x_rm,   Nc, Hp)

            tol = FT === Float64 ? 1e-10 : 1f-3
            @test isapprox(lhs, rhs; rtol = tol, atol = tol * abs(lhs))
        end
    end

    @testset "no-convection columns: adjoint preserves λ to round-trip ε" begin
        # When cmfmc is identically zero everywhere, the forward is the
        # identity and the adjoint must be the identity too (up to one
        # ULP from the `(λ·m_k)/m_k` round-trip the Pass-2 adjoint
        # performs even when all coefficients vanish — at m_k = 1e15
        # that's ~1 ULP in the lambda).
        FT = Float64
        Nc, Hp, Nz = 4, 1, 4
        N = Nc + 2 * Hp
        mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)

        panels_m = ntuple(_ -> fill(FT(1e15), N, N, Nz), 6)
        cell_areas = ntuple(_ -> fill(FT(1e12), Nc, Nc), 6)
        cmfmc = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
        dtrain = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
        forcing = ConvectionForcing(cmfmc, dtrain, nothing)

        rng = MersenneTwister(7)
        lambda = ntuple(_ -> zeros(FT, N, N, Nz), 6)
        for p in 1:6, k in 1:Nz,
            j in (Hp + 1):(Hp + Nc),
            i in (Hp + 1):(Hp + Nc)
            lambda[p][i, j, k] = randn(rng, FT)
        end
        lambda_before = deepcopy(lambda)

        ws = CMFMCWorkspace(lambda; cell_metrics = cell_areas)
        _apply_cs_convection_adjoint!(lambda, panels_m, forcing,
                                       CMFMCConvection(), FT(60.0), ws, mesh)

        # Identity-modulo-round-trip-ε: every cell is preserved to
        # within a few ULPs (≪ rtol = 1e-12 is fine even at m_k=1e15).
        @test all(isapprox(lambda[p], lambda_before[p];
                            rtol = 1e-12, atol = 0) for p in 1:6)
    end
end

# A deep convective column over uneven air masses: cloud base at layer 18 of 24,
# so the sub-cloud mean sums seven layers, and DTRAIN distinct from the
# detrainment derived from CMFMC (or missing).
function _deep_cmfmc_case(::Type{FT}, Nc, Hp, Nz, with_dtrain) where FT
    N = Nc + 2Hp
    rng = MersenneTwister(11)
    panels_m = ntuple(_ -> [FT(1e15) * (1 + FT(0.3) * rand(rng, FT)) for _ in 1:N, _ in 1:N, _ in 1:Nz], 6)
    cmfmc = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    dtrain = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    for p in 1:6
        cmfmc[p][:, :, 6:18] .= FT(0.02)            # updraft between the cloud base (layer 18) and layer 5
        cmfmc[p][:, :, 10:12] .= FT(0.03)           # entrainment, then detrainment aloft
        dtrain[p][:, :, 5] .= FT(0.015)
        dtrain[p][:, :, 9] .= FT(0.005)
    end
    return panels_m, ConvectionForcing(cmfmc, with_dtrain ? dtrain : nothing, nothing)
end

@testset "CMFMC adjoint identity: deep column, several substeps" begin
    for FT in (Float64, Float32), with_dtrain in (true, false)
        Nc, Hp, Nz = 3, 1, 24
        N = Nc + 2Hp
        mesh = CubedSphereMesh(; Nc = Nc, Hp = Hp, FT = FT)
        cell_areas = ntuple(_ -> fill(FT(1e12), Nc, Nc), 6)
        panels_m, forcing = _deep_cmfmc_case(FT, Nc, Hp, Nz, with_dtrain)
        rng = MersenneTwister(29)
        x = ntuple(_ -> zeros(FT, N, N, Nz), 6)
        y = ntuple(_ -> zeros(FT, N, N, Nz), 6)
        for p in 1:6, k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
            x[p][i, j, k] = randn(rng, FT)
            y[p][i, j, k] = randn(rng, FT)
        end
        dt = FT(120_000)                            # CMFMC CFL > 1: several substeps
        Lx = ntuple(p -> copy(x[p]), 6)
        ws = CMFMCWorkspace(Lx; cell_metrics = cell_areas)
        let state = AtmosTransport.State.CubedSphereState(AtmosTransport.State.DryBasis, mesh, panels_m; tracer=Lx)
            nz = size(panels_m[1], 3)
            FT = eltype(panels_m[1])
            vc = AtmosTransport.Grids.HybridSigmaPressure(zeros(FT,nz+1), collect(range(zero(FT),one(FT); length=nz+1)))
            grid = AtmosTransport.Grids.AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT)
            AtmosTransport.Operators.apply!(state, forcing, grid, CMFMCConvection(), dt; workspace=ws)
            foreach((dst, src) -> copyto!(dst, view(src,:,:,:,1)), Lx, state.tracers_raw)
        end
        @test ws.cached_n_sub[] > 1
        LTy = ntuple(p -> copy(y[p]), 6)
        _apply_cs_convection_adjoint!(LTy, panels_m, forcing, CMFMCConvection(), dt,
                                      CMFMCWorkspace(LTy; cell_metrics = cell_areas), mesh)
        lhs, rhs = _inner_interior(y, Lx, Nc, Hp), _inner_interior(LTy, x, Nc, Hp)
        tol = FT === Float64 ? 1e-10 : 1f-4
        @test isapprox(lhs, rhs; rtol = tol, atol = tol * abs(lhs))
    end
end
