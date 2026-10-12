#!/usr/bin/env julia
"""
Tests for diffusion pressure recovery and hydrostatic `dz` helpers. Checks
cover structured and face-indexed CPU arrays, cubed-sphere tuples, the
runtime humidity-selection contract, and optional CUDA execution. Metal is
covered by `test/diagnostic/test_metal_kernels.jl` on supported CI hosts.

  1. Numerical correctness vs hand-computed dz at a few sample columns.
     For each `(i, j, k)`, dz should reduce to
         dz = R · T_v / g · delp / p_ctr
     with `T_v = T · (1 + 0.61·qv)`. We seed nonuniform T and qv and
     verify the kernel matches the hand formula.

  2. Cubed-sphere panel-tuple dispatch produces the same per-cell dz
     as the single-array variant on a synthetic per-panel column.

  3. dz_v != dz_c when virtual T differs from `T_ref = 260 K` — the
     whole point of D6.

"""

using Test
import AtmosTransport
using .AtmosTransport.Operators.Diffusion: fill_dz_hydrostatic_virtualT!,
                                            fill_dz_hydrostatic_constT!,
                                            recover_total_surface_pressure!
using .AtmosTransport.Parameters: R_DRY_AIR, STANDARD_GRAVITY
using .AtmosTransport.Models: _diffusion_pressure_humidity,
                               _diffusion_total_surface_pressure!
using .AtmosTransport.Operators: DiffusionWorkspace
using .AtmosTransport.Operators.Diffusion: _stage_hybrid_coefficients!

const _R_DRY  = R_DRY_AIR
const _G_REF  = STANDARD_GRAVITY

const HAS_CUDA_DZ = try
    using CUDA
    CUDA.functional()
catch
    false
end

@testset "dry and total surface pressure give equivalent diffusion dz" begin
    FT = Float64
    Nx, Ny, Nz = 3, 2, 4
    ak = FT[100, 500, 1200, 800, 0]
    bk = FT[0, 0.1, 0.35, 0.7, 1]
    ps_total = FT[96_000 + 300i + 200j for i in 1:Nx, j in 1:Ny]
    qv = FT[0.001 + 0.003k + 0.0002i for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    ps_dry = similar(ps_total)
    for j in 1:Ny, i in 1:Nx
        ps_dry[i, j] = sum((1 - qv[i, j, k]) *
                           ((ak[k + 1] - ak[k]) +
                            (bk[k + 1] - bk[k]) * ps_total[i, j]) for k in 1:Nz)
    end

    recovered = similar(ps_total)
    recover_total_surface_pressure!(recovered, ps_dry, qv, ak, bk)
    dz_total, dz_dry = zeros(FT, Nx, Ny, Nz), zeros(FT, Nx, Ny, Nz)
    fill_dz_hydrostatic_constT!(dz_total, ps_total, ak, bk)
    fill_dz_hydrostatic_constT!(dz_dry, recovered, ak, bk)
    @test recovered ≈ ps_total rtol = 32eps(FT)
    @test dz_dry ≈ dz_total rtol = 64eps(FT)

    if HAS_CUDA_DZ
        recovered_gpu = CUDA.zeros(FT, Nx, Ny)
        recover_total_surface_pressure!(recovered_gpu, CUDA.CuArray(ps_dry),
                                        CUDA.CuArray(qv), ak, bk)
        dz_gpu = CUDA.zeros(FT, Nx, Ny, Nz)
        fill_dz_hydrostatic_constT!(dz_gpu, recovered_gpu, ak, bk)
        @test Array(recovered_gpu) ≈ ps_total rtol = 32eps(FT)
        @test Array(dz_gpu) ≈ dz_total rtol = 64eps(FT)
    end
end

@testset "dry-pressure recovery handles pure-pressure and unobservable coordinates" begin
    FT = Float64

    # Surface pressure does not enter an A-only coordinate, so dry→total
    # recovery is both unnecessary and mathematically unobservable. Preserve
    # the finite input instead of evaluating 0/0. Exercise both runtime shapes.
    ak_pressure = FT[0, 90_000]
    bk_pressure = FT[0, 0]
    ps_matrix = fill(FT(90_000), 3, 2)
    qv_matrix = zeros(FT, 3, 2, 1)
    recovered_matrix = fill(FT(NaN), 3, 2)
    recover_total_surface_pressure!(recovered_matrix, ps_matrix, qv_matrix,
                                    ak_pressure, bk_pressure)
    @test recovered_matrix == ps_matrix

    ps_vector = fill(FT(90_000), 5)
    qv_vector = zeros(FT, 5, 1)
    recovered_vector = fill(FT(NaN), 5)
    recover_total_surface_pressure!(recovered_vector, ps_vector, qv_vector,
                                    ak_pressure, bk_pressure)
    @test recovered_vector == ps_vector

    dz_matrix = zeros(FT, 3, 2, 1)
    dz_vector = zeros(FT, 5, 1)
    fill_dz_hydrostatic_constT!(dz_matrix, recovered_matrix, ak_pressure, bk_pressure)
    fill_dz_hydrostatic_constT!(dz_vector, recovered_vector, ak_pressure, bk_pressure)
    @test all(isfinite, dz_matrix)
    @test all(isfinite, dz_vector)

    # When B does affect pressure but saturated/unphysical humidity removes
    # every observable B increment, recovery must fail explicitly rather than
    # emitting NaN/Inf into the following dz kernel.
    ak_sigma = FT[0, 0]
    bk_sigma = FT[0, 1]
    @test_throws ArgumentError recover_total_surface_pressure!(
        similar(ps_matrix), ps_matrix, ones(FT, 3, 2, 1), ak_sigma, bk_sigma)
    @test_throws ArgumentError recover_total_surface_pressure!(
        similar(ps_vector), ps_vector, ones(FT, 5, 1), ak_sigma, bk_sigma)
end

@testset "diffusion runtime humidity selection and missing-humidity fallback" begin
    qv_start = fill(0.01, 2, 2, 2)
    qv_vdiff = fill(0.02, 2, 2, 2)
    @test _diffusion_pressure_humidity((; qv_start, vdiff = (; qv = qv_vdiff))) === qv_start
    @test _diffusion_pressure_humidity((; qv_start = nothing,
                                        vdiff = (; qv = qv_vdiff))) === qv_vdiff
    @test _diffusion_pressure_humidity((; qv_start = nothing, vdiff = nothing)) === nothing

    ps = fill(90_000.0, 2, 2)
    workspace = DiffusionWorkspace(ones(2, 2, 2))
    coefficients = _stage_hybrid_coefficients!(workspace.hybrid_coefficients,
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.5, 1.0])
    returned = Ref{Any}()
    @test_logs (:warn, r"dry surface pressure without endpoint or VDIFF humidity") begin
        returned[] = _diffusion_total_surface_pressure!(
            AtmosTransport.DrySurfacePressure(), workspace.factors,
            ps, [0.0, 0.0, 0.0], [0.0, 0.5, 1.0], nothing, coefficients)
    end
    @test returned[] === ps
end

@testset "fill_dz_hydrostatic_virtualT! — single 3D array matches hand formula" begin
    FT = Float64
    Nx, Ny, Nz = 3, 3, 6

    # Build a simple hybrid sigma-pressure grid: pure pressure at top,
    # pure sigma at bottom.
    ak = collect(LinRange(FT(100.0), FT(0.0), Nz + 1))       # Pa at top → 0
    bk = collect(LinRange(FT(0.0),  FT(1.0),  Nz + 1))        # σ scaling at bottom

    # Surface pressure varies a bit.
    ps = FT[100000.0 + 200 * (i + j) for i in 1:Nx, j in 1:Ny]

    # Layer-mean T and qv with a non-trivial profile.
    t_lyr  = zeros(FT, Nx, Ny, Nz)
    qv_lyr = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz
        t_lyr[:, :, k]  .= FT(220.0 + 8 * k)         # 228 K (TOA) → 268 K (surface)
        qv_lyr[:, :, k] .= FT(0.001 + 0.002 * k)     # rises near surface
    end

    dz = zeros(FT, Nx, Ny, Nz)
    fill_dz_hydrostatic_virtualT!(dz, t_lyr, qv_lyr, ps, ak, bk)

    # Spot-check a few cells against the hand formula.
    for (i, j, k) in ((1, 1, 1), (2, 2, 3), (3, 3, Nz))
        delp  = (ak[k+1] - ak[k]) + (bk[k+1] - bk[k]) * ps[i, j]
        p_ctr = 0.5 * (ak[k] + ak[k+1] + (bk[k] + bk[k+1]) * ps[i, j])
        Tv    = t_lyr[i, j, k] * (1 + 0.61 * qv_lyr[i, j, k])
        expected = _R_DRY * Tv / _G_REF * delp / p_ctr
        @test isapprox(dz[i, j, k], expected; rtol = 1e-12)
    end
end

@testset "fill_dz_hydrostatic_virtualT! — clamps negative qv to zero" begin
    # Tiny post-regrid negative-qv values shouldn't make T_v drop below T.
    FT = Float64
    Nx, Ny, Nz = 2, 2, 4
    ak = collect(LinRange(FT(100.0), FT(0.0), Nz + 1))
    bk = collect(LinRange(FT(0.0),  FT(1.0),  Nz + 1))
    ps = fill(FT(100000.0), Nx, Ny)
    t_lyr = fill(FT(250.0), Nx, Ny, Nz)
    qv_neg = fill(FT(-1e-8), Nx, Ny, Nz)       # numerical noise
    qv_zero = fill(FT(0.0), Nx, Ny, Nz)
    dz_neg = zeros(FT, Nx, Ny, Nz)
    dz_zero = zeros(FT, Nx, Ny, Nz)
    fill_dz_hydrostatic_virtualT!(dz_neg, t_lyr, qv_neg, ps, ak, bk)
    fill_dz_hydrostatic_virtualT!(dz_zero, t_lyr, qv_zero, ps, ak, bk)
    @test dz_neg ≈ dz_zero atol = 1e-14
end

@testset "fill_dz_hydrostatic_virtualT! — CS panel-tuple dispatch" begin
    FT = Float64
    Nc, Nz = 4, 5
    ak = collect(LinRange(FT(100.0), FT(0.0), Nz + 1))
    bk = collect(LinRange(FT(0.0),  FT(1.0),  Nz + 1))

    # Six different panels with different profiles to confirm per-panel data.
    dz_panels = ntuple(p -> zeros(FT, Nc, Nc, Nz), 6)
    t_panels  = ntuple(p -> fill(FT(220 + 5 * p), Nc, Nc, Nz), 6)
    qv_panels = ntuple(p -> fill(FT(0.002 * p), Nc, Nc, Nz), 6)
    ps_panels = ntuple(p -> fill(FT(95000 + 1000 * p), Nc, Nc), 6)

    fill_dz_hydrostatic_virtualT!(dz_panels, t_panels, qv_panels,
                                   ps_panels, ak, bk)

    # Reference: per-panel single-array call.
    for p in 1:6
        ref = zeros(FT, Nc, Nc, Nz)
        fill_dz_hydrostatic_virtualT!(ref, t_panels[p], qv_panels[p],
                                       ps_panels[p], ak, bk)
        @test dz_panels[p] ≈ ref atol = 1e-14
    end
end

@testset "fill_dz_hydrostatic_virtualT! vs constT — differ when T_v != T_ref" begin
    # The whole point of D6 is that the two functions should disagree
    # when virtual T differs significantly from 260 K. Sanity check.
    FT = Float64
    Nx, Ny, Nz = 2, 2, 4
    ak = collect(LinRange(FT(100.0), FT(0.0), Nz + 1))
    bk = collect(LinRange(FT(0.0),  FT(1.0),  Nz + 1))
    ps = fill(FT(100000.0), Nx, Ny)
    t_lyr  = fill(FT(290.0), Nx, Ny, Nz)   # well above 260 K T_ref
    qv_lyr = fill(FT(0.015), Nx, Ny, Nz)
    dz_v = zeros(FT, Nx, Ny, Nz)
    dz_c = zeros(FT, Nx, Ny, Nz)
    fill_dz_hydrostatic_virtualT!(dz_v, t_lyr, qv_lyr, ps, ak, bk)
    fill_dz_hydrostatic_constT!(dz_c, ps, ak, bk)
    # T_v = 290 · 1.00915 ≈ 292.6, T_ref = 260. dz_v / dz_c ≈ 1.125.
    @test all(dz_v .> dz_c)
    @test isapprox(dz_v[1, 1, 1] / dz_c[1, 1, 1], 292.6 / 260.0; rtol = 1e-3)
end
