import AtmosTransport
include(joinpath(@__DIR__, "..", "helpers", "diffusion_geometry.jl"))

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
using .AtmosTransport.Operators.Diffusion: fill_dz_hydrostatic_virtualT_staged!,
                                            fill_dz_hydrostatic_constT_staged!,
                                            recover_total_surface_pressure_staged!
using .AtmosTransport.Parameters: R_DRY_AIR, STANDARD_GRAVITY
using .AtmosTransport.Models: _diffusion_pressure_humidity,
                               _diffusion_total_surface_pressure!
using .AtmosTransport.Operators: DiffusionWorkspace
using .AtmosTransport.Operators.Diffusion: stage_hybrid_coefficients!

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
    recover_total_surface_pressure_staged!(recovered, ps_dry, qv, diffusion_coefficients(qv, ak, bk); sync = true)
    dz_total, dz_dry = zeros(FT, Nx, Ny, Nz), zeros(FT, Nx, Ny, Nz)
    fill_dz_hydrostatic_constT_staged!(dz_total, ps_total, diffusion_coefficients(dz_total, ak, bk); sync = true)
    fill_dz_hydrostatic_constT_staged!(dz_dry, recovered, diffusion_coefficients(dz_dry, ak, bk); sync = true)
    @test recovered ≈ ps_total rtol = 32eps(FT)
    @test dz_dry ≈ dz_total rtol = 64eps(FT)

    if HAS_CUDA_DZ
        recovered_gpu = CUDA.zeros(FT, Nx, Ny)
        recover_total_surface_pressure_staged!(recovered_gpu, CUDA.CuArray(ps_dry), CUDA.CuArray(qv), diffusion_coefficients(CUDA.CuArray(qv), ak, bk); sync = true)
        dz_gpu = CUDA.zeros(FT, Nx, Ny, Nz)
        fill_dz_hydrostatic_constT_staged!(dz_gpu, recovered_gpu, diffusion_coefficients(dz_gpu, ak, bk); sync = true)
        @test Array(recovered_gpu) ≈ ps_total rtol = 32eps(FT)
        @test Array(dz_gpu) ≈ dz_total rtol = 64eps(FT)
    end
end

@testset "Float64 hybrid coefficients stage into Float32 workspaces" begin
    FT = Float32
    Nx, Ny, Nz = 2, 2, 3
    A64 = Float64[125.125, 80.25, 20.5, 0.0]
    B64 = Float64[0.0, 0.2, 0.625, 1.0]
    A32, B32 = FT.(A64), FT.(B64)
    ps_total = FT[95_000 + 100i + 50j for i in 1:Nx, j in 1:Ny]
    qv = FT[0.002 + 0.001k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    ps_dry = FT[sum((one(FT) - qv[i, j, k]) *
                    ((A32[k + 1] - A32[k]) +
                     (B32[k + 1] - B32[k]) * ps_total[i, j]) for k in 1:Nz)
                for i in 1:Nx, j in 1:Ny]

    workspace = DiffusionWorkspace(ones(FT, Nx, Ny, Nz))
    staged = stage_hybrid_coefficients!(workspace.hybrid_coefficients, A64, B64)
    @test eltype(staged.A) === FT
    @test eltype(staged.B) === FT
    @test staged.A == A32
    @test staged.B == B32

    recovered64 = zeros(FT, Nx, Ny)
    recovered32 = similar(recovered64)
    recover_total_surface_pressure_staged!(recovered64, ps_dry, qv, diffusion_coefficients(qv, A64, B64); sync = true)
    recover_total_surface_pressure_staged!(recovered32, ps_dry, qv, diffusion_coefficients(qv, A32, B32); sync = true)
    @test recovered64 == recovered32

    dz64 = zeros(FT, Nx, Ny, Nz)
    dz32 = similar(dz64)
    fill_dz_hydrostatic_constT_staged!(dz64, recovered64, diffusion_coefficients(dz64, A64, B64); sync = true)
    fill_dz_hydrostatic_constT_staged!(dz32, recovered32, diffusion_coefficients(dz32, A32, B32); sync = true)
    @test dz64 == dz32
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
    recover_total_surface_pressure_staged!(recovered_matrix, ps_matrix, qv_matrix, diffusion_coefficients(qv_matrix, ak_pressure, bk_pressure); sync = true)
    @test recovered_matrix == ps_matrix

    ps_vector = fill(FT(90_000), 5)
    qv_vector = zeros(FT, 5, 1)
    recovered_vector = fill(FT(NaN), 5)
    recover_total_surface_pressure_staged!(recovered_vector, ps_vector, qv_vector, diffusion_coefficients(qv_vector, ak_pressure, bk_pressure); sync = true)
    @test recovered_vector == ps_vector

    dz_matrix = zeros(FT, 3, 2, 1)
    dz_vector = zeros(FT, 5, 1)
    fill_dz_hydrostatic_constT_staged!(dz_matrix, recovered_matrix, diffusion_coefficients(dz_matrix, ak_pressure, bk_pressure); sync = true)
    fill_dz_hydrostatic_constT_staged!(dz_vector, recovered_vector, diffusion_coefficients(dz_vector, ak_pressure, bk_pressure); sync = true)
    @test all(isfinite, dz_matrix)
    @test all(isfinite, dz_vector)

    # When B does affect pressure but saturated/unphysical humidity removes
    # every observable B increment, recovery must fail explicitly rather than
    # emitting NaN/Inf into the following dz kernel.
    ak_sigma = FT[0, 0]
    bk_sigma = FT[0, 1]
    @test_throws ArgumentError recover_total_surface_pressure_staged!(similar(ps_matrix), ps_matrix, ones(FT, 3, 2, 1), diffusion_coefficients(ones(FT, 3, 2, 1), ak_sigma, bk_sigma); sync = true)
    @test_throws ArgumentError recover_total_surface_pressure_staged!(similar(ps_vector), ps_vector, ones(FT, 5, 1), diffusion_coefficients(ones(FT, 5, 1), ak_sigma, bk_sigma); sync = true)

    # Validation is performed on the actual accumulated slope inside the
    # recovery kernel, including non-finite inputs and arithmetic overflow.
    ps_one = fill(FT(90_000), 1, 1)
    dest_one = similar(ps_one)
    for bad_qv in (fill(FT(-Inf), 1, 1, 1), fill(FT(NaN), 1, 1, 1))
        @test_throws ArgumentError recover_total_surface_pressure_staged!(dest_one, ps_one, bad_qv, diffusion_coefficients(bad_qv, ak_sigma, bk_sigma); sync = true)
    end
    qv_overflow = fill(-floatmax(FT), 1, 1, 2)
    @test_throws ArgumentError recover_total_surface_pressure_staged!(dest_one, ps_one, qv_overflow, diffusion_coefficients(qv_overflow, FT[0, 0, 0], FT[0, 1, 2]); sync = true)

    # One shared device-style status flag covers all six panel launches.
    panel_ps = ntuple(_ -> fill(FT(90_000), 1, 1), 6)
    panel_dest = ntuple(_ -> zeros(FT, 1, 1), 6)
    panel_qv = ntuple(p -> p == 4 ? fill(FT(NaN), 1, 1, 1) : zeros(FT, 1, 1, 1), 6)
    @test_throws ArgumentError recover_total_surface_pressure_staged!(panel_dest, panel_ps, panel_qv, diffusion_coefficients(panel_qv, ak_sigma, bk_sigma); sync = true)

    pure_panel_dest = ntuple(_ -> fill(FT(NaN), 1, 1), 6)
    recover_total_surface_pressure_staged!(pure_panel_dest, panel_ps, ntuple(_ -> zeros(FT, 1, 1, 1), 6), diffusion_coefficients(ntuple(_ -> zeros(FT, 1, 1, 1), 6), ak_pressure, bk_pressure); sync = true)
    @test all(pure_panel_dest[p] == panel_ps[p] for p in 1:6)
end

@testset "cubed-sphere dz helpers preflight every panel and coefficient length" begin
    FT = Float64
    Nc, Nz = 2, 2
    A = FT[100, 50, 0]
    B = FT[0, 0.5, 1]
    ps = ntuple(_ -> fill(FT(95_000), Nc, Nc), 6)
    qv = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)

    recovered = ntuple(_ -> fill(FT(-1), Nc, Nc), 6)
    bad_ps = ntuple(p -> p == 6 ? zeros(FT, Nc + 1, Nc) : ps[p], 6)
    @test_throws DimensionMismatch recover_total_surface_pressure_staged!(recovered, bad_ps, qv, diffusion_coefficients(qv, A, B); sync = true)
    @test all(all(==(-one(FT)), recovered[p]) for p in 1:6)
    @test_throws DimensionMismatch recover_total_surface_pressure_staged!(recovered, ps, qv, diffusion_coefficients(qv, A[1:end-1], B); sync = true)

    dz = ntuple(_ -> fill(FT(-1), Nc, Nc, Nz), 6)
    @test_throws DimensionMismatch fill_dz_hydrostatic_constT_staged!(dz, bad_ps, diffusion_coefficients(dz, A, B); sync = true)
    @test all(all(==(-one(FT)), dz[p]) for p in 1:6)
    @test_throws DimensionMismatch fill_dz_hydrostatic_constT_staged!(dz, ps, diffusion_coefficients(dz, A, B[1:end-1]); sync = true)

    t = ntuple(_ -> fill(FT(250), Nc, Nc, Nz), 6)
    bad_t = ntuple(p -> p == 5 ? fill(FT(250), Nc, Nc, Nz + 1) : t[p], 6)
    @test_throws DimensionMismatch fill_dz_hydrostatic_virtualT_staged!(dz, bad_t, qv, ps, diffusion_coefficients(dz, A, B); sync = true)
    @test all(all(==(-one(FT)), dz[p]) for p in 1:6)
    @test_throws DimensionMismatch fill_dz_hydrostatic_virtualT_staged!(dz, t, qv, ps, diffusion_coefficients(dz, A[1:end-1], B); sync = true)
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
    coefficients = stage_hybrid_coefficients!(workspace.hybrid_coefficients,
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

@testset "fill_dz_hydrostatic_virtualT_staged! — single 3D array matches hand formula" begin
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
    fill_dz_hydrostatic_virtualT_staged!(dz, t_lyr, qv_lyr, ps, diffusion_coefficients(dz, ak, bk); sync = true)

    # Spot-check a few cells against the hand formula.
    for (i, j, k) in ((1, 1, 1), (2, 2, 3), (3, 3, Nz))
        delp  = (ak[k+1] - ak[k]) + (bk[k+1] - bk[k]) * ps[i, j]
        p_ctr = 0.5 * (ak[k] + ak[k+1] + (bk[k] + bk[k+1]) * ps[i, j])
        Tv    = t_lyr[i, j, k] * (1 + 0.61 * qv_lyr[i, j, k])
        expected = _R_DRY * Tv / _G_REF * delp / p_ctr
        @test isapprox(dz[i, j, k], expected; rtol = 1e-12)
    end
    dz_const = similar(dz)
    fill_dz_hydrostatic_constT_staged!(dz_const, ps, diffusion_coefficients(dz_const, ak, bk); sync = true)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp = (ak[k+1] - ak[k]) + (bk[k+1] - bk[k]) * ps[i,j]
        pc = ((ak[k] + ak[k+1]) + (bk[k] + bk[k+1]) * ps[i,j]) / 2
        @test dz_const[i,j,k] ≈ _R_DRY * 260 / _G_REF * dp / pc rtol=1e-14
    end

end

@testset "fill_dz_hydrostatic_virtualT_staged! — clamps negative qv to zero" begin
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
    fill_dz_hydrostatic_virtualT_staged!(dz_neg, t_lyr, qv_neg, ps, diffusion_coefficients(dz_neg, ak, bk); sync = true)
    fill_dz_hydrostatic_virtualT_staged!(dz_zero, t_lyr, qv_zero, ps, diffusion_coefficients(dz_zero, ak, bk); sync = true)
    @test dz_neg ≈ dz_zero atol = 1e-14
end

@testset "fill_dz_hydrostatic_virtualT_staged! — CS panel-tuple dispatch" begin
    FT = Float64
    Nc, Nz = 4, 5
    ak = collect(LinRange(FT(100.0), FT(0.0), Nz + 1))
    bk = collect(LinRange(FT(0.0),  FT(1.0),  Nz + 1))

    # Six different panels with different profiles to confirm per-panel data.
    dz_panels = ntuple(p -> zeros(FT, Nc, Nc, Nz), 6)
    t_panels  = ntuple(p -> fill(FT(220 + 5 * p), Nc, Nc, Nz), 6)
    qv_panels = ntuple(p -> fill(FT(0.002 * p), Nc, Nc, Nz), 6)
    ps_panels = ntuple(p -> fill(FT(95000 + 1000 * p), Nc, Nc), 6)

    fill_dz_hydrostatic_virtualT_staged!(dz_panels, t_panels, qv_panels, ps_panels, diffusion_coefficients(dz_panels, ak, bk); sync = true)

    # Reference: per-panel single-array call.
    for p in 1:6
        ref = zeros(FT, Nc, Nc, Nz)
        fill_dz_hydrostatic_virtualT_staged!(ref, t_panels[p], qv_panels[p], ps_panels[p], diffusion_coefficients(ref, ak, bk); sync = true)
        @test dz_panels[p] ≈ ref atol = 1e-14
    end
end
