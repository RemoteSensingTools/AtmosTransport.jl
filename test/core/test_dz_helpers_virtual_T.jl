#!/usr/bin/env julia
"""
Tests for the virtual-temperature `dz` helper (D6 fix). Three checks:

  1. Numerical correctness vs hand-computed dz at a few sample columns.
     For each `(i, j, k)`, dz should reduce to
         dz = R · T_v / g · delp / p_ctr
     with `T_v = T · (1 + 0.61·qv)`. We seed nonuniform T and qv and
     verify the kernel matches the hand formula.

  2. Cubed-sphere panel-tuple dispatch produces the same per-cell dz
     as the single-array variant on a synthetic per-panel column.

  3. dz_v != dz_c when virtual T differs from `T_ref = 260 K` — the
     whole point of D6.

NOTE: the runtime fallback `_fill_dz_for_diffusion!` in
`src/Models/driven_physics_refresh.jl` (which warns + reverts to constT when
the window lacks `vdiff`) is exercised indirectly by the regression
test suite when LocalHoltslagBovilleKzField runs are loaded. We don't
unit-test it here because it requires constructing a mock
`DrivenSimulation` with a synthetic window payload — out of scope for
the kernel-level helper tests.
"""

using Test
import AtmosTransport
using .AtmosTransport.Operators.Diffusion: fill_dz_hydrostatic_virtualT!,
                                            fill_dz_hydrostatic_constT!
using .AtmosTransport.Parameters: R_DRY_AIR, STANDARD_GRAVITY

const _R_DRY  = R_DRY_AIR
const _G_REF  = STANDARD_GRAVITY

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
