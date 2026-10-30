#!/usr/bin/env julia
# Unit + invariance tests for the TM5 boundary-layer diffusion column kernel
# (`tm5_bldiff_kvh_column!`). These check the physical behaviour of the
# Holtslag-Boville scheme on idealized columns rather than bit-matching TM5
# (a full reference comparison is the end-to-end validation step). The column
# builder uses ERA5-like fine near-surface spacing so shallow stable PBLs
# actually contain interfaces.

using Test

import AtmosTransport
using .AtmosTransport.Preprocessing: BLDiffConstants, tm5_bldiff_kvh_column!,
                                     tm5_bldiff_center_kz_column!, tm5_bldiff_dkg_column!, BLDiffColumnScratch

"""Build a consistent bottom-up column (index 1 = surface) with ERA5-like fine
near-surface resolution. Returns `(T, q, u, v, p_edge, z_edge)`."""
function _idealized_column(::Type{FT}; surface_p = 1.0e5) where {FT}
    z_edge = FT[0, 20, 50, 90, 140, 200, 280, 380, 500, 650, 830, 1050,
                1320, 1650, 2050, 2550, 3150, 3900, 4850, 6050, 7600, 9600,
                12200, 15600, 20000]
    Nz = length(z_edge) - 1
    c  = BLDiffConstants{FT}()
    zc = [(z_edge[l] + z_edge[l + 1]) / 2 for l in 1:Nz]
    T  = [max(FT(288) - FT(6e-3) * zc[l], FT(215)) for l in 1:Nz]
    q  = [max(FT(9e-3) * exp(-zc[l] / 2500), FT(1e-6)) for l in 1:Nz]
    u  = [FT(3) + FT(4e-3) * zc[l] for l in 1:Nz]
    v  = zeros(FT, Nz)
    p_edge = zeros(FT, Nz + 1); p_edge[1] = FT(surface_p)
    for l in 1:Nz
        p_edge[l + 1] = p_edge[l] *
            exp(-c.grav * (z_edge[l + 1] - z_edge[l]) / (c.r_air * T[l]))
    end
    return (; T, q, u, v, p_edge, z_edge, zc, Nz, c)
end

_run(col, hflux, lhflux, ustar) = begin
    kvh = zeros(eltype(col.T), col.Nz)
    pblh = tm5_bldiff_kvh_column!(kvh, col.T, col.q, col.u, col.v,
                                  col.p_edge, col.z_edge,
                                  eltype(col.T)(hflux), eltype(col.T)(lhflux),
                                  eltype(col.T)(ustar), col.c)
    (; kvh, pblh)
end

@testset "TM5 bldiff column kernel" begin

    @testset "output shape, finiteness, sign" begin
        col = _idealized_column(Float64)
        for (h, lh, us) in ((250.0, 150.0, 0.5), (-30.0, 10.0, 0.15), (1.0, 3.0, 0.3))
            r = _run(col, h, lh, us)
            @test length(r.kvh) == col.Nz
            @test r.kvh[end] == 0.0           # no flux through the model top
            @test all(isfinite, r.kvh)
            @test all(>=(0), r.kvh)           # diffusivity is non-negative
            @test r.pblh >= col.c.pblh_min    # PBL height is floored
        end
    end

    @testset "convective PBL is deeper than the stable PBL" begin
        col = _idealized_column(Float64)
        unstable = _run(col, 250.0, 150.0, 0.50)
        stable   = _run(col, -30.0,  10.0, 0.15)
        neutral  = _run(col,   1.0,   3.0, 0.30)
        @test unstable.pblh > neutral.pblh > stable.pblh
        @test maximum(unstable.kvh) > 10 * maximum(stable.kvh)
        @test maximum(unstable.kvh) > 20.0
        kmax = argmax(unstable.kvh)
        @test 100.0 < col.zc[kmax] < unstable.pblh
    end

    @testset "Kvh is non-negative across non-monotonic profiles (clamp guard)" begin
        # TM5's prescribed entrainment `0.2·wheatv/dθv` is unclamped and goes
        # large-negative when the PBL-top interface is locally unstable
        # (dθv/dz ≤ 0) — seen on ~0.0003% of real Dec-2021 N320 cells (down to
        # ~-700 m² s⁻¹). The kernel floors the override at kvh_min. This sweeps
        # superadiabatic / non-monotonic θ profiles as a non-negativity
        # regression guard; the entrainment clamp's specific real-data trigger
        # is validated by the Dec-2021 binary regen (validate_tm5_kz_payload).
        base = _idealized_column(Float64)
        c, Nz, zc = base.c, base.Nz, base.zc
        for s in 0:5
            T = [z < 2500 ? 295.0 - (12 + 2s) * 1e-3 * z : base.T[l]
                 for (l, z) in enumerate(zc)]
            T = max.(T, 215.0)
            for (h, lh, us) in ((300.0, 180.0, 0.55), (200.0, 120.0, 0.45))
                kvh = zeros(Nz)
                tm5_bldiff_kvh_column!(kvh, T, base.q, base.u, base.v,
                                       base.p_edge, base.z_edge, h, lh, us, c)
                @test all(isfinite, kvh)
                @test minimum(kvh) >= 0.0
            end
        end
    end



    @testset "Float32 path runs and agrees with Float64 to single precision" begin
        col64 = _idealized_column(Float64)
        col32 = _idealized_column(Float32)
        r64 = _run(col64, 250.0, 150.0, 0.50)
        r32 = _run(col32, 250.0, 150.0, 0.50)
        @test eltype(r32.kvh) == Float32
        @test isapprox(r32.pblh, r64.pblh; rtol = 1e-3)
        @test isapprox(Float64.(r32.kvh), r64.kvh; rtol = 1e-2, atol = 1e-2)
    end

    @testset "free troposphere uses the Louis branch, not the BL formula" begin
        col = _idealized_column(Float64)
        r = _run(col, -30.0, 10.0, 0.15)        # shallow stable PBL (~100 m)
        # Interfaces well above the PBL get the free-troposphere diffusivity,
        # which for this weakly-sheared, stably-stratified column is far below
        # the in-PBL floor (kvh_min = 0.1).
        above = findall(l -> col.z_edge[l + 1] > 3 * r.pblh && l < col.Nz, 1:col.Nz)
        @test !isempty(above)
        @test all(r.kvh[above] .< col.c.kvh_min)
    end

    @testset "entrainment flux is prescribed at the PBL-top interface" begin
        col = _idealized_column(Float64)
        c = col.c
        kvh = zeros(col.Nz)
        h, lh, us = 250.0, 150.0, 0.50
        pblh = tm5_bldiff_kvh_column!(kvh, col.T, col.q, col.u, col.v,
                                      col.p_edge, col.z_edge, h, lh, us, c)
        # Locate the interface whose bracketing layer centres straddle the PBL
        # top — the one the scheme overrides with 0.2·w_heatv / (dθv/dz).
        l = findfirst(l -> col.zc[l] < pblh < col.zc[l + 1], 1:col.Nz-1)
        @test l !== nothing
        # Reconstruct the expected entrainment value from the column state.
        vt = c.r_vap / c.r_air - 1
        θ(k)  = col.T[k] * (c.p_ref / ((col.p_edge[k] + col.p_edge[k+1]) / 2))^(c.r_air / c.cp_air)
        θv(k) = θ(k) * (1 + vt * col.q[k])
        ρ = (col.p_edge[1] - col.p_edge[2]) / (c.grav * (col.z_edge[2] - col.z_edge[1]))
        w_heat = h / (ρ * c.cp_air)
        w_qflx = lh / (ρ * c.l_vap)
        w_heatv = w_heat + vt * θ(1) * w_qflx
        dθv = (θv(l + 1) - θv(l)) / (col.zc[l + 1] - col.zc[l])
        @test kvh[l] ≈ 0.2 * w_heatv / dθv  rtol = 1e-6
    end

    @testset "top-down column driver maps to centres consistently" begin
        # Build a top-down (k=1 TOA → k=Nz surface) hybrid column and run the
        # driver. Cross-check against the bottom-up kernel on the same column.
        c = BLDiffConstants{Float64}()
        Nz = 30
        A = collect(range(0.0, 1.0e4; length = Nz + 1))          # TOA→surface
        B = collect(range(0.0, 1.0;    length = Nz + 1))
        ps = 1.0e5
        # Top-down profiles: surface (k=Nz) warm/moist, TOA (k=1) cold/dry.
        σ = [(A[k] + B[k] * ps) / ps for k in 1:Nz+1]            # edge sigma
        σc = [(σ[k] + σ[k+1]) / 2 for k in 1:Nz]
        T = [215.0 + 73.0 * σc[k] for k in 1:Nz]                  # warmer toward surface
        q = [max(1e-6, 9e-3 * σc[k]^3) for k in 1:Nz]
        u = [3.0 + 6.0 * (1 - σc[k]) for k in 1:Nz]; v = zeros(Nz)

        scratch = BLDiffColumnScratch{Float64}(Nz)
        kz = zeros(Nz)
        pblh = tm5_bldiff_center_kz_column!(kz, T, q, u, v, ps, 250.0, 150.0, 0.5,
                                            A, B, c, scratch)
        @test length(kz) == Nz
        @test all(isfinite, kz)
        @test all(>=(0), kz)
        @test pblh >= c.pblh_min
        # Centre Kz is largest in the lower troposphere (near the surface, high k)
        # and ~0 in the upper levels (low k).
        @test kz[Nz] >= 0                      # surface layer present
        @test maximum(kz[1:Nz÷3]) < maximum(kz[2Nz÷3:Nz])
        # The centre values bracket the bottom-up interface profile they average.
        @test maximum(kz) <= maximum(scratch.kvh) + 1e-9

        # Algebraic cross-check of the flip + interface→centre mapping: the
        # driver left the flipped bottom-up profiles in `scratch`, so call the
        # bottom-up kernel directly on them and reproduce the centre mapping by
        # hand. This catches any flip-direction, off-by-one, or storage-index
        # error that the plausibility checks above would miss.
        kvh_direct = zeros(Nz)
        tm5_bldiff_kvh_column!(kvh_direct, scratch.T, scratch.q, scratch.u,
                               scratch.v, scratch.p_edge, scratch.z_edge,
                               250.0, 150.0, 0.5, c)
        for l in 1:Nz
            below = l == 1 ? 0.0 : kvh_direct[l - 1]
            above = kvh_direct[l]
            @test kz[Nz + 1 - l] ≈ (below + above) / 2  atol = 1e-12
        end
    end

    @testset "vanishing surface forcing collapses toward weak diffusivity" begin
        col = _idealized_column(Float64)
        calm = _run(col, 0.0, 0.0, 0.05)
        # With no buoyancy flux and a near-calm surface, the in-PBL diffusivity
        # sits at the floor and the free troposphere is near zero.
        @test maximum(calm.kvh) < 1.0
        @test calm.pblh == col.c.pblh_min       # PBL collapses to its floor
    end
end

@testset "TM5 bldiff emits the vendored dkg formula" begin
    FT = Float64
    Nz = 6
    A = collect(range(0.0, 2000.0; length=Nz + 1))
    B = collect(range(0.0, 1.0; length=Nz + 1))
    ps = FT(1e5)
    T = collect(range(220.0, 288.0; length=Nz))
    q = collect(range(0.0, 0.01; length=Nz))
    u = collect(range(12.0, 3.0; length=Nz)); v = zeros(FT, Nz)
    m = FT[1, 2, 4, 7, 10, 12] .* 1e13
    scratch = BLDiffColumnScratch{FT}(Nz)
    dkg = zeros(FT, Nz)
    tm5_bldiff_dkg_column!(dkg, T, q, u, v, m, ps, 200.0, 100.0, 0.4,
                           A, B, BLDiffConstants{FT}(), scratch)
    for k in 1:Nz-1
        l = Nz - k
        expected = max(scratch.kvh[l], 0) * 2 * (m[k] + m[k+1]) /
                   (scratch.dz[k] + scratch.dz[k+1])^2
        @test dkg[k] ≈ expected rtol=2e-15
    end
    @test dkg[end] == 0
end
