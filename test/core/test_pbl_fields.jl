# Production boundary-layer fields and their scalar closure invariants.
using Test
import AtmosTransport
using Adapt
using AtmosTransport.Operators.Diffusion: PBLPhysicsParameters,
    LocalHoltslagBovilleKzField, refresh_local_holtslag_boville_kz_cache!,
    _beljaars_viterbo_kz, _obukhov_length, _prandtl_inverse

@testset "TimeVaryingField — _beljaars_viterbo_kz" begin
    p = PBLPhysicsParameters{Float64}()

    @testset "above 1.2 h_pbl returns Kz_bg exactly" begin
        h_pbl = 1000.0
        for z in (1200.0, 1500.0, 5000.0, 1e5)
            Kz = _beljaars_viterbo_kz(z, h_pbl, 0.3, -50.0, 1.0, p)
            @test Kz === p.Kz_bg
        end
    end

    @testset "stable BL spot value (z < h)" begin
        # L_ob > 0 (stable). Kz = u* κ z (1-z/h)² / (1 + 5z/L)
        h_pbl = 1000.0
        us    = 0.3
        z     = 100.0
        L_ob  = 50.0
        expected = us * p.kappa_vk * z * (1 - z/h_pbl)^2 / (1 + 5 * z / L_ob)
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, 1.0, p)
        @test Kz ≈ expected atol=1e-12
        @test p.Kz_min <= Kz <= p.Kz_max  # inside clamps
    end

    @testset "unstable surface layer spot value (z < 0.1 h)" begin
        # L_ob < 0. z < 0.1 h → surface-layer branch.
        # Kz = u* κ z (1-z/h)² (1 - β_h z/L)^(1/3) × Pr_inv
        h_pbl, us, z, L_ob, Pr_inv = 1000.0, 0.3, 50.0, -40.0, 2.0
        expected = us * p.kappa_vk * z * (1 - z/h_pbl)^2 *
                   cbrt(1 - p.β_h * z / L_ob) * Pr_inv
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, Pr_inv, p)
        @test Kz ≈ expected atol=1e-12
    end

    @testset "unstable mixed layer spot value (0.1 h < z < h)" begin
        # L_ob < 0. z ≥ 0.1 h → mixed-layer branch.
        # w_m = u* (1 - 0.1 β_h h/L)^(1/3); Kz = w_m κ z (1-z/h)² × Pr_inv
        h_pbl, us, z, L_ob, Pr_inv = 1000.0, 0.3, 500.0, -40.0, 2.0
        w_m = us * cbrt(1 - 0.1 * p.β_h * h_pbl / L_ob)
        expected = w_m * p.kappa_vk * z * (1 - z/h_pbl)^2 * Pr_inv
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, Pr_inv, p)
        @test Kz ≈ expected atol=1e-12
    end

    @testset "clamping to Kz_max" begin
        # Very small L_ob (strongly unstable), large Pr_inv → would blow past 500
        h_pbl, us, z, L_ob, Pr_inv = 1000.0, 2.0, 500.0, -5.0, 10.0
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, Pr_inv, p)
        @test Kz === p.Kz_max
    end

    @testset "clamping to Kz_min" begin
        # Very small ustar → Kz → 0; gets clamped up to Kz_min
        h_pbl, us, z, L_ob, Pr_inv = 1000.0, 1e-6, 10.0, 50.0, 1.0
        Kz = _beljaars_viterbo_kz(z, h_pbl, us, L_ob, Pr_inv, p)
        @test Kz === p.Kz_min
    end

    @testset "taper zone (h ≤ z < 1.2 h): linear blend to Kz_bg" begin
        # At z = h_pbl exactly, taper has frac = (1.2h - h) / (0.2h) = 1 and
        # z_eff = min(z, h-1) = h-1, so (1-z_eff/h)² ≈ 1/h² — the raw Kz is
        # tiny. After clamp + taper-blend at frac=1 the returned value equals
        # the clamped in-PBL value. At z = 1.1h, frac = 0.5 → 50/50 blend.
        # At z = 1.2h, frac = 0 → Kz_bg.
        h_pbl, us, L_ob, Pr_inv = 1000.0, 0.3, 50.0, 1.0
        Kz_at_h   = _beljaars_viterbo_kz(h_pbl,     h_pbl, us, L_ob, Pr_inv, p)
        Kz_at_11h = _beljaars_viterbo_kz(1.1*h_pbl, h_pbl, us, L_ob, Pr_inv, p)
        Kz_at_12h = _beljaars_viterbo_kz(1.2*h_pbl, h_pbl, us, L_ob, Pr_inv, p)

        @test Kz_at_12h === p.Kz_bg
        # Blend formula: Kz = Kz_bg + frac × (Kz_in_pbl_clamped - Kz_bg)
        expected_11h = p.Kz_bg + 0.5 * (Kz_at_h - p.Kz_bg)
        @test Kz_at_11h ≈ expected_11h atol=1e-12
        # Direction of the blend depends on whether the clamped in-PBL value
        # exceeds Kz_bg (common in strong mixing) or sits below (as here,
        # where (1-zh)² ≈ 1/h² drives the raw Kz below Kz_min so it clamps
        # to Kz_min = 0.01 < Kz_bg = 0.1 → Kz increases through the taper).
        @test Kz_at_h <= Kz_at_11h <= Kz_at_12h
    end


end

@testset "TimeVaryingField — _obukhov_length" begin
    p = PBLPhysicsParameters{Float64}()

    @testset "positive hflux → unstable (L < 0)" begin
        L_ob, H_kin = _obukhov_length(100.0, 0.3, 295.0, p)
        @test H_kin > 0
        @test L_ob < 0
    end

    @testset "negative hflux → stable (L > 0)" begin
        L_ob, H_kin = _obukhov_length(-50.0, 0.3, 280.0, p)
        @test H_kin < 0
        @test L_ob > 0
    end

    @testset "zero hflux → finite large L via safety offset" begin
        L_ob, H_kin = _obukhov_length(0.0, 0.3, 285.0, p)
        @test H_kin === 0.0
        @test isfinite(L_ob)
        @test abs(L_ob) > 1e9   # very large, no singularity
    end

    @testset "hand-verified spot value" begin
        # hflux=100, u*=0.3, t2m=295
        # H_kin = 100 / (1.225 * 1004.64) = 0.0813 K m/s
        # L_ob = -295 × 0.027 / (0.41 × 9.80665 × 0.0813 (+ offset))
        hflux, us, t2m = 100.0, 0.3, 295.0
        H_kin_ref = hflux / (p.rho_ref * p.cp_dry)
        H_safe_ref = H_kin_ref + sign(H_kin_ref + 1e-20) * 1e-10
        L_ob_ref = -t2m * us^3 / (p.kappa_vk * p.gravity * H_safe_ref)
        L_ob, H_kin = _obukhov_length(hflux, us, t2m, p)
        @test H_kin ≈ H_kin_ref atol=1e-12
        @test L_ob  ≈ L_ob_ref  atol=1e-6
    end
end

@testset "TimeVaryingField — _prandtl_inverse" begin
    p = PBLPhysicsParameters{Float64}()

    @testset "stable branch returns 1" begin
        # L_ob > 0, H_kin can be anything
        @test _prandtl_inverse(1000.0, 0.3, 0.1, 285.0, 50.0, p) === 1.0
        @test _prandtl_inverse(1000.0, 0.3, -0.1, 285.0, 50.0, p) === 1.0
    end

    @testset "non-positive H_kin returns 1" begin
        # Even with L_ob < 0 (unstable), if H_kin ≤ 0 formula doesn't apply
        @test _prandtl_inverse(1000.0, 0.3, 0.0, 285.0, -50.0, p) === 1.0
        @test _prandtl_inverse(1000.0, 0.3, -0.05, 285.0, -50.0, p) === 1.0
    end

    @testset "tiny h_pbl returns 1" begin
        @test _prandtl_inverse(5.0, 0.3, 0.1, 285.0, -50.0, p) === 1.0
    end

    @testset "unstable convective → > 1" begin
        # Classic daytime setup
        Pr_inv = _prandtl_inverse(1000.0, 0.3, 0.08, 295.0, -25.0, p)
        @test Pr_inv > 1.0
        @test isfinite(Pr_inv)
    end
end

@testset "TimeVaryingField — production WindowPBLKzField" begin
    for FT in (Float32, Float64), hflux in (FT(-100), zero(FT), FT(100))
        Nx, Ny, Nz, Hp = 2, 2, 10, 1
        cache = ntuple(_ -> fill(FT(-1), Nx, Ny, Nz), 6)
        f = AtmosTransport.WindowPBLKzField(cache)
        surface = (pblh = ntuple(_ -> fill(FT(3000), Nx, Ny), 6),
                   ustar = ntuple(_ -> fill(FT(0.3), Nx, Ny), 6),
                   hflux = ntuple(_ -> fill(hflux, Nx, Ny), 6),
                   t2m = ntuple(_ -> fill(FT(295), Nx, Ny), 6))
        areas = fill(FT(1e8), Nx, Ny)
        mass = ntuple(_ -> fill(FT(10132.5) * FT(1e8) / f.params.gravity,
                                Nx + 2Hp, Ny + 2Hp, Nz), 6)
        AtmosTransport.refresh_pbl_kz_cache!(f, surface, mass, areas; halo_width = Hp)
        for p in 1:6
            data = f.panels[p].data
            @test all(isfinite, data)
            @test all(f.params.Kz_min .<= data .<= f.params.Kz_max)
            @test data[1, 1, :] == data[2, 2, :]
            @test data[1, 1, 1] === f.params.Kz_bg
            @test data[1, 1, 2] === f.params.Kz_bg
            # Independent two-pass hydrostatic height + scalar closure reference.
            dp = mass[p][Hp+1, Hp+1, 1] * f.params.gravity / areas[1, 1]
            RTg = f.params.cp_dry / FT(3.5) * FT(295) / f.params.gravity
            dz = [dp * RTg / max((FT(k-1)*dp + FT(k)*dp)/FT(2), FT(1)) for k in 1:Nz]
            L, H = _obukhov_length(hflux, FT(0.3), FT(295), f.params)
            Pr = _prandtl_inverse(FT(3000), FT(0.3), H, FT(295), L, f.params)
            heights = [sum(dz[k+1:end]) + dz[k]/FT(2) for k in 1:Nz]
            expected = [_beljaars_viterbo_kz(z, FT(3000), FT(0.3), L, Pr, f.params) for z in heights]
            @test data[1, 1, :] ≈ expected rtol = 32eps(FT)
        end
        adapted = Adapt.adapt(Array, f)
        AtmosTransport.refresh_pbl_kz_cache!(adapted, surface, mass, areas; halo_width = Hp)
        @test all(p -> adapted.panels[p].data == f.panels[p].data, 1:6)
    end
end


@testset "TimeVaryingField — GCHP Holtslag-Boville VDIFF Kz" begin
    FT = Float64
    Nx, Ny, Nz = 2, 2, 4
    params = PBLPhysicsParameters{FT}()
    areas = fill(FT(1.0e10), Nx, Ny)
    delp = fill(FT(25000), Nz)
    air_col = delp .* areas[1, 1] ./ params.gravity
    air_mass = ntuple(_ -> reshape(repeat(air_col, inner = Nx * Ny), Nx, Ny, Nz), 6)
    cache = ntuple(_ -> fill(FT(-1), Nx, Ny, Nz), 6)
    field = LocalHoltslagBovilleKzField(cache; params)

    pblh = FT(1800)
    ustar = FT(0.35)
    hflux = FT(80)
    t2m = FT(295)
    surface = (
        pblh  = ntuple(_ -> fill(pblh, Nx, Ny), 6),
        ustar = ntuple(_ -> fill(ustar, Nx, Ny), 6),
        hflux = ntuple(_ -> fill(hflux, Nx, Ny), 6),
        t2m   = ntuple(_ -> fill(t2m, Nx, Ny), 6),
    )
    t_profile = FT[245, 255, 270, 285]
    q_profile = FT[1e-5, 5e-4, 2e-3, 6e-3]
    u_profile = fill(FT(12), Nz)
    v_profile = fill(FT(-4), Nz)
    vdiff = (
        u  = ntuple(_ -> reshape(repeat(u_profile, inner = Nx * Ny), Nx, Ny, Nz), 6),
        v  = ntuple(_ -> reshape(repeat(v_profile, inner = Nx * Ny), Nx, Ny, Nz), 6),
        t  = ntuple(_ -> reshape(repeat(t_profile, inner = Nx * Ny), Nx, Ny, Nz), 6),
        qv = ntuple(_ -> reshape(repeat(q_profile, inner = Nx * Ny), Nx, Ny, Nz), 6),
    )

    refresh_local_holtslag_boville_kz_cache!(field, surface, vdiff, air_mass,
                                            areas; halo_width = 0)

    R_dry = params.cp_dry / FT(3.5)
    tv = max.(t_profile, FT(180)) .* (one(FT) .+ FT(0.61) .* max.(q_profile, zero(FT)))
    L_ob, H_kin = _obukhov_length(hflux, ustar, t2m, params)
    Pr_inv = _prandtl_inverse(pblh, ustar, H_kin, t2m, L_ob, params)
    z_col = zero(FT)
    p_top = zero(FT)
    for k in 1:Nz
        p_bot = p_top + delp[k]
        p_mid = max((p_top + p_bot) / FT(2), FT(1))
        z_col += delp[k] * R_dry * tv[k] / (params.gravity * p_mid)
        p_top = p_bot
    end
    expected = Vector{FT}(undef, Nz)
    z_above = z_col
    for k in 1:Nz
        dz = delp[k] * R_dry * tv[k] /
             (params.gravity * max((sum(delp[1:k-1]) + sum(delp[1:k])) / FT(2), FT(1)))
        z_center = z_above - dz / FT(2)
        expected[k] = _beljaars_viterbo_kz(z_center, pblh, ustar, L_ob,
                                           Pr_inv, params)
        z_above -= dz
    end

    @test all(isfinite, field.host_cache[1])
    @test all(field.host_cache[1] .>= params.Kz_bg)
    for panel in 1:6
        @test field.host_cache[panel][1, 1, :] ≈ expected rtol=1e-12 atol=1e-12
        @test field.host_cache[panel][2, 2, :] ≈ expected rtol=1e-12 atol=1e-12
    end
end

# ==========================================================================
# StepwiseField (plan 17 Commit 1)
# ==========================================================================
#
# Validates the piecewise-constant-in-time concrete type of
# `AbstractTimeVaryingField`. Covers:
#   - Construction: rank bounds, boundary monotonicity, sample/boundary
#                   length consistency
#   - field_value: reads the current window's sample at (idx...)
#   - update_field!: binary-search correctness, idempotence, out-of-range
#                    errors, half-open convention
#   - Sub-step additivity: integral over [t1, t2] + [t2, t3] == [t1, t3]
#   - Type stability of field_value
#   - Kernel-safety on CPU backend
#   - Adapt / GPU adaptation (samples → CuArray, current_window → CuArray)
