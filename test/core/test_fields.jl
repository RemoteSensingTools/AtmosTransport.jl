"""
Unit tests for the `TimeVaryingField` abstraction.

Validates the minimum concrete type `ConstantField{FT, N}` against the
field interface:

1. `field_value(f, idx)` returns the stored scalar regardless of index
2. `update_field!(f, t)` is a no-op (returns `f`, no state change)
3. Type stability at every rank used downstream (N ∈ {0, 2, 3})
4. Kernel-safety on CPU (callable inside a KA kernel)

The concrete diffusion field types (`ProfileKzField`, etc.) extend this suite.
"""

using Test
using AtmosTransport: AbstractTimeVaryingField, ConstantField, ProfileKzField,
                      PreComputedKzField, DerivedKzField, PBLPhysicsParameters,
                      StepwiseField, LocalHoltslagBovilleKzField,
                      field_value, update_field!, integral_between,
                      refresh_local_holtslag_boville_kz_cache!
using AtmosTransport.State.Fields: _beljaars_viterbo_kz, _obukhov_length,
                                    _prandtl_inverse
using KernelAbstractions: @kernel, @index, get_backend, synchronize
using Adapt

# GPU support — optional; only run `@testset "... GPU"` blocks if CUDA is up.
const HAS_GPU_FIELDS = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

@testset "TimeVaryingField — ConstantField" begin

    @testset "scalar (N=0) — Float64" begin
        f = ConstantField{Float64, 0}(2.098e-6)
        @test f isa AbstractTimeVaryingField{Float64, 0}
        @test field_value(f, ()) === 2.098e-6
        @test update_field!(f, 0.0) === f
        @test update_field!(f, 1.0e9) === f
        # idempotent updates leave field_value unchanged
        update_field!(f, 1.0)
        @test field_value(f, ()) === 2.098e-6
    end

    @testset "scalar (N=0) — Float32" begin
        f = ConstantField{Float32, 0}(3.0f0)
        @test field_value(f, ()) === 3.0f0
        @test update_field!(f, 0.0) === f
    end

    @testset "Real → FT coercion in constructor" begin
        # Integer / other-float input coerced to FT
        f = ConstantField{Float64, 0}(1)
        @test field_value(f, ()) === 1.0
        g = ConstantField{Float32, 0}(2.5)   # Float64 → Float32
        @test field_value(g, ()) === 2.5f0
    end

    @testset "surface (N=2)" begin
        f = ConstantField{Float64, 2}(42.0)
        @test f isa AbstractTimeVaryingField{Float64, 2}
        @test field_value(f, (1, 1)) === 42.0
        @test field_value(f, (1000, 500)) === 42.0
    end

    @testset "volume (N=3) — anticipating Kz in plan 16b" begin
        f = ConstantField{Float64, 3}(1.5)
        @test f isa AbstractTimeVaryingField{Float64, 3}
        @test field_value(f, (1, 1, 1)) === 1.5
        @test field_value(f, (144, 72, 32)) === 1.5
    end

    @testset "rank-mismatched index does not dispatch" begin
        f = ConstantField{Float64, 3}(1.0)
        # Cannot call field_value with a 2-tuple on a rank-3 field
        @test_throws MethodError field_value(f, (1, 1))
        # Cannot call with empty tuple on rank-3
        @test_throws MethodError field_value(f, ())
    end

    @testset "type stability" begin
        f = ConstantField{Float64, 0}(1.0)
        @test @inferred(field_value(f, ())) === 1.0

        g = ConstantField{Float32, 3}(0.5f0)
        @test @inferred(field_value(g, (1, 2, 3))) === 0.5f0
    end

    @testset "kernel-safety — CPU backend" begin
        # field_value must be callable from inside a KA kernel without
        # allocation or dynamic dispatch. Launch a trivial kernel that
        # writes the field's value into every cell of an output array.
        f = ConstantField{Float64, 0}(7.25)
        out = zeros(Float64, 4, 3, 2)
        backend = get_backend(out)

        @kernel function _fill_from_scalar_field!(out, field)
            i, j, k = @index(Global, NTuple)
            @inbounds out[i, j, k] = field_value(field, ())
        end

        _fill_from_scalar_field!(backend, 64)(out, f; ndrange = size(out))
        synchronize(backend)
        @test all(out .== 7.25)

        # Rank-3 field
        g = ConstantField{Float64, 3}(-1.0)
        fill!(out, 0.0)
        @kernel function _fill_from_volume_field!(out, field)
            i, j, k = @index(Global, NTuple)
            @inbounds out[i, j, k] = field_value(field, (i, j, k))
        end
        _fill_from_volume_field!(backend, 64)(out, g; ndrange = size(out))
        synchronize(backend)
        @test all(out .== -1.0)
    end
end

@testset "TimeVaryingField — ProfileKzField" begin

    @testset "construction + type bounds" begin
        profile = [0.1, 0.5, 1.0, 2.0, 5.0]
        f = ProfileKzField(profile)
        @test f isa AbstractTimeVaryingField{Float64, 3}
        @test f isa ProfileKzField{Float64}

        f_rg = ProfileKzField(profile; spatial_rank = 2)
        @test f_rg isa AbstractTimeVaryingField{Float64, 2}

        g = ProfileKzField(Float32[0.0, 1.0, 2.0])
        @test g isa AbstractTimeVaryingField{Float32, 3}
    end

    @testset "rank-2 field_value selects the k coordinate" begin
        profile = [10.0, 20.0, 30.0, 40.0]
        f = ProfileKzField(profile; spatial_rank = 2)
        @test field_value(f, (1, 1)) === 10.0
        @test field_value(f, (7, 2)) === 20.0
        @test field_value(f, (99, 4)) === 40.0
    end

    @testset "field_value selects the k coordinate" begin
        profile = [10.0, 20.0, 30.0, 40.0, 50.0]
        f = ProfileKzField(profile)
        @test field_value(f, (1, 1, 1)) === 10.0
        @test field_value(f, (1, 1, 2)) === 20.0
        @test field_value(f, (1, 1, 3)) === 30.0
        @test field_value(f, (1, 1, 5)) === 50.0
    end

    @testset "field_value ignores i, j" begin
        profile = [1.0, 2.0, 3.0]
        f = ProfileKzField(profile)
        # Same k, different (i, j) → same value
        @test field_value(f, (1,   1,  2)) === 2.0
        @test field_value(f, (100, 50, 2)) === 2.0
        @test field_value(f, (1,   72, 2)) === 2.0
        @test field_value(f, (144, 1,  2)) === 2.0
    end

    @testset "update_field! is a no-op" begin
        profile = [1.0, 2.0, 3.0]
        f = ProfileKzField(profile)
        @test update_field!(f, 0.0) === f
        @test update_field!(f, 1.0e9) === f
        # Profile unchanged
        @test f.profile == [1.0, 2.0, 3.0]
        # Subsequent reads still return the originals
        @test field_value(f, (1, 1, 1)) === 1.0
        @test field_value(f, (1, 1, 3)) === 3.0
    end

    @testset "type stability" begin
        f = ProfileKzField([1.0, 2.0, 3.0])
        @test @inferred(field_value(f, (1, 1, 1))) === 1.0

        g = ProfileKzField(Float32[0.5f0, 1.5f0])
        @test @inferred(field_value(g, (1, 1, 2))) === 1.5f0
    end

    @testset "rank-mismatched index does not dispatch" begin
        f = ProfileKzField([1.0, 2.0, 3.0])
        @test_throws MethodError field_value(f, (1, 1))
        @test_throws MethodError field_value(f, ())
    end

    @testset "kernel-safety — CPU backend" begin
        # Verify field_value is callable from inside a KA kernel.
        # Write a k-varying profile into every column of an output array.
        profile = [10.0, 20.0, 30.0, 40.0]
        f = ProfileKzField(profile)
        out = zeros(Float64, 3, 2, 4)
        backend = get_backend(out)

        @kernel function _fill_from_profile_field!(out, field)
            i, j, k = @index(Global, NTuple)
            @inbounds out[i, j, k] = field_value(field, (i, j, k))
        end

        _fill_from_profile_field!(backend, 64)(out, f; ndrange = size(out))
        synchronize(backend)

        # Every column should match the profile
        for i in 1:3, j in 1:2
            @test out[i, j, :] == profile
        end
    end

    if HAS_GPU_FIELDS
        @testset "Adapt to GPU backing" begin
            # The struct is parametric on `V <: AbstractVector{FT}` so the
            # same type handles both `Vector` (CPU) and `CuArray` (GPU)
            # backings. `Adapt.adapt_structure` swaps the profile storage
            # at kernel-launch time without losing FT.
            profile_cpu = Float32[0.5f0, 1.0f0, 1.5f0, 2.0f0]
            f_cpu = ProfileKzField(profile_cpu)
            f_gpu = Adapt.adapt(CUDA.CuArray, f_cpu)

            @test f_gpu isa ProfileKzField{Float32}
            @test f_gpu.profile isa CUDA.CuArray{Float32, 1}

            # Confirm field_value works through the kernel on GPU
            out = CUDA.zeros(Float32, 3, 2, 4)
            backend = get_backend(out)

            @kernel function _fill_from_profile_gpu!(out, field)
                i, j, k = @index(Global, NTuple)
                @inbounds out[i, j, k] = field_value(field, (i, j, k))
            end

            _fill_from_profile_gpu!(backend, 64)(out, f_gpu; ndrange = size(out))
            synchronize(backend)

            out_cpu = Array(out)
            for i in 1:3, j in 1:2
                @test out_cpu[i, j, :] == profile_cpu
            end
        end
    end
end

@testset "TimeVaryingField — PreComputedKzField" begin

    @testset "construction + type bounds" begin
        data = rand(Float64, 4, 3, 5)
        f = PreComputedKzField(data)
        @test f isa AbstractTimeVaryingField{Float64, 3}
        @test f isa PreComputedKzField{Float64}

        data_rg = rand(Float64, 8, 5)
        f_rg = PreComputedKzField(data_rg)
        @test f_rg isa AbstractTimeVaryingField{Float64, 2}

        data32 = rand(Float32, 2, 2, 3)
        g = PreComputedKzField(data32)
        @test g isa AbstractTimeVaryingField{Float32, 3}
    end

    @testset "rank-2 field_value respects (cell, k)" begin
        data = Array{Float64, 2}(undef, 4, 5)
        for c in 1:4, k in 1:5
            data[c, k] = 100 * c + k
        end
        f = PreComputedKzField(data)

        @test field_value(f, (1, 1)) === 101.0
        @test field_value(f, (4, 5)) === 405.0
        @test field_value(f, (2, 3)) === 203.0
    end

    @testset "field_value respects (i, j, k) independently" begin
        # Construct data where each cell's value is a unique fingerprint
        data = Array{Float64, 3}(undef, 4, 3, 5)
        for i in 1:4, j in 1:3, k in 1:5
            data[i, j, k] = 100 * i + 10 * j + k
        end
        f = PreComputedKzField(data)

        @test field_value(f, (1, 1, 1)) === 111.0
        @test field_value(f, (4, 3, 5)) === 435.0
        @test field_value(f, (2, 1, 3)) === 213.0
        # Varying only i or only j must change the value (vs. ProfileKzField)
        @test field_value(f, (1, 1, 2)) !== field_value(f, (2, 1, 2))
        @test field_value(f, (1, 1, 2)) !== field_value(f, (1, 2, 2))
    end

    @testset "update_field! is a no-op" begin
        data = [1.0; 2.0; 3.0 ;; 4.0; 5.0; 6.0 ;;; 7.0; 8.0; 9.0 ;; 10.0; 11.0; 12.0]
        # Built as 3×2×2. Content doesn't matter; we test identity + preservation.
        f = PreComputedKzField(data)
        pre = copy(f.data)
        @test update_field!(f, 0.0) === f
        @test update_field!(f, 1.0e9) === f
        @test f.data == pre
    end

    @testset "caller-owned storage: external mutation visible" begin
        # The field does not own the array — mutating the backing array
        # (e.g. when met-window advances) is immediately visible through
        # field_value.
        data = zeros(Float64, 2, 2, 2)
        f = PreComputedKzField(data)
        @test field_value(f, (1, 1, 1)) === 0.0

        data[1, 1, 1] = 42.0
        @test field_value(f, (1, 1, 1)) === 42.0

        fill!(data, -1.0)
        @test field_value(f, (2, 2, 2)) === -1.0
    end

    @testset "type stability" begin
        data = rand(Float64, 3, 3, 3)
        f = PreComputedKzField(data)
        @test @inferred(field_value(f, (1, 1, 1))) isa Float64

        data32 = rand(Float32, 2, 2, 2)
        g = PreComputedKzField(data32)
        @test @inferred(field_value(g, (1, 1, 1))) isa Float32
    end

    @testset "rank-mismatched construction rejected" begin
        # Rank-2 and rank-3 are both valid for topology-specific Kz
        # fields; other ranks are rejected.
        @test PreComputedKzField(rand(Float64, 3, 3)) isa AbstractTimeVaryingField{Float64, 2}
        @test_throws MethodError PreComputedKzField(rand(Float64, 7))
        @test_throws MethodError PreComputedKzField(rand(Float64, 2, 2, 2, 2))
    end

    @testset "kernel-safety — CPU backend" begin
        # Launch a KA kernel that reads every element of the backing array
        # through field_value and copies it into an output buffer. If the
        # field path matches direct array indexing, the output equals the
        # input elementwise.
        Nx, Ny, Nz = 3, 4, 5
        src = reshape(collect(1.0:Nx*Ny*Nz), Nx, Ny, Nz)
        f = PreComputedKzField(src)
        out = zeros(Float64, Nx, Ny, Nz)
        backend = get_backend(out)

        @kernel function _copy_from_volume_field!(out, field)
            i, j, k = @index(Global, NTuple)
            @inbounds out[i, j, k] = field_value(field, (i, j, k))
        end

        _copy_from_volume_field!(backend, 64)(out, f; ndrange = size(out))
        synchronize(backend)
        @test out == src
    end
end

@testset "TimeVaryingField — PBLPhysicsParameters defaults" begin
    p = PBLPhysicsParameters{Float64}()
    @test p.β_h      === 15.0
    @test p.Kz_bg    === 0.1
    @test p.Kz_min   === 0.01
    @test p.Kz_max   === 500.0
    @test p.kappa_vk === 0.41
    @test p.gravity  === 9.80665
    @test p.cp_dry   === 1004.64
    @test p.rho_ref  === 1.225

    # FT propagation
    p32 = PBLPhysicsParameters{Float32}()
    @test p32.β_h isa Float32
    @test p32.Kz_bg === 0.1f0

    # kwarg overrides
    p_custom = PBLPhysicsParameters{Float64}(β_h = 20.0, Kz_min = 0.001)
    @test p_custom.β_h === 20.0
    @test p_custom.Kz_min === 0.001
    @test p_custom.Kz_bg === 0.1   # defaults preserved
end

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

    @testset "type stability" begin
        p64 = PBLPhysicsParameters{Float64}()
        @test @inferred(_beljaars_viterbo_kz(100.0, 1000.0, 0.3, -40.0, 1.5, p64)) isa Float64
        p32 = PBLPhysicsParameters{Float32}()
        @test @inferred(_beljaars_viterbo_kz(100f0, 1000f0, 0.3f0, -40f0, 1.5f0, p32)) isa Float32
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

# Helpers for DerivedKzField tests — module scope so `where {FT}` binds correctly
_make_delp_field(Nx, Ny, delp_col::Vector{FT}) where FT = begin
    data = Array{FT}(undef, Nx, Ny, length(delp_col))
    for i in 1:Nx, j in 1:Ny, k in eachindex(delp_col)
        data[i, j, k] = delp_col[k]
    end
    PreComputedKzField(data)
end

_constant_surface(::Type{FT}; pblh, ustar, hflux, t2m) where FT = (
    pblh  = ConstantField{FT, 2}(pblh),
    ustar = ConstantField{FT, 2}(ustar),
    hflux = ConstantField{FT, 2}(hflux),
    t2m   = ConstantField{FT, 2}(t2m),
)

@testset "TimeVaryingField — DerivedKzField" begin

    @testset "construction + type bounds" begin
        FT = Float64
        Nx, Ny, Nz = 2, 3, 5
        surface = _constant_surface(FT; pblh=1000.0, ustar=0.3, hflux=100.0, t2m=295.0)
        delp    = _make_delp_field(Nx, Ny, fill(20000.0, Nz))
        cache   = zeros(FT, Nx, Ny, Nz)
        f       = DerivedKzField(; surface, delp, cache)

        @test f isa AbstractTimeVaryingField{Float64, 3}
        @test f isa DerivedKzField{Float64}
        @test f.params isa PBLPhysicsParameters{Float64}
    end

    @testset "constructor rejects surface fields of wrong rank" begin
        FT = Float64
        bad_surface = (
            pblh  = ConstantField{FT, 3}(1000.0),   # should be rank-2
            ustar = ConstantField{FT, 2}(0.3),
            hflux = ConstantField{FT, 2}(100.0),
            t2m   = ConstantField{FT, 2}(295.0),
        )
        delp  = _make_delp_field(2, 2, fill(20000.0, 3))
        cache = zeros(FT, 2, 2, 3)
        @test_throws ArgumentError DerivedKzField(;
            surface=bad_surface, delp=delp, cache=cache)
    end

    @testset "update_field! populates cache" begin
        FT = Float64
        Nx, Ny, Nz = 2, 2, 10
        surface = _constant_surface(FT; pblh=3000.0, ustar=0.3,
                                    hflux=100.0, t2m=295.0)
        # 10 layers, roughly 10 kPa each, surface = 101325 Pa
        delp    = _make_delp_field(Nx, Ny, fill(10132.5, Nz))
        cache   = fill(-1.0, Nx, Ny, Nz)   # sentinel so we see writes
        f       = DerivedKzField(; surface, delp, cache)

        # Before update — sentinel present
        @test all(f.cache .== -1.0)

        update_field!(f, 0.0)

        # After update — all cells written, no negatives
        @test all(f.cache .>= 0.0)
        @test all(isfinite, f.cache)
        # Horizontal uniformity (all (i,j) columns have same surface fields and delp)
        @test f.cache[1, 1, :] == f.cache[2, 2, :]
        @test f.cache[1, 2, :] == f.cache[2, 1, :]
        # All within clamps [Kz_min, Kz_max]
        @test all(f.cache .>= f.params.Kz_min - 1e-12)
        @test all(f.cache .<= f.params.Kz_max + 1e-12)
    end

    @testset "high levels (above 1.2 h_pbl) → Kz_bg" begin
        # With h_pbl = 500 m and tall layers, the top cells sit above 1.2h
        FT = Float64
        Nx, Ny, Nz = 2, 2, 10
        surface = _constant_surface(FT; pblh=500.0, ustar=0.3,
                                    hflux=100.0, t2m=295.0)
        delp    = _make_delp_field(Nx, Ny, fill(10132.5, Nz))
        cache   = zeros(FT, Nx, Ny, Nz)
        f       = DerivedKzField(; surface, delp, cache)
        update_field!(f, 0.0)

        # k=1 is TOA; top few levels are well above 1.2 × 500 = 600 m
        @test f.cache[1, 1, 1] === f.params.Kz_bg
        @test f.cache[1, 1, 2] === f.params.Kz_bg
    end

    @testset "field_value reads from cache" begin
        FT = Float64
        Nx, Ny, Nz = 2, 2, 5
        surface = _constant_surface(FT; pblh=1500.0, ustar=0.3,
                                    hflux=100.0, t2m=295.0)
        delp    = _make_delp_field(Nx, Ny, fill(20265.0, Nz))
        cache   = zeros(FT, Nx, Ny, Nz)
        f       = DerivedKzField(; surface, delp, cache)
        update_field!(f, 0.0)

        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            @test field_value(f, (i, j, k)) === f.cache[i, j, k]
        end
    end

    @testset "changing surface fields changes cache on next update" begin
        # Mutable surface field: wrap hflux in a PreComputedKzField
        # so we can flip it, re-update, and see a different cache.
        FT = Float64
        Nx, Ny, Nz = 2, 2, 5

        hflux_arr = fill(100.0, Nx, Ny)
        # Use a trivial 2D "constant-but-mutable" wrapper: simplest is to
        # just re-construct the field with a new ConstantField value and
        # re-call update_field!.
        surface_pos = _constant_surface(FT; pblh=1500.0, ustar=0.3,
                                        hflux=100.0, t2m=295.0)
        surface_neg = _constant_surface(FT; pblh=1500.0, ustar=0.3,
                                        hflux=-50.0, t2m=295.0)
        delp = _make_delp_field(Nx, Ny, fill(20265.0, Nz))

        cache1 = zeros(FT, Nx, Ny, Nz)
        f1 = DerivedKzField(; surface=surface_pos, delp, cache=cache1)
        update_field!(f1, 0.0)

        cache2 = zeros(FT, Nx, Ny, Nz)
        f2 = DerivedKzField(; surface=surface_neg, delp, cache=cache2)
        update_field!(f2, 0.0)

        # The PBL-level cells (below 1.2 × 1500 = 1800 m) must differ —
        # unstable vs stable branch → substantially different Kz.
        # The top of the column is above taper for both, so those cells
        # are the same (Kz_bg). Check the bottom cell (surface layer).
        @test cache1[1, 1, Nz] != cache2[1, 1, Nz]
    end

    @testset "type stability of field_value" begin
        FT = Float64
        surface = _constant_surface(FT; pblh=1000.0, ustar=0.3,
                                    hflux=100.0, t2m=295.0)
        delp    = _make_delp_field(2, 2, fill(20000.0, 3))
        cache   = zeros(FT, 2, 2, 3)
        f       = DerivedKzField(; surface, delp, cache)
        update_field!(f, 0.0)
        @test @inferred(field_value(f, (1, 1, 1))) isa Float64
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
#   - integral_between: exact within a single window, multi-window sums,
#                       zero for empty overlap, monotone (t1 <= t2 check)
#   - Sub-step additivity: integral over [t1, t2] + [t2, t3] == [t1, t3]
#   - Type stability of field_value
#   - Kernel-safety on CPU backend
#   - Adapt / GPU adaptation (samples → CuArray, current_window → CuArray)

@testset "TimeVaryingField — StepwiseField" begin

    @testset "construction — rank-1 (N=0 spatial, 1 time dim only)" begin
        # Degenerate: rank-0 (scalar) field with time dimension
        samples  = Float64[1.0, 2.0, 3.0]      # 3 windows
        bounds   = Float64[0.0, 1.0, 2.0, 3.0]  # 4 boundaries
        f = StepwiseField(samples, bounds)
        @test f isa AbstractTimeVaryingField{Float64, 0}
        @test f isa StepwiseField
    end

    @testset "construction — rank-2 (surface)" begin
        # (Nx, Ny, N_win) = (2, 3, 4)
        samples = reshape(collect(1.0:24.0), 2, 3, 4)
        bounds  = [0.0, 10.0, 20.0, 30.0, 40.0]
        f = StepwiseField(samples, bounds)
        @test f isa AbstractTimeVaryingField{Float64, 2}
    end

    @testset "construction — rank-3 (volume)" begin
        # (Nx, Ny, Nz, N_win) = (2, 3, 4, 2)
        samples = reshape(collect(1.0:48.0), 2, 3, 4, 2)
        bounds  = [0.0, 100.0, 200.0]
        f = StepwiseField(samples, bounds)
        @test f isa AbstractTimeVaryingField{Float64, 3}
    end

    @testset "constructor — invalid inputs throw" begin
        # Wrong boundary length
        @test_throws ArgumentError StepwiseField(Float64[1.0, 2.0],
                                                 Float64[0.0, 1.0])
        # Unsorted boundaries
        @test_throws ArgumentError StepwiseField(Float64[1.0, 2.0],
                                                 Float64[0.0, 2.0, 1.0])
    end

    @testset "field_value reads from current window" begin
        # rank-2, 3 windows, values differ by window only for simple check
        samples = zeros(Float64, 2, 2, 3)
        samples[:, :, 1] .= 10.0
        samples[:, :, 2] .= 20.0
        samples[:, :, 3] .= 30.0
        bounds  = [0.0, 1.0, 2.0, 3.0]
        f = StepwiseField(samples, bounds)

        update_field!(f, 0.5)
        @test field_value(f, (1, 1)) === 10.0
        @test field_value(f, (2, 2)) === 10.0

        update_field!(f, 1.5)
        @test field_value(f, (1, 1)) === 20.0

        update_field!(f, 2.999)
        @test field_value(f, (1, 1)) === 30.0
    end

    @testset "update_field! — half-open window convention" begin
        # boundary convention: [b[n], b[n+1]) → b[n+1] is NOT in window n
        samples = Float64[10.0, 20.0, 30.0]
        bounds  = Float64[0.0, 1.0, 2.0, 3.0]
        f = StepwiseField(samples, bounds)

        # Left-closed
        update_field!(f, 0.0);  @test field_value(f, ()) === 10.0
        update_field!(f, 1.0);  @test field_value(f, ()) === 20.0
        update_field!(f, 2.0);  @test field_value(f, ()) === 30.0

        # Interior
        update_field!(f, 0.5);   @test field_value(f, ()) === 10.0
        update_field!(f, 1.999); @test field_value(f, ()) === 20.0

        # t == b[end] is outside (right-open)
        @test_throws ArgumentError update_field!(f, 3.0)
    end

    @testset "update_field! — out of range throws" begin
        f = StepwiseField(Float64[1.0, 2.0], Float64[0.0, 10.0, 20.0])
        @test_throws ArgumentError update_field!(f, -0.1)
        @test_throws ArgumentError update_field!(f, 20.0)
        @test_throws ArgumentError update_field!(f, 100.0)
    end

    @testset "update_field! is idempotent" begin
        samples = Float64[1.0, 2.0, 3.0]
        bounds  = Float64[0.0, 1.0, 2.0, 3.0]
        f = StepwiseField(samples, bounds)

        update_field!(f, 1.5)
        @test field_value(f, ()) === 2.0
        # Calling twice at the same t gives the same result
        update_field!(f, 1.5)
        @test field_value(f, ()) === 2.0

        # Return value is f for chaining
        @test update_field!(f, 1.5) === f
    end

    @testset "integral_between — within a single window" begin
        # rate = 5.0 over [0, 10). Integral over [2, 7] = 5 * 5 = 25
        samples = Float64[5.0]
        bounds  = Float64[0.0, 10.0]
        f = StepwiseField(samples, bounds)
        @test integral_between(f, 2.0, 7.0, ()) == 25.0
    end

    @testset "integral_between — spans multiple windows" begin
        # rate 1 from [0, 10), rate 2 from [10, 20), rate 5 from [20, 30)
        samples = Float64[1.0, 2.0, 5.0]
        bounds  = Float64[0.0, 10.0, 20.0, 30.0]
        f = StepwiseField(samples, bounds)
        # [5, 25] = 1 × 5 + 2 × 10 + 5 × 5 = 50
        @test integral_between(f, 5.0, 25.0, ()) == 50.0
        # Full range sums to 1 × 10 + 2 × 10 + 5 × 10 = 80
        @test integral_between(f, 0.0, 30.0, ()) == 80.0
    end

    @testset "integral_between — substep additivity" begin
        # Critical property for sub-stepping: sum of integrals over
        # contiguous pieces equals integral over the union.
        samples = Float64[3.0, 7.0, 11.0]
        bounds  = Float64[0.0, 1.0, 2.0, 3.0]
        f = StepwiseField(samples, bounds)

        # Piecewise across window boundaries
        total = integral_between(f, 0.0, 3.0, ())
        pieces = integral_between(f, 0.0, 0.5, ()) +
                 integral_between(f, 0.5, 1.5, ()) +
                 integral_between(f, 1.5, 2.7, ()) +
                 integral_between(f, 2.7, 3.0, ())
        @test total ≈ pieces
    end

    @testset "integral_between — t1 > t2 throws" begin
        f = StepwiseField(Float64[1.0], Float64[0.0, 10.0])
        @test_throws ArgumentError integral_between(f, 5.0, 2.0, ())
    end

    @testset "integral_between — rank-2 spatial index" begin
        # Different spatial points can have different rates per window.
        samples = zeros(Float64, 2, 2, 2)
        samples[1, 1, 1] = 1.0;  samples[1, 1, 2] = 2.0
        samples[2, 2, 1] = 10.0; samples[2, 2, 2] = 20.0
        bounds = Float64[0.0, 5.0, 10.0]
        f = StepwiseField(samples, bounds)

        @test integral_between(f, 0.0, 10.0, (1, 1)) == 1.0 * 5 + 2.0 * 5
        @test integral_between(f, 0.0, 10.0, (2, 2)) == 10.0 * 5 + 20.0 * 5
    end

    @testset "type stability of field_value" begin
        # Rank-0
        f0 = StepwiseField(Float64[1.0, 2.0], Float64[0.0, 1.0, 2.0])
        update_field!(f0, 0.5)
        @test @inferred(field_value(f0, ())) === 1.0

        # Rank-3
        f3 = StepwiseField(reshape(Float32[1:24;], 2, 3, 4, 1),
                           Float32[0.0, 100.0])
        update_field!(f3, 50.0)
        @test @inferred(field_value(f3, (1, 1, 1))) isa Float32
    end

    @testset "rank-mismatched index does not dispatch" begin
        f = StepwiseField(reshape(Float64[1:6;], 2, 3), Float64[0.0, 1.0, 2.0, 3.0])
        # rank-1 field (N=1), (i,j) 2-tuple should fail
        @test_throws MethodError field_value(f, (1, 2))
        @test_throws MethodError field_value(f, ())
    end

    @testset "kernel-safety — CPU backend" begin
        # Verify field_value callable from inside a KA kernel after
        # update_field!. Writes the current window's sample into an output.
        samples = reshape(collect(1.0:24.0), 2, 3, 4)   # (Nx, Ny, N_win)
        bounds  = Float64[0.0, 1.0, 2.0, 3.0, 4.0]
        f = StepwiseField(samples, bounds)

        update_field!(f, 2.5)  # window 3
        # Expected samples[:, :, 3] = reshape(13:18, 2, 3)
        expected = reshape(13.0:18.0, 2, 3)

        out = zeros(Float64, 2, 3)
        backend = get_backend(out)

        @kernel function _fill_from_surface_field!(out, field)
            i, j = @index(Global, NTuple)
            @inbounds out[i, j] = field_value(field, (i, j))
        end

        _fill_from_surface_field!(backend, 16)(out, f; ndrange = size(out))
        synchronize(backend)
        @test out == expected

        # Updating the window AFTER kernel launch changes the read.
        update_field!(f, 0.5)  # window 1
        expected1 = reshape(1.0:6.0, 2, 3)
        fill!(out, 0.0)
        _fill_from_surface_field!(backend, 16)(out, f; ndrange = size(out))
        synchronize(backend)
        @test out == expected1
    end

    @testset "Adapt — structural conversion" begin
        # Test that adapt_structure preserves values and structure on the
        # trivial `Array -> Array` adaptor case (CPU passthrough). Real GPU
        # case mirrors the pattern and is covered by the HAS_GPU_FIELDS
        # block when CUDA is available.
        samples = reshape(collect(1.0:12.0), 2, 3, 2)
        bounds  = Float64[0.0, 1.0, 2.0]
        f       = StepwiseField(samples, bounds)

        # Adapt to Array (no-op on CPU, exercises the adapt_structure path)
        f_adapt = Adapt.adapt(Array, f)
        @test f_adapt isa StepwiseField
        @test f_adapt.samples == f.samples
        @test f_adapt.boundaries == f.boundaries

        # Adapted field's current_window is preserved (same value as source).
        update_field!(f, 1.5)
        f_adapt2 = Adapt.adapt(Array, f)
        @test f_adapt2.current_window[1] == 2
    end

    @testset "Mass-accounting invariant (plan 17 acceptance criterion)" begin
        # User-requested check: Σ (rate × dt × step_count) over the run
        # window equals the Earth-total emissions integrated over the run.
        #
        # Model the rate as a rank-2 StepwiseField with per-cell kg/s.
        # Run 40 time steps of dt = 0.25 s across a 10 s window.
        # Total emitted mass per cell = ∫ rate(t) dt.
        Nx, Ny = 3, 4
        # 2 windows of 5s each. rate[i,j,1] = (i+j), rate[i,j,2] = 2*(i+j).
        samples = zeros(Float64, Nx, Ny, 2)
        for j in 1:Ny, i in 1:Nx
            samples[i, j, 1] = Float64(i + j)
            samples[i, j, 2] = Float64(2 * (i + j))
        end
        bounds = Float64[0.0, 5.0, 10.0]
        f = StepwiseField(samples, bounds)

        # Method 1: explicit integral from the field
        earth_total_expected = 0.0
        for j in 1:Ny, i in 1:Nx
            earth_total_expected += integral_between(f, 0.0, 10.0, (i, j))
        end

        # Method 2: Riemann sum with rate × dt at each substep time,
        # emulating how the kernel would accumulate mass.
        dt = 0.25
        n_steps = 40
        riemann_sum = 0.0
        for n in 1:n_steps
            t = (n - 1) * dt   # emit at beginning of step (kernel convention)
            update_field!(f, t)
            for j in 1:Ny, i in 1:Nx
                riemann_sum += field_value(f, (i, j)) * dt
            end
        end

        # For stepwise-constant fields, step-beginning Riemann = exact integral
        # provided steps align with window boundaries. Here dt = 0.25 divides
        # 5.0 cleanly (20 substeps per window) so the result is exact.
        @test riemann_sum ≈ earth_total_expected
    end

    if HAS_GPU_FIELDS
        @testset "Adapt — GPU adaptation" begin
            samples = reshape(collect(1.0:24.0), 2, 3, 4)
            bounds  = Float64[0.0, 1.0, 2.0, 3.0, 4.0]
            f_cpu   = StepwiseField(samples, bounds)
            update_field!(f_cpu, 2.5)   # window 3

            f_gpu = Adapt.adapt(CUDA.CuArray, f_cpu)
            @test f_gpu.samples isa CUDA.CuArray{Float64, 3}
            @test f_gpu.current_window isa CUDA.CuArray{Int, 1}

            # Kernel launch on GPU reading from the adapted field
            out = CUDA.zeros(Float64, 2, 3)
            @kernel function _gpu_fill_from_surface!(out, field)
                i, j = @index(Global, NTuple)
                @inbounds out[i, j] = field_value(field, (i, j))
            end
            _gpu_fill_from_surface!(get_backend(out), 16)(out, f_gpu;
                                                         ndrange = size(out))
            synchronize(get_backend(out))

            # Expected: samples[:, :, 3] = reshape(13:18, 2, 3)
            @test Array(out) == reshape(13.0:18.0, 2, 3)
        end
    end
end
