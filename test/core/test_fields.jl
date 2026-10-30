"""
Unit tests for the `TimeVaryingField` abstraction.

Validates the minimum concrete type `ConstantField{FT, N}` against the
field interface:

1. `field_value(f, idx)` returns the stored scalar regardless of index
2. `update_field!(f, t)` is a no-op (returns `f`, no state change)
3. Type stability at every rank used downstream (N ∈ {0, 2, 3})
4. Kernel-safety on CPU (callable inside a KA kernel)

Production boundary-layer caches extend this suite.
"""

using Test
import AtmosTransport
using AtmosTransport: AbstractTimeVaryingField, ConstantField,
                      PreComputedKzField,
                      StepwiseField,
                      field_value, update_field!
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

@testset "TimeVaryingField — PreComputedKzField" begin



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
        # Varying only i or only j must change the value
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

@kernel function _read_adapted_stepwise!(out, field)
    i, j = @index(Global, NTuple)
    out[i, j] = field_value(field, (i, j))
end

@testset "adapted StepwiseField remains host-updatable" begin
    samples = cat(fill(1f0, 3, 2), fill(2f0, 3, 2); dims = 3)
    f = StepwiseField(samples, Float32[0, 1, 2])
    for to in (HAS_GPU_FIELDS ? (Array, CUDA.CuArray) : (Array,))
        adapted = Adapt.adapt(to, f)
        output = to(zeros(Float32, 3, 2))
        backend = get_backend(output)
        for (time, window) in ((1.5f0, 2), (0.5f0, 1))
            update_field!(adapted, time)
            _read_adapted_stepwise!(backend)(output, adapted; ndrange = size(output))
            synchronize(backend)
            @test Array(output) == samples[:, :, window]
        end
    end
end

@testset "TimeVaryingField — StepwiseField" begin







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

        earth_total_expected = sum(15.0 * (i + j) for j in 1:Ny, i in 1:Nx)

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
