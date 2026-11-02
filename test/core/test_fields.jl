# Numerical reads and time selection of production fields. Constructors,
# rank dispatch and Adapt(Array) storage pins are covered by these kernel reads.
using Test, AtmosTransport, Adapt
using AtmosTransport: ConstantField, PreComputedKzField, StepwiseField,
                      field_value, update_field!
using KernelAbstractions: @kernel, @index, get_backend, synchronize

const HAS_GPU_FIELDS = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

@kernel function read_field!(out, field, ::Val{N}) where N
    idx = @index(Global, NTuple)
    @inbounds out[idx...] = field_value(field, ntuple(i -> idx[i], N))
end

function read_on_backend(field, shape, FT; to=Array, rank=length(shape))
    out = to(zeros(FT, shape))
    backend = get_backend(out)
    read_field!(backend)(out, field, Val(rank); ndrange=shape)
    synchronize(backend)
    return Array(out)
end

@testset "TimeVaryingField — ConstantField" begin
    # Rank zero is read by chemistry; ranks two/three feed surface/volume
    # kernels. Read each rank on the CPU, in both production precisions.
    for FT in (Float32, Float64), shape in ((), (3,2), (3,2,4))
        value = isempty(shape) ? FT(7.25) : length(shape) == 3 ? FT(-1) : FT(42)
        f = ConstantField{FT,length(shape)}(value)
        for time in (0.0, 1.0, 1e9)
            update_field!(f,time)
            if isempty(shape)
                @test field_value(f,()) === value
                @test all(==(value), read_on_backend(f,(3,2,4),FT;rank=0))
            else
                @test all(==(value), read_on_backend(f,shape,FT))
            end
        end
    end
end

@testset "TimeVaryingField — PreComputedKzField" begin
    for FT in (Float32, Float64), shape in ((4,5), (4,3,5))
        data = reshape(FT.(1:prod(shape)),shape)
        f = PreComputedKzField(data)
        @test read_on_backend(f,shape,FT) == data
        # A met-window refresh mutates caller-owned data; field reads must
        # observe every new element, with the same independent indices.
        data .*= -2
        update_field!(f,1e9)
        @test read_on_backend(f,shape,FT) == data
    end
end

@testset "TimeVaryingField — StepwiseField" begin
    @test_throws ArgumentError StepwiseField([1.0,2.0],[0.0,1.0])
    @test_throws ArgumentError StepwiseField([1.0,2.0],[0.0,2.0,1.0])

    for FT in (Float32,Float64)
        f = StepwiseField(FT[10,20,30],FT[0,1,2,3])
        for (time,expected) in ((0,10),(0.5,10),(1,20),(1.999,20),(2,30),(2.999,30),
                                 (1.5,20),(1.5,20))
            update_field!(f,time)
            @test field_value(f,()) === FT(expected)
        end
        for time in (-0.1,3.0,100.0)
            @test_throws ArgumentError update_field!(f,time)
        end
    end

    @testset "adapted StepwiseField remains host-updatable" begin
        # Distinct values at every cell/window expose axis or window mistakes.
        for FT in (Float32,Float64)
            samples = reshape(FT.(1:24),2,3,4)
            f = StepwiseField(samples,FT[0,1,2,3,4])
            for to in (HAS_GPU_FIELDS ? (Array,CUDA.CuArray) : (Array,))
                update_field!(f,1.5)
                adapted = Adapt.adapt(to,f)
                @test read_on_backend(adapted,(2,3),FT;to) == samples[:,:,2]
                for (time,window) in ((2.5,3),(0.5,1),(3.5,4))
                    update_field!(adapted,time)
                    @test read_on_backend(adapted,(2,3),FT;to) == samples[:,:,window]
                end
            end
        end
    end

    @testset "Mass-accounting invariant" begin
        samples = [Float64((i+j)*k) for i in 1:3,j in 1:4,k in 1:2]
        f = StepwiseField(samples,[0.0,5.0,10.0])
        total = 0.0
        for n in 0:39
            update_field!(f,n*0.25)
            total += sum(field_value(f,(i,j))*0.25 for i in 1:3,j in 1:4)
        end
        @test total ≈ sum(15.0*(i+j) for i in 1:3,j in 1:4)
    end
end
