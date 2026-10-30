#!/usr/bin/env julia

using Test

using AtmosTransport

const Arch = AtmosTransport.Architectures

struct MockCUDAArray end
struct MockROCArray end
struct MockMetalArray end

@testset "execution architecture selection" begin
    cpu = Arch.architecture_from_config(Dict("backend" => "cpu"))
    @test cpu isa Arch.CPU
    @test !Arch.is_gpu(cpu)
    @test Arch.array_adapter(cpu) === Array
    @test Arch.architecture_label(cpu) == "CPU"
    @test Arch.array_adapter_for(zeros(Float32, 2, 2)) === Array

    # A working-tree CLI may include AtmosTransport after loading CUDA or
    # Metal. In that mode package extensions cannot attach to the local module,
    # so adapter discovery must also recognize arrays through loaded runtimes.
    loaded = Dict{Base.PkgId, Any}(
        Arch._RUNTIME_PACKAGE_IDS.CUDA => (CuArray = MockCUDAArray,),
        Arch._RUNTIME_PACKAGE_IDS.AMDGPU => (ROCArray = MockROCArray,),
        Arch._RUNTIME_PACKAGE_IDS.Metal => (MtlArray = MockMetalArray,),
    )
    @test Arch._loaded_gpu_adapter(MockCUDAArray(), loaded) === MockCUDAArray
    @test Arch._loaded_gpu_adapter(MockROCArray(), loaded) === MockROCArray
    @test Arch._loaded_gpu_adapter(MockMetalArray(), loaded) === MockMetalArray
    @test Arch._loaded_gpu_adapter(zeros(Float32, 2), loaded) === Array

    @test Arch.architecture_from_config(Dict("use_gpu" => false)) isa Arch.CPU
    @test Arch.architecture_from_config(Dict("backend" => "cuda")) isa Arch.GPU{:cuda}
    @test Arch.GPU("metal") isa Arch.GPU{:metal}
    @test Arch._architecture_request("cpu") isa Arch.CPURequest
    @test Arch._architecture_request("cuda") isa Arch.CUDARequest
    @test Arch._architecture_request("metal") isa Arch.MetalRequest
    @test Arch._architecture_request("auto") isa Arch.AutoGPURequest
    @test Arch.backend_name(Arch.GPU(:cuda)) === :cuda
    @test Arch.is_gpu(Arch.GPU(:metal))
    @test_throws MethodError Arch.GPU()
    @test Arch.GPU(:rocm) isa Arch.GPU{:rocm}
    @test Arch.GPU("rocm") isa Arch.GPU{:rocm}
    @test Arch._architecture_request("rocm") isa Arch.ROCmRequest
    @test Arch.backend_name(Arch.GPU(:rocm)) === :rocm
    @test Arch.assert_float_type!(Arch.GPU(:rocm), Float64) === nothing
    @test_throws ArgumentError Arch.GPU(:unknown)
    @test_throws ArgumentError Arch.architecture_from_config(Dict("use_gpu" => true,
                                                                  "backend" => "cpu"))
    @test Arch.architecture_from_config(Dict("backend" => "rocm")) isa Arch.GPU{:rocm}
    @test_throws ArgumentError Arch.architecture_from_config(Dict("backend" => "unknown"))
    @test_throws ArgumentError Arch.architecture_from_config(Dict("backend" => :cuda))

    for alias in ("host", "nvidia", "apple", "apple_metal", "apple-metal", "gpu", "CPU")
        err = try
            Arch.architecture_from_config(Dict("backend" => alias))
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("spelling \"$(alias)\" was removed", sprint(showerror, err))
    end

    architecture_source = read(joinpath(@__DIR__, "..", "..", "src", "Architectures.jl"), String)
    driven_source = join((read(joinpath(@__DIR__, "..", "..", "src", "Models", f), String)
                          for f in ("Runner/DrivenSimulation.jl", "Runner/driven_window_state.jl",
                                    "Physics/driven_physics_refresh.jl", "Core/driven_stepping.jl")))
    @test !occursin(r"\bisdefined\(Main|\bgetproperty\(Main|Core\.eval\(Main", architecture_source)
    @test !occursin(r"\bisdefined\(Main|\bgetproperty\(Main|Core\.eval\(Main", driven_source)
end

@testset "Metal requires Float32" begin
    metal = Arch.architecture_from_config(Dict("backend" => "metal"))
    @test metal isa Arch.GPU{:metal}
    @test Arch.assert_float_type!(metal, Float32) === nothing
    @test_throws ArgumentError Arch.assert_float_type!(metal, Float64)
end

@testset "runtime kernels avoid hard Float64 accumulation" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    files = [
        "src/Operators/Convection/CMFMC/cmfmc_kernels.jl",
    ]
    forbidden = r"Float64\(|zero\(Float64\)|::Float64"
    for file in files
        src = read(joinpath(repo, file), String)
        @test !occursin(forbidden, src)
    end
end

@testset "DrivenRunner resolves one concrete architecture" begin
    cfg = Dict("architecture" => Dict("backend" => "cpu"),
               "numerics" => Dict("float_type" => "Float64"))
    arch = AtmosTransport.Models.DrivenRunner._cfg_architecture(cfg)
    @test arch isa Arch.CPU
    @test Arch.array_adapter(arch) === Array
    @test Arch.architecture_label(arch) == "CPU"
    @test Arch.synchronize_architecture!(arch) === nothing

    runner_source = read(joinpath(@__DIR__, "..", "..", "src", "Models",
                                  "Runner", "DrivenRunner.jl"), String)
    @test occursin("Base.invokelatest(_run_driven_simulation, cfg, arch)",
                   runner_source)
end

@testset "canonical CLI activates package extensions" begin
    cli_source = read(joinpath(@__DIR__, "..", "..", "scripts",
                               "run_transport.jl"), String)
    @test occursin(r"(?m)^using AtmosTransport$", cli_source)
    @test !occursin(r"include\(.*src.*AtmosTransport\.jl", cli_source)
end

# `launch!` instantiates the kernel for the backend and workgroup, runs it over
# `ndrange` with the arguments in order, and synchronizes unless `sync = false`.
import KernelAbstractions
struct _CountingBackend <: KernelAbstractions.Backend
    syncs :: Base.RefValue{Int}
end
KernelAbstractions.synchronize(b::_CountingBackend) = (b.syncs[] += 1; nothing)

@testset "launch! runs the kernel and synchronizes by default" begin
    calls = Any[]
    fake_kernel(backend, workgroup) = (args...; ndrange) -> push!(calls, (workgroup, ndrange, args))
    backend = _CountingBackend(Ref(0))
    AtmosTransport.Architectures.launch!(fake_kernel, backend, (8, 8), (4, 3), :a, 2)
    @test calls == [((8, 8), (4, 3), (:a, 2))]
    @test backend.syncs[] == 1
    AtmosTransport.Architectures.launch!(fake_kernel, backend, 256, 10, :b; sync = false)
    @test calls[end] == (256, 10, (:b,))
    @test backend.syncs[] == 1
end

# Point operations: one body, run as a host loop or as a (fused) kernel.
module _PointOpTests
using AtmosTransport.Architectures: AbstractPointOp
import AtmosTransport.Architectures: index_space, apply_point!
# Writes `value` into `ctx.out[slot]` over an n × m block.
struct Mark <: AbstractPointOp
    slot::Int; n::Int; m::Int; value::Float64
end
index_space(op::Mark, ctx) = (op.n, op.m)
@inline apply_point!(op::Mark, ctx, i, j) = (ctx.out[i, j, op.slot] = op.value; nothing)
# Adds 1 to whatever `Mark` slot 1 wrote: only correct after it (a dependency).
struct Increment <: AbstractPointOp end
index_space(::Increment, ctx) = size(ctx.out)[1:2]
@inline apply_point!(::Increment, ctx, i, j) = (ctx.out[i, j, 1] += 1; nothing)
# Binding replaces the context with a different shape and counts the bindings:
# the index space must come from the unbound context, binding must happen once.
struct Rebind <: AbstractPointOp end
const REBINDS = Ref(0)
index_space(::Rebind, ctx) = size(ctx.out)[1:2]
import AtmosTransport.Architectures: bound_context
bound_context(::Rebind, ctx) = (REBINDS[] += 1; (; target = ctx.out))
@inline apply_point!(::Rebind, ctx, i, j) = (ctx.target[i, j, 1] = 7.0; nothing)
# `Mark` that asks for the kernel on the host; binding is counted, which only
# the host loops of a `Fused` set do (its kernel runs on the unbound context).
struct ParallelMark <: AbstractPointOp
    slot::Int; n::Int; m::Int; value::Float64
end
import AtmosTransport.Architectures: host_parallel
host_parallel(::ParallelMark) = true
index_space(op::ParallelMark, ctx) = (op.n, op.m)
bound_context(::ParallelMark, ctx) = (REBINDS[] += 1; ctx)
@inline apply_point!(op::ParallelMark, ctx, i, j) = (ctx.out[i, j, op.slot] = op.value; nothing)
# A rank-1 operation, which cannot be fused with the rank-2 ones above.
struct Line <: AbstractPointOp
    n::Int
end
index_space(op::Line, ctx) = (op.n,)
@inline apply_point!(::Line, ctx, i) = nothing
# Counts the visits of every index of `ctx.out`.
struct Visit <: AbstractPointOp end
index_space(::Visit, ctx) = size(ctx.out)
@inline apply_point!(::Visit, ctx, I...) = (ctx.out[I...] += 1; nothing)
end
using ._PointOpTests: Mark, Increment, Line, Rebind, REBINDS, ParallelMark

@testset "Point operations: host loop, fused kernel, sequence" begin
    Arch = AtmosTransport.Architectures
    ops = (Mark(1, 4, 3, 1.0), Mark(2, 2, 3, 2.0), Mark(3, 4, 1, 3.0))
    expected = zeros(4, 3, 3)
    expected[:, :, 1] .= 1; expected[1:2, :, 2] .= 2; expected[:, 1, 3] .= 3
    fused = Arch.Fused(ops...)
    @test Arch.index_space(fused, nothing) == (4, 3, 3)
    @test only(Base.return_types(Arch.Fused, Tuple{Mark, Mark, Mark})) === typeof(fused)
    @test isbits(fused)
    @test Arch.Fused(ops) === fused
    # A fused launch needs at least one operation.
    @test_throws ArgumentError Arch.Fused()
    @test_throws ArgumentError Arch.Fused(())
    # Members must be point operations, and fused operations must share one rank.
    @test_throws MethodError Arch.Fused((Mark(1, 4, 3, 1.0), 2))
    @test_throws MethodError Arch.Sequence((Mark(1, 4, 3, 1.0), 2))
    @test_throws ArgumentError Arch.index_space(Arch.Fused(Mark(1, 4, 3, 1.0), Line(5)), nothing)
    @test_throws ArgumentError Arch.launch!(Arch.Fused(Mark(1, 4, 3, 1.0), Line(5)),
                                            (; out = zeros(4, 3, 3)), KernelAbstractions.CPU())

    # The separate-launch policy (Metal) launches each operation with its bound
    # context and gives the fused result.
    @test Arch.fusion_policy(KernelAbstractions.CPU()) === Arch.FuseLaunches()
    separate = (; out = zeros(4, 3, 3))
    Arch._launch_fused!(Arch.SeparateLaunches(), fused, separate, KernelAbstractions.CPU(), 4)
    @test separate.out == expected
    for launch in ((op, ctx) -> Arch.launch!(op, ctx, KernelAbstractions.CPU()),
                   (op, ctx) -> Arch._launch_fused!(Arch.SeparateLaunches(), Arch.Fused(op),
                                                    ctx, KernelAbstractions.CPU(), 4))
        REBINDS[] = 0
        rebound = (; out = zeros(4, 3, 1))
        launch(Rebind(), rebound)
        @test REBINDS[] == 1 && all(==(7.0), rebound.out)
    end

    host = (; out = zeros(4, 3, 3))
    Arch.launch!(fused, host, KernelAbstractions.CPU())
    @test host.out == expected

    # The device path (one kernel over the flat index, slot dispatch, masking)
    # on the CPU backend, with a workgroup that does not divide the space.
    kernel = (; out = zeros(4, 3, 3))
    Arch._launch_point_kernel!(fused, kernel, (4, 3, 3), KernelAbstractions.CPU(), 5)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    @test kernel.out == expected

    seq = (; out = zeros(4, 3, 3))
    Arch.launch!(Arch.Sequence(Mark(1, 4, 3, 1.0), Increment()), seq, KernelAbstractions.CPU())
    @test all(==(2.0), seq.out[:, :, 1])

    # `host_parallel` operations run the kernel on the host: same result, and
    # the fused set is not bound operation by operation.
    @test !Arch.host_parallel(fused)
    parallel = Arch.Fused(map(op -> ParallelMark(op.slot, op.n, op.m, op.value), ops))
    @test Arch.host_parallel(parallel) && Arch.host_parallel(Arch.Fused(ops[1], parallel.ops[2]))
    REBINDS[] = 0
    threaded = (; out = zeros(4, 3, 3))
    Arch.launch!(parallel, threaded, KernelAbstractions.CPU())
    @test threaded.out == expected && REBINDS[] == 0
end

# A device launch runs over the flat index of the space; every index of the
# space must be visited exactly once, in either index width.
@testset "flat point-operation launch visits every index once" begin
    Arch = AtmosTransport.Architectures
    space = (3, 4, 5, 2)
    for T in (Int32, Int)
        @test [Arch._point_index(space, T(r)) for r in 0:(prod(space) - 1)] ==
              vec(Tuple.(CartesianIndices(space)))
    end
    @test Arch._point_index((7,), Int32(6)) == (7,)
    # The 32-bit width covers spaces of up to typemax(Int32) indices.
    @test Arch._point_index_type(Int(typemax(Int32))) === Int32
    @test Arch._point_index_type(Int(typemax(Int32)) + 1) === Int
    big = (3, 2^31)                                  # a dimension beyond Int32
    @test Arch._point_index(big, 3 * 2^31 - 1) == big
    @test Arch._point_index(big, 3 * (2^31 - 1) + 2) == (3, 2^31)
    @test Arch._point_index((2^16, 2^15), Int32(typemax(Int32))) == (2^16, 2^15)
    visits = (; out = zeros(Int, 3, 4, 5))
    for workgroup in (1, 7, 256)
        fill!(visits.out, 0)
        Arch._launch_point_kernel!(_PointOpTests.Visit(), visits, size(visits.out),
                                   KernelAbstractions.CPU(), workgroup)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        @test all(==(1), visits.out)
    end
    # The kernel itself in the 64-bit width.
    fill!(visits.out, 0)
    Arch._point_op_kernel!(KernelAbstractions.CPU(), 7)(_PointOpTests.Visit(), visits,
                                                        size(visits.out), Int; ndrange = length(visits.out))
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    @test all(==(1), visits.out)
    # An empty index space launches nothing.
    @test Arch._launch_point_kernel!(_PointOpTests.Visit(), (; out = zeros(Int, 0, 4, 5)), (0, 4, 5),
                                     KernelAbstractions.CPU(), 256) === nothing
end

@testset "select_panel returns panel p for p in 1:6" begin
    panels = ntuple(p -> fill(p, 2), 6)
    for p in 1:6
        @test AtmosTransport.Architectures.select_panel(panels, p) === panels[p]
    end
end

# `copy_arrays!`: one launch for a set of copies, equal to their `copyto!`s.
@testset "copy_arrays! copies every pair (host loop and device kernel path)" begin
    Arch = AtmosTransport.Architectures
    src = (rand(Float32, 5, 4, 3), rand(Float32, 4, 5, 3), rand(Float32, 4, 4, 4))
    dest = map(a -> fill(NaN32, size(a)), src)
    @test Arch.copy_arrays!(dest, src) === dest
    @test all(map(==, dest, src))
    @test Arch.copy_arrays!((), ()) === ()
    # The device path: one kernel over the longest array, shorter pairs masked.
    kernel = map(a -> fill(NaN32, size(a)), src)
    space = Arch.index_space(Arch.LinearCopy(), (; dest = kernel, src))
    @test space == (maximum(length, src),)
    Arch._launch_point_kernel!(Arch.LinearCopy(), (; dest = kernel, src), space,
                               KernelAbstractions.CPU(), 16)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    @test all(map(==, kernel, src))
    @test_throws DimensionMismatch Arch.copy_arrays!((zeros(Float32, 4, 5, 3),), (src[1],))
    @test_throws ArgumentError Arch.copy_arrays!(dest, src[1:2])
end

# An array that reports another backend, to build a batch that mixes backends.
struct _OtherBackendArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(a::_OtherBackendArray) = size(a.data)
Base.getindex(a::_OtherBackendArray, i::Int...) = a.data[i...]
Base.setindex!(a::_OtherBackendArray, v, i::Int...) = (a.data[i...] = v)
const _OTHER_BACKEND = _CountingBackend(Ref(0))
KernelAbstractions.get_backend(::_OtherBackendArray) = _OTHER_BACKEND

@testset "copy_arrays! launches once only when every array shares one backend" begin
    Arch = AtmosTransport.Architectures
    src = (rand(Float32, 3, 2), rand(Float32, 2, 3))
    @test Arch.single_launch_copy(map(similar, src), src)
    other = _OtherBackendArray(zeros(Float32, 2, 3))
    # Eligibility looks at every pair, not only the first.
    @test !Arch.single_launch_copy((similar(src[1]), other), src)
    @test !Arch.single_launch_copy(map(similar, src), (src[1], _OtherBackendArray(copy(src[2]))))
    @test Arch.single_launch_copy((other,), (_OtherBackendArray(copy(src[2])),))
    # A mixed batch falls back to one `copyto!` per pair (a launch on the fake
    # backend would throw).
    first_other = _OtherBackendArray(zeros(Float32, 3, 2))
    dest = (first_other, similar(src[2]))
    @test Arch.copy_arrays!(dest, src) === dest
    @test first_other.data == src[1] && dest[2] == src[2]
    mixed = (similar(src[1]), other)
    @test Arch.copy_arrays!(mixed, src) === mixed
    @test mixed[1] == src[1] && other.data == src[2]
end
