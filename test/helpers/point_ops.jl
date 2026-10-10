# Point-operation checks shared by the opt-in device diagnostics: one operation,
# a fused launch of operations with different index extents, and a dependent
# sequence must write exactly what the host loops write. `device` moves a host
# array to the device (e.g. `CuArray`, `MtlArray`); data are Float32.
using Test, AtmosTransport, KernelAbstractions
const PointArch = AtmosTransport.Architectures

module _PointOpDeviceTests
using AtmosTransport.Architectures: AbstractPointOp
import AtmosTransport.Architectures: index_space, apply_point!
# Writes `factor * src` into slot `slot` of `out` over an n1 × n2 × n3 block.
struct ScaleInto <: AbstractPointOp
    slot::Int; n1::Int; n2::Int; n3::Int; factor::Float32
end
index_space(op::ScaleInto, ctx) = (op.n1, op.n2, op.n3)
@inline apply_point!(op::ScaleInto, ctx, i, j, k) =
    (ctx.out[i, j, k, op.slot] = op.factor * ctx.src[i, j, k]; nothing)
# Adds `src` to slot 1: correct only after `ScaleInto` slot 1 (a dependency).
struct AddSource <: AbstractPointOp end
index_space(::AddSource, ctx) = size(ctx.src)
@inline apply_point!(::AddSource, ctx, i, j, k) =
    (ctx.out[i, j, k, 1] += ctx.src[i, j, k]; nothing)
# Binding replaces the context with a different shape (counted on the host):
# the index space must come from the unbound context, binding must happen once.
struct Rebind <: AbstractPointOp end
const REBINDS = Ref(0)
index_space(::Rebind, ctx) = size(ctx.src)
import AtmosTransport.Architectures: bound_context
bound_context(::Rebind, ctx) = (REBINDS[] += 1; (; target = ctx.out, source = ctx.src))
@inline apply_point!(::Rebind, ctx, i, j, k) =
    (ctx.target[i, j, k, 1] = 3f0 * ctx.source[i, j, k]; nothing)
end
using ._PointOpDeviceTests: ScaleInto, AddSource, Rebind, REBINDS

function check_point_ops_on_device(device)
    # Extents that are not multiples of a workgroup; cells an operation does
    # not cover keep the sentinel.
    Nx, Ny, Nz = 37, 29, 7
    src = Float32[sin(0.3f0 * i) * cos(0.2f0 * j) + 0.05f0 * k
                  for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    sentinel = -999f0
    host_fields(nslot) = (; src = copy(src), out = fill(sentinel, Nx, Ny, Nz, nslot))
    device_fields(nslot) = (; src = device(src), out = device(fill(sentinel, Nx, Ny, Nz, nslot)))

    @testset "launch! of one point operation matches the host loop" begin
        op = ScaleInto(1, Nx, Ny, Nz, 2f0)
        host = host_fields(1)
        PointArch.launch!(op, host, KernelAbstractions.CPU())
        @test host.out[:, :, :, 1] == 2f0 .* src
        for workgroup in (32, 256)
            dev = device_fields(1)
            PointArch.launch!(op, dev, get_backend(dev.out); workgroup)
            @test Array(dev.out) == host.out
        end
    end

    @testset "Fused operations of different extents match the host loops" begin
        # Five slots give an uneven slot tree; operations 2 to 5 each cover
        # less than the fused index space, so the kernel masks their indices.
        ops = (ScaleInto(1, Nx, Ny, Nz, 2f0), ScaleInto(2, 20, Ny, Nz, 0.5f0),
               ScaleInto(3, Nx, 11, 3, -3f0), ScaleInto(4, 1, 1, Nz, 0.1f0),
               ScaleInto(5, Nx, Ny, 1, 1.5f0))
        fused = PointArch.Fused(ops...)
        @test PointArch.index_space(fused, nothing) == (Nx, Ny, Nz, 5)
        expected = fill(sentinel, Nx, Ny, Nz, 5)
        for op in ops
            block = (1:op.n1, 1:op.n2, 1:op.n3)
            expected[block..., op.slot] .= op.factor .* src[block...]
        end
        host = host_fields(5)
        PointArch.launch!(fused, host, KernelAbstractions.CPU())
        @test host.out == expected
        dev = device_fields(5)
        PointArch.launch!(fused, dev, get_backend(dev.out))
        @test Array(dev.out) == host.out
    end

    @testset "A single launch and separate launches bind the context once" begin
        backend = get_backend(device(src))
        for launch in ((op, ctx) -> PointArch.launch!(op, ctx, backend),
                       (op, ctx) -> (PointArch._launch_fused!(PointArch.SeparateLaunches(),
                                                              PointArch.Fused(op), ctx, backend, 256);
                                     KernelAbstractions.synchronize(backend)))
            REBINDS[] = 0
            dev = device_fields(1)
            launch(Rebind(), dev)
            @test REBINDS[] == 1
            @test Array(dev.out)[:, :, :, 1] == 3f0 .* src
        end
    end

    @testset "Sequence launches a dependent operation after the first" begin
        seq = PointArch.Sequence(ScaleInto(1, Nx, Ny, Nz, 2f0), AddSource())
        host = host_fields(1)
        PointArch.launch!(seq, host, KernelAbstractions.CPU())
        @test host.out[:, :, :, 1] == 2f0 .* src .+ src
        dev = device_fields(1)
        PointArch.launch!(seq, dev, get_backend(dev.out))
        @test Array(dev.out) == host.out
    end
end
