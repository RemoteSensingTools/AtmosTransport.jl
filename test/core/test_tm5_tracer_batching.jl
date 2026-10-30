using Test, Random, LinearAlgebra
using AtmosTransport
import KernelAbstractions

const C = AtmosTransport.Operators.Convection

# Exercise host selection without querying or initializing any GPU.
struct BatchTestBackend <: KernelAbstractions.GPU end

@testset "Collaborative convection batching: host selection" begin
    backend = BatchTestBackend()
    op = C.TM5.TM5Convection(use_collab_lu=true)
    for nt in (1, 6, 7, 12, 32, 65, 129)
        @test C.TM5._should_use_collab(op, 85, nt, Float32, backend)
    end
    @test !C.TM5._use_collab_path(op, 85, 65, Float32, KernelAbstractions.CPU())
    @test !C.TM5._use_collab_path(op, 85, 65, Float64, backend)
    @test !C.TM5._use_collab_path(op, 66, 6, Float64, backend)
    @test !C.TM5._use_collab_path(op, 66, 6, Float64, KernelAbstractions.CPU())
    @test !C.TM5._use_collab_path(op, 66, 6, Float16, backend)
    @test !C.TM5._use_collab_path(C.TM5.TM5Convection(), 85, 65, Float32, backend)
    @test_throws ArgumentError C.TM5._should_use_collab(op, 86, 65, Float32, backend)
    @test_throws ArgumentError C.TM5._should_use_collab(op, 85, 0, Float32, backend)
    capped = C.TM5.TM5Convection(use_collab_lu=true, lmax_conv=85)
    @test C.TM5._should_use_collab(capped, 137, 65, Float32, backend)
    merged = C.TM5.TM5Convection(use_collab_lu=true, n_merge=2)
    @test C.TM5._should_use_collab(merged, 137, 65, Float32, backend)
    matrix = C.CMFMC.CMFMCMatrixConvection(use_collab_lu=true)
    @test C.TM5._should_use_collab(matrix.inner, 85, 65, Float32, backend)
end

@testset "Shared RHS helper: partial batches, pivots, and active rows" begin
    rng = MersenneTwister(9042026)
    width = C.TM5._TM5_COLLAB_TRACER_BATCH
    for FT in (Float32, Float64), n in (1, 8, 60, 85), lo in unique((1, min(4, n)))
        A = Matrix{FT}(I, n, n)
        active = n-lo+1
        A[lo:n, lo:n] .= randn(rng, FT, active, active) + FT(active)*I
        if active > 1
            A[lo, lo] = FT(0.01)
            A[lo+1, lo] = FT(10)
        end
        lu = copy(A)
        piv = fill(Int32(-1), n) # Inactive pivot entries must never be read.
        C.TM5._tm5_lu!(lu, piv, n; icltop_eff=lo)
        active > 1 && @test piv[lo] != lo
        saved_lu, saved_piv = copy(lu), copy(piv)
        for nt in (1, 6, 7, 12, 32, 65, 129)
            rhs = randn(rng, FT, n, nt)
            reference, actual = copy(rhs), similar(rhs)
            foreach(tracer -> C.TM5._tm5_solve_shared_tracer!(reference, lu, piv, n, max(Int(lo), 1), tracer), 1:nt)
            shared_rhs = fill(FT(NaN), n, width)
            for first in 1:width:nt
                count = min(width, nt-first+1)
                fill!(shared_rhs, FT(NaN))
                shared_rhs[:, 1:count] .= @view rhs[:, first:(first+count-1)]
                for slot in 1:count
                    C.TM5._tm5_solve_shared_tracer!(shared_rhs, lu, piv, n, lo, slot)
                end
                actual[:, first:(first+count-1)] .= @view shared_rhs[:, 1:count]
                @test all(isnan, @view shared_rhs[:, (count+1):width])
            end
            bits = FT === Float32 ? UInt32 : UInt64
            @test reinterpret(bits, vec(actual)) == reinterpret(bits, vec(reference))
            @test actual[1:(lo-1), :] == rhs[1:(lo-1), :]
            @test norm(Float64.(A)*Float64.(actual)-Float64.(rhs)) /
                  (norm(A)*norm(actual)+norm(rhs)) < 100eps(FT)
        end
        @test isequal(lu, saved_lu)
        @test piv == saved_piv
    end
end
