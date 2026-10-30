using Test, Random, LinearAlgebra
using AtmosTransport
const C = AtmosTransport.Operators.Convection
const AD = AtmosTransport.Adjoints

@testset "Bidiagonal eligibility follows active pivot history" begin
    @test C.TM5._tm5_identity_pivots(Int32[-1,-1,3,4,5], 5, 3)
    @test !C.TM5._tm5_identity_pivots(Int32[-1,-1,4,4,5], 5, 3)
    A = Float64[0.01 2 3; 2 4 5; 0 6 7]
    pivots = zeros(Int,3)
    C.TM5._tm5_hessenberg_lu!(A,pivots,3)
    @test pivots[1] == 2
    @test !C.TM5._tm5_identity_pivots(pivots,3,1)
end

@testset "Bidiagonal RHS solves agree with general LU solves" begin
    rng = MersenneTwister(9052026)
    for FT in (Float32,Float64), n in (1,2,8,60,85), lo in unique((1,min(4,n)))
        # A well-conditioned upper-Hessenberg matrix with guaranteed no swaps.
        # Inactive rows are identity; inactive pivot slots deliberately invalid.
        A = Matrix{FT}(I,n,n)
        for k in lo:n
            A[k,k] = FT(n+1) + rand(rng,FT)
            for j in (k+1):n
                A[k,j] = FT(0.1)*randn(rng,FT)
            end
            k > lo && (A[k,k-1] = -rand(rng,FT))
        end
        lu, pivots = copy(A), fill(Int32(-1),n)
        C.TM5._tm5_hessenberg_lu!(lu,pivots,n; icltop_eff=lo)
        @test C.TM5._tm5_identity_pivots(pivots,n,lo)
        @test all(iszero(lu[k,j]) for j in lo:n for k in (j+2):n)
        saved = copy(lu)
        bits = FT === Float32 ? UInt32 : UInt64
        for nt in (1,7,32,65,129)
            rhs = randn(rng,FT,n,nt)
            rhs[:,1] .= 0 # Include zeros and a sparse tracer.
            rhs[n,1] = one(FT)
            reference = copy(rhs)
            foreach(tracer -> C.TM5._tm5_solve_shared_tracer!(reference, lu, pivots, n, max(Int(lo), 1), tracer), 1:nt)
            actual = copy(rhs)
            foreach(tracer -> C.TM5._tm5_solve_bidiagonal_tracer!(actual, lu, n, max(Int(lo), 1), tracer), 1:nt)
            @test reinterpret(bits,vec(actual)) == reinterpret(bits,vec(reference))
            @test actual[1:(lo-1),:] == rhs[1:(lo-1),:]
            @test norm(Float64.(A)*Float64.(actual)-Float64.(rhs)) /
                  (norm(A)*norm(actual)+norm(rhs)) < 100eps(FT)
            # Same helper on a strided column view, as in the per-thread kernels.
            packed = zeros(FT,2,3,n,nt)
            column = view(packed,2,2,:,:)
            column .= rhs
            foreach(tracer -> C.TM5._tm5_solve_bidiagonal_tracer!(column, lu, n, max(Int(lo), 1), tracer), 1:nt)
            @test column == reference
            # Shared-buffer batches, including the final partial batch.
            shared = zeros(FT,n,6)
            for first in 1:6:nt
                count = min(6,nt-first+1)
                shared[:,1:count] .= rhs[:,first:(first+count-1)]
                for slot in 1:count
                    C.TM5._tm5_solve_bidiagonal_tracer!(shared,lu,n,lo,slot)
                end
                @test shared[:,1:count] == reference[:,first:(first+count-1)]
            end
        end
        x, y = randn(rng,FT,n), randn(rng,FT,n)
        general, fast = copy(x), copy(x)
        C.TM5._tm5_solve_shared_tracer!(reshape(general,:,1),lu,pivots,n,lo,1)
        C.TM5._tm5_solve_bidiagonal_tracer!(reshape(fast,:,1),lu,n,lo,1)
        @test reinterpret(bits,fast) == reinterpret(bits,general)
        general_adj, fast_adj = copy(y), copy(y)
        C.TM5._tm5_solve_vector_transpose!(general_adj,lu,pivots,n; icltop_eff=lo)
        C.TM5._tm5_solve_vector_transpose_bidiagonal!(fast_adj,lu,n; icltop_eff=lo)
        @test reinterpret(bits,fast_adj) == reinterpret(bits,general_adj)
        @test abs(dot(fast,y)-dot(x,fast_adj)) <
              100eps(FT)*max(norm(fast)*norm(y),norm(x)*norm(fast_adj))
        @test isequal(lu,saved)
    end
end
