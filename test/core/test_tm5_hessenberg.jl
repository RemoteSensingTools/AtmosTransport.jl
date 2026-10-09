using Test, Random, LinearAlgebra
using AtmosTransport
import KernelAbstractions as KA
const C = AtmosTransport.Operators.Convection
const AD = AtmosTransport.Adjoints

@testset "Hessenberg LU retains general pivots and triangular solves" begin
    rng = MersenneTwister(9042026)
    swaps = 0
    for FT in (Float32,Float64), n in (1,2,8,25,85), trial in 1:8
        A = triu(randn(rng, FT, n, n), -1)
        if n > 1
            A[1,1], A[2,1] = FT(0.01), FT(2)
        end
        dense, structured = copy(A), copy(A)
        pd, ps = zeros(Int,n), zeros(Int,n)
        C._tm5_lu!(dense, pd, n)
        C._tm5_hessenberg_lu!(structured, ps, n)
        swaps += count(ps .!= 1:n)
        @test dense == structured
        @test pd == ps
        rhs = randn(rng, FT, n, 7)
        q, reference = copy(rhs), copy(rhs)
        C._tm5_solve!(reference, dense, pd, n, 7)
        C._tm5_solve!(q, structured, ps, n, 7)
        bits = FT === Float32 ? UInt32 : UInt64
        @test reinterpret(bits, vec(q)) == reinterpret(bits, vec(reference))
        @test norm(Float64.(A)*Float64.(q)-Float64.(rhs)) /
              (norm(A)*norm(q)+norm(rhs)) < 100eps(FT)
        y = randn(rng, FT, n)
        adjoint_q = copy(y)
        AD._tm5_solve_vector_transpose!(adjoint_q, structured, ps, n)
        @test norm(transpose(Float64.(A))*Float64.(adjoint_q)-Float64.(y)) /
              (norm(A)*norm(adjoint_q)+norm(y)) < 100eps(FT)
    end
    @test swaps > 0
    # Zero levels retain the existing no-op contract.
    @test C._tm5_hessenberg_lu!(zeros(0,0), Int[], 0) === nothing
end

function physical_rates(rng, FT, n, top, downdraft)
    cmfmc, dtrain = zeros(FT,1,1,n+1), zeros(FT,1,1,n)
    cmfmc[1,1,(top+1):n] .= FT(0.1) .* rand(rng,FT,n-top)
    dtrain[1,1,top:n] .= FT(0.01) .* rand(rng,FT,n-top+1)
    e, d = similar(dtrain), similar(dtrain)
    C._derive_cmfmc_matrix_rates_ll_kernel!(KA.CPU(), (1,1))(
        e, d, cmfmc, dtrain, n; ndrange=(1,1))
    KA.synchronize(KA.CPU())
    ed, dd = zeros(FT,n), zeros(FT,n)
    ed[top+1] = downdraft
    dd[n] = downdraft
    (vec(e), vec(d), ed, dd)
end

@testset "Production forward and adjoint choose exact column structure" begin
    rng = MersenneTwister(9042026)
    for FT in (Float32,Float64), n in (8,25,85,137), top in (1,4),
        down in (zero(FT), FT(0.01), eps(FT)^2)
        rates = physical_rates(rng, FT, n, top, down)
        m = FT(50) .+ FT(150) .* rand(rng,FT,n)
        dt = FT(1800)
        ct, _, lfs = C._tm5_diagnose_cloud_dims(rates[2],rates[3],n)
        lo = min(lfs,max(ct,2)-1)
        @test (lfs > n) == iszero(down) # Even tiny positive downdrafts stay dense.
        A = Matrix{FT}(I,n,n)
        C._tm5_build_conv1!(A, rates..., m, ct, lfs, dt, n;
                           f=A, amu=zeros(FT,n+1), amd=zeros(FT,n+1))
        if iszero(down)
            @test all(iszero(A[i,j]) for j in 1:n for i in (j+2):n)
        end
        dense, pd = copy(A), fill(-1,n)
        C._tm5_lu!(dense, pd, n; icltop_eff=lo)
        rhs = rand(rng,FT,n,65) .* m
        reference = copy(rhs)
        for tracer in 1:65                       # plain dense solve + column mass ledger
            C._tm5_conserving_solve_tracer!(reference, dense, pd, n, lo, tracer, false)
        end

        # The automatic production path must agree with explicitly dense LU.
        q, work, piv = copy(rhs), Matrix{FT}(I,n,n), fill(-1,n)
        C._tm5_solve_column!(q, m, rates..., work, piv, zeros(Int,3), dt;
            f_buf=work, amu_buf=zeros(FT,n+1), amd_buf=zeros(FT,n+1))
        bits = FT === Float32 ? UInt32 : UInt64
        @test reinterpret(bits, vec(q)) == reinterpret(bits, vec(reference))
        @test work == dense
        @test piv == pd
        @test minimum(q) >= 0
        # The plain solve conserves to rounding (the matrix is closed), and the
        # column mass ledger makes it exact up to one rounding in one cell.
        plain = copy(rhs)
        C._tm5_solve!(plain, dense, pd, n, 65; icltop_eff=lo)
        @test maximum(abs.(sum(Float64, plain; dims=1) - sum(Float64, rhs; dims=1)) ./
                      sum(Float64, rhs; dims=1)) < 100eps(FT)
        @test all(t -> abs(sum(big, q[:,t]) - sum(big, rhs[:,t])) <= eps(FT) * maximum(q[:,t]), 1:65)

        # Exercise the column entry points used by CS adjoint replay too.
        x, y = copy(rhs[:,1]), randn(rng,FT,n)
        fwd, adj = copy(x), copy(y)
        for (solve!, profile) in ((AD._tm5_solve_column_vector!,fwd),
                                  (AD._tm5_solve_column_vector_adjoint!,adj))
            work = Matrix{FT}(I,n,n)
            solve!(profile, m, rates..., work, fill(-1,n), zeros(Int,3), dt;
                   f_buf=work, amu_buf=zeros(FT,n+1), amd_buf=zeros(FT,n+1))
        end
        adj_reference = copy(y)
        AD._tm5_solve_vector_transpose!(adj_reference, dense, pd, n; icltop_eff=lo)
        @test reinterpret(bits, fwd) == reinterpret(bits, reference[:,1])
        @test reinterpret(bits, adj) == reinterpret(bits, adj_reference)
        @test abs(dot(fwd,y)-dot(x,adj)) < 100eps(FT)*max(norm(fwd)*norm(y),norm(x)*norm(adj))
    end
end
