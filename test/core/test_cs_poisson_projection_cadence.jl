#!/usr/bin/env julia

using Test
using Random
using LinearAlgebra

using AtmosTransport

const P = AtmosTransport.Preprocessing

@testset "CS Poisson PCG projection cadence" begin
    Nc = 12
    mesh = CubedSphereMesh(; Nc=Nc, convention=GEOSNativePanelConvention())
    ft = P.build_cs_global_face_table(Nc, mesh.connectivity)
    degree = P.cs_cell_face_degree(ft)

    rng = MersenneTwister(20260429)
    rhs0 = randn(rng, ft.nc)
    rhs0 .-= sum(rhs0) / length(rhs0)

    function solve_with(project_every)
        scratch = P.CSPoissonScratch(ft.nc)
        psi = zeros(Float64, ft.nc)
        rhs = copy(rhs0)
        res, it = P.solve_cs_poisson_pcg!(
            psi, rhs, ft, degree, (r=scratch.r, p=scratch.p, Ap=scratch.Ap, z=scratch.z);
            tol=1e-11, max_iter=2000, project_every=project_every)
        P._cs_graph_laplacian_mul!(scratch.Ap, psi, ft, degree)
        return (; psi, residual=maximum(abs.(scratch.Ap .- rhs0)), solver_res=res, it)
    end

    legacy = solve_with(1)
    periodic = solve_with(8)

    @test legacy.residual < 1e-9
    @test periodic.residual < 1e-9
    @test isapprox(periodic.psi, legacy.psi; rtol=1e-8, atol=1e-8)
    @test abs(sum(periodic.psi) / length(periodic.psi)) < 1e-12
end
