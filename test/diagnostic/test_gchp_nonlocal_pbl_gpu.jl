# GEOS-Chem non-local PBL: the CUDA column kernel and the profile deposit match
# the CPU. Set CUDA_VISIBLE_DEVICES explicitly and opt in before running.
using Test
if get(ENV, "ATMOSTR_RUN_NONLOCAL_PBL_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in GEOS-Chem non-local PBL GPU tests"
else
    using CUDA, Adapt
    using AtmosTransport
    using AtmosTransport.State: panel_field
    using AtmosTransport.Operators.SurfaceFlux: SurfaceFluxSource, SurfaceFluxOperator,
                                                emission_deposit, apply_surface_flux!
    using AtmosTransport.Operators.Diffusion: ImplicitVerticalDiffusion, DiffusiveSurfaceFluxBoundary
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__, "..", "helpers", "gchp_nonlocal_pbl.jl"))
    @testset "CUDA GEOS-Chem non-local PBL" begin
        for T in (Float32, Float64), hflux in (250.0, -30.0)
            cpu, _ = refreshed_pbl_field(T; hflux)
            gpu, _ = refreshed_pbl_field(T, x -> Adapt.adapt(CuArray, x); hflux)
            rtol = T === Float32 ? 1e-4 : 1e-10
            for p in 1:6
                @test Array(panel_field(gpu, p).data) ≈ panel_field(cpu, p).data rtol = rtol
                @test Array(gpu.emission_profile[p]) ≈ cpu.emission_profile[p] rtol = rtol atol = 1e-6
            end
            # Fresh emissions spread by the GPU profile conserve mass like the CPU.
            rate = ntuple(_ -> fill(T(1), NONLOCAL_NC, NONLOCAL_NC), 6)
            q0() = ntuple(_ -> fill(T(1e4), NONLOCAL_NC + 2, NONLOCAL_NC + 2, NONLOCAL_NZ, 1), 6)
            run(field, arr) = begin
                op = ImplicitVerticalDiffusion(; kz_field = field,
                                               surface_flux_coupling = DiffusiveSurfaceFluxBoundary())
                q = map(arr, q0())
                src = SurfaceFluxSource(:x, map(arr, rate))
                apply_surface_flux!(q, SurfaceFluxOperator(src), nothing, 600.0, nothing, nothing;
                                    tracer_names = (:x,), halo_width = 1, deposit = emission_deposit(op))
                map(Array, q)
            end
            q_cpu, q_gpu = run(cpu, identity), run(gpu, CuArray)
            @test sum(sum, q_gpu) ≈ sum(sum, q_cpu) rtol = rtol
            @test q_gpu[1] ≈ q_cpu[1] rtol = rtol
        end
    end
end
