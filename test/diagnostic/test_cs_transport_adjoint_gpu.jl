# Opt-in checks on an explicitly selected CUDA device.
# ATMOSTR_RUN_TRANSPORT_ADJOINT_GPU_TESTS=1 ATMOSTR_ADJOINT_GPU_NAME=V100 \
# CUDA_VISIBLE_DEVICES=<authorized UUID> julia --project=. test/diagnostic/test_cs_transport_adjoint_gpu.jl
using Test

if get(ENV, "ATMOSTR_RUN_TRANSPORT_ADJOINT_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in transporting adjoint GPU tests"
else
    using CUDA, AtmosTransport
    CUDA.functional(true) || error("CUDA is unavailable")
    expected_device = get(ENV, "ATMOSTR_ADJOINT_GPU_NAME", "A100")
    isempty(expected_device) && error("ATMOSTR_ADJOINT_GPU_NAME must identify the selected device")
    @assert occursin(expected_device, CUDA.name(CUDA.device())) "Wrong device for adjoint tests"
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__, "..", "core", "test_cs_ppm_adjoint_footprint.jl"))

    # A smooth nonzero base avoids limiter switches in centered finite
    # differences. The corner observation exercises panel-edge transport.
    # Float32 is compared with its CPU gradient; Float64 also checks the
    # gradient against perturbations of the GPU forward model.
    @testset "CUDA transporting footprint gradients" begin
        for FT in (Float64, Float32), scheme in (
                AT.PPMScheme(AT.NoLimiter()), AT.PPMScheme(), AT.PPMScheme(AT.CW84Limiter()),
                AT.LinRoodPPMScheme(5), AT.LinRoodPPMScheme(7))
            @testset "$FT $scheme" begin
                mesh, m, rm, am, bm, cm = _transport_cs_problem(; Nc=4, Nz=4, nsteps=3, FT)
                _fill_smooth_tracer!(rm, m, mesh)
                rates = [ntuple(6) do p
                    [FT(sin(0.31s + 0.17p + 0.23i - 0.19j))
                     for i in 1:mesh.Nc, j in 1:mesh.Nc]
                end for s in 1:3]
                gm, grm = map(CuArray, m), map(CuArray, rm)
                gam, gbm, gcm, grates = map(_to_gpu_steps, (am, bm, cm, rates))
                obj = AT.CSColumnMeanObjective(1, 1, 1)
                dt = FT(1.5)
                cpu = AT.cs_surface_emission_footprint(rm, m, am, bm, cm, mesh, obj; scheme, dt)
                gpu = AT.cs_surface_emission_footprint(grm, gm, gam, gbm, gcm, mesh, obj; scheme, dt)
                predicted = _dot_footprint(gpu, grates)
                tolerance = FT == Float64 ? 1e-10 : 2e-5
                for s in 1:3, p in 1:6
                    @test gpu.footprints[s][p] isa CuArray
                    @test isapprox(Array(gpu.footprints[s][p]), cpu.footprints[s][p];
                                   rtol=tolerance, atol=tolerance * 1e-3)
                end
                if FT == Float64
                    eps_dir = FT(2e-6)
                    jp = AT.run_cs_footprint_forward(grm, gm, gam, gbm, gcm, mesh, obj;
                        scheme, dt, emission_rates=_scaled_rates(grates, eps_dir))
                    jm = AT.run_cs_footprint_forward(grm, gm, gam, gbm, gcm, mesh, obj;
                        scheme, dt, emission_rates=_scaled_rates(grates, -eps_dir))
                    fd = (jp - jm) / (2eps_dir)
                    @info "GPU directional gradient" scheme predicted fd relative_error=abs((predicted - fd) / fd)
                    @test isapprox(predicted, fd; rtol=1e-6, atol=1e-9)
                end
                for checkpoint in (AT.StrideCheckpoint(2), AT.RevolveCheckpoint())
                    replay = AT.cs_surface_emission_footprint(grm, gm, gam, gbm, gcm, mesh, obj;
                                                             scheme, dt, checkpoint)
                    for s in 1:3, p in 1:6
                        @test isapprox(Array(replay.footprints[s][p]), Array(gpu.footprints[s][p]);
                                       rtol=tolerance, atol=tolerance * 1e-3)
                    end
                end
                CUDA.synchronize()
            end
        end
    end

    # Exercise the public single-panel wrappers too, including nonzero halo
    # seeds carried between substeps and accumulation into an existing seed.
    function _single_panel_reverse(rm, m, am, bm, mesh, order)
        entry, rm, m = Adv.record_linrood_substep!(rm, m, am[1], bm[1], mesh; ord=order)
        tape = [entry]
        for step in 2:length(am)
            entry, rm, m = Adv.record_linrood_substep!(rm, m, am[step], bm[step], mesh; ord=order)
            push!(tape, entry)
        end
        seed_rm = fill!(similar(rm), one(eltype(rm)))
        seed_m = fill!(similar(m), eltype(m)(0.3))
        lambda_rm, lambda_m = copy(seed_rm), copy(seed_m)
        Adv.apply_linrood_multi_substep_adjoint!(
            lambda_rm, lambda_m, seed_rm, seed_m, tape, am, bm, mesh)
        return rm, m, lambda_rm, lambda_m
    end

    @testset "CUDA single-panel Lin-Rood recording and reverse" begin
        for FT in (Float64, Float32), order in (Val(5), Val(7))
            mesh, m, rm, am, bm, _ = _transport_cs_problem(; Nc=4, Nz=4, nsteps=2, FT)
            _fill_smooth_tracer!(rm, m, mesh)
            ax = [copy(Adv._cs_flux_x_interior(a[1], mesh.Nc, mesh.Hp)) for a in am]
            by = [copy(Adv._cs_flux_y_interior(b[1], mesh.Nc, mesh.Hp)) for b in bm]
            cpu = _single_panel_reverse(rm[1], m[1], ax, by, mesh, order)
            gpu = _single_panel_reverse(CuArray(rm[1]), CuArray(m[1]),
                                        map(CuArray, ax), map(CuArray, by), mesh, order)
            tolerance = FT == Float64 ? 1e-10 : 2e-5
            for (actual, expected) in zip(gpu, cpu)
                @test actual isa CuArray
                @test isapprox(Array(actual), expected; rtol=tolerance, atol=tolerance * 1e-3)
            end
        end
    end
end
