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

    function _to_gpu_steps(steps)
        [ntuple(p -> CUDA.CuArray(step[p]), 6) for step in steps]
    end

    @testset "GPU execution keeps footprint arrays on device" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=3, nsteps=1, FT=Float32)
        panels_m_g = ntuple(p -> CUDA.CuArray(panels_m[p]), 6)
        panels_rm_g = ntuple(p -> CUDA.CuArray(panels_rm[p]), 6)
        panels_am_g = _to_gpu_steps(panels_am)
        panels_bm_g = _to_gpu_steps(panels_bm)
        panels_cm_g = _to_gpu_steps(panels_cm)

        result = AT.cs_surface_emission_footprint(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, AT.CSLayerMeanObjective(1, 2, 2, 3);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(1))

        @test result.footprints[1][1] isa CUDA.CuArray
        fp = Array(result.footprints[1][1])
        expected = Float32(1) / Array(panels_m_g[1])[mesh.Hp + 2, mesh.Hp + 2, 3]
        @test fp[2, 2] ≈ expected rtol=1f-5
        pinned_result = AT.cs_surface_emission_footprint(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, AT.CSLayerMeanObjective(1, 2, 2, 3);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(1),
            tape_storage=:pinned_host)
        @test pinned_result.footprints[1][1] isa CUDA.CuArray
        @test Array(pinned_result.footprints[1][1]) ≈ fp rtol=1f-6

        # Plan 26 Phase A.1 — mmap tape on GPU should reach the same
        # device-side result. We compare against the pinned-host result
        # (the closest sibling) for bit-exact parity, since both paths
        # round-trip through a host snapshot of the recorded panels.
        mmap_result = AT.cs_surface_emission_footprint(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, AT.CSLayerMeanObjective(1, 2, 2, 3);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(1),
            tape_storage=:mmap)
        @test mmap_result.footprints[1][1] isa CUDA.CuArray
        @test Array(mmap_result.footprints[1][1]) ==
              Array(pinned_result.footprints[1][1])

        diffusion_op, diffusion_ws = _cs_diffusion_context(
            mesh, panels_rm_g[1]; kz=2.0f0, dz=50.0f0)
        diff_result = AT.cs_surface_emission_footprint(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, AT.CSLayerMeanObjective(1, 2, 2, 2);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(300),
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws)
        @test diff_result.footprints[1][1] isa CUDA.CuArray
        @test Array(diff_result.footprints[1][1])[2, 2] > 0
        pinned_diff_result = AT.cs_surface_emission_footprint(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, AT.CSLayerMeanObjective(1, 2, 2, 2);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(300),
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws,
            tape_storage=:pinned_host)
        @test pinned_diff_result.footprints[1][1] isa CUDA.CuArray
        @test Array(pinned_diff_result.footprints[1][1]) ≈
              Array(diff_result.footprints[1][1]) rtol=1f-6

        mmap_diff_result = AT.cs_surface_emission_footprint(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, AT.CSLayerMeanObjective(1, 2, 2, 2);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(300),
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws,
            tape_storage=:mmap)
        @test mmap_diff_result.footprints[1][1] isa CUDA.CuArray
        @test Array(mmap_diff_result.footprints[1][1]) ==
              Array(pinned_diff_result.footprints[1][1])

        convection_op, convection_forcing, convection_ws =
            _cs_tm5_convection_context(mesh, panels_m_g)
        conv_result = AT.cs_surface_emission_footprint(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, AT.CSLayerMeanObjective(1, 2, 2, 3);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(20),
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)
        @test conv_result.footprints[1][1] isa CUDA.CuArray
        @test sum(abs, Array(conv_result.footprints[1][1])) > 0

        cmfmc_op, cmfmc_forcing, cmfmc_ws =
            _cs_cmfmc_convection_context(mesh, panels_m_g)
        cmfmc_result = AT.cs_surface_emission_footprint(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, AT.CSLayerMeanObjective(1, 2, 2, 2);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(20),
            convection_op=cmfmc_op,
            convection_forcing=cmfmc_forcing,
            convection_workspace=cmfmc_ws)
        @test cmfmc_result.footprints[1][1] isa CUDA.CuArray
        @test sum(abs, Array(cmfmc_result.footprints[1][1])) > 0

        gpu_values = ntuple(_ -> CUDA.zeros(Float32, mesh.Nc, mesh.Nc), 6)
        gpu_control = AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:step1, 1),
            gpu_values)
        gpu_observations = [
            AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 3),
                             0.01f0, 0.2f0),
        ]
        gpu_4dvar = AT.cs_surface_flux_4dvar(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, gpu_observations, gpu_control;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(1))
        @test isfinite(gpu_4dvar.cost)
        @test gpu_4dvar.gradients[1][1] isa CUDA.CuArray

        gpu_solve = AT.cs_surface_flux_4dvar_optimize(
            panels_rm_g, panels_m_g, panels_am_g, panels_bm_g, panels_cm_g,
            mesh, gpu_observations, gpu_control;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=Float32(1),
            optimizer=AT.CSGradientDescent(1, 0.1f0, sqrt(eps(Float32)),
                                            0.5f0, 0.0f0, true, false))
        @test gpu_solve.controls[1].value[1] isa CUDA.CuArray
    end

    @testset "GPU host tape storage keeps limited-PPM m and rm reads apart (#57)" begin
        # `:pinned_host` and `:mmap` read staged states back through
        # device caches; a limited-PPM sweep record reads its air mass and
        # tracer mass together, so one shared cache handed the reverse
        # kernel `m === rm` (footprint error ~0.8). NoLimiter tapes no rm
        # and could not see it.
        FT = Float64
        mesh = AT.CubedSphereMesh(Nc = 4, Hp = 3, FT = FT)
        Nc, Hp, Nz, nsteps = mesh.Nc, mesh.Hp, 5, 4
        N = Nc + 2Hp
        m = ntuple(p -> [FT(2.0 + 0.25k + 0.01p + 0.03sin(0.7i + 0.3j + p))
                         for i in 1:N, j in 1:N, k in 1:Nz], 6)
        rm = ntuple(p -> [m[p][i, j, k] * FT(0.08 + 0.01sin(0.27i + 0.13j + 0.19k + 0.07p))
                          for i in 1:N, j in 1:N, k in 1:Nz], 6)
        Adv.fill_panel_halos!(m, mesh; dir = 0)
        Adv.fill_panel_halos!(rm, mesh; dir = 0)
        face(dims, range_i, range_j, f) = ntuple(p -> begin
            a = zeros(FT, dims...)
            for k in 1:Nz, j in range_j, i in range_i
                a[i, j, k] = f(p, i, j, k)
            end
            CUDA.CuArray(a)
        end, 6)
        am = [face((N + 1, N, Nz), Hp+1:Hp+Nc+1, Hp+1:Hp+Nc,
                   (p, i, j, k) -> 0.05sin(0.2s + 0.3p + 0.7i + 0.4j + 0.2k)) for s in 1:nsteps]
        bm = [face((N, N + 1, Nz), Hp+1:Hp+Nc, Hp+1:Hp+Nc+1,
                   (p, i, j, k) -> 0.04cos(0.3s + 0.5p + 0.4i + 0.6j + 0.1k)) for s in 1:nsteps]
        cm = [ntuple(p -> begin
                  c = zeros(FT, N, N, Nz + 1)
                  for k in 2:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
                      c[i, j, k] = -0.03(1 + 0.3sin(i + j + k + p + s))
                  end
                  CUDA.CuArray(c)
              end, 6) for s in 1:nsteps]
        m_g = ntuple(p -> CUDA.CuArray(m[p]), 6)
        rm_g = ntuple(p -> CUDA.CuArray(rm[p]), 6)
        objective = AT.CSLayerMeanObjective(2, 2, 3, Nz)
        maxrel(a, b) =
            maximum(maximum(abs.(Array(a.footprints[s][p]) .- Array(b.footprints[s][p])))
                    for s in eachindex(a.footprints), p in 1:6) /
            maximum(maximum(abs.(Array(b.footprints[s][p])))
                    for s in eachindex(b.footprints), p in 1:6)
        for limiter in (AT.MonotoneLimiter(), AT.CW84Limiter()),
            checkpoint in (AT.FullCheckpoint(), AT.StrideCheckpoint(2))
            footprint(storage) = AT.cs_surface_emission_footprint(
                rm_g, m_g, am, bm, cm, mesh, objective;
                scheme = AT.PPMScheme(limiter), dt = FT(30),
                tape_storage = storage, checkpoint)
            reference = footprint(:device)
            @test reference.footprints[1][1] isa CUDA.CuArray
            @test maxrel(footprint(:pinned_host), reference) <= 1e-12
            @test maxrel(footprint(:mmap), reference) <= 1e-12
        end
    end

    # A smooth nonzero base avoids limiter switches in centered finite
    # differences. The corner observation exercises panel-edge transport.
    # Float32 is compared with its CPU gradient; Float64 also checks the
    # gradient against perturbations of the GPU forward model.
    # The FV3 vertical sweeps (OF-94) run on 8 layers so that the interior
    # layers' Huynh constraint and positivity limiter are exercised.
    fv3 = AT.PPMScheme(; vertical = AT.Operators.Advection.FV3ScalarProfile())
    fv3_signed = AT.PPMScheme(; vertical = AT.Operators.Advection.FV3ScalarProfile(; positive_definite = false))
    @testset "CUDA transporting footprint gradients" begin
        for FT in (Float64, Float32), (scheme, Nz) in (
                (AT.PPMScheme(AT.NoLimiter()), 4), (AT.PPMScheme(), 4), (AT.PPMScheme(AT.CW84Limiter()), 4),
                (AT.LinRoodPPMScheme(5), 4), (AT.LinRoodPPMScheme(7), 4),
                (AT.LinRoodPPMScheme(5; vertical = fv3), 8), (AT.LinRoodPPMScheme(7; vertical = fv3_signed), 8))
            @testset "$FT $scheme" begin
                mesh, m, rm, am, bm, cm = _transport_cs_problem(; Nc=4, Nz, nsteps=3, FT)
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
                    # The same identity holds to <= 3e-11 for every scheme here
                    # on CPU (Lin-Rood was 5.1e-8 before the issue #10 fix);
                    # 1e-8 leaves room for GPU rounding.
                    @test isapprox(predicted, fd; rtol=1e-8)
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
