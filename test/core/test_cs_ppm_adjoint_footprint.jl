include(joinpath(@__DIR__, "..", "helpers", "cs_footprint_problem.jl"))

@testset "CS split-sweep surface-emission footprint prototype" begin
    @testset "No-transport analytic footprint" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=3, nsteps=2)
        dt = 2.5
        obj = AT.CSLayerMeanObjective(1, 2, 2, 3)

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt)

        @test result.lag_steps == [1, 0]
        expected = dt / panels_m[1][mesh.Hp + 2, mesh.Hp + 2, 3]
        for step in 1:2
            @test result.footprints[step][1][2, 2] ≈ expected rtol=1e-8
            leakage = sum(abs, result.footprints[step][1]) - abs(result.footprints[step][1][2, 2])
            for p in 2:6
                leakage += sum(abs, result.footprints[step][p])
            end
            @test leakage < 1e-10
        end

        col = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh,
            AT.CSColumnMeanObjective(1, 2, 2);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt)
        column_mass = sum(panels_m[1][mesh.Hp + 2, mesh.Hp + 2, :])
        @test col.footprints[2][1][2, 2] ≈ dt / column_mass rtol=1e-8
    end

    @testset "Layer and column Jacobians aggregate user time windows" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=3, nsteps=3)
        dt = 2.0
        objectives = [
            AT.CSLayerMeanObjective(1, 2, 2, 3),
            AT.CSColumnMeanObjective(1, 2, 2),
        ]
        windows = [
            AT.CSSurfaceFluxWindow(:step2, 2),
            AT.CSSurfaceFluxWindow(:all_steps, 1:3),
            AT.CSSurfaceFluxWindow(:mean_first_two, 1:2; normalize=true),
        ]
        jac = AT.cs_surface_flux_jacobian(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, objectives, windows;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt)

        @test size(jac.footprints) == (2, 3)
        @test jac.windows[2].name == :all_steps
        per_step = jac.per_step_results[1]
        for p in 1:6
            @test jac.footprints[1, 1][p] ≈ per_step.footprints[2][p]
            @test jac.footprints[1, 2][p] ≈
                  per_step.footprints[1][p] .+
                  per_step.footprints[2][p] .+
                  per_step.footprints[3][p]
            @test jac.footprints[1, 3][p] ≈
                  0.5 .* (per_step.footprints[1][p] .+
                          per_step.footprints[2][p])
        end
    end

    @testset "4D-Var directional finite differences: $physics" for physics in (:none, :tm5, :cmfmc)
        coupled = physics !== :none
        nsteps, Nz = coupled ? (2, 5) : (3, 3)
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(; Nc = 3, Nz, nsteps)
        dt = coupled ? 20.0 : 2.0
        zero_panel = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
        controls, observations, directions = if coupled
            cs = [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize = true), zero_panel)]
            obs = [AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.01, 0.25)]
            d = physics === :tm5 ?
                ntuple(p -> [sin(0.17p + 0.29i - 0.31j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6) :
                ntuple(p -> [cos(0.23p + 0.19i - 0.11j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6)
            (cs, obs, [d])
        else
            background = ntuple(_ -> fill(0.05, mesh.Nc, mesh.Nc), 6)
            cs = [AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:first_step, 1), zero_panel;
                                          background, sigma = 0.2),
                  AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:late_window, 2:3; normalize = true), zero_panel)]
            obs = [AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.03, 0.4),
                   AT.CSObservation(3, AT.CSColumnMeanObjective(1, 2, 2), 0.02, 0.3)]
            ds = [ntuple(p -> [sin(0.11p + 0.17i - 0.13j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6),
                  ntuple(p -> [cos(0.07p + 0.19i + 0.23j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6)]
            (cs, obs, ds)
        end
        kwargs = if physics === :tm5
            diffusion_op, diffusion_workspace = _cs_diffusion_context(mesh, panels_rm[1]; kz = 2.0, dz = 50.0)
            convection_op, convection_forcing, convection_workspace = _cs_tm5_convection_context(mesh, panels_m)
            (; diffusion_op, diffusion_workspace, convection_op, convection_forcing, convection_workspace)
        elseif physics === :cmfmc
            convection_op, convection_forcing, convection_workspace = _cs_cmfmc_convection_context(mesh, panels_m)
            (; convection_op, convection_forcing, convection_workspace)
        else
            (;)
        end
        evaluate(cs) = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, observations, cs;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt, kwargs...)
        result = evaluate(controls)
        if !coupled
            @test result.cost ≈ result.observation_cost + result.background_cost
            @test haskey(result.gradient_by_name, :first_step)
            @test haskey(result.gradient_by_name, :late_window)
        end
        h = coupled ? 1e-7 : 1e-6
        shifted(a) = [_shift_control(c, d, a) for (c, d) in zip(controls, directions)]
        fd = (evaluate(shifted(h)).cost - evaluate(shifted(-h)).cost) / (2h)
        @test _dot_control_gradient(result.gradients, directions) ≈ fd rtol = (coupled ? 5e-5 : 3e-5) atol = 1e-10
    end

    @testset "Footprint directional finite differences: $label" for (label, scheme) in (
            (:upwind, AT.UpwindScheme()), (:slopes, AT.SlopesScheme(AT.NoLimiter())),
            (:linear_ppm, AT.PPMScheme(AT.NoLimiter())), (:cfl, AT.UpwindScheme()),
            (:monotone, AT.PPMScheme(AT.MonotoneLimiter())), (:cw84, AT.PPMScheme(AT.CW84Limiter())))
        cfl = label === :cfl
        limited = label in (:monotone, :cw84)
        Nc, Nz, nsteps = cfl ? (4, 3, 1) : (limited ? 4 : 3, 6, 2)
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _transport_cs_problem(; Nc, Nz, nsteps)
        limited && _fill_smooth_tracer!(panels_rm, panels_m, mesh)
        dt, flux_scale = cfl ? (1.0, 1.5) : (1.5, 1.0)
        if cfl
            # ×25 needs several palindrome subcycles while retaining valid masses.
            for flux_steps in (panels_am, panels_bm, panels_cm), panel in flux_steps[1]
                panel .*= 25
            end
            n_pal = Adv._cs_static_palindrome_subcycle_count(
                panels_am[1], panels_bm[1], panels_cm[1], panels_m,
                mesh.Nc, mesh.Hp, Nz, 0.95; flux_scale)
            @test n_pal > 1
        end
        objective = AT.CSColumnMeanObjective(1, 2, 2)
        rates = [ntuple(p -> [cfl ? sin(0.17p + 0.31i - 0.23j) :
                                      sin(0.31step + 0.17p + 0.23i - 0.19j)
                              for i in 1:mesh.Nc, j in 1:mesh.Nc], 6) for step in 1:nsteps]
        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, objective;
            scheme, dt, flux_scale)
        h = limited ? 1e-6 : 2e-6
        forward(a) = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, objective;
            scheme, dt, flux_scale, emission_rates = _scaled_rates(rates, a))
        fd = (forward(h) - forward(-h)) / (2h)
        @test _dot_footprint(result, rates) ≈ fd rtol = (limited || cfl ? 3e-5 : 2e-5) atol = 1e-10
    end

    @testset "4D-Var forward and gradient share non-unit flux_scale" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _transport_cs_problem(Nc=3, Nz=3, nsteps=2)
        values = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
        control = AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize=true), values)
        observations = [
            AT.CSObservation(2, AT.CSColumnMeanObjective(1, 2, 2), 0.02, 0.3),
        ]
        flux_scale = 2.0
        result = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, control;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=1.0, flux_scale)

        direction = ntuple(p ->
            [cos(0.19p + 0.13i - 0.11j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6)
        epsilon = 1e-6
        j_plus = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, _shift_control(control, direction, epsilon);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=1.0, flux_scale).cost
        j_minus = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, _shift_control(control, direction, -epsilon);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=1.0, flux_scale).cost

        finite_difference = (j_plus - j_minus) / (2epsilon)
        @test _dot_control_gradient(result.gradients, [direction]) ≈
              finite_difference rtol=5e-5 atol=1e-10
    end

    @testset "Generated footprint includes GCHP VDIFF local-Kz diffusion adjoint" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=5, nsteps=2)
        dt = 300.0
        scheme = AT.PPMScheme(AT.NoLimiter())
        diffusion_op, diffusion_ws = _cs_gchp_vdiff_diffusion_context(
            mesh, panels_m; dz=55.0)
        obj = AT.CSLayerMeanObjective(1, 2, 2, 4)

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws)

        @test result.footprints[2][1][2, 2] > 0

        rates = [ntuple(6) do p
            [sin(0.31step + 0.17p + 0.23i - 0.19j)
             for i in 1:mesh.Nc, j in 1:mesh.Nc]
        end for step in 1:2]

        eps_dir = 1e-7
        j_plus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            emission_rates=_scaled_rates(rates, eps_dir),
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws)
        j_minus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            emission_rates=_scaled_rates(rates, -eps_dir),
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws)

        fd = (j_plus - j_minus) / (2eps_dir)
        predicted = _dot_footprint(result, rates)
        @test predicted ≈ fd rtol=3e-5 atol=1e-10

        stale_cache = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc, size(panels_m[1], 3)), 6)
        stale_op = AT.ImplicitVerticalDiffusion(;
            kz_field = AT.LocalHoltslagBovilleKzField(stale_cache))
        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            diffusion_op=stale_op,
            diffusion_workspace=diffusion_ws)
    end

    @testset "Vector mass panels preserve seeded footprint bits ($FT)" for FT in (Float32, Float64)
        mesh, panels_m, _, panels_am, panels_bm, panels_cm =
            _transport_cs_problem(Nc=3, Nz=3, nsteps=2, FT=FT)
        masses = collect(panels_m)
        @test masses isa Vector{Array{FT, 3}}
        seed = ntuple(p -> [FT(sin(i + 2j + 3k + p))
                            for i in axes(masses[p], 1), j in axes(masses[p], 2),
                                k in axes(masses[p], 3)], 6)
        # The former panel helper indexed any container and allocated/filled
        # each panel before advancing. Supply that base through the public API.
        old_zero = ntuple(6) do p
            a = similar(masses[p])
            fill!(a, zero(eltype(a)))
            a
        end
        kwargs = (; scheme=AT.PPMScheme(AT.NoLimiter()), dt=FT(1))
        reference = AT.cs_surface_emission_footprint_from_seed(
            seed, masses, panels_am, panels_bm, panels_cm, mesh;
            kwargs..., base_panels_rm0=old_zero)
        for container in (masses, panels_m)
            result = AT.cs_surface_emission_footprint_from_seed(
                seed, container, panels_am, panels_bm, panels_cm, mesh; kwargs...)
            @test result.lag_steps == reference.lag_steps
            @test result.dt === reference.dt
            @test length(result.footprints) == length(reference.footprints)
            for p in 1:6
                @test reinterpret(UInt8, vec(result.initial_adjoint[p])) ==
                      reinterpret(UInt8, vec(reference.initial_adjoint[p]))
                for step in eachindex(reference.footprints)
                    @test reinterpret(UInt8, vec(result.footprints[step][p])) ==
                          reinterpret(UInt8, vec(reference.footprints[step][p]))
                end
            end
        end
    end

    @testset "Explicit final adjoint seed matches built-in objective seed" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=3, nsteps=1)
        dt = 1.0
        obj_result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh,
            AT.CSLayerMeanObjective(1, 2, 2, 3);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt)

        seed = ntuple(p -> zeros(Float64, size(panels_m[p])), 6)
        seed[1][mesh.Hp + 2, mesh.Hp + 2, 3] =
            1.0 / panels_m[1][mesh.Hp + 2, mesh.Hp + 2, 3]
        seed_result = AT.cs_surface_emission_footprint_from_seed(
            seed, panels_m, panels_am, panels_bm, panels_cm, mesh;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt)

        for p in 1:6
            @test seed_result.footprints[1][p] ≈ obj_result.footprints[1][p]
        end

        @test_throws ArgumentError AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh,
            AT.CSSeedObjective();
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt)
    end

end
