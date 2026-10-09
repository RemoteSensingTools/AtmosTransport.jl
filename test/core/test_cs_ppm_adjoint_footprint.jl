#!/usr/bin/env julia

using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

const HAS_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

function _constant_cs_problem(; Nc=4, Nz=3, nsteps=2, FT=Float64)
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=3, FT=FT)
    N = mesh.Nc + 2mesh.Hp

    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25k + 0.01p)
        end
        m
    end
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir=0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir=0)

    panels_am_steps = [ntuple(_ -> zeros(FT, N + 1, N, Nz), 6) for _ in 1:nsteps]
    panels_bm_steps = [ntuple(_ -> zeros(FT, N, N + 1, Nz), 6) for _ in 1:nsteps]
    panels_cm_steps = [ntuple(_ -> zeros(FT, N, N, Nz + 1), 6) for _ in 1:nsteps]
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

function _transport_cs_problem(; Nc=4, Nz=3, nsteps=2, FT=Float64)
    mesh, panels_m, panels_rm, _, _, _ = _constant_cs_problem(; Nc, Nz, nsteps, FT)
    N = mesh.Nc + 2mesh.Hp
    Hp = mesh.Hp

    panels_am_steps = Vector{Any}(undef, nsteps)
    panels_bm_steps = Vector{Any}(undef, nsteps)
    panels_cm_steps = Vector{Any}(undef, nsteps)
    for step in 1:nsteps
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc + 1
                am[i, j, k] = FT(0.015) * sin(FT(0.2step + 0.3p + 0.7i + 0.4j + 0.2k))
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc + 1, i in Hp + 1:Hp + mesh.Nc
                bm[i, j, k] = FT(0.012) * cos(FT(0.3step + 0.5p + 0.4i + 0.6j + 0.1k))
            end
            bm
        end
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            for k in 2:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc
                cm[i, j, k] = -FT(0.010) * (one(FT) + FT(0.1) * sin(FT(i + j + k + p + step)))
            end
            cm
        end
    end
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

function _fill_smooth_tracer!(panels_rm, panels_m, mesh)
    FT = eltype(panels_rm[1])
    N = mesh.Nc + 2mesh.Hp
    Nz = size(panels_rm[1], 3)
    for p in 1:6
        for k in 1:Nz, j in 1:N, i in 1:N
            c = FT(0.08) + FT(0.01) * sin(FT(0.27i + 0.13j + 0.19k + 0.07p))
            panels_rm[p][i, j, k] = panels_m[p][i, j, k] * c
        end
    end
    Adv.fill_panel_halos!(panels_rm, mesh; dir=0)
    return panels_rm
end

function _dot_footprint(result, rates)
    total = 0.0
    for step in eachindex(result.footprints), p in 1:6
        total += sum(result.footprints[step][p] .* rates[step][p])
    end
    return total
end

function _dot_control_gradient(gradients, directions)
    total = 0.0
    for c in eachindex(gradients), p in 1:6
        total += sum(gradients[c][p] .* directions[c][p])
    end
    return total
end

function _scaled_rates(rates, scale)
    return [ntuple(p -> scale .* rates[step][p], 6) for step in eachindex(rates)]
end

function _shift_control(control, direction, scale)
    value = ntuple(p -> control.value[p] .+ scale .* direction[p], 6)
    return AT.CSSurfaceFluxControl(control.window, value;
        background=control.background,
        sigma=control.sigma)
end

function _cs_diffusion_context(mesh, prototype; kz=2.0, dz=50.0)
    FT = eltype(prototype)
    panels = ntuple(_ -> prototype, 6)
    ws = AT.DiffusionWorkspace(panels, mesh.Hp, 0)
    for p in 1:6
        fill!(ws.layer_thickness[p], FT(dz))
    end
    kz_field = AT.CubedSphereField(ntuple(_ -> AT.ConstantField{FT, 3}(FT(kz)), 6))
    op = AT.ImplicitVerticalDiffusion(; kz_field)
    return op, ws
end

function _cs_gchp_vdiff_diffusion_context(mesh, panels_m; dz=50.0)
    FT = eltype(panels_m[1])
    Nc = mesh.Nc
    Nz = size(panels_m[1], 3)
    ws = AT.DiffusionWorkspace(panels_m, mesh.Hp, 0)
    for p in 1:6
        fill!(ws.layer_thickness[p], FT(dz))
    end

    surface = AT.PBLSurfaceForcing(
        ntuple(p -> [FT(900 + 15p + 2i + j) for i in 1:Nc, j in 1:Nc], 6),
        ntuple(p -> [FT(0.22 + 0.01p + 0.001i) for i in 1:Nc, j in 1:Nc], 6),
        ntuple(p -> [FT(65 + 3p + i - 0.5j) for i in 1:Nc, j in 1:Nc], 6),
        ntuple(p -> [FT(285 + 0.2p + 0.05i - 0.03j) for i in 1:Nc, j in 1:Nc], 6),
    )
    vdiff = (
        u = ntuple(p -> [FT(4 + 0.3p + 0.02i - 0.01j + 0.05k)
                         for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6),
        v = ntuple(p -> [FT(-2 + 0.1p - 0.01i + 0.03j - 0.02k)
                         for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6),
        t = ntuple(p -> [FT(285 - 5k + 0.1p + 0.02i)
                         for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6),
        qv = ntuple(p -> [FT(0.006 / k + 0.0001p)
                          for i in 1:Nc, j in 1:Nc, k in 1:Nz], 6),
    )
    host_cache = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    kz_field = AT.LocalHoltslagBovilleKzField(host_cache)
    AT.refresh_local_holtslag_boville_kz_cache!(
        kz_field, surface, vdiff, panels_m, mesh.cell_areas;
        halo_width = mesh.Hp)
    op = AT.ImplicitVerticalDiffusion(; kz_field)
    return op, ws
end

function _cs_tm5_convection_context(mesh, panels_m)
    FT = eltype(panels_m[1])
    Nc = mesh.Nc
    Nz = size(panels_m[1], 3)
    entu = ntuple(_ -> begin
        e = similar(panels_m[1], FT, Nc, Nc, Nz)
        fill!(e, zero(FT))
        e[:, :, 2:min(4, Nz - 1)] .= FT(0.03)
        e
    end, 6)
    detu = ntuple(_ -> begin
        e = similar(panels_m[1], FT, Nc, Nc, Nz)
        fill!(e, zero(FT))
        e[:, :, 2:min(4, Nz - 1)] .= FT(0.02)
        e
    end, 6)
    entd = ntuple(_ -> begin
        e = similar(panels_m[1], FT, Nc, Nc, Nz)
        fill!(e, zero(FT))
        e[:, :, 3:min(4, Nz - 1)] .= FT(0.01)
        e
    end, 6)
    detd = ntuple(_ -> begin
        e = similar(panels_m[1], FT, Nc, Nc, Nz)
        fill!(e, zero(FT))
        e[:, :, 3:min(4, Nz - 1)] .= FT(0.005)
        e
    end, 6)
    forcing = AT.ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd))
    metrics = ntuple(_ -> begin
        a = similar(panels_m[1], FT, Nc, Nc)
        fill!(a, one(FT))
        a
    end, 6)
    ws = AT.TM5Workspace(panels_m; tile_columns=Nc * Nc, cell_metrics=metrics)
    return AT.TM5Convection(), forcing, ws
end

function _cs_cmfmc_convection_context(mesh, panels_m)
    FT = eltype(panels_m[1])
    Nc = mesh.Nc
    Nz = size(panels_m[1], 3)
    cmfmc = ntuple(_ -> begin
        c = similar(panels_m[1], FT, Nc, Nc, Nz + 1)
        fill!(c, zero(FT))
        Nz >= 2 && (c[:, :, 2] .= FT(0.012))
        Nz >= 3 && (c[:, :, 3] .= FT(0.020))
        Nz >= 4 && (c[:, :, 4] .= FT(0.015))
        Nz >= 5 && (c[:, :, 5] .= FT(0.008))
        c
    end, 6)
    dtrain = ntuple(_ -> begin
        d = similar(panels_m[1], FT, Nc, Nc, Nz)
        fill!(d, zero(FT))
        Nz >= 2 && (d[:, :, 2] .= FT(0.006))
        Nz >= 3 && (d[:, :, 3] .= FT(0.005))
        Nz >= 4 && (d[:, :, 4] .= FT(0.003))
        d
    end, 6)
    forcing = AT.ConvectionForcing(cmfmc, dtrain, nothing)
    metrics = ntuple(_ -> begin
        a = similar(panels_m[1], FT, Nc, Nc)
        fill!(a, one(FT))
        a
    end, 6)
    ws = AT.CMFMCWorkspace(panels_m; cell_metrics=metrics)
    return AT.CMFMCConvection(), forcing, ws
end

function _to_gpu_steps(steps)
    [ntuple(p -> CUDA.CuArray(step[p]), 6) for step in steps]
end

@testset "CS split-sweep surface-emission footprint prototype" begin
    @testset "unsupported convection variants fail before footprint recording" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=4, nsteps=1)
        objective = AT.CSLayerMeanObjective(1, 2, 2, 3)

        variants = (
            (AT.CMFMCConvection(clamp=true), "clamp = false"),
            (AT.TM5Convection(use_collab_lu=true), "full-column, unmerged"),
            (AT.TM5Convection(lmax_conv=3), "full-column, unmerged"),
            (AT.TM5Convection(n_merge=2), "full-column, unmerged"),
            (AT.CMFMCMatrixConvection(use_collab_lu=true), "full-column, unmerged"),
        )

        for (op, expected) in variants
            message = try
                AT.cs_surface_emission_footprint(
                    panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                    mesh, objective; convection_op=op)
                nothing
            catch err
                sprint(showerror, err)
            end
            @test message isa String
            @test occursin(expected, something(message, ""))
        end
    end

    @testset "No-transport analytic footprint" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=3, nsteps=2)
        dt = 2.5
        obj = AT.CSLayerMeanObjective(1, 2, 2, 3)

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt, epsilon=1e-6)

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
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt, epsilon=1e-6)
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

    @testset "Prototype 4D-Var cost gradient matches directional finite difference" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=3, nsteps=3)
        dt = 2.0
        zero_panel = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
        background = ntuple(_ -> fill(0.05, mesh.Nc, mesh.Nc), 6)
        controls = [
            AT.CSSurfaceFluxControl(
                AT.CSSurfaceFluxWindow(:first_step, 1), zero_panel;
                background=background, sigma=0.2),
            AT.CSSurfaceFluxControl(
                AT.CSSurfaceFluxWindow(:late_window, 2:3; normalize=true),
                zero_panel),
        ]
        observations = [
            AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.03, 0.4),
            AT.CSObservation(3, AT.CSColumnMeanObjective(1, 2, 2), 0.02, 0.3),
        ]

        result = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, controls;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt)

        @test result.cost ≈ result.observation_cost + result.background_cost
        @test haskey(result.gradient_by_name, :first_step)
        @test haskey(result.gradient_by_name, :late_window)

        directions = [
            ntuple(p -> [sin(0.11p + 0.17i - 0.13j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6),
            ntuple(p -> [cos(0.07p + 0.19i + 0.23j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6),
        ]
        eps_dir = 1e-6
        plus_controls = [_shift_control(controls[i], directions[i], eps_dir) for i in eachindex(controls)]
        minus_controls = [_shift_control(controls[i], directions[i], -eps_dir) for i in eachindex(controls)]
        j_plus = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, plus_controls;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt).cost
        j_minus = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, minus_controls;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt).cost
        fd = (j_plus - j_minus) / (2eps_dir)
        predicted = _dot_control_gradient(result.gradients, directions)
        @test predicted ≈ fd rtol=3e-5 atol=1e-10
    end

    @testset "Prototype 4D-Var optimizer reduces cost" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=3, nsteps=2)
        dt = 2.0
        values = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
        control = AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize=true),
            values)
        observations = [
            AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.05, 0.2),
        ]

        initial = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, control;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt)
        solve = AT.cs_surface_flux_4dvar_optimize(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, control;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt,
            iterations=4,
            initial_step=0.25)

        @test solve.iterations >= 1
        @test solve.cost_history[1] ≈ initial.cost
        @test solve.last.cost < initial.cost
        @test solve.cost_history[end] <= solve.cost_history[1]
        @test solve.gradient_norm_history[end] <= solve.gradient_norm_history[1]
    end

    @testset "Prototype 4D-Var gradient includes diffusion and TM5 convection" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=5, nsteps=2)
        dt = 20.0
        diffusion_op, diffusion_ws = _cs_diffusion_context(
            mesh, panels_rm[1]; kz=2.0, dz=50.0)
        convection_op, convection_forcing, convection_ws =
            _cs_tm5_convection_context(mesh, panels_m)
        values = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
        control = AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize=true),
            values)
        observations = [
            AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.01, 0.25),
        ]

        result = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, control;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt,
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)

        direction = ntuple(p -> [sin(0.17p + 0.29i - 0.31j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6)
        eps_dir = 1e-7
        j_plus = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, _shift_control(control, direction, eps_dir);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt,
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws).cost
        j_minus = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, _shift_control(control, direction, -eps_dir);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt,
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws).cost
        fd = (j_plus - j_minus) / (2eps_dir)
        predicted = _dot_control_gradient(result.gradients, [direction])
        @test predicted ≈ fd rtol=5e-5 atol=1e-10
    end

    @testset "Prototype 4D-Var gradient includes CMFMC convection" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=5, nsteps=2)
        dt = 20.0
        convection_op, convection_forcing, convection_ws =
            _cs_cmfmc_convection_context(mesh, panels_m)
        values = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
        control = AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize=true),
            values)
        observations = [
            AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.01, 0.25),
        ]

        result = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, control;
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)

        direction = ntuple(p -> [cos(0.23p + 0.19i - 0.11j) for i in 1:mesh.Nc, j in 1:mesh.Nc], 6)
        eps_dir = 1e-7
        j_plus = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, _shift_control(control, direction, eps_dir);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws).cost
        j_minus = AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, _shift_control(control, direction, -eps_dir);
            scheme=AT.PPMScheme(AT.NoLimiter()), dt=dt,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws).cost
        fd = (j_plus - j_minus) / (2eps_dir)
        predicted = _dot_control_gradient(result.gradients, [direction])
        @test predicted ≈ fd rtol=5e-5 atol=1e-10
    end

    @testset "Generated footprint matches directional finite difference" begin
        for scheme in (AT.UpwindScheme(), AT.SlopesScheme(AT.NoLimiter()), AT.PPMScheme(AT.NoLimiter()))
            @testset "$(typeof(scheme))" begin
                mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
                    _transport_cs_problem(Nc=3, Nz=6, nsteps=2)
                dt = 1.5
                obj = AT.CSColumnMeanObjective(1, 2, 2)

                result = AT.cs_surface_emission_footprint(
                    panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
                    scheme=scheme, dt=dt, epsilon=1e-6)

                rates = [ntuple(6) do p
                    [sin(0.31step + 0.17p + 0.23i - 0.19j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
                end for step in 1:2]

                eps_dir = 2e-6
                j_plus = AT.run_cs_footprint_forward(
                    panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
                    scheme=scheme, dt=dt,
                    emission_rates=_scaled_rates(rates, eps_dir))
                j_minus = AT.run_cs_footprint_forward(
                    panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
                    scheme=scheme, dt=dt,
                    emission_rates=_scaled_rates(rates, -eps_dir))

                fd = (j_plus - j_minus) / (2eps_dir)
                predicted = _dot_footprint(result, rates)
                @test predicted ≈ fd rtol=2e-5 atol=1e-10
            end
        end
    end

    @testset "Footprint uses the production palindrome CFL schedule" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _transport_cs_problem(Nc=4, Nz=3, nsteps=1)
        for flux_steps in (panels_am, panels_bm, panels_cm),
            panel in flux_steps[1]
            panel .*= 40
        end

        flux_scale = 1.5
        n_pal = Adv._cs_static_palindrome_subcycle_count(
            panels_am[1], panels_bm[1], panels_cm[1], panels_m,
            mesh.Nc, mesh.Hp, size(panels_m[1], 3), 0.95;
            flux_scale)
        @test n_pal > 1

        scheme = AT.UpwindScheme()
        objective = AT.CSColumnMeanObjective(1, 2, 2)
        rates = [ntuple(6) do p
            [sin(0.17p + 0.31i - 0.23j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
        end]
        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, objective; scheme, dt=1.0, flux_scale)

        epsilon = 2e-6
        j_plus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, objective;
            scheme, dt=1.0, flux_scale,
            emission_rates=_scaled_rates(rates, epsilon))
        j_minus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, objective;
            scheme, dt=1.0, flux_scale,
            emission_rates=_scaled_rates(rates, -epsilon))

        finite_difference = (j_plus - j_minus) / (2epsilon)
        @test _dot_footprint(result, rates) ≈ finite_difference rtol=3e-5 atol=1e-10
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

    @testset "Limited PPM footprint replays nonlinear branch tape ($(nameof(typeof(limiter))))" for limiter in
            (AT.MonotoneLimiter(), AT.CW84Limiter())
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _transport_cs_problem(Nc=4, Nz=6, nsteps=2)
        _fill_smooth_tracer!(panels_rm, panels_m, mesh)
        dt = 1.5
        scheme = AT.PPMScheme(limiter)
        obj = AT.CSColumnMeanObjective(1, 2, 2)

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt)

        rates = [ntuple(6) do p
            [sin(0.31step + 0.17p + 0.23i - 0.19j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
        end for step in 1:2]

        eps_dir = 1e-6
        j_plus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            emission_rates=_scaled_rates(rates, eps_dir))
        j_minus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            emission_rates=_scaled_rates(rates, -eps_dir))

        fd = (j_plus - j_minus) / (2eps_dir)
        predicted = _dot_footprint(result, rates)
        @test predicted ≈ fd rtol=3e-5 atol=1e-10
    end

    @testset "Limited PPM footprint follows nonzero base emissions" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _transport_cs_problem(Nc=4, Nz=6, nsteps=2)
        dt = 1.5
        scheme = AT.PPMScheme()
        obj = AT.CSColumnMeanObjective(1, 2, 2)

        base_rates = [ntuple(6) do p
            [0.03 + 0.005sin(0.11step + 0.17p + 0.23i + 0.19j)
             for i in 1:mesh.Nc, j in 1:mesh.Nc]
        end for step in 1:2]
        direction = [ntuple(6) do p
            [cos(0.29step + 0.13p - 0.31i + 0.07j)
             for i in 1:mesh.Nc, j in 1:mesh.Nc]
        end for step in 1:2]

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt, base_emission_rates=base_rates)

        eps_dir = 1e-6
        plus_rates = [ntuple(p -> base_rates[step][p] .+
                                  eps_dir .* direction[step][p], 6)
                      for step in 1:2]
        minus_rates = [ntuple(p -> base_rates[step][p] .-
                                   eps_dir .* direction[step][p], 6)
                       for step in 1:2]
        j_plus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt, emission_rates=plus_rates)
        j_minus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt, emission_rates=minus_rates)

        fd = (j_plus - j_minus) / (2eps_dir)
        predicted = _dot_footprint(result, direction)
        @test predicted ≈ fd rtol=3e-5 atol=1e-10
    end

    @testset "Limited PPM branch tape includes implicit vertical diffusion" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _transport_cs_problem(Nc=4, Nz=6, nsteps=2)
        _fill_smooth_tracer!(panels_rm, panels_m, mesh)
        dt = 20.0
        scheme = AT.PPMScheme()
        diffusion_op, diffusion_ws = _cs_diffusion_context(
            mesh, panels_rm[1]; kz=2.0, dz=50.0)
        obj = AT.CSLayerMeanObjective(1, 2, 2, 4)

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws)

        rates = [ntuple(6) do p
            [sin(0.21step + 0.13p + 0.37i + 0.11j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
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
    end

    @testset "Generated footprint includes implicit vertical diffusion" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=5, nsteps=2)
        dt = 300.0
        scheme = AT.PPMScheme(AT.NoLimiter())
        diffusion_op, diffusion_ws = _cs_diffusion_context(
            mesh, panels_rm[1]; kz=2.0, dz=50.0)
        obj = AT.CSLayerMeanObjective(1, 2, 2, 4)

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            diffusion_op=diffusion_op,
            diffusion_workspace=diffusion_ws)

        @test result.footprints[2][1][2, 2] > 0

        rates = [ntuple(6) do p
            [cos(0.21step + 0.13p + 0.37i + 0.11j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
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

    @testset "Generated footprint includes TM5 convection transpose" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=5, nsteps=2)
        dt = 20.0
        scheme = AT.PPMScheme(AT.NoLimiter())
        convection_op, convection_forcing, convection_ws =
            _cs_tm5_convection_context(mesh, panels_m)
        obj = AT.CSLayerMeanObjective(1, 2, 2, 3)

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)

        @test sum(abs, result.footprints[2][1]) > 0

        rates = [ntuple(6) do p
            [sin(0.27step + 0.09p + 0.31i - 0.07j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
        end for step in 1:2]

        eps_dir = 1e-7
        j_plus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            emission_rates=_scaled_rates(rates, eps_dir),
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)
        j_minus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            emission_rates=_scaled_rates(rates, -eps_dir),
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)

        fd = (j_plus - j_minus) / (2eps_dir)
        predicted = _dot_footprint(result, rates)
        @test predicted ≈ fd rtol=3e-5 atol=1e-10
    end

    @testset "Generated footprint includes CMFMC convection transpose" begin
        mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
            _constant_cs_problem(Nc=3, Nz=5, nsteps=2)
        dt = 20.0
        scheme = AT.PPMScheme(AT.NoLimiter())
        convection_op, convection_forcing, convection_ws =
            _cs_cmfmc_convection_context(mesh, panels_m)
        obj = AT.CSLayerMeanObjective(1, 2, 2, 3)

        result = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)

        @test sum(abs, result.footprints[2][1]) > 0

        rates = [ntuple(6) do p
            [cos(0.19step + 0.15p + 0.21i - 0.09j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
        end for step in 1:2]

        eps_dir = 1e-7
        j_plus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            emission_rates=_scaled_rates(rates, eps_dir),
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)
        j_minus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
            scheme=scheme, dt=dt,
            emission_rates=_scaled_rates(rates, -eps_dir),
            convection_op=convection_op,
            convection_forcing=convection_forcing,
            convection_workspace=convection_ws)

        fd = (j_plus - j_minus) / (2eps_dir)
        predicted = _dot_footprint(result, rates)
        @test predicted ≈ fd rtol=5e-5 atol=1e-10
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

    if HAS_GPU
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
                iterations=1,
                initial_step=0.1f0)
            @test gpu_solve.controls[1].value[1] isa CUDA.CuArray
        end
    end
end
