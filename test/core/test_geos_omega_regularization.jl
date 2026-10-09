#!/usr/bin/env julia

using Test

import AtmosTransport
using .AtmosTransport.Preprocessing: build_target_geometry

const P = AtmosTransport.Preprocessing

@testset "GEOS OMEGA regularization" begin
    grid = build_target_geometry(
        Dict{String, Any}(
            "type" => "cubed_sphere",
            "Nc" => 4,
            "panel_convention" => "geos_native",
        ),
        Float64,
    )
    Nc = grid.Nc
    nc = 6 * Nc * Nc

    @testset "pressure taper and validation" begin
        bounds = (50.0, 80.0, 300.0, 350.0)
        @test P._omega_pressure_weight(40.0, bounds) == 0.0
        @test P._omega_pressure_weight(50.0, bounds) == 0.0
        @test P._omega_pressure_weight(80.0, bounds) == 1.0
        @test P._omega_pressure_weight(200.0, bounds) == 1.0
        @test P._omega_pressure_weight(350.0, bounds) == 0.0
        @test P._omega_pressure_weight(500.0, bounds) == 0.0
        @test_throws ArgumentError P._validate_omega_regularization(
            P.OmegaRegularization(pressure_taper_hpa = (80.0, 50.0, 300.0, 350.0)))
        @test_throws ArgumentError P._validate_omega_regularization(
            P.OmegaRegularization(max_relative_flux_correction = 1.1))
        @test_throws ArgumentError P._validate_omega_regularization(
            P.OmegaRegularization(smoothing_fraction = 0.126))
        @test_throws ArgumentError P._validate_omega_regularization(
            P.OmegaRegularization(max_bottom_flux_correction = 1.1))
    end

    @testset "seam-aware smoother is conservative" begin
        field = [sin(0.73 * c) + (isodd(c) ? 1.0 : -1.0) for c in 1:nc]
        initial = copy(field)
        scratch = similar(field)
        smoothed = P._smooth_cs_graph_conservative!(field, scratch,
                                                     grid.face_table, 3, 0.10)
        @test sum(smoothed) ≈ sum(initial) atol = 1e-12
        @test sum(abs2, smoothed .- sum(smoothed) / nc) <
              sum(abs2, initial .- sum(initial) / nc)
    end

    @testset "target preserves endpoints and resolved scales outside UTLS" begin
        Nz = 5
        native_cm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz + 1), 6)
        omega_cm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz + 1), 6)
        # Alternating grid-scale OMEGA discrepancy at the 100 and 250 hPa
        # interfaces, both inside the fully active pressure window.
        for p in 1:6, j in 1:Nc, i in 1:Nc
            checker = isodd(i + j + p) ? 1.0 : -1.0
            omega_cm[p][i, j, 3] = 2.0 * checker
            omega_cm[p][i, j, 4] = -checker
        end
        omega_vdiv = ntuple(p ->
            omega_cm[p][:, :, 1:Nz] .- omega_cm[p][:, :, 2:(Nz + 1)], 6)

        # Interface pressures: 0, 40, 100, 250, 350, 1000 hPa.
        dp_hpa = (40.0, 60.0, 150.0, 100.0, 650.0)
        m = ntuple(p -> begin
            panel = zeros(Float64, Nc, Nc, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                panel[i, j, k] = dp_hpa[k] * 100.0 *
                                 grid.mesh.cell_areas[i, j] / AtmosTransport.Parameters.STANDARD_GRAVITY
            end
            panel
        end, 6)
        target = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), 6)
        scratch = P.OmegaRegularizationScratch(
            ntuple(_ -> zeros(Float64, Nc, Nc, Nz + 1), 6),
            zeros(nc), zeros(nc), zeros(nc), zeros(nc),
            falses(Nz),
        )
        options = P.OmegaRegularization()
        P._regularize_omega_target!(target, native_cm, omega_vdiv, m, m, grid,
                                    AtmosTransport.Parameters.STANDARD_GRAVITY, 1.0, options, scratch)
        @test scratch.active_levels == Bool[false, true, true, true, false]

        for p in 1:6
            # Telescoping target retains the native top and surface endpoints.
            @test maximum(abs, dropdims(sum(target[p]; dims = 3); dims = 3)) < 1e-12
            implied = zeros(Float64, Nc, Nc, Nz + 1)
            for k in 1:Nz
                implied[:, :, k + 1] .= implied[:, :, k] .- target[p][:, :, k]
            end
            @test implied[:, :, 1] ≈ native_cm[p][:, :, 1] atol = 1e-12
            @test implied[:, :, 2] ≈ native_cm[p][:, :, 2] atol = 1e-12 # 40 hPa
            @test implied[:, :, 5] ≈ native_cm[p][:, :, 5] atol = 1e-12 # 350 hPa
            @test implied[:, :, 6] ≈ native_cm[p][:, :, 6] atol = 1e-12
        end
        @test any(panel -> maximum(abs, panel) > 0.0, target)
    end

    @testset "per-level horizontal correction cap" begin
        Nz = 2
        am = ntuple(_ -> ones(Float64, Nc + 1, Nc, Nz), 6)
        bm = ntuple(_ -> ones(Float64, Nc, Nc + 1, Nz), 6)
        dm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), 6)
        vdiv = ntuple(p -> begin
            a = zeros(Float64, Nc, Nc, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                a[i, j, k] = 100.0 * sin(0.91 * (i + Nc * (j - 1) +
                                                     Nc * Nc * (p - 1) + 3k))
            end
            a
        end, 6)
        for k in 1:Nz
            level_mean = sum(sum(panel[:, :, k]) for panel in vdiv) / nc
            for panel in vdiv
                panel[:, :, k] .-= level_mean
            end
        end

        P._OMEGA_LEVEL_PARALLEL[] = false
        diagnostics = P._reconstruct_omega_target!(
            am, bm, dm, vdiv, grid, 1.0;
            max_relative_correction = 0.05,
            active_levels = Bool[true, false],
        )
        @test diagnostics.max_relative_correction <= 0.05 * (1 + 1e-12)
        @test diagnostics.max_relative_correction > 0.049
        @test diagnostics.max_local_relative_correction > 0.0
        @test diagnostics.relative_correction_by_level[2] == 0.0
        @test all(panel[:, :, 2] == ones(Nc + 1, Nc) for panel in am)
        @test all(panel[:, :, 2] == ones(Nc, Nc + 1) for panel in bm)
        @test all(all(isfinite, panel) for panel in am)
        @test all(all(isfinite, panel) for panel in bm)
    end
end
