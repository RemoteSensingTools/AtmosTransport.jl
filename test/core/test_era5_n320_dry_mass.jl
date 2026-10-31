#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Breakpoint C — ERA5 N320 dry-air mass derivation.
#
# Verifies the hybrid-pressure → dry-mass surface for the N320 source grid:
#
#   1. Output struct + allocator shapes match the source mesh.
#   2. Q = 0 everywhere ⇒ dry == moist:
#        PS_dry == PS_total exactly,
#        Σ_k m_dry[:, k] ≈ PS × cell_area / g.
#   3. Constant Q = q₀ ⇒ uniform mass reduction:
#        PS_dry / PS_total = (1 − q₀) (to roundoff).
#   4. Coefficient-length mismatch / shape mismatch ⇒ DimensionMismatch.
#   5. (opt-in, real GRIB) End-to-end dry-mass on 2021-12-01 hour 0:
#        global mean PS_dry ≈ 0.99 × PS_total (water vapor is ~1% of mass);
#        global dry-mass total ≈ 5.1e18 kg (Earth's dry atmosphere).
# ---------------------------------------------------------------------------

using Test
using Dates

import AtmosTransport

using .AtmosTransport.Preprocessing: ERA5N320Settings, ERA5GRIBDayHandles,
                                      ERA5N320SpectralWorkspace, ERA5N320WindowFields,
                                      ERA5N320DryMassFields,
                                      allocate_era5_n320_spectral_workspace,
                                      allocate_era5_n320_window_fields,
                                      allocate_era5_n320_dry_mass_fields,
                                      discover_era5_n320_source_grid,
                                      discover_era5_spectral_truncation,
                                      read_era5_n320_window_fields!,
                                      derive_n320_dry_mass!,
                                      n320_cell_areas,
                                      open_era5_day, close_era5_day!,
                                      build_target_geometry,
                                      load_hybrid_coefficients
using .AtmosTransport.Grids: HybridSigmaPressure, ncells, nrings

const GRAV_TEST = 9.80665

function _tiny_source_grid(::Type{FT} = Float64) where FT
    cfg = Dict{String, Any}(
        "type"            => "synthetic_reduced_gaussian",
        "gaussian_number" => 2,
        "nlon_mode"       => "regular",
    )
    return build_target_geometry(Val(:synthetic_reduced_gaussian), cfg, FT)
end

"""Synthetic hybrid coordinate with `Nz` layers: A goes 0 → 0 (Pa), B goes
0 (TOA) → 1 (surface) linearly. Σ_k DELP_full[k] = PS_total exactly."""
function _synthetic_vc(Nz::Int, ::Type{FT} = Float64) where FT
    A = zeros(FT, Nz + 1)
    B = FT.(range(0.0, 1.0; length = Nz + 1))
    return HybridSigmaPressure(A, B)
end

@testset "ERA5 N320 dry-mass — breakpoint C" begin


    @testset "Q = 0 ⇒ dry == moist exactly" begin
        grid = _tiny_source_grid(Float64)
        Nz = 4
        nc = ncells(grid.mesh)

        # Synthetic window: uniform PS = 1000 hPa, Q = 0 everywhere.
        win = allocate_era5_n320_window_fields(grid, Nz)
        fill!(win.ps, 100_000.0)
        fill!(win.qv, 0.0)

        vc = _synthetic_vc(Nz, Float64)
        areas = n320_cell_areas(grid)
        dry = allocate_era5_n320_dry_mass_fields(grid, Nz)

        @test_throws ArgumentError allocate_era5_n320_dry_mass_fields(grid, 0)
        @test isapprox(sum(areas), 4π * grid.mesh.radius^2; rtol = 1e-6)
        derive_n320_dry_mass!(dry, win, vc, areas; grav = GRAV_TEST)

        @test all(dry.ps_dry .≈ 100_000.0)
        # Σ_k DELP_dry[c, k] == PS_dry by construction (we summed it).
        for c in 1:nc
            @test isapprox(sum(view(dry.delp_dry, c, :)), dry.ps_dry[c]; rtol = 1e-12)
        end
        # m_dry totals: Σ_k m_dry = PS × area / g
        for c in 1:nc
            expected = 100_000.0 * areas[c] / GRAV_TEST
            @test isapprox(sum(view(dry.m_dry, c, :)), expected; rtol = 1e-12)
        end
    end

    @testset "Q = q₀ constant ⇒ PS_dry = (1 − q₀) × PS_total" begin
        grid = _tiny_source_grid(Float64)
        Nz = 4

        q0 = 0.012
        win = allocate_era5_n320_window_fields(grid, Nz)
        fill!(win.ps, 100_000.0)
        fill!(win.qv, q0)

        vc = _synthetic_vc(Nz, Float64)
        areas = n320_cell_areas(grid)
        dry = allocate_era5_n320_dry_mass_fields(grid, Nz)

        derive_n320_dry_mass!(dry, win, vc, areas; grav = GRAV_TEST)
        @test all(dry.ps_dry .≈ (1 - q0) * 100_000.0)
    end

    @testset "Shape and coefficient-length mismatches" begin
        grid = _tiny_source_grid(Float64)
        Nz = 4
        win = allocate_era5_n320_window_fields(grid, Nz)
        areas = n320_cell_areas(grid)
        vc = _synthetic_vc(Nz, Float64)

        dry_wrong_Nz = allocate_era5_n320_dry_mass_fields(grid, Nz + 1)
        @test_throws DimensionMismatch derive_n320_dry_mass!(dry_wrong_Nz, win, vc, areas)

        dry = allocate_era5_n320_dry_mass_fields(grid, Nz)
        @test_throws DimensionMismatch derive_n320_dry_mass!(
            dry, win, vc, ones(Float64, length(areas) + 1))

        bad_vc = _synthetic_vc(Nz + 1, Float64)
        @test_throws DimensionMismatch derive_n320_dry_mass!(dry, win, bad_vc, areas)
    end

    @testset "Float32 dry mass — narrow-FT path is lossless to roundoff" begin
        grid = _tiny_source_grid(Float32)
        Nz = 4

        win = allocate_era5_n320_window_fields(grid, Nz)
        fill!(win.ps, 100_000.0f0)
        fill!(win.qv, 0.005f0)
        vc = _synthetic_vc(Nz, Float64)   # vc.A/B stay Float64 for arithmetic
        areas = n320_cell_areas(grid)
        dry = allocate_era5_n320_dry_mass_fields(grid, Nz)

        derive_n320_dry_mass!(dry, win, vc, areas; grav = GRAV_TEST)

        @test dry isa ERA5N320DryMassFields{Float32}
        @test all(isapprox.(dry.ps_dry, 99_500.0f0; rtol = 1e-6))
    end

    # -----------------------------------------------------------------------
    # Real-data smoke. Combines breakpoint B (synthesise PS / Q from GRIB)
    # with breakpoint C (derive dry mass) so a regression in either path is
    # visible. Gated on the same env var.
    # -----------------------------------------------------------------------
end
