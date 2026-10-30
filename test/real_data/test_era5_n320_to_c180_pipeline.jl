#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Breakpoint F — full per-window N320 → C180 preprocessing pipeline.
#
# Wires breakpoints B (spectral synthesis), C (dry-mass), D (regrid), and E
# (convection) into a single `process_era5_n320_window!` call and verifies:
#
#   1. The pipeline allocator produces a consistent bundle and validates
#      coefficient-file vs Nz mismatch.
#   2. `include_convection = false` produces a pipeline with
#      `convection_fields === nothing`.
#   3. End-to-end smoke for 2021-12-01 hour 6 against the real N320
#      archive: every output field (n320 window, dry mass, convection,
#      C180 regridded) reaches its expected physical range.
#
# All three need the real N320 GRIB archive: set `ATMOS_ERA5_N320_ROOT`
# (otherwise the file records one skipped test). Hence the real_data tier
# (`test/runtests.jl --real-data`).
# ---------------------------------------------------------------------------

using Test
using Dates

import AtmosTransport

using .AtmosTransport.Preprocessing: ERA5N320Settings, ERA5GRIBDayHandles,
                                      ERA5N320ToC180Pipeline,
                                      allocate_era5_n320_to_c180_pipeline,
                                      process_era5_n320_window!,
                                      open_era5_day, close_era5_day!,
                                      build_target_geometry
using .AtmosTransport.Grids: nrings

function _tiny_cs_target(::Type{FT} = Float64; Nc = 24) where FT
    cfg = Dict{String, Any}(
        "type"             => "cubed_sphere",
        "Nc"               => Nc,
        "panel_convention" => "geos_native",
        "definition"       => "gmao_equal_distance",
    )
    return build_target_geometry(Val(:cubed_sphere), cfg, FT)
end

@testset "ERA5 N320 → C180 pipeline — breakpoint F" begin

    # -----------------------------------------------------------------------
    # Real-data smoke. The pipeline allocator needs to discover the source
    # mesh + spectral truncation from the day's core GRIB, so a hermetic
    # path would need to mock the GRIB headers. Instead we gate the entire
    # testset on `ATMOS_ERA5_N320_ROOT` — the file exists, the test runs.
    # -----------------------------------------------------------------------
    real_root_env = get(ENV, "ATMOS_ERA5_N320_ROOT", "")
    if isempty(real_root_env) || !isdir(real_root_env)
        @info "Skipping breakpoint F (set ATMOS_ERA5_N320_ROOT to enable)."
        @test_skip false
    else

    @testset "Allocator produces a consistent bundle" begin
        settings = ERA5N320Settings(; root_dir = real_root_env,
                                      include_convection = true,
                                      include_surface = true)
        handles = open_era5_day(settings, Date(2021, 12, 1))
        try
            dst = _tiny_cs_target(Float32; Nc = 24)
            pipeline = allocate_era5_n320_to_c180_pipeline(handles, dst;
                                                            Nz = 137,
                                                            include_convection = true)
            @test pipeline isa ERA5N320ToC180Pipeline{Float32}
            @test pipeline.target_grid === dst
            @test pipeline.convection_fields !== nothing
            @test nrings(pipeline.source_grid.mesh) == 640
            @test length(pipeline.cell_areas) == 542080
            @test pipeline.spectral_ws.T  == 639
            @test pipeline.spectral_ws.Nz == 137

            # No-convection variant.
            pipeline_no_conv = allocate_era5_n320_to_c180_pipeline(handles, dst;
                                                                    Nz = 137,
                                                                    include_convection = false)
            @test pipeline_no_conv.convection_fields === nothing
        finally
            close_era5_day!(handles)
        end
    end

    @testset "Coefficient-file vs Nz mismatch is rejected" begin
        settings = ERA5N320Settings(; root_dir = real_root_env)
        handles  = open_era5_day(settings, Date(2021, 12, 1))
        try
            dst = _tiny_cs_target(Float32; Nc = 24)
            @test_throws DimensionMismatch allocate_era5_n320_to_c180_pipeline(
                handles, dst; Nz = 72)
        finally
            close_era5_day!(handles)
        end
    end

    @testset "process_era5_n320_window! — 2021-12-01 hour 6 → C24" begin
        settings = ERA5N320Settings(; root_dir = real_root_env,
                                      include_convection = true)
        handles  = open_era5_day(settings, Date(2021, 12, 1))
        try
            dst = _tiny_cs_target(Float32; Nc = 24)
            pipeline = allocate_era5_n320_to_c180_pipeline(handles, dst;
                                                            Nz = 137,
                                                            include_convection = true)
            process_era5_n320_window!(pipeline, handles, Date(2021, 12, 1), 6)

            # N320 source-mesh outputs.
            win = pipeline.window_fields
            @test all(175.0    .<= win.t  .<= 330.0)
            @test all(0.0      .<= win.qv .<= 0.03)
            @test all(30_000.0 .<= win.ps .<= 110_000.0)
            @test 96_000.0 <= sum(win.ps) / length(win.ps) <= 102_000.0

            # Dry-mass derivation on source mesh.
            dry = pipeline.dry_fields
            ratio = sum(dry.ps_dry) / sum(win.ps)
            @test 0.985 <= ratio <= 1.0
            @test 4.8e18 <= sum(dry.m_dry) <= 5.3e18

            # Convection forecast read.
            conv = pipeline.convection_fields
            @test all(conv.udmf .>= -1e-12)
            @test all(abs.(conv.udmf) .<= 2.0)
            @test maximum(conv.udmf) > 1e-6
            @test minimum(conv.ddmf) < -1e-4

            # Regridded C24 scalar fields. Per-panel mean PS in band, T/Q in
            # physical range across all 6 panels.
            c180 = pipeline.c180_fields
            for p in 1:6
                panel_ps_mean = sum(c180.ps[p]) / length(c180.ps[p])
                @test 90_000.0 <= panel_ps_mean <= 102_000.0
            end
            t_all  = vcat([vec(c180.t[p])  for p in 1:6]...)
            qv_all = vcat([vec(c180.qv[p]) for p in 1:6]...)
            @test all(170.0 .<= t_all  .<= 330.0)
            @test all(0.0   .<= qv_all .<= 0.03)
        finally
            close_era5_day!(handles)
        end
    end
    end
end

using AtmosTransport.Preprocessing: ERA5N320Settings, open_era5_day, close_era5_day!, discover_era5_n320_source_grid, allocate_era5_n320_convection_fields, read_era5_n320_convection_window!

@testset "Real N320 convection input" begin
    real_root_env = get(ENV, "ATMOS_ERA5_N320_ROOT", "")
    if !isempty(real_root_env) && isdir(real_root_env)
        @testset "Real N320 convection smoke (2021-12-01 hour 6)" begin
            settings = ERA5N320Settings(; root_dir = real_root_env,
                                          include_convection = true)
            handles = open_era5_day(settings, Date(2021, 12, 1))
            try
                @test handles.convection_path !== nothing

                grid = discover_era5_n320_source_grid(handles.core_path; FT = Float64)
                Nz = 137
                f = allocate_era5_n320_convection_fields(grid, Nz)
                read_era5_n320_convection_window!(f, handles, grid.mesh,
                                                   Date(2021, 12, 1), 6)

                # ERA5 convective mass fluxes are time-mean kg m⁻² s⁻¹.
                # Updraft is non-negative; downdraft is mostly non-positive
                # with small (~1e-6 kg m⁻² s⁻¹) positive noise from the
                # forecast time-averaging — sign-flip bug would invert the
                # dominant tail. Magnitudes peak near ~1 kg m⁻² s⁻¹ in the
                # deepest convection.
                @test all(f.udmf .>= -1e-12)
                @test all(abs.(f.udmf) .<= 2.0)
                @test all(abs.(f.ddmf) .<= 2.0)
                @test minimum(f.ddmf) < -1e-4               # downdraft tail is genuinely negative
                @test count(f.ddmf .> 1e-4) == 0            # no large positive outliers

                # Detrainment rates are non-negative magnitudes.
                @test all(f.udrf .>= -1e-12)
                @test all(f.ddrf .>= -1e-12)
                # Sanity: at least one cell has non-trivial convection at
                # hour 6 UTC (deep tropics are convectively active).
                @test maximum(f.udmf) > 1e-6
            finally
                close_era5_day!(handles)
            end
        end
    else
        @info "Skipping real N320 convection smoke (set ATMOS_ERA5_N320_ROOT to enable)."
        @test_skip false
    end
end
