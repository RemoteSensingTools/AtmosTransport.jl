#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Breakpoint D — conservative regrid from the N320 source mesh to a
# C-tier cubed-sphere target.
#
# Verifies the regrid surface for ERA5 → cubed sphere:
#
#   1. Output struct + workspace allocate with the right shapes.
#   2. Regrid of a uniform PS field returns the same value on every C-target
#      cell to floating-point roundoff (idempotency check).
#   3. Regrid is mass-conserving: ∫ PS · dA preserved between the N320 source
#      and the C-target destination.
#   4. Per-panel mean PS lands in a physical range after regridding real PS.
#   5. (opt-in, real GRIB) End-to-end smoke for 2021-12-01 hour 0:
#      regrid PS / U / V / T / Q from the actual N320 archive to a small CS
#      target and check physical ranges + means.
#
# A small `Nc = 8` CS target keeps the hermetic regridder build under a few
# seconds.
# ---------------------------------------------------------------------------

using Test
using Dates

import AtmosTransport

using .AtmosTransport.Preprocessing: ERA5N320Settings, ERA5GRIBDayHandles,
                                      ERA5N320WindowFields,
                                      ERA5C180RegridFields, ERA5C180RegridWorkspace,
                                      allocate_era5_n320_spectral_workspace,
                                      allocate_era5_n320_window_fields,
                                      allocate_era5_c180_regrid_fields,
                                      allocate_era5_c180_regrid_workspace,
                                      regrid_n320_to_c180!,
                                      discover_era5_n320_source_grid,
                                      discover_era5_spectral_truncation,
                                      read_era5_n320_window_fields!,
                                      open_era5_day, close_era5_day!,
                                      build_target_geometry,
                                      CubedSphereTargetGeometry
using .AtmosTransport.Grids: ncells, nrings, cell_area

function _tiny_source_grid(::Type{FT} = Float64; nlon_mode = "regular") where FT
    cfg = Dict{String, Any}(
        "type"            => "synthetic_reduced_gaussian",
        "gaussian_number" => 8,
        "nlon_mode"       => nlon_mode,
    )
    return build_target_geometry(Val(:synthetic_reduced_gaussian), cfg, FT)
end

function _tiny_cs_target(::Type{FT} = Float64; Nc = 8) where FT
    cfg = Dict{String, Any}(
        "type"             => "cubed_sphere",
        "Nc"               => Nc,
        "panel_convention" => "geos_native",
        "definition"       => "gmao_equal_distance",
    )
    return build_target_geometry(Val(:cubed_sphere), cfg, FT)
end

@testset "ERA5 N320 → C180 regrid — breakpoint D" begin

    @testset "Output + workspace shapes" begin
        src = _tiny_source_grid(Float64)
        dst = _tiny_cs_target(Float64; Nc = 8)
        Nz = 4

        fields = allocate_era5_c180_regrid_fields(dst, Nz)
        @test fields isa ERA5C180RegridFields{Float64}
        @test all(size(fields.ps[p]) == (8, 8)     for p in 1:6)
        @test all(size(fields.u[p])  == (8, 8, Nz) for p in 1:6)
        @test all(size(fields.v[p])  == (8, 8, Nz) for p in 1:6)
        @test all(size(fields.t[p])  == (8, 8, Nz) for p in 1:6)
        @test all(size(fields.qv[p]) == (8, 8, Nz) for p in 1:6)

        @test_throws ArgumentError allocate_era5_c180_regrid_fields(dst, 0)

        mktempdir() do cache_dir
            ws = allocate_era5_c180_regrid_workspace(src, dst, Nz;
                                                      cache_dir = cache_dir)
            @test ws isa ERA5C180RegridWorkspace
            @test length(ws.dst_flat_2d) == 6 * 8 * 8
            @test size(ws.dst_flat_3d)   == (6 * 8 * 8, Nz)
            @test size(ws.src_flat_3d)   == (ncells(src.mesh), Nz)

            # Cache file should exist after first call; second call reuses it.
            files = readdir(cache_dir)
            @test any(startswith.(files, "regridder_"))

            ws2 = allocate_era5_c180_regrid_workspace(src, dst, Nz;
                                                       cache_dir = cache_dir)
            @test ws2 isa ERA5C180RegridWorkspace
        end
    end

    @testset "Uniform PS regrid recovers the source value ($nlon_mode)" for nlon_mode in ("regular", "octahedral")
        # Octahedral rings change length from ring to ring, which leaves uncovered
        # slivers between the chord-edged source cells; the coverage division undoes them.
        src = _tiny_source_grid(Float64; nlon_mode)
        dst = _tiny_cs_target(Float64; Nc = 8)
        Nz = 2

        ws  = allocate_era5_c180_regrid_workspace(src, dst, Nz)
        fields = allocate_era5_c180_regrid_fields(dst, Nz)

        # The coverage division must not hide a broken regridder: every source
        # polygon is intersected in full, and only the octahedral rings leave gaps.
        R = ws.regridder
        @test sum(R.dst_areas .* ws.coverage) ≈ sum(R.src_areas) rtol = 1e-12
        if nlon_mode == "regular"
            @test all(c -> abs(c - 1) < 1e-6, ws.coverage)
        else
            @test 0.99 < minimum(ws.coverage) < 0.999 && maximum(ws.coverage) < 1.01
        end

        # Build a synthetic source window with PS = 101325 Pa everywhere
        # and zero 3D fields; U/V/T/Q being identically zero exercises the
        # regrid kernel for 3D inputs without introducing a confounding signal.
        win = allocate_era5_n320_window_fields(src, Nz)
        fill!(win.ps, 101325.0)
        regrid_n320_to_c180!(fields, win, ws, dst)

        # Conservative regrid on small CS meshes leaves a few corner cells
        # with residual ~5e-4 Pa drift relative to the source (the
        # MultiTreeWrapper bounding-box pruning misses tiny intersections at
        # panel-edge cells). Relative error stays at 1e-8 across the panel.
        for p in 1:6
            rel_dev = maximum(abs.(fields.ps[p] .- 101325.0)) / 101325.0
            @test rel_dev < 1e-6
            @test all(abs.(fields.u[p]) .< 1e-12)
            @test all(abs.(fields.v[p]) .< 1e-12)
        end
    end

    @testset "Area-weighted PS is conserved on regrid" begin
        src = _tiny_source_grid(Float64)
        dst = _tiny_cs_target(Float64; Nc = 8)
        Nz = 1

        ws  = allocate_era5_c180_regrid_workspace(src, dst, Nz)
        fields = allocate_era5_c180_regrid_fields(dst, Nz)
        win = allocate_era5_n320_window_fields(src, Nz)

        # Spatially varying PS: latitude-dependent so the regrid actually
        # does work, but smooth so conservative regrid stays bounded.
        mesh_src = src.mesh
        for j in 1:nrings(mesh_src)
            ring_start = mesh_src.ring_offsets[j]
            ring_end   = mesh_src.ring_offsets[j + 1] - 1
            ps_j = 100_000.0 + 2_000.0 * sind(mesh_src.latitudes[j])
            for c in ring_start:ring_end
                win.ps[c] = ps_j
            end
        end

        # Compute source area-weighted integral.
        src_areas = [Float64(cell_area(mesh_src, c)) for c in 1:ncells(mesh_src)]
        src_integral = sum(win.ps .* src_areas)

        regrid_n320_to_c180!(fields, win, ws, dst)

        # CubedSphereMesh.cell_area takes (i, j) — every panel shares the
        # same per-(i, j) area matrix because CS panels are isotropic. The
        # global cell ordering is `(p-1)·Nc² + (j-1)·Nc + i`.
        mesh_dst = dst.mesh
        dst_integral = 0.0
        for p in 1:6
            panel = fields.ps[p]
            for j in 1:mesh_dst.Nc, i in 1:mesh_dst.Nc
                dst_integral += panel[i, j] * Float64(cell_area(mesh_dst, i, j))
            end
        end
        @test isapprox(dst_integral, src_integral; rtol = 1e-3)
    end

    @testset "Float32 round-trip" begin
        src = _tiny_source_grid(Float32)
        dst = _tiny_cs_target(Float32; Nc = 8)
        Nz = 2

        ws  = allocate_era5_c180_regrid_workspace(src, dst, Nz)
        fields = allocate_era5_c180_regrid_fields(dst, Nz)
        win = allocate_era5_n320_window_fields(src, Nz)
        fill!(win.ps, 101325.0f0)
        fill!(win.t, 250.0f0)
        regrid_n320_to_c180!(fields, win, ws, dst)

        @test fields isa ERA5C180RegridFields{Float32}
        for p in 1:6
            @test all(isapprox.(fields.ps[p], 101325.0f0; rtol = 1e-5))
            @test all(isapprox.(fields.t[p],  250.0f0;    rtol = 1e-5))
        end
    end

    # -----------------------------------------------------------------------
    # Real-data smoke. Wires B + D end-to-end so a regression in either path
    # is visible. Uses a small Nc = 24 CS target to keep regridder build
    # under 1-2 minutes (full C180 takes longer and we cover that in F).
    # -----------------------------------------------------------------------
    real_root_env = get(ENV, "ATMOS_ERA5_N320_ROOT", "")
    if !isempty(real_root_env) && isdir(real_root_env)
        @testset "Real N320 → C24 regrid (2021-12-01 hour 0)" begin
            settings = ERA5N320Settings(; root_dir = real_root_env)
            handles  = open_era5_day(settings, Date(2021, 12, 1))
            try
                T = discover_era5_spectral_truncation(handles.core_path)
                src = discover_era5_n320_source_grid(handles.core_path; FT = Float64)
                dst = _tiny_cs_target(Float64; Nc = 24)
                Nz = 137

                spec_ws = allocate_era5_n320_spectral_workspace(src, T, Nz)
                win = allocate_era5_n320_window_fields(src, Nz)
                read_era5_n320_window_fields!(win, spec_ws, handles, Date(2021, 12, 1), 0)

                regrid_ws = allocate_era5_c180_regrid_workspace(src, dst, Nz)
                fields = allocate_era5_c180_regrid_fields(dst, Nz)
                regrid_n320_to_c180!(fields, win, regrid_ws, dst)

                # Per-panel PS sanity: every panel mean within a reasonable
                # band around the global ~985 hPa.
                for p in 1:6
                    panel_mean = sum(fields.ps[p]) / length(fields.ps[p])
                    @test 90_000.0 <= panel_mean <= 102_000.0
                end

                # T and Q stay in physical ranges after regridding.
                t_all = vcat([vec(fields.t[p]) for p in 1:6]...)
                qv_all = vcat([vec(fields.qv[p]) for p in 1:6]...)
                @test all(170.0 .<= t_all  .<= 330.0)
                @test all(0.0   .<= qv_all .<= 0.03)
            finally
                close_era5_day!(handles)
            end
        end
    else
        @info "Skipping real N320 → CS regrid smoke (set ATMOS_ERA5_N320_ROOT to enable)."
        @test_skip false
    end
end
