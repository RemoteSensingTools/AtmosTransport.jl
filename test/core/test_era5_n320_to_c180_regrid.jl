#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Breakpoint D — conservative regrid from the N320 source mesh to a
# C-tier cubed-sphere target.
#
# Verifies the regrid surface for ERA5 → cubed sphere:
#
#   1. The physical N320 sphere (IFS radius) stays distinct from the
#      target regrid sphere.
#   2. Regrid of a uniform PS field returns the same value on every C-target
#      cell to floating-point roundoff (Float32 and Float64).
#   3. Regrid is mass-conserving: ∫ PS · dA preserved between the N320 source
#      and the C-target destination.
#   4. (opt-in, real GRIB via ATMOS_ERA5_N320_ROOT) N320 decode of
#      2021-12-01 hour 0 (grid discovery, T/Q/PS/U/V ranges and means) and
#      the pipeline allocator's coefficient-file vs Nz guard.
#
# A small `Nc = 8` CS target keeps the hermetic regridder build under a few
# seconds.
# ---------------------------------------------------------------------------

using Test
using Dates

import AtmosTransport

using .AtmosTransport.Preprocessing: ERA5N320Settings, open_era5_day, close_era5_day!,
                                     allocate_era5_n320_to_c180_pipeline,
                                     discover_era5_spectral_truncation,
                                     discover_era5_n320_source_grid,
                                     allocate_era5_n320_spectral_workspace,
                                     read_era5_n320_window_fields!, ERA5GRIBDayHandles,
                                     ERA5N320WindowFields, ERA5C180RegridFields,
                                     ERA5C180RegridWorkspace,
                                     allocate_era5_n320_window_fields,
                                     allocate_era5_c180_regrid_fields,
                                     allocate_era5_c180_regrid_workspace,
                                     regrid_n320_to_c180!, build_target_geometry,
                                     CubedSphereTargetGeometry
using .AtmosTransport.Grids: ncells, nrings, cell_area
using .AtmosTransport.Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS

function _tiny_source_grid(::Type{FT} = Float64; nlon_mode = "regular",
                           radius = EARTH_RADIUS) where FT
    cfg = Dict{String, Any}(
        "type"            => "synthetic_reduced_gaussian",
        "gaussian_number" => 8,
        "nlon_mode"       => nlon_mode,
    )
    grid = build_target_geometry(Val(:synthetic_reduced_gaussian), cfg, FT)
    FT(radius) == grid.mesh.radius && return grid
    mesh = AtmosTransport.ReducedGaussianMesh(
        grid.mesh.latitudes, grid.mesh.nlon_per_ring; FT, radius = FT(radius))
    return AtmosTransport.Preprocessing.ReducedGaussianTargetGeometry{
        FT, typeof(mesh)}(mesh, grid.geometry_source_grib, grid.gaussian_number,
                          copy(grid.nlon_per_ring), copy(grid.lats),
                          [FT.(AtmosTransport.Grids.ring_longitudes(mesh, j))
                           for j in 1:AtmosTransport.Grids.nrings(mesh)])
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


    @testset "physical N320 sphere is distinct from target regrid sphere" begin
        src = _tiny_source_grid(Float64; radius = IFS_EARTH_RADIUS)
        dst = _tiny_cs_target(Float64; Nc = 8)
        ws = allocate_era5_c180_regrid_workspace(src, dst, 1)
        @test src.mesh.radius == IFS_EARTH_RADIUS
        @test dst.mesh.radius == EARTH_RADIUS
        @test sum(ws.regridder.src_areas) ≈ 4π * EARTH_RADIUS^2 rtol = 1e-6
        @test sum(cell_area(src.mesh, c) for c in 1:ncells(src.mesh)) ≈
              4π * IFS_EARTH_RADIUS^2 rtol = 1e-6
    end

    @testset "Uniform PS regrid recovers the source value ($nlon_mode)" for nlon_mode in ("regular", "octahedral"), FT in (Float32, Float64)
        # Octahedral rings change length from ring to ring, which leaves uncovered
        # slivers between the chord-edged source cells; the coverage division undoes them.
        src = _tiny_source_grid(FT; nlon_mode)
        dst = _tiny_cs_target(FT; Nc = 8)
        Nz = 2

        ws  = allocate_era5_c180_regrid_workspace(src, dst, Nz)
        fields = allocate_era5_c180_regrid_fields(dst, Nz)

        # The coverage division must not hide a broken regridder: every source
        # polygon is intersected in full, and only the octahedral rings leave gaps.
        R = ws.regridder
        @test sum(R.dst_areas .* ws.coverage) ≈ sum(R.src_areas) rtol = (FT === Float64 ? 1e-12 : 1e-6)
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
        @test isapprox(dst_integral, src_integral; rtol = 2e-6)
    end


    # -----------------------------------------------------------------------
    # Real-data smoke. Wires B + D end-to-end so a regression in either path
    # is visible. Uses a small Nc = 24 CS target to keep regridder build
    # under 1-2 minutes (full C180 takes longer and we cover that in F).
    # -----------------------------------------------------------------------
end

@testset "Native N320 input (optional archive)" begin
    real_root_env = get(ENV, "ATMOS_ERA5_N320_ROOT", "")
    if !isempty(real_root_env) && isdir(real_root_env)
        @testset "Real N320 decode and pipeline coefficient contract" begin
            settings = ERA5N320Settings(; root_dir = real_root_env)
            handles  = open_era5_day(settings, Date(2021, 12, 1))

            try
                dst = _tiny_cs_target(Float32; Nc = 24)
                @test_throws DimensionMismatch allocate_era5_n320_to_c180_pipeline(handles, dst; Nz = 72)
                pipeline = allocate_era5_n320_to_c180_pipeline(handles, dst; Nz = 137, include_convection = false)
                @test pipeline.convection_fields === nothing
                T = discover_era5_spectral_truncation(handles.core_path)
                @test T == 639

                grid = discover_era5_n320_source_grid(handles.core_path; FT = Float64)
                @test grid.gaussian_number == 320
                @test nrings(grid.mesh)    == 640
                @test minimum(grid.nlon_per_ring) >= 18
                @test maximum(grid.nlon_per_ring) == 1280

                Nz = 137
                ws = allocate_era5_n320_spectral_workspace(grid, T, Nz)
                f  = allocate_era5_n320_window_fields(grid, Nz)

                read_era5_n320_window_fields!(f, ws, handles, Date(2021, 12, 1), 0)

                # Physical-range gates. ECMWF model atmosphere covers ~175 K
                # (mesopause) to ~330 K (summer Sahara surface), Q is bounded
                # by ~30 g/kg in the tropical boundary layer, and the lowest
                # surface pressures on Earth are over the Himalayas/Tibet
                # plateau (~490 hPa over the highest summits).
                @test all(175.0     .<= f.t  .<= 330.0)
                @test all(0.0       .<= f.qv .<= 0.03)
                @test all(30_000.0  .<= f.ps .<= 110_000.0)

                # Wind ranges checked on cells with |φ| ≤ 89° — the two polar
                # rings of N320 are at ±89.78° where `1/cos(φ)` amplifies any
                # spectral synthesis noise. Stratospheric jets reach ~150 m/s.
                mesh = grid.mesh
                interior_cells = vcat([mesh.ring_offsets[j]:(mesh.ring_offsets[j + 1] - 1)
                                       for j in 1:nrings(mesh)
                                       if abs(mesh.latitudes[j]) <= 89.0]...)
                @test all(-200.0 .<= f.u[interior_cells, :] .<= 200.0)
                @test all(-200.0 .<= f.v[interior_cells, :] .<= 200.0)

                # Global means: PS within 2% of the ~985 hPa global average
                # for December (cell-area-weighted is better but cells are
                # equator-weighted in N320, which is close enough).
                ps_mean = sum(f.ps) / length(f.ps)
                @test 96_000.0 <= ps_mean <= 102_000.0

                # Global mean temperature integrated across all 137 levels
                # is near 240 K for a standard atmosphere weighted by mass.
                # We use a simple level-arithmetic mean here, which is biased
                # cold relative to mass-weighted (stratosphere over-represented)
                # but stays in a reproducible range.
                t_mean = sum(f.t) / length(f.t)
                @test 230.0 <= t_mean <= 260.0
            finally
                close_era5_day!(handles)
            end
        end
    else
        @info "Skipping real N320 window-reader smoke (set ATMOS_ERA5_N320_ROOT to enable)."
        @test_skip false
    end
end
