#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan indexed-baking-valiant Commit 5 — GEOS → CS passthrough orchestrator
#
# Verifies:
#   1. process_day(date, ::CSGrid, ::AbstractGEOSSettings, vertical; out_path)
#      writes a v4 binary at out_path with the expected size and shape.
#   2. The binary's write-time replay gate is below the requested tolerance.
#   3. The GEOS native path column-balances fluxes to the raw dry endpoint
#      and writes a binary that closes replay against that endpoint.
#   4. `geos_native_to_face_flux!` correctly maps MFXC → am_v4 with
#      panel-edge halos populated by mirror sync.
# ---------------------------------------------------------------------------

using Test
using Dates
using NCDatasets
using TOML

import AtmosTransport
using .AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention,
                              EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST,
                              HybridSigmaPressure, n_levels
using .AtmosTransport.Preprocessing: GEOSITSettings, process_day,
                                      build_target_geometry,
                                      geos_native_to_face_flux!,
                                      sync_all_cs_boundary_mirrors!,
                                      load_hybrid_coefficients,
                                      plan_vertical,
                                      MergeAbovePressure,
                                      CS_PANEL_COUNT
using .AtmosTransport.MetDrivers: TransportBinaryReader, CubedSphereBinaryGeometry,
                                  binary_geometry, load_window!,
                                  load_flux_delta_window!,
                                  verify_window_continuity_cs

const FT_TEST = Float64
const NC = 8
const NPANEL = 6
const NZ = 4
const MFXC_C5 = 80.0
const MFYC_C5 = -40.0

function assert_cs_window_replay(reader::TransportBinaryReader, win::Int;
                                 tol::Real = 1e-12)
    window = load_window!(reader, win)
    deltas = load_flux_delta_window!(reader, win)
    steps = reader.header.steps_per_window_by_window[win]
    m_next = ntuple(p -> window.m[p] .+ deltas.dm[p], 6)
    diag = verify_window_continuity_cs(window.m, window.am, window.bm,
                                       window.cm, m_next, steps)
    @test diag.max_rel_err < tol
    @test all(all(isfinite, panel) for panel in window.am)
    @test all(all(isfinite, panel) for panel in window.bm)
    @test all(all(isfinite, panel) for panel in window.cm)
    @test all(all(>(0), panel) for panel in m_next)
    return window
end

# Inline the synthetic fixture writers (same shape as test_geos_reader.jl;
# duplicating ~50 lines is cleaner than cross-test importing).

function _synthetic_hybrid_coefs_c5(nz::Int)
    nz == 4 || error("synthetic coefs only defined for nz=4 here")
    A = [0.0, 0.0, 0.0, 0.0, 0.0]
    B = [0.0, 0.05, 0.20, 0.55, 1.0]
    return A, B
end

function _defvar3d_c5!(ds, name, A; units::String="")
    v = defVar(ds, name, eltype(A), ("Xdim","Ydim","nf","lev","time");
               attrib = ["units" => units])
    v[:, :, :, :, :] = A
    return v
end

function _defvar2d_xyz_t_c5!(ds, name, A; units::String="")
    v = defVar(ds, name, eltype(A), ("Xdim","Ydim","nf","time");
               attrib = ["units" => units])
    v[:, :, :, :] = A
    return v
end

"""Write a synthetic GEOS-IT day under `dir`, bottom-up levels, 24-window cadence.
For replay-gate testing we default to **zero horizontal fluxes** so the
divergence is identically zero and the cm column closes trivially. Set
`mfxc_const`/`mfyc_const` to test other paths.
"""
function _write_synthetic_geosit_day_c5!(dir::String, datestr::String;
                                         qv_const::Float64 = 0.005,
                                         mfxc_const::Float64 = 0.0,
                                         mfyc_const::Float64 = 0.0,
                                         ntimes_a1::Int = 24,
                                         ntimes_i1::Int = 25)
    mkpath(dir)
    A_top, B_top = _synthetic_hybrid_coefs_c5(NZ)
    delp_per_layer_bu = Float64[]
    for k in NZ:-1:1
        ΔA = A_top[k+1] - A_top[k]
        ΔB = B_top[k+1] - B_top[k]
        push!(delp_per_layer_bu, ΔA + ΔB * 100_000.0)
    end

    a1 = joinpath(dir, "GEOSIT.$(datestr).CTM_A1.C$(NC).nc")
    NCDataset(a1, "c") do ds
        ds.dim["Xdim"] = NC
        ds.dim["Ydim"] = NC
        ds.dim["nf"]   = NPANEL
        ds.dim["lev"]  = NZ
        ds.dim["time"] = ntimes_a1
        delp_bu = fill(0.0, NC, NC, NPANEL, NZ, ntimes_a1)
        for k in 1:NZ
            delp_bu[:, :, :, k, :] .= delp_per_layer_bu[k]
        end
        _defvar3d_c5!(ds, "DELP", delp_bu;       units="Pa")
        _defvar3d_c5!(ds, "MFXC", fill(mfxc_const, NC, NC, NPANEL, NZ, ntimes_a1);
                       units="Pa m+2")
        _defvar3d_c5!(ds, "MFYC", fill(mfyc_const, NC, NC, NPANEL, NZ, ntimes_a1);
                       units="Pa m+2")
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib=["positive" => "down"])
    end

    i1 = joinpath(dir, "GEOSIT.$(datestr).CTM_I1.C$(NC).nc")
    NCDataset(i1, "c") do ds
        ds.dim["Xdim"] = NC
        ds.dim["Ydim"] = NC
        ds.dim["nf"]   = NPANEL
        ds.dim["lev"]  = NZ
        ds.dim["time"] = ntimes_i1
        _defvar2d_xyz_t_c5!(ds, "PS", fill(1000.0, NC, NC, NPANEL, ntimes_i1);
                             units="hPa")
        _defvar3d_c5!(ds, "QV", fill(qv_const, NC, NC, NPANEL, NZ, ntimes_i1);
                       units="kg kg-1")
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib=["positive" => "down"])
    end

    return a1, i1
end

function _write_synthetic_coefs_toml_c5!(path::String)
    A, B = _synthetic_hybrid_coefs_c5(NZ)
    open(path, "w") do io
        write(io, """
            [metadata]
            n_levels = $(NZ)
            n_interfaces = $(NZ + 1)

            [coefficients]
            a = $(repr(A))
            b = $(repr(B))
            """)
    end
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "GEOS → CS passthrough" begin
    tmpdir   = mktempdir()
    daydir   = joinpath(tmpdir, "20211201")
    nextdir  = joinpath(tmpdir, "20211202")
    _write_synthetic_geosit_day_c5!(daydir,  "20211201";
                                    mfxc_const = MFXC_C5, mfyc_const = MFYC_C5)
    _write_synthetic_geosit_day_c5!(nextdir, "20211202";
                                    mfxc_const = MFXC_C5, mfyc_const = MFYC_C5)
    coef_path = joinpath(tmpdir, "synthetic_coefs.toml")
    _write_synthetic_coefs_toml_c5!(coef_path)

    settings = GEOSITSettings(
        root_dir = tmpdir, Nc = NC,
        mass_flux_dt = 450.0,
        coefficients_file = coef_path,
    )

    grid = build_target_geometry(
        Dict{String,Any}("type" => "cubed_sphere", "Nc" => NC,
                          "panel_convention" => "geos_native"),
        FT_TEST,
    )

    vertical = (
        merged_vc = load_hybrid_coefficients(coef_path),
        Nz = NZ,
        Nz_native = NZ,
    )
    merged_plan = plan_vertical(
        MergeAbovePressure(pressure_Pa = 20_000.0,
                           target_min_thickness_Pa = Inf),
        vertical.merged_vc)
    vertical_merged = (
        plan = merged_plan,
        transform = merged_plan.transform,
        merged_vc = merged_plan.merged_vc,
        native_vc = merged_plan.native_vc,
        merge_map = merged_plan.merge_map,
        groups = merged_plan.groups,
        vertical_mapping_method = :merge_above_pressure,
        Nz = merged_plan.Nz_output,
        Nz_native = merged_plan.Nz_native,
    )

    @testset "geos_native_to_face_flux! shape + values" begin
        # Use a SINGLE constant for both am_native and bm_native — that way,
        # regardless of which neighbour edge a panel's west/south halo comes
        # from (X-edge or Y-edge, with possibly flipped orientation), the
        # synced halo magnitude is unambiguous.
        K = FT_TEST(42)
        am_native = ntuple(p -> fill(K, NC, NC, NZ), 6)
        bm_native = ntuple(p -> fill(K, NC, NC, NZ), 6)
        am_v4 = ntuple(p -> zeros(FT_TEST, NC + 1, NC, NZ), 6)
        bm_v4 = ntuple(p -> zeros(FT_TEST, NC, NC + 1, NZ), 6)
        scale = FT_TEST(2.0)
        geos_native_to_face_flux!(am_v4, bm_v4, am_native, bm_native,
                                  grid.mesh.connectivity, NC, NZ, scale)
        # Interior + east boundary: am_v4[i+1, j, k] = MFXC[i, j, k] * scale.
        # West-boundary halo (i=1) is filled from a neighbour's canonical
        # edge by `sync_all_cs_boundary_mirrors!` and has magnitude 42*scale
        # too (sign may flip per panel orientation, hence the abs check).
        for p in 1:6
            for k in 1:NZ, j in 1:NC, i in 1:NC
                @test am_v4[p][i + 1, j, k] ≈ K * scale
            end
            for k in 1:NZ, j in 1:NC
                @test abs(am_v4[p][1, j, k]) ≈ K * scale
            end
            for k in 1:NZ, i in 1:NC, j in 1:NC
                @test bm_v4[p][i, j + 1, k] ≈ K * scale
            end
            for k in 1:NZ, i in 1:NC
                @test abs(bm_v4[p][i, 1, k]) ≈ K * scale
            end
        end
    end

    @testset "nested GEOS CS block coarsening helpers" begin
        R = Val(2)
        src3 = reshape(collect(FT_TEST, 1:16), 4, 4, 1)
        dst3 = zeros(FT_TEST, 2, 2, 1)
        AtmosTransport.Preprocessing._coarsen_sum3!(dst3, src3, R)
        @test dst3[:, :, 1] == [
            sum(src3[1:2, 1:2, 1]) sum(src3[1:2, 3:4, 1]);
            sum(src3[3:4, 1:2, 1]) sum(src3[3:4, 3:4, 1])
        ]

        src_area = fill(FT_TEST(1), 4, 4)
        src2 = reshape(collect(FT_TEST, 1:16), 4, 4)
        dst2 = zeros(FT_TEST, 2, 2)
        AtmosTransport.Preprocessing._coarsen_area_weighted2!(dst2, src2, src_area, R)
        @test dst2 ≈ [
            sum(src2[1:2, 1:2]) / 4 sum(src2[1:2, 3:4]) / 4;
            sum(src2[3:4, 1:2]) / 4 sum(src2[3:4, 3:4]) / 4
        ]

        src_x = ones(FT_TEST, 5, 4, 1)
        dst_x = zeros(FT_TEST, 3, 2, 1)
        AtmosTransport.Preprocessing._coarsen_xface_sum!(dst_x, src_x, R)
        @test all(dst_x .== 2)

        src_y = ones(FT_TEST, 4, 5, 1)
        dst_y = zeros(FT_TEST, 2, 3, 1)
        AtmosTransport.Preprocessing._coarsen_yface_sum!(dst_y, src_y, R)
        @test all(dst_y .== 2)
    end

    @testset "process_day writes a valid CS binary with reproducible bytes" begin
        out_path = joinpath(tmpdir, "out_cs.bin")
        first = process_day(Date(2021, 12, 1), grid, settings, vertical;
                            out_path = out_path,
                            dt_met_seconds = 3600.0,
                            FT = FT_TEST,
                            mass_basis = :dry,
                            replay_tol = 1e-12)
        @test isfile(out_path)
        @test first.elapsed > 0
        @test first.worst_replay_rel < 1e-12
        @test first.out_path == out_path

        # Open the binary back and confirm the header round-trips.
        reader = TransportBinaryReader(out_path; FT=FT_TEST)
        try
            @test binary_geometry(reader) isa CubedSphereBinaryGeometry
            @test reader.header.geometry.Nc == NC
            @test reader.header.geometry.npanel == 6
            @test reader.header.nlevel == NZ
            @test reader.header.nwindow == 24                        # GEOS hourly
            @test reader.header.steps_per_window == 8
            @test reader.header.mass_basis === :dry
            @test reader.header.geometry.panel_convention === :geos_native
            @test :dm in reader.header.payload_sections              # delta enabled
            @test reader.header.raw_header["runtime_substep_contract"] == "binary_schedule"
            @test reader.header.raw_header["geos_mass_endpoint"] == "raw_dry_endpoint"
            @test reader.header.raw_header["geos_horizontal_balance"] ==
                  "column_poisson_to_endpoint"

            window = assert_cs_window_replay(reader, 1)
            @test any(panel -> any(x -> x != 0, panel), window.am)
            @test any(panel -> any(x -> x != 0, panel), window.bm)
        finally
            close(reader)
        end

        second_path = joinpath(tmpdir, "out_cs_second.bin")
        second = process_day(Date(2021, 12, 1), grid, settings, vertical;
                             out_path = second_path,
                             dt_met_seconds = 3600.0,
                             FT = FT_TEST,
                             mass_basis = :dry,
                             replay_tol = 1e-12)
        @test isfile(second_path)
        @test second.out_path == second_path
        @test second.worst_replay_rel ≈ first.worst_replay_rel
        @test second.worst_replay_abs ≈ first.worst_replay_abs
        @test second.worst_replay_win == first.worst_replay_win
        @test read(second_path) == read(out_path)
        for p in 1:6
            @test second.final_m[p] == first.final_m[p]
        end

        # `write_replay_check = false` skips the gate (no worst replay window)
        # and is recorded; the transported mass is the same.
        off_path = joinpath(tmpdir, "out_cs_off.bin")
        off = process_day(Date(2021, 12, 1), grid, settings, vertical;
                          out_path = off_path, dt_met_seconds = 3600.0, FT = FT_TEST,
                          mass_basis = :dry, replay_tol = 1e-12, write_replay_check = false)
        @test off.worst_replay_win == 0
        @test all(p -> off.final_m[p] == first.final_m[p], 1:6)
        reader = TransportBinaryReader(off_path; FT = FT_TEST)
        try
            @test reader.header.raw_header["write_replay_check"] === false
        finally
            close(reader)
        end
        reader = TransportBinaryReader(out_path; FT = FT_TEST)
        try
            @test !haskey(reader.header.raw_header, "write_replay_check")
        finally
            close(reader)
        end
    end

    @testset "process_day supports GEOS native vertical merge above pressure" begin
        @test vertical_merged.groups == [1:2, 3:3, 4:4]
        @test vertical_merged.Nz == 3
        parsed_vertical = AtmosTransport.Preprocessing._build_native_vertical_setup(
            Dict{String, Any}("transform" => "merge_above_pressure",
                              "pressure_hPa" => 200.0),
            vertical.merged_vc,
            FT_TEST)
        @test parsed_vertical.Nz == vertical_merged.Nz
        @test parsed_vertical.groups == vertical_merged.groups

        out_path = joinpath(tmpdir, "out_cs_merged.bin")
        result = process_day(Date(2021, 12, 1), grid, settings, vertical_merged;
                             out_path = out_path,
                             dt_met_seconds = 3600.0,
                             FT = FT_TEST,
                             mass_basis = :dry,
                             replay_tol = 1e-12)
        @test isfile(out_path)
        @test result.worst_replay_rel < 1e-12

        reader = TransportBinaryReader(out_path; FT=FT_TEST)
        try
            @test reader.header.nlevel == 3
            @test reader.header.raw_header["vertical_Nz_output"] == 3
            window = assert_cs_window_replay(reader, 1)
            @test size(window.am[1], 3) == 3
            @test size(window.bm[1], 3) == 3
            @test size(window.cm[1], 3) == 4
        finally
            close(reader)
        end
    end
end
