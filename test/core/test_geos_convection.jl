#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan geos-followups Section C — GCHP-style convection (CMFMC + DTRAIN)
# wiring through the GEOS source path.
#
# Verifies:
#   1. `has_convection(::GEOSSettings)` follows `settings.include_convection`.
#   2. `allocate_raw_window` allocates `cmfmc` / `dtrain` slots iff convection
#      is enabled (and leaves them `nothing` otherwise).
#   3. `read_window!` populates `raw.cmfmc` / `raw.dtrain` from A3mstE / A3dyn
#      under the 3-hourly hold-constant binding (windows 1-3 → idx 1, …,
#      windows 22-24 → idx 8). Level orientation is flipped consistently.
#   4. `process_day` emits `:cmfmc` and `:dtrain` payload sections in the
#      v4 binary header when convection is on.
#   5. With convection OFF the binary's `payload_sections` excludes
#      `:cmfmc` / `:dtrain` (no silent capability mismatch).
# ---------------------------------------------------------------------------

using Test
using Dates
using NCDatasets

import AtmosTransport
using .AtmosTransport.Grids: GEOSNativePanelConvention
using .AtmosTransport.Preprocessing: GEOSITSettings, open_day, close_day!,
                                      read_window!, allocate_raw_window,
                                      has_convection, has_pbl_eflux, process_day,
                                      build_target_geometry,
                                      load_hybrid_coefficients
using .AtmosTransport.MetDrivers: TransportBinaryReader

const FT_TEST = Float64
const NC = 8
const NPANEL = 6
const NZ = 4
const NWIN_DAY = 24
const NA3 = 8                                # 3-hourly cadence

# ---------------------------------------------------------------------------
# Synthetic A3 fixture writer.
# Writes A3mstE.CMFMC (interfaces, Nz+1 lev) and A3dyn.DTRAIN (centers, Nz lev)
# alongside the existing CTM_A1 / CTM_I1 fixtures from `test_geos_reader.jl`.
# Each A3 timestep gets a unique constant value so the per-window binding is
# observable in the test.
# ---------------------------------------------------------------------------

function _synthetic_hybrid_coefs_c(nz::Int)
    nz == 4 || error("synthetic coefs only defined for nz=4 here")
    return [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.05, 0.20, 0.55, 1.0]
end

function _defvar3d!(ds, name, A; units::String="")
    v = defVar(ds, name, eltype(A), ("Xdim", "Ydim", "nf", "lev", "time");
               attrib = ["units" => units])
    v[:, :, :, :, :] = A
    return v
end

function _defvar2d_xyz_t!(ds, name, A; units::String="")
    v = defVar(ds, name, eltype(A), ("Xdim", "Ydim", "nf", "time");
               attrib = ["units" => units])
    v[:, :, :, :] = A
    return v
end

"""
Write a synthetic GEOS-IT day under `dir` including A3mstE / A3dyn.
CMFMC at A3 time index `t` is set to the constant value `100.0 + t`
(t=1..8) so we can verify the 3-hourly-hold-constant binding. DTRAIN
similarly is `200.0 + t`.

Levels: bottom-up convention (k=1 = surface) so the reader's flip
exercises real code.
"""
function _write_synthetic_geosit_day_with_a3!(dir::String, datestr::String;
                                              ntimes_a1::Int = 24,
                                              ntimes_i1::Int = 25)
    mkpath(dir)
    A_top, B_top = _synthetic_hybrid_coefs_c(NZ)
    delp_per_layer_bu = Float64[]
    for k in NZ:-1:1
        ΔA = A_top[k+1] - A_top[k]
        ΔB = B_top[k+1] - B_top[k]
        push!(delp_per_layer_bu, ΔA + ΔB * 100_000.0)
    end

    # CTM_A1 — same as test_geos_cs_passthrough's fixture
    a1 = joinpath(dir, "GEOSIT.$(datestr).CTM_A1.C$(NC).nc")
    NCDataset(a1, "c") do ds
        ds.dim["Xdim"] = NC; ds.dim["Ydim"] = NC; ds.dim["nf"] = NPANEL
        ds.dim["lev"]  = NZ; ds.dim["time"] = ntimes_a1
        delp_bu = fill(0.0, NC, NC, NPANEL, NZ, ntimes_a1)
        for k in 1:NZ
            delp_bu[:, :, :, k, :] .= delp_per_layer_bu[k]
        end
        _defvar3d!(ds, "DELP", delp_bu;             units = "Pa")
        _defvar3d!(ds, "MFXC", zeros(NC, NC, NPANEL, NZ, ntimes_a1); units = "Pa m+2")
        _defvar3d!(ds, "MFYC", zeros(NC, NC, NPANEL, NZ, ntimes_a1); units = "Pa m+2")
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib = ["positive" => "down"])
    end

    # CTM_I1
    i1 = joinpath(dir, "GEOSIT.$(datestr).CTM_I1.C$(NC).nc")
    NCDataset(i1, "c") do ds
        ds.dim["Xdim"] = NC; ds.dim["Ydim"] = NC; ds.dim["nf"] = NPANEL
        ds.dim["lev"]  = NZ; ds.dim["time"] = ntimes_i1
        _defvar2d_xyz_t!(ds, "PS", fill(1000.0, NC, NC, NPANEL, ntimes_i1); units = "hPa")
        _defvar3d!(ds, "QV", fill(0.005, NC, NC, NPANEL, NZ, ntimes_i1);    units = "kg kg-1")
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib = ["positive" => "down"])
    end

    # A1 — hourly surface PBL fields for diffusion.
    a1surf = joinpath(dir, "GEOSIT.$(datestr).A1.C$(NC).nc")
    NCDataset(a1surf, "c") do ds
        ds.dim["Xdim"] = NC; ds.dim["Ydim"] = NC; ds.dim["nf"] = NPANEL
        ds.dim["time"] = ntimes_a1
        _defvar2d_xyz_t!(ds, "PBLH",  fill(1000.0, NC, NC, NPANEL, ntimes_a1); units = "m")
        _defvar2d_xyz_t!(ds, "USTAR", fill(0.30,   NC, NC, NPANEL, ntimes_a1); units = "m s-1")
        _defvar2d_xyz_t!(ds, "HFLUX", fill(120.0,  NC, NC, NPANEL, ntimes_a1); units = "W m-2")
        _defvar2d_xyz_t!(ds, "T2M",   fill(295.0,  NC, NC, NPANEL, ntimes_a1); units = "K")
        _defvar2d_xyz_t!(ds, "EFLUX", fill(45.0,   NC, NC, NPANEL, ntimes_a1); units = "W m-2")
    end

    # A3mstE — CMFMC at interfaces (NZ+1 lev), 8 timesteps
    a3mste = joinpath(dir, "GEOSIT.$(datestr).A3mstE.C$(NC).nc")
    NCDataset(a3mste, "c") do ds
        ds.dim["Xdim"] = NC; ds.dim["Ydim"] = NC; ds.dim["nf"] = NPANEL
        ds.dim["lev"]  = NZ + 1; ds.dim["time"] = NA3
        cmfmc = zeros(Float32, NC, NC, NPANEL, NZ + 1, NA3)
        for t in 1:NA3
            cmfmc[:, :, :, :, t] .= Float32(100.0 + t)
        end
        _defvar3d!(ds, "CMFMC", cmfmc; units = "kg m-2 s-1")
        defVar(ds, "lev", collect(1:(NZ + 1)), ("lev",); attrib = ["positive" => "down"])
    end

    # A3dyn — DTRAIN at centers (NZ lev), 8 timesteps
    a3dyn = joinpath(dir, "GEOSIT.$(datestr).A3dyn.C$(NC).nc")
    NCDataset(a3dyn, "c") do ds
        ds.dim["Xdim"] = NC; ds.dim["Ydim"] = NC; ds.dim["nf"] = NPANEL
        ds.dim["lev"]  = NZ; ds.dim["time"] = NA3
        dtrain = zeros(Float32, NC, NC, NPANEL, NZ, NA3)
        for t in 1:NA3
            dtrain[:, :, :, :, t] .= Float32(200.0 + t)
        end
        _defvar3d!(ds, "DTRAIN", dtrain; units = "kg m-2 s-1")
        # Horizontal winds for the GEOS-Chem VDIFF fields (include_vdiff_fields)
        _defvar3d!(ds, "U", fill(Float32(5.0),  NC, NC, NPANEL, NZ, NA3); units = "m s-1")
        _defvar3d!(ds, "V", fill(Float32(-3.0), NC, NC, NPANEL, NZ, NA3); units = "m s-1")
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib = ["positive" => "down"])
    end

    # I3 — 3-hourly temperature for the GEOS-Chem VDIFF fields
    i3 = joinpath(dir, "GEOSIT.$(datestr).I3.C$(NC).nc")
    NCDataset(i3, "c") do ds
        ds.dim["Xdim"] = NC; ds.dim["Ydim"] = NC; ds.dim["nf"] = NPANEL
        ds.dim["lev"]  = NZ; ds.dim["time"] = NA3
        _defvar3d!(ds, "T", fill(Float32(280.0), NC, NC, NPANEL, NZ, NA3); units = "K")
        defVar(ds, "lev", collect(1:NZ), ("lev",); attrib = ["positive" => "down"])
    end

    return a1, i1, a1surf, a3mste, a3dyn
end

function _write_synthetic_coefs_toml!(path::String)
    A, B = _synthetic_hybrid_coefs_c(NZ)
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

@testset "GEOS convection wiring (Section C)" begin
    tmpdir   = mktempdir()
    daydir   = joinpath(tmpdir, "20211201")
    nextdir  = joinpath(tmpdir, "20211202")
    _write_synthetic_geosit_day_with_a3!(daydir,  "20211201")
    _write_synthetic_geosit_day_with_a3!(nextdir, "20211202")
    coef_path = joinpath(tmpdir, "synthetic_coefs.toml")
    _write_synthetic_coefs_toml!(coef_path)

    settings_off = GEOSITSettings(
        root_dir = tmpdir, Nc = NC,
        coefficients_file = coef_path,
        include_convection = false,
    )
    settings_on = GEOSITSettings(
        root_dir = tmpdir, Nc = NC,
        coefficients_file = coef_path,
        include_convection = true,
    )
    settings_full = GEOSITSettings(
        root_dir = tmpdir, Nc = NC,
        coefficients_file = coef_path,
        include_surface = true,
        include_convection = true,
    )
    settings_vdiff = GEOSITSettings(
        root_dir = tmpdir, Nc = NC,
        coefficients_file = coef_path,
        include_surface = true,
        include_vdiff_fields = true,
    )

    grid = build_target_geometry(
        Dict{String,Any}("type" => "cubed_sphere", "Nc" => NC,
                          "panel_convention" => "geos_native"),
        FT_TEST,
    )
    vertical = (merged_vc = load_hybrid_coefficients(coef_path),
                Nz = NZ, Nz_native = NZ)

    @testset "has_convection follows settings.include_convection" begin
        @test has_convection(settings_off) === false
        @test has_convection(settings_on)  === true
    end

    @testset "allocate_raw_window respects include_convection" begin
        raw_off = allocate_raw_window(settings_off; FT = FT_TEST, Nz = NZ)
        @test raw_off.cmfmc  === nothing
        @test raw_off.dtrain === nothing

        raw_on  = allocate_raw_window(settings_on; FT = FT_TEST, Nz = NZ)
        @test raw_on.cmfmc  isa NTuple{6, Array{FT_TEST, 3}}
        @test raw_on.dtrain isa NTuple{6, Array{FT_TEST, 3}}
        @test size(raw_on.cmfmc[1])  == (NC, NC, NZ + 1)        # interfaces
        @test size(raw_on.dtrain[1]) == (NC, NC, NZ)            # centers

        raw_full = allocate_raw_window(settings_full; FT = FT_TEST, Nz = NZ)
        @test raw_full.surface !== nothing
        @test size(raw_full.surface.pblh[1]) == (NC, NC)
    end

    @testset "3-hourly hold-constant binding + dry-basis correction" begin
        # Synthetic CMFMC moist value at A3 idx t = 100+t (t=1..8); DTRAIN
        # similarly is 200+t. Synthetic qv = 0.005 (constant), so the dry
        # factor is (1 − 0.005) = 0.995 uniformly. Verify:
        #   1. Window→A3 binding: windows 1-3 → t=1, 4-6 → t=2, …, 22-24 → t=8.
        #   2. Dry-basis correction is applied (× 0.995).
        # If either changes we expect the literal product to fail to within
        # 1e-12, catching unit / basis regressions explicitly.
        dry_factor = 1.0 - 0.005
        handles = open_day(settings_on, Date(2021, 12, 1))
        try
            raw = allocate_raw_window(settings_on; FT = FT_TEST, Nz = NZ)
            for win in [1, 3, 4, 6, 7, 22, 24]
                read_window!(raw, settings_on, handles, Date(2021, 12, 1), win)
                expected_a3_idx = (win - 1) ÷ 3 + 1
                @test all(isapprox.(raw.cmfmc[1],
                                    Float64(100 + expected_a3_idx) * dry_factor;
                                    rtol = 1e-12))
                @test all(isapprox.(raw.dtrain[1],
                                    Float64(200 + expected_a3_idx) * dry_factor;
                                    rtol = 1e-12))
            end
        finally
            close_day!(handles)
        end
    end

    @testset "surface PBL fields are read and written" begin
        handles = open_day(settings_full, Date(2021, 12, 1))
        try
            raw = allocate_raw_window(settings_full; FT = FT_TEST, Nz = NZ)
            read_window!(raw, settings_full, handles, Date(2021, 12, 1), 1)
            @test all(raw.surface.pblh[1]  .== 1000.0)
            @test all(raw.surface.ustar[1] .== 0.30)
            @test all(raw.surface.hflux[1] .== 120.0)
            @test all(raw.surface.t2m[1]   .== 295.0)
        finally
            close_day!(handles)
        end

        out_full = joinpath(tmpdir, "with_surface_and_conv.bin")
        process_day(Date(2021, 12, 1), grid, settings_full, vertical;
                    out_path = out_full,
                    dt_met_seconds = 3600.0, FT = FT_TEST,
                    mass_basis = :dry, replay_tol = 1e-12)
        reader = TransportBinaryReader(out_full; FT = FT_TEST)
        try
            @test :pblh  in reader.header.payload_sections
            @test :ustar in reader.header.payload_sections
            @test :pbl_hflux in reader.header.payload_sections
            @test :t2m   in reader.header.payload_sections
            win = AtmosTransport.MetDrivers.load_window!(reader, 1)
            @test win.surface.pblh[1][1, 1] == 1000.0
            @test win.surface.t2m[1][1, 1] == 295.0
        finally
            close(reader)
        end
    end

    @testset "GEOS-IT EFLUX reaches the binary with the VDIFF fields" begin
        # geoschem_nonlocal_vdiff needs :pbl_eflux; GEOS-IT A1 carries EFLUX.
        @test has_pbl_eflux(settings_vdiff) === true
        @test has_pbl_eflux(settings_full) === false     # no VDIFF fields requested
        handles = open_day(settings_vdiff, Date(2021, 12, 1))
        try
            raw = allocate_raw_window(settings_vdiff; FT = FT_TEST, Nz = NZ)
            read_window!(raw, settings_vdiff, handles, Date(2021, 12, 1), 1)
            @test all(p -> all(==(45.0), p), raw.surface.eflux)
        finally
            close_day!(handles)
        end

        out_vdiff = joinpath(tmpdir, "with_vdiff.bin")
        process_day(Date(2021, 12, 1), grid, settings_vdiff, vertical;
                    out_path = out_vdiff,
                    dt_met_seconds = 3600.0, FT = FT_TEST,
                    mass_basis = :dry, replay_tol = 1e-12)
        reader = TransportBinaryReader(out_vdiff; FT = FT_TEST)
        try
            @test :pbl_eflux in reader.header.payload_sections
            @test has_pbl_eflux(reader)
            win = AtmosTransport.MetDrivers.load_window!(reader, 1)
            @test win.surface.eflux[1][1, 1] == 45.0
        finally
            close(reader)
        end
    end

    @testset "binary payload includes :cmfmc / :dtrain when on" begin
        out_on  = joinpath(tmpdir, "with_conv.bin")
        process_day(Date(2021, 12, 1), grid, settings_on, vertical;
                    out_path = out_on,
                    dt_met_seconds = 3600.0, FT = FT_TEST,
                    mass_basis = :dry, replay_tol = 1e-12)
        reader = TransportBinaryReader(out_on; FT = FT_TEST)
        try
            @test :cmfmc  in reader.header.payload_sections
            @test :dtrain in reader.header.payload_sections
        finally
            close(reader)
        end
    end

    @testset "binary excludes :cmfmc / :dtrain when off" begin
        out_off = joinpath(tmpdir, "no_conv.bin")
        process_day(Date(2021, 12, 1), grid, settings_off, vertical;
                    out_path = out_off,
                    dt_met_seconds = 3600.0, FT = FT_TEST,
                    mass_basis = :dry, replay_tol = 1e-12)
        reader = TransportBinaryReader(out_off; FT = FT_TEST)
        try
            @test !(:cmfmc  in reader.header.payload_sections)
            @test !(:dtrain in reader.header.payload_sections)
        finally
            close(reader)
        end
    end
end
