#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan indexed-baking-valiant Commit 4 — TOML descriptor → typed settings
#
# Verifies that `load_met_settings(toml_path; root_dir, ...)` reads a
# `config/met_sources/*.toml` file and returns the correct concrete
# `AbstractMetSettings` subtype with [preprocessing]-derived defaults
# applied and explicit kwargs overriding.
# ---------------------------------------------------------------------------

using Test
using TOML

import AtmosTransport
using .AtmosTransport.Preprocessing: load_met_settings, GEOSITSettings, GEOSFPSettings,
                                      MERRA2Settings, AbstractGEOSSettings, AbstractMetSettings,
                                      geosfp_native_hourly_ctm_path, merra2_path,
                                      open_merra2_day, close_merra2_day!,
                                      detect_merra2_level_order, NASAArchive, GEOSChemArchive, SurfaceFirst, TopDown, read_merra2_window_fields,
                                      read_merra2_next_day_endpoint, read_merra2_physics_window,
                                      convective_cloud_base!,
                                      has_surface, has_convection, has_vdiff_fields
using Dates
using NCDatasets

const REPO_ROOT = joinpath(@__DIR__, "..", "..")

@testset "Met source loader" begin

    @testset "GEOS-IT TOML → GEOSITSettings" begin
        toml = joinpath(REPO_ROOT, "config", "met_sources", "geosit.toml")
        s = load_met_settings(toml; root_dir = "/tmp/geosit_test")
        @test s isa GEOSITSettings
        @test s isa AbstractGEOSSettings
        @test s isa AbstractMetSettings
        @test s.root_dir          == "/tmp/geosit_test"
        @test s.Nc                == 180
        @test s.mass_flux_dt      == 450.0
        @test s.level_orientation === :auto
        @test s.include_surface === false
        @test s.include_convection === false
        @test endswith(s.coefficients_file, "geos_L72_coefficients.toml")
    end

    @testset "GEOS-FP TOML → native C720 GEOSFPSettings" begin
        toml = joinpath(REPO_ROOT, "config", "met_sources", "geosfp.toml")
        tmp = mktempdir()
        daydir = joinpath(tmp, "20211201")
        mkpath(daydir)
        fname = "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20211201_1330.V01.nc4"
        touch(joinpath(daydir, fname))
        s = load_met_settings(toml; root_dir = tmp)
        @test s isa GEOSFPSettings
        @test s.Nc == 720
        @test s.mass_flux_dt == 450.0
        @test s.include_surface === false
        @test s.include_convection === false
        @test s.physics_dir == ""
        @test s.physics_layout === :auto
        @test geosfp_native_hourly_ctm_path(s, Date(2021, 12, 1), 13) ==
              joinpath(daydir, fname)

        legacy = "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20211201_1400.V01.nc4"
        touch(joinpath(daydir, legacy))
        @test geosfp_native_hourly_ctm_path(s, Date(2021, 12, 1), 14) ==
              joinpath(daydir, legacy)
    end

    @testset "kwargs override TOML defaults" begin
        toml = joinpath(REPO_ROOT, "config", "met_sources", "geosit.toml")
        s = load_met_settings(toml;
                              root_dir = "/tmp/geosit_test",
                              mass_flux_dt = 900.0,
                              level_orientation = :bottom_up,
                              include_surface = true,
                              include_convection = true,
                              physics_dir = "/tmp/geosfp_physics",
                              physics_layout = :latlon_025)
        @test s.mass_flux_dt       == 900.0
        @test s.level_orientation  === :bottom_up
        @test s.include_surface    === true
        @test s.include_convection === true
        @test s.physics_dir        == "/tmp/geosfp_physics"
        @test s.physics_layout     === :latlon_025
    end

    @testset "MERRA-2 layouts and physics settings" begin
        toml = joinpath(REPO_ROOT, "config", "met_sources", "merra2.toml")
        s = load_met_settings(toml; root_dir = "/tmp/merra2_test")
        @test s isa MERRA2Settings
        @test s.archive isa NASAArchive
        @test !s.include_surface && !s.include_convection && !s.include_vdiff_fields
        @test !has_surface(s) && !has_convection(s) && !has_vdiff_fields(s)
        @test merra2_path(s, Date(2021, 12, 1), :inst3) ==
              "/tmp/merra2_test/M2I3NVASM/2021/12/MERRA2_400.inst3_3d_asm_Nv.20211201.nc4"

        gc = load_met_settings(joinpath(REPO_ROOT, "config", "met_sources", "merra2_geoschem.toml");
                               root_dir = "/tmp/merra2_gc")
        @test gc.archive isa GEOSChemArchive
        @test gc.include_surface && gc.include_convection && gc.include_vdiff_fields
        @test gc.include_convective_cloud_base
        @test has_surface(gc) && has_convection(gc) && has_vdiff_fields(gc)
        @test merra2_path(gc, Date(2022, 3, 4), :tavg3) ==
              "/tmp/merra2_gc/2022/03/MERRA2.20220304.A3dyn.05x0625.nc4"
        @test merra2_path(gc, Date(2022, 3, 4), :a3mste) ==
              "/tmp/merra2_gc/2022/03/MERRA2.20220304.A3mstE.05x0625.nc4"

        write_cfg(pre) = begin
            path = tempname() * ".toml"
            write(path, "[source]\nname = \"MERRA-2\"\n[preprocessing]\n" * pre)
            path
        end
        # Physics fields exist only in the GEOS-Chem archive.
        @test_throws ArgumentError load_met_settings(write_cfg("include_convection = true\n");
                                                     root_dir = "/tmp/m")
        # The GEOS-Chem archive has no instantaneous winds.
        @test_throws ArgumentError load_met_settings(
            write_cfg("layout = \"geoschem\"\nwinds_collection = \"inst3\"\n"); root_dir = "/tmp/m")
        @test_throws ArgumentError load_met_settings(write_cfg("layout = \"other\"\n");
                                                     root_dir = "/tmp/m")
        @test_throws ArgumentError load_met_settings(write_cfg("include_tm5_diffusion = true\n");
                                                     root_dir = "/tmp/m")
        # The cloud base needs the convection payload.
        @test_throws ArgumentError load_met_settings(
            write_cfg("layout = \"geoschem\"\ninclude_convective_cloud_base = true\n"); root_dir = "/tmp/m")
    end

    @testset "MERRA-2 level order is detected from QV" begin
        root = mktempdir()
        settings = MERRA2Settings(; root_dir = root, archive = GEOSChemArchive())
        profile = Float32[1e-2 * exp(-(k - 1) / 8) for k in 1:72]   # surface first: moist → dry
        function write_nc(path, vars)
            mkpath(dirname(path))
            NCDataset(path, "c") do ds
                defDim(ds, "lon", 576); defDim(ds, "lat", 361)
                defDim(ds, "lev", 72); defDim(ds, "time", 1)
                for (name, prof) in vars
                    defVar(ds, name, Float32, ("lon", "lat", "lev", "time"))[:, :, :, :] =
                        repeat(reshape(prof, 1, 1, 72, 1), 576, 361, 1, 1)
                end
                defVar(ds, "PS", Float32, ("lon", "lat", "time"))[:, :, :] = fill(1f5, 576, 361, 1)
            end
            return path
        end
        d1, d2 = Date(2021, 12, 1), Date(2021, 12, 2)
        write_nc(merra2_path(settings, d1, :inst3), ("QV" => profile,))            # surface first
        write_nc(merra2_path(settings, d2, :inst3), ("QV" => reverse(profile),))   # top down
        ramp = Float32.(1:72)
        write_nc(merra2_path(settings, d1, :tavg3), ("U" => ramp, "V" => -ramp))

        h = open_merra2_day(settings, d1; next_day_handle = true)
        try
            @test h.level_order isa SurfaceFirst
            @test h.next_level_order isa TopDown
            f = read_merra2_window_fields(h, 1, 72; FT = Float64)
            @test f.qv[1, 1, :] ≈ reverse(profile)                 # top-down after the flip
            @test f.u[1, 1, :] == reverse(ramp)                     # same flip for every collection
            nxt = read_merra2_next_day_endpoint(h, 72; FT = Float64)
            @test nxt.qv[1, 1, :] ≈ reverse(profile)               # already top-down: no flip
        finally
            close_merra2_day!(h)
        end

        flat = write_nc(joinpath(mktempdir(), "flat.nc"), ("QV" => fill(1f-3, 72),))
        NCDataset(flat) do ds
            @test_throws ErrorException detect_merra2_level_order(ds, flat)
        end

        # Physics fields: same flip for the 73 CMFMC edges and the DTRAIN/T
        # layers, A1 surface fields averaged over the window's three hours.
        phys = MERRA2Settings(; root_dir = root, archive = GEOSChemArchive(), include_surface = true,
                              include_convection = true, include_vdiff_fields = true,
                              include_convective_cloud_base = true)
        function write_profiles(path, nlev, vars)
            NCDataset(path, "c") do ds
                defDim(ds, "lon", 576); defDim(ds, "lat", 361)
                defDim(ds, "lev", nlev); defDim(ds, "time", 1)
                for (name, prof) in vars
                    defVar(ds, name, Float32, ("lon", "lat", "lev", "time"))[:, :, :, :] =
                        repeat(reshape(prof, 1, 1, nlev, 1), 576, 361, 1, 1)
                end
            end
        end
        rm(merra2_path(phys, d1, :inst3)); rm(merra2_path(phys, d1, :tavg3))
        write_nc(merra2_path(phys, d1, :inst3), ("QV" => profile, "T" => 200f0 .+ ramp))
        write_profiles(merra2_path(phys, d1, :tavg3), 72, ("U" => ramp, "V" => ramp, "DTRAIN" => ramp))
        edges = Float32.(0:72)
        write_profiles(merra2_path(phys, d1, :a3mste), 73, ("CMFMC" => edges,))
        rain = Float32[k in 5:20 ? 1f-8 : 0f0 for k in 1:72]           # surface first: layers 5-20
        write_profiles(merra2_path(phys, d1, :a3mstc), 72, ("DQRCU" => rain,))
        NCDataset(merra2_path(phys, d1, :a1), "c") do ds
            defDim(ds, "lon", 576); defDim(ds, "lat", 361); defDim(ds, "time", 24)
            for name in ("PBLH", "USTAR", "HFLUX", "EFLUX", "T2M")
                defVar(ds, name, Float32, ("lon", "lat", "time"))[:, :, :] =
                    repeat(reshape(Float32.(100 .* (1:24)), 1, 1, 24), 576, 361, 1)
            end
        end
        h = open_merra2_day(phys, d1; next_day_handle = false)
        try
            @test h.level_order isa SurfaceFirst
            raw = read_merra2_physics_window(h, 1, 72; FT = Float64)
            @test raw.cmfmc[1, 1, :] == reverse(edges)               # TOA edge first
            @test raw.dtrain[1, 1, :] == reverse(ramp)
            @test raw.t[1, 1, :] == 200 .+ reverse(ramp)
            @test size(raw.pblh) == (576, 361, 3)                    # hourly records 1-3
            @test raw.pblh[1, 1, :] == [100, 200, 300]
            @test raw.eflux[1, 1, :] == [100, 200, 300]
            @test raw.dqrcu[1, 1, :] == reverse(rain)
            cb = ntuple(_ -> zeros(1, 1), 6)
            convective_cloud_base!(cb, ntuple(_ -> reshape(raw.dqrcu[1, 1, :], 1, 1, 72), 6))
            @test cb[1][1, 1] == 72 - 5 + 1                          # surface-first layer 5
            convective_cloud_base!(cb, ntuple(_ -> zeros(1, 1, 72), 6))
            @test cb[1][1, 1] == 72                                  # no rain: surface layer
            @test_throws ArgumentError read_merra2_physics_window(h, 9, 72; FT = Float64)
        finally
            close_merra2_day!(h)
        end
    end

    @testset "arco_surface_pressure is ERA5-N320-only" begin
        for (name, why) in (("MERRA-2", "MERRA-2"), ("GEOS-IT", "GEOS"))
            bad = tempname() * ".toml"
            open(bad, "w") do io
                print(io, """
                    [source]
                    name = "$name"
                    [grid]
                    Nc = 180
                    [preprocessing]
                    arco_surface_pressure = true
                    """)
            end
            @test_throws ArgumentError load_met_settings(bad; root_dir = "/tmp/$why")
        end
    end

    @testset "unsupported source name errors loudly" begin
        # Synthesize a tiny TOML with an unknown source name.
        path = tempname() * ".toml"
        open(path, "w") do io
            print(io, """
                [source]
                name = "FAKE-SOURCE"

                [grid]
                Nc = 8
                """)
        end
        @test_throws ErrorException load_met_settings(path; root_dir = "/tmp/x")
    end

    @testset "missing TOML errors with file path" begin
        @test_throws ErrorException load_met_settings("/nonexistent/path.toml";
                                                       root_dir = "/tmp")
    end
end

@testset "MERRA-2 hourly windows: endpoint interpolation and A1 hours" begin
    P = AtmosTransport.Preprocessing
    a = ntuple(p -> Float32.(rand(3, 3) .+ p), 6)
    b = ntuple(p -> Float32.(rand(3, 3) .- p), 6)
    dst = ntuple(_ -> zeros(Float32, 3, 3), 6)
    @test P._lerp_panels!(dst, a, b, 0) == a                       # exact block start
    @test P._lerp_panels!(dst, a, b, 1) == b                       # exact block end
    @test P._lerp_panels!(dst, a, b, 1 / 3)[2] ≈ (2a[2] .+ b[2]) ./ 3
    @test P.merra2_windows_per_block(10800) == 1
    @test P.merra2_windows_per_block(3600) == 3
    @test_throws ArgumentError P.merra2_windows_per_block(1800)

    # Surface fields of window h average the A1 hours inside it.
    settings = MERRA2Settings(; root_dir = "/tmp/x", archive = GEOSChemArchive(), include_surface = true)
    hour(x) = (pblh = ntuple(_ -> fill(x, 2, 2), 6), ustar = ntuple(_ -> fill(x, 2, 2), 6),
               hflux = ntuple(_ -> fill(x, 2, 2), 6), t2m = ntuple(_ -> fill(x, 2, 2), 6))
    qv = ntuple(_ -> zeros(2, 2, 3), 6)
    pipe = (; c180_fields = (; qv), phys = (; surface_hours = (hour(100.0), hour(200.0), hour(300.0))))
    out = P.allocate_merra2_window_physics(settings, 2, 3, Float64)
    window_pblh(h, nsub) = P.merra2_window_physics!(out, settings, pipe, pipe, h, nsub).surface.pblh[1][1, 1]
    @test window_pblh(1, 1) == 200                                 # 3-hourly: mean of hours 1-3
    @test [window_pblh(h, 3) for h in 1:3] == [100, 200, 300]      # hourly: one A1 record each
end
