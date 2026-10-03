#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Breakpoint A — ERA5N320Settings + ERA5GRIBDayHandles
#
# Verifies the settings + day-handle surface for the native-GRIB ERA5 source:
#
#   1. ERA5N320Settings is a concrete AbstractERA5GRIBSettings subtype that
#      defaults to top-down level orientation, matching ERA5 on-disk storage.
#   2. era5_grib_path joins root_dir + subdir + filename correctly for each
#      stream, errors on unknown streams, and uses the YYYYMMDD date format.
#   3. open_era5_day asserts today's required files exist (errors when not),
#      reads next-day endpoints when they exist, and tolerates a missing
#      next-day file by setting next_core_path = nothing.
#   4. close_era5_day! / close_day! are idempotent.
#   5. windows_per_day is 24 for every UTC date. The
#      `has_surface / has_convection / has_vdiff_fields` traits report
#      whether the `RawWindow` optional fields can be populated — on this
#      branch the answer is unconditionally `false` for ERA5 because the
#      per-window pipeline produces its own structured output rather than
#      a `RawWindow`, so the traits stay `false` even when the matching
#      `include_*` settings flag is `true`.
#
# Hermetic. Uses mktempdir + touched placeholder GRIBs so the test runs in
# CI without any real-data dependency. A separate `@testset` runs only when
# the real N320 archive is on disk to keep the local-only smoke documented.
# ---------------------------------------------------------------------------

using Test
using Dates

import AtmosTransport
using .AtmosTransport.Preprocessing: AbstractMetSettings,
                                      AbstractERA5GRIBSettings,
                                      ERA5GRIBSettings, ERA5N320Settings,
                                      ERA5GRIBDayHandles,
                                      open_day, close_day!, open_era5_day, close_era5_day!,
                                      era5_grib_path,
                                      load_met_settings,
                                      windows_per_day, has_convection,
                                      has_surface, has_vdiff_fields

"""Materialise an empty placeholder file at `path`, creating its parent dir.

Path resolution only checks `isfile`, so zero-byte placeholders are enough
to drive the reader through every branch without staging real GRIBs.
"""
function _placeholder!(path::AbstractString)
    mkpath(dirname(path))
    touch(path)
    return path
end

"""Materialise placeholder GRIBs for `dates` × `streams` under `root`."""
function _materialise_placeholders(root::AbstractString,
                                   dates::AbstractVector{<:Date},
                                   streams::AbstractVector{Symbol})
    settings = ERA5N320Settings(; root_dir = root)
    for date in dates, stream in streams
        _placeholder!(era5_grib_path(settings, date, stream))
    end
    return settings
end

@testset "ERA5 native-GRIB reader — breakpoint A" begin

    @testset "Type hierarchy + defaults" begin
        @test ERA5N320Settings <: AbstractERA5GRIBSettings <: AbstractMetSettings
        @test ERA5N320Settings === ERA5GRIBSettings{:n320}

        s = ERA5N320Settings(; root_dir = "/tmp/era5-test")
        @test s.root_dir              == "/tmp/era5-test"
        @test s.include_surface       === false
        @test s.include_convection    === false
        @test s.level_orientation     === :top_down
        @test endswith(s.coefficients_file, "era5_L137_coefficients.toml")
    end

    @testset "era5_grib_path resolution" begin
        s = ERA5N320Settings(; root_dir = "/tmp/era5-test")
        date = Date(2021, 12, 1)

        @test era5_grib_path(s, date, :core) ==
              joinpath("/tmp/era5-test", "ml_an_native_core", "era5_core_20211201.grib")
        @test era5_grib_path(s, date, :convection) ==
              joinpath("/tmp/era5-test", "ml_fc_convection", "era5_convection_20211201.grib")
        @test era5_grib_path(s, date, :surface) ==
              joinpath("/tmp/era5-test", "sfc_an_native", "era5_surface_20211201.grib")

        # Unknown stream is a hard error, not a silent typo.
        @test_throws ArgumentError era5_grib_path(s, date, :wind)
    end

    @testset "open_era5_day — required files + endpoints" begin
        mktempdir() do root
            settings = _materialise_placeholders(root,
                [Date(2021, 12, 1), Date(2021, 12, 2)],
                [:core, :convection, :surface])

            settings_full = ERA5N320Settings(;
                root_dir = root,
                include_surface = true,
                include_convection = true,
            )

            h = open_era5_day(settings_full, Date(2021, 12, 1))
            @test h isa ERA5GRIBDayHandles{typeof(settings_full)}
            @test h.date == Date(2021, 12, 1)
            @test isfile(h.core_path)
            @test h.convection_path !== nothing && isfile(h.convection_path)
            @test h.surface_path    !== nothing && isfile(h.surface_path)
            @test h.next_core_path  !== nothing && isfile(h.next_core_path)
            @test endswith(h.next_core_path, "era5_core_20211202.grib")

            # Generic open_day dispatch lands on the same code path.
            h2 = open_day(settings_full, Date(2021, 12, 1))
            @test h2.core_path == h.core_path
        end
    end

    @testset "open_era5_day — gated streams stay nothing" begin
        mktempdir() do root
            # Only the `core` stream exists on disk; surface/convection are gated.
            _materialise_placeholders(root, [Date(2021, 12, 1)], [:core])
            settings_core_only = ERA5N320Settings(; root_dir = root)

            h = open_era5_day(settings_core_only, Date(2021, 12, 1))
            @test h.convection_path === nothing
            @test h.surface_path    === nothing
            @test h.next_core_path  === nothing   # no next-day file on disk
        end
    end

    @testset "open_era5_day — ARCO per-variable surface layout" begin
        mktempdir() do root
            date = Date(2021, 12, 1)
            _materialise_placeholders(root, [date], [:core])
            surface_dir = joinpath(root, "sfc_an_native", "arco", "20211201")
            mkpath(surface_dir)
            _placeholder!(joinpath(surface_dir, "surface_pressure.nc"))

            settings = ERA5N320Settings(;
                root_dir = root,
                include_surface = true,
                arco_surface_pressure = true,
            )
            h = open_era5_day(settings, date; next_day_handle = false)
            @test h.surface_path == surface_dir
            @test h.arco_sp_path == joinpath(surface_dir, "surface_pressure.nc")
        end
    end

    @testset "open_era5_day — missing required files are errors" begin
        mktempdir() do root
            # Nothing on disk → core read must error, and the message must
            # name the missing path so logs are debuggable.
            settings = ERA5N320Settings(; root_dir = root)
            @test_throws "ERA5 core GRIB not found" open_era5_day(settings, Date(2021, 12, 1))
            @test_throws "era5_core_20211201.grib"  open_era5_day(settings, Date(2021, 12, 1))

            _materialise_placeholders(root, [Date(2021, 12, 1)], [:core])

            # Convection requested but missing.
            s_conv = ERA5N320Settings(; root_dir = root, include_convection = true)
            @test_throws "ERA5 convection GRIB not found" open_era5_day(s_conv, Date(2021, 12, 1))

            # Surface requested but missing in BOTH layouts (GRIB and ARCO).
            s_surf = ERA5N320Settings(; root_dir = root, include_surface = true)
            @test_throws "ERA5 surface data not found" open_era5_day(s_surf, Date(2021, 12, 1))
            @test_throws "era5_surface_20211201.grib" open_era5_day(s_surf, Date(2021, 12, 1))
            @test_throws "arco" open_era5_day(s_surf, Date(2021, 12, 1))

            # An ARCO directory with no netCDFs does not satisfy the gate.
            arco_dir = joinpath(root, "sfc_an_native", "arco", "20211201")
            mkpath(arco_dir)
            @test_throws "ERA5 surface data not found" open_era5_day(s_surf, Date(2021, 12, 1))
        end
    end

    @testset "open_era5_day — ARCO per-variable surface layout satisfies include_surface" begin
        mktempdir() do root
            _materialise_placeholders(root, [Date(2021, 12, 1)], [:core])
            arco_dir = joinpath(root, "sfc_an_native", "arco", "20211201")
            _placeholder!(joinpath(arco_dir, "boundary_layer_height.nc"))

            s_surf = ERA5N320Settings(; root_dir = root, include_surface = true)
            h = open_era5_day(s_surf, Date(2021, 12, 1))
            @test h.surface_path == arco_dir

            # A surface GRIB, when present, still wins over the ARCO layout.
            _materialise_placeholders(root, [Date(2021, 12, 1)], [:surface])
            h2 = open_era5_day(s_surf, Date(2021, 12, 1))
            @test endswith(h2.surface_path, "era5_surface_20211201.grib")
        end
    end

    @testset "open_era5_day — level_orientation is validated" begin
        mktempdir() do root
            _materialise_placeholders(root, [Date(2021, 12, 1)], [:core])
            s_bad = ERA5N320Settings(;
                root_dir = root, level_orientation = :sideways)
            @test_throws ArgumentError open_era5_day(s_bad, Date(2021, 12, 1))
        end
    end

    @testset "next_day_handle=false skips the endpoint lookup" begin
        mktempdir() do root
            _materialise_placeholders(root,
                [Date(2021, 12, 1), Date(2021, 12, 2)], [:core])
            s = ERA5N320Settings(; root_dir = root)

            h = open_era5_day(s, Date(2021, 12, 1); next_day_handle = false)
            @test h.next_core_path === nothing
        end
    end

    @testset "close_day! is idempotent" begin
        mktempdir() do root
            _materialise_placeholders(root, [Date(2021, 12, 1)], [:core])
            s = ERA5N320Settings(; root_dir = root)
            h = open_era5_day(s, Date(2021, 12, 1); next_day_handle = false)

            @test close_era5_day!(h) === nothing
            @test close_era5_day!(h) === nothing
            @test close_day!(h)      === nothing
        end
    end

    @testset "load_met_settings — ERA5-N320 dispatch" begin
        mktempdir() do root
            toml_path = joinpath(root, "era5_n320.toml")
            open(toml_path, "w") do io
                println(io, """
                [source]
                name = "ERA5-N320"

                [preprocessing]
                include_surface = true
                include_convection = true
                """)
            end

            s = load_met_settings(toml_path; root_dir = root)
            @test s isa ERA5N320Settings
            @test s.root_dir             == root
            @test s.include_surface      === true
            @test s.include_convection   === true
            @test s.level_orientation    === :top_down

            # Per-call kwargs override TOML values.
            s_override = load_met_settings(toml_path;
                root_dir = root, include_surface = false)
            @test s_override.include_surface === false

            open(toml_path, "a") do io
                println(io, "include_vdiff_fields = false")
            end
            @test_throws ArgumentError load_met_settings(toml_path; root_dir = root)
        end
    end

    @testset "load_met_settings — unknown source name errors clearly" begin
        mktempdir() do root
            toml_path = joinpath(root, "bogus.toml")
            open(toml_path, "w") do io
                println(io, """
                [source]
                name = "NONEXISTENT-MET-SOURCE"
                """)
            end
            @test_throws "Unsupported met source" load_met_settings(toml_path; root_dir = root)
        end
    end

    @testset "Trait predicates + windows_per_day" begin
        s_off = ERA5N320Settings(; root_dir = "/tmp/era5-test")
        s_on  = ERA5N320Settings(;
            root_dir = "/tmp/era5-test",
            include_surface = true,
            include_convection = true,
        )

        # Both `s_off` and `s_on` advertise `false` for all three traits
        # until the corresponding RawWindow writer paths land. The traits
        # describe what RawWindow can carry, not what `open_era5_day` reads.
        for s in (s_off, s_on)
            @test has_surface(s)      === false
            @test has_convection(s)   === false
            @test has_vdiff_fields(s) === false
        end

        # 24 hourly windows, regardless of date.
        @test windows_per_day(s_off, Date(2021, 12, 1)) == 24
        @test windows_per_day(s_off, Date(2024,  2, 29)) == 24
    end

    # Opt-in real-data smoke. The full N320 download archive is ~150 GB and
    # only present on the workstation, so we keep this off by default to keep
    # CI hermetic. Toggle by setting ATMOS_ERA5_N320_ROOT to the archive root.
    real_root_env = get(ENV, "ATMOS_ERA5_N320_ROOT", "")
    if !isempty(real_root_env) && isdir(real_root_env)
        @testset "Real N320 archive smoke ($(real_root_env))" begin
            settings = ERA5N320Settings(;
                root_dir = real_root_env,
                include_surface = true,
                include_convection = true,
            )
            h = open_era5_day(settings, Date(2021, 12, 1))
            @test isfile(h.core_path)
            @test h.convection_path !== nothing && isfile(h.convection_path)
            @test h.surface_path    !== nothing && isfile(h.surface_path)
        end
    else
        @info "Skipping real N320 archive smoke (set ATMOS_ERA5_N320_ROOT to enable)."
        @test_skip false
    end

end
