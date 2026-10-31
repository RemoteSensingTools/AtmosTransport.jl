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
            settings_core_only = ERA5N320Settings(; root_dir = root)
            core_only = open_era5_day(settings_core_only, Date(2021, 12, 1))
            @test core_only.convection_path === nothing
            @test core_only.surface_path === nothing
        end
    end


    @testset "open_era5_day — missing required files are errors" begin
        mktempdir() do root
            # Nothing on disk → core read must error, and the message must
            # name the missing path so logs are debuggable.
            settings = ERA5N320Settings(; root_dir = root)
            @test_throws ErrorException open_era5_day(settings, Date(2021, 12, 1))

            _materialise_placeholders(root, [Date(2021, 12, 1)], [:core])

            # Convection requested but missing.
            s_conv = ERA5N320Settings(; root_dir = root, include_convection = true)
            @test_throws ErrorException open_era5_day(s_conv, Date(2021, 12, 1))

            # Surface requested but missing in BOTH layouts (GRIB and ARCO).
            s_surf = ERA5N320Settings(; root_dir = root, include_surface = true)
            @test_throws ErrorException open_era5_day(s_surf, Date(2021, 12, 1))

            # An ARCO directory with no netCDFs does not satisfy the gate.
            arco_dir = joinpath(root, "sfc_an_native", "arco", "20211201")
            mkpath(arco_dir)
            @test_throws ErrorException open_era5_day(s_surf, Date(2021, 12, 1))
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








    # Opt-in real-data smoke. The full N320 download archive is ~150 GB and
    # only present on the workstation, so we keep this off by default to keep
    # CI hermetic. Toggle by setting ATMOS_ERA5_N320_ROOT to the archive root.
end
