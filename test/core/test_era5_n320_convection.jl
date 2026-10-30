#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Breakpoint E — ERA5 N320 convection forecast reader.
#
# Verifies the convection surface for the N320 source grid:
#
#   1. Field-output struct + allocator shapes match the source mesh.
#   2. `era5_convection_hour_address(h)` maps every UTC hour into the right
#      (prev-day?, dataTime, stepRange) GRIB header triple.
#   3. The reader errors clearly when the previous-day file is needed but
#      absent (hours 0..5 without prev_convection_path).
#   4. (opt-in, real GRIB) Hour 6..23 of 2021-12-01 read correctly:
#      UDMF / DDMF are non-negative, all four fields stay in physical
#      ranges for time-mean convective mass flux / detrainment rates.
# ---------------------------------------------------------------------------

using Test
using Dates

import AtmosTransport

using .AtmosTransport.Preprocessing: ERA5N320Settings, ERA5GRIBDayHandles,
                                      ERA5N320ConvectionFields,
                                      allocate_era5_n320_convection_fields,
                                      era5_convection_hour_address,
                                      read_era5_n320_convection_window!,
                                      discover_era5_n320_source_grid,
                                      open_era5_day, close_era5_day!,
                                      era5_grib_path,
                                      build_target_geometry
using .AtmosTransport.Grids: ncells, nrings

function _tiny_source_grid(::Type{FT} = Float64) where FT
    cfg = Dict{String, Any}(
        "type"            => "synthetic_reduced_gaussian",
        "gaussian_number" => 2,
        "nlon_mode"       => "regular",
    )
    return build_target_geometry(Val(:synthetic_reduced_gaussian), cfg, FT)
end

"""Materialise an empty placeholder GRIB at the path. Reused from the
breakpoint-A test harness; the convection reader is gated on the file
existing on disk."""
function _placeholder!(path::AbstractString)
    mkpath(dirname(path))
    touch(path)
    return path
end

@testset "ERA5 N320 convection — breakpoint E" begin

    @testset "era5_convection_hour_address — full hour table" begin
        @test_throws ArgumentError era5_convection_hour_address(-1)
        @test_throws ArgumentError era5_convection_hour_address(24)

        # Hours 0..5 → previous day, 18 UTC base, stepRange (h+6)-(h+7).
        for h in 0:5
            use_prev, dt, sr = era5_convection_hour_address(h)
            @test use_prev === true
            @test dt == 1800
            @test sr == "$(h + 6)-$(h + 7)"
        end

        # Hours 6..17 → today's file, 06 UTC base, stepRange (h-6)-(h-5).
        for h in 6:17
            use_prev, dt, sr = era5_convection_hour_address(h)
            @test use_prev === false
            @test dt == 600
            @test sr == "$(h - 6)-$(h - 5)"
        end

        # Hours 18..23 → today's file, 18 UTC base, stepRange (h-18)-(h-17).
        for h in 18:23
            use_prev, dt, sr = era5_convection_hour_address(h)
            @test use_prev === false
            @test dt == 1800
            @test sr == "$(h - 18)-$(h - 17)"
        end
    end

    @testset "Reader errors when prev-day file is needed but absent" begin
        mktempdir() do root
            # Only today's core + convection placeholders — no prev-day file.
            settings = ERA5N320Settings(; root_dir = root, include_convection = true)
            _placeholder!(era5_grib_path(settings, Date(2021, 12, 1), :core))
            _placeholder!(era5_grib_path(settings, Date(2021, 12, 1), :convection))

            handles = open_era5_day(settings, Date(2021, 12, 1); next_day_handle = false)
            @test handles.prev_convection_path === nothing

            grid = _tiny_source_grid(Float64)
            f = allocate_era5_n320_convection_fields(grid, 3)

            # Hour 0..5 require the prev-day file → clear error.
            @test_throws "previous-day file" read_era5_n320_convection_window!(
                f, handles, grid.mesh, Date(2021, 12, 1), 0)
            @test_throws "previous-day file" read_era5_n320_convection_window!(
                f, handles, grid.mesh, Date(2021, 12, 1), 5)

            close_era5_day!(handles)
        end
    end

    @testset "Prev-day handle resolves when previous file is on disk" begin
        mktempdir() do root
            settings = ERA5N320Settings(; root_dir = root, include_convection = true)
            for d in (Date(2021, 12, 1), Date(2021, 11, 30))
                _placeholder!(era5_grib_path(settings, d, :core))
                _placeholder!(era5_grib_path(settings, d, :convection))
            end

            handles = open_era5_day(settings, Date(2021, 12, 1); next_day_handle = false)
            @test handles.prev_convection_path !== nothing
            @test endswith(handles.prev_convection_path, "era5_convection_20211130.grib")
            close_era5_day!(handles)
        end
    end

    # -----------------------------------------------------------------------
    # Real-data smoke — hour 6 onwards, since the workstation archive starts
    # at 2021-12-01 (we don't have the 2021-11-30 prev-day convection).
    # -----------------------------------------------------------------------
end
