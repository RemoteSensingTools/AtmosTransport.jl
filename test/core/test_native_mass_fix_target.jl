#!/usr/bin/env julia
#
# `[mass_fix] mode = "initial_endpoint"` pins every window to the dry mass at the
# first window start. Only the GEOS native path implements it; the other native
# sources (MERRA-2, ERA5 N320) must refuse it instead of writing unpinned binaries.

using Test
using AtmosTransport
using .AtmosTransport.Parameters: STANDARD_GRAVITY, GLOBAL_MEAN_DRY_SURFACE_PRESSURE
using .AtmosTransport.Preprocessing: AbstractMetSettings, AbstractGEOSSettings,
    _native_mass_fix_target_kg

struct NonGEOSSource <: AbstractMetSettings end
struct GEOSLikeSource <: AbstractGEOSSettings end

@testset "native [mass_fix] target" begin
    grid = (mesh = (cell_areas = fill(1.0e10, 2, 2),),)     # 4 cells per panel
    pin(mode) = Dict("mass_fix" => Dict("enable" => true, "mode" => mode))

    @test isnan(_native_mass_fix_target_kg(pin("initial_endpoint"), grid, GEOSLikeSource()))
    err = try
        _native_mass_fix_target_kg(pin("initial_endpoint"), grid, NonGEOSSource())
    catch e
        e
    end
    @test err isa ErrorException && occursin("GEOS native sources only", err.msg)

    @test _native_mass_fix_target_kg(pin("target_ps_dry"), grid, NonGEOSSource()) ≈
          GLOBAL_MEAN_DRY_SURFACE_PRESSURE * 6 * 4e10 / STANDARD_GRAVITY
    @test isnan(_native_mass_fix_target_kg(Dict{String, Any}(), grid, NonGEOSSource()))   # pin off

    stale = Dict("mass_fix" => Dict("enable" => true, "target_ps_dry_pa" => 98325.0))
    stale_err = try
        _native_mass_fix_target_kg(stale, grid, NonGEOSSource())
    catch e
        e
    end
    @test stale_err isa ArgumentError
    @test occursin("target_ps_dry_pa", sprint(showerror, stale_err))
    @test occursin("GLOBAL_MEAN_DRY_SURFACE_PRESSURE", sprint(showerror, stale_err))
end

@testset "the header records the configured dry-ps target exactly" begin
    record = AtmosTransport.Preprocessing.mass_fix_target_ps_dry_record
    area = 6 * 2025 * 2.5e10                      # a C45-sized sphere [m²]
    target_kg = GLOBAL_MEAN_DRY_SURFACE_PRESSURE * area / STANDARD_GRAVITY
    for g in (STANDARD_GRAVITY, Float32(STANDARD_GRAVITY))
        @test record(GLOBAL_MEAN_DRY_SURFACE_PRESSURE, target_kg, area, g) ===
              GLOBAL_MEAN_DRY_SURFACE_PRESSURE
    end
    # Without a configured target (GEOS `initial_endpoint`, or a kg-only
    # caller) the record is the mean dry pressure of the pinned mass, which a
    # Float32 `g` moves off the configured value.
    @test record(NaN, target_kg, area, STANDARD_GRAVITY) ≈
          GLOBAL_MEAN_DRY_SURFACE_PRESSURE rtol = 1e-15
    @test record(NaN, target_kg, area, Float32(STANDARD_GRAVITY)) !=
          GLOBAL_MEAN_DRY_SURFACE_PRESSURE
end
