"""
Unit tests for `PerTracerFluxMap` + `SurfaceFluxSource` — plan 17 Commit 2.

Validates:
- Construction variants (variadic, tuple, vector of sources)
- Duplicate tracer names are rejected at construction
- Non-`SurfaceFluxSource` entries are rejected
- `flux_for(map, :name)` returns the source, or `nothing` for absent
- Iteration and indexing behaviour
- `tracer_names(map)` matches storage order
- Backward-compat: `SurfaceFluxSource` still usable by-name after
  the plan 17 Commit 2 migration from `src/Models/` to `src/Operators/SurfaceFlux/`
- `Adapt.adapt_structure` path: CPU passthrough preserves structure
  and per-source rate arrays
- GPU adaptation (when CUDA.functional()): rate arrays become CuArrays
"""

using Test
using AtmosTransport
using AtmosTransport: SurfaceFluxSource, PerTracerFluxMap, flux_for
using AtmosTransport.Operators.SurfaceFlux: tracer_names
using Adapt

const HAS_GPU_FLUX = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

@testset "PerTracerFluxMap — plan 17 Commit 2" begin

    @testset "construction — variadic SurfaceFluxSource..." begin
        src1 = SurfaceFluxSource(:CO2,   fill(1.0, 4, 3))
        src2 = SurfaceFluxSource(:SF6,   fill(2.0, 4, 3))
        src3 = SurfaceFluxSource(:Rn222, fill(3.0, 4, 3))

        map = PerTracerFluxMap(src1, src2, src3)
        @test map isa PerTracerFluxMap
        @test length(map) == 3
        # Stored in declaration order
        @test tracer_names(map) === (:CO2, :SF6, :Rn222)
    end

    @testset "construction — Tuple backed" begin
        src1 = SurfaceFluxSource(:CO2, fill(1.0, 4, 3))
        src2 = SurfaceFluxSource(:SF6, fill(2.0, 4, 3))
        # Direct Tuple input
        map = PerTracerFluxMap((src1, src2))
        @test length(map) == 2
        # Equivalent to variadic form
        map2 = PerTracerFluxMap(src1, src2)
        @test tracer_names(map) == tracer_names(map2)
    end

    @testset "construction — Vector of sources" begin
        sources = [SurfaceFluxSource(:CO2, fill(1.0, 4, 3)),
                   SurfaceFluxSource(:SF6, fill(2.0, 4, 3))]
        map = PerTracerFluxMap(sources)
        @test length(map) == 2
        @test tracer_names(map) === (:CO2, :SF6)
    end

    @testset "empty map is allowed" begin
        # Empty tuple gives a valid no-emission map
        map = PerTracerFluxMap(())
        @test length(map) == 0
        @test tracer_names(map) === ()
        @test flux_for(map, :CO2) === nothing
    end

    @testset "construction rejects duplicates" begin
        src1 = SurfaceFluxSource(:CO2, fill(1.0, 4, 3))
        src1b = SurfaceFluxSource(:CO2, fill(2.0, 4, 3))   # same name!
        @test_throws ArgumentError PerTracerFluxMap(src1, src1b)
    end

    @testset "construction rejects non-SurfaceFluxSource entries" begin
        src = SurfaceFluxSource(:CO2, fill(1.0, 4, 3))
        bad = (tracer_name = :SF6, cell_mass_rate = fill(1.0, 4, 3))
        @test_throws ArgumentError PerTracerFluxMap((src, bad))
    end

    @testset "flux_for returns source or nothing" begin
        co2_rate = fill(1.0, 4, 3)
        src = SurfaceFluxSource(:CO2, co2_rate)
        map = PerTracerFluxMap(src)

        found = flux_for(map, :CO2)
        @test found isa SurfaceFluxSource
        @test found.tracer_name === :CO2
        @test found.cell_mass_rate === co2_rate   # no copy

        @test flux_for(map, :SF6) === nothing
    end

    @testset "iteration + indexing" begin
        src1 = SurfaceFluxSource(:A, fill(1.0, 2, 2))
        src2 = SurfaceFluxSource(:B, fill(2.0, 2, 2))
        src3 = SurfaceFluxSource(:C, fill(3.0, 2, 2))
        map = PerTracerFluxMap(src1, src2, src3)

        # iteration yields sources in storage order
        collected = collect(map)
        @test length(collected) == 3
        @test collected[1].tracer_name === :A
        @test collected[3].tracer_name === :C

        # indexing
        @test map[1].tracer_name === :A
        @test map[end].tracer_name === :C

        # comprehensions
        @test [s.tracer_name for s in map] == [:A, :B, :C]
    end

    @testset "tracer_names matches storage order" begin
        # Construction order preserved in tracer_names (critical for
        # `iterate(flux_map)` semantics used by the apply! kernel).
        src_b = SurfaceFluxSource(:B, fill(1.0, 2, 2))
        src_a = SurfaceFluxSource(:A, fill(1.0, 2, 2))
        src_c = SurfaceFluxSource(:C, fill(1.0, 2, 2))
        map = PerTracerFluxMap(src_b, src_a, src_c)
        @test tracer_names(map) === (:B, :A, :C)
    end

    @testset "Adapt — structural conversion (CPU passthrough)" begin
        rate1 = fill(1.0, 4, 3)
        rate2 = fill(2.0, 4, 3)
        src1 = SurfaceFluxSource(:CO2, rate1)
        src2 = SurfaceFluxSource(:SF6, rate2)
        map = PerTracerFluxMap(src1, src2)

        map_adapt = Adapt.adapt(Array, map)
        @test map_adapt isa PerTracerFluxMap
        @test length(map_adapt) == 2
        @test tracer_names(map_adapt) === (:CO2, :SF6)
        # Values preserved
        @test flux_for(map_adapt, :CO2).cell_mass_rate == rate1
        @test flux_for(map_adapt, :SF6).cell_mass_rate == rate2
    end

    @testset "Adapt on SurfaceFluxSource alone (2D and 3D rates)" begin
        # 2D rate (structured grid surface slice)
        rate2d = fill(1.5, 3, 4)
        src2d = SurfaceFluxSource(:X, rate2d)
        src2d_ad = Adapt.adapt(Array, src2d)
        @test src2d_ad.tracer_name === :X
        @test src2d_ad.cell_mass_rate == rate2d

        # Adaptation itself is rank-agnostic; shape validation occurs when a
        # source is installed on a concrete state.
        rate3d = fill(0.5, 3, 4, 2)
        src3d = SurfaceFluxSource(:Y, rate3d)
        src3d_ad = Adapt.adapt(Array, src3d)
        @test src3d_ad.cell_mass_rate == rate3d
    end

    if HAS_GPU_FLUX
        @testset "Adapt — GPU adaptation" begin
            rate1 = fill(1.0, 4, 3)
            rate2 = fill(2.0, 4, 3)
            src1 = SurfaceFluxSource(:CO2, rate1)
            src2 = SurfaceFluxSource(:SF6, rate2)
            map = PerTracerFluxMap(src1, src2)

            map_gpu = Adapt.adapt(CUDA.CuArray, map)
            @test map_gpu isa PerTracerFluxMap
            @test length(map_gpu) == 2

            # Each source's rate is a CuArray on the device
            co2_on_gpu = flux_for(map_gpu, :CO2)
            @test co2_on_gpu.cell_mass_rate isa CUDA.CuArray{Float64, 2}
            # Round-trip to host returns the original values
            @test Array(co2_on_gpu.cell_mass_rate) == rate1

            sf6_on_gpu = flux_for(map_gpu, :SF6)
            @test Array(sf6_on_gpu.cell_mass_rate) == rate2
        end
    end
end
