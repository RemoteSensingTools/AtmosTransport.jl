"""
Unit tests for `PerTracerFluxMap` + `SurfaceFluxSource` — plan 17 Commit 2.

Validates:
- Duplicate tracer names are rejected at construction
- `flux_for(map, :name)` returns the source, or `nothing` for absent
- Iteration and indexing behaviour
- `tracer_names(map)` matches storage order
- Backward-compat: `SurfaceFluxSource` still usable by-name after
  the plan 17 Commit 2 migration from `src/Models/` to `src/Operators/SurfaceFlux/`
- `SurfaceFluxOperator` resolves the map's tracer names against the state:
  `emitting_tracer_indices`, map tracers absent from the state are skipped,
  and a structured `(Nx, Ny, Nz, Nt)` buffer and its unrolled face-indexed
  `(Nx*Ny, Nz, Nt)` form receive the same increments
"""

using Test
using AtmosTransport
using AtmosTransport: SurfaceFluxSource, PerTracerFluxMap, flux_for
using AtmosTransport.Operators.SurfaceFlux: tracer_names, emitting_tracer_indices,
                                            apply_surface_flux!


@testset "PerTracerFluxMap — plan 17 Commit 2" begin

    @testset "construction — Vector of sources" begin
        sources = [SurfaceFluxSource(:CO2, fill(1.0, 4, 3)),
                   SurfaceFluxSource(:SF6, fill(2.0, 4, 3))]
        map = PerTracerFluxMap(sources)
        @test length(map) == 2
        @test tracer_names(map) === (:CO2, :SF6)
    end


    @testset "construction rejects duplicates" begin
        src1 = SurfaceFluxSource(:CO2, fill(1.0, 4, 3))
        src1b = SurfaceFluxSource(:CO2, fill(2.0, 4, 3))   # same name!
        @test_throws ArgumentError PerTracerFluxMap(src1, src1b)
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

    @testset "SurfaceFluxOperator resolves map tracers against the state" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 2
        state = CellState(MoistBasis, ones(FT, Nx, Ny, Nz);
                          CO2 = ones(FT, Nx, Ny, Nz), SF6 = ones(FT, Nx, Ny, Nz),
                          Rn222 = ones(FT, Nx, Ny, Nz))

        op = SurfaceFluxOperator(SurfaceFluxSource(:SF6,   fill(1.0, Nx, Ny)),
                                 SurfaceFluxSource(:Rn222, fill(2.0, Nx, Ny)))
        @test emitting_tracer_indices(op, state) === (2, 3)
        @test emitting_tracer_indices(NoSurfaceFlux(), state) === ()

        # A map tracer the state does not carry (CH4) is skipped, not an error.
        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(2.0, Nx, Ny)),
                                 SurfaceFluxSource(:CH4, fill(9.9, Nx, Ny)))
        apply!(state, nothing, nothing, op, 1.5)
        @test all(state.tracers.CO2[:, :, Nz] .≈ 1.0 + 2.0 * 1.5)
        @test all(state.tracers.CO2[:, :, 1] .== 1.0)
        @test all(state.tracers.SF6 .== 1.0)

        # Structured and unrolled face-indexed buffers get the same increments.
        Nt, Nz4 = 2, 4
        rate_x = reshape(FT.(1:(Nx * Ny)), Nx, Ny) ./ 10
        rate_y = fill(FT(0.5), Nx, Ny)
        q_struct = zeros(FT, Nx, Ny, Nz4, Nt)
        q_face = zeros(FT, Nx * Ny, Nz4, Nt)
        apply_surface_flux!(q_struct, SurfaceFluxOperator(SurfaceFluxSource(:X, rate_x),
                                                          SurfaceFluxSource(:Y, rate_y)),
                            nothing, FT(7), nothing, nothing; tracer_names = (:X, :Y))
        apply_surface_flux!(q_face, SurfaceFluxOperator(SurfaceFluxSource(:X, vec(copy(rate_x))),
                                                        SurfaceFluxSource(:Y, vec(copy(rate_y)))),
                            nothing, FT(7), nothing, nothing; tracer_names = (:X, :Y))
        @test any(!iszero, q_face)
        @test reshape(q_struct, Nx * Ny, Nz4, Nt) ≈ q_face
    end

end
