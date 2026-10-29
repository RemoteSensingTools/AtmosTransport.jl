"""
Plan 17 Commit 6 — `TransportModel.emissions` + `with_emissions` +
`DrivenSimulation` cleanup.

Verifies:
1. `with_emissions(model, op)` returns a new model with replaced
   emissions operator; other fields are storage-shared.
2. `step!` with active `emissions` adds surface mass via the palindrome's
   S slot — end-to-end composed with diffusion + chemistry, on lat-lon and
   reduced-Gaussian grids; tracers absent from the state are skipped.

The `DrivenSimulation` side (sim-level `surface_sources` installed into the
wrapped model with `with_emissions`) is covered by `test_driven_simulation.jl`
("DrivenSimulation applies bottom-layer surface sources").
"""

using Test
using AtmosTransport
using AtmosTransport: CellState, MoistBasis,
                      UpwindScheme, SlopesScheme, NoLimiter,
                      StructuredFaceFluxState, DryBasis,
                      allocate_face_fluxes, AdvectionWorkspace,
                      NoDiffusion, NoChemistry,
                      SurfaceFluxSource, SurfaceFluxOperator, NoSurfaceFlux,
                      TransportModel, step!, with_chemistry, with_diffusion,
                      with_emissions,
                      ExponentialDecay, ConstantField

using AtmosTransport: HybridSigmaPressure, CPU, LatLonMesh, ReducedGaussianMesh,
                      AtmosGrid, ncells

# =====================================================================
# Helpers
# =====================================================================

function _make_latlon_model(FT, Nx, Ny, Nz;
                            chemistry = NoChemistry(),
                            diffusion = NoDiffusion(),
                            emissions = NoSurfaceFlux(),
                            init = 1.0,
                            tracer_names = (:CO2,))
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    # Build a hybrid-sigma-pressure vertical with Nz layers.
    a_ifc = collect(range(FT(0), FT(1000); length = Nz + 1))
    b_ifc = collect(range(FT(0), FT(1); length = Nz + 1))
    vertical = HybridSigmaPressure(a_ifc, b_ifc)
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    air  = ones(FT, Nx, Ny, Nz)
    kwargs = NamedTuple{tracer_names}(ntuple(_ -> fill(FT(init), Nx, Ny, Nz),
                                             length(tracer_names)))
    state  = CellState(MoistBasis, air; kwargs...)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz;
                                  FT = FT, basis = MoistBasis)
    return TransportModel(state, fluxes, grid, UpwindScheme();
                          chemistry = chemistry,
                          diffusion = diffusion,
                          emissions = emissions)
end

function _make_rg_model(FT, Nz;
                        chemistry = NoChemistry(),
                        diffusion = NoDiffusion(),
                        emissions = NoSurfaceFlux(),
                        init = 1.0,
                        tracer_names = (:CO2,))
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT)
    a_ifc = collect(range(FT(0), FT(1000); length = Nz + 1))
    b_ifc = collect(range(FT(0), FT(1); length = Nz + 1))
    vertical = HybridSigmaPressure(a_ifc, b_ifc)
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    air = ones(FT, ncells(mesh), Nz)
    kwargs = NamedTuple{tracer_names}(ntuple(_ -> fill(FT(init), ncells(mesh), Nz),
                                             length(tracer_names)))
    state = CellState(MoistBasis, air; kwargs...)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz;
                                  FT = FT, basis = MoistBasis)
    return TransportModel(state, fluxes, grid, UpwindScheme();
                          chemistry = chemistry,
                          diffusion = diffusion,
                          emissions = emissions)
end

@testset "TransportModel emissions — plan 17 Commit 6" begin

    @testset "with_emissions replaces operator, storage-shared" begin
        FT = Float64
        Nx, Ny, Nz = 3, 2, 2
        model = _make_latlon_model(FT, Nx, Ny, Nz)

        @test model.emissions isa NoSurfaceFlux

        rate = fill(2.0, Nx, Ny)
        op   = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
        model2 = with_emissions(model, op)

        @test model2.emissions === op
        @test model2.emissions !== model.emissions
        # Storage-shared — not a deep copy
        @test model2.state === model.state
        @test model2.fluxes === model.fluxes
        @test model2.workspace === model.workspace
        @test model2.advection === model.advection
        @test model2.chemistry === model.chemistry
        @test model2.diffusion === model.diffusion
    end

    @testset "step! with active emissions adds surface mass at palindrome center" begin
        FT = Float64
        Nx, Ny, Nz = 3, 2, 2

        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(2.0, Nx, Ny)))
        model = _make_latlon_model(FT, Nx, Ny, Nz; init = 0.0, emissions = op)

        dt = 5.0
        step!(model, dt)

        # With NoDiffusion + NoChemistry + zero fluxes (default in helper
        # — the test fluxes are zeroed by allocate_face_fluxes), the full
        # step collapses to S-only emission: surface adds rate * dt = 10.0.
        @test all(model.state.tracers.CO2[:, :, Nz] .≈ 10.0)
        # Upper layers untouched (no diffusion)
        @test all(model.state.tracers.CO2[:, :, 1] .== 0.0)
    end

    @testset "step! composes advection+diffusion+emissions+chemistry" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 3

        rate = fill(1.0, Nx, Ny)
        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, rate))
        decay = ExponentialDecay{FT, 1, Tuple{ConstantField{FT, 0}}}(
            (ConstantField{FT, 0}(0.0),),   # zero decay rate
            (:CO2,))

        model = _make_latlon_model(FT, Nx, Ny, Nz;
                                    init     = 0.0,
                                    emissions = op,
                                    chemistry = decay)

        dt = 3.0
        step!(model, dt)

        # Zero decay + zero fluxes → same result as pure S (no chemistry effect)
        @test all(model.state.tracers.CO2[:, :, Nz] .≈ rate[1,1] * dt)
    end

    @testset "ReducedGaussian step! with active emissions adds surface mass at palindrome center" begin
        FT = Float64
        Nz = 2
        ncell = ncells(ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT))

        op = SurfaceFluxOperator(SurfaceFluxSource(:CO2, fill(2.0, ncell)))
        model = _make_rg_model(FT, Nz; init = 0.0, emissions = op)

        dt = 5.0
        step!(model, dt)

        @test all(model.state.tracers.CO2[:, Nz] .≈ 10.0)
        @test all(model.state.tracers.CO2[:, 1] .== 0.0)
    end

    @testset "ReducedGaussian step! updates emitting tracers and skips absent ones" begin
        FT = Float64
        Nz = 3
        ncell = ncells(ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT))

        op = SurfaceFluxOperator(
            SurfaceFluxSource(:CO2, fill(1.0, ncell)),
            SurfaceFluxSource(:SF6, fill(0.5, ncell)),
            SurfaceFluxSource(:CH4, fill(9.9, ncell)),
        )
        model = _make_rg_model(FT, Nz;
                               init = 0.0,
                               emissions = op,
                               tracer_names = (:CO2, :SF6))

        step!(model, 4.0)

        @test all(model.state.tracers.CO2[:, Nz] .≈ 4.0)
        @test all(model.state.tracers.SF6[:, Nz] .≈ 2.0)
        @test all(model.state.tracers.CO2[:, 1] .== 0.0)
        @test all(model.state.tracers.SF6[:, 1] .== 0.0)
    end
end
