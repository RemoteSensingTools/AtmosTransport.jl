# RG multi-tracer emission composition: emitting and absent tracers.
# LL emission and diffusion composition lives in test_driven_simulation.jl.
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
