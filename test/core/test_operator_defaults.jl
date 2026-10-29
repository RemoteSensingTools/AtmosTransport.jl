using Test
using AtmosTransport
using AtmosTransport.Operators.StepPlans: StepOperators

@testset "all optional operators default to their No* type" begin
    # Advection is a required argument. The other four operator slots default
    # independently in each topology's TransportModel constructor and in the
    # StepOperators constructor used by replay. Inspect those actual defaults.
    vertical = HybridSigmaPressure([0.0, 100.0, 300.0], [0.0, 0.0, 1.0])
    meshes = (LatLonMesh(; Nx = 3, Ny = 2),
              ReducedGaussianMesh([-45.0, 45.0], [4, 4]),
              CubedSphereMesh(; Nc = 2, Hp = 1))
    for mesh in meshes
        grid = AtmosGrid(mesh, vertical, CPU())
        state = if mesh isa CubedSphereMesh
            n = mesh.Nc + 2mesh.Hp
            panels = ntuple(_ -> ones(n, n, 2), 6)
            CubedSphereState(DryBasis, mesh, panels; CO2 = ntuple(_ -> zeros(n, n, 2), 6))
        else
            shape = mesh isa LatLonMesh ? (3, 2, 2) : (8, 2)
            CellState(ones(shape); CO2 = zeros(shape))
        end
        fluxes = allocate_face_fluxes(mesh, 2; FT = Float64, basis = DryBasis)
        model = TransportModel(state, fluxes, grid, NoAdvection())
        ops = StepOperators(; state, fluxes, grid, advection = NoAdvection(),
                             advection_workspace = nothing)
        for composition in (model, ops)
            @test composition.chemistry isa NoChemistry
            @test composition.diffusion isa NoDiffusion
            @test composition.emissions isa NoSurfaceFlux
            @test composition.convection isa NoConvection
        end
    end
end
