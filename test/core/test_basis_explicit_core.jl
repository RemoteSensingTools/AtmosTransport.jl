#!/usr/bin/env julia

using Test
using Adapt

using AtmosTransport
using .AtmosTransport.Grids: StructuredFluxTopology, face_cells, face_length,
                             face_normal, n_levels, planet_parameters
using .AtmosTransport.Operators: AbstractConstantScheme, AbstractLinearScheme,
                                 AbstractQuadraticScheme, MonotoneLimiter,
                                 strang_split!
using .AtmosTransport.Parameters: PlanetParameters
using .AtmosTransport.State: FaceIndexedFluxState, StructuredFaceFluxState,
                             mass_basis

const HAS_CUDA_FOR_ADAPT = try
    using CUDA
    CUDA.functional()
catch
    false
end

struct _IncompleteHorizontalMesh <: AtmosTransport.Grids.AbstractHorizontalMesh end
struct _IncompleteStructuredMesh <: AtmosTransport.Grids.AbstractStructuredMesh end
Base.eltype(::_IncompleteHorizontalMesh) = Float64
Base.eltype(::_IncompleteStructuredMesh) = Float64

struct _IncompleteVerticalCoordinate{FT} <: AtmosTransport.Grids.AbstractVerticalCoordinate{FT} end

@testset "PlanetParameters and AtmosGrid" begin
    params = PlanetParameters(; FT=Float32, radius=6.0f6, gravity=9.8f0, reference_pressure=1.0f5)
    mesh = LatLonMesh(; FT=Float32, Nx=4, Ny=3, radius=params.radius)
    vc = HybridSigmaPressure(Float32[0, 100, 300], Float32[0, 0, 1])
    grid = @inferred AtmosGrid(mesh, vc, AtmosTransport.CPU(); planet=params)

    @test planet_parameters(grid) == params
    @test radius(grid) == params.radius
    @test gravity(grid) == params.gravity
    @test reference_pressure(grid) == params.reference_pressure
    @test grid.radius == params.radius
    @test floattype(grid) === Float32
end

@testset "Basis-explicit core types" begin
    @test UpwindScheme <: AbstractConstantScheme
    @test SlopesScheme <: AbstractLinearScheme
    @test PPMScheme <: AbstractQuadraticScheme
    m = ones(Float64, 4, 3, 2)
    state_dry = @inferred CellState(DryBasis, m; CO2=copy(m) .* 400e-6)
    state_moist = @inferred CellState(MoistBasis, copy(m); CO2=copy(m) .* 400e-6)

    @test mass_basis(state_dry) isa DryBasis
    @test mass_basis(state_moist) isa MoistBasis

    mesh = LatLonMesh(; Nx=4, Ny=3)
    vc = HybridSigmaPressure([0.0, 100.0, 300.0], [0.0, 0.0, 1.0])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU())
    flux_dry = allocate_face_fluxes(mesh, 2; FT=Float64, basis=DryBasis)
    flux_moist = allocate_face_fluxes(StructuredFluxTopology(), 4, 3, 2; FT=Float64, basis=MoistBasis)

    @test mass_basis(flux_dry) isa DryBasis
    @test mass_basis(flux_moist) isa MoistBasis

    ws = AdvectionWorkspace(state_dry.air_mass)

    @test_throws MethodError apply!(state_dry, flux_moist, grid, UpwindScheme(), 1800.0; workspace=ws)

    face_rm = ones(Float64, ncells(mesh), 2)
    face_m = ones(Float64, ncells(mesh), 2)
    face_flux = zeros(Float64, nfaces(mesh), 2)
    face_ws = AdvectionWorkspace(face_m)
    err = try
        AtmosTransport.Operators.Advection.sweep_horizontal!(
            face_rm, face_m, face_flux, mesh, SlopesScheme(), face_ws)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("face-indexed meshes supports UpwindScheme only", sprint(showerror, err))
end

@testset "physics families share AbstractOperator" begin
    root = AtmosTransport.Operators.AbstractOperator
    @test UpwindScheme() isa root
    @test NoDiffusion() isa root
    @test NoSurfaceFlux() isa root
    @test NoConvection() isa root
    @test NoChemistry() isa root
end

@testset "State and flux containers reject inconsistent storage" begin
    air = ones(Float64, 4, 3, 2)
    raw = zeros(Float64, 4, 3, 2, 1)

    @test CellState(DryBasis, air, raw, (:CO2,)).tracer_names == (:CO2,)
    @test_throws DimensionMismatch CellState(
        DryBasis, air, zeros(Float64, 4, 2, 2, 1), (:CO2,))
    @test_throws DimensionMismatch CellState(
        DryBasis, air, zeros(Float64, 4, 3, 2, 2), (:CO2,))
    @test_throws ArgumentError CellState(
        DryBasis, air, zeros(Float64, 4, 3, 2, 2), (:CO2, :CO2))
    @test_throws ArgumentError CellState(
        DryBasis, air, zeros(Float32, 4, 3, 2, 1), (:CO2,))

    @test_throws DimensionMismatch StructuredFaceFluxState{DryBasis}(
        zeros(4, 3, 2), zeros(4, 4, 2), zeros(4, 3, 3))
    @test_throws ArgumentError StructuredFaceFluxState{DryBasis}(
        zeros(Float32, 5, 3, 2), zeros(Float64, 4, 4, 2), zeros(Float64, 4, 3, 3))
    @test_throws DimensionMismatch FaceIndexedFluxState{DryBasis}(
        zeros(8, 2), zeros(4, 2))

    mesh = CubedSphereMesh(; FT=Float64, Nc=2, Hp=1)
    n = mesh.Nc + 2 * mesh.Hp
    panels_air = ntuple(_ -> ones(Float64, n, n, 2), 6)
    panels_raw = ntuple(_ -> zeros(Float64, n, n, 2, 1), 6)
    @test CubedSphereState(
        DryBasis, panels_air, panels_raw, (:CO2,); halo_width=1).halo_width == 1
    bad_panels_raw = ntuple(p -> zeros(Float64, n, n, p == 6 ? 1 : 2, 1), 6)
    @test_throws DimensionMismatch CubedSphereState(
        DryBasis, panels_air, bad_panels_raw, (:CO2,); halo_width=1)
    @test_throws DimensionMismatch CubedSphereState(
        DryBasis, panels_air, panels_raw, (:CO2,); halo_width=2)

    mixed_met = @test_deprecated AtmosTransport.State.MetState(
        ones(Float64, 4, 3), ones(Float32, 4, 3, 2))
    @test eltype(mixed_met.ps) === Float64
    @test eltype(mixed_met.q) === Float32
end

@testset "Abstract grid contracts fail with actionable errors" begin
    mesh = _IncompleteHorizontalMesh()
    smesh = _IncompleteStructuredMesh()
    vc = _IncompleteVerticalCoordinate{Float64}()

    for f in (
        () -> ncells(mesh),
        () -> nfaces(mesh),
        () -> cell_area(mesh, 1),
        () -> face_length(mesh, 1),
        () -> face_normal(mesh, 1),
        () -> face_cells(mesh, 1),
        () -> cell_faces(mesh, 1),
        () -> nx(smesh),
        () -> ny(smesh),
    )
        err = try
            f()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("must be implemented", sprint(showerror, err))
    end

    for f in (
        () -> n_levels(vc),
        () -> pressure_at_interface(vc, 1, 1000.0),
        () -> pressure_at_level(vc, 1, 1000.0),
        () -> level_thickness(vc, 1, 1000.0),
        () -> AtmosTransport.Grids.b_diff(vc, 1),
    )
        err = try
            f()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("must be implemented", sprint(showerror, err))
    end
end

@testset "Standalone runtime smoke test" begin
    FT = Float64
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    m = ones(FT, Nx, Ny, Nz)
    state = CellState(DryBasis, copy(m); CO2=copy(m) .* FT(400e-6))
    fluxes = allocate_face_fluxes(StructuredFluxTopology(), Nx, Ny, Nz; FT=FT, basis=DryBasis)

    m0 = total_air_mass(state)
    rm0 = total_mass(state, :CO2)

    model = @inferred TransportModel(state, fluxes, grid, UpwindScheme())
    @test_throws ArgumentError Simulation(model; Δt=0.0, stop_time=1.0)
    @test_throws ArgumentError Simulation(model; Δt=1.0, stop_time=NaN)
    sim = Simulation(model; Δt=FT(1800), stop_time=FT(3600))
    run!(sim)

    @test sim.iteration == 2
    @test total_air_mass(sim.model.state) ≈ m0 atol=eps(FT) * m0 * 10
    @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(FT) * rm0 * 10

    @test_throws ArgumentError Simulation(model; Δt=FT(1800), stop_time=FT(2700))
    final_callback_fired = Ref(false)
    decimal = Simulation(model; Δt=FT(0.1), stop_time=FT(1.0),
                         callbacks=(final = s -> final_callback_fired[] =
                             s.time >= s.stop_time,))
    run!(decimal)
    @test decimal.iteration == 10
    @test decimal.time == FT(1.0)
    @test final_callback_fired[]

    state_slopes = CellState(DryBasis, copy(m); CO2=copy(m) .* FT(400e-6))
    fluxes_slopes = allocate_face_fluxes(StructuredFluxTopology(), Nx, Ny, Nz; FT=FT, basis=DryBasis)
    model_slopes = @inferred TransportModel(state_slopes, fluxes_slopes, grid, SlopesScheme())
    sim_slopes = Simulation(model_slopes; Δt=FT(1800), stop_time=FT(3600))
    run!(sim_slopes)

    @test sim_slopes.iteration == 2
    @test total_air_mass(sim_slopes.model.state) ≈ m0 atol=eps(FT) * m0 * 10
    @test total_mass(sim_slopes.model.state, :CO2) ≈ rm0 atol=eps(FT) * rm0 * 10
end

@testset "Structured x-direction static CFL pilot" begin
    FT = Float64
    Nx, Ny, Nz = 4, 1, 1
    m = ones(FT, Nx, Ny, Nz)
    rm = copy(m) .* FT(400e-6)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)

    # Periodic inflow through face 1 / Nx+1 plus stronger outflow through face 2.
    # Static outflow-based CFL: cell 1 loses max(am[2],0)=1.5 per step, m=1.0, so
    # CFL=1.5 → n_sub = ceil(1.5/1.0) = 2. The pre-plan-13 evolving-mass pilot
    # returned n_sub=3 because it flagged a post-first-pass transient CFL=1.0
    # equality as a violation; the static pilot accepts CFL<=cfl_limit.
    am[1, 1, 1] = FT(1.0)
    am[2, 1, 1] = FT(1.5)
    am[Nx + 1, 1, 1] = FT(1.0)

    ws = AtmosTransport.Operators.Advection.AdvectionWorkspace(m)
    nsub = AtmosTransport.Operators.Advection._x_subcycling_pass_count(am, m, ws, FT(1))
    @test nsub == 2

    m0 = sum(m)
    rm0 = sum(rm)
    flux_scale = inv(FT(nsub))
    for _ in 1:nsub
        AtmosTransport.Operators.Advection.sweep_x!(rm, m, am, UpwindScheme(), ws, flux_scale)
    end

    @test minimum(m) ≥ -eps(FT) * 10
    @test sum(m) ≈ m0 atol=eps(FT) * m0 * 10
    @test sum(rm) ≈ rm0 atol=eps(FT) * rm0 * 10
    @test all(isfinite, m)
    @test all(isfinite, rm)
end

@testset "Clustered x-sweeps accept Int32 cluster_sizes" begin
    FT = Float64
    m0 = ones(FT, 4, 1, 1)
    rm0 = copy(m0) .* FT(400e-6)
    am = zeros(FT, 5, 1, 1)

    for scheme in (UpwindScheme(), SlopesScheme(MonotoneLimiter()))
        m = copy(m0)
        rm = copy(rm0)
        ws = AtmosTransport.Operators.Advection.AdvectionWorkspace(m; cluster_sizes_cpu=Int32[2])
        AtmosTransport.Operators.Advection.sweep_x!(rm, m, am, scheme, ws)
        @test m == m0
        @test rm == rm0
        @test all(isfinite, m)
        @test all(isfinite, rm)
    end
end

@testset "RussellLerner y-sweep stays finite with zero-mass donor" begin
    FT = Float32
    m = ones(FT, 1, 3, 1)
    rm = fill(FT(0.4), 1, 3, 1) .* m
    m[1, 1, 1] = zero(FT)
    rm[1, 1, 1] = zero(FT)
    bm = zeros(FT, 1, 4, 1)
    bm[1, 2, 1] = FT(0.1)
    ws = AtmosTransport.Operators.Advection.AdvectionWorkspace(m)

    AtmosTransport.Operators.Advection.sweep_y!(rm, m, bm, SlopesScheme(MonotoneLimiter()), ws)

    @test all(isfinite, m)
    @test all(isfinite, rm)
    @test sum(rm) ≈ FT(0.8)
    @test rm[1, 1, 1] == zero(FT)
end

@testset "Face-indexed horizontal subcycling preserves positivity" begin
    FT = Float64
    Nz = 1
    mesh = ReducedGaussianMesh(FT[0], [4]; FT=FT)
    vc = HybridSigmaPressure(FT[0, 100], FT[0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    m = ones(FT, ncells(mesh), Nz)
    rm = reshape(FT[1, 0, 0, 0], :, 1)
    state = CellState(DryBasis, copy(m); CO2=copy(rm))
    fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)
    fluxes.horizontal_flux .= zero(FT)
    fluxes.cm .= zero(FT)

    # Cell 1 sees one strong outflow and one weaker inflow in the same sweep.
    # Static outflow-based CFL = 1.2 → n_sub = ceil(1.2/1.0) = 2. The
    # pre-plan-13 evolving-mass pilot returned n_sub=3 because it flagged a
    # post-first-pass transient CFL=1.0 equality as a violation.
    fluxes.horizontal_flux[1, 1] = FT(0.4)
    fluxes.horizontal_flux[2, 1] = FT(1.2)

    ws = AtmosTransport.Operators.Advection.AdvectionWorkspace(state.air_mass)
    nsub = AtmosTransport.Operators.Advection._horizontal_face_subcycling_pass_count(
        fluxes.horizontal_flux, state.air_mass, mesh, ws, FT(1))
    @test nsub == 2

    flux_scale = inv(FT(nsub))
    for _ in 1:nsub
        AtmosTransport.Operators.Advection.sweep_horizontal!(
            state.tracers.CO2, state.air_mass, fluxes.horizontal_flux, mesh,
            UpwindScheme(), ws, flux_scale)
    end

    q = mixing_ratio(state, :CO2)
    @test minimum(state.air_mass) > zero(FT)
    @test minimum(q) ≥ -eps(FT) * 10
    @test maximum(q) ≤ one(FT) + eps(FT) * 10
end

@testset "GPU static CFL enforces max_n_sub" begin
    FT = Float32
    if HAS_CUDA_FOR_ADAPT
        m = CUDA.fill(FT(1), 4, 1, 1)
        am = cu(reshape(FT[10, 10, 0, 0, 10], 5, 1, 1))
        ws = AtmosTransport.Operators.Advection.AdvectionWorkspace(m)
        @test_throws ArgumentError AtmosTransport.Operators.Advection._x_subcycling_pass_count(
            am, m, ws, FT(1); max_n_sub=4)
    else
        @test true
    end
end

@testset "CubedSphere runtime uses dedicated panel-native types" begin
    mesh = CubedSphereMesh(; FT=Float64, Nc=4)
    @test_throws ArgumentError cell_area(mesh, 1)
    @test_throws ArgumentError face_cells(mesh, 1)

    vc = HybridSigmaPressure([0.0, 100.0, 300.0], [0.0, 0.0, 1.0])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=Float64)
    m = ones(Float64, 12, 4, 2)
    state = CellState(DryBasis, copy(m); CO2=copy(m) .* 400e-6)
    fluxes = StructuredFaceFluxState{DryBasis}(zeros(Float64, 13, 4, 2), zeros(Float64, 12, 5, 2), zeros(Float64, 12, 4, 3))
    ws = AdvectionWorkspace(state.air_mass)

    @test_throws ArgumentError TransportModel(state, fluxes, grid, UpwindScheme())
    @test_throws ArgumentError apply!(state, fluxes, grid, UpwindScheme(), 1800.0; workspace=ws)
    @test_throws ArgumentError strang_split!(state, fluxes, grid, UpwindScheme(); workspace=ws)

    FT = Float64
    Nz = 2
    N = mesh.Nc + 2 * mesh.Hp
    panels_m = ntuple(_ -> ones(FT, N, N, Nz), 6)
    panels_rm = ntuple(_ -> fill(FT(400e-6), N, N, Nz), 6)
    cs_state = CubedSphereState(DryBasis, mesh, panels_m; CO2=panels_rm)
    cs_fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)

    ws_a = AtmosTransport.Operators.Advection.CSAdvectionWorkspace(mesh, Nz)
    ws_b = AtmosTransport.Operators.Advection.CSAdvectionWorkspace(mesh, Nz)
    AtmosTransport.Operators.Advection._record_cs_subcycle_growth!(ws_a, 2, 3, 4)
    @test ws_a.max_subcycles[] == (2, 3, 4)
    @test ws_b.max_subcycles[] == (1, 1, 1)
    ws_adapted = Adapt.adapt(Array, ws_a)
    @test ws_adapted.max_subcycles[] == ws_a.max_subcycles[]
    @test ws_adapted.max_subcycles !== ws_a.max_subcycles

    m0 = total_air_mass(cs_state)
    rm0 = total_mass(cs_state, :CO2)

    cs_model = @inferred TransportModel(cs_state, cs_fluxes, grid, UpwindScheme())
    cs_sim = Simulation(cs_model; Δt=FT(1800), stop_time=FT(3600))
    run!(cs_sim)

    @test cs_sim.iteration == 2
    @test total_air_mass(cs_sim.model.state) ≈ m0 atol=eps(FT) * m0 * 10
    @test total_mass(cs_sim.model.state, :CO2) ≈ rm0 atol=eps(FT) * rm0 * 10
end

@testset "Face-connected reduced-Gaussian smoke test" begin
    FT = Float64
    Nz = 2
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    m = ones(FT, ncells(mesh), Nz)
    state = CellState(DryBasis, copy(m); CO2=copy(m) .* FT(400e-6))
    fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)

    @test fluxes isa FaceIndexedFluxState{DryBasis}
    @test mass_basis(fluxes) isa DryBasis

    m0 = total_air_mass(state)
    rm0 = total_mass(state, :CO2)

    model = @inferred TransportModel(state, fluxes, grid, UpwindScheme())
    sim = Simulation(model; Δt=FT(1800), stop_time=FT(3600))
    run!(sim)

    @test sim.iteration == 2
    @test total_air_mass(sim.model.state) ≈ m0 atol=eps(FT) * m0 * 10
    @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(FT) * rm0 * 10

end

@testset "Face-connected reduced-Gaussian GPU matches CPU for Upwind" begin
    FT = Float64
    Nz = 2
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    m = ones(FT, ncells(mesh), Nz)
    q = reshape(range(FT(390e-6), FT(410e-6); length=ncells(mesh) * Nz), ncells(mesh), Nz)
    rm = q .* m
    fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)
    fluxes.horizontal_flux .= zero(FT)
    fluxes.cm .= zero(FT)
    fluxes.horizontal_flux[1, 1] = FT(0.10)
    fluxes.horizontal_flux[2, 1] = FT(-0.04)
    fluxes.horizontal_flux[5, 2] = FT(0.06)
    fluxes.cm[:, 2] .= reshape(range(FT(-0.03), FT(0.03); length=ncells(mesh)), :, 1)

    state_cpu = CellState(DryBasis, copy(m); CO2=copy(rm))
    model_cpu = TransportModel(state_cpu, deepcopy(fluxes), grid, UpwindScheme())
    step!(model_cpu, FT(1800))

    if HAS_CUDA_FOR_ADAPT
        model_gpu = Adapt.adapt(CUDA.CuArray, TransportModel(CellState(DryBasis, copy(m); CO2=copy(rm)),
                                                             deepcopy(fluxes), grid, UpwindScheme()))
        @test model_gpu.workspace.advection_ws.face_left isa CUDA.CuArray{Int32, 1}
        @test model_gpu.workspace.advection_ws.face_right isa CUDA.CuArray{Int32, 1}

        step!(model_gpu, FT(1800))

        @test Array(model_gpu.state.air_mass) ≈ model_cpu.state.air_mass atol=eps(FT) * 200 rtol=eps(FT) * 200
        @test Array(model_gpu.state.tracers.CO2) ≈ model_cpu.state.tracers.CO2 atol=eps(FT) * 200 rtol=eps(FT) * 200
    else
        @test_skip false
    end
end

@testset "Face-connected unsupported reconstruction families fail honestly" begin
    FT = Float64
    Nz = 2
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    m = ones(FT, ncells(mesh), Nz)
    state = CellState(DryBasis, copy(m); CO2=copy(m) .* FT(400e-6))
    fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)
    ws = AdvectionWorkspace(m)

    @test_throws ArgumentError apply!(state, fluxes, grid, SlopesScheme(), FT(1800); workspace=ws)
    @test_throws ArgumentError apply!(state, fluxes, grid, PPMScheme(), FT(1800); workspace=ws)
end


@testset "Adapt.jl container conversions" begin
    FT = Float64
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)
    m = ones(FT, Nx, Ny, Nz)
    state = CellState(DryBasis, copy(m); CO2=copy(m) .* FT(400e-6))
    fluxes = allocate_face_fluxes(StructuredFluxTopology(), Nx, Ny, Nz; FT=FT, basis=DryBasis)
    model = TransportModel(state, fluxes, grid, UpwindScheme())

    model_host = Adapt.adapt(Array, model)
    @test model_host.state.air_mass isa Array{FT,3}
    @test model_host.fluxes.am isa Array{FT,3}
    @test model_host.workspace.advection_ws.rm_A isa Array{FT,3}
    @test model_host.grid === model.grid

    if HAS_CUDA_FOR_ADAPT
        model_gpu = Adapt.adapt(CUDA.CuArray, model)
        @test model_gpu.state.air_mass isa CUDA.CuArray{FT,3}
        @test model_gpu.fluxes.am isa CUDA.CuArray{FT,3}
        @test model_gpu.workspace.advection_ws.rm_A isa CUDA.CuArray{FT,3}
        @test model_gpu.grid === model.grid
    end
end
