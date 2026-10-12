#!/usr/bin/env julia

using Test

using AtmosTransport
using .AtmosTransport.MetDrivers: driver_grid
using .AtmosTransport.Models: current_qv, substep_index, window_index
const Runner = AtmosTransport.Models.DrivenRunner

function write_driven_latlon_binary(path::AbstractString;
                                    FT::Type{<:AbstractFloat}=Float64,
                                    source_flux_sampling::Symbol=:window_start_endpoint,
                                    flux_sampling::Symbol=:window_constant,
                                    window_mass_scales::Tuple{Vararg{Real}}=(1, 1),
                                    mass_basis::Symbol=:moist)
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)

    windows = [
        begin
            m = fill(FT(window_mass_scales[win]), Nx, Ny, Nz)
            am = zeros(FT, Nx + 1, Ny, Nz)
            bm = zeros(FT, Nx, Ny + 1, Nz)
            cm = zeros(FT, Nx, Ny, Nz + 1)
            ps = fill(FT(95000 + 100win), Nx, Ny)
            qv_start = fill(FT(0.01win), Nx, Ny, Nz)
            qv_end = fill(FT(0.01win + 0.01), Nx, Ny, Nz)
            dam = fill(FT(0.2win), Nx + 1, Ny, Nz)
            dbm = fill(FT(0.4win), Nx, Ny + 1, Nz)
            dcm = fill(FT(0.6win), Nx, Ny, Nz + 1)
            dm = fill(FT(0.8win), Nx, Ny, Nz)
            (; m, am, bm, cm, ps, qv_start, qv_end, dam, dbm, dcm, dm)
        end for win in 1:length(window_mass_scales)
    ]

    write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=1800.0,
                           steps_per_window=2,
                           mass_basis,
                           source_flux_sampling=source_flux_sampling,
                           flux_sampling=flux_sampling,
                           extra_header=Dict(
                               "poisson_balance_target_scale" => 0.25,
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ))
    return grid
end

@testset "runner initial condition uses start_window air mass" begin
    mktemp() do path, io
        close(io)
        write_driven_latlon_binary(path; FT = Float64,
                                   window_mass_scales = (1, 2),
                                   mass_basis = :dry)
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "run" => Dict("start_window" => 2, "stop_window" => 2,
                          "prefetch_windows" => false),
            "advection" => Dict("scheme" => "upwind"),
            "tracers" => Dict("co2" => Dict(
                "init" => Dict("kind" => "uniform", "background" => 4e-4))),
            "output" => Dict("enabled" => false),
        )
        model = run_driven_simulation(cfg)
        @test all(mixing_ratio(model.state, :co2) .== 4e-4)
        @test all(model.state.air_mass .== 2.0)
    end
end

@testset "lat-lon runner requires a global longitude span" begin
    @test Runner._require_global_longitude_span(
        LatLonMesh(; FT = Float64, Nx = 4, Ny = 3)) === nothing
    regional = LatLonMesh(; FT = Float64, Nx = 4, Ny = 3,
                          longitude = (-20, 20))
    @test_throws ArgumentError Runner._require_global_longitude_span(regional)
end

function write_driven_reduced_binary(path::AbstractString;
                                     FT::Type{<:AbstractFloat}=Float64,
                                     source_flux_sampling::Symbol=:window_start_endpoint,
                                     window_mass_scales::Tuple{Vararg{Real}}=(1, 1))
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)

    windows = [
        begin
            m = fill(FT(window_mass_scales[win]), ncell, nlevel)
            hflux = zeros(FT, nface_h, nlevel)
            cm = zeros(FT, ncell, nlevel + 1)
            ps = fill(FT(90000 + 100win), ncell)
            qv_start = fill(FT(0.01win), ncell, nlevel)
            qv_end = fill(FT(0.01win + 0.01), ncell, nlevel)
            (; m, hflux, cm, ps, qv_start, qv_end)
        end for win in 1:length(window_mass_scales)
    ]

    write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=1800.0,
                           steps_per_window=2,
                           mass_basis=:moist,
                           source_flux_sampling=source_flux_sampling,
                           extra_header=Dict(
                               "poisson_balance_target_scale" => 0.25,
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ))
    return grid
end

@testset "DrivenSimulation optional no-reset window carry preserves mixing ratio" begin
    mktemp() do path, io
        close(io)
        write_driven_latlon_binary(path; FT=Float64, window_mass_scales=(1, 2))

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        grid = driver_grid(driver)
        state = CellState(MoistBasis, ones(Float64, 4, 3, 2); CO2=fill(400e-6, 4, 3, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        sim = DrivenSimulation(model, driver;
                               start_window=1,
                               stop_window=2,
                               air_mass_reset_mode=:none)

        step!(sim)
        step!(sim)
        step!(sim)

        @test window_index(sim) == 2
        @test all(isapprox.(sim.model.state.air_mass, 1.0; atol=eps(Float64) * 10))
        @test all(isapprox.(mixing_ratio(sim.model.state, :CO2), 400e-6; atol=eps(Float64) * 10))

        close(driver)
    end
end

@testset "DrivenSimulation air-mass reset modes choose VMR or tracer-mass conservation" begin
    mktemp() do path, io
        close(io)
        write_driven_latlon_binary(path; FT=Float64, window_mass_scales=(1, 2))

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        grid = driver_grid(driver)
        ncell = 4 * 3 * 2

        function build_sim(; kwargs...)
            state = CellState(MoistBasis, ones(Float64, 4, 3, 2);
                              CO2 = fill(400e-6, 4, 3, 2))
            fluxes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
            model = TransportModel(state, fluxes, grid, UpwindScheme())
            return DrivenSimulation(model, driver; start_window=1, stop_window=2, kwargs...)
        end

        sim_vmr = build_sim(air_mass_reset_mode=:preserve_vmr)
        step!(sim_vmr)
        step!(sim_vmr)
        step!(sim_vmr)
        @test window_index(sim_vmr) == 2
        @test all(isapprox.(sim_vmr.model.state.air_mass, 2.0; atol=eps(Float64) * 10))
        @test all(isapprox.(mixing_ratio(sim_vmr.model.state, :CO2), 400e-6;
                            atol=eps(Float64) * 10))
        @test total_mass(sim_vmr.model.state, :CO2) ≈ ncell * 2.0 * 400e-6

        sim_mass = build_sim()
        @test sim_mass.air_mass_reset_mode === :preserve_tracer_mass
        step!(sim_mass)
        step!(sim_mass)
        step!(sim_mass)
        @test window_index(sim_mass) == 2
        @test all(isapprox.(sim_mass.model.state.air_mass, 2.0; atol=eps(Float64) * 10))
        @test total_mass(sim_mass.model.state, :CO2) ≈ ncell * 400e-6
        @test all(isapprox.(mixing_ratio(sim_mass.model.state, :CO2), 200e-6;
                            atol=eps(Float64) * 10))

        close(driver)
    end
end

@testset "DrivenSimulation applies bottom-layer surface sources" begin
    mktemp() do path, io
        close(io)
        write_driven_latlon_binary(path; FT=Float64, window_mass_scales=(1,))

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        grid = driver_grid(driver)
        state = CellState(MoistBasis, ones(Float64, 4, 3, 2);
                          natural_co2=fill(400e-6, 4, 3, 2),
                          fossil_co2=zeros(Float64, 4, 3, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        source = AtmosTransport.SurfaceFluxSource(:fossil_co2, fill(2.0, 4, 3))
        sim = DrivenSimulation(model, driver;
                               start_window=1,
                               stop_window=1,
                               surface_sources=(source,))

        step!(sim)

        @test all(isapprox.(sim.model.state.tracers.natural_co2, 400e-6; atol=eps(Float64) * 10))
        @test all(iszero, sim.model.state.tracers.fossil_co2[:, :, 1])
        @test all(isapprox.(sim.model.state.tracers.fossil_co2[:, :, 2], 3600.0; atol=eps(Float64) * 10))
        @test total_mass(sim.model.state, :fossil_co2) ≈ 4 * 3 * 3600.0 atol=eps(Float64) * 100

        close(driver)
    end
end

@testset "DrivenSimulation applies bottom-layer surface sources on ReducedGaussian" begin
    mktemp() do path, io
        close(io)
        write_driven_reduced_binary(path; FT=Float64, window_mass_scales=(1,))

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        grid = driver_grid(driver)
        ncell = ncells(grid.horizontal)

        state = CellState(MoistBasis, ones(Float64, ncell, 2);
                          natural_co2=fill(400e-6, ncell, 2),
                          fossil_co2=zeros(Float64, ncell, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        source = AtmosTransport.SurfaceFluxSource(:fossil_co2, fill(2.0, ncell))
        sim = DrivenSimulation(model, driver;
                               start_window=1,
                               stop_window=1,
                               surface_sources=(source,))

        step!(sim)

        @test all(isapprox.(sim.model.state.tracers.natural_co2, 400e-6; atol=eps(Float64) * 10))
        @test all(iszero, sim.model.state.tracers.fossil_co2[:, 1])
        @test all(isapprox.(sim.model.state.tracers.fossil_co2[:, 2], 3600.0; atol=eps(Float64) * 10))
        @test total_mass(sim.model.state, :fossil_co2) ≈ ncell * 3600.0 atol=eps(Float64) * 100

        close(driver)
    end
end

@testset "DrivenSimulation ReducedGaussian run! accumulates emissions across windows" begin
    mktemp() do path, io
        close(io)
        write_driven_reduced_binary(path; FT=Float64, window_mass_scales=(1, 1))

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        grid = driver_grid(driver)
        ncell = ncells(grid.horizontal)

        state = CellState(MoistBasis, ones(Float64, ncell, 2);
                          fossil_co2=zeros(Float64, ncell, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        source = AtmosTransport.SurfaceFluxSource(:fossil_co2, fill(2.0, ncell))
        sim = DrivenSimulation(model, driver;
                               start_window=1,
                               stop_window=2,
                               surface_sources=(source,))

        run!(sim)

        @test sim.iteration == 4
        @test sim.time == 7200.0
        @test window_index(sim) == 2
        @test total_mass(sim.model.state, :fossil_co2) ≈ ncell * 2.0 * 1800.0 * 4 atol=eps(Float64) * 100
        @test all(iszero, sim.model.state.tracers.fossil_co2[:, 1])
        @test all(isapprox.(sim.model.state.tracers.fossil_co2[:, 2], 2.0 * 1800.0 * 4; atol=eps(Float64) * 10))

        close(driver)
    end
end

@testset "DrivenSimulation ReducedGaussian supports diffusion plus surface sources" begin
    mktemp() do path, io
        close(io)
        write_driven_reduced_binary(path; FT=Float64, window_mass_scales=(1,))

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        grid = driver_grid(driver)
        ncell = ncells(grid.horizontal)

        air_mass = ones(Float64, ncell, 2)
        tracer = zeros(Float64, ncell, 2)
        tracer[:, 2] .= 100.0
        state = CellState(MoistBasis, air_mass; fossil_co2=tracer)
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        kz = ConstantField{Float64, 2}(1.0)
        diffusion = ImplicitVerticalDiffusion(; kz_field=kz)
        model = TransportModel(state, fluxes, grid, UpwindScheme(); diffusion=diffusion)
        fill!(model.workspace.diffusion_ws.layer_thickness, 100.0)

        source = AtmosTransport.SurfaceFluxSource(:fossil_co2, fill(2.0, ncell))
        sim = DrivenSimulation(model, driver;
                               start_window=1,
                               stop_window=1,
                               surface_sources=(source,))

        m0 = total_mass(sim.model.state, :fossil_co2)
        run!(sim)

        @test sim.iteration == 2
        @test sim.time == 3600.0
        @test all(sim.model.state.tracers.fossil_co2[:, 1] .> 0.0)
        @test total_mass(sim.model.state, :fossil_co2) ≈ m0 + ncell * 2.0 * 1800.0 * 2 rtol=1e-7

        close(driver)
    end
end

@testset "DrivenSimulation window-forcing runtime" begin
    mktemp() do path, io
        close(io)
        grid_ref = write_driven_latlon_binary(path; FT=Float64)

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        grid = driver_grid(driver)
        @test grid.horizontal isa LatLonMesh
        @test nx(grid.horizontal) == nx(grid_ref.horizontal)
        @test ny(grid.horizontal) == ny(grid_ref.horizontal)

        state = CellState(MoistBasis, ones(Float64, 4, 3, 2); CO2=fill(400e-6, 4, 3, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model = @inferred TransportModel(state, fluxes, grid, UpwindScheme())
        sim = DrivenSimulation(model, driver; start_window=1, stop_window=2,
                               air_mass_reset_mode=:none)
        @test sim.interpolate_fluxes_within_window == false

        @test sim.Δt == 1800.0
        @test sim.window_dt == 3600.0
        @test sim.steps_per_window == 2
        @test window_index(sim) == 1
        @test substep_index(sim) == 1
        @test current_qv(sim) !== nothing
        @test all(current_qv(sim) .== 0.01)

        m0 = total_air_mass(sim.model.state)
        rm0 = total_mass(sim.model.state, :CO2)

        state_window = CellState(MoistBasis, ones(Float64, 4, 3, 2); CO2=fill(400e-6, 4, 3, 2))
        fluxes_window = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model_window = TransportModel(state_window, fluxes_window, grid, UpwindScheme())
        sim_window = DrivenSimulation(model_window, driver; start_window=1, stop_window=2,
                                      air_mass_reset_mode=:none)
        run_window!(sim_window)
        @test sim_window.iteration == 2
        @test sim_window.time == 3600.0
        @test window_index(sim_window) == 1

        step!(sim)
        @test sim.iteration == 1
        @test sim.time == 1800.0
        @test window_index(sim) == 1
        @test substep_index(sim) == 2
        @test all(iszero, sim.model.fluxes.am)
        @test all(iszero, sim.model.fluxes.bm)
        @test all(iszero, sim.model.fluxes.cm)
        @test all(isapprox.(sim.expected_air_mass, 1.2; atol=eps(Float64) * 10))
        @test all(isapprox.(current_qv(sim), 0.0125; atol=eps(Float64) * 10))

        step!(sim)
        @test sim.iteration == 2
        @test sim.time == 3600.0
        @test window_index(sim) == 1
        @test substep_index(sim) == 1
        @test all(iszero, sim.model.fluxes.am)
        @test all(iszero, sim.model.fluxes.bm)
        @test all(iszero, sim.model.fluxes.cm)
        @test all(isapprox.(sim.expected_air_mass, 1.6; atol=eps(Float64) * 10))
        @test all(isapprox.(current_qv(sim), 0.0175; atol=eps(Float64) * 10))

        step!(sim)
        @test sim.iteration == 3
        @test sim.time == 5400.0
        @test window_index(sim) == 2
        @test substep_index(sim) == 2
        @test all(iszero, sim.model.fluxes.am)
        @test all(iszero, sim.model.fluxes.bm)
        @test all(iszero, sim.model.fluxes.cm)
        @test all(isapprox.(sim.expected_air_mass, 1.4; atol=eps(Float64) * 10))
        @test all(isapprox.(current_qv(sim), 0.0225; atol=eps(Float64) * 10))

        run!(sim)
        @test sim.iteration == 4
        @test sim.time == 7200.0
        @test window_index(sim) == 2
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10
        @test_throws ArgumentError step!(sim)

        state_slopes = CellState(MoistBasis, ones(Float64, 4, 3, 2); CO2=fill(400e-6, 4, 3, 2))
        fluxes_slopes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model_slopes = @inferred TransportModel(state_slopes, fluxes_slopes, grid, SlopesScheme())
        sim_slopes = DrivenSimulation(model_slopes, driver; start_window=1, stop_window=1,
                                      air_mass_reset_mode=:none)
        m0_slopes = total_air_mass(sim_slopes.model.state)
        rm0_slopes = total_mass(sim_slopes.model.state, :CO2)
        run_window!(sim_slopes)
        @test sim_slopes.iteration == 2
        @test total_air_mass(sim_slopes.model.state) ≈ m0_slopes atol=eps(Float64) * m0_slopes * 10
        @test total_mass(sim_slopes.model.state, :CO2) ≈ rm0_slopes atol=eps(Float64) * rm0_slopes * 10

        close(driver)
    end
end

@testset "full-window flux scaling guard (flux_kind contract)" begin
    # The runtime divides full-window fluxes by 2*steps_per_window before each
    # substep. Only the CS flux state implements that scaling; every other
    # flux state must FAIL LOUDLY — a silent no-op would run LL/RG transport
    # with ~2*steps_per_window-times overstrength winds.
    Models = AtmosTransport.Models
    FT = Float64
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT)
    fluxes_rg = allocate_face_fluxes(mesh, 2; FT = FT, basis = DryBasis)
    @test_throws ArgumentError Models._scale_runtime_fluxes!(fluxes_rg, FT(0.5))

    # CS flux state: am/bm/cm scale, nothing else does.
    cs_mesh = CubedSphereMesh(; FT = FT, Nc = 4, Hp = 1)
    fluxes_cs = allocate_face_fluxes(cs_mesh, 2; FT = FT, basis = DryBasis)
    for p in 1:6
        fluxes_cs.am[p] .= 2.0
        fluxes_cs.bm[p] .= 4.0
        fluxes_cs.cm[p] .= 8.0
    end
    Models._scale_runtime_fluxes!(fluxes_cs, FT(0.5))
    @test all(all(fluxes_cs.am[p] .== 1.0) for p in 1:6)
    @test all(all(fluxes_cs.bm[p] .== 2.0) for p in 1:6)
    @test all(all(fluxes_cs.cm[p] .== 4.0) for p in 1:6)
end

@testset "cubed-sphere window refresh copies diffusion forcing" begin
    FT = Float64
    mesh = CubedSphereMesh(; FT, Nc=2, Hp=1)
    Nz = 2
    N = mesh.Nc + 2mesh.Hp

    function make_window(value)
        air_mass = ntuple(_ -> fill(FT(value), N, N, Nz), 6)
        ps = ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc), 6)
        fluxes = allocate_face_fluxes(mesh, Nz; FT, basis=DryBasis)
        surface = AtmosTransport.MetDrivers.PBLSurfaceForcing(
            ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc), 6),
            ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc), 6),
            ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc), 6),
            ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc), 6))
        vdiff = (
            u = ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc, Nz), 6),
            v = ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc, Nz), 6),
            t = ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc, Nz), 6),
            qv = ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc, Nz), 6))
        dkg = ntuple(_ -> fill(FT(value), mesh.Nc, mesh.Nc, Nz), 6)
        return AtmosTransport.MetDrivers.TransportWindow(
            air_mass, ps, fluxes; surface, vdiff, dkg)
    end

    destination = make_window(1)
    source = make_window(2)
    AtmosTransport.Models._copy_window_payload!(destination, source)

    @test all(all(==(FT(2)), destination.vdiff.u[p]) for p in 1:6)
    @test all(all(==(FT(2)), destination.vdiff.v[p]) for p in 1:6)
    @test all(all(==(FT(2)), destination.vdiff.t[p]) for p in 1:6)
    @test all(all(==(FT(2)), destination.vdiff.qv[p]) for p in 1:6)
    @test all(all(==(FT(2)), destination.dkg[p]) for p in 1:6)
end
