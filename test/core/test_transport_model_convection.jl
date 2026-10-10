#!/usr/bin/env julia

using Test

using AtmosTransport

const _REALISTIC_AIR_MASS_KG = 1e16

function _make_convection_grid(; FT = Float64, Nx = 4, Ny = 3, Nz = 5)
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    return AtmosGrid(mesh, vertical, CPU(); FT = FT)
end

function _make_convection_model(; FT = Float64, Nx = 4, Ny = 3, Nz = 5,
                                 convection::AbstractConvection = NoConvection(),
                                 convection_forcing::ConvectionForcing = ConvectionForcing())
    grid = _make_convection_grid(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz)
    air_mass = fill(FT(_REALISTIC_AIR_MASS_KG), Nx, Ny, Nz)
    tracer = zeros(FT, Nx, Ny, Nz)
    tracer[:, :, Nz] .= FT(1e-6) .* air_mass[:, :, Nz]
    state = CellState(air_mass; CO2 = tracer)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)
    return TransportModel(state, fluxes, grid, UpwindScheme();
                          convection = convection,
                          convection_forcing = convection_forcing)
end

function _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.02), top_detrain = FT(0.01))
    cmfmc = zeros(FT, Nx, Ny, Nz + 1)
    cmfmc[:, :, 4] .= peak * FT(0.5)
    cmfmc[:, :, 3] .= peak
    cmfmc[:, :, 2] .= peak * FT(0.5)

    dtrain = zeros(FT, Nx, Ny, Nz)
    dtrain[:, :, 1] .= top_detrain
    return ConvectionForcing(cmfmc, dtrain, nothing)
end

function _make_rg_cmfmc_forcing(FT, ncell, Nz; peak = FT(0.02), top_detrain = FT(0.01))
    cmfmc = zeros(FT, ncell, Nz + 1)
    cmfmc[:, 4] .= peak * FT(0.5)
    cmfmc[:, 3] .= peak
    cmfmc[:, 2] .= peak * FT(0.5)

    dtrain = zeros(FT, ncell, Nz)
    dtrain[:, 1] .= top_detrain
    return ConvectionForcing(cmfmc, dtrain, nothing)
end

function _make_rg_convection_model(; FT = Float64, Nz = 5,
                                    convection::AbstractConvection = NoConvection(),
                                    convection_forcing::ConvectionForcing = ConvectionForcing())
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    ncell = ncells(mesh)

    air_mass = fill(FT(_REALISTIC_AIR_MASS_KG), ncell, Nz)
    tracer = zeros(FT, ncell, Nz)
    tracer[:, Nz] .= FT(1e-6) .* air_mass[:, Nz]
    state = CellState(air_mass; CO2 = tracer)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)
    return TransportModel(state, fluxes, grid, UpwindScheme();
                          convection = convection,
                          convection_forcing = convection_forcing)
end

struct _ConvectionWindowDriver{FT, GridT, WindowT} <: AbstractMetDriver
    grid    :: GridT
    windows :: Vector{WindowT}
    dt      :: FT
    steps   :: Int
    binary_contract :: Bool
end

AtmosTransport.total_windows(driver::_ConvectionWindowDriver) = length(driver.windows)
AtmosTransport.window_dt(driver::_ConvectionWindowDriver) = driver.dt
AtmosTransport.steps_per_window(driver::_ConvectionWindowDriver) = driver.steps
AtmosTransport.load_transport_window(driver::_ConvectionWindowDriver, win::Int) = driver.windows[win]
AtmosTransport.driver_grid(driver::_ConvectionWindowDriver) = driver.grid
AtmosTransport.air_mass_basis(::_ConvectionWindowDriver) = :dry
AtmosTransport.MetDrivers.flux_interpolation_mode(::_ConvectionWindowDriver) = :constant
AtmosTransport.MetDrivers.supports_native_vertical_flux(::_ConvectionWindowDriver) = true
AtmosTransport.MetDrivers.uses_binary_substep_contract(driver::_ConvectionWindowDriver) =
    driver.binary_contract
AtmosTransport.supports_convection(::_ConvectionWindowDriver) = true

struct _CountingChemistry <: AbstractChemistryOperator
    calls    :: Base.RefValue{Int}
    total_dt :: Base.RefValue{Float64}
end

function AtmosTransport.Operators.apply!(state, meteo, grid,
                                         op::_CountingChemistry, dt;
                                         workspace = nothing)
    op.calls[] += 1
    op.total_dt[] += Float64(dt)
    return state
end

function _make_convection_window_driver(; FT = Float64, steps = 1,
                                        binary_contract = false)
    grid = _make_convection_grid(FT = FT)
    Nx, Ny, Nz = 4, 3, 5
    air_mass = fill(FT(_REALISTIC_AIR_MASS_KG), Nx, Ny, Nz)
    ps = fill(FT(95_000), Nx, Ny)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)

    forcing_a = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.02), top_detrain = FT(0.01))
    forcing_b = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.5), top_detrain = FT(0.1))

    window_a = TransportWindow(air_mass, ps, fluxes; convection = forcing_a)
    window_b = TransportWindow(air_mass, ps, fluxes; convection = forcing_b)
    driver = _ConvectionWindowDriver{FT, typeof(grid), typeof(window_a)}(
        grid, [window_a, window_b], FT(1800), Int(steps), Bool(binary_contract))
    return driver, forcing_a, forcing_b
end

@testset "TransportModel convection runtime" begin
    @testset "default model carries NoConvection and no convection workspace" begin
        model = _make_convection_model()
        @test model.convection isa NoConvection
        @test model.workspace.convection_ws === nothing
    end

    @testset "with_convection installs structured CMFMC workspace" begin
        base = _make_convection_model()
        updated = with_convection(base, CMFMCConvection())

        @test updated.convection isa CMFMCConvection
        @test updated.convection_forcing === base.convection_forcing
        @test updated.state === base.state
        @test updated.fluxes === base.fluxes
        @test updated.grid === base.grid
        @test updated.advection === base.advection
        @test updated.chemistry === base.chemistry
        @test updated.diffusion === base.diffusion
        @test updated.emissions === base.emissions
        @test updated.workspace !== base.workspace
        @test updated.workspace.advection_ws === base.workspace.advection_ws
        @test updated.workspace.convection_ws isa CMFMCWorkspace
        @test updated.workspace.diffusion_ws === nothing
    end

    @testset "step! with default NoConvection stays bit-exact" begin
        model_a = _make_convection_model()
        model_b = _make_convection_model()

        for _ in 1:3
            step!(model_a, 1800.0)
            step!(model_b, 1800.0)
        end

        @test model_a.state.tracers_raw == model_b.state.tracers_raw
    end

    @testset "step! with CMFMCConvection redistributes vertically" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 5
        forcing = _make_cmfmc_forcing(FT, Nx, Ny, Nz)

        model_ctrl = _make_convection_model(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz)
        model_conv = _make_convection_model(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz)
        model_conv = with_convection(model_conv, CMFMCConvection())
        model_conv = with_convection_forcing(model_conv, forcing)

        rm_before = copy(model_conv.state.tracers_raw)
        total_before = sum(rm_before)

        step!(model_ctrl, FT(1800))
        step!(model_conv, FT(1800))

        @test model_ctrl.state.tracers_raw == rm_before
        @test model_conv.state.tracers_raw != rm_before
        # CMFMCConvection is now the GCHP RAS scheme written in true interface-
        # flux-divergence form (cmfmc_kernels.jl Pass 2): every interior interface
        # flux telescopes, so column dry-tracer mass is conserved to machine
        # precision — no fixer, no matrix. (Previously a simplified 2-term
        # tendency drifted ~10⁻¹.) The non-conservative GCHP clamp is omitted.
        @test isapprox(sum(model_conv.state.tracers_raw), total_before; rtol = 1e-12)
        @test model_conv.state.tracers.CO2[1, 1, Nz] < rm_before[1, 1, Nz, 1]
        @test maximum(model_conv.state.tracers.CO2[:, :, 1:(Nz - 1)]) > 0
    end

    @testset "step! reads convection forcing from the model" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 5

        weak = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.02), top_detrain = FT(0.01))
        strong = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.5), top_detrain = FT(0.1))

        model_weak = with_convection(_make_convection_model(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz), CMFMCConvection())
        model_strong = with_convection(_make_convection_model(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz), CMFMCConvection())
        model_weak = with_convection_forcing(model_weak, weak)
        model_strong = with_convection_forcing(model_strong, strong)

        step!(model_weak, FT(1800))
        step!(model_strong, FT(1800))

        @test model_weak.state.tracers_raw != model_strong.state.tracers_raw
    end

    @testset "ReducedGaussian step! with CMFMCConvection redistributes vertically" begin
        FT = Float64
        Nz = 5
        ncell = ncells(ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT))
        forcing = _make_rg_cmfmc_forcing(FT, ncell, Nz; peak = FT(0.02), top_detrain = FT(0.01))

        model = with_convection(_make_rg_convection_model(FT = FT, Nz = Nz), CMFMCConvection())
        model = with_convection_forcing(model, forcing)
        rm_before = copy(model.state.tracers_raw)

        step!(model, FT(1800))

        # Reduced-Gaussian CMFMCConvection conserves column mass to machine
        # precision via the same interface-flux-divergence form (see the LL
        # testset above for the derivation).
        @test abs(sum(model.state.tracers_raw) - sum(rm_before)) / sum(rm_before) < 1e-12
        @test model.state.tracers_raw != rm_before
    end
end

@testset "DrivenSimulation convection runtime" begin
    FT = Float64
    driver, forcing_a, forcing_b = _make_convection_window_driver(FT = FT)

    state = CellState(fill(FT(_REALISTIC_AIR_MASS_KG), 4, 3, 5);
                      CO2 = fill(FT(1e-6 * _REALISTIC_AIR_MASS_KG), 4, 3, 5))
    fluxes = allocate_face_fluxes(driver.grid.horizontal, 5; FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, driver.grid, UpwindScheme();
                           convection = CMFMCConvection())

    sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 2)

    @test sim.model.workspace.convection_ws isa CMFMCWorkspace
    @test sim.model.convection_forcing !== forcing_a
    @test sim.model.convection_forcing.cmfmc !== forcing_a.cmfmc
    @test sim.model.convection_forcing.cmfmc == forcing_a.cmfmc

    step!(sim)
    @test sim.model.workspace.convection_ws.cached_n_sub[] == 1
    @test sim.model.workspace.convection_ws.cache_valid[] == true

    step!(sim)
    @test window_index(sim) == 2
    @test sim.model.convection_forcing.cmfmc == forcing_b.cmfmc
    @test sim.model.workspace.convection_ws.cached_n_sub[] > 1
    @test sim.model.workspace.convection_ws.cache_valid[] == true
end

@testset "binary substep contract keeps chemistry at window cadence" begin
    FT = Float64

    driver_step, _, _ = _make_convection_window_driver(
        FT = FT, steps = 4, binary_contract = false)
    state_step = CellState(fill(FT(_REALISTIC_AIR_MASS_KG), 4, 3, 5);
                           CO2 = fill(FT(1e-6 * _REALISTIC_AIR_MASS_KG), 4, 3, 5))
    fluxes_step = allocate_face_fluxes(driver_step.grid.horizontal, 5; FT = FT, basis = DryBasis)
    model_step = TransportModel(state_step, fluxes_step, driver_step.grid, UpwindScheme())
    chem_step = _CountingChemistry(Ref(0), Ref(0.0))
    sim_step = DrivenSimulation(model_step, driver_step;
                                start_window = 1, stop_window = 1,
                                chemistry = chem_step)
    run!(sim_step)

    @test chem_step.calls[] == 4
    @test chem_step.total_dt[] == 1800.0

    driver_window, _, _ = _make_convection_window_driver(
        FT = FT, steps = 4, binary_contract = true)
    state_window = CellState(fill(FT(_REALISTIC_AIR_MASS_KG), 4, 3, 5);
                             CO2 = fill(FT(1e-6 * _REALISTIC_AIR_MASS_KG), 4, 3, 5))
    fluxes_window = allocate_face_fluxes(driver_window.grid.horizontal, 5; FT = FT, basis = DryBasis)
    model_window = TransportModel(state_window, fluxes_window, driver_window.grid, UpwindScheme())
    chem_window = _CountingChemistry(Ref(0), Ref(0.0))
    sim_window = DrivenSimulation(model_window, driver_window;
                                  start_window = 1, stop_window = 1,
                                  chemistry = chem_window)
    run!(sim_window)

    @test chem_window.calls[] == 1
    @test chem_window.total_dt[] == 1800.0

    # `[run] physics_cadence = "substep"` runs chemistry every advection substep on
    # the same binary-contract driver (cadence A/B comparisons).
    state_sub = CellState(fill(FT(_REALISTIC_AIR_MASS_KG), 4, 3, 5);
                          CO2 = fill(FT(1e-6 * _REALISTIC_AIR_MASS_KG), 4, 3, 5))
    fluxes_sub = allocate_face_fluxes(driver_window.grid.horizontal, 5; FT = FT, basis = DryBasis)
    model_sub = TransportModel(state_sub, fluxes_sub, driver_window.grid, UpwindScheme())
    chem_sub = _CountingChemistry(Ref(0), Ref(0.0))
    sim_sub = DrivenSimulation(model_sub, driver_window; start_window = 1, stop_window = 1,
                               chemistry = chem_sub, physics_cadence = "substep")
    run!(sim_sub)
    @test chem_sub.calls[] == 4
    @test chem_sub.total_dt[] == 1800.0
    # Both cadences keep the binary's window-end air-mass reset.
    @test AtmosTransport.Models._binary_window_contract(sim_sub)
    @test !AtmosTransport.Models._uses_binary_transport_schedule(sim_sub)
    @test AtmosTransport.Models._uses_binary_transport_schedule(sim_window)

    resolve = AtmosTransport.Models._resolve_physics_cadence
    withenv("ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS" => nothing) do
        @test resolve(nothing) === :window
        @test resolve(:substep) === :substep
        @test_throws ArgumentError resolve("hourly")
    end
    withenv("ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS" => "1") do
        @test (@test_logs (:warn, r"deprecated") resolve(nothing)) === :substep
        @test_throws ArgumentError resolve(:window)
    end
end

@testset "DrivenSimulation keeps convection runtime on model FT" begin
    FT = Float32
    driver, forcing_a, _ = _make_convection_window_driver(FT = FT)

    state = CellState(fill(FT(_REALISTIC_AIR_MASS_KG), 4, 3, 5);
                      CO2 = fill(FT(1e-6 * _REALISTIC_AIR_MASS_KG), 4, 3, 5))
    fluxes = allocate_face_fluxes(driver.grid.horizontal, 5; FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, driver.grid, UpwindScheme();
                           convection = CMFMCConvection())

    sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)

    @test typeof(sim.Δt) === FT
    @test typeof(sim.window_dt) === FT
    @test eltype(sim.model.convection_forcing.cmfmc) === FT
    @test sim.model.convection_forcing.cmfmc == forcing_a.cmfmc
end

@testset "New DrivenSimulation invalidates convection from a previous driver" begin
    # Splitting the same two windows across two daily drivers must agree with
    # advancing inside one driver, even though the numerical workspace persists.
    for op in (CMFMCConvection(), CMFMCMatrixConvection())
        driver, _, _ = _make_convection_window_driver(binary_contract=true)
        make_model() = begin
            model = _make_convection_model(convection=op)
            for k in 1:5
                model.state.tracers_raw[:,:,k,:] .= k * 1e-6 * _REALISTIC_AIR_MASS_KG
            end
            model
        end
        continuous = DrivenSimulation(make_model(),driver)
        run!(continuous)
        driver1 = typeof(driver)(driver.grid,[driver.windows[1]],driver.dt,driver.steps,true)
        driver2 = typeof(driver)(driver.grid,[driver.windows[2]],driver.dt,driver.steps,true)
        day1 = DrivenSimulation(make_model(),driver1)
        run!(day1)
        workspace = day1.model.workspace.convection_ws
        day2 = DrivenSimulation(day1.model,driver2;
                                initialize_air_mass=false,start_time=1800)
        @test day2.model.workspace.convection_ws === workspace
        run!(day2)
        @test day2.model.state.tracers_raw ≈ continuous.model.state.tracers_raw rtol=1e-13
        if op isa CMFMCConvection
            @test workspace.cached_n_sub[] == continuous.model.workspace.convection_ws.cached_n_sub[]
        else
            @test workspace.derived_entu == continuous.model.workspace.convection_ws.derived_entu
            @test workspace.derived_detu == continuous.model.workspace.convection_ws.derived_detu
        end
    end
end

using OffsetArrays
include(joinpath(@__DIR__, "..", "helpers", "cmfmc_cfl.jl"))

@testset "CMFMC CFL scan equals the per-interface definition" begin
    for FT in (Float32, Float64)
        f = cmfmc_cfl_fixture(FT)
        ref = reference_cmfmc_max_cfl(f)
        got = scan_cmfmc_max_cfl(f, identity, ref.dt)
        @test got.ll === ref.ll
        @test got.fi === ref.fi
        @test got.cs === ref.cs
        @test all(x -> x isa FT && x > 0, (got.ll, got.fi, got.cs))
        @test all(check_single_hot(f, identity))
        @test all(check_nan_mass_skips_interfaces(f, identity))

        # A NaN layer mass skips its interfaces; a NaN flux makes the scan NaN.
        f.ll.air_mass[5, 4, 3] = FT(NaN)
        f.cs.air_mass[2][f.Hp + 1, f.Hp + 3, 2] = FT(NaN)
        ref = reference_cmfmc_max_cfl(f)
        got = scan_cmfmc_max_cfl(f, identity, ref.dt)
        @test got.ll === ref.ll && isfinite(got.ll)
        @test got.cs === ref.cs && isfinite(got.cs)
        f.fi.cmfmc[3, 2] = FT(NaN)
        @test isnan(scan_cmfmc_max_cfl(f, identity, ref.dt).fi)
    end
end

# A one-based host array type with no KernelAbstractions backend method.
struct _PlainHostArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(a::_PlainHostArray) = size(a.data)
Base.getindex(a::_PlainHostArray, i::Int...) = a.data[i...]

@testset "CMFMC CFL scan copies non-Array CPU inputs to host Arrays" begin
    # Same values in reversed memory layout: indexing differs from parent indexing.
    reversed(a) = (p = ntuple(d -> ndims(a) + 1 - d, ndims(a));
                   PermutedDimsArray(permutedims(a, p), p))
    one_based(a) = OffsetArray(a, ntuple(_ -> 0, ndims(a)))
    shifted(a) = OffsetArray(a, ntuple(_ -> -1, ndims(a)))
    for FT in (Float32, Float64)
        f = cmfmc_cfl_fixture(FT)
        ref = reference_cmfmc_max_cfl(f)
        for wrap in (reversed, one_based, _PlainHostArray)
            got = scan_cmfmc_max_cfl(f, wrap, ref.dt)
            @test got.ll === ref.ll
            @test got.fi === ref.fi
            @test got.cs === ref.cs
        end
        # Non-one-based axes are rejected instead of read with shifted indices.
        @test_throws DimensionMismatch CMFMCConv._cmfmc_max_cfl(
            shifted(f.ll.cmfmc), f.ll.air_mass, f.ll.areas, ref.dt)
        @test_throws DimensionMismatch CMFMCConv._cmfmc_max_cfl(
            f.fi.cmfmc, shifted(f.fi.air_mass), f.fi.areas, ref.dt)
        @test_throws DimensionMismatch CMFMCConv._cmfmc_max_cfl(
            f.cs.cmfmc, f.cs.air_mass, map(shifted, f.cs.areas), ref.dt)
    end
end

@testset "CMFMC CFL sub-step count" begin
    for FT in (Float32, Float64)
        # Powers of two keep the ratio exact: bmass = 2^30 kg / 2^20 m² = 2^10 kg/m²
        # and dt = 2^10 s, so cmfmc·dt/bmass equals the cmfmc value.
        Nx, Ny, Nz = 3, 2, 4
        air_mass = fill(FT(2)^30, Nx, Ny, Nz)
        ws = CMFMCConv.CMFMCWorkspace(air_mass; cell_metrics = fill(FT(2)^20, Ny))
        function n_sub(worst; allow_clamp = false)
            cmfmc = zeros(FT, Nx, Ny, Nz + 1)
            cmfmc[2, 1, 3] = worst
            CMFMCConv.invalidate_cmfmc_cache!(ws)
            return CMFMCConv._get_or_compute_n_sub!(ws, cmfmc, air_mass, ws.cell_metrics,
                                                    1024; allow_clamp)
        end

        # n_sub = max(1, ceil(worst / 0.5)).
        @test n_sub(zero(FT)) == 1
        for k in (1, 2, 7, 48)
            @test n_sub(prevfloat(FT(k) / 2)) == k
            @test n_sub(FT(k) / 2) == k
            @test n_sub(nextfloat(FT(k) / 2)) == k + 1
        end

        # Unclamped, the safety ceiling is inclusive and one more sub-step throws;
        # clamped, the count is capped instead.
        n_max = CMFMCConv._CMFMC_N_SUB_MAX
        cap = CMFMCConv._CMFMC_CLAMP_N_SUB_CAP
        @test n_sub(FT(n_max) / 2) == n_max
        @test_throws ArgumentError n_sub(nextfloat(FT(n_max) / 2))
        @test n_sub(FT(cap) / 2; allow_clamp = true) == cap
        @test n_sub(nextfloat(FT(cap) / 2); allow_clamp = true) == cap
        @test n_sub(nextfloat(FT(n_max) / 2); allow_clamp = true) == cap
        # A huge finite ratio (beyond `Int`) hits the ceiling or the cap, not an InexactError.
        @test_throws ArgumentError n_sub(FT(1e19))
        @test n_sub(FT(1e19); allow_clamp = true) == cap

        # Non-finite fluxes are rejected in both modes.
        for bad in (FT(NaN), FT(Inf), FT(-Inf)), allow_clamp in (false, true)
            @test_throws ArgumentError n_sub(bad; allow_clamp)
        end
    end
end
