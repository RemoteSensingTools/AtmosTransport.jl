#!/usr/bin/env julia

using Test

using AtmosTransport
include(joinpath(@__DIR__, "..", "helpers", "convection_runtime.jl"))

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

# `NoConvection` is a dead branch: after warm-up, `apply!` allocates nothing.
function _no_convection_apply_allocs(state, forcing, grid, dt)
    apply!(state, forcing, grid, NoConvection(), dt; workspace = nothing)
    return @allocated apply!(state, forcing, grid, NoConvection(), dt; workspace = nothing)
end

@testset "TransportModel convection runtime" begin

    @testset "NoConvection apply! is an allocation-free identity, also on cubed-sphere states" begin
        FT = Float64
        grid = _make_convection_grid(FT = FT, Nx = 2, Ny = 2)
        m = ones(FT, 2, 2, 5)
        state = CellState(m; CO2 = FT(400e-6) .* m)
        @test _no_convection_apply_allocs(state, ConvectionForcing(), grid, FT(1)) == 0

        Nc, Hp, Nz = 2, 1, 2
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        cs_grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1]), CPU(); FT = FT)
        dims = (Nc + 2Hp, Nc + 2Hp, Nz)
        cs_state = CubedSphereState(DryBasis, mesh, ntuple(_ -> ones(FT, dims), 6);
                                    CO2 = ntuple(_ -> fill(FT(400e-6), dims), 6))
        air_before = deepcopy(cs_state.air_mass)
        raw_before = deepcopy(cs_state.tracers_raw)
        result = apply!(cs_state, ConvectionForcing(), cs_grid, NoConvection(), FT(1800);
                        workspace = nothing)
        @test result === cs_state
        @test cs_state.air_mass == air_before
        @test cs_state.tracers_raw == raw_before
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

@testset "DrivenSimulation convection runtime and model precision" begin
    for FT in (Float32,Float64), tm5 in (false,true)
        driver, forcing_a, forcing_b = _make_convection_window_driver(;FT,tm5)
        m = copy(driver.windows[1].air_mass)
        tracer = zeros(FT,4,3,5)
        tracer[:,:,5] .= FT(1e-6).*m[:,:,5]
        state = CellState(m;CO2=tracer)
        fluxes = allocate_face_fluxes(driver.grid.horizontal,5;FT,basis=DryBasis)
        op = tm5 ? TM5Convection() : CMFMCConvection()
        model = TransportModel(state,fluxes,driver.grid,UpwindScheme();convection=op)
        sim = DrivenSimulation(model,driver;start_window=1,stop_window=2)
        payload(f) = tm5 ? f.tm5_fields.entu : f.cmfmc
        @test typeof(sim.Δt) === FT
        @test typeof(sim.window_dt) === FT
        @test eltype(payload(sim.model.convection_forcing)) === FT
        @test payload(sim.model.convection_forcing) == payload(forcing_a)
        initial = copy(sim.model.state.tracers_raw)
        mass0 = sum(Float64,initial)
        step!(sim)
        @test isapprox(sum(Float64,sim.model.state.tracers_raw),mass0;rtol=100eps(FT))
        step!(sim)
        @test window_index(sim) == 2
        @test payload(sim.model.convection_forcing) == payload(forcing_b)
        @test isapprox(sum(Float64,sim.model.state.tracers_raw),mass0;rtol=100eps(FT))
        @test sim.model.state.tracers_raw != initial
        @test sum(sim.model.state.tracers_raw[:,:,5,1]) < sum(initial[:,:,5,1])
    end
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
