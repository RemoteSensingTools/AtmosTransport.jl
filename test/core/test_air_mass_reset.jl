using AtmosTransport, Test
const ResetPlans = AtmosTransport.Operators.StepPlans
const ResetAdvection = AtmosTransport.Operators.Advection

# The preserve-VMR reset rescales all tracers in one broadcast over the packed
# array; it must equal the per-tracer formula bit for bit, including cells
# whose old air mass is at or below the floor.
_reset_reference!(rm, old, new, floor_m) =
    (@. rm = ifelse(old > floor_m, rm / old * new, zero(eltype(rm))); rm)

@testset "Air-mass reset over the packed tracer array" begin
    for FT in (Float32, Float64)
        floor_m = eps(FT)
        @testset "CellState $FT" begin
            Nx, Ny, Nz, Nt = 4, 3, 2, 5
            old = FT[1 + i + 2j + 3k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
            old[2, 1, 1] = 0
            old[3, 2, 2] = floor_m
            new = FT[2 + 0.5i + j - k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
            tracers = (Symbol("t$n") => FT[n * (i + j + k) / 7 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
                       for n in 1:Nt)
            state = CellState(copy(old); tracers...)
            expected = [_reset_reference!(copy(selectdim(state.tracers_raw, 4, n)), old, new, floor_m)
                        for n in 1:Nt]
            ResetPlans._reset_air_mass!(state, new, nothing, :preserve_vmr)
            @test all(selectdim(state.tracers_raw, 4, n) == expected[n] for n in 1:Nt)
            @test state.air_mass == new
        end
        @testset "CellState reduced-Gaussian shape $FT" begin
            Ncell, Nz, Nt = 7, 3, 4
            old = FT[1 + c + 2k for c in 1:Ncell, k in 1:Nz]
            old[4, 2] = 0
            new = FT[3 + 0.25c - k for c in 1:Ncell, k in 1:Nz]
            tracers = (Symbol("t$n") => FT[n * (c + k) / 5 for c in 1:Ncell, k in 1:Nz] for n in 1:Nt)
            state = CellState(copy(old); tracers...)
            expected = [_reset_reference!(copy(selectdim(state.tracers_raw, 3, n)), old, new, floor_m)
                        for n in 1:Nt]
            ResetPlans._reset_air_mass!(state, new, nothing, :preserve_vmr)
            @test all(selectdim(state.tracers_raw, 3, n) == expected[n] for n in 1:Nt)
            @test state.air_mass == new
        end
        @testset "CubedSphereState $FT" begin
            Nc, Hp, Nz, Nt = 3, 1, 2, 5
            mesh = CubedSphereMesh(; Nc, Hp, FT)
            N = Nc + 2Hp
            old = ntuple(p -> FT[p + i + 2j + 3k for i in 1:N, j in 1:N, k in 1:Nz], 6)
            old[2][2, 2, 1] = 0
            new = ntuple(p -> FT[2p + 0.5i + j - k for i in 1:N, j in 1:N, k in 1:Nz], 6)
            tracers = (Symbol("t$n") => ntuple(p -> FT[n * (p + i + j + k) / 7
                                                        for i in 1:N, j in 1:N, k in 1:Nz], 6)
                       for n in 1:Nt)
            state = CubedSphereState(DryBasis, mesh, map(copy, old); tracers...)
            reference = deepcopy(state)
            new_filled = map(copy, new)
            ResetAdvection.fill_panel_halos!(new_filled, mesh; dir = 1)
            for p in 1:6, n in 1:Nt
                _reset_reference!(selectdim(reference.tracers_raw[p], 4, n), old[p],
                                  new_filled[p], floor_m)
            end
            foreach(copyto!, reference.air_mass, new_filled)
            ResetPlans._refresh_state_halos!(reference, mesh)
            ResetPlans._reset_air_mass!(state, map(copy, new), mesh, :preserve_vmr)
            @test all(state.tracers_raw[p] == reference.tracers_raw[p] for p in 1:6)
            @test all(state.air_mass[p] == reference.air_mass[p] for p in 1:6)
        end
    end
end

# ---------------------------------------------------------------------------
# Runs without advection: the air mass follows the met window every step
# (`[run] air_mass_update`, default "step" for `NoAdvection`).
#
# Nothing moves the air mass of an advection-off run inside a met window, so
# before this update it stayed at the window-start air mass until the
# window-end reset, and diffusion and convection weighted the layers by masses
# up to a window old. Now every step starts by resetting it to the window's air
# mass at the end of that step, `m + λ dm`, as GEOS-Chem recomputes its air
# mass before every step (AirQnt).
# ---------------------------------------------------------------------------

using Logging
include(joinpath(@__DIR__, "..", "helpers", "convection_runtime.jl"))
const ResetMD = AtmosTransport.MetDrivers

# Two met windows on the lat-lon convection grid whose air mass changes by a
# layer- and column-dependent `dm` over each window (the second window starts
# where the first ends, plus a jump the window-advance reset installs).
function _moving_air_mass_driver(FT; steps = 4)
    grid = _make_convection_grid(FT = FT)
    Nx, Ny, Nz = 4, 3, 5
    m1 = FT[_REALISTIC_AIR_MASS_KG * (1 + 0.05k + 0.01i) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    dm1 = FT[1e-2 * _REALISTIC_AIR_MASS_KG * (k - 3) * (1 + 0.1i) * (1 + 0.2j) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    m2 = m1 .+ dm1 .+ FT(1e-3 * _REALISTIC_AIR_MASS_KG)
    dm2 = -FT(0.5) .* dm1
    ps = fill(FT(95_000), Nx, Ny)
    window(m, dm, peak) = begin
        fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)
        deltas = ResetMD.StructuredFluxDeltas(zero(fluxes.am), zero(fluxes.bm), zero(fluxes.cm), dm)
        TransportWindow(m, ps, fluxes; deltas,
                        convection = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak, top_detrain = FT(0.01)))
    end
    windows = [window(m1, dm1, FT(0.05)), window(m2, dm2, FT(0.08))]
    return _ConvectionWindowDriver{FT, typeof(grid), eltype(windows)}(grid, windows, FT(3600), steps, true)
end

function _moving_air_mass_sim(driver; advection = NoAdvection(), kwargs...)
    FT = eltype(driver.windows[1].air_mass)
    m = copy(driver.windows[1].air_mass)
    rm = FT[1e-6 * m[i, j, k] * (1 + 0.3k) for i in axes(m, 1), j in axes(m, 2), k in axes(m, 3)]
    state = CellState(m; CO2 = rm)
    fluxes = allocate_face_fluxes(driver.grid.horizontal, size(m, 3); FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, driver.grid, advection; convection = CMFMCConvection())
    return DrivenSimulation(model, driver; kwargs...)
end

_window_air_mass_at(window, λ) = ResetMD.expected_air_mass!(similar(window.air_mass), window, λ)

@testset "Advection off: the air mass follows the met window every step" begin
    for FT in (Float64, Float32), cadence in (:substep, :window)
        driver = _moving_air_mass_driver(FT)
        steps = driver.steps
        sim = _moving_air_mass_sim(driver; physics_cadence = cadence)
        @test first(sim.plan.step) == ResetPlans.AirMassResetStage(:preserve_tracer_mass,
                                                                    ResetPlans.StepEndAirMass())
        tracer_mass = sum(Float64, sim.model.state.tracers_raw)
        for w in 1:2, k in 1:steps
            step!(sim)
            # The step's operators ran on the window's air mass at the step end.
            @test sim.model.state.air_mass == _window_air_mass_at(driver.windows[w], k / steps)
            @test isapprox(sum(Float64, sim.model.state.tracers_raw), tracer_mass;
                           rtol = FT === Float64 ? 1e-13 : 1e-6)
        end
    end

    # Convection mixes on the step's layer masses: the run equals convection
    # applied by hand to the tracer after installing each step's end air mass.
    for FT in (Float64, Float32)
        driver = _moving_air_mass_driver(FT)
        steps = driver.steps
        sim = _moving_air_mass_sim(driver; physics_cadence = :substep)
        reference = _moving_air_mass_sim(driver; physics_cadence = :substep)
        state = reference.model.state
        op, ws = reference.model.convection, reference.model.workspace.convection_ws
        for w in 1:2
            forcing = driver.windows[w].convection
            AtmosTransport.invalidate_cmfmc_cache!(ws)
            for k in 1:steps
                copyto!(state.air_mass, _window_air_mass_at(driver.windows[w], k / steps))
                AtmosTransport.Operators.apply!(state, forcing, driver.grid, op, sim.Δt; workspace = ws)
                step!(sim)
                @test sim.model.state.tracers_raw == state.tracers_raw
            end
        end
        # The window-start air mass (the opt-out) mixes differently.
        old = _moving_air_mass_sim(driver; physics_cadence = :substep, air_mass_update = :window)
        step!(old)
        @test old.model.state.air_mass == driver.windows[1].air_mass
        fresh = _moving_air_mass_sim(driver; physics_cadence = :substep)
        step!(fresh)
        @test old.model.state.tracers_raw != fresh.model.state.tracers_raw
    end

    # preserve_vmr resets keep the mixing ratio instead (and so the tracer
    # mass follows the air mass); the target is the same.
    driver = _moving_air_mass_driver(Float64)
    sim = _moving_air_mass_sim(driver; physics_cadence = :substep, air_mass_reset_mode = :preserve_vmr)
    @test first(sim.plan.step) == ResetPlans.AirMassResetStage(:preserve_vmr, ResetPlans.StepEndAirMass())
    step!(sim)
    @test sim.model.state.air_mass == _window_air_mass_at(driver.windows[1], 1 / driver.steps)
end

@testset "air_mass_update: defaults, opt-out and rejections" begin
    driver = _moving_air_mass_driver(Float64)
    has_step_reset(sim) = any(s -> s isa ResetPlans.AirMassResetStage{ResetPlans.StepEndAirMass},
                              ResetPlans.plan_stages(sim.plan))
    # Opt-out: the window-start air mass until the window-end reset.
    sim = _moving_air_mass_sim(driver; air_mass_update = "window")
    @test !has_step_reset(sim)
    for _ in 1:driver.steps - 1
        step!(sim)
        @test sim.model.state.air_mass == driver.windows[1].air_mass
    end
    step!(sim)
    @test sim.model.state.air_mass == _window_air_mass_at(driver.windows[1], 1)
    # No reset mode: nothing to update the air mass with.
    @test !has_step_reset(_moving_air_mass_sim(driver; air_mass_reset_mode = :none))
    @test_throws ArgumentError _moving_air_mass_sim(driver; air_mass_update = :step,
                                                    air_mass_reset_mode = :none)
    @test_throws ArgumentError _moving_air_mass_sim(driver; air_mass_update = :hourly)

    # Advection on: the plan is the one without the option, bit for bit the
    # same run; "step" is rejected (advection moves the air mass itself).
    for cadence in (:substep, :window), mode in (:preserve_tracer_mass, :preserve_vmr, :none)
        sim = _moving_air_mass_sim(driver; advection = UpwindScheme(), physics_cadence = cadence,
                                   air_mass_reset_mode = mode)
        @test sim.plan == ResetPlans.StepPlan(; physics_cadence = cadence, window_contract = true,
                                              air_mass_reset = mode)
    end
    a = _moving_air_mass_sim(driver; advection = UpwindScheme())
    b = _moving_air_mass_sim(driver; advection = UpwindScheme(), air_mass_update = :window)
    run!(a); run!(b)
    @test a.model.state.tracers_raw == b.model.state.tracers_raw
    @test a.model.state.air_mass == b.model.state.air_mass
    err = try _moving_air_mass_sim(driver; advection = UpwindScheme(), air_mass_update = :step)
          catch e; e end
    @test err isa ArgumentError && occursin("without advection", err.msg)

    # The resolution rule itself, and the plan.
    resolve(v; advects, mode = :preserve_tracer_mass) =
        ResetPlans.resolve_air_mass_update(v; advects_air_mass = advects, air_mass_reset = mode)
    @test resolve(nothing; advects = false) === :step
    @test resolve(nothing; advects = false, mode = :none) === :window
    @test resolve(nothing; advects = true) === :window
    @test resolve("window"; advects = false) === :window
    @test resolve("step"; advects = false, mode = "preserve_vmr") === :step
    @test_throws ArgumentError resolve("step"; advects = true)
    @test_throws ArgumentError resolve("Step"; advects = false)
    @test ResetPlans.advects_air_mass(NoAdvection()) === false
    @test ResetPlans.advects_air_mass(UpwindScheme()) === true
    @test_throws ArgumentError ResetPlans.StepPlan(; air_mass_update = :step)
    plan = ResetPlans.StepPlan(; window_contract = true, air_mass_reset = :preserve_tracer_mass,
                               air_mass_update = :step)
    @test plan.step == (ResetPlans.AirMassResetStage(:preserve_tracer_mass, ResetPlans.StepEndAirMass()),
                        ResetPlans.TransportStage())
    @test occursin("step end", sprint(show, MIME"text/plain"(), plan))

    # No adjoint record (NoAdvection itself has none either).
    gap = AtmosTransport.Adjoints.adjoint_support(
        ResetPlans.AirMassResetStage(:preserve_tracer_mass, ResetPlans.StepEndAirMass()))
    @test gap isa String && occursin("air_mass_update", gap)
end

@testset "air_mass_update in validate_config and the output metadata" begin
    mktemp() do path, io
        cfg(run; scheme = "none") = Dict{String, Any}(
            "input" => Dict{String, Any}("binary_paths" => [path]),
            "advection" => Dict{String, Any}("scheme" => scheme), "run" => run)
        errors(c) = with_logger(() -> last(validate_config(c)), NullLogger())
        @test isempty(errors(cfg(Dict{String, Any}())))
        @test isempty(errors(cfg(Dict{String, Any}("air_mass_update" => "window"))))
        @test isempty(errors(cfg(Dict{String, Any}("air_mass_update" => "step"))))
        @test isempty(errors(cfg(Dict{String, Any}("air_mass_update" => "window"); scheme = "upwind")))
        @test any(contains("air_mass_update"), errors(cfg(Dict{String, Any}("air_mass_update" => "hourly"))))
        @test any(contains("without advection"),
                  errors(cfg(Dict{String, Any}("air_mass_update" => "step"); scheme = "upwind")))
        @test any(contains("air_mass_reset_mode"),
                  errors(cfg(Dict{String, Any}("air_mass_update" => "step", "air_mass_reset_mode" => "none"))))
    end
    # The resolved choice is recorded with the run's physics metadata, only for
    # runs without advection (advection-on outputs keep their attribute set).
    fake_driver = (; reader = (; header = (; raw_header = Dict{String, Any}(), format_version = 1)))
    attribute(scheme, run) = get(AtmosTransport.Models.Physics.runtime_physics_attributes(
        Dict{String, Any}("advection" => Dict{String, Any}("scheme" => scheme), "run" => run),
        fake_driver, (), :dry, String[]), "physics_air_mass_update", nothing)
    @test attribute("none", Dict{String, Any}()) == "step"
    @test attribute("none", Dict{String, Any}("air_mass_update" => "window")) == "window"
    @test attribute("none", Dict{String, Any}("air_mass_reset_mode" => "none")) == "window"
    @test attribute("upwind", Dict{String, Any}()) === nothing
end

include(joinpath(@__DIR__, "..", "helpers", "cs_adjoint_step_plan.jl"))

@testset "Advection off on the cubed sphere: per-step air mass, conservation, no adjoint" begin
    # Three windows of 2, 3 and 2 steps with their own air mass and dm.
    for cadence in (:substep, nothing)
        c = Case(scheme = AT.NoAdvection(), conv = :cmfmc, emissions = false, cadence = cadence)
        sim = _simulation(c)
        windows = sim.driver.windows
        total(panels) = sum(p -> sum(Float64, p), _interior(panels))
        mass0 = total(_rm(sim))
        for (w, n) in enumerate(c.steps), k in 1:n
            step!(sim)
            target = AT.MetDrivers.expected_air_mass!(map(similar, windows[w].air_mass), windows[w], k / n)
            @test _interior(sim.model.state.air_mass) == _interior(target)
            @test isapprox(total(_rm(sim)), mass0; rtol = 1e-13)
        end
    end
    # Advection-off runs have no cubed-sphere adjoint; the guard names both
    # the scheme and the per-step reset.
    sim = _simulation(Case(scheme = AT.NoAdvection(), conv = :cmfmc, emissions = false))
    err = try AT.require_adjoint_support(sim); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("air_mass_update", err.msg) && occursin("NoAdvection", err.msg)
    @test_throws ArgumentError AT.cs_adjoint_inputs(sim; tracer = :CO2)
end
