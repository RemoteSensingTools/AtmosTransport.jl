# Shared small driven CS configuration for CPU parity and opt-in GPU reset tests.
using Test
import AtmosTransport
const AT = AtmosTransport
const ADJ = AT.Adjoints
const SP = AT.Operators.StepPlans

const NC = 4
const NZ = 5
const S = 4e15          # kg per cell and layer (≈ 1500 kg m⁻² on C4 cells)
const WINDOW_DT = 3600.0

_close(a::AbstractArray, b::AbstractArray) = isapprox(a, b; rtol = 1e-13)
_close(a::Tuple, b::Tuple) = all(map(_close, a, b))
exact(a, b) = Threads.nthreads() == 1 ? a == b : _close(a, b)

# ---------------------------------------------------------------------------
# A mock cubed-sphere met driver: three windows with different air mass, mass
# fluxes, window mass change, convection forcing and (optionally) surface
# pressure, so the layer thickness of the diffusion changes per window, and
# either flux storage convention.
# ---------------------------------------------------------------------------

struct _PlanDriver{FT, G, W} <: AT.AbstractMetDriver
    grid::G
    windows::Vector{W}
    dt::FT
    steps::Vector{Int}
    contract::Bool
    flux_kind::Symbol
end
AT.total_windows(d::_PlanDriver) = length(d.windows)
AT.window_dt(d::_PlanDriver) = d.dt
AT.steps_per_window(d::_PlanDriver) = d.steps[1]
AT.MetDrivers.steps_per_window_schedule(d::_PlanDriver) = d.steps
AT.load_transport_window(d::_PlanDriver, w::Int) = d.windows[w]
AT.driver_grid(d::_PlanDriver) = d.grid
AT.air_mass_basis(::_PlanDriver) = :dry
AT.MetDrivers.flux_interpolation_mode(::_PlanDriver) = :constant
AT.MetDrivers.supports_native_vertical_flux(::_PlanDriver) = true
AT.MetDrivers.uses_binary_substep_contract(d::_PlanDriver) = d.contract
AT.MetDrivers.flux_kind(d::_PlanDriver) = d.flux_kind
AT.supports_convection(::_PlanDriver) = true

const GRID = AT.AtmosGrid(AT.CubedSphereMesh(Nc = NC, Hp = 3, FT = Float64),
                          AT.HybridSigmaPressure([0.0, 100, 300, 600, 1000, 2000],
                                                 [0.0, 0, 0.1, 0.3, 0.7, 1]),
                          AT.CPU(); FT = Float64)
const MESH = GRID.horizontal
const HP = MESH.Hp
const NP = NC + 2HP

function _tm5_forcing(w)
    mk(lo, hi, v) = ntuple(p -> (e = zeros(NC, NC, NZ); e[:, :, lo:min(hi, NZ - 1)] .= v * (1 + 0.1w); e), 6)
    return AT.ConvectionForcing(nothing, nothing,
        (; entu = mk(2, 4, 0.06), detu = mk(2, 4, 0.04), entd = mk(3, 4, 0.02), detd = mk(3, 4, 0.01)))
end

function _cmfmc_forcing(w)
    cmfmc = ntuple(_ -> (c = zeros(NC, NC, NZ + 1);
                         c[:, :, 2] .= 0.012 * (1 + 0.1w); c[:, :, 3] .= 0.020; c[:, :, 4] .= 0.015; c), 6)
    dtrain = ntuple(_ -> (d = zeros(NC, NC, NZ); d[:, :, 2] .= 0.006; d[:, :, 3] .= 0.005; d), 6)
    return AT.ConvectionForcing(cmfmc, dtrain, nothing)
end

function _window(w, conv; vary_ps = false)
    m = ntuple(p -> [S * (2.0 + 0.25k + 0.01p + 0.02w + 0.03sin(0.7i + 0.3j + p + w))
                     for i in 1:NP, j in 1:NP, k in 1:NZ], 6)
    dm = ntuple(p -> [S * 0.002cos(0.4i + 0.2j + 0.3k + p + w) for i in 1:NP, j in 1:NP, k in 1:NZ], 6)
    am = ntuple(p -> (a = zeros(NP + 1, NP, NZ);
        for k in 1:NZ, j in HP+1:HP+NC, i in HP+1:HP+NC+1
            a[i, j, k] = S * 0.05sin(0.2w + 0.3p + 0.7i + 0.4j + 0.2k)
        end; a), 6)
    bm = ntuple(p -> (b = zeros(NP, NP + 1, NZ);
        for k in 1:NZ, j in HP+1:HP+NC+1, i in HP+1:HP+NC
            b[i, j, k] = S * 0.04cos(0.3w + 0.5p + 0.4i + 0.6j + 0.1k)
        end; b), 6)
    cm = ntuple(p -> (c = zeros(NP, NP, NZ + 1);
        for k in 2:NZ, j in HP+1:HP+NC, i in HP+1:HP+NC
            c[i, j, k] = -S * 0.03(1 + 0.3sin(i + j + k + p + w))
        end; c), 6)
    forcing = conv === :tm5 ? _tm5_forcing(w) :
              conv in (:cmfmc, :cmfmc_matrix) ? _cmfmc_forcing(w) : nothing
    ps = vary_ps ? 95_000.0 - 3000.0w : 95_000.0
    return AT.TransportWindow(m, ntuple(_ -> fill(ps, NC, NC), 6),
                              AT.CubedSphereFaceFluxState{AT.DryBasis}(am, bm, cm);
                              deltas = AT.MetDrivers.CubedSphereFluxDeltas(dm), convection = forcing,
                              dkg = ntuple(_ -> fill(S * 1e-5, NC, NC, NZ), 6))
end

_driver(; conv = :none, contract = true, steps = [2, 3, 2], vary_ps = false,
        flux_kind = :substep_mass_amount) =
    (ws = [_window(w, conv; vary_ps) for w in eachindex(steps)];
     _PlanDriver{Float64, typeof(GRID), eltype(ws)}(GRID, ws, WINDOW_DT, steps, contract, flux_kind))

const RATES = ntuple(p -> [S * 1e-6 * (1 + 0.3sin(i + 2j + p)) for i in 1:NC, j in 1:NC], 6)

_convection_op(conv) = conv === :tm5 ? AT.TM5Convection() :
                       conv === :cmfmc ? AT.CMFMCConvection() :
                       conv === :cmfmc_matrix ? AT.CMFMCMatrixConvection() : AT.NoConvection()

# Kz [m² s⁻¹] = kz / k on level k. The hydrostatic layers of the mock grid are
# kilometres thick: `kz = 30` diffuses weakly, `kz = 3e4` strongly.
_diffusion_op(d, kz = 30.0) = d === :none ? AT.NoDiffusion() :
    d === :dkg ? AT.ImplicitVerticalDiffusion(;
        kz_field = AT.PrecomputedCSDkgField(ntuple(_ -> fill(S * 1e-5, NC, NC, NZ), 6))) :
    AT.ImplicitVerticalDiffusion(;
    kz_field = AT.CubedSphereField(ntuple(_ -> AT.PreComputedKzField(repeat(reshape([kz / k for k in 1:NZ], 1, 1, NZ), NC, NC, 1)), 6)),
    surface_flux_coupling = d === :split ? AT.SplitSurfaceFluxCoupling() : AT.DiffusiveSurfaceFluxBoundary())

_initial_rm(air) = ntuple(p -> [air[p][i, j, k] * (0.08 + 0.01sin(0.27i + 0.13j + 0.19k + 0.07p))
                                for i in 1:NP, j in 1:NP, k in 1:NZ], 6)

# A configuration of the matrix.
Base.@kwdef struct Case
    scheme = AT.UpwindScheme()
    conv::Symbol = :none
    diffusion::Symbol = :none
    emissions::Bool = true
    cadence = nothing
    contract::Bool = true
    reset::Symbol = :preserve_tracer_mass
    steps::Vector{Int} = [2, 3, 2]
    kz::Float64 = 30.0
    vary_ps::Bool = false
    flux_kind::Symbol = :substep_mass_amount
end

_label(c::Case) = "$(nameof(typeof(c.scheme))) $(something(c.cadence, :window)) " *
    "contract=$(c.contract) conv=$(c.conv) diffusion=$(c.diffusion) emissions=$(c.emissions) " *
    "reset=$(c.reset)" * (c.vary_ps ? " per-window ps" : "") *
    (c.flux_kind === :substep_mass_amount ? "" : " $(c.flux_kind)")

# The production simulation of a case (not yet run) and its initial state.
function _simulation(c::Case; rate_scale = 1.0, rm_perturbation = nothing)
    driver = _driver(; conv = c.conv, contract = c.contract, steps = c.steps,
                     vary_ps = c.vary_ps, flux_kind = c.flux_kind)
    air = ntuple(p -> copy(driver.windows[1].air_mass[p]), 6)
    rm = _initial_rm(air)
    rm_perturbation === nothing || (rm = ntuple(p -> rm[p] .+ rm_perturbation[p], 6))
    state = AT.CubedSphereState(AT.DryBasis, MESH, air; CO2 = rm)
    model = AT.TransportModel(state, AT.allocate_face_fluxes(MESH, NZ; basis = AT.DryBasis),
                              GRID, c.scheme; diffusion = _diffusion_op(c.diffusion, c.kz),
                              convection = _convection_op(c.conv))
    sources = c.emissions ? (AT.SurfaceFluxSource(:CO2, ntuple(p -> rate_scale .* RATES[p], 6)),) : ()
    return AT.DrivenSimulation(model, driver; surface_sources = sources,
                               physics_cadence = c.cadence, air_mass_reset_mode = c.reset)
end

_rm(sim) = AT.get_tracer(sim.model.state, :CO2)
_interior(panels) = ntuple(p -> panels[p][HP+1:HP+NC, HP+1:HP+NC, :], 6)

# The adjoint inputs of a case, captured from a second, identical production
# simulation (`cs_adjoint_inputs`): the face fluxes of every step as production
# refreshes them, the convection forcing and diffusion layer thickness of each
# met window, the schedule, and fresh workspaces.
function _adjoint_inputs(c::Case; kw...)
    x = AT.cs_adjoint_inputs(_simulation(c; kw...); tracer = :CO2)
    return (; rm0 = x.panels_rm0, m0 = x.panels_m0, am = x.panels_am_steps,
            bm = x.panels_bm_steps, cm = x.panels_cm_steps, kw = x.kwargs)
end

# The keywords of the footprint entry points (base emissions, no `surface_flux`).
_footprint_kwargs(inp) = (; Base.structdiff(inp.kw, NamedTuple{(:surface_flux, :emission_rates)})...,
                          base_emission_rates = inp.kw.emission_rates)

_replay(inp; kw...) = first(ADJ.Footprint._cs_replay_for_call(inp.rm0, inp.m0, inp.am, inp.bm, inp.cm, MESH;
                                                     inp.kw..., kw...))
