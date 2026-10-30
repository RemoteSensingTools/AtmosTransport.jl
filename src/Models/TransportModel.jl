const _TRANSPORT_MODEL_OVERVIEW = """
    TransportModel(state, fluxes, grid, advection; kwargs...)

Minimal Oceanigans-style model object for standalone `src` transport runs.

Carries advection, chemistry, vertical diffusion, surface emissions,
and convection operators. Constant-rate exponential decay is Strang-split
around the transport block's emissions:
    decay(dt/2) → transport_block(dt) → convection_block(dt) → decay(dt/2)

Other chemistry operators, including exponential decay with a time-varying
rate, retain the unsplit order:
    transport_block(dt) → convection_block(dt) → chemistry_block(dt)

where `transport_block` runs the full palindrome with diffusion and
emissions at the center:

    X → Y → Z → V(dt/2) → S(dt) → V(dt/2) → Z → Y → X      (emissions active)
    X → Y → Z → S(dt) → V(dt) → Z → Y → X                  (diffusive boundary coupling)
    X → Y → Z → V(dt) → Z → Y → X                          (no emissions)

`step!(model, dt)` executes the full runtime composition. Constant-rate
exponential decay runs as two exact half-steps around transport/emissions;
other chemistry, including time-varying-rate decay, runs once after transport
and convection.

Defaults `chemistry = NoChemistry()`, `diffusion = NoDiffusion()`,
`emissions = NoSurfaceFlux()`, and `convection = NoConvection()` make the
inactive operator slots compile to no-op dispatches.

# Convection fields

- `convection :: ConvT` — operator type, defaults to `NoConvection()`.
  Concrete subtypes are `CMFMCConvection` and `TM5Convection`.
  `NoConvection` is a compile-time dead branch in `step!`.
- `convection_forcing :: CF` — per-step forcing container. Defaults to
  `ConvectionForcing()` (all-nothing placeholder). `DrivenSimulation`
  construction allocates real buffers via
  `allocate_convection_forcing_like`; `_refresh_forcing!` populates them
  from `sim.window.convection` at the first substep of each window.

Helpers `with_convection(model, op)` and
`with_convection_forcing(model, forcing)` parallel
`with_chemistry` / `with_diffusion` / `with_emissions`.
"""
"""
    TransportModelWorkspace(advection_ws, diffusion_ws; convection_ws=nothing)

Independent preallocated storage for advection, diffusion, and convection.
Inactive operators use nothing; Adapt.adapt moves populated workspaces to the
requested array backend.
"""
struct TransportModelWorkspace{AdvT, DiffT, ConvT}
    advection_ws  :: AdvT
    diffusion_ws  :: DiffT
    convection_ws :: ConvT
end

TransportModelWorkspace(advection_ws, diffusion_ws; convection_ws = nothing) =
    TransportModelWorkspace{typeof(advection_ws), typeof(diffusion_ws),
                            typeof(convection_ws)}(
        advection_ws, diffusion_ws, convection_ws)

function Adapt.adapt_structure(to, workspace::TransportModelWorkspace)
    advection_ws = Adapt.adapt(to, workspace.advection_ws)
    diffusion_ws = workspace.diffusion_ws === nothing ? nothing :
                   Adapt.adapt(to, workspace.diffusion_ws)
    convection_ws = workspace.convection_ws === nothing ? nothing :
                    Adapt.adapt(to, workspace.convection_ws)
    return TransportModelWorkspace(advection_ws, diffusion_ws;
                                   convection_ws = convection_ws)
end

_diffusion_workspace_for(::NoDiffusion, state) = nothing
_diffusion_workspace_for(::AbstractDiffusion, state) = DiffusionWorkspace(state)


# The cubed-sphere advection workspace of a scheme (`_cs_advection_workspace_for`)
# is defined with the step plan (`Operators.StepPlans`), which the adjoint replay
# shares.

# The check a workspace carries (`NoBinaryCFLCheck()` for workspaces without one).
_workspace_binary_cfl_check(ws::CSAdvectionWorkspace) = ws.binary_cfl_check
_workspace_binary_cfl_check(ws) = NoBinaryCFLCheck()

function _with_convection_workspace(workspace::TransportModelWorkspace,
                                    convection_ws)
    return workspace.convection_ws === convection_ws ? workspace :
           TransportModelWorkspace(workspace.advection_ws,
                                   workspace.diffusion_ws;
                                   convection_ws)
end


struct TransportModel{StateT, FluxT, GridT, SchemeT, WorkspaceT,
                       ChemT, DiffT, EmT, ConvT, CF}
    state              :: StateT
    fluxes             :: FluxT
    grid               :: GridT
    advection          :: SchemeT
    workspace          :: WorkspaceT
    chemistry          :: ChemT
    diffusion          :: DiffT
    emissions          :: EmT
    convection         :: ConvT     # default NoConvection()
    convection_forcing :: CF        # default ConvectionForcing() placeholder
end

@doc _TRANSPORT_MODEL_OVERVIEW TransportModel

function _rebuild_model(model::TransportModel;
                        state = model.state,
                        fluxes = model.fluxes,
                        grid = model.grid,
                        advection = model.advection,
                        workspace = model.workspace,
                        chemistry = model.chemistry,
                        diffusion = model.diffusion,
                        emissions = model.emissions,
                        convection = model.convection,
                        convection_forcing = model.convection_forcing)
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace), typeof(chemistry),
                          typeof(diffusion), typeof(emissions), typeof(convection),
                          typeof(convection_forcing)}(
        state, fluxes, grid, advection, workspace, chemistry, diffusion,
        emissions, convection, convection_forcing)
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:LatLonMesh},
                        advection::AbstractAdvectionScheme;
                        advection_workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusion = NoDiffusion(),
                        diffusion_workspace = _diffusion_workspace_for(diffusion, state),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvection = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    workspace_model = TransportModelWorkspace(
        advection_workspace, diffusion_workspace;
        convection_ws = convection_workspace_for(convection, state, grid))
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace_model),
                          typeof(chemistry), typeof(diffusion), typeof(emissions),
                          typeof(convection), typeof(convection_forcing)}(
        state, fluxes, grid, advection, workspace_model,
        chemistry, diffusion, emissions, convection, convection_forcing)
end

function TransportModel(state::CellState{B},
                        fluxes::FaceIndexedFluxState{B},
                        grid::AtmosGrid,
                        advection::AbstractAdvectionScheme;
                        advection_workspace = AdvectionWorkspace(state; mesh=grid.horizontal),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusion = NoDiffusion(),
                        diffusion_workspace = _diffusion_workspace_for(diffusion, state),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvection = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    workspace_model = TransportModelWorkspace(
        advection_workspace, diffusion_workspace;
        convection_ws = convection_workspace_for(convection, state, grid))
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace_model),
                          typeof(chemistry), typeof(diffusion), typeof(emissions),
                          typeof(convection), typeof(convection_forcing)}(
        state, fluxes, grid, advection, workspace_model,
        chemistry, diffusion, emissions, convection, convection_forcing)
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:CubedSphereMesh},
                        advection::AbstractAdvectionScheme;
                        advection_workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusion = NoDiffusion(),
                        diffusion_workspace = _diffusion_workspace_for(diffusion, state),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvection = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphere transport now uses CubedSphereState + CubedSphereFaceFluxState; CellState + StructuredFaceFluxState remains unsupported for CubedSphereMesh"))
end

function TransportModel(state::CubedSphereState{B},
                        fluxes::CubedSphereFaceFluxState{B},
                        grid::AtmosGrid{<:CubedSphereMesh},
                        advection::AbstractAdvectionScheme;
                        binary_cfl_check::AbstractBinaryCFLCheck = NoBinaryCFLCheck(),
                        advection_workspace = _cs_advection_workspace_for(advection, state, grid;
                                                                          binary_cfl_check),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusion = NoDiffusion(),
                        diffusion_workspace = _diffusion_workspace_for(diffusion, state),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvection = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    binary_cfl_check isa NoBinaryCFLCheck ||
        _workspace_binary_cfl_check(advection_workspace) === binary_cfl_check ||
        throw(ArgumentError("binary_cfl_check = $(binary_cfl_check), but the supplied " *
                            "advection_workspace was built without it"))
    workspace_model = TransportModelWorkspace(
        advection_workspace, diffusion_workspace;
        convection_ws = convection_workspace_for(convection, state, grid))
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace_model),
                          typeof(chemistry), typeof(diffusion), typeof(emissions),
                          typeof(convection), typeof(convection_forcing)}(
        state, fluxes, grid, advection, workspace_model,
        chemistry, diffusion, emissions, convection, convection_forcing)
end

"""
    with_chemistry(model::TransportModel, chemistry)

Return a copy of `model` with its chemistry operator replaced. All other
fields share storage with the original. Chemistry is installed into the
model rather than held at the sim level, so this helper is primarily
useful for tests that want to swap chemistry on a constructed model.
"""
function with_chemistry(model::TransportModel, chemistry::AbstractChemistryOperator)
    return _rebuild_model(model; chemistry)
end

"""
    with_diffusion(model::TransportModel, diffusion)

Return a copy of `model` with its diffusion operator replaced. All other
fields share storage with the original. Parallel to [`with_chemistry`](@ref);
useful for installing a diffusion operator into a model that was
constructed with the default `NoDiffusion()`.
"""
function with_diffusion(model::TransportModel, diffusion::AbstractDiffusion)
    diffusion_ws = _diffusion_workspace_for(diffusion, model.state)
    workspace = TransportModelWorkspace(
        model.workspace.advection_ws, diffusion_ws;
        convection_ws = model.workspace.convection_ws)
    return _rebuild_model(model; workspace, diffusion)
end

"""
    with_emissions(model::TransportModel, emissions)

Return a copy of `model` with its surface-emissions operator replaced.
All other fields share storage with the original. Parallel to
[`with_chemistry`](@ref) and [`with_diffusion`](@ref); used by
`DrivenSimulation` to install the sim-level
`surface_sources` tuple as a `SurfaceFluxOperator` inside the wrapped
model, so the palindrome's S slot runs at the right place in the
transport block without sim-level post-step hacks.
"""
function with_emissions(model::TransportModel, emissions::AbstractSurfaceFluxOperator)
    return _rebuild_model(model; emissions)
end

"""
    with_convection(model::TransportModel, convection)

Return a copy of `model` with its convection operator replaced.
All other fields — including
`convection_forcing` — share storage with the original.

Note: `with_convection` does NOT allocate convection-forcing
buffers. The model-side `ConvectionForcing()` placeholder stays
as-is. `DrivenSimulation` construction is responsible for allocating
real buffers via `allocate_convection_forcing_like` after the first
window loads. For tests that bypass the sim
layer, use `with_convection_forcing(model, forcing)` to inject
allocated buffers directly. The model workspace is re-wrapped as
needed so concrete operators can carry their own scratch storage
without disturbing the advection workspace.
"""
function with_convection(model::TransportModel, convection::AbstractConvection)
    workspace = _with_convection_workspace(
        model.workspace, convection_workspace_for(convection, model.state, model.grid))
    return _rebuild_model(model; workspace, convection)
end

"""
    with_convection_forcing(model::TransportModel, forcing::ConvectionForcing)

Return a copy of `model` with its per-step convection-forcing
container replaced. All other fields — including
the `convection` operator — share storage with the original.

Used by `DrivenSimulation` construction to install the allocated
forcing buffers after the first window loads.
Also useful for tests that inject forcing directly without going
through the sim's `_refresh_forcing!` path.
"""
function with_convection_forcing(model::TransportModel, forcing::ConvectionForcing)
    return _rebuild_model(model; convection_forcing = forcing)
end

function Adapt.adapt_structure(to, model::TransportModel)
    state              = Adapt.adapt(to, model.state)
    fluxes             = Adapt.adapt(to, model.fluxes)
    workspace          = Adapt.adapt(to, model.workspace)
    diffusion          = Adapt.adapt(to, model.diffusion)
    emissions          = Adapt.adapt(to, model.emissions)
    convection_forcing = Adapt.adapt(to, model.convection_forcing)
    return _rebuild_model(model; state, fluxes, workspace, diffusion,
                          emissions, convection_forcing)
end

"""
    step_operators(model::TransportModel) -> StepOperators

The state, fluxes and operators of `model` in the form the stages of a
[`StepPlan`](@ref) act on.
"""
step_operators(model::TransportModel) =
    StepOperators(model.state, model.fluxes, model.grid, model.advection,
                  model.workspace.advection_ws, model.diffusion,
                  model.workspace.diffusion_ws, model.emissions, model.convection,
                  model.workspace.convection_ws, model.convection_forcing,
                  model.chemistry)

# The plan `TransportModel.step!` executes: the full operator composition at
# every step (no met windows).
const _MODEL_STEP_PLAN = StepPlan(; physics_cadence = :substep)

"""
    step!(model::TransportModel, dt; meteo = nothing)

Advance `model.state` by one full runtime step: the `step` stages of the
substep [`StepPlan`](@ref). Exponential decay uses the symmetric composition
decay(dt/2) → transport/emissions(dt) → convection(dt) → decay(dt/2). Other
chemistry operators keep the transport/emissions(dt) → convection(dt) →
chemistry(dt) order.

With defaults `diffusion = NoDiffusion()`, `emissions = NoSurfaceFlux()`,
`chemistry = NoChemistry()`, `convection = NoConvection()`, every live
component is a dead branch and the call is bit-exact equivalent to
the advection-only path.

`meteo` is optional and defaults to `nothing`; pass a real
meteorology object (`AbstractMetDriver`) or a `DrivenSimulation`
to thread `current_time(meteo)` through operators
that consume time-varying fields.
"""
function step!(model::TransportModel, dt; meteo = nothing)
    execute_step_stages!(_MODEL_STEP_PLAN, step_operators(model), StepTiming(dt), meteo)
    return nothing
end

export TransportModel, step!
export with_chemistry, with_diffusion, with_emissions
export with_convection, with_convection_forcing
