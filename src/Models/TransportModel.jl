"""
    TransportModel

Minimal Oceanigans-style model object for standalone `src` transport runs.

Carries advection, chemistry, vertical diffusion, surface emissions,
and convection operators. The composition target from
`OPERATOR_COMPOSITION.md` §3.1 is:

    transport_block(dt)   →   convection_block(dt)   →   chemistry_block(dt)

where `transport_block` runs the full palindrome with diffusion and
emissions at the center:

    X → Y → Z → V(dt/2) → S(dt) → V(dt/2) → Z → Y → X      (emissions active)
    X → Y → Z → V(dt) → Z → Y → X                          (no emissions)

`step!(model, dt)` executes the full runtime composition:
transport block → convection block → chemistry block.

Defaults `chemistry = NoChemistry()`, `diffusion = NoDiffusion()`,
`emissions = NoSurfaceFlux()`, `convection = NoConvection()` keep
pre-refactor behaviour for callers that don't opt in.

# Convection fields

- `convection :: ConvT` — operator type, defaults to `NoConvection()`.
  Concrete subtypes are `CMFMCConvection` and `TM5Convection`.
  `NoConvection` is a compile-time dead branch in `step!`.
- `convection_forcing :: CF` — per-step forcing container. Defaults to
  `ConvectionForcing()` (all-nothing placeholder). `DrivenSimulation`
  construction allocates real buffers via
  `allocate_convection_forcing_like`; `_refresh_forcing!` populates them
  from `sim.window.convection` each substep.

Helpers `with_convection(model, op)` and
`with_convection_forcing(model, forcing)` parallel
`with_chemistry` / `with_diffusion` / `with_emissions`.
"""
struct TransportModelWorkspace{AdvT, ConvT, ChemT, DiffT}
    advection_ws  :: AdvT
    convection_ws :: ConvT
    chemistry_ws  :: ChemT
    diffusion_ws  :: DiffT
end

TransportModelWorkspace(advection_ws; convection_ws = nothing, chemistry_ws = nothing,
                        diffusion_ws = nothing) =
    TransportModelWorkspace(advection_ws, convection_ws, chemistry_ws, diffusion_ws)

function Base.getproperty(workspace::TransportModelWorkspace, name::Symbol)
    if name === :advection_ws || name === :convection_ws || name === :chemistry_ws || name === :diffusion_ws
        return getfield(workspace, name)
    end
    return getproperty(getfield(workspace, :advection_ws), name)
end

function Base.propertynames(workspace::TransportModelWorkspace, private::Bool = false)
    return (:advection_ws, :convection_ws, :chemistry_ws, :diffusion_ws,
            propertynames(getfield(workspace, :advection_ws), private)...)
end

function Adapt.adapt_structure(to, workspace::TransportModelWorkspace)
    advection_ws = Adapt.adapt(to, workspace.advection_ws)
    convection_ws = workspace.convection_ws === nothing ? nothing :
                    Adapt.adapt(to, workspace.convection_ws)
    chemistry_ws = workspace.chemistry_ws === nothing ? nothing :
                   Adapt.adapt(to, workspace.chemistry_ws)
    diffusion_ws = if workspace.diffusion_ws !== nothing &&
                      workspace.advection_ws !== nothing &&
                      workspace.diffusion_ws.w_scratch === workspace.advection_ws.w_scratch &&
                      workspace.diffusion_ws.dz_scratch === workspace.advection_ws.dz_scratch
        ColumnDiffusionWorkspace(advection_ws.w_scratch, advection_ws.dz_scratch)
    else
        Adapt.adapt(to, workspace.diffusion_ws)
    end
    return TransportModelWorkspace(advection_ws; convection_ws, chemistry_ws, diffusion_ws)
end

_convection_workspace_for(::NoConvection, state, grid) = nothing
_chemistry_workspace_for(::NoChemistry, state, grid) = nothing
_chemistry_workspace_for(::AbstractChemistryOperator, state, grid) = nothing
function _chemistry_workspace_for(operator::CompositeChemistry, state, grid)
    chemistry_workspace_plan(operator, state, grid)
    workspaces = map(operator.schemes) do scheme
        _chemistry_workspace_for(scheme, state, grid)
    end
    return CompositeChemistryWorkspace(workspaces)
end

function Adapt.adapt_structure(to, workspace::CompositeChemistryWorkspace)
    return CompositeChemistryWorkspace(Adapt.adapt(to, workspace.workspaces))
end

_cs_advection_workspace_for(::AbstractAdvectionScheme,
                            state::CubedSphereState,
                            grid::AtmosGrid{<:CubedSphereMesh}) =
    CSAdvectionWorkspace(grid.horizontal, state.air_mass[1];
                         n_tracers = ntracers(state))

_cs_advection_workspace_for(::LinRoodPPMScheme,
                            state::CubedSphereState,
                            grid::AtmosGrid{<:CubedSphereMesh}) =
    CSLinRoodAdvectionWorkspace(grid.horizontal, state.air_mass[1]; n_tracers=ntracers(state))

# No advection → no scratch buffers (NoAdvection alone doesn't need any).
# When the runtime stacks diffusion on top of NoAdvection (the
# "diffusion-only" experiment), the model constructor below detects that
# case and allocates only a ColumnDiffusionWorkspace when diffusion is active.
_cs_advection_workspace_for(::NoAdvection,
                            state::CubedSphereState,
                            grid::AtmosGrid{<:CubedSphereMesh}) = nothing

_cmfmc_cell_metrics(mesh::LatLonMesh) = cell_areas_by_latitude(mesh)
_cmfmc_cell_metrics(mesh::ReducedGaussianMesh) = [cell_area(mesh, c) for c in 1:ncells(mesh)]
_cmfmc_cell_metrics(mesh::CubedSphereMesh) = ntuple(_ -> mesh.cell_areas, 6)

# CMFMCConvection — one CMFMCWorkspace per topology with cached
# cell metrics the CFL scan needs.
_convection_workspace_for(::CMFMCConvection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:LatLonMesh}) where {B, A, Raw <: AbstractArray{<:Any, 4}} =
    CMFMCWorkspace(state.air_mass; cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
_convection_workspace_for(::CMFMCConvection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:ReducedGaussianMesh}) where {B, A, Raw <: AbstractArray{<:Any, 3}} =
    CMFMCWorkspace(state.air_mass; cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
_convection_workspace_for(::CMFMCConvection,
                          state::CubedSphereState{B},
                          grid::AtmosGrid{<:CubedSphereMesh}) where {B} =
    CMFMCWorkspace(state.air_mass; cell_metrics = _cmfmc_cell_metrics(grid.horizontal))

# TM5Convection — one TM5Workspace per topology with cached cell
# metrics. TM5's original matrix divides kg/m²/s convective fluxes
# by layer mass per unit area; runtime state stores kg per cell, so
# the kernels need cell areas just like the CMFMC path. The operator
# also carries a `tile_workspace_gib::FT` budget; the `TM5Workspace`
# constructor turns that into a tile
# column count via `derive_tile_columns`, so the same code path
# covers all three topologies.
_convection_workspace_for(op::TM5Convection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:LatLonMesh}) where {B, A, Raw <: AbstractArray{<:Any, 4}} =
    TM5Workspace(state.air_mass;
                 tile_workspace_gib = op.tile_workspace_gib,
                 defer_scratch = op.use_collab_lu,
                 cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
_convection_workspace_for(op::TM5Convection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:ReducedGaussianMesh}) where {B, A, Raw <: AbstractArray{<:Any, 3}} =
    TM5Workspace(state.air_mass;
                 tile_workspace_gib = op.tile_workspace_gib,
                 defer_scratch = op.use_collab_lu,
                 cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
_convection_workspace_for(op::TM5Convection,
                          state::CubedSphereState{B},
                          grid::AtmosGrid{<:CubedSphereMesh}) where {B} =
    TM5Workspace(state.air_mass;
                 tile_workspace_gib = op.tile_workspace_gib,
                 defer_scratch = op.use_collab_lu,
                 cell_metrics = _cmfmc_cell_metrics(grid.horizontal))

# CMFMCMatrixConvection — derives (entu, detu) from GEOS (cmfmc, dtrain) and
# routes through the same TM5 LU machinery. Workspace composes a TM5Workspace
# with rate-cache slabs sized to mirror dtrain (one entry per layer center).
_convection_workspace_for(op::CMFMCMatrixConvection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:LatLonMesh}) where {B, A, Raw <: AbstractArray{<:Any, 4}} =
    CMFMCMatrixWorkspace(state.air_mass;
                         tile_workspace_gib = op.inner.tile_workspace_gib,
                         defer_scratch = op.inner.use_collab_lu,
                         cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
_convection_workspace_for(op::CMFMCMatrixConvection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:ReducedGaussianMesh}) where {B, A, Raw <: AbstractArray{<:Any, 3}} =
    CMFMCMatrixWorkspace(state.air_mass;
                         tile_workspace_gib = op.inner.tile_workspace_gib,
                         defer_scratch = op.inner.use_collab_lu,
                         cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
_convection_workspace_for(op::CMFMCMatrixConvection,
                          state::CubedSphereState{B},
                          grid::AtmosGrid{<:CubedSphereMesh}) where {B} =
    CMFMCMatrixWorkspace(state.air_mass;
                         tile_workspace_gib = op.inner.tile_workspace_gib,
                         defer_scratch = op.inner.use_collab_lu,
                         cell_metrics = _cmfmc_cell_metrics(grid.horizontal),
                         halo_width = grid.horizontal.Hp)

# Fallback for future operators — keep LAST so the specific
# methods above take precedence. Returns `nothing` so installing an
# unknown operator on the model compiles; DrivenSimulation's
# validator catches it at runtime with a clear error.
_convection_workspace_for(::AbstractConvection, state, grid) = nothing

function _with_convection_workspace(workspace, convection_ws)
    if workspace isa TransportModelWorkspace
        return workspace.convection_ws === convection_ws ?
               workspace :
               TransportModelWorkspace(workspace.advection_ws; convection_ws,
                                       chemistry_ws = workspace.chemistry_ws,
                                       diffusion_ws = workspace.diffusion_ws)
    end
    return TransportModelWorkspace(workspace; convection_ws = convection_ws)
end

function _with_chemistry_workspace(workspace, chemistry_ws)
    if workspace isa TransportModelWorkspace
        return workspace.chemistry_ws === chemistry_ws ? workspace :
               TransportModelWorkspace(workspace.advection_ws;
                                       convection_ws = workspace.convection_ws,
                                       chemistry_ws, diffusion_ws = workspace.diffusion_ws)
    end
    return TransportModelWorkspace(workspace; chemistry_ws)
end

function _with_diffusion_workspace(workspace::TransportModelWorkspace, op, state, grid)
    scratch = if op isa NoDiffusion
        nothing
    elseif workspace.diffusion_ws !== nothing
        workspace.diffusion_ws
    elseif workspace.advection_ws !== nothing
        # Reuse legacy column buffers when supplied by an advection workspace;
        # ownership and dispatch are explicit, without allocating duplicate arrays.
        ColumnDiffusionWorkspace(workspace.advection_ws.w_scratch,
                                 workspace.advection_ws.dz_scratch)
    elseif state isa CubedSphereState
        ColumnDiffusionWorkspace(state.air_mass, grid.horizontal.Nc)
    else
        ColumnDiffusionWorkspace(state.air_mass)
    end
    return TransportModelWorkspace(workspace.advection_ws;
        convection_ws=workspace.convection_ws, chemistry_ws=workspace.chemistry_ws,
        diffusion_ws=scratch)
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

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:LatLonMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusion = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvection = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    workspace_model = _with_convection_workspace(
        workspace, _convection_workspace_for(convection, state, grid))
    workspace_model = _with_chemistry_workspace(
        workspace_model, _chemistry_workspace_for(chemistry, state, grid))
    workspace_model = _with_diffusion_workspace(workspace_model, diffusion, state, grid)
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
                        workspace = AdvectionWorkspace(state; mesh=grid.horizontal),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusion = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvection = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    workspace_model = _with_convection_workspace(
        workspace, _convection_workspace_for(convection, state, grid))
    workspace_model = _with_chemistry_workspace(
        workspace_model, _chemistry_workspace_for(chemistry, state, grid))
    workspace_model = _with_diffusion_workspace(workspace_model, diffusion, state, grid)
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
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusion = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvection = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphere transport now uses CubedSphereState + CubedSphereFaceFluxState; CellState + StructuredFaceFluxState remains unsupported for CubedSphereMesh"))
end

function TransportModel(state::CubedSphereState{B},
                        fluxes::CubedSphereFaceFluxState{B},
                        grid::AtmosGrid{<:CubedSphereMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = _cs_advection_workspace_for(advection, state, grid),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusion = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvection = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    workspace_model = _with_convection_workspace(
        workspace, _convection_workspace_for(convection, state, grid))
    workspace_model = _with_chemistry_workspace(
        workspace_model, _chemistry_workspace_for(chemistry, state, grid))
    workspace_model = _with_diffusion_workspace(workspace_model, diffusion, state, grid)
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
    chemistry_ws = _chemistry_workspace_for(chemistry, model.state, model.grid)
    workspace = _with_chemistry_workspace(model.workspace, chemistry_ws)
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(workspace), typeof(chemistry),
                          typeof(model.diffusion), typeof(model.emissions),
                          typeof(model.convection), typeof(model.convection_forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        workspace, chemistry, model.diffusion, model.emissions,
        model.convection, model.convection_forcing)
end

"""
    with_diffusion(model::TransportModel, diffusion)

Return a copy of `model` with its diffusion operator replaced. All other
fields share storage with the original. Parallel to [`with_chemistry`](@ref);
useful for installing a diffusion operator into a model that was
constructed with the default `NoDiffusion()`.
"""
function with_diffusion(model::TransportModel, diffusion::AbstractDiffusion)
    workspace = _with_diffusion_workspace(model.workspace, diffusion, model.state, model.grid)
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(workspace), typeof(model.chemistry),
                          typeof(diffusion), typeof(model.emissions),
                          typeof(model.convection), typeof(model.convection_forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        workspace, model.chemistry, diffusion, model.emissions,
        model.convection, model.convection_forcing)
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
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(model.workspace), typeof(model.chemistry),
                          typeof(model.diffusion), typeof(emissions),
                          typeof(model.convection), typeof(model.convection_forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, model.diffusion, emissions,
        model.convection, model.convection_forcing)
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
        model.workspace, _convection_workspace_for(convection, model.state, model.grid))
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(workspace), typeof(model.chemistry),
                          typeof(model.diffusion), typeof(model.emissions),
                          typeof(convection), typeof(model.convection_forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        workspace, model.chemistry, model.diffusion, model.emissions,
        convection, model.convection_forcing)
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
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(model.workspace), typeof(model.chemistry),
                          typeof(model.diffusion), typeof(model.emissions),
                          typeof(model.convection), typeof(forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, model.diffusion, model.emissions,
        model.convection, forcing)
end

function Adapt.adapt_structure(to, model::TransportModel)
    state              = Adapt.adapt(to, model.state)
    fluxes             = Adapt.adapt(to, model.fluxes)
    workspace          = Adapt.adapt(to, model.workspace)
    diffusion          = Adapt.adapt(to, model.diffusion)
    emissions          = Adapt.adapt(to, model.emissions)
    convection_forcing = Adapt.adapt(to, model.convection_forcing)
    return TransportModel{typeof(state), typeof(fluxes), typeof(model.grid),
                          typeof(model.advection), typeof(workspace),
                          typeof(model.chemistry), typeof(diffusion),
                          typeof(emissions), typeof(model.convection),
                          typeof(convection_forcing)}(
        state, fluxes, model.grid, model.advection, workspace,
        model.chemistry, diffusion, emissions,
        model.convection, convection_forcing)
end

"""
    transport_step!(model::TransportModel, dt; meteo = nothing)

Advance `model.state` by one transport step: advection with
vertical diffusion at the palindrome center, surface emissions
wrapped by the two V half-steps when active.

Binary-scheduled driven runs use this block at the per-window
advection substep cadence stored in the transport binary. Convection
and chemistry are separate physics blocks and can run at the met-window
cadence via [`convection_chemistry_step!`](@ref).
"""
function transport_step!(model::TransportModel, dt; meteo = nothing)
    SectionTimer.@section :advection apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace.advection_ws,
           diffusion_op = model.diffusion,
           diffusion_workspace = model.workspace.diffusion_ws,
           emissions_op = model.emissions,
           meteo = meteo)
    return nothing
end

"""
    convection_chemistry_step!(model::TransportModel, dt; meteo = nothing)

Advance the non-transport physics blocks once: convection block →
chemistry block.
"""
function convection_chemistry_step!(model::TransportModel, dt; meteo = nothing)
    _convection_block!(model.convection, model, dt)
    SectionTimer.@section :chemistry chemistry_block!(
        model.state, meteo, model.grid, model.chemistry, dt;
        workspace = model.workspace.chemistry_ws)
    return nothing
end

@inline _convection_block!(::NoConvection, model::TransportModel, dt) = nothing

function _convection_block!(op::AbstractConvection, model::TransportModel, dt)
    SectionTimer.@section :convection apply!(model.state, model.convection_forcing,
                                             model.grid, op, dt;
                                             workspace = model.workspace.convection_ws)
    return nothing
end

"""
    step!(model::TransportModel, dt; meteo = nothing)

Advance `model.state` by one full runtime step: transport block
(advection with vertical diffusion at the palindrome center, surface
emissions wrapped by the two V half-steps when active) → convection
block → chemistry block.

With defaults `diffusion = NoDiffusion()`, `emissions = NoSurfaceFlux()`,
`chemistry = NoChemistry()`, `convection = NoConvection()`, every live
component is a dead branch and the call is bit-exact equivalent to
pre-refactor advection.

`meteo` is optional and defaults to `nothing`; pass a real
meteorology object (`AbstractMetDriver`) or a `DrivenSimulation`
to thread `current_time(meteo)` through operators
that consume time-varying fields.
"""
function step!(model::TransportModel, dt; meteo = nothing)
    transport_step!(model, dt; meteo = meteo)
    convection_chemistry_step!(model, dt; meteo = meteo)
    return nothing
end

export TransportModel, step!, transport_step!, convection_chemistry_step!
export with_chemistry, with_diffusion, with_emissions
export with_convection, with_convection_forcing
