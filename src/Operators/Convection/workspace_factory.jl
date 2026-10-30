# Runtime convection workspace factories, including topology metrics.
convection_workspace_for(::NoConvection, state, grid) = nothing

_cmfmc_cell_metrics(mesh::LatLonMesh) = cell_areas_by_latitude(mesh)
_cmfmc_cell_metrics(mesh::ReducedGaussianMesh) = [cell_area(mesh, c) for c in 1:ncells(mesh)]
_cmfmc_cell_metrics(mesh::CubedSphereMesh) = ntuple(_ -> mesh.cell_areas, 6)

# CMFMCConvection — one CMFMCWorkspace per topology with cached
# cell metrics the CFL scan needs.
convection_workspace_for(::CMFMCConvection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:LatLonMesh}) where {B, A, Raw <: AbstractArray{<:Any, 4}} =
    CMFMCWorkspace(state.air_mass; cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
convection_workspace_for(::CMFMCConvection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:ReducedGaussianMesh}) where {B, A, Raw <: AbstractArray{<:Any, 3}} =
    CMFMCWorkspace(state.air_mass; cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
convection_workspace_for(::CMFMCConvection,
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
convection_workspace_for(op::TM5Convection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:LatLonMesh}) where {B, A, Raw <: AbstractArray{<:Any, 4}} =
    TM5Workspace(state.air_mass;
                 tile_workspace_gib = op.tile_workspace_gib,
                 defer_scratch = op.use_collab_lu,
                 cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
convection_workspace_for(op::TM5Convection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:ReducedGaussianMesh}) where {B, A, Raw <: AbstractArray{<:Any, 3}} =
    TM5Workspace(state.air_mass;
                 tile_workspace_gib = op.tile_workspace_gib,
                 defer_scratch = op.use_collab_lu,
                 cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
convection_workspace_for(op::TM5Convection,
                          state::CubedSphereState{B},
                          grid::AtmosGrid{<:CubedSphereMesh}) where {B} =
    TM5Workspace(state.air_mass;
                 tile_workspace_gib = op.tile_workspace_gib,
                 defer_scratch = op.use_collab_lu,
                 cell_metrics = _cmfmc_cell_metrics(grid.horizontal))

# CMFMCMatrixConvection — derives (entu, detu) from GEOS (cmfmc, dtrain) and
# routes through the same TM5 LU machinery. Workspace composes a TM5Workspace
# with rate-cache slabs sized to mirror dtrain (one entry per layer center).
convection_workspace_for(op::CMFMCMatrixConvection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:LatLonMesh}) where {B, A, Raw <: AbstractArray{<:Any, 4}} =
    CMFMCMatrixWorkspace(state.air_mass;
                         tile_workspace_gib = op.inner.tile_workspace_gib,
                         defer_scratch = op.inner.use_collab_lu,
                         cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
convection_workspace_for(op::CMFMCMatrixConvection,
                          state::CellState{B, A, Raw},
                          grid::AtmosGrid{<:ReducedGaussianMesh}) where {B, A, Raw <: AbstractArray{<:Any, 3}} =
    CMFMCMatrixWorkspace(state.air_mass;
                         tile_workspace_gib = op.inner.tile_workspace_gib,
                         defer_scratch = op.inner.use_collab_lu,
                         cell_metrics = _cmfmc_cell_metrics(grid.horizontal))
convection_workspace_for(op::CMFMCMatrixConvection,
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
convection_workspace_for(::AbstractConvection, state, grid) = nothing
