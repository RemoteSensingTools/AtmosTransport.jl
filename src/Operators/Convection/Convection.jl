"""
    Convection

Convective transport operators.

This submodule ships the convection-operator hierarchy:

- [`AbstractConvection`](@ref) — supertype for all convective
  transport operators.
- [`NoConvection`](@ref) — identity no-op; default for configurations
  without active convection. Dispatch is a compile-time dead branch
  in `TransportModel.step!` so the presence of the convection block
  has zero cost for users who don't opt in.

Operators:

- `ConvectionForcing` in `..MetDrivers` carries the convective forcing
  arrays (with `copy_convection_forcing!`,
  `allocate_convection_forcing_like`, and window-struct integration).
- `CMFMCConvection` (GCHP path, CMFMC+DTRAIN kernel with mandatory CFL
  sub-cycling and well-mixed sub-cloud).
- `TM5Convection` (TM5 column solver) with `TM5Workspace`, running on
  all three topologies (LL / RG / CS).

The `step!`-level runtime block wires convection across all three
topologies (LL / RG / CS).

## `apply!` contract

    apply!(state::CellState{B},
           forcing::ConvectionForcing,
           grid::AtmosGrid,
           op::AbstractConvection,
           dt::Real;
           workspace) where {B <: AbstractMassBasis}

The operator takes `ConvectionForcing` directly (not a transport
window or driver). `_refresh_forcing!` populates
`model.convection_forcing` at the first substep of each window by copying
from `sim.window.convection`. `TransportModel.step!` executes the convection
block between transport and chemistry. No `meteo` kwarg — the forcing
arrays are the time information; the operator does not call
`current_time`.

## Face-indexed scope

`CMFMCConvection` now supports structured LatLon, face-indexed
ReducedGaussian, and panel-native CubedSphere state layouts. The CS
path keeps forcing panel-native too: the driver loads `cmfmc` /
`dtrain` as per-panel tuples and the operator applies the same
column-local logic on each panel interior.

`TM5Convection` ships the same three-topology scope.

The no-op `NoConvection` path accepts any state shape — it's a pure
dead branch.
"""
module Convection

using ...State: CellState, CubedSphereState
using ...Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, cell_areas_by_latitude
using ...MetDrivers: ConvectionForcing
import ..apply!
import ..AbstractConvection             # global root from src/Operators/AbstractOperators.jl

export AbstractConvection, NoConvection
export CMFMCConvection, AbstractCloudBase, CMFMCEdgeCloudBase, ArchivedCloudBase
export CMFMCWorkspace, invalidate_cmfmc_cache!
export TM5Convection
export TM5Workspace
export CMFMCMatrixConvection                    # GEOS-derived rates → TM5 LU (conservative CMFMC)
export CMFMCMatrixWorkspace, invalidate_cmfmc_matrix_cache!
export apply_convection!

function apply_convection! end
include("operators.jl")
include("workspace_shared.jl")
include("TM5/TM5.jl")
using .TM5: TM5Convection, TM5Workspace, derive_tile_columns
include("CMFMC/CMFMC.jl")
using .CMFMC: CMFMCConvection, AbstractCloudBase, CMFMCEdgeCloudBase, ArchivedCloudBase,
              CMFMCWorkspace, invalidate_cmfmc_cache!, CMFMCMatrixConvection,
              CMFMCMatrixWorkspace, invalidate_cmfmc_matrix_cache!
using ...Grids: cell_area, ncells
import ..operator_state, ..restore_operator_state!, ..reset_operator_state!
using ..Operators: _copy_state, _zero_state!
include("workspace_factory.jl")
include("workspace_state.jl")
export convection_workspace_for, derive_tile_columns
end # module Convection
