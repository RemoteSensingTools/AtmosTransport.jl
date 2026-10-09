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
`model.convection_forcing` each substep by copying from
`sim.window.convection`. `TransportModel.step!` executes the convection
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

using Adapt
import KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize,
                          @uniform, get_backend, synchronize
using ...State: CellState, CubedSphereState
using ...Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, cell_areas_by_latitude
using ...MetDrivers: ConvectionForcing
using ...Architectures: _kahan_add, _neumaier_sum, _ledger_residual
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

include("operators.jl")
include("convection_workspace.jl")   # CMFMCWorkspace + TM5Workspace
include("cmfmc_kernels.jl")          # kernels + inline helpers
include("CMFMCConvection.jl")        # struct + apply! methods
include("tm5_column_solve.jl")       # backend-agnostic column solver
include("tm5_kernels.jl")            # @kernel wrappers per topology
include("TM5Convection.jl")          # struct + apply! methods
include("cmfmc_matrix_kernels.jl")   # GEOS (cmfmc,dtrain) → TM5 (entu,detu) derivation kernels
include("CMFMCMatrixConvection.jl")  # struct + apply! routing through TM5 LU

end # module Convection
