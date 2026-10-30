module TM5

using Adapt
import KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize,
                          @uniform, get_backend, synchronize
using ....State: CellState, CubedSphereState
using ....Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, cell_areas_by_latitude
using ....MetDrivers: ConvectionForcing
using ....Architectures: _kahan_add, _neumaier_sum, _ledger_residual, _ledger_deposit,
                        _load_ledger_carry, _store_ledger_carry!, launch!, select_panel
import ...apply!
import ..Convection: AbstractConvection, apply_convection!
using ..Convection: _cmfmc_metric_buffer, _empty_ledger_carry, _fitted_ledger_carry

export TM5Convection, TM5Workspace, derive_tile_columns

include("workspace.jl")
include("tm5_column_solve.jl")
include("tm5_kernels.jl")
include("TM5Convection.jl")
include("adjoint_kernels.jl")

end # module TM5
