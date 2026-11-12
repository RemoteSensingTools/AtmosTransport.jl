module IFS

using Adapt
using KernelAbstractions: @kernel, @index, @Const, get_backend
using ....State: CellState, CubedSphereState
using ....Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh
using ....MetDrivers: ConvectionForcing
using ....Architectures: _ledger_deposit, _ledger_residual, _load_ledger_carry, _store_ledger_carry!,
                        launch!, select_panel
import ...apply!
import ..Convection: AbstractConvection, apply_convection!
using ..Convection: _cmfmc_metric_buffer, _empty_ledger_carry, _fitted_ledger_carry
using ..TM5: _tm5_column_sum

export IFSConvection, IFSWorkspace, AbstractIFSTracerVariant, IFSCuctracer, IFSMonotoneUpstream

include("workspace.jl")
include("IFSConvection.jl")
include("ifs_column.jl")
include("ifs_kernels.jl")

end # module IFS
