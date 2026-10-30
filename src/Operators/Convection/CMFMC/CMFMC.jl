module CMFMC

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
using ..TM5: TM5Convection, TM5Workspace, _tm5_column_sum, _tm5_restore_column_mass!, _tm5_template

export AbstractCloudBase, ArchivedCloudBase, CMFMCConvection, CMFMCEdgeCloudBase, CMFMCMatrixConvection, CMFMCMatrixWorkspace, CMFMCWorkspace, invalidate_cmfmc_cache!, invalidate_cmfmc_matrix_cache!

include("workspace.jl")
include("cmfmc_kernels.jl")
include("CMFMCConvection.jl")
include("cmfmc_matrix_kernels.jl")
include("CMFMCMatrixConvection.jl")
include("adjoint_kernels.jl")

end # module CMFMC
