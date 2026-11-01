module MassClosure

export balance_cs_level!, cs_level_scratches, cs_column_convergence

using LinearAlgebra: mul!, dot, cholesky, Symmetric
using SparseArrays: SparseMatrixCSC, sparse
using ...Parameters: DEFAULT_POISSON_BALANCE_TOL, DEFAULT_POISSON_BALANCE_MAX_ITER,
    DEFAULT_POISSON_PROJECT_EVERY, HYBRID_LAYER_MIN_DB, COLUMN_CLOSURE_TOL_KG
using ...Grids: cell_centers, unit3, arc3, slerp3, unit_xyz_from_radians, seam_mirror_sign, dot3,
    cross3, CS_PANEL_COUNT, CSGlobalFaceTable, cs_edge_face_location, cs_flat_index, cs_panel_ij
using ...Grids: LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, cs_face_edge_lengths,
    cs_corner_xyz, PanelConnectivity, ncells, nfaces, cell_area, panel_cell_center_lonlat,
    panel_cell_local_tangent_basis, default_panel_connectivity, reciprocal_edge, EDGE_NORTH,
    EDGE_SOUTH, EDGE_EAST, EDGE_WEST
using ..Common: AbstractColumnWeights, MassWeightedColumn, ERA5SpectralSettings,
    validate_column_balance_weights
import ..Common: check_column_weight_levels, column_weights, spread_column_correction!,
    target_balance_geometry

target_balance_geometry(ft::CSGlobalFaceTable) =
    (cs_cell_face_degree(ft), CSPoissonScratch(ft.nc))

export AbstractWindRegrid,
    AlongFaceFilter,
    COLUMN_WEIGHT_KINDS,
    CSPoissonScratch,
    CSVectorFaceGeometry,
    CartesianWindRegrid,
    CompressedLaplacian,
    HybridBWeightedColumn,
    HybridMassWeightedColumn,
    LineIntegralFaceFluxes,
    OuterCellPair,
    ScalarWindRegrid,
    apply_cs_flux_correction!,
    balance_compressed_column_fluxes!,
    balance_cs_column_mass_fluxes!,
    balance_cs_global_mass_fluxes!,
    build_compressed_laplacian,
    column_weights,
    compute_cs_cm_pressure_fixer!,
    cs_cell_face_degree,
    cs_face_fluxes!,
    cs_graph_laplacian_mul!,
    cs_vector_face_fluxes!,
    diagnose_balanced_cs_cm!,
    diagnose_cs_cm!,
    fill_cs_layer_thickness!,
    geos_native_to_face_flux!,
    line_integral_face_fluxes!,
    point_weights,
    project_mean_zero!,
    recheck_cs_column_closure,
    reconstruct_cs_fluxes!,
    recover_ll_cell_center_winds!,
    regrid_winds!,
    rotate_winds_to_panel_local!,
    solve_compressed_poisson_pcg!,
    solve_cs_poisson_pcg!,
    spread_column_correction!,
    sync_all_cs_boundary_mirrors!,
    sync_cs_mirrors!

include("ring_closure.jl")
include("cs_poisson_solver.jl")
include("column_weights.jl")
include("cs_poisson_balance.jl")
include("ring_poisson_balance.jl")
include("cs_flux_reconstruction.jl")
include("cs_wind_rotation.jl")
include("cs_native_fluxes.jl")
include("face_line_integrals.jl")
include("flux_construction.jl")
include("tm5_target_balance.jl")

end
