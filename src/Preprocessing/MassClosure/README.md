# MassClosure

Horizontal flux construction, wind rotation, Poisson balance and vertical mass-flux diagnosis.

## File map

- [`MassClosure.jl`](MassClosure.jl) — module imports, public API and include order.
- [`column_weights.jl`](column_weights.jl) — hybrid and mass weights; Core interface implementations.
- [`cs_flux_reconstruction.jl`](cs_flux_reconstruction.jl) — panel-average and vector face construction.
- [`cs_native_fluxes.jl`](cs_native_fluxes.jl) — GEOS native flux staggering and pressure-fixer closure.
- [`cs_poisson_balance.jl`](cs_poisson_balance.jl) — layer/column balance and explicit-endpoint cm diagnosis.
- [`cs_poisson_solver.jl`](cs_poisson_solver.jl) — CS graph Laplacian, PCG, correction and seam mirrors.
- [`cs_wind_rotation.jl`](cs_wind_rotation.jl) — geographic to panel winds and LL flux wind recovery.
- [`face_line_integrals.jl`](face_line_integrals.jl) — source-grid midpoint quadrature and shared thickness adapters.
- [`flux_construction.jl`](flux_construction.jl) — typed face methods, validation and wind regridding.
- [`ring_closure.jl`](ring_closure.jl) — shared mean-zero projection.
- [`ring_poisson_balance.jl`](ring_poisson_balance.jl) — compressed RG graph Laplacian and column balance.

## Depends on

Core supplies target geometry and the column-weight protocols. Grids owns face
indexing and geometry; Parameters supplies constants.
This stage follows Core → Vertical → Spectral in pipeline order and imports no
parent facade, target, TransportBinary, source or PhysicsInputs implementation.

## Used by

The Preprocessing facade preserves script, Models and test import sites.
Deferred target and source workflows call these stage APIs.

## Shared helpers owned here

- `AbstractFaceLengths`
- `AbstractWindRegrid`
- `AlongFaceFilter`
- `COLUMN_WEIGHT_KINDS`
- `CSPoissonScratch`
- `CSVectorFaceGeometry`
- `CartesianWindRegrid`
- `CellCenterlineLengths`
- `ColumnResidualWeight`
- `CompressedLaplacian`
- `DryMassFluxThickness`
- `EdgeLengths`
- `FACE_FLUX_KINDS`
- `FACE_INTERPOLATION_KINDS`
- `FACE_LENGTH_KINDS`
- `FLUX_THICKNESS_KINDS`
- `HybridBWeightedColumn`
- `HybridMassWeightedColumn`
- `LineIntegralFaceFluxes`
- `MoistFluxThickness`
- `OuterCellPair`
- `PanelAverageFluxes`
- `ScalarWindRegrid`
- `VectorFaceFluxes`
- `WIND_REGRID_KINDS`
- `apply_cs_flux_correction!`
- `balance_compressed_column_fluxes!`
- `balance_cs_column_mass_fluxes!`
- `balance_cs_global_mass_fluxes!`
- `build_compressed_laplacian`
- `check_column_weight_levels`
- `column_weights`
- `compute_cs_cm_pressure_fixer!`
- `cs_cell_bases`
- `cs_cell_face_degree`
- `cs_column_balance_projected_linf`
- `cs_column_roundoff_tolerance!`
- `cs_face_corners`
- `cs_face_divergence!`
- `cs_face_fluxes!`
- `cs_graph_laplacian_mul!`
- `cs_linf`
- `cs_vector_face_fluxes!`
- `diagnose_balanced_cs_cm!`
- `diagnose_cs_cm!`
- `east_north_to_cartesian`
- `face_flux_method`
- `face_fluxes!`
- `fill_cs_layer_thickness!`
- `flux_header_face_interpolation`
- `flux_header_wind_regrid`
- `flux_thickness_method`
- `geos_native_to_face_flux!`
- `has_source_line_integrals`
- `line_integral_face_fluxes!`
- `point_weights`
- `prepare_cell_winds!`
- `project_mean_zero!`
- `recheck_cs_column_closure`
- `reconstruct_cs_fluxes!`
- `recover_ll_cell_center_winds!`
- `regrid_winds!`
- `require_cs_column_balance`
- `rotate_winds_to_panel_local!`
- `solve_compressed_poisson_pcg!`
- `solve_cs_poisson_pcg!`
- `source_cell_centers`
- `spread_column_correction!`
- `sync_all_cs_boundary_mirrors!`
- `sync_cs_mirrors!`
- `validate_flux_construction`
- `validate_flux_thickness`
- `validate_flux_time_sampling`
- `vector_face_value`
- `wind_basis_trig`
- `build_wind_regrid`
- `xface_length`
- `yface_length`

GEOS also uses `balance_cs_level!`, `cs_level_scratches` and
`cs_column_convergence`; canonical face order and FT column sums are unchanged.

## Forward<->adjoint mirrors

None. These are offline CPU preprocessing routines.

## Numerical contract and future spectral construction

Flux construction (`face_flux_method`, `face_fluxes!`, `line_integral_face_fluxes!`),
balance (`balance_cs_column_mass_fluxes!`, `balance_cs_global_mass_fluxes!`,
`balance_compressed_column_fluxes!`, `solve_cs_poisson_pcg!`,
`solve_compressed_poisson_pcg!`) and cm diagnosis (`diagnose_cs_cm!`,
`diagnose_balanced_cs_cm!`) remain
separately callable. The later TM5 spectral strategy can share the graph/face
geometry, wind rotation, Laplacian/PCG primitives, correction mapping and seam
synchronization here. Its direct spectral quadrature is separate from the
existing source-grid line-integral method.

The dry pressure mass fix stays separately callable in Spectral through
`pin_global_mean_ps!`, `pin_global_mean_ps_using_qv!` and
`recompute_native_mass_fields!`; LL balance remains there through
`LLPoissonWorkspace`, `balance_mass_fluxes!`, `balance_column_mass_fluxes!`.
Final dry repair can reuse `MetDrivers.recompute_cm_from_dm_target!`.
This split implements no TM5 spectral strategy or new numerical policy.

Keep CS and RG solver projection/reduction policies distinct. Keep FT CS cm
recurrence, Float64 LL/RG recurrence, thresholds, floors and weight booking
unchanged. GEOS diagnostic pressure-fixer closures remain available. Duplicates
whose other copies belong to Targets, TransportBinary or Sources are tracked in
the PP2 handoff for their owning PRs.
