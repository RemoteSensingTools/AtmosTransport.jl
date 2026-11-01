# MassClosure

Horizontal flux construction, wind rotation, Poisson balance and vertical mass-flux diagnosis.

## File map

- [`MassClosure.jl`](MassClosure.jl) — module imports, public API and include order.
- [`column_weights.jl`](column_weights.jl) — hybrid and mass weights; Common interface implementations.
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

Common supplies target geometry and the column-weight protocols. Grids owns face
indexing and geometry; Parameters supplies constants.
This stage follows Common → Vertical → Spectral in pipeline order and imports no
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

None. These are offline preprocessing routines; direct spectral quadrature
uses portable CPU/CUDA kernels.

## Numerical contract and spectral construction

Flux construction (`face_flux_method`, `face_fluxes!`, `line_integral_face_fluxes!`),
balance (`balance_cs_column_mass_fluxes!`, `balance_cs_global_mass_fluxes!`,
`balance_compressed_column_fluxes!`, `solve_cs_poisson_pcg!`,
`solve_compressed_poisson_pcg!`) and cm diagnosis (`diagnose_cs_cm!`,
`diagnose_balanced_cs_cm!`) remain
separately callable. The TM5 spectral strategy shares the target geometry and balance primitives.
Its direct spectral quadrature is separate from the existing source-grid
line-integral method.

The dry pressure mass fix stays separately callable in Spectral through
`pin_global_mean_ps!`, `pin_global_mean_ps_using_qv!` and
`recompute_native_mass_fields!`; LL balance remains there through
`LLPoissonWorkspace`, `balance_mass_fluxes!`, `balance_column_mass_fluxes!`.
Final dry repair can reuse `MetDrivers.recompute_cm_from_dm_target!`.
The optional TM5 strategy adds the independent W constraint before the shared
dry repair; legacy numerical policies remain in their existing dispatch.

CS and RG retain their distinct solvers. Column mean-zero projections use compensated
Float64 reductions; layer/standalone defaults keep their original sums; the FT CS cm recurrence, Float64 LL/RG recurrence,
thresholds, floors and vertical weight booking are unchanged. GEOS diagnostic pressure-fixer closures remain available. Duplicates
whose other copies belong to Targets, TransportBinary or Sources are tracked in
the PP2 handoff for their owning PRs.

# shared: DirectRingPoisson, solve_compressed_poisson_direct!, balance_compressed_level!

`tm5_target_balance.jl` contains only the reusable compressed direct solver.
Spectral owns quadrature and endpoint construction; Targets owns layer
orchestration, dry conversion and endpoint averaging. MassClosure imports
neither Spectral nor Vertical. The direct solver is reused for RG layer and
final dry column repair; the PCG and direct paths share the column global-mode treatment.

## Column global mode

For fixed dry targets, LL (owned by Spectral), RG PCG/direct and CS column balances shift the solver
target by `R̄ (M_c / mean(M) - 1)` using current column mass and compensated
global reductions. Original endpoint tendencies remain the vertical-closure
and replay targets. The residual global mode therefore follows column mass
instead of being uniform per cell. The standalone CS per-level target is
unchanged. Replay tolerances and CS typed closure tolerances are unchanged.

This reaches spectral LL/RG/CS and TM5 LL/RG, LL-binary-to-CS and ERA5 N320
regrids, MERRA-2, and GEOS endpoint-balanced/moisture-filtered/OMEGA closures wherever
they call the shared column routines. Native mass pins remain separate.
The manual's mass-conservation page records the equations and full-day evidence.

On ERA5 2021-12-01 (1°, Float64 dry mass, 34 levels, 24 windows), replay
maxima decrease from 1.471e-12 to 5.823e-15 for TM5 and from
1.064e-12 to 6.107e-15 for point-sampled construction. Counts above
1e-12 decrease from 536 and 6 to zero. Maximum |global ΔM| per window decreases
from 135,695.388 to 666.004 kg (TM5), and from
121,982.453 to 659.033 kg (point-sampled).

The mass-proportional global-mode shift is enabled only for fixed dry-mass
targets, after Common's compensated mass-normalized budget guard (`1e-14`
for Float64; lower precision allows two-endpoint storage rounding). Moist and
unpinned paths retain their uniform global mode and original endpoint tendency.
TM5 LL/RG layer solves and standalone CS per-level solves retain original
projection arithmetic; only column calls select compensated projection dispatch.
