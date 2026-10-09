# Adjoints

Reverse-mode (discrete adjoint) kernels for cubed-sphere transport. This
module also hosts the footprint and 4D-Var code.

`Adjoints` is the module that contains this folder,
[`../Footprint/`](../Footprint/), and [`../Inversion/`](../Inversion/). Files in
this folder hold the per-operator reverse kernels: split-sweep advection, seam
exchange, halo fill, vertical diffusion, convection, and the Lin-Rood tape.
Meteorology (air mass and fluxes) is fixed input. Only tracer mass is
differentiated. The runtime (`Models`) does not use this module; it is called
by scripts, tests, and user code.

## How It Is Loaded

[`../AtmosTransport.jl`](../AtmosTransport.jl) includes `Tape/Tape.jl` and then
`Adjoints/Adjoints.jl` (after `Operators`), followed by `using .Adjoints`.
[`Adjoints.jl`](Adjoints.jl) includes files in an order that matters,
because later files use types defined earlier:
`ObjectiveSeeding.jl` and `Footprint/FootprintResult.jl`; then the
`Inversion/` type, IO, covariance, and preconditioner files; then inline
helpers (`CSAdjointWorkspace`, `_add_surface_rates!`); then the kernel files
in this folder ending with `LinRoodTape.jl`; then the remaining
`Footprint/` files; and finally `Inversion/Jacobian.jl`, `CostGradient.jl`,
and `Optimizer.jl`. The export block at the end of `Adjoints.jl` is the
public surface for all three folders. `AtmosTransport` makes these names
reachable as `AtmosTransport.<name>` but does not re-export them.

## Entry Points

- Scheme support unions in [`Adjoints.jl`](Adjoints.jl):
  `CSAdjointLinearScheme` (`UpwindScheme`, `SlopesScheme{NoLimiter}`,
  `PPMScheme{NoLimiter, SameAsHorizontal}`), `CSAdjointNonlinearScheme`
  (`PPMScheme{MonotoneLimiter, SameAsHorizontal}`), `CSAdjointLinRoodScheme`
  (`LinRoodPPMScheme{<:Any, UpwindScheme}`), and their union
  `CSAdjointSupportedScheme`. The FV3 vertical profile is not supported.
- `CSAdjointWorkspace(mesh, prototype)` in [`Adjoints.jl`](Adjoints.jl):
  per-panel transpose scratch plus the 12-seam seed cache.
- Objectives in [`ObjectiveSeeding.jl`](ObjectiveSeeding.jl):
  `CSLayerMeanObjective`, `CSColumnMeanObjective`, `CSSeedObjective`, and
  `evaluate_objective`.
- Reverse kernels called by `_walk_window_reverse!`
  (`../Footprint/ReverseLoop.jl`): `_adjoint_scheme_sweep!`,
  `_adjoint_fill_panel_halos!`, `_apply_cs_diffusion_adjoint!`,
  `_apply_cs_convection_adjoint!`, `_apply_cs_linrood_horizontal_adjoint!`.
- User-facing drivers live in `../Footprint/` (`cs_surface_emission_footprint`)
  and `../Inversion/` (`cs_surface_flux_4dvar`).

## File Map

- [`Adjoints.jl`](Adjoints.jl) — module, imports from `Operators`/`Tape`,
  scheme unions, include order, workspace, surface-rate kernels, exports
- [`ObjectiveSeeding.jl`](ObjectiveSeeding.jl) — objective types, forward
  evaluation, final-time seed `_seed_objective!`, per-step surface read-out
  `_accumulate_surface_footprint!`
- [`AdvectionAdjoint.jl`](AdvectionAdjoint.jl) — per-scheme face
  coefficients, x/y/z sweep adjoint kernels, `_adjoint_scheme_sweep!`.
  The monotone-PPM variant is the linearization around the recorded `rm`,
  computed with six-component derivative tuples (`_d6_*`).
- [`CubedSphereSeams.jl`](CubedSphereSeams.jl) — transpose of the paired
  physical-seam exchange in `../Operators/Advection/CubedSphereSeams.jl`
- [`HaloAdjoint.jl`](HaloAdjoint.jl) — transpose of `fill_panel_halos!` and
  `copy_corners!`. Halo seeds are atomically added into the source interior
  cells.
- [`DiffusionAdjoint.jl`](DiffusionAdjoint.jl) — transposed implicit
  vertical diffusion for CS `Kz` fields and for precomputed `Dkg` fields,
  plus input validation
- [`ConvectionAdjoint.jl`](ConvectionAdjoint.jl) — TM5 LU solves and their
  transposes, CMFMC and TM5 column kernels, and forward-replay/adjoint arms
  for `NoConvection`, `CMFMCConvection`, `TM5Convection`, and
  `CMFMCMatrixConvection`
- [`LinRoodTape.jl`](LinRoodTape.jl) — `_CSLinRoodHorizRecord`,
  `_record_cs_linrood_tape`, `_apply_cs_linrood_horizontal_adjoint!`,
  `_linrood_run_forward_step!`. The per-kernel Lin-Rood adjoints live in
  `../Operators/Advection/linrood_adjoint_kernels.jl`.

## Common Tasks

- Supporting a new advection scheme: add it to a `CSAdjoint*Scheme` union,
  add face-coefficient and `_add_*_face_adjoint!` methods in
  [`AdvectionAdjoint.jl`](AdvectionAdjoint.jl), and extend the recorder
  dispatch in `../Footprint/TapeRecording.jl`.
- Supporting a new convection or diffusion variant: add methods for
  `_apply_cs_convection_forward!` and `_apply_cs_convection_adjoint!`
  (or `_apply_cs_diffusion_adjoint!`), then update the `_require_*`
  guards so unsupported settings fail before any tape is recorded.
- Debugging a footprint/finite-difference mismatch: check that the forward
  replay matches production first (see Invariants), then run the
  kernel-level identity tests listed below.

## Invariants

- Each reverse kernel is the transpose of the forward operator the tape
  replays. For monotone PPM and Lin-Rood, it transposes the linearization
  around the recorded state, so limiter branches are frozen. Kernel tests
  check `⟨y, L·x⟩ = ⟨Lᵀ·y, x⟩` to rounding error.
- The forward replays must match production: CMFMC replay uses the
  production `_cmfmc_cs_panel_column_kernel!`, and TM5 column solves are
  checked bit-for-bit. The Lin-Rood vertical sweep is `UpwindScheme()`
  because that is what `_sweep_z!` runs in production.
- Forward settings that the adjoint cannot transpose are rejected:
  `CMFMCConvection(clamp = true)` or an archived cloud base; `TM5Convection`
  with `use_collab_lu`, `lmax_conv != 0`, or `n_merge != 1`; GEOS-Chem
  non-local VDIFF (`GCHPNonlocalPBLField`); a Holtslag-Boville `Kz` cache
  that is all zero or non-finite.
- Lin-Rood tapes accept only `tape_storage = :device`; `tape_path` is
  rejected.
- Level `Nz` is the surface layer. Surface rates are added to level `Nz`,
  and footprints are read from `lambda[:, :, Nz]` on the interior
  `Hp+1:Hp+Nc` cells.
- `FT` is taken from the panel eltype. `dt` and `flux_scale` are converted
  to `FT`. Kernels use KernelAbstractions, so CPU and GPU share one code
  path.

## Related Docs And Tests

- [`../../docs/src/theory/adjoint_status.md`](../../docs/src/theory/adjoint_status.md),
  [`../../docs/src/getting_started/adjoints.md`](../../docs/src/getting_started/adjoints.md),
  [`../../docs/src/api/adjoints.md`](../../docs/src/api/adjoints.md)
- Kernel transpose tests in [`../../test/core/`](../../test/core/):
  `test_linrood_kernel_adjoints.jl` (Lin-Rood), `test_cmfmc_adjoint_identity.jl`,
  `test_cmfmc_matrix_convection.jl`, `test_tm5_hessenberg.jl`,
  `test_tm5_bidiagonal_solve.jl` (convection),
  `test_diffusion_mass_flux_conservation.jl`,
  `test_precomputed_dkg_binary_payload.jl` (diffusion),
  `test_cs_seam_exchange.jl`, `test_cubed_sphere_advection.jl` (seams), and
  `test_fv3_vertical_profile.jl` (which schemes are supported)
- End-to-end footprint tests: see [`../Footprint/README.md`](../Footprint/README.md).
  Opt-in GPU checks: `../../test/diagnostic/test_cs_transport_adjoint_gpu.jl`
  and `test_linrood_adjoint_integration.jl`
