# Adjoints

Reverse-mode (discrete adjoint) kernels for cubed-sphere transport. This
module also hosts the footprint and 4D-Var code.

`Adjoints` is the module that contains this folder,
[`../Footprint/`](../Footprint/), and [`../Inversion/`](../Inversion/). Files in
this folder hold the per-operator reverse kernels: split-sweep advection, seam
exchange, halo fill, vertical diffusion, convection, and the Lin-Rood tape.
Meteorology (air mass and fluxes) is fixed input. Only tracer mass is
differentiated. The forward model the reverse pass transposes is the
production `StepPlan` (`../Operators/StepPlan/`), replayed with the production
operators; [`AdjointSupport.jl`](AdjointSupport.jl) is the setup guard that
rejects a plan with an operator the tape cannot transpose. The runtime
(`Models`) uses only `CSStepSchedule` and `require_adjoint_support` (for a
`DrivenSimulation`, `../Models/adjoint_schedule.jl`); the adjoint is called by
scripts, tests, and user code.

## How It Is Loaded

[`../AtmosTransport.jl`](../AtmosTransport.jl) includes `Tape/Tape.jl` and then
`Adjoints/Adjoints.jl` (after `Operators`), followed by `using .Adjoints`.
[`Adjoints.jl`](Adjoints.jl) includes files in an order that matters,
because later files use types defined earlier:
`ObjectiveSeeding.jl`, `Footprint/FootprintResult.jl` and
`Footprint/StepForcing.jl`; then the
`Inversion/` type, IO, covariance, and preconditioner files; then inline
helpers (`CSAdjointWorkspace`); then the kernel files in this folder ending
with `LinRoodTape.jl`; then `Footprint/StepReplay.jl`, `AdjointSupport.jl`
and the remaining `Footprint/` files; and finally `Inversion/Jacobian.jl`, `CostGradient.jl`,
and `Optimizer.jl`. The export block at the end of `Adjoints.jl` is the
public surface for all three folders. `AtmosTransport` makes these names
reachable as `AtmosTransport.<name>` but does not re-export them.

## Entry Points

- Scheme support unions in [`Adjoints.jl`](Adjoints.jl):
  `CSAdjointLinearScheme` (`UpwindScheme`, `SlopesScheme{NoLimiter}`,
  `PPMScheme{NoLimiter, SameAsHorizontal}`), `CSAdjointNonlinearScheme`
  (`PPMScheme{MonotoneLimiter}` and `PPMScheme{CW84Limiter}`, both with
  `SameAsHorizontal`), `CSAdjointFV3Vertical` (`PPMScheme` with
  `FV3ScalarProfile`, recorded only as Lin-Rood's vertical sweep),
  `CSAdjointLinRoodScheme` (`LinRoodPPMScheme` with upwind or FV3 vertical
  sweeps), and their union `CSAdjointSupportedScheme`. The split-sweep
  `PPMScheme` with the FV3 vertical profile is not supported.
  `require_adjoint_support(scheme)` is the dispatch-based gate on this union:
  it returns a supported scheme and otherwise throws an error naming the
  missing piece (used by the inversion driver for its `[advection]` table).
- The setup guard in [`AdjointSupport.jl`](AdjointSupport.jl):
  `adjoint_support(x)` (`nothing` for a supported plan stage, scheme or
  operator, otherwise the reason; unsupported is the fallback) and
  `require_adjoint_support(plan, ops...)`, which every entry point calls
  before recording and which throws an error naming every unsupported
  piece.
- `CSAdjointWorkspace(mesh, prototype; column_scratch = false)` in
  [`Adjoints.jl`](Adjoints.jl): per-panel transpose scratch, the 12-seam seed
  cache, and (with `column_scratch = true`) the per-column storage of the FV3
  vertical sweep's reverse pass.
- Objectives in [`ObjectiveSeeding.jl`](ObjectiveSeeding.jl):
  `CSLayerMeanObjective`, `CSColumnMeanObjective`, `CSPlacedPointObjective`
  (an in-situ sample placed in the column; its layer is chosen per sampled
  state from the column meteorology, so only `cs_surface_flux_4dvar` takes
  it), `CSSeedObjective`, and `evaluate_objective`.
- Reverse kernels called by `_walk_window_reverse!`
  (`../Footprint/ReverseLoop.jl`): `_adjoint_scheme_sweep!`,
  `_adjoint_fill_panel_halos!`, `_apply_cs_diffusion_adjoint!`,
  `_apply_cs_convection_adjoint!`, `_apply_cs_linrood_horizontal_adjoint!`.
- User-facing drivers live in `../Footprint/` (`cs_surface_emission_footprint`)
  and `../Inversion/` (`cs_surface_flux_4dvar`).

## File Map

- [`Adjoints.jl`](Adjoints.jl) — module, imports from `Operators`/`Tape`,
  scheme unions, include order, workspace, exports
- [`AdjointSupport.jl`](AdjointSupport.jl) — the setup guard
  (`adjoint_support`, `require_adjoint_support` for a plan and its operators
  and for a replay)
- [`ObjectiveSeeding.jl`](ObjectiveSeeding.jl) — objective types, forward
  evaluation, final-time seed `_seed_objective!` (zero the adjoint state, then
  add the objective's linear functional from `../Footprint/StepForcing.jl`;
  the seed values are unchanged), per-step surface read-out
  `_accumulate_surface_footprint!`
- [`AdvectionAdjoint.jl`](AdvectionAdjoint.jl) — per-scheme face
  coefficients, x/y/z sweep adjoint kernels, `_adjoint_scheme_sweep!`.
  The monotone-PPM variant is the linearization around the recorded `rm`,
  computed with six-component derivative tuples (`_d6_*`). The FV3 vertical
  method calls the column kernel in
  `../Operators/Advection/vertical_fv3_profile_adjoint.jl`.
- [`CubedSphereSeams.jl`](CubedSphereSeams.jl) — transpose of the paired
  physical-seam exchange in `../Operators/Advection/CubedSphereSeams.jl`
- [`HaloAdjoint.jl`](HaloAdjoint.jl) — transpose of `fill_panel_halos!` and
  `copy_corners!`. Halo seeds are atomically added into the source interior
  cells.
- [`DiffusionAdjoint.jl`](DiffusionAdjoint.jl) — transposed implicit
  vertical diffusion for CS `Kz` fields and for precomputed `Dkg` fields,
  plus input validation
- [`ConvectionAdjoint.jl`](ConvectionAdjoint.jl) — TM5 LU solves and their
  transposes, the CMFMC and TM5 adjoint column kernels, and the adjoint arms
  for `NoConvection`, `CMFMCConvection` (with the recorded CFL sub-step
  count), `TM5Convection`, and `CMFMCMatrixConvection`;
  `_apply_cs_convection_forward!` (production `apply_convection!` on
  single-tracer panels) serves the per-operator identity tests only
- [`LinRoodTape.jl`](LinRoodTape.jl) — `_CSLinRoodHorizRecord` (built from
  the production `fv_tp_2d_cs!`'s recorder hooks) and
  `_apply_cs_linrood_horizontal_adjoint!`. The per-kernel Lin-Rood adjoints
  live in `../Operators/Advection/linrood_adjoint_*.jl`.

## Common Tasks

- Supporting a new advection scheme: add it to a `CSAdjoint*Scheme` union,
  add face-coefficient and `_add_*_face_adjoint!` methods in
  [`AdvectionAdjoint.jl`](AdvectionAdjoint.jl), and extend the
  `_record_sweep!` dispatch in `../Footprint/TapeRecording.jl`.
- Supporting a new convection or diffusion variant: add the adjoint arm
  (`_apply_cs_convection_adjoint!` or `_apply_cs_diffusion_adjoint!`) and an
  `adjoint_support` method returning `nothing` in
  [`AdjointSupport.jl`](AdjointSupport.jl). The forward is production's: the
  replay calls the operator's `apply!` through the step plan.
- Supporting a new plan stage or operator type: the guard rejects it until it
  has an `adjoint_support` method, a recorder hook in production, a tape
  record and a reverse arm in `_walk_window_reverse!`.
- Debugging a footprint/finite-difference mismatch: run
  `test_cs_adjoint_step_plan_parity.jl` (replay == production), then the
  kernel-level identity tests listed below.

## Invariants

- Each reverse kernel is the transpose of the forward operator the tape
  replays. For monotone PPM, Lin-Rood and the FV3 vertical profile, it
  transposes the linearization around the recorded state, so limiter branches
  are frozen. Kernel tests check `⟨y, L·x⟩ = ⟨Lᵀ·y, x⟩` to rounding error.
- The forward replay is production: `_replay_steps!`
  (`../Footprint/StepReplay.jl`) executes the production `StepPlan` with the
  production operators and their persistent state (the surface source's Kahan
  compensation, the TM5/CMFMC/Dkg ledger carries, the CMFMC CFL cache, all
  reset to a fresh run's at replay start and saved with every checkpoint),
  and the tape is recorded through the production step's recorder hooks.
  `test_cs_adjoint_step_plan_parity.jl` checks replay, recording and
  checkpoint restarts against the production `DrivenSimulation` bit for bit.
  The ledger carries and the Kahan compensation shape the forward's rounding
  only; the reverse kernels transpose the exact operators.
  The FV3 reverse replays the forward profile functions themselves, on
  `_LinearForm` numbers, so its branches are the forward's.
- A sweep record holds the state entering the sweep: the reverse
  coefficients divide by the pre-sweep donor mass. The production
  palindromes call `record_sweep!` before each sweep.
- Forward settings that the adjoint cannot transpose are rejected by the
  guard before recording: `CMFMCConvection(clamp = true)` or an archived cloud
  base; `TM5Convection` with `use_collab_lu`, `lmax_conv != 0`, or
  `n_merge != 1`; GEOS-Chem non-local VDIFF (`GCHPNonlocalPBLField`);
  chemistry other than `NoChemistry`; any stage, scheme or operator without an
  `adjoint_support` method. A Holtslag-Boville `Kz` cache that is all zero or
  non-finite is rejected by the diffusion input validation.
- Lin-Rood tapes accept only `tape_storage = :device`; `tape_path` is
  rejected.
- The reverse walk reads staged states with a read role
  (`_tape_panels(slot, AirMassRead())`, `TracerMassRead()` for a limited
  sweep's tracer mass): host-side tape storage keeps one device cache per
  role, so one kernel's two staged operands are distinct buffers.
- Level `Nz` is the surface layer. The surface source adds its rates to
  level `Nz`, and footprints are read from `lambda[:, :, Nz]` on the interior
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
  `test_cs_seam_exchange.jl`, `test_cubed_sphere_advection.jl` (seams),
  `test_cs_linrood_fv3_adjoint.jl` (FV3 vertical sweep), and
  `test_fv3_vertical_profile.jl` (which schemes are supported)
- End-to-end footprint tests: see [`../Footprint/README.md`](../Footprint/README.md).
  Opt-in GPU checks: `../../test/diagnostic/test_cs_transport_adjoint_gpu.jl`
  and `test_linrood_adjoint_integration.jl`
