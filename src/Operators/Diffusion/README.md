# Diffusion

Implicit vertical diffusion for the transport runtime.

This folder owns backward-Euler vertical diffusion, its column solves, boundary-layer closures, matching adjoint kernels and
conservative Dkg factorization, and the topology-specific runtime adapters that let the same operator family
run on structured, face-indexed, and panel-native cubed-sphere state.

## Entry Points

- Operator types:
  [`operators.jl`](operators.jl)
  defines `AbstractDiffusion`, `NoDiffusion`, and
  `ImplicitVerticalDiffusion`
- Model-facing runtime entrypoint:
  [`operators.jl`](operators.jl)
  provides `apply!(state, meteo, grid, op, dt; workspace)`
- Array-level runtime entrypoint:
  [`operators.jl`](operators.jl)
  provides `apply_vertical_diffusion_vmr!`
- Surface-flux coupling:
  [`operators.jl`](operators.jl) defines the `AbstractSurfaceFluxCoupling`
  policies, `surface_flux_coupling(op)` and `surface_flux_center!`, the one
  definition of the transport-palindrome center order:
  `V(dt/2) → S(dt) → V(dt/2)` (`SplitSurfaceFluxCoupling`, the default) or
  `S(dt) → V(dt)` (`DiffusiveSurfaceFluxBoundary`). Production palindromes
  call it through `TransportMidpoint` (`../StepPlan/midpoint.jl`);
  the cubed-sphere adjoint replays and records the same composition
- Kernel implementations:
  [`diffusion_kernels.jl`](diffusion_kernels.jl) and
  [`conservative_dkg.jl`](conservative_dkg.jl)

Kz fields use one column
kernel for every topology: one work item per column builds the interface
exchange `D_{k+½}` and the Thomas ratios once, then solves each tracer for the
new mixing ratio on the departure from the column minimum. Tracer-mass states
(`apply_vertical_diffusion_vmr!`) are then updated in flux form,
`rm_k ← (rm_k + F_{k−½}) − F_{k+½}` with `F_{k+½} = Δt D_{k+½} (g_k − g_{k+1})`,
so each interface flux leaves one layer and enters the next with the same
rounded value. Writing back `m_k (g_k + c)` instead (the matrix form) lost
mass with one sign in `Float32`, because the rounding of the diagonal
`1 + Δt (D_{k−½} + D_{k+½})/m_k` is the same in every column of the
pure-pressure levels (finding OF-113). The flux-form rounding that remains,
which can keep one sign where the fluxes keep one direction, goes through the
same column mass ledger with persistent carry as the Dkg solve. Layers
without exchange stay bit-exact. Lat-lon, reduced-Gaussian and all six
cubed-sphere panels each take one launch. The numerics are documented at the
top of [`diffusion_kernels.jl`](diffusion_kernels.jl). Tests compare the
production tracer-mass path with an independent Backward-Euler matrix.

Precomputed cubed-sphere Dkg acts directly on tracer mass through paired
retention/transfer passes. It shares factors across tracers, carries rounding
residuals in registers, and preserves layers with no exchange exactly. There is
no tracer-total normalization. A column mass ledger returns each column's
`Float32` rounding residual to its largest exchanging layer; the part that
layer cannot hold persists in `DiffusionWorkspace.ledger_carry`, one value per
column and tracer, and is added with the column's next residual (see the
[single-precision page](../../../docs/src/theory/float32_conservation.md)).
Packed tracers use carry slot `t`; a single-tracer call selects its slot with
`tracer = idx` (the per-tracer Lin-Rood and reduced-Gaussian paths) and keeps
no carry without it (adjoint replays). The Kz flux-form ledger uses the same
carry and slots on every topology. On CUDA, packed states with multiple tracers factor each column once and then
solve distinct column/tracer pairs in parallel. Factors are read-only during
those solves; no extra workspace is allocated. CPU and Metal retain the fused
column loop. The CUDA launch regression compares all stored values, including
halos, against that serial kernel.

## Supported Layouts

- Structured:
  `q_raw :: (Nx, Ny, Nz, Nt)` with rank-3 Kz fields
- Face-indexed reduced Gaussian:
  `q_raw :: (ncells, Nz, Nt)` or `(ncells, Nz)` with rank-2 Kz fields
- Cubed sphere:
  `NTuple{6}` of halo-padded `(Nc + 2Hp, Nc + 2Hp, Nz)` tracer panels,
  with panel-native Kz wrapped in
  [`../../State/Fields/CubedSphereField.jl`](../../State/Fields/CubedSphereField.jl)

## File Map

- [`Diffusion.jl`](Diffusion.jl) — submodule assembly and public exports
- [`diffusion_kernels.jl`](diffusion_kernels.jl) — the numerics (matrix
  solve for the mixing ratio, flux-form tracer-mass update with column
  ledger), exchange sources (`KzExchange`), and the
  KernelAbstractions kernels for structured, face-indexed, and six-panel
  cubed-sphere solves
- [`conservative_dkg.jl`](conservative_dkg.jl) — mass-conservative bidiagonal
  factorization for precomputed cubed-sphere Dkg, with shared packed factors
- [`dz_helpers.jl`](dz_helpers.jl) — hydrostatic layer-thickness helper
  kernels and host wrappers shared by vertical diffusion paths
- [`workspace.jl`](workspace.jl) — topology-aware `DiffusionWorkspace`
  allocation (column scratch `exchange` and `elimination`, the persistent
  ledger carry) and device adaptation
- [`operators.jl`](operators.jl) — operator hierarchy, surface-flux
  coupling policies and `surface_flux_center!`, constructor validation,
  state-level `apply!`, array-level `apply_vertical_diffusion_vmr!`

- [`pbl_physics.jl`](pbl_physics.jl) — Beljaars-Viterbo constants, scalar
  closure and shared column bodies. WindowPBL uses isothermal geometry;
  LocalHB retains virtual-temperature geometry and shear enhancement.
- [`WindowPBLKzField.jl`](WindowPBLKzField.jl) and
  [`LocalHoltslagBovilleKzField.jl`](LocalHoltslagBovilleKzField.jl) — the
  concrete window-cached fields, kernels and host fallback staging
- [`window_kz_cache.jl`](window_kz_cache.jl) — shared constructor, panel
  access, Adapt, area cache and device-refresh orchestration
- [`gchp_nonlocal_pbl.jl`](gchp_nonlocal_pbl.jl) — non-local GEOS-Chem PBL
  physics and refresh. Its field container remains in State/Fields for
  the fenced SurfaceFlux consumer.
- [`workspace_validation.jl`](workspace_validation.jl) — forward, adjoint
  and advection workspace checks plus the shared ledger selector
- [`adjoint_kernels.jl`](adjoint_kernels.jl) — Kz transpose solve and Dkg
  transpose partitions; tape and objective wiring stays in Adjoints

## Common Tasks

- Adding a new Kz field type:
  make it satisfy the field contract in `State/Fields`, then validate it
  through [`operators.jl`](operators.jl)
- Debugging a workspace mismatch:
  read the shape checks in [`operators.jl`](operators.jl) before looking
  at kernels
- Debugging cubed-sphere diffusion:
  check the panel-native workspace shape and `CubedSphereField`
  handling before touching arithmetic
- Verifying arithmetic changes:
  compare the tracer-mass result from
  [`diffusion_kernels.jl`](diffusion_kernels.jl) with an independent
  Backward-Euler solve
  (`test/core/test_diffusion_flux_form.jl`): conservation alone does not
  show that a solve did anything
- Tracing runtime integration:
  follow `TransportModel.step!` into `../StepPlan/midpoint.jl`

## Cross-Dependencies

- [`../../State/`](../../State/) provides tracer storage, `eachtracer`,
  and time-varying field contracts
- [`../../MetDrivers/`](../../MetDrivers/) provides `current_time`,
  which time-varying Kz fields consume
- [`../StepPlan/midpoint.jl`](../StepPlan/midpoint.jl) supplies diffusion at
  the abstract Advection midpoint and owns physics preflight
- The cubed-sphere adjoint (`../../Adjoints/`, `../../Adjoints/Footprint/`) orders its
  replayed and recorded midpoint with `surface_flux_center!`
- [`../../Models/Core/TransportModel.jl`](../../Models/Core/TransportModel.jl)
  determines whether diffusion is active in a given run

## Related Docs And Tests

- Runtime/block ordering:
  [`../../Models/Core/TransportModel.jl`](../../Models/Core/TransportModel.jl) and
  [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
- Topology status:
  [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md)
- Tests: [`../../../test/core/test_diffusion_flux_form.jl`](../../../test/core/test_diffusion_flux_form.jl)
  (flux form against Backward Euler, bit-exact isolated layers, `Float32`
  drift of the matrix and flux forms, CUDA when available),
  [`../../../test/core/test_diffusion_mass_flux_conservation.jl`](../../../test/core/test_diffusion_mass_flux_conservation.jl),
  [`../../../test/core/test_precomputed_dkg_binary_payload.jl`](../../../test/core/test_precomputed_dkg_binary_payload.jl),
  [`../../../test/core/test_conservative_dkg.jl`](../../../test/core/test_conservative_dkg.jl),
  and [`../../../test/core/test_driven_simulation.jl`](../../../test/core/test_driven_simulation.jl).
  V100/A100 checks are opt-in through
  [`../../../test/diagnostic/test_conservative_dkg_gpu.jl`](../../../test/diagnostic/test_conservative_dkg_gpu.jl).

## Depends on

State/Fields (generic field containers and GCHP storage), State (tracer
storage), Architectures (launches and compensated ledgers), Parameters
(physical constants), MetDrivers (clock), Adapt and KernelAbstractions.
Diffusion loads before Tape and Adjoints and imports neither.

## Used by

StepPlan (transport midpoint), SurfaceFlux (coupling and emission profile),
Models (runtime recipes and window refresh), and Adjoints (transpose kernels).

## Shared helpers owned here

- `check_diffusion_workspace`, `preflight_diffusion`: public storage ownership,
  tracer-count, precision, backend and shape validation before advection.
- `HybridCoefficientWorkspace`, `stage_hybrid_coefficients!`: reusable
  coefficient storage; `recover_total_surface_pressure_staged!`,
  `fill_dz_hydrostatic_constT_staged!`, `fill_dz_hydrostatic_virtualT_staged!`
  consume those coefficients and require explicit `sync`. The allocating
  wrappers were removed because only tests called them.
- `WindowPBLKzField`, `LocalHoltslagBovilleKzField`, `PBLPhysicsParameters`,
  their refresh functions and `refresh_gchp_nonlocal_pbl!` own the runtime
  boundary-layer calculations. `DerivedKzField` and `ProfileKzField` had no
  production callers and were removed; use the window cache or
  `PreComputedKzField` for spatial inputs.
- `surface_flux_coupling`, `surface_flux_center!`: the shared midpoint order.
- The following numerical kernels are explicitly shared with Adjoints;
  CUDA's tile hooks remain shared with its extension.

# shared: _vertical_diffusion_cs_single_adjoint_kernel!, _vertical_diffusion_cs_single_dkg_adjoint_kernel!, _cs_dkg_mass_workgroupsize, _cs_dkg_tracer_workgroupsize

## Forward<->adjoint mirrors

`diffusion_kernels.jl` and `conservative_dkg.jl` pair with
`adjoint_kernels.jl` in this folder. `_kz_exchange`, `_safe_inverse_mass`,
`_dkg_transfer_ratios` and `_dkg_small_fraction` share identical expressions.
The transpose's neighbour-mass normalization, scalar-carry loop, rounding
partitions and launch order are unchanged. Ledger rounding has no adjoint.
`Adjoints/DiffusionAdjoint.jl` retains tape validation, sequence selection
and operator-adjoint wiring.

WindowPBL and LocalHB each use one inline body from host and kernel paths.
They keep distinct column bodies: LocalHB uses per-layer virtual temperature,
`delp * R_dry * tv / (gravity * p_mid)`, a different surface-temperature floor
and shear enhancement. WindowPBL uses the precomputed isothermal
`R_dry * T_sfc / gravity`. Host area staging converts the entire area array to
FT; the kernel converts each loaded area. Neither path has been replaced by
the other. Runtime hydrostatic dz and preprocessing retain their different
operation orders, pressure floors and humidity clamps.

Deferred numerical changes: GCHP's Float64 `cbrt` (BF6), cross-operator
mixing-ratio floors and LocalHB's shear-dependent background floor. Diffusion
retains its absorbing `m <= 0` convention; Advection and diagnostics keep
their respective policies.

Host fallback staging uses Architectures.host_array. Zeroed workspace scratch
uses Architectures.zeros_like, retaining one allocation and fill per array.
