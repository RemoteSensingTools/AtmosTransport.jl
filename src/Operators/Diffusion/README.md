# Diffusion

Implicit vertical diffusion for the transport runtime.

This folder owns backward-Euler vertical diffusion, its Thomas reference and
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
  call it through `palindrome_center!` (`../Advection/StrangSplitting.jl`);
  the cubed-sphere adjoint replays and records the same composition
- Numerical reference pieces:
  [`thomas_solve.jl`](thomas_solve.jl)
  provides the allocation-free `solve_tridiagonal!` reference
- Kernel implementations:
  [`diffusion_kernels.jl`](diffusion_kernels.jl) and
  [`conservative_dkg.jl`](conservative_dkg.jl)

Kz fields (and the VMR-level reference solve of any field) use one column
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
top of [`diffusion_kernels.jl`](diffusion_kernels.jl). The VMR-level
`apply_vertical_diffusion!` (mixing ratios in and out) keeps the matrix form
as the reference solve.

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
- [`thomas_solve.jl`](thomas_solve.jl) — reference Thomas solve and
  coefficient builder
- [`diffusion_kernels.jl`](diffusion_kernels.jl) — the numerics (matrix
  solve for the mixing ratio, flux-form tracer-mass update with column
  ledger), exchange sources (`KzExchange`, `DkgExchange`), and the
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
  keep [`thomas_solve.jl`](thomas_solve.jl) and the kernel formulas in
  [`diffusion_kernels.jl`](diffusion_kernels.jl) aligned, and compare the
  tracer-mass result with an independent Backward-Euler solve
  (`test/core/test_diffusion_flux_form.jl`): conservation alone does not
  show that a solve did anything
- Tracing runtime integration:
  follow calls from `TransportModel.step!` into advection midpoint hooks
  in `../Advection/StrangSplitting.jl` (RG: `../Advection/strang_apply.jl`)

## Cross-Dependencies

- [`../../State/`](../../State/) provides tracer storage, `eachtracer`,
  and time-varying field contracts
- [`../../MetDrivers/`](../../MetDrivers/) provides `current_time`,
  which time-varying Kz fields consume
- [`../Advection/StrangSplitting.jl`](../Advection/StrangSplitting.jl),
  [`../Advection/multitracer_strang.jl`](../Advection/multitracer_strang.jl) and
  [`../Advection/strang_apply.jl`](../Advection/strang_apply.jl)
  embed diffusion at the transport midpoint
- The cubed-sphere adjoint (`../../Adjoints/`, `../../Footprint/`) orders its
  replayed and recorded midpoint with `surface_flux_center!`
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  determines whether diffusion is active in a given run

## Related Docs And Tests

- Runtime/block ordering:
  [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl) and
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
