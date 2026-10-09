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
- Numerical reference pieces:
  [`thomas_solve.jl`](thomas_solve.jl)
  provides the allocation-free `solve_tridiagonal!` reference
- Kernel implementations:
  [`diffusion_kernels.jl`](diffusion_kernels.jl) and
  [`conservative_dkg.jl`](conservative_dkg.jl)

Precomputed cubed-sphere Dkg acts directly on tracer mass through paired
retention/transfer passes. It shares factors across tracers, carries rounding
residuals in registers, and preserves layers with no exchange exactly. There is
no tracer-total normalization. The public VMR-level solver retains its Thomas
implementation; state-mass entry points dispatch to the conservative path.
On CUDA, packed states with multiple tracers factor each column once and then
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
- [`diffusion_kernels.jl`](diffusion_kernels.jl) — KernelAbstractions
  kernels for structured, face-indexed, and cubed-sphere panel solves
- [`conservative_dkg.jl`](conservative_dkg.jl) — mass-conservative bidiagonal
  factorization for precomputed cubed-sphere Dkg, with shared packed factors
- [`dz_helpers.jl`](dz_helpers.jl) — hydrostatic layer-thickness helper
  kernels and host wrappers shared by vertical diffusion paths
- [`workspace.jl`](workspace.jl) — topology-aware `DiffusionWorkspace`
  allocation and device adaptation
- [`operators.jl`](operators.jl) — operator hierarchy, constructor
  validation, state-level `apply!`, array-level `apply_vertical_diffusion_vmr!`

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
  [`diffusion_kernels.jl`](diffusion_kernels.jl) aligned
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
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  determines whether diffusion is active in a given run

## Related Docs And Tests

- Runtime/block ordering:
  [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl) and
  [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
- Topology status:
  [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md)
- Tests: [`../../../test/core/test_diffusion_mass_flux_conservation.jl`](../../../test/core/test_diffusion_mass_flux_conservation.jl),
  [`../../../test/core/test_precomputed_dkg_binary_payload.jl`](../../../test/core/test_precomputed_dkg_binary_payload.jl),
  [`../../../test/core/test_conservative_dkg.jl`](../../../test/core/test_conservative_dkg.jl),
  and [`../../../test/core/test_driven_simulation.jl`](../../../test/core/test_driven_simulation.jl).
  V100/A100 checks are opt-in through
  [`../../../test/diagnostic/test_conservative_dkg_gpu.jl`](../../../test/diagnostic/test_conservative_dkg_gpu.jl).
