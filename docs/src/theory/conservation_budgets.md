# Conservation budgets

The conservation contract from [Mass conservation](@ref) is enforced
by **explicit `@test` assertions** in the test suite. This page lists
the tests that anchor each conservation property, with file:line
citations and the exact tolerance bounds so a reader can verify by
inspection.

All tolerance values quoted here come straight from the test source;
none of them are bit-exact. Bit-exact conservation isn't achievable
in floating-point — what the tests assert is closure within
machine-precision noise floors that scale with `FT`.

## Uniform tracer preservation

The simplest non-trivial conservation property: a tracer initialised
to a uniform constant value `χ_0` everywhere should stay close to
`χ_0` after multi-step advection on a non-trivial flow.

| Test (`test/core/test_advection_kernels.jl`) | Asserts | Tolerance |
|---|---|---|
| CPU uniform-invariance test sets | `maximum(abs(χ_out − 4.0e-4)) / 4.0e-4 < 1e-6` after one Strang step on a sinusoidal flow. CPU coverage spans `Upwind` / `Slopes` / `PPM`. | `< 1e-6` (both F32 and F64) |
| GPU uniform-invariance test set | same assertion on `CuArray`-backed state — currently for **`Upwind`**. The Slopes / PPM uniform-invariance cases are CPU-only. | `< 1e-6` |

The test fixture builds a synthetic LL `36 × 18 × 4` setup
together with a sinusoidal zonal `am`
tapered with latitude and a `cm` diagnosed from continuity to keep
the synthetic forcing self-consistent. CFL ≈ 0.15 at the equator.
Schemes covered: `UpwindScheme`, `SlopesScheme`, `PPMScheme` — each
in its own `@testset` block. CPU coverage spans all three; **GPU
uniform-invariance coverage is `Upwind` only**. The Slopes / PPM uniform tests
run on CPU only.

## Mass-budget conservation

Same kernel, but on a non-uniform tracer (a meridional gradient
`rm_grad_cpu`):

| Test (`test/core/test_advection_kernels.jl`) | Asserts | Tolerance |
|---|---|---|
| CPU mass conservation, uniform | `abs(Σ m_out − Σ m_cpu) / Σ m_cpu < tol` and the same for tracer mass | F64: `< 1e-13`. F32: `< 1e-5` |
| CPU mass conservation, gradient over 4 steps | same assertion after 4 Strang steps with a gradient IC | F64: `< 1e-12`. F32: `< 5e-5` |
| GPU mass conservation, gradient over 4 steps | same on a `CuArray`-backed state | F64: `< 1e-12`. F32: `< 5e-5` |
| CPU non-trivial transport | `maximum(abs(rm_out − rm_grad_cpu)) > 0` — sanity that the run actually moved tracer mass (not just held it constant) | strict `>` |

`test/core/test_cubed_sphere_advection.jl` covers the PPM-on-CS variants of the
same suite, exercising the panel-edge halo sync end to end.

## CPU / GPU agreement

Bit-exact CPU/GPU agreement is **not** asserted: floating-point
associativity differs between sequential CPU sums and parallel GPU
reductions, and on F64 the GPU's FMA instruction can produce a
small ULP-scale difference per multiply-add. The test suite uses
ULP-bounded tolerances that vary by scheme and step count:

| Test (`test/core/test_advection_kernels.jl`) | Tolerance |
|---|---|
| Upwind, 1 step | `4 * eps(FT)` per cell |
| Upwind, 4 steps | `16 * eps(FT)` per cell |
| Slopes, 4 steps | `16 * eps(FT)` per cell |
| PPM, 4 steps | `16 * eps(FT)` per cell |

`LinRoodPPMScheme` is not in this CPU/GPU agreement matrix; it has a
forward runtime coverage in the opt-in orphan test
`test/orphan/test_cubed_sphere_runtime.jl`, but no per-step GPU comparison.

The CPU/GPU agreement check runs ONLY when CUDA.jl is loaded; it's
gated by `HAS_GPU` at the top of the test file. CI runs the CPU
side; GPU coverage is exercised on machines with hardware.

## Cross-window mass closure (replay gate)

Per-window replay is the contract that lets the runtime stream window
N+1 starting from window N's evolved endpoint without drift. Tested
in `test/core/test_replay_consistency.jl`:

| Subtest (line) | Asserts |
|---|---|
| `verify_window_continuity_ll` with continuity-consistent data | `tol_rel ≤ 1e-12` (F64) / `1e-6` (F32). Both pass on the synthetic fixture. |
| Deliberately broken `cm` storage | The storage-level gate fires `@test_throws ErrorException` when the binary's stored `cm` violates the explicit-`dm` closure by more than `replay_tolerance(FT)`. A separate subtest checks the diagnostic residual against the same broken field. |
| Final-window inconsistent `cm` | The LL storage replay-gate test deliberately injects an inconsistent final-window `cm` and asserts `@test_throws ErrorException` — i.e. the gate **does** detect inconsistency at the day boundary, not silently pass it as zero-tendency. |

The same gate runs at preprocessing write time (always) and at
runtime load time (opt-in via `[met_data] validate_replay = true` or
`ATMOSTR_REPLAY_CHECK = 1`). The test exercises the function
directly so the contract is validated independently of the
preprocessor / runtime drivers.

## Cross-topology (regridding) conservation

When mass moves between topologies (LL → CS in the spectral
preprocessor; LL → RG in the LL→RG variant), the conservative
regridder preserves total mass exactly but may shift the per-level
distribution by `O(10⁻⁶)`.

| Test | Asserts |
|---|---|
| `test/regridding/test_conservation.jl` | `sum(m_dest) ≈ sum(m_source)` under LL↔CS conservative regrid. |
| `test/core/test_ll_to_cs_regrid_script.jl` | End-to-end LL → CS regrid pipeline produces a binary whose stored `m`, summed globally, matches the LL source's stored `m` to `1e-6` relative — the script-level acceptance gate, looser than the kernel-level `1e-13`. |

`m` is regridded with `regrid_3d_to_cs_panels!(..., ExtensiveCellField())`
(`cs_transport_helpers.jl`), which converts to density on the source mesh and
back to cell mass on the target, so each level's global integral is conserved
to the regridder's tolerance. The Poisson balance then closes continuity and
the write-time replay gate verifies it.

## Initial-condition mass conservation

When a uniform-VMR initial condition is constructed, the conversion
`χ × air_mass` should yield a tracer-mass field whose ratio to
`air_mass` is `χ` everywhere. Trivially true for a uniform IC;
matters when the IC interpolates from a different mesh.

| Test | Asserts |
|---|---|
| `test/core/test_initial_condition_io.jl` | The `file` / `netcdf` IC kinds round-trip a known field through the IC pipeline and assert the recovered mixing ratio matches the source within tolerance. |
| `test/core/test_basis_explicit_core.jl` | Dry-basis IC interpretation: `[tracers.co2.init] kind = "uniform"; background = 4.0e-4` produces a tracer field whose `mixing_ratio(state, :CO2)` is `4.0e-4` exactly when `air_mass` is on dry basis. |

## Preprocessor contract suite

The unified preprocessor (`run_unified_preprocessor_day!`)
anchors its window-contract invariants with a dedicated test set:

| Test | Asserts |
| --- | --- |
| `test_ll_preprocessor_contract.jl` | `LatLonContract` + `LatLonSpectralWindowWorkspace` + LL writer satisfy mass-balance and per-substep positivity gates. |
| `test_rg_preprocessor_contract.jl` | Same on the face-indexed RG topology, including the boundary-stub-flux invariant for pole-singular faces. |
| `test_cs_preprocessor_contract.jl` | Same on cubed-sphere, including the palindrome-budget positivity check. |
| `test_preprocessor_unified_driver.jl` | The driver dispatches correctly across (source × vertical × target). |
| `test_preprocessor_writer_adapters.jl` | Each `AbstractBinaryWriter` matches its paired `AbstractWindowContract` on basis and topology at the type level. |
| `test_{ll,rg,cs}_spectral_unified_driver.jl` | End-to-end day builds for the spectral path. |

## Inversion / adjoint suite

The CS surface-emission footprint and 4D-Var stack are anchored by:

| Test | Asserts |
| --- | --- |
| `test_cs_ppm_adjoint_footprint.jl` | Kernel-level transposition + finite-difference VJPs for split-sweep schemes. |
| `test_linrood_kernel_adjoints.jl` | Kernel-level transposition for the LinRood adjoint kernels (ORD=5 and ORD=7). |
| `test_cs_inversion_truth_recovery.jl` | Synthetic-truth-recovery end-to-end through `cs_surface_flux_4dvar_solve` + L-BFGS. |
| `test_cs_4dvar_preconditioned.jl`, `test_cs_lbfgs.jl`, `test_cs_optimizer_dispatch.jl` | Preconditioning, optimizer dispatch, and gradient identities. |
| `test_cs_covariance.jl`, `test_cs_preconditioning.jl` | `apply_B_half!` / `apply_B_half_adjoint!` / `apply_B_half_inverse!` identities. |
| `test_cs_observations_io.jl`, `test_cs_observation_binding.jl`, `test_cs_departures_io.jl` | Observation IO + bind-to-mesh + departures-file round-trip. |
| `test_cs_stride_checkpoint.jl`, `test_cs_tape_mmap_roundtrip.jl` | Tape storage and checkpoint scheduler correctness, including `RevolveCheckpoint` cases. |

## Test-pass status

The `test/runtests.jl` entry point discovers more than 100 files under
`test/core/`; they run without external meteorology. CI runs that tier and the
regridding tier on pull requests and pushes to `main`. Real-data tests are opt-in via
`--real-data` or `--all` and require locally staged inputs.

Total core-suite case count is in the thousands; CI reports per-test
pass / fail breakdown on every run.

## What's next

- [Validation status](@ref) — what we've validated end-to-end against
  external reference data, vs what's still synthetic-fixture-only.
- [Adjoint status](@ref) — the verified support boundary and remaining gaps.
