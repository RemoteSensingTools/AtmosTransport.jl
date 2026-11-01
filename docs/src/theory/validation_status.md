# Validation status

This page is the honest current-state report of what AtmosTransport has
been validated against, what hasn't been validated yet, and the
floating-point tolerances that hold in each case. The goal is to let an
atmospheric-transport practitioner decide quickly whether the level of
validation here meets their needs — and where the gaps are.

## Verification vs validation

The terms in their canonical sense:

- **Verification** ("are we solving the equations correctly?") — the
  test suite covers this for the three schemes
  (`UpwindScheme`, `SlopesScheme`, `PPMScheme`) that live in the
  shared kernel test matrix in `test/core/test_advection_kernels.jl`:
  uniform-invariance, mass-budget, and CPU/GPU-agreement tests all
  run for those three. `LinRoodPPMScheme` has a CPU CS runtime smoke test in
  `test/core/test_cubed_sphere_runtime.jl`, plus core kernel and
  adjoint tests, but is not covered by the same per-step CPU/GPU matrix. The replay gate enforces the
  discrete-conservation contract on every preprocessor write and
  every opt-in runtime load, regardless of scheme. See
  [Conservation budgets](@ref) for the per-test breakdown.
- **Validation** ("are we solving the right equations?") — this page.
  Validation is comparison against external reference data (TM5, GCHP,
  observations, …) and is fundamentally less complete than
  verification.

If you only need the first — reproduction of a published algorithm
under your own forcing within tolerances documented in the test
suite — verification is solid (for the schemes that are in the
test matrix) and you can proceed. If you need cross-model
comparison or observational match, read the gaps below.

## What HAS been validated

### Synthetic-fixture suite (verification, comprehensive)

More than 100 files under `test/core/` run on pull requests and pushes to `main` via the `CI`
workflow, with no external-data dependency. `test/runtests.jl` discovers that
tier dynamically; `test/regridding/` is also part of the default CI baseline.
Anchor tables:

Hosted jobs are CPU-only. CUDA-gated comparisons skip without a functional GPU;
the separate opt-in CUDA diagnostics provide the hardware evidence below.
Passing hosted CI therefore does not certify CUDA, ROCm, or Metal execution. ROCm hardware checks are
opt-in and pending; see the [ROCm runbook](../development/rocm.md).

| Property | Test files | Status |
|---|---|---|
| Uniform tracer invariance under a synthetic flow (relative error `< 1e-6`) | `test/core/test_advection_kernels.jl` covers CPU for `Upwind` / `Slopes` / `PPM`; GPU uniform coverage is `Upwind`. `LinRoodPPMScheme` is not in this matrix. | green where exercised |
| Global mass conservation (gradient IC, 4 steps) | `test/core/test_advection_kernels.jl` (CPU+GPU), `test/core/test_cubed_sphere_advection.jl` | green |
| Cross-window replay closure | `test/core/test_replay_consistency.jl` | green |
| Cross-day continuity (synthetic GEOS C8 fixture) | `test/core/test_geos_cs_passthrough.jl` | green |
| GEOS native CS preprocessor end-to-end (synthetic fixture) | `test/core/test_geos_reader.jl`, `test/core/test_geos_cs_passthrough.jl`, `test/core/test_geos_convection.jl` | green |
| Conservative regrid mass closure | `test/regridding/test_conservation.jl`, `test/core/test_ll_to_cs_regrid_script.jl` (script-level tolerance `1e-6`) | green |
| CPU / GPU agreement (4 ULP for Upwind 1-step; 16 ULP for Slopes / PPM 4-step, F32 and F64) | CUDA-gated test sets in `test/core/test_advection_kernels.jl`; LinRood is outside this comparison matrix | green where exercised |
| Operator dispatch (palindrome ordering and no-op branches) | `test/core/test_transport_model_convection.jl`, `test/core/test_tm5_convection.jl`, `test/core/test_diffusion_palindrome_contract.jl` | green |

Total core-suite cases: thousands; CI breaks down pass/fail per file.

### Advection coverage in the September 2026 V100 experiments

These are numerical and performance checks on tofu's V100 using the C90 L66
ERA5 archive. They are not a matched ranking of advection algorithms or an
end-to-end comparison against another model.

The subsequent [L40S/V100 release checks](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/567bc96b/scripts/benchmarks/results/release_fp32_l40s_20260906/README.md)
passed all ten maintained GPU diagnostic files on L40S, including transporting
adjoints and output reductions. A matched Float32 C90/L66 full-physics day with
six or 32 tracers produced exactly matching current-code L40S/V100 column means.
Those results describe CUDA. A separate Apple M5 Pro smoke test is recorded below.

### Apple Metal forward verification

The user-provided September 6 logs from an **Apple M5 Pro with 20 GPU cores**
show both six- and 32-tracer runs passing on source `567bc96b`. The bundle uses
Float32 C90/L66 meteorology for two hours (two windows), standard PPM, exact
TM5 Dkg diffusion, full-column collaborative TM5 convection and column output.
It verifies `MtlArray` state with scalar indexing disabled, finite output, two
completed snapshots, mass conservation and agreement with bundled CUDA output.

| Tracers | Warmed elapsed time | Maximum relative mass drift | Maximum column relative L2 difference from CUDA |
|---|---|---|---|
| 6 | 2.899 s | `5.4687e-8` | `8.8808e-8` |
| 32 | 8.279 s | `5.6299e-8` | `8.9079e-8` |

The environment was Julia 1.12.6, Metal.jl 1.10.3 and KernelAbstractions 0.9.42
on macOS 26.5.2. Times are single warmed runs including setup and output;
they are not repeated benchmark medians. Transport executes in Float32;
output totals and column accumulation use Float64 host slabs. This establishes
the tested forward path on Apple hardware, while Metal adjoints and broader
operator coverage remain open.

### Scheme-specific coverage

| Algorithm | Measured coverage | Evidence |
|---|---|---|
| Standard split `PPMScheme(MonotoneLimiter())` (`scheme="ppm"`) | Full-day performance and conservation; 7- and 31-day conservation in Float32/Float64. The month uses six tracers; the week also checks 32 in Float32. | [31-day results](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/9fd853822a9322403c55316c08c47cf04d647f83/scripts/benchmarks/results/main_monthly_mass_v100_20260906/README.md), [7-day results](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/9fd853822a9322403c55316c08c47cf04d647f83/scripts/benchmarks/results/main_weekly_mass_v100_20260906/README.md) |
| `LinRoodPPMScheme(7)` (`scheme="linrood", ppm_order=7`) | Six-tracer full-day Float32/Float64 seam experiment plus a matched four-month CATRINE comparison against split PPM. | [Lin–Rood seam experiment](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/9fd853822a9322403c55316c08c47cf04d647f83/scripts/benchmarks/results/main_mass_seams_v100_20260905/README.md), `docs/memos/2026-10-07_v0.5_transport_fidelity_backlog.md` |
| `LinRoodPPMScheme(5)` | Kernel/adjoint checks plus the matched four-month CATRINE comparison against split PPM and LR7. | `test/core/test_linrood_kernel_adjoints.jl`, `docs/memos/2026-10-07_v0.5_transport_fidelity_backlog.md` |
| Unlimited standard PPM (`PPMScheme(NoLimiter())`) | Focused seam/adjoint checks, outside the long real-input comparisons. | `test/core/test_cs_seam_exchange.jl`, `test/diagnostic/test_cs_seam_exchange_gpu.jl` |

The 31-day maximum daily relative mass drift, `5.84e-7` in Float32 and
`1.98e-16` in Float64, applies to **standard monotone PPM with TM5 convection
and conservative Dkg diffusion**. It is not a bound for Lin–Rood or all limiter
choices. Small negative column means occur even in the standard PPM run;
conservation and field boundedness are separate checks.

The proposed A100 comparison of standard monotone PPM, LR5, and LR7 is closed
as superseded by the completed four-month CATRINE five-variant comparison. It
holds the forcing and physics fixed and reports field errors against GCHP; the
full results and configuration differences are in
`docs/memos/2026-10-07_v0.5_transport_fidelity_backlog.md`. Standard PPM and
Lin–Rood differ vertically as well as horizontally unless the FV3 vertical
profile is selected, so field differences apply to the complete configured
algorithm. See [Advection schemes](@ref) for dispatch and order definitions.

### Real-data preprocessor smoke tests (verification with real input)

| Path | What was verified | Status |
|---|---|---|
| ERA5 spectral → LL 72×37 F32, Dec 2021 | preprocessor closes write-time replay gate; runtime steps cleanly; conservation tested via uniform IC | green (proven on disk) |
| ERA5 spectral → LL 144×73 F32, Dec 2021 | same | green |
| ERA5 spectral → CS C24 F32, Dec 2021 | same; F32-CS path requires the `f3b3abf` fix to spectral_synthesis.jl | green (post-`f3b3abf`) |
| ERA5 spectral → CS C90 F32, Dec 2021 | same | green (post-`f3b3abf`) |
| GEOS-IT C180 → CS C180 F64 native | preprocessor closes write-time replay gate; binary loads cleanly via `inspect_transport_binary.jl`; per-window snapshot output verified. The 2026-04-25 unified-chain validation flagged a runtime GPU step-1 blocker on the unmerged-vertical C180 binary; the production response targets the `merge_above_pressure = 0.25 hPa` 64-level product with adaptive substeps. Status against that product is tracked in current Catrine config notes. | preprocessor green; current C180 runtime path uses merged + adaptive-substep binaries |

### Model parity (TM5)

| What | Tests | Status |
|---|---|---|
| ERA5 → TM5 four-field conversion (`entu/detu/entd/detd`, `ec2tm_from_rates!`) against TM5's `tmm.F90` packing, `ECconv_to_TMconv` and `meteo.F90` closure | `test/core/test_tm5_preprocessing_rates.jl` and `test/core/test_era5_c180_tm5_convection.jl` against an independent transcription (`test/helpers/tm5_ecconv_reference.jl`) | **green since the issue-#54 fix**: bit for bit on an exactly representable column, to rounding on 500 random deep columns, input fluxes recovered at every interface. Before the fix the conversion placed ERA5 level k one interface low and dropped the cloud-top layer; the tests then compared against a copy of that loop, so the earlier "green" did not check the convention |
| TM5 convection operator and runtime coupling | `test/core/test_tm5_preprocessing.jl`, `test/core/test_tm5_vs_cmfmc_parity.jl`, `test/core/test_tm5_driven_simulation.jl` | green |
| `SlopesScheme` vs TM5's slopes advection (`dynamu` in `advectx.F90`) | `test/core/test_slopes_tm5_reference.jl` against an independent transcription of `dynamu` | **differs**: same Russell–Lerner face flux (TM5 with its moment re-diagnosed by minmod reproduces `SlopesScheme` to rounding), but TM5 carries prognostic moments and `SlopesScheme` does not; `SlopesScheme` is markedly more diffusive (L2 error 4.6–6.8 times TM5's after 10 revolutions of a 1-D Gaussian at Courant numbers 0.05–0.75) |

The ERA5 → TM5 conversion is checked against an independent transcription of
TM5's own routines; the earlier tests compared it with a copy of the code under
test and so missed the index errors of issue #54 (see
[ERA5 → TM5 fields](@ref tm5-ec2tm)). No advection option reproduces TM5's
slopes advection; a prognostic-moment scheme is postponed (see
[Advection schemes](@ref)).

## Remaining end-to-end validation gaps

The following gaps remain even where production comparisons now exist:

| Gap | Why it matters | Status |
|---|---|---|
| **GCHP parity for full-physics CS runs** | A 25-month CATRINE C90 comparison and controlled four-month transport comparisons on MERRA-2 forcing are complete and documented. Residual polar differences remain, and no automated multi-month regression is maintained. | production comparison complete; regression automation open |
| **CATRINE D7.1 intercomparison** | Four-tracer (CO2, fossil CO2, SF6, 222Rn) full-physics production runs span 25 months; the four-month controlled comparison covers five transport variants and documents their differences against GCHP in `docs/memos/2026-10-07_v0.5_transport_fidelity_backlog.md`. | multi-month comparison documented; maintained end-to-end regression not wired |
| **Observational closure** | Comparison of model output (column CO2, surface SF6 etc.) against an observational network (NOAA in-situ + TCCON / OCO satellite) | not started |
| **Multi-month GPU production runs** | CATRINE production runs span 25 months and controlled scheme comparisons span four months; CI still has no multi-week real-data GPU regression. | production evidence exists; maintained regression open |
| **Adjoint kernels** | See [Adjoint status](@ref). The `AdjointTape` (device, pinned-host and mmap storage; full and stride checkpointing), the supported advection/convection and halo reverse paths, and the 4D-Var driver are on CI. The 4D-Var gradient takes one forward run, which samples the observations and records the tape, and one reverse pass for all observations. Gaps: optimized/clamped convection variants, logarithmic-memory (binomial Revolve) checkpointing, vertical observation binding, and TM5-4DVAR cross-validation. | partial (shipped) |

## Floating-point tolerance practice

Tolerances vary by operation; the canonical sources:

| Operation | F64 tolerance | F32 tolerance | Reference |
|---|---|---|---|
| Per-window replay gate (per cell, relative to the summed budget-term magnitudes) | `1e-12` | `1e-5` | `src/MetDrivers/ReplayContinuity.jl::replay_tolerance(FT)` |
| Window-continuity verification (test variant) | `1e-12` | `1e-6` | `test/core/test_replay_consistency.jl` |
| Per-step uniform-tracer invariance (relative) | `1e-6` | `1e-6` | `test/core/test_advection_kernels.jl` |
| 4-step total mass conservation (gradient IC, structured grid) | `1e-12` | `5e-5` | `test/core/test_advection_kernels.jl` |
| CPU/GPU advection agreement | `4 * eps(FT)` (Upwind 1-step) / `16 * eps(FT)` (Upwind / Slopes / PPM 4-step) | same as F64 column | CUDA-gated test sets in `test/core/test_advection_kernels.jl`; LinRood is not in this matrix. |
| Conservative regrid mass closure (script-level acceptance) | `≤ 1e-6` rel | same | `test/core/test_ll_to_cs_regrid_script.jl` |
| Cross-day GEOS chain continuity | machine epsilon (`5.94e-16` F64 measured) | `~3.5e-7` F32 measured | preprocessor stdout from `process_day` |

The F64 tolerances reflect double-precision noise floors at production
resolutions; F32 tolerances reflect single-precision accumulation.
Production runs on the L40S GPU use F32 by default — the F32 noise
floor is the operational tolerance.

## What this means for users

If you are doing:

- **Advection algorithm research** → verification is solid, F32 / F64
  noise-floor agreement is well-tested. Proceed.
- **CO2 intercomparison studies that need GCHP-equivalent fidelity** →
  the underlying operators are TM5-faithful or GCHP-style, except that no
  advection option reproduces TM5's prognostic-slopes scheme. The
  end-to-end intercomparison report has not been written. Run a
  side-by-side and compare yourself;
  `scripts/diagnostics/catrine_compare_vs_geoschem.py` is the starting point.
- **Inverse modelling that needs an adjoint** → the adjoint and
  4D-Var stack ship on CS. See [Adjoint status](@ref) for the supported
  scheme matrix and the remaining gaps (optimized/clamped convection
  variants, column-mean observation binding, and TM5-4DVAR
  cross-validation).
- **Validation against observations** → not in scope today; the
  forward model has the fidelity, but the observation-comparison
  diagnostics are external.

## Where to read next

- [Adjoint status](@ref) — what the README claims vs what actually
  ships.
- [Conservation budgets](@ref) — the explicit `@test` assertions
  that anchor the verification claims above.
- [TOML schema](@ref) — configuration reference for the runs that drive
  the validation work above.
