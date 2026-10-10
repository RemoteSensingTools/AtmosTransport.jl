# Conservative precomputed Dkg diffusion

> The reproducer scripts `ablation.jl`, `check_outputs.jl`, `check_profile_outputs.jl`, `check_totals.jl`, `column_probe.jl`, `factorization_probe.jl`, `full_day.jl`, `production_columns.jl`, `profile.jl` and `weak_exchange.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_dkg_mass_v100_20260905/<file>`.

Solving the implicit diffusion equation directly in tracer mass reduces the
remaining Float32 drift after paired advection seams. It preserves weak physical
transfers and uses no normalization of column or global totals. The
[implementation report](../../../../docs/memos/2026-09-05_main_dkg_mass.md)
records the derivation, validation, review, and limitations.

## Full-day results

The baseline is `5cb5a7f9` (paired split seams). Both versions use the same
C90 L66 ERA5 archive, PPM, TM5 convection, precomputed Dkg diffusion, no emissions,
and `preserve_tracer_mass` air resets. Six pressure-layer tracers each target
1e35 molecules; all 25 requested hourly snapshots are complete.

| Maximum absolute relative total drift | Baseline | Conservative Dkg |
|---|---:|---:|
| Six tracers, Float32, final | 8.168654720603056e-7 | 3.138139946426091e-7 |
| Six tracers, Float64, final | 9.914172177906698e-16 | 0 |
| Six tracers, Float64, all hours | 9.914172177906698e-16 | 1.9828344355813393e-16 |
| 32 tracers, Float32, final | 8.203781159418402e-7 | 3.138139946426091e-7 |

Float32 improves another 2.6-fold beyond the seam fix (about 71-fold versus the
original six-tracer PPM result). Float64 compensated final totals happen to
match initialization exactly in this run; that is not a universal guarantee.
Every final Float32 column-mean field is closer in relative L2 to the Float64
run: baseline errors 1.43e-6–2.04e-6 become 1.12e-6–1.37e-6. Float64 uses legacy
convection while Float32 uses collaborative LU; this is an end-to-end comparison,
not an isolated diffusion error estimate. Small negative column means from
transport remain (about -4.27e-11 mol/mol).

The exploratory full-day target of 1e-7 **was not met**. Its four failures are
retained in `exploratory_target_failure.txt`. An intermediate algorithm reached
9.75e-8 but erased weak transfers and was rejected. The final archived validators
check completeness, finite values, identical initial totals, and reduction of
the worst drift across the same workload; they explicitly report the unmet
target. No maintained numerical test tolerance was relaxed. Adoption rests on
the independent field/weak-transfer checks as well as improved totals.

## Attribution and independent physics checks

Ablation on the seam-only baseline gives maximum final Float32 drift:

| Enabled processes | Drift |
|---|---:|
| Advection, convection, diffusion | 8.168655e-7 |
| Advection and convection | 5.047727e-9 |
| Advection and diffusion | 8.789195e-7 |
| Advection only | 4.823358e-9 |

This identifies diffusion as the main remaining source on this archive. The
unchanged public VMR Thomas solver provides a comparison, and an independently
assembled Float64 tridiagonal mass matrix provides the column reference.
`production_columns.jl` samples 216 actual meteorological columns with four
layer impulses and exchange ratios up to about 39.9 at dt=360 s:

| Float32 column measure | VMR Thomas | Conservative mass |
|---|---:|---:|
| Worst relative mass drift | 1.231886e-6 | 1.028200e-7 |
| Worst relative field L2 error versus independent Float64 solve | 1.067056e-6 | 1.458508e-7 |

Maintained tests also cover signed profiles, uniform backgrounds, isolated
layers, closed interfaces, zero carriers, and stiffness up to 1e4. Analytic
weak-transfer checks reach dt*D/m=1e-14 and check recipient-relative errors and
adjoint sensitivities. For a 4e6 kg donor, empty recipient, 1e10 kg equal air
masses, and 100 kg/s exchange over one second, Float32 transfers 0.04 kg. The
rejected retention-subtraction candidate transferred zero. A Float32 donor at
4e6 kg has spacing 0.25 kg: ordinary state storage cannot represent every debit
while also preserving such small recipient amounts. Lower drift alone cannot
justify deleting the exchange.

## Runtime and memory

The 32-tracer workload uses 255 substeps over all 24 windows, with column means
and compensated totals at hours 0 and 24. One warmup precedes two measured runs.

| Measure | Paired-seam baseline | Conservative Dkg |
|---|---:|---:|
| Median whole-run seconds | 34.5738578205 | 38.6349346815 |
| Measured seconds | 34.383–34.765 | 38.549–38.721 |
| Cumulative host allocation | 6.419 GB | 6.414 GB |

The added arithmetic costs about 11.7% whole-run time on this V100. No persistent
workspace is added; the existing per-column factor buffer is reused for every
tracer. No new Float64 arithmetic is introduced into Float32 kernels. Peak
host/device memory was not measured. Runs are sequential with warm caches,
not interleaved statistical estimates. Timer sections can nest and include
waits, so their times cannot be summed; CSV allocation zeroes are unmeasured.
Profile TOML `final_totals` are ordinary reductions; the conservation values
above use compensated NetCDF totals in `profile_checks.txt`.

## Validation and archived files

- Complete CPU suite: 120 core files and 628 regridding assertions, exit 0.
- Focused CPU diffusion: 12,087 assertions on both Julia 1.12.6 and 1.10.12.
- V100 diffusion: 6,989,800 assertions through 65 tracers, both precisions,
  independent solves, scalar/packed identity, transpose and weak exchanges.
- Full-day output checks: 140 six-tracer and 369 32-tracer assertions.
- Strict documentation build and 160 documentation/link/map checks pass.
- Aqua: 10 assertions. JET: 152 reports versus 148 before; four additional
  known KernelAbstractions launch/propagated-return reports for two new kernels.

`after/` stores final TOMLs and timing CSVs; baseline data remain in
[`../main_split_mass_seams_v100_20260905/`](../main_split_mass_seams_v100_20260905/README.md).
Run `check_totals.jl` to recheck the small archived TOMLs without a GPU or
meteorological input. Large NetCDFs remain on tofu. `production_columns.txt`, `cpu_checks.txt`,
`gpu_checks.txt`, `output_checks.txt`, and `profile_checks.txt` contain final
results. `column_probe.*` and `factorization_probe.*` are historical derivation
probes, **not the final production algorithm**; they do not validate weak
transfers. `weak_exchange.*` records the production weak-transfer probe.

## Reproduction

Use an isolated checkout with the same input described in the
[seam experiment](../main_split_mass_seams_v100_20260905/README.md):
3,114,263,552 bytes, dry format 4, explicit dm, C90 L66. The first interior
level-33 air mass is 2.829608e12 kg in Float32 `(96,96,66)` panels; maximum Dkg
is 2.0023729e11 kg/s in `(90,90,66)` arrays. This archive holds experimental
three-hour convection against hourly transport; it is not forcing validation.

The measured export is `/tmp/atmos-conservative-dkg` on tofu, with Julia 1.12.6,
CUDA.jl 5.11.3 and CUDA runtime 12.6. Preserve those environment pins on V100.
GPU 0 is Tesla V100-PCIE-16GB, UUID below; GPU scalar indexing is disabled.
Run from the configured checkout, using an explicit Julia binary if necessary:

```bash
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
julia --startup-file=no --project=. /path/to/full_day.jl after

ATMOSTR_TIMERS=1 \
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
julia --startup-file=no --project=. /path/to/profile.jl after

ATMOSTR_RUN_DKG_GPU_TESTS=1 ATMOSTR_DKG_GPU_NAME=V100 \
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
julia --startup-file=no --project=. test/diagnostic/test_conservative_dkg_gpu.jl
```

`full_day.jl` writes `/tmp/atmos-dkg-drift-after/` and accepts `ATMOSTR_MASS_INPUT`
as an archive override. `profile.jl` writes `/tmp/atmos-dkg-day-profile-after/`.
The NetCDF validators use those paths and the preceding split experiment's
paths. Run validators, `production_columns.jl`, and core tests with GPUs hidden
(`CUDA_VISIBLE_DEVICES=`). No Metal or A100 validation was performed here.
