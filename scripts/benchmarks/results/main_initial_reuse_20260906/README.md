# Reuse private cubed-sphere initialization buffers

> The reproducer scripts `check_outputs.jl`, `initialization.jl` and `profile.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_initial_reuse_20260906/<file>`.

The driven runner now reuses one tuple of interior VMR arrays between analytic
initializers, then copies each result into its independent packed state slot.
The public allocating builder still owns fresh arrays. Native/file builders
may return replacement arrays to the runner; no public workspace API is added.

## Actual C90 L66 initialization

`initialization.jl` loads the actual dry format-4 ERA5 input used by the GPU
experiments, probes shapes/units, and compares the preceding allocating loop
with the reused-buffer loop on identical pressure-layer configurations. It
warms both paths and interleaves their order across five measured repetitions.
Both packed tracer arrays and tracer ordering match exactly.

| Tracers | Previous cumulative allocation | Reused buffers | Saved |
|---|---:|---:|---:|
| 6 | 166,917,840 bytes | 102,762,560 bytes | 64,155,280 bytes |
| 32 | 890,246,216 bytes | 492,483,704 bytes | 397,762,512 bytes |

The remaining allocation includes the final packed state, one interior VMR
tuple, and per-tracer pressure-layer selection arrays. Values are cumulative
allocation, not peak memory. Median CPU setup times in this sample are
0.0811 → 0.0680 s for six tracers and 0.4214 → 0.3792 s for 32; other CPU
work may overlap, so these times are descriptive rather than an isolated
performance claim. Allocation savings and field identity are the primary result.

The probe uses Float32 `(96,96,66)` halo-padded air panels, `(90,90)` surface
pressure panels, and 66 vertical layers. First-interior level-33 air mass is
2.829608e12 kg and surface pressure is 100869.164 Pa. Each pressure-layer
tracer targets 1e35 molecules. `initialization.toml` retains every sample.

## Full-day production check

The same 32-tracer V100 day drops from 6.414270756 to 6.016210056 GB cumulative
host allocation (about 398 MB saved). Median whole-run time is 29.238 s versus
29.543 s before; the small timing difference is not a statistically established
speedup. All 150 output arrays across both measured samples are identical;
454 output/completeness checks pass. One warmup precedes two measured runs.
The change reduces allocation and preserves transport output, including the
3.14e-7 maximum final relative mass drift. Peak memory is not measured in this
comparison.

## Validation and review

- Initial-condition and packing tests: 558 packing, 934 independent pressure-layer,
  and 133 initial-condition I/O assertions pass.
- New ownership checks: 54 mixed analytic and 14 signed native-file assertions.
- Julia 1.10.12: all 626 packing/ownership and 934 pressure-layer checks pass.
- Aqua 10 and JET 152 against unchanged 152 pass.
- Complete CPU suite: 120 core files plus 628 regridding assertions, exit 0.
- Strict documentation build: exit 0; all 160 local link/map checks pass.

Critical Codex self-review checked buffer ownership, unchanged public allocation,
full overwrites between pressure/uniform/latitude/Gaussian modes, signed native
values, independent packed slots and input arrays, dictionary-dependent tracer
ordering, shape checks before reuse writes, and preservation of the pressure
selection/normalization arithmetic. Comments in the touched runner setup now
state storage and basis behavior rather than obsolete extraction references.
No extra agent was used.

The baseline is `5db7c780` on `revamp/current-main`. CPU measurements use Julia
1.12.6, four Julia threads, one OpenBLAS thread, and `CUDA_VISIBLE_DEVICES=`.
The temporary Julia 1.10 environment develops this same worktree. No repository
dependency files are changed. The GPU candidate export is
`/tmp/atmos-initial-reuse` on tofu, retaining CUDA.jl 5.11.3/runtime 12.6.
`profile.jl` and `check_outputs.jl` reproduce the separate full-day comparison;
run with the authorized V100 UUID and environment from the
[parallel Dkg experiment](../main_dkg_parallel_v100_20260906/README.md).
