# Release notes

## Unreleased

### Runtime and output

- New `[output.observations]` contract for sampling tracer profiles at
  observation points (OCO-2 Lite soundings, NOAA ObsPack sites, generic point
  tables) from the containing model cell at met-window ends. The table is
  parsed by `observation_output_spec` and checked by `validate_config`, which
  rejects unknown keys and requires an absolute run origin. Both runners
  sample at t = 0 and at every met-window end: soundings are blended linearly
  between the two bracketing window ends (or taken from the nearest), sites
  get the intake-layer value from hypsometric heights. Output goes to
  append-only `_soundings` / `_sites` NetCDF files (see the output schema).
  Gridded snapshot output is unchanged.
- Point events (satellite soundings, ObsPack records, station time lists)
  carry an intake height and get `<tracer>_intake` from the layer containing
  it; station tables choose `EveryWindow`, `TimeRange`, or `TimeList`
  schedules per row and accept `altitude` with `elevation`; repeated site ids
  merge their time lists. OCO sources take a typed `quality_filter`
  (`"flag_max"`, `"flag_values"` for categorical flags such as the MIP
  `assimilate_flag`, or `"none"` to co-sample every record). The editor schema documents every choice with its Julia
  type and is checked against the parser by a test; site tables have their own
  schema (`schemas/observation_sites.schema.json`).
- User guide `docs/memos/2026-10-02_observation_sampling_guide.md` with a
  runnable OCO-2 MIP + TCCON example
  (`config/examples/observation_sampling_oco2mip.toml`, site table
  `config/examples/tccon_ggg2020_sites.csv`).
- All runtime NetCDF writes, including the background daily snapshot task,
  share one lock (`with_netcdf_lock`); netcdf-c is not thread-safe.
  Observation rows queue while the lock is busy instead of stalling the run.
- Daily output files (snapshots and observations) now use the day index when
  a binary name carries no date; previously every day wrote the same file.
  Two binaries that still resolve to one daily file fail before transport.
- `[output.fields].tracers = "name"` (a single string) no longer throws a
  `MethodError`; it selects that one tracer as documented.

### Verification and remaining limits

- New core tests cover config parsing, cell location on every topology
  against independent references, source readers (including a real OCO-2
  Lite file check), the device gather, pressure and height
  reconstruction on GEOS L72, the sampler's time bookkeeping, and end-to-end
  lat-lon and cubed-sphere runs, including identical results across a
  binary handoff. An opt-in CUDA test covers the gather on an L40S.
- Station placement uses a constant temperature except on cubed-sphere
  binaries with GCHP VDIFF temperatures. Averaging kernels are applied
  offline.

## 0.4.0 — 2026-09-06

This release changes cubed-sphere numerical results and runtime/output
behavior. Keep the version, forcing binary, run TOML and dependency manifest
with reproducibility records; recheck conservation and scientific diagnostics
when upgrading from 0.3.0.

### Numerical changes

- Cubed-sphere split advection and Lin–Rood now share conservative face
  transfers across panel seams. Previous runs could accumulate tracer mass
  drift at those seams.
- Precomputed TM5 Dkg diffusion advances conservative tracer storage with
  bidiagonal factors. It preserves weak exchange into empty layers, isolated
  layers and signed totals more accurately. Its adjoint uses the matching
  transpose. This is a numerical correction, so old and new output need not
  match bit for bit.
- CUDA PPM launch tiles, separate Dkg factor/tracer launches and batched
  convection solves reduce work without lowering configured precision.
  CUDA collaborative Float64 convection supports full columns through 73
  levels within its shared-memory budget; Float32 retains its portable depth
  limit. Explicit `lmax_conv`/`n_merge` choices remain numerical approximations.
- Lin–Rood recording and reverse halo propagation use backend kernels and
  pass transporting CUDA adjoint checks with scalar indexing disabled.

### Runtime and output

- Input drivers and device workspaces are reused across daily files. Rolling
  staging owns its cache and checks retained source identity; failed prefetch
  and output tasks are consumed during cleanup.
- Single-file NetCDF output streams selected snapshots and records
  `completed_snapshots`. A failed run may leave partial output; file existence
  is not evidence of completion. Reopening a stream to resume is unsupported.
- Selected NetCDF output avoids retaining full tracer volumes on the host.
  Signed Float64 `<tracer>_total_mass` diagnostics are captured independently
  of spatial output precision. Metal uses bounded host slabs for Float64
  totals and column accumulation; CUDA retains device Float64 reductions.
- `SnapshotFrame` and `SnapshotWriteOptions` are part of the curated top-level
  API. Write and resource-close failures are preserved together rather than
  allowing one to mask the other.
- Runtime preflight reports malformed tables, including `[input.staging]`
  and tracer subtables, before opening binaries. Window indices must be
  integers. Multiple input files require their complete window ranges;
  omit `stop_window` for a full multi-day run. Nine obsolete 48-hour configs
  now live under `config/runs/likely_legacy/`.

### Dependencies and documentation

- Add explicit `FileWatching` and `HDF5_jll` dependencies for staging ownership
  and NetCDF error-handler handling. HDF5_jll compatibility accepts 1.14 and 2.
- Remove self-version bounds from test and benchmark environments so release
  bumps do not invalidate them. Julia 1.10 remains the minimum supported version.
- Add an executed emission-footprint tutorial with a finite-difference check
  and a beginner's guide to adjoints, observations, priors and inversion.
  Correct preprocessing configuration/balance explanations and distinguish
  hosted CPU CI from separate GPU verification.

### Verification and remaining limits

The branch records Julia 1.10/1.12 CPU test runs, strict documentation builds,
all ten maintained L40S GPU diagnostics, V100 adjoint/output checks, and
real-input Float32 C90/L66 comparisons. The matched L40S day benchmark improves
from 7.35 to 4.54 seconds for six tracers and from 47.47 to 13.84 seconds for
32 tracers. The 32-tracer 0.3.0 baseline uses its supported legacy solver,
because that release's collaborative solver is capped at six tracers. These
measurements include runtime setup and I/O and apply to the documented workload;
see [benchmark evidence](scripts/benchmarks/results/release_fp32_l40s_20260906/README.md).

The C90/L66 Float32 forward smoke test passed on an Apple M5 Pro (20 GPU
cores), with six and 32 tracers, full TM5 collaborative convection, Dkg
diffusion and column output. Warmed runs took 2.90 and 8.28 seconds; maximum
column relative L2 differences from CUDA were below `9e-8` and relative mass
drift below `5.7e-8`. See the [Metal verification record](docs/memos/release_readiness_20260906.md).
This does not establish coverage of Metal adjoints or every operator. Optimized,
clamped or reduced-column convection paths do not all have supported adjoints.
The shipped inversion CLI remains synthetic-only; real-data inversion assembly
and external TM5-4DVAR cross-validation are separate tasks. Numerical checks
and performance measurements are not a complete observational validation.

## 0.3.0

The preceding release is the baseline for the changes above. This changelog
was introduced for 0.4.0; it does not reconstruct earlier release histories.
