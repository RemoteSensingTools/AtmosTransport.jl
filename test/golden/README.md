# Golden outputs

> A full `check` writes about 25 GB of case outputs into `<new>`; put it on a
> large local disk and delete it once the comparison is recorded.

A refactor must not change results. `run_goldens.jl` runs the production entry
points (`scripts/preprocessing/preprocess_transport_binary.jl` and
`scripts/run_transport.jl`) on the fixed cases of [`cases.toml`](cases.toml) and
compares every output with a recorded reference, bit for bit. A change that is
meant to alter results states its deltas from the same comparison.

## Usage

```bash
# Record the reference at the commit that defines it.
julia --project=. test/golden/run_goldens.jl record /temp1/$USER/goldens/ref_<commit>

# After a change: run the cases again into a new directory and compare.
julia --project=. test/golden/run_goldens.jl check \
    /temp1/$USER/goldens/ref_<commit> /temp1/$USER/goldens/<label>

# Compare two existing trees.
julia --project=. test/golden/run_goldens.jl compare <ref> <new>
```

`check` prints one line per case, `identical` or `DIFFERS` with the number of
differing values and the largest difference per variable, the wall times of
both runs (see [Wall times](#wall-times)), and notes on anything besides the
code that differed between the runs (the filled config, Julia, the Manifest,
threads, GPU, input files). It exits with status 1 if a case differs, fails or
wrote nothing to compare. Runtime cases of a check read the reference's
preprocessed binaries, so a runtime difference is never a preprocessing
difference.

Select cases with `--cases=a,b` (whatever their tags), `--tags=run,gpu`
(cases carrying all of these) or `--skip-tags=slow`. The cases run in three
[worker sessions](#sessions) (preprocessing, CPU runs, GPU runs). On wurst
(L40S) a full check of the 28 cases takes about 42 minutes (about 71 with one
process per case): preprocessing 17, CPU runs 16 and GPU runs 8 minutes. The
preprocessing cases need 35 GB of memory (L117). GPU cases run on the visible
CUDA device (`CUDA_VISIBLE_DEVICES`); `--threads=N` sets the Julia threads of
each session (default 8) and `--timeout=MIN` the minutes after which a hung
case is stopped (default 45, about six times the slowest case; see
[Sessions](#sessions)). Unknown options and tags selecting nothing are
errors.

A case never overwrites: every selected case directory must be new. To
re-record one case, delete its directory in the reference first
(`rm -r <ref>/<case>`), then `record <ref> --cases=<case>`. Each case's
`status.toml` is written as soon as the case finishes, so an interrupted run
keeps the status of its finished cases; the cases it had not finished have a
directory without `status.toml` (`compare` reports them as not run). To run
those again, delete their directories and run with `--cases=` naming them.

Compare only with a reference recorded on the same host, GPU model and
Manifest, by the same user: results are reproducible run to run on one machine,
not across hardware, and binary headers carry expanded source paths. Before a
reference is used, run `check` against it once from the same commit: every
case must come back identical (the harness does not enforce this repeat run).

## What is compared

- **NetCDF** (`*.nc`): every variable's element type, dimensions and stored
  values, floats bit for bit (a NaN payload or the sign of a zero counts), and
  every attribute with its type, except the provenance attributes `creation_date`,
  `framework_commit`, `framework_dirty`, `runtime`, `hostname`, `user` and
  `history`.
- **Transport binaries** (`*.bin`): the JSON header (its key set and values)
  without the provenance keys
  `git_commit`, `git_dirty`, `creation_time`, `script_path`,
  `script_mtime_unix`, `generation_fingerprint` and `dirty_nonce`, and every
  payload bit. Payload differences are reported in the binary's float type.
- **Other files**: byte for byte. `config.toml` (the filled case config),
  `log.txt` and `status.toml` at the top of a case directory are bookkeeping,
  and directories starting with `_` are caches.

Attributes such as `grid` and the binary's `horizontal_topology` print Julia
type names: a refactor that renames a type changes them, which is a stated
attribute delta of that commit.

Each case runs with:

- **Regridding weights recomputed** in the case's `_regrid_cache` (the
  preprocessor's `regridder_cache_dir` and, for runtime sources,
  `ATMOSTR_REGRID_CACHE_DIR`), so a change to the weights shows up in the
  outputs instead of being hidden by a stale cache.
- **No `ATMOSTR_*` or `ERA5_N320_PROFILE` switches** inherited from the
  parent environment (they can change what a run does or writes); the
  scrubbed names are listed in `status.toml`.

`status.toml` records whether the case succeeded (`ok`, and the `error` of a
failure; the log has the backtrace), its wall time and when it finished, its
session (`session.label`, its `position` in the session, the number of `cases`
the session's worker started with, what the worker had to `restore` after
the case, and `poisoned` if freeing memory after the case failed), the commit and whether `src`, `ext`, `scripts`, `config`,
`Project.toml` or `test/golden` had local changes, the run environment (Julia,
Manifest hash, threads, GPU), the size and modification time of every input
file the config names (directories such as the raw met archives are not
fingerprinted), and any non-finite values in the NetCDF outputs (recorded and
warned about, not an error).

## Sessions

Starting Julia, loading the package and compiling the code a case needs take
much of a case's time (a C90 GPU day steps in about 5 s once compiled), so the
cases share Julia processes. `run_goldens.jl` groups the selected cases by what
a process fixes when it starts:

- the entry point (`kind`: preprocessing or runtime);
- the filled config's `[architecture]` table, which decides whether CUDA (or
  Metal) is loaded, compared as written;
- the Julia threads (`--threads`).

The environment is otherwise the same for every case. With the current cases
that makes three sessions: the preprocessing cases, the CPU runs and the GPU
runs. Sessions run one after the other, in the order of their first case in
`cases.toml`, and each keeps the case order. A case that reads another case's
outputs (`@CASE:<name>@`) must run after it: `record` checks this before it
runs anything (the runtime cases of a `check` read the reference's outputs).

Each session is one [`golden_worker.jl`](golden_worker.jl) process. It first
includes the entry scripts its cases use, each into its own module, which
loads AtmosTransport and its dependencies once: package initialisation, such as
the `ECCODES_*` variables set when GRIB loads, belongs to the session as it
belonged to each process before. For each case it then calls the script's
`main(args)`, the function the script runs from the command line, and

- sends stdout, stderr (also of C libraries) and the logging system to the
  case's `log.txt`;
- sets `ATMOSTR_REGRID_CACHE_DIR` to the case's `_regrid_cache` and runs the
  case in the repository directory;
- on an exception, prints it with its backtrace to the log, records the case
  as failed and goes on with the next case;
- afterwards restores the global logger, the working directory and `ENV`
  (anything the case changed is undone and listed in `session.restored`), runs
  a full garbage collection and, when CUDA is loaded, returns its cached device
  memory (`CUDA.reclaim()`).

After every case the worker rewrites a results file, and before a case it
names the running one. `run_goldens.jl` starts the worker without waiting for
it and reads that file every few seconds: it writes each case's `status.toml`
as soon as the case is reported finished, and it knows what the worker is
doing when it ends or hangs.

- **Crash.** If the worker dies during a case (a crash, an `exit`, the OOM
  killer), that case fails with the exit code or signal, and the remaining
  cases continue in a new worker; so do they if it dies between cases. If it
  dies before its first case (while loading the entry scripts), the remaining
  cases fail with `session.position = 0`, since a new worker would die the
  same way.
- **Timeout.** If a case runs longer than `--timeout` minutes (default 45),
  the harness stops the worker (SIGTERM, on which Julia prints a backtrace of
  its main thread into the case's `log.txt`; SIGKILL 30 s later if it is
  still alive), the case fails with `the case ran longer than the timeout`,
  and the remaining cases continue in a new worker. The same limit applies to
  loading the entry scripts (then the remaining cases fail) and to the cleanup
  after a case: that case was already reported, so it keeps its result, and
  the remaining cases continue in a new worker.
- **Sticky CUDA error.** A device error such as an illegal address leaves the
  CUDA context unusable: every later CUDA call of the process fails. If the
  `synchronize` or `reclaim` after a case throws, the worker marks the session
  poisoned (in the results file and in that case's `session.poisoned`; the
  case's own outcome stands), logs the error with its backtrace and exits with
  code 75, and the remaining cases continue in a new worker. A worker that
  does not exit within 30 s of reporting this is stopped.

The summary printed at the end has one line per worker; a worker that did not
end normally (exit status 0 after reporting every case it was given) carries a
note such as `worker exited with 3 during <case>`,
`worker was stopped after the timeout (45 min) during <case>` or `session
poisoned after <case>`.

### Determinism

A session must write the outputs one process per case wrote, bit for bit; the
first full session check came back identical on all 27 cases against a
reference recorded with one process per case. State that cases share in a
process, and why it does not change results:

| State | Why results do not depend on it |
|---|---|
| Random numbers | Nothing in `src/` or the entry scripts draws any; a process is randomly seeded, so no reproducible golden can depend on them. |
| Regridding weights | Built or read from disk only (`build_regridder(...; cache_dir)` keeps no in-memory cache), in the case's directory: `ATMOSTR_REGRID_CACHE_DIR` is read when weights are built, and the preprocessing configs set `regridder_cache_dir`. |
| Other caches and module state | Spectral coefficient caches live on disk (`ATMOSTR_SPECTRAL_CACHE_DIR` is scrubbed). Module-level mutable state in `src/` is the section timer (off: `ATMOSTR_TIMERS` is scrubbed), the memo of the source revision (provenance, not compared) and two flags of the GEOS native preprocessor, set before every use. Preset level lists are copied before use. |
| FFTW plans | All `ESTIMATE` (no measured plans), so a plan does not depend on what was planned before. |
| CUDA memory pool | Returned to the driver between cases. Reused device memory could only matter to a kernel that reads memory it did not write, which the comparison would expose. |
| Logging | `[logging] level` applies to the run only (`with_log_level`), the entry scripts scope their console logger to `main`, and the worker restores the global logger; nothing calls `disable_logging`. |
| Environment | `src/` only reads variables; the worker restores `ENV` after each case anyway. |
| Working directory | `src/` never changes it; the cases run in the repository, as the processes did. |
| Threads | A session has the threads a case process had; the runner's re-execution with 2 threads happens only from the command line. Cases run on the main task, like a script, so `@threads :static` loops behave the same. |
| Background tasks | Window prefetch and input staging tasks finish before a run returns. |
| GPU package | The command line loads CUDA before AtmosTransport; in a session the first GPU case loads it through the library (`ensure_runtime!`), as for any caller that does not preload it. The methods are the same. |

### Wall times

`wall_seconds` of a session case is the case alone. The first cases of a
session pay compilation that later cases reuse, and the worker reports the time
it took to load the entry scripts separately, so a session wall time compares
only with the same case at the same position of a session. `check` prints the
wall times with how each run ran, for example
`wall 177 s (own process) → 12 s (session case 3 of 12)`, and never treats them
as a result; a failed case prints `failed` (its wall time is not known when
its worker died). At the end of the runs the harness prints each worker's wall
time, loading time and time in its cases.

The comparison ([`compare.jl`](compare.jl)) reads files with NCDatasets, JSON3
and Mmap, not through AtmosTransport, so a refactor of the readers cannot hide
a change.

## Cases

| Case | Exercises |
|---|---|
| `pre_ll72` | ERA5 spectral synthesis → 5° lat-lon, merged levels, dry-mass fix, Float64 |
| `pre_o24` | ERA5 spectral → octahedral reduced-Gaussian O24, humidity-aware dry-mass pin, Float64 |
| `pre_c24` | ERA5 spectral → C24 through a staging grid, Float64 |
| `pre_merra2_c90` | MERRA-2 → C90 L72 with GCHP's flux construction (vector fluxes, Cartesian regrid, FV3 filter), adaptive substeps, Float32 |
| `pre_geosit_c45` | GEOS-IT native C180 → C45 L72 block coarsening, GEOS panels, endpoint column balance with the source's default weights, adaptive substeps, Float32 |
| `pre_era5_n320_c90_l117` | ERA5 N320 → C90 L117, line-integral window-mean fluxes, TM5 diffusion fields (`slow`: about 10 min, 35 GB) |
| `run_ll72_*` | lat-lon PPM (CPU Float64), slopes (GPU Float32), upwind (CPU Float32) |
| `run_o24_upwind_cpu_f64` | reduced-Gaussian upwind (CPU Float64) |
| `run_c24_*` | cubed-sphere PPM (CPU Float64), Lin-Rood and upwind (CPU Float32), slopes (GPU Float32) |
| `run_c90_merra2_gpu_f32` | the CATRINE MERRA-2 configuration: PPM with FV3 kord-8 vertical, CMFMC with the DQRCU cloud base, non-local VDIFF, time-varying stepwise sources, native-C90 initial states |
| `run_c90_merra2_*` | the same with Lin-Rood (FV3 or upwind vertical), Holtslag–Boville VDIFF, and on CPU in Float64 (`slow`) |
| `run_c90_era5_tm5_gpu_f32` | the CATRINE ERA5 configuration: TM5 convection (collaborative LU), dkg diffusion with the diffusive surface-flux boundary, protocol initial states |
| `run_c90_era5_*` | the same over two daily binaries (`multiday`), with `n_merge = 2` and `lmax_conv = 50`, with Beljaars–Viterbo local Kz, and on CPU in Float64 (`slow`) |
| `run_c90_era5_l117_gpu_f32` | dkg diffusion, no convection, on the L117 binary of `pre_era5_n320_c90_l117` (`slow`) |

The small grids (one day, 2021-12-01) use static CAMS, GridFED and Zhang
Rn-222 sources (time-varying sources are cubed-sphere only), constant Kz,
decay, a Gaussian blob and a uniform 400 ppm tracer. The C90 cases run
2022-01-15 from the 2021-12-01 initial states.

Not covered yet: GEOS native identity-C180 and GEOS-FP C720 preprocessing,
GEOS surface/PBL, convection, VDIFF, OMEGA regularization and chain-mass paths,
the binary snapshot writer, observation sampling, TM5 convection attachment,
adjoints and inversions (their kernels accumulate with atomics, so they need a
tolerance or a dot-product test rather than bit identity), Float64 on GPU and
Metal.

## Inputs

The cases read data on the group's servers (`~/data/AtmosTransport`): ERA5
N320 raw fields, the GEOS-Chem MERRA-2 archive, the CATRINE emissions, initial
states and native-C90 fluxes, and GCHP's 2021-12-01 03 UTC output (initial
state). Three input sets are frozen copies in `/temp1/cfranken/goldens/inputs`:

- `era5_0.5x0.5/` holds the ERA5 fields of `pre_ll72`, `pre_o24` and
  `pre_c24`, which read the spectral `lnsp` and `vo_d` files
  (`spectral_hourly/`) and the model-level thermodynamics
  (`physics/era5_thermo_ml_*.nc`) of 2021-12-01 and 02. The 2021-11-30
  thermodynamics and the convection files of the three days are kept but not
  read. No maintained tool downloads these files any more. `SHA256SUMS` in
  that folder lists their checksums; `check` does not fingerprint input
  directories, so verify them with
  `cd /temp1/cfranken/goldens/inputs/era5_0.5x0.5 && sha256sum -c SHA256SUMS`.
- The ERA5 C90 L66 binaries with TM5 convection attached (2022-01-15 and 16),
  because convection attachment is not part of the preprocessor. Since the
  v0.5 re-record (2026-10-13) they are byte-identical copies of two days of the
  v0.5 production set `transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v05`
  (ERA5 N320 → C90 L66, `hybrid_mass` column weights, dry flux thickness,
  98,325 Pa dry-mass target, TM5 attachment v4 with dry convection rates; code
  f4ac1f65). `inputs/SHA256SUMS` lists them and `inputs/README.md` records
  their provenance; the previous pair is kept in
  `inputs/superseded_2026-10-12_l66_hb_gchp_v3attach/`.
- `geosit_c180_20211201/` holds only the GEOS-IT native C180 collections used
  by `pre_geosit_c45`: 2021-12-01 CTM_A1 and CTM_I1 plus the 2021-12-02 CTM_I1
  endpoint. Surface, convection and VDIFF collections are deliberately absent.
  Verify them with
  `cd /temp1/cfranken/goldens/inputs/geosit_c180_20211201 && sha256sum -c SHA256SUMS`.

A changed input file named in a config appears as a note in `check`.

## Known failures

Cases tagged `known_failure` are skipped, and absent from the comparison,
unless `--tags` or `--cases` names them. No case carries the tag at present.
The two reduced-Gaussian cases carried it until the preprocessor passed its own
write-time replay gate (Poisson balance sign, then the humidity-aware dry-mass
pin; items A1 and A11 of `docs/memos/2026-10-09_refactor_log.md`).

## Adding a case

Add a `[[case]]` to `cases.toml` with a config template in `configs/`
(`@OUTPUT@`, `@CASE:<name>@` and `@REPO@` are filled; each `set` key must
exist in the template, each `add` key must not), record it into the reference with
`record <ref> --cases=<name>`, and run `check <ref> <new> --cases=<name>` to
confirm it is reproducible.
