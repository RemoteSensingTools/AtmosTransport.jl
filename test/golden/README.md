# Golden outputs

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
both runs, and notes on anything besides the code that differed between the
runs (the filled config, Julia, the Manifest, threads, GPU, input files). It
exits with status 1 if a case differs, fails or wrote nothing to compare.
Runtime cases of a check read the reference's preprocessed binaries, so a
runtime difference is never a preprocessing difference.

Select cases with `--cases=a,b` (whatever their tags), `--tags=run,gpu`
(cases carrying all of these) or `--skip-tags=slow`. A check of the runtime
cases only, `--tags=run --skip-tags=slow`, takes about 20 minutes; the
preprocessing cases add about 30 minutes and 35 GB of memory (L117). GPU cases
run on the visible CUDA device (`CUDA_VISIBLE_DEVICES`); `--threads=N` sets
the Julia threads of each case process (default 8). Unknown options and tags
selecting nothing are errors.

A case never overwrites: every selected case directory must be new. To
re-record one case, delete its directory in the reference first
(`rm -r <ref>/<case>`), then `record <ref> --cases=<case>`.

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

Each case process runs with:

- **Regridding weights recomputed** in the case's `_regrid_cache` (the
  preprocessor's `regridder_cache_dir` and, for runtime sources,
  `ATMOSTR_REGRID_CACHE_DIR`), so a change to the weights shows up in the
  outputs instead of being hidden by a stale cache.
- **No `ATMOSTR_*` or `ERA5_N320_PROFILE` switches** inherited from the
  parent environment (they can change what a run does or writes); the
  scrubbed names are listed in `status.toml`.

`status.toml` also records the commit and whether `src`, `ext`, `scripts`,
`config`, `Project.toml` or `test/golden` had local changes, the run
environment (Julia, Manifest hash, threads, GPU), the size and modification
time of every input file the config names (directories such as the raw met
archives are not fingerprinted), and any non-finite values in the NetCDF
outputs (recorded and warned about, not an error).

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

Not covered yet: GEOS native cubed-sphere preprocessing (MFXC/MFYC, cm
closures), the binary snapshot writer, observation sampling, TM5 convection
attachment, adjoints and inversions (their kernels accumulate with atomics, so
they need a tolerance or a dot-product test rather than bit identity),
Float64 on GPU and Metal.

## Inputs

The cases read data on the group's servers (`~/data/AtmosTransport`): ERA5
N320 raw fields, the GEOS-Chem MERRA-2 archive, the CATRINE emissions, initial
states and native-C90 fluxes, and GCHP's 2021-12-01 03 UTC output (initial
state). Two inputs are frozen copies in `/temp1/cfranken/goldens/inputs`:

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
  because convection attachment is not part of the preprocessor.

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
