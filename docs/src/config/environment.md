# Environment variables

Run behavior belongs in the TOML configuration. The environment variables
below are for paths, metadata, profiling and diagnostics; none of them changes
results unless noted. `test/core/test_environment_variables.jl` checks that the
package reads no others.

## Paths and metadata

| Variable | Effect | Read in |
|---|---|---|
| `ATMOSTRANSPORT_DATA_ROOT` | Root of `$ATMOSTRANSPORT_DATA_ROOT` in configured paths (default `~/data/AtmosTransport`). Any other `$NAME` or `${NAME}` in a path expands to that variable. | `expand_data_path` (`src/AtmosTransport.jl`) |
| `ATMOSTR_REGRID_CACHE_DIR` | Cache of conservative regridding weights for surface fluxes (default `~/.cache/AtmosTransport/cr_regridding`). | `src/Models/initial_conditions/surface_flux.jl` |
| `ATMOSTR_SPECTRAL_CACHE_DIR` | Default spectral-coefficient cache of the ERA5 preprocessor; `[cache] spectral_coefficients_dir` overrides it. | `src/Preprocessing/configuration.jl` |
| `ATMOSTR_SCRATCH_DIR`, `ATMOSTR_TMPDIR`, `SCRATCH`, `TMPDIR`, `TEMP`, `TMP` | Candidate directories, in this order, for extracting ERA5 surface archives. | `src/Preprocessing/era5_surface_reader.jl` |
| `ATMOSTR_INSTITUTION` | `institution` attribute of NetCDF snapshot and observation files. | `src/Output/netcdf_schema.jl`, `src/Output/observations/observation_netcdf.jl` |
| `USER`, `USERNAME` | `user` attribute of NetCDF snapshot files. | `src/Output/netcdf_schema.jl` |
| `NO_COLOR`, `TERM` | ANSI colors in the run summary only when `NO_COLOR` is unset or empty and `TERM` is set and not `dumb`. | `src/Models/runner/configuration.jl` |

## Profiling and diagnostics

| Variable | Effect | Read in |
|---|---|---|
| `ATMOSTR_TIMERS`, `ATMOSTR_NVTX`, `ATMOSTR_ALLOC_TIMERS` | Section timers, NVTX ranges, and allocation counts within the timers (`1`, `true`, `on` or `yes`; `ATMOSTR_ALLOC_TIMERS` only together with `ATMOSTR_TIMERS`). | `src/Diagnostics/SectionTimer.jl` |
| `ATMOSTR_PROFILE_GPU` | With timers on (`1`, `true`, `on` or `yes`): synchronize after each cubed-sphere sweep kernel and time the launch and the wait separately (slower). | `src/Operators/Advection/cs_sweep_common.jl` |
| `ATMOSTR_ASSERT_CS_BINARY_CFL` | `1`: recompute the cubed-sphere CFL subcycle count and stop if the binary's substep schedule asks for fewer (slower). | `src/Operators/Advection/CubedSphereStrang.jl` |
| `ATMOSTR_DISABLE_PREFETCH` | `1`: load each met window synchronously instead of prefetching the next one on a second thread. | `src/Models/driven_window_state.jl` |
| `ATMOS_OMEGA_TIMING` | Timing output of the GEOS OMEGA-consistent `cm` closure in the preprocessor. | `src/Preprocessing/transport_binary/cubed_sphere_geos.jl` |
| `ERA5_N320_PROFILE` | `1`: per-window timing of the ERA5 N320 preprocessor. | `src/Preprocessing/sources/era5_n320_window.jl`, `era5_n320_to_cs.jl` |

## Preprocessing switches

| Variable | Effect | Read in |
|---|---|---|
| `ATMOSTR_NO_WRITE_REPLAY_CHECK` | `1`: skip the write-time replay-continuity gate of the preprocessor (diagnostic only). The binary records `write_replay_check = false`; the inspector marks it and the runtime warns when it opens it. | `write_replay_check_enabled` (`src/Preprocessing/configuration.jl`) |

## Removed

These were replaced by configuration keys and are now ignored.

| Variable | Replacement |
|---|---|
| `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS` | `[run] physics_cadence = "substep"` |
| `ATMOSTR_REPLAY_CHECK` | `[input] validate_replay = true` |
| `ATMOSTR_NO_REPLAY_CHECK` | none: the load-time replay check runs only when requested |
| `ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE` | `[numerics] balance_mode = "per_layer"` |

## Command-line runner

`scripts/run_transport.jl` reads these before the package loads:

| Variable | Effect |
|---|---|
| `JULIA_NUM_THREADS` | Thread count. When it is unset and Julia runs one thread, the runner restarts itself with two threads (also after an explicit `--threads=1`). |
| `ATMOSTR_NO_AUTO_THREADS` | `1`: do not restart with two threads (needed to run with one thread). |
| `ATMOSTR_PROFILE_MODE` | `full`: CUDA activity profile of the whole run. `window`: profile a window of the run and then **end the process**, so the run does not finish. |
| `ATMOSTR_PROFILE_WARMUP_SEC`, `ATMOSTR_PROFILE_DUR_SEC` | Start (default 120 s) and length (default 60 s) of the `window` profile. |
