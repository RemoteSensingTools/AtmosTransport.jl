# AtmosTransport.jl refactor roadmap: Oceananigans-inspired steps (from 2026-10-09)

Starting point: branch `refactor/wip` at HEAD `f6437e89`, with a clean working tree. The golden harness (stored reference outputs, compared byte for byte) has **27 cases**: 22 runtime runs and 5 preprocessing runs. Earlier notes said 28; that was wrong. Every step below is one commit, gets a Codex review, and updates its README or docs in the same commit.

Terms used throughout:
- **Bit-identical by construction:** the same floating-point operations run in the same order, so results cannot change.
- **Host sync:** the CPU waits for the GPU to finish its queued work.
- **Workgroup:** the block of GPU threads launched together.
- **PTX:** NVIDIA's virtual GPU instruction set that the compiler emits (the driver compiles it to SASS machine code). Unchanged PTX/SASS shows the kernel code is unchanged; speed is unchanged only if the launch configuration and the surrounding host code are unchanged as well.
- **Dispatch:** Julia picks a method from the type of an argument.
- **Kernel golden:** a bitwise reference for one kernel, proposed in M7.

---

## 1. Principles

### What we adopt from Oceananigans

1. **Names in the config, types in the code.** A string or Symbol from the TOML is turned into a type once, when the config is read. After that, behavior is chosen by dispatch, not by `if kind === ...` chains. We already do this for advection, diffusion, convection and chemistry. Surface fluxes and GEOS closures do not yet.
2. **Components declare what they need, and one validator checks it once.** Errors appear at startup and say how to fix the problem ("regenerate the binary with include_vdiff_fields = true").
3. **One way to launch a GPU kernel.** All launches go through `Architectures.launch!`, and every launch states its workgroup and whether it syncs.
4. **Physics written once, chosen by small marker types.** A direction (x, y, z) or a column layout (lat-lon, reduced Gaussian, cubed sphere) is a type. Per-direction kernel names stay explicit and searchable.
5. **Every file format has a writer and a library reader.** Scripts read sections by name instead of computing byte offsets.
6. **Readable objects.** A one-line `summary` and a tree `show` for each type a user sees, and output files that record which physics ran.
7. **Tests mirror `src/`**, can run on one selected architecture, and come with import hygiene, a bibliography and a developer guide.

### What we deliberately do not copy

- **Anything that changes rounding or launch shape:** `@muladd`, heuristic workgroups, and workgroup sizes derived from the grid. Oceananigans does not promise bit-identical results; we do.
- **Unsplit tendency kernels.** We keep the split Strang sweeps with ping-pong buffers, the per-sweep positivity budgets and binary replay.
- **"No host syncs at all."** Removing the halo syncs measured 3.6 % slower; how that interacts with the asynchronous window prefetch remains to be measured.
- **Global mutable defaults objects.** Options pass through constructors.
- **Changes to KernelAbstractions internals** (`KernelParameters`, `OffsetStaticSize`). They redefine methods inside another package, and we pin KernelAbstractions 0.9.
- **`@eval`-generated method families.** They are hard to search and hard for scientists to read.
- **Automatically adding missing requirements.** A missing binary section cannot be created at run time; the only correct response is an error that says to regenerate the binary.
- **Per-iteration callbacks, and shortening Δt to hit output times.** These break binary replay and add GPU syncs.
- **Heavy machinery:** `FieldTimeSeries` (halos, time interpolation), the `ParallelTestRunner` worker pool, a mutable callback registry, and grid-level topology type parameters.

---

## 2. Milestones, in recommended order

Milestones in different areas can run in parallel. Results-changing steps are marked **RESULTS** and always come last within their area.

### M0. Safety nets (no `src/` changes)

**Goal:** a failing test no longer hides later failures, and the golden inputs cannot disappear.

1. The test runner keeps going after a failing file and prints the 15 slowest files (`test/runtests.jl`, `test/README.md`). First record today's Pass total by summing the per-testset summaries, because no single total exists yet.
2. Stop compiling the package twice: use `using AtmosTransport` in `test/regridding/runtests.jl` and `test/core/test_era5_n320_vertical_merge.jl`.
3. **Freeze the ERA5 0.5° golden inputs (urgent).** `pre_ll72`, `pre_o24` and `pre_c24` read `~/data/.../era5/0.5x0.5/{physics,spectral_hourly}`. No maintained tool can recreate those files: their downloaders were deleted in `028c1c34`. Copy them to `/temp1/cfranken/goldens/inputs`, record checksums, and point the three configs at the copies (`test/golden/configs/pre_*.toml`, `test/golden/README.md`).
4. Record the golden reference at `f6437e89` or later, then repeat `check` on the same commit.
5. Fix `CONTRIBUTING.md`. It names about ten types and functions that do not exist and an outdated docs command. Add it to the files the link checker covers.

**Gates:** same Pass total; same test file list; goldens identical. **Risk:** none. **Effort:** S (each step).

### M1. Catch silent misconfiguration at startup

**Goal:** a typo or a misaligned setting fails before any binary is opened. Several real cases exist today:
- Two `c45` configs set `order = 7`, which is never read, so they run Lin-Rood PPM5 instead of PPM7.
- A misspelled `[diffussion]` table silently disables diffusion.
- One misaligned snapshot hour silently loses every later snapshot.

Steps:
1. Check snapshot schedules and output settings at setup (`src/Models/runner/output.jl`, `src/Output/runtime_output.jl`, `DrivenRunner.jl`):
   - Requested hours must fall on met-window ends.
   - A path without times, or times without a path, is an error.
   - `format = "binary_mmap"` on lat-lon or reduced-Gaussian grids is rejected at setup. Today it fails only at the first write, after a day of transport.
   - `[output.fields]` combined with `binary_mmap` gets a warning.
2. One shared key checker with "did you mean ...?", and one `_config_bool` instead of three copies (`observation_sources.jl`, `Output.jl`, `Models.jl`, `Preprocessing.jl`).
3. Known-key tables for each runtime section, checked in `validate_config`:
   - Include `[run]` `Hp`, `halo_padding` and `tracer_name`.
   - Advection keys under `[run]` are valid only when `[advection]` is absent (legacy form).
   - Report flat `[tracers.X]` keys that sit next to an `init` subtable; today they are dropped silently.
   - Start as warnings; whether they become errors is an owner decision.
4. `validate_config` also parses the physics specs, `air_mass_reset_mode`, `physics_cadence` and the output spec, so all config errors are reported together.
5. Schema/parser agreement test: each schema enum must equal what the parsers accept. This adds `geoschem_nonlocal_vdiff` (used by a golden config), `expected_nlevel`, `required_preprocessor_contract`, `regridding`, and the `[output]` aliases marked deprecated. `additionalProperties = false` stays off for now (see Section 3).

**Gates:**
- new tests;
- run `validate_config` on every TOML in `config/runs` and `config/examples` and list the results in the commit message;
- goldens identical;
- `test_jet.jl` and `test_aqua.jl`.

**Risk:** none. Fixing the `c45` configs changes what they run, so that waits for the owner. **Effort:** M.

### M2. One capability check per binary section

**Goal:** remove a validation layer whose checks mostly can never fire, and close a real gap: in multi-file runs, `cmfmc_matrix` convection checks the `dtrain` section only for the first file.

1. Fold the runner's capability check into the recipe validator:
   - Add a `_runtime_has_dtrain` predicate.
   - Delete `_validate_capability_match` and the `_validate_convection_capability` methods in `runner/model_setup.jl`, plus the now-unused import.
   - Fix the stale text in `DrivenRunner.jl:16`, `src/Models/README.md:76` and `runner/README.md:27,71`.
2. One error message per diffusion kind. The "regenerate with include_..." hints move into `validate_runtime_diffusion`.
   - Keep a one-line check in `materialize` that reuses the same message. Without it, the exported `build_runtime_diffusion` would stop validating.
   - Fix the negative tests so they hit the capability message rather than an unrelated `_pbl_cache_shape` error.

**Gates:** `test_cs_driven_builders.jl` (new testset), `test_cs_multifile_equivalence.jl` (the only multi-file `cmfmc_matrix` coverage), `test_transport_model_convection.jl`, goldens. **Risk:** none. **Effort:** S + S.

### M3. Replay checks become config keys and leave a record

**Goal:**
- A binary written with its continuity check skipped can no longer pass for one that passed it.
- The load-time replay check gets a TOML key.
- No new environment variable reads appear in `src/`.

1. Allowlist test for environment-variable reads, plus `docs/src/config/environment.md`. There are 37 reads in 23 files today.
   - Key the allowlist by variable name plus an allowed-file pattern, not exact file paths, so later file moves do not break it.
   - Skip strings and comments, and special-case the two helpers that read variable names built at run time.
   - Document the variables read by `scripts/run_transport.jl`, including `ATMOSTR_NO_AUTO_THREADS`.
2. One resolver function for the write-time replay check, replacing 9 inline reads.
3. `[numerics] write_replay_check`:
   - The header gets `write_replay_check = false` only when the check is off, so default headers stay byte-identical.
   - The inspector and the driver warn about such binaries.
   - The lat-lon to cubed-sphere regrid path is driven by a script, not TOML, so `scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl` gets a `--no-write-replay-check` flag.
   - The GEOS and regrid writers must resolve the setting before building the header.
4. `[input] validate_replay` replaces `ATMOSTR_REPLAY_CHECK`. The double-negative `ATMOSTR_NO_REPLAY_CHECK` goes away.
   - Rewrite the three error messages that tell users to set it.
   - Fix the false `[met_data] validate_replay` claim in `conservation_budgets.md:85`.
   - CHANGELOG entry.

**Gates:** preprocessing and runtime goldens identical; new tests on a small lat-lon and a small cubed-sphere binary with the check off. **Risk:** none. **Effort:** S, S, M, S.

### M4. A library reader for snapshot binaries (`binary_mmap`)

**Goal:** this binary output format is used by 17 active run configs. Its only converter is untested, and five scripts each parse the format by hand in slightly different ways. This is the most valuable item in the readers group.

1. Cubed-sphere tag round-trip helpers in `src/Grids/`:
   - The default definition is chosen from the **definition tag**, not the panel convention. Mixed pairs are legal, and the file records no longitude offset.
   - Test all four definition × convention pairs.
   - Use the helpers in `Output/binary_writer.jl` and `cubed_sphere_reader.jl`.
2. `open_snapshot_binary` (`src/Output/binary_reader.jl`), with:
   - validation on open;
   - reads of one field at one frame without loading the file;
   - `snapshot_mesh`.
   It must also read older files that have no `tracer_total_mass`, use one shared generic function with `Visualization.snapshot_times` so the names do not clash, and extend `State.mass_basis` explicitly.
3. Rebuild the converter `scripts/postprocess/binary_to_netcdf.jl` on the reader. The test reference must use the Float32 standard-definition mesh that the reader rebuilds.
4. Port the four extractor scripts.

**Gates:**
- write-then-read is bit-exact;
- old and new converter output on one production daily file is identical;
- old and new extractor outputs are identical;
- `test_output_snapshots.jl`, `test_binary_planet_radius.jl`, `test_jet.jl` (Grids is gated by a JET count);
- docs build.

No golden covers this format; the goldens only cover the shared tag code.

**Risk:** none. **Effort:** S, M, S, M.

### M5. Transport-binary header and single-section reader

**Goal:** scripts read one named section of one window. They no longer depend on private tables, or on the geometry block being empty.

1. `read_transport_header(path)` and `disk_float_type`:
   - Port `inspect_transport_binary.jl` and the coarsener's open-twice probe.
   - `compare_preprocessors.jl` stays on raw JSON: it must be able to compare outputs that fail today's format checks. Only its dead function is removed.
2. `section_shape`, `section_elements`, `section_range`, `section_view` and `load_section(!)`, built on one table.
   - The four existing private layout tables become thin wrappers over it, so the step removes tables instead of adding a fifth. It is integer-only and bit-identical, and keeps their error behavior.
   - A cross-check test confirms the counts match the header's `n_*` keys and sum to `elems_per_window`.
   - Update the "add a payload section" checklist in the README.
3. Port `check_mass_balance_dec2021.jl` and `diagnose_tm5_active_layers.jl`. Copy each panel into scratch before summing, so the summation order stays the same.

**Gates:** every section from `load_section` is bit-equal to what `load_window!` and the other existing loaders return; script outputs identical before and after; `test_binary_inspector.jl`. **Risk:** none. **Effort:** S, M, S.

### M6. Loose ends (small, independent)

1. Correct the header comment in the four TRENDY batch configs (it names a generator that was never committed).
2. Script constants: use `EARTH_RADIUS`, `STANDARD_GRAVITY` and `IFS_EARTH_RADIUS` in five active scripts, with an explicit `using AtmosTransport.Parameters: ...`. These constants are not exported at the top level, so without the import the scripts fail at run time.
3. One set of block-coarsening sum helpers in a neutral file, `transport_binary/cs_block_restriction.jl`. The two area-weighted helpers stay separate, with a comment explaining their precision difference. Verify with `test_cs_binary_coarsener.jl`, its synthetic fixture and `test_geos_cs_passthrough.jl`.
4. Once the owner picks an ERA5 provenance option (Section 4), a test that every `scripts/...` path cited in `src/`, active configs, tests and docs exists. Patterns are treated as globs and directories; the old deny-list is removed in the same commit.

**Risk:** none; step 3 is bit-identical by construction. **Effort:** S each.

### M7. Kernel-level goldens and a codegen check (needed before any kernel work)

**Goal:** most kernels that M9–M13 touch have no golden case today:
- lat-lon and reduced-Gaussian convection;
- the single-tracer diffusion kernels;
- all adjoint kernels;
- cubed-sphere CW84 in the z direction on GPU.

1. A `kind = "kernel"` case type in `test/golden/`. Required harness changes:
   - Outputs use a `.raw` extension. `.bin` files are parsed as transport binaries and would crash the comparison.
   - A per-case `threads` setting.
   - No config file needed.
   Cases: lat-lon and cubed-sphere sweeps (Upwind, Slopes, PPM, CW84; CPU F64/F32, GPU F32), TM5 per-thread and collaborative, CMFMC and CMFMC-matrix, Kz/dkg packed and single, and the adjoint sweeps plus the CMFMC adjoint on CPU with 1 thread. GPU atomic additions are not reproducible run to run, so those use tolerances. The reduced-Gaussian horizontal sweep is compared bitwise on CPU only.
2. `scripts/checks/check_kernel_codegen.jl` records LLVM and PTX/SASS **for each production specialization** (scheme × limiter × float type × array type × CUDA workgroup override), not one launch per kernel. Unchanged PTX/SASS, with unchanged launch configuration and host-side code, means the kernel code is unchanged; anything else needs a GPU timing run.

**Gates:** recording twice gives identical results. **Risk:** none. **Effort:** M + M.

### M8. Run loop: shared logic in one place, without merging the two loops yet

**Goal:** the lat-lon and cubed-sphere run loops (about 600 lines) duplicate snapshot, flush and setup logic and have already diverged, causing bugs. Fix that duplication without the risky full merge.

1. Characterization tests. **First measure** whether split and combined multi-file runs are bitwise equal today; the existing cubed-sphere test only checks tracers to `rtol = 1e-12`.
   - Lat-lon: compare final state and snapshot/observation times. Lat-lon rejects time-varying sources, so the clock cannot be observed end to end.
   - Cubed sphere: needs a time-varying `cs_native` fixture.
   - Extend the existing batch-vs-window test with nonzero fluxes instead of duplicating it.
   - Keep the source-text check on `start_time` until step 3 replaces it with a unit test.
2. A shared `SnapshotCursor` and one capture function. Delete the unused `_flush_daily_output!`.
3. Shared per-binary open/finish helpers, with the `[run]` settings parsed once. Keep `scripts/diagnostics/tracer_budget_closure.jl` working.
4. The cubed-sphere loop steps one met window at a time with `run_window!`, which runs the same sequence of steps. Update the docstrings.
5. One `stop_window` rule, checked right after the first driver opens on both grid types, so no partial output file is left behind. The cubed-sphere loop currently clamps silently.
6. *(Optional, RESULTS, owner)* With `start_window > 1`, build the initial condition on the start window's air mass. Today the initial mixing ratio is scaled by m(window 1)/m(start window). No golden uses `start_window > 1`.

**Gates:**
- all 22 runtime goldens identical, including the slow and 2-day cases;
- GPU wall time within noise on `run_c90_merra2_gpu_f32` and the 2-day case;
- `test_jet.jl` and `test_aqua.jl`.

**Risk:** steps 2–4 are bit-identical by construction; step 5 changes only error behavior. **Effort:** M overall.

### M9. Output files record the physics that ran (RESULTS: attributes only)

**Goal:** A/B campaigns (PPM vs Lin-Rood, CW84, `n_merge`, cadence) produce output files that are self-describing; today they differ only by path.

1. Writer plumbing:
   - An `attributes` keyword, including the end-of-run path `_write_output_frames!`.
   - A fresh dictionary for each asynchronous daily write, to avoid races.
   - The empty key is omitted from snapshot-binary headers, so they stay byte-identical.
2. `run_metadata_attributes` in TOML vocabulary:
   - the physics choices;
   - the **effective** physics cadence plus the binary's `runtime_substep_contract`, not the raw TOML value;
   - the input binaries with their preprocessor contract and commit;
   - float type, backend, and a digest of the config.
3. Wire it through the M8 helpers and the observation writer; document it in `output_schema.md`.

**Gates:** `check` shows only "new global attributes" differences in all 22 runtime cases. Then re-record the reference and repeat `check` on the same commit. **Risk:** changes results (attributes only). **Effort:** S, M, S.

### M10. Surface-flux datasets as types

**Goal:** close silent emission errors:
- an unknown `kind` name is treated as a generic file;
- files in `mol m-2 s-1` are read as `kg m-2 s-1`.

1. Characterization tests for every kind, including:
   - the cross cases: `lmdz` with tonnes units on the static and time-varying paths, and a generic file with tonnes units;
   - a file with no units attribute (like the Zhang golden input);
   - an unregistered name.
2. Dataset types (GridFED, EDGAR SF6, Zhang Rn-222, CAMS/LMDZ, native cubed-sphere, generic). Each docstring states the dataset's native units and conversion. One function reads `kind`.
3. One reorientation and unit path for the static and time-varying loaders. It takes an explicit branch-set argument, because the static loader has an EDGAR/tonnes branch that the time-varying loader lacks.
4. `validate_config` checks the surface-flux tables before any binary is opened.
5. **RESULTS (owner):** strict units:
   - `mol`/`g`-based and unknown units become errors;
   - a missing units attribute never counts as a conflict;
   - a declared kind that contradicts a non-empty units attribute is an error.
   All golden inputs are already accepted (checked with `ncdump`).

**Gates:** step-1 tests unchanged after steps 2–4; focused surface-flux tests; runtime goldens identical. **Effort:** S, M, S, S, S.

### M11. Kernel launches through `launch!`, part 1 (outside advection)

**Rule for this milestone and M13:**
- No host sync is added or removed.
- Launches inside panel or tile loops pass `sync = false`, so each loop keeps its single sync after the loop (`launch!` defaults to `sync = true`).

1. `launch!` with a KernelAbstractions-chosen workgroup:
   - Use an explicit marker type, not `nothing`, because `nothing` already means "no tile" in a CUDA hook.
   - Use the `Vararg{Any,N}` signature so long argument lists stay concretely typed. Test inference and zero allocations with 15 or more arguments.
   - The docstring says "KernelAbstractions 0.9.x (0.9.41–0.9.44 checked)".
2. Cubed-sphere surface flux (2 sites).
3. Chemistry, State/Fields and Output (8 sites). Includes the import lines in `Fields.jl`, `Output.jl` and `Chemistry.jl`.
4. Convection (15 sites). Drop the redundant `workgroupsize` keyword on TM5 collaborative launches.

**Gates:** focused tests per module; CPU and GPU goldens identical; codegen check. **Risk:** bit-identical by construction. **Effort:** S, S, S, M.

### M12. Remove the duplicated single-tracer sweep kernels

1. Lat-lon/RG ("structured"): the 3-D sweeps call the packed multi-tracer kernels with one tracer.
   - Delete 3 kernels and a duplicated `@eval` family; move their physics docstrings onto the packed kernels.
   - Correct the false "Oceananigans" citation.
   - The test keeps an independent hand-written reference; update `Advection/README.md`.
2. Cubed-sphere z sweep: the same change.
   - Reshape on each call. Cached array aliases would silently stop aliasing on GPU.
   - Workgroup stays 256.
   - The only production effect is Lin-Rood with the upwind vertical scheme.
3. *(Optional, together with step 2)* Cubed-sphere x and y sweeps. These are used only by footprint and tape replay, so they are covered by tests only.

**Gates:**
- `run_c90_merra2_linrood_upwindz_gpu_f32` and `run_c24_linrood_cpu_f32`;
- `test_cs_ppm_adjoint_footprint.jl`;
- kernel goldens;
- `bench_cs_advection_gpu.jl --scheme=linrood5`.

**Risk:** step 1 is bit-identical by construction; steps 2–3 need goldens to confirm. **Effort:** M each.

### M13. Kernel launches through `launch!`, part 2 (advection) and documented sync rules

1. Lat-lon/RG sweeps (5 source sites).
2. Cubed-sphere split sweeps (21 sites). The profiling wrapper keeps its sync decision; rename its closure argument from `launch!` to `enqueue!`.
3. Halo exchange and seams (8 sites). Syncs and launch order stay exactly as they are.
4. Lin-Rood forward (17 sites). Correct the false "avoid repeated compilation" comment.
5. Guard test: any `ndrange` outside `Architectures.jl` fails, in either the `ndrange =` form or the `; ndrange` shorthand.
6. Documentation in `kernel_architecture.md`, with the **corrected** synchronization model:
   - On CUDA, CUDA.jl synchronizes a buffer's previous stream when the buffer is used from another stream (`Base.convert(::Type{CuPtr}, ::Managed)` calls `maybe_synchronize`, CUDA.jl `src/memory.jl`). The prefetch task fills window buffers on its own stream, so the main task's first use of them waits for those copies. How much of the measured halo "pacing" this explains has not been measured.
   - On Metal there is no such automatic wait, so the existing operator-end syncs are what protect the prefetch handoff.
   - Also: a table of workgroup sizes, and a fix for the false claim at line 46.

**Gates:**
- CPU and GPU goldens on all cubed-sphere and Lin-Rood cases;
- codegen check;
- warm GPU timing on an idle GPU (`bench_cs_advection_gpu.jl` at C90/C180, after checking `nvidia-smi` and machine load), which must be within noise.

**Risk:** bit-identical by construction. **Effort:** S, M, S, M, S, S.

### M14. Write each face-flux scheme once (one body for x, y and z)

1. Documentation fixes in `reconstruction.jl`:
   - Every direction has a mass floor; there is no z-only floor.
   - Keep the `deps/tm5/base/src/advectx.F90` path (`deps/` exists but is gitignored) and correct the routine names to `dynamu`/`dynamv`/`dynamw`.
   - State that lat-lon transport assumes a global longitude span, and add a check or record it as a known limitation.
2. Direction types and boundary-rule types, with yes/no helpers (`_is_closed`, `_is_interior`) and masked forms built on them. Update the README file maps.
3. Upwind written once (one commit), then Slopes/PPM/CW84 for y and z (one commit), then x through the same bodies (one commit). The `_x/_y/_zface_tracer_flux` names stay as one-line wrappers, so no caller changes.
4. *(Optional)* The adjoint driver dispatches on direction instead of Symbol chains, and the adjoint face methods are written once per scheme (12 methods become 4). Verified with adjoint kernel goldens (CPU, 1 thread) and the dot-product identity tests.

**Gates:**
- codegen check per specialization;
- the lat-lon and cubed-sphere advection goldens;
- kernel goldens for cubed-sphere CW84 z on GPU F32, which no runtime golden covers;
- if PTX changes, timing runs.

**Risk:** bit-identical by construction (confirmed by the gates); adjoint step 4 needs goldens. **Effort:** S, S, S/M/M, M.

### M15. Column kernels shared across lat-lon, reduced-Gaussian and cubed-sphere grids

1. Column-layout types:
   - Each array argument is explicitly classed as halo-padded state or interior. The CMFMC scratch array is padded, not interior.
   - Pilot on the TM5 per-thread kernels (3 into 1), which exercise both padding and the area lookup.
   - The cubed-sphere CMFMC-matrix kernel simply reuses the lat-lon kernel; they are already identical.
2. TM5 collaborative kernels, 3 into 1 (about 470 duplicated lines removed).
   - Update the tests and benchmarks that launch these kernels by name.
   - **Metal Float32 smoke run** on the gpu-metal runner.
   - PTX registers and shared memory must be identical, or `bench_tm5_convection.jl` must stay within noise (convection dominates wall time).
3. CMFMC, 3 into 1. Must update `src/Adjoints/Adjoints.jl:51` and `ConvectionAdjoint.jl`; otherwise the package fails to load. Verify with `test_cmfmc_adjoint_identity.jl`.
4. Diffusion Kz solve kernels by layout (packed 3 into 1, single 2 into 1). The Kz and dkg families stay separate.

**Gates:** TM5, CMFMC and diffusion goldens; lat-lon/RG kernel goldens; Metal smoke runs. **Risk:** steps 1, 2 and 4 bit-identical by construction; step 3 needs goldens. **Effort:** S, M, M, M.

### M16. Readable objects (`summary` / `show`)

1. Summaries for advection schemes, limiters and vertical reconstructions, using today's labels.
2. Summaries for the diffusion, convection (`lmax_conv`, `n_merge`, `use_collab_lu`), chemistry, surface-flux and output-schedule types.
3. Trees for state, `TransportModel`, the physics recipe and `DrivenSimulation`.
   - Trees go in the `text/plain` display method; plain 2-argument `show` equals `summary`, so error messages stay one line.
   - Never index arrays (that would be scalar indexing on GPU).
4. Build the run-start log from these summaries, adding cadence and reset mode.

**Gates:**
- `test_jet.jl` and `test_aqua.jl`;
- a GPU test that displays the GPU-side model, state and simulation with scalar indexing disallowed (the run-start log prints only the host-side recipe, so goldens never cover this);
- goldens identical.

**Risk:** none. **Effort:** S, M, M, S.

### M17. Test layout and one GPU switch

1. Recursive test discovery, plus `--group=a,b` / `ATMOSTR_TEST_GROUP` and `--list`. Do not recurse into `test/regridding`; error only when the whole selection is empty.
2. Test paths via `pkgdir(AtmosTransport)` instead of a new helper file; update the rule in `test/README.md:50`.
3. Move the files into `test/core/<group>/` in three `git mv` batches, updating doc paths.
   - Three diagnostic files include core test files by path; move their shared builders to `test/helpers` instead.
   - Verify with `--tiers=diagnostic` on CPU.
4. GPU switch `ATMOSTR_TEST_ARCHITECTURE`:
   - The helper loads CUDA or Metal at the top of its file, to avoid Julia "method too new" (world-age) errors.
   - It works only through `julia --project=test test/runtests.jl`. `Pkg.test`'s sandbox cannot see CUDA, so under `Pkg.test` it fails with a message saying so.
   - Convert the 5 files that probe for CUDA themselves. The footprint test must change in the same commit as the diagnostics that include it.
   - The 12 diagnostic GPU files stay CUDA-only and skip on Metal.
5. Run the operator groups on GPU. CPU keeps its current float types; tolerances are set explicitly per float type.

**Gates:** same Pass totals and file list; GPU pass counts unchanged before and after. **Risk:** none. **Effort:** S, S, M×3, M, M.

### M18. Developer guide and bibliography

1. A Developer section in the manual:
   - a testing page;
   - "adding an operator", using the existing `ExponentialDecay` as the worked example (not a naive duplicate of it). Add KernelAbstractions to `docs/Project.toml` if the example shows a kernel;
   - "adding an advection scheme" and "adding a met driver";
   - shrink the "Extending" section of `CLAUDE.md` to a link.
2. Bibliography with DocumenterCitations:
   - `refs.bib` with about 28 works and verified DOIs, and a References page;
   - the link checker ignores `@cite` links;
   - narrow the DocumenterVitepress compat to ≥0.2.7, which ships the citations extension;
   - convert the theory pages and **only the docstrings that render** (exported types). Private-function docstrings keep free-text references.
3. About 8 doctests with stable output (meshes, architectures, `flux_application_seconds`).

**Gates:** docs build with `warnonly = false`; link test. **Risk:** none. **Effort:** M, S/S/S, S.

### M19. Import hygiene (ExplicitImports)

1. **First, a separate commit:** the CUDA extension imports Tape internals from `AtmosTransport.Tape`, their owner. Today it imports them through Adjoints, where they look unused and would be deleted. Check on wurst that the extension still loads.
2. ExplicitImports test:
   - stale imports are enforced;
   - implicit-import counts and owner violations are recorded as count baselines, as JET does, rather than fixed all at once.
3. Explicit imports in the leaf modules only (Architectures, Grids, State).
   - Note: a missing name fails when the function is first **called**, not when the package loads.
   - Run the goldens that exercise each converted module.

**Risk:** bit-identical by construction. **Effort:** S, S, M.

### M20. Remove the deprecated environment fallbacks (after one tagged release)

Delete the fallbacks for the horizontal-balance, physics-cadence and replay variables. For one more release, a set variable raises an error that names its replacement key. **Risk:** none. **Effort:** S.

### M21. GEOS native preprocessing (needs owner decisions first)

1. Add GEOS golden coverage; none exists today.
   - GEOS-IT C180 → C90 with the default closure.
   - Optionally, a slow native-C180 `omega_regularized` case.
   - Extend the synthetic C8 test to the other closures.
2. The GEOS workspace stores the typed horizontal balance instead of a Symbol (bit-identical). Update `scripts/diagnostics/omega_prepare_bench.jl`.
3. Then, per owner decision: retire the diagnostic closures.
4. Then closure types, using the Symbol→type boundary pattern; update `omega_prepare_bench.jl`.
5. Split the per-window closure code into one method per closure only after the OMEGA golden exists.

**Effort:** M, S, S, M, M.

---

## 3. Dropped or deferred, with reasons

**Synchronization**
- **Removing host syncs** (seams, Lin-Rood forward, lat-lon per-sweep, adjoint per-panel, and the per-panel cubed-sphere diffusion syncs, which a reviewer found are also candidates): deferred.
  - Removing the halo syncs measured 3.6 % slower; where the wait goes instead (for example into the prefetch handoff, which CUDA.jl orders across streams) needs an NVTX measurement.
  - Prerequisite: measure with NVTX where `prefetch_fetch_wait` time goes. If the pacing comes from CUDA's per-buffer waits, the real fix is an event-based window handoff, not deleting syncs.
  - On Metal, no operator-end sync may be removed without a Metal multi-window equality test.
- **Explicit prefetch barrier:** fold into the measurement step above. On CUDA the handoff is ordered by CUDA.jl's cross-stream synchronization (M13 step 6); on Metal an explicit barrier is needed.

**Kernel launches and kernels**
- **Adjoint and inversion `launch!` conversion (58 sites):** deferred. Off the production path, no golden coverage, and heavy use of atomic additions on GPU, so bitwise checks are impossible there.
- **Optional workgroup changes:** the Nt = 1 packed CUDA tile for Lin-Rood, workgroup-hook consolidation, the halo rule and cheaper wrap (A5), and `@inbounds` in face bodies (A6). A6 as written would do nothing; if redone with `@propagate_inbounds` it removes the only out-of-bounds protection. Defer all until a profile shows they matter.
- **B1** (direction types replacing `Val(1)`/`Val(2)` in seams; 9+ files, cosmetic), **B4** (merging the x/y host files) and **C6** (conversion kernels by layout): low value for the effort.

**Configuration and environment variables**
- **`[run] assert_binary_cfl` and `[input] prefetch_windows`:** deferred. Debug-only switches on hot GPU paths that M13 also edits. Listed as documented debug variables instead.
- **Payload-requirements table (capability d3):** optional; changes public output (new inspector rows and `binary_capabilities` fields).
- **`additionalProperties = false` in the schema:** deferred until the 91 active configs that use the legacy `[run] scheme` key are migrated or listed as deprecated. It currently contradicts `test_public_config_and_cli.jl:182`.
- **Low-value environment moves:** `ATMOSTR_INSTITUTION` as an `[output]` key, and folding `ERA5_N320_PROFILE` and `ATMOS_OMEGA_TIMING` into SectionTimer.

**Run loop and output**
- **Merging the two run loops (M8 step "a5") and run-callback objects:** conditional. After M8 the remaining duplication is small, and callbacks pay off only when a concrete feature needs them (window-end NaN check, restart, new output kind). A public callbacks API is dropped.
- **Boundary air-mass mismatch diagnostic on cubed sphere:** dropped as framed. On lat-lon it is computed after the air mass has been reset, so in the default mode it is exactly zero. Decide whether to measure it before the reset or delete it.
- **Embedding the full resolved config TOML in output files:** deferred. It needs a comparison exemption and exposes absolute paths; a config digest covers the use case.

**Readers and scripts**
- **Rebuilding the internal window loaders on the section table (a4):** deferred. It touches every runtime loader for tidiness only and would make the loaders more lenient. Making the private tables thin wrappers in M5 gets most of the benefit.
- **Writing `A_ifc`/`B_ifc` into snapshot binaries:** dropped; the NetCDF writer never reads the vertical grid. Longitude offset and mesh float type are optional additions.
- **Gravity change in `compare_era5_geosit_met.jl`:** dropped. The script cannot run on any current binary: it reads header keys that cubed-sphere v4 binaries lack, and its default input is gone. Archive or repair it as one decision.
- **Porting `per_column_depth_histogram.jl`:** archive it instead.
- **Area-weighted coarsening precision (e2, RESULTS):** deferred. An ulp-level change to three production GEOS Float32 configs with no golden coverage.

**Tests and docs**
- **Enforced allocation budgets:** keep as a report only. Counts depend on thread count and on the unpinned KernelAbstractions version in CI, and they cannot see GPU costs. Code changes driven by them are deferred.
- **CI sharding:** deferred until per-file timings exist.
- **Doctests of summaries:** wait for M16. **Cubed-sphere tutorial:** optional.
- **ExplicitImports beyond the leaf modules:** deferred until the Adjoints private-name policy is settled.
- **GEOS step M21.5 (per-window closure methods):** deferred until a native-C180 OMEGA golden exists. Otherwise the candidate closure would be moved with no golden coverage.

---

## 4. Decisions needed (recommendation in brackets)

**Configuration**
1. Unknown config keys: warn for one release and then error, or error now? [warn, then error]
2. Legacy forms (`[run] scheme` in 91 active configs, top-level `[init]`, flat tracer keys, `[output]` aliases): migrate, or keep as deprecated? [keep as deprecated; migrate the configs over time]
3. The two `c45` binary-format A/B configs: rename `order` to `ppm_order`, which changes them from PPM5 to PPM7, or retire them? [retire]
4. Environment keys: names and sections (`[numerics] write_replay_check`, `[input] validate_replay`); write the header key only when the check is off [yes]; deprecation window [one tagged release].

**Run loop and output**
5. Output metadata: on by default, re-recording the 22 runtime goldens with attribute-only changes, or opt-in? Attribute prefix (`physics_*` or `atmostr_*`), and whether to bump `output_contract`. [on by default; `physics_*`; no bump]
6. Run loop:
   - `stop_window` above the window count: error or clamp? [error]
   - Drop the lat-lon batch fast path? [yes, after timing]
   - Fix the `start_window > 1` initial condition (RESULTS)? [yes]
   - Single `start_time` rule (integer-exact sum)? [yes]

**Surface fluxes and ERA5 inputs**
7. Surface fluxes:
   - Unregistered kind names: error and migrate the two `oceanflux` configs to `kind = "file"`, or warn? [error]
   - Approve strict units (M10 step 5)? [yes]
8. ERA5 split-file inputs (no producer exists):
   - A: restore the two legacy downloaders [recommended];
   - B: native download recipes;
   - C: declare them frozen.
   Long term: move the ERA5 goldens to the N320 GRIB path?

**GEOS**
9. GEOS closures:
   - which GEOS goldens to record;
   - retire `pressure_fixer`, `moisture_filtered`, `pfix_corrected` and `omega_full_replacement`? [retire]
   - trim the 27 aliases (`omega` silently means `omega_regularized`)?

**Kernels**
10. Kernels:
    - keep `launch!`'s `sync = true` default [yes];
    - remove the deprecated exported `diagnose_cm_from_continuity_ka!` (breaking: needs a version bump and release notes);
    - export `launch!`?
    - spend GPU time on the sync measurement?
    - which machine is the timing reference (L40S F32 or A100 F64)?
    - acceptance threshold [≥1% warm wall time over ≥3 repeats].
11. Names and placement:
    - direction and boundary-rule types (Oceananigans names `XDirection`/`Periodic`/`Bounded`, unexported, in Grids?);
    - column-layout types (an Operators-level file?);
    - summary vocabulary (physics names or TOML keys).

**Readers**
12. Readers:
    - export level: MetDrivers/Output only, or also top level? The top-level export cap is 160. [module level only]
    - should `open_snapshot` also accept snapshot binaries?
    - archive or repair `compare_era5_geosit_met.jl` [archive]; port or archive `check_mass_balance_dec2021.jl`.

**Tests, docs and imports**
13. Tests:
    - groups nested under `test/core/<group>` with tiers kept [yes];
    - variable names `ATMOSTR_TEST_GROUP` / `ATMOSTR_TEST_ARCHITECTURE`;
    - GPU runs explicit opt-in only, ending the auto-probing [yes];
    - a self-hosted GPU CI runner, and which devices are authorized.
14. Docs: citation style [author-year]; whether the Codex-review rule goes in public docs.
15. Imports: are the 57 private names Adjoints imports an accepted internal interface? Raise Julia compat to 1.11 to use `public`? [document them; stay on 1.10 for now]
16. Capability d3: accept the new inspector rows and `binary_capabilities` fields, or skip? [skip for now]

---

## 5. What is in flight and how it fits

Both pieces of in-flight work have landed; the working tree is clean.

- **`[run] physics_cadence`** is committed (`8a29efd8`). It replaces `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS`, and the environment variable is still honored with a deprecation warning.
  - Its fallback is removed in M20 after one release.
  - It is parsed early by M1 step 4, included in the known-key tables, and recorded in output files by M9. M9 must record the **effective** cadence, because the setting changes anything only on binaries whose `runtime_substep_contract` is `binary_schedule`.
  - Plans that list it as a dependency can drop that dependency.
- **`Architectures.launch!`** is committed (`f6437e89`). 27 sites are converted: Diffusion 21 + 3 and SurfaceFlux 3. About 136 hand-written launches remain.
  - The rollout continues in M11 (outside advection), M12 (fold the duplicate kernels first, so fewer sites need converting) and M13 (advection).
  - New hazard: `launch!` defaults to `sync = true`. Any launch inside a panel or tile loop must pass `sync = false` to keep today's single sync after the loop, especially in convection and in the cubed-sphere sweeps, which deliberately avoid per-kernel host syncs on GPU.
  - The two remaining cubed-sphere surface-flux sites become their own small commit (M11 step 2).
- **Golden reference:** record it at `f6437e89` or later, with a repeat `check` on the same commit, before M0 step 4 and before any later milestone that relies on goldens.
- **Uncommitted work in the main checkout** (outside this worktree: the streaming-binary permission fix and the TRENDY rerun guide) is not affected by this roadmap.
---

## 6. Owner decisions (2026-10-09, evening)

- All recommendations in Section 4 are adopted.
- Timing references: the L40S on wurst for Float32; the A100 on curry for
  Float64 and for Float32-versus-Float64 comparisons on the same hardware.
  Acceptance threshold as recommended (≥ 1 % of warm wall time over ≥ 3
  repeats).
- ERA5 has no benchmark to match yet. ERA5 runs must not deviate in unphysical
  ways from MERRA-2-based runs; MERRA-2 runs are the plausibility reference,
  not a target. The ERA5 spectral preprocessing path is the default.
- Metal: tested on the owner's Mac laptop (self-contained bundle).
- Since this roadmap was written, the halo exchange (M13 step 3) was rebuilt on
  point operations (`Architectures.AbstractPointOp`, `Fused`, `Sequence`), with
  one fused launch for the edges and, when a sweep direction asks for them, one for the corners on CUDA (default fusion policy), and separate launches with bound panel arrays on Metal; see the refactor log, "GPU performance and point
  operations".
