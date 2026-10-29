# Full-Codebase Review Notes — 2026-05-17

Historical review of `f2500fe`, not an active defect list. Several paths below
have since been removed or repaired. See the
[September 6 review](2026-09-06_revamp_review.md) and
[current-main integration record](2026-09-05_main_integration.md) for verified
status; old performance rankings were hypotheses, not measurements.

Synthesis of six parallel reviewers (julia-style-reviewer) each covering a
non-overlapping slice of the active `src/` tree (~63K lines, ~150 files).
Findings collected here for future documentation generation and as a
backlog for cleanup work. Cited line numbers were valid at HEAD = `f2500fe`
on the `convection` branch.

Scope of the six slices:
1. **Foundations** — Architectures, Parameters, Quantities, Grids, State, Kernels.
2. **Preprocessing** — `src/Preprocessing/` including `transport_binary/`.
3. **Advection** — `src/Operators/Advection/`.
4. **Convection/Diffusion/Chemistry/SurfaceFlux** — non-advection operators.
5. **MetDrivers / Downloads / Regridding**.
6. **Models / Output / Adjoints / Tape / Footprint / Inversion / Diagnostics / Visualization**.

---

## 1. Broken Contract Pathways

Public type contracts where method coverage is inconsistent, invariants are
asserted in one place but not enforced elsewhere, or abstractions leak.

### 1.1 Operator `apply!` dispatch holes
- **`NoConvection.apply!` is missing the `CubedSphereState` arm**
  ([Convection/operators.jl:58](../../src/Operators/Convection/operators.jl#L58)).
  Every other no-op operator (`NoDiffusion`, `NoSurfaceFlux`) has both
  `CellState` and `CubedSphereState` arms — `NoConvection` will `MethodError`
  on a CS run with convection disabled.
- **CMFMC has no error arm for `CubedSphereState` on a non-CS mesh**
  ([CMFMCConvection.jl:318](../../src/Operators/Convection/CMFMCConvection.jl#L318));
  a mismatched call falls through to a generic message about
  face-indexed-state-on-RG that is misleading.
- **`sweep_horizontal!` is generated only for `AbstractConstantScheme`**
  ([StrangSplitting.jl:564](../../src/Operators/Advection/StrangSplitting.jl#L564)).
  `apply!`-level guards stop a misrouted scheme there but `sweep_horizontal!`
  called directly with `SlopesScheme`/`PPMScheme` hits a runtime `MethodError`
  inside `_horizontal_face_tendency!`.

### 1.2 Abstract-type fanouts that are split per concrete subtype
- **`flux_basis` / `mass_basis` defined per concrete `AbstractFaceFluxState`**
  ([FaceFluxState.jl:207](../../src/State/FaceFluxState.jl#L207)). A new
  subtype silently loses both methods until the user trips a `MethodError`.
  A one-line `@inline flux_basis(::AbstractFaceFluxState{B}) where B = B()`
  closes this.
- **`AbstractCubedSphereField` is parallel to, not derived from,
  `AbstractTimeVaryingField`**
  ([CubedSphereField.jl:12](../../src/State/Fields/CubedSphereField.jl#L12)).
  Operators that accept `AbstractTimeVaryingField` cannot accept a
  `CubedSphereField`; consumers must branch with `isa` at every call site.
- **`AbstractHorizontalMesh` documents 7 required methods**
  ([AbstractMeshes.jl:20](../../src/Grids/AbstractMeshes.jl#L20)) but has no
  abstract-level fallbacks that error with a useful message. New mesh types
  produce silent `MethodError`s instead of "must implement face_length".
- **`AbstractVerticalCoordinate.b_diff` is undocumented**
  ([VerticalCoordinates.jl:46](../../src/Grids/VerticalCoordinates.jl#L46));
  callers of `b_diff` on an arbitrary VC will `MethodError`.

### 1.3 Concrete drivers that don't honor their declared interface
- **`PreprocessedERA5Driver` subtypes `AbstractMassFluxMetDriver` but
  implements none of the 7 required methods** (`total_windows`, `window_dt`,
  `steps_per_window`, `steps_per_window_schedule`, `load_transport_window`,
  `driver_grid`, `air_mass_basis`)
  ([ERA5/DryFluxBuilder.jl:25](../../src/MetDrivers/ERA5/DryFluxBuilder.jl#L25)).
  It works because nothing calls it as a driver — but the subtyping advertises
  a contract that does not exist. Either implement the interface or drop the
  supertype with a docstring rename ("closure-strategy tag").
- **`ERA5BinaryReader` is missing a `has_qv_endpoints` forwarding stub**
  ([MetDrivers.jl:36](../../src/MetDrivers/MetDrivers.jl#L36) for the present
  stubs). Callers fall through to a method-not-found error rather than
  receiving `false`. ERA5 v5 binaries cannot declare
  `humidity_sampling = :window_endpoints` until this is added.

### 1.4 Cubed-sphere palindrome halo timing asymmetry
- The forward Y/X legs of `strang_split_cs!` / `strang_split_cs_mt!` rely on
  trailing halo exchanges from the preceding sweep
  ([CubedSphereStrang.jl:950](../../src/Operators/Advection/CubedSphereStrang.jl#L950)),
  while the reverse legs do an additional leading exchange
  ([CubedSphereStrang.jl:977](../../src/Operators/Advection/CubedSphereStrang.jl#L977)
  and `:989`).
  For `n_sub == 1` this is one unnecessary exchange per leg per palindrome.
  For `n_sub > 1` the forward and reverse halves do asymmetric counts. Not a
  conservation violation but should be normalized: "halo at the start of each
  pass; skip the trailing intra-loop exchange on the last pass".

### 1.5 Plan 41 typed contract — one production CS path still outside it
- **`regrid_ll_binary_to_cs` does not use the typed contracts at all**
  ([cubed_sphere_regrid.jl](../../src/Preprocessing/transport_binary/cubed_sphere_regrid.jl)).
  It owns its own loop variables, calls `verify_cs_window_contract!` and
  `update_cs_positivity_accumulator` directly, and manually does
  `mv(tmp_path, out_path)` instead of using
  `promote_streaming_binary!`/`quarantine_streaming_binary!`. It is the only
  remaining CS producer that bypasses Plan 41 P4's unified driver. It also
  rebuilds the LL→CS regridder per call (30–60 s for C90/C180); it accepts no
  `PreprocessorRunCache` and so cannot share with the spectral path.

### 1.6 Diffusion lower-boundary approximation

**September 6 correction:** the small-diffusivity restriction in the original
paragraph below is incorrect for fixed linear backward-Euler coefficients.
Adding surface mass before the solve gives the same right-hand side as adding
that source directly to the system. Full GCHP VDIFF parity remains unvalidated;
see the [current ordering contract](../src/for_tm5_gchp_users/operators_on_binaries.md#surface-flux-and-diffusion-ordering).

- **`DiffusiveSurfaceFluxBoundary` is documented as "GCHP VDIFF lower-boundary
  placement"** ([Diffusion/operators.jl:22](../../src/Operators/Diffusion/operators.jl#L22))
  but the implementation adds surface flux mass to the bottom cell and then
  runs the Thomas solve, rather than modifying the `b[Nz]/d[Nz]` entries of
  the tridiagonal. These are equivalent only when `Kz × dt / dz²` is small.
  Either correct the implementation to match GCHP or change the docstring to
  state the approximation.

### 1.7 CMFMC fallback wrong-device array
- **`_cmfmc_dtrain_array` builds a CPU `Array{FT}(undef, ...)`** when
  `forcing.dtrain === nothing`
  ([CMFMCConvection.jl:106](../../src/Operators/Convection/CMFMCConvection.jl#L106),
  `:120`). If `cmfmc` is a `CuArray`/`MtlArray`, the kernel sees this
  CPU-backed `@Const(dtrain)` and either launch-fails or silently reads host
  memory. Replace with `similar(cmfmc, FT, Nx_c, Ny_c, Nz_c)` so the fallback
  inherits the device backend.

### 1.8 Misc smaller contract holes
- **Resolved:** `replay_window_pair` under both LL and RG drivers loaded window
  `k+1` twice per iteration. The unified driver now reuses `fluxes_next`
  ([driver.jl](../../src/MetDrivers/transport_binary/driver.jl#L106)).
- **CS `PreprocessorRunCache` is not threaded into `regrid_ll_binary_to_cs`**
  (see §1.5) — separate cache instance per day batch, regridder rebuilt.
- **LinRood: `strang_split_linrood_ppm!` (public) runs 4 fillz passes and
  omits the midpoint hook**
  ([LinRood.jl:927](../../src/Operators/Advection/LinRood.jl#L927)) while
  `_strang_split_linrood_ppm_cs!` (private, production) runs 3 with the hook.
  Either delegate or deprecate the public API.

---

## 2. Obsolete Code Hidden in `src/`

### 2.1 Definitely dead — safe to remove
- **`preferred_era5_api`**
  ([Downloads/python_interop.jl:37](../../src/Extras/Downloads/python_interop.jl#L37))
  — exported but no caller in `src/`, `test/`, or `scripts/`.
- **`OPeNDAPProtocol.execute!` is a permanent `error()` stub**
  ([Downloads/pipeline.jl:203](../../src/Extras/Downloads/pipeline.jl#L203)) and
  `MERRA2Source._default_protocol` returns `"opendap"`. The entire MERRA-2
  download path is permanently broken at runtime. Remove or implement.
- **`PressureTendencyClosure`, `NativeVerticalFluxClosure`** — exported
  subtypes that immediately `throw(ArgumentError("Phase 3+..."))`
  ([MetDrivers/DryFluxBuilder.jl:50](../../src/MetDrivers/DryFluxBuilder.jl#L50)).
- **`_cs_static_subcycle_count`** (the per-direction version) — production
  path uses only `_cs_static_palindrome_subcycle_count`
  ([CubedSphereStrang.jl:734](../../src/Operators/Advection/CubedSphereStrang.jl#L734)).
  Has tests only.
- **`_cmfmc_wellmix_subcloud!` standalone helper** — kernel body inlines the
  same logic; the standalone is unreachable
  ([cmfmc_kernels.jl:187](../../src/Operators/Convection/cmfmc_kernels.jl#L187)).
- **`use_limiter::Bool` argument of `_sweep_z!`** — every caller passes `true`,
  none read it
  ([CubedSphereStrang.jl:1141](../../src/Operators/Advection/CubedSphereStrang.jl#L1141)).
- **`_apply_surface_source!` + `_surface_shape`** in
  [SurfaceFlux/sources.jl:105](../../src/Operators/SurfaceFlux/sources.jl#L105)
  — verify no `DrivenSimulation` caller post-plan-17; if confirmed dead,
  move to `src_legacy/`.

### 2.2 Probably dead — verify before removing
- **`write_day_binary!`**
  ([latlon_contracts.jl:726](../../src/Preprocessing/transport_binary/latlon_contracts.jl#L726))
  — pre-P3 single-pass writer. Current production path uses
  `LatLonDeferredBinaryWriter`. No `src/` caller; check `scripts/` and `test/`.
- **`_ppm_edge_values_ord4` / `_ord6`** in
  [ppm_subgrid_distributions.jl:86](../../src/Operators/Advection/ppm_subgrid_distributions.jl#L86):
  `LinRoodPPMScheme` validates `ORD ∈ {5, 7}` only, so the ORD=4/6 dispatch
  arms are unreachable.
- **`DrivenSimulation._apply_surface_sources!` and the `surface_sources`/
  `chemistry` fields** — Plan 17 Commit 6 moved these into `step!`. Marked
  "for future adaptive helpers". If no helpers are planned, remove.
- **`Simulation.jl::run!`** — used only in unit tests; `while sim.time <
  sim.stop_time` is the classic float off-by-one pattern. Document or remove.

### 2.3 Active but vestigial naming / forwarding shims
- **`AbstractMassFluxBasis`, `MoistMassFluxBasis`, `DryMassFluxBasis` aliases**
  ([Basis.jl:28](../../src/State/Basis.jl#L28)) — three additional names for
  the existing `AbstractMassBasis`/`MoistBasis`/`DryBasis`. They are exported
  and appear in docstrings, creating two synonymous names for the same three
  types.
- **`StructuredTopology`, `FaceConnectedTopology` aliases**
  ([AbstractMeshes.jl:81](../../src/Grids/AbstractMeshes.jl#L81)) — backward
  names for `StructuredFluxTopology`/`FaceIndexedFluxTopology`.
- **`state.air_dry_mass` `getproperty` alias** for `air_mass` on
  `CellState{DryBasis}` and `CubedSphereState`
  ([CellState.jl:146](../../src/State/CellState.jl#L146)). Type parameter
  already encodes the basis.
- **`CSLinRoodAdvectionWorkspace.getproperty` forwarding**
  ([LinRood.jl:164](../../src/Operators/Advection/LinRood.jl#L164)) — defeats
  type inference on every workspace field access. Remove and require
  `workspace.cs.field`.
- **`Grid = AtmosGrid` alias** ([AbstractMeshes.jl:155](../../src/Grids/AbstractMeshes.jl#L155))
  — not exported, not referenced.
- **`apply_poisson_balance!(contract = nothing)` branch**
  ([latlon_contracts.jl:271](../../src/Preprocessing/transport_binary/latlon_contracts.jl#L271))
  — pre-typed-gate fallback; production always passes a typed contract.

---

## 3. Documentation Notes — paste-ready facts

Phrased for direct use in `docs/reference/` and `docs/memos/`.

### 3.1 `FROM_TM5.md` corrections / additions
- The operator table at `FROM_TM5.md:20` references `TiedtkeConvection`; the
  production type has been **`TM5Convection`** since Plan 23.
- **Convection placement deviation**: we place `Convection` *after*
  FV dynamics (matching GCHP), not interleaved into the palindrome (TM5).
  This is described in prose at `FROM_TM5.md:88–108` but not in the
  quick-reference table — add a "Convection placement" row. Reference
  implementation: `gchp_chunk_mod.F90:1164-1174`.

### 3.2 `ADVECTION_SCHEMES.md` invariants to add
- **CS palindrome positivity budget**: `2·(out_x + out_y + out_z)/m_start`,
  computed in `_cs_static_palindrome_subcycle_count`
  ([CubedSphereStrang.jl:803](../../src/Operators/Advection/CubedSphereStrang.jl#L803)).
  Factor of 2 because the palindrome traverses each direction twice;
  `out_x = max(0,-ax_lo) + max(0,ax_hi)` (Lin-Rood 1996 refinement) rather
  than `max(|ax_lo|,|ax_hi|)`.
- **Multi-tracer launch counts**: split-sweep CS and structured LL both fuse
  the multi-tracer X/Y/Z passes into **6 launches total** (not 6·Nt). The
  LinRood `CSLinRoodStyle` path does *not* fuse and runs ~7 launches per
  tracer per palindrome (5 horizontal substeps + 2 vertical) — i.e. ~7·Nt
  launches per step.
- **ORD=5 vs ORD=7**: ORD=7 inside `_ppm_edge_values_ord7` delegates to
  `_ppm_edge_values_ord5` and adds an `_apply_ord7_boundary` correction at
  face indices `1` and `Nc+1`. ORD=7 is "ORD=5 with discontinuous-boundary
  correction at gnomonic edges", not a wider interior stencil.
- **fillz pass count**: 3 passes in the production CS LinRood path
  (`_strang_split_linrood_ppm_cs!`), matching GCHP's `fv_fill.F90`. The
  public-API `strang_split_linrood_ppm!` runs 4 (divergent — see §1.8).

### 3.3 `PREPROCESSING_GUIDE.md` invariants to add
- **Window-contract pairing rule** ([window_contracts.jl:111](../../src/Preprocessing/transport_binary/window_contracts.jl#L111)):
  every write must pair exactly one `AbstractWindowContract{G,FT}` with an
  `AbstractBinaryWriter{G,FT,Basis}`. The `G` type parameter guarantees the
  topology match at compile time; `Basis` guarantees the on-disk label
  matches the runtime reader.
- **cm-closure point — dry-basis invariant**: `recompute_cm_from_dm_target!`
  must run *after* `balance_cs_column_mass_fluxes!` (or the LL/RG analogue);
  `cm[:,:,1] = cm[:,:,Nz+1] = 0` is enforced afterward. Initializing `cm`
  from `divergence(am, bm)` before balance produces the bug fixed in Plan 39.
  References:
  [latlon_contracts.jl:249](../../src/Preprocessing/transport_binary/latlon_contracts.jl#L249),
  [cubed_sphere_spectral.jl:296](../../src/Preprocessing/transport_binary/cubed_sphere_spectral.jl#L296).
- **Per-window adaptive substep selection** (GEOS-native CS only):
  `_geos_select_steps_for_window!` runs up to 8 refinement iterations per
  window. The final schedule is patched into the streaming binary header via
  `driver_before_close_writer!` → `set_streaming_steps_per_window_schedule!`.
  Runtime must read `steps_per_window_by_window` to apply per-window substep
  counts.
- **GEOS pressure-fixer endpoint rule**: GEOS-native binaries write the *raw*
  GEOS DELP_dry endpoint as `m_target`, not the pressure-fixer-derived
  endpoint. The pressure-fixer's implied endpoint can go negative in thin
  upper layers; the raw target is more robust for the column balance and `cm`
  diagnosis. Header records `"geos_mass_endpoint" => "raw_dry_endpoint"`.
- **ERA5 spectral ps pin**: `pin_global_mean_ps!` runs once per window when
  `mass_fix_enable=true`. Per-window offsets (Pa) land in
  `ps_offsets_pa_per_window` in the header. Without the pin, raw ERA5
  analysis drifts in total mass.
- **Boundary-stub flux invariant (RG)**: `face_left[f] ≤ 0 ||
  face_right[f] ≤ 0` identifies a pole-singularity stub face; runtime
  silently discards `hflux` there (`StrangSplitting.jl:279`). Writers must
  zero boundary stubs at write time;
  `verify_boundary_stub_flux_rg` gate checks this.
- **Surface flux unit contract**: emission sources supplied to operators are
  `kg/s` per cell, *already area-integrated*
  ([SurfaceFlux/sources.jl:17](../../src/Operators/SurfaceFlux/sources.jl#L17)).
  EDGAR/GFED `kg/m²/s` rasters must be multiplied by cell area before
  ingestion. Currently missing from the guide.
- **`fill_dz_hydrostatic_constT!` requires `(ps, ak_ifc, bk_ifc)` in the
  binary**: any diffusion-enabled run depends on these fields being present.

### 3.4 `BINARY_FORMAT_V5.md` schema fields
- `format_version = 3` requires both:
  - `steps_per_window_by_window :: Vector{Int}` (length = nwindow)
  - `poisson_balance_target_scale_by_window :: Vector{Float64}` (length = nwindow)
- The scalar `steps_per_window` in the header must equal `maximum(schedule)`.
- The `time_step_schedule` string field is `"constant"` or `"per_window"`
  depending on whether all entries are equal.
- `CubedSphereBinaryHeader.raw_header :: Dict{String, Any}` is the
  forward-compatible probe used at runtime, e.g.
  `get(hdr.raw_header, "runtime_substep_contract", nothing)` to detect the
  CS binary-schedule contract.
- Document `ps_offsets_pa_per_window` (ERA5 spectral mass-fix offsets).

### 3.5 `ARCHITECTURE.md` — TransportModel composition order
At each `step!(model)`:
```
transport_block:   X → Y → Z → V(dt/2) → S(dt) → V(dt/2) → Z → Y → X
convection_block:  apply!(state, convection_forcing, grid, convection, dt)
chemistry_block:   chemistry_block!(state, meteo, grid, chemistry, dt)
```
Binary-scheduled drivers skip the full `step!` and call `transport_step!`
per substep, then `convection_chemistry_step!` once per window at
`current_window_end_iteration`.

### 3.6 `GRID_CONVENTIONS.md` notes
- `LatLonMesh.lon_shift_rad = deg2rad(λᶜ[1])` must be supplied to the
  spectral preprocessor's `spectral_to_grid!` for `(-180, 180)` meshes
  ([LatLonMesh.jl:93](../../src/Grids/LatLonMesh.jl#L93)).
- `HybridSigmaPressure` level numbering: **k=1 is TOA, k=Nz is surface**
  throughout; `A`/`B` arrays have length `Nz+1` at interfaces.
  `level_thickness > 0` for increasing pressure, so positive `cm` is
  *downward*, consistent with `FaceFluxState` `(Nx, Ny, Nz+1)` layout where
  `cm[i,j,k+1]` is the flux at the bottom of layer `k`.
- `ReducedGaussianMesh` flattens **south-to-north, west-to-east within ring**:
  `c = ring_offsets[j] + i - 1`. Inverse of structured LL (which is i-fastest).
- `PanelEdge.orientation`: 0 = aligned, 2 = reversed (1 is reserved/unused).
  Stored as `Int` deliberately to leave room for future transpositions.
- Panel 5 in `GEOSNativePanelConvention` is the Americas panel, rotated 90°
  CW; local axes X=south, Y=east in native GEOS arrays. The `_panel_xyz`
  rotation applies `(η, -ξ, ...)`
  ([CubedSphereMesh.jl:566](../../src/Grids/CubedSphereMesh.jl#L566)).

### 3.7 `FROM_GCHP.md` new file recommended
- Convection placement (after FV) matches GCHP, citing
  `gchp_chunk_mod.F90:1164-1174`.
- Multi-tracer launch comparison: ours 6 fused launches per palindrome (LL/CS
  split-sweep); GCHP `fv_tracer2d.F90:532-549` runs the tracer loop *inside*
  the FV step with 4× PPM reconstructions per face per tracer.
- Putman-Lin cross-flux averaging reference: `tp_core.F90:200-213`.

### 3.8 New memo recommended — `DIFFUSION_AND_CONVECTION.md`
- Describe `DiffusiveSurfaceFluxBoundary` ordering `S(dt) → V(dt)` and
  document the GCHP-VDIFF approximation gap (§1.6).

---

## 4. Non-Julian Pathways

`if/elseif`, Symbol comparison, or `isa`-chain dispatch where multiple
dispatch would be cleaner or measurably faster.

### 4.1 GPU/CPU dispatch via `parent(arr) isa Array`
- **`StrangSplitting.jl:403`-`455`**: `parent(rm) isa Array` chooses CPU vs
  GPU path. CLAUDE.md explicitly forbids this — use `get_backend(rm)` and
  dispatch on `KA.CPU` vs the GPU backend type. The `parent isa Array` form
  also breaks for views of `CuArray`.

### 4.2 Symbol/Int direction dispatch in advection
- **`_cs_static_subcycle_count`**: `if direction === :x ... :y ... :z`
  ([CubedSphereStrang.jl:747](../../src/Operators/Advection/CubedSphereStrang.jl#L747))
  — function is dead in production (see §2.1) but pattern should not migrate
  to the live palindrome pilot.
- **`_adjoint_scheme_sweep!` direction Symbol if-chain**
  ([AdvectionAdjoint.jl:616](../../src/Adjoints/AdvectionAdjoint.jl#L616)
  and `:649`): repeated in two overloads. Promote `_CSSweepRecord.direction`
  to `Val{D}` and split into three 1-method overloads.

### 4.3 GPU warp-divergent branches inside hot kernels
- **`minmod_ppm` in `ppm_subgrid_distributions.jl:36`** has real `if/else`
  branches; the `_minmod3` in `limiters.jl:82` is already branchless. Replace
  the former with the latter.
- **`huynh_second_constraint` denominator guard
  ([ppm_subgrid_distributions.jl:63](../../src/Operators/Advection/ppm_subgrid_distributions.jl#L63))**:
  `if abs(denom) < 10*eps(FT); return zero(FT); end`. Inside a kernel-called
  `@inline` function. Use `ifelse(mag < 10*eps(FT), zero(FT), result)`.

### 4.4 Type-tag dispatch on already-typed values
- **`TransportModel.convection_chemistry_step!`**:
  `if !(model.convection isa NoConvection) ...`
  ([TransportModel.jl:393](../../src/Models/TransportModel.jl#L393)).
  Split into two `_apply_convection!` methods on `NoConvection` vs
  `AbstractConvection`.
- **`DrivenRunner._validate_capability_match` re-parses `conv_kind` from
  config Dict** ([DrivenRunner.jl:501](../../src/Models/DrivenRunner.jl#L501))
  even though `recipe.convection` is already typed. Dispatch on the typed
  field.
- **CS diffusion workspace via `hasproperty`/`getproperty`**
  ([Diffusion/operators.jl:336](../../src/Operators/Diffusion/operators.jl#L336),
  `:371`): runtime reflection masquerading as a contract. Replace with a
  typed `CSDiffusionWorkspace` struct so missing fields error at
  construction.

### 4.5 Custom `getproperty` overrides with runtime Symbol comparison
- `AtmosGrid.getproperty` forwards `:radius`, `:gravity`, `:reference_pressure`
  to `g.planet.*` ([AbstractMeshes.jl:157](../../src/Grids/AbstractMeshes.jl#L157)).
  In hot paths, prefer the function form `radius(g)`/`gravity(g)`.
- `CellState.getproperty` aliases `:air_dry_mass → :air_mass` and routes
  `:tracers` through a `TracerAccessor` wrapper
  ([CellState.jl:145](../../src/State/CellState.jl#L145)). `state.tracers.CO2`
  re-wraps on every call; in hot paths use `get_tracer(state, :CO2)`.
- `CSLinRoodAdvectionWorkspace.getproperty` (already noted in §2.3) is the
  worst — it breaks inference for *every* field access.

### 4.6 Symbol round-trip on mass basis
- `DrivenRunner.jl:459` derives a concrete `BasisT` type from
  `air_mass_basis(driver)::Symbol`; at lines 1077 and 1106 the same code
  converts `BasisT` back to a Symbol to pass to `write_snapshot_netcdf(...,
  mass_basis::Symbol)`. The frame already carries a typed `mass_basis`
  field. Either (a) make `write_snapshot_netcdf` dispatch on
  `AbstractMassBasis` or (b) read the symbol from `frame.mass_basis`.

### 4.7 String-key configuration dispatch
- `_output_partition` parses `split` string into `SingleOutputFile`/
  `DailyOutputFiles` ([runtime_output.jl:203](../../src/Output/runtime_output.jl#L203))
  — fine at TOML-parse time, not a hot path. But `output_split` accessor
  returns a Symbol downstream, used at 6 call sites in `DrivenRunner`; the
  typed partition should be threaded through instead of re-symbolized.
- `met_source` factory in `loader.jl:37` has a redundant `name == "GEOS-IT"`
  comparison nested under a `name == "GEOS-IT" || name == "GEOS-FP"` check.
  Use `get(_SOURCE_CONSTRUCTORS, name, nothing)` over a const Dict.
- `entrypoint.jl:336–340` uses `isa`-chain to enumerate spectral-supported
  topologies; converting to one-method-per-topology `_assert_spectral_support`
  closes the open-extension problem.

---

## 5. Hidden Performance Issues

### 5.1 Type instability with `Any` fields / `Ref{Any}` / `Vector{Any}`
- **`Ref{Any}(nothing)` area caches**:
  [WindowPBLKzField.jl:17](../../src/State/Fields/WindowPBLKzField.jl#L17),
  [GCHPHoltslagBovilleKzField.jl:19](../../src/State/Fields/GCHPHoltslagBovilleKzField.jl#L19).
  Every `_cached_backend_cell_areas!` call boxes/unboxes. Fix:
  `Ref{Union{Nothing, Matrix{FT}}}(nothing)` (or fully parametric).
- **`LatLonSpectralWindowWorkspace.last_hour_next::Any`**
  ([latlon_spectral.jl:10](../../src/Preprocessing/transport_binary/latlon_spectral.jl#L10))
  defeats inference in `flush_final_windows!` and writer paths. Add a type
  parameter `L`, default `Nothing`.
- **Tape ops `Vector{Any}` — three sites**:
  [TapeRecording.jl:248](../../src/Footprint/TapeRecording.jl#L248),
  [TapeRecording.jl:411](../../src/Footprint/TapeRecording.jl#L411),
  [LinRoodTape.jl:347](../../src/Adjoints/LinRoodTape.jl#L347). The reverse
  loop iterates `Iterators.reverse(ops)` and dispatches with
  `op isa _CSSweepRecord`/`_CSHaloRecord`/etc. — every iteration is dynamic.
  The `_CSTapeOp` union exists. Use
  `const _CSAllTapeOp = Union{_CSTapeOp, _CSLinRoodHorizRecord}; ops =
  _CSAllTapeOp[]`. **Highest-leverage performance fix in the adjoint layer.**
- **`PinnedHostCSTapeStorage.device_cache::Any`, `synchronize::Any`**
  ([TapeStorage.jl:41](../../src/Tape/TapeStorage.jl#L41)). Constrain to
  `Union{Nothing, NTuple{6}}` and `Union{Nothing, Function}`.
- **`CSSurfaceFluxJacobianResult.objectives::Vector{AbstractCSFootprintObjective}`**
  ([Observations.jl:71](../../src/Inversion/Observations.jl#L71)). Add type
  parameter `OT <: AbstractCSFootprintObjective`.
- **`PreprocessorRunCache.entries::Dict{Symbol, Any}`**
  ([window_contracts.jl:210](../../src/Preprocessing/transport_binary/window_contracts.jl#L210))
  — populated once per run, so runtime cost is negligible, but new key types
  require type-assertion at every call site.
- **`ERA5SpectralSettings.nt::NamedTuple` with `getproperty` forwarding shim**
  ([met_readers.jl:321](../../src/Preprocessing/met_readers.jl#L321)). Every
  `settings.field_name` access is a runtime symbol lookup.

### 5.2 Hot-loop allocations
- **`ReducedGaussianMesh.cell_faces` allocates `Int[]` then `push!`**
  ([ReducedGaussianMesh.jl:353](../../src/Grids/ReducedGaussianMesh.jl#L353)).
  Per-cell call, so on an octahedral N1280 mesh this is millions of small
  allocations per transport step. Return `NTuple{N, Int}` or use an in-place
  output buffer.
- **`_copy_interior!` allocates a `UnitRange` per call**
  ([CubedSphereStrang.jl:608](../../src/Operators/Advection/CubedSphereStrang.jl#L608))
  — 36+ calls per Strang step. `@views` and `@inline` it.
- **LinRood per-substep face-buffer allocations** — 5 × `ntuple(6, …)`
  allocations per `_record_linrood_horizontal_substep!` call
  ([LinRoodTape.jl:123](../../src/Adjoints/LinRoodTape.jl#L123)) +
  `rm_buf`/`m_buf` at `:195`. For C180 14-day adjoint runs this is ~690K GPU
  allocations on the recording forward pass alone. Pre-allocate in a
  workspace.
- **`_cmfmc_dtrain_array` allocates per met window** when `dtrain == nothing`
  ([CMFMCConvection.jl:106](../../src/Operators/Convection/CMFMCConvection.jl#L106))
  — beyond the GPU correctness issue (§1.7), it allocates ~15 MB per call
  for ERA5 288×181×72. Pre-allocate in `CMFMCWorkspace`.
- **`_validate_window_cm_sanity` allocates a full window per iteration**
  ([driver.jl](../../src/MetDrivers/transport_binary/driver.jl#L55))
  at driver construction — 24 windows × ~4 arrays × ~8 MB = ~768 MB
  temporary allocation. Pre-allocate buffers before the loop.
- **`diagnose_cm_from_continuity_vc!` allocated `Δb` per call**; the unused
  wrapper was removed for v0.5.0 rather than optimized.
- **Per-window TM5 buffer allocation in `_store_window_tm5_fields!`**
  ([latlon_workspaces.jl:723](../../src/Preprocessing/transport_binary/latlon_workspaces.jl#L723))
  and **`store_qv_output!`** at `:463` — 4+1 arrays per window per day.
  Pre-allocate the inner arrays at `allocate_window_storage` time (mirror the
  pattern used for `all_m`).
- **`cubed_sphere_geos.jl:669`** allocates 6 panel `copy()`s per window so
  `convert_cs_mass_target_to_delta!` doesn't clobber `m_next_target`.
  `dm_v4` is already pre-allocated — `copyto!` into it instead.
- **`_horizontal_face_outgoing_ratio`** (`StrangSplitting.jl:645`) allocates
  a per-level broadcast `outgoing ./ max.(m[:, k], eps(FT))`; structured
  path uses single-pass reduction without temporaries.

### 5.3 Wrong-device or scalar-GPU paths
- **`VerticalRemapWorkspace` allocates `zeros(FT, ...)` only**
  ([VerticalRemap.jl:65](../../src/Operators/Advection/VerticalRemap.jl#L65))
  — CPU-only. Any GPU LinRood path that reaches vertical remap hits scalar
  indexing errors. Mirror `LinRoodWorkspace`'s `array_type::Type{<:AbstractArray}`
  pattern.
- **`_cmfmc_dtrain_array` returns CPU `Array` on GPU runs** (already noted
  §1.7) — correctness *and* performance.

### 5.4 GPU launch overhead
- **HB Kz / WindowPBLKz: 6 sequential 2D KA launches in a `for panel in 1:6`**
  loop with a single barrier at the end
  ([WindowPBLKzField.jl:131](../../src/State/Fields/WindowPBLKzField.jl#L131),
  [GCHPHoltslagBovilleKzField.jl:178](../../src/State/Fields/GCHPHoltslagBovilleKzField.jl#L178)).
  Convert to a single 3D-with-panel launch
  (`ndrange = (Nc, Nc, 6)`).
- **CS surface flux: `synchronize(backend)` inside per-panel + per-source
  double loop** ([SurfaceFlux/operators.jl:289](../../src/Operators/SurfaceFlux/operators.jl#L289))
  — 6 × N_sources unnecessary host stalls per `apply!`. The LL/RG paths sync
  once at the end. Move the sync out.
- **`_cs_static_palindrome_subcycle_count` runs 6 sequential `mapreduce`**
  over panels ([CubedSphereStrang.jl:793](../../src/Operators/Advection/CubedSphereStrang.jl#L793))
  — each call blocks the host on GPU. Bypassed in the n_sub=1 path so this is
  only painful on contract-less runs.
- **`_compute_air_mass_kernel!` hardcodes 256 workgroup**
  (the obsolete generic cell-kernel prototype). Omit and let
  KA pick.

### 5.5 Repeated work per substep
- **`findfirst(==(src.tracer_name), tracer_names)` per `apply_surface_flux!`
  call** ([SurfaceFlux/operators.jl:165](../../src/Operators/SurfaceFlux/operators.jl#L165),
  `:189`, `:273`). `emitting_tracer_indices` already resolves these; thread
  the cache into the kernel launch.
- **`ExponentialDecay.apply!` resolves indices per call**
  ([Chemistry.jl:155](../../src/Operators/Chemistry/Chemistry.jl#L155)) via
  `ntuple(N) do n; tracer_index(...) end`. Cache a `Ref{Union{Nothing,
  NTuple{N, Int32}}}` lazily at first call.
- **`detect_python_env` runs 4 Python subprocesses per `download_data!`**
  ([Downloads/python_interop.jl:14](../../src/Extras/Downloads/python_interop.jl#L14))
  — never cached. Module-level `Ref` + lazy fill.

### 5.6 I/O patterns
- **`load_surface_window!(CubedSphereBinaryReader)` reads the full window**
  ([CubedSphereBinaryReader.jl:489](../../src/MetDrivers/CubedSphereBinaryReader.jl#L489))
  and discards everything except `.surface`. Add a section-skipping surface-only
  loader (`load_flux_delta_window!` already demonstrates the pattern at
  `:459`).
- **Double `load_window!(k+1)` per LL/RG replay validation iteration**
  (§1.8); fold the second load away.
- **`inspect_binary` JSON-parses the header twice** (one in `_peek_grid_type`
  and again when `TransportBinaryReader` is constructed) — minor for the CLI,
  but doubled per call from any batch tool.

### 5.7 Other
- **`R_dry = p.cp_dry / FT(3.5)` hardcoded in three places**
  ([DerivedKzField.jl:248](../../src/State/Fields/DerivedKzField.jl#L248),
  [GCHPHoltslagBovilleKzField.jl:88](../../src/State/Fields/GCHPHoltslagBovilleKzField.jl#L88),
  [WindowPBLKzField.jl:58](../../src/State/Fields/WindowPBLKzField.jl#L58)) —
  ideal-diatomic assumption (cp = 7/2 R) not in `PlanetParameters`. Risk of
  drift; one helper or one extra `PlanetParameters` field would centralize.
- **`DerivedKzField._recompute_kz_cache!` does two vertical passes**
  ([DerivedKzField.jl:263](../../src/State/Fields/DerivedKzField.jl#L263)) —
  pass 1 computes `z_col`; pass 2 computes Kz given `z_col`. Combine into one
  pass.
- **`_verify_rg_balanced_window!` returns a type-unstable NamedTuple** when
  `write_replay_on=false`
  ([reduced_transport_helpers.jl:1390](../../src/Preprocessing/reduced_transport_helpers.jl#L1390))
  — `diag.replay` is `nothing` vs typed NamedTuple. Use a zero-filled
  sentinel.

---

## Cross-Reviewer Top-10 Punch List

Ordered by leverage (impact × effort) for future cleanup work.

1. **Tape op vectors → typed union**
   ([TapeRecording.jl:248,411](../../src/Footprint/TapeRecording.jl#L248),
    [LinRoodTape.jl:347](../../src/Adjoints/LinRoodTape.jl#L347))
   — single highest-leverage type-stability fix. Affects every reverse loop.
2. **Pre-allocate LinRood substep face buffers** — `~690K` GPU allocations
   on C180 14-day adjoint runs ([LinRoodTape.jl:123](../../src/Adjoints/LinRoodTape.jl#L123)).
3. **`regrid_ll_binary_to_cs` migrate to unified Plan 41 driver** — closes the
   last architectural gap in the CS preprocessor; also unlocks
   `PreprocessorRunCache` sharing (30–60 s saved per day at C180).
4. **`Ref{Any}(nothing)` area caches → typed `Ref{Union{Nothing, Matrix{FT}}}`**
   ([WindowPBLKzField.jl:17](../../src/State/Fields/WindowPBLKzField.jl#L17),
    [GCHPHoltslagBovilleKzField.jl:19](../../src/State/Fields/GCHPHoltslagBovilleKzField.jl#L19)).
5. **`NoConvection.apply!(::CubedSphereState, ...)` missing** — `MethodError`
   on a "disable convection" CS run.
6. **CMFMC GPU correctness: `_cmfmc_dtrain_array` → `similar(cmfmc, ...)`**
   ([CMFMCConvection.jl:106](../../src/Operators/Convection/CMFMCConvection.jl#L106),
    `:120`).
7. **`parent(rm) isa Array` GPU dispatch → `get_backend(rm)`**
   ([StrangSplitting.jl:403](../../src/Operators/Advection/StrangSplitting.jl#L403)).
8. **CS palindrome halo timing asymmetry** — normalize forward and reverse
   exchange order; eliminates 1+ unnecessary exchange per palindrome.
9. **Symbol round-trip on `mass_basis`** — make `write_snapshot_netcdf`
   dispatch on `AbstractMassBasis`, drop both `BasisT → Symbol` conversions
   and the 6 `output_split` Symbol comparisons in DrivenRunner.
10. **Dead-code purge**: `preferred_era5_api`, `OPeNDAPProtocol.execute!`
    stub, `MERRA2Source`, `PressureTendencyClosure`/`NativeVerticalFluxClosure`
    stubs, ORD=4/6 PPM dispatch arms, `use_limiter` arg of `_sweep_z!`,
    `_cs_static_subcycle_count`, `_cmfmc_wellmix_subcloud!`,
    `write_day_binary!`, `_apply_surface_source!`.

---

## Doc-Generation Hooks

For future `docs/reference/` additions, the most impactful files to author or
update are:

- `docs/reference/FROM_GCHP.md` (new) — multi-tracer fusion comparison,
  Putman-Lin reference, convection-placement match.
- `docs/reference/FROM_TM5.md` — fix the `TiedtkeConvection` reference at
  `:20`; add a "Convection placement" row to the quick-reference table.
- `docs/reference/ADVECTION_SCHEMES.md` — add §3.2 invariants
  (palindrome budget formula, multi-tracer launch count, ORD=5/7 distinction,
  fillz count).
- `docs/reference/PREPROCESSING_GUIDE.md` — add §3.3 invariants
  (window-contract pairing, cm-closure point, per-window adaptive substep,
  GEOS PF endpoint rule, ERA5 ps pin, RG boundary-stub, surface flux units,
  `fill_dz_hydrostatic_constT!` requirement).
- `docs/reference/BINARY_FORMAT_V5.md` — schedule schema fields
  (`steps_per_window_by_window`, `poisson_balance_target_scale_by_window`,
  `time_step_schedule`, `raw_header`).
- `docs/reference/GRID_CONVENTIONS.md` — clarify `lon_shift_rad`, vertical
  level numbering, RG flattening convention, `PanelEdge.orientation` 0/2
  convention.
- `docs/reference/ARCHITECTURE.md` — record the TransportModel composition
  order and binary-scheduled-driver call pattern.
- `docs/memos/DIFFUSION_AND_CONVECTION.md` (new) — document
  `DiffusiveSurfaceFluxBoundary` ordering and the GCHP-VDIFF approximation
  caveat.
