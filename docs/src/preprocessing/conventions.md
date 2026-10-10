# Conventions cheat sheet

The single-page reference for unit conventions, hard-coded constants,
auto-detected formats, and replay tolerances. Useful when you're
debugging an unexpected binary header or trying to consume an
external dataset that doesn't quite match the runtime contract.

## `mass_flux_dt = 450 s` — the FV3 dynamics step

GEOS / FV3 native `MFXC` and `MFYC` are accumulated over the
**dynamics timestep** (default 450 s for GEOS-IT C180), not the
1-hour met window. Treating them as per-second rates without dividing
by `mass_flux_dt` makes transport ~8× too slow.

| Where | What |
|---|---|
| `[preprocessing] mass_flux_dt_seconds` (source-descriptor TOML, e.g. `config/met_sources/geosit.toml`) | The scalar a user must set. Defaults to `450.0`. |
| `src/Preprocessing/sources/geos_read_window.jl::read_window!` | Where the division happens (`am ∝ MFXC / mass_flux_dt`); the GEOS-CS orchestrator then converts the per-second rate to the binary's `flux_kind = :substep_mass_amount` units before write. |
| `state.air_mass` after one window | Should remain `O(1)` relative to its starting value; an `O(8)` jump means the constant got missed. |

Spectral preprocessing does not have an equivalent constant — Holton
synthesis already produces per-second rates.

## Dry-basis conversion

`DELP_dry = DELP_moist · (1 − qv)` is applied at the cell-center
level; for face-staggered fluxes the dry factor is the average of
the two adjacent cell-center `(1 − qv)` values.

| Path | Function | Where |
|---|---|---|
| Spectral ERA5 (LL/RG/CS) | `apply_dry_basis_native!` | `src/Preprocessing/mass_support.jl` — runs after spectral synthesis, before binary write. |
| GEOS native CS | `endpoint_dry_mass!` (centers) + face-averaging | `src/Preprocessing/sources/geos_panels.jl` — runs at endpoint reconstruction. |
| GEOS convection forcing (cmfmc / dtrain) | `_moist_to_dry_cmfmc!`, `_moist_to_dry_dtrain!` | `src/Preprocessing/sources/geos_read_window.jl` — runs in the convection-read block. |

`MFXC` / `MFYC` from GEOS are special: per the GMAO product
documentation they are **already on a dry basis**. Don't apply the
correction twice.

## Level orientation

GEOS-IT files are **bottom-up** (level index 1 is the surface, level
index `Nz` is the top of the model). The runtime expects **top-down**
(level 1 = TOA, level `Nz` = surface). The reader auto-detects and
flips once at read time.

```julia
src/Preprocessing/sources/geos.jl::detect_level_orientation(ctm_a1)
    -> :bottom_up | :top_down
```

The heuristic compares `mean(DELP[:, :, :, 1, 1])` against
`mean(DELP[:, :, :, Nz, 1])` — the surface DELP is `O(1000 Pa)` and
the TOA DELP is `O(1 Pa)`, so the ordering is unambiguous.

NetCDF `lev:positive` attributes are not consulted in the current GEOS reader;
level orientation comes from the DELP-magnitude heuristic above. A non-GMAO
source for which that heuristic is invalid needs a dedicated reader method
instead of relying on this path.

ERA5 spectral output is already top-down; no flip needed.

## Panel conventions (cubed-sphere only)

| Convention | Symbol | Equatorial panels | North | South |
|---|---|---|---|---|
| Gnomonic | `:gnomonic` | 1, 2, 3, 4 | 5 | 6 |
| GEOS-native | `:geos_native` | 1, 2, 4, 5 | 3 | 6 |

The preprocessor writes the active convention into the binary header
as `panel_convention`. The runtime reads it back and constructs the
matching `PanelConnectivity` (panel-edge wiring used to rotate fluxes
across panel boundaries). **Picking the wrong convention silently
produces panel-edge artifacts** — every diagnostic tool emits the
panel layout schematic so this is easy to verify visually; see the
[Cubed-sphere section of the Grids page](@ref Grids) for the side-by-side
schematic.

## Units in the binary payload

| Section | Unit | Basis | Layout |
|---|---|---|---|
| `:m`        | **kg per cell** | matches `mass_basis` header | `(Nx, Ny, Nz)` LL / `(ncells, Nz)` RG / `NTuple{6, (Nc, Nc, Nz)}` CS |
| `:am`       | **kg per substep** (`flux_kind = :substep_mass_amount`; multiply by `steps_per_window` for per-window total) | matches header | LL: `(Nx+1, Ny, Nz)`; CS: `(Nc+1, Nc, Nz)` per panel (canonical + halo) |
| `:bm`       | **kg per substep** | matches header | LL: `(Nx, Ny+1, Nz)`; CS: `(Nc, Nc+1, Nz)` per panel |
| `:hflux`    | **kg per substep** | matches header | RG only — face-indexed `(nface_h, Nz)` |
| `:cm`       | **kg per substep** | matches header | extends `Nz` by 1 along the vertical axis |
| `:ps`       | Pa | dry if `mass_basis = :dry` | `(Nx, Ny)` LL etc. |
| `:dm`       | **kg per cell** | matches header | per-window mass tendency for the explicit-`dm` replay closure |
| `:qv_start`/`:qv_end` | kg / kg | (water-mass mixing ratio is basis-agnostic) | optional endpoint pair — set when `[output] include_qv = true` |
| `:cmfmc`    | kg / m² / s | dry-converted by the GEOS reader | NZ+1 interfaces |
| `:dtrain`   | kg / m² / s | dry-converted by the GEOS reader | NZ centers |
| `:entu`, `:detu`, `:entd`, `:detd` | kg / m² / s | dry | NZ centers, all four required together for `TM5Convection` |
| `:pblh`, `:ustar`, `:pbl_hflux`, `:t2m` | SI surface quantities | basis-agnostic | per-cell surface fields; all four together feed `WindowPBLKzField` |
| `:vdiff_u`, `:vdiff_v`, `:vdiff_t`, `:vdiff_qv` | m/s, K, kg/kg at substep cadence | dry where applicable | NZ centers, feeds `LocalHoltslagBovilleKzField` |

The transport-binary core fields (`:m`, `:am`, `:bm`, `:cm`, `:dm`)
are stored as **per-cell mass** and **per-substep mass amount**
(`flux_kind = :substep_mass_amount`), not per-area / per-second
fluxes. The runtime advection step consumes them directly without
unit conversion. Convection forcing arrays (`:cmfmc`, `:dtrain`,
`:entu`, …) come from external met physics and stay in their
native per-area-per-second units; the operator handles the
multiplication by `dt` at apply time.

!!! note "Current format only"
    Maintained writers emit transport-binary version 4 and maintained readers
    reject earlier formats. Use `inspect_binary` to verify data rather than
    inferring a format from a directory name.

## Replay tolerances

| FT | `replay_tolerance(FT)` |
|---|---|
| `Float64` | `1e-10` |
| `Float32` | `1e-4` |

Defined in `src/MetDrivers/ReplayContinuity.jl`. Used by:

- **Write-time gate** in every preprocessing path
  (`verify_storage_continuity_*!`, `verify_write_replay_cs!`).
  Failures abort the run.
- **Opt-in load-time gate** at runtime. Enable with `[input]
  validate_replay = true`, or by constructing the driver with
  `validate_replay = true`. Failures throw an `ArgumentError` pointing at
  the worst-cell location.

The asymmetry between F64 and F32 reflects the noise floor: F32
arithmetic at production resolutions accumulates rounding to ~`1e-5`
per substep, so a `1e-10` gate would false-positive on healthy F32
binaries.

## Per-source data layout cheat sheet

### ERA5 spectral

```
~/data/AtmosTransport/met/era5/
└── 0.5x0.5/
    ├── spectral_hourly/
    │   ├── era5_spectral_YYYYMMDD_lnsp.gb     # log surface pressure
    │   └── era5_spectral_YYYYMMDD_vo_d.gb     # vorticity + divergence
    └── physics/
        └── era5_thermo_ml_YYYYMMDD.nc          # specific humidity
```

### ERA5 N320 native GRIB

Vorticity, divergence, temperature and `lnsp` are spectral and are synthesized
at the `ReducedGaussianMesh` cell centres, `(i − ½) Δλ` in each ring. Specific
humidity and the convective mass fluxes are `reduced_gg` grid points whose
first point is at 0°, so point `i` sits at `(i − 1) Δλ`; the reader
interpolates them to the cell centres (`_reorder_grib_reduced_gg_to_mesh!`)
and rejects files whose first point is elsewhere.

### GEOS-IT C180 native

```
~/data/AtmosTransport/met/geosit/C180/native/
├── GEOSIT.YYYYMMDD.CTM_A1.C180.nc              # MFXC, MFYC, DELP (hourly)
├── GEOSIT.YYYYMMDD.CTM_I1.C180.nc              # PS, QV (hourly instantaneous)
├── GEOSIT.YYYYMMDD.A3mstE.C180.nc              # CMFMC (3-hourly)  — convection
└── GEOSIT.YYYYMMDD.A3dyn.C180.nc               # DTRAIN (3-hourly) — convection
```

The day after the requested range is also required (for the last
window's forward-flux endpoints).

## v4 header fields (preprocessor-set)

| Field | Required? | Notes |
| --- | --- | --- |
| `format_version` | yes | Must be `4` — every other version is rejected. |
| `steps_per_window_by_window :: Vector{Int}` | yes | Length `nwindow`. Per-window adaptive substep count (GEOS-CS) or constant (spectral). |
| `poisson_balance_target_scale_by_window :: Vector{Float64}` | yes | Length `nwindow`. Scale factor used by the per-window balance step. |
| `time_step_schedule` | yes | `"constant"` if all entries of `steps_per_window_by_window` are equal, else `"per_window"`. |
| `runtime_substep_contract` | optional | `"binary_schedule"` (CS only) — tells the runtime to advance using the per-window schedule. |
| `geos_mass_endpoint` | optional (CS GEOS only) | `"raw_dry_endpoint"` — records that the mass target is the raw GEOS dry endpoint, not the pressure-fixer endpoint. |
| `ps_offsets_pa_per_window` | optional (LL spectral only) | Per-window PS offsets applied by `pin_global_mean_ps!`. |
| `preprocessor_contract` | optional (CS only) | Free-form tag, e.g. `"plan41_variable_substeps"`. |

## What's next

- [Inspecting output](../getting_started/inspecting_output.md) — verify a freshly-built binary.
- [Concepts: binary format](../concepts/binary_format.md) — the full
  header schema and capability surface.
