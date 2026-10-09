# GEOS native cubed-sphere

The GEOS path takes **GEOS-IT C180** native NetCDF — the FV3
dynamical core's own cubed-sphere output — and writes a transport
binary on the same cubed-sphere grid (no horizontal regrid). GEOS-FP
native C720 hourly CTM files use the same source contract, with one
file per UTC hour. It preserves native FV3 horizontal mass fluxes, balances
them against raw next-hour dry-air endpoints, and diagnoses the vertical mass
flux required by that discrete continuity target.

GEOS-FP C720 support covers the native hourly CTM layout
(`GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HH30.V01.nc4`; test fixtures may
also use `HH00`). Surface/convection physics from
the 0.25° GEOS-FP products can be attached by setting
`[source] physics_dir`, `include_surface = true`, and
`include_convection = true`; the day handle validates and embeds those
payloads into the same transport binary.

## How the native fluxes are balanced

The native `MFXC` and `MFYC` fields provide the starting horizontal mass-flux
amounts. After conversion to the transport grid, mass basis and time schedule,
they must close against the selected next-hour dry-air endpoint.

The current path uses `balance_cs_column_mass_fluxes!`: it integrates each
column vertically, applies a **global cubed-sphere Poisson correction** to
those column totals, and distributes the face corrections back over levels
using local air-mass weights. This avoids an independent correction at every
level while retaining the column budget needed for closed top and bottom
boundaries.

The preprocessor then diagnoses vertical mass flux with `diagnose_cs_cm!`
from the corrected horizontal divergence and explicit layer mass tendency.
The implementation and its correction policy live in
`src/Preprocessing/cs_poisson_balance.jl` and the native-source driver
`src/Preprocessing/transport_binary/cubed_sphere_geos.jl` (the per-window
closures in `geos_cs_window.jl` and `geos_cs_omega.jl` beside it).

The endpoint convention is the **raw dry endpoint** rather than the
endpoint implied by an FV3-style pressure fixer. The pressure-fixer
endpoint can go slightly negative in thin upper layers; the raw
endpoint is robust and the header records
`"geos_mass_endpoint" => "raw_dry_endpoint"` for traceability.

## Required input per day

| Collection | Cadence | Variables |
|---|---|---|
| `CTM_A1` | hourly (window-averaged) | `MFXC`, `MFYC`, `DELP` |
| `CTM_I1` | hourly (instantaneous) | `PS`, `QV` |
| `A1` | hourly | `PBLH`, `USTAR`, `HFLUX`, `T2M` *(GEOS-IT native or GEOS-FP physics fallback, only if surface is enabled)* |
| `A3mstE` | 3-hourly (window-averaged) | `CMFMC` *(only if convection is enabled)* |
| `A3dyn` | 3-hourly (window-averaged) | `DTRAIN` *(only if convection is enabled)* |

GEOS-IT file-naming convention: `GEOSIT.YYYYMMDD.<COLLECTION>.C180.nc`.
GEOS-FP native CTM file-naming convention:
`GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HH30.V01.nc4`. The
preprocessor needs the **next day's hour-0** for the last window's
forward-flux endpoint, mirroring the spectral path; the trailing peek
at `<end+1>` is unavoidable.

## GEOS preprocessing TOML

A working GEOS-IT C180 → CS C180 preprocessing config has two TOML
files: a small **preprocessing TOML** (per-run knobs) and a separate
**source descriptor TOML** (shared with the runtime, declares the
source's invariants):

```toml
# config/preprocessing/geosit_c180_to_cs180.toml
[source]
toml               = "config/met_sources/geosit.toml"   # source descriptor (below)
root_dir           = "~/data/AtmosTransport/met/geosit_c180/raw_catrine"
include_surface    = true     # PBL surface payload (pblh, ustar, hflux, t2m)
include_convection = true     # CMFMC + DTRAIN payload
include_vdiff_fields = true   # GCHP VDIFF payload (u, v, T, qv at substep cadence)

[output]
directory  = "~/data/AtmosTransport/met/geosit/C180/preprocessed/merge025hpa_adaptive_f32"
mass_basis = "dry"                             # binary header

[grid]
type                = "cubed_sphere"
Nc                  = 180
panel_convention    = "geos_native"
regridder_cache_dir = "~/.cache/AtmosTransport/cr_regridding"

[vertical]
transform    = "merge_above_pressure"
threshold_pa = 25.0                              # merge above 0.25 hPa → 64-level product
coefficients = "config/geos_L72_coefficients.toml"
# The default "identity" transform keeps all 72 levels; production
# C180 uses "merge_above_pressure" with threshold_pa = 25.0 (0.25 hPa)
# to fold the very thin upper-mesosphere layers into one, which makes
# the palindrome positivity budget feasible.

[numerics]
float_type            = "Float32"
dt_met_seconds        = 3600.0           # CTM cadence (hourly for GEOS-IT)
substep_schedule      = "adaptive_cfl"   # per-window adaptive substep count
substep_cfl_target    = 0.95
min_steps_per_window  = 2
max_steps_per_window  = 16
```

The source descriptor (referenced via `[source] toml = …`) declares
collection mappings and FV3-specific invariants. The actual
descriptor TOML is in `config/met_sources/geosit.toml`; the keys the
preprocessor reads sit under `[preprocessing]`:

```toml
# config/met_sources/geosit.toml (extract — full file is longer)
[preprocessing]
mass_flux_dt_seconds  = 450.0     # FV3 dynamics step — see Conventions
level_orientation     = "auto"    # bottom_up | top_down | auto
collections_required  = ["CTM_A1", "CTM_I1"]
collections_optional  = ["A3mstE", "A3dyn"]   # used when convection is on
```

The preprocessing TOML's `[source]` block can override
`include_surface`, `include_convection`, `include_vdiff_fields`, and,
for GEOS-FP, `physics_dir` / `physics_layout`. GEOS-IT reads native
A1/A3 files next to CTM_A1/CTM_I1. GEOS-FP reads native C720 CTM files
from `root_dir` and physics fallback files from `physics_dir`.

## Adaptive substep schedule

The CS GEOS-native path supports per-window adaptive substep counts.
When `[numerics].substep_schedule = "adaptive_cfl"`:

1. `_geos_select_steps_for_window!` runs up to 8 refinement iterations
   per window, evaluating the palindrome positivity budget
   `2·(out_x + out_y + out_z) / m_start` against `substep_cfl_target`.
2. The final per-window substep count is collected into a
   `steps_per_window_by_window :: Vector{Int}` of length 24.
3. `driver_before_close_writer!` patches the schedule into the streaming
   binary header via `set_streaming_steps_per_window_schedule!`.
4. The header carries `runtime_substep_contract = "binary_schedule"`,
   which tells the runtime to advance using the per-window schedule
   rather than a single scalar.

C180 production binaries use this. The scalar `steps_per_window` in
the header equals `maximum(schedule)`; the runtime reads
`steps_per_window_by_window` instead and gets per-window granularity.

## GCHP VDIFF preprocessing payload

When `[source].include_vdiff_fields = true`, the preprocessor writes
four extra payload sections:

| Section | Contents |
| --- | --- |
| `:vdiff_u`, `:vdiff_v` | Substep-cadence horizontal wind components |
| `:vdiff_t`, `:vdiff_qv` | Substep-cadence temperature and specific humidity |

These feed the `LocalHoltslagBovilleKzField` local Kz at runtime.
The binary's capability surface advertises `gchp_vdiff = true` once all
four are present. The runtime's `[diffusion].kind =
"geoschem_holtslag_boville_vdiff"` requires this capability.

## Per-window pipeline

For each of the 24 hourly windows:

1. **Read** (`src/Preprocessing/sources/geos.jl::read_window!`):
    - Open `CTM_A1` for hourly `MFXC`, `MFYC`, `DELP` (window-constant).
    - Open `CTM_I1` for instantaneous `PS`, `QV` at hour `n` and `n+1`.
    - Expose `MFXC` / `MFYC` as a rate-like diagnostic by dividing by
      `mass_flux_dt = 450 s`. CTM_A1 stores a dry pressure-area transport
      amount for one 450 s dynamics step, so the GEOS-CS writer stores each
      Strang half-sweep face flux as `MFXC / (2g)` and reuses that amount for
      the 8 substeps in the hourly window. The logged `flux_scale` is
      `450 / (2g) = 22.94361479`.
    - Auto-detect level orientation
      (`detect_level_orientation`): GEOS-IT files are **bottom-up**
      (k=1 surface), the runtime expects **top-down** (k=1 TOA).
      The reader flips once at read time so all downstream code sees
      the runtime convention.

2. **Endpoint dry-mass reconstruction**
   (`endpoint_dry_mass!`):
    - From `PS_total` + `QV` + the hybrid-σ-pressure coefficients,
      reconstruct `DELP_dry`, `PS_dry`, ensuring `Σ DELP_dry = PS_dry`
      to machine precision.

3. **Native MFXC/MFYC → v4 face flux layout**
   (`geos_native_to_face_flux!`):
    - GEOS-IT MFXC/MFYC are **already on a dry basis** (per the GMAO
      product manual). No `(1 − qv)` correction is applied.
    - Re-stagger from the FV3 layout (cell-centered values that
      represent the east / north face) into the v4 layout
      (`am[i+1, j, k] = MFXC[i, j, k]`).
    - Sync west / south halos via `_propagate_cs_outflow_to_halo!`.

4. **Raw dry endpoint target + column balance**
   (`balance_cs_column_mass_fluxes!`):
    - The next-hour raw GEOS `DELP_dry` endpoint is transformed with the
      same vertical plan as `m_cur`.
    - Native horizontal fluxes are column-balanced to that endpoint for the
      selected per-window substep count.

5. **Diagnosed vertical mass flux**
   (`diagnose_cs_cm!`):
    - The binary stores `dm = (m_next_raw - m_cur) / (2 * n_sub[win])`.
    - `cm` is diagnosed from the balanced horizontal fluxes and this `dm`, so
      runtime replay closes against the same raw endpoint used by the
      positivity gate.
    - With `chain_mass = true`, the raw endpoint is chained into the next
      window/day. With `chain_mass = false`, every window seeds `m_cur` from
      the raw GEOS start endpoint.

6. **Write-time replay gate** — same contract as the spectral path
   (`verify_write_replay_cs!`); failures abort.

7. **Cross-day chain** — when `chain_mass = true`, the day's `final_m`
   is threaded as `seed_m` for the next day's `process_day` invocation.
   When `chain_mass = false`, no cross-day seed is returned; each day/window
   is re-seeded from raw GEOS mass.

## GCHP-style convection wiring

When `[source] include_convection = true`:

1. The reader opens `A3mstE` and `A3dyn` (3-hourly NetCDF). If either
   collection is missing, the open fails loudly — there is no silent
   fallback.

2. **3-hourly hold-constant binding** — every hourly window `w`
   reads from A3 record index `(w − 1) ÷ 3 + 1`. Windows 1, 2, 3 see
   the same A3 record; windows 22, 23, 24 see the 8th. The dry-basis
   correction is still per-window (see step 3).

3. **Dry-basis correction**
   (`_moist_to_dry_dtrain!`, `_moist_to_dry_cmfmc!`):
    - GMAO ships `CMFMC` and `DTRAIN` as **moist-air** mass fluxes,
      kg / m² / s. The runtime needs them on the same dry basis as
      `state.air_mass`, so the reader multiplies by `(1 − qv_face)`
      using the window-mean QV (average of `t_n` and `t_{n+1}`
      humidity).
    - `CMFMC` lives at NZ+1 interfaces; the dry factor at face `k`
      is the four-corner mean of QV at the two adjacent centers
      averaged over the two endpoints. `DTRAIN` is at centers; the
      dry factor is the simple two-endpoint mean.

4. **Binary write** — the `:cmfmc` and `:dtrain` payload sections
   land in the binary; `inspect_binary` advertises
   `cmfmc_convection = true` once they're present. The runtime's
   `CMFMCConvection` operator picks them up via the
   `ConvectionForcing` carrier.

If `include_convection = false`, the orchestrator's per-window
NamedTuple omits the `:cmfmc` / `:dtrain` keys and the writer
no-ops them (the binary's `payload_sections` does not list them and
runtime convection is automatically gated off).

## Performance notes

- A 1-day GEOS-IT C180 → CS C180 F32 preprocess is ~1 minute on a
  recent workstation; with `include_convection = true`, add ~10 s per
  day for the A3 reads. `include_vdiff_fields = true` adds another
  ~10 s per day for the substep-cadence VDIFF fields.
- The column-balance + `diagnose_cs_cm!` path has **zero global
  Poisson iteration** — every step is a per-column scan, embarrassingly
  parallel. There's no diminishing return from adding cores.
- Real GEOS-IT data closes write-time replay at machine precision for
  each written window. Worst replay relative errors on real C180 days
  land around `5e-08` in F32.

## What's next

- [Regridding](regridding.md) — how the conservative weights are built
  and cached (relevant for cross-topology paths).
- [Conventions cheat sheet](conventions.md) — `mass_flux_dt = 450 s`,
  panel conventions, units cheat sheet.
