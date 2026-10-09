# ERA5 spectral path

The ERA5 spectral path takes ECMWF model-level **vorticity (VO)**,
**divergence (D)**, and **log surface pressure (LNSP)** GRIB fields,
synthesizes mass-conserving wind / mass-flux fields on a target grid
via Holton-style continuity-consistent reconstruction, and writes a
transport binary (`format_version = 4`).

All three target topologies (LL, RG, CS) run through the unified driver
`run_unified_preprocessor_day!` with the source axis fixed to
`ERA5SpectralReader` and the target axis chosen via `[grid].type`.

## Required input

| File pattern | Contents | Hourly? |
|---|---|---|
| `era5_spectral_YYYYMMDD_lnsp.gb` | log(surface pressure) spectral coefficients | yes |
| `era5_spectral_YYYYMMDD_vo_d.gb` | vorticity + divergence spectral coefficients (137 model levels) | yes |
| `era5_thermo_ml_YYYYMMDD.nc` | specific humidity (for dry-basis conversion + mass-fix) | yes |

The thermo file is mandatory if `[output] mass_basis = "dry"` (the
default and only currently-tested binary basis for runtime
consumption); without it, the dry-basis correction `(1 − qv)` cannot
be applied. ECMWF distributes spectral data via CDS / MARS; the unified
downloader is `scripts/downloads/download_data.jl`, with ERA5 recipes under
`config/downloads/`.

The preprocessor **prefers** one extra day past the requested range
so the last window of each day can read its forward-flux endpoints
from the next day's hour-0 instantaneous fields. **All three target
paths fall back to a zero-tendency final-window closure** when the
next-day file is missing — the binary still builds, and replay
continuity holds, but the very last window of the last day uses an
implicit `dm = 0` rather than a real next-day delta. As a rule of
thumb, download `[start, end+1]` for production runs; a one-day development
job can use the fallback if its final-window tendency is not being analyzed.

## Targets

The spectral source supports all three target topologies; pick the
one that matches your downstream run:

| Target | Topology-specific workspace |
| --- | --- |
| `LatLonTargetGeometry` | `LatLonSpectralWindowWorkspace` (`src/Preprocessing/transport_binary/latlon_spectral.jl`) |
| `ReducedGaussianTargetGeometry` | `ReducedGaussianSpectralWindowWorkspace` (`src/Preprocessing/reduced_window_buffer.jl`) |
| `CubedSphereTargetGeometry` | `CubedSphereSpectralWindowWorkspace` (`src/Preprocessing/transport_binary/cubed_sphere_spectral.jl`) |

All three workspaces subtype `AbstractWindowWorkspace{G, FT}` and are
allocated once per day by `allocate_window_workspace`. The
`PreprocessorRunCache` (allocated once per run) holds the shared
expensive objects: the LL→CS conservative regridder, the RG
compressed-Laplacian factorization. These persist across days within
a single batch invocation.

The CS target adds a **lat-lon staging step** between spectral
synthesis and CS panels — spectral-to-CS-direct is mathematically
possible but the conservative regrid via an intermediate LL grid is
cheaper and matches the LL/RG paths' precision. The staging
resolution is configurable via `staging_nlon`, `staging_nlat` in the
`[grid]` block; the defaults `max(4·Nc, 360)` × `max(2·Nc + 1, 181)`
are what the LL bundle binaries use.

## Per-window pipeline

For each of the day's 24 hourly windows:

1. **Spectral synthesis** (`spectral_synthesis.jl::spectral_to_native_fields!`)
    - Reconstruct LNSP → PS via inverse spectral transform.
    - Reconstruct VO + D → (u_stag, v_stag) via the standard IFS
      streamfunction / velocity-potential decomposition, then to mass
      fluxes (am, bm) directly from the Holton continuity formula.
    - 137 native model levels are synthesized; the `level_top` /
      `level_bot` config window picks a slab.

2. **Vertical level merging** (`vertical_coordinates.jl::merge_thin_levels`,
   `select_levels_echlevs`)
    - The `[grid] echlevs` mode (e.g. `"ml137_tropo34"`) selects a
      vertical-level recipe — typically a tropospheric subset merged
      to ~34 layers via the TM5 r1112 echlevs scheme.
    - `merge_min_thickness_Pa = 1000.0` floors the merged layer
      thickness so no merged layer is thinner than ~1 km in the
      tropopause region.

3. **Mass-fix** (`mass_support.jl::pin_global_mean_ps_using_qv!`)
    - Optional, gated by `[mass_fix] enable = true`.
    - Computes a **uniform PS offset per window** so the global-mean
      dry surface pressure equals `target_ps_dry_pa`.
        - LL and CS targets use the **hourly QV field** for the
          offset.
        - RG targets use the constant **`qv_global_climatology`**
          scalar (the per-cell hourly QV is dry-converted later in
          the pipeline; the mass-fix step itself uses the
          climatology for cost reasons).
    - The preprocessor **applies the offset in place** while building
      the window — the binary's stored PS, mass, and fluxes are
      already pinned. The **LL writer** additionally records the
      per-window offsets in the binary header as
      `ps_offsets_pa_per_window` for diagnostic traceability; the
      **CS and RG writers** carry only `mass_fix_enabled` (and a
      handful of related metadata flags), no per-window offset
      vector. The runtime never re-applies the offsets in any case —
      there is nothing left to apply.
    - Without `[mass_fix]`, raw ERA5 spectral analysis drifts by tens
      of Pa per window, which compounds to noticeable
      day-boundary tracer-mass jumps over a multi-week run.

4. **Dry-basis conversion** (`mass_support.jl::apply_dry_basis_native!`)
    - Multiplies cell-centered DELP and PS by `(1 − qv)`; multiplies
      face-staggered mass fluxes by face-averaged `(1 − qv_face)`.
    - Applied **before** the binary write; the binary is dry-basis
      end-to-end.

5. **Poisson balance** — applied per target, with target-specific
   solver and finalisation:
    - **LL**: `apply_poisson_balance!` (in
      `latlon_contracts.jl`) uses an FFT-based Poisson solve over
      the periodic-longitude grid; followed by
      `recompute_cm_from_dm_target!` so the explicit-`dm` closure
      holds at write-time replay.
    - **RG**: a CG-based solve over the compressed face Laplacian,
      streaming previous-window state forward; the per-day binary
      is built window-by-window rather than accumulated.
    - **CS** (spectral source): a CG-based solve on the per-panel
      cubed-sphere Laplacian; cm is then derived via
      `diagnose_cs_cm!` rather than `recompute_cm_from_dm_target!`.
    - All three converge to the same current dry-basis tolerance and
      satisfy `‖m_evolved − m_stored_{n+1}‖/‖m_{n+1}‖ ≤
      replay_tolerance(FT)` at the write-time gate.

6. **Write + replay gate** — `verify_storage_continuity_*!` re-runs the
   forward stepper on the just-written window and checks
   `‖m_evolved − m_stored_{n+1}‖ / ‖m_stored_{n+1}‖ ≤ replay_tolerance(FT)`
   (`1e-10` for Float64, `1e-4` for Float32). Failures throw rather
   than letting the bad binary land.

## A worked LL preprocessing config

```toml
# Preprocesses 5° (72 × 37) ERA5 spectral → daily transport binary, F32.
# (config/preprocessing/era5_ll72x37_advresln_dec2021_f32.toml)
[input]
spectral_dir = "~/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly"
thermo_dir   = "~/data/AtmosTransport/met/era5/0.5x0.5/physics"
coefficients = "config/era5_L137_coefficients.toml"

[output]
directory  = "~/data/AtmosTransport/met/era5/ll72x37_advresln/transport_binary_v4_tropo34_dec2021_f32"
mass_basis = "dry"

[grid]
type = "latlon"
nlon = 72
nlat = 37
level_top = 1
level_bot = 137
echlevs = "ml137_tropo34"
merge_min_thickness_Pa = 1000.0

[numerics]
float_type   = "Float32"
dt           = 900.0
met_interval = 3600.0

[mass_fix]
enable                = true
target_ps_dry_pa      = 98726.0
qv_global_climatology = 0.00247
```

The CS variant adds two `[grid]` keys (`Nc`, `regridder_cache_dir`,
optionally `staging_nlon` / `staging_nlat`) and a different
`type = "cubed_sphere"`. See
`config/preprocessing/era5_cs_c24_transport_binary*.toml` for a
working example.

The synthetic-RG variant uses
`type = "synthetic_reduced_gaussian"` + `gaussian_number = N` +
`nlon_mode = "octahedral"`; see `era5_o090_transport_binary.toml`.

## Performance notes

- The slowest step is **spectral synthesis**: at LL 720×361 it's
  ~30 s per window F64 on a recent 8-core CPU, ~10 s F32.
  Multiplied by 24 windows × 30 days, a one-month F32 run is ~2 h.
- The CS variant's **conservative regrid** dominates at high `Nc`.
  `PreprocessorRunCache` builds the regridder once per batch run; the
  JLD2 cache (under `~/.cache/AtmosTransport/cr_regridding/`) makes
  the regridder persist across batch invocations.
- **Replay-gate verification** doubles the window's compute (one
  forward step per window). It is on by default; set the env var
  `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` to skip (not recommended for
  production binaries).
- All three target topologies (LL, RG, CS) run on F32 in production.

## What's next

- [GEOS native cubed-sphere](@ref) — the FV3-native path (no
  spectral synthesis; a column-integrated Poisson correction closes fluxes against
  the raw next-hour dry endpoint).
- [Regridding](@ref) — how the conservative weights are built and
  cached.
- [Conventions cheat sheet](@ref) — units, tolerances, panel
  conventions.
