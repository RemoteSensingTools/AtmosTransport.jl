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

The spectral path writes no convection sections. Its former
`[tm5_convection]` table (ERA5 convection NetCDF to TM5 fields on the
lat-lon grid) placed the fields 180° off in longitude and one hour early and
was removed; a config that still has the table is rejected. TM5 convection
comes from the ERA5 N320 → cubed-sphere path (`include_convection = true`)
or from the attached TM5-converted CATRINE archive.

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

For a cubed-sphere target, `[numerics]` defaults to
`face_fluxes = "vector"`, `face_lengths = "edge"`,
`face_interpolation = "fv3"`, and `wind_regrid = "cartesian"`. The staging
wind is converted to fixed Cartesian components before conservative regrid,
projected back to east/north at cube-cell centres, interpolated to the true
great-circle face, and projected on its normal. The stored half-substep mass
amount is

```math
F_f = \frac{\Delta t}{g}\,\ell_f\,(\mathbf{v}_f\cdot\mathbf{n}_f)\,\Delta p_f.
```

Alternatively `face_fluxes = "line_integral"` evaluates

```math
F_f = \frac{\Delta t}{g}\int_f (\mathbf{v}\cdot\mathbf{n}_f)\,\Delta p\,dl
```

from the staging-grid winds at 16 midpoints of each face. With the default
`flux_time_sampling = "window_mean"`, raw face amounts are constructed at both
window endpoints and averaged before column balance. Adaptive retries rebuild
that mean from the untouched endpoint amounts at the new half-substep duration;
they never rescale an already balanced field.

## Zonal truncation on each latitude ring

Every synthesis of a spectral field ``\hat f_n^m`` (triangular truncation
`T`, ECMWF's fully normalised ``\tilde P_n^m``) onto a latitude ring of `N`
equally spaced points ``λ_k = λ_0 + 2πk/N`` evaluates

```math
f(λ_k, φ) = G_0(φ) + 2\,\mathrm{Re} \sum_{m=1}^{M} G_m(φ)\, e^{i m λ_k},
\qquad
G_m(φ) = \sum_{n=m}^{T} \hat f_n^m\, \tilde P_n^m(\sin φ),
\qquad
M = \min\!\left(T, \left\lfloor \frac{N-1}{2} \right\rfloor\right),
```

as one unnormalised inverse FFT of length `N`. The limit `M` is
`ring_max_wavenumber` in `src/Preprocessing/spectral_synthesis.jl`; the
lat-lon `spectral_to_grid!`, the reduced Gaussian `spectral_to_ring!` and the
batched ERA5 N320 `synthesize_reduced!` all use it. Each wavenumber
`1 ≤ m ≤ M` occupies the two FFT bins `m` and `N − m`. On an even ring the
term `m = N/2` would occupy a single bin, and its samples
``e^{i N λ_k / 2} = e^{i N λ_0 / 2} (−1)^k`` make it depend on where the ring
starts: it is ``2\,\mathrm{Re}\,G`` at the cell faces (``λ_0 = 0``, where `u`
is synthesised) and ``−2\,\mathrm{Im}\,G`` at the cell centres
(``λ_0 = π/N``, where `lnsp` is synthesised). The term is therefore dropped.
This matches ecTrans's rule for linear reduced grids,
`NMEN = MIN(NSMAX, (NLOEN − 1)/2)`
(`src/trans/common/internal/setup_geom_mod.F90:48`, ecTrans `aff81b9`).

On lat-lon targets and the cubed-sphere staging grid `T = N/2 − 1`, so the
limit never binds. On a reduced Gaussian target `T` is set by the widest ring
(`max(nlon)/2 − 1`, 55 for O24), and each narrower ring drops its
wavenumbers above `M`. The ERA5 N320 reader synthesises at the file
truncation, T639, on rings of 18 to 1280 points.

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
    - `[grid] echlevs` names a TM5-style interface selection (see
      [Vertical level presets](@ref era5-vertical-level-presets)). The
      native layers between two selected interfaces are summed into one
      output layer.
    - Without `echlevs`, `merge_thin_levels` merges adjacent native
      layers until each is at least `merge_min_thickness_Pa` thick at
      standard surface pressure (1000 Pa = 10 hPa). With `echlevs` the
      key is ignored.

3. **Mass-fix** (`mass_support.jl::pin_global_mean_ps_using_qv!`)
    - Optional, gated by `[mass_fix] enable = true`.
    - Computes a **uniform PS offset per window** so the global-mean
      dry surface pressure equals
      `Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE` (98,325 Pa for every
      meteorological source).
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
    - **RG**: `balance_compressed_column_fluxes!`, a CG-based solve of
      the column budget over the compressed face Laplacian, streaming
      previous-window state forward; the per-day binary is built
      window-by-window rather than accumulated.
    - Every target balances the column-integrated fluxes and spreads
      each face's correction over the layers with `[numerics]
      column_balance_weights` (air mass in hybrid layers, `hybrid_mass`,
      by default), so `cm` keeps the vertical motion (see
      [Vertical transport](../theory/vertical_transport.md) section 2).
    - **CS** (spectral source): a CG-based solve on the per-panel
      cubed-sphere Laplacian; cm is then derived via
      `diagnose_cs_cm!` rather than `recompute_cm_from_dm_target!`.
    - All three converge to the same current dry-basis tolerance and
      pass the per-cell write-time replay gate (`replay_tolerance(FT)`).

6. **Write + replay gate** — `verify_storage_continuity_*!` re-runs the
   forward stepper on the just-written window and checks every cell,
   `|m_evolved − m_stored_{n+1}| ≤ replay_tolerance(FT) · (|m_n| +
   |m_stored_{n+1}| + 2 n Σ_faces |F|)` (`1e-12` for Float64, `1e-5` for
   Float32). Failures throw rather than letting the bad binary land. The gate
   checks that the payload is self-consistent, not that the winds are right;
   see [What the replay gate certifies](@ref).

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
qv_global_climatology = 0.00247
```

The target is deliberately not configurable per dataset. A stale
`target_ps_dry_pa` or `target_total_kg` key is an error naming the shared
constant.

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
  forward step per window). It is on by default; set
  `[numerics] write_replay_check = false` to skip it (not recommended for
  production binaries; the binary then records `write_replay_check = false`
  and the runtime warns about it).
- All three target topologies (LL, RG, CS) run on F32 in production.

## What's next

- [GEOS native cubed-sphere](@ref) — the FV3-native path (no
  spectral synthesis; a column-integrated Poisson correction closes fluxes against
  the raw next-hour dry endpoint).
- [Regridding](@ref) — how the conservative weights are built and
  cached.
- [Conventions cheat sheet](@ref) — units, tolerances, panel
  conventions.
