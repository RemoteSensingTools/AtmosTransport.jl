```@meta
CurrentModule = AtmosTransport
```

# The binary pipeline

The transport binary is the **I/O contract** between preprocessing and
runtime. Everything else in AtmosTransport is downstream of this file.
This page describes how a daily binary is built, what guarantees it
carries, and why we trade disk space for an I/O model that bypasses
NetCDF entirely.

## What a daily transport binary actually is

```mermaid
flowchart LR
    H[JSON header<br/>padded to header_bytes]
    PAY[Flat payload<br/>fixed bytes per window]
    H --> PAY
    subgraph Header
        direction TB
        H1[grid_type, mass_basis]
        H2[A_ifc, B_ifc, nlevel]
        H3[dt_met_seconds, steps_per_window_by_window]
        H4[payload_sections]
        H5[poisson_balance_target_scale_by_window]
    end
    subgraph Payload
        direction TB
        P1[per window: m, am, bm, cm, ps]
        P2[+ optional: dm, qv_start/qv_end, cmfmc, entu/detu/entd/detd, dkg, ...]
    end
```

Preprocessing normally writes one file per day. JSON metadata comes first,
padded to the declared `header_bytes`; after that, every meteorological window
has the same section order and element stride. The reader memory-maps the flat
payload and computes `window_offset + k * elems_per_window`. Use
`inspect_binary(path)` or `scripts/diagnostics/inspect_transport_binary.jl`
instead of parsing padded header bytes by hand.

File size depends directly on grid resolution, vertical levels, precision,
window count, and optional sections. Inspect `payload_sections` before
comparing two products; a full-physics binary is intentionally much larger
than an advection-only binary.

## Why "one daily binary" instead of NetCDF

TM5's boundary archive and GCHP's MAPL ExtData layer are flexible interfaces
to collections of meteorological variables. AtmosTransport instead performs
that interpretation once during preprocessing. Runtime then sees one header,
one fixed section order, and one numerical schedule. The flat payload also
avoids chunk decompression during model stepping; it stores raw `Float32` or
`Float64` values.

The trade-off is **disk**. The transport payload is uncompressed so its
sections have fixed offsets and can be copied directly from the memory map.
Actual read time depends on filesystem, cache state, payload size, precision
conversion, and—on accelerators—the host-to-device copy. The runtime reports
those stages separately when `ATMOSTR_TIMERS=1`.

!!! tip "If disk is tight"
    Binaries may be compressed *at rest* with a general-purpose tool such as
    `zstd` and decompressed before a run. Compression ratio and decompression
    time vary with the fields and precision. The runtime itself requires the
    uncompressed version-4 file.

## The mass-conservation contract

Every binary that the runtime is willing to read satisfies the structural
version-4 contract. The unified preprocessor additionally runs numerical
positivity and replay checks before publishing a file. Runtime replay checking
is optional because it redoes work already performed by that writer; enable it
for imported, copied, or otherwise suspect files.

### Dry-basis cm closure

The vertical mass flux `cm[i,j,Nz+1]` is **explicitly** diagnosed from
the explicit `dm` (per-substep mass delta) field via
`recompute_cm_from_dm_target!` *after* the horizontal Poisson balance
runs. The fall-out invariant is

```
m[t+1] = m[t] + dm = m[t] + Δt · (∂xa + ∂yb + ∂zc)
```

with `dm` written to disk and `cm` reconstructed from it. This means the
runtime uses the preprocessor's explicit mass target rather than diagnosing a
new target from horizontal divergence at run time.

For GEOS-native cubed-sphere sources we additionally use the raw GEOS
`DELP_dry` endpoint as the mass target. The pressure-fixer's implied
endpoint can go negative in thin upper layers; the raw endpoint is
robust, the header records `"geos_mass_endpoint" => "raw_dry_endpoint"`,
and the column balance and `cm` diagnosis both target it.

### Write-time replay gate

For every written window, the preprocessor evolves `m_n` forward with the
stored flux fields and asserts, in every cell,

```
|m_evolved - m_stored[n+1]|  ≤  tol · (|m_n| + |m_stored[n+1]| + 2 n Σ_faces |F|)
```

with `tol = 1e-5` (Float32) or `1e-12` (Float64) and `n` substeps per
window. Output is staged under a temporary name; a failed product is removed
instead of being promoted to its canonical path.

Since `cm` is diagnosed from the same horizontal fluxes just before the
check, the gate reduces to the column mass budget that the balance has just
closed. It certifies a self-consistent payload, not correct winds: a
rotational flux error or a level flip of `am`/`bm` passes. See
[What the replay gate certifies](@ref).

### Per-window adaptive substeps

The header carries `steps_per_window_by_window :: Vector{Int}` and the
runtime reads it to set per-window substep counts. GEOS-native CS
preprocessing chooses each window's count adaptively from the
palindrome positivity budget — see
[Operators on top of the binary](operators_on_binaries.md#adaptive-substeps).
The v4 reader requires this schedule. Older formats are unsupported and must
be regenerated; loading does not repeat the write-time conservation check.

## How the two preprocessing paths build the binary

There are two production paths today: **ERA5 spectral** (mostly LL,
RG; CS via subsequent regrid), and **GEOS native** (CS only). Both
land in the same v4 binary schema.

### Path A — ERA5 spectral

ERA5 ships log-PS, vorticity, divergence, temperature, and humidity as
spectral coefficients on a Gaussian grid via the CDS API. The
spectral path turns those into mass-flux fields.

```mermaid
flowchart TD
    A1[CDS GRIB:<br/>vort, div, log_ps, T, qv] --> A2[Spectral synthesis<br/>per pressure level]
    A2 --> A3[Pin global-mean ps<br/>removes ERA5 mass drift]
    A3 --> A4[Build dry am, bm<br/>from divergence + ps]
    A4 --> A5[Poisson balance horizontal<br/>fluxes against dm]
    A5 --> A6[Diagnose cm explicitly<br/>from dm + balanced am, bm]
    A6 --> A7[Write window]
    A7 --> A8[Replay-gate the window]
    A8 --> A9{All windows<br/>complete?}
    A9 -->|no| A2
    A9 -->|yes| A10[Patch header,<br/>promote to canonical path]
```

Three checkpoints in this pipeline are load-bearing for TM5 users:

- **`pin_global_mean_ps!`** aligns each ERA5 window with the configured
  global dry-pressure target before mass fluxes are built. The JSON header
  records `ps_offsets_pa_per_window` for traceability.
- **Poisson balance.** Horizontal fluxes from spectral divergence have
  a non-zero divergence residual at the discrete grid level; we solve
  the topology-specific Poisson problem against the explicit mass tendency.
  Every topology (LL, RG, CS) corrects the vertically integrated column and
  distributes the correction over levels. TM5's `BalanceMassFluxes_m`
  balances each level instead, but its target includes the IFS vertical flux;
  without one, a per-level balance leaves no vertical mass flux. Vertical flux
  diagnosis follows balance.
- **`recompute_cm_from_dm_target!`** runs *after* balance, not before.
  Initializing `cm` from `divergence(am, bm)` before balance is the
  wrong dependency order because balance changes the horizontal divergence;
  post-balance closure is the required invariant.

A TM5-style construction of these fluxes is planned beside this path:
quadratures of the retained spectral fields along the target faces, TM5's
independent vertical flux, and a per-layer balance against the hybrid mass
tendency. [TM5-style spectral mass fluxes](../preprocessing/tm5_spectral_fluxes.md)
gives its equations, the differences from TM5, and the binary version-5 keys
that identify it.

### Path B — GEOS native CS

GEOS-IT (and eventually GEOS-FP) write hourly native cubed-sphere
NetCDF files with the dynamics-step-integrated mass fluxes (`MFXC`,
`MFYC`, `MFZ`) on the `mass_flux_dt = 450 s` substep. The native
path consumes those directly.

```mermaid
flowchart TD
    B1[GEOS-IT NetCDF<br/>DELP, MFXC, MFYC, MFZ] --> B2[Convert DELP_moist→DELP_dry<br/>using QV]
    B2 --> B3[FV3-style pressure-fixer<br/>against raw DELP_dry endpoint]
    B3 --> B4[Adaptive substep selection<br/>per palindrome positivity budget]
    B4 --> B5[Column balance dry fluxes]
    B5 --> B6[Diagnose cm from dm_dry]
    B6 --> B7[Write window + replay gate]
    B7 --> B8{All windows<br/>complete?}
    B8 -->|no| B1
    B8 -->|yes| B9[Patch steps_per_window_by_window<br/>schedule into header]
```

Two GCHP-relevant facts:

- **Adaptive substeps** are chosen by `_geos_select_steps_for_window!`
  using the palindrome positivity budget (see next page). The chosen
  count goes into `steps_per_window_by_window[k]`; the scalar
  `steps_per_window` is `maximum(schedule)`. The runtime reads the
  per-window vector, not the scalar.
- **Endpoint convention.** We use the raw GEOS dry-endpoint, not the
  pressure-fixer endpoint, as the mass target. This is documented as
  `"geos_mass_endpoint" => "raw_dry_endpoint"` in the header.

## Optional sections and their capabilities

A binary advertises its capabilities through `payload_sections`. The
runtime refuses to wire an operator that depends on a section that
isn't present.

| Section(s) | Operator unlocked |
| --- | --- |
| `:dm` plus topology-specific flux deltas | Endpoint interpolation and opt-in load-time replay checks |
| `:qv_start`/`:qv_end` | Specific-humidity endpoints for interpolation and moist-bookkeeping helpers |
| `:cmfmc` (+ optional `:dtrain`) | `CMFMCConvection` (GCHP-style) |
| `:entu`, `:detu`, `:entd`, `:detd` (all four) | `TM5Convection` (TM5 four-field updraft/downdraft) |
| `:pblh`, `:ustar`, `:pbl_hflux`, `:t2m` | Cubed-sphere PBL-derived diffusion fields |
| `:vdiff_u`, `:vdiff_v`, `:vdiff_t`, `:vdiff_qv` | `LocalHoltslagBovilleKzField` when the PBL surface fields are also present |
| `:dkg` | `PrecomputedCSDkgField` interface exchange |

The capability surface is queryable from Julia:

```julia
using AtmosTransport
caps = inspect_binary("/path/to/transport.bin")
# (advection = true, replay_gate = true, tm5_convection = false,
#  cmfmc_convection = true, surface_pressure = true, humidity = true,
#  mass_basis = :dry, grid_type = :cubed_sphere, ...)
```

The CLI `scripts/diagnostics/inspect_transport_binary.jl` pretty-prints
the same information and is the recommended first stop when a binary
behaves unexpectedly.

## How the runtime reads it back

```mermaid
sequenceDiagram
    participant FS as Filesystem
    participant MMAP as mmap
    participant RDR as Reader
    participant HOST as Host window
    participant RT as Runtime/backend loop
    FS->>MMAP: open + mmap full payload
    MMAP-->>RDR: virtual address
    loop per window k
        RDR->>RDR: offset = header_bytes + k * elems_per_window
        RDR->>HOST: copy/convert required sections
        HOST->>RT: use on CPU or copy to device buffers
        RT->>RT: run Strang palindrome (substep × steps_per_window[k])
    end
```

Four details matter:

- **The mmap is CPU-side storage.** Window loaders copy required sections into
  typed host arrays. That copy also converts precision when the configured
  runtime `FT` differs from the on-disk float type.
- **Per-window stride is constant.** `elems_per_window` is computed
  from the header at construction time. Walking from window `k` to
  window `k+1` is a single addition, regardless of which optional
  sections are present.
- **Cubed-sphere loading adds halos.** On-disk panels are unpadded;
  `load_transport_window` constructs the halo-padded runtime fields.
- **GPU runs perform an explicit backend copy.** Persistent device-side window
  buffers are refreshed from the host load. With multiple Julia threads, the
  next host window can be prefetched while the current window is computed.

The runtime side of the contract lives in:

- `src/MetDrivers/BinaryFormat/BinaryFormat.jl`
  (header schema and section-aware reader)
- `src/MetDrivers/BinaryReaders/cubed_sphere_reader.jl`
  (cubed-sphere geometry specializations)
- `src/MetDrivers/BinaryReaders/driver.jl`
  (window loop + replay-gate)

## Comparison with the TM5 and GCHP I/O models

| Concern | TM5 tm5-meteo archive | GCHP MAPL ExtData | AtmosTransport binary |
| --- | --- | --- | --- |
| Runtime input grouping | Boundary archive | ExtData collections | Normally one v4 file per day |
| Schema | Archive conventions | NetCDF attributes and connectors | Padded JSON header + fixed payload |
| Per-window operation | Archive reads | NetCDF/ExtData reads | Offset lookup + typed host copy |
| Mass-balance gate | TM5 mass_correction (write-time) | Per-operator at run time | Write-time + opt-in load-time |
| Compression | gzip (per-variable) | NetCDF DEFLATE | None at rest, optional zstd at rest |
| Accelerator transfer | Runtime-specific | Runtime-specific | Host window → persistent backend buffer |

The binary is the smallest possible commitment to "the runtime should
not have to think about I/O." Once you accept that one-line tenet, the
rest of the contract — dry basis, explicit `dm`, replay gate, per-window
schedule — follows.

## What to read next

- For the operator-side consequences of having `m`, `am`, `bm`, `cm`,
  `dm` on hand at every step, jump to
  [Operators on top of the binary](operators_on_binaries.md).
- For runtime data movement, prefetch, and profiling, see
  [Kernel architecture](kernel_architecture.md).
- For the on-disk schema details (every field, the JSON layout, the
  CS-specific extras), see [Binary format](../concepts/binary_format.md).
