> **Archived research note.** This dated investigation is preserved for
> scientific provenance. Paths, run status, and implementation details may
> have changed; use the maintained manual for current workflows.

> **Correction (2026-10-11).** The former 98,726 Pa target came from converting
> Trenberth and Smith's dry-air mass with constant gravity on a sphere. That
> conversion cannot reproduce their pressure because their estimate includes
> Earth's shape and varying gravity. All current paths instead use the measured
> shared modern-reanalysis dry mean, 98,325 Pa, from
> `Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE`. The config target key was
> removed, and the native correction is distributed by `Delta B (1-q)`.
# GEOS Preprocessing: Mass-Flux Balance and Global Dry-Air Conservation

> **Status (2026-05-29).** Global dry-air pins are **shipped** for both native
> CS-producing paths in commit `8646772` ("Pin global dry mass in native
> preprocessing"): **GEOS-CS** and **ERA5 N320→C180** both pin the **measured**
> dry-air mass (after the Q-based dry-DELP derivation) to the **same fixed
> absolute target** (`mode = "target_ps_dry"`, 98,325 Pa,
> ⟨dry mass⟩ = 5.114e18 kg on the IFS sphere). Sharing one target across the two paths is what
> makes GEOS-driven and ERA-N320-driven binaries mutually consistent (one
> dry-air baseline; see §7). Enable per config via `[mass_fix]`. The local
> mass-flux balance (column or per-layer) is orthogonal and unchanged. Per-day
> production-binary validation numbers will be filled into §4 once a pinned day
> completes.

This is the GEOS-cubed-sphere companion to
[`../memos/GLOBAL_MEAN_PS_FIX.md`](../memos/GLOBAL_MEAN_PS_FIX.md) (which
documents the already-shipped ERA5 spectral pin) and
[`FROM_GCHP.md`](FROM_GCHP.md) (concept mapping). Read both for context.

---

## 1. The one theorem that governs everything

On a closed sphere there is no domain boundary, so for any flux-form transport

```
∮ ∇·F dV = 0          (divergence theorem, no boundary)
```

and with the vertical boundary condition `cm(k=top) = cm(k=surface) = 0` the
same holds per column for the vertical mass flux. Three guarantees follow, and
they hold **independently of how good or bad the input meteorology is**:

1. **Global dry-air mass is conserved to roundoff** by any flux-form runtime
   that evolves air mass by flux divergence with no reset and no global
   rescale.
2. **Global inert-tracer mass = initial + ∫emissions to roundoff**, regardless
   of any inconsistency in the binary's stored air mass `m`. This is why a
   fossil-CO₂ burden tracks the imposed source integral even when the binary
   endpoint drifts.
3. **A flux-form model can never reproduce a globally non-conservative
   endpoint.** If the stored endpoint `m_next` has a different global integral
   than `m_cur`, no horizontal flux field can bridge that gap — its global
   divergence is zero by the theorem. The mismatch is forced into the
   write-time replay residual, not into runtime state.

Guarantee 3 is the crux of this entire page: **the Poisson/CG balance cannot
remove a global dry-air drift, and it is a category error to expect it to.**

---

## 2. Two independent problems, often conflated

"The GEOS run is not conservative" actually decomposes into two separate
issues that must not be confused:

### Problem A — local replay closure (binary-internal)

Does the stored flux field reproduce the stored per-window mass tendency,
cell by cell?

```
m_next - m_cur  ?=  -(∂am/∂x + ∂bm/∂y) - Δcm     (per cell)
```

This is what `diagnose_cs_cm!` closes and what the write-time replay gate in
[`../../src/Preprocessing/transport_binary/cubed_sphere_contracts.jl`](../../src/Preprocessing/transport_binary/cubed_sphere_contracts.jl)
verifies. The local mass-flux balance
(`balance_cs_column_mass_fluxes!` / `balance_cs_global_mass_fluxes!` in
[`../../src/Preprocessing/transport_binary/cubed_sphere_geos.jl`](../../src/Preprocessing/transport_binary/cubed_sphere_geos.jl))
is what tightens it.

### Problem B — global dry-air conservation (physics)

Is the global integral of the stored endpoint dry-air mass constant across
windows and days?

Dry air has no global sources or sinks, so the answer should be "yes, to
roundoff." Raw GEOS analysis says "no" — the global dry mass derived from
moist `PS` minus `Q` drifts ~1e-6/day because of analysis increments,
moisture conversion, and conservative-regrid error. This is **Problem B**, and
it is a *preprocessing* property of the stored endpoint, not a runtime leak.

**Problem A is solvable by the balance. Problem B is not** (theorem §1.3).
Conflating them leads to the trap of tightening the balance and being
surprised the global residual is unchanged.

---

## 3. The balance approaches (and one pin)

| Mode | What it does | Fixes A? | Fixes B? | Cost | Adjoint |
|---|---|---|---|---|---|
| **Raw GEOS** | Wind-derived fluxes, no balance | no | no | none | trivial (met-only) |
| **Column balance** | Per-column flux adjustment to match column-integrated `dm` | partial (column-integrated) | no | cheap | met-only, not differentiated for emission adjoint |
| **Per-layer reclose** | Global CG Leray projection per layer onto `∇·F = dm` | yes (local to ~1e-7) | no | ~1600 CG iter/window | iterative solve — harder to differentiate if winds were a control |
| **GCHP Cameron-Smith** | Algebraic zonal-band + meridional-mode + `dbk` vertical spread | column-integrated only | no (online avoids B) | cheap | algebraic, easy to differentiate |
| **Column balance + global pin** *(recommended)* | Column balance **plus** uniform `ps` offset pinning ⟨dry mass⟩ | yes | **yes** | cheap | met-only |

Key points:

- **Per-layer reclose is the L2-optimal local correction** (a true Leray
  projection): the minimum-norm flux adjustment achieving exact local
  continuity, isotropic, no zonal-band artifact. On polar-clustered
  N320→C180 geometry its isotropy is a genuine advantage over the
  Cameron-Smith zonal-band ansatz.
- **None of the local modes touch Problem B.** The pin is orthogonal: it acts
  on the global-mean subspace the projection discards as its singular mode.
- **The pin + a local balance compose.** The pin fixes the global mean; the
  balance fixes local divergence. They are independent subspaces, so it is not
  "pin vs reclose" — the only open question is whether *column* balance's
  local fidelity suffices or *per-layer reclose*'s tighter local replay
  (~1e-7) is needed for regional gradients. That is empirically decidable, not
  assumable.

---

## 4. Why the balance cannot fix the global mean (with measured proof)

Measured on the GEOS-IT C180 Dec 1 2021 binary (investigation 2026-05-29):

| Metric | Old binary | Per-layer reclosed |
|---|---|---|
| worst replay rel | 1.558e-6 | 1.153e-7 |
| day replay L1 | 2.238e13 | 6.042e12 |
| column L1 | 5.684e11 | 1.684e11 |
| **signed global residual (kg dry air)** | **-6.306445e11** | **-6.306458e11** |

Reclosing improved local replay 3–14× but left the global residual
**unchanged to 6 significant figures**. That is the divergence theorem (§1.3)
in measured form: the balance redistributes locally and cannot alter the
global integral.

The same investigation also showed the *runtime* is globally conservative,
once Float32 summation noise is removed:

| Metric | F32 | F64 |
|---|---|---|
| reclosed runtime global dry-air Δ over 24 h | -8.87e11 kg | **~+7.45e2 kg** |

The scary F32 −8.87e11 (~1.7e-7 relative) was global-reduction cancellation
precision, **not** a real flux-divergence imbalance — at F64 it collapses to
roundoff. So the reclosed fluxes remain globally divergence-free, the runtime
state remains globally conservative, and the unmatchable part of `dm` lives
entirely in the binary's replay residual where it never propagates into the
transported state. Fossil-CO₂ burden stayed source-consistent in both F32 and
F64.

**Conclusion: the runtime is correct; Problem B lives in the stored endpoint.**

---

## 5. Runtime contract (why the runtime is not the problem)

Verified in
[`../../src/Models/DrivenSimulation.jl`](../../src/Models/DrivenSimulation.jl):

- **Air-mass carry (default).** `reset_air_mass_each_window = false`
  (default). The prognostic `state.air_mass` is evolved purely by the stored
  fluxes; under the `:window_constant` contract its flux divergence integrates
  to `m_next - m_cur` over each window, so it tracks the binary endpoint
  without an explicit reset (`DrivenSimulation.jl:673-676`).
- **Advection denominator.** The tracer is divided by / re-multiplied by the
  *same* runtime air-mass field the fluxes evolve — so inert tracer mass is
  conserved to roundoff (`StrangSplitting.jl`).
- **Diagnostics denominator.** Column-mean VMR / XCO₂ use the runtime
  prognostic `state.air_mass`, **not** the binary's stored `m` — so the VMR
  diagnostic is internally self-consistent with the transported tracer.
- **No global rescale.** The runtime never forces global air mass to the GEOS
  endpoint. (The legacy runtime `mass_fixer` band-aid is retired.)

The optional `reset_air_mass_each_window = true` path resets air mass to the
stored endpoint **while preserving VMR**, which re-injects/removes tracer mass
at each window — see the policy fork below.

---

## 6. The forced policy fork (and how the pin dissolves it)

As long as the stored endpoint drifts (Problem B unfixed), these two are
mutually exclusive at the ~1e-6 level:

- **Policy A — conserve tracer mass (default).** Trust flux continuity. Tracer
  mass closes exactly (Σemissions = Δburden). Global dry air pinned constant
  by the scheme. Cost: the air-mass *trajectory* diverges from GEOS-Chem by
  the unremovable global-mean part.
- **Policy B — match GEOS-Chem trajectory** (`reset_air_mass_each_window =
  true`). Air-mass trajectory matches GEOS-Chem. Cost: preserving VMR across a
  mass jump breaks exact tracer-mass conservation.

**Pinning the global-mean dry-air mass in the preprocessor dissolves the
fork.** If the stored endpoint is itself globally conservative, the flux-form
runtime can reach it, and Policy A and Policy B converge — exact tracer mass
*and* a GEOS-matching trajectory simultaneously. This is exactly what the ERA5
spectral path already achieves with `pin_global_mean_ps!`.

---

## 7. The three pinned paths

There are now **three** offline paths that pin the global mean, plus GCHP which
doesn't need to. Two produce cubed-sphere binaries (**GEOS-CS** and
**ERA5 N320→C180**) and both pin *measured* dry mass to the *same* target —
that pairing is what makes their binaries mutually consistent.

| | ERA5 spectral → LL/RG/CS | **ERA5 N320→C180** | **GEOS-CS** | GCHP |
|---|---|---|---|---|
| Local balance | Poisson | column / per-layer | column / per-layer | algebraic Cameron-Smith (zonal-band) |
| Global dry-air pin | **yes** (`pin_global_mean_ps!`) | **yes** (`mode="target_ps_dry"`) | **yes** (`mode="target_ps_dry"`) | n/a (online, never stitches snapshots) |
| Pin quantity | total `ps` → dry via climatological ⟨q_v⟩=0.00247 | **measured** dry mass directly | **measured** dry mass directly | — |
| Fixed target | `Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE` | `98325.0` (shared) | `98325.0` (shared) | — |
| Native fluxes | reconstructed from winds | reconstructed from winds | native dry MFXC/MFYC | dry mass flux |
| `ps` after pin | uniform shift on `ps` | recomputed from pinned mass | recomputed from pinned mass | — |
| Endpoint source | spectral LNSP → ps | N320 PS − Q → dry mass (regridded) | moist PS − Q → dry mass | live GMAO stream at 450 s |
| Why pin needed | offline snapshot stitching | offline snapshot stitching | offline snapshot stitching | not needed — continuous budget |
| Cadence | hourly window | hourly window | hourly window | 450 s dynamics step |

**ERA-N320 and GEOS-CS are the consistent pair.** Both pin the *measured* dry
mass (each has the real per-column Q) to the *same* `98325.0` Pa target
(⟨dry mass⟩ = 5.114e18 kg on the IFS sphere), and both recompute `ps` from the pinned mass. So a
4D-Var inversion can mix ERA-N320-driven and GEOS-driven binaries with no
dry-air baseline jump in XCO₂. The ERA *spectral* path (the original
`pin_global_mean_ps!`) instead works on total `ps` and converts the dry target
with a climatological ⟨q_v⟩ — a ~12 Pa approximation the `GLOBAL_MEAN_PS_FIX.md`
memo flags as a deferred upgrade (§9 there); the two native CS paths do not need
that proxy because they have the gridded Q. And because GEOS MFXC/MFYC are
*natively dry*, there is no moist-`ps` flux reconstruction for the offset to
desync — the pin only shifts endpoint masses, the balance reconciles the
unchanged dry fluxes
against the pinned endpoints, and `ps` is recomputed from the pinned dry mass.

GCHP structurally avoids Problem B because it never reconstructs dry mass from
snapshots — its dycore carries a closed dry-air budget forward continuously.
Our offline design inherits Problem B and the pin is how we close it. See
[`FROM_GCHP.md`](FROM_GCHP.md) and the GCHP pressure-fixer analysis for the
full comparison (Cameron-Smith is a fast *online* fixer for a trusted stream;
it does not help an offline binary whose endpoint is unpinned, and its
zonal-band structure fits our polar-clustered target poorly).

---

## 8. Procedure

1. **GEOS-CS global dry-air pin** (`mode = "target_ps_dry"`). Compute the
   *measured* global dry-air mass after the Q-based dry-DELP derivation, then
   apply a uniform offset so it equals a fixed target
   `GLOBAL_MEAN_DRY_SURFACE_PRESSURE · A_header / g`. Recompute `ps` from
   the pinned dry mass. The correction in layer `k` is
   `Delta B_k Delta ps (1-q_k) A/g`; pure-pressure layers are unchanged.
2. **Keep the local balance.** Pin and balance are orthogonal; the pin fixes
   the global mean the balance discards as its singular mode. Because MFXC/MFYC
   are native dry fluxes, the pin only shifts endpoint masses; the balance then
   reconciles the unchanged fluxes against the pinned endpoints.
3. **Decide column vs per-layer empirically** from the regional-XCO₂-vs-GEOS-Chem
   comparison (§3): column balance + pin is the expected cost-efficient winner;
   per-layer reclose + pin is the maximum-fidelity option if regional gradients
   need the tighter local replay.

### Why a fixed absolute target, not "pin to first endpoint"

Pinning each day to its own first processed endpoint removes drift *within* a
run but leaves an arbitrary absolute baseline that depends on which day/window
started it — Dec 1 and Dec 6 would carry different global dry masses. A fixed
absolute target removes both the drift and the baseline ambiguity, which buys
two things beyond reproducibility:

- **Cross-dataset comparability.** Using the *same* 98325 Pa target for ERA and
  GEOS gives both driver families one shared dry-air baseline, so an inversion
  can mix ERA-driven and GEOS-driven binaries without a baseline jump in XCO₂.
- **Multi-day binary coherence — the precondition for the preprocessing
  architecture.** Every day in an archive shares one global baseline whether it
  is built serially-chained (GEOS `chain_mass=true`: day D's pinned final mass
  and day D+1's pinned first window agree globally, so the seed handoff carries
  no global-mean jump) or **day-threaded** (the ERA5 case — independently
  processed days only stay globally consistent under a fixed target; a
  first-endpoint policy would give each thread its own per-day baseline and
  silently inject day-boundary discontinuities). See the day-threading trait in
  `met_readers.jl` (`supports_day_threading`).

### Validation checklist for a pinned rebuild

| Check | Expected |
|---|---|
| Stored endpoint global dry-mass Δ/day | drops from ~5e12 kg → roundoff |
| Replay residual — global component | vanishes (pin removed the unmatchable part) |
| Replay residual — local component | unchanged (set by the chosen balance) |
| F32 **and** F64 burden vs source closure | unchanged (pin is denominator-side; if burden *moves*, something is wrongly coupled) |
| Regional XCO₂ vs GEOS-Chem | the column-vs-per-layer decision metric |

---

## 9. References

- ERA5 precedent (shipped): [`../memos/GLOBAL_MEAN_PS_FIX.md`](../memos/GLOBAL_MEAN_PS_FIX.md)
- GCHP concept mapping: [`FROM_GCHP.md`](FROM_GCHP.md)
- Advection mass-flux design: [`MASS_FLUX_EVOLUTION.md`](MASS_FLUX_EVOLUTION.md)
- Binary preprocessing architecture: [`BINARY_PREPROCESSING_ARCHITECTURE.md`](BINARY_PREPROCESSING_ARCHITECTURE.md)
- Code (pin shipped in commit `8646772`):
  - `src/Preprocessing/transport_binary/cubed_sphere_geos.jl` — GEOS-CS balance + cm closure; the shared pin helpers `_pin_cs_global_air_mass!` + `_ps_from_air_mass!`
  - `src/Preprocessing/transport_binary/era5_n320_regrid.jl` — ERA5 N320→C180 pin call sites (`pin_endpoint_mass!` after each `derive_c180_dry_mass!`)
  - `src/Preprocessing/transport_binary/entrypoint.jl` — `_native_mass_fix_target_kg` + `[mass_fix]` parse/wiring (shared by all native sources)
  - `src/Preprocessing/transport_binary/cubed_sphere_contracts.jl` — write-time replay + positivity gates
  - `src/Preprocessing/mass_support.jl`, `src/Preprocessing/transport_binary/latlon_workspaces.jl` — ERA5 spectral `pin_global_mean_ps!` (climatological-q_v variant)
  - `src/Models/DrivenSimulation.jl` — runtime air-mass carry policy
- External: Trenberth, K. E., and L. Smith, 2005: "The Mass of the Atmosphere:
  A Constraint on Global Analyses." *J. Climate* **18**, 864-875.

---

*Created 2026-05-29 from the GEOS-IT C180 Dec 2021 mass-conservation
investigation. Update in the same commit as any change to the GEOS-CS balance
or the (future) GEOS-CS global pin.*
