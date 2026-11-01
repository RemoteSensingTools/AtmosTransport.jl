# Mass conservation

This page derives the discrete conservation contract that holds end-to-end
across the AtmosTransport pipeline (preprocessor → binary → runtime →
snapshot output) and quantifies the floating-point tolerances at which it
holds.

## The contract

For every cell `c` and every CFL-safe advection sub-step:

```math
m^{n+1}_c \;=\; m^n_c \;+\; \sum_f \sigma_{c,f}\,F^n_f
```

where `F^n_f` is the per-substep mass-amount through face `f` (units: kg per
substep, the binary's `flux_kind = :substep_mass_amount`) and `σ_{c,f} ∈
{+1, -1}` is the inflow / outflow sign convention for face `f` at cell `c`.
The same identity holds for every tracer mass `μ^n_{c, t}`:

```math
\mu^{n+1}_{c,t} \;=\; \mu^n_{c,t} \;+\; \sum_f \sigma_{c,f}\,\Phi^n_{f,t}
```

with `Φ^n_{f, t}` the tracer-mass flux through face `f`. Because `m` and
the tracer-mass slice live on the same staggering and use the same `F`,
`Φ` triple-pairing, the **mixing ratio `χ = μ / m` of a passive tracer
that starts uniform stays uniform exactly** under advection — modulo
floating-point.

## Telescoping divergence — why advection conserves total mass

For a closed periodic domain (lat-lon with periodic longitude) summing the
update over all cells gives:

```math
\sum_c m^{n+1}_c \;=\; \sum_c m^n_c \;+\; \sum_c \sum_f \sigma_{c,f} F^n_f
```

Each interior face appears in the sum twice — once with `+F` (the cell on
the inflow side) and once with `−F` (the cell on the outflow side) — so
the double-sum collapses to **boundary fluxes only**:

```math
\sum_c \sum_f \sigma_{c,f} F^n_f \;=\; \sum_{f \in \partial \Omega} F^n_f
```

For a closed sphere there is no boundary, so the right-hand side
identically vanishes mathematically — and to floating-point in the
implementation, provided both cells use the same face flux at the same
update stage. Writing each panel's update in flux form is insufficient
if neighboring panels reconstruct different tracer fluxes. The structured
X / Y / Z kernels in `src/Operators/Advection/Structured/multitracer_kernels.jl` use
shared face values. The telescoping argument does NOT hold
for schemes written in advective form (e.g. `χ ∂_t = -u ∂_x χ`),
which is why every advection scheme in this repository is flux-form.

The same telescoping argument extends to:

- Cubed-sphere Lin–Rood via shared final panel-edge tracer fluxes
  (`src/Operators/Advection/LinRoodSeams.jl`) and mirrored air-mass fluxes.
  The q-space Lin–Rood path requires CFL-safe inputs: its existing cell-local
  emergency flux scaling breaks conservation when activated.
- Cubed-sphere Upwind, Slopes, and PPM via one cached physical seam transfer
  applied to both neighbors in the same directional group
  (`src/Operators/Advection/CubedSphere/CubedSphereSeams.jl`). This also pairs rotated X/Y
  contacts. Halo exchange alone is insufficient; see
  [Panel-edge halo treatment (cubed sphere)](@ref).
- The reduced-Gaussian grid via the LCM-based ring-boundary face
  segmentation in `_boundary_counts(nlon_per_ring)` in
  `src/Grids/ReducedGaussianMesh.jl`. Each ring-pair boundary is split
  into `lcm(nlon[j], nlon[j+1])` segments so the outflow from ring `j`
  exactly equals the inflow to ring `j+1`.

## Vertical closure

The discrete continuity equation for the vertical mass flux `cm` is:

```math
m^{n+1}_c \;=\; m^n_c \;-\; 2\,\text{steps}\,(am_{e} - am_{w} \;+\; bm_{n} - bm_{s} \;+\; cm_{k+1} - cm_{k})
```

with the surface boundary `cm` at `k = N_z + 1` pinned to zero (no mass
flux through the ground). All three production preprocessing paths
produce `cm` via the **same diagnostic**:

1. **LL spectral preprocessor.** Runs the column FFT Poisson balance
   over the periodic-longitude grid (`balance_column_mass_fluxes!`),
   then calls `recompute_cm_from_dm_target!` to diagnose `cm` from the
   explicit `(am, bm, dm)` field.
2. **RG spectral preprocessor.** Runs the column Poisson balance on
   the compressed ring Laplacian (`balance_compressed_column_fluxes!`),
   then calls the same `recompute_cm_from_dm_target!` diagnostic.
3. **GEOS native CS preprocessor.** Column-balances horizontal mass
   fluxes against the raw next-hour dry endpoint
   (`balance_cs_column_mass_fluxes!`), then calls `diagnose_cs_cm!`
   (the cubed-sphere analogue of `recompute_cm_from_dm_target!`).

In every case the binary lands with a `cm` field that closes the
explicit-`dm` continuity equation to floating-point tolerance — the
**write-time replay gate** then verifies it before the binary file is
committed to disk.

The production GEOS-CS path does not use the separate FV3-style pressure-fixer
`cm` diagnostic; it closes against the raw dry endpoint described above.

## Dry-mass pin and the column global mode

The humidity-aware spectral pin adds one uniform surface-pressure offset,

```math
\delta p_s = \frac{p_{\mathrm{dry,target}}\sum_c A_c -
 \sum_c A_c\sum_k(\Delta A_k+\Delta B_k p_{s,c})(1-q_{c,k})}
 {\sum_c A_c\sum_k\Delta B_k(1-q_{c,k})}.
```

Native-layer dry-pressure and sensitivity sums and all three global reductions
use compensated Float64 summation. The small target-minus-current numerator
retains the compensated low parts and uses a fused multiply-add for the target
pressure times area. The pin uses local scalar accumulators, without scratch
allocations or shared mutable state.

Horizontal faces conserve global mass and cannot remove a remaining global
endpoint mismatch. Let the column residual be convergence minus endpoint
mass tendency, `R_c`, with arithmetic mean `R̄`, and let
`q_c = M_c / mean(M)` for current column mass `M_c = Σ_k m_{c,k}`.
For a fixed global dry-mass target, the column Poisson solve uses the shifted tendency
`T_c + R̄(q_c - 1)` and accurately projects its right-hand side to zero mean.
This is equivalent to distributing the shift across levels with fractions
`m_{c,k}/M_c`. It leaves `R̄ q_c`, rather than a constant kilogram residual in
every cell. Small polar cell budgets therefore do not amplify the global mode.
Vertical closure still uses the original endpoint tendency and the configured
layer weights. Raw residuals and the global mean remain diagnostic quantities.

Before applying this shift, a separate compensated global-budget guard requires

```math
\frac{\left|\sum_{c,k}(m^{\mathrm{next}}_{c,k}-m_{c,k})\right|}
 {\sum_{c,k}m_{c,k}} \leq \mathtt{GLOBAL\_DRY\_MASS\_BUDGET\_RTOL}=10^{-14}.
```

The expected global change is zero when the dry-mass pin is enabled. LL uses
the equivalent numerator `|2 n Σ dm|` from its per-application tendency, with
the actual window substep count `n`; RG and CS sum endpoint differences.
The tolerance allows roughly two orders of magnitude above the measured
post-pin Float64 roundoff of about `1e-16` relative per window (largest measured
changes: 666.004 and 659.033 kg out of about `5.1e18` kg). Lower-precision
stored endpoints allow the larger of this tolerance and `2 eps(FT)`, a bound
for rounding both endpoints. Reductions use the stored values in Float64.
The guard runs regardless of whether replay is enabled and uses no flux scale;
circulation and substep refinement cannot dilute a global-budget error.
A polar regression with `m_next = m*(1+1e-10)`, 100 substeps and divergence-free
zonal flux `am = 0.25m` per application is rejected; an `O(eps(Float64))`
mismatch passes. Replay alone can accept the former case because its
denominator grows with circulation.

For moist-basis or unpinned windows, a physical global mass change is permitted.
These paths neither assert a zero global budget nor apply the mass-proportional
shift. They retain the uniform nullspace treatment, original endpoint tendency,
and the existing local replay and CS closure checks. Conservative horizontal
faces cannot close a genuine global source or sink: it must be represented by
a physical boundary/source term or remain in the raw residual. Local checks
can reject an incompatible budget, but are not an independent global guarantee.

This treatment applies to LL FFT, RG PCG/direct and CS column solves. The
spectral pin applies to point-sampled LL/RG/CS and TM5 LL/RG construction.
The same CS column balance is used by spectral CS, LL-binary-to-CS regridding,
ERA5 N320 regridding, MERRA-2 and the GEOS endpoint-balanced, moisture-filtered and OMEGA closures.
Native sources retain their separate mass-pin routine. The separately callable
CS per-level balance retains its original target and arithmetic. The TM5 LL and
RG layer solves also retain their original arithmetic: compensated projections
are selected by separate dispatch only for column solves.

The production call sites are:

| Routine | Callers and coverage |
|---|---|
| `pin_global_mean_ps_using_qv!` | `construct_tm5_endpoint` (TM5 LL/RG), `apply_mass_fix_if_needed!` (point-sampled LL and spectral CS staging), `pin_convert_merge_window!` (point-sampled RG) |
| `balance_column_mass_fluxes!` | `apply_poisson_balance!` (point-sampled and TM5 LL) |
| `balance_compressed_column_fluxes!` | RG `balance_window!` (point-sampled), RG `process_day` for `TM5SpectralFluxes` (direct solver) |
| `balance_cs_column_mass_fluxes!` | `cs_spectral_contract_diag!`, `regrid_ll_binary_to_cs`, `process_era5_n320_to_cs_day`, MERRA-2 `_balance_window!`, and GEOS `_geos_prepare_window_for_steps!` (moisture-filtered, both OMEGA modes and endpoint-balanced branches) |
| `balance_cs_global_mass_fluxes!` | `balance_cs_column_mass_fluxes!` on single-level column buffers, with mass weighting only for fixed dry targets; standalone per-level callers retain original target and projection arithmetic |
| `balance_mass_fluxes!` | TM5 LL `balance_tm5_layers!` uses the original FFT RHS; column calls explicitly select compensated RHS projection |
| `project_mean_zero!` | TM5 RG `balance_compressed_level!` → `solve_compressed_poisson_direct!`, standalone RG/CS solvers: original `sum(v)/length(v)`; column solvers explicitly select compensated dispatch |
| `check_fixed_global_mass` | LL/RG/CS column entry points only when dry mass is pinned; spectral settings, native pin flags, or a dry regrid source's recorded pin enable it |

## Measured 1° dry-mass and replay correction

CPU validation uses ERA5 2021-12-01 on 360 × 180 cells, Float64 dry mass,
34 `ml137_tropo34a` levels, and all 24 windows (52,876,800 budgets per
construction). The point-sampled construction retains its default T179
truncation; TM5 retains T639. The after runs use the stock `1e-12` replay gate
and adaptive positivity checks. TM5 was regenerated through the binary writer;
point-sampled endpoints use the bounded hourly diagnostic driver with the
production pin, dry conversion, merge, balance and closure.

| Construction | median | p99 | p99.99 | maximum | count > 1e-12 |
|---|---:|---:|---:|---:|---:|
| TM5 before | 2.0295e-15 | 5.1465e-14 | 5.6883e-13 | 1.4709e-12 | 536 |
| TM5 after | 0.0000e+00 | 2.0675e-16 | 1.1806e-15 | 5.8233e-15 | 0 |
| Point-sampled before | 1.4122e-15 | 3.4638e-14 | 2.8047e-13 | 1.0639e-12 | 6 |
| Point-sampled after | 0.0000e+00 | 2.0077e-16 | 8.5414e-16 | 6.1071e-15 | 0 |

Largest absolute global mass change per window falls from 135,695.388
to 666.004 kg for TM5 and from 121,982.453 to 659.033 kg
for point-sampled construction. The remaining mass changes and endpoint target
errors are reported separately from the local replay metric; they cannot be
removed by conservative horizontal faces. Maximum relative endpoint target
errors after the change are 6.651e-17
(TM5) and 7.338e-17
(point-sampled), measured with compensated mass totals against the dry-pressure
area integral.

All spectral `pre_*` golden outputs are expected to change at rounding level.
The same column-mode treatment also applies to native-source and regrid
callers; it does not change the separate native mass-pin routine or the
Float64 `1e-12` / Float32 `1e-5` replay thresholds.

## Replay-gate tolerance

Conservation in floating-point is verified per window and per cell. With
`n` substeps per window, the stored per-substep face fluxes `F` of cell `c`
must carry its mass from one window to the next:

```math
\left| m_{c,n} - 2n \sum_{\text{faces}} F_{\text{out}} + 2n \sum_{\text{faces}} F_{\text{in}} - m_{c,n+1} \right|
\;\le\; \tau(\text{FT}) \left( |m_{c,n}| + |m_{c,n+1}| + 2n \sum_{\text{faces}} |F| \right)
```

The right-hand side is the sum of the magnitudes of the budget's terms,
which bounds the rounding error of evaluating the sum (Higham, *Accuracy
and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, ch. 4).
Measuring each cell against its own budget keeps the test as strict in the
thin top layers as in thick boundary-layer cells; a test normalised by the
largest cell mass would be 10⁴–10⁵ times looser there.

`replay_tolerance(FT)` (`src/MetDrivers/ReplayContinuity.jl`) returns the
constants of `src/Parameters/NumericalDefaults.jl`:

| FT | `τ(FT)` | worst current binary |
|---|---|---|
| `Float64` | `1e-12` | `6.5e-14` |
| `Float32` | `1e-5`  | `1.0e-6` |

Float64 binaries are limited by the residual of the Poisson flux balance;
Float32 binaries by Float32 rounding (about 9 `eps(Float32)`). The measured values cover the 2026-10 golden binaries
and samples of the C90 ERA5, MERRA-2 and C180 GEOS-IT production
binaries. A binary loaded in a different float type than it is stored in is
checked at the tolerance of the less precise type.

These replay gates check carrier-air continuity. They do not bound tracer
drift or establish consistency of reconstructed tracer fluxes at panel seams;
tracer conservation needs its own tests. Because `cm` is diagnosed from the
same horizontal fluxes just before the check, the gate certifies that the
stored payload is self-consistent, not that the winds are right: see
[What the replay gate certifies](@ref).

The gate fires twice in the lifecycle of a binary:

1. **Write-time** (on by default, in the preprocessor). A binary that fails
   is rejected at write time — the preprocessor errors out rather than
   producing a known-bad file. `[numerics] write_replay_check = false` is an
   explicit diagnostic escape hatch; do not use it for production binaries. A binary
   written that way records `write_replay_check = false` in its header.
2. **Load-time** (opt-in, in the runtime). Set `[input] validate_replay =
   true` (the `validate_replay` keyword of `TransportBinaryDriver`). Off by
   default because it doubles binary load time; recommended for any
   new binary configuration before a long production simulation.

## Implicit diffusion and roundoff

The precomputed cubed-sphere Dkg path solves backward Euler directly for
tracer mass. For exchange rate `D[k]` between layers `k` and `k+1`, define
`u[k] = dt*D[k]/m[k]` and `d[k] = dt*D[k]/m[k+1]`. The mass-space matrix has
diagonal `1 + u[k] + d[k-1]`, lower diagonal `-u[k-1]`, and upper diagonal
`-d[k]`. Its columns sum to one: a closed diffusion column conserves mass.

Ordinary Float32 Thomas elimination in VMR space can lose that cancellation
through coefficient rounding and mass/VMR conversion. The state-mass entry
point instead uses two column-conservative bidiagonal factors. Their inverses
are directed retention/transfer passes. Starting with `v[0] = 0`, their ratios
are

```math
r_k = \frac{u_k}{1+v_{k-1}}, \qquad
v_k = \frac{d_k}{1+r_k}.
```

The downward pass retains fraction `1/(1+r[k])`; the upward pass retains
`1/(1+v[k-1])`. Each passes the complementary mass to the adjacent layer.
This is a factorization of the same backward-Euler equation, not a correction
or rescaling of the tracer total. Factors are shared across all tracers.

The implementation computes the smaller partition directly, because subtracting
a rounded retention from one can erase a weak exchange into an empty layer.
Compensated incoming sums reduce loss of small layer contributions in stiff
columns. A removable constant background is chosen from the concentration
range's endpoint closest to zero; profiles spanning zero use zero. Layers
with no exchange retain their input values exactly. These operations need no
additional persistent workspace and use Float32 arithmetic on Float32 states.

The adjoint reverses the two passes with the same ratios. Tests cover signed
and positive tracers, weak transfers down to `dt*D/m = 1e-14`, stiff columns,
zero exchange, and constant total-mass gradients.

Kz diffusion fields (the lat-lon and reduced-Gaussian runs, and local-Kz
cubed-sphere runs) keep the Thomas solve, but only for the new mixing ratio
`g`. The tracer mass is then updated in flux form: each interface flux
`F[k] = dt*D[k]*(g[k] - g[k+1])` is computed once and applied with opposite
signs to layers `k` and `k+1`, so the column total telescopes. Writing back
`m[k]*g[k]` instead lost mass with one sign in `Float32`, because the rounding
of the diagonal `1 + u[k] + d[k-1]` is the same in every column of the
pure-pressure levels (2.0e-6 of a tracer per day in a lat-lon L72 run). The
remaining rounding goes through the same column ledger with persistent carry
as the Dkg solve ([Tracer Conservation in Single Precision](@ref)). The
VMR-level reference solve keeps the matrix form.
The conservation statement assumes positive carrier masses and closed column
boundaries; the existing zero-carrier sink convention is preserved separately.
Floating-point storage still introduces roundoff, so conservation of the
mathematical operator is not a promise of identical stored totals in every run.

On one C90 L66, 24-hour V100 PPM workload after paired advection seams,
this diffusion path reduces the maximum final Float32 relative tracer-total
drift from `8.17e-7` to `3.14e-7`. All six final Float64 compensated totals
match initialization; their maximum hourly relative drift is `1.98e-16`.
The exploratory Float32 target `1e-7` is not met. All six final Float32
column-mean fields are closer to the Float64 result. The initial conservative solve
costs `38.635` s median for the 32-tracer day; distributing CUDA tracer solves
across threads reduces this to `29.543` s with identical output and no new
persistent workspace. The earlier VMR solve with paired advection seams takes
`34.574` s.

Across seven consecutive daily files with the same setup, worst final relative
Float32 drift falls from `3.62e-6` with the earlier diffusion solve to `5.64e-7`.
All six final column-mean fields are closer to the Float64 reference, whose
largest daily total drift is `1.98e-16`. A 32-tracer week has the same maximum
final Float32 drift.

![Daily mass drift before and after conservative diffusion](../assets/weekly_mass_v100.svg)

Extending the conservative setup through all 31 December inputs gives a
maximum daily Float32 drift of `5.84e-7` on day 10 and a final drift of
`2.48e-7`. Float64 stays within `1.98e-16`. All first-week total series match
the earlier runs exactly. The declining Float32 endpoint reflects net error
accumulation and cancellation; it does not replace the worst-daily measure.
These daily samples cover one forcing sequence and do not bound within-day
maxima, other archives, or longer durations.

![Daily mass drift through one month](../assets/monthly_mass_v100.svg)

The native final tracer-storage comparison across all panels and layers has
maximum Float32-to-Float64 relative L2 difference `1.74e-6`. Both precisions
still produce small negative column means earlier in the month, about
`-1.46e-11` mol/mol. Conserved totals and finite output do not establish
positivity or reference-model field accuracy.

## Dry-basis vs moist-basis

The pipeline ships **dry-basis by default** (`mass_basis = :dry` in the
binary header). The mathematical content of the conservation law is
identical on either basis — what changes is the meaning of the stored `m`:

| `mass_basis` | `m` represents | Tracer VMR semantics |
|---|---|---|
| `:dry` | `m_dry = m_moist · (1 − qv)` per cell | dry VMR (`χ_dry = μ / m_dry`) |
| `:moist` | total air mass per cell | moist VMR (`χ_moist = μ / m_moist`) |

Conversions happen at the boundaries:

- **Preprocessing.** `apply_dry_basis_native!` in
  `src/Preprocessing/Spectral/mass_support.jl` multiplies cell-centered
  `m, dp, ps` and face-averaged `am, bm` by `(1 − qv_face)` with the
  appropriate face-averaging convention. After this step every payload
  field in the binary is dry.
- **Runtime.** `state.air_mass` carries dry mass. Tracer storage in
  `state.tracers_raw` is **mass**, not VMR; the dry-VMR contract is
  enforced at the IC boundary (uniform-value initial conditions
  interpret `4.0e-4` as dry VMR and convert to mass via `χ × m_dry` at
  construction) and at the snapshot-output boundary
  (`<tracer>_column_mean = column-integrated tracer mass /
  column-integrated air mass` is dry by construction).
- **Convection forcing.** The CMFMC and DTRAIN fields shipped by
  GMAO are moist-basis. The GEOS reader's
  `moist_to_dry_cmfmc!` / `moist_to_dry_dtrain!` apply the
  `(1 − qv_face)` correction so the convection operator consumes
  forcing on the same basis as `state.air_mass`.

**Mixing a moist binary with a dry-basis runtime is rejected at
`DrivenSimulation` construction time** — not at raw binary open, but
well before any windows actually step.

## Where the math meets the code

| Concept | File |
|---|---|
| X-sweep kernel (flux-form telescoping) | `src/Operators/Advection/Structured/multitracer_kernels.jl` |
| Strang palindrome (X→Y→Z / V/S/V / Z→Y→X) | `src/Operators/Advection/Structured/StrangSplitting.jl` (`strang_split!`) and `CubedSphereStrang.jl` (`strang_split_cs_mt!`) |
| Diagnose `cm` from `dm` target (LL / RG) | `src/MetDrivers/ReplayContinuity.jl::recompute_cm_from_dm_target!` |
| Diagnose `cm` from `dm` target (CS) | `src/Preprocessing/MassClosure/cs_poisson_balance.jl::diagnose_cs_cm!` |
| Column balance against next-hour endpoint (GEOS-CS) | `src/Preprocessing/MassClosure/cs_poisson_balance.jl::balance_cs_column_mass_fluxes!` |
| Write-time replay gate | `src/MetDrivers/ReplayContinuity.jl::verify_window_continuity_*!` |
| `replay_tolerance(FT)` | `src/MetDrivers/ReplayContinuity.jl` |
| Substep positivity gates | `src/Preprocessing/Targets/{CubedSphere,LatLon,ReducedGaussian}/*_contracts.jl` |
| Dry-basis correction | `src/Preprocessing/Spectral/mass_support.jl::apply_dry_basis_native!` |
| Basis-mismatch enforcement | `src/Models/Runner/DrivenSimulation.jl` (construction) |

## What's next

- [Advection schemes](@ref) — per-scheme order, monotonicity, CFL.
- [Conservation budgets](@ref) — what the test suite actually asserts.
- [Validation status](@ref) — what's been validated against external
  reference data; what hasn't.
