# Tracer Conservation in Single Precision

AtmosTransport runs production transport in `Float32`. Data-centre GPUs such
as the NVIDIA L40S execute `Float64` at about 1/64 of the `Float32` rate,
Apple Metal has no `Float64` kernels, and single precision halves the memory
and bandwidth of the tracer, air-mass and flux fields. `Float32` resolves
about seven significant digits, and a year of transport applies tens of
thousands of operator steps to every cell. This page describes how the model
nevertheless conserves global tracer mass in `Float32` to about 1e-6 of the
burden per year, how that was measured, which numerical techniques it relies
on, and what remains.

## How single precision loses mass

Round-to-nearest is unbiased for a single operation. Global tracer mass
drifts only when rounding errors share a sign across many cells and steps.
Three mechanisms caused such drifts in the model:

1. **Absorbed increments.** Adding less than half an ulp of the target leaves
   the target unchanged, and adding a few ulps rounds the same way every
   time. A surface flux added to a 400 ppm CO₂ cell, or a weak diffusive
   exchange, loses or gains the same fraction on every step.
2. **Near-identity linear solves.** Implicit convection and diffusion solve
   `(I + B) x = q` with `‖B‖ ≪ 1`. Elimination in `Float32` rounds the
   diagonal `1 + bₖₖ` and every update of a large `qₖ` by a small
   correction. Each solve then changes the column total by about `1e-8`, in
   proportion to the tracer mass and with a fixed sign.
3. **Geometry and constants rounded before use.** A cell area, regridding
   weight, decay factor or time step computed in `Float32` carries a fixed
   relative error. That error repeats on every application, for example in
   every area-weighted emission.

Flux-form advection applies each face flux with opposite signs to its two
cells, including across cube-panel seams. The global total therefore changes
only through the rounding of updated cell values. That rounding is mostly
noise, with a small residual bias (see [What remains](#What-remains)).

## How it was measured

Each run carries a uniform 400 ppm background tracer without sources. Its
global mass is invariant, so any change is arithmetic error. SF₆ starts from
its observed state. For tracers with emissions, a `Float32` run is compared
with a `Float64` run of the same code. Global totals are the
`<tracer>_total_mass` output variables, accumulated during the run in
`Float64` with compensated summation, independent of the storage precision.
Runs switch operators on one at a time. For the noise analysis, the 3-hourly
increments of a total are tested for a non-zero mean (`t` = mean divided by
its standard error).

| Archive | Levels, windows | Diffusion | Convection | Length |
|---|---|---|---|---|
| ERA5 | C90 L66, hourly | TM5 `dkg` | TM5, collaborative LU | 3 days, no sources |
| MERRA-2 (GEOS-Chem archive) | C90 L72, hourly | GEOS-Chem non-local VDIFF | CMFMC, DQRCU cloud base | 7 days, with sources |

`Float32` runs used an NVIDIA L40S and `Float64` runs an A100, starting
2021-12-01. In the `Float64` MERRA-2 runs the background total stays constant
to the last bit with every operator set. A `Float32` run starts 2.5e-8 below
its `Float64` twin because `Float32(400e-6)` is 2.5e-8 low. That offset is
constant and is removed from the comparisons below.

## Results

**ERA5, `Float32`, 3 days, no sources: relative change of global mass.**

| Operators | Background, before → after | SF₆, before → after |
|---|---|---|
| advection | +8.6e-9 | −1.2e-9 |
| advection + `dkg` diffusion | +9.0e-9 → +8.4e-9 | −1.24e-8 → +2.5e-9 |
| advection + TM5 convection | −9.7e-7 → +7.0e-9 | −9.9e-7 → −2.0e-9 |
| all operators | −9.7e-7 → +9.2e-9 | −1.0e-6 → +2.1e-9 |

**MERRA-2, 7 days with sources: `Float32` − `Float64`.** Background, CO₂
and SF₆ are relative to the global burden. Fossil CO₂ starts at zero and is
relative to the emitted mass.

| Operators | Background | CO₂ | SF₆ | Fossil CO₂ |
|---|---|---|---|---|
| advection, before | +8.0e-9 | −5.5e-9 | −4.6e-9 | +8.2e-7 |
| advection, after | +8.0e-9 | −4.2e-9 | −4.1e-9 | −4.0e-8 |
| all operators, before | −9.6e-9 | −2.4e-7 | +1.1e-7 | +8.5e-7 |
| all operators, after | −2.3e-9 | −2.0e-8 | −3.8e-9 | −1.9e-8 |

Over the week, the CO₂ burden changes by 3.0e-4 through net fluxes and the
SF₆ burden by 6.9e-4. Before the fixes, `Float32` therefore lost 8e-4 of the
net CO₂ flux, almost all of it in the non-local boundary-layer scheme
(advection + diffusion: −2.6e-7; advection alone: −5.5e-9).

**Individual mechanisms, measured directly.**

| Quantity | `Float32` before | After |
|---|---|---|
| C90 cell area, max. relative error | 1.3e-4 | 5.7e-8 (½ ulp) |
| C90 Δx, Δy, max. relative error | 1.8e-5 | 4.4e-8 |
| Rn-222 decayed fraction per step (Δt = 300–900 s) | up to 3.4e-5 | ≤ 3.2e-8 |
| GridFED → C90 regridding, F32 vs F64 global flux | +8.3e-7 | 6e-10 (rates rounded) |
| GridFED → C90 regridding, `Float64` loss | 2.1e-6 | 0 |
| Model clock, 7 steps per hour, after one year | 3.1 h behind | exact at window ends |

A `Float32` clock happens to stay exact over two years with 300, 400 or
900 s steps, but adaptive substepping can choose any step count; with seven
steps per hour it is 24 h ahead after two years.

## Techniques

### Kahan-compensated accumulation

Every surface source keeps a persistent per-cell compensation `c`. The
surface-layer update is

```
y = x − c;  t = s + y;  c = (t − s) − y;  s = t
```

so increments below the target's resolution carry forward instead of being
lost (Kahan, 1965). After `n` steps the field has gained `n·x + c`, up to
unbiased roundings of about `eps·x` per step; `c` is mass deposited ahead of
schedule (negative: behind).

### Error-free transformations

`TwoSum` (Møller, 1965; Knuth, 1969) returns `s = fl(a + b)` and the exact
error `e = (a + b) − s` using six `Float32` additions. GEOS-Chem's non-local
boundary-layer scheme deposits fresh emissions over several layers. Each
upper-layer addition is a `TwoSum`, so the mass that rounding kept out of a
layer is known. The surface layer receives the emission minus what actually
landed above, through the Kahan update. Each column thus gains the emitted
mass even when the profile fractions do not sum to one in `Float32`.

On NVIDIA GPUs, `ptxas` fuses a multiply and a dependent add into one FMA
(verified in the generated SASS for the deposit). The share `x·fₖ` may then
enter the sums unrounded, and the bookkeeping is exact to about ½ ulp of the
share per layer, with random sign.

### Column mass ledger

The TM5 convection matrix and the `dkg` diffusion exchange conserve column
mass exactly: their columns sum to one. Their `Float32` solves do not. Around
each solve, the ledger

1. sums the participating cells before and after with compensated summation
   (Neumaier, 1974, in the branch-free `TwoSum` form of Ogita et al., 2005);
2. forms the residual `(s₀ − s₁) + (c₀ − c₁)`. The solve changes the column
   total by far less than a factor of two, so `s₀ − s₁` is exact (Sterbenz's
   lemma);
3. adds the residual to the largest participating cell.

The residual is small against that cell: a few ulps for TM5 convection
(1–2e-8 of the active column) and about one ulp for `dkg` diffusion (≈5e-10
of the column). Spread over the column it would round away again. Added to
one cell it survives unless it is below half an ulp, which happens in 2–10 %
of TM5 columns and 20–30 % of `dkg` columns. For `dkg` the ledger therefore
removes the bias statistically, not column by column. The perturbation, a few
ulps of one cell, is below the solve's own rounding there.

The ledger corrects rounding, not physics. A residual larger than
`16·n·eps·|q_max|`, for `n` participating cells with largest value `q_max`, is
left in place. That bound is about 1e-6 of the column in `Float32`, far above
the measured residuals, so a matrix that does not conserve mass still shows up
in the budgets.

The ledger runs only where the operator conserves mass by construction. The
linear-algebra routines stay plain solves. TM5 convection uses the cloud
levels; `dkg` diffusion uses layers that exchange mass, leaves isolated
layers bit-exact, and keeps the absorbing convention of columns containing a
zero-mass cell. The adjoint's forward replay applies the same ledger, so
recorded states match the forward run.

A delta-form solve, `(I + B) Δ = −B q` followed by `q += Δ`, would remove the
bias at its source and also improve local accuracy. It needs the matrix
before factorisation for every tracer batch, which exceeds the shared-memory
budget of the collaborative GPU kernel.

### `Float64` geometry and constants, rounded once

Quantities computed once and used many times are evaluated in `Float64` on
the host and rounded to the run precision a single time:

- cubed-sphere corners, cell areas and edge lengths;
- conservative-regridding geometry. Meshes of either precision regrid on a
  `Float64` sphere and share cached weights. Source grids stored in
  `Float32` (GridFED) take their spacing from the full span, and extents
  within 1e-3 of a cell of ±90° or 360° snap to them. Otherwise the regridder
  misses a sliver of the sphere;
- the decay decrement `d = expm1(−λΔt)`, applied in kernels as `c + c·d`;
- the model clock, computed from the window and step counters, so window ends
  fall exactly on multiples of the window length.

Kernels never see `Float64` values. They stay type-stable in the run
precision, as Metal requires and the L40S rewards.

Preprocessing builds the same mesh. Binaries written with
`float_type = "Float32"` before this change used `Float32` cell areas, with
errors up to 1.3e-4 per cell at C90, 6.8e-4 at C180 and 9.7e-3 at C720
(global area −1.6e-8, −3.1e-7 and +2.1e-6), wherever the preprocessor
converts between pressure and mass. Such binaries should be regenerated,
especially at C180 and finer.

### Diagnostics in `Float64`

Totals meant to reveal `1e-8` drifts cannot be `Float32` reductions.
Snapshot totals, `total_mass` and the log totals share one compensated
`Float64` reduction: in place on CPU and CUDA, in bounded host slabs on Metal.

## What remains

**Advection.** The 3-hourly change of the background total has a standard
deviation of 5.5e-10. Its mean is +1.4e-10 (MERRA-2, `t` = 1.9) to +3.6e-10
(ERA5, `t` = 3.2), i.e. at most about +1e-6 per year. SF₆ shows no detectable
mean (`t` = −1.0). The cause of the bias for the uniform background has not
been identified. Removing storage rounding altogether would need double-single
tracer storage (two `Float32` words per value), doubling memory traffic. That
is not warranted at this level.

**CMFMC convection.** The remaining CO₂ difference with all operators
(−2.0e-8 of the burden per week, about −1e-6 per year) matches what CMFMC
convection added to advection alone before the fixes (−2.2e-8). CMFMC applies
its explicit flux-form update without a column ledger.

**Emission rates.** Rounding each cell's rate to `Float32` changes global
emission totals by less than 1e-9.

**Initial conditions.** Converting mixing ratios to tracer mass rounds once
(the 400 ppm offset above). Output fields are stored in `Float32`; budgets
must use the `_total_mass` variables, not sums of output fields.

**Not covered.** The ledger does not cover the VMR-form Thomas diffusion path
(local Kz fields), at about 2–3e-6 per month in `Float32`, or the clamped
CMFMC variant, at about 5e-6 per month. Neither is used by the production
configurations.

Global mass fixers, which rescale tracer fields to a target total, would hide
operator errors instead of removing them and are not used.

## Reproducing

`scripts/diagnostics/float32_conservation_attribution.py write-configs --out
DIR` writes the run configurations for both archives and precisions. After
the runs, `... budget --out DIR` prints the tables above: drifts, `Float32` −
`Float64` net of the initial offset (fossil relative to the emitted mass), and
the 3-hourly increment statistics.

`test/core/test_float32_conservation.jl` checks the error-free and compensated
sums, that the ledger leaves a leaking matrix visible, the geometry, the decay
decrement, the source-grid extents and the profile deposit.
`test/core/test_tm5_hessenberg.jl` and `test/helpers/conservative_dkg.jl`
require the convection and diffusion ledgers to conserve each column to within
one ulp of its largest cell. `test/core/test_multiday_source_clock.jl` checks
that the clock lands on window ends with seven steps per window.

## References

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*,
  2nd ed., SIAM, ch. 4.
- Kahan, W. (1965). Further remarks on reducing truncation errors.
  *Commun. ACM* 8, 40.
- Knuth, D. E. (1969). *The Art of Computer Programming*, Vol. 2,
  *Seminumerical Algorithms*, §4.2.2.
- Møller, O. (1965). Quasi double-precision in floating point addition.
  *BIT* 5, 37–50.
- Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur Summation
  endlicher Summen. *Z. Angew. Math. Mech.* 54, 39–51.
- Ogita, T., Rump, S. M., and Oishi, S. (2005). Accurate sum and dot
  product. *SIAM J. Sci. Comput.* 26, 1955–1988.
