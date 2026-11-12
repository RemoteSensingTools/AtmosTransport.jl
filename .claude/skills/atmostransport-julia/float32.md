# Single precision

`Float32` is the production precision on GPUs. The measurements and techniques are in
`docs/src/theory/float32_conservation.md`; read it before changing an update that must
conserve mass.

## What goes wrong

A `Float32` value has about seven digits. A change smaller than half an ulp of the stored
value is lost when it is added, and for a tracer with a large background (CO2 at 400 ppm)
the lost parts have one sign. An operator that conserves mass exactly in real arithmetic then
drifts.

## Techniques used here

- **Update by increments.** Compute the change of the mixing ratio `δ` and add `m * δ` to the
  stored tracer mass; do not rewrite the mass as `c_new * m`. A cell that should not change
  then stays bit-identical; the increments that rounding drops are what the column ledger
  returns. Measure the form: for Kz diffusion, adding the two face fluxes separately drifted
  less than adding their net (the theory page).
- **Flux form.** What leaves one cell enters its neighbour as the same rounded number.
- **Compensated sums** for totals and for sums of many terms: `_neumaier_sum(f, ks, T)`
  returns a pair `(s, c)` whose total is `s + c`; `_kahan_add` compensates `Float32` and
  `Float16` and is a plain add in `Float64`.
- **Column mass ledger** around an implicit solve or an update that rounds each layer: the
  residual between the compensated column sums before and after goes to the column's largest
  cell, and what that cell cannot hold is carried to the next call (`_ledger_residual`,
  `_ledger_deposit`). The ledger accepts only rounding-sized residuals, at most 16 n ulps of
  the largest cell with n the number of cells summed; a larger one is a real loss and must
  stay visible. It must not hide a scheme that does not conserve: the `Float64` update
  conserves to rounding without it.
- **Cumulative sums of stored fields do not close.** A flux rebuilt from entrainment minus
  detrainment does not return to zero exactly; end it structurally (at the last detraining
  layer), not by hoping the remainder is zero.
- **Geometry and constants in `Float64`, rounded once**; diagnostics of totals in `Float64`.
- **The same decisions in both precisions.** A threshold that makes a `Float32` run convect
  in other columns than the `Float64` run makes them different models.

## What a test must show

- Column or global mass before and after with exact sums (`BigFloat` or compensated), over
  many calls, not one. Drift must stay bounded, not grow with the number of calls.
- A uniform field over many calls on frozen and on changing forcing.
- `Float32` against `Float64` on the same case, with the difference stated.
