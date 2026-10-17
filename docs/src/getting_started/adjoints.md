# [Learning adjoints and inversions](@id Learning-adjoints)

Start here after the [Quickstart](@ref) and [State & basis](@ref). You only need
to understand that prescribed meteorology transports an emission field into
mixing ratios, and that the model stores mixing ratio times air mass.

## Three different questions

| Calculation | Question | Result |
|---|---|---|
| Forward model | What would these emissions produce at a receptor? | A modeled mixing ratio or other observation quantity. |
| Adjoint / footprint | How sensitive is that result to emissions in each cell and earlier step? | A derivative for each emission input. |
| Inversion | Which emissions best explain observations while respecting prior information? | An optimized control field for the chosen cost and constraints. |

A receptor is a location and time at which we evaluate the model. An
observation operator selects or averages model values to describe that receptor.
It might select one cell and layer, form a column mean, or apply a supplied
final-state sensitivity. A **scalar objective** is the one number whose
derivative we want. For a footprint it can be a receptor mixing ratio; for an
inversion it is usually a cost measuring observation and prior mismatches.

## First compute and check a footprint

Run the small, self-contained tutorial:

```bash
julia --project=. docs/literate/first_adjoint.jl
```

It moves emissions upward in a synthetic C3 column, computes sensitivity at a
receptor above the source, and checks an adjoint derivative against two
perturbed forward runs. Success ends with `FIRST_ADJOINT_PASSED`. Read the
[executed tutorial](@ref First-emission-footprint) for the arrays, units,
indexing conventions and numerical results. The first run includes Julia
compilation; later runs reuse compiled methods.

Writing the forward model as ``F(E)`` and the receptor operator as ``H``,
the tutorial uses ``J(E)=H(F(E))``. Its footprint ``g=\partial J/\partial E``
predicts a small change through

```math
\delta J \approx \sum_{t,p,i,j} g_{t,p,i,j}\,\delta E_{t,p,i,j}.
```

The reverse pass applies transposes of the recorded operator derivatives in
reverse order. It propagates sensitivities; it does not run winds backward or
undo numerical diffusion. One scalar objective gives sensitivities to many
emission inputs. Many independent receptors require additional reverse work;
the Jacobian API organizes that work.

The low-level footprint API holds meteorology fixed and differentiates with
respect to per-cell **model-storage emission rates**. Those are not directly
physical kg-species m⁻² s⁻¹. Convert inventory units, cell area and molecular
weight consistently. The tutorial explains the convention before constructing
its arrays. Footprints are not probabilities, source attribution fractions or
posterior uncertainty estimates.

## Then run a synthetic inversion

```bash
julia --project=. scripts/inversions/cs_4dvar.jl \
    config/inversions/example_synthetic.toml
```

This separate example uses zero transport fluxes to isolate the inversion
assembly. It starts from zero emissions, fits one synthetic surface-layer
observation, and prints initial/final cost, the observation/background cost
components, iteration count and final gradient norm. A successful optimization
reduces the total cost. One observation cannot determine a whole global field:
the covariance and prior influence the unobserved cells.

For emission controls ``x``, a conventional cost is

```math
\mathcal{J}(x)=\frac12\sum_k
\left(\frac{H_k(F(x))-y_k}{\sigma_k}\right)^2
+\frac12(x-x_b)^T B^{-1}(x-x_b).
```

Here ``y_k`` and ``\sigma_k`` are the observations and their uncertainties;
``x_b`` and ``B`` describe the prior and its covariance. The example's linear
preconditioner writes ``x=x_b+B^{1/2}\chi``, so the optimizer works with
``\chi`` and the background cost becomes ``\frac12\chi^T\chi``.

The gradient of the observation term needs one adjoint run, not one per
observation. With fixed meteorology each ``H_k`` is a linear function of the
tracer mass after its model step ``s_k``, with weights ``\mathbf{w}_k`` from
that step's air mass. A forward run gives every ``H_k(F(x))``; the reverse
pass then starts from zero at the last observation step and, at the end of
each step ``s``, adds the weighted departures of the observations sampled
there,

```math
\lambda_s = M_{s+1}^T\lambda_{s+1}
+ \sum_{k:\,s_k=s}\frac{H_k(F(x))-y_k}{\sigma_k^2}\,\mathbf{w}_k ,
```

where ``M_{s+1}^T`` is the transposed model step. Read at each step's surface
emissions, ``\lambda`` gives the gradient for all observations at once, as in
TM5-4DVAR. The [single-sweep gradient](@ref single-sweep-4dvar-gradient)
section of the adjoint status page has the details.

| Config section | Role in the example |
|---|---|
| `[mesh]`, `[time]`, `[meteo]` | C3 grid, two steps and synthetic constant meteorology. |
| `[advection]` | The scheme whose adjoint gives the gradient, parsed as in a forward run. The example uses unlimited PPM (`limiter = "none"`, accepted only by the inversion driver), which keeps the cost exactly quadratic; with zero fluxes the choice does not change the result. |
| `[[observations.entries]]` | Receptor indices, step, observed value and uncertainty. |
| `[control]` | One spatial field shared by named emission steps. `normalize=true` divides its contribution across those steps. |
| `[covariance]` | Prior-error scale and spatial correlation. |
| `[preconditioner]` | Map optimizer variables to physical control values. |
| `[optimizer]` | L-BFGS iteration and stopping settings. |

Try changing `observations.entries.value` or `covariance.sigma_value` in a
copy of the TOML and compare the two cost components. A smaller observation
uncertainty puts more weight on fitting the observation. A tighter prior
penalizes departures from the background more strongly. A lower cost is a
check of the optimization, not proof that inferred emissions are scientifically
correct or unique.

## What you can use today

The adjoint API covers supported **cubed-sphere** advection, diffusion and
convection paths. Check [Adjoint status](@ref) before changing the tutorial's
scheme or adding physics. In particular, optimized collaborative, truncated
and merged convection branches do not yet have a supported footprint reverse
pass. Nonlinear limiters require derivatives around a specified base trajectory;
finite differences that switch limiter branches need careful interpretation.

Recording every step is simplest for this tiny example. Long runs trade memory
for recomputation using `StrideCheckpoint(k)` or the bisection-based
`RevolveCheckpoint()`. Lin–Rood currently requires device tape storage. These
choices and the exact supported combinations belong in [Adjoint status](@ref).

The forward and inversion TOML files are **different interfaces**. Adding
`[adjoint]` to a forward run does not create an inversion, and the inversion
driver rejects forward-run tables such as `[diffusion]` or `[convection]`
that it does not read. The shipped inversion
CLI only assembles synthetic constant meteorology and inline observations;
real transport binaries and observation files require programmatic assembly.
`config/inversions/example_c48.toml` is a future-driver sketch, not a working
real-data recipe. Metal forward smoke tests have passed, but Metal adjoints and comparison
with TM5-4DVAR on real observations remain open.

Continue with [Adjoints and surface-flux inversion](../for_tm5_gchp_users/adjoints.md)
for the workflow mapping, or [Adjoints and checkpointing API](@ref) for exact
function signatures.
