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
tracer mass after the model steps it samples (one step ``s``, or the steps of
a time window with weights ``\omega_{ks}`` summing to 1), with weights
``\mathbf{w}_{ks}`` from those steps' air mass. A forward run gives every
``H_k(F(x))`` and records the tape; the reverse pass then starts from zero
at the last step with an assimilated observation and, at the end of each step
``s``, adds the weighted departures of the observations sampled there,

```math
\lambda_s = M_{s+1}^T\lambda_{s+1}
+ \sum_{k \text{ sampled at } s}\frac{H_k(F(x))-y_k}{\sigma_k^2}\,\omega_{ks}\,\mathbf{w}_{ks} ,
```

where ``M_{s+1}^T`` is the transposed model step. Read at each step's surface
emissions, ``\lambda`` gives the gradient for all observations at once, as in
TM5-4DVAR. The [single-sweep gradient](@ref single-sweep-4dvar-gradient)
section of the adjoint status page has the details.

| Config section | Role in the example |
|---|---|
| `[mesh]`, `[time]`, `[meteo]` | C3 grid, two steps and synthetic constant meteorology. |
| `[advection]` | The scheme whose adjoint gives the gradient, parsed as in a forward run. The example uses unlimited PPM (`limiter = "none"`, accepted only by the inversion driver), which keeps the cost exactly quadratic; with zero fluxes the choice does not change the result. |
| `[[observations.entries]]` | Receptor indices, step, observed value and uncertainty (or `[observations].file`, an observation file, below). |
| `[control]` | One spatial field shared by named emission steps. `normalize=true` divides its contribution across those steps. |
| `[covariance]` | Prior-error scale and spatial correlation. |
| `[preconditioner]` | Map optimizer variables to physical control values. |
| `[optimizer]` | L-BFGS iteration and stopping settings. |

Instead of inline entries, `[observations]` can read an observation file
(schema v2, written with `write_observations`). Point records (`kind =
"point"`: surface stations, towers, aircraft) are placed in the model column
as the forward observation sampler places them, by their height above the
ground or their pressure; column records (`kind = "column"`) are column
means, or, when they carry a retrieval block (the retrieval's layers,
pressure weights, averaging kernel and prior), averaging-kernel columns. The
file needs the model start time and a column of physical air mass, the
`synthetic_hydrostatic` meteorology:

```toml
[time]
nsteps = 3
dt_seconds = 600.0
start = "2024-06-15T00:00:00"

[meteo]
source = "synthetic_hydrostatic"
interface_pressures_pa = [1000.0, 30000.0, 70000.0, 100000.0]   # total, top first
specific_humidity = [1e-5, 2e-3, 1e-2]                         # per layer, kg/kg

[observations]
file = "observations.nc"
tracer = "CO2"
# flagged_records = "cosample"   # simulate and report records with a nonzero flag, never assimilate them

[observations.placement]
vertical_coordinates = ["height_above_ground", "pressure"]
# max_surface_pressure_mismatch_pa = 5000.0   # averaging-kernel columns; default no limit

# [observations.cell_averaging]   # column records of one model cell and overpass
# mode = "average"                # or "inflate_model_error"
# time_tolerance_s = 60.0
```

OCO-2 and OCO-3 records of the v11 MIP 10-second files become observation
files with `scripts/preprocessing/oco2_mip_to_observations.jl` (one file per
day; every record, those not to be assimilated with a nonzero flag), and are
compared with the model through the
[hat projection](@ref hat-level-projection) of their 20-level retrievals; see
[OCO-2 MIP records](@ref oco2-mip-records) for the converter, co-sampling and
cell averages.

Each sample reads the layer that contains it at its model step (a record with
`time_window_s > 0` averages the steps of its window), so a 10 m intake reads
the surface layer and a 5 km aircraft sample a layer near 540 hPa in the same
cell. The [observation operators](@ref placed-observation-operator) and
[averaging-kernel columns](@ref retrieval-column-operator) sections of the
adjoint status page give the equations.

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

Recording every step on the device is simplest for this tiny example. Long
runs pass a `tape = AdjointTape(; storage, checkpoint)`: host or on-disk
storage when the tape does not fit on the device, and `StrideCheckpoint(k)`
to trade one extra forward run for a tape of `k` steps. Lin–Rood currently
requires device tape storage. These choices and the exact supported
combinations belong in [Adjoint status](@ref).

The forward and inversion TOML files are **different interfaces**. Adding
`[adjoint]` to a forward run does not create an inversion (in the inversion
driver, `[adjoint.tape]` chooses the tape), and the inversion driver rejects
forward-run tables such as `[diffusion]` or `[convection]` that it does not
read. The shipped inversion
CLI assembles synthetic meteorology (constant, or a hydrostatic column) with
inline observations or an observation file; real transport binaries require
programmatic assembly (`bind_to_mesh` with a `CSObservationColumnMet` for
`cs_surface_flux_4dvar`).
`config/inversions/example_c48.toml` is a future-driver sketch, not a working
real-data recipe. Metal forward smoke tests have passed, but Metal adjoints and comparison
with TM5-4DVAR on real observations remain open.

Continue with [Adjoints and surface-flux inversion](../for_tm5_gchp_users/adjoints.md)
for the workflow mapping, or [Adjoints and checkpointing API](@ref) for exact
function signatures.
