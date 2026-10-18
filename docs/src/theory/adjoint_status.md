# Adjoint status

This page marks the current boundary of the adjoint implementation: what can
be run and tested today, and which combinations are deliberately rejected.
The LinRood adjoint and TM5-style inversion scaffold are implemented and
covered by the core test suite, but real-data parity with TM5-4DVAR remains a
validation gap.

For a first introduction, start with [Learning adjoints and inversions](@ref Learning-adjoints)
and the [executed footprint tutorial](@ref First-emission-footprint). This page
is the support reference, not the starting lesson.

## What is shipped

### Forward operators (unchanged)

The forward transport model — advection (four schemes: `UpwindScheme`,
`SlopesScheme`, `PPMScheme`, `LinRoodPPMScheme`), convection (CMFMC +
TM5), implicit vertical diffusion (Beljaars–Viterbo / Holtslag–Boville
Kz fields), and surface flux sources — is implemented for the runtime's
supported topology/backend combinations and covered by the test suite documented in
[Conservation budgets](@ref).

### Tape, checkpoint, and reverse pass

`src/Adjoints/` ships a full discrete-adjoint pipeline for the
cubed-sphere split-sweep path, with a separate kernel-level pipeline
for `LinRoodPPMScheme` integrated through `cs_surface_emission_footprint`.

The shipped pieces, all exported through `AtmosTransport.Adjoints`:

| Surface | What it does |
| --- | --- |
| `cs_surface_emission_footprint` | Forward → tape → reverse pass driver for CS, returning `dJ/dE_t` at every surface step |
| `cs_surface_emission_footprint_from_seed` | Same, but accepts an explicit final `dJ/drm` seed for arbitrary observation operators |
| `cs_surface_flux_jacobian` | Batches layer/column objectives on one recorded tape (one reverse walk per objective), aggregates per-step footprints into named user windows |
| `cs_surface_flux_4dvar` | 4D-Var cost / gradient with named controls, scalar observations sampled at one step or over a time window (layer, column-mean, point samples placed in the column, and averaging-kernel columns), diagonal background; one taped run and one reverse pass per gradient, whatever the number of observations ([single-sweep gradient](@ref single-sweep-4dvar-gradient), [observation operators](@ref placed-observation-operator), [averaging-kernel columns](@ref retrieval-column-operator)) |
| `cs_surface_flux_4dvar_optimize` | Dependency-free gradient-descent driver for the above |
| `cs_surface_flux_4dvar_solve` | Production driver with optimizer choice (`CSGradientDescent` / `CSLBFGS`) |

The adjoint supports the following advection schemes (full union in
`CSAdjointSupportedScheme`):

- `UpwindScheme()`
- `SlopesScheme(NoLimiter())`
- `PPMScheme(NoLimiter())`
- `PPMScheme(MonotoneLimiter())` and `PPMScheme(CW84Limiter())` — via a stored tracer-branch tape around the base trajectory
- `LinRoodPPMScheme(5)` and `LinRoodPPMScheme(7)`, with either vertical
  sweep: FV3's `kord = 8` profile (`vertical = "fv3_kord8"`, the configured
  default, or `"fv3_kord8_signed"`) or upwind (`vertical = "upwind"`). The FV3
  sweeps are transposed at the limiter branches of the recorded state; see
  [the adjoint of the FV3 vertical sweep](@ref fv3-vertical-adjoint)

The split-sweep `PPMScheme` with the FV3 vertical profile has no adjoint;
neither do limited `SlopesScheme` and `NoAdvection`.
`require_adjoint_support(scheme)` returns a supported scheme and otherwise
throws an error that names the missing piece. The inversion driver applies it
to its `[advection]` table; a Lin–Rood table without `vertical` selects
`fv3_kord8` there as in forward runs.

The supporting kernel adjoints in
`src/Operators/Advection/linrood_adjoint_*.jl` are
transposition-tested (`test/core/test_linrood_kernel_adjoints.jl`) and
finite-difference VJP-tested via single-panel and cross-panel halo
compositions. The full LinRood footprint, including its two Upwind Z
half-sweeps, matches central differences to about `1e-11` relative on a smooth
base field. `test/core/test_cs_linrood_footprint_fd.jl` asserts `1e-9` with
upward vertical flux and a surface-layer objective, the case most sensitive to
which air mass the Z records hold. Every tape records the air mass entering a
Z sweep, because the upwind adjoint coefficient `F / m_donor` uses the
pre-sweep donor mass; the FV3 profile's records also hold the tracer mass
entering the sweep. `test/core/test_cs_linrood_fv3_adjoint.jl` checks the FV3
vertical sweeps' transpose in 256-bit arithmetic and the Lin–Rood footprints
with them against central differences (`1e-9`, upward and downward flux,
ORD 5 and 7, both profiles), and Full/Stride/Revolve parity.

The opt-in `test/diagnostic/test_cs_transport_adjoint_gpu.jl` also exercises
transporting PPM and LinRood footprints on CUDA with scalar indexing disabled.
It checks Float32/Float64 CPU/GPU agreement, Float64 directional finite
differences, and full/stride/revolve checkpoint parity, plus the single-panel
recording and reverse wrappers. LinRood copy-back and halo-gradient updates
run on the array backend; the tape path does not require host scalar reads.

The CS reverse pass also supports the default, unclamped/full-column/unmerged
forms of `TM5Convection`, `CMFMCConvection`, and `CMFMCMatrixConvection`.
CMFMC and CMFMC-matrix have column adjoint-identity tests; the TM5/CMFMC
footprint paths are checked against finite-difference emission gradients in
`test_cs_ppm_adjoint_footprint.jl`.

### Checkpoint schedules

`src/Tape/CheckpointSchedule.jl` ships three checkpoint policies,
selectable via the `checkpoint` kwarg of `cs_surface_emission_footprint`,
`cs_surface_emission_footprint_from_seed`, `cs_surface_flux_jacobian` and
`cs_surface_flux_4dvar`:

| Policy | Memory | Recompute |
| --- | --- | --- |
| `FullCheckpoint` | O(N) state snapshots | none — fastest reverse pass |
| `StrideCheckpoint(stride)` | O(N/stride) | replays each stride forward |
| `RevolveCheckpoint()` | O(log N) | recursive bisection; each step may be replayed once per recursion level |

`RevolveCheckpoint()` has no snapshot-budget argument. It is a recursive
bisection variant rather than the optimal binomial Revolve: peak snapshots are
proportional to the recursion depth and total replay work is
`O(N log N)`. It is covered in `test/core/test_cs_stride_checkpoint.jl`.

### Tape storage backends

Split-sweep tape records can live on the device (the default), in pinned-host
memory, or on disk via mmap — selectable via the `tape_storage`
kwarg (`:device` / `:pinned_host` / `:mmap`). Storage backends are
defined in `src/Tape/TapeStorage.jl` and `src/Tape/MmapTapeStorage.jl`;
the mmap path is covered by `test_cs_tape_mmap_roundtrip.jl`. LinRood tape
records currently require `:device` storage.

### Inversion scaffold

`src/Inversion/` ships:

| Module | Surface |
| --- | --- |
| `Covariance.jl` | `DiagonalCSCovariance`, `IsotropicGaussianCSCovariance`, `apply_B_half!`, `apply_B_half_adjoint!`, `apply_B_half_inverse!` |
| `Preconditioning.jl` | `apply_preconditioner!`, `LinearOptimType`, `LogNormalOptimType` |
| `Optimizer.jl` | `AbstractCSOptimizer`, `CSGradientDescent`, `CSLBFGS` (via multiple-dispatch surfaces) |
| `Jacobian.jl` | Footprint-Jacobian assembly for batched observation operators |
| `CostGradient.jl` | 4D-Var cost / gradient evaluator with preconditioning |
| `Observations.jl`, `ObservationColumns.jl`, `ObservationBinding.jl`, `ObservationsIO.jl` | Typed observation surface (time sampling, placed point objectives, column meteorology, stencils) + on-disk schema v2 |
| `DeparturesIO.jl` | Departures-file IO |

These are all on CI. Full inversion tests:
`test_cs_4dvar_single_sweep.jl`, `test_cs_4dvar_placed_observations.jl`,
`test_cs_4dvar_preconditioned.jl`, `test_cs_lbfgs.jl`,
`test_cs_inversion_driver.jl`, `test_cs_inversion_truth_recovery.jl`,
`test_cs_iteration_log.jl`, `test_cs_observations_io.jl`,
`test_cs_observation_binding.jl`, `test_cs_departures_io.jl`,
`test_cs_covariance.jl`, `test_cs_preconditioning.jl`,
`test_cs_optimizer_dispatch.jl`.

### [Single-sweep 4D-Var gradient](@id single-sweep-4dvar-gradient)

`cs_surface_flux_4dvar` computes the observation term of the cost and its
gradient with one taped forward run and one reverse pass, whatever the number
of observations, as TM5-4DVAR does.

**Observations as linear functionals.** An observation ``i`` is sampled after
the model steps ``s \in S_i`` of its time sampling (one step, or the steps of
a time window with weights ``\omega_{is}`` summing to 1; see
[observation operators](@ref placed-observation-operator)), after convection:
the state runtime output and the forward observation sampler see. The
meteorology is fixed, so the air mass ``m`` of every step is known before any
tracer is transported, and the observation operator is a fixed sparse linear
functional of the tracer masses ``\mathbf{r}_s`` at those steps,

```math
H_i(x) = c_i + \sum_{s \in S_i} \omega_{is} \sum_e w_{ise}\, r_s[e],
```

with weights from the replayed air mass at each ``s``:
``w = 1/\max(m, \epsilon)`` on one cell for `CSLayerMeanObjective` and for
the layer a `CSPlacedPointObjective` reads at that step (``\epsilon`` the
machine epsilon, the floor of its evaluation), ``w = 1/\sum_k m_k`` on every
level of a column for `CSColumnMeanObjective`, and ``c_i = 0``; an
averaging-kernel column (`CSRetrievalColumnObjective`) has
``w_k = H_k/\max(m_k, \epsilon)`` on every level and its prior term as the
offset ``c_i`` ([averaging-kernel columns](@ref retrieval-column-operator)).
Its gradient ``\partial H_i/\partial\mathbf{r}_s`` is
``\omega_{is}\mathbf{w}_{is}``. An instantaneous observation (one step,
``\omega = 1``) gives exactly the functional it had before time windows
existed.

**One reverse sweep.** Write ``M_s`` for the tangent-linear model of step
``s`` (the transport palindrome with the surface-flux center, then
convection) and ``\lambda_s`` for the adjoint state at the end of step ``s``.
The cost ``\mathcal{J}_\mathrm{obs} = \tfrac12\sum_i ((H_i(x) - y_i)/\sigma_i)^2``
has the gradient of a sum, so all observations share one recursion,

```math
\lambda_{N} = f_{N}, \qquad
\lambda_{s} = M_{s+1}^{T}\,\lambda_{s+1} + f_{s}, \qquad
f_{s} = \sum_{i:\,s \in S_i} \frac{H_i(x) - y_i}{\sigma_i^{2}}\,\omega_{is}\,\mathbf{w}_{is} ,
```

run from the last sampled step ``N`` back to the first step. Every
recorder ends each step with an end-of-step marker (`_CSStepEndRecord`, after
the convection record); the reverse walk adds ``f_s`` there, before it
transposes that step's convection. This is the TM5-4DVAR structure: its
forcing enters just before the adjoint of the operator after which the forward
run sampled. Where the walk reaches the midpoint of step ``t`` (the
palindrome center, where the step's surface emissions enter) it reads
``\Delta t\,\lambda`` as ``\partial\mathcal{J}_\mathrm{obs}/\partial E_t``,
and each control window sums its steps.

The forcing entries are grouped by step, panel and cell; the injection
kernel gives each cell one work item, which adds the cell's entries in input
order, so the injection is deterministic without atomics. With
`FullCheckpoint` one evaluation therefore costs a forward run for ``H(x)``
and the weights, one taped forward run to the last observation step, and one
reverse pass; `StrideCheckpoint` adds its checkpoint propagation pass and
`RevolveCheckpoint` its ``O(\log N)`` recomputation, as for a footprint. Steps
after the last observation do not affect the cost and get a zero gradient.
`checkpoint`, `tape_storage` and `tape_path` choose how that one taped run is
stored. `test/core/test_cs_4dvar_single_sweep.jl` checks the gradient
against one footprint per observation (the previous algorithm, rebuilt in the
test) for upwind, unlimited slopes and PPM, monotone PPM and Lin–Rood with
diffusion of both surface-flux couplings, TM5 and CMFMC convection and the
three checkpoint schedules, and for CW84-limited PPM and Lin–Rood with the
FV3 vertical sweep
(``10^{-12}`` relative in Float64, ``10^{-5}`` in Float32), and counts tape
reads to confirm one recording and one reverse pass per gradient.

`cs_surface_flux_jacobian` likewise records its tape once and walks it once
per objective (with `FullCheckpoint`); each walk is the one a separate
`cs_surface_emission_footprint` call makes.

### [Observation operators: point samples in the column and time windows](@id placed-observation-operator)

Observation files use schema v2 (`schemas/cs_observations_v2.toml`): Float64
coordinates, a `kind` (`"point"` for an in-situ sample, `"column"` for a
column average), the vertical fields `elevation_m`, `intake_height_m`,
`altitude_m` and `pressure_pa` (NaN when a record does not give them), a
`time_window_s`, a quality `flag` and a `model_sigma`. `bind_to_mesh` turns
each record into a `CSObservation`; its `sigma` is
``\sigma = (\sigma_\mathrm{obs}^2 + \sigma_\mathrm{mod}^2)^{1/2}``.

**Cell.** The record binds to the cell that contains its point (the
grid-box stencil), found with the forward observation sampler's locator, so a
record is sampled and inverted in the same cell. Float32 coordinates (schema
v1) could put a point within rounding of a cell edge into the neighbouring
cell; v1 files are rejected.

**Vertical placement.** A point record becomes a `CSPlacedPointObjective`,
placed once by the forward sampler's rule (`Output.resolve_placement`) with
the `vertical_coordinates` of the placement settings, in preference order
(default height above ground, then pressure):

- by its height above the true ground ``h``: `intake_height_m`, else
  `altitude_m − elevation_m`, else `altitude_m` minus a DEM elevation at the
  point. ``h`` is measured up from the model surface, whatever the model
  surface height: the model ground lies above or below the true ground,
  and a mountain station belongs in the model's lowest layers, not at its
  absolute altitude. Absolute altitudes are an explicit option
  (`absolute_altitude_above_agl_m`);
- by its pressure ``p_\mathrm{obs}`` (a total pressure).

A point record that gives none of the coordinates in use is unplaced: an
error, or dropped and counted with `unplaced_records = "skip"`. A column
record without a retrieval block becomes a `CSColumnMeanObjective`; one with
a block becomes an [averaging-kernel column](@ref retrieval-column-operator).

The layer is chosen at every sampled state ``s`` from that state's replayed
dry air mass ``m_k`` (kg per cell, ``k = 1`` at the top) and the column
meteorology (`CSObservationColumnMet`: model-top pressure
``p_\mathrm{top}``, gravity ``g``, layer temperature ``T_k``, specific
humidity ``q_k``), with the forward sampler's function
`Output.sample_column_layer!`. With the cell area ``A``, the dry interface
pressures and the hypsometric interface heights above the model surface are

```math
p^d_{1/2} = p_\mathrm{top}, \quad
p^d_{k+1/2} = p^d_{k-1/2} + \frac{g\,m_k}{A}, \qquad
z_{N+1/2} = 0, \quad
z_{k-1/2} = z_{k+1/2} + \frac{R_d T_k}{g}\ln\frac{p^d_{k+1/2}}{p^d_{k-1/2}} .
```

A record placed by height reads the layer with ``z_{k+1/2} \le h < z_{k-1/2}``
(the lowest layer for ``h \le 0``). A record placed by pressure reads the
layer with ``p_{k-1/2} \le p_\mathrm{obs} < p_{k+1/2}``, where the
interfaces are total pressures from the dry air mass and the humidity,
``p_{k+1/2} = p_{k-1/2} + g\,m_k / (A\,(1-q_k))`` (`pressure_basis =
"total"`, the default), or the dry ones (`pressure_basis = "dry"`, for
meteorology without humidity). The sampled value is the dry mixing ratio of
that layer, ``h_s = r_{s,k_s} / \max(m_{s,k_s}, \epsilon)``. The layer
depends only on the meteorology, not on the tracer, so it is fixed in an
evaluation and the functional weight ``1/\max(m_{s,k_s}, \epsilon)`` is the
exact derivative of the evaluated value: every vertical coordinate enters the
adjoint by construction.

**Time sampling.** Step ``s`` covers ``I_s = [t_0 + (s-1)\Delta t, t_0 +
s\Delta t)`` and is represented by the state at its end. A record without a
window samples the step whose interval contains its time ``t``
(``\omega = 1``). A record with a window ``W > 0`` samples

```math
\omega_s = \frac{|I_s \cap [t - W/2,\, t + W/2] \cap [t_0,\, t_0 + N\Delta t)|}
                 {\sum_{s'} |I_{s'} \cap [t - W/2,\, t + W/2] \cap [t_0,\, t_0 + N\Delta t)|},
\qquad H = \sum_s \omega_s h_s :
```

steps fully inside the window weigh ``\Delta t``, a step partly inside weighs
its overlap, and the window is truncated at the run's ends and renormalised.
A window narrower than a step inside one interval is the instantaneous
sample. Whether a record is in the run is decided by its time ``t``
(`out_of_range_policy`); `ClampOutOfRange` keeps a record outside the run:
the part of its window inside the run, truncated and renormalised, or the
first or last step when it has no window or the window misses the run
(departure rows mark such records `clamped`). Window records need the run
length (`nsteps`), which bounds their steps.

This time sampling differs from the forward observation sampler's: the
sampler takes its samples at met-window ends (a linear blend of the two ends
around the record time, or the nearest end), while the inversion takes the
state at the end of the model step containing the record time (up to one
step ``\Delta t`` after it) or the Δt-weighted mean over its window. Both
place a sample in the same layer of the same state and read the same value
there; for an arbitrary record time they sample different states.

`CS4DVarResult.stencils` returns, per observation, the cell, the sampled
steps and weights, the level read at each step, the placement code and the
height or pressure that placed it; `build_departure_set` joins them to the
records (by id) in schema-v2 departure files (`schemas/cs_departures_v2.toml`).

**Relation to TM5-4DVAR.** Matched: the sample is taken right after a fixed
operator (here the end of the step, after convection) and its forcing enters
just before that operator's transpose, in one reverse sweep; windows are
Δt/ΣΔt means with weights fixed at binding; the horizontal stencil is fixed
per observation (TM5 also offers slopes and trilinear stencils; ours is the
grid box, a type that can gain a trilinear form for both the sampler and the
inversion); the vertical stencil is chosen from the same meteorology forward
and in reverse. Different: points are placed by their height above the true
ground measured from the model surface (TM5 compares sea-level altitudes with
heights on the model orography and never samples layer 1); pressure placement
is in the adjoint too (TM5's adjoint uses the altitude only, so its gradient
of pressure-coordinate points is wrong); a step partly inside a window counts
its overlap (TM5 counts the half steps fully inside); the basis is dry; and
records carry their flag, elevation, intake height and pressure.

`test/core/test_cs_4dvar_placed_observations.jl` checks parity with the
forward sampler on one state (same layer and value through
`build_observation_sampler` in nearest-window mode, records sampled at the
window end that closes their model step, and through `bind_to_mesh` +
`cs_surface_flux_4dvar`), a
10 m intake against a 5 km aircraft sample, total against dry pressure
placement, layers that follow the replayed air mass step by step, a window
observation's value and gradient against the Δt-weighted sum of its
single-step observations (``10^{-12}``), the single sweep against
per-observation footprints for placed and window observations (``10^{-12}``,
upwind, monotone PPM and Lin–Rood, TM5 convection, Full and Stride
checkpoints), and central finite differences of the cost.

### [Averaging-kernel columns](@id retrieval-column-operator)

A retrieved column-averaged dry mole fraction (OCO-2, GOSAT, TROPOMI, TCCON)
is not the column mean of the true profile: the retrieval weights the
atmosphere with its averaging kernel and fills in its prior where it has
little sensitivity. A column record that carries a **retrieval block**
(schema v2, `schemas/cs_observations_v2.toml`) is compared with the model
through that retrieval's own vertical description, with the operator of
TM5-4DVAR (`deps/tm5-cy3-4dvar/base/f2py/src/averaging_kernels.F90`, the OCO-2
routine `applyaveragingkernel` and the rebin `inv_rebin`). A block describes
either layer means (`retrieval_mapping = "layer_rebin"`, this section) or
values on levels with the profile linear between them (`"hat_levels"`, the
OCO-2/OCO-3 retrievals; see [level retrievals](@ref hat-level-projection)).
A layer block gives the retrieval's ``N`` layers from the surface up:

- interfaces ``p^r_1 > p^r_2 > \dots > p^r_{N+1} \ge 0`` (Pa, total pressure),
  with ``p^r_1 = p^r_s`` the retrieval surface pressure and ``p^r_{N+1} =
  p^r_t`` its top;
- pressure weights ``h_i \ge 0`` with ``\sum_i h_i = 1`` (checked to
  ``10^{-4}``: a deficit ``\delta`` would bias the column by ``\delta X``);
- the column averaging kernel ``a_i``, normalized, so that
  ``\partial X/\partial x_i = h_i a_i`` on the retrieval layers;
- the prior profile ``x_{a,i}`` (dry mole fraction, mol mol⁻¹; files state
  the unit, ppm and ppb are converted). The operator adds it to the model's
  dry mole fractions, so the record value and errors and the transported
  tracer are in mol mol⁻¹ too: such a file states a mole-fraction unit on
  `value`, `value_sigma` and `model_sigma` (converted to mol mol⁻¹), and a
  record must be an absolute column average, a value in (0, 1] with errors
  not above it;
- optionally `retrieval_profile_sigma`, an uncertainty ``\sigma_i`` of the
  *model* profile on the retrieval layers (mol mol⁻¹), for the model-error
  term below.

`bind_to_mesh` turns such a record into a `CSRetrievalColumnObjective`. At
every sampled state the model's interface pressures ``p_{k+1/2}`` of its cell
are built as for pressure placement: total pressures from the replayed dry air
mass and the specific humidity, ``p_{k+1/2} = p_{k-1/2} + g\,m_k/(A(1-q_k))``
(`pressure_basis = "total"`, the default; retrieval pressures are total
pressures), or the dry interfaces (`"dry"`). Model surface and top are
``p_s = p_{N_z+1/2}`` and ``p_t = p_{1/2}``. Then (`Output.retrieval_column_weights!`):

1. **Stretch.** The model interfaces are mapped linearly onto the
   retrieval's pressure range, the model surface onto the retrieval surface
   and the model top onto the retrieval top:

   ```math
   s_{k+1/2} = \mu\,p_{k+1/2} + c, \qquad
   \mu = \frac{p^r_s - p^r_t}{p_s - p_t}, \qquad
   c = \tfrac12\big[(p^r_s - \mu p_s) + (p^r_t - \mu p_t)\big].
   ```

2. **Rebin.** Each retrieval layer takes the pressure-weighted mean of the
   stretched model layers it overlaps:

   ```math
   R_{ik} = \frac{\big|[s_{k-1/2},\, s_{k+1/2}] \cap [p^r_{i+1},\, p^r_i]\big|}
                 {p^r_i - p^r_{i+1}}, \qquad (R x)_i = \sum_k R_{ik}\, x_k ,
   ```

   with the outermost retrieval interfaces taken from the stretched model
   column (TM5's choice; they equal ``p^r_s`` and ``p^r_t`` to rounding).
   Every row of ``R`` sums to 1 and ``\sum_i R_{ik}(p^r_i - p^r_{i+1}) =
   s_{k+1/2} - s_{k-1/2}``: the rebin conserves the column integral.

3. **Column.**

   ```math
   X = \sum_i h_i a_i\, (R x)_i + \sum_i h_i (1 - a_i)\, x_{a,i}
     = \sum_k H_k\, x_k + c, \qquad
   H_k = \sum_i h_i a_i R_{ik},
   ```

   on the model's dry mixing ratios ``x_k = r_k/\max(m_k, \epsilon)``.

The pressures depend on the meteorology only, so ``X`` is linear in the
tracer mass with weights ``w_k = H_k/\max(m_k, \epsilon)`` on every level of
the cell and the offset ``c`` (`_linear_functional` is the exact derivative of
the evaluation; a window record takes the ``\omega_s``-weighted sum over its
steps, each step with its own pressures). The adjoint forcing is TM5's
per-layer ``H \cdot (X - y)/\sigma^2``.

**What the stretch does physically.** The stretch keeps every model layer's
share of the column, ``(s_{k+1/2} - s_{k-1/2})/(p^r_s - p^r_t) = (p_{k+1/2} -
p_{k-1/2})/(p_s - p_t)``: it moves the model column onto the retrieval's
pressure range without adding or removing air, and it absorbs both a
surface-pressure mismatch ``\Delta p_s = p_s - p^r_s`` and a different top.
With ``a = 1`` and weights normalized to the retrieval column the operator
returns the model's pressure-weighted column mean for any ``\Delta p_s``. A
mismatch changes only where the model's layers sit against the vertical
structure of the kernel. Where the model surface lies below the sounding's
true ground (``\Delta p_s > 0``, smoothed model orography under a mountain
sounding), the model's boundary layer is compressed onto the retrieval's
lowest layers instead of being cut off below the retrieval surface, which
would remove boundary-layer air, and its enhancements, from the column; a
30 hPa mismatch moves every interface by about 3 % of its pressure. This is
why the default is no limit, as in TM5. The same holds at the top: when the
retrieval top lies well below the model top (a retrieval grid that stops at
10 hPa, say), the stretch packs the whole model column, stratosphere
included, into the retrieval's pressure range, as TM5 does, so ``X`` is a
full-column mean of the model, not a partial column above the retrieval
top.

**Surface-pressure mismatch policy.** `max_surface_pressure_mismatch_pa`
(`bind_to_mesh` keyword; `[observations.placement]` key of the inversion
driver; default no limit) bounds ``|\Delta p_s|``, the time-weighted model
minus retrieval surface pressure over the sampled steps, with the model
surface pressure of the stretch's pressure basis (with `"dry"`, ``\Delta
p_s`` also contains the vapour weight, about 0.5 % of the surface pressure).
A record above the limit follows the unplaced policy: with
`unplaced_records = "error"` (default) the 4D-Var stops and names the
records; with `"skip"` the record is reported (simulated value, residual,
its stencil with `assimilated = false`) but adds nothing to the cost or the
gradient, and the number of such records is logged. The model surface
pressure is known only at the sampled states, so the check runs in the
4D-Var's forward replay; it depends on the meteorology only, so the same
records are left out in every iteration.

**Model-error term.** When the record gives a model-profile uncertainty
``\sigma_i`` (`retrieval_profile_sigma`), the observation error is
``\sigma^2 = \sigma^2_\mathrm{obs} + \sigma^2_\mathrm{mod} + \sum_i (h_i a_i \sigma_i)^2``
(`Output.retrieval_model_error_variance`): the operator's weights on the
retrieval layers, squared, times the profile variance, the layers taken
independent. It depends on the record only, not on the state or the
meteorology, so it is fixed when the record is bound and the cost stays
quadratic in every iteration (TM5 freezes its ``\sigma_\mathrm{mod}`` at the
first iteration for the same reason). It is analogous to, not the same as,
TM5's term: TM5's CH4 TROPOMI routine (`averaging_kernels.F90`, line 669)
evaluates ``\sum_k H_k^2 \sigma_k^2`` with the weights ``H = (h a) R`` on the
model layers and the spread of the modelled profiles, and its OCO-2 routine
adds a column model error instead. The input is an uncertainty of the
*model* profile (representation or transport error). Do not pass the
retrieval's a-priori uncertainty: the smoothing error it causes is already
represented by the averaging kernel.

**Diagnostics.** The stencil of an averaging-kernel column
(`CSObservationStencil`, placement code `RETRIEVAL_COLUMN_PLACEMENT_CODE` =
−3, level 0) carries ``\Delta p_s``, the model profile on the retrieval
layers ``R x`` (time-weighted) and `assimilated`; departure files add them
with the profile model sigma (`schemas/cs_departures_v2.toml`).

**Relation to TM5-4DVAR.** Matched: the stretch (including its averaging of
the two offsets), the overlap matrix of `inv_rebin` with its outer
interfaces, the column and its per-layer forcing; the test reproduces values
computed with TM5's Fortran on fixed columns to ``10^{-14}`` (the stretched
interfaces exactly; ``H`` can differ in the last bit from the Fortran's
`matmul` summation order). Different:

- the operator is in the model, a linear functional of the tracer mass at
  each sampled step (a window record averages the operator over its steps),
  where TM5 writes the Δt-averaged profile and surface pressure to a track
  file and applies the kernel once in Python;
- dry basis: ``x`` is the dry mixing ratio and the interfaces are total
  pressures rebuilt from the dry air mass and the humidity; TM5 uses moist
  ``rm/m`` and ``AT + BT\,p_s`` of its moist surface pressure;
- the pressure weights come from the record and must sum to 1; TM5's OCO-2
  routine computes ``h_i = \Delta p_i/p^r_s``, which leaves out the column
  above a retrieval top above 0 Pa;
- an explicit mismatch policy, where TM5's OCO-2 path has none and its CH4
  TROPOMI path sets a huge σ above a tolerance (the same effect as `"skip"`);
- a rebin that fails TM5's checks (possible only for a retrieval layer
  thinner than the stretch's rounding) is an error, not a silent zero
  departure;
- the model-profile uncertainty is given on the retrieval layers by the
  record; TM5's CH4 TROPOMI path uses the spread of the modelled profile on
  model layers from its track file.

`test/core/test_cs_4dvar_retrieval_columns.jl` checks agreement with TM5's
Fortran (reference values computed with `averaging_kernels.F90` itself, with
the model surface below and above the retrieval's and the retrieval top above
and below the model top) and with a line-by-line transcription on random
columns, conservation by the rebin, the pressure-weighted and air-mass-weighted
column means for ``a = 1`` on the model's levels, the schema block (units,
ordering, NaN padding, errors), the single sweep against a seeded footprint
per observation and step (``10^{-12}``; upwind, monotone PPM, Lin–Rood, TM5
convection, Full and Stride checkpoints), central finite differences with
window records, the mismatch policy, and the departure diagnostics.

### [Level retrievals (OCO-2, OCO-3): the hat projection](@id hat-level-projection)

The OCO-2 and OCO-3 retrievals (Lite files and the v11 MIP 10-second files)
give their prior and column averaging kernel on 20 **levels**,
``\sigma_j = p_j/p_s`` from ``10^{-4}`` (top) to 1 (surface), "defined on layer
boundaries": the retrieval's state vector holds the profile at the levels and
the profile is linear in pressure between them. Their pressure weights are
the integrals of the levels' hat functions ``\varphi_j`` (``\varphi_j(p_j) =
1``, falling linearly to 0 at the neighbouring levels): the trapezoid rule,
half-thickness end levels, with a dry-air correction. A record whose block
has `retrieval_mapping = "hat_levels"` (`Output.HatLevelProjection`) stores
the 20 level pressures, weights ``h_j``, kernel ``a_j`` and prior ``x_{a,j}``
surface first, and the model column is projected onto the hats.

**Projection.** The model interfaces are stretched as for layer retrievals,
the model surface onto the bottom level and the model top onto the top level
(``s_{k\pm1/2}``, each layer keeping its share of the column). The level values
``c_j`` of the model profile are its L2 (Galerkin) projection onto the hats in
dry air: the piecewise-linear profile ``\sum_j c_j \varphi_j`` closest to the
model profile, weighted by dry air,

```math
\sum_i M_{ji}\, c_i = b_j, \qquad
M_{ji} = \int \varphi_j \varphi_i \,\mathrm{d}p_\mathrm{dry}, \qquad
b_j = \int \varphi_j\, x\, \mathrm{d}p_\mathrm{dry} = \sum_k B_{jk} x_k, \qquad
B_{jk} = \frac{m_k}{s_{k+1/2} - s_{k-1/2}} \int_{s_{k-1/2}}^{s_{k+1/2}} \varphi_j(p)\, \mathrm{d}p ,
```

each stretched layer's dry air ``m_k`` spread evenly over its pressure
thickness; ``M`` is tridiagonal (neighbouring hats overlap), so
``P = M^{-1} B`` costs one tridiagonal solve per column and step. The column is
the operator of the layer case with ``P`` in place of the rebin ``R``:

```math
X = X_a + \sum_j h_j a_j (c_j - x_{a,j})
  = \sum_k H_k x_k + c, \qquad H = (h \circ a)^\mathsf{T} M^{-1} B,
```

with ``h``, ``a``, ``x_a`` and the prior column ``X_a`` from the file (the
MIP protocol's eq. 2; the kernel is already normalized: it is not multiplied
by ``h``). ``X_a`` is the file's `xco2_apriori`, carried by the converter and
the forward sampler's reader (`retrieval_prior_column` in observation files);
a record without one uses ``X_a = \sum_j h_j x_{a,j}``, the form of the layer
case. The 10-s averages of the weights and of the prior were taken separately,
so the two differ: by up to about 0.01 ppm (−0.012 ppm for sounding
20210804153946), a shift of every simulated column. ``X`` is still linear in the tracer: the
4D-Var functional is exact as before, with weights ``H_k/\max(m_k,
\epsilon)`` built from the replayed air mass and pressures of each sampled
step.

**Why the projection.** The basis contains the constants (the hats sum to 1
over the stretched column), so the projection preserves the column integral:
the rows of ``P`` sum to 1, and with the hat integrals of the model's dry air
as weights, ``h^m_j = \sum_k B_{jk} / \sum_k m_k``, the projected profile has
the model's dry column mean exactly, ``\sum_j h^m_j c_j = \sum_k m_k x_k /
\sum_k m_k``. The file's weights are the same integrals with the retrieval's
own humidity, so the column is conserved with them to the difference of the
two dry-air corrections. And a projection reproduces every profile its basis
can represent: a model profile that is piecewise linear on the levels, the
retrieval's own representation, gives its level values back exactly (to the
discretization of the model's layer means, second order in the layer
thickness), so the kernel is applied to the level values it was derived for.
A model profile equal to the retrieval's prior gives the prior column, to the
projection's discretization: on 960 OCO-2 records of 2019-07-01, with the
prior sampled as the layer means of a fine column aligned with the levels
(760 layers), ``X - X_a`` is −7e-6 ppm (5–95 %: −9e-6 to −4e-6). On a GEOS L72
column, whose layers do not line up with the levels, ``|X - \sum_j h_j
x_{a,j}|`` ranges from 1e-5 to 2.1e-3 ppm and single level values ``c_j``
differ from ``x_{a,j}`` by up to 0.57 ppm (independent review): the level
values carry the model's coarse vertical sampling, the column much less. The hat
average ``\int \varphi_j x / \int \varphi_j`` (``M`` lumped onto its diagonal),
which is not a projection, puts that column −0.0185 ppm low (−0.021 to
−0.012), because the prior is curved in the stratosphere; for a +10 ppm
boundary layer below ``\sigma = 0.95`` it is 0.018 ppm below the projection.
``P``, the inverse of the consistent mass matrix times ``B``, has negative
entries in general (about −0.21 to −0.23 in every row on a GEOS L72 column),
which a linear operator allows; the column weights ``H`` stayed positive on
OCO-2 records (on L72, ``\min_k H_k`` was 0.23–0.39 of their mean).

**The top level.** The top level sits at ``\sigma = 10^{-4}`` (10 Pa), not at
0. The stretch maps the whole model column, from its top (about 1 Pa for
GEOS, 0 for ERA5) to its surface, onto ``[10^{-4} p_s, p_s]``, every layer
keeping its share, so nothing is cut off: the model stratosphere above 53 hPa
falls under the top half-hat (``h \approx 0.026``) and the model's air above
10 Pa (``10^{-4}`` of the column) is squeezed into it. The weights sum to 1
over the levels.

**TM5-4DVAR's 19 layers.** `Output.tm5_layer_column` builds TM5-4DVAR's layer
form of the same record (`proj/tracer/CO2/py/Observations.py`, 7016–7052;
`Satellite.py`, 707–717): the 20 levels as the interfaces of 19 layers, the
kernel and prior interpolated to the layer midpoints, the weights
``\Delta p_i``, then the layer operator (rebin ``R``). TM5 divides the
thicknesses by ``p_s``, so its weights sum to ``1 - 10^{-4}``; this mode
normalizes them to 1 (TM5's column is 0.9999 times this one). It is for
comparisons with TM5-4DVAR co-samples (converter `mapping = :tm5`, forward
sampler `averaging_kernel_mapping = "tm5"`). For a +10 ppm boundary layer
below ``\sigma = 0.95`` on the prior, the mode is 0.009 ppm above the hat
projection (5–95 %: +0.002 to +0.017; 960 records of 2019-07-01).

**The MIP point formula.** The MIP protocol writes the column as
``X = X_\mathrm{prior} + \sum_j h_j a_j (c_j - c_{\mathrm{prior},j})`` with
``c_j`` the model profile at the levels, without saying how to transform the
model to them. Point values are not column-conserving: for a +10 ppm
boundary layer below ``\sigma = 0.95`` (``\Delta X`` = +0.49 ppm) the point
form is 0.232 ppm below the hat projection (5–95 %: 0.230–0.236 ppm; 960
records of 2019-07-01).
At ``c = x_a``, TM5's unnormalized weights put its column 0.033 ppm below the
point form. Neither point values nor boxes centred on the levels are offered.

**Tests.** `test/core/test_retrieval_hat_projection.jl`: the projection
matrix against an independent construction (exact integration of the hats and
their products), rows summing to 1, column conservation with the hat
integrals, a piecewise-linear profile reproduced at every level, ends and
kinks included, within ``\Delta \max|v'|/N^2`` for ``N`` layers per interval
(second order), a model profile equal to the prior giving the prior column
within 1e-5 ppm (synthetic and real MIP records), TM5 mode against a
transcription of TM5's Python, level records in schema v2.
`test/core/test_cs_4dvar_oco2_mip.jl`: the single sweep against a seeded
footprint per observation and step for level columns, windows and TM5-mode
records (``10^{-12}``, upwind and monotone PPM), and central finite
differences.

### [OCO-2 MIP records: converter, co-sampling, cell averages](@id oco2-mip-records)

**Converter.** `read_oco2_mip_observations(path; ...)` reads an OCO-2 or OCO-3
v11 MIP 10-second file (`OCO2_b11.2_10sec_GOOD_r2/r3.nc4`,
`OCO3_b11_10sec_GOOD_r2.nc4`) into schema-v2 column records with a level
block, and `convert_oco2_mip_observations(path, output; partition = :day)`
writes one file per day (`{YYYYMMDD}`) or month (`{YYYYMM}`):

```bash
julia --project=. scripts/preprocessing/oco2_mip_to_observations.jl \
    OCO2_b11.2_10sec_GOOD_r2.nc4 'obs/oco2_mip_{YYYYMMDD}.nc' \
    --start 2019-01-01 --stop 2019-02-01 --data-types LN,LG,OG --assimilate-flags 1
```

`sounding_id` → `id`; `date[1:6]` → `date_components` (truncated to the
second, as TM5 reads it); `latitude`, `longitude`; `kind = "column"`;
`altitude` → `elevation_m`; `time_window_s = 0`; `xco2_2019_scale`,
`xco2_uncertainty` and `model_error` (or a constant, `--model-sigma`) →
`value`, `value_sigma`, `model_sigma` (ppm → mol/mol); the levels
(``\sigma \cdot p_s``, hPa → Pa), weights, kernel and prior (ppm → mol/mol),
turned surface first, and the prior column `xco2_apriori`. Every record of the time range is written: `flag = 0`
when its `assimilate_flag` is in the experiment's set (`--assimilate-flags`,
1 by default) and its data type too (`--data-types`: LN, LG, OG by default;
`SAM` names OCO-3's snapshot area maps, `data_type = 0`),
else `flag = 1 + assimilate_flag` (1 evaluation-only scenes, 2 assimilable
but outside the data types, 3 withheld). `instrument_type` is
`<instrument>_<data type>` (`OCO2_OG`); the files record the source file,
the options and the mapping as root attributes. OCO-2 and OCO-3 sounding ids
can coincide, so keep the instruments in separate sets.

**Forward-only co-sampling.** The MIP needs co-samples of every record,
withheld and evaluation-only ones included. `bind_to_mesh(...;
flagged_records = "cosample")` (`CosampleFlaggedRecords`; the inversion
driver's `[observations].flagged_records`) binds records with a nonzero flag
as co-samples, `CSObservation(...; assimilate = false)`: the forward replay
simulates them, `CS4DVarResult` reports their simulated values, residuals and
stencils (`assimilated = false`), but they add nothing to the cost or the
gradient, are not screened by the surface-pressure limit, and do not extend
the taped run: the cost is bit for bit that without them, and the reverse pass
receives bit-identical forcing and tape length (the gradient is bit for bit
that without them single-threaded; with several threads the adjoint kernels
are not bit-reproducible from one call to the next, at about ``10^{-22}`` of
the largest entry, with or without co-samples).
Departure files carry each row's `flag` (`build_departure_set`), so one file
holds the co-samples of all records. The default (`"drop"`) is unchanged.

**Cell averages.** About 59 % of the assimilated OCO-2 10-s records share a
C90 cell with another record of the same orbit (1.43 per orbit and cell, about
9 s apart). `cell_averaging = CellAverage(; time_tolerance_s = 60,
correlation = 0)` (`[observations.cell_averaging]`, `mode = "average"`)
averages the assimilated column records of one tracer, cell and step sampling
that lie within `time_tolerance_s` of the earliest record of their group. With
``w_i = 1/\sigma_i^2`` (`value_sigma`):

```math
\bar y = \frac{\sum_i w_i y_i}{\sum_i w_i}, \quad
\bar H = \frac{\sum_i w_i H_i}{\sum_i w_i}, \quad
\sigma^2_\mathrm{meas} = \frac{\sum_i\sum_j w_i w_j \rho_{ij}\sigma_i\sigma_j}{(\sum_i w_i)^2}, \quad
\bar\sigma^2 = \sigma^2_\mathrm{meas} + \frac{\sum_i w_i \sigma^2_{\mathrm{mod},i}}{\sum_i w_i},
```

``\rho_{ii} = 1`` and ``\rho_{ij} = `` `correlation` between distinct spans (0:
the within-span correlation is already in ``\sigma_i``); the model error
(`model_sigma` and the profile term) is added once, not reduced.
``\bar H`` is a `CSCellAverageObjective`, exact for the linear operators: its
gradient is the ``w``-weighted combination of the members'. The members stay
as co-samples, so departures keep one row per record, with the average's
`cell_average_id`, and the average is a row of its own, the term of the cost
(placement code −4, `id = −cell_average_id`, ``\bar y``, ``\bar H x``,
``\sigma_\mathrm{meas}`` in `value_sigma`, the model error in `model_sigma`,
``\rho`` in `cell_average_correlation`): over the assimilated rows,
``\sum`` `normalized_departure`² ``= 2 J_\mathrm{obs}``. Averaging needs
`max_surface_pressure_mismatch_pa = Inf` (a `CSCellAverageObjective` built by
hand is checked the same way). The alternative,
`inflate_cell_model_error(set, mesh, t_start, dt; ...)` (`mode =
"inflate_model_error"`), keeps the records and multiplies each record's
`model_sigma²` by the number ``k`` of records of its group; the profile-error
term of an averaging-kernel column is not inflated.

**Forward sampler.** An `oco2_lite` source of `[output.observations]` with
`averaging_kernel_tracers = ["co2"]` (opt-in) also reads each sounding's
retrieval (`Output.read_oco2_retrieval_columns`) and writes
`co2_ak_column`, the same operator (`averaging_kernel_mapping = "hat"` or
`"tm5"`) on the sampled profile and the interface pressures of
`pressure_basis`. Without the key the output is unchanged. The test checks
that, in nearest-window mode, the sampler's value equals the 4D-Var's
simulated value of the same record converted to schema v2 (``10^{-13}``), and
that the default sounding file is identical.

## What is not yet shipped

| Item | Notes |
| --- | --- |
| **Optimal binomial Revolve** | `RevolveCheckpoint` ships as the bisection variant — logarithmic memory but not the Griewank–Walther optimal recompute count. Optimal binomial Revolve is the next refinement. |
| **Optimized/clamped convection adjoints** | CMFMC `clamp = true` and TM5/CMFMC-matrix collaborative, truncated, or merged solves are rejected by the footprint API until their exact branches are taped and transposed. |
| **TM5-4DVAR cross-validation** | Synthetic truth-recovery via `test_cs_inversion_truth_recovery.jl` is on CI; a side-by-side parity run against TM5-4DVAR on real data has not been published. |
| **Inversion driver on transport binaries** | `scripts/inversions/cs_4dvar.jl` binds observation files only on its synthetic meteorology (`synthetic_hydrostatic`); a 4D-Var on transport-binary meteorology is assembled with the Julia API (`bind_to_mesh`, `CSObservationColumnMet`, `cs_surface_flux_4dvar`). |
| **Tangent-linear model** | Forward TL is not exposed as a separate driver. If you need it, use the reverse pass plus identity seeding. |

## Adjoint-readiness in the forward design

Three concrete forward-design choices that pay off in the adjoint:

- **Vertical diffusion** — generic Kz kernels expose the tridiagonal
  coefficients `(a, b, c)` and transpose the backward-Euler solve with its
  mass/VMR scaling. The precomputed Dkg mass path exposes the retention
  fractions of two conservative bidiagonal solves. Its reverse applies
  those passes' transposes in reverse order, preserving a constant total-mass
  gradient exactly for positive carrier masses.
- **Convection (CMFMC + TM5)** — `apply!` takes a `ConvectionForcing`
  carrier explicitly so the operator does not call `current_time`
  internally; this keeps the operator pure-functional in the time
  variable. The TM5 reverse pass rebuilds the same per-column matrix
  and solves with the transposed LU factors.
- **Advection** — the adjoint composes each operator's transpose in reverse
  order. The palindrome helps organize that reversal, but forward transport
  is not its own inverse. Stored states supply the limiter branches and
  reconstructed-face sensitivities needed by the reverse pass.

## How to use the adjoint today

The maintained command-line path is the inversion driver:

```bash
julia --project=. scripts/inversions/cs_4dvar.jl \
    config/inversions/example_synthetic.toml
```

That synthetic-only configuration owns observations, controls, covariance and
optimizer choices, and an `[advection]` table that is parsed like a forward
run's (`limiter = "none"`, the unlimited PPM, is accepted only here). Unknown
tables and keys are errors. `[observations]` takes inline entries or a
schema-v2 observation `file`, whose point records `[observations.placement]`
places in the column of the `synthetic_hydrostatic` meteorology (the keys of
`[output.observations]` placement, parsed by the same code). It does not
expose checkpoint, diffusion, convection or real-binary settings.
For the lower-level Julia API, start with the
[footprint tutorial](@ref First-emission-footprint), then
`test/core/test_cs_inversion_truth_recovery.jl` for a larger inversion assembly.
Use `RevolveCheckpoint()` (without arguments)
for recursive bisection, or `StrideCheckpoint(K)` for an explicit interval.
LinRood checkpoints currently require `tape_storage = :device`; split-sweep
schemes also support `:pinned_host` and `:mmap`.

## Where to read next

- [Validation status](@ref) — what the forward model has been
  validated against.
- [Conservation budgets](@ref) — the explicit verification tests
  the forward operators pass.
- [Adjoints on top of the binary](../for_tm5_gchp_users/adjoints.md) —
  TM5-4DVAR / GIGC-adjoint user perspective on how the pipeline maps
  to those workflows.
