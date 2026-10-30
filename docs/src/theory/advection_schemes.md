# Advection schemes

The runtime ships **four** flux-form advection schemes, each behind the
abstract type `AbstractAdvectionScheme` declared in
`src/Operators/Advection/Reconstruction/schemes.jl`. Upwind, slopes, and standard PPM use
directional sweeps. Lin–Rood couples the two horizontal directions; configured
runs default its vertical sweep to FV3's `kord = 8` tracer profile, while the
low-level Julia constructor retains upwind. On the cubed sphere, PPM can also
select that profile with `vertical = "fv3_kord8"`. The model-facing operator interface dispatches to the
appropriate transport path.

| Scheme | Smooth-flow accuracy | Monotone? | Positive? | LL | CS | RG |
|---|---|---|---|---|---|---|
| `UpwindScheme` | 1st order (donor cell) | yes (trivially) | preserves a non-negative input under its CFL contract; also supports signed tracers | yes | yes | **yes — RG's only option today** |
| `SlopesScheme{L}` | 2nd order in smooth regions (van Leer / Russell-Lerner) | yes if `L = MonotoneLimiter` (default) | no zero clamp in the default signed path; `PositivityLimiter` is explicit opt-in | yes | yes | no (the face-indexed Strang path restricts to `AbstractConstantScheme`) |
| `PPMScheme{L,V}` | 2nd order: limited Colella–Woodward (1984) edges set a Russell–Lerner slope. With `L = CW84Limiter` the flux integrates the limited parabola (the complete CW84 scheme). With `V = FV3ScalarProfile` the vertical sweep integrates FV3's `kord = 8` parabola (3rd order or better in smooth regions) | profile-limited with `MonotoneLimiter`, but the 1-D update over- and undershoots; with `CW84Limiter` monotone in uniform 1-D flow (Courant number ≤ 1) | signed; small negative column means observed in CS runs with `MonotoneLimiter`. `CW84Limiter` keeps a non-negative tracer non-negative in every sweep in which no cell exports more than its air mass. The FV3 vertical profile is positive definite | yes (default vertical only) | yes — covered by `test/core/test_cubed_sphere_advection.jl` and `test/core/test_fv3_vertical_profile.jl` | no (same rejection) |
| `LinRoodPPMScheme{ORD}` | piecewise-parabolic; `ORD ∈ {5, 7}` selects the boundary stencil | profile-limited, but the full split update can undershoot | signed; not positivity-preserving | n/a | yes — uses FV3 cross-term advection (`fv_tp_2d_cs!`) | n/a |

The accuracy column describes the **per-face
reconstruction** in smooth regions; near limiters / discontinuities
high-order reconstruction can drop to first order locally. `LinRoodPPMScheme`'s
`ORD ∈ {5, 7}` selects the **boundary stencil** for cross-panel
faces; the in-panel reconstruction is the same FV3 PPM either way. ORD=7 is
not a globally seventh-order transport method. Spatial reconstruction order
also does not establish the temporal order of the complete split update.

## Selecting a scheme

The TOML runner defaults to `scheme = "upwind"` when the selector is omitted.
Choose the algorithm explicitly when comparing runs:

| `[advection]` settings | Runtime scheme | Vertical transport |
|---|---|---|
| `scheme = "upwind"` | `UpwindScheme()` | upwind |
| `scheme = "slopes"` | `SlopesScheme(MonotoneLimiter())` | slopes |
| `scheme = "ppm"` | `PPMScheme(MonotoneLimiter())` | PPM-informed slopes, as horizontally |
| `scheme = "ppm", limiter = "cw84"` | `PPMScheme(CW84Limiter())` | complete CW84 PPM, as horizontally |
| `scheme = "ppm", vertical = "fv3_kord8"` | `PPMScheme(MonotoneLimiter(), FV3ScalarProfile())` (CS only) | FV3 `scalar_profile`, `kord = 8`, positive definite |
| `scheme = "ppm", vertical = "fv3_kord8_signed"` | `PPMScheme(MonotoneLimiter(), FV3ScalarProfile(; positive_definite = false))` (CS only) | FV3 `scalar_profile`, `kord = 8`, for signed tracers |
| `scheme = "linrood", ppm_order = 5` | `LinRoodPPMScheme(5; vertical = PPMScheme(; vertical = FV3ScalarProfile()))` (CS only) | FV3 `scalar_profile`, `kord = 8` (default) |
| `scheme = "linrood", ppm_order = 7` | `LinRoodPPMScheme(7; vertical = PPMScheme(; vertical = FV3ScalarProfile()))` (CS only) | FV3 `scalar_profile`, `kord = 8` (default) |
| `scheme = "linrood", vertical = "upwind"` | `LinRoodPPMScheme(5; vertical = UpwindScheme())` (CS only) | upwind (explicit legacy choice) |
| `scheme = "linrood", ppm_order = 7, vertical = "fv3_kord8"` | `LinRoodPPMScheme(7; vertical = PPMScheme(; vertical = FV3ScalarProfile()))` (CS only) | FV3 `scalar_profile`, `kord = 8` |

Omitting `ppm_order` for Lin–Rood selects 5, and omitting `vertical` selects
the GCHP-matched `fv3_kord8` profile, in forward runs and in the inversion
driver alike: every Lin–Rood vertical choice has an adjoint (see
[the FV3 profile's adjoint](@ref fv3-vertical-adjoint)). The Julia
constructors `LinRoodAdvectionSpec(order)` and `LinRoodPPMScheme()` remain
upwind. Setting `ppm_order` with `scheme = "ppm"`
is an error: standard split PPM has no order selector. `vertical` is only
accepted with `scheme = "ppm"` or `"linrood"`, and `limiter` (`"monotone"`, the
default, or `"cw84"`) only with `scheme = "ppm"`. `NoLimiter` and
`PositivityLimiter` are selected through the Julia constructors.
Packed tracer arrays, GPU workgroup sizes, and copy-back or ping-pong execution
are implementation choices within a scheme, not additional algorithms.

## Diagnosed slopes with the Russell–Lerner flux (`SlopesScheme`)

Per-cell linear reconstruction with a monotone limiter by default:

```math
χ_c(x) \;\approx\; \chi_c \;+\; s_c \,(x - x_c)
```

where `s_c` is the slope estimate. The face flux through the right-of-cell-`c`
face uses the **Courant-fraction formula** (Russell & Lerner, 1981):

```math
F^{n+1/2}_{c+1/2} \;=\; F^n_{c+1/2}\,(\chi^n_c + s^n_c \cdot (1 - \alpha_{c+1/2})/2)
```

with `α = F / m` the local Courant fraction (`F` is the per-substep
mass amount stored in the binary, so the `Δt` factor is already
absorbed into `F` — see [Mass conservation](@ref) for the
`flux_kind = :substep_mass_amount` convention). The slope `s_c` is
re-diagnosed from the neighbouring cell means at every sweep (van Leer 1977;
`MonotoneLimiter` takes the minmod of the centred and twice the one-sided
differences), so the scheme carries no slope state between sweeps. The
production kernel is `_slopes_face_flux` in
`src/Operators/Advection/Reconstruction/reconstruction.jl`, with the canonical formula
derivation in that function's docstring.

**Properties:**
- 2nd order in smooth regions even with `NoLimiter` (the centered
  slope itself is a 2nd-order reconstruction; what `MonotoneLimiter`
  buys you is monotonicity at discontinuities, at the cost of
  dropping locally to 1st order at limiter saturation).
- Monotonic with `MonotoneLimiter` (van Leer minmod); does not
  produce new extrema across faces.
- Valid for signed mixing ratios and equivariant under a constant VMR offset.
  The monotone profile is limited relative to neighbouring values, never
  relative to tracer zero.
- Mass-conservative to floating-point via the flux-form telescoping
  argument (see [Mass conservation](@ref) for the precise statement
  and round-off bounds).

### Relation to TM5's slopes advection

`SlopesScheme` is not TM5's slopes scheme, although it uses the same face
flux. TM5 (`dynamu`, `dynamv`, `dynamw` in `advectx.F90`, `advecty.F90`,
`advectz.F90`) carries the first moments `rxm`, `rym`, `rzm` of every tracer
as prognostic state. Each sweep transports them with their own flux and
update equation, and the moments are also updated by TM5's convection and
emissions. TM5 limits a moment only for positivity, `|rxm| ≤ rm`, and does
not use a minmod limiter.

`test/core/test_slopes_tm5_reference.jl` checks the relation against an
independent transcription of TM5's `dynamu` on a periodic row
(`test/helpers/tm5_dynamu_reference.jl`):

- With `rxm` re-diagnosed by the minmod rule at the start of every sweep, the
  TM5 update reproduces `SlopesScheme()` to rounding, and with `rxm := 0` it
  reproduces `UpwindScheme()`. The flux formula is shared; the moment
  equation is what is missing.
- With the prognostic moment, TM5 is much less diffusive. A Gaussian of
  half-width 6 cells on a 120-cell ring keeps 0.952 of its peak after 10
  revolutions at Courant number 0.5 in TM5, against 0.719 with `SlopesScheme`
  (0.830 with `PPMScheme`). After 10 revolutions the `SlopesScheme` L2 error
  is 4.6 to 6.8 times TM5's at Courant numbers 0.05 to 0.75.
- TM5's positivity clamp refers to zero, so TM5 is not equivariant under a
  constant offset (transporting `c − 1/2` is not transporting `c` and
  subtracting `1/2`). `SlopesScheme` limits against neighbouring values and is
  offset-equivariant, which signed anomaly tracers require.

!!! warning "Known limitation: no prognostic-moment (TM5) slopes scheme"
    No advection option reproduces TM5's slopes advection. A
    prognostic-moment scheme is postponed: it would need moment state for
    every tracer carried through emissions, convection and diffusion, moment
    rotation at cubed-sphere panel seams, an adjoint, and a limiter that is
    safe for signed tracers. Runs made with `scheme = "slopes"` are
    diagnosed-slope (van Leer) results, not TM5-equivalent ones.

## PPM-informed slopes (`PPMScheme`)

`PPMScheme` builds the piecewise parabola of Colella & Woodward (1984) in each
cell,

```math
χ_c(ξ) \;=\; χ^L_c + ξ\bigl(Δχ_c + χ^{(6)}_c (1 - ξ)\bigr), \qquad 0 \le ξ \le 1,
```

with `Δχ_c = χ^R_c − χ^L_c` and `χ^{(6)}_c = 6(χ_c − ½(χ^L_c + χ^R_c))`. The
edge values come from the uniform-index fourth-order formula
`χ_{c+½} = 7/12 (χ_c + χ_{c+1}) − 1/12 (χ_{c−1} + χ_{c+2})` and are limited with
the Colella–Woodward monotonicity conditions (`_ppm_limit_profile`).

The face flux does not integrate the parabola. Only the limited edge on the
outflow side is used: it sets the edge offset `s_x = m_c (χ^R_c − χ_c)` (outflow
to the right) of the Russell–Lerner flux above (`_slopes_face_flux`). The update
is therefore second order, with a PPM-informed slope. Donor layers within two
layers of the model top or surface have no slope (upwind flux).

This default is not monotone: the unlimited fourth-order edge can lie outside
the range of its neighbouring cell means, and the linear flux ignores the
parabola's curvature. A box or spike advected 100 steps around a 40-cell ring
undershoots to −0.16 (box, Courant number 0.7) and overshoots to 1.015 (box,
0.3) on a [0, 1] field (`test/core/test_ppm_cw84_limiter.jl`).

**Complete CW84 PPM (`limiter = "cw84"`, `PPMScheme(CW84Limiter())`).** Three
changes give the scheme of Colella & Woodward (1984):

1. Edges use van Leer-limited slopes (CW84 eqs. 1.7–1.8),
   ```math
   χ_{c+1/2} = \tfrac12 (χ_c + χ_{c+1}) - \tfrac16 (δχ_{c+1} - δχ_c), \qquad
   δχ_c = \operatorname{minmod}\bigl(\tfrac12 (χ_{c+1} - χ_{c-1}),\, 2(χ_{c+1} - χ_c),\, 2(χ_c - χ_{c-1})\bigr),
   ```
   so each edge lies between its two neighbouring cell means. Where neither
   slope is limited this equals the fourth-order edge above.
2. The profile is limited with the same monotonicity conditions.
3. The flux is the mass flux times the parabola's mean over the swept
   fraction `α = |F|/m` of the donor cell (CW84 eq. 1.12). For outflow to the
   right this is `χ_c + (1 − α)(b_R − α b_0)` with `b_R = χ^R_c − χ_c` and the
   curvature `b_0 = (χ^L_c − χ_c) + (χ^R_c − χ_c)`; it enters
   `_slopes_face_flux` as the moment `s_x = m_c (b_R − α b_0)`
   (`_ppm_raw_moments`).

The stencil is the same six cells, so halo widths do not change. In uniform
one-dimensional flow with Courant numbers up to 1 the update is monotone; on the ring
above it stays in [0, 1] in Float32 and Float64. Every limited parabola is
bounded by the neighbouring cell means, so within one sweep a cell's tracer
mass stays non-negative as long as the fractions of the cell swept out through
its two faces sum to at most 1. When the runtime chooses the subcycle count,
its palindrome budget guarantees this for every pass: each pass removes at
most `cfl_limit / 2` of the mass the cell holds when the pass starts (see
"CFL handling and subcycling" below). Cubed-sphere
binaries with the `binary_schedule` substep contract run `n = 1` and rely on
the write-time gate, which checks the same budget against the window's masses
rather than each step's, so there positivity is measured, not proven. Advecting one period of a sine at Courant number 0.4, the L2 error falls
by 4.05 from 80 to 160 cells (3.46 for the default) and is 18% lower at 160
cells; at 20 cells it is 29% higher, because the profile limiter flattens the
sine's extrema. The cubed-sphere adjoint supports both the monotone and CW84
limiters (`_ppm_face_coeffs`); the run log labels the option `PPM, CW84`.

**Vertical reconstruction.** On the cubed sphere,
`PPMScheme(; vertical = FV3ScalarProfile())` (TOML `vertical = "fv3_kord8"`)
replaces the vertical face flux with the profile GEOS-Chem High Performance
uses for tracers: FV3's `scalar_profile` with `kord = 8`. Its edges solve a
compact tridiagonal system weighted by layer air mass. They are limited with
large-scale constraints, Huynh's second constraint and a positive-definite
limiter. The flux integrates the donor layer's parabola over the swept
fraction. The positive-definite profile flattens layers whose mean is not
positive, so tracers that become negative (flux anomalies, for example) need
`vertical = "fv3_kord8_signed"` (FV3's `iv = 1`). Equations, limiter details and tests are in
[Vertical transport](vertical_transport.md), section 4.

**Measured accuracy.** A Gaussian bump is translated 30 layers on a uniform
grid; `test/core/test_fv3_vertical_profile.jl` holds a version of this as a
test. Doubling the bump width from 6 to 12 to 24 layers cuts the RMS error:
- 4× for `slopes` and the default PPM (second order);
- 11–23× for the FV3 vertical profile (third order or better);
- 2× for upwind.

The FV3 profile is 4–900 times more accurate than the default for widths of
3–24 layers and Courant numbers of 0.01–0.3 (output:
`/temp1/cfranken/scratch/fv3_vertical/translation_probe.txt`).

**Properties:**
- Second order in smooth regions (horizontal sweeps and default vertical);
  local accuracy drops to first order at limiter saturation.
- Monotonicity controlled by the limiter parameter; without a
  limiter PPM is **not monotone** and can produce small oscillations
  near discontinuities. The FV3 vertical profile is positive definite but
  not monotone.
- Shipped on lat-lon AND cubed-sphere structured layouts (the CS
  case is exercised by `test/core/test_cubed_sphere_advection.jl`).
  The FV3 vertical profile is cubed-sphere only.
  Face-connected PPM for the **reduced-Gaussian** topology is not
  currently wired — and neither is `SlopesScheme` on RG. The
  face-indexed Strang path restricts to `AbstractConstantScheme`, so
  **`UpwindScheme` is the only advection option for RG production
  runs today**.

For cubed-sphere runs that need the FV3 cross-term advection at panel edges and
permit signed undershoots, `LinRoodPPMScheme` (next section) is the relevant
variant. Default monotone PPM limits the reconstructed profile, but the full
CS update has produced small negative column means in real-input tests.
Use upwind under its CFL contract when non-negativity is required, or validate
the higher-order scheme's bounds on the intended workload.

## [Lin-Rood PPM with cross-term (`LinRoodPPMScheme{ORD}`)](@id Lin-Rood-PPM-with-cross-term)

The cubed-sphere variant. Extends PPM with the **two-step Lin-Rood
splitting** (`fv_tp_2d_cs!` in
`src/Operators/Advection/linrood_horizontal.jl`) so the X and Y sweeps see each
other's intermediate fluxes via the inner-edge flux-and-slope rotation
that FV3 uses internally. The runtime pairs this horizontal update with
vertical upwind. Two edge-value families are selectable:

| `ORD` | Interior reconstruction | Panel-edge treatment |
|---|---|---|
| `5` | Colella–Woodward fourth-order edges with extremum flattening | none: the fourth-order edges read the neighbor panel's halo cells |
| `7` | Same as ORD=5 | FV3's cube-edge values (below) |

With `ORD = 7` every face carries one edge value, used by both cells beside it,
as FV3's `al(i)` in `tp_core.F90` (`xppm`/`yppm`, `iord < 8`). With face `f`
between cells `f−1` and `f` and the panel edges at faces `1` and `Nc + 1`:

```math
\begin{aligned}
q_{1} &= \tfrac12\Bigl[\bigl(\tfrac32 q_0 - \tfrac12 q_{-1}\bigr) + \bigl(\tfrac32 q_1 - \tfrac12 q_2\bigr)\Bigr]
  && \text{panel edge (Putman \& Lin 2007, App. C, eq. 47)}\\
q_{2} &= \tfrac{5}{14} q_1 + \tfrac{11}{14} q_2 - \tfrac{2}{14} q_3,\qquad
q_{0} = -\tfrac{2}{14} q_{-2} + \tfrac{11}{14} q_{-1} + \tfrac{5}{14} q_0
  && \text{one-sided, beside the edge}\\
q_{f} &= \tfrac{7}{12}(q_{f-1} + q_f) - \tfrac{1}{12}(q_{f-2} + q_{f+1})
  && \text{elsewhere,}
\end{aligned}
```

and mirrored at `Nc`, `Nc + 1`, `Nc + 2` (`_ppm_edge_values_ord7`). No edge of a
cell beside a panel edge is built from a stencil that crosses it, so cell 1
has one parabola: the seam value on its left, the one-sided value on its
right. FV3 weights the two seam extrapolations by the cell widths `dxa`; they
are equal on both sides of an equiangular panel edge, and the code uses equal
weights. Up to version 0.5.0 the seam value replaced only the edges at faces
`1` and `Nc + 1`, so the face kernels rebuilt cell 1 with a cross-seam
fourth-order left edge when they evaluated face 2 (issue #62). In solid-body
rotation of a Gaussian blob about a tilted axis (one revolution in `16 Nc`
steps), the consistent edges lower the global relative L2 error of ORD=7 by
10–31% at C8–C64 (both panel conventions); in the two cell rows beside each
panel edge the error changes by −9% to +17%. The forward and adjoint face kernels
evaluate the same edge function (the adjoint on dual numbers), so the
adjoint stays exact.

`PPMScheme` is the strict-structured PPM; `LinRoodPPMScheme` is the
cross-term-aware CS-native variant. Lin-Rood no longer runs the zero-referenced
`fillz` repair, because that repair destroys negative anomaly mass. The tradeoff
is explicit: a non-negative input can develop small negative undershoots even
while carrier air mass remains positive and global tracer storage is conserved.
Choose it for signed/anomaly transport or after validating that this boundedness
tradeoff is acceptable for the tracer being simulated.

A divergence-damping (del-2) operator is layered on top
(`_divergence_damping_cs_kernel!` in `LinRood.jl`) to suppress the
small numerical noise that survives at the panel boundaries.

## Limiters

Four selectable limiter types are declared in `schemes.jl`; their formulas
live in `src/Operators/Advection/Reconstruction/limiters.jl` and, for the PPM edges and
moments, `reconstruction.jl`:

| Limiter | What it enforces | Use case |
|---|---|---|
| `NoLimiter()` | unlimited centered slope / parabola | smooth-flow benchmarks where you want the order-N error rate without limiter clipping |
| `MonotoneLimiter()` (default) | van Leer minmod slope: `minmod(central, 2forward, 2backward)`, where `central = (forward + backward)/2`. Bounds the reconstructed profile relative to neighboring values and supports signed tracers. | production runs, including anomaly tracers |
| `CW84Limiter()` (`PPMScheme` only) | the complete Colella–Woodward PPM: van Leer-limited edges, the monotone profile limiter, and the swept-parabola flux. Monotone in uniform 1-D flow; non-negative in every sweep in which no cell exports more than its air mass. | tracers that must stay non-negative under PPM (fossil and other emission tracers) |
| `PositivityLimiter()` | keeps the sign of the centered slope and clips its magnitude to `\|s\| ≤ 2c`, so both face values `c ± s/2` stay non-negative (the moment form is the clamp `\|s_x\| ≤ r_m`). Rising and falling profiles are treated alike. **Weaker than `MonotoneLimiter`**: positivity-only, may still create new local maxima from large gradients. Requires a non-negative tracer. | tracers that must stay non-negative (mole fractions, water vapor, aerosol concentrations) AND tolerate occasional new maxima |

Limiter primitives are written branchless (`ifelse(a*b > 0, ..., 0)`)
in `limiters.jl` so they don't trigger warp divergence on the GPU.

The default monotone path deliberately has no post-step tracer clamp. Bounds
on a local reconstructed profile do not establish positivity of the complete
multidimensional CS update; monitor transported-field minima separately from
mass totals. `PositivityLimiter` is a separate, explicitly zero-referenced
policy and must not be used for anomaly tracers.

## CFL handling and subcycling

The binary's stored fluxes are **per-substep mass amounts**
(`flux_kind = :substep_mass_amount`), already pre-divided by the
substep count for the active window. The Courant number is therefore
a pure mass ratio — no `Δt` factor:

```math
\alpha_{c+1/2} \;=\; \frac{|F^n_{c+1/2}|}{m^n_c}
```

One transport step applies each direction twice (`X Y Z | midpoint | Z Y X`,
`H V | midpoint | V H` on reduced Gaussian), so the runtime budgets one
palindrome's worth of outflow per cell against the smaller of the cell's
masses at the start and at the end of the step,

```math
r_c \;=\; \frac{2\,\bigl(\mathrm{out}_x + \mathrm{out}_y + \mathrm{out}_z\bigr)}{\min\bigl(m_c,\, m_c^{\mathrm{end}}\bigr)},
\qquad \mathrm{out}_d = \max(0, -F_{d,\mathrm{lo}}) + \max(0, F_{d,\mathrm{hi}}),
\qquad m_c^{\mathrm{end}} = m_c + 2 \sum_d \bigl(F_{d,\mathrm{lo}} - F_{d,\mathrm{hi}}\bigr),
```

and when `max r_c` exceeds `cfl_limit` it **subcycles** the whole step:

```julia
n = ceil(max_c r_c / cfl_limit)
```

and runs the symmetric composition

```math
S(Δt) \;=\; \bigl[X_h Y_h Z_h\bigr]^{n} \circ V(Δt) \circ \bigl[Z_h Y_h X_h\bigr]^{n},
\qquad h = \frac{Δt}{2n},
```

where `X_h` is an x sweep with every flux scaled by `1/n` and operator
products act right to left: `n` forward triples `X → Y → Z`, the midpoint
operators `V` once, and `n` reverse triples `Z → Y → X`. With
`ψ(h) = Z_h Y_h X_h` the forward half is `ψ^n` and the reverse half
`(ψ^*)^n`, where `ψ^*(h) = X_h Y_h Z_h = ψ(−h)^{-1}` is its adjoint, so
`S(−Δt) = S(Δt)^{-1}`: the composition is symmetric, hence second order like
the palindrome itself, with the same `6n` sweeps; at `n = 1` it is the plain
palindrome. FV3's `fv_tracer2d` subcycles the same way (its whole 2-D
operator `nsplt` times at flux / `nsplt`).

**What the budget guarantees.** The midpoint operators do not change air
mass, and each block of passes (one `X_h Y_h Z_h` or `Z_h Y_h X_h`) changes a
cell's mass by the same amount, `Σ_d (F_{d,lo} − F_{d,hi}) / n`. The masses
between blocks therefore lie on the straight line from `m_c` to
`m_c^end`, so all of them are at least `min(m_c, m_c^end)`. Within one block
the passes remove at most `(out_x + out_y + out_z) / n ≤ (cfl_limit / 2)
min(m_c, m_c^end)`. Hence:

- every pass removes at most `cfl_limit / 2` of the mass the cell holds when
  that pass starts, and
- the air mass stays at least `(1 − cfl_limit / 2) min(m_c, m_c^end) > 0`
  throughout the step.

A reference mass of `m_c` alone, as the count used before this budget,
bounds only the first block, because a cell that drains within the step
starts its later blocks well below `m_c`. With one cell of mass 1 among
cells of mass 100, inflow 1.51 in x and outflow 2.0 in y (`m_c^end = 0.02`),
that rule chose `n = 5`, the reverse half drove the air mass negative, and PPM produced
a mixing ratio of `−1.7 × 10^43`. The reference mass `min(m_c, m_c^end)` is
the one of the write-time positivity gate (`min(m, m_next)`, with the step's
own end mass for `m_next`); here it gives `n = 211`. A cell with outflow
whose reference mass is not positive has no valid count (the flux set would
leave it with no or negative mass), and the count throws an `ArgumentError`.
The budget is sufficient, not tight: it assumes that a block's passes
remove all of their outflow before any of their inflow arrives. `test/core/test_subcycle_composition.jl`
checks it on the draining cell above (x and y swapped too), solid-body
rotation and strain on the cubed sphere, strain on lat-lon and an overturning
circulation on reduced Gaussian. Up to version 0.5.0 each direction was
subcycled on its own, `X^n Y^n Z^n | Z^n Y^n X^n`: the `n` passes of one leg
then remove the full-flux outflow of that direction before any other direction
refills the cell, so subcycling bounded the Courant number of a pass but not
the mass loss within a leg, and exactly non-divergent flows drove air mass
negative (issue #61).

`_palindrome_subcycle_count` and the per-cell `_palindrome_outflow_ratio` in
`src/Operators/Advection/Structured/subcycling.jl` are shared by all split-sweep
topologies; `_cs_static_palindrome_subcycle_count` (`CubedSphere/cs_subcycling.jl`),
`_structured_palindrome_subcycle_count` and
`_face_indexed_palindrome_subcycle_count` (`subcycling.jl`) evaluate them on
each flux layout. See
[Operators on top of the binary](../for_tm5_gchp_users/operators_on_binaries.md#palindrome-subcycling-and-the-positivity-budget)
for the write-time gate and the binary schedule, which sets `n = 1` on
cubed-sphere binaries with the `binary_schedule` substep contract.

All split-sweep topologies share one default `cfl_limit`, `DEFAULT_CFL_LIMIT =
0.95`, so when the runtime chooses the count no pass removes more than 47.5%
of a cell's current mass and every cell keeps at least 52.5% of
`min(m_c, m_c^end)`:

| Topology | Default `cfl_limit` | Notes |
| --- | --- | --- |
| Lat-lon (structured-grid Strang, `strang_split!`) | `0.95` | Palindrome budget (was a per-direction `1.0` up to 0.5.0). |
| Reduced Gaussian (face-indexed Strang) | `0.95` | Palindrome budget over `H` and `V`; only `UpwindScheme` reachable today. |
| Cubed-sphere (`SlopesScheme` / `PPMScheme`, `strang_split_cs_mt!`) | `0.95` | Palindrome budget. |
| Cubed-sphere (`LinRoodPPMScheme`) | not used | The LinRood path relies on the binary's adaptive substep schedule. Its in-kernel 0.9 × cell-mass clipping limits carrier outflow; it does **not** guarantee tracer positivity. |

If you're running `LinRoodPPMScheme` on a flow with locally large
Courant numbers, the recourse is to halve `dt` in the run config
rather than rely on the operator to subcycle internally.

## Strang palindrome and temporal accuracy

The full transport step is the time-symmetric composition

```text
S(Δt) = X(Δt/2) Y(Δt/2) Z(Δt/2) ∘ V(Δt) ∘ Z(Δt/2) Y(Δt/2) X(Δt/2)
```

(with `V(Δt) → V(Δt/2) S(Δt) V(Δt/2)` when surface flux is on). Strang
composition gives second-order splitting accuracy for sufficiently accurate
subflows. The palindrome alone does not establish second-order accuracy of
the implemented transport: reconstruction, halo evolution, and each subflow's
time integration must also be validated. A cubed-sphere seam fixture shows
first-order timestep refinement for both the original split PPM and its paired
seam update. Conservation and temporal accuracy are separate requirements.

Convection and chemistry are NOT inside the palindrome — they are
applied once per met window, post-palindrome — because their natural
cadence is the met window rather than the advection sub-step, and they
do not commute with advection at the per-substep level.

## Panel-edge halo treatment (cubed sphere)

`PanelConnectivity` (`src/Grids/PanelConnectivity.jl`) carries a
table of panel-edge mappings: for each panel `p` and each edge `e ∈
{west, east, south, north}`, what is the neighbour panel and what's the
edge orientation (`0 = aligned`, `2 = reversed`)? The default
connectivity methods encode the GEOS-FP / GEOS-IT 6-panel arrangement and its
gnomonic alternative.

Halo exchange supplies neighboring tracer and air-mass values for the
reconstruction stencil. Matching air-mass fluxes alone does not guarantee
matching tracer fluxes: both cells sharing a face must use the same tracer
exchange at the same update stage.

Lin–Rood now shares the mean of the two panels' inner and outer face mixing
ratios before applying either panel's final horizontal divergence. Only panel
boundary faces are changed. Tangential indices follow the connectivity map;
mixing ratios are scalars, and the mirrored air-mass flux supplies the normal
sign. This makes the two tracer transfers cancel to roundoff. The adjoint
applies the transpose of the same averaging operation.

The dimensionally split cubed-sphere Upwind, Slopes, and PPM paths assign each
physical seam to the lower-numbered panel and that panel edge's local axis.
Each X/Y group updates its panel-interior faces and all seams assigned to it.
A seam transfer is reconstructed once, before any panel changes, then applied
with opposite signs to both neighboring cells in the same group. This includes
rotated contacts whose neighbor edge lies on the other local axis. The local
panel kernels mask boundary fluxes so each physical exchange occurs once.
Air and signed tracer mass therefore cancel across contacts; mirrored input
mass fluxes remain required for consistency with the binary's continuity budget.

The forward tape records these same grouped sweeps. At fixed meteorology, the
adjoint first collects both neighbors' output seeds, reverses interior sweeps,
and then differentiates the shared seam reconstruction. This preserves the
tracer-gradient contract without changing the tape format. The forward cache
uses `12 × Nc × Nz × (Nt + 1)` values, including air mass: 9.41 MB for C90 L66,
32 tracers, Float32. It has no six-tracer cap; Lin–Rood does not allocate this
split-sweep buffer.

In a six-tracer C90 L66, 24-hour V100 test with TM5 convection and Dkg diffusion,
sharing Lin–Rood seam estimates reduces maximum final relative tracer-total
drift from `3.77e-5` to `6.98e-7` in Float32 and from `3.80e-5` to `7.93e-16`
in Float64. The paired split PPM update reduces the corresponding maxima
from `2.24e-5` to `8.17e-7` in Float32 and from `2.28e-5` to `9.91e-16` in
Float64, with no global normalization. These are measured results for one
forcing archive, not universal error bounds. Positive initial layers still
develop negative column means: about `-2.09e-10` mol/mol in Lin–Rood before
and after its fix, and `-4.27e-11` mol/mol in the corrected split PPM run.
Conserving totals does not establish positivity or reference-model agreement.
These seam measurements predate the subsequent
[conservative Dkg diffusion](mass_conservation.md#Implicit-diffusion-and-roundoff)
change, which further reduces the PPM Float32 maximum to `3.14e-7` on the same
six-tracer workload.

An independent tilted solid-body rotation of a smooth Gaussian tracer on C8,
C16, and C32 grids, with both panel conventions, gives slightly smaller
area-weighted field errors for paired split PPM at quarter and full rotations.
Float64 mass drift remains below `6.1e-15`. This checks transported fields as
well as totals; it is not a general temporal-order or positivity guarantee.

## Reduced-Gaussian per-ring face segmentation

Adjacent rings on a reduced-Gaussian grid have different cell counts
(`nlon[j+1] ≠ nlon[j]`), so the boundary between them cannot be
covered by a one-to-one face mapping. `_boundary_counts(nlon_per_ring)`
in `src/Grids/ReducedGaussianMesh.jl` segments each ring-pair
boundary into `lcm(nlon[j], nlon[j+1])` mini-faces — every cell on
ring `j` contributes `lcm/nlon[j]` mini-faces, and every cell on ring
`j+1` receives `lcm/nlon[j+1]` mini-faces. Mass-flux pairing is then
exact and the telescoping argument holds across the whole grid.

The cost is a higher face count than the cell count would suggest:
adjacent rings with `nlon = 108, 112` produce `lcm = 3024` mini-faces
between them. Performance-tuning notes live beside the implementation.

## Where the schemes meet the code

| Concept | File / function |
| --- | --- |
| Scheme abstract root | `src/Operators/Advection/Reconstruction/schemes.jl::AbstractAdvectionScheme` |
| `UpwindScheme`, `SlopesScheme{L}`, `PPMScheme{L,V}`, `LinRoodPPMScheme{ORD,Z}` | `src/Operators/Advection/Reconstruction/schemes.jl` |
| FV3 vertical profile (`FV3ScalarProfile`) | `src/Operators/Advection/CubedSphere/vertical_fv3_profile.jl` |
| FV3 vertical-profile adjoint (frozen-limiter transpose) | `src/Operators/Advection/CubedSphere/vertical_fv3_profile_adjoint.jl::_sweep_z_panels_fv3_adjoint!` |
| Limiter primitives (branchless, GPU-safe) | `src/Operators/Advection/Reconstruction/limiters.jl` |
| Slopes face flux (Russell-Lerner formula) | `src/Operators/Advection/Reconstruction/reconstruction.jl::_slopes_face_flux` |
| Structured-grid Strang palindrome | `src/Operators/Advection/Structured/StrangSplitting.jl::strang_split!` |
| Cubed-sphere Strang palindrome | `src/Operators/Advection/CubedSphere/CubedSphereStrang.jl::strang_split_cs_mt!` |
| CFL subcycle counts (lat-lon, reduced Gaussian) | `subcycling.jl::_structured_palindrome_subcycle_count`, `_face_indexed_palindrome_subcycle_count` |
| CFL subcycle count (CS) | `CubedSphere/cs_subcycling.jl::_cs_static_palindrome_subcycle_count` |
| Shared palindrome budget | `Structured/subcycling.jl::_palindrome_subcycle_count` |
| CS multi-tracer fused kernels (X / Y / Z) | `src/Operators/Advection/CubedSphere/cs_sweep_common.jl` |
| CS paired split seam exchange | `src/Operators/Advection/CubedSphere/CubedSphereSeams.jl` |
| CS paired split seam adjoint | `src/Operators/Advection/CubedSphere/CubedSphereSeamsAdjoint.jl` |
| CS panel-edge halo sync | `src/Grids/PanelConnectivity.jl` + `cs_native_fluxes.jl::_propagate_cs_outflow_to_halo!` |
| Lin-Rood cross-term + del-2 damping | `src/Operators/Advection/linrood_horizontal.jl` (driver), `LinRood.jl` (kernels, damping) |

## What's next

- [Conservation budgets](@ref) — the synthetic-fixture tests that
  verify the conservation contract bit-by-bit.
- [Validation status](@ref) — what we've actually validated end-to-end
  vs what's still on the to-do list.
