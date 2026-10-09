# Advection schemes

The runtime ships **four** flux-form advection schemes, each behind the
abstract type `AbstractAdvectionScheme` declared in
`src/Operators/Advection/schemes.jl`. Upwind, slopes, and standard PPM use
directional sweeps. Lin–Rood couples the two horizontal directions and uses
upwind vertically by default. On the cubed sphere, PPM and Lin–Rood can
instead use FV3's vertical tracer profile (`vertical = "fv3_kord8"`). The model-facing operator interface dispatches to the
appropriate transport path.

| Scheme | Smooth-flow accuracy | Monotone? | Positive? | LL | CS | RG |
|---|---|---|---|---|---|---|
| `UpwindScheme` | 1st order (donor cell) | yes (trivially) | preserves a non-negative input under its CFL contract; also supports signed tracers | yes | yes | **yes — RG's only option today** |
| `SlopesScheme{L}` | 2nd order in smooth regions (van Leer / Russell-Lerner) | yes if `L = MonotoneLimiter` (default) | no zero clamp in the default signed path; `PositivityLimiter` is explicit opt-in | yes | yes | no (the face-indexed Strang path restricts to `AbstractConstantScheme`) |
| `PPMScheme{L,V}` | 2nd order: limited Colella–Woodward (1984) edges set a Russell–Lerner slope. With `V = FV3ScalarProfile` the vertical sweep integrates FV3's `kord = 8` parabola (3rd order or better in smooth regions) | profile-limited with `MonotoneLimiter`; full CS update can undershoot | signed; small negative column means observed in CS runs. The FV3 vertical profile is positive definite | yes (default vertical only) | yes — covered by `test/core/test_cubed_sphere_advection.jl` and `test/core/test_fv3_vertical_profile.jl` | no (same rejection) |
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
| `scheme = "ppm", vertical = "fv3_kord8"` | `PPMScheme(MonotoneLimiter(), FV3ScalarProfile())` (CS only) | FV3 `scalar_profile`, `kord = 8`, positive definite |
| `scheme = "ppm", vertical = "fv3_kord8_signed"` | `PPMScheme(MonotoneLimiter(), FV3ScalarProfile(; positive_definite = false))` (CS only) | FV3 `scalar_profile`, `kord = 8`, for signed tracers |
| `scheme = "linrood", ppm_order = 5` | `LinRoodPPMScheme(5)` (CS only) | upwind |
| `scheme = "linrood", ppm_order = 7` | `LinRoodPPMScheme(7)` (CS only) | upwind |
| `scheme = "linrood", ppm_order = 7, vertical = "fv3_kord8"` | `LinRoodPPMScheme(7; vertical = PPMScheme(; vertical = FV3ScalarProfile()))` (CS only) | FV3 `scalar_profile`, `kord = 8` |

Omitting `ppm_order` for Lin–Rood selects 5. Setting it with `scheme = "ppm"`
is an error: standard split PPM has no order selector. `vertical` is only
accepted with `scheme = "ppm"`. Alternative limiter
objects are selected through the Julia constructors, not a TOML limiter key.
Packed tracer arrays, GPU workgroup sizes, and copy-back or ping-pong execution
are implementation choices within a scheme, not additional algorithms.

## Russell-Lerner slopes (`SlopesScheme`)

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
`flux_kind = :substep_mass_amount` convention). This is implemented
unchanged from TM5's `advectx__slopes` / `advecty__slopes` routines;
the production kernel is
`_slopes_face_flux` in
`src/Operators/Advection/reconstruction.jl`, with the canonical
formula derivation in that function's docstring.

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
`src/Operators/Advection/LinRood.jl`) so the X and Y sweeps see each
other's intermediate fluxes via the inner-edge flux-and-slope rotation
that FV3 uses internally. The runtime pairs this horizontal update with
vertical upwind. Two edge-value families are selectable:

| `ORD` | Interior reconstruction | Panel-edge treatment |
|---|---|---|
| `5` | Huynh-constrained PPM | default edge-value family |
| `7` | Same as ORD=5 | special cubed-sphere boundary correction |

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

Three selectable limiter types are declared in `schemes.jl`; their formulas
live in `src/Operators/Advection/limiters.jl`:

| Limiter | What it enforces | Use case |
|---|---|---|
| `NoLimiter()` | unlimited centered slope / parabola | smooth-flow benchmarks where you want the order-N error rate without limiter clipping |
| `MonotoneLimiter()` (default) | van Leer minmod slope: `minmod(central, 2forward, 2backward)`, where `central = (forward + backward)/2`. Bounds the reconstructed profile relative to neighboring values and supports signed tracers. | production runs, including anomaly tracers |
| `PositivityLimiter()` | one-sided clip that drops the slope where the reconstruction would go negative at a face. **Weaker than `MonotoneLimiter`**: positivity-only, may still create new local maxima from large gradients. | tracers that must stay non-negative (mole fractions, water vapor, aerosol concentrations) AND tolerate occasional new maxima |

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

If the maximum `α` over the domain exceeds the active `cfl_limit`,
the runtime **subcycles** that direction:

```julia
n_sub = ceil(max_α / cfl_limit)
```

and runs the direction `n_sub` times with flux scaling `F → F / n_sub`.
The X / Y / Z subcycle counts can differ; all six sweeps in the
Strang palindrome use the same per-direction count to preserve time
symmetry.

`_subcycling_pass_count` in
`src/Operators/Advection/subcycling.jl` is the per-direction
counter; the structured per-direction max-α helpers are
`_x_subcycling_pass_count` / `_y_subcycling_pass_count` /
`_z_subcycling_pass_count` in the same file. The CS analogue is the
palindrome-aware `_cs_static_palindrome_subcycle_count` in
`src/Operators/Advection/cs_subcycling.jl`. The CS palindrome
budget sums all six legs and uses
`2·(out_x + out_y + out_z) / m_start` — see
[Operators on top of the binary](../for_tm5_gchp_users/operators_on_binaries.md#cubed-sphere-palindrome-and-the-positivity-budget)
for the derivation.

The `cfl_limit` defaults are baked into the per-topology Strang
entry points:

| Topology | Default `cfl_limit` | Notes |
| --- | --- | --- |
| Lat-lon (structured-grid Strang, `strang_split!`) | `1.0` | |
| Reduced Gaussian (face-indexed Strang) | `1.0` | Only `UpwindScheme` reachable today. |
| Cubed-sphere (`SlopesScheme` / `PPMScheme`, `strang_split_cs!`) | `0.95` | Palindrome-budget metric. |
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
| Scheme abstract root | `src/Operators/Advection/schemes.jl::AbstractAdvectionScheme` |
| `UpwindScheme`, `SlopesScheme{L}`, `PPMScheme{L,V}`, `LinRoodPPMScheme{ORD,Z}` | `src/Operators/Advection/schemes.jl` |
| FV3 vertical profile (`FV3ScalarProfile`) | `src/Operators/Advection/vertical_fv3_profile.jl` |
| Limiter primitives (branchless, GPU-safe) | `src/Operators/Advection/limiters.jl` |
| Slopes face flux (Russell-Lerner formula) | `src/Operators/Advection/reconstruction.jl::_slopes_face_flux` |
| Structured-grid Strang palindrome | `src/Operators/Advection/StrangSplitting.jl::strang_split!` |
| Cubed-sphere Strang palindrome | `src/Operators/Advection/CubedSphereStrang.jl::strang_split_cs!` |
| CFL subcycle counters (structured) | `subcycling.jl::_subcycling_pass_count`, `_x/_y/_z_subcycling_pass_count` |
| CFL subcycle counters (CS) | `cs_subcycling.jl::_cs_static_palindrome_subcycle_count` |
| CS multi-tracer fused kernels (X / Y / Z) | `src/Operators/Advection/multitracer_kernels.jl` |
| CS paired split seam exchange | `src/Operators/Advection/CubedSphereSeams.jl` |
| CS paired split seam adjoint | `src/Adjoints/CubedSphereSeams.jl` |
| CS panel-edge halo sync | `src/Grids/PanelConnectivity.jl` + `cs_transport_helpers.jl::_propagate_cs_outflow_to_halo!` |
| Lin-Rood cross-term + del-2 damping | `src/Operators/Advection/LinRood.jl` |

## What's next

- [Conservation budgets](@ref) — the synthetic-fixture tests that
  verify the conservation contract bit-by-bit.
- [Validation status](@ref) — what we've actually validated end-to-end
  vs what's still on the to-do list.
