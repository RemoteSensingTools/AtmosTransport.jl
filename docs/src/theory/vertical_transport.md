# Vertical Transport: From Winds to Tracer Columns

This page follows the steps that set vertical tracer transport in
AtmosTransport, from met winds to the vertical tracer flux, and compares each
step with TM5 and GEOS-Chem High Performance (GCHP). It describes the code as
of the CATRINE C90 benchmark (2026-10) and is updated as options are added.
Open items are tracked in `docs/memos/2026-10-07_v0.5_transport_fidelity_backlog.md`.

The vertical index `k` counts layers from the model top (`k = 1`) to the
surface (`k = Nz`). Interface `k + ½` lies below layer `k`. Hybrid interfaces
are `p_{k+½} = A_{k+½} + B_{k+½} p_s`, with `ΔA_k`, `ΔB_k` the differences
across layer `k`. Above the hybrid region (about 177 hPa for GEOS L72) every
layer is pure pressure, `ΔB_k = 0`.

## 1. Horizontal mass fluxes

Preprocessing turns met winds into horizontal air-mass fluxes through the
cell faces of the cubed sphere, for each layer and substep.

**MERRA-2** (`src/Preprocessing/transport_binary/merra2_latlon_regrid.jl`).
- Inputs: A3dyn 3-hour mean winds (0.5° × 0.625°); I3 instantaneous surface
  pressure and humidity.
- `u_east` and `v_north` are regridded conservatively to C90 cell means, as
  two scalars.
- Hourly windows split each 3-hour block; air mass, `p_s`, humidity and
  temperature are linear in time.

Two constructions of the face fluxes are available
(`[preprocessing] face_fluxes`, MERRA-2 path):

- **`panel_average`** (default, historical,
  `src/Preprocessing/cs_transport_helpers.jl`, `cs_face_fluxes!`).
  - Each cell's wind is projected onto its panel's local face-normal
    directions.
  - The two adjacent cells' projections are averaged.
  - The result is multiplied by the mean moist layer thickness `Δp_k`, a
    length, and the time step, and divided by `g`.
  - The length is the centerline width of the cell on one side of the face
    (`face_lengths = "cell_centerline"`). `"edge"` uses the face's own
    great-circle length instead.
  - At a panel seam only the cell on the owning panel is used.
- **`vector`** (`cs_vector_face_fluxes!`).
  - The two cells' winds are combined as 3-D vectors.
  - At a panel seam the result is further interpolated along the edge to the
    face midpoint, as FV3 does.
  - It is then projected onto the face's own unit normal and multiplied by
    the face's great-circle length and the interpolated layer thickness.

**Why the seams matter.** The cells on both sides of a panel seam are skewed
along the edge. They are mirror images across the seam, so both centres are
shifted the same way along the edge and the shift does not cancel. The point
between the two centres lies 16 % (RMS), up to 25 %, of a face length along the
edge from the face midpoint; in the panel interior it is 2e-4. (The two centres
are equally far from the face, to 1e-12.) Averaging the two cells therefore
takes the wind at the wrong point.

For a solid-body rotation, which has no divergence, the spurious divergence
relative to the face flux at C90 is as below
(`/temp1/cfranken/scratch/overnight_2026_10_07/solid_body_divergence.jl`;
`test/core/test_cs_face_fluxes.jl` checks the same at C48):

| | interior cells | seam cells |
|---|---|---|
| `panel_average` | 1e-4 | 4e-3 |
| `vector` | 2e-7 | 5e-6 |

In pure-pressure layers nothing corrects this divergence (section 2), so it
becomes vertical motion. Against MERRA-2's own vertical velocity (2022-07-15,
C90, p < 150 hPa), the `panel_average` error at seam cells is larger than the
true signal: RMSE 8.8e-3 Pa/s against an RMS of 7.2e-3. With `vector` it is
1.3e-3, as in the interior. In the extratropics at 30–200 hPa the RMSE falls
from 8.4e-3 to 4.6e-3 Pa/s.

`flux_thickness = "dry_mass"` uses the dry layer thickness `g m_dry / A` that
the fluxes transport, instead of the moist `Δp`.

**ERA5** (`.../era5_n320_regrid.jl`). Instantaneous hourly winds on the N320
reduced Gaussian grid, regridded the same way. Surface pressure comes from
the 0.25° ARCO product. Each hour's winds are held over the following hourly
window.

**GCHP** takes the same MERRA-2 A3dyn winds. MAPL regrids them conservatively
to the cube as a vector (three Cartesian components). FV3's
`fv_computeMassFluxes` then restaggers them to the faces (A → D → C grid,
fourth-order interior interpolation, along-edge interpolation at cube edges)
and forms mass fluxes and Courant numbers with dry pressure. The winds are held over each 3-hour block;
`PS1`/`PS2` are interpolated in time to each 600 s step (GCHP run-directory
setting `IMPORT_MASS_FLUX_FROM_EXTDATA = .false.` for MERRA-2).

**TM5** integrates `U·Δp` along each cell edge from the spectral fields
(`tmm.F90`), with at least two points per spectral wavelength.

## 2. Column mass closure (pressure fixer)

The reconstructed fluxes never close the column mass budget exactly. The
mismatch between flux convergence and the met surface-pressure tendency is
about 0.1–0.2 Pa/s RMS before correction. AtmosTransport closes each column:

1. Solve a Poisson problem on the cube for a correction to the
   column-integrated face fluxes, so that column convergence equals the
   change of column air mass over the window (`balance_cs_column_mass_fluxes!`,
   `src/Preprocessing/cs_poisson_balance.jl`).
2. Spread each face's column correction `δF` over the levels with weights
   `w_k`: `δF_k = w_k δF / Σ_j w_j`.

The weights are selectable (`[preprocessing] column_balance_weights`, MERRA-2
path):

| Option | Weights `w_k` | Consequence |
|---|---|---|
| `mass` (current default) | layer air mass of the two adjacent cells | The correction reaches every layer, including the stratosphere |
| `hybrid_b` | `ΔB_k` | Only hybrid layers are corrected; the semantics of TM5 and GCHP |
| `hybrid_mass` | layer air mass where `ΔB_k > 0`, else 0 | Like `mass` in the troposphere, no correction in pure-pressure layers |

The vertical mass flux (section 3) inherits the correction as
`Σ_{j ≤ k} (w_j/Σw − ΔB_j) · δ`, a mode that is coherent in the vertical.
With `mass` weights it is proportional to `p/p_s − B`, about 0.1 of the
column mismatch at 100 hPa and 0.2 near the extratropical tropopause.

**Measured effect.** These tests compare the vertical velocity diagnosed
from `cm` with MERRA-2's own (A3dyn OMEGA, C90, 3-hour means, on 2022-01-15
and 2022-07-15).
- At 30–100 hPa poleward of 30°, the RMS of the `mass`-weighted vertical
  velocity is 1.0–3.5 times that of OMEGA. The excess is largest at high
  summer latitudes, where the resolved motion is weakest. The correlation is
  0.28–0.93. With `hybrid_mass` the RMS ratio is 0.97–1.10 and the
  correlation 0.87–0.99.
- Below 200 hPa, `hybrid_mass` and `mass` agree with OMEGA equally well
  (correlations within 0.01).
- `hybrid_b` matches `hybrid_mass` in the stratosphere. In the troposphere
  it is close but slightly worse, mostly in the southern storm track
  (60–30°S). The RMS difference from OMEGA, in 1e-3 Pa/s, is:
  - at 500–850 hPa: 69.9 against 65.8 (January), 76.5 against 69.4 (July);
  - at 200–500 hPa: 59.4 against 52.3 (January), 65.7 against 56.2 (July).

Per-band values: `/temp1/cfranken/scratch/omega_3way_2022{0115,0715}_dB.txt`
(memo `2026-10-07_v0.5_transport_fidelity_backlog.md`, item 1). A first
`hybrid_b` build had a constructor bug: its weights were the B value at each
layer's top interface, not `ΔB`. Its results are superseded by these.

In a four-month MERRA-2 run (December 2021 – March 2022, PPM, compared with
GCHP), `hybrid_mass` cut the growth of the bias above 100 hPa from +0.54 to
+0.19 ppm CO₂ and from +0.066 to +0.023 ppt SF₆. The slope against GCHP
above 100 hPa rose from 0.89 to 0.97. Near-surface RMSE (below 910 hPa)
changed little:
- CO₂ fell from 0.383 to 0.370 ppm and SF₆ by 5 %;
- fossil CO₂ and Rn-222 rose by 0.3–0.4 %.

Column RMSE was unchanged. `hybrid_b` gives the same stratospheric
improvement (+0.19 ppm CO₂) with near-surface RMSE between the two: CO₂
0.377 ppm.

**GCHP has no column closure.** FV3 transports tracers horizontally with the
uncorrected fluxes. It then remaps the Lagrangian layers onto `A + B p_s,adv`,
where `p_s,adv` is the surface pressure the flux convergence implies
(`fv_tracer2d.F90`, `offline_tracer_advection`). The mismatch with the met
surface pressure stays in `p_s,adv`. The pressure is reset from the met
fields at the next step. Each tracer's global mass is then rescaled, with the
factor applied only from the first layer with `B > 0` downward
(`fv_tracer2d.F90`, about line 977). The implied vertical motion distributes the
column convergence by `ΔB`.

**TM5** holds its vertical flux at the IFS value and Poisson-corrects the
horizontal fluxes level by level. The column tendency it closes is
`ΔB_k (p_s2 − p_s1)`, so the mismatch never reaches pure-pressure layers.

## 3. Vertical mass flux

The vertical air-mass flux through interface `k + ½` follows from continuity
for each substep (`diagnose_cs_cm!`):

`cm_{k+½} = cm_{k−½} + (∇·F)_k − Δm_k`, with `cm_½ = 0` at the top.

`Δm_k` is the window's change of layer air mass per substep. A residual left
at the surface interface is spread with the same weights as in section 2. The
binary stores `cm` for each window; it is constant within the window. GCHP
has no stored vertical flux: its vertical motion is implicit in the remap.
TM5 recomputes `cm` at every dynamics step from the time-interpolated
fluxes.

## 4. Vertical advection

**Splitting.** Each substep applies `X Y Z [diffusion, emissions] Z Y X`
(`src/Operators/Advection/CubedSphereStrang.jl`). The C90 MERRA-2 runs of
December 2021 – March 2022 used 6–13 substeps per hourly window (mostly
8–11), that is 12–26 vertical sweeps per hour.
Convection and chemistry run once per window. Air mass is updated in every
sweep with the same fluxes that move the tracer, so tracer mass is conserved
exactly.

**Flux form.** In each vertical sweep
`rm_k ← rm_k + f_{k−½} − f_{k+½}`, with `f = α (rm_donor + (1 − α) s_donor)`
for downward flow (and the mirror form for upward flow), where
`α = cm / m_donor` and `s` is a tracer-mass slope.

**`scheme = "ppm"`** (production). The slope comes from the PPM edge value on
the outflow side of the donor cell:
- edge values use the uniform-index fourth-order formula
  `q_{k+½} = 7/12 (q_k + q_{k+1}) − 1/12 (q_{k−1} + q_{k+2})`, without
  layer-thickness weighting;
- the edges are limited with the Colella–Woodward (1984) monotonicity
  conditions (flattening at extrema, overshoot correction);
- the limited edge sets a linear slope toward the outflow face, which is
  bounded by the cell content (`_limited_moment`), and the flux uses the
  Russell–Lerner form above (`_slopes_face_flux`,
  `src/Operators/Advection/reconstruction.jl`).

The scheme is therefore second order, with a PPM-informed slope; the parabola
is never integrated. The top two and bottom two layers fall back to upwind.
In 3-D tests it behaves almost like `scheme = "slopes"` (minmod slopes).

**`scheme = "ppm"`, `vertical = "fv3_kord8"`** (cubed sphere) uses the profile
GCHP uses for tracers, FV3's `scalar_profile` with `kord = 8` and `iv = 0`
(`fv_mapz.F90`), in the same flux form
(`src/Operators/Advection/vertical_fv3_profile.jl`). Within layer `k`, with
`s ∈ [0, 1]` running from the top to the bottom edge,

```math
q(s) = q_L + s\,\bigl[(q_R - q_L) + q_6 (1 - s)\bigr], \qquad q_6 = 3\,(2\bar q_k - q_L - q_R).
```

1. *Edges.* The edge values solve FV3's compact tridiagonal system. It is
   fourth-order accurate on a grid whose spacing is the layer air mass, so thin
   and thick layers are weighted correctly.
2. *Large-scale constraints.* Each interior edge lies between its two
   neighbouring layer means, except at a local extremum of the means. There it
   may overshoot on the extremum side only, and at a minimum it stays
   non-negative.
3. *Layer limiters.*
   - Layers 3 to `Nz − 2`: Huynh's second constraint bounds each edge using the
     neighbouring slopes, then a positive-definite limiter removes negative
     interior minima.
   - Layers 2 and `Nz − 1`: the standard PPM monotonicity limiter; at a local
     extremum of the means the layer is flattened.
   - Top and bottom layers: the outer edge is made non-negative, then the
     layer is monotone.
4. *Flux.* The tracer flux through an interface is the air-mass flux `F`
   times the mean of the donor parabola over the swept fraction
   `α = |F| / m_donor ≤ 1`:

```math
\bar q_\downarrow = q_R - \tfrac{α}{2}\bigl[(q_R - q_L) - (1 - \tfrac{2α}{3})\, q_6\bigr], \qquad
\bar q_\uparrow = q_L + \tfrac{α}{2}\bigl[(q_R - q_L) + (1 - \tfrac{2α}{3})\, q_6\bigr],
```

for downward flow (bottom of the layer above) and upward flow (top of the
layer below).

In cumulative air-mass coordinates, this sweep is FV3's conservative remap of
the layers (`mapn_tracer`, which uses the positive-definite profile) onto
interfaces shifted by the swept mass, `M_e → M_e − F_e`. In exact arithmetic
the two are identical, provided that:
- no interface flux exceeds its donor layer's air mass;
- the fluxes through the model top and the surface are zero.

One sweep is one remap with the same profile. GCHP remaps once per 600 s after
its horizontal step and then rescales tracer mass globally. `fillz` is off by
default (`fill = .false.` in `fv_arrays.F90`). Here
the remaps follow the binary's `cm`, twice per substep (per palindrome
subcycle).

Each interface flux is computed once per column and enters both adjacent
layers with opposite signs, so the fluxes cancel exactly in the column sum.
Only the rounding of the cell updates remains, in Float32 as in Float64.

The profile is non-negative everywhere when the layer means are. A layer's
outflow is then the integral of its own non-negative parabola over the swept
parts. The sweep therefore stays non-negative as long as the swept fractions
of a layer's two faces add up to at most one, `α_top + α_bottom ≤ 1`. The
runtime's subcycling budgets the total outflow of all six sweeps of a substep
against the starting layer mass (`cfl_limit = 0.95`). That budget is a static
proxy, not a proof for the evolving mass, but it keeps the sum below one in
practice.

**Signed tracers.** The profile above is FV3's positive-definite one
(`iv = 0`), which GCHP uses for all tracers. It flattens layers whose mean is
not positive and clamps edges at zero, so tracers that become negative, such
as flux anomalies, fall back to first order where they are negative.
`vertical = "fv3_kord8_signed"` selects FV3's profile for signed fields
(`iv = 1`). It omits the three non-negativity steps and is symmetric under
`q → −q`, except where neighbouring layer means are exactly equal: FV3's
large-scale constraint resolves those ties with its local-minimum branch
whatever the sign.

Differences from GCHP:
- GCHP remaps Lagrangian layers once per 600 s step; here the profile drives
  each flux-form vertical sweep, with interface fluxes from the binary's `cm`.
- GCHP rescales each tracer's global mass in the hybrid layers after every
  step; this model conserves tracer mass without rescaling.

Tests (`test/core/test_fv3_vertical_profile.jl`):
- The edge values and limited parabolas equal, bit for bit, an independent
  transcription of `fv_mapz.F90` for both profiles; the positive-definite
  parabolas are non-negative inside every layer.
- Beyond a Courant number of one the flux is capped at the donor's content.
- The fluxes equal the swept integrals of that reference. The whole update
  equals an independent remap onto the shifted interfaces, to 1e-11.
- Uniform fields are preserved, column mass telescopes and results stay
  non-negative, in both precisions.
- A six-panel sweep equals the column kernel column by column, the
  single-tracer and packed sweeps agree, and `TransportModel` builds the
  column scratch and runs the sweep.

Checks on an L40S (C90, L72, four tracers, 200–400 random sweeps at
Courant ≤ 0.3; scripts and output in `/temp1/cfranken/scratch/fv3_vertical/`):
- CPU and GPU air mass are bit-identical.
- Relative tracer-mass changes are below 3e-15 in Float64 and 5e-9 in Float32.
  Under the same Float32 test the default PPM changes the uniform background
  tracer by up to 1.4e-8.
- One-month C90 MERRA-2 run (December 2021, production Courant numbers,
  `hybrid_mass` binaries):
  - Float32 − Float64 global tracer totals: −3e-8 (CO₂), +6e-8 (SF₆) and
    −2e-10 (fossil CO₂).
  - The default scheme, from the 25-month twin on `mass` binaries (so the
    closure differs as well), gives −4e-8, +5e-8 and +8e-9 for the same month.
  - Column-mean differences match (RMS 0.009 ppm CO₂ for both;
    `/temp1/cfranken/catrine_protocol/compare_f32_f64_dec2021/`).
- A six-panel sweep costs 1.2 ms in Float32 against 0.47 ms for the default.
  In a four-month C90 run the wall time per window is unchanged
  (0.58–0.60 s).

**`scheme = "linrood"`** uses FV3-style cross-term advection horizontally and,
by default, first-order upwind vertically. That is far too diffusive: in a
four-month MERRA-2 test the stratospheric CO₂ bias against GCHP grew six times
faster than with `ppm`. With `vertical = "fv3_kord8"` its vertical sweeps use
the FV3 profile above.

**GCHP** has no Eulerian vertical advection. After the horizontal step it
remaps the Lagrangian layers onto the hybrid levels with PPM
(`kord_tr = 8`: thickness-weighted edges, Huynh constraint, positivity),
once per 600 s step. **TM5** carries prognostic tracer-mass slopes that are
advected with the tracer, with a positivity-only limiter and xyz–zyx
splitting.

**Numerical diffusion.** In 1-D column tests on the L72 grid
(`/temp1/cfranken/scratch/tm5_advection/results_matrix.txt`), a 5-km
vertical wave is advected at ω = 0.0073 Pa/s. That is close to the mean
MERRA-2 |ω|; the median per-sweep Courant number is 8e-4. The effective
diffusivities at 100–30 hPa are 0.57 m²/s for `ppm`, 0.30 for TM5's
prognostic slopes, 0.08 for FV3 `kord 8` (Lagrangian remap and our flux form
alike) and 2.7 for upwind. For a smooth monotone profile, `ppm` and FV3 are
near zero, while both slopes schemes and upwind are not.

In 1-D translation tests (a Gaussian bump moved 30 layers;
`/temp1/cfranken/scratch/fv3_vertical/translation_probe.txt`):
- The error of the FV3 profile falls 11–23× when the bump width doubles
  (third order or better). It falls 4× for `ppm` and `slopes` (second order)
  and 2× for upwind.
- The FV3 profile is 4–900 times more accurate than `ppm` for widths of 3–24
  layers and Courant numbers of 0.01–0.3.

In four-month MERRA-2 runs against GCHP (December 2021 – March 2022), the FV3
profile reduced the growth of the CO₂ bias above 100 hPa by about 0.05 ppm,
with either column-balance weighting:
- with `mass` weights, from +0.54 to +0.49 ppm;
- with `hybrid_mass` weights, from +0.19 to +0.14 ppm.

The FV3 profile also lowered near-surface RMSE for every tracer, by 1–3 %.
The column closure (section 2) is the larger of the two effects.

## 5. Sub-grid vertical transport

Convection and boundary-layer mixing add vertical transport that the met
winds do not resolve. See the convection (CMFMC with the DQRCU cloud base;
TM5 matrix convection) and diffusion (GEOS-Chem non-local VDIFF; TM5 `dkg`)
documentation. In the benchmark, ERA5 runs use the TM5 schemes and MERRA-2
runs the GEOS-Chem schemes.

## 6. Summary

| Step | TM5 | GCHP 14.7 (MERRA-2) | AtmosTransport |
|---|---|---|---|
| Horizontal fluxes | Spectral edge integrals | FV3 from conservatively regridded A3dyn winds | Two-cell means of conservatively regridded winds × layer thickness |
| Column closure | Per-level Poisson, vertical flux fixed, ΔB | None; mismatch in advected `p_s`, reset each step | Column Poisson; weights `mass` (default), `hybrid_b`, `hybrid_mass` |
| Vertical motion | IFS vertical velocity | Implied by Lagrangian remap (ΔB) | `cm` from continuity, constant per window |
| Vertical scheme | Prognostic slopes, positivity limiter | Lagrangian remap, PPM `kord 8` | Flux form; PPM-informed slope with CW84 limiter (`ppm`), or FV3 `kord 8` parabola (`vertical = "fv3_kord8"`) |
| Time step | Global `ndyn`, CFL-limited | 600 s | 10–13 substeps per hourly window |

GEOS-Chem is a reference, not the physical truth. The choices above are
judged against the met model's own diagnostics and against observations, as
well as against GCHP.

## References

- Colella, P., and Woodward, P. R. (1984). The piecewise parabolic method
  (PPM) for gas-dynamical simulations. *J. Comput. Phys.* 54, 174–201.
- Eastham, S. D., et al. (2018). GEOS-Chem High Performance (GCHP v11-02c).
  *Geosci. Model Dev.* 11, 2941–2953.
- Huynh, H. T. (1996). Schemes and constraints for advection. *Fifteenth
  International Conference on Numerical Methods in Fluid Dynamics*, Monterey,
  CA, Springer, 498–503.
- Krol, M., et al. (2005). The two-way nested global chemistry-transport zoom
  model TM5. *Atmos. Chem. Phys.* 5, 417–432.
- Lin, S.-J. (2004). A "vertically Lagrangian" finite-volume dynamical core
  for global models. *Mon. Weather Rev.* 132, 2293–2307.
- Lin, S.-J., and Rood, R. B. (1996). Multidimensional flux-form
  semi-Lagrangian transport schemes. *Mon. Weather Rev.* 124, 2046–2070.
- Putman, W. M., and Lin, S.-J. (2007). Finite-volume transport on various
  cubed-sphere grids. *J. Comput. Phys.* 227, 55–78.
- Russell, G. L., and Lerner, J. A. (1981). A new finite-differencing scheme
  for the tracer transport equation. *J. Appl. Meteorol.* 20, 1483–1498.
