# TM5-style spectral mass fluxes

This page specifies `TM5SpectralFluxes`, a second construction of the ERA5
spectral mass fluxes for lat-lon (LL), reduced-Gaussian (RG) and direct
cubed-sphere (CS) targets. It
follows the TM5 meteorological preprocessor (`tmm`): the face fluxes are
quadratures of the retained spectral winds and surface pressure along the
target faces, an independent vertical flux is integrated from the spectral
divergence, and the horizontal fluxes are balanced layer by layer against the
hybrid mass tendency. The existing construction (face- and centre-sampled
winds times a face pressure thickness, see [ERA5 spectral path](spectral_era5.md))
stays the default; the design calls it `PointSampledSpectralFluxes`.

LL and RG are implemented by TM5-FLUX-1; direct CS is implemented by TM5-FLUX-2b. The point-sampled construction stays
the default. Select the new route explicitly:

```toml
[preprocessing]
spectral_flux_construction = "tm5_spectral"
spectral_synthesis = "cpu" # reference; omit for GPU (CLI loads CUDA)
# spectral_synthesis_float = "auto" # GPU: auto, Float64 or Float32
# quadrature_refinement_steps = 2

[numerics]
flux_time_sampling = "window_mean"
```

The selector is `spectral_flux_construction`, since the existing
`flux_construction` settings describe the face-method family. Existing configs
keep their interpretation and default arithmetic. The new route requires a global target, the hourly ERA5 interval and dry
basis, a next-day 00 UTC spectral and humidity endpoint, and retained native
interfaces (`echlevs` or layer merging). Pressure-overlap vertical remapping
is outside this construction. The hybrid-B
extent must run from 0 to 1, with `level_top=1` and zero top A. These contracts are checked during configuration planning, before reading meteorology. GPU selection on the legacy point-sampled route
is rejected explicitly; its opt-in is deferred.

The page follows the design of 2026-10-30 with the owner answers of that day.
Statements marked *estimate* are estimates, not measurements.

## Sources

TM5 citations are `file:line` in the NOAA CarbonTracker TM5 repository
(`ct-unified-github` at commit `2d60209`); pydyn citations are at pydyn commit
`0db6e6d`. pydyn is a NumPy transcription of the TM5 dynamic core; it starts
from the TM5 meteorological archive and has no spectral stage.

| Abbreviation | Repository path |
|---|---|
| `gts` | `base/trunk/src/grid_type_sh.F90` |
| `gi` | `base/trunk/src/grid_interpol.F90` |
| `tmm` | `base/trunk/src/tmm.F90` |
| `met` | `base/trunk/src/meteo.F90` (the ct-unified overlay `proj/ct-unified/trunk/src/meteo__pclim.F90` carries the same `Meteo_Setup_Mass`) |
| `adv` | `base/trunk/src/advect.F90` |
| `GTS` | `proj/ct-unified/branches/speedup/src/grid_type_ll__speedup.F90` |
| `g3d` | `base/trunk/src/grid_3d.F90` |
| `hyb` | `base/trunk/src/grid_type_hyb.F90` |
| `binas` | `base/trunk/src/binas.F90` |
| `mdat` | `proj/ct-unified/trunk/src/meteodata__pclim.F90` |
| `acfl` | `proj/ct-unified/trunk/src/advectm_cfl.F90` |
| pydyn | `pydyn/balance.py`, `pydyn/kernels.py`, `pydyn/forward.py` |

## Conventions

- Longitude ``λ``, latitude ``φ``, ``μ = \sin φ``, ``g = 9.80665`` m s⁻².
  ``R`` is the radius of the target mesh (`EARTH_RADIUS = 6.371e6` m, equal to
  TM5's `ae`, `binas:51`); it sets face lengths and cell areas. ``R_v`` is the
  `vod2uv` radius that enters ``U`` and ``V`` (§2). TM5 uses ``R = R_v`` = `ae`;
  after owner decision Q4 the two differ.
- Layers count from the top (``k = 1``) to the surface (``k = N``); interface
  ``e`` is the top of layer ``e``, so interfaces run ``e = 1 … N+1``. Hybrid
  interfaces are ``p_e = A_e + B_e p_s``, with ``ΔA_k = A_{k+1} − A_k`` and
  ``ΔB_k = B_{k+1} − B_k ≥ 0``. TM5's `FillMassChange` takes `abs(db)`
  (`g3d:366`), so the sign convention of ``ΔB`` does not matter there.
- ``H = \ln p_s`` (the spectral LNSP field). Fluxes ``F^u``, ``F^v`` and the
  vertical flux ``W`` are rates in kg s⁻¹ integrated over the whole face.
  Raw ``W`` is upward-positive; our stored `cm` is downward-positive.
- TM5 and pydyn arrays are bottom-up. Every reference comparison goes through
  one explicit level-order and sign adapter.

## 1. Truncation, refinement and ring evaluation

**Retained truncation.** TM5 keeps the source truncation: `ShTruncation = T`
(`gi:1587`). For ERA5 this is TL639. The existing route synthesises at the
target truncation (`target_spectral_truncation`, ``T = N_λ/2 − 1`` on LL).

**Refinement.** Each cell edge of length ``Δ`` degrees is divided into

```math
r(T, Δ) = \max\!\left(1, \left\lceil \frac{2\,Δ}{s_T} \right\rceil\right),
\qquad s_T = \frac{360°}{2(T+1)}
```

intervals (`ShRefinement`, `gi:1602–1630`; `nstep = 2` at `gi:1627`). The
call sites carry the comment "5 points per sh resolution" (`gi:1677`, `gi:1822`,
`gi:2458`), but the function uses 2. For TL639 and a 1° cell, ``s_T = 0.28125°``
and ``r = 8``.

**Oversampled rings.** For ``K`` output longitudes, TM5 doubles ``L`` until
``J = K L ≥ 2T + 1``, runs one FFT of length ``J`` and keeps every ``L``-th value
(`gts:1687–1692`, `gts:1731`). The existing `fill_ring_spectrum!`
(`src/Preprocessing/Spectral/spectral_synthesis.jl`) already supports this: on
a buffer of length ``J ≥ 2T + 1``, `ring_max_wavenumber(T, J) = T`, so every
retained zonal wavenumber is synthesised. The unnormalised `bfft` is kept.

**Legendre corner.** TM5 sets ``P_T^T = 0`` deliberately (`gts:713`,
`gts:829–830`). `compute_legendre_column!` evaluates it; this is a documented
difference (§13).

**Cost.** The retained TL639 work is approximately 101 times the legacy
T179, 1° Legendre work: eight times as many refined rows, multiplied by
``639·641/(179·181)``. The earlier estimate of 50 times omitted half the
latitude refinement. Batched GEMM reduces the actual wall-time ratio.
With this implementation on one CPU thread (Julia 1.12.6, BLAS one thread),
a real T639/137-level instant on 360×180 LL takes 84.4 s for synthesis and
quadrature; plan construction takes 8.6 s. A complete window using two
identical copies of that real endpoint and ``q=0`` takes 171–183 s through
pin, merge, balance, dry conversion, closure and replay, excluding plan
construction and GRIB decode. This is a stage benchmark, not a changing
real-hour pair or a golden reference.

## 2. Vorticity and divergence to pseudo-winds

With

```math
δ_{mn} = -\frac{1}{n}\sqrt{\frac{n^2 - m^2}{4n^2 - 1}}, \qquad
σ_{mn} = -\frac{m}{n(n+1)}
```

(`gts:2037`, `gts:2042`), the spectral coefficients of ``U = u\cos φ`` and
``V = v\cos φ`` are

```math
\begin{aligned}
U_n^m &= R_v\left[δ_{mn} ζ_{n-1}^m + i σ_{mn} D_n^m - δ_{m,n+1} ζ_{n+1}^m\right],\\
V_n^m &= R_v\left[-δ_{mn} D_{n-1}^m + i σ_{mn} ζ_n^m + δ_{m,n+1} D_{n+1}^m\right]
\end{aligned}
```

(`gts:1837–1849`). TM5's `shi_vod2uv` treats the truncation edge as follows:

| Row | TM5 (`gts:1949–2045`) |
|---|---|
| ``m = n = 0`` | upper-neighbour term only (`gts:1993–1994`) |
| ``n = T−1``, ``m ≤ T−2`` | upper neighbour ``ζ_T``, ``D_T`` omitted (`gts:2013–2014`) |
| ``n = m = T−1`` | the ``(m, m)`` branch: ``iσ`` term **and** upper neighbour ``(T−1, T)`` included (`gts:1996–1997`) |
| ``n = T``, ``m ≤ T−1`` | lower neighbour only: the ``iσ`` term and the upper neighbour are dropped (`gts:2021–2022`) |
| ``(T, T)`` | zero (`gts:2030–2031`) |

The source gives EMOS compatibility as the reason (vorticity and divergence
assumed truncated at ``T−1``, `gts:1851–1852`). For ``R_v`` it uses `ae`,
6.371 × 10⁶ m (`gts:1951`).

Ours is the existing `vod2uv!` (`src/Preprocessing/Spectral/spectral_synthesis.jl`):
it keeps every neighbour term inside ``[m, T]``, truncates the output at
``n = T`` and uses `IFS_EARTH_RADIUS`, 6.371229 × 10⁶ m, for ``R_v``: the
radius consistent with the ERA5 coefficients (owner decision Q4). The relative
radius difference is ``3.6 × 10^{-5}``.

**Remaining truncation.** For vorticity and divergence truncated at ``T``, the
wind has a nonzero degree ``T + 1``:
``U_{T+1}^m = R_v\,δ_{m,T+1} ζ_T^m`` and ``V_{T+1}^m = −R_v\,δ_{m,T+1} D_T^m``.
TM5 and the legacy `vod2uv!` drop this row. The new construction's
`complete_wind_coefficients` retains it, using the existing transform within
``[m,T]`` and adding the exact neighbour at ``T+1``. This is required for the
Gauss test to converge for a source mode at degree ``T``. The single-mode
tests size both retained contributions explicitly.

## 3. Cell surface pressure: the discrete rule

TM5 evaluates ``e^{H}`` at the refined sample points (`gi:1893`), integrates
each refined row with piecewise-linear interval quadrature in ``λ``
(`IntervalQuad_Lin`, `gi:1907`), integrates the row results with
cosine-weighted linear quadrature in ``φ`` (`IntervalQuad_Cos_Lin`, `gi:1926`)
and divides by the cell area (key `'exp,aver'`, `tmm:1876`). The cell value
``\bar p_{s,c}`` is therefore the exact integral of a piecewise-linear
interpolant of ``e^{H}``; it is neither ``\iint e^{H}\,dA / S_c`` nor
``\exp`` of the mean of ``H``. The construction implements this rule
with the same refinement. The layer mass is

```math
M_{c,k} = \frac{S_c}{g}\left(ΔA_k + ΔB_k\,\bar p_{s,c}\right).
```

## 4. Horizontal face fluxes per native layer

For a meridional face at ``λ_f`` between ``φ_S`` and ``φ_N``, and a zonal face at
``φ_f`` between ``λ_W`` and ``λ_E``,

```math
\begin{aligned}
F^u_{f,k} &= \frac{R}{g}\int_{φ_S}^{φ_N} \frac{U_k(λ_f, φ)}{\cos φ}
             \left[ΔA_k + ΔB_k\, e^{H(λ_f, φ)}\right] dφ,\\
F^v_{f,k} &= \frac{R}{g}\int_{λ_W}^{λ_E} V_k(λ, φ_f)
             \left[ΔA_k + ΔB_k\, e^{H(λ, φ_f)}\right] dλ
\end{aligned}
```

(`tmm:2135–2138`, `tmm:2167–2170`; integrands `gi:3429` and `gi:3784`;
units kg s⁻¹, `tmm:1966`). Native-layer fluxes are summed into the retained
layers (`FillLevels(..., 'sum')`, `tmm:2147`, `tmm:2179`). The product of a
mean wind and a mean face thickness is not this integral: the covariance of
wind and surface pressure along the face matters. TM5 also integrates face
surface pressures (`spu`, `spv`, `tmm:2142`, `tmm:2174`); the plan does not
need them.

**Poles.** TM5's `IntLon` gives zero meridional flux at either pole
(`gi:3781–3782`); ours does the same. TM5 has two polar-row quirks:

- `IntLat` sets the pole sample to the sum of the adjacent row divided by
  `size(llf, 1)`. That row holds the ``N_λ + 1`` face longitudes ``0 … N_λ``,
  so TM5 divides by ``N_λ + 1`` and counts the periodic face twice (`gi:3373`,
  `gi:3396`, `gi:3444`).
- `IntArea_shi_ll_fgh` uses the adjacent-row mean at the pole even for the
  ``D \cos φ`` integrand, which is zero there (`gi:2498–2511`, `gi:2588–2593`).

The construction uses open (pole-free) quadrature nodes in the polar rows instead; the
difference is listed in §13.

## 5. Independent vertical flux

Per native layer, TM5 integrates the horizontal mass divergence directly from
the spectral fields. With the two radii of the conventions, the implemented form is

```math
C_k = \frac{R R_v}{g}\iint D_k\left(ΔA_k + ΔB_k e^{H}\right)\cos φ\,dλ\,dφ
    + \frac{R}{g}\iint \frac{e^{H} ΔB_k}{\cos φ}
      \left(U_k\,∇_λ H + V_k\,∇_φ H\right) dλ\,dφ,
```

with ``∇_λ H = ∂H/∂λ`` and ``∇_φ H = (1 − μ^2)\,∂H/∂μ`` (`tmm:2428`, `tmm:2446`,
`tmm:2450`, combined at `tmm:2468`). TM5 writes the first factor as
``R^2/g`` (`tmm:2468`) because it has ``R = R_v``. Applying Gauss's theorem to
the face fluxes of §4 gives exactly this expression: the divergence ``D`` is
defined on the IFS sphere of radius ``R_v``, ``U`` and ``V`` carry ``R_v``, and
the face lengths carry ``R``. The design note wrote ``R^2/g``; that is corrected
here. With ``R^2/g``, ``W`` and the convergence of the face fluxes would differ
by ``3.6 × 10^{-5}`` of the ``D`` term in every layer, and the discrete Gauss
test of §14 would stall at that level. ``C_k`` is the layer-integrated
divergence ``g^{-1}\iint ∇\cdot(\mathbf v\,Δp_k)\,dA`` [kg s⁻¹]. With the
downward partial sums ``I_e = \sum_{k<e} C_k`` and the column total
``I_s = \sum_k C_k``,

```math
W_e = I_e - B_e I_s, \qquad W_1 = W_{N+1} = 0
```

is the upward vertical mass flux at interface ``e`` (`tmm:2466–2473`,
`tmm:2517`, `tmm:2522`; TM5 forms ``B_e I_s − I_e`` and negates it at
`tmm:2568`). It satisfies the hybrid layer budget
``-C_k + W_{k+1} - W_k = -ΔB_k I_s``. Retained interfaces are selected, not
summed: `FillLevels` (call at `tmm:2558`) chooses the fill mode `'combine'`
when the target levels are a selection of the source levels
(`hyb:1050–1058`, `hyb:2408–2420`). The key `'bottom'` passed at `tmm:2558` is
used only by the `'interpol'` mode (`hyb:2436–2438`).

``W`` must come from the spectral divergence, not from the finite-volume
divergence of the target fluxes; otherwise the per-layer constraint of §6
becomes circular.

**Spectral gradient.** For ``X = H``, the meridional gradient coefficients are

```math
\left(∇_φ H\right)_n^m = (n+2)\,ε_{m,n+1} H_{n+1}^m - (n-1)\,ε_{m,n} H_{n-1}^m,
\qquad ε_{m,n} = \sqrt{\frac{n^2 - m^2}{4n^2 - 1}},
```

from ``(1 − μ^2)\,dP_n^m/dμ = −n ε_{m,n+1} P_{n+1}^m + (n+1) ε_{m,n} P_{n-1}^m``.
TM5's coefficient comment gives the same rows for ``n < T`` (`gts:1788–1795`).
The same comment states the ``n = T`` row as
``−(T−1) ε_{m,T} X_{m,T−1}`` (`gts:1794`), but the executable branch computes
``+(n−1) ε_{m,n} X_{m,n−1}`` (`gts:1804`): a sign error. TM5 also truncates the
gradient at degree ``T``, dropping the ``n = T + 1`` term
``−T ε_{m,T+1} H_T^m``. The plan computes the exact gradient, with output up to
degree ``T + 1`` (owner decision Q3). The error reaches ``W`` through
``V ∇_φ H`` (`tmm:2450`).

TM5 also computes the surface-pressure tendency `tsp` (`tmm:2499–2507`). Only
the PRISM coupling path consumes it (`met:1645–1718`), so the plan does not
build it.

## 6. Per-layer balance

TM5 balances each layer's horizontal fluxes against the hybrid mass tendency
of the window ``[a, b]``:

```math
\dot M_k = \frac{S}{g}\,ΔB_k\,\frac{p_s(b) - p_s(a)}{b - a}
```

(`FillMassChange`, `g3d:366`; `met:1798`, `met:1811`). The layer residual is
``r_k = H_k + W_{k+1} − W_k − \dot M_k``, with ``H_k`` the horizontal
convergence (`GTS:2666–2669`, written here top-down). The horizontal target is
therefore

```math
H_k^\ast = \dot M_k - \left(W_{k+1} - W_k\right),
```

using the ``W`` of the endpoint being balanced. Both endpoint flux sets of a
window are balanced against the same ``\dot M_k`` (`met:1921–1924`,
`met:2033–2036`).

**LL solver.** Per level, TM5 solves a periodic five-point index-space Poisson
problem by FFT: eigenvalues ``2(\cos(2π i/N_λ) + \cos(2π j/N_φ) − 2)``
(`GTS:3028`), zero mode removed (`GTS:3031`, `GTS:3166`), then the west and
south gauges are subtracted so the outer faces stay closed (`GTS:3172–3192`).
The per-layer global-mean residual is left uncorrected; it cancels in the
column sum. pydyn transcribes this solve (`pydyn/balance.py:183–201`).
Our dry-pin route first makes the target solvable: for cell c, solve against
``H^*_{c,k} - (S_c/\sum S)\sum_{c'}H^*_{c',k}``. The pressure guard removes
only the known global water tendency (§9); final dry repair imposes the dry budget.

**CS solver.** The direct target uses the same ``H_k^*`` and area-weighted
removal of its global mode. The CS face table stores outward-positive flux,
so its explicit target is ``-H_k^*``. `DirectCSPoisson` factors a pinned
principal minor of the global degree-four graph Laplacian once per serial run and
reuses it for both endpoint sets, every layer and every window. Corrections
update canonical and mirror entries with the face table's seam signs.
This is TM5's layer equation on our CS graph, rather than TM5's periodic
FFT five-point solver. The existing CS PCG endpoint target keeps its
historical arithmetic through a neutral target-functor split.

A one-thread CPU synthetic solver benchmark gives the following stage times
(68 right-hand sides, graph built independently of meteorology):

| Grid | Cells | Factorization | 68 direct solves | One PCG solve |
|---|---:|---:|---:|---:|
| C90 | 48,600 | 1.01 s | 0.46 s | 0.30 s (705 iterations) |
| C180 | 194,400 | 0.72 s | 1.56 s | 1.59 s (1433 iterations) |

The first factorization includes compilation. Process peak RSS was 0.77 GB
at C90 and 1.06 GB after C180; these are process peaks, not factor sizes.
The smooth synthetic RHS leaves maximum relative direct residuals
``5.15×10^{-9}`` and ``2.49×10^{-8}``, respectively, from conditioning of the
pinned graph. Acceptance uses the physical pressure-normalized guards below;
these timings are neither full-day performance nor GPU measurements.

**Guards.** `CheckMassBalance` compares the column convergence with the
surface-pressure tendency, relative to the mean surface pressure [s⁻¹]
(`GTS:2864–2936`):

| Stage | rms limit | max limit | On failure |
|---|---|---|---|
| pre-check | ``10^{-4}`` | ``10^{-3}`` | abort (`met:1893–1894`, `met:1901–1903`) |
| post-check | ``10^{-7}`` | ``10^{-6}`` | repeat the balance, up to `MB_MAX_TRIES = 3` passes (`met:1445`, `met:1917–1920`), then abort (`met:1943–1946`, `met:2055–2058`) |

These are repeated Poisson passes, not Richardson extrapolation. pydyn raises
on the pre-check (`pydyn/balance.py:211`) but returns `ok = False` after the
third failed post-check (`pydyn/balance.py:212–218`), and its forward driver
ignores the flag (`pydyn/forward.py:134`). The construction follows the Fortran and
raises an error.

**Known difference.** The existing `balance_mass_fluxes!`
(`src/Preprocessing/Spectral/mass_support.jl`) skips a level whose largest
absolute residual is below ``10^{-10}``; TM5 never skips. The skip is harmless
and recorded here.

## 7. Vertical closure and the explicit-dm identity

TM5 interpolates the balanced ``pu``, ``pv`` to each dynamic-step midpoint
(`mdat:760`, `mdat:879`; base `meteodata.F90:764`, `:773`) and calls `dynam0`.
In bottom-up indexing with ``P = \sum_l H_l`` and ``h = Δt/2``
(`adv:210`), it forms

```math
\mathrm{cm}_l = h\left[B_{l+1} P - \sum_{k>l} H_k\right]
```

upward at the upper face of layer ``l`` (`adv:241–256`, `adv:265–266`,
`adv:290`), with ``\mathrm{am} = h\,pu`` and ``\mathrm{bm} = h\,pv``
(`adv:274`, `adv:282`). The archived vertical flux is not advected. The
IO-server B-pivot is disabled in this TM5 version
(`b_pivot_active = .false.`, "TODO #332 B-pivot REVERT", `acfl:928–938`).
pydyn transcribes `dynam0` as `fluxes_from_meteo` (`pydyn/kernels.py:89–112`).

**Identity.** The existing explicit-dm closure
`recompute_cm_from_dm_target!` (`src/MetDrivers/ReplayContinuity.jl`)
integrates ``C_{k+1} = C_k − D_k − Δm_k`` from ``C_1 = 0`` at the top
(top-down, downward-positive). In bottom-up form its upward flux at the top of
layer ``l`` is the mass gained by the layers above it minus what enters them
horizontally. With ``Δm_k = h (B_k − B_{k+1}) P`` and ``B_{L+1} = 0`` at the top,

```math
\sum_{k>l}\left(Δm_k - h H_k\right) = h\left[B_{l+1}P - \sum_{k>l} H_k\right],
```

which is TM5's `cm`, with no assumption about the column residual. TM5's `sd`
also sums from the top (`adv:253–256`); the two differ only in where ``h`` and
``ΔB\,P`` are rounded (about ``10^{-12}`` scaled, not bitwise). The column
residual is zero in exact arithmetic; at rounding level,
`recompute_cm_from_dm_target!` folds it back over the column. With the
endpoint ``Δm`` the same function is today's explicit-dm closure, so the plan
has no separate closure type.
`recompute_cm_from_divergence!` is already the ``ΔB \times`` pit form and stays a
staging helper.

## 8. Pipeline on LL, RG and CS targets

E marks a shared existing function, N a new implemented function.

| # | Stage | Functions | Home |
|---|---|---|---|
| 1 | Read one hour at a time and the next day's hour 00 | `with_spectral_hours`, `read_hour0_spectral`, `validate_spectral_header`; ecCodes indexes and per-hour caches. Full source truncation. Next-day 00 is required. | `Spectral/spectral_hour_io.jl` |
| 2 | Humidity on native levels | E `read_window_qv!` (LL), `load_rg_qv!` (RG), unchanged | `Targets/` |
| 3 | Ring evaluation | `RingSynthesisPlan`, `synthesize_rings`, `synthesize_native_fields`, `complete_wind_coefficients`, `lnsp_gradient`; batched GEMM and oversized inverse FFT | `Spectral/` |
| 4 | Cell PS, face fluxes, ``C_k`` and ``W`` (§3–§5) | N `integrate_surface_pressure!`, `integrate_face_fluxes!`, `integrate_vertical_flux!`; LL boxes, RG per-ring boxes and LCM face segments (§10) | `Spectral/spectral_integral_fluxes.jl` |
| 5 | Dry pin on the integrated PS, fluxes not recomputed (Q1) | E `pin_global_mean_ps_using_qv!` | `Spectral/mass_support.jl` |
| 6 | Moist masses from the pinned PS | `construct_tm5_endpoint`: area × (ΔA+ΔB PS)/g | `Spectral/tm5_endpoint.jl` |
| 7 | Target levels | `TM5EndpointPlan` prepares `plan_vertical` once per day; `apply_vertical!` with `MassField` and `MassFluxField` sums; N `InterfaceMassFluxField` (selects retained interfaces with the body of `ConvectionInterfaceFlux`); E `merge_qv!` weighted by the pinned moist masses. The NOAA levels are `ml137_tropo34a`, not `ml137_tropo34`. | `Vertical/`, `Spectral/` |
| 8 | Per window: balance both endpoint sets against the window's ``\dot M`` (§6) | `balance_window_layers!`, `balance_tm5_layers!`; LL `balance_mass_fluxes!`, RG `balance_compressed_level!` with reused `DirectRingPoisson`; guards and up to three passes | `Targets/`, `MassClosure/` |
| 9 | Dry carrier on target layers, for each balanced endpoint set with its own instant's ``q`` | `dry_target_fluxes!`; LL reuses `apply_dry_basis_native!`, RG uses arithmetic face humidity | `Targets/` |
| 10 | Window flux: mean of the two dry endpoint sets (Q2) | `balance_window_layers!` | `Targets/LatLon/tm5_layer_balance.jl` |
| 11 | Substep schedule, amounts, column balance, `cm`, replay gates | E schedule; amounts via `MetDrivers.flux_application_seconds`; `apply_poisson_balance!` → `balance_column_mass_fluxes!` (LL), `balance_compressed_column_fluxes!` (RG); `recompute_cm_from_dm_target!` with the dry endpoint ``Δm`` | `Targets/`, `MassClosure/`, `MetDrivers/` |
| 12 | Write | E writers; version-5 header with the construction identity | `TransportBinary/`, `MetDrivers/BinaryFormat/` |

For CS, `Spectral/cs_spectral_integrals.jl` implements stages 3–4 on regular
cell-centred rings without pole nodes. The CS target prepares the shared
`cs_spectral_face_geometry` scalar stencils, face normals and conservative box overlaps once,
then supplies them to Spectral without a reverse dependency on MassClosure.
`CSSpectralIntegralWorkspace` applies face and overlap maps on the synthesis
backend with portable KernelAbstractions gathers. Refined three-dimensional
samples remain there through integration; only reduced target fields return
per batch. Surface PS uses Float64; the CS three-dimensional kernels use the
selected synthesis precision with typed constants, without Float64 promotion.

`Targets/CubedSphere/tm5_spectral_day.jl` orchestrates the shared post-integral
pin and moist merge, per-layer CS balance, adjacent-cell face humidity,
endpoint drying and mean. Final dry amounts use the existing CS column
balance, `diagnose_balanced_cs_cm!`, replay and positivity contracts and
streaming writer. After residual booking, the direct CS route explicitly zeros
the rounded surface-interface remainder before replay, enforcing the closed
physical boundary. Positivity retries rebuild amounts from pristine dry rates.
The mandatory next-day endpoint closes the final window; no held-endpoint
fallback is used. The optional live Calc_Kzz group is assembled through the
shared CS PBL writer. Two moist endpoints and one output window are retained.

CS writes `TM5SpectralFluxesHeader` with unchanged v5 identity fields and no
new payload sections. Ordinary provenance keys are
`cs_face_quadrature = "great_circle_midpoint_cardinal16_v2"`,
`samples_per_face`, `refined_nlon`, `refined_nlat` and
`cs_cell_quadrature = "conservative_refined_box_mean_v1"`, plus the actual
synthesis backend/precision. Existing v5 LL/RG files require no new keys.

Folders are under `src/Preprocessing/` unless they start with `MetDrivers/`.
The order differs on purpose from the existing route:

- **Existing route:** pin → recompute fluxes → dry conversion on native levels
  → merge → column balance → `cm`.
- **TM5 route:** moist integration → pin (no recompute) → moist merge →
  moist per-layer balance with ``W`` → dry conversion of each endpoint set on
  target layers → window mean → shared dry column balance and closure.

**Substep retries.** The balanced moist rates [kg s⁻¹] do not depend on the
substep schedule, so a positivity-driven retry recomputes only stage 11 from
the stored rates; no field already balanced in amount units is rescaled. This
is the rule of the cubed-sphere spectral writer
(`src/Preprocessing/Targets/CubedSphere/cubed_sphere_spectral.jl`).

**Memory.** Both targets decode one spectral hour at a time with ecCodes
indexes; the optional decoded cache contains one hour per JLD2 file. The
next-day reader also reads only hour 00. Constructed endpoints retain merged
PS, mass, humidity, face rates and W, without native C. Only two endpoints
survive at a time. RG closes and writes each window immediately. LL closes
each window and keeps one completed lookahead window to form the historical
delta payload; it retains neither a day of outputs nor a day of pristine rates.
Synthesis batches bound the refined-grid field buffers. Float64 surface and
3-D plans share Legendre tables; duplicate face rings share their synthesis.
The coefficient working set at T639/L137 is approximately 1.80 GB per hour,
plus the required next-day hour (3.60 GB combined), versus 43.25 GB for the
previous 24-hour source allocation alone. These decimal sizes exclude plans,
quadrature and target buffers. The handoff records the CPU measured extrapolation.

## 9. Mass consistency under the dry pin

TM5 matches the area mean of the total surface pressure to `p_global =
98500` Pa by an additive offset (`binas:133`; `met:1535–1537`,
`met:1764–1766`; the offset is added uniformly at `GTS:1107–1115`). We keep our
dry pin, `pin_global_mean_ps_using_qv!`, which shifts the surface pressure
additively so that the global dry mass implied by the native humidity equals
`GLOBAL_MEAN_DRY_SURFACE_PRESSURE` (98325 Pa).

- **No normalisation factor is needed.** Pin the integrated PS with the same
  native ``q`` that is merged later, then merge ``q`` with the pinned moist
  masses; `merge_qv!` preserves ``\sum q M`` exactly. Together,
  ``\sum_K (1 − \bar q_K) M_K`` on target layers equals the pinned native dry
  mass in exact arithmetic.
- **Global water term.** The dry pin leaves a global moist tendency equal
  to the global water change. Before each layer solve, remove the layer's
  area-weighted global target, so the target is globally solvable. The guard
  compares the column against the moist tendency minus the known global
  water tendency distributed by area; global dry-mass drift still fails.
  This prevents an index-space mean from becoming a polar pressure error.
  Previously a global fractional water rate G/M appeared as
  ``(G/M)(\bar S/S_c)`` in the guard, with mean/min area ratio about 290 at
  0.25°. Thus even a ``10^{-9}`` s⁻¹ global estimate could give nearly
  ``3×10^{-7}`` s⁻¹ at the smallest cells. The scaled-cap regression exceeds
  the old maximum threshold and passes after the source projection. Final
  dry column repair still imposes the stored dry endpoint budget.
- **Local E − P term.** The dry conversion leaves a local column residual of
  about ``E − P`` relative to the column mass (*estimate:* about ``10^{-9}``
  s⁻¹ for 3 mm per day, up to ``3 × 10^{-7}`` s⁻¹ for 10 mm per hour of heavy
  rain). The shared dry column balance removes it, as in the existing route.
- **``q = 0`` limit.** After the column balance, the dry route's `cm` equals the
  hybrid-B closure in exact arithmetic (covered by the closure and dry-limit tests).

## 10. Reduced Gaussian and direct cubed-sphere specifics

Owner decision Q11: RG ships with LL from the first flux pull request.

- **Geometry.** RG cells are longitude-latitude boxes on each ring with
  midpoint ring boundaries; meridional faces between rings are latitude-arc
  segments of the least-common-multiple subdivision
  (`src/Grids/ReducedGaussianMesh.jl`).
- **Quadrature.** TM5's one-dimensional rules apply exactly with a per-ring
  ``Δλ``. A prefix integral of the refined samples along each face ring gives
  every LCM segment in ``O(1)``. Each ring's FFT length is a multiple of its
  ``N_λ`` that is at least ``2T + 1``.
- **Balance.** Per-level balance on the existing compressed ring Laplacian.
  The column PCG takes about 19 s per window at O90 (measured, header of
  `src/Preprocessing/MassClosure/ring_poisson_balance.jl`); the per-layer
  route needs ``2N_z`` solves per window, so it needs a factor-once direct
  solve of the same Laplacian, reused across levels, endpoint sets and
  windows. The existing column PCG path stays unchanged. On O90 (35,640 cells,
  2,250,376 faces), one-thread compression takes 0.18–0.41 s, factorization
  0.70–0.88 s, and 68 reused layer solves 0.40–0.43 s.

The direct RG spectral path already shares the LL primitives
(`ReducedSpectralSynthesis`; `spectral_to_ring!` and
`compute_reduced_horizontal_fluxes!` in
`src/Preprocessing/Targets/ReducedGaussian/reduced_transport_helpers.jl`).

**Direct CS quadrature.** Let ``Δ_{min}`` be the smallest target face length
in degrees and ``r = quadrature_refinement(T, Δ_{min}, nstep)``. Open,
cell-centred synthesis rings have spacing at most ``Δ_{min}/r`` and an even
longitude count retaining degree ``T+1``. At C90/T639 the shape is
3056×1528; at C90/T255 it is 1528×764. These are oversampled relative to the
retained bandwidth. The run cache retains synthesis plans, face/overlap maps
and the graph factor across matching days.

For the constant Cartesian normal ``N_f`` directed left to right,

```math
F_{f,k}=\frac{1}{g}\int_f
    (\mathbf v_k\cdot N_f)(ΔA_k+ΔB_k e^H)\,dl.
```

The face map uses tensor-product **16-point cardinal interpolation** in
longitude and latitude of the sampled Cartesian wind–pressure product. It
preserves sampled pressure–wind covariance. It integrates that interpolant
with equally spaced **midpoints in great-circle arc length**, including the
slerp Jacobian. This differs from TM5's uniform-latitude/longitude trapezoids
of ``UΔp/\cos φ`` and ``VΔp``. The sample count is
``N=\max(16,4r,\lceil(T+2)θ_{max}/0.1\rceil)``, where ``θ_{max}`` is the
largest target edge angle in radians. It is 112 at C90/T639 and 45 at
C90/T255. The tighter midpoint spacing bounds the error of modes oscillating
*along* the face as well as the interpolation error across it. Increasing
only the old midpoint count would leave its transverse bilinear damping.
The previous 24-point bilinear C90/T639 rule attenuated the 637/637 meridian
probe by 20.70%; it is no longer the direct TM5 CS default. The existing
point-sampled CS, LL/RG and general `LineIntegralFaceFluxes` paths retain
their original arithmetic.

A zero face integral has no finite relative error. Tests therefore check
complex harmonic phases against an absolute integral envelope and also
report ordinary relative error when the integral is at least 10% of that
envelope. The envelope is the integral of the absolute harmonic amplitude;
underflowed modes below ``10^{-13}`` of the largest envelope are excluded.
Every retained scalar harmonic through T255 and T639 is tested on C90-like
meridians (including halfway between source longitudes), a tilted actual C90
edge, a polar edge and an equatorial arc, against independent 256-node
Gauss integration. Vector harmonics and converged TM5 comparisons are tested
in the same file, `test_tm5_cs_harmonic_accuracy.jl`. The maximum ordinary
relative error is 0.04075% against exact integration and 0.04091% against
converged TM5; the maximum envelope-normalized error is 0.02747%. These
sweeps cover all 205,120 retained scalar modes and both vector families at
T639, and all 32,896 at T255, on each of five faces. Selected high modes
also exercise the portable gather in Float32 and Float64. This is a measured bound
on those faces and test fields, not a theorem for arbitrary pressure spectra:
``e^H\mathbf v`` is not band limited even when ``H`` and the wind are.

**Accuracy reference.** The CS path targets converged physical integrals
(owner decision, 2026-11-02). Default TM5 IntLon has one-degree retained-mode
errors of −18.7414% at T255 and −16.6374% at T639; its default and exact
integrals cannot simultaneously satisfy a 0.1% bound. Tests preserve those
counterexamples and compare accuracy with the IntLat/IntLon rule at converged
refinement (1024 subdivisions). Section 13 gives the measured cause, the
comment/code refinement mismatch and the real-ERA5 difference. General tilted
CS edges have no corresponding TM5 meridian/parallel face.

**Polar evaluation.** A scalar transform of pseudo-wind coefficients obtains
small ``U`` by subtraction before division by ``\cos φ``. On the C90/T639
outermost ring at 89.941099°, the Float32 degree-one zonal rotation had
5.9582% relative error (physical ``u=0.00567218`` m/s). CS now evaluates
vector harmonics from latitude derivatives of associated Legendre functions
where ``|\cos φ|<0.025``. Derivative tables use ``\cos φ`` directly and a
differentiated recurrence; their device tables, potential coefficients,
GEMMs, FFTs and replacement kernel all use the selected 3-D precision.
No Float64 arithmetic is introduced in Float32 device kernels. The same
probe's relative error is ``4.51×10^{-7}`` at T639 and ``4.86×10^{-7}`` at
T255. Tests also exercise tilted rotation and divergent degree-one modes.
Selected near-truncation vector modes on production rings have maximum
Float32 relative wind error ``3.40×10^{-7}`` against direct Float64 values.
Surface synthesis retains the existing Float64 policy.

The old bilinear source sampler **clamps** latitude poleward of its outermost
ring. The new high-order CS map instead continues Cartesian products through
the poles by reflecting latitude and shifting longitude by π. This supplies
centred stencils even for face samples closer to a pole than the outermost
ring. Neither policy reproduces TM5's duplicate-longitude polar mean or its
substituted ``D\cos φ`` endpoint. Ordinary LL/RG polar policy is unchanged.

PS remains the conservative overlap mean of ``e^H`` box samples. Independent
``C_k`` retains the same overlap on the §5 forcing density, including
``R_v/(Rg) D_k Δp_k`` and
``p_s ΔB_k(U_k H_λ+V_k H_φ)/(Rg\cos^2 φ)``. Its integral feeds W and the
existing layer solve. The face accuracy change does not turn that box rule
into an exact spectral cell integral; Gauss/closure/replay tests cover its
separate discretization and budget.

**Cost and choice.** `cs_exact_samples` supplies a portable, batched direct
vector-harmonic evaluation at arbitrary face points without interpolation.
It uses selected-precision Legendre/derivative recurrence, Fourier sums and
compensated accumulation. Bounded chunks need no per-point Legendre table.
A warmed T639 eight-level, 64-point chunk took 1.388 s on one CPU thread.
Straight scaling to every C90 face midpoint and two L137 endpoints gives
8.09 million seconds per window, versus 123 seconds for the previous
ring-synthesis/face-gather workload. This is an extrapolation of the direct
implementation, not an optimized non-uniform FFT or exact-latitude ring
implementation. A separate exact-sample-latitude geometry count, grouping
absolute latitudes within ``10^{-12}`` radians and exploiting hemisphere
symmetry, finds 685,507 distinct latitudes at C90 and 1,366,010 at C180.
A conventional retained Legendre table would require 1.128/2.249 TB in
Float64 (half in Float32), or 449/894 times the current latitude-contraction
work before exact longitude evaluation. Chunking removes that storage
requirement while retaining the contraction work. These are measured
geometry counts and storage/work estimates, not NUFFT timings.
The measured direct point implementation exceeds the 5× threshold by a
wide margin, so the high-order oversampled-ring alternative is the default. Its integrated
stencils are combined into one reusable scalar face stencil, with each
Cartesian normal stored once. The gather reconstructs weights in the original
X/Y/Z order, preserving the expanded map's arithmetic, including zero-normal
products. Float32 stores an Int8 rounding offset per component/stencil entry
so that the reconstructed weight equals the original once-rounded Float64
geometry product, without Float64 device arithmetic. This avoids both a
samples×16² allocation per level and triplicated scalar geometry.

Before geometry compaction, C90/T639 warmed batch costs were 3.376 s for eight-level U/V/D synthesis,
0.143 s for polar replacement and 1.325 s for the face gather (previous gather:
0.201 s). Two L137 endpoint equivalents cost 166 s versus 123 s, a 1.35×
ratio. These window estimates exclude common surface/cell overlap, balance,
closure and IO work. Setup took 55.6 s for the face map. The former expanded Float64
portable gather stored 1.540 GB, shared synthesis plans 3.129 GB, and polar
plans 0.118 GB; estimated live synthesis/face buffers total 7.07 GB, excluding
source hourly coefficients, cell overlap and graph factors. Setup temporarily
retains the 2.165 GB host sparse map. Sizes are decimal GB; times exclude
compilation after warmup except the first geometry setup.

A CPU-only GPU model at 500 GB/s effective gather bandwidth gives 1.41 s
per two-endpoint L137 C90 window-equivalent for the new gather alone
(1.48 s including the padded final batches). A 1 TFLOP/s FP64
recurrence model gives 27,533 s for the direct point method. These are
estimates, not GPU measurements, and omit GEMM/FFT, launch and transfer costs.
Actual GPU execution remains untested in this CPU-only run.

The corresponding pre-compaction C180/T639 benchmark retains 3056×1528 rings and uses 56 face
midpoints. Warmed eight-level synthesis, polar replacement and gather cost
3.382, 0.133 and 3.853 s; the old gather costs 0.364 s. Two-endpoint L137
equivalents are 252 s versus 128 s (1.97×). The expanded gather stored 4.864 GB, estimated live
synthesis/face storage is 10.41 GB, and the temporary host sparse map is
6.595 GB. Face-map setup takes 119.8 s. Direct-point scaling gives 16.20
million CPU seconds per window. Under the same GPU assumptions the new
gather is 4.44 s (4.67 s with padding) and the direct recurrence 55,066 s. These timings use Julia
and BLAS one thread, batch eight, Float64, with no actual GPU execution.
The benchmark is retained at
`scripts/benchmarks/bench_cs_spectral_quadrature.jl`.
Set `ATMOSTR_BENCH_EXACT_LATITUDES=1` to reproduce the exact-ring geometry
count, and `ATMOSTR_BENCH_FULL_WINDOW=1` to time complete face-path windows.
The latter includes all 36 padded batches plus the production Cartesian
pressure–wind/forcing kernel. C90/C180 measured 198.1/292.1 s, with peak
process RSS 14.63/23.79 GB, including setup and benchmark scratch in the peak. This still
excludes common surface synthesis, cell overlap, balance, closure and IO;
it is not a full preprocessing-window timing. The day driver reuses the
previous endpoint, so steady-state face work is approximately 99.1 s/window
at C90 and 146.1 s/window at C180; a 24-window day builds 25 endpoints,
giving approximately 103.2/152.2 s/window before the common stages and
amortized setup. The GPU gather model likewise halves to 0.74/2.33 s per
steady-state window with endpoint reuse. These full-window timings and GPU
models describe the expanded geometry before the storage optimization.

The compact map shares scalar stencil indices/weights and each face normal.
A paired CPU benchmark against the pre-compaction fix measures:

| T639, Float64, batch eight | C90 | C180 |
|---|---|---|
| Expanded → compact portable geometry (decimal GB) | 1.539983 → **0.515920** | 4.863692 → **1.631599** |
| Compact host geometry (GB; transient during setup) | 0.687375 | 2.173391 |
| Expanded → compact setup (s) | 54.52 → 48.98 | 119.51 → 105.31 |
| Expanded → compact gather median (s; three warmed trials) | 1.2820 → 1.2348 | 4.1868 → 3.8669 |
| Compact Float32 geometry including rounding offsets (GB) | 0.471987 | 1.491874 |

The live-buffer estimates above decrease by the geometry reduction alone:
approximately 6.05/7.18 GB instead of 7.07/10.41 GB. These are estimates,
not new peak-memory measurements. The benchmark process itself retains both
maps and reference scratch; its 44.68 GB peak is not a production requirement.
Reconstruction preserves every weight and the X/Y/Z accumulation order.
All 4.86 million production-sized output values checked across C90/C180 and
Float32/Float64 match the pre-compaction fix **bitwise**. The CS integration
and day fixtures also match bitwise (76 and 4,356 numeric leaves respectively,
including all 24 windows in Float64, Float32 and live-PBL fixtures). Timings
use one Julia/BLAS CPU thread; setup includes first-call compilation, while
gather medians exclude warmup. No GPU execution is claimed. The compact
gather model would read 36 rather than 60 bytes per scalar entry/level
(12 for shared index/weight and 24 for three source values), but this is a
traffic estimate rather than a GPU measurement.

**Real endpoint check.** Local ERA5 T639 GRIB endpoints on 2021-12-01 at
00:00 and 01:00 were checked at native levels 1, 70 and 137 on four C90
faces (central, tilted, polar and panel boundary). Actual ring-synthesized
Cartesian pressure–wind products use the production cardinal weights and
are compared with 256-node Gauss integration of direct spectral point
values. All eight endpoint/face checks pass; maximum scaled relative error
is 0.002885%. This checks individual endpoint rates before balancing; it
is not a full-day ERA5 W or transport comparison. The reproducible opt-in
file is `test/real_data/test_tm5_cs_endpoint_accuracy.jl`.

## 11. Types, dispatch and configuration

```julia
abstract type AbstractSpectralFluxConstruction end
struct PointSampledSpectralFluxes <: AbstractSpectralFluxConstruction end  # default: today's bodies
struct TM5SpectralFluxes <: AbstractSpectralFluxConstruction
    refinement_steps :: Int   # TM5 ShRefinement nstep = 2
    vod2uv_radius    :: Float64     # IFS_EARTH_RADIUS (Q4)
end
```

| Dispatch point | Default method | `TM5SpectralFluxes` method |
|---|---|---|
| `spectral_resolution(c, grid, T_source)` | `target_spectral_truncation(grid)` | `T_source` |
| LL `process_window!(c, …)`, RG `synthesize_and_merge_window!(c, …)` | current body, untouched | stages 3–7 |
| `process_day(c, …)` | legacy daily workflow | hourly read, two endpoints, `balance_window_layers!` for stages 8–10, per-window closure/write |

- Backend choices are parsed into `CPUSynthesis` / `GPUSynthesis`, with
  `AutoSynthesisPrecision` or a float type; dispatch constructs the actual plan.
- `InterfaceMassFluxField <: AbstractFieldKind`
  (a kg s⁻¹ interface flux is not a convective flux, even where the selection
  body is the same).
- Direct CS dispatches on `TM5SpectralFluxes` and uses
  `CSSpectralIntegralWorkspace`; legacy face-method choices retain their meaning.
- `[preprocessing].spectral_flux_construction` is parsed once with
  `ConfigChecks.parse_choice`; absence adds no fields to historical settings,
  preserving their fingerprint representation. `flux_construction` and the
  existing face-method keys retain their meaning. `TM5SpectralFluxes` has the
  fixed IFS radius and a positive `quadrature_refinement_steps` (default 2).
- `ConfigChecks.check_known_keys` rejects unknown preprocessing keys, and native
  sources reject all four spectral-only keys. Planning rejects moist basis and sampling other than
  `window_mean`. The day driver requires next-day 00 UTC. The new route
  requires retained native interfaces, rather than pressure-overlap remapping.
- `[preprocessing] include_tm5_pbl = true` can be combined with this construction
  on LL, RG and CS. The v5 binary retains the `tm5_spectral` construction identity
  and includes the complete live Calc_Kzz input group. Set `input.surface_dir`
  and `input.thermo_dir`, then select `[diffusion] kind = "tm5_calc_kzz"` at
  runtime (see [TM5 boundary layer](../theory/tm5_boundary_layer.md)).
  Both constructions share the day-local PBL input cache, area-mean surface
  fields and T/q assembly, including next-day 00 UTC endpoints.
  T/q merge weights use each construction's stored window-start total surface
  pressure. Merged T/q can therefore differ between constructions beyond
  rounding, consistent with each binary's hybrid layers.
- The default path stays bitwise unchanged; every flux pull request gates the
  `pre_ll72`, `pre_o24` and `pre_c24` goldens.

```@docs
AtmosTransport.Preprocessing.TM5SpectralFluxes
AtmosTransport.Preprocessing.PointSampledSpectralFluxes
```

## 12. Binary identity

A `TM5SpectralFluxes` binary is written as **format version 5**: version 4
plus the construction identity. This part is implemented (TM5-0); see
[Binary format](../concepts/binary_format.md) for the full contract.

| Header key | Value |
|---|---|
| `format_version` | `5` |
| `flux_construction` | `"tm5_spectral"` or `"point_sampled_spectral"` |
| `flux_construction_revision` | revision of the construction (tm5_spectral only) |
| `spectral_truncation_used` | retained truncation ``T`` (tm5_spectral only) |
| `quadrature_refinement_steps` | `nstep` of the refinement rule (tm5_spectral only) |
| `vod2uv_radius_m` | `vod2uv` radius in m (tm5_spectral only) |

The reader decodes the keys into `TM5SpectralFluxesHeader` or
`PointSampledSpectralFluxesHeader`; a version-4 header decodes to
`NoFluxConstructionHeader`. Readers accept versions 4 and 5. Writers produce
version 5 only when the identity is recorded, so every existing output stays
version 4. The mesh radius stays in the existing `planet_radius_m`. The
window flux uses the existing version-4 sampling tags
(`source_flux_sampling = "window_mean"`, `flux_sampling = "window_constant"`);
the new LL and RG route records these tags from its construction.

The v5 headers also record `spectral_synthesis` (`cpu`/`gpu`),
`spectral_synthesis_float` (the actual Float64/Float32 plan type), and
`spectral_surface_float` (Float64). CUDA's math mode is set to PEDANTIC
for each transform and restored even on exceptions; FAST_MATH is rejected.

## 13. Differences from TM5

We reproduce TM5's equations and modelling choices, evaluated accurately; we
**do not reproduce TM5's numerical errors**. In particular, the direct CS path
targets converged face integrals, rather than agreement with TM5's default
under-resolved quadrature (owner decision, 2026-11-02). There is no
TM5-default-compatible CS quadrature option. The two groups below distinguish
numerical corrections from modelling or convention choices; unisolated sizes
are explicitly marked **not measured**.

**Deliberate corrections of TM5 numerical errors**

| Where TM5 differs | Here | Why | How much |
|---|---|---|---|
| ``n=T`` Nabla sign error and gradient truncated at ``T`` (§5; `grid_type_sh.F90:1804`) | Exact LNSP gradient through degree ``T+1`` | The top-row sign in TM5 is a code error; differentiation also generates a nonzero extra degree | Sign-only single-mode difference ``2(T-1)ε_{m,T}H_{T-1}^m``; real L96 forcing proxy 0.41% global / 26.9% cap divergence RMS (qualifications below). Separate missing-degree atmospheric effect: **not measured** |
| ``P_T^T=0`` (§1) | Evaluate ``P_T^T`` | A retained Legendre function must not be suppressed | Real L137 global Δu RMS about ``3.5×10^{-4}`` m/s; unit real corner amplitude ``2P_T^T`` |
| `vod2uv` edge rows ``n=T−1``, ``m=T−1``, ``n=T`` (§2) | Retain every neighbour and ``iσ`` term inside ``[m,T]`` | Complete the vector transform at truncation | Combined truncation-edge probe: L137 cap Δu/Δv RMS 0.140/0.176 m/s (3.53/4.39%); omitted terms ``−R_v δ_{m,T}ζ_T^m``, ``iR_v σ_{m,T}D_T^m`` and V counterparts; special ``m=n=T−1`` branch differs by exactly zero |
| Wind row ``n=T+1`` dropped (§2) | Complete transform retains it | Required by the vector transform and Gauss identity | Unit VO/D contributes ``±R_v δ_{m,T+1}``; isolated atmospheric size **not measured** |
| Polar endpoint ``1/(N_λ+1)`` mean; adjacent-row ``D\cos φ`` substitution (§4) | LL/RG open polar nodes; CS stable vector evaluation on open rings (§10) | Remove the duplicated-longitude mean and artificial endpoint; avoid cancellation of small pseudo-winds before division by ``\cos φ`` | L137 Fu cap RMS about 8%, L1 cap divergence RMS 40% in the probe below. Unit cosine has spurious mean ``1/(N_λ+1)``; constant-D endpoint adds ``(h/2)sin(h)``. CS Float32 outer-ring rotation error 5.9582% → ``4.51×10^{-7}`` relative at T639; 0.8471% → ``4.86×10^{-7}`` at T255 |
| Default IntLat/IntLon refinement versus converged CS face quadrature (§10) | Cardinal-16 Cartesian interpolation and resolved arc midpoints; coordinate-face checks use converged IntLat/IntLon | Integrate the retained oscillations accurately; default trapezoids have a measurable integration error | Unit retained-mode default misses −18.7414% (T255), −16.6374% (T639). Real ERA5 default/converged difference: **0.02655% relative L2 over 48 sampled rates**; **1.7009%** at 80°E, 30–31°N, L137, 2021-12-01 01:00 UTC (**−501,352 vs −492,967 kg/s**). CS harmonic/exact maximum 0.04075% on the §10 sweep; real CS endpoint maximum 0.002885% |
| `π` literal `3.14159265358979` (`binas:35`) | Julia's ``π`` | Use the available constant precision | Relative difference ``1.0×10^{-15}`` |

The default face-integration error is the **composite trapezoid error of a
coarsely sampled oscillation**, not an error in the exact face integral or a
CS-only interpolation effect: a 1° ``V=cos((T-2)λ)`` probe reproduces
−18.7414% / −16.6374%, agreeing with the analytic trapezoid response to roundoff.
The code uses **2**, despite comments saying **5** points per spectral
resolution; even setting 5 leaves −2.5518% / −3.1995%, rather than removing
the error.

The inspected source is `deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/`:
`grid_interpol.F90:1624–1626` defines
``s_T=360°/[2(T+1)]``, ``nstep=2`` and
``r=max(1,ceil(nstep·Δ/s_T))``. Its call-site comments say “5 points per sh
resolution” (`grid_interpol.F90:1675`, also 1820, 2006, 2224, 2713 and 2941).
The CarbonTracker tree agrees (`base/trunk/src/grid_interpol.F90:1626–1628`,
comment at 1677). These routines are in **grid_interpol**, rather than
`grid_type_sh.F90`, which contains the spectral primitives and Nabla bug.
IntLat applies the trapezoid to ``U·Δp/\cos φ`` in latitude
(`grid_interpol.F90:3248,3275`); IntLon applies it to ``V·Δp`` in longitude
(`grid_interpol.F90:3609,3629`), with ``U=u\cos φ``, ``V=v\cos φ``.
The common radius/gravity factors do not affect the relative quadrature error.

For 1° faces, default T255/T639 subdivisions are **3/8**, giving spacings
**0.333333°/0.125°**. The shortest retained longitudinal wavelengths
``360°/T`` are 1.411765°/0.563380°, hence **4.2353/4.5070 samples per
wavelength** (spacing 0.23611/0.221875 of a wavelength). They are above the
Nyquist sampling minimum of two, but far too coarse for accurate trapezoid
integration. The following measurements integrate the retained ``m=T-2``
cosine from 0° to 1° with constant pressure thickness. For each sinusoid,
the measured relative response equals ``x cot(x)-1``, where
``x=mΔ/(2r)`` in radians, to better than ``10^{-12}`` in relative error.

| ``nstep`` | T255: subdivisions / error | T639: subdivisions / error |
|---|---|---|
| 2 | 3 / −18.741384% | 8 / −16.637396% |
| 3 | 5 / −6.585504% | 11 / −8.661243% |
| 5 | 8 / −2.551821% | 18 / −3.199522% |
| 8 | 12 / −1.130926% | 29 / −1.227786% |

Holding the default spacing fixed isolates the along-face frequency:

| Along-face ``m`` (T255 / T639) | T255 error | T639 error |
|---|---|---|
| 1 / 1 | −0.00028205% | −0.00003966% |
| 63 / 159 | −1.121983% | −1.004755% |
| 126 / 318 | −4.518499% | −4.043503% |
| 253 / 637 | −18.741384% | −16.637396% |

Thus high **along-face** frequency causes the miss. A near-sectoral harmonic
varying slowly along a meridian can instead expose transverse interpolation
error (§10); it does not test this trapezoid defect. The analytic response
and direct sums separate those two causes. The reproducible 32-case sweep
(`nstep=2,3,5,8`, four wavenumbers at each truncation) is produced by
`scripts/diagnostics/tm5_default_quadrature.jl harmonics OUT`; its `day` and
`native-day` commands measure the ERA5 impact below. See
`scripts/diagnostics/README.md` for definitions and commands.

**Real ERA5 1° LL day: integration-only comparison.** On 2021-12-01,
all 24 hourly UTC analyses and all 137 native levels were integrated on every
meridian and parallel face, using identical complete wind coefficients and surface pressure on both
sides. TM5-default here means the eight-subdivision IntLat/IntLon rule,
including its meridian polar endpoint means; the reference uses 16-node
Gauss latitude integration and Fourier primitives of longitude products.
Doubling to 32 Gauss nodes and 11,520 rather than 5,760 product longitudes
at hour 00 (levels 1/70/137) changes U/V relative L2 by only
``8.23×10^{-13}`` / ``3.96×10^{-16}``. The independently evaluated review
case is −501,352.1089 versus −492,967.1089 kg/s, confirming its 1.7009%
difference.

Below, Δ is default minus converged. RMS is unweighted over faces and
hours, in kg/s; relative L2 is ``rms(Δ)/rms(reference)``. Rates use the IFS
radius on both sides, as in the independent review; this common scale does
not affect relative errors. Caps are absolute
face latitude >85°; meridian faces use interval centres and parallel faces
their latitude boundary. The parallel statistics include zero polar rates.
The last column excludes absolute latitude ≥89°, separating ordinary
trapezoid error from the outer polar interval. The cap meridian discrepancy
includes TM5's polar endpoint-mean policy; it must not all be attributed to
the sinusoidal trapezoid response above.

| Native level / rate | Global Δ RMS (kg/s) | Global relative L2 | Cap Δ RMS (kg/s) | Cap relative L2 | Relative L2, latitude <89° in magnitude |
|---|---|---|---|---|---|
| 1 / Fu (meridian) | 2,180.504 | 0.316011% | 9,246.98858 | 2.844701% | 0.009494% |
| 1 / Fv (parallel) | 19.426 | 0.013821% | 0.01043 | 0.000075% | 0.013821% |
| 60 / Fu (meridian) | 200,272.242 | 0.187393% | 847,350.73076 | 2.419952% | 0.013917% |
| 60 / Fv (parallel) | 8,662.252 | 0.026106% | 1.87134 | 0.000142% | 0.026106% |
| 70 / Fu (meridian) | 216,055.058 | 0.107259% | 908,277.32706 | 2.102330% | 0.014501% |
| 70 / Fv (parallel) | 17,769.944 | 0.023305% | 3.47345 | 0.000215% | 0.023305% |
| 96 / Fu (meridian) | 633,818.123 | 0.211811% | 2,643,025.22090 | 2.653127% | 0.039553% |
| 96 / Fv (parallel) | 50,661.231 | 0.037187% | 23.91611 | 0.000536% | 0.037187% |
| 137 / Fu (meridian) | 84,455.719 | 0.568389% | 353,524.19361 | 3.715678% | 0.093682% |
| 137 / Fv (parallel) | 6,792.446 | 0.078181% | 5.67749 | 0.001383% | 0.078182% |

The full 137-level profile and the 32 harmonic cases are retained in
`scripts/benchmarks/results/tm5_quadrature_20261102/`, with definitions and
reproduction commands. The raw day has 6,576 hour/level/component statistics
covering 427,308,480 face/level/time rates, including zero polar parallel
rates. All 144 statistics from the independent three-level day agree exactly
with their entries in the all-level pass. Pooling every level and component
gives relative L2 **0.252418% globally**, **3.32820% in caps**, and
**0.0370244% with absolute latitude <89°**. The maximum daily per-level Fu
relative L2 is 0.670047% at level 129; the Fv maximum is 0.078181% at level
137. The full native-level pass used
about 48 CPU minutes including setup and a discarded partial hour; the two
preliminary/convergence passes bring the measurement work to about one CPU
hour. Independent hourly chunks reduced wall time; no GPU was used.

These are raw analysis face rates, before pin, balance, drying or merging,
rather than a balanced transport-day or W comparison. Restricting caps to
``85°<|latitude|<89°`` gives Fu relative L2 0.01135%, 0.02259%, 0.09181%
at levels 1/70/137 (Fv: 0.00007582%, 0.00021812%, 0.00142049%). The
large Fu differences are concentrated in the outer polar degree, where TM5
also substitutes pole means. This comparison does not isolate that
substitution from trapezoid error within that degree; ordinary trapezoid
error remains visible away from it. The original 48-rate
review sample above has a different spatial/level selection and is not a
global or day-wide statistic.

The [TM5-FLUX-2a real-day memo](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/main/docs/memos/2026-11-01_tm5_flux_real_day.md)
records the default-versus-TM5 construction comparison, balance diagnostics
and flux × PBL transport experiment. Reproduction commands and CSV definitions
are in `scripts/diagnostics/README.md`.

The atmospheric sizes below are measured in the independent **2026-10-31
WIND_DIFF_PROBE** report (`WIND_DIFF_PROBE.md` in that experiment's output directory),
using the original ERA5 T639 analysis coefficients for 2022-01-15 00–01 UTC.
They cover native levels 1, 60, 96 and 137 on a common 1° target, before pin,
balance, drying or merging. Caps mean |latitude| > 85°. They are raw flux and
wind differences, not final transport errors or full-day statistics.

| Difference | Real T639 size (probe sections: winds, fluxes, Nabla proxy) |
|---|---|
| Truncation-edge wind rows | L137 cap RMS Δu=0.140 m/s (3.53%), Δv=0.176 m/s (4.39%); cap maxima about 1.1 m/s. On the nearest extra open polar rings, maxima rise to about 11 m/s. |
| Point sampling versus face integration | Global raw net-divergence RMS changes by 9–26%; face RMS by 0.3–6.1%, cap net divergence by 8–54%. Includes wind–pressure covariance and face averaging. |
| Open polar quadrature versus TM5 endpoint rules | L137 Fu cap RMS changes by about 8%; L1 cap net-divergence RMS by 40%. The top-level reference divergence is small. |
| Wind radius convention | 0.003594% coherent wind scale change (approximately 0.0036%). |
| n=T Nabla sign | No direct wind/face effect. L96 forcing proxy δC is 0.41% of global and 26.9% of cap horizontal-divergence RMS. The proxy uses 1° latitude midpoints and eight longitude trapezoids, not full refined IntArea or a full-column W comparison. |
| Suppressed Legendre corner | L137 global u RMS about 3.5×10⁻⁴ m/s; negligible aloft and in polar caps; redundant once the corner wind coefficient is zeroed. |
| Special n=m=T−1 branch | Exactly zero isolated difference on the measured coefficients. |


The unit-coefficient values below are reproducible analytic test anchors;
they are not atmospheric amplitudes. The separate missing T+1 gradient,
full-column forcing and final balanced transport effects were not isolated
by this four-level probe.

For the unit-coefficient tests at ``T=7``, the missing ``n=T+1`` LNSP
gradient at ``m=2`` is −3.39549875051. A unit ``ζ_T^1`` has the omitted
``n=T−1`` wind coefficient +451573.541727 m s⁻¹ and ``n=T+1`` coefficient
−395852.518767 m s⁻¹. A unit ``D_T^2`` has the omitted ``n=T`` zonal
coefficient −227543.892857i m s⁻¹; the ``m=T`` corner is −796403.625i m s⁻¹.
These are unit spectral amplitudes, not measured atmospheric winds. The
special ``m=n=T−1`` branch contributes zero discrepancy. At polar interval
``h=0.0125`` radians the substituted D endpoint adds 7.81229655107 × 10⁻⁵
to the unit-integrand quadrature.


**Deliberate modelling or convention choices**

| Where TM5 differs | Here | Why | How much |
|---|---|---|---|
| Total-pressure area mean additively matched to 98500 Pa (§9) | Dry pin after flux integration, without recomputing fluxes | Fix our dry carrier mass while preserving TM5's ordering of integration before pinning | Global water term, *estimate* ``10^{-11}``–``10^{-9}`` s⁻¹ |
| Forecast `h06h18tr3` records and lead bookkeeping; balanced fluxes interpolated to dynamic-step midpoints | Hourly analyses; average two endpoint flux sets, each balanced against the same tendency | Use the available analysis archive and its window-mean flux contract | Endpoint mean exactly integrates the assumed linear interpolation; forecast-versus-analysis effect **not measured** |
| NOAA 1° archive plus offline coarsener | Integrate directly on the target | Avoid an intermediate archive and integrate the target's own boundaries | Direct coarse integrals can differ from sums of 1° fluxes; magnitude **not measured** |
| Mass-weighted coarsening of 1° T and Q | Reader's target-cell point ``q``, merged with pinned moist masses | Humidity is supplied as a grid-point field rather than spectral coefficients | Isolated difference **not measured** |
| `vod2uv` radius `ae=6.371×10^6` m | IFS radius ``R_v=6.371229×10^6`` m | Match the radius used to define the ERA5 coefficients | 0.003594% coherent wind scale change (relative ``3.6×10^{-5}``) |
| Single-radius ``R²/g`` multiplying D in ``C_k`` | ``R R_v/g`` (§5) | Keep independent W consistent with winds synthesized using the IFS radius and faces using the mesh radius | Coefficient ratio ``R_v/R``; for ``R=6.371×10^6`` m, relative ``3.6×10^{-5}``; isolated final W effect **not measured** |
| Latitude/longitude trapezoids on coordinate faces | CS uses midpoints in great-circle arc length, with the arc Jacobian (§10) | CS edges are great circles; their natural integration coordinate is arc length | Isolated rule-versus-coordinate effect **not measured**; combined CS accuracy bounds in §10 |
| IntLat and IntLon on meridians and parallels | Integrate ``(v·N_f)Δp`` on each CS edge with its constant Cartesian normal | General CS faces are neither meridians nor parallels, so IntLat/IntLon are not directly applicable | No one-to-one TM5 face comparison on tilted edges; isolated effect **not measured**; coincident coordinate-face convergence is tested |
| Polar endpoint means on LL | CS uses stable vector evaluation, Cartesian pole reflection and longitude shift; ``asin`` argument clamped to ``[-1,1]`` for roundoff | Continue smooth Cartesian products through the pole without an artificial endpoint mean. The previous bilinear latitude clamp is superseded on the direct CS path | Vector cancellation improvement above; separate reflection/clamping atmospheric effect **not measured** |
| IntArea on LL boxes | Conservative refined-box overlap for CS PS and independent C | Supply cell means and forcing on CS polygons without a new spectral polygon integral | Separate exact-cell-integral difference **not measured**; Gauss convergence, closure and replay tests in §10 |
| Balance never skips a level | Skip residuals below ``10^{-10}`` in `balance_mass_fluxes!` | Avoid solving negligible residuals | Uncorrected residual bounded by ``10^{-10}`` at the skip; final transport difference **not measured** |
| `tsp` and `'bottom'` FillLevels key | Not built | No consumer in our construction | Atmospheric effect **not measured**; neither output is consumed here |
| Check_CFL, polar clustering, prognostic slopes, diffusion inside convection, window-end convection | Existing runtime | This change implements spectral flux construction, rather than a TM5 runtime replacement | Combined runtime difference **not measured** |

## 14. Validation tolerances

| Boundary | Criterion |
|---|---|
| Spectral primitives (analytic) | Float64, ``10^{-12}`` scaled, with an absolute floor tied to the signal scale |
| Quadrature | Observed convergence order matches the rule; the discrete Gauss test (face fluxes against ``C_k`` per cell) converges with refinement |
| Per-layer balance against pydyn | about ``10^{-12}`` scaled; retry decisions exact away from thresholds. FFTW and NumPy FFT differ at rounding level (`pydyn/balance.py:9–12`), so bitwise equality is not claimed |
| Closure identity (§7) | ``≤ 10^{-12}`` scaled |
| Binary and replay | exact same-type round trip; replay ``10^{-12}`` (Float64) and ``10^{-5}`` (Float32) |
| Float32 storage | equal to the matched Float64 result rounded to Float32 |

A tolerance is never loosened to cover a wrong grid, product, basis or missing
modes, and an absent or skipped fixture is never reported as passed. pydyn has
no spectral stage, so it can check stages 8 and 11 (balance and `dynam0`) but
not stages 3–4. Synthetic pydyn fixtures for stages 8 and 11 are local-only,
kept outside the repository and not committed. The parity loader and tests load fixtures from
`ATMOSTR_TM5_FIXTURES_DIR` and skip with a clear message when the variable
is unset or the required fixtures are absent.

## 15. Owner decisions (2026-10-30)

- **Q1** Apply our dry pin after flux integration, without recomputing the
  fluxes (TM5 order).
- **Q2** Write the mean of the two window-end flux sets, each balanced against
  the same window tendency (`window_mean`).
- **Q3** Implement the correct mathematics at the truncation edge and document
  every TM5 difference and its size; no quirk is reproduced.
- **Q4** `vod2uv` uses the IFS radius ``6.371229 × 10^6`` m.
- **Q9** Running pydyn's pure functions locally on synthetic inputs is
  approved. A request for a live TM5 dump waits and will be combined with the
  report of the ``n = T`` Nabla sign error (`gts:1804`).
- **Q11** Reduced Gaussian ships with lat-lon from the first flux pull request.

On 2026-10-31 the owner bundled the pull requests: TM5-0 (local-only fixtures, this page
and the page [TM5 boundary-layer diffusion](../theory/tm5_boundary_layer.md),
binary format gate), TM5-FLUX-1 (S3–S6:
plumbing, spectral evaluator and quadrature, moist target construction,
per-layer balance, end-to-end LL and RG) and TM5-FLUX-2 (S7 first real day
with attributable deltas, S8 cubed-sphere direct target).

## 16. Implementation status

| Item | Status |
|---|---|
| Binary version 5: `flux_construction` identity keys, readers accepting versions 4 and 5, inspector display | Implemented (TM5-0) |
| pydyn balance and `dynam0` fixtures | Generated locally outside the repository; not committed; loader and parity tests implemented using `ATMOSTR_TM5_FIXTURES_DIR`, with a clear skip when unset or fixtures are absent |
| Construction types, configuration, dispatch hooks, `InterfaceMassFluxField` (S3) | Implemented (TM5-FLUX-1) |
| Spectral evaluator and quadrature for LL and RG rings (S4) | Implemented (TM5-FLUX-1) |
| Integrated PS, face fluxes and ``W`` (S5) | Implemented (TM5-FLUX-1) |
| Per-layer balance, TM5 guards, RG factor-once solve (S6a) | Implemented (TM5-FLUX-1) |
| End to end for LL and RG, new cases `pre_ll72_tm5`, `pre_o24_tm5` (S6b) | Implemented; golden definitions only, references generated separately |
| First real day and attributable deltas (S7) | [Real-day comparison memo](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/main/docs/memos/2026-11-01_tm5_flux_real_day.md) and `scripts/diagnostics/compare_flux_constructions.jl` (TM5-FLUX-2a) |
| Cubed-sphere direct target (S8) | Implemented (TM5-FLUX-2b): refined rings, portable face/overlap gathers, independent W, direct per-layer graph solve, dry endpoint mean and shared CS write gates. Synthetic, exhaustive retained-harmonic and real ERA5 endpoint checks plus CPU cost measurements; `pre_c24_tm5`, `pre_c24_tm5_tm5pbl`, `run_c24_tm5_cpu_f64` definitions added. Reference recording and GPU measurements remain separate. |

## 17. Synthesis backends and measurements

`CPURingSynthesis` uses batched host GEMM and unnormalised FFTW;
`GPUSpectralSynthesis` uses the same portable KernelAbstractions pack/scatter/
gather stages, device GEMM and device FFT. Refined samples stay on the device
through quadrature, whose accumulation is Float64. Batches are limited to
eight field levels in the day driver. Surface LNSP and both gradients always
use Float64. Core does not load CUDA. With CUDA loaded, automatic precision
uses Float64 on A100/H100/B200-class devices and Float32 for 3-D fields on
L40S-class devices. `FAST_MATH` is rejected and the CUDA extension selects
pedantic math through `CUDA.math_mode!`, restoring the caller's mode after each transform. GPU execution is opt-in in tests:

```bash
ATMOSTR_TEST_SPECTRAL_GPU=1 julia --project=test test/core/test_spectral_synthesis_backends.jl
```

The preprocessing CLI loads CUDA when this construction requests GPU
synthesis. Library callers load their backend explicitly before planning.
No silent CPU fallback occurs. The GPU runtime and performance
checks require a GPU session; the CPU-array version exercises the portable
kernels in ordinary CI. The spike's mirrored-table optimization, pinned
staging and copy-free cuFFT execution are performance follow-ups; this
implementation uses bounded batches and standard array/FFT APIs.

Measured synthetic Gauss residuals ``||C + H||₂ / ||C||₂`` with refinement
steps 2, 4, 8 are 0.0322, 0.00778, 0.00252 on LL and 0.0138, 0.00453,
0.00135 on RG (degree 4, two layers, variable pressure and a truncation-edge
divergence mode). Constant-pressure solid-body rotation gives exactly zero
``C`` and meridional flux. Degree-7 quirk sizing gives 5.76461 for the
sign-only gradient difference with a unit ``H_6^2`` coefficient and 1.77259
for ``P_7^7(0)``; a repeated-endpoint cosine mean at 16 longitudes is
0.0588235. These are analytic signal sizes, not climatological errors.

The 2026-10-31 real-wind probe found that point versus face integration
changes raw net-divergence RMS by 9–26% in four native layers of one ERA5
window, before balance, pinning and drying. It also measured near-surface
polar-cap wind edge differences of about 0.14 m/s in u and 0.176 m/s in v.
These establish the importance of face integration and the edge terms;
they are not measurements of final balanced cm or tracer differences.

## References

- Bregman, B., Segers, A., Krol, M., Meijer, E., and van Velthoven, P. (2003).
  On the use of mass-conserving wind fields in chemistry-transport models.
  *Atmos. Chem. Phys.* 3, 447–457.
- Krol, M., et al. (2005). The two-way nested global chemistry-transport zoom
  model TM5. *Atmos. Chem. Phys.* 5, 417–432.
- ECMWF IFS documentation, Part III: Dynamics and numerical procedures (the
  vorticity-divergence to wind relations used by `vod2uv!`).
