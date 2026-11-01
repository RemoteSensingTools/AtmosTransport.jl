# TM5-style spectral mass fluxes

This page specifies `TM5SpectralFluxes`, a second construction of the ERA5
spectral mass fluxes for lat-lon (LL) and reduced-Gaussian (RG) targets. It
follows the TM5 meteorological preprocessor (`tmm`): the face fluxes are
quadratures of the retained spectral winds and surface pressure along the
target faces, an independent vertical flux is integrated from the spectral
divergence, and the horizontal fluxes are balanced layer by layer against the
hybrid mass tendency. The existing construction (face- and centre-sampled
winds times a face pressure thickness, see [ERA5 spectral path](spectral_era5.md))
stays the default; the design calls it `PointSampledSpectralFluxes`.

!!! warning "Planned, not implemented"
    Of everything on this page, only the binary version-5 identity keys exist
    (bundle TM5-0, see [Binary format](../concepts/binary_format.md)). Every
    stage, type and function marked *planned* below is the design of the
    TM5-FLUX-1 and TM5-FLUX-2 pull requests. The last section lists the status.

The page follows the design of 2026-10-30, with the owner answers of that
day. Statements marked *estimate* are design estimates, not measurements.

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

**Cost (estimate).** At TL639 on a 1° target there are 8 refined rows per cell
row. Legendre work per row grows with ``T^2`` (``639^2/179^2 ≈ 13``) and the row
count by about 4.5, about 50 times today's 1° Legendre work, for 137 levels of
three fields plus LNSP and its gradient per instant. The plan reuses the
batched multi-field synthesis of `ReducedSpectralSynthesis` and the threaded
synthesis; S5 records a benchmark.

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
Both TM5 and `vod2uv!` drop this row. Whether S4 extends the wind output to
``T + 1`` or keeps the existing `vod2uv!` is decided in S4; the effect is sized
by the planned single-mode tests.

## 3. Cell surface pressure: the discrete rule

TM5 evaluates ``e^{H}`` at the refined sample points (`gi:1893`), integrates
each refined row with piecewise-linear interval quadrature in ``λ``
(`IntervalQuad_Lin`, `gi:1907`), integrates the row results with
cosine-weighted linear quadrature in ``φ`` (`IntervalQuad_Cos_Lin`, `gi:1926`)
and divides by the cell area (key `'exp,aver'`, `tmm:1876`). The cell value
``\bar p_{s,c}`` is therefore the exact integral of a piecewise-linear
interpolant of ``e^{H}``; it is neither ``\iint e^{H}\,dA / S_c`` nor
``\exp`` of the mean of ``H``. The planned construction implements this rule
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

The plan uses open (pole-free) quadrature nodes in the polar rows instead; the
difference is listed in §13.

## 5. Independent vertical flux

Per native layer, TM5 integrates the horizontal mass divergence directly from
the spectral fields. With the two radii of the conventions, the planned form is

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
ignores the flag (`pydyn/forward.py:134`). The plan follows the Fortran and
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

## 8. Pipeline on LL and RG targets (planned)

E marks an existing function, N a planned new one.

| # | Stage | Functions | Home |
|---|---|---|---|
| 1 | Read the spectral day and the next day's hour 00 | E `read_day_spectral_streaming`, `validate_spectral_header`. No truncation. The next-day 00 record is required: no zero-tendency fallback. | `Spectral/spectral_io.jl` |
| 2 | Humidity on native levels | E `read_window_qv!` (LL), `load_rg_qv!` (RG), unchanged | `Targets/` |
| 3 | Ring evaluation | E `compute_legendre_column!`, `fill_ring_spectrum!` (oversized FFT), `vod2uv!`; N exact LNSP gradient | `Spectral/` |
| 4 | Cell PS, face fluxes, ``C_k`` and ``W`` (§3–§5) | N `integrate_surface_pressure!`, `integrate_face_fluxes!`, `integrate_vertical_flux!`; LL boxes, RG per-ring boxes and LCM face segments (§10) | `Spectral/`, `MassClosure/` |
| 5 | Dry pin on the integrated PS, fluxes not recomputed (Q1) | E `pin_global_mean_ps_using_qv!` | `Spectral/mass_support.jl` |
| 6 | Moist masses from the pinned PS | E `compute_dp!`, `compute_air_mass!` (LL), `compute_reduced_dp_and_mass!` (RG) | `Spectral/`, `Targets/ReducedGaussian/` |
| 7 | Target levels | E `plan_vertical`, `apply_vertical!` with `MassField` and `MassFluxField` sums; N `InterfaceMassFluxField` (selects retained interfaces with the body of `ConvectionInterfaceFlux`); E `merge_qv!` weighted by the pinned moist masses. The NOAA levels are `ml137_tropo34a`, not `ml137_tropo34`. | `Vertical/`, `Spectral/` |
| 8 | Per window: balance both endpoint sets against the window's ``\dot M`` (§6) | LL: E `balance_mass_fluxes!` one level at a time with target ``H^\ast``; RG: per-level ring solve from a neutral split of `balance_compressed_column_fluxes!`; N TM5 guards and passes | `Spectral/`, `MassClosure/` |
| 9 | Dry carrier on target layers, for each balanced endpoint set with its own instant's ``q`` | E `apply_dry_basis_native!`, `apply_dry_basis_reduced!` | `Spectral/`, `Targets/ReducedGaussian/` |
| 10 | Window flux: mean of the two dry endpoint sets (Q2) | N, a few lines | `TransportBinary/` |
| 11 | Substep schedule, amounts, column balance, `cm`, replay gates | E schedule; amounts via `MetDrivers.flux_application_seconds`; `apply_poisson_balance!` → `balance_column_mass_fluxes!` (LL), `balance_window!` (RG); `recompute_cm_from_dm_target!` with the dry endpoint ``Δm`` | `Targets/`, `MassClosure/`, `MetDrivers/` |
| 12 | Write | E writers; version-5 header with the construction identity | `TransportBinary/`, `MetDrivers/BinaryFormat/` |

Folders are under `src/Preprocessing/` unless they start with `MetDrivers/`.
The order differs on purpose from the existing route:

- **Existing route:** pin → recompute fluxes → dry conversion on native levels
  → merge → column balance → `cm`.
- **Planned route:** moist integration → pin (no recompute) → moist merge →
  moist per-layer balance with ``W`` → dry conversion of each endpoint set on
  target layers → window mean → shared dry column balance and closure.

**Substep retries.** The balanced moist rates [kg s⁻¹] do not depend on the
substep schedule, so a positivity-driven retry recomputes only stage 11 from
the stored rates; no field already balanced in amount units is rescaled. This
is the rule of the cubed-sphere spectral writer
(`src/Preprocessing/Targets/CubedSphere/cubed_sphere_spectral.jl`).

**Memory (estimate).** Raw rates for the 25 instants of a 1° L34 day take about
1.4 GB in Float64; if that is too much, two instants per window are streamed.

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
- **Global water term.** The moist per-layer balance sees a global moist
  tendency equal to the water change; the zero-mode removal leaves it in
  place. *Estimate:* ``10^{-11}`` to ``10^{-9}`` s⁻¹, far below the
  ``10^{-7}`` post-check. TM5's total-pressure pin avoids the term by
  construction. This is the one physics consequence of keeping our pin.
- **Local E − P term.** The dry conversion leaves a local column residual of
  about ``E − P`` relative to the column mass (*estimate:* about ``10^{-9}``
  s⁻¹ for 3 mm per day, up to ``3 × 10^{-7}`` s⁻¹ for 10 mm per hour of heavy
  rain). The shared dry column balance removes it, as in the existing route.
- **``q = 0`` limit.** After the column balance, the dry route's `cm` equals the
  hybrid-B closure in exact arithmetic (a planned S6b test).

## 10. Reduced Gaussian specifics

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
  windows. The existing column PCG path stays unchanged; S6a carries a
  benchmark gate.

The direct RG spectral path already shares the LL primitives
(`ReducedSpectralSynthesis`; `spectral_to_ring!` and
`compute_reduced_horizontal_fluxes!` in
`src/Preprocessing/Targets/ReducedGaussian/reduced_transport_helpers.jl`).

## 11. Types, dispatch and configuration (planned)

```julia
abstract type AbstractSpectralFluxConstruction end
struct PointSampledSpectralFluxes <: AbstractSpectralFluxConstruction end  # default: today's bodies
struct TM5SpectralFluxes{R} <: AbstractSpectralFluxConstruction
    refinement_steps :: Int   # TM5 ShRefinement nstep = 2
    vod2uv_radius    :: R     # IFS_EARTH_RADIUS (Q4)
end
```

| Dispatch point | Default method | `TM5SpectralFluxes` method |
|---|---|---|
| `spectral_resolution(c, grid, T_source)` | `target_spectral_truncation(grid)` | `T_source` |
| LL `process_window!(c, …)`, RG `synthesize_and_merge_window!(c, …)` | current body, untouched | stages 3–7 |
| `balance_window_layers!(c, storage, …)` before the shared flush | no-op | stages 8–10 |

- The only other new nominal type is `InterfaceMassFluxField <: AbstractFieldKind`
  (a kg s⁻¹ interface flux is not a convective flux, even where the selection
  body is the same).
- In S8 the cubed-sphere method `SpectralIntegralFaceFluxes` joins the
  `face_fluxes` family (`face_flux_method`, with a new `AbstractFaceFluxMethod`
  over `PanelAverageFluxes`, `VectorFaceFluxes` and `LineIntegralFaceFluxes`).
- Configuration: one key, parsed once into the type; an absent key means the
  default with exactly today's settings. The design proposes
  `[preprocessing] flux_construction = "tm5_spectral"`, but "flux construction"
  already names the face-flux options read from `[preprocessing]` and
  `[numerics]` (`parse_flux_construction`, `FLUX_CONSTRUCTION_KEYS`,
  `src/Preprocessing/MassClosure/flux_construction.jl`). S3 must choose a key
  that does not collide, or rename; this is an open design point. Planning
  rejects cubed-sphere targets until S8, `mass_basis = "moist"`, and any
  `flux_time_sampling` other than `"window_mean"`.
- The default path stays bitwise unchanged; every flux pull request gates the
  `pre_ll72`, `pre_o24` and `pre_c24` goldens.

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
taking the LL and RG tags from the settings is planned for S3.

## 13. Differences from TM5

| TM5 | Here | Reason or size |
|---|---|---|
| Additive match of the total-pressure area mean to 98500 Pa (§9) | Our dry pin, applied after flux integration, fluxes not recomputed | Owner decision Q1; global water term, *estimate* ``10^{-11}``–``10^{-9}`` s⁻¹ |
| Forecast records (`h06h18tr3`), valid-time and lead bookkeeping, balanced fluxes interpolated to each dynamic-step midpoint | Hourly analyses; the window flux is the mean of the two endpoint sets, each balanced against the same tendency | Owner decision Q2; the mean is the exact window integral of TM5's linear interpolation |
| NOAA 1° archive plus offline coarsener | Direct integration on the target grid | Owner decision; *estimate:* a direct coarse integral differs from the sum of 1° fluxes |
| Mass-weighted coarsening of 1° T and Q | ``q`` from the existing reader (point values on target cells), merged with the pinned moist masses | ``q`` is a grid-point field, not spectral |
| ``n = T`` Nabla sign error and gradient truncated at ``T`` (§5) | Exact gradient to degree ``T + 1`` | Q3; sized by the planned single-mode tests |
| ``P_T^T = 0`` (§1) | ``P_T^T`` evaluated | Q3; affects the ``(T, T)`` coefficient only; sized by the planned tests |
| `vod2uv` edge rows ``n = T−1``, ``m = T−1`` and ``n = T`` (§2) | All neighbour terms inside ``[m, T]`` | Q3; sized by the planned single-mode tests |
| Polar rows: ``1/(N_λ + 1)`` mean and adjacent-row mean for ``D\cos φ`` (§4) | Open polar nodes | Q3; sized by the planned polar tests |
| ``π`` literal `3.14159265358979` (`binas:35`) | Julia's ``π`` | relative ``1.0 × 10^{-15}`` |
| `vod2uv` radius `ae`, 6.371 × 10⁶ m | `IFS_EARTH_RADIUS`, 6.371229 × 10⁶ m | Q4; relative ``3.6 × 10^{-5}`` |
| One radius: ``R^2/g`` in the ``D`` term of ``C_k`` | ``R R_v/g`` (§5) | follows from Q4; keeps ``W`` consistent with the face fluxes |
| Wind row ``n = T + 1`` dropped (§2) | also dropped by `vod2uv!` | shared truncation; whether S4 keeps it is decided in S4, sized by the planned single-mode tests |
| Balance never skips a level | `balance_mass_fluxes!` skips levels with residual ``< 10^{-10}`` | harmless |
| `tsp` and the `'bottom'` `FillLevels` key | Not built | not consumed |
| Check_CFL, polar clustering, prognostic slopes, diffusion inside convection, window-end convection | Runtime unchanged | These bound any "TM5-style" claim: this is a flux construction, not TM5 runtime parity |

## 14. Tolerances (planned targets)

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
kept outside the repository and not committed. The parity loader and tests
are planned for S6a (TM5-FLUX-1): they will load fixtures from
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
| pydyn balance and `dynam0` fixtures | Generated locally outside the repository; not committed; loader and parity tests planned for S6a using `ATMOSTR_TM5_FIXTURES_DIR`, with a clear skip when unset or fixtures are absent |
| Construction types, configuration, dispatch hooks, `InterfaceMassFluxField` (S3) | Planned (TM5-FLUX-1) |
| Spectral evaluator and quadrature for LL and RG rings (S4) | Planned (TM5-FLUX-1) |
| Integrated PS, face fluxes and ``W`` (S5) | Planned (TM5-FLUX-1) |
| Per-layer balance, TM5 guards, RG factor-once solve (S6a) | Planned (TM5-FLUX-1) |
| End to end for LL and RG, new goldens `pre_ll72_tm5`, `pre_o24_tm5` (S6b) | Planned (TM5-FLUX-1) |
| First real day and attributable deltas (S7) | Planned (TM5-FLUX-2) |
| Cubed-sphere direct target (S8) | Planned (TM5-FLUX-2) |

## References

- Bregman, B., Segers, A., Krol, M., Meijer, E., and van Velthoven, P. (2003).
  On the use of mass-conserving wind fields in chemistry-transport models.
  *Atmos. Chem. Phys.* 3, 447–457.
- Krol, M., et al. (2005). The two-way nested global chemistry-transport zoom
  model TM5. *Atmos. Chem. Phys.* 5, 417–432.
- ECMWF IFS documentation, Part III: Dynamics and numerical procedures (the
  vorticity-divergence to wind relations used by `vod2uv!`).
