# TM5 boundary-layer diffusion (`Calc_Kzz`)

This page specifies `TM5CalcKzzField`, an implemented interface-exchange (`dkg`)
field for the existing `ImplicitVerticalDiffusion` operator. It transcribes
TM5's live `Calc_Kzz`: the friction velocity, the hydrostatic interface
heights, the Holtslag–Boville boundary-layer scheme `bldiff`, and the
conversion of the heat diffusivity into an exchange rate in kg s⁻¹ across each
level interface. It is a new field type; the existing diffusion schemes and the
preprocessing port `tm5_bldiff.jl` stay unchanged.

Lat-lon and reduced-Gaussian column physics, v5 inputs and runtime integration
are implemented in TM5-PBL-1. Cubed-sphere and adjoint support remain planned
for TM5-PBL-2. Local pydyn fixtures are read only through
`ATMOSTR_TM5_FIXTURES_DIR`; they are never committed (§14).

The page follows the design of 2026-10-30, with the owner answers of that
day. Statements marked *estimate* are design estimates, not measurements.

## Sources

TM5 citations are `file:line` in the NOAA CarbonTracker TM5 repository
(`ct-unified-github` at commit `2d60209`); pydyn citations are at pydyn commit
`0db6e6d`.

| Abbreviation | Repository path |
|---|---|
| `K` | `proj/ct-unified/trunk/src/diffusion_kernels.F90` (`ustar_calc`, `bldiff_core`) |
| `SP` | `proj/ct-unified/branches/speedup/src/diffusion__speedup.F90` (`Calc_Kzz`) |
| `PC` | `proj/ct-unified/trunk/src/meteo__pclim.F90` (the active overlay of `base/trunk/src/meteo.F90`) |
| `MI` | `proj/ct-unified/trunk/src/modelIntegration.F90` |
| `conv` | `base/trunk/src/convection.F90` |
| `binas` | `base/trunk/src/binas.F90` |
| `lev` | `proj/levels/ml137/tropo34/src/dims_levels__ml137_tropo34.F90` |
| `rc` | `base/trunk/rc/meteo-tm5-noaa.rc` |
| pydyn | `pydyn/diffusion.py`, `pydyn/simple_diffusion.py` |

## Conventions

The equations of §2–§7 use TM5's bottom-up indexing: layers ``l = 1 … L``
from the surface, interface ``l`` is the bottom of layer ``l`` (interface 1 is
the surface, interface ``L+1`` the top). The runtime is top-down (``k = 1`` at
the top). One adapter reverses the layer centres and all ``L + 1`` interfaces;
the column physics has no orientation conditionals.

The mask function of TM5 is

```math
J(a, b) = \mathrm{nint}\!\left(0.5 + \mathrm{sign}(0.5,\, a - b)\right)
```

(`jqif`, `K:1026–1037`): ``J = 1`` where ``a − b`` is ``+0`` or positive, and
``J = 0`` where it is negative or ``−0``. In Julia this is
``1 − \mathrm{signbit}(a − b)``. pydyn implements it the same way
(`pydyn/diffusion.py:71–73`).

## 1. Inputs and cadence in TM5

`Calc_Kzz` is called from `Proces_Update` (`MI:544`), which runs once per
dynamic step (`base/trunk/src/tm5.F90:1303`, after `Meteo_Setup_Other` at
`:1297`). Its inputs:

- Interface pressures `phlb` and layer masses ``M`` come from `spm`, the
  surface pressure linearly interpolated to the middle of the step
  (`SP:954–966`; `spm` at `PC:2209–2218`).
- The heights ``G`` use the advected prognostic surface pressure `sp`
  (`PC:1570`, `PC:5994`, `PC:6006–6020`).
- The winds come from the balanced, time-interpolated mass fluxes `pu`, `pv`.
- For ERA5 (`rc:287–298`): model-level fields such as ``T`` and ``q`` are
  3-hourly and interpolated (`interp3`), instantaneous surface fields hourly
  (`interp1`), accumulated surface fields are hourly interval means (`aver1`);
  orography and land-sea mask are constant and the Olsson/Ganzeveld roughness
  `srols` is monthly (`rc:319–324`).

**Cache mode.** NOAA production has read a precomputed `dkg` cache since
2026-05-07 and runs live `Calc_Kzz` only on a cache miss
(`base/trunk/src/diffusion.F90:72–76`; `rc:424–434`). pydyn reports that this
cache differs from live `Calc_Kzz` by about ``7 × 10^{-6}`` rms relative per
hour (`docs/state.md:382–386` in pydyn), and pydyn's physics dumps were made
with the cache. The parity target here is live `Calc_Kzz` (owner decision
Q10).

## 2. Friction velocity

`ustar_calc` (`K:175–283`, called by `dd_calc_ustar`, `K:137–170`), NOAA
default branch. The TM5-4DVAR land variant (`K:260–264`) is off by default
(`K:100`), and the speedup overlay aborts if it is requested (`SP:255–259`).

With the land fraction ``x`` (TM5 stores percent, `K:216`), the surface stress
``τ = \sqrt{\mathrm{ewss}^2 + \mathrm{nsss}^2}`` and ``V_{10} = \sqrt{u_{10}^2 + v_{10}^2}``:

```math
\begin{aligned}
u_{*,\mathrm{sea}} &= \max\!\left(0.001,\ \sqrt{τ/ρ}\right), \qquad ρ = 1.2\ \mathrm{kg\,m^{-3}},\\
u_{*,\mathrm{land}} &= \frac{κ V_{10}}{\ln(10/z_h)}\,\frac{\ln(75/z_h)}{\ln(10/z_l)},
\qquad z_h = \min(z_{0,\mathrm{ECMWF}},\ 0.03),\quad z_l = \max(z_{0,\mathrm{Olsson}},\ 0.01),\\
u_* &= x\,u_{*,\mathrm{land}} + (1 - x)\,u_{*,\mathrm{sea}}
\end{aligned}
```

(`K:189`, `K:228–233`, `K:265–270`, `K:277`). The 10-m wind is carried up the
log profile of the clipped ECMWF roughness to 75 m, where the wind is assumed
independent of the surface (`K:254–256`), and back down with the Olsson
roughness; the NOAA branch uses ``\ln(10/z_l)`` in the last factor, the 4DVAR
branch ``\ln(75/z_l)``. For ``x = 0`` the land terms are 0 and no land
logarithm is evaluated. `bldiff` uses ``u_s = \max(u_*, 0.01)`` with
``κ = 0.4`` (`K:630`, `binas:152`). pydyn: `ustar` (`pydyn/diffusion.py:76–89`).

**Roughness (owner decision Q8).** TM5's ``z_{0,\mathrm{ECMWF}}`` is the
ECMWF surface roughness `sr`, listed among the analysis surface fields
(`rc:119`), and ``z_{0,\mathrm{Olsson}}`` is the monthly Olsson/Ganzeveld
database `srols` (`rc:120`, `rc:323–324`). Q8 replaces only `srols`: its role
(``z_l``, floored at 0.01 m) is taken by the ERA5 forecast surface roughness
`fsr`. `sr` keeps its role (``z_h``, clipped at 0.03 m); the input writer accepts `sr` when present and otherwise uses `fsr` for both roles. If ERA5 provides the roughness only as
`fsr`, both roles use the same field ``z_0``, and the land formula has three
regimes:

| ``z_0`` [m] | ``z_h`` | ``z_l`` | ``u_{*,\mathrm{land}}`` |
|---|---|---|---|
| ``z_0 ≤ 0.01`` | ``z_0`` | ``0.01`` | ``κ V_{10} \ln(75/z_0) / [\ln(10/z_0)\,\ln 1000]`` |
| ``0.01 < z_0 ≤ 0.03`` | ``z_0`` | ``z_0`` | ``κ V_{10} \ln(75/z_0) / \ln^2(10/z_0)`` |
| ``z_0 > 0.03`` | ``0.03`` | ``z_0`` | ``κ V_{10} \ln 2500 / [\ln(1000/3)\,\ln(10/z_0)]`` |

In every case the formula needs ``z_l < 10`` m, so that ``\ln(10/z_l) > 0``;
TM5 has no guard for it. The size of the change against `srols` is unknown
until both datasets are compared on the same grid.

## 3. Interface heights

`Meteo_Produce_Gph` (`PC:6006–6020`), with ``p_l = a_l + b_l p_s`` at the
interfaces:

```math
G_1 = \frac{\mathrm{oro}}{g}, \qquad
G_{l+1} = G_l + \frac{287.05\, T_l (1 + 0.608\, q_l)}{g}\ln\frac{p_l}{p_{l+1}}
\quad (l = 1 … L-1), \qquad G_{L+1} = 200\ \mathrm{km}.
```

The top value is assigned, so ``\ln(p/0)`` is never taken. Layer centres are
interpolated linearly in pressure (`K:595`, `K:601–603`):

```math
\bar p_l = \tfrac12 (p_l + p_{l+1}),\quad
f_l = \frac{p_l - \bar p_l}{p_l - p_{l+1}},\quad
z_l = (G_l - G_1)(1 - f_l) + (G_{l+1} - G_1) f_l .
```

The top centre height is interpolated toward 200 km, and the boundary-layer
height falls back to it when no crossing is found (§6). pydyn:
`geopotential` (`pydyn/diffusion.py:92–104`).

## 4. Thermodynamics and surface buoyancy

Constants (`K:486–509`, `K:873`, `binas`):

| Symbol | Value | Source |
|---|---|---|
| ``g`` | 9.80665 m s⁻² | `binas:55` |
| ``κ`` | 0.4 | `binas:152` |
| ``c_p`` | 1004 J kg⁻¹ K⁻¹ | `binas:137` |
| ``L_v`` | ``2.5 × 10^6`` J kg⁻¹ | `binas:140` |
| ``p_0`` | ``10^5`` Pa | `K:493` |
| ``R_a`` | ``8.3144 × 10^3/28.94 ≈ 287.2979`` J kg⁻¹ K⁻¹ | `K:490`, `binas:98`, `binas:116` |
| ``ε``, ``χ`` | ``R_a/461.51 ≈ 0.622517``, ``1/ε − 1 ≈ 0.606382`` | `K:492`, `K:503` |
| ``κ_p`` | ``R_a/c_p ≈ 0.286153`` | `K:502` |
| ``\mathrm{Ri}_c``, `zacb`, `tiny` | 0.3, 100, ``10^{-9}`` | `K:505–507` |
| `sffrac`, `ssfrac`, ``β_h = β_m`` | 0.1, 0.1, 15 | `K:488`, `K:495`, `K:509` |
| `fak`, `fakn` | 8.5, 7.2 | `K:499–500` |
| ``K_\mathrm{free,min}``, ``K_\mathrm{BL,min}``, ``h_\mathrm{min}`` | ``10^{-15}``, 0.1, 100 m | `K:486`, `K:873`, `K:855` |

Three dry-air gas constants appear and are kept separate. The heights use the
literal 287.05 (`PC:6014`) with the virtual-temperature factor 0.608. `bldiff`
uses ``R_a = R/M_a`` with TM5's molar mass 28.94 g mol⁻¹ (`binas:98`), which
gives 287.2979. The existing preprocessing port uses
`TM5_CONSTANTS.r_air = 287.307` (`src/Parameters/PhysicalConstants.jl`),
``3.2 × 10^{-5}`` above 287.2979; it stays untouched, and the live
`Calc_Kzz` constants are separate named values.

With ``\bar p_l`` from §3 (`K:595–597`, `K:624–630`, `K:752–754`):

```math
\begin{aligned}
θ_l &= T_l \left(p_0/\bar p_l\right)^{κ_p}, \qquad θ_{v,l} = θ_l (1 + χ q_l),\\
w'θ' &= -\mathrm{sshf}\,\frac{(G_2 - G_1)\, g}{(p_1 - p_2)\, c_p}, \qquad
w'q' = -\mathrm{slhf}\,\frac{(G_2 - G_1)\, g}{(p_1 - p_2)\, L_v},\\
θ_s &= θ_{v,1}, \qquad H_v = w'θ' + χ\, θ_1\, w'q',\\
L_\mathrm{ob} &= -\frac{θ_s\, u_s^3}{g κ \left[H_v + \mathrm{sign}(10^{-10}, H_v)\right]} .
\end{aligned}
```

`sshf` and `slhf` are ECMWF downward-positive fluxes [W m⁻²]; the factor
``(G_2 − G_1) g/(p_1 − p_2)`` is the inverse density of the lowest layer.

## 5. Winds and free-atmosphere diffusivity

TM5 recovers the winds from the face mass fluxes (`K:644–668`):
``u_l = d_x (pu_e + pu_w)/(2 M_l)`` and ``v_l = d_y (pv_n + pv_s)/(2 M_l)``
with ``d_x = a_e Δλ \cos φ_j`` and ``d_y = a_e Δφ`` (``a_e`` = `ae`). pydyn:
`_winds` (`pydyn/diffusion.py:113–120`).

At the interface above layer ``l``, with ``z = G_{l+1} − G_1`` (`K:684–727`):

```math
\begin{aligned}
λ(z) &= 300\ \text{m}\ (z ≤ 1000\ \text{m}), \qquad 30 + 270\, e^{1 - z/1000}\ \text{otherwise},\\
ℓ^2 &= \left[1/λ + 1/(κ z)\right]^{-2},\\
S^2 &= \frac{\max\!\left(Δu^2 + Δv^2,\ 10^{-10}\right)}{(z_{l+1} - z_l)^2},\\
α &= \frac{\ln p_{l+1} - \ln \bar p_l}{\ln \bar p_{l+1} - \ln \bar p_l}, \qquad
θ_{v,\mathrm{ifc}} = θ_{v,l} + α\,(θ_{v,l+1} - θ_{v,l}),\\
\mathrm{Ri} &= \frac{g\,(θ_{v,l+1} - θ_{v,l})}{θ_{v,\mathrm{ifc}}\,(z_{l+1} - z_l)\, S^2}, \qquad
K_n = ℓ^2 \sqrt{S^2},\\
K_\mathrm{free} &= \begin{cases}
  \max\!\left(10^{-15},\ K_n / [1 + 10\,\mathrm{Ri}(1 + 8\,\mathrm{Ri})]\right) & \mathrm{Ri} ≥ 0,\\
  \max\!\left(10^{-15},\ K_n \sqrt{\max(1 - 18\,\mathrm{Ri},\ 0)}\right) & \mathrm{Ri} < 0.
\end{cases}
\end{aligned}
```

The shear floor applies before the division by ``Δz^2``, and ``\mathrm{Ri} = 0``
takes the stable branch (``J(\mathrm{Ri}, 0) = 1``).

## 6. Boundary-layer height: two passes

Vogelezang and Holtslag (1996), as cited at `K:736`; `K:749–855`. For each
layer centre,

```math
V_l^2 = \max\!\left((u_l - u_1)^2 + (v_l - v_1)^2 + 100\, u_s^2,\ 10^{-9}\right)
```

(`K:762–764`, `K:827–829`), and the bulk Richardson number is

```math
\mathrm{Rib}_l = \frac{g\,(θ_{v,l} - θ_\mathrm{ref})\,(z_l - z_1)}{θ_s V_l^2}
  + \mathrm{sign}\!\left(10^{-10},\ \cdot\right).
```

Each pass scans upward to the first crossing of ``\mathrm{Ri}_c = 0.3`` and
interpolates linearly between centre heights; with no crossing it takes the
highest centre (`K:800`, `K:854`). Work flags freeze the result once found,
and an offset of 0.1 after the crossing keeps later denominators nonzero
(`K:791–796`, `K:845–850`).

- **Pass 1** starts at ``l = 1`` with ``θ_\mathrm{ref} = θ_s`` (`K:760–800`).
  At ``l = 1``, ``z_l − z_1 = 0`` gives ``\mathrm{Rib}_1 = +10^{-10}``, which
  never crosses, and the interpolation there would start from the surface
  (`K:771–772`, `K:780–782`).
- **Pass 2** starts at ``l = 2`` (`K:807–854`). With ``j = J(H_v, 0)`` and the
  pass-1 height ``h_1``:
  ``w_{sc} = u_s [j(1 − 1.5 h_1/L_\mathrm{ob}) + (1 − j)]^{1/3}``,
  ``Δθ = 8.5 H_v / w_{sc}`` and ``θ_\mathrm{ref} = θ_s + j\,Δθ``; the
  denominator keeps ``θ_s``.
- The result is ``h = \max(h_2, 100\ \mathrm{m})`` (`K:855`).

The scan uses the full ``L``-layer profile, not the profile cut at the mixing
cutoff, and ERA5's boundary-layer height is not used.

## 7. Boundary-layer diffusivity, Prandtl number and entrainment

`K:887–966`, at the interface above layer ``l`` with ``z = G_{l+1} − G_1``.

**Masks.** ``I = J(h, z)`` (inside the boundary layer), ``j_U = J(H_v, 0)``,
``j_S = I(1 − j_U)`` (stable), ``j_C = I j_U`` (convective), ``η = I z/h``,
surface layer ``j_4 = j_C J(0.1, η)``, mixed layer above it
``j_5 = j_C (1 − J(0.1, η))`` (`K:891–910`).

**Stable and neutral (Beljaars and Viterbo 1998, `K:866`).**

```math
\begin{aligned}
ℓ_\mathrm{BL}^2 &= \left[1/(κ z) + 1/450\right]^{-2}, \qquad j_R = J(\mathrm{Ri}, 0),\\
f &= \frac{j_R}{1 + 10\,\mathrm{Ri}\sqrt{1 + j_R\,\mathrm{Ri}}} + (1 - j_R),\\
K_\mathrm{base} &= j_S \max\!\left(ℓ_\mathrm{BL}^2 \sqrt{S^2}\, f,\ 0.1\right)
  + (1 - j_S)\, K_\mathrm{free}
\end{aligned}
```

(`K:916–924`).

**Unstable (Holtslag and Boville 1993, `K:639`).** With
``a = 1 − 1.5\,h/L_\mathrm{ob}``:

```math
\begin{aligned}
K_m &= u_s κ z (1 - z/h)^2 (1 - 15 z/L_\mathrm{ob})^{1/3} && \text{(surface layer)},\\
K_m &= u_s a^{1/3}\, κ z (1 - z/h)^2 && \text{(mixed layer)},\\
w_* &= \left(\lvert H_v\rvert\, g\, h/θ_s\right)^{1/3},\\
\mathrm{Pr} &= j_C\left[\frac{a^{1/3}}{\sqrt a}
  + \frac{0.1\,κ\, 7.2\, w_*}{u_s a^{1/3}}\right] + (1 - j_C),\\
K_{vh} &= j_C \max\!\left(K_m/\mathrm{Pr},\ 0.1\right) + (1 - j_C)\, K_\mathrm{base}
\end{aligned}
```

(`K:928–961`; in the Prandtl number ``L_\mathrm{ob}`` is replaced by ``−1`` outside
the convective boundary layer, `K:952`). TM5 divides by ``\mathrm{Pr}``; the
existing `_prandtl_inverse` (`src/Operators/Diffusion/pbl_physics.jl`) is not
its reciprocal and is not reused.

**Entrainment.** At the interface whose centre heights straddle ``h``,
``j_E = j_U J(h, z_l) J(z_{l+1}, h)`` (`K:903–905`), and

```math
K_\mathrm{entr} = \frac{0.2\, H_v}{(θ_{v,l+1} - θ_{v,l})/(z_{l+1} - z_l)}, \qquad
K_{vh} ← j_E K_\mathrm{entr} + (1 - j_E) K_{vh}
```

(`K:964–966`). TM5 applies no threshold, ceiling or floor here; §9 describes
the one guard we add.

## 8. Exchange rate and the mixing cutoff

```math
D_l = \max(0, K_{vh,l})\, \frac{2\,(M_{l+1} + M_l)}{(G_{l+2} - G_l)^2}
\quad (l = 1 … l_\mathrm{max} - 1), \qquad D_{l_\mathrm{max}} = 0
```

(`K:980–988`), in kg s⁻¹, built directly from the interface diffusivity
without a centre-``K_z`` detour. pydyn: `dkg` (`pydyn/diffusion.py:256–263`).

**tropo34.** NOAA's 34 levels (`lev:43–47`) are AtmosTransport's
`ml137_tropo34a` (`ECHLEVS_ML137_TROPO34A` in
`src/Parameters/MetSourceConstants.jl`), not `ml137_tropo34`. There
``l_\mathrm{max} = 26`` (`lev:34`; the comment at `lev:32–33` places the top of
layer 26, ECMWF half-level 57, near 86 hPa). In top-down indexing, layers
``k = 9 … 34`` are mixed. A compact array ``D[k]`` (exchange between ``k`` and
``k+1``) has ``D[1:8] = 0`` and ``D[34] = 0``; a full interface array is active
at slots 10 … 34. The runtime stores the compact form; the adapter tests check its full-interface placement.

**Other level sets (owner decision Q7).** The cutoff is the pressure equivalent
of tropo34a level 26: mix only the layers whose reference mid-layer pressure
exceeds the pressure of ECMWF L137 half-level 57. From
`config/era5_L137_coefficients.toml`, ``a_{57} = 8608.525391`` Pa and
``b_{57} = 5.9 × 10^{-5}``, so ``p_{57} = 8614.34`` Pa at the reference surface
pressure 98500 Pa of the existing GEOS-Chem layer-count rule (8614.43 Pa at
``10^5`` Pa; the dependence on the reference is below 0.2 Pa). On tropo34a this
gives exactly 26 layers, because half-level 57 is an interface of that set; on
other sets it keeps the same physical depth. The implementation shares
the rule of `gchp_pbl_layer_count` (`src/Operators/Diffusion/gchp_nonlocal_pbl.jl`),
leaving the GEOS-Chem results bitwise unchanged. The cutoff is stored as the exact
`8614.336891` Pa literal in `Parameters`, rather than reading the coefficient
file at runtime. A coefficient-file test verifies the literal and the tropo34a count.
Free-atmosphere ``K`` and the boundary-layer height still use the full column.

## 9. Guards and the entrainment NaN

| Item | TM5 | Implemented here |
|---|---|---|
| ``u_*`` floors | sea ``u_* ≥ 0.001``; ``z_l ≥ 0.01``; ``z_h ≤ 0.03``; ``u_s ≥ 0.01`` | same |
| Shear and bulk ``V^2`` | floor ``10^{-10}`` before ``Δz^2``; ``\max(… + 100 u_s^2, 10^{-9})`` | same |
| ``L_\mathrm{ob}``, Rib | signed ``10^{-10}`` offsets | same |
| ``h`` and ``K`` floors | ``h ≥ 100`` m; ``K_\mathrm{free} ≥ 10^{-15}``; boundary-layer ``K ≥ 0.1`` where TM5 applies it | same |
| Masked branches | arithmetic masks ``j x + (1 − j) y`` evaluate both branches | selection `ifelse(j, x, y)`; selects the same value wherever both branches are finite |
| Entrainment | unguarded (below) | `K_vh = ifelse(jE & isfinite(K_entr), K_entr, K_BL)` |
| ``D`` | ``\max(0, K_{vh})``, then zero above the cutoff | same |

**The entrainment NaN.** If ``θ_{v,l+1} = θ_{v,l}`` at an interface,
``K_\mathrm{entr} = ±∞`` (NaN if also ``H_v = 0``). Outside the entrainment
interface ``j_E = 0``, and ``0 · ∞`` = NaN propagates into ``K_{vh}`` even
though the entrainment branch is not selected. Fortran's `MAX` with a NaN
argument is processor-dependent, so TM5's ``D`` there is compiler-defined; an
infinite ``D`` makes the solve singular. pydyn reproduces the NaN
(`pydyn/diffusion.py:20–25`, `:250–252`); its readable version
`simple_diffusion.py` selects the used branch instead (`:19–21`).

The guard keeps a finite ``K_\mathrm{entr}``, also a negative one, which
``\max(0, ·)`` then clips as in TM5, and otherwise keeps the boundary-layer
value. It selects the same value as TM5's mask arithmetic wherever TM5 is
finite and differs only where TM5 is non-finite or compiler-defined. The
column function returns the number of guarded interfaces, and the refresh
warns once per run when the guard changes an entrainment value below the mixing
cutoff. Unused singular branches and interfaces above the cutoff do not enter
the warning count; the column also returns the full singular-expression count. The scheme does not fail fast:
that would abort long runs on rare columns, more often in Float32. The
parity path uses no `@fastmath`, no reassociation, and integer powers such
as `u_s^3`, the form
pydyn uses. Julia's `exp`, `log` and power functions are not the C library's,
so parity is tested against the tolerances of §14, not bit for bit.

**Two other unused-branch paths.** TM5's masks also evaluate two expressions
that are non-finite only in the branch that is not used:

- *Equal consecutive bulk Richardson numbers.* Before the crossing
  (``J(\mathrm{Ri}_c, \mathrm{Rib}_l) = 1``), the interpolation term
  ``(\mathrm{Ri}_c − \mathrm{Rib}_{l−1})/(\mathrm{Rib}_l − \mathrm{Rib}_{l−1})``
  is evaluated anyway; equal values give a division by zero, and the mask
  ``0 · ∞`` makes ``h`` NaN (`K:784–785`, `K:839–840`; pydyn
  `simple_diffusion.py:19–21` names this case). At an actual crossing the two
  values cannot be equal, because the previous one is at most 0.3 and the new
  one exceeds it.
- *``\mathrm{Ri} = −0.1`` exactly.* With ``j_R = 0`` the stable term of ``f``
  is ``0/(1 − 1) = 0/0`` (`K:921`), and the NaN reaches ``K_\mathrm{base}``
  through ``\max(\mathrm{NaN}, 0.1)``, which is processor-dependent.

Branch selection takes the used branch in both cases (the
previous ``h``; ``f = 1``), as it does for entrainment. These selections are implemented with the column physics.

## 10. Integration in AtmosTransport (implemented for LL/RG)

**One field type inside the existing operator.**

```julia
struct TM5CalcKzzField{FT,N,D<:AbstractArray{FT,N},S,P,M} <: AbstractTimeVaryingField{FT,N}
    dkg::D       # topology-native compact interface exchange [kg/s]
    scratch::S   # bottom-up column inputs and intermediates
    params::P    # precision-specific TM5 constants and mixing cutoff
    metrics::M   # mesh metrics built by Models.Physics, adapted with the field
    guard_warned::Base.RefValue{Bool} # host-side once-per-run warning state
end
```

- Files: `src/Operators/Diffusion/TM5CalcKzzField.jl` (type, Adapt,
  `refresh_tm5_calc_kzz!`) and `src/Operators/Diffusion/tm5_calc_kzz.jl` (one
  pure `@inline` column function for CPU and GPU, generic in the float type,
  with no Float64 promotion). The column function returns ``D`` and its
  intermediates (``u_*``, ``G``, ``z``, ``θ_v``, Ri, both height passes,
  ``K_{vh}``, full and active guard counts); column tests call it directly.
- **Solve.** `DkgExchange` reads interface exchange directly. Its factor
  method feeds the existing structured and face-indexed backward-Euler solves,
  including their flux-form tracer-mass update and column ledger. No second
  diffusion operator or new solve kernel is introduced. The new field's
  cubed-sphere route remains deferred.
- **Refresh.** A new `_refresh_pbl_kz_for_window!` method
  (`src/Models/Runner/driven_physics_refresh.jl`) runs once per met window,
  after the window-advance air-mass reset; `_refresh_dz_for_window!` gets a
  no-op method. No new hook is needed.
- **Configuration.** `[diffusion] kind = "tm5_calc_kzz"`, with the existing
  surface-flux couplings (`SplitSurfaceFluxCoupling` by default,
  `DiffusiveSurfaceFluxBoundary` with `surface_flux_boundary = true`, which
  reduced-Gaussian runs reject today). ``l_\mathrm{max}`` is computed when the
  field is materialised (§8).
- `PrecomputedCSDkgField` and `GCHPNonlocalPBLField` stay untouched.

**Inputs (owner decision Q5).** Pressure, carrier mass and winds come from the
runtime state; the binary adds only ``T``, ``q`` and the surface fields, so the
scheme works with either flux construction and on existing binaries plus the
new section group.

| `Calc_Kzz` input | TM5 | Implemented |
|---|---|---|
| Pressure (for ``G``, ``\bar p``, density) | ``G`` from the advected `sp`; `phlb` and ``M`` from the midpoint `spm` | one profile from the window-start carrier mass: ``p_{k+1} = p_k + g\, m_k/(S (1 - q_k))`` from ``p_1 = A_1``, top-down; the ``(1 − q)`` factor only on a dry basis |
| ``M`` in the winds and in ``D`` | midpoint moist mass | runtime carrier mass ``m`` (dry by default); the solve uses the same ``m``, so TM5's ``D/m`` structure is kept |
| ``u``, ``v`` | balanced midpoint `pu`, `pv` over the midpoint ``M`` | stored window amounts: ``u = d_x (\mathrm{am}_w + \mathrm{am}_e)/(2 m s)`` with ``s`` = `flux_application_seconds(window_dt, steps, flux_kind)` and ``d_x = A/L_\mathrm{face}``; reduced Gaussian sums the face segments per cell edge; the cubed sphere rotates panel components with `panel_cell_local_tangent_basis` |
| ``T``, ``q`` | 3-hourly, interpolated to the step midpoint | `tm5_t`, `tm5_q`: mean of the two hourly analysis endpoints on target levels, merged with mass weights |
| ``u_{10}``, ``v_{10}``, `sr` | hourly, interpolated | mean of the two hourly endpoints; `sr` if available, otherwise `fsr` |
| `sshf`, `slhf`, `ewss`, `nsss` | hourly forecast interval means | ERA5 reanalysis hourly accumulations over the window divided by its length |
| `lsm`, `oro` | constant | constant; `lsm` stored as a fraction |
| `srols` | monthly Olsson/Ganzeveld | ERA5 forecast surface roughness `fsr` (Q8) |

*Estimate:* using dry instead of moist mass ratios changes the recovered winds
by ``O(Δq)`` across a face, negligible next to the accepted difference between
analysis and forecast inputs. `tm5_q` is used instead of the `qv_start` and
`qv_end` sections because cubed-sphere binaries do not carry humidity
endpoints.

**Binary section group `tm5_pbl` (implemented, TM5-PBL-1).** Optional, valid only
in format version 5, all-or-nothing, one record per window:

| Sections | Shape | Units and sign |
|---|---|---|
| `tm5_t`, `tm5_q` | cell | K; kg kg⁻¹ of total air |
| `tm5_lsm` | surface | fraction 0–1 |
| `tm5_sr`, `tm5_fsr` | surface | m (`tm5_fsr` replaces `srols`, Q8) |
| `tm5_u10`, `tm5_v10` | surface | m s⁻¹ |
| `tm5_ewss`, `tm5_nsss` | surface | N m⁻², interval mean |
| `tm5_sshf`, `tm5_slhf` | surface | W m⁻², ECMWF downward-positive (unlike the existing upward-positive `pbl_hflux`) |
| `tm5_oro` | surface | m² s⁻² |

*Estimate:* about 40 % of a C90 L34 window payload.

Enable the input writer with `[preprocessing] include_tm5_pbl = true`,
`input.thermo_dir` and `input.surface_dir`. This is an hourly ERA5 spectral
LL/RG option. The thermo sidecar must contain `t` in K and `q` in kg/kg;
the surface archive must contain the ten primitives, including the next day's
00 UTC endpoint. NetCDF, split ZIP and ARCO surface layouts are supported.
Surface primitives (`lsm`, stresses, heat fluxes, `sr`/`fsr`, 10 m winds and
orography) are **cell-area means** on the actual LL or RG mesh, using the package's
shared conservative regridder with intensive-field semantics. In particular,
`lsm` is the coastal cell's land fraction, rather than the value at its centre.
Excursions of the remapped fraction within 1e-12 of [0, 1] are clamped:
rounding in the area mean leaves it up to about 3e-15 above 1 in all-land cells
(LL 72×37, 2021-12-01). Larger excursions fail the binary validator. This
follows TM5's two-dimensional `area-aver` input convention
(`base/trunk/src/tmm.F90:1062–1065`). Roughness uses an arithmetic area mean.
The shared regridder's spherical-polygon approximation and RG ring-boundary
slivers also apply here (see [conservative regridding](../preprocessing/regridding.md)). T/q retain coordinate-aware
bilinear sampling at target cell centres and are merged with native moist
layer-pressure weights before the endpoint mean. The LL and RG thermo paths
share the existing bilinear sampler. Day-local input ownership reuses the shared
hourly thermo endpoint, surface readers (including split-ZIP extraction), and
surface remap weights. `grid.target_vertical` pressure-overlap remapping and
`grid.level_top != 1` are rejected before synthesis; at least two output layers
are required. ERA5 reanalysis accumulations are hourly interval totals labelled
by their ending time, not cumulative forecast-since-initialisation values.
The reader validates units and divides by the supplied interval; already
exported rates keep their values and ECMWF heat-flux signs.

Select the runtime with:

```toml
[diffusion]
kind = "tm5_calc_kzz"
surface_flux_boundary = false
```

The `(1-q)` pressure correction dispatches on **carrier-mass basis**, rather
than `surface_pressure_basis`: these are independent contracts. In particular,
the normal dry-mass binary records total surface pressure but still needs the
dry-to-total layer-mass correction. Archived `ps` does not set the runtime
pressure profile.

Runtime column scratch holds approximately `16 × ncells × nlevels` floating-point
values, plus one integer guard count per cell; the D cache adds one field of
`ncells × nlevels`. PBL height is a local scalar, with no extra stored diagnostic.
For 34 layers and about 542,000 cells this is approximately 1.18 GB of Float32
scratch before D, metrics and solver workspaces. LL72/O24 sizes and CPU timing
are measured in the implementation handoff; large RG runs must budget this
allocation. A smaller-scratch implementation remains future work.

## 11. Differences from TM5

| Topic | TM5 | Implemented here | Note |
|---|---|---|---|
| Refresh cadence | every dynamic step | once per hourly met window | follows the analysis-window decision |
| Pressure | two: advected `sp` for ``G``, midpoint `spm` for `phlb` and ``M`` | one, from the window-start runtime state | refreshed once per window, so the difference spans up to the surface-pressure change over the window |
| Inputs | forecast fields, midpoint interpolation | analysis window means; accumulations over the window | surface primitives are area means on both LL and RG; T/q use point interpolation (§10), whereas TM5 supports area/mass averaging for 3-D inputs |
| Mass basis of ``D`` | moist `spm` mass | runtime carrier mass (dry) | ``D/m`` structure kept; cancels for constant ``q`` |
| Roughness | Olsson/Ganzeveld `srols` for ``z_l`` | ERA5 forecast roughness `fsr` for ``z_l``; `sr` keeps its role (§2) | Q8; size not yet measured |
| Wind recovery length | ``d_x = a_e Δλ \cos φ_j`` | ``d_x = A/L_x``, ``d_y = 2A/(L_s+L_n)`` | effective face-metric lengths; on uniform LL, the relative differences are approximately ``−Δφ^2/24`` zonally and ``+Δφ^2/12`` meridionally |
| Entrainment NaN | propagates or compiler-defined | guarded (§9) | differs only where TM5 is non-finite |
| Masked arithmetic | both branches evaluated | `ifelse` selection | same value where finite |
| Cutoff | ``l_\mathrm{max} = 26`` on tropo34 | pressure equivalent on any level set (§8) | Q7; identical on tropo34a |
| Solve | dense inverse of a matrix combined with convection (`conv:475–476`, `conv:521–529`), split `x y z v s c c s v z y x` (`base/trunk/src/dims.F90:93–94`) | separate backward-Euler diffusion solve at the palindrome centre | Q6: the combined matrix is deferred; the splitting difference remains |
| Precision | Float64 on CPU | Float32 on GPU in production; parity target is Float64 on CPU | Q12; tests report Float32 crossing changes separately |
| Data | production `dkg` cache | live `Calc_Kzz` | Q10 |
| Runtime composition | diffusion inside convection, Check_CFL, slopes, polar clustering | runtime unchanged | bounds any "TM5-style" claim |

TM5's combined matrix also divides by the prognostic mass `m_dat`
(`conv:314`, `conv:475–476`), whereas ``D`` was built from the `spm` mass.

## 12. Relation to the existing schemes

All existing schemes stay bitwise unchanged.

| Aspect | `tm5_bldiff.jl` (preprocessing) | `WindowPBLKzField`, `LocalHoltslagBovilleKzField` | `TM5CalcKzzField` |
|---|---|---|---|
| Where it runs | preprocessing; writes `:dkg` for cubed-sphere binaries (runtime `kind = "tm5_dkg"`) | runtime, layer-centre ``K_z`` | runtime, interface ``D``, once per window |
| Entrainment | TM5 form only for ``Δθ_v/Δz > 10^{-4}`` K m⁻¹ (strict) and a finite ``K ≤ 1000`` m² s⁻¹, then floored at 0.1 m² s⁻¹; else the mixed-layer value | none | TM5 form, guarded only where non-finite |
| Source | port of TM5 cy3-4dvar `bldiff` (file header) | Beljaars–Viterbo with PBL scaling; GEOS VDIFF local scheme | TM5 ct-unified live `Calc_Kzz` |

`_beljaars_viterbo_kz`, `_prandtl_inverse` and `_shear_enhanced_kz` are not
repurposed. A leaf function of `tm5_bldiff.jl` may be shared only where
constants, arithmetic order and mask semantics agree; it then moves to the
earlier `Diffusion` module and the old wrapper stays bitwise unchanged.

## 13. Adjoint and tape (planned, TM5-PBL-2)

- For fixed ``D`` the solve is linear in the tracer, so the adjoint is the
  transpose solve. The planned adjoint (B5) transposes whichever forward solve
  the cubed sphere uses. The ``K_z`` transpose kernels in
  `src/Operators/Diffusion/adjoint_kernels.jl` compute the exchange inline
  with `_kz_exchange`, so a `DkgExchange` factor method there needs a small
  refactor; the existing cubed-sphere `dkg` adjoint
  (`_vertical_diffusion_cs_single_dkg_adjoint_kernel!`, same file) transposes
  the conservative solve of `src/Operators/Diffusion/conservative_dkg.jl`.
- With the default `air_mass_reset_mode` (anything other than `:none`), the
  refresh runs after the reset, so ``D`` depends only on the binary's window
  data; the adjoint recomputes it and needs no tape.
- With `:none`, ``D`` depends on the prognostic mass and is taped per window:
  ``N_c^2 × 6 × l_\mathrm{max}`` values, for C90 with ``l_\mathrm{max} ≈ 26``
  about ``1.26 × 10^6`` values, 5 MB in Float32 per window, 121 MB per day.
- ``D`` is treated as forcing: no derivatives through ``u_*`` or ``K`` with
  respect to the state.
- Adjoint coverage is cubed sphere only, as for the 4D-Var path.

## 14. Tests and tolerances

| Stage | Criterion |
|---|---|
| ``u_*``, heights, winds | Float64, ``10^{-12}`` scaled; branches exact; pure land, pure sea, coast and both friction floors |
| Free ``K`` and Ri | ``10^{-12}`` scaled away from singular conditioning; floors at the right place |
| Boundary-layer height | crossing indices exact away from thresholds; ``h`` to ``10^{-11}`` relative |
| ``K_{vh}``, entrainment | ``10^{-11}``–``10^{-10}`` scaled for finite, well-conditioned columns; non-finite TM5 cases against the §9 rule |
| ``D`` and cutoff | inactive zeros exact; with injected ``K_{vh}``, ``M``, ``G``: ``10^{-12}`` scaled |
| Solve | backward residual ``10^{-13}``–``10^{-12}`` (Float64); isolated layers bit-exact; constant mixing ratio and mass within solver rounding |
| Float32 against Float64 CPU, synthetic fixture columns (well conditioned) | active K and D: ``2 × 10^{-4}`` per nonzero entry; D column L1: ``2 × 10^{-5}``; all K (including above cutoff): ``3 × 10^{-3}``; branch flips reported separately |
| Float32 against Float64 CPU, real ERA5 windows | D per nonzero entry: median ``≤ 3 × 10^{-5}``, p99 ``≤ 5 × 10^{-4}``, maximum ``≤ 5 × 10^{-2}``; D column L1: p99 ``≤ 10^{-3}``, maximum ``≤ 3 × 10^{-2}``; branch flips reported |
| CUDA Float32 against CPU Float32 | D per nonzero entry: p99 ``≤ 2 × 10^{-4}``, maximum ``≤ 5 × 10^{-3}`` |

**Edge inventory (synthetic).** Neutral ``H_v = +0`` and ``−0``; small stable
flux; strong convection; zero and weak shear; no Ri crossing; a crossing
exactly at a centre; the ``h`` floor; interfaces exactly at ``h`` and ``0.1 h``;
the 1000 m mixing-length transition; two equal consecutive bulk Richardson
numbers; ``\mathrm{Ri} = −0.1`` exactly; weak, negative and zero inversions; a
negative ``K_\mathrm{entr}`` clipped by ``D``; the sea stress floor; pure ocean
with invalid land roughness; a coast; mountain orography; the 200 km top; a
full 34-layer scan with 26 mixed layers.

**pydyn fixtures.** Local-only fixtures hold synthetic columns run through
pydyn's pure functions (`ustar`, `geopotential`, `_winds`, `bldiff`, `dkg`,
`calc_kzz`; owner decision Q9). They are generated and kept outside the
repository, are not committed, and reproduce TM5's unguarded arithmetic,
including the entrainment NaN. The parity loader and tests in `test/core/test_tm5_calc_kzz.jl` load fixtures from `ATMOSTR_TM5_FIXTURES_DIR` and skip
with a clear message when the variable is unset or the required fixtures are
absent. Matching the `Calc_Kzz` equations does not establish CFL, advection-moment or
full-model parity with TM5, and an absent or skipped fixture is never reported
as passed.

Float64 CPU comparison on the 16 local synthetic columns: maximum relative
error in `K_vh` is `6.03e-14`, in D `6.03e-14`; maximum absolute errors are
`7.06e-21 m²/s` and `7.47e-14 kg/s`, respectively. Heights, both PBL-height
passes, friction velocity and Obukhov length match exactly on this fixture.
All 32 Float32 PBL-height crossing indices agree with Float64, and all 16
singular-expression counts agree. Measured Float32 D errors are `1.67734e-4`
maximum per nonzero entry and `1.97981e-5` maximum column L1 relative error;
active K has maximum relative error `1.63822e-4`. Across all K interfaces,
including those above the mixing cutoff, the maximum is `2.68030e-3`.
These measurements cover the finite TM5 values; non-finite entrainment checks
use the documented guard. They are column parity, not full-model TM5 parity.

**Float32 on real data.** The fixture bounds do not hold on real ERA5 columns.
Measured on 2021-12-01 (O24, windows at 00, 01, 02 and 12 UTC, identical
window-start inputs), the Float32 D error relative to Float64 per nonzero entry
has median `1.2e-5`, p99 `1.3e-4`, p99.9 `4.5e-4` and maximum `1.0e-2`; the
column L1 error has p99 `2.7e-4` and maximum `9.1e-3`. About 1.8 % of entries
exceed `1e-4` and at most 0.033 % exceed `1e-3`. No PBL-height crossing index or
stability sign changes. The error comes from the conditioning of the scheme,
not from a Float32 defect. K is a function of the gradient Richardson number,
whose numerator ``Δθ_v = θ_{v,l+1} − θ_{v,l}`` is a difference of two values near
300 K. At weak-shear, near-neutral interfaces (``|Δθ_v| ≈ 10^{-3}``–``10^{-2}`` K) the
relative condition number of K with respect to T reaches ``4.7 × 10^5``. Rounding
only the Float64 inputs to Float32 already gives `1.1e-3`–`5.8e-3`, and relative
input noise of `1e-7` gives `3.3e-3`–`7.8e-2`. A Float32 column that forms every
``Δθ_v`` exactly (in Float64, rounded once) still has maximum errors of
`1.1e-3`–`4.0e-3`, the input-rounding level. In one 900 s diffusion solve, the Float32 D changes a surface
plume's mixing ratio by at most `3.2e-4` of the column maximum.

**CUDA.** On an NVIDIA L40S, CUDA Float32 D differs from CPU Float32 by p99
`5e-5` and at most `7.7e-4` (O24) and `1.3e-3` (LL 72×37, 00 UTC). Pressures
and winds agree bitwise; ``θ`` differs by up to 2 ulp, geopotential by up to 3 ulp
and heights by up to 8 ulp (libdevice `powf`/`logf` and FMA contraction, not
separated), and the same conditioning amplifies these differences. CUDA Float32 against CPU Float64 on LL 72×37 at
00 UTC has a maximum of `3.1e-3` and column L1 maximum `1.9e-3`. Three-window
Float32 CUDA runs with live Calc_Kzz change the total air mass by `-3.1e-9`
(O24) and `-1.9e-9` (LL 72×37).

## 15. Owner decisions (2026-10-30)

- **Q5** Pressure, air mass and winds come from the runtime state; the binary
  adds only ``T``, ``q`` and the surface fields.
- **Q6** The combined convection and diffusion matrix is deferred until pure
  diffusion parity passes; it must not require changes to the surface-flux
  operator.
- **Q7** On level sets other than tropo34a, the cutoff is the pressure
  equivalent of tropo34a level 26 (§8).
- **Q8** The ERA5 forecast surface roughness replaces the Olsson/Ganzeveld
  `srols`; the difference is documented (§2).
- **Q9** Local generation of pydyn fixtures from synthetic inputs is approved;
  the fixtures are not committed. A request for a live-`Calc_Kzz` dump waits.
- **Q10** The parity target is TM5's live `Calc_Kzz`, not the NOAA `dkg` cache.
- **Q12** Float32 on the GPU is acceptable for production; parity
  tests use Float64 on the CPU.

On 2026-10-31 the owner bundled the pull requests: TM5-0 (local-only fixtures, this page
and the page
[TM5-style spectral mass fluxes](../preprocessing/tm5_spectral_fluxes.md),
binary format gate), TM5-PBL-1 (B2 column physics
in Float64 on the CPU, B3 binary group and writers, B4 lat-lon and
reduced-Gaussian runtime) and TM5-PBL-2 (B5 cubed sphere and adjoint). B6, the
combined convection and diffusion matrix, stays deferred.

## 16. Implementation status

| Item | Status |
|---|---|
| This page; binary version-5 format gate | Implemented (TM5-0) |
| pydyn `Calc_Kzz` column fixtures | Generated locally outside the repository; not committed; loader and parity tests implemented using `ATMOSTR_TM5_FIXTURES_DIR`, with a clear skip when unset or fixtures are absent |
| B2: bldiff constants, column function, shared layer-count rule | Implemented (TM5-PBL-1) |
| B3: `tm5_pbl` section group, ERA5 surface readers, lat-lon and reduced-Gaussian writers | Implemented (TM5-PBL-1) |
| B4: `TM5CalcKzzField`, `DkgExchange`, lat-lon and reduced-Gaussian runtime, new golden case definitions | Implemented; references generated by the main session |
| B5: cubed sphere, adjoint | Planned (TM5-PBL-2) |
| B6: combined convection and diffusion matrix | Deferred (Q6) |

## References

- Beljaars, A. C. M., and Viterbo, P. (1998). Role of the boundary layer in a
  numerical weather prediction model. In *Clear and Cloudy Boundary Layers*,
  A. A. M. Holtslag and P. G. Duynkerke (eds.), Royal Netherlands Academy of
  Arts and Sciences, Amsterdam, 287–304.
- Holtslag, A. A. M., and Boville, B. A. (1993). Local versus nonlocal
  boundary-layer diffusion in a global climate model. *J. Climate* 6,
  1825–1842.
- Jacobson, M. Z. *Fundamentals of Atmospheric Modeling*, sections 8.3
  (friction velocity) and 8.4 (surface roughness lengths), cited in TM5 at
  `K:220–236`.
- Krol, M., et al. (2005). The two-way nested global chemistry-transport zoom
  model TM5. *Atmos. Chem. Phys.* 5, 417–432.
- Vogelezang, D. H. P., and Holtslag, A. A. M. (1996). Evaluation and model
  impacts of alternative boundary-layer height formulations. *Bound.-Layer
  Meteorol.* 81, 245–269.
