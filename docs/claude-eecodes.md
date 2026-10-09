# eccodes / ecTrans vs Julia spectral preprocessing

_Assessment date: 2026-05-05._
_Branch: `convection` @ `1d75203`._
_Scope: spectral-coefficient → near-surface u, v path. No code changes proposed; this is a written audit only._

## 1. Summary

The Julia spectral path in [src/Preprocessing/spectral_synthesis.jl](../src/Preprocessing/spectral_synthesis.jl) and [src/Preprocessing/spectral_io.jl](../src/Preprocessing/spectral_io.jl) reproduces the **TM5-cy3-4DVar** spherical-harmonic synthesis to within one documented (m,n) loop-bound and one discretisation choice (point evaluation vs cell quadrature). It correctly consumes ECMWF ERA5 spectral GRIB1 messages as decoded by **eccodes 2.35.0** as long as those messages have `JS = KS = MS = 0` and `laplacianOperator P = 0` — which is the operational ERA5 case. Three concrete hazards can bias near-surface u, v:

1. **Synthesis-time spectral truncation at `m = Nlon/2`** (no oversampling).
   On the default `T=639, Nlon=720` ERA5 setup, modes `m ∈ 361..639` are silently discarded. ecTrans on its native TL639/N320 reduced-Gaussian carries all of them. This loses small-scale structure of u near coastlines, sharp gradients and orography.

2. **Cell-center point evaluation rather than refined cell-boundary quadrature** for U at the west face. The `1/cosφ` divisor is applied to a single point sample at `φ_c` rather than integrated `1/cosφ × U(λ_w, φ) dp(λ_w, φ) dφ` across the face — TM5 does the latter (`grid_interpol.F90:3127-3309` `IntLat_sh_ll` with `1/cos(lat)` inside the integrand at line 3263). Polar bias O(1–2%) for the second-most polar row on a 0.5° mesh.

3. **No GRIB-header sanity guard** for `J/K/M`, `JS/KS/MS`, `laplacianOperator`. Silent on a malformed spectral message; latent only.

The recursion coefficients `δ(m,n) = -ε(m,n)/n` and `σ(m,n) = -m/(n(n+1))` in `vod2uv!` ([spectral_synthesis.jl:69-84](../src/Preprocessing/spectral_synthesis.jl#L69-L84)) match the IFS Part III definitions and are equivalent to ecTrans's representation (which divides by `n(n+1)` on the neighbours and by `RA` at the end, vs Julia/TM5 multiplying by `R_E`). The fully-normalised Belousov three-term Legendre recurrence in `compute_legendre_column!` ([spectral_synthesis.jl:33-62](../src/Preprocessing/spectral_synthesis.jl#L33-L62)) matches TM5 `sh_Pnm` ([grid_type_sh.F90:753-833](../deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90)) line for line. The FFT convention (FFTW unnormalised `bfft`) matches IFS practice.

One note in Julia's favour: TM5 truncates the central neighbour term at `n = T-1` (an EMOS-library quirk) while Julia carries it through `n = T` with zero outside; **ecTrans does the same as Julia**, not as TM5. So at this single point Julia is closer to the modern ECMWF reference than TM5 is.

## 2. eccodes: how spectral coefficients are decoded

### 2.1 Storage order

GRIB1 ERA5 spectral messages use `data_g1complex_packing` (sub-truncation + low-resolution simple packing). When you call `codes_get_double_array(handle, "values", val, len)`, eccodes runs the unpack template in `grib_accessor_class_data_complex_packing.cc:675-896`, which fills `val` **column-major over m**:

```
val[ 0..1] = (n=0, m=0) Re,Im        # Im is forced to 0 (line 879-880)
val[ 2..3] = (n=1, m=0) Re,Im
...
val[2T..2T+1] = (n=T, m=0)
val[2T+2..2T+3] = (n=1, m=1)
val[2T+4..2T+5] = (n=2, m=1)
...
```

That is exactly the order

```julia
for m in 0:T
    for n in m:T
        spec[n+1, m+1] = complex(vals[idx], vals[idx+1])
        idx += 2
    end
end
```

which is what `read_spectral_coeffs!` does in [spectral_io.jl:23-30](../src/Preprocessing/spectral_io.jl#L23-L30). **Bit-correct match.**

### 2.2 m=0 imaginary part

eccodes explicitly zeros the imaginary part of every `m=0` coefficient on read (`grib_accessor_class_data_complex_packing.cc:879-880`: "These values should always be zero, but as they are packed, it is necessary to force them back to zero"). Julia's complex pair is therefore correct by construction. Confirmed against `data_sh_packed.cc:354-355` for the older accessor (same zeroing).

### 2.3 Sub-truncation packing (JS / KS / MS / P)

ERA5 IFS spectral messages can have two sub-blocks:

- A high-precision IEEE-32 raw block of size `(JS+1)(JS+2)` for the low-degree coefficients `n ≤ JS`.
- A simple-packed (low-resolution) block for `n > JS`.

For the JS-block, eccodes does **no** laplacian scaling on read when `laplacianOperator P = 0` (the operational ERA5 case): `grib_accessor_class_data_complex_packing.cc:850-859` decodes the IEEE floats verbatim. For the low-resolution part:

```c
val[i++] = d * (T)((grib_decode_unsigned_long(...) * s) + reference_value)
            * scals[lup];
val[i++] = d * (T)(...) * scals[lup];
```

where `scals[lup]` is the inverse of `pow(n*(n+1), laplacianOperator)` (lines 822-836). For ERA5, `P = 0` ⇒ `scals[i] = 1`, so the on-the-wire coefficients ARE the physical spectral values and Julia's reader is correct.

If `P ≠ 0` (some IFS post-processed forecast spectral exports use P=1 or 2 to whiten the spectrum for compression), eccodes reverses the scaling automatically and Julia still gets correct values via the standard `codes_get_double_array("values", …)` path. **The Julia reader does not need a separate inversion.**

What Julia does **not** do:

- Check `laplacianOperator`, `subSetJ`, `subSetK`, or `subSetM` from the GRIB header.
- Check the ccall return code in [spectral_io.jl:11-19](../src/Preprocessing/spectral_io.jl#L11-L19).

If someone hands the preprocessor a malformed spectral GRIB or a non-IFS spectral product with `pen_j ≠ pen_k ≠ pen_m`, the read could silently fill garbage. eccodes will throw `GRIB_DECODING_ERROR` internally, but Julia's wrapper discards it. **[low-but-real risk for non-ERA5 inputs]**.

### 2.4 GRIB1 vs GRIB2

- GRIB1 spectral simple packing: `grib_accessor_class_data_g1shsimple_packing.cc` — prepends an IBM-float real part to a normally simple-packed array; same (m,n) ordering.
- GRIB2 simple packing: `grib_accessor_class_data_g2shsimple_packing.cc` — same shape.
- GRIB2 complex packing: `grib_accessor_class_data_g2complex_packing.cc:102` shows `0, /* unpack_double */` — **no unpack** in eccodes 2.35. GRIB2 spectral therefore has to be GRIB2-simple, IEEE32, or GRIB2-complex-with-IEEE-floats.

ERA5 native streams are GRIB1 complex packed (`packingType=spectral_complex`, edition=1). The Julia reader will fail (return-code-discarded, but no values produced) on any GRIB2 spectral input — this is fine because none are in the ERA5 stream we consume.

### 2.5 Cross-check vs Julia spectral_io

`read_spectral_coeffs!` ([spectral_io.jl:5-31](../src/Preprocessing/spectral_io.jl#L5-L31)) calls `codes_get_double_array(handle, "values", …)` which dispatches to `data_g1complex_packing` → super class `data_complex_packing` → `unpack<double>`. The unpack returns the (m, n) ordering above; Julia rebuilds it correctly. **Match.** `T = msg["J"]` is read at line 20; `J`, `K`, `M` are pentagonal resolution parameters (eccodes `pen_j`, etc.) and are equal for triangular ERA5 truncation. There is no rescaling of the JS sub-block on read; eccodes already returns unscaled values.

## 3. ecTrans: official ECMWF spectral→grid path

### 3.1 vord2uv recursion

Reference: [`ecmwf-ifs/ectrans/src/trans/cpu/internal/vdtuv_mod.F90`](https://github.com/ecmwf-ifs/ectrans/blob/main/src/trans/cpu/internal/vdtuv_mod.F90), KM ≠ 0 branch (lines 117-129):

```fortran
DO JI=2,ISMAX+3-KM
  PU(JI,IR) = -ZKM*ZLAPIN(JI)*PDIV(JI,II) + &
              ZN(JI+1)*ZEPSNM(JI)*ZLAPIN(JI+1)*PVOR(JI+1,IR) - &
              ZN(JI-2)*ZEPSNM(JI-1)*ZLAPIN(JI-1)*PVOR(JI-1,IR)
  PU(JI,II) = +ZKM*ZLAPIN(JI)*PDIV(JI,IR) + ...
  PV(JI,IR) = -ZKM*ZLAPIN(JI)*PVOR(JI,II) - &
              ZN(JI+1)*ZEPSNM(JI)*ZLAPIN(JI+1)*PDIV(JI+1,IR) + &
              ZN(JI-2)*ZEPSNM(JI-1)*ZLAPIN(JI-1)*PDIV(JI-1,IR)
  PV(JI,II) = ...
ENDDO
```

with arrays in *reversed* meridional order (`IJ = ISMAX+3-JN`, lines 86-88). The physical meaning:

- `RLAPIN(n) = 1/(n(n+1))` (a "Laplacian inverse" applied to neighbouring degrees)
- `RN(n)` is a Legendre normalisation factor
- `PEPSNM(m,n) = ε(m,n) = √((n²-m²)/(4n²-1))` — same as Julia `_delta` × `n`, same as TM5

The wrapper [`vd2uv_mod.F90`](https://github.com/ecmwf-ifs/ectrans/blob/main/src/trans/cpu/internal/vd2uv_mod.F90) finishes with:

```fortran
ZA_R = 1.0_JPRB / REAL(RA, JPRB)
PU(JFLD, INM) = ZIA(J+2, IR+IUL-1) * ZA_R
```

i.e., **divides by Earth radius** `RA` (`TPM_CONSTANTS::RA`).

ecTrans is therefore working with `PVOR / PDIV` interpreted as streamfunction/velocity-potential coefficients (they are pre-divided by `n(n+1)` via `ZLAPIN` on the neighbours), and the final `1/RA` scales the gradient back to physical units (m/s).

The Julia/TM5 form works directly with VO, DI and uses `δ(m,n) = -ε(m,n)/n` multiplied by `R = a_e`. **The two formulations are mathematically equivalent; this is a representation choice, not a bug.**

### 3.2 Loop bounds — Julia matches ecTrans, not TM5

| Implementation | n-loop bounds for the central δ_{n+1}·VO_{n+1} term |
|---|---|
| Julia [`vod2uv!`](../src/Preprocessing/spectral_synthesis.jl#L114-L115) | `m..T`, with `vo_np1 = 0` for `n == T` |
| TM5 [`sh_vod2uv`](../deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90) lines 1900-1918 | `m..T-1` (then a separate `(m,T-1)` and `(m,T)` cap) — EMOS-library bug emulation |
| ecTrans [`VDTUV`](https://github.com/ecmwf-ifs/ectrans/blob/main/src/trans/cpu/internal/vdtuv_mod.F90) line 117 | `JI=2..ISMAX+3-KM` — equivalent to Julia (full range with array over-dim) |

**Julia matches ecTrans. TM5's truncation is an artefact of EMOS-library compatibility** noted at `grid_type_sh.F90:1847-1848`. ecTrans dropped that emulation; Julia did not adopt it. So at this single technical point, Julia is closer to modern ECMWF practice than TM5.

### 3.3 Normalisation

ecTrans uses fully-normalised P̃_n^m (`∫ P̃² dμ = 1`), same as TM5 and Julia. The Belousov three-term recurrence in Julia ([spectral_synthesis.jl:33-62](../src/Preprocessing/spectral_synthesis.jl#L33-L62)) matches TM5 [`sh_Pnm`](../deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90) (lines 753-833) line for line:

| Step | Julia | TM5 |
|---|---|---|
| `P̃_0^0 = 1` | `P[1,1] = 1.0` (line 38) | `Pnm(1) = 1.0` (line 797) |
| `P̃_m^m = √((2m+1)/(2m)) cosφ · P̃_{m-1}^{m-1}` | line 42 | `fmm = fmmp · rmu / √(2m)` (line 802) |
| `P̃_{m+1}^m = √(2m+3) sinφ · P̃_m^m` | line 47 | line 808 |
| Three-term: `P̃_n^m = a·μ·P̃_{n-1}^m − b·P̃_{n-2}^m`, `a = √((4n²-1)/(n²-m²))`, `b = √(((2n+1)(n-m-1)(n+m-1))/((2n-3)(n²-m²)))` | lines 53-58 | lines 815-823 (form `(μ·P_{n-1} − ε₁·P_{n-2})/ε`, algebraically identical) |

**Match.** Julia's file header at [spectral_synthesis.jl:11-13](../src/Preprocessing/spectral_synthesis.jl#L11-L13) cites IFS Documentation Part III §2.3 directly.

### 3.4 Oversampling / dealiasing — biggest discretisation gap

ECMWF operational ERA5 archives spectral truncation `T639` (or `TL639`). The accompanying Gaussian grid is the **linear reduced Gaussian** `N320` (640+1+640 latitudes, ~1280 longitudes near the equator, fewer near the poles).

Julia preprocessor on the regular `720 × 361` lat-lon mesh (`config/preprocessing/era5_ll720x361_*.toml`):

- `Nlon = 720`, `T = 639`. Nyquist `m_max = 360`. Julia synthesises `m ∈ 0..min(T, 360)` ([spectral_synthesis.jl:180](../src/Preprocessing/spectral_synthesis.jl#L180)). **Modes `m ∈ 361..639` are silently discarded** — no anti-aliasing filter, just truncation.
- `Nlat = 361` is regular-spaced, not Gaussian. ecTrans uses Gaussian latitudes for exact quadrature.

ECMWF's recommendation for a regular lat-lon target is `Nlon ≥ 2T+1 = 1279` for linear, `≥ 3T+1 = 1918` for quadratic. **Julia's 720 is well below either.**

For near-surface u, v this loses ~30–40% of the zonal wavenumbers' contribution to small-scale structure (coastlines, orography, fronts). Total energy loss is typically a few percent because ERA5 power decays with `m`, but for high-gradient regions the local effect is larger. **[medium risk on small-scale wind structure; small risk on planetary-scale wind]**.

### 3.5 cos(φ) handling

ecTrans returns U·cosφ, V·cosφ in spectral form to the inverse Legendre + FFT step (standard ECMWF practice — the meridional operator stays well-conditioned at the poles). The division by cosφ to recover physical wind happens in user code (or, in IFS, on the grid-point dynamics side via metric tensors).

Julia matches: `vod2uv!` produces `u_spec, v_spec` interpreted as U·cosφ and V·cosφ ([spectral_synthesis.jl:99-104](../src/Preprocessing/spectral_synthesis.jl#L99-L104)), and the division is deferred to `compute_mass_fluxes!` ([spectral_synthesis.jl:362-380](../src/Preprocessing/spectral_synthesis.jl#L362-L380)):

```julia
am[i, j, k] = u_stag[i, j, k] / cos_lat * dp_face * R_g * dlat * half_dt
```

Cell-center `cos_lat = grid.cos_lat[j]` is computed from the **cell-center** latitude. This is the midpoint approximation of the cell-edge integral. The area-average of `1/cosφ` across a cell `[φ_s, φ_n]` is **not** `1/cos(φ_c)` — TM5 evaluates `(da + db·exp(LNSP))/cos(φ)` *inside* its refined latitudinal quadrature ([grid_interpol.F90:3263](../deps/tm5-cy3-4dvar/base/src/grid_interpol.F90)). For low-to-mid latitudes the midpoint error is `O(Δφ²)` and sub-percent; near the poles where `1/cosφ` blows up the error grows.

### 3.6 Pole handling

ecTrans uses Gaussian latitudes that never hit the pole. Julia's regular `-90..90` mesh has cell-center latitudes `±89.75°` for `Nlat=361` (never exactly ±90), so `1/cosφ` is finite. At the polar **rows** of `am`, [spectral_synthesis.jl:357-358](../src/Preprocessing/spectral_synthesis.jl#L357-L358) zeros the zonal mass flux. At polar boundaries of `bm`, the meridional flux is zeroed too. **Closed-pole convention; matches TM5; does not match ecTrans because ecTrans never sees a pole.**

## 4. Julia code audit — end to end

What happens to a single ERA5 spectral message:

1. **GRIB read**: [`read_day_spectral_streaming`](../src/Preprocessing/spectral_io.jl) iterates GRIB messages. For each VO/D, `read_spectral_coeffs!` calls `codes_get_double_array` and unpacks into a complex `(T+1)×(T+1)` matrix per (level, hour). Level ordering uses `msg["level"]` (1..137), with `k=1`=TOA → `k=137`=surface — already the runtime convention. No re-ordering.

2. **Phase-shift setup** in `spectral_to_native_fields!` ([spectral_synthesis.jl:423](../src/Preprocessing/spectral_synthesis.jl#L423)):
   - `sp_shift = deg2rad(grid.lons[1])` — for cell-center scalars (sp, v_cc).
   - `u_edge_shift = deg2rad(first(grid.mesh.λᶠ))` — for west-edge u_cc.
   This is the fix for the 180°-off bug noted at [spectral_synthesis.jl:444-456](../src/Preprocessing/spectral_synthesis.jl#L444-L456). Looks correct for both `lons = (0, 360)` and `lons = (-180, 180)`.

3. **vod2uv!** per level, threaded across levels (one buffer set per thread). Bug-hunt details:
   - `_sigma(m, 0) = 0` (line 82) and `_sigma(0, n) = 0` (line 83). The `m=0` zero is physically correct — at zonal mean there is no `i·m·D` rotation contribution.
   - Neighbours: `vo_nm1 = (n > m) ? vo_spec[n, m+1] : 0`, `vo_np1 = (n < T) ? vo_spec[n+2, m+1] : 0`. The second bound differs from TM5 (which would be `n < T-1`); matches ecTrans.

4. **`spectral_to_grid!`** ([spectral_synthesis.jl:160](../src/Preprocessing/spectral_synthesis.jl#L160)). Per latitude:
   - Build `P̃_n^m(sin φ_j)`.
   - For each `m ∈ 0..min(T, Nlon/2)`, sum `Σ_{n=m}^T spec[n+1, m+1] · P̃_n^m`.
   - **Truncation at `m = Nlon/2 - 1`**, `Nlon/2` Nyquist bin left zero. Loses one extra mode at exactly Nyquist (typically negligible). The bigger issue is `m > Nlon/2` is dropped entirely (see §3.4).
   - Apply zonal phase shift `Gm *= exp(im · m · lon_shift_rad)`.
   - Conjugate-symmetric fill, unnormalised `bfft`. Correct (matches ECMWF unnormalised inverse FFT convention).

5. **`stagger_winds!`** ([spectral_synthesis.jl:236-254](../src/Preprocessing/spectral_synthesis.jl#L236-L254)). u_stag is a copy of u_cc (synthesised at the west face — phase-shifted appropriately). v_stag is a 2-cell average across the south face. Pole rows zeroed.

6. **`compute_mass_fluxes!`** ([spectral_synthesis.jl:341-411](../src/Preprocessing/spectral_synthesis.jl#L341-L411)).
   `am = u_stag / cos_lat × dp_face × R_g × dlat × half_dt`. All evaluated at cell-center latitude `φ_j`. **Midpoint approximation, see §3.5**.

7. **`cm`** is diagnosed from horizontal divergence + `dB[k] · pit` (lines 386-407). Independent of u, v accuracy except through column-integrated convergence.

### 4.1 CS path

[`_synth_and_regrid_to_cs!`](../src/Preprocessing/transport_binary/cubed_sphere_spectral.jl) synthesises to a regular LL **staging** mesh (default `staging_nlon × staging_nlat`), regrids to CS panels, then reconstructs CS face fluxes from the regridded *cell-center winds* ([cs_flux_reconstruction.jl](../src/Preprocessing/cs_flux_reconstruction.jl) `reconstruct_cs_fluxes!`). The Poisson balance ([cs_poisson_balance.jl](../src/Preprocessing/cs_poisson_balance.jl)) projects the divergence to be exactly consistent with `m_next - m_cur`, which **wipes out** small pointwise spectral synthesis error at the cost of redistributing it as a small uniform offset on each face. So on the CS target the dominant error is **regrid + Poisson balance**, not point-evaluation in spectral synthesis.

### 4.2 Bug-hunt summary

| Location | Issue | Bias on near-surface u,v | Severity |
|---|---|---|---|
| [spectral_synthesis.jl:180](../src/Preprocessing/spectral_synthesis.jl#L180) | `m`-truncation at `Nlon/2`; `T=639 > 360` | Loses high-m modes; localised impact at coastlines/orography | medium |
| [spectral_synthesis.jl:362-364](../src/Preprocessing/spectral_synthesis.jl#L362-L364) | Cell-center `1/cos_lat` instead of integrated `1/cosφ` | Polar U bias O(1–2%) at 89° | medium |
| [vod2uv!](../src/Preprocessing/spectral_synthesis.jl#L108-L135) bound `n=T` | One extra mode vs TM5 EMOS-truncation; matches ecTrans | Tiny, top-of-spectrum only | negligible |
| [spectral_io.jl:11-19](../src/Preprocessing/spectral_io.jl#L11-L19) | ccall return-code discarded; no `J/K/M`, `JS/KS/MS`, `laplacianOperator` guard | Silent on malformed GRIB | low (latent) |
| [stagger_winds!](../src/Preprocessing/spectral_synthesis.jl#L248-L249) pole | Closed pole; matches Julia stagger contract | n/a | n/a |
| [reconstruct_cs_fluxes!](../src/Preprocessing/cs_flux_reconstruction.jl) | Face flux from 2-cell-center average + cell-center `dp` | Sub-percent on fine staging mesh; absorbed by Poisson balance | low |

## 5. TM5-4DVar reference points

- [`sh_vod2uv`](../deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90) lines 1852-1939: multiplies by `R = ae` from `Binas` module (line 1854). Same direction as Julia.
- The `(m,T-1)` cap at line 1908-1911 retains both `δ(m,n)·VO_{n-1}` and `i·σ(m,n)·D_n` but **drops** `δ(m,n+1)·VO_{n+1}`; the `(m,T)` cap at line 1914-1918 keeps only `δ(m,n)·VO_{n-1}`. EMOS-style T-1 truncation. ecTrans's recurrence carries `n+1` to `T+1` and zeros the out-of-range PVOR via array over-dimensioning. Julia carries `n+1` to `T+1` and zeros via the explicit `n < T ? ... : 0`. **Julia ≡ ecTrans, both differ from TM5 by one spectral mode.**
- [`Aver_sh_ll`](../deps/tm5-cy3-4dvar/base/src/grid_interpol.F90) at lines 1627-1739: refined longitude/latitude quadrature with `IntervalQuad_Lin` and `IntervalQuad_Cos_Lin`. The cell-boundary integration the Julia path skips.
- [`IntLat_sh_ll`](../deps/tm5-cy3-4dvar/base/src/grid_interpol.F90) at lines 3127-3309: applies `(da + db·exp(H))/cos(lat)` *inside* the latitudinal quadrature (line 3263). U-flux computation. `1/cos(lat)` is inside the integrand, evaluated on the refined latitude. This is what Julia's midpoint-`1/cos(φ_c)` is approximating.

## 6. Bottom line — match status table

| Step | Julia | ecTrans | TM5 4DVar | eccodes | Match? |
|---|---|---|---|---|---|
| GRIB unpack ordering | (m, n) col-major, R-then-I | n/a | n/a | (m, n) col-major (`data_complex_packing.cc:846-882`) | ✓ |
| m=0 imag forced 0 | by construction (eccodes wrote 0) | n/a | n/a | yes (`data_complex_packing.cc:879-880`) | ✓ |
| Sub-truncation laplacian rescale | not handled in Julia, eccodes does it | n/a | n/a | yes if P ≠ 0 (`data_complex_packing.cc:830-836`) | ✓ for ERA5 (P=0) |
| `J/K/M` consistency check | no | n/a | n/a | yes (asserts) | ⚠ Julia silent |
| Legendre normalisation | fully-normalised P̃ | fully-normalised P̃ | fully-normalised P̃ | n/a | ✓ |
| `δ(m,n)`, `σ(m,n)` formulas | `-ε(m,n)/n`, `-m/(n(n+1))` | uses `RN·EPSNM·LAPIN` (equiv via `RA` factor) | identical to Julia | n/a | ✓ |
| `n=T` neighbour | included with zero | included with array over-dim zero | truncated at `T-1` (EMOS) | n/a | Julia ≡ ecTrans, differs from TM5 |
| Earth-radius scale | × R_E | / RA (different rep.) | × ae | n/a | ✓ (rep. equivalent) |
| Cell quadrature for U at face | midpoint at φ_c | grid-side spectral evaluation, IFS uses Gaussian quadrature | refined quadrature with `1/cosφ` inside integrand | n/a | ⚠ Julia midpoint only |
| `1/cosφ` placement | applied to point sample at cell center | applied per Gaussian latitude after Legendre | inside the lat integrand | n/a | ⚠ |
| FFT origin / phase shift | `exp(i·m·lon_shift)` per cell-edge or center | post-FFT shift not used (Gaussian grid native) | similar phase-shift in `Eval_Lons` | n/a | ✓ |
| FFT normalisation | bfft (unnormalised) | unnormalised | unnormalised | n/a | ✓ |
| Pole handling | closed pole (am=0 at j=1, j=Nlat; bm=0 at j=1, j=Nlat+1) | no pole (Gaussian) | closed pole | n/a | n/a (different grid types) |
| Dealiasing | hard truncate at `m=Nlon/2` (loses m>360 on T=639,Nlon=720) | TL/TQ Gaussian (e.g. N320 for T639) | refined LL (1280×640 commonly) | n/a | ⚠ Julia under-resolved |

## 7. Likely sources of near-surface u,v difference, ranked

1. **Synthesis-time spectral truncation at `m = Nlon/2`** ([spectral_synthesis.jl:180](../src/Preprocessing/spectral_synthesis.jl#L180)). On default `T=639, Nlon=720` ERA5, ~44% of zonal wavenumbers are discarded with no anti-aliasing filter. Affects small-scale structure of u near coastlines and orography most strongly. Probably the largest single contribution in a ground-truth U comparison.

2. **Midpoint vs cell-boundary integration of `U/cosφ × dp` for `am`** ([spectral_synthesis.jl:362-364](../src/Preprocessing/spectral_synthesis.jl#L362-L364)). Polar bias in the meridional profile of zonal-mean U; sub-percent at mid-latitudes, O(1–2%) at 89.5°.

3. **Half-Nyquist underfill.** The conjugate-symmetry fill ([spectral_synthesis.jl:196-198](../src/Preprocessing/spectral_synthesis.jl#L196-L198)) goes up to `m = Nlon/2 - 1`. Bin `Nlon/2` (Nyquist) is left zero. For even `Nlon`, loses a single mode (typically negligible).

4. **TM5-style `(m,T-1)` truncation NOT used.** Differs from TM5 by one high-(m,n) mode per zonal wavenumber. **Julia matches modern ecTrans here**, so this is not a defect; just a difference vs the older TM5 reference.

5. **Conservative regrid + Poisson balance on the CS path** absorbs most of (1)-(2) into a uniform per-face correction. So on the CS target the dominant error becomes **regrid resolution** rather than spectral synthesis.

6. **`spectral_io` does not validate `pen_j == pen_k == pen_m == J`** — would fail silently on a malformed GRIB. Latent only.

## 8. Suggested numerical probes

In priority order:

1. **Spectrum check.** For a representative ERA5 day, plot the zonal power spectrum of `am` and compare against a reference computed by running the same VO/D through eccodes-tools `grib_ls --json` → ecTrans Python bindings (or `epygram`/`gribset`). The Julia spectrum should drop to zero at `m = 360` on a 720-lon mesh. If it does not, there is a wraparound or aliasing bug.

2. **Polar U bias probe.** Compute zonal-mean U at level 137 (surface) on the Julia `720×361` mesh from a single ERA5 hour. Compare to a direct ECMWF TL639/N320 reduced-Gaussian → 720×361 conservative regrid via `cdo remapbic` or `metview` (which avoids the synthesis entirely). Difference should be ~O(0.5%) at mid-latitudes, growing to ~O(2%) at 89.5°. Larger means the midpoint approximation is biting.

3. **Truncation-mode test.** Run with `T_target = 360` (matches the 720-lon Nyquist exactly) and compare resulting `u, v` against the default `T_target = 639`. The difference is the silently-dropped-mode contribution.

4. **GRIB sanity assertion** (latent-only, no immediate effect on physics). A one-off check that `msg["J"] == msg["K"] == msg["M"]` and `msg["JS"] == 0` (or `laplacianOperator == 0`) in `read_spectral_coeffs!`. Should be a no-op on every ERA5 input today; catches malformed inputs in the future.

5. **Round-trip check.** Take Julia-synthesised `u_cc` on the 720×361 mesh, run forward spectral analysis (project onto `Y_n^m`) up to `T = 360`, then synthesise again. Idempotent to floating-point precision when the original `T ≤ 360`. With `T = 639`, the double-synthesis reveals the mode loss.

6. **Single-mode probe.** Set `vo[m=300, n=300] = 1+0i` and synthesise. Verify the gridpoint pattern matches the closed-form `Y_300^300(λ, φ)`. Isolates the Legendre + FFT chain from the `vod2uv` chain.

## 9. Citations

### Local file:line

eccodes 2.35.0 (under `/home/cfranken/code/eccodes-2.35.0-Source/`):
- `src/grib_accessor_class_data_complex_packing.cc:198-303` — `calculate_pfactor`; sub-truncation rescale derivation.
- `src/grib_accessor_class_data_complex_packing.cc:675-896` — `unpack` template; the actual decoding loop (m-outer / n-inner).
- `src/grib_accessor_class_data_complex_packing.cc:822-836` — `scals[i] = 1.0 / pow(i*(i+1), laplacianOperator)`; sub-truncation laplacian rescale.
- `src/grib_accessor_class_data_complex_packing.cc:874-882` — low-resolution coefficient unpack with `* scals[lup]`; m=0 imag forced to 0.
- `src/grib_accessor_class_data_sh_packed.cc:331-362` — legacy `data_sh_packed` unpack, identical ordering.
- `src/grib_accessor_class_data_sh_unpacked.cc:322-353` — `data_sh_unpacked`, with explicit GRIBEX-bug branch.
- `src/grib_accessor_class_spectral_truncation.cc:144-156` — J/K/M consistency check (triangular vs rhomboidal vs trapezoidal).
- `src/grib_accessor_class_data_g1complex_packing.cc:139-235` — GRIB1 wrapper: `pen_j = J`, etc.
- `src/grib_accessor_class_data_g2complex_packing.cc:79-164` — GRIB2 wrapper: has pack but **no unpack** in 2.35.
- `src/grib_accessor_class_data_g1shsimple_packing.cc:117-150` — GRIB1 simple SH unpack (prepends real_part to coded_values).
- `src/grib_accessor_class_data_g2shsimple_packing.cc:127-157` — GRIB2 simple SH unpack (same shape as GRIB1).
- `src/grib_accessor_class_statistics_spectral.cc:191-205` — spectral statistics; confirms (m,n) Re/Im interleaved layout.
- `definitions/grib1/data.spectral_complex.def:14-101` — definition glue: `data.values` → `data_g1complex_packing`.
- `definitions/grib1/data.spectral_simple.def:21-39` — definition for non-complex spectral.

Julia preprocessor:
- [src/Preprocessing/spectral_io.jl:5-31](../src/Preprocessing/spectral_io.jl#L5-L31) — `read_spectral_coeffs!`; the GRIB → matrix step.
- [src/Preprocessing/spectral_synthesis.jl:33-62](../src/Preprocessing/spectral_synthesis.jl#L33-L62) — `compute_legendre_column!`; Belousov three-term recurrence.
- [src/Preprocessing/spectral_synthesis.jl:69-84](../src/Preprocessing/spectral_synthesis.jl#L69-L84) — `_delta`, `_sigma` definitions.
- [src/Preprocessing/spectral_synthesis.jl:108-135](../src/Preprocessing/spectral_synthesis.jl#L108-L135) — `vod2uv!`; VO/D → U·cosφ, V·cosφ in spectral space.
- [src/Preprocessing/spectral_synthesis.jl:160-217](../src/Preprocessing/spectral_synthesis.jl#L160-L217) — `spectral_to_grid!`; Legendre + FFT synthesis with phase shift and Nyquist truncation.
- [src/Preprocessing/spectral_synthesis.jl:236-254](../src/Preprocessing/spectral_synthesis.jl#L236-L254) — `stagger_winds!`; c-grid placement and pole zeroing.
- [src/Preprocessing/spectral_synthesis.jl:341-411](../src/Preprocessing/spectral_synthesis.jl#L341-L411) — `compute_mass_fluxes!`; face-flux calculation; cell-center `cos_lat`.
- [src/Preprocessing/spectral_synthesis.jl:423-497](../src/Preprocessing/spectral_synthesis.jl#L423-L497) — `spectral_to_native_fields!`; the day-loop driver.
- [src/Preprocessing/transport_binary/latlon_spectral.jl:33-149](../src/Preprocessing/transport_binary/latlon_spectral.jl#L33-L149) — LL day-process pipeline.
- [src/Preprocessing/transport_binary/cubed_sphere_spectral.jl:133-185](../src/Preprocessing/transport_binary/cubed_sphere_spectral.jl#L133-L185) — `_synth_and_regrid_to_cs!`; staging LL → CS regrid + flux reconstruct.
- [src/Preprocessing/cs_transport_helpers.jl](../src/Preprocessing/cs_transport_helpers.jl) — `recover_ll_cell_center_winds!`; flux→wind on staging LL.
- [src/Preprocessing/cs_flux_reconstruction.jl](../src/Preprocessing/cs_flux_reconstruction.jl) — `reconstruct_cs_fluxes!`; wind→flux on CS panels.

TM5 cy3-4DVar:
- [deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90](../deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90) lines 753-833 — `sh_Pnm`; Belousov recurrence (matches Julia).
- [deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90](../deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90) lines 1852-1939 — `sh_vod2uv`; T-1 truncation; multiplies by R=ae.
- [deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90](../deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90) lines 1434-1528 — `shi_Eval_Lons_Ulat` / `shgrid_Eval_Lons_Ulat`; single-lat evaluation (no boundary integral).
- [deps/tm5-cy3-4dvar/base/src/grid_interpol.F90](../deps/tm5-cy3-4dvar/base/src/grid_interpol.F90) lines 1627-1739 — `Aver_sh_ll`; refined cell-boundary quadrature.
- [deps/tm5-cy3-4dvar/base/src/grid_interpol.F90](../deps/tm5-cy3-4dvar/base/src/grid_interpol.F90) lines 3127-3309 — `IntLat_sh_ll`; refined lat integration with `1/cos(lat)` inside integrand at line 3263.

### ecTrans (GitHub URLs, current `main`)

- https://github.com/ecmwf-ifs/ectrans/blob/main/src/trans/cpu/internal/vd2uv_mod.F90 — `VD2UV` wrapper, divides by `RA`.
- https://github.com/ecmwf-ifs/ectrans/blob/main/src/trans/cpu/internal/vdtuv_mod.F90 — core recurrence (lines 95-132 KM=0 / KM≠0 branches); references Temperton 1991 MWR p.1303 at line 55.
- https://github.com/ecmwf-ifs/ectrans/blob/main/src/trans/cpu/internal/vd2uv_ctl_mod.F90 — driver loop calling `VD2UV` per zonal wavenumber.
- https://github.com/ecmwf-ifs/ectrans/blob/main/src/trans/cpu/internal/leinv_mod.F90 — inverse Legendre on a grid latitude.
- https://github.com/ecmwf-ifs/ectrans/blob/main/src/trans/cpu/external/vordiv_to_uv.F90 — public-API entry point.
- https://github.com/ecmwf-ifs/ectrans/tree/main/src/trans/cpu/internal — full file listing (vd2uv_ctl_mod, vd2uv_mod, vdtuv_mod, vdtuvad_mod, uvtvd_mod, uvtvdad_mod, leinv_mod, ledir_mod).
- https://github.com/ecmwf-ifs/ectrans/blob/main/README.md — high-level overview.
- https://sites.ecmwf.int/docs/ectrans/ — ecTrans documentation index.

### IFS Documentation / textbook references

- IFS Documentation Cy48r1 Part III: Dynamics and Numerical Procedures, ECMWF, 2023. §2.2 (spherical-harmonic representation), §2.3 (fully normalised P̃ recurrence), Eq. 2.14-2.15 (vorticity-divergence to UV reconstruction). Cited at [spectral_synthesis.jl:11-13](../src/Preprocessing/spectral_synthesis.jl#L11-L13).
- Temperton C., 1991: "On scalar and vector transform methods for global spectral models." *Monthly Weather Review* 119, 1303-1307. Cited by ecTrans `vdtuv_mod.F90:55` and `vd2uv_mod.F90:59` — the ultimate reference for the recurrence everyone uses.

---

_Tags used: ✓ = verified match; ⚠ = real difference; n/a = not applicable. Speculative claims (numerical magnitudes for biases) tagged inline. No code changes proposed; this document records the audit only._
