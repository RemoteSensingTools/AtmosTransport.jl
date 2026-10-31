# ============================================================================
# Spherical-harmonic spectral → gridpoint synthesis for ERA5 preprocessor.
#
# Converts spectral coefficients (VO, D, LNSP) to gridpoint fields (u, v, sp)
# at target grid latitudes using:
#   1. Associated Legendre polynomials P_n^m(sin φ) via three-term recurrence
#   2. Vorticity/divergence → wind reconstruction (IFS convention)
#   3. Backward FFT for longitude synthesis
#
# References:
#   Temperton (1991) "On scalar and vector transform methods for global SH"
#   IFS Documentation Part III: Dynamics, ECMWF (2021), Ch. 2.2
#   TM5 cy3-4dvar meteo.F90:1200-1350 (spectral inversion)
# ============================================================================

"""
    compute_legendre_column!(P, T, sin_lat)

Fill the fully-normalised associated Legendre polynomial table `P[n+1, m+1]`
for spectral truncation `T` at latitude `φ` (given as `sin_lat = sin(φ)`).

Uses the standard three-term recurrence (Belousov 1962):

    P_m^m     = √((2m+1)/(2m)) · cos(φ) · P_{m-1}^{m-1}     (sectoral)
    P_{m+1}^m = √(2m+3) · sin(φ) · P_m^m                      (semi-sectoral)
    P_n^m     = a · sin(φ) · P_{n-1}^m − b · P_{n-2}^m          (general)

where `a`, `b` include the normalisation factors for the fully-normalised
convention used by ECMWF spectral fields (IFS Doc Part III, Eq. 2.3).

Output: `P` is `(T+1) × (T+1)`, 1-indexed, with `P[n+1, m+1] = P̃_n^m(sin φ)`.
"""
function compute_legendre_column!(P::AbstractMatrix{<:Real}, T::Int, sin_lat::Real)
    # cos(φ) from sin(φ) avoids the tan(φ) pole singularity
    cos_lat = sqrt(1.0 - sin_lat^2)

    fill!(P, 0.0)
    P[1, 1] = 1.0   # P̃_0^0 = 1

    # Sectoral recurrence: P̃_m^m from P̃_{m-1}^{m-1}
    for m in 1:T
        P[m + 1, m + 1] = sqrt((2m + 1.0) / (2m)) * cos_lat * P[m, m]
    end

    # Semi-sectoral: P̃_{m+1}^m from P̃_m^m
    for m in 0:(T - 1)
        P[m + 2, m + 1] = sqrt(2m + 3.0) * sin_lat * P[m + 1, m + 1]
    end

    # General three-term recurrence: P̃_n^m from P̃_{n-1}^m and P̃_{n-2}^m
    for m in 0:T
        for n in (m + 2):T
            n2 = n * n
            m2 = m * m
            a = sqrt((4.0 * n2 - 1.0) / (n2 - m2))
            b = sqrt(((2n + 1.0) * (n - m - 1.0) * (n + m - 1.0)) /
                     ((2n - 3.0) * (n2 - m2)))
            P[n + 1, m + 1] = a * sin_lat * P[n, m + 1] - b * P[n - 1, m + 1]
        end
    end
    return nothing
end

"""
    ring_max_wavenumber(T, nlon) -> Int

Highest zonal wavenumber synthesised on a latitude ring of `nlon` equally
spaced points from a field of triangular truncation `T`:

    M = min(T, ⌊(nlon − 1) / 2⌋).

Every spectral synthesizer uses this limit (`spectral_to_grid!`,
`spectral_to_ring!` and `synthesize_reduced!`, through
[`fill_ring_spectrum!`](@ref) for the first two).

On a ring of `N` points `λ_k = λ_0 + 2πk/N` (k = 0, …, N − 1) the real field

    f(λ_k) = G_0 + 2 Re Σ_{m=1}^{M} G_m e^{i m λ_k}

is the unnormalised inverse DFT of bins `m` and `N − m` holding `G_m e^{i m λ_0}`
and its conjugate, which are distinct bins for every `m < N/2`. For `m = N/2`
on an even ring they are one bin: the samples are
`e^{i N λ_k / 2} = e^{i N λ_0 / 2} (−1)^k`, so the term `2 Re(G e^{i m λ_k})`
samples as `2 Re(G e^{i N λ_0 / 2}) (−1)^k`. That depends on the ring's
origin: `2 Re G` at the faces (`λ_0 = 0`) and `−2 Im G` at the cell centres
(`λ_0 = π/N`), so one coefficient cannot be represented consistently on the
staggered points of a ring. The term is dropped, and every `1 ≤ m ≤ M` enters
with its mirror (weight 2).

This matches ECMWF ecTrans's rule for linear reduced grids,
`NMEN = MIN(NSMAX, (NLOEN − 1)/2)`
(`src/trans/common/internal/setup_geom_mod.F90:48`, ecTrans `aff81b9`), including
ERA5's linear N320 grid at T639. On lat-lon and cubed-sphere staging targets
`target_spectral_truncation` sets `T = nlon ÷ 2 − 1`, so `M = T` there. On a
reduced Gaussian target `T` is set by the widest ring, and each narrower ring
drops its wavenumbers above `M`.
"""
ring_max_wavenumber(T::Integer, nlon::Integer) = min(Int(T), (Int(nlon) - 1) ÷ 2)

"""
    fill_ring_spectrum!(buf, spec, P, T, lon_shift_rad) -> buf

Fill `buf` (length `N`, the ring's number of points) with the zonal Fourier
coefficients of one latitude ring, in the layout of an unnormalised inverse FFT
(FFTW `bfft`):

    G_m       = e^{i m λ_0} Σ_{n=m}^{T} f̂_n^m P̃_n^m(sin φ),   0 ≤ m ≤ M,
    buf[1]     = G_0,
    buf[m + 1] = G_m,  buf[N − m + 1] = conj(G_m)              (1 ≤ m ≤ M),

and zero in every other bin, with `M = ring_max_wavenumber(T, N)`. Then
`bfft(buf)[k + 1] = G_0 + 2 Re Σ_{m=1}^{M} G_m e^{2πimk/N} = f(λ_0 + 2πk/N, φ)`.
`spec[n + 1, m + 1] = f̂_n^m` are the spectral coefficients, `P[n + 1, m + 1]`
the Legendre column `P̃_n^m(sin φ)` of the ring's latitude
([`compute_legendre_column!`](@ref)) and `λ_0 = lon_shift_rad` the longitude of
the first point (radians), applied in its own precision.
"""
function fill_ring_spectrum!(buf::AbstractVector{ComplexF64},
                             spec::AbstractMatrix{ComplexF64},
                             P::AbstractMatrix{<:Real},
                             T::Int,
                             lon_shift_rad::Real)
    N = length(buf)
    fill!(buf, zero(ComplexF64))
    for m in 0:ring_max_wavenumber(T, N)
        Gm = zero(ComplexF64)
        @inbounds for n in m:T
            Gm += spec[n + 1, m + 1] * P[n + 1, m + 1]
        end
        if lon_shift_rad != 0 && m > 0
            Gm *= exp(im * m * lon_shift_rad)
        end
        @inbounds buf[m + 1] = Gm
        if m > 0
            @inbounds buf[N - m + 1] = conj(Gm)
        end
    end
    return buf
end

"""
    _delta(m, n)

IFS/TM5 helper coefficient used in the vorticity/divergence to wind transform.
"""
@inline function _delta(m::Int, n::Int)
    n == 0 && return 0.0
    n2 = Float64(n * n)
    m2 = Float64(m * m)
    return -sqrt((n2 - m2) / (4n2 - 1)) / n
end

"""
    _sigma(m, n)

IFS/TM5 helper coefficient used in the vorticity/divergence to wind transform.
"""
@inline function _sigma(m::Int, n::Int)
    (n == 0 || m == 0) && return 0.0
    return -Float64(m) / (Float64(n) * (n + 1))
end

"""
    vod2uv!(u_spec, v_spec, vo_spec, d_spec, T)

Convert one level of spectral relative vorticity (VO) and divergence (D) into
spectral zonal (U) and meridional (V) wind coefficients using the IFS/TM5
reconstruction formula (IFS Doc Part III, Eq. 2.14-2.15):

    Û_n^m = R_E · [ δ(m,n)·VO_{n-1}^m + i·σ(m,n)·D_n^m − δ(m,n+1)·VO_{n+1}^m ]
    V̂_n^m = R_E · [−δ(m,n)·D_{n-1}^m + i·σ(m,n)·VO_n^m + δ(m,n+1)·D_{n+1}^m ]

where `R_E` is the Earth radius [m], `δ(m,n) = −√((n²−m²)/(4n²−1)) / n`
is the coupling coefficient, and `σ(m,n) = −m / (n(n+1))` is the asymmetric
rotation coefficient. `i` is the imaginary unit.

The output `u_spec`, `v_spec` are the spectral coefficients of `U·cos(φ)` and
`V·cos(φ)` respectively (the scaled pseudo-winds used by ECMWF), so the
gridpoint synthesis via `spectral_to_grid!` produces the cos-scaled wind.
Division by cos(φ) to obtain the physical wind is done later in
`compute_mass_fluxes!`.

Input arrays are `(T+1) × (T+1)` 1-indexed: `spec[n+1, m+1] = Ŝ_n^m`.
"""
function vod2uv!(u_spec::Matrix{ComplexF64}, v_spec::Matrix{ComplexF64},
                 vo_spec::AbstractMatrix{ComplexF64}, d_spec::AbstractMatrix{ComplexF64},
                 T::Int)
    fill!(u_spec, zero(ComplexF64))
    fill!(v_spec, zero(ComplexF64))

    for m in 0:T
        for n in m:T
            delta_mn = _delta(m, n)       # δ(m, n)
            delta_mn1 = _delta(m, n + 1)  # δ(m, n+1)
            sigma_mn = _sigma(m, n)       # σ(m, n) = -m/(n(n+1))

            # Spectral neighbours (zero outside [m, T]):
            vo_nm1 = n > m ? vo_spec[n, m + 1] : zero(ComplexF64)      # VO_{n-1}^m
            vo_np1 = n < T ? vo_spec[n + 2, m + 1] : zero(ComplexF64)  # VO_{n+1}^m
            d_nm1 = n > m ? d_spec[n, m + 1] : zero(ComplexF64)        # D_{n-1}^m
            d_np1 = n < T ? d_spec[n + 2, m + 1] : zero(ComplexF64)    # D_{n+1}^m
            vo_n = vo_spec[n + 1, m + 1]  # VO_n^m
            d_n = d_spec[n + 1, m + 1]    # D_n^m

            # IFS Eq 2.14: Û = R_E · (δ·VO_{n-1} + i·σ·D_n − δ_{n+1}·VO_{n+1})
            u_spec[n + 1, m + 1] = IFS_EARTH_RADIUS * (delta_mn * vo_nm1 + im * sigma_mn * d_n - delta_mn1 * vo_np1)
            # IFS Eq 2.15: V̂ = R_E · (−δ·D_{n-1} + i·σ·VO_n + δ_{n+1}·D_{n+1})
            v_spec[n + 1, m + 1] = IFS_EARTH_RADIUS * (-delta_mn * d_nm1 + im * sigma_mn * vo_n + delta_mn1 * d_np1)
        end
    end
    return nothing
end

"""
    spectral_to_grid!(field, spec, T, lats, Nlon, P_buf, fft_buf; lon_shift_rad=0.0, ...)

Synthesize a single 2D spectral field onto a regular lat-lon grid.

For each latitude `φ ∈ lats`:
1. Build the associated Legendre column `P̃_n^m(sin φ)`.
2. Sum over `n` for each `m` to get the zonal harmonic coefficient
   `G_m(φ) = Σ_n spec[n+1, m+1] · P̃_n^m(sin φ)`.
3. Apply longitude phase shift: `G_m *= exp(i · m · lon_shift_rad)`.
   This shifts the output so index 1 represents the field at physical
   longitude `lon_shift_rad` (radians). Use `lon_shift_rad = deg2rad(λᶜ[1])`
   to align with a mesh whose first cell center is at `λᶜ[1]`.
4. Fill negative-frequency bins via conjugate symmetry (real-valued output).
5. Inverse FFT → gridpoint values along the longitude ring.

Steps 2–4 are [`fill_ring_spectrum!`](@ref), shared with the reduced Gaussian
ring synthesis `spectral_to_ring!`.

**Longitude convention**: the FFT's natural origin is lon=0°. Without a
shift, `field[1, j]` is at lon=0°. With `lon_shift_rad = deg2rad(-178.125)`,
`field[1, j]` is at lon=-178.125° (first cell center of a `[-180, 180)` mesh).

**Truncation**: wavenumbers `0 ≤ m ≤ ring_max_wavenumber(T, Nlon) =
min(T, (Nlon − 1) ÷ 2)` are synthesised, each `m > 0` with full weight; the
term `m = Nlon/2` of an even `Nlon` is dropped (see
[`ring_max_wavenumber`](@ref)). Lat-lon targets use `T = Nlon ÷ 2 − 1`, so
nothing is dropped there.
"""
function spectral_to_grid!(field::AbstractMatrix{<:Real},
                           spec::AbstractMatrix{ComplexF64},
                           T::Int,
                           lats::AbstractVector{<:Real},
                           Nlon::Int,
                           P_buf::AbstractMatrix{<:Real},
                           fft_buf::AbstractVector{ComplexF64};
                           fft_out::Union{Nothing, AbstractVector{ComplexF64}}=nothing,
                           bfft_plan=nothing,
                           lon_shift_rad::Real=0.0)
    length(fft_buf) == Nlon ||
        throw(DimensionMismatch("FFT buffer length $(length(fft_buf)) ≠ Nlon = $Nlon"))

    for j in eachindex(lats)
        sin_lat = sind(lats[j])
        compute_legendre_column!(P_buf, T, sin_lat)

        # Steps 1–3: G_m(φ) = Σ_n Ŝ_n^m P̃_n^m(sin φ), phase-shifted to
        # lon_shift_rad, in bins m and Nlon − m for 0 ≤ m ≤ ring_max_wavenumber.
        fill_ring_spectrum!(fft_buf, spec, P_buf, T, lon_shift_rad)

        # Step 4: Backward (unnormalised) FFT → gridpoint values along ring.
        # FFTW `bfft` computes Σ_m G_m · exp(2πi·m·k/N) for k=0..N-1, which
        # is the Fourier synthesis WITHOUT the 1/N normalisation (matching
        # ECMWF's spectral convention where forward coefficients already
        # contain the normalisation).
        f_lon = if bfft_plan !== nothing && fft_out !== nothing
            mul!(fft_out, bfft_plan, fft_buf)
            fft_out
        else
            bfft(fft_buf)
        end

        @inbounds for i in 1:Nlon
            field[i, j] = real(f_lon[i])
        end
    end
    return nothing
end

"""
    stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nlon, Nlat, Nz)

Map spectral-synthesised winds onto the Arakawa C-grid staggered positions:

- `u_stag[i, j, k]`: zonal wind at the **west face** of cell `(i, j)`.
  For i=1..Nlon, copied from `u_cc` (which is synthesised at west-edge
  longitudes via `spectral_to_grid!` with the face shift). Index
  `Nlon+1` wraps periodically to `u_cc[1, j, k]`.
- `v_stag[i, j, k]`: meridional wind at the **south face** of cell `(i, j)`.
  Averaged from `v_cc` at cell centers j-1 and j. Pole boundaries (j=1,
  j=Nlat+1) are zeroed (no meridional mass flux through the poles).

Stagger convention: `am[i, j, k] = u_stag[i, j, k] × dp_face × ...` is
the eastward mass flux through the west face of cell `(i, j)`. Positive
flux = mass moving eastward.
"""
function stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nlon, Nlat, Nz)
    # u_stag at west edges — direct copy since u_cc is already synthesised there
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        u_stag[i, j, k] = u_cc[i, j, k]
    end
    # Periodic wrap: east face of last cell = west face of first cell
    @inbounds for k in 1:Nz, j in 1:Nlat
        u_stag[Nlon + 1, j, k] = u_cc[1, j, k]
    end

    # v_stag at south edges — average v_cc at j-1 and j (cell centers)
    @inbounds for k in 1:Nz, j in 2:Nlat, i in 1:Nlon
        v_stag[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end
    # No meridional flux at the poles (closed latitude boundaries)
    v_stag[:, 1, :] .= 0       # south pole
    v_stag[:, Nlat + 1, :] .= 0  # north pole
    return nothing
end

"""
    compute_dp!(dp, ps, dA, dB, Nlon, Nlat, Nz)

Compute native-layer pressure thickness [Pa] from hybrid A/B coefficients
and surface pressure. Uses `|dA[k] + dB[k] × ps|` (absolute value handles
either level-ordering convention).

Vertical convention: `k=1` is TOA, `k=Nz` is the surface-adjacent level.
`dA[k] = A_ifc[k+1] - A_ifc[k]` and `dB[k] = B_ifc[k+1] - B_ifc[k]`
where `A_ifc` [Pa] and `B_ifc` [dimensionless] define the half-level
pressures: `p_half[k] = A_ifc[k] + B_ifc[k] × ps`.
"""
function compute_dp!(dp, ps, dA, dB, Nlon, Nlat, Nz)
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        dp[i, j, k] = abs(dA[k] + dB[k] * ps[i, j])
    end
    return nothing
end

"""
    compute_air_mass!(m_arr, dp, area, Nlon, Nlat, Nz)

Convert pressure thickness to air mass per cell [kg]:

    m[i, j, k] = dp[i, j, k] × area[i, j] / g

where `area[i, j]` is the horizontal cell area [m²] and `g` is the
gravitational acceleration [m/s²].
"""
function compute_air_mass!(m_arr, dp, area, Nlon, Nlat, Nz)
    inv_g = 1.0 / STANDARD_GRAVITY  # [s²/m]
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        m_arr[i, j, k] = dp[i, j, k] * area[i, j] * inv_g  # [Pa × m² × s²/m] = [kg]
    end
    return nothing
end

"""
    compute_mass_fluxes!(am, bm, u_stag, v_stag, ps, dA, dB, grid, half_dt, Nz)

Compute the native-level horizontal mass fluxes `am` and `bm`.

## Horizontal fluxes

Zonal mass flux through the west face of cell `(i, j)`:

    am[i, j, k] = (u / cos φ) × Δp_face × (R / g) × Δφ × Δt/2   [kg]

where `u` is the spectral pseudo-wind `U·cos(φ)`, so `u / cos φ` recovers the
physical zonal wind. `Δp_face`
is the Jensen-corrected face-level pressure thickness
`|dA + dB × exp((ln(ps_L) + ln(ps_R)) / 2)|`, and `Δt/2 = half_dt` is the
v4 binary's per-substep flux convention (Strang splitting applies fluxes
twice per full step: forward + reverse).

Meridional mass flux through the south face of cell `(i, j)`:

    bm[i, j, k] = v × Δp_face × (R / g) × Δλ × Δt/2   [kg]

where `v` is the spectral pseudo-wind `V·cos(φ)`. The east-west face length's
`cos(φ)` factor is therefore already present in `v`, so there is no additional
explicit latitude-cosine multiplier in the `bm` formula.

Sign convention: positive `am` = eastward mass flux; positive `bm` = northward.

The pole rows `j = 1` and `j = Nlat` are ordinary cells, centred at
`±(90° − Δφ/2)`: their zonal faces carry the synthesized flux like every other
row. Only the meridional faces on the poles (`bm[:, 1, :]`, `bm[:, Nlat+1, :]`,
at `φ = ±90°`) have zero length and carry no flux. Closing the pole-row zonal
faces would leave a flow that crosses the pole no path through the polar cap;
the column balance would then re-route `bm` at every latitude (issue #58; the
same zeroing was removed once before, see
`docs/resources/bug_archive/SESSION1_GRID_CM_FIX_2026-04-03.md`, section 3).

The stored vertical flux is diagnosed after column balance from the explicit
forward mass endpoint; no native vertical staging estimate is needed.
"""
function compute_mass_fluxes!(am, bm, u_stag, v_stag, ps,
                              dA, dB, grid::LatLonTargetGeometry, half_dt, Nz)
    Nlon, Nlat = nlon(grid), nlat(grid)
    # The spectral wind synthesis above uses the IFS radius because vorticity
    # and divergence are defined on the physical IFS sphere.  These face
    # integrals, however, belong to the output mesh and must use its recorded
    # radius so their areas agree with `planet_radius_m` at runtime.
    R_g = grid.mesh.radius / STANDARD_GRAVITY   # [m × s²/m] = [s²/m²] — converts dp × R / g to mass/area
    dlon = grid.dlon        # [radians] — longitudinal grid spacing
    dlat = grid.dlat        # [radians] — latitudinal grid spacing
    # --- Zonal mass flux: am[i, j, k] at west face of cell (i, j) ---
    # Positive am = eastward. Every row carries its flux, the pole rows
    # included: they are cells centred at ±(90° − Δφ/2), not pole points.
    # Do NOT zero am in rows 1 and Nlat (issue #58; bug archive
    # SESSION1_GRID_CM_FIX_2026-04-03.md, section 3); the regression test is
    # the pole-row testsets of test/core/test_spectral_targets.jl.
    # Face ps uses Jensen correction: exp(mean(ln ps)) instead of mean(exp(ln ps)).
    # log(ps) is computed inline to avoid allocating a scratch matrix.
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:(Nlon + 1)
        i_l = i == 1 ? Nlon : i - 1   # periodic wrap: west neighbour of i=1 is Nlon
        i_r = i <= Nlon ? i : 1        # periodic wrap: east neighbour of Nlon+1 is 1
        ps_face = exp((log(ps[i_l, j]) + log(ps[i_r, j])) / 2)  # Jensen-corrected
        dp_face = abs(dA[k] + dB[k] * ps_face)  # face-level pressure thickness [Pa]
        # u_stag is U·cos(φ) from spectral synthesis; dividing by cos(φ)
        # recovers the physical zonal wind u [m/s].
        # am = u × R/g × Δp × Δφ × Δt/2   [kg per half-step]
        am[i, j, k] = u_stag[i, j, k] / grid.cos_lat[j] * dp_face * R_g * dlat * half_dt
    end

    # --- Meridional mass flux: bm[i, j, k] at south face of cell (i, j) ---
    # Positive bm = northward. Pole boundaries (j=1, j=Nlat+1) are zeroed.
    @inbounds for k in 1:Nz, j in 1:(Nlat + 1), i in 1:Nlon
        if j == 1 || j == Nlat + 1
            bm[i, j, k] = 0  # no meridional flux through poles (closed boundary)
        else
            j_s = j - 1   # cell south of this face
            j_n = j        # cell north of this face
            ps_face = exp((log(ps[i, j_s]) + log(ps[i, j_n])) / 2)
            dp_face = abs(dA[k] + dB[k] * ps_face)
            # v_stag is V·cos(φ) from spectral synthesis. That cos factor is
            # the east-west face-width metric, so no extra cos appears here.
            # bm = v_cos × R/g × Δp × Δλ × Δt/2   [kg per half-step]
            bm[i, j, k] = v_stag[i, j, k] * dp_face * R_g * dlon * half_dt
        end
    end

    return nothing
end

"""
    spectral_to_native_fields!(...)

Transform one analysis hour of ERA5 spectral `lnsp`, `vo`, and `d` into native
gridpoint mass/flux fields on the configured target geometry.

This routine builds only the moist native fields. Dry-basis conversion and any
humidity-aware mass fix are applied later once the matching hourly `q` field is
available.
"""
function spectral_to_native_fields!(
    m_arr::AbstractArray{<:Real, 3}, am::AbstractArray{<:Real, 3}, bm::AbstractArray{<:Real, 3},
    sp::AbstractMatrix{<:Real},
    u_cc::AbstractArray{<:Real, 3}, v_cc::AbstractArray{<:Real, 3},
    u_stag::AbstractArray{<:Real, 3}, v_stag::AbstractArray{<:Real, 3},
    dp::AbstractArray{<:Real, 3},
    lnsp_spec::AbstractMatrix{ComplexF64},
    vo_hour::AbstractArray{ComplexF64, 3}, d_hour::AbstractArray{ComplexF64, 3},
    T::Int, level_range::UnitRange{Int}, ab,
    grid::LatLonTargetGeometry, half_dt::Real,
    P_buf::AbstractMatrix{<:Real}, fft_buf::AbstractVector{ComplexF64},
    field_2d::AbstractMatrix{<:Real},
    P_buf_t, fft_buf_t, fft_out_t, u_spec_t, v_spec_t, field_2d_t,
    bfft_plans)

    Nlon = nlon(grid)
    Nlat = nlat(grid)
    Nz = length(level_range)

    # Compute spectral synthesis longitude phase shifts FROM THE MESH
    # coordinates, not from `dlon / 2`. The old `center_shift = dlon/2`
    # only aligned with a `longitude=(0,360)` mesh where the first cell
    # center is at `1.875°`. For the default `longitude=(-180,180)` mesh
    # the first cell center is at `-178.125°` and the first west face
    # is at `-180°`, so `spectral_to_grid!` needs a negative shift to
    # move the FFT's natural origin (physical lon = 0°) to the mesh's
    # first point. Prior bug: every LL v4 binary was 180° off in
    # longitude relative to its stored `λᶜ`. See 2026-04-11 AGENT_CHAT
    # entry for full diagnosis.
    #
    # `spectral_to_grid!` applies `Gm *= exp(im * m * lon_shift_rad)`,
    # which makes FFT output index 1 represent the field at physical
    # lon = `lon_shift_rad`. So `lon_shift_rad = deg2rad(λᶜ[1])` for
    # cell-centered scalars (sp, v_cc) and `deg2rad(λᶠ[1])` for the
    # west-face-located u_cc.
    sp_shift = deg2rad(grid.lons[1])
    u_edge_shift = deg2rad(first(grid.mesh.λᶠ))

    spectral_to_grid!(field_2d, lnsp_spec, T, grid.lats, Nlon, P_buf, fft_buf;
                      fft_out=fft_out_t[1], bfft_plan=bfft_plans[1],
                      lon_shift_rad=sp_shift)
    @. sp = exp(field_2d)

    Threads.@threads for k in 1:Nz
        level = level_range[k]
        tid = Threads.threadid()

        vod2uv!(u_spec_t[tid], v_spec_t[tid],
                @view(vo_hour[:, :, level]),
                @view(d_hour[:, :, level]),
                T)

        # u_cc is located at the west face in longitude (cell edges)
        spectral_to_grid!(field_2d_t[tid], u_spec_t[tid], T,
                          grid.lats, Nlon, P_buf_t[tid], fft_buf_t[tid];
                          fft_out=fft_out_t[tid], bfft_plan=bfft_plans[tid],
                          lon_shift_rad=u_edge_shift)
        u_cc[:, :, k] .= field_2d_t[tid]

        # v_cc is located at cell center in longitude
        spectral_to_grid!(field_2d_t[tid], v_spec_t[tid], T,
                          grid.lats, Nlon, P_buf_t[tid], fft_buf_t[tid];
                          fft_out=fft_out_t[tid], bfft_plan=bfft_plans[tid],
                          lon_shift_rad=sp_shift)
        v_cc[:, :, k] .= field_2d_t[tid]
    end

    stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nlon, Nlat, Nz)
    compute_dp!(dp, sp, ab.dA, ab.dB, Nlon, Nlat, Nz)
    compute_air_mass!(m_arr, dp, grid.area, Nlon, Nlat, Nz)
    compute_mass_fluxes!(am, bm, u_stag, v_stag, sp,
                         ab.dA, ab.dB, grid, half_dt, Nz)

    return nothing
end

"""
    recompute_native_mass_fields!(transform, ab, grid, half_dt)

Recompute native `dp`, moist cell mass, and mass fluxes after a uniform
surface-pressure adjustment has been applied to `transform.sp`.
"""
function recompute_native_mass_fields!(transform,
                                       ab,
                                       grid::LatLonTargetGeometry,
                                       half_dt::Real)
    Nlon = nlon(grid)
    Nlat = nlat(grid)
    Nz = size(transform.m_arr, 3)

    compute_dp!(transform.dp, transform.sp, ab.dA, ab.dB, Nlon, Nlat, Nz)
    compute_air_mass!(transform.m_arr, transform.dp, grid.area, Nlon, Nlat, Nz)
    compute_mass_fluxes!(transform.am_arr, transform.bm_arr,
                         transform.u_stag, transform.v_stag, transform.sp,
                         ab.dA, ab.dB, grid, half_dt, Nz)
    return nothing
end
