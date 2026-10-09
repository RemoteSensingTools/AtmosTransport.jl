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

**Longitude convention**: the FFT's natural origin is lon=0°. Without a
shift, `field[1, j]` is at lon=0°. With `lon_shift_rad = deg2rad(-178.125)`,
`field[1, j]` is at lon=-178.125° (first cell center of a `[-180, 180)` mesh).

**Truncation**: only wavenumbers `m ∈ [0, min(T, Nlon/2)]` are retained to
avoid aliasing above the Nyquist frequency.
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
    Nfft = Nlon  # FFT length = number of longitude grid points

    for j in eachindex(lats)
        sin_lat = sind(lats[j])
        compute_legendre_column!(P_buf, T, sin_lat)

        # Step 1: Sum over total wavenumber n for each zonal wavenumber m
        # to get the latitude-dependent harmonic coefficient G_m(φ).
        # Only m ∈ [0, Nlon/2] is retained (Nyquist truncation).
        fill!(fft_buf, zero(ComplexF64))
        for m in 0:min(T, div(Nfft, 2))
            Gm = zero(ComplexF64)
            @inbounds for n in m:T
                # G_m += Ŝ_n^m · P̃_n^m(sin φ)
                Gm += spec[n + 1, m + 1] * P_buf[n + 1, m + 1]
            end
            # Step 2: Apply longitude phase shift. Multiplying by exp(i·m·Δλ)
            # shifts the output grid so index 1 represents lon = lon_shift_rad.
            if lon_shift_rad != 0.0 && m > 0
                Gm *= exp(im * m * lon_shift_rad)
            end
            fft_buf[m + 1] = Gm   # 1-indexed: fft_buf[1] = m=0 (DC)
        end

        # Step 3: Fill negative-frequency bins via conjugate symmetry
        # (real-valued field ⇒ G_{−m} = conj(G_m)).
        for m in 1:min(T, div(Nfft, 2) - 1)
            fft_buf[Nfft - m + 1] = conj(fft_buf[m + 1])
        end

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
    compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, ps, dA, dB, grid, half_dt, Nz)

Compute the native-level horizontal mass fluxes `am`, `bm` and diagnose the
native-level vertical flux `cm` from discrete continuity.

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

Pole rows `j=1` and `j=Nlat` have `am = 0` (no zonal transport through poles).

## Vertical flux

`cm[i, j, k]` is diagnosed from the discrete continuity equation per column,
with a B-coefficient redistribution term for the hybrid coordinate:

    cm[k+1] = −Σ_{l=1}^{k} div_h[l] + B[k] × Σ_{l=1}^{Nz} div_h[l]

where `div_h = (am[i] − am[i+1]) + (bm[j] − bm[j+1])` is the horizontal
convergence. `cm[1] = 0` (TOA) and `cm[Nz+1] = 0` (surface) by construction.

These raw `cm` values are only a staging estimate. Before writing a transport
binary, the preprocessing path applies a column-integrated mass correction to
the horizontal fluxes and diagnoses the stored `cm` from the explicit forward
mass endpoint. The legacy per-layer horizontal Poisson correction is available
only as an opt-in diagnostic mode because it can distort physical horizontal
winds when no independent vertical mass flux is supplied.
"""
function compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, ps,
                              dA, dB, grid::LatLonTargetGeometry, half_dt, Nz)
    Nlon, Nlat = nlon(grid), nlat(grid)
    R_g = IFS_EARTH_RADIUS / STANDARD_GRAVITY   # [m × s²/m] = [s²/m²] — converts dp × R / g to mass/area
    dlon = grid.dlon        # [radians] — longitudinal grid spacing
    dlat = grid.dlat        # [radians] — latitudinal grid spacing
    # --- Zonal mass flux: am[i, j, k] at west face of cell (i, j) ---
    # Positive am = eastward. Pole rows (j=1, j=Nlat) are zeroed.
    # Face ps uses Jensen correction: exp(mean(ln ps)) instead of mean(exp(ln ps)).
    # log(ps) is computed inline to avoid allocating a scratch matrix.
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:(Nlon + 1)
        i_l = i == 1 ? Nlon : i - 1   # periodic wrap: west neighbour of i=1 is Nlon
        i_r = i <= Nlon ? i : 1        # periodic wrap: east neighbour of Nlon+1 is 1
        ps_face = exp((log(ps[i_l, j]) + log(ps[i_r, j])) / 2)  # Jensen-corrected
        dp_face = abs(dA[k] + dB[k] * ps_face)  # face-level pressure thickness [Pa]
        cos_lat = grid.cos_lat[j]
        if j == 1 || j == Nlat
            am[i, j, k] = 0  # no zonal flux at pole rows
        else
            # u_stag is U·cos(φ) from spectral synthesis; dividing by cos(φ)
            # recovers the physical zonal wind u [m/s].
            # am = u × R/g × Δp × Δφ × Δt/2   [kg per half-step]
            am[i, j, k] = u_stag[i, j, k] / cos_lat * dp_face * R_g * dlat * half_dt
        end
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

    # --- Vertical flux cm: diagnosed from horizontal divergence via continuity ---
    # cm[i, j, 1] = 0 (TOA boundary) and cm[i, j, Nz+1] = 0 (surface boundary).
    fill!(cm, zero(eltype(cm)))
    @inbounds for j in 1:Nlat, i in 1:Nlon
        # Total column horizontal convergence (sum over all levels):
        # pit = Σ_k div_h[k] = Σ_k [(am_east - am_west) + (bm_north - bm_south)]
        pit = 0.0
        for k in 1:Nz
            pit += (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                   (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
        end
        # Accumulate cm from top (k=1) downward. At each interface k+1:
        #   cm[k+1] = cm[k] − div_h[k] + dB[k] × pit
        # where dB[k] = B[k+1] − B[k] redistributes the total-column
        # divergence into the hybrid coordinate's ps-dependent component.
        # This ensures cm[1] = 0 (TOA) and cm[Nz+1] ≈ 0 (surface) when
        # the column-integrated horizontal fluxes are balanced.
        acc = 0.0
        for k in 1:Nz
            div_h = (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                    (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
            acc = acc - div_h + Float64(dB[k]) * pit
            cm[i, j, k + 1] = eltype(cm)(acc)
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
    cm::AbstractArray{<:Real, 3}, sp::AbstractMatrix{<:Real},
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
    compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, sp,
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
    compute_mass_fluxes!(transform.am_arr, transform.bm_arr, transform.cm_arr,
                         transform.u_stag, transform.v_stag, transform.dp, transform.sp,
                         ab.dA, ab.dB, grid, half_dt, Nz)
    return nothing
end
