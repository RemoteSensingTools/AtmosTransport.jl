# ===========================================================================
# TM5 boundary-layer diffusion (`bldiff`) — column eddy-diffusivity kernel.
#
# This is a faithful port of TM5's `bldiff` subroutine
# (`deps/tm5-cy3-4dvar/base/src/diffusion.F90`), the Holtslag & Boville (1993,
# J. Climate 6, 1825-1842) non-local boundary-layer scheme. It is the diffusion
# TM5 actually uses for ERA5-driven runs, and the reason neither of our existing
# runtime Kz fields reproduces TM5 (they are local schemes; this one carries the
# non-local counter-gradient and a prescribed entrainment flux at the PBL top).
#
# Physics, in the order the column kernel evaluates it:
#
#   1. Thermodynamic profile — potential temperature θ, virtual potential
#      temperature θ_v, and layer-centre heights.
#   2. Free-troposphere diffusivity `Kv_free` — a local Louis-type scheme with a
#      height-dependent asymptotic mixing length, driven by the gradient
#      Richardson number Ri = (static stability) / (wind shear)².
#   3. Boundary-layer height `h` — a bulk-Richardson parcel method
#      (Vogelezang & Holtslag 1996), evaluated twice: once neutrally, then again
#      with a convective temperature excess under unstable conditions.
#   4. Surface fluxes — the sensible and latent surface heat fluxes (W m⁻²,
#      upward-positive) are converted to kinematic form and combined into the
#      virtual heat flux that sets the Monin-Obukhov length L.
#   5. Eddy diffusivity `Kvh` — stable/neutral air uses the local Louis form;
#      the unstable surface layer and mixed layer use convective scaling with a
#      non-local Prandtl-number enhancement; the entrainment layer at the PBL
#      top is overwritten with a prescribed flux.
#
# Output convention: `Kvh[l]` is the heat diffusivity (m² s⁻¹) at the **top** of
# layer `l`, i.e. the interface between layers `l` and `l+1`. `Kvh[Nz] = 0`
# (no flux through the model top).
#
# Vertical convention: this kernel takes **bottom-up** columns (index 1 = the
# surface layer, index `Nz` = the model top), matching the Fortran. Callers that
# hold top-down (k=1 = TOA) profiles flip them at the boundary; see
# `tm5_bldiff_kvh!`.
# ===========================================================================

"""
    BLDiffConstants{FT}

Physical constants for the TM5 boundary-layer diffusion scheme. Defaults
reproduce TM5's `binas` module and the hard-coded `bldiff` parameters; they are
exposed as a struct so tests can probe sensitivities without editing the kernel.

Thermodynamic constants

  * `grav`    — gravitational acceleration (m s⁻²)
  * `cp_air`  — specific heat of dry air at constant pressure (J kg⁻¹ K⁻¹)
  * `r_air`   — specific gas constant for dry air (J kg⁻¹ K⁻¹)
  * `r_vap`   — specific gas constant for water vapour (J kg⁻¹ K⁻¹)
  * `l_vap`   — latent heat of vaporisation (J kg⁻¹)
  * `vkarman` — von Kármán constant
  * `p_ref`   — reference pressure for potential temperature (Pa)

Derived: `kappa = r_air / cp_air`, `vt_coef = r_vap/r_air - 1` (the 0.61 in the
virtual-temperature correction).

Scheme parameters (Holtslag & Boville constants)

  * `ri_crit`     — critical bulk Richardson number defining the PBL top
  * `sf_frac`     — surface-layer fraction of the PBL depth
  * `beta_m/_h`   — stability-function slopes for momentum/heat
  * `fak`, `fakn` — convective excess factor and counter-gradient factor
  * `ustar_shear` — coefficient of the ustar shear term in the bulk Richardson
                    number (`zacb` in TM5)
  * `kvh_min`     — floor on the in-PBL diffusivity (m² s⁻¹)
  * `kv_free_min` — floor on the free-troposphere diffusivity (m² s⁻¹)
  * `pblh_min`    — floor on the diagnosed PBL height (m)
"""
Base.@kwdef struct BLDiffConstants{FT <: AbstractFloat}
    grav        :: FT = TM5_CONSTANTS.gravity
    cp_air      :: FT = TM5_CONSTANTS.cp_air
    r_air       :: FT = TM5_CONSTANTS.r_air      # = Rgas·1000 / M_air = 8.3144·1000 / 28.94 (TM5 binas)
    r_vap       :: FT = TM5_CONSTANTS.r_vap
    l_vap       :: FT = TM5_CONSTANTS.l_vap
    vkarman     :: FT = TM5_CONSTANTS.karman
    p_ref       :: FT = TM5_CONSTANTS.p_ref
    ri_crit     :: FT = 0.3
    sf_frac     :: FT = 0.1
    beta_m      :: FT = 15.0
    beta_h      :: FT = 15.0
    fak         :: FT = 8.5
    fakn        :: FT = 7.2
    ustar_shear :: FT = 100.0
    kvh_min     :: FT = 0.1
    kv_free_min :: FT = 1.0e-15
    pblh_min    :: FT = 100.0
    # Validity bounds on the prescribed-entrainment override `K = 0.2·w_heatv/dθv`
    # at the PBL top. The formula assumes a STABLE cap (dθv/dz > 0); it is applied
    # only when dθv is meaningfully positive AND the resulting K is finite and
    # within `[·, kvh_max]`. Outside that envelope (vanishing cap dθv→0 → +Inf;
    # non-stable cap dθv ≤ 0 → negative; or an absurdly large K) the column keeps
    # its regular in-PBL mixed-layer diffusivity instead — see the entrainment
    # branch in `tm5_bldiff_kvh_column!`.
    dthv_entr_min :: FT = 1.0e-4   # K m⁻¹: floor on a "meaningful" inversion
    kvh_max       :: FT = 1.0e3    # m² s⁻¹: physical ceiling on a sane diffusivity
end

"""
    BLDiffDiag{FT}()

Per-thread counters for the boundary-layer diffusion fallbacks, accumulated by
[`tm5_bldiff_kvh_column!`] over the columns of one window so the preprocessor can
log how often (and how hard) the entrainment override was skipped or the output
guard fired — a few stray cells is expected; widespread fallback means a met
problem to investigate, not silently fill.
"""
mutable struct BLDiffDiag{FT <: AbstractFloat}
    entr_fallback :: Int   # entrainment override skipped (cap absent/non-stable/absurd)
    kvh_floored   :: Int   # final output guard tripped (non-finite Kvh → kvh_min)
    max_kz        :: FT    # max diffusivity written (range sanity check)
end
BLDiffDiag{FT}() where {FT} = BLDiffDiag{FT}(0, 0, zero(FT))
@inline _reset!(d::BLDiffDiag{FT}) where {FT} =
    (d.entr_fallback = 0; d.kvh_floored = 0; d.max_kz = zero(FT); d)

# TM5's branchless step function `jqif`: 1 if a ≥ b, else 0. Kept as integer
# masks (rather than `if`s) so the multi-branch diffusivity expressions read the
# same as the reference and stay branch-predictable. The `≥` (not `>`) matches
# TM5's `nint(0.5 + sign(0.5, a-b))`, which yields 1 at exact equality — the
# only case where `>` would diverge from the Fortran.
@inline _step(a::T, b::T) where {T <: AbstractFloat} = a >= b ? one(T) : zero(T)
@inline _step(a, b) = a >= b ? 1 : 0

"""
    tm5_bldiff_kvh_column!(kvh, T, q, u, v, p_edge, z_edge,
                           hflux, lhflux, ustar, c) -> pblh

Compute the TM5 boundary-layer heat diffusivity profile for one atmospheric
column and return the diagnosed PBL height (m).

All profile inputs are **bottom-up** (index 1 = surface layer). Layer-centre
quantities have length `Nz`; half-level (interface) quantities have length
`Nz + 1`, with index 1 at the surface and index `Nz + 1` at the model top.

Arguments

  * `kvh`    — output, length `Nz`; `kvh[l]` is the diffusivity (m² s⁻¹) at the
               top of layer `l`. `kvh[Nz]` is set to 0.
  * `T`      — layer-centre temperature (K)
  * `q`      — layer-centre specific humidity (kg kg⁻¹)
  * `u`, `v` — layer-centre wind components (m s⁻¹)
  * `p_edge` — half-level pressure (Pa), length `Nz + 1`
  * `z_edge` — half-level geopotential height above the surface (m), length
               `Nz + 1`, with `z_edge[1] = 0`
  * `hflux`  — surface sensible heat flux (W m⁻², upward-positive)
  * `lhflux` — surface latent heat flux (W m⁻², upward-positive)
  * `ustar`  — friction velocity (m s⁻¹)
  * `c`      — [`BLDiffConstants`](@ref)

The scalar surface fluxes already use the upward-positive convention produced by
the ERA5 surface reader, so — unlike the Fortran, which flips ERA5's
downward-positive sign — no sign flip is applied here.
"""
function tm5_bldiff_kvh_column!(kvh::AbstractVector{FT},
                                T::AbstractVector{FT},
                                q::AbstractVector{FT},
                                u::AbstractVector{FT},
                                v::AbstractVector{FT},
                                p_edge::AbstractVector{FT},
                                z_edge::AbstractVector{FT},
                                hflux::FT,
                                lhflux::FT,
                                ustar::FT,
                                c::BLDiffConstants{FT};
                                diag::Union{Nothing, BLDiffDiag{FT}} = nothing) where {FT}
    Nz = length(T)
    @assert length(kvh) == Nz
    @assert length(p_edge) == Nz + 1 && length(z_edge) == Nz + 1

    kappa    = c.r_air / c.cp_air
    vt_coef  = c.r_vap / c.r_air - one(FT)     # ≈ 0.61 (virtual-temperature)
    ccon     = c.sf_frac * c.vkarman
    onethird = one(FT) / 3

    # -- 1. Thermodynamic profile -----------------------------------------
    # θ, θ_v at layer centres; layer-centre height interpolated from the
    # half-level heights linearly in pressure.
    θ   = similar(T)
    θv  = similar(T)
    zc  = similar(T)          # layer-centre height above surface (m)
    @inbounds for l in 1:Nz
        p_mid  = (p_edge[l] + p_edge[l + 1]) / 2
        θ[l]   = T[l] * (c.p_ref / p_mid)^kappa
        θv[l]  = θ[l] * (one(FT) + vt_coef * q[l])
        # Linear-in-pressure weight of the upper edge within the layer.
        w      = (p_edge[l] - p_mid) / (p_edge[l] - p_edge[l + 1])
        zc[l]  = z_edge[l] * (one(FT) - w) + z_edge[l + 1] * w
    end

    # -- 2. Free-troposphere diffusivity & gradient Richardson number ------
    # Evaluated at each interior interface l (top of layer l). `Kv_free`
    # and `Ri` are stored per layer; index Nz is unused (no interface above).
    Kv_free = similar(T); fill!(Kv_free, zero(FT))
    Ri      = similar(T); fill!(Ri, zero(FT))
    shear2  = similar(T); fill!(shear2, zero(FT))
    @inbounds for l in 1:Nz-1
        z_iface = z_edge[l + 1]                       # height of this interface
        # Height-dependent asymptotic mixing length: 300 m near the surface,
        # relaxing to ~30 m aloft (Holtslag & Boville).
        below = _step(FT(1000), z_iface)
        λc = below * FT(300) +
             (1 - below) * (FT(30) + FT(270) * exp(one(FT) - z_iface / 1000))
        mix2 = (one(FT) / (one(FT) / λc + one(FT) / (c.vkarman * z_iface)))^2

        # Wind shear squared across the interface (floored to avoid /0).
        dz = zc[l + 1] - zc[l]
        du2 = (u[l + 1] - u[l])^2 + (v[l + 1] - v[l])^2
        s2 = max(du2, FT(1e-10)) / dz^2
        shear2[l] = s2

        # Static stability from θ_v interpolated to the interface (linear in
        # log-pressure), then the gradient Richardson number.
        p_lo = (p_edge[l]     + p_edge[l + 1]) / 2
        p_hi = (p_edge[l + 1] + p_edge[l + 2]) / 2
        arg  = (log(p_edge[l + 1]) - log(p_lo)) / (log(p_hi) - log(p_lo))
        θv_iface = θv[l] + arg * (θv[l + 1] - θv[l])
        stab = c.grav / θv_iface * (θv[l + 1] - θv[l]) / dz
        Ri[l] = stab / s2

        # Louis stability functions: stable (Ri>0) vs unstable (Ri<0) branch.
        f_unstable = sqrt(max(one(FT) - 18 * Ri[l], zero(FT)))
        f_stable   = one(FT) / (one(FT) + 10 * Ri[l] * (one(FT) + 8 * Ri[l]))
        Kv_neutral = mix2 * sqrt(s2)
        stableq = _step(Ri[l], zero(FT))
        Kv_free[l] = stableq * max(c.kv_free_min, Kv_neutral * f_stable) +
                     (1 - stableq) * max(c.kv_free_min, Kv_neutral * f_unstable)
    end

    # -- 3. Surface fluxes → virtual heat flux & Monin-Obukhov length ------
    # Kinematic conversion W m⁻² → (K m s⁻¹) and (m s⁻¹) using the lowest
    # layer's density ρ = Δp / (g Δz); the surface fluxes are already
    # upward-positive, so no sign flip (cf. the Fortran).
    ρ_surf = (p_edge[1] - p_edge[2]) / (c.grav * (z_edge[2] - z_edge[1]))
    w_heat = hflux  / (ρ_surf * c.cp_air)         # K m s⁻¹
    w_qflx = lhflux / (ρ_surf * c.l_vap)          # m s⁻¹ (kinematic moisture)

    θv_surf = θ[1] * (one(FT) + vt_coef * q[1])   # θ_v at the lowest level
    w_heatv = w_heat + vt_coef * θ[1] * w_qflx    # virtual (buoyancy) heat flux
    us = max(ustar, FT(0.01))
    L = -θv_surf * us^3 /
        (c.grav * c.vkarman * (w_heatv + copysign(FT(1e-10), w_heatv)))

    # -- 4. Boundary-layer height (two-pass bulk Richardson) ---------------
    pblh = max(_pbl_height(u, v, θ, q, zc, us, w_heatv, θv_surf, L, c), c.pblh_min)

    # -- 5. Eddy diffusivity profile ---------------------------------------
    unstable = w_heatv > zero(FT)
    # Convective velocity scale w* (only meaningful when unstable).
    wstar = (abs(w_heatv) * c.grav * pblh / θv_surf)^onethird
    @inbounds for l in 1:Nz-1
        z = z_edge[l + 1]
        in_pbl = _step(pblh, z)                  # 1 inside the PBL
        zh = in_pbl * z / pblh                   # fractional height in PBL
        zL = in_pbl * z / L + (1 - in_pbl)       # z/L (=1 outside PBL)
        oneminus = in_pbl * (one(FT) - zh)^2

        unst    = in_pbl * (unstable ? 1 : 0)    # unstable & in PBL
        in_sfc  = _step(c.sf_frac, zh)           # within surface layer
        unst_sfc = unst * in_sfc
        unst_mix = unst * (1 - in_sfc)

        # Stable / neutral local diffusivity (Louis form at this interface).
        f_stab = let r = Ri[l]
            q0 = _step(r, zero(FT))
            q0 / (one(FT) + 10 * r * sqrt(one(FT) + q0 * r)) + (1 - q0)
        end
        Kv_local = mix2_at(z, c.vkarman) * sqrt(shear2[l]) * f_stab
        # Outside the PBL fall back to the free-troposphere value; inside, use
        # the local stable diffusivity. Unstable-BL interfaces are overwritten
        # below, so this initial value only survives where `unst == 0`.
        Kvh = (1 - in_pbl) * Kv_free[l] + in_pbl * max(Kv_local, c.kvh_min)

        if unstable
            # Unstable surface layer: K = u* k z (1-z/h)² (1-βh z/L)^{1/3}.
            slask_sfc = (one(FT) - c.beta_h * zL)
            K_sfc = us * c.vkarman * z * oneminus * cbrt_pos(slask_sfc)
            # Unstable mixed layer: convective velocity scale at the surface-
            # layer top, K = w_sc k z (1-z/h)².
            slask_mix = one(FT) - c.sf_frac * c.beta_h * pblh / L
            w_sc = us * cbrt_pos(slask_mix)
            K_mix = w_sc * c.vkarman * z * oneminus

            K_unst = unst_sfc * K_sfc + unst_mix * K_mix +
                     (1 - unst_sfc - unst_mix) * Kvh

            # Prandtl number: constant through the convective BL, carrying the
            # non-local (counter-gradient) enhancement via `fakn`.
            denom = one(FT) - c.sf_frac * c.beta_h * pblh / L
            term1 = cbrt_pos(denom)
            term2 = sqrt(denom)
            term3 = ccon * c.fakn * wstar / (us * cbrt_pos(denom))
            Pr = term1 / term2 + term3

            Kvh = unst * max(K_unst / Pr, c.kvh_min) + (1 - unst) * Kvh
        end

        # Entrainment layer: at the interface that straddles the PBL top under
        # unstable conditions, override with the prescribed entrainment flux
        # K_entr = 0.2 w_heatv / (dθ_v/dz).
        if unstable
            below_top = _step(pblh, zc[l]) * _step(zc[l + 1], pblh)
            if below_top == 1
                dθv = (θv[l + 1] - θv[l]) / (zc[l + 1] - zc[l])
                # The prescribed entrainment K = 0.2·w_heatv/dθv assumes a STABLE
                # cap (dθv/dz > 0). The straddling interface is occasionally
                # OUTSIDE that validity envelope:
                #   * dθv/dz ≤ 0 (non-stable / superadiabatic cap) → negative K
                #     (anti-diffusion the runtime solve cannot accept);
                #   * dθv/dz → 0 (vanishing inversion) → K → +Inf, or 0/0 → NaN;
                #   * a tiny-but-positive dθv → finite but absurdly large K.
                # In every such case the formula is out of bounds, so we KEEP the
                # column's regular in-PBL mixed-layer diffusivity (continuous, the
                # right sign/limit of mixing) rather than overriding. The override
                # is applied only for a meaningfully positive dθv giving a finite,
                # sane K — then floored at kvh_min. Earlier code wrote the +Inf/NaN
                # into the derived exchange coefficient, which NaN'd runtime diffusion (Dec-11
                # 2021 N320 blew the 14-day run to NaN). Skips are counted (`diag`)
                # so widespread fallback surfaces as a met problem, not silently.
                K_entr = FT(0.2) * w_heatv / dθv
                if dθv > c.dthv_entr_min && isfinite(K_entr) && K_entr <= c.kvh_max
                    Kvh = max(K_entr, c.kvh_min)
                else
                    diag === nothing || (diag.entr_fallback += 1)
                end
            end
        end

        # Final guard: the derived Kz MUST be finite — a single non-finite
        # diffusivity NaN-propagates through the runtime implicit solve. With the
        # entrainment fix above this should never fire; if it does (an unforeseen
        # degeneracy) it is counted, not silently absorbed.
        if isfinite(Kvh)
            kvh[l] = Kvh
        else
            kvh[l] = c.kvh_min
            diag === nothing || (diag.kvh_floored += 1)
        end
        diag === nothing || (diag.max_kz = max(diag.max_kz, kvh[l]))
    end
    kvh[Nz] = zero(FT)
    return pblh
end

# Mixing length squared in the surface/mixed layer (TM5 `cml2` with the 450 m
# asymptote used inside the diffusivity loop, distinct from the free-trop λc).
@inline mix2_at(z, vkarman) =
    (one(z) / (one(z) / (vkarman * z) + one(z) / oftype(z, 450)))^2

# Cube root that returns 0 for non-positive arguments — mirrors TM5's use of
# `slask^{1/3}` only where the stability factor is physically positive.
@inline cbrt_pos(x) = x > zero(x) ? cbrt(x) : zero(x)

# --- Boundary-layer height: bulk-Richardson parcel method -----------------
# Two passes (Vogelezang & Holtslag 1996): a neutral pass, then an unstable
# pass that lifts a parcel with a convective temperature excess. The PBL top is
# where the bulk Richardson number first crosses `ri_crit`, found by linear
# interpolation in Ri between the bracketing layer centres.
function _pbl_height(u, v, θ, q, zc, us, w_heatv, θv_surf, L,
                     c::BLDiffConstants{FT}) where {FT}
    Nz = length(θ)
    binm     = c.beta_m * c.sf_frac
    vt_coef  = c.r_vap / c.r_air - one(FT)
    onethird = one(FT) / 3
    tiny     = FT(1e-9)
    u1, v1, z1 = u[1], v[1], zc[1]

    # Bulk Richardson against a fixed parcel virtual temperature `θv_parcel`,
    # returning the interpolated crossing height (or the model top if never
    # exceeded).
    bulk_ri_height = function (θv_parcel::FT)
        h = zero(FT)
        searching = true
        ri_prev = zero(FT)
        @inbounds for l in 2:Nz
            shear = (u[l] - u1)^2 + (v[l] - v1)^2 + c.ustar_shear * us^2
            shear = max(shear, tiny)
            θv_l = θ[l] * (one(FT) + vt_coef * q[l])
            ri = c.grav * (θv_l - θv_parcel) * (zc[l] - z1) / (shear * θv_surf)
            ri += copysign(FT(1e-10), ri)
            ri = searching ? ri : zero(FT)
            below = _step(c.ri_crit, ri)          # 1 while Ri < ri_crit
            if below == 0
                h = zc[l - 1] +
                    (c.ri_crit - ri_prev) / (ri - ri_prev) * (zc[l] - zc[l - 1])
            end
            searching = (below == 1) && searching
            ri_prev = searching ? ri : ri + FT(0.1)
        end
        return searching ? zc[Nz] : h
    end

    # Pass 1 — neutral parcel (θ_v at the lowest level).
    h1 = bulk_ri_height(θv_surf)

    # Pass 2 — add a convective temperature excess under unstable conditions.
    jq = _step(w_heatv, zero(FT))
    fmt = (jq * (one(FT) - binm * h1 / L) + (1 - jq))^onethird
    w_sc = us * fmt
    excess = w_heatv * c.fak / w_sc
    θv_parcel = θv_surf + jq * excess
    return bulk_ri_height(θv_parcel)
end

# ===========================================================================
# Top-down column driver: from the model's native top-down profiles (k=1 = TOA,
# k=Nz = surface) to a layer-centre Kz diagnostic. The v4 writer stores the
# derived interface exchange `dkg`, not this intermediate Kz profile.
#
# The runtime diffusion kernel reads Kz at layer CENTRES and averages adjacent
# centres to interfaces, whereas `bldiff` produces Kz at INTERFACES. We map the
# interface profile to centres by averaging the two interfaces bounding each
# layer; the runtime re-average then reproduces a faithful, lightly-smoothed
# interface diffusivity (the same centre convention the GCHP runtime Kz field
# uses).
# ===========================================================================

"""
    BLDiffColumnScratch{FT}(Nz)

Per-column work buffers for [`tm5_bldiff_center_kz_column!`]. Allocate once per
thread and reuse across the millions of columns in a global preprocess to avoid
per-column allocation.
"""
Base.@kwdef struct BLDiffColumnScratch{FT <: AbstractFloat}
    T      :: Vector{FT}     # bottom-up layer-centre temperature (Nz)
    q      :: Vector{FT}
    u      :: Vector{FT}
    v      :: Vector{FT}
    p_edge :: Vector{FT}     # bottom-up half-level pressure (Nz+1)
    z_edge :: Vector{FT}     # bottom-up half-level height above surface (Nz+1)
    kvh    :: Vector{FT}     # bottom-up interface diffusivity (Nz)
    dz     :: Vector{FT}     # top-down layer thickness (Nz)
    diag   :: BLDiffDiag{FT} # per-thread fallback counters (reused across columns)
end

# Keyword construction (field order can change without breaking the sizing call).
BLDiffColumnScratch{FT}(Nz::Integer) where {FT} = BLDiffColumnScratch{FT}(
    T = zeros(FT, Nz), q = zeros(FT, Nz), u = zeros(FT, Nz), v = zeros(FT, Nz),
    p_edge = zeros(FT, Nz + 1), z_edge = zeros(FT, Nz + 1),
    kvh = zeros(FT, Nz), dz = zeros(FT, Nz), diag = BLDiffDiag{FT}())

"""
    tm5_bldiff_center_kz_column!(kz, T, q, u, v, ps, hflux, lhflux, ustar,
                                 A, B, c, scratch) -> pblh

Compute the layer-centre TM5 eddy diffusivity (m² s⁻¹) for one column given the
model's **top-down** profiles, writing into `kz` (top-down, length `Nz`) and
returning the diagnosed PBL height (m).

Arguments

  * `kz`            — output, top-down layer-centre Kz (m² s⁻¹), length `Nz`
  * `T, q, u, v`    — top-down layer-centre temperature (K), specific humidity
                      (kg kg⁻¹), and winds (m s⁻¹)
  * `ps`            — surface pressure (Pa)
  * `hflux, lhflux` — surface sensible / latent heat flux (W m⁻², upward-positive)
  * `ustar`         — friction velocity (m s⁻¹)
  * `A, B`          — hybrid-σ half-level coefficients, length `Nz + 1`, ordered
                      TOA→surface, so `p_edge_topdown[k] = A[k] + B[k]·ps`
  * `c`             — [`BLDiffConstants`](@ref)
  * `scratch`       — [`BLDiffColumnScratch`](@ref) sized to `Nz`

The hydrostatic layer thicknesses are derived internally via
`dz_hydrostatic_virtual!`; heights are integrated upward from a zero surface
reference. The column is flipped to the kernel's bottom-up convention, the
interface diffusivities are mapped to centres, and the result is flipped back to
top-down before storing.
"""
function tm5_bldiff_center_kz_column!(kz::AbstractVector{FT},
                                      T::AbstractVector{FT},
                                      q::AbstractVector{FT},
                                      u::AbstractVector{FT},
                                      v::AbstractVector{FT},
                                      ps::FT, hflux::FT, lhflux::FT, ustar::FT,
                                      A::AbstractVector, B::AbstractVector,
                                      c::BLDiffConstants{FT},
                                      scratch::BLDiffColumnScratch{FT}) where {FT}
    Nz = length(T)

    # Top-down layer thickness (m) from the virtual-temperature hydrostatic
    # integral, then bottom-up edges: index 1 = surface (z = 0, p = ps).
    dz = scratch.dz
    dz_hydrostatic_virtual!(dz, T, q, ps, A, B, Nz)
    pe, ze = scratch.p_edge, scratch.z_edge
    @inbounds begin
        ze[1] = zero(FT)
        pe[1] = ps
        for l in 1:Nz
            k = Nz + 1 - l                       # top-down layer feeding bottom-up l
            ze[l + 1] = ze[l] + dz[k]
            pe[l + 1] = FT(A[k] + B[k] * ps)     # top-down upper edge of layer k
        end
        # Flip the centre profiles to bottom-up.
        for l in 1:Nz
            k = Nz + 1 - l
            scratch.T[l] = T[k]; scratch.q[l] = q[k]
            scratch.u[l] = u[k]; scratch.v[l] = v[k]
        end
    end

    pblh = tm5_bldiff_kvh_column!(scratch.kvh, scratch.T, scratch.q,
                                  scratch.u, scratch.v, pe, ze,
                                  hflux, lhflux, ustar, c; diag = scratch.diag)

    # Map bottom-up interface kvh → bottom-up centre Kz (average of the two
    # interfaces bounding each layer; the surface interface carries no flux),
    # then flip back to top-down for storage.
    kvh = scratch.kvh
    @inbounds for l in 1:Nz
        below = l == 1 ? zero(FT) : kvh[l - 1]   # interface under layer l
        above = kvh[l]                            # interface over layer l (kvh[Nz]=0)
        kz[Nz + 1 - l] = (below + above) / 2
    end
    return pblh
end

"""
    tm5_bldiff_dkg_column!(dkg, T, q, u, v, air_mass, ps, hflux, lhflux,
                           ustar, A, B, c, scratch) -> pblh

Compute TM5's interface dry-air exchange directly on one target-grid column:

```text
dkg[k] = Kvh[k] * 2 * (m[k] + m[k+1]) / (dz[k] + dz[k+1])^2
```

`T` is temperature [K], `q` is specific humidity [kg kg⁻¹], `u` and `v` are
wind [m s⁻¹], `air_mass` is dry cell mass [kg], `ps` is surface pressure [Pa],
`hflux` and `lhflux` are upward-positive sensible and latent heat flux
[W m⁻²], and `ustar` is friction velocity [m s⁻¹]. `A` [Pa] and dimensionless
`B` define top-down hybrid interfaces.

Output is top-down: `dkg[k]` exchanges layers `k` and `k+1` [kg s⁻¹], and
`dkg[end] == 0` is the surface no-flux boundary. The function mutates `dkg`
and `scratch`, allocates no column storage, and returns the diagnosed PBL
height [m]. Non-negative `dkg` makes the implicit column solve conserve tracer
mass exactly apart from floating-point roundoff.
"""
function tm5_bldiff_dkg_column!(dkg::AbstractVector{FT},
                                 T::AbstractVector{FT}, q::AbstractVector{FT},
                                 u::AbstractVector{FT}, v::AbstractVector{FT},
                                 air_mass::AbstractVector{FT},
                                 ps::FT, hflux::FT, lhflux::FT, ustar::FT,
                                 A::AbstractVector, B::AbstractVector,
                                 c::BLDiffConstants{FT},
                                 scratch::BLDiffColumnScratch{FT}) where {FT}
    Nz = length(T)
    length(dkg) == length(air_mass) == Nz || throw(DimensionMismatch(
        "dkg, air_mass, and thermodynamic profiles must have the same length"))
    # Populate the exact interface Kvh and virtual-temperature dz. The temporary
    # centre-Kz output is immediately overwritten, avoiding another column buffer.
    pblh = tm5_bldiff_center_kz_column!(dkg, T, q, u, v, ps, hflux, lhflux,
                                        ustar, A, B, c, scratch)
    @inbounds for k in 1:Nz-1
        l_bottom_up = Nz - k
        sum_dz = scratch.dz[k] + scratch.dz[k + 1]
        dkg[k] = max(scratch.kvh[l_bottom_up], zero(FT)) * FT(2) *
                 (air_mass[k] + air_mass[k + 1]) / (sum_dz * sum_dz)
    end
    dkg[Nz] = zero(FT)
    return pblh
end
