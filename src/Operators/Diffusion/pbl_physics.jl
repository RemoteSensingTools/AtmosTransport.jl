"""
    PBLPhysicsParameters{FT}(; kwargs...)

Physical constants and scheme parameters for the Beljaars-Viterbo
(revised LTG) boundary-layer diffusivity parameterization, as implemented
in TM5 (`diffusion.F90`, `bldiff`). Defaults reproduce the values used in
the legacy `PBLDiffusion`. `WindowPBLKzField` and
`LocalHoltslagBovilleKzField` all read their constants, surface-input
floors and profile shape from here.

| Field                       | Default   | Units       | Meaning                           |
|-----------------------------|-----------|-------------|-----------------------------------|
| `β_h`                       | 15.0      | —           | Businger-Dyer stability parameter |
| `Kz_bg`                     | 0.1       | m²/s        | Free-tropospheric background Kz   |
| `Kz_min`                    | 0.01      | m²/s        | Safety floor inside the PBL       |
| `Kz_max`                    | 500.0     | m²/s        | Safety ceiling inside the PBL     |
| `kappa_vk`                  | 0.41      | —           | von Kármán constant (`VON_KARMAN`) |
| `gravity`                   | 9.80665   | m/s²        | Gravitational acceleration (`STANDARD_GRAVITY`) |
| `cp_dry`                    | 1004.64   | J/(kg·K)    | Dry-air specific heat at const. p (`CP_DRY_AIR`) |
| `rho_ref`                   | 1.225     | kg/m³       | Reference density for H_kin (`STANDARD_SEA_LEVEL_AIR_DENSITY`) |
| `h_pbl_min`                 | 100.0     | m           | Floor on the input PBL height     |
| `ustar_min`                 | 0.01      | m/s         | Floor on the input friction velocity |
| `T_sfc_min`                 | 200.0     | K           | Floor on the input surface air temperature |
| `surface_layer_fraction`    | 0.1       | —           | Surface layer as a fraction of the PBL (TM5 `sffrac`) |
| `entrainment_zone_fraction` | 0.2       | —           | Entrainment-zone depth above the PBL top, as a fraction of the PBL height |

`R_dry` is derived from `cp_dry` as `cp_dry / 3.5` (ideal diatomic,
`cp = 7/2 R_dry`), matching the legacy convention.
"""
Base.@kwdef struct PBLPhysicsParameters{FT <: AbstractFloat}
    β_h      :: FT = FT(15.0)
    Kz_bg    :: FT = FT(0.1)
    Kz_min   :: FT = FT(0.01)
    Kz_max   :: FT = FT(500.0)
    kappa_vk :: FT = FT(VON_KARMAN)
    gravity  :: FT = FT(STANDARD_GRAVITY)
    cp_dry   :: FT = FT(CP_DRY_AIR)
    rho_ref  :: FT = FT(STANDARD_SEA_LEVEL_AIR_DENSITY)
    h_pbl_min                 :: FT = FT(100)
    ustar_min                 :: FT = FT(0.01)
    T_sfc_min                 :: FT = FT(200)
    surface_layer_fraction    :: FT = FT(0.1)
    entrainment_zone_fraction :: FT = FT(0.2)
end

@inline _pbl_cbrt(x::Float64) = cbrt(x)
@inline _pbl_cbrt(x::FT) where FT <: AbstractFloat = x^(one(FT) / FT(3))

# =========================================================================
# Pure helper functions — line-for-line port from the legacy PBL
# diffusion module. Git archaeology: commit ec2d2c0 contains
# src_legacy/Diffusion/pbl_diffusion.jl.
# =========================================================================

"""
    _beljaars_viterbo_kz(z, h_pbl, ustar, L_ob, Pr_inv, p) -> FT

Return Kz [m²/s] at height `z` using the revised Louis-Tiedtke-Geleyn
scheme (Beljaars & Viterbo 1998). `L_ob` is the Obukhov length,
`Pr_inv = Kh/Km ≥ 1` the Prandtl-number inverse (unstable convective
amplification; pass `one(FT)` for stable/neutral).

Line-for-line port of `_pbl_kz` in the legacy PBL diffusion module
(git commit ec2d2c0, path `src_legacy/Diffusion/pbl_diffusion.jl:66`).
Pure: no side effects, no allocation — safe to call inside a kernel.
"""
@inline function _beljaars_viterbo_kz(z, h_pbl, ustar, L_ob, Pr_inv,
                                      p::PBLPhysicsParameters{FT}) where FT
    κ = p.kappa_vk
    # Top of the entrainment zone. With the default fraction 0.2,
    # `one(FT) + FT(0.2) == FT(1.2)` in Float32 and Float64, the float types
    # the model runs in (wider types may round differently).
    h_taper = (one(FT) + p.entrainment_zone_fraction) * h_pbl

    # Above the entrainment zone: background only
    if z >= h_taper
        return p.Kz_bg
    end

    z_eff = min(z, h_pbl - FT(1))   # avoid z/h = 1 singularity
    zh = z_eff / h_pbl
    zzh2 = (FT(1) - zh)^2            # (1 - z/h)² shape function

    if L_ob < zero(FT)
        # Unstable BL
        if z_eff < p.surface_layer_fraction * h_pbl
            # Surface layer (TM5 sffrac)
            Kz = ustar * κ * z_eff * zzh2 * _pbl_cbrt(FT(1) - p.β_h * z_eff / L_ob)
        else
            # Mixed layer
            w_m = ustar * _pbl_cbrt(FT(1) - p.surface_layer_fraction * p.β_h * h_pbl / L_ob)
            Kz = w_m * κ * z_eff * zzh2
        end
        # Convective Prandtl correction (Kh ≥ Km)
        Kz *= Pr_inv
    else
        # Stable BL
        Kz = ustar * κ * z_eff * zzh2 / (FT(1) + FT(5) * z_eff / L_ob)
    end

    Kz = clamp(Kz, p.Kz_min, p.Kz_max)

    # Entrainment-zone taper (h_pbl ≤ z < h_taper): linear blend to Kz_bg
    if z >= h_pbl
        frac = (h_taper - z) / (p.entrainment_zone_fraction * h_pbl)
        Kz = p.Kz_bg + frac * (Kz - p.Kz_bg)
    end

    return Kz
end

"""
    _obukhov_length(hflux, ustar, t2m, p) -> (L_ob, H_kin)

Obukhov length `L_ob` [m] from surface sensible heat flux `hflux`
[W/m²], friction velocity `ustar` [m/s], and 2-m temperature `t2m`
[K]. Also returns the kinematic heat flux `H_kin` [K·m/s], which
the Prandtl-inverse calculation reuses.

Implements `L_ob = -T_sfc · u*³ / (κ · g · H_kin)` with `H_kin = H_sfc / (ρ · cp)`.
Matches the legacy PBL diffusion module (git commit ec2d2c0, path
`src_legacy/Diffusion/pbl_diffusion.jl:161`). A 1e-10 offset (signed
by `H_kin`) prevents a division singularity in the exactly neutral
case `H_kin = 0`.
"""
@inline function _obukhov_length(hflux, ustar, t2m,
                                 p::PBLPhysicsParameters{FT}) where FT
    H_kin  = hflux / (p.rho_ref * p.cp_dry)
    H_safe = H_kin + sign(H_kin + FT(1e-20)) * FT(1e-10)
    L_ob   = -t2m * ustar^3 / (p.kappa_vk * p.gravity * H_safe)
    return L_ob, H_kin
end

"""
    _prandtl_inverse(h_pbl, ustar, H_kin, t2m, L_ob, p) -> FT

Inverse Prandtl number `Kh/Km ≥ 1` for the unstable convective BL,
following TM5 (`diffusion.F90:1213-1230`). Returns `one(FT)` whenever
conditions are not clearly unstable (`L_ob ≥ 0`, `H_kin ≤ 0`, or
`h_pbl ≤ 10 m`). Port of the legacy PBL diffusion module (git
commit ec2d2c0, path `src_legacy/Diffusion/pbl_diffusion.jl:198-210`).
"""
@inline function _prandtl_inverse(h_pbl, ustar, H_kin, t2m, L_ob,
                                  p::PBLPhysicsParameters{FT}) where FT
    if L_ob < zero(FT) && H_kin > zero(FT) && h_pbl > FT(10)
        fL     = max(FT(1) - FT(0.5) * h_pbl / L_ob, FT(1))
        x_h    = _pbl_cbrt(fL)
        w_star = _pbl_cbrt(H_kin * p.gravity * h_pbl / t2m)
        Pr_inv = x_h / sqrt(fL) + FT(7.2) * w_star / (ustar * x_h)
        return max(Pr_inv, one(FT))
    end
    return one(FT)
end

abstract type AbstractWindowKzCache{FT} <: AbstractCubedSphereField{FT} end

@inline function _window_pbl_column!(cache, air_mass, pblh, ustar, hflux, t2m, areas, p, Hp, i, j)
    Nz = size(cache, 3)
    FT = eltype(cache)
    R_dry = p.cp_dry / FT(CP_OVER_R_DIATOMIC)

    h_pbl = max(FT(pblh[i, j]), p.h_pbl_min)
    us = max(FT(ustar[i, j]), p.ustar_min)
    H_sfc = FT(hflux[i, j])
    T_sfc = max(FT(t2m[i, j]), p.T_sfc_min)

    L_ob, H_kin = _obukhov_length(H_sfc, us, T_sfc, p)
    Pr_inv = _prandtl_inverse(h_pbl, us, H_kin, T_sfc, L_ob, p)
    R_T_over_g = R_dry * T_sfc / p.gravity

    z_col = zero(FT)
    p_top = zero(FT)
    @inbounds for k in 1:Nz
        delp_k = FT(air_mass[i + Hp, j + Hp, k]) * p.gravity / FT(areas[i, j])
        p_bot = p_top + delp_k
        p_mid = max((p_top + p_bot) / FT(2), FT(1))
        z_col += delp_k * R_T_over_g / p_mid
        p_top = p_bot
    end

    z_above = z_col
    p_top = zero(FT)
    @inbounds for k in 1:Nz
        delp_k = FT(air_mass[i + Hp, j + Hp, k]) * p.gravity / FT(areas[i, j])
        p_bot = p_top + delp_k
        p_mid = max((p_top + p_bot) / FT(2), FT(1))
        dz_k = delp_k * R_T_over_g / p_mid
        z_center = z_above - dz_k / FT(2)
        cache[i, j, k] = _beljaars_viterbo_kz(z_center, h_pbl, us, L_ob,
                                              Pr_inv, p)
        z_above -= dz_k
        p_top = p_bot
    end
    return nothing
end

@inline function _local_hb_column!(cache, air_mass, pblh, ustar, hflux, t2m, u, v, t, qv, areas, p, Hp, i, j)
    Nz = size(cache, 3)
    FT = eltype(cache)
    R_dry = p.cp_dry / FT(CP_OVER_R_DIATOMIC)

    h_pbl = max(FT(pblh[i, j]), p.h_pbl_min)
    us = max(FT(ustar[i, j]), p.ustar_min)
    H_sfc = FT(hflux[i, j])
    T_sfc = max(max(FT(t2m[i, j]), FT(t[i, j, Nz])), p.T_sfc_min)

    L_ob, H_kin = _obukhov_length(H_sfc, us, T_sfc, p)
    Pr_inv = _prandtl_inverse(h_pbl, us, H_kin, T_sfc, L_ob, p)

    z_col = zero(FT)
    p_top = zero(FT)
    @inbounds for k in 1:Nz
        tv = _virtual_temperature(t[i, j, k], qv[i, j, k], FT)
        delp_k = FT(air_mass[i + Hp, j + Hp, k]) * p.gravity / FT(areas[i, j])
        p_bot = p_top + delp_k
        p_mid = max((p_top + p_bot) / FT(2), FT(1))
        z_col += delp_k * R_dry * tv / (p.gravity * p_mid)
        p_top = p_bot
    end

    z_above = z_col
    p_top = zero(FT)
    prev_z = zero(FT)
    prev_theta = zero(FT)
    prev_u = zero(FT)
    prev_v = zero(FT)
    @inbounds for k in 1:Nz
        tv = _virtual_temperature(t[i, j, k], qv[i, j, k], FT)
        delp_k = FT(air_mass[i + Hp, j + Hp, k]) * p.gravity / FT(areas[i, j])
        p_bot = p_top + delp_k
        p_mid = max((p_top + p_bot) / FT(2), FT(1))
        dz_k = delp_k * R_dry * tv / (p.gravity * p_mid)
        z_center = z_above - dz_k / FT(2)
        theta = _local_hb_theta(tv, p_mid, p)
        base_kz = _beljaars_viterbo_kz(z_center, h_pbl, us, L_ob, Pr_inv, p)
        kz = base_kz
        if k > 1
            kz = _shear_enhanced_kz(base_kz, z_center, prev_z,
                                    theta, prev_theta,
                                    FT(u[i, j, k]), prev_u,
                                    FT(v[i, j, k]), prev_v, p)
        end
        cache[i, j, k] = kz
        prev_z = z_center
        prev_theta = theta
        prev_u = FT(u[i, j, k])
        prev_v = FT(v[i, j, k])
        z_above -= dz_k
        p_top = p_bot
    end
    return nothing
end

@inline _potential_temperature(T, p, κ, p_ref) = T * (p_ref / p)^κ
