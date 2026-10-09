"""
    PBLPhysicsParameters{FT}(; kwargs...)

Physical constants for the Beljaars-Viterbo (revised LTG) boundary-
layer diffusivity parameterization, as implemented in TM5
(`diffusion.F90`, `bldiff`). Defaults reproduce the values used in
the legacy `PBLDiffusion`.

| Field      | Default   | Units       | Meaning                           |
|------------|-----------|-------------|-----------------------------------|
| `β_h`      | 15.0      | —           | Businger-Dyer stability parameter |
| `Kz_bg`    | 0.1       | m²/s        | Free-tropospheric background Kz   |
| `Kz_min`   | 0.01      | m²/s        | Safety floor inside the PBL       |
| `Kz_max`   | 500.0     | m²/s        | Safety ceiling inside the PBL     |
| `kappa_vk` | 0.41      | —           | von Kármán constant               |
| `gravity`  | 9.80665   | m/s²        | Gravitational acceleration        |
| `cp_dry`   | 1004.64   | J/(kg·K)    | Dry-air specific heat at const. p |
| `rho_ref`  | 1.225     | kg/m³       | Reference density for H_kin       |

`R_dry` is derived from `cp_dry` as `cp_dry / 3.5` (ideal diatomic,
`cp = 7/2 R_dry`), matching the legacy convention.
"""
Base.@kwdef struct PBLPhysicsParameters{FT <: AbstractFloat}
    β_h      :: FT = FT(15.0)
    Kz_bg    :: FT = FT(0.1)
    Kz_min   :: FT = FT(0.01)
    Kz_max   :: FT = FT(500.0)
    kappa_vk :: FT = FT(0.41)
    gravity  :: FT = FT(STANDARD_GRAVITY)
    cp_dry   :: FT = FT(CP_DRY_AIR)
    rho_ref  :: FT = FT(1.225)
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
    h_taper = FT(1.2) * h_pbl

    # Above the entrainment zone: background only
    if z >= h_taper
        return p.Kz_bg
    end

    z_eff = min(z, h_pbl - FT(1))   # avoid z/h = 1 singularity
    zh = z_eff / h_pbl
    zzh2 = (FT(1) - zh)^2            # (1 - z/h)² shape function

    if L_ob < zero(FT)
        # Unstable BL
        if z_eff < FT(0.1) * h_pbl
            # Surface layer (TM5 sffrac = 0.1)
            Kz = ustar * κ * z_eff * zzh2 * _pbl_cbrt(FT(1) - p.β_h * z_eff / L_ob)
        else
            # Mixed layer
            w_m = ustar * _pbl_cbrt(FT(1) - FT(0.1) * p.β_h * h_pbl / L_ob)
            Kz = w_m * κ * z_eff * zzh2
        end
        # Convective Prandtl correction (Kh ≥ Km)
        Kz *= Pr_inv
    else
        # Stable BL
        Kz = ustar * κ * z_eff * zzh2 / (FT(1) + FT(5) * z_eff / L_ob)
    end

    Kz = clamp(Kz, p.Kz_min, p.Kz_max)

    # Entrainment-zone taper (h_pbl ≤ z < 1.2 h_pbl): linear blend to Kz_bg
    if z >= h_pbl
        frac = (h_taper - z) / (FT(0.2) * h_pbl)
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

# =========================================================================
# DerivedKzField
# =========================================================================

"""
    DerivedKzField(; surface, delp, cache, params = PBLPhysicsParameters{FT}())

A rank-3 `AbstractTimeVaryingField{FT, 3}` whose values are Kz [m²/s]
computed per column from Beljaars-Viterbo surface-field closure.

# Inputs

| Name       | Type                                            | Role                                  |
|------------|-------------------------------------------------|---------------------------------------|
| `surface`  | `NamedTuple` of four rank-2 `TimeVaryingField`s | `(pblh, ustar, hflux, t2m)` fields    |
| `delp`     | rank-3 `TimeVaryingField`                       | Layer pressure thickness [Pa], k=1 at TOA |
| `cache`    | `AbstractArray{FT, 3}`                          | Backing storage for computed Kz, size `(Nx, Ny, Nz)` |
| `params`   | `PBLPhysicsParameters{FT}`                      | Physical constants                    |

# Interface

- `field_value(f, (i, j, k))` → `f.cache[i, j, k]`. Kernel-safe.
- `update_field!(f, t)` →
  1. Refreshes every input field at time `t` (surface + delp).
  2. Recomputes every cell's Kz on the cache backend.

Kz is stored at **cell centers**. Consumers that need interface Kz
(e.g. a Thomas solve) should interpolate between adjacent k-levels.

# Vertical convention

`k = 1` is TOA; `k = Nz` is surface. This matches the legacy
`_pbl_diffuse_kernel!` which accumulates `p_top` upward from `0` Pa
and integrates hydrostatic `dz` downward. Configs that provide
ground-up delp must flip the k axis before passing in.

# Example (test-style construction)

```julia
surface = (
    pblh  = ConstantField{Float64, 2}(1000.0),
    ustar = ConstantField{Float64, 2}(0.3),
    hflux = ConstantField{Float64, 2}(100.0),
    t2m   = ConstantField{Float64, 2}(295.0),
)
delp  = PreComputedKzField(fill(3000.0, 4, 3, 10))   # 10 layers of 3 kPa
cache = zeros(Float64, 4, 3, 10)
f     = DerivedKzField(; surface, delp, cache)

update_field!(f, 0.0)
Kz_surface_cell = field_value(f, (1, 1, 10))   # populated m²/s
```
"""
struct DerivedKzField{FT, SF, DELP, A,
                      P <: PBLPhysicsParameters{FT}} <: AbstractTimeVaryingField{FT, 3}
    surface :: SF
    delp    :: DELP
    cache   :: A
    params  :: P

    function DerivedKzField{FT, SF, DELP, A, P}(surface::SF, delp::DELP,
                                                cache::A, params::P) where {
            FT <: AbstractFloat,
            SF <: NamedTuple{(:pblh, :ustar, :hflux, :t2m)},
            DELP <: AbstractTimeVaryingField{FT, 3},
            A <: AbstractArray{FT, 3},
            P <: PBLPhysicsParameters{FT}}
        for (name, field) in pairs(surface)
            field isa AbstractTimeVaryingField{FT, 2} ||
                throw(ArgumentError("surface.$name must be an AbstractTimeVaryingField{$FT, 2}; got $(typeof(field))"))
        end
        new{FT, SF, DELP, A, P}(surface, delp, cache, params)
    end
end

function DerivedKzField(; surface::NamedTuple, delp, cache::AbstractArray{FT, 3},
                        params = nothing) where FT
    resolved = params === nothing ? PBLPhysicsParameters{FT}() : params
    resolved isa PBLPhysicsParameters{FT} ||
        throw(ArgumentError("params must be a PBLPhysicsParameters{$FT}; got $(typeof(resolved))"))
    DerivedKzField{FT, typeof(surface), typeof(delp), typeof(cache), typeof(resolved)}(
        surface, delp, cache, resolved)
end

function Adapt.adapt_structure(to, f::DerivedKzField{FT}) where FT
    surface = Adapt.adapt(to, f.surface)
    delp = Adapt.adapt(to, f.delp)
    cache = Adapt.adapt(to, f.cache)
    return DerivedKzField{FT, typeof(surface), typeof(delp), typeof(cache),
                          typeof(f.params)}(surface, delp, cache, f.params)
end

@inline field_value(f::DerivedKzField, idx::NTuple{3, Int}) =
    @inbounds f.cache[idx[1], idx[2], idx[3]]

function update_field!(f::DerivedKzField, t::Real)
    update_field!(f.surface.pblh,  t)
    update_field!(f.surface.ustar, t)
    update_field!(f.surface.hflux, t)
    update_field!(f.surface.t2m,   t)
    update_field!(f.delp, t)
    _recompute_kz_cache!(f)
    return f
end

"""
    _recompute_kz_cache!(f::DerivedKzField) -> f

Fill `f.cache[i, j, k]` with Kz at cell-center height, using the
current state of the surface fields and `delp`. One backend-portable kernel
runs per horizontal column and performs the vertical pressure-to-height
integration locally, so adapted fields refresh without host scalar indexing.
"""
@kernel function _derived_kz_cache_kernel!(cache, surface, delp, p)
    i, j = @index(Global, NTuple)
    Nz = size(cache, 3)
    FT = eltype(cache)
    R_dry = p.cp_dry / FT(3.5)

    h_pbl = max(field_value(surface.pblh,  (i, j)), FT(100))
    us    = max(field_value(surface.ustar, (i, j)), FT(0.01))
    H_sfc = field_value(surface.hflux, (i, j))
    T_sfc = max(field_value(surface.t2m,   (i, j)), FT(200))

    L_ob, H_kin = _obukhov_length(H_sfc, us, T_sfc, p)
    Pr_inv = _prandtl_inverse(h_pbl, us, H_kin, T_sfc, L_ob, p)
    R_T_over_g = R_dry * T_sfc / p.gravity

    z_col = zero(FT)
    p_top = zero(FT)
    @inbounds for k in 1:Nz
        delp_k = field_value(delp, (i, j, k))
        p_bot = p_top + delp_k
        p_mid = max((p_top + p_bot) / FT(2), FT(1))
        z_col += delp_k * R_T_over_g / p_mid
        p_top = p_bot
    end

    z_above = z_col
    p_top = zero(FT)
    @inbounds for k in 1:Nz
        delp_k = field_value(delp, (i, j, k))
        p_bot = p_top + delp_k
        p_mid = max((p_top + p_bot) / FT(2), FT(1))
        dz_k = delp_k * R_T_over_g / p_mid
        z_center = z_above - dz_k / FT(2)
        cache[i, j, k] = _beljaars_viterbo_kz(z_center, h_pbl, us, L_ob,
                                              Pr_inv, p)
        z_above -= dz_k
        p_top = p_bot
    end
end

function _recompute_kz_cache!(f::DerivedKzField)
    cache = f.cache
    Nx, Ny, _ = size(cache)
    backend = get_backend(cache)
    kernel! = _derived_kz_cache_kernel!(backend)
    kernel!(cache, f.surface, f.delp, f.params; ndrange = (Nx, Ny))
    synchronize(backend)
    return f
end
