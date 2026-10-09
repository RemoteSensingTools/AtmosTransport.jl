"""
    GCHPVdiffParameters{FT}()

Constants of GEOS-Chem's non-local boundary-layer scheme (`vdiff_mod.F90`,
after Holtslag & Boville 1993, J. Climate 6, 1825), with GEOS-Chem's physical
constants (`physconstants.F90`).
"""
Base.@kwdef struct GCHPVdiffParameters{FT}
    g              :: FT = GEOSCHEM_CONSTANTS.gravity      # gravity [m s⁻²]
    R_dry          :: FT = GEOSCHEM_CONSTANTS.r_dry        # dry-air gas constant [J kg⁻¹ K⁻¹]
    cp_dry         :: FT = GEOSCHEM_CONSTANTS.cp_dry       # dry-air heat capacity [J kg⁻¹ K⁻¹]
    L_vap          :: FT = GEOSCHEM_CONSTANTS.l_vap        # latent heat of vaporization [J kg⁻¹]
    ε_virtual      :: FT = GEOSCHEM_CONSTANTS.r_vap / GEOSCHEM_CONSTANTS.r_dry - 1   # Rv/Rd − 1 (free-troposphere θv)
    ε_virtual_pbl  :: FT = VIRTUAL_TEMPERATURE_FACTOR    # pbldif's literal for the surface θv
    p_ref          :: FT = GEOSCHEM_CONSTANTS.p_ref        # potential-temperature reference [Pa]
    karman         :: FT = GEOSCHEM_CONSTANTS.karman
    β_m            :: FT = 15.0          # unstable momentum profile
    β_h            :: FT = 15.0          # unstable heat profile
    β_s            :: FT = 5.0           # stable profile
    fakn           :: FT = 7.2           # counter-gradient constant
    surface_frac   :: FT = 0.1           # surface layer as a fraction of the PBL
    mixing_length  :: FT = 30.0          # free-troposphere mixing length [m]
    kz_min         :: FT = 0.01          # free-troposphere minimum K [m² s⁻¹]
    ustar_min      :: FT = 1e-6          # guard against u★ = 0 (GEOS-Chem has none here) [m s⁻¹]
    p_pbl_min      :: FT = 4e4           # PBL scheme confined below this reference pressure [Pa]
    p_surface_ref  :: FT = 98500         # reference surface pressure for that level count [Pa]
end

"""
    GCHPNonlocalPBLField(Nc, Nz, FT; params = GCHPVdiffParameters{FT}())

GEOS-Chem's non-local PBL mixing on the cubed sphere, refreshed every met
window from the archived PBL height, surface fluxes (`hflux`, `eflux`,
`ustar`) and the VDIFF profiles (`u`, `v`, `t`, `qv`):

  - `dkg` — dry-air exchange `A ρ K / Δz` [kg s⁻¹] between top-down layers
    `k` and `k+1` (TM5 convention; zero at the surface), solved by the
    conservative implicit `:dkg` diffusion. `K` is GEOS-Chem's: a
    Richardson-number closure in the free troposphere and the
    Holtslag–Boville profile inside the PBL.
  - `emission_profile` — the fraction of a column's fresh surface emission
    that each layer receives: GEOS-Chem's counter-gradient term
    `∂(ρ K γ)/∂z`, with `γ = cgs · F/ρ_sfc`, is linear in the surface flux
    `F`, so it moves the fraction `w = ρ K cgs / ρ_sfc` of each step's
    emission upward through every interface. Layer `k` keeps
    `w_below − w_above`; the fractions sum to one.

GEOS-Chem mixes on moist air and closes the column with a mass fix; here the
exchange uses the dry density, so the implicit solve is conservative by
construction.
"""
struct GCHPNonlocalPBLField{FT, D <: PrecomputedCSDkgField{FT}, P <: NTuple{6, AbstractArray{FT, 3}},
                            Q <: GCHPVdiffParameters{FT}, A <: AbstractMatrix{FT}} <: AbstractCSDkgField{FT}
    dkg              :: D
    emission_profile :: P       # (Nc, Nc, Nz) per panel
    p_mid            :: P       # scratch: moist mid-layer pressure [Pa]
    z_mid            :: P       # scratch: mid-layer height above the surface [m]
    params           :: Q
    cell_areas       :: A       # (Nc, Nc) [m²], copied in at every refresh
end

GCHPNonlocalPBLField(dkg::PrecomputedCSDkgField{FT}, profile, p_mid, z_mid, params, areas) where FT =
    GCHPNonlocalPBLField{FT, typeof(dkg), typeof(profile), typeof(params), typeof(areas)}(
        dkg, profile, p_mid, z_mid, params, areas)

function GCHPNonlocalPBLField(Nc::Integer, Nz::Integer, ::Type{FT};
                              params = GCHPVdiffParameters{FT}()) where FT
    panels() = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    profile = panels()
    for p in 1:6
        profile[p][:, :, Nz] .= one(FT)          # until the first refresh: surface layer
    end
    return GCHPNonlocalPBLField(PrecomputedCSDkgField(panels()), profile, panels(), panels(), params,
                                zeros(FT, Nc, Nc))
end

@inline panel_field(f::GCHPNonlocalPBLField, p::Integer) = panel_field(f.dkg, p)
update_field!(f::GCHPNonlocalPBLField, ::Real) = f

Adapt.adapt_structure(to, f::GCHPNonlocalPBLField) =
    GCHPNonlocalPBLField(Adapt.adapt(to, f.dkg), Adapt.adapt(to, f.emission_profile),
                         Adapt.adapt(to, f.p_mid), Adapt.adapt(to, f.z_mid), f.params,
                         Adapt.adapt(to, f.cell_areas))

# Potential temperature of a layer.
@inline _potential_temperature(T, p, prm) = T * (prm.p_ref / p)^(prm.R_dry / prm.cp_dry)

# Free troposphere: Richardson-number closure K = ℓ² |∂v/∂z| f(Ri) (vdiff_mod.F90).
@inline function _free_troposphere_kz(Δu, Δv, Δz, θv_above, θv_below, ℓ², prm)
    FT = typeof(Δz)
    shear² = max((Δu^2 + Δv^2) / Δz^2, FT(1e-30))     # GCHP floors at 1e-36 m² s⁻² before /Δz²
    N² = prm.g * 2 * (θv_above - θv_below) / ((θv_above + θv_below) * Δz)
    Ri = N² / shear²
    f_stab = Ri < 0 ? sqrt(max(1 - 18Ri, zero(FT))) : 1 / (1 + 10Ri * (1 + 8Ri))
    return max(prm.kz_min, ℓ² * sqrt(shear²) * f_stab)
end

# Surface-layer similarity of the column (pbldif with archived PBL height).
@inline function _pbl_surface_state(w_θv, θv_s, u★, h, prm)
    FT = typeof(h)
    L = -θv_s * u★^3 / (prm.g * prm.karman * (w_θv + copysign(FT(1e-10), w_θv)))  # Obukhov length
    unstable = w_θv > 0
    φm⁻¹ = unstable ? cbrt(1 - prm.β_m * prm.surface_frac * h / L) : one(FT)
    φh⁻¹ = unstable ? sqrt(1 - prm.β_h * prm.surface_frac * h / L) : one(FT)
    w_m = u★ * φm⁻¹                                         # velocity scale
    w★ = unstable ? cbrt(w_θv * prm.g * h / θv_s) : zero(FT) # convective velocity
    fak3 = unstable ? prm.fakn * w★ / w_m : zero(FT)
    return (; L, unstable, φm⁻¹, φh⁻¹, w_m, fak3)
end

# Holtslag–Boville K and counter-gradient coefficient at height z inside the PBL.
@inline function _pbl_kz(z, h, u★, sfc, prm)
    FT = typeof(z)
    ζh, ζL = z / h, z / sfc.L
    shape = ζh <= 1 ? ζh * (1 - ζh)^2 : zero(FT)
    if !sfc.unstable
        K = u★ * h * prm.karman * shape / (ζL <= 1 ? 1 + prm.β_s * ζL : prm.β_s + ζL)
        return K, zero(FT)
    elseif ζh < prm.surface_frac                             # unstable surface layer
        term = cbrt(1 - prm.β_m * ζL)
        Pr = term / sqrt(1 - prm.β_h * ζL)
        return u★ * h * prm.karman * shape * term / Pr, zero(FT)
    else                                                     # unstable outer layer
        Pr = sfc.φm⁻¹ / sfc.φh⁻¹ + prm.surface_frac * prm.karman * sfc.fak3   # ccon·fak3/fak
        return sfc.w_m * h * prm.karman * shape / Pr, sfc.fak3 / (h * sfc.w_m)
    end
end

@kernel function _gchp_nonlocal_pbl_kernel!(dkg, profile, p_mid, z_mid,
                                            @Const(air_mass), @Const(area),
                                            @Const(pblh), @Const(ustar), @Const(hflux),
                                            @Const(eflux), @Const(u), @Const(v),
                                            @Const(t), @Const(qv), prm, p_top, n_pbl, Hp)
    i, j = @index(Global, NTuple)
    FT = eltype(dkg)
    Nz = size(dkg, 3)
    A = FT(area[i, j])
    g, R = prm.g, prm.R_dry
    moist_Δp(k) = g * air_mass[i + Hp, j + Hp, k] / (A * (1 - qv[i, j, k]))
    θv(k, ε) = _potential_temperature(t[i, j, k], p_mid[i, j, k], prm) * (1 + ε * qv[i, j, k])

    @inbounds begin
        # Moist mid-layer pressure from the dry air mass and humidity.
        p_edge = FT(p_top)
        for k in 1:Nz
            Δp = moist_Δp(k)
            p_mid[i, j, k] = p_edge + Δp / 2
            p_edge += Δp
        end
        # Mid-layer heights above the surface (hypsometric, virtual temperature).
        p_below, z_below = p_edge, zero(FT)
        for k in Nz:-1:1
            H = R / g * t[i, j, k] * (1 + prm.ε_virtual * qv[i, j, k])   # scale height
            p_above = 2p_mid[i, j, k] - p_below
            z_mid[i, j, k] = z_below + H * log(p_below / p_mid[i, j, k])
            z_below += H * log(p_below / p_above)
            p_below = p_above
        end

        # Surface fluxes as kinematic fluxes at the lowest layer.
        ρ_s = p_mid[i, j, Nz] / (R * t[i, j, Nz])
        θ_s = _potential_temperature(t[i, j, Nz], p_mid[i, j, Nz], prm)
        w_θ = hflux[i, j] / (ρ_s * prm.cp_dry)
        w_q = eflux[i, j] / (prm.L_vap * ρ_s)
        w_θv = w_θ + prm.ε_virtual_pbl * θ_s * w_q
        h = FT(pblh[i, j])
        u★ = max(FT(ustar[i, j]), prm.ustar_min)
        sfc = _pbl_surface_state(w_θv, θv(Nz, prm.ε_virtual_pbl), u★, h, prm)

        # Interfaces k = 2…Nz, between layers k−1 (above) and k (below).
        profile[i, j, 1] = zero(FT)                 # nothing crosses the model top
        p_edge = FT(p_top)
        for k in 2:Nz
            p_edge += moist_Δp(k - 1)
            Δz = z_mid[i, j, k - 1] - z_mid[i, j, k]
            ℓ² = k == 2 ? zero(FT) : prm.mixing_length^2
            K = _free_troposphere_kz(u[i, j, k - 1] - u[i, j, k], v[i, j, k - 1] - v[i, j, k], Δz,
                                     θv(k - 1, prm.ε_virtual), θv(k, prm.ε_virtual), ℓ², prm)
            cgs = zero(FT)
            if z_mid[i, j, k] < h && k >= Nz - n_pbl + 2
                K_pbl, cgs = _pbl_kz((z_mid[i, j, k] + z_mid[i, j, k - 1]) / 2, h, u★, sfc, prm)
                K = max(K_pbl, K)
            end
            T_edge = (t[i, j, k - 1] + t[i, j, k]) / 2
            ρ = p_edge / (R * T_edge)
            ρ_dry = ρ * (1 - (qv[i, j, k - 1] + qv[i, j, k]) / 2)
            dkg[i, j, k - 1] = A * ρ_dry * K / Δz
            profile[i, j, k] = ρ * K * cgs / ρ_s      # fraction w carried up through edge k
        end
        dkg[i, j, Nz] = zero(FT)
        # Layer k keeps w(edge below) − w(edge above); the surface edge carries 1.
        for k in 1:Nz
            w_below = k < Nz ? profile[i, j, k + 1] : one(FT)
            profile[i, j, k] = w_below - profile[i, j, k]
        end
    end
end

"""
    gchp_pbl_layer_count(A, B, prm) -> Int

GEOS-Chem's `npbl` (`Max_PblHt_For_Vdiff`): the bottom layers whose reference
mid-layer pressure exceeds `prm.p_pbl_min`; the PBL profile is confined to
the interfaces among them. GEOS-Chem takes the reference profile from the
global-mean model pressure; here it is the hybrid coordinate at
`prm.p_surface_ref`.
"""
function gchp_pbl_layer_count(A, B, prm::GCHPVdiffParameters)
    p_ref(k) = (A[k] + A[k + 1] + (B[k] + B[k + 1]) * prm.p_surface_ref) / 2
    Nz = length(A) - 1
    k = findlast(k -> p_ref(k) < prm.p_pbl_min, 1:Nz)
    return max(1, Nz - something(k, 0))
end

"""
    refresh_gchp_nonlocal_pbl!(field, surface, vdiff, air_mass, cell_areas, vertical; halo_width)

Recompute `field.dkg` and `field.emission_profile` for the current met
window. `surface` must carry `eflux`; `vertical` is the hybrid coordinate
(`A`, `B`; `A[1]` is the model-top pressure [Pa]).
"""
function refresh_gchp_nonlocal_pbl!(field::GCHPNonlocalPBLField, surface, vdiff,
                                    air_mass::NTuple{6}, cell_areas::AbstractMatrix, vertical;
                                    halo_width::Integer = 0)
    (surface === nothing || surface.eflux === nothing) && throw(ArgumentError(
        "GEOS-Chem non-local VDIFF needs pblh/ustar/pbl_hflux/t2m and pbl_eflux in the transport window"))
    vdiff === nothing && throw(ArgumentError(
        "GEOS-Chem non-local VDIFF needs vdiff_u/vdiff_v/vdiff_t/vdiff_qv in the transport window"))
    copyto!(field.cell_areas, cell_areas)
    areas = field.cell_areas
    backend = get_backend(field.p_mid[1])
    kernel! = _gchp_nonlocal_pbl_kernel!(backend)
    for p in 1:6
        dkg = panel_field(field, p).data
        kernel!(dkg, field.emission_profile[p], field.p_mid[p], field.z_mid[p],
                air_mass[p], areas, surface.pblh[p], surface.ustar[p], surface.hflux[p],
                surface.eflux[p], vdiff.u[p], vdiff.v[p], vdiff.t[p], vdiff.qv[p],
                field.params, first(vertical.A), gchp_pbl_layer_count(vertical.A, vertical.B, field.params),
                Int(halo_width); ndrange = size(dkg)[1:2])
    end
    synchronize(backend)
    return field
end
