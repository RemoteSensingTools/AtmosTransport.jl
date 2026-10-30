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
