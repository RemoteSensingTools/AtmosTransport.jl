"""
    LocalHoltslagBovilleKzField(host_cache; params = PBLPhysicsParameters{FT}())

Panel-native Kz cache for the GEOS/GCHP VDIFF runtime path.

The cache is refreshed from the active window's GEOS VDIFF payload
(`vdiff_u`, `vdiff_v`, `vdiff_t`, `vdiff_qv`) plus PBL surface fields.
This is a **local** Holtslag-Boville Kz closure: it derives column
geometry + free-tropospheric shear enhancement from GEOS T/qv/wind
profiles, with the existing Beljaars-Viterbo surface-layer shape inside
the diagnosed PBL.

# ⚠ NOT full GCHP VDIFF parity

Two intentional divergences from GCHP's `vdiff_mod.F90`:

  1. **No non-local counter-gradient term.** GCHP's `pbldif`
     (`vdiff_mod.F90:1237-1254`) computes counter-gradient terms
     `cgs = fak3/(pblh·wm)`, `cgh = khfs·cgs` and injects them into
     the θ and qv profiles before the implicit solve. We apply only the
     LOCAL Kz, so daytime PBL lofting of surface-emitted tracers is
     systematically weaker than GCHP — typically 10–30 % less mass
     above the PBL top for surface sources. Use
     `kind = "tm5_beljaars_viterbo_local_kz"` if you want to be explicit
     that no parameterization claims GCHP parity.
  2. **Surface-flux coupling default mismatch.** GCHP applies emissions
     as a boundary condition inside one combined turbulence step
     (`vdiff_mod.F90:679`, `gchp_chunk_mod.F90:1296`). Our default
     `SplitSurfaceFluxCoupling` does `V(dt/2) → S(dt) → V(dt/2)` Strang
     instead. For GCHP parity in long integrations, set
     `[surface_flux].coupling = "boundary"` (which selects
     `DiffusiveSurfaceFluxBoundary`). The recipe validator emits a
     warning at config-load when this Kz field is selected without the
     boundary-coupling switch.

See `memory/diffusion_full_pipeline_audit_2026_05_25.md` for the full
audit chain and pending mass-flux conservation fix.

# Why no non-local kernel here

In GCHP's full VDIFF the non-local term enters as an additive RHS source
in the implicit tridiagonal solve, NOT a modification of `Kz` itself:

    ∂(m·q)/∂t = ∂/∂z [ρ·Kz·(∂q/∂z - γ_q)]

where `γ_q = a·w*·q*/wm²/pblh` is a scaled convective-velocity expression
of the surface flux. A real non-local kernel would need:
  1. Convective-velocity diagnostics (`w*`, `wm`, etc.) added to the
     per-window refresh — currently only `pblh, ustar, hflux, t2m` are
     in the surface payload.
  2. A per-tracer `γ_q` derivation. For our offline pipeline this is
     ill-defined because surface emissions are applied by
     `apply_surface_flux!` as a SEPARATE operator outside the diffusion
     solve, so the surface-source pattern that the GCHP counter-gradient
     exists to handle doesn't appear inside our diffusion step.
  3. A new RHS-source term in `_vertical_diffusion_cs_*` kernels +
     a matching adjoint kernel.

This is deferred indefinitely; the field is correctly named `Local…`
so users opting into GCHP-style VDIFF know what they get.

"""
struct LocalHoltslagBovilleKzField{FT, F <: AbstractTimeVaryingField{FT, 3}, H,
                                  P <: PBLPhysicsParameters{FT}, A} <: AbstractWindowKzCache{FT}
    panels     :: NTuple{6, F}
    host_cache :: H
    params     :: P
    area_cache :: A
end

@inline function _virtual_temperature(t, qv, ::Type{FT}) where FT
    return max(FT(t), FT(180)) * (one(FT) + FT(VIRTUAL_TEMPERATURE_FACTOR) * max(FT(qv), zero(FT)))
end

# Virtual potential temperature, κ = R/cp of ideal diatomic dry air.
@inline _local_hb_theta(tv, p_mid, ::PBLPhysicsParameters{FT}) where FT =
    _potential_temperature(tv, max(p_mid, FT(1)), one(FT) / FT(CP_OVER_R_DIATOMIC), FT(THETA_REFERENCE_PRESSURE))

@inline function _shear_enhanced_kz(base_kz, z_lower, z_upper, theta_lower,
                                    theta_upper, u_lower, u_upper,
                                    v_lower, v_upper,
                                    p::PBLPhysicsParameters{FT}) where FT
    dz = max(abs(z_upper - z_lower), FT(1))
    du_dz = (u_upper - u_lower) / dz
    dv_dz = (v_upper - v_lower) / dz
    shear2 = du_dz * du_dz + dv_dz * dv_dz
    shear = sqrt(max(shear2, zero(FT)))
    shear <= FT(1e-6) && return base_kz

    theta_mid = max((theta_lower + theta_upper) / FT(2), FT(100))
    dtheta_dz_up = (theta_upper - theta_lower) / dz
    n2 = p.gravity / theta_mid * dtheta_dz_up
    ri = n2 / max(shear2, FT(1e-12))
    ric = FT(0.25)
    stability = ri <= zero(FT) ? one(FT) : max(zero(FT), one(FT) - ri / ric)^2
    l_mix = min(p.kappa_vk * max(min(z_lower, z_upper), FT(1)), FT(150))
    shear_kz = l_mix * l_mix * shear * stability
    return clamp(max(base_kz, p.Kz_bg + shear_kz), p.Kz_bg, p.Kz_max)
end

@kernel function _local_hb_kz_cs_panel_kernel!(cache, @Const(air_mass),
                                              @Const(pblh), @Const(ustar),
                                              @Const(hflux), @Const(t2m),
                                              @Const(u), @Const(v),
                                              @Const(t), @Const(qv),
                                              @Const(areas), params, Hp)
    i, j = @index(Global, NTuple)
    _local_hb_column!(cache, air_mass, pblh, ustar, hflux, t2m, u, v, t, qv, areas, params, Hp, i, j)
end

@inline function _vdiff_ready(vdiff)
    vdiff === nothing && return false
    return all(p -> _backend_ready_array(vdiff.u[p]) &&
                    _backend_ready_array(vdiff.v[p]) &&
                    _backend_ready_array(vdiff.t[p]) &&
                    _backend_ready_array(vdiff.qv[p]), 1:6)
end

function refresh_local_holtslag_boville_kz_cache!(field::LocalHoltslagBovilleKzField{FT},
                                                  surface,
                                                  vdiff,
                                                  air_mass::NTuple{6},
                                                  cell_areas::AbstractMatrix;
                                                  halo_width::Integer = 0) where FT
    surface === nothing &&
        throw(ArgumentError("[diffusion] kind=\"geoschem_holtslag_boville_vdiff\" requires pblh/ustar/pbl_hflux/t2m surface fields in the transport window"))
    vdiff === nothing &&
        throw(ArgumentError("[diffusion] kind=\"geoschem_holtslag_boville_vdiff\" requires vdiff_u/vdiff_v/vdiff_t/vdiff_qv sections in the transport window"))
    if _try_refresh_window_kz_backend!(field, surface, vdiff, air_mass,
                                              cell_areas; halo_width = halo_width)
        return field
    end

    Hp = Int(halo_width)
    areas = FT.(_pbl_host_array(cell_areas))
    p = field.params

    @inbounds for panel in 1:6
        cache = field.host_cache[panel]
        mhost = _pbl_host_array(air_mass[panel])
        pblh  = _surface_panel(surface, :pblh,  panel)
        ustar = _surface_panel(surface, :ustar, panel)
        hflux = _surface_panel(surface, :hflux, panel)
        t2m   = _surface_panel(surface, :t2m,   panel)
        u3 = _pbl_host_array(vdiff.u[panel])
        v3 = _pbl_host_array(vdiff.v[panel])
        t3 = _pbl_host_array(vdiff.t[panel])
        q3 = _pbl_host_array(vdiff.qv[panel])
        Nc, Ny, Nz = size(cache)

        for j in 1:Ny, i in 1:Nc
            _local_hb_column!(cache, mhost, pblh, ustar, hflux, t2m, u3, v3, t3, q3, areas, p, Hp, i, j)
        end

        copyto!(field.panels[panel].data, cache)
    end
    return field
end

export LocalHoltslagBovilleKzField, refresh_local_holtslag_boville_kz_cache!
