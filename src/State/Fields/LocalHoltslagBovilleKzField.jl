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
                                  P <: PBLPhysicsParameters{FT}, A} <: AbstractCubedSphereField{FT}
    panels     :: NTuple{6, F}
    host_cache :: H
    params     :: P
    area_cache :: A
end

function LocalHoltslagBovilleKzField(host_cache::NTuple{6, Array{FT, 3}};
                                    params = PBLPhysicsParameters{FT}()) where FT
    params isa PBLPhysicsParameters{FT} ||
        throw(ArgumentError("params must be a PBLPhysicsParameters{$FT}; got $(typeof(params))"))
    panels = ntuple(p -> PreComputedKzField(host_cache[p]), 6)
    area_cache = _typed_area_cache_ref(FT, host_cache[1], size(host_cache[1], 1),
                                       size(host_cache[1], 2))
    return LocalHoltslagBovilleKzField{FT, typeof(panels[1]), typeof(host_cache),
                                      typeof(params), typeof(area_cache)}(
        panels, host_cache, params, area_cache)
end

@inline panel_field(f::LocalHoltslagBovilleKzField, p::Integer) = f.panels[Int(p)]
# Window-constant cadence by design — see audit memo. VDIFF source
# fields (`vdiff_u/v/t/qv`) + PBL surface (`pblh/ustar/hflux/t2m`)
# are hourly-archive-loaded and refreshed only on met-window advance;
# Kz inherits that cadence. Sub-window refresh would need surface
# interpolation between adjacent archive snapshots, which is not
# wired up. The systematic error is small relative to the typical
# diurnal evolution of these fields over an hour.
update_field!(f::LocalHoltslagBovilleKzField, ::Real) = f

function Adapt.adapt_structure(to, f::LocalHoltslagBovilleKzField)
    panels = Adapt.adapt(to, f.panels)
    data1 = panels[1].data
    area_cache = _typed_area_cache_ref(_local_hb_eltype(f), data1,
                                       size(data1, 1), size(data1, 2))
    return LocalHoltslagBovilleKzField{_local_hb_eltype(f), typeof(panels[1]),
                                      typeof(f.host_cache), typeof(f.params),
                                      typeof(area_cache)}(
        panels, f.host_cache, f.params, area_cache)
end

@inline _local_hb_eltype(::LocalHoltslagBovilleKzField{FT}) where FT = FT

@inline function _virtual_temperature(t, qv, ::Type{FT}) where FT
    return max(FT(t), FT(180)) * (one(FT) + FT(VIRTUAL_TEMPERATURE_FACTOR) * max(FT(qv), zero(FT)))
end

@inline function _potential_temperature(tv, p_mid, ::PBLPhysicsParameters{FT}) where FT
    kappa = one(FT) / FT(3.5)
    return tv * (FT(100000) / max(p_mid, FT(1)))^kappa
end

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
    Nz = size(cache, 3)
    FT = eltype(cache)
    p = params
    R_dry = p.cp_dry / FT(3.5)

    h_pbl = max(FT(pblh[i, j]), FT(100))
    us = max(FT(ustar[i, j]), FT(0.01))
    H_sfc = FT(hflux[i, j])
    T_sfc = max(max(FT(t2m[i, j]), FT(t[i, j, Nz])), FT(200))

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
        theta = _potential_temperature(tv, p_mid, p)
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
end

function _cached_backend_cell_areas!(field::LocalHoltslagBovilleKzField{FT},
                                     cell_areas::AbstractMatrix,
                                     reference) where FT
    cache = field.area_cache[]
    if cache === nothing || size(cache) != size(cell_areas) ||
       eltype(cache) != FT
        cache = similar(reference, FT, size(cell_areas))
        copyto!(cache, cell_areas)
        field.area_cache[] = cache
    end
    return cache
end

@inline function _vdiff_ready(vdiff)
    vdiff === nothing && return false
    return all(p -> _backend_ready_array(vdiff.u[p]) &&
                    _backend_ready_array(vdiff.v[p]) &&
                    _backend_ready_array(vdiff.t[p]) &&
                    _backend_ready_array(vdiff.qv[p]), 1:6)
end

function _try_refresh_local_hb_kz_cache_backend!(field::LocalHoltslagBovilleKzField{FT},
                                                surface,
                                                vdiff,
                                                air_mass::NTuple{6},
                                                cell_areas::AbstractMatrix;
                                                halo_width::Integer = 0) where FT
    data1 = field.panels[1].data
    (_backend_ready_array(data1) &&
     all(p -> _backend_ready_array(air_mass[p]), 1:6) &&
     _backend_ready_surface(surface) &&
     _vdiff_ready(vdiff)) || return false

    areas = _cached_backend_cell_areas!(field, cell_areas, data1)
    backend = get_backend(data1)
    Hp = Int(halo_width)
    @inbounds for panel in 1:6
        cache = field.panels[panel].data
        kernel! = _local_hb_kz_cs_panel_kernel!(backend)
        kernel!(cache, air_mass[panel], surface.pblh[panel],
                surface.ustar[panel], surface.hflux[panel],
                surface.t2m[panel], vdiff.u[panel], vdiff.v[panel],
                vdiff.t[panel], vdiff.qv[panel], areas, field.params, Hp;
                ndrange = (size(cache, 1), size(cache, 2)))
    end
    synchronize(backend)
    return true
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
    if _try_refresh_local_hb_kz_cache_backend!(field, surface, vdiff, air_mass,
                                              cell_areas; halo_width = halo_width)
        return field
    end

    Hp = Int(halo_width)
    areas = FT.(_host_array(cell_areas))
    p = field.params
    R_dry = p.cp_dry / FT(3.5)

    @inbounds for panel in 1:6
        cache = field.host_cache[panel]
        mhost = _host_array(air_mass[panel])
        pblh  = _surface_panel(surface, :pblh,  panel)
        ustar = _surface_panel(surface, :ustar, panel)
        hflux = _surface_panel(surface, :hflux, panel)
        t2m   = _surface_panel(surface, :t2m,   panel)
        u3 = _host_array(vdiff.u[panel])
        v3 = _host_array(vdiff.v[panel])
        t3 = _host_array(vdiff.t[panel])
        q3 = _host_array(vdiff.qv[panel])
        Nc, Ny, Nz = size(cache)

        for j in 1:Ny, i in 1:Nc
            h_pbl = max(FT(pblh[i, j]), FT(100))
            us = max(FT(ustar[i, j]), FT(0.01))
            H_sfc = FT(hflux[i, j])
            T_sfc = max(max(FT(t2m[i, j]), FT(t3[i, j, Nz])), FT(200))
            L_ob, H_kin = _obukhov_length(H_sfc, us, T_sfc, p)
            Pr_inv = _prandtl_inverse(h_pbl, us, H_kin, T_sfc, L_ob, p)

            z_col = zero(FT)
            p_top = zero(FT)
            for k in 1:Nz
                tv = _virtual_temperature(t3[i, j, k], q3[i, j, k], FT)
                delp_k = FT(mhost[i + Hp, j + Hp, k]) * p.gravity / areas[i, j]
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
            for k in 1:Nz
                tv = _virtual_temperature(t3[i, j, k], q3[i, j, k], FT)
                delp_k = FT(mhost[i + Hp, j + Hp, k]) * p.gravity / areas[i, j]
                p_bot = p_top + delp_k
                p_mid = max((p_top + p_bot) / FT(2), FT(1))
                dz_k = delp_k * R_dry * tv / (p.gravity * p_mid)
                z_center = z_above - dz_k / FT(2)
                theta = _potential_temperature(tv, p_mid, p)
                base_kz = _beljaars_viterbo_kz(z_center, h_pbl, us, L_ob,
                                               Pr_inv, p)
                kz = base_kz
                if k > 1
                    kz = _shear_enhanced_kz(base_kz, z_center, prev_z,
                                            theta, prev_theta,
                                            FT(u3[i, j, k]), prev_u,
                                            FT(v3[i, j, k]), prev_v, p)
                end
                cache[i, j, k] = kz
                prev_z = z_center
                prev_theta = theta
                prev_u = FT(u3[i, j, k])
                prev_v = FT(v3[i, j, k])
                z_above -= dz_k
                p_top = p_bot
            end
        end

        copyto!(field.panels[panel].data, cache)
    end
    return field
end

export LocalHoltslagBovilleKzField, refresh_local_holtslag_boville_kz_cache!
