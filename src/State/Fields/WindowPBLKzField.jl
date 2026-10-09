"""
    WindowPBLKzField(host_cache; params = PBLPhysicsParameters{FT}())

Panel-native Kz cache for cubed-sphere window-driven PBL diffusion.

`host_cache` is an `NTuple{6, Array{FT,3}}` with interior panel shape
`(Nc, Nc, Nz)`. The runtime refreshes it from the active transport window's
raw PBL surface fields and dry air mass whenever the met window advances.
The diffusion kernels read the wrapped `PreComputedKzField`s.

# Cadence — window-constant by design

`update_field!(::WindowPBLKzField, ::Real)` is a deliberate no-op. The
surface fields that feed the Kz closure (`pblh`, `ustar`, `pbl_hflux`,
`t2m`) are loaded once per met window (hourly archive); the diagnosed
Kz inherits that cadence. Refreshing per substep without first
interpolating the surface fields would produce identical output — so
the no-op accurately reflects the data flow.

This matches the surface-state smoothness on the typical 1 h archive
cadence (pblh and friends evolve on a 10 min - 1 h characteristic
scale). The associated systematic error in tracer diffusion is well
below the operator-level mass-conservation tolerance (10⁻⁷).
TM5 and GCHP refresh Kz at every dynamic step against state that's
also updated each dynamic step; matching that cadence on offline
transport would require sub-hourly surface forcing and per-substep
linear interpolation, neither of which is wired up here. Tracked in
`memory/diffusion_full_pipeline_audit_2026_05_25.md`.
"""
struct WindowPBLKzField{FT, F <: AbstractTimeVaryingField{FT, 3}, H,
                        P <: PBLPhysicsParameters{FT}, A} <: AbstractCubedSphereField{FT}
    panels     :: NTuple{6, F}
    host_cache :: H
    params     :: P
    area_cache :: A
end

function WindowPBLKzField(host_cache::NTuple{6, Array{FT, 3}};
                          params = PBLPhysicsParameters{FT}()) where FT
    params isa PBLPhysicsParameters{FT} ||
        throw(ArgumentError("params must be a PBLPhysicsParameters{$FT}; got $(typeof(params))"))
    panels = ntuple(p -> PreComputedKzField(host_cache[p]), 6)
    area_cache = _typed_area_cache_ref(FT, host_cache[1], size(host_cache[1], 1),
                                       size(host_cache[1], 2))
    return WindowPBLKzField{FT, typeof(panels[1]), typeof(host_cache),
                            typeof(params), typeof(area_cache)}(
        panels, host_cache, params, area_cache)
end

@inline panel_field(f::WindowPBLKzField, p::Integer) = f.panels[Int(p)]
# Window-constant cadence by design — see struct docstring.
update_field!(f::WindowPBLKzField, ::Real) = f

function Adapt.adapt_structure(to, f::WindowPBLKzField)
    panels = Adapt.adapt(to, f.panels)
    data1 = panels[1].data
    area_cache = _typed_area_cache_ref(_window_pbl_eltype(f), data1,
                                       size(data1, 1), size(data1, 2))
    return WindowPBLKzField{_window_pbl_eltype(f), typeof(panels[1]),
                            typeof(f.host_cache), typeof(f.params),
                            typeof(area_cache)}(
        panels, f.host_cache, f.params, area_cache)
end

@inline _window_pbl_eltype(::WindowPBLKzField{FT}) where FT = FT

function _typed_area_cache_ref(::Type{FT}, reference, nx::Integer, ny::Integer) where FT
    probe = similar(reference, FT, Int(nx), Int(ny))
    return Ref{Union{Nothing, typeof(probe)}}(nothing)
end

_host_array(a::Array) = a
_host_array(a) = Array(a)

@inline function _surface_panel(surface, name::Symbol, p::Int)
    return _host_array(getfield(surface, name)[p])
end

@kernel function _window_pbl_kz_cs_panel_kernel!(cache, @Const(air_mass),
                                                 @Const(pblh), @Const(ustar),
                                                 @Const(hflux), @Const(t2m),
                                                 @Const(areas), params, Hp)
    i, j = @index(Global, NTuple)
    Nz = size(cache, 3)
    FT = eltype(cache)
    p = params
    R_dry = p.cp_dry / FT(CP_OVER_R_DIATOMIC)

    h_pbl = max(FT(pblh[i, j]), FT(100))
    us = max(FT(ustar[i, j]), FT(0.01))
    H_sfc = FT(hflux[i, j])
    T_sfc = max(FT(t2m[i, j]), FT(200))

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
end

@inline _backend_ready_array(a::Array) = false
@inline _backend_ready_array(_a) = true

function _backend_ready_surface(surface)
    surface === nothing && return false
    return all(p -> _backend_ready_array(surface.pblh[p]) &&
                    _backend_ready_array(surface.ustar[p]) &&
                    _backend_ready_array(surface.hflux[p]) &&
                    _backend_ready_array(surface.t2m[p]), 1:6)
end

function _cached_backend_cell_areas!(field::WindowPBLKzField{FT},
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

function _try_refresh_pbl_kz_cache_backend!(field::WindowPBLKzField{FT},
                                            surface,
                                            air_mass::NTuple{6},
                                            cell_areas::AbstractMatrix;
                                            halo_width::Integer = 0) where FT
    data1 = field.panels[1].data
    (_backend_ready_array(data1) &&
     all(p -> _backend_ready_array(air_mass[p]), 1:6) &&
     _backend_ready_surface(surface)) || return false

    areas = _cached_backend_cell_areas!(field, cell_areas, data1)
    backend = get_backend(data1)
    Hp = Int(halo_width)
    @inbounds for panel in 1:6
        cache = field.panels[panel].data
        kernel! = _window_pbl_kz_cs_panel_kernel!(backend)
        kernel!(cache, air_mass[panel], surface.pblh[panel],
                surface.ustar[panel], surface.hflux[panel],
                surface.t2m[panel], areas, field.params, Hp;
                ndrange = (size(cache, 1), size(cache, 2)))
    end
    synchronize(backend)
    return true
end

"""
    refresh_pbl_kz_cache!(field, surface, air_mass, cell_areas; halo_width)

Recompute `field` from a window's raw PBL surface forcing and dry air mass.
`air_mass` may be halo-padded; `halo_width` selects the interior. The computed
host cache is copied back into the field panels, which may be CPU or device
arrays.
"""
function refresh_pbl_kz_cache!(field::WindowPBLKzField{FT},
                               surface,
                               air_mass::NTuple{6},
                               cell_areas::AbstractMatrix;
                               halo_width::Integer = 0) where FT
    surface === nothing &&
        throw(ArgumentError("[diffusion] kind=\"pbl\" requires pblh/ustar/hflux/t2m surface fields in the transport window"))
    if _try_refresh_pbl_kz_cache_backend!(field, surface, air_mass, cell_areas;
                                          halo_width = halo_width)
        return field
    end
    Hp = Int(halo_width)
    areas = FT.(_host_array(cell_areas))
    p = field.params
    R_dry = p.cp_dry / FT(CP_OVER_R_DIATOMIC)

    @inbounds for panel in 1:6
        cache = field.host_cache[panel]
        mhost = _host_array(air_mass[panel])
        pblh  = _surface_panel(surface, :pblh,  panel)
        ustar = _surface_panel(surface, :ustar, panel)
        hflux = _surface_panel(surface, :hflux, panel)
        t2m   = _surface_panel(surface, :t2m,   panel)
        Nc, Ny, Nz = size(cache)

        for j in 1:Ny, i in 1:Nc
            h_pbl = max(FT(pblh[i, j]),  FT(100))
            us    = max(FT(ustar[i, j]), FT(0.01))
            H_sfc = FT(hflux[i, j])
            T_sfc = max(FT(t2m[i, j]),   FT(200))

            L_ob, H_kin = _obukhov_length(H_sfc, us, T_sfc, p)
            Pr_inv = _prandtl_inverse(h_pbl, us, H_kin, T_sfc, L_ob, p)
            R_T_over_g = R_dry * T_sfc / p.gravity

            z_col = zero(FT)
            p_top = zero(FT)
            for k in 1:Nz
                delp_k = FT(mhost[i + Hp, j + Hp, k]) * p.gravity / areas[i, j]
                p_bot = p_top + delp_k
                p_mid = max((p_top + p_bot) / FT(2), FT(1))
                z_col += delp_k * R_T_over_g / p_mid
                p_top = p_bot
            end

            z_above = z_col
            p_top = zero(FT)
            for k in 1:Nz
                delp_k = FT(mhost[i + Hp, j + Hp, k]) * p.gravity / areas[i, j]
                p_bot = p_top + delp_k
                p_mid = max((p_top + p_bot) / FT(2), FT(1))
                dz_k = delp_k * R_T_over_g / p_mid
                z_center = z_above - dz_k / FT(2)
                cache[i, j, k] = _beljaars_viterbo_kz(z_center, h_pbl, us,
                                                      L_ob, Pr_inv, p)
                z_above -= dz_k
                p_top = p_bot
            end
        end

        copyto!(field.panels[panel].data, cache)
    end
    return field
end

export WindowPBLKzField, refresh_pbl_kz_cache!
