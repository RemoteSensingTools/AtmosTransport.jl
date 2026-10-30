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
                        P <: PBLPhysicsParameters{FT}, A} <: AbstractWindowKzCache{FT}
    panels     :: NTuple{6, F}
    host_cache :: H
    params     :: P
    area_cache :: A
end

function _typed_area_cache_ref(::Type{FT}, reference, nx::Integer, ny::Integer) where FT
    probe = similar(reference, FT, Int(nx), Int(ny))
    return Ref{Union{Nothing, typeof(probe)}}(nothing)
end

_pbl_host_array(a::Array) = a
_pbl_host_array(a) = Array(a)

@inline function _surface_panel(surface, name::Symbol, p::Int)
    return _pbl_host_array(getfield(surface, name)[p])
end

@kernel function _window_pbl_kz_cs_panel_kernel!(cache, @Const(air_mass),
                                                 @Const(pblh), @Const(ustar),
                                                 @Const(hflux), @Const(t2m),
                                                 @Const(areas), params, Hp)
    i, j = @index(Global, NTuple)
    _window_pbl_column!(cache, air_mass, pblh, ustar, hflux, t2m, areas, params, Hp, i, j)
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
    if _try_refresh_window_kz_backend!(field, surface, nothing, air_mass, cell_areas;
                                          halo_width = halo_width)
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
        Nc, Ny, Nz = size(cache)

        for j in 1:Ny, i in 1:Nc
            _window_pbl_column!(cache, mhost, pblh, ustar, hflux, t2m, areas, p, Hp, i, j)
        end

        copyto!(field.panels[panel].data, cache)
    end
    return field
end

export WindowPBLKzField, refresh_pbl_kz_cache!
