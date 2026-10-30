function (::Type{F})(host_cache::NTuple{6, Array{FT, 3}};
                          params = PBLPhysicsParameters{FT}()) where {FT, F <: AbstractWindowKzCache}
    params isa PBLPhysicsParameters{FT} ||
        throw(ArgumentError("params must be a PBLPhysicsParameters{$FT}; got $(typeof(params))"))
    panels = ntuple(p -> PreComputedKzField(host_cache[p]), 6)
    area_cache = _typed_area_cache_ref(FT, host_cache[1], size(host_cache[1], 1),
                                       size(host_cache[1], 2))
    return F{FT, typeof(panels[1]), typeof(host_cache),
                            typeof(params), typeof(area_cache)}(
        panels, host_cache, params, area_cache)
end

@inline panel_field(f::AbstractWindowKzCache, p::Integer) = f.panels[Int(p)]
# Window-constant cadence by design — see struct docstring.
update_field!(f::AbstractWindowKzCache, ::Real) = f

function Adapt.adapt_structure(to, f::AbstractWindowKzCache)
    panels = Adapt.adapt(to, f.panels)
    data1 = panels[1].data
    area_cache = _typed_area_cache_ref(_window_cache_eltype(f), data1,
                                       size(data1, 1), size(data1, 2))
    return _window_cache_type(f){_window_cache_eltype(f), typeof(panels[1]),
                            typeof(f.host_cache), typeof(f.params),
                            typeof(area_cache)}(
        panels, f.host_cache, f.params, area_cache)
end

@inline _window_cache_eltype(::AbstractWindowKzCache{FT}) where FT = FT

_window_cache_type(::WindowPBLKzField) = WindowPBLKzField
_window_cache_type(::LocalHoltslagBovilleKzField) = LocalHoltslagBovilleKzField

function _cached_backend_cell_areas!(field::AbstractWindowKzCache{FT},
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


function _try_refresh_window_kz_backend!(field::AbstractWindowKzCache{FT},
                                            surface, vdiff,
                                            air_mass::NTuple{6},
                                            cell_areas::AbstractMatrix;
                                            halo_width::Integer = 0) where FT
    data1 = field.panels[1].data
    (_backend_ready_array(data1) &&
     all(p -> _backend_ready_array(air_mass[p]), 1:6) &&
     _backend_ready_surface(surface) &&
     _window_vdiff_ready(field, vdiff)) || return false

    areas = _cached_backend_cell_areas!(field, cell_areas, data1)
    backend = get_backend(data1)
    Hp = Int(halo_width)
    @inbounds for panel in 1:6
        cache = field.panels[panel].data
        kernel! = _window_kz_kernel(field)(backend)
        kernel!(cache, air_mass[panel], surface.pblh[panel],
                surface.ustar[panel], surface.hflux[panel], surface.t2m[panel],
                _window_vdiff_args(field, vdiff, panel)..., areas, field.params, Hp;
                ndrange = (size(cache, 1), size(cache, 2)))
    end
    synchronize(backend)
    return true
end

_window_kz_kernel(::WindowPBLKzField) = _window_pbl_kz_cs_panel_kernel!
_window_kz_kernel(::LocalHoltslagBovilleKzField) = _local_hb_kz_cs_panel_kernel!
_window_vdiff_ready(::WindowPBLKzField, _) = true
_window_vdiff_ready(::LocalHoltslagBovilleKzField, vdiff) = _vdiff_ready(vdiff)
_window_vdiff_args(::WindowPBLKzField, _, p) = ()
_window_vdiff_args(::LocalHoltslagBovilleKzField, vdiff, p) =
    (vdiff.u[p], vdiff.v[p], vdiff.t[p], vdiff.qv[p])
