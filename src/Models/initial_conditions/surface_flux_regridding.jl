function _renormalize_surface_flux_rate!(rate::AbstractArray{FT}, source::FileSurfaceFluxField) where FT
    isfinite(source.native_total_mass_rate) || return rate
    sampled_total = Float64(sum(rate))
    sampled_total > 0 || return rate
    scale = source.native_total_mass_rate / sampled_total
    rate .*= FT(scale)
    return rate
end

"""Parse regridding method from config, with a caller-owned default."""
_regridding_method(cfg, default::AbstractString = "bilinear") =
    Symbol(lowercase(String(get(cfg, "regridding", default))))

"""
    _build_surface_flux_regridder(lon, lat, dst_mesh) -> regridder

Build the conservative LL→`dst_mesh` regridder for a source flux grid
defined by `lon`/`lat`, with Float64 geometry for every run precision.
Factored out so the time-varying path can build the regridder once and
reuse it across every time slice instead of rebuilding it per slice.
"""
function _build_surface_flux_regridder(lon::Vector{Float64}, lat::Vector{Float64},
                                       dst_mesh::AbstractHorizontalMesh)
    src_mesh = _build_source_latlon_mesh(lon, lat; radius = dst_mesh.radius)
    return build_regridder(src_mesh, dst_mesh; cache_dir = _regrid_cache_dir())
end

"""
    _apply_surface_flux_regridder(regridder, raw2d, FT; report=true) -> Vector{FT}

Conservatively regrid one 2-D flux-density slice `raw2d` [kg/m²/s] using a
pre-built `regridder`, returning a flat per-cell mass rate [kg/s]
(area-integrated by the regridder's `dst_areas`).
"""
function _apply_surface_flux_regridder(regridder, raw2d::AbstractMatrix, ::Type{FT};
                                       report::Bool = true) where FT
    src_flat = vec(Float64.(raw2d))
    n_dst = length(regridder.dst_areas)
    dst_flat = zeros(Float64, n_dst)
    apply_regridder!(dst_flat, regridder, src_flat)

    # Convert flux density [kg/m²/s] → mass rate [kg/s] using regridder areas
    rate = Array{FT}(undef, n_dst)
    for c in 1:n_dst
        rate[c] = FT(dst_flat[c] * regridder.dst_areas[c])
    end

    if report
        # Report global mass conservation (warn only; conservative regrid is exact to ~FP ulps)
        src_total = sum(src_flat .* regridder.src_areas)
        dst_total = sum(Float64.(rate))
        rel_err = abs(dst_total - src_total) / max(abs(src_total), 1e-30)
        @info @sprintf("  Conservative regrid: src_total=%.6e  dst_total=%.6e  rel_err=%.2e kg/s",
                       src_total, dst_total, rel_err)
        rel_err > 1e-6 && @warn @sprintf("  Conservative regrid mass conservation warning: rel_err=%.2e", rel_err)
    end

    return rate
end

"""
    _conservative_surface_flux_rate(source, dst_mesh, FT) -> Vector{FT}

Conservatively regrid flux density [kg/m²/s] onto `dst_mesh`; return a
flat vector of per-cell mass rates [kg/s] (already area-integrated).
Callers reshape/wrap per topology.
"""
function _conservative_surface_flux_rate(source::FileSurfaceFluxField,
                                         dst_mesh::AbstractHorizontalMesh,
                                         ::Type{FT}) where FT
    regridder = _build_surface_flux_regridder(source.lon, source.lat, dst_mesh)
    return _apply_surface_flux_regridder(regridder, source.raw, FT)
end
