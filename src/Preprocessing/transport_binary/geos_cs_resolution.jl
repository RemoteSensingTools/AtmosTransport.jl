# GEOS native CS preprocessing: global dry-mass pin, panel-convention check, native → target resolution strategies and payloads.
# Split from cubed_sphere_geos.jl (refactor phase 4); included by Preprocessing.jl in this order.

function _cs_total_air_mass(panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray})
    total = 0.0
    for p in 1:CS_PANEL_COUNT
        total += sum(Float64, panels_m[p])
    end
    return total
end

function _cs_total_area(cell_areas::AbstractMatrix)
    return CS_PANEL_COUNT * sum(Float64, cell_areas)
end

"""
    _pin_cs_global_air_mass!(panels_m, cell_areas, g, target_kg)

Apply a uniform dry-surface-pressure offset to a cubed-sphere dry-air mass
state so its global mass equals `target_kg`. The column offset is distributed
vertically in proportion to each column's existing dry layer mass, preserving
the endpoint's vertical shape while removing the nonphysical global mean.
"""
function _pin_cs_global_air_mass!(panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray{FT, 3}},
                                  cell_areas::AbstractMatrix,
                                  g::FT,
                                  target_kg::Real) where FT
    current = _cs_total_air_mass(panels_m)
    area_total = _cs_total_area(cell_areas)
    delta_kg = Float64(target_kg) - current
    delta_ps = delta_kg * Float64(g) / area_total
    Nz = size(panels_m[1], 3)

    @inbounds for p in 1:CS_PANEL_COUNT
        m = panels_m[p]
        for j in axes(m, 2), i in axes(m, 1)
            delta_col = delta_ps * Float64(cell_areas[i, j]) / Float64(g)
            col = 0.0
            for k in 1:Nz
                col += Float64(m[i, j, k])
            end
            if col > 0.0
                for k in 1:Nz
                    m[i, j, k] = FT(Float64(m[i, j, k]) + delta_col * Float64(m[i, j, k]) / col)
                end
            else
                per_layer = delta_col / Nz
                for k in 1:Nz
                    m[i, j, k] = FT(Float64(m[i, j, k]) + per_layer)
                end
            end
        end
    end

    final = _cs_total_air_mass(panels_m)
    return (before_kg = current,
            after_kg = final,
            target_kg = Float64(target_kg),
            delta_ps_pa = delta_ps,
            residual_kg = final - Float64(target_kg))
end

_validate_geos_native_panel_convention(::GEOSNativePanelConvention) = nothing
function _validate_geos_native_panel_convention(conv)
    error("GEOS-CS passthrough requires panel_convention=`geos_native` on " *
          "the target geometry; got $(typeof(conv)).")
end

_geos_next_endpoint_available(handles::GEOSDayHandles) =
    handles.next_ctm_i1 !== nothing
_geos_next_endpoint_available(handles::GEOSFPNativeDayHandles) =
    handles.next_ctm !== nothing

abstract type AbstractGEOSCSResolutionStrategy end
struct GEOSCSIdentityStrategy <: AbstractGEOSCSResolutionStrategy end
struct GEOSCSBlockCoarsenStrategy{R} <: AbstractGEOSCSResolutionStrategy end

_geos_cs_resolution_strategy(settings::AbstractGEOSSettings, grid::CubedSphereTargetGeometry) =
    _geos_cs_resolution_strategy(Val(settings.Nc), Val(grid.Nc))

_geos_cs_resolution_strategy(::Val{N}, ::Val{N}) where {N} =
    GEOSCSIdentityStrategy()

function _geos_cs_resolution_strategy(::Val{Nsrc}, ::Val{Ndst}) where {Nsrc, Ndst}
    (Nsrc > Ndst && Nsrc % Ndst == 0) ||
        throw(ArgumentError(
            "GEOS-CS conversion supports native passthrough or nested block coarsening only; " *
            "source Nc=$(Nsrc), target Nc=$(Ndst)."))
    return GEOSCSBlockCoarsenStrategy{Nsrc ÷ Ndst}()
end

_geos_cs_strategy_name(::GEOSCSIdentityStrategy) = "identity"
_geos_cs_strategy_name(::GEOSCSBlockCoarsenStrategy{R}) where {R} =
    "nested_block_coarsen_$(R)x$(R)"

function _coarsen_sum3!(dst::AbstractArray{FT, 3},
                        src::AbstractArray{FT, 3},
                        ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            s += src[ii, jj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _coarsen_area_weighted2!(dst::AbstractMatrix{FT},
                                  src::AbstractMatrix{FT},
                                  src_area::AbstractMatrix{FT},
                                  ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        num = zero(FT)
        den = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            a = src_area[ii, jj]
            num += src[ii, jj] * a
            den += a
        end
        dst[i, j] = num / den
    end
    return dst
end

function _coarsen_area_weighted3!(dst::AbstractArray{FT, 3},
                                  src::AbstractArray{FT, 3},
                                  src_area::AbstractMatrix{FT},
                                  ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        num = zero(FT)
        den = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            a = src_area[ii, jj]
            num += src[ii, jj, k] * a
            den += a
        end
        dst[i, j, k] = num / den
    end
    return dst
end

function _coarsen_xface_sum!(dst::AbstractArray{FT, 3},
                             src::AbstractArray{FT, 3},
                             ::Val{R}) where {FT, R}
    Nc = size(dst, 2)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        fi = (i - 1) * R + 1
        s = zero(FT)
        for fj in ((j - 1) * R + 1):(j * R)
            s += src[fi, fj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _coarsen_yface_sum!(dst::AbstractArray{FT, 3},
                             src::AbstractArray{FT, 3},
                             ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        fj = (j - 1) * R + 1
        s = zero(FT)
        for fi in ((i - 1) * R + 1):(i * R)
            s += src[fi, fj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _geos_strategy_workspace(::GEOSCSIdentityStrategy,
                                  settings::AbstractGEOSSettings,
                                  grid::CubedSphereTargetGeometry,
                                  ::Type{FT}, Nz_native::Int,
                                  Nz_out::Int) where FT
    return nothing
end

function _geos_strategy_workspace(::GEOSCSBlockCoarsenStrategy{R},
                                  settings::AbstractGEOSSettings,
                                  grid::CubedSphereTargetGeometry,
                                  ::Type{FT}, Nz_native::Int,
                                  Nz_out::Int) where {FT, R}
    source_mesh = source_grid(settings; FT = FT)
    Nsrc = settings.Nc
    Nc = grid.Nc
    panels_3d_src() = ntuple(_ -> zeros(FT, Nsrc, Nsrc, Nz_native), CS_PANEL_COUNT)
    panels_xface_src() = ntuple(_ -> zeros(FT, Nsrc + 1, Nsrc, Nz_native), CS_PANEL_COUNT)
    panels_yface_src() = ntuple(_ -> zeros(FT, Nsrc, Nsrc + 1, Nz_native), CS_PANEL_COUNT)
    panels_2d_dst() = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    panels_3d_dst(nlev) = ntuple(_ -> zeros(FT, Nc, Nc, nlev), CS_PANEL_COUNT)
    return (
        source_mesh = source_mesh,
        source_cell_areas = source_mesh.cell_areas,
        fine_m_kg = panels_3d_src(),
        fine_am_v4 = panels_xface_src(),
        fine_bm_v4 = panels_yface_src(),
        cmfmc_native = settings.include_convection ? panels_3d_dst(Nz_native + 1) : nothing,
        dtrain_native = settings.include_convection ? panels_3d_dst(Nz_native) : nothing,
        surface = (settings.include_surface || settings.include_vdiff_fields) ? (
            pblh = panels_2d_dst(),
            ustar = panels_2d_dst(),
            hflux = panels_2d_dst(),
            t2m = panels_2d_dst(),
        ) : nothing,
        vdiff_native = settings.include_vdiff_fields ? (
            u = panels_3d_dst(Nz_native),
            v = panels_3d_dst(Nz_native),
            t = panels_3d_dst(Nz_native),
            qv = panels_3d_dst(Nz_native),
        ) : nothing,
        vdiff_weights_native = settings.include_vdiff_fields ?
            panels_3d_dst(Nz_native) : nothing,
    )
end

function _geos_fluxes_to_target!(::GEOSCSIdentityStrategy, _ws,
                                 am_v4, bm_v4, raw, grid,
                                 Nc::Int, Nz::Int, flux_scale)
    geos_native_to_face_flux!(am_v4, bm_v4, raw.am, raw.bm,
                              grid.mesh.connectivity, Nc, Nz, flux_scale)
    return nothing
end

function _geos_fluxes_to_target!(::GEOSCSBlockCoarsenStrategy{R}, ws,
                                 am_v4, bm_v4, raw, _grid,
                                 Nc::Int, Nz::Int, flux_scale) where R
    Nsrc = Nc * R
    geos_native_to_face_flux!(ws.fine_am_v4, ws.fine_bm_v4, raw.am, raw.bm,
                              ws.source_mesh.connectivity, Nsrc, Nz, flux_scale)
    for p in 1:CS_PANEL_COUNT
        _coarsen_xface_sum!(am_v4[p], ws.fine_am_v4[p], Val(R))
        _coarsen_yface_sum!(bm_v4[p], ws.fine_bm_v4[p], Val(R))
    end
    return nothing
end

function _geos_seed_mass!(::GEOSCSIdentityStrategy, _ws, m_cur, raw,
                          cell_areas, inv_g, Nc::Int, Nz::Int)
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(m_cur[p], raw.m[p], cell_areas, inv_g)
    end
    return nothing
end

function _geos_seed_mass!(::GEOSCSBlockCoarsenStrategy{R}, ws, m_cur, raw,
                          _cell_areas, inv_g, _Nc::Int, _Nz::Int) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m[p], ws.source_cell_areas, inv_g)
        _coarsen_sum3!(m_cur[p], ws.fine_m_kg[p], Val(R))
    end
    return nothing
end

function _geos_target_mass!(::GEOSCSIdentityStrategy, _ws, m_next, raw,
                            cell_areas, inv_g, Nc::Int, Nz::Int)
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(m_next[p], raw.m_next[p], cell_areas, inv_g)
    end
    return nothing
end

function _geos_target_mass!(::GEOSCSBlockCoarsenStrategy{R}, ws, m_next, raw,
                            _cell_areas, inv_g, _Nc::Int, _Nz::Int) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m_next[p], ws.source_cell_areas, inv_g)
        _coarsen_sum3!(m_next[p], ws.fine_m_kg[p], Val(R))
    end
    return nothing
end

_geos_surface_payload!(::GEOSCSIdentityStrategy, _ws, raw) = raw.surface

function _geos_surface_payload!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.surface === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted2!(ws.surface.pblh[p], raw.surface.pblh[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.ustar[p], raw.surface.ustar[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.hflux[p], raw.surface.hflux[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.t2m[p], raw.surface.t2m[p], ws.source_cell_areas, Val(R))
    end
    return ws.surface
end

_geos_vdiff_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.vdiff
_geos_vdiff_native_weights!(::GEOSCSIdentityStrategy, _ws, raw,
                            _cell_areas, _inv_g) = raw.m

function _geos_vdiff_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.vdiff === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.vdiff_native.u[p], raw.vdiff.u[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.v[p], raw.vdiff.v[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.t[p], raw.vdiff.t[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.qv[p], raw.vdiff.qv[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.vdiff_native
end

function _geos_vdiff_native_weights!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw,
                                     _cell_areas, inv_g) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m[p],
                                  ws.source_cell_areas, inv_g)
        _coarsen_sum3!(ws.vdiff_weights_native[p], ws.fine_m_kg[p], Val(R))
    end
    return ws.vdiff_weights_native
end

_geos_cmfmc_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.cmfmc
_geos_dtrain_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.dtrain

function _geos_cmfmc_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.cmfmc === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.cmfmc_native[p], raw.cmfmc[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.cmfmc_native
end

function _geos_dtrain_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.dtrain === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.dtrain_native[p], raw.dtrain[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.dtrain_native
end

function _geos_cmfmc_payload!(workspace)
    workspace.cmfmc_v4 === nothing && return nothing
    native = _geos_cmfmc_native_target!(workspace.strategy, workspace.strategy_ws,
                                        workspace.raw)
    native === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.cmfmc_v4[p], native[p], workspace.plan,
                        ConvectionInterfaceFlux())
    end
    return workspace.cmfmc_v4
end

function _geos_dtrain_payload!(workspace)
    workspace.dtrain_v4 === nothing && return nothing
    native = _geos_dtrain_native_target!(workspace.strategy, workspace.strategy_ws,
                                         workspace.raw)
    native === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.dtrain_v4[p], native[p], workspace.plan,
                        ConvectionTendencyField())
    end
    return workspace.dtrain_v4
end

function _geos_vdiff_payload!(workspace)
    workspace.vdiff_v4 === nothing && return nothing
    native = _geos_vdiff_native_target!(workspace.strategy, workspace.strategy_ws,
                                        workspace.raw)
    native === nothing && return nothing
    weights = _geos_vdiff_native_weights!(workspace.strategy, workspace.strategy_ws,
                                          workspace.raw, workspace.cell_areas,
                                          workspace.inv_g)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.vdiff_v4.u[p], native.u[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.v[p], native.v[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.t[p], native.t[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.qv[p], native.qv[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
    end
    return workspace.vdiff_v4
end
