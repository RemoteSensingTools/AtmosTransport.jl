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
    _pin_cs_global_air_mass!(panels_m, panels_qv, cell_areas, g, target_kg, dB)

Apply a uniform moist-surface-pressure offset to a cubed-sphere dry-air mass
state so its global mass equals `target_kg`. For layer `k`, the correction is

```math
\\Delta m_k = \\Delta B_k\\,\\Delta p_s\\,(1-q_k)\\,A/g.
```

This is the same hybrid-coordinate correction as the spectral path: layers
with `\\Delta B_k = 0` are unchanged. `panels_qv` and `dB` are on the same
target vertical grid as `panels_m`.
"""
function _pin_cs_global_air_mass!(panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray{FT, 3}},
                                  panels_qv::NTuple{CS_PANEL_COUNT, <:AbstractArray{<:Real, 3}},
                                  cell_areas::AbstractMatrix,
                                  g::FT,
                                  target_kg::Real,
                                  dB::AbstractVector) where FT
    current = _cs_total_air_mass(panels_m)
    Nz = size(panels_m[1], 3)
    length(dB) == Nz || throw(DimensionMismatch(
        "native dry-mass pin dB length $(length(dB)) does not match Nz=$Nz"))
    sensitivity_kg_per_pa = 0.0
    @inbounds for p in 1:CS_PANEL_COUNT
        m = panels_m[p]
        qv = panels_qv[p]
        size(qv) == size(m) || throw(DimensionMismatch(
            "native dry-mass pin qv panel $p shape $(size(qv)) does not match mass $(size(m))"))
        for k in 1:Nz, j in axes(m, 2), i in axes(m, 1)
            q = clamp(Float64(qv[i, j, k]), 0.0, 0.999999)
            sensitivity_kg_per_pa += Float64(dB[k]) * (1.0 - q) *
                                     Float64(cell_areas[i, j]) / Float64(g)
        end
    end
    delta_ps = _mass_target_offset(target_kg, current, sensitivity_kg_per_pa)

    @inbounds for p in 1:CS_PANEL_COUNT
        m = panels_m[p]
        qv = panels_qv[p]
        for k in 1:Nz, j in axes(m, 2), i in axes(m, 1)
            q = clamp(Float64(qv[i, j, k]), 0.0, 0.999999)
            delta_m = Float64(dB[k]) * delta_ps * (1.0 - q) *
                      Float64(cell_areas[i, j]) / Float64(g)
            m[i, j, k] = FT(Float64(m[i, j, k]) + delta_m)
        end
    end

    final = _cs_total_air_mass(panels_m)
    return (before_kg = current,
            after_kg = final,
            target_kg = Float64(target_kg),
            delta_ps_pa = delta_ps,
            residual_kg = final - Float64(target_kg))
end

"""
    _pin_cs_global_air_mass_from_sensitivity!(panels_m, sensitivity, target_kg)

Apply the global pin using a precomputed extensive sensitivity
`dm/dps = dB * (1-q) * area/g` in kg/Pa. Building this field on native levels
and summing it through a vertical transform preserves humidity covariance
inside merged hybrid layers.
"""
function _pin_cs_global_air_mass_from_sensitivity!(
        panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray{FT, 3}},
        sensitivity::NTuple{CS_PANEL_COUNT, <:AbstractArray{<:Real, 3}},
        target_kg::Real) where FT
    for p in 1:CS_PANEL_COUNT
        size(sensitivity[p]) == size(panels_m[p]) || throw(DimensionMismatch(
            "native dry-mass pin sensitivity panel $p shape $(size(sensitivity[p])) " *
            "does not match mass $(size(panels_m[p]))"))
    end
    current = _cs_total_air_mass(panels_m)
    sensitivity_kg_per_pa = _cs_total_air_mass(sensitivity)
    delta_ps = _mass_target_offset(target_kg, current, sensitivity_kg_per_pa)
    for p in 1:CS_PANEL_COUNT
        @. panels_m[p] += FT(delta_ps) * sensitivity[p]
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

_geos_next_endpoint_available(handles::Union{GEOSDayHandles, GEOSFPNativeDayHandles}) =
    handles.next_ctm_i1 !== nothing

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

# Block kernels of the coarsen strategy: cell sums and area-weighted means are
# `Regridding.coarsen_sum!` / `coarsen_area_weighted!` (Float64 accumulation and
# one rounding for the means), face sums `_coarsen_xface_sum!` /
# `_coarsen_yface_sum!` (`cubed_sphere_coarsen.jl`). The ratio `R` of the
# strategy type is passed as an `Int`.

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
            eflux = has_pbl_eflux(settings) ? panels_2d_dst() : nothing,
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
        _coarsen_xface_sum!(am_v4[p], ws.fine_am_v4[p], R)
        _coarsen_yface_sum!(bm_v4[p], ws.fine_bm_v4[p], R)
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
        coarsen_sum!(m_cur[p], ws.fine_m_kg[p], R)
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
        coarsen_sum!(m_next[p], ws.fine_m_kg[p], R)
    end
    return nothing
end

# Dry-mass sensitivity to a uniform moist surface-pressure shift. It is built
# on the source grid as ΔB(1-q)A/g [kg Pa⁻¹], then conservatively coarsened.
# The caller subsequently applies the vertical plan by summing layers.
function _geos_pin_sensitivity_native!(::GEOSCSIdentityStrategy, _ws,
                                       sensitivity_native, qv, dB,
                                       cell_areas, inv_g)
    for p in 1:CS_PANEL_COUNT
        s = sensitivity_native[p]
        q = qv[p]
        @inbounds for k in axes(s, 3), j in axes(s, 2), i in axes(s, 1)
            s[i, j, k] = eltype(s)(Float64(dB[k]) *
                (1.0 - clamp(Float64(q[i, j, k]), 0.0, 0.999999)) *
                Float64(cell_areas[i, j]) * Float64(inv_g))
        end
    end
    return sensitivity_native
end

function _geos_pin_sensitivity_native!(::GEOSCSBlockCoarsenStrategy{R}, ws,
                                       sensitivity_native, qv, dB,
                                       _cell_areas, inv_g) where R
    for p in 1:CS_PANEL_COUNT
        fine = ws.fine_m_kg[p]
        q = qv[p]
        @inbounds for k in axes(fine, 3), j in axes(fine, 2), i in axes(fine, 1)
            fine[i, j, k] = eltype(fine)(Float64(dB[k]) *
                (1.0 - clamp(Float64(q[i, j, k]), 0.0, 0.999999)) *
                Float64(ws.source_cell_areas[i, j]) * Float64(inv_g))
        end
        coarsen_sum!(sensitivity_native[p], fine, R)
    end
    return sensitivity_native
end

_geos_surface_payload!(::GEOSCSIdentityStrategy, _ws, raw) = raw.surface

function _geos_surface_payload!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.surface === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        coarsen_area_weighted!(ws.surface.pblh[p], raw.surface.pblh[p], ws.source_cell_areas, R)
        coarsen_area_weighted!(ws.surface.ustar[p], raw.surface.ustar[p], ws.source_cell_areas, R)
        coarsen_area_weighted!(ws.surface.hflux[p], raw.surface.hflux[p], ws.source_cell_areas, R)
        coarsen_area_weighted!(ws.surface.t2m[p], raw.surface.t2m[p], ws.source_cell_areas, R)
        ws.surface.eflux === nothing ||
            coarsen_area_weighted!(ws.surface.eflux[p], raw.surface.eflux[p], ws.source_cell_areas, R)
    end
    return ws.surface
end

_geos_vdiff_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.vdiff
_geos_vdiff_native_weights!(::GEOSCSIdentityStrategy, _ws, raw,
                            _cell_areas, _inv_g) = raw.m

function _geos_vdiff_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.vdiff === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        coarsen_area_weighted!(ws.vdiff_native.u[p], raw.vdiff.u[p],
                               ws.source_cell_areas, R)
        coarsen_area_weighted!(ws.vdiff_native.v[p], raw.vdiff.v[p],
                               ws.source_cell_areas, R)
        coarsen_area_weighted!(ws.vdiff_native.t[p], raw.vdiff.t[p],
                               ws.source_cell_areas, R)
        coarsen_area_weighted!(ws.vdiff_native.qv[p], raw.vdiff.qv[p],
                               ws.source_cell_areas, R)
    end
    return ws.vdiff_native
end

function _geos_vdiff_native_weights!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw,
                                     _cell_areas, inv_g) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m[p],
                                  ws.source_cell_areas, inv_g)
        coarsen_sum!(ws.vdiff_weights_native[p], ws.fine_m_kg[p], R)
    end
    return ws.vdiff_weights_native
end

_geos_cmfmc_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.cmfmc
_geos_dtrain_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.dtrain

function _geos_cmfmc_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.cmfmc === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        coarsen_area_weighted!(ws.cmfmc_native[p], raw.cmfmc[p],
                               ws.source_cell_areas, R)
    end
    return ws.cmfmc_native
end

function _geos_dtrain_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.dtrain === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        coarsen_area_weighted!(ws.dtrain_native[p], raw.dtrain[p],
                               ws.source_cell_areas, R)
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
