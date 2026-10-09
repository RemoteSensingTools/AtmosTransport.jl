# GEOS native reader: the canonical read_window! surface and the 3-hourly convection binding.
# Split from geos.jl (refactor phase 4); included by Preprocessing.jl in this order.

# ---------------------------------------------------------------------------
# Canonical AbstractMetSettings interface implementations.
# ---------------------------------------------------------------------------

"""
    open_day(settings::GEOSSettings, date::Date; next_day_handle=true) -> GEOSDayHandles

Canonical-contract alias for `open_geos_day`. The orchestrator calls this
once per day and threads the returned handles through every per-window
`read_window!`.
"""
open_day(settings::GEOSSettings{:geosit}, date::Date; next_day_handle::Bool=true,
         adjacent_omega::Bool=false) =
    open_geos_day(settings, date; next_day_handle=next_day_handle,
                  adjacent_omega=adjacent_omega)
# GEOS-FP native has no `:omega_consistent` path yet (it errors on
# include_vdiff_fields); accept the kwarg for a uniform trait surface and ignore.
open_day(settings::GEOSSettings{:geosfp}, date::Date; next_day_handle::Bool=true,
         adjacent_omega::Bool=false) =
    open_geosfp_native_day(settings, date; next_day_handle=next_day_handle)

"""Canonical-contract alias for `close_geos_day!`."""
close_day!(handles::GEOSDayHandles) = close_geos_day!(handles)
close_day!(handles::GEOSFPNativeDayHandles) = close_geosfp_native_day!(handles)

"""
    source_grid(settings::GEOSSettings) -> CubedSphereMesh

The native source mesh GEOS data is archived on (`Nc × Nc` per panel,
GEOS-native panel convention).
"""
function source_grid(settings::GEOSSettings; FT::Type{<:AbstractFloat}=Float64)
    return CubedSphereMesh(; Nc = settings.Nc, FT = FT,
                            convention = GEOSNativePanelConvention(),
                            radius = FT(IFS_EARTH_RADIUS))
end

"""
    allocate_raw_window(settings::GEOSSettings; FT, Nz) -> RawWindow

Pre-allocate a per-window workspace for the GEOS reader: 6 zero-filled
panel arrays each for `m`, `m_next`, `qv`, `qv_next`, `am`, `bm` (shape
`(Nc, Nc, Nz)`) and `ps`, `ps_next` (shape `(Nc, Nc)`).

When `settings.include_convection`, also allocates `cmfmc` (interfaces,
shape `(Nc, Nc, Nz + 1)` per panel) and `dtrain` (centers, shape
`(Nc, Nc, Nz)` per panel). Cross-topology winds (`u`, `v`) stay
`nothing` here — they are produced by the orchestrator only when the
target grid differs from the source.
"""
function allocate_raw_window(settings::GEOSSettings; FT::Type{<:AbstractFloat}, Nz::Int)
    Nc = settings.Nc
    panels_3d() = ntuple(_ -> zeros(FT, Nc, Nc, Nz),     6)
    panels_3d_iface() = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    panels_2d() = ntuple(_ -> zeros(FT, Nc, Nc),         6)
    m       = panels_3d(); ps      = panels_2d(); qv      = panels_3d()
    m_next  = panels_3d(); ps_next = panels_2d(); qv_next = panels_3d()
    am      = panels_3d(); bm      = panels_3d()
    need_surface = settings.include_surface || settings.include_vdiff_fields
    surface = need_surface ? (
        pblh  = panels_2d(),
        ustar = panels_2d(),
        hflux = panels_2d(),
        t2m   = panels_2d(),
    ) : nothing
    vdiff = settings.include_vdiff_fields ? (
        u  = panels_3d(),
        v  = panels_3d(),
        t  = panels_3d(),
        qv = panels_3d(),
    ) : nothing
    cmfmc   = settings.include_convection ? panels_3d_iface() : nothing
    dtrain  = settings.include_convection ? panels_3d()       : nothing
    return RawWindow{FT, typeof(ps), typeof(m)}(
        m,       ps,      qv,
        m_next,  ps_next, qv_next,
        am,      bm,
        nothing, nothing,
        surface, vdiff,
        cmfmc,   dtrain,
    )
end

"""
    read_window!(raw, settings, handles, date, win_idx) -> raw

    Fill `raw` in place with one window of GEOS data on the source CS grid.
    Both endpoints (t_n, t_{n+1}) carry dry DELP + dry PS reconstructed from
    PS_total via the hybrid coordinate, plus the original QV. Dynamics-step
    `am`/`bm` are MFXC/MFYC scaled by `1/mass_flux_dt`.

The signature matches the canonical `AbstractMetSettings` contract
declared in `met_sources.jl::read_window!`.
"""
function read_window!(raw::RawWindow{FT}, settings::GEOSSettings,
                      handles::GEOSDayHandles, date::Date, win_idx::Int) where {FT}
    nw = windows_per_day(settings, date)
    1 <= win_idx <= nw || error("window $win_idx out of range 1..$nw")

    or = handles.orientation
    vc = handles.vc

    # ---- Endpoints: PS (units-aware → Pa), QV ----
    ps_factor_today = _ps_pa_factor(handles.ctm_i1["PS"]; FT=FT)
    ps_n_total = ntuple(p -> _read_panels_2d(handles.ctm_i1, "PS", win_idx; FT=FT)[p] .* ps_factor_today, 6)
    qv_n_panels = _read_panels_3d(handles.ctm_i1, "QV", win_idx, or; FT=FT)

    if win_idx < nw
        ps_np1_total = ntuple(p -> _read_panels_2d(handles.ctm_i1, "PS", win_idx + 1; FT=FT)[p] .* ps_factor_today, 6)
        qv_np1_panels = _read_panels_3d(handles.ctm_i1, "QV", win_idx + 1, or; FT=FT)
    elseif handles.next_ctm_i1 !== nothing
        ps_factor_next = _ps_pa_factor(handles.next_ctm_i1["PS"]; FT=FT)
        ps_np1_total = ntuple(p -> _read_panels_2d(handles.next_ctm_i1, "PS", 1; FT=FT)[p] .* ps_factor_next, 6)
        qv_np1_panels = _read_panels_3d(handles.next_ctm_i1, "QV", 1, or; FT=FT)
    else
        error("last window ($win_idx of $nw) has no next-day CTM_I1 endpoint; " *
              "open the day with `next_day_handle=true` and ensure the next-day file is on disk")
    end

    # ---- Dynamics-step horizontal mass transport (dry, /mass_flux_dt) ----
    inv_dt = inv(FT(settings.mass_flux_dt))
    mfxc_raw = _read_panels_3d(handles.ctm_a1, "MFXC", win_idx, or; FT=FT)
    mfyc_raw = _read_panels_3d(handles.ctm_a1, "MFYC", win_idx, or; FT=FT)

    # ---- Fill `raw` in place. Endpoint dry-mass derivation lives in
    #      `endpoint_dry_mass!` and writes into the raw buffer directly. ----
    @assert raw.qv      !== nothing "GEOS RawWindow must carry qv"
    @assert raw.qv_next !== nothing "GEOS RawWindow must carry qv_next"
    for p in 1:6
        copyto!(raw.qv[p],      qv_n_panels[p])
        copyto!(raw.qv_next[p], qv_np1_panels[p])
        endpoint_dry_mass!(raw.m[p],      raw.ps[p],      ps_n_total[p],   raw.qv[p],      vc)
        endpoint_dry_mass!(raw.m_next[p], raw.ps_next[p], ps_np1_total[p], raw.qv_next[p], vc)
        _scale_flux!(raw.am[p], mfxc_raw[p], inv_dt)
        _scale_flux!(raw.bm[p], mfyc_raw[p], inv_dt)
    end

    if settings.include_surface || settings.include_vdiff_fields
        _read_geos_surface_window!(raw, handles, win_idx)
    end
    settings.include_vdiff_fields &&
        _read_geos_vdiff_window!(raw, handles, win_idx, or)

    # ---- Optional convection forcing (GCHP RAS / Grell-Freitas inputs) ----
    if settings.include_convection
        _read_geos_convection_window!(raw, handles, win_idx, or)
    end

    return raw
end

function read_window!(raw::RawWindow{FT}, settings::GEOSSettings{:geosfp},
                      handles::GEOSFPNativeDayHandles, date::Date,
                      win_idx::Int) where {FT}
    nw = windows_per_day(settings, date)
    1 <= win_idx <= nw || error("window $win_idx out of range 1..$nw")

    ds_n = handles.ctm[win_idx]
    ds_np1 = if win_idx < nw
        handles.ctm[win_idx + 1]
    elseif handles.next_ctm !== nothing
        handles.next_ctm
    else
        error("last GEOS-FP window ($win_idx of $nw) has no next-day hourly CTM endpoint")
    end

    or = handles.orientation
    vc = handles.vc

    ps_factor_n = _ps_pa_factor(ds_n["PS"]; FT=FT)
    ps_factor_np1 = _ps_pa_factor(ds_np1["PS"]; FT=FT)
    ps_n_total = ntuple(p -> _read_panels_2d(ds_n, "PS", 1; FT=FT)[p] .* ps_factor_n, 6)
    ps_np1_total = ntuple(p -> _read_panels_2d(ds_np1, "PS", 1; FT=FT)[p] .* ps_factor_np1, 6)
    qv_n_panels = _read_panels_3d(ds_n, "QV", 1, or; FT=FT)
    qv_np1_panels = _read_panels_3d(ds_np1, "QV", 1, or; FT=FT)

    inv_dt = inv(FT(settings.mass_flux_dt))
    mfxc_raw = _read_panels_3d(ds_n, "MFXC", 1, or; FT=FT)
    mfyc_raw = _read_panels_3d(ds_n, "MFYC", 1, or; FT=FT)

    @assert raw.qv      !== nothing "GEOS-FP RawWindow must carry qv"
    @assert raw.qv_next !== nothing "GEOS-FP RawWindow must carry qv_next"
    for p in 1:6
        copyto!(raw.qv[p],      qv_n_panels[p])
        copyto!(raw.qv_next[p], qv_np1_panels[p])
        endpoint_dry_mass!(raw.m[p],      raw.ps[p],      ps_n_total[p],   raw.qv[p],      vc)
        endpoint_dry_mass!(raw.m_next[p], raw.ps_next[p], ps_np1_total[p], raw.qv_next[p], vc)
        _scale_flux!(raw.am[p], mfxc_raw[p], inv_dt)
        _scale_flux!(raw.bm[p], mfyc_raw[p], inv_dt)
    end
    if settings.include_surface
        _read_geosfp_surface_window!(raw, handles.physics, win_idx)
    end
    if settings.include_convection
        _read_geosfp_convection_window!(raw, handles.physics, win_idx, or)
    end
    return raw
end

function _read_geos_surface_window!(raw::RawWindow{FT},
                                    handles::GEOSDayHandles,
                                    win_idx::Int) where {FT}
    handles.a1 === nothing &&
        error("settings.include_surface=true but A1 handle is missing; ensure the A1 collection is on disk")
    raw.surface === nothing &&
        error("RawWindow.surface must be allocated when surface output is enabled")

    pblh  = _read_panels_2d(handles.a1, "PBLH",  win_idx; FT=FT)
    ustar = _read_panels_2d(handles.a1, "USTAR", win_idx; FT=FT)
    hflux = _read_panels_2d(handles.a1, "HFLUX", win_idx; FT=FT)
    t2m   = _read_panels_2d(handles.a1, "T2M",   win_idx; FT=FT)

    for p in 1:6
        copyto!(raw.surface.pblh[p],  pblh[p])
        copyto!(raw.surface.ustar[p], ustar[p])
        copyto!(raw.surface.hflux[p], hflux[p])
        copyto!(raw.surface.t2m[p],   t2m[p])
    end
    return raw
end

function _read_geos_vdiff_window!(raw::RawWindow{FT},
                                  handles::GEOSDayHandles,
                                  win_idx::Int,
                                  orientation::Symbol) where {FT}
    handles.a3dyn === nothing &&
        error("settings.include_vdiff_fields=true but A3dyn handle is missing; VDIFF needs U/V")
    handles.i3 === nothing &&
        error("settings.include_vdiff_fields=true but I3 handle is missing; VDIFF needs T")
    raw.vdiff === nothing &&
        error("RawWindow.vdiff must be allocated when GCHP VDIFF fields are enabled")
    raw.qv === nothing &&
        error("RawWindow.qv must be allocated when GCHP VDIFF fields are enabled")

    # GEOS A3dyn and I3 are both 3-hourly instantaneous/averaged blocks in
    # this archive, so U/V and T intentionally share the same window block.
    a3_idx = _a3_index_for_window(win_idx)
    u = _read_panels_3d(handles.a3dyn, "U", a3_idx, orientation; FT=FT)
    v = _read_panels_3d(handles.a3dyn, "V", a3_idx, orientation; FT=FT)
    t = _read_panels_3d(handles.i3, "T", a3_idx, orientation; FT=FT)
    for p in 1:CS_PANEL_COUNT
        copyto!(raw.vdiff.u[p],  u[p])
        copyto!(raw.vdiff.v[p],  v[p])
        copyto!(raw.vdiff.t[p],  t[p])
        copyto!(raw.vdiff.qv[p], raw.qv[p])
    end
    _validate_geos_vdiff_panels!(raw.vdiff, win_idx)
    return raw
end

function _read_geosfp_surface_window!(raw::RawWindow{FT},
                                      ::GEOSFPNoPhysics,
                                      win_idx::Int) where {FT}
    error("GEOS-FP include_surface=true but no physics fallback reader was opened for window $(win_idx)")
end

function _read_geosfp_surface_window!(raw::RawWindow{FT},
                                      physics::GEOSFPCSPhysicsFallback,
                                      win_idx::Int) where {FT}
    physics.a1 === nothing &&
        error("GEOS-FP surface fallback requires A1 but no A1 handle is open")
    raw.surface === nothing &&
        error("RawWindow.surface must be allocated when GEOS-FP surface output is enabled")

    pblh_name  = _find_var_name(physics.a1, GEOS_SURFACE_VAR_CANDIDATES[:pblh])
    ustar_name = _find_var_name(physics.a1, GEOS_SURFACE_VAR_CANDIDATES[:ustar])
    hflux_name = _find_var_name(physics.a1, GEOS_SURFACE_VAR_CANDIDATES[:hflux])
    t2m_name   = _find_var_name(physics.a1, GEOS_SURFACE_VAR_CANDIDATES[:t2m])

    pblh  = _read_panels_2d(physics.a1, pblh_name,  win_idx; FT=FT)
    ustar = _read_panels_2d(physics.a1, ustar_name, win_idx; FT=FT)
    hflux = _read_panels_2d(physics.a1, hflux_name, win_idx; FT=FT)
    t2m   = _read_panels_2d(physics.a1, t2m_name,   win_idx; FT=FT)

    for p in 1:CS_PANEL_COUNT
        copyto!(raw.surface.pblh[p],  pblh[p])
        copyto!(raw.surface.ustar[p], ustar[p])
        copyto!(raw.surface.hflux[p], hflux[p])
        copyto!(raw.surface.t2m[p],   t2m[p])
    end
    _validate_geos_surface_panels!(raw.surface, something(physics.a1_path, "<GEOS-FP A1>"), win_idx)
    return raw
end

function _read_geosfp_surface_window!(raw::RawWindow{FT},
                                      physics::GEOSFPLatLonPhysicsFallback,
                                      win_idx::Int) where {FT}
    physics.a1 === nothing &&
        error("GEOS-FP lat-lon surface fallback requires A1 but no A1 handle is open")
    raw.surface === nothing &&
        error("RawWindow.surface must be allocated when GEOS-FP surface output is enabled")

    pblh_name  = _find_var_name(physics.a1, GEOS_SURFACE_VAR_CANDIDATES[:pblh])
    ustar_name = _find_var_name(physics.a1, GEOS_SURFACE_VAR_CANDIDATES[:ustar])
    hflux_name = _find_var_name(physics.a1, GEOS_SURFACE_VAR_CANDIDATES[:hflux])
    t2m_name   = _find_var_name(physics.a1, GEOS_SURFACE_VAR_CANDIDATES[:t2m])

    pblh_ll  = _read_latlon_slice(physics.a1, pblh_name,  win_idx, Val(2), FT)
    ustar_ll = _read_latlon_slice(physics.a1, ustar_name, win_idx, Val(2), FT)
    hflux_raw = _read_latlon_slice(physics.a1, hflux_name, win_idx, Val(2), FT)
    hflux_ll = _hflux_to_upward_wm2(hflux_raw, physics.a1, hflux_name, FT)
    t2m_ll   = _read_latlon_slice(physics.a1, t2m_name,   win_idx, Val(2), FT)

    _interpolate_ll_to_panels!(raw.surface.pblh,  pblh_ll,  physics)
    _interpolate_ll_to_panels!(raw.surface.ustar, ustar_ll, physics)
    _interpolate_ll_to_panels!(raw.surface.hflux, hflux_ll, physics)
    _interpolate_ll_to_panels!(raw.surface.t2m,   t2m_ll,   physics)
    _validate_geos_surface_panels!(raw.surface, something(physics.a1_path, "<GEOS-FP A1>"), win_idx)
    return raw
end

function _read_geosfp_convection_window!(raw::RawWindow{FT},
                                         ::GEOSFPNoPhysics,
                                         win_idx::Int,
                                         orientation::Symbol) where {FT}
    error("GEOS-FP include_convection=true but no physics fallback reader was opened for window $(win_idx)")
end

function _read_geosfp_convection_window!(raw::RawWindow{FT},
                                         physics::GEOSFPCSPhysicsFallback,
                                         win_idx::Int,
                                         orientation::Symbol) where {FT}
    physics.a3mste === nothing &&
        error("GEOS-FP convection fallback requires A3mstE but no handle is open")
    physics.a3dyn === nothing &&
        error("GEOS-FP convection fallback requires A3dyn but no handle is open")
    @assert raw.cmfmc  !== nothing "RawWindow.cmfmc must be allocated when convection is enabled"
    @assert raw.dtrain !== nothing "RawWindow.dtrain must be allocated when convection is enabled"

    a3_idx = _a3_index_for_window(win_idx)
    cmfmc_name = _find_var_name(physics.a3mste, GEOS_CONVECTION_VAR_CANDIDATES[:cmfmc])
    dtrain_name = _find_var_name(physics.a3dyn, GEOS_CONVECTION_VAR_CANDIDATES[:dtrain])
    cmfmc_raw  = _read_panels_3d(physics.a3mste, cmfmc_name,  a3_idx, orientation; FT=FT)
    dtrain_raw = _read_panels_3d(physics.a3dyn,  dtrain_name, a3_idx, orientation; FT=FT)
    for p in 1:CS_PANEL_COUNT
        copyto!(raw.cmfmc[p],  cmfmc_raw[p])
        copyto!(raw.dtrain[p], dtrain_raw[p])
    end
    _moist_to_dry_dtrain!(raw.dtrain, raw.qv, raw.qv_next)
    _moist_to_dry_cmfmc!(raw.cmfmc,  raw.qv, raw.qv_next)
    _validate_geos_convection_panels!(raw.cmfmc, "CMFMC", something(physics.a3mste_path, "<GEOS-FP A3mstE>"), win_idx)
    _validate_geos_convection_panels!(raw.dtrain, "DTRAIN", something(physics.a3dyn_path, "<GEOS-FP A3dyn>"), win_idx)
    return raw
end

function _read_geosfp_convection_window!(raw::RawWindow{FT},
                                         physics::GEOSFPLatLonPhysicsFallback,
                                         win_idx::Int,
                                         orientation::Symbol) where {FT}
    physics.a3mste === nothing &&
        error("GEOS-FP lat-lon convection fallback requires A3mstE but no handle is open")
    physics.a3dyn === nothing &&
        error("GEOS-FP lat-lon convection fallback requires A3dyn but no handle is open")
    @assert raw.cmfmc  !== nothing "RawWindow.cmfmc must be allocated when convection is enabled"
    @assert raw.dtrain !== nothing "RawWindow.dtrain must be allocated when convection is enabled"

    # `_read_latlon_slice` calls `_time_index_for_geosfp_physics`
    # internally, which already maps `win_idx` (hourly, 1..24) to the
    # appropriate A3 record (3-hourly, 1..8) via
    # `_a3_index_for_window`. Pass `win_idx` directly here — the CS
    # fallback above uses `_read_panels_3d`, which has no such
    # internal mapping and therefore precomputes `a3_idx`.
    cmfmc_name = _find_var_name(physics.a3mste, GEOS_CONVECTION_VAR_CANDIDATES[:cmfmc])
    dtrain_name = _find_var_name(physics.a3dyn, GEOS_CONVECTION_VAR_CANDIDATES[:dtrain])
    cmfmc_ll  = _read_latlon_slice(physics.a3mste, cmfmc_name,  win_idx, Val(3), FT)
    dtrain_ll = _read_latlon_slice(physics.a3dyn,  dtrain_name, win_idx, Val(3), FT)
    if orientation === :bottom_up
        cmfmc_ll = reverse(cmfmc_ll; dims = 3)
        dtrain_ll = reverse(dtrain_ll; dims = 3)
    end
    size(cmfmc_ll, 3) == size(raw.cmfmc[1], 3) ||
        throw(DimensionMismatch("GEOS-FP CMFMC fallback has $(size(cmfmc_ll, 3)) levels; expected $(size(raw.cmfmc[1], 3))"))
    size(dtrain_ll, 3) == size(raw.dtrain[1], 3) ||
        throw(DimensionMismatch("GEOS-FP DTRAIN fallback has $(size(dtrain_ll, 3)) levels; expected $(size(raw.dtrain[1], 3))"))
    _interpolate_ll_to_panels!(raw.cmfmc,  cmfmc_ll,  physics)
    _interpolate_ll_to_panels!(raw.dtrain, dtrain_ll, physics)
    _moist_to_dry_dtrain!(raw.dtrain, raw.qv, raw.qv_next)
    _moist_to_dry_cmfmc!(raw.cmfmc,  raw.qv, raw.qv_next)
    _validate_geos_convection_panels!(raw.cmfmc, "CMFMC", something(physics.a3mste_path, "<GEOS-FP A3mstE>"), win_idx)
    _validate_geos_convection_panels!(raw.dtrain, "DTRAIN", something(physics.a3dyn_path, "<GEOS-FP A3dyn>"), win_idx)
    return raw
end

# ---------------------------------------------------------------------------
# Convection: 3-hourly hold-constant binding from the per-day A3 datasets.
#
# A3 collections are time-averaged 3-hourly (8 time records per day,
# centered at 01:30, 04:30, …, 22:30). We bind every hourly preprocessing
# window to the A3 record that covers it: windows 1–3 → A3 idx 1, windows
# 4–6 → idx 2, …, windows 22–24 → idx 8. Result: cmfmc / dtrain are the
# same across the 3-hour block; the dry-basis correction applied below
# uses the per-window window-mean qv (average of t_n and t_{n+1}), so
# the dry forcing varies hourly even though the underlying moist flux
# is held constant across the 3-hour block.
#
# Why dry-basis correction matters: GMAO archives CMFMC and DTRAIN as
# moist-air mass fluxes (kg moist air / m² / s), but the v4 binary
# carries `mass_basis = :dry` and the runtime tracer state's
# `air_mass` is dry. The convection operator transports
# tracers proportional to `f / m × dt`, so f must be on the same basis
# as m. We multiply by `(1 − qv_face)` here so the consumer sees a
# dry-air mass flux throughout the chain.
# ---------------------------------------------------------------------------

@inline _a3_index_for_window(win::Int) = (win - 1) ÷ 3 + 1

function _read_geos_convection_window!(raw::RawWindow{FT},
                                       handles::GEOSDayHandles,
                                       win_idx::Int,
                                       orientation::Symbol) where {FT}
    handles.a3mste === nothing &&
        error("settings.include_convection=true but A3mstE handle is missing; " *
              "did you call `open_day(...; next_day_handle=true)` on a settings " *
              "with `include_convection=true`?")
    handles.a3dyn === nothing &&
        error("settings.include_convection=true but A3dyn handle is missing; " *
              "ensure the A3dyn collection is on disk for this date")
    @assert raw.cmfmc  !== nothing "RawWindow.cmfmc must be allocated when convection is enabled"
    @assert raw.dtrain !== nothing "RawWindow.dtrain must be allocated when convection is enabled"

    a3_idx = _a3_index_for_window(win_idx)
    cmfmc_raw  = _read_panels_3d(handles.a3mste, "CMFMC",  a3_idx, orientation; FT=FT)
    dtrain_raw = _read_panels_3d(handles.a3dyn,  "DTRAIN", a3_idx, orientation; FT=FT)
    for p in 1:6
        copyto!(raw.cmfmc[p],  cmfmc_raw[p])
        copyto!(raw.dtrain[p], dtrain_raw[p])
    end
    _moist_to_dry_dtrain!(raw.dtrain, raw.qv, raw.qv_next)
    _moist_to_dry_cmfmc!(raw.cmfmc,  raw.qv, raw.qv_next)
    return raw
end

# DTRAIN at centers: face index k → center k. Dry factor is the
# window-mean of (1 − qv) at the same cell.
function _moist_to_dry_dtrain!(dtrain::NTuple{6, Array{FT,3}},
                               qv::NTuple{6, Array{FT,3}},
                               qv_next::NTuple{6, Array{FT,3}}) where {FT}
    @inbounds for p in 1:6
        d = dtrain[p]; q1 = qv[p]; q2 = qv_next[p]
        Nc1, Nc2, Nz = size(d)
        for k in 1:Nz, j in 1:Nc2, i in 1:Nc1
            qv_avg = (q1[i, j, k] + q2[i, j, k]) * FT(0.5)
            d[i, j, k] *= (one(FT) - qv_avg)
        end
    end
    return dtrain
end

# CMFMC at interfaces (Nz+1): face k sits between centers k−1 (above)
# and k (below) under TOA-first orientation. Use the four-corner mean
# (two centers × two endpoints) for interior faces; collapse to the
# single adjacent center at the model top (k=1) and surface (k=Nz+1).
function _moist_to_dry_cmfmc!(cmfmc::NTuple{6, Array{FT,3}},
                              qv::NTuple{6, Array{FT,3}},
                              qv_next::NTuple{6, Array{FT,3}}) where {FT}
    @inbounds for p in 1:6
        c = cmfmc[p]; q1 = qv[p]; q2 = qv_next[p]
        Nc1, Nc2, Nz1 = size(c)
        Nz = Nz1 - 1
        for k in 1:Nz1, j in 1:Nc2, i in 1:Nc1
            qv_face = if k == 1
                (q1[i, j, 1] + q2[i, j, 1]) * FT(0.5)
            elseif k == Nz1
                (q1[i, j, Nz] + q2[i, j, Nz]) * FT(0.5)
            else
                (q1[i, j, k-1] + q1[i, j, k] +
                 q2[i, j, k-1] + q2[i, j, k]) * FT(0.25)
            end
            c[i, j, k] *= (one(FT) - qv_face)
        end
    end
    return cmfmc
end

# Per-window 2D variable handle by name (NCDataset[name]).
_read_panels_2d(ds::NCDataset, name, win_idx; FT) =
    _read_panels_2d(ds[name], win_idx; FT=FT)

_read_panels_3d(ds::NCDataset, name, win_idx, orientation; FT) =
    _read_panels_3d(ds[name], win_idx, orientation; FT=FT)
