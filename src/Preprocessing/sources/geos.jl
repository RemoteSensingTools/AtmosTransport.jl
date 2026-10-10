# ===========================================================================
# Native GEOS-IT / GEOS-FP NetCDF reader for the preprocessor.
#
# GEOS data lives on a cubed-sphere grid (C180 for GEOS-IT, C720 for GEOS-FP),
# 72 hybrid sigma-pressure levels, daily files split by collection:
#
#   CTM_A1   hourly  MFXC, MFYC, DELP   (window-constant dynamics-step
#                                       horizontal mass transport, DELP)
#   CTM_I1   hourly  PS, QV             (instantaneous endpoints)
#   A3dyn    3-hr    DTRAIN, U, V       (held constant in 3-hr blocks; convection)
#   A3mstE   3-hr    CMFMC              (convection, edge-based)
#
# Critical conventions baked into this reader:
#
#   * CTM_A1 MFXC/MFYC are dry dynamics-step face masses in pressure-area
#     units (`MFXC ≈ CX * DELP * cell_area`). `mass_flux_dt = 450 s` is still
#     used because `read_window!` exposes a rate-like RawWindow
#     (`MFXC / mass_flux_dt`) for diagnostics and legacy contracts. The v4
#     writer converts each raw amount to one Strang half-sweep amount and
#     reuses it for the 8 GEOS substeps inside the hourly CTM window.
#
#   * GEOS-IT stores levels bottom-to-top (k=1 = surface). GEOS-FP stores them
#     top-to-bottom. Both are flipped (where needed) to AtmosTransport's
#     top-to-bottom convention. Auto-detection compares DELP[k=1] vs
#     DELP[k=Nz]; the surface side has the larger pressure thickness.
#
#   * DELP and PS in the GEOS archive are MOIST (total atmosphere). MFXC and
#     MFYC are ALREADY DRY mass fluxes per GMAO and the in-tree diagnostic
#     `scripts/diagnostics/heritage/compare_era5_geosit_met.jl`
#     (`am_moist = MFXC / (g·dt_dyn) / (1−qv)`).
#     The reader converts DELP and PS to dry via the hybrid coordinate plus
#     QV; MFXC and MFYC pass through unchanged apart from the RawWindow
#     rate normalization described above.
#
#   * `m` and `m_next` in the produced `RawWindow` are DELP_dry at the two
#     window endpoints, reconstructed from PS_total via the hybrid coordinate.
#     `Σ m_k = ps_dry` at every endpoint to roundoff. The orchestrator can
#     either chain raw endpoint mass across windows (`chain_mass = true`) or
#     seed every window from these raw GEOS endpoints (`chain_mass = false`).
#     Written windows balance horizontal fluxes to the raw endpoint and
#     diagnose `cm` from that same target for replay.
# ===========================================================================

abstract type AbstractGEOSSettings <: AbstractMetSettings end

"""
    GEOSSettings{flavor} <: AbstractGEOSSettings

Settings for one of the two supported GEOS flavors:

- `flavor = :geosit` — GEOS-IT (file pattern `GEOSIT.{date}.{collection}.C{Nc}.nc`).
- `flavor = :geosfp` — GEOS-FP (file pattern `GEOSFP.{date}.{collection}.C{Nc}.nc`).

Auto-detection of level orientation runs at `open_geos_day` time when
`level_orientation = :auto`. Set explicitly to `:bottom_up` or `:top_down`
to skip the heuristic.
"""
Base.@kwdef struct GEOSSettings{flavor} <: AbstractGEOSSettings
    root_dir            :: String
    Nc                  :: Int
    mass_flux_dt        :: Float64 = 450.0
    level_orientation   :: Symbol  = :auto    # :auto, :bottom_up, :top_down
    include_surface     :: Bool    = false
    include_convection  :: Bool    = false
    include_vdiff_fields :: Bool   = false
    physics_dir         :: String  = ""       # GEOS-FP 0.25°/CS fallback for surface + convection
    physics_layout      :: Symbol  = :auto    # :auto, :latlon_025, :cubed_sphere
    coefficients_file   :: String  = "config/geos_L72_coefficients.toml"
end

const GEOSITSettings = GEOSSettings{:geosit}
const GEOSFPSettings = GEOSSettings{:geosfp}

# ---------------------------------------------------------------------------
# File-naming dispatch on flavor.
#
# GEOS-IT C180 archive uses *daily* files: `GEOSIT.YYYYMMDD.<collection>.C180.nc`.
# GEOS-FP C720 native archive uses *hourly* files (one file per UTC hour):
# `GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HHMM.V01.nc4`.
# ---------------------------------------------------------------------------

"""
    geos_collection_path(settings::GEOSITSettings, date::Date, collection) -> String

Resolve the on-disk path of one GEOS-IT collection for `date`. Searches a
flat `root_dir` and the per-day `root_dir/YYYYMMDD/` layout.
"""
function geos_collection_path(settings::GEOSSettings{:geosit}, date::Date,
                              collection::AbstractString)
    datestr = Dates.format(date, "yyyymmdd")
    fname   = "GEOSIT.$(datestr).$(collection).C$(settings.Nc).nc"
    flat    = joinpath(settings.root_dir, fname)
    daily   = joinpath(settings.root_dir, datestr, fname)
    isfile(flat)  && return flat
    isfile(daily) && return daily
    error("GEOS-IT file not found: tried $flat and $daily")
end

function geosfp_native_hourly_ctm_path(settings::GEOSSettings{:geosfp},
                                       date::Date,
                                       hour::Integer)
    0 <= hour <= 23 || throw(ArgumentError("GEOS-FP hour must be 0..23, got $hour"))
    datestr = Dates.format(date, "yyyymmdd")
    hh = lpad(string(hour), 2, '0')
    stem = "GEOS.fp.asm.tavg_1hr_ctm_c$(lpad(settings.Nc, 4, '0'))_v72.$(datestr)_"
    # WashU's tavg_1hr archive is normally stamped at the window centre
    # (HH30). Some local fixtures and older mirrors used HH00, so keep it as
    # a compatibility fallback.
    candidates = String[]
    for minute in ("30", "00")
        fname = "$(stem)$(hh)$(minute).V01.nc4"
        push!(candidates, joinpath(settings.root_dir, fname))
        push!(candidates, joinpath(settings.root_dir, datestr, fname))
    end
    for path in candidates
        isfile(path) && return path
    end
    error("GEOS-FP native hourly CTM file not found: tried " * join(candidates, ", "))
end

function geos_collection_path(settings::GEOSSettings{:geosfp}, date::Date,
                              collection::AbstractString)
    collection in ("CTM_A1", "CTM_I1", "tavg_1hr_ctm_c0720_v72", "ctm") ||
        throw(ArgumentError("GEOS-FP native path resolver only supports hourly CTM collections; got $(collection)"))
    return geosfp_native_hourly_ctm_path(settings, date, 0)
end

abstract type AbstractGEOSFPPhysicsFallback end
struct GEOSFPNoPhysics <: AbstractGEOSFPPhysicsFallback end

mutable struct GEOSFPCSPhysicsFallback <: AbstractGEOSFPPhysicsFallback
    a1          :: Union{Nothing, NCDataset}
    a3mste      :: Union{Nothing, NCDataset}
    a3dyn       :: Union{Nothing, NCDataset}
    a1_path     :: Union{Nothing, String}
    a3mste_path :: Union{Nothing, String}
    a3dyn_path  :: Union{Nothing, String}
end

mutable struct GEOSFPLatLonPhysicsFallback <: AbstractGEOSFPPhysicsFallback
    a1          :: Union{Nothing, NCDataset}
    a3mste      :: Union{Nothing, NCDataset}
    a3dyn       :: Union{Nothing, NCDataset}
    a1_path     :: Union{Nothing, String}
    a3mste_path :: Union{Nothing, String}
    a3dyn_path  :: Union{Nothing, String}
    lons        :: Vector{Float64}
    lats        :: Vector{Float64}
    target_lons :: NTuple{CS_PANEL_COUNT, Matrix{Float64}}
    target_lats :: NTuple{CS_PANEL_COUNT, Matrix{Float64}}
end

function _geosfp_physics_collection_candidates(root::String, date::Date,
                                               collection::AbstractString,
                                               Nc::Int,
                                               layout::Symbol)
    datestr = Dates.format(date, "yyyymmdd")
    dirs = (root, joinpath(root, datestr))
    names = if layout === :cubed_sphere
        (
            "GEOSFP.$(datestr).$(collection).C$(Nc).nc",
            "GEOSFP.$(datestr).$(collection).C$(lpad(Nc, 4, '0')).nc",
            "GEOSFP_CS$(Nc).$(datestr).$(collection).nc",
            "GEOSFP_CS$(lpad(Nc, 4, '0')).$(datestr).$(collection).nc",
        )
    elseif layout === :latlon_025
        ("GEOSFP.$(datestr).$(collection).025x03125.nc",)
    else
        throw(ArgumentError("unsupported GEOS-FP physics layout $(layout)"))
    end
    return [joinpath(dir, name) for dir in dirs for name in names]
end

function _resolve_geosfp_physics_path(root::String, date::Date,
                                      collection::AbstractString,
                                      Nc::Int, layout::Symbol)
    for path in _geosfp_physics_collection_candidates(root, date, collection, Nc, layout)
        isfile(path) && return path
    end
    return nothing
end

function _select_geosfp_physics_layout(settings::GEOSSettings{:geosfp},
                                       date::Date,
                                       required_collection::AbstractString)
    layout = settings.physics_layout
    if layout === :auto
        root = expand_data_path(settings.physics_dir)
        _resolve_geosfp_physics_path(root, date, required_collection,
                                     settings.Nc, :cubed_sphere) !== nothing &&
            return :cubed_sphere
        _resolve_geosfp_physics_path(root, date, required_collection,
                                     settings.Nc, :latlon_025) !== nothing &&
            return :latlon_025
        return :auto
    elseif layout in (:cubed_sphere, :latlon_025)
        return layout
    end
    throw(ArgumentError("GEOS-FP physics_layout must be auto, cubed_sphere, or latlon_025; got $(layout)"))
end

function _required_geosfp_physics_collection(settings::GEOSSettings{:geosfp})
    settings.include_surface && return "A1"
    settings.include_convection && return "A3mstE"
    return ""
end

function _open_required_geosfp_physics(root::String, date::Date,
                                       collection::AbstractString,
                                       Nc::Int, layout::Symbol)
    path = _resolve_geosfp_physics_path(root, date, collection, Nc, layout)
    path === nothing && error("GEOS-FP physics fallback file not found for $(date) collection $(collection) " *
                              "layout=$(layout). Tried " *
                              join(_geosfp_physics_collection_candidates(root, date, collection, Nc, layout), ", "))
    return NCDataset(path, "r"), path
end

function _open_geosfp_physics_fallback(settings::GEOSSettings{:geosfp},
                                       date::Date)
    (settings.include_surface || settings.include_convection) || return GEOSFPNoPhysics()
    isempty(settings.physics_dir) &&
        throw(ArgumentError(
            "GEOS-FP include_surface/include_convection requires [source].physics_dir " *
            "pointing at GEOSFP.YYYYMMDD.{A1,A3mstE,A3dyn}.025x03125.nc or " *
            "pre-regridded GEOSFP.YYYYMMDD.<collection>.C$(settings.Nc).nc files."))

    root = expand_data_path(settings.physics_dir)
    required = _required_geosfp_physics_collection(settings)
    layout = _select_geosfp_physics_layout(settings, date, required)
    layout === :auto &&
        error("Could not auto-detect GEOS-FP physics fallback layout in $(root) for $(date) $(required)")

    a1 = nothing; a1_path = nothing
    if settings.include_surface
        a1, a1_path = _open_required_geosfp_physics(root, date, "A1", settings.Nc, layout)
    end

    a3mste = nothing; a3mste_path = nothing
    a3dyn = nothing; a3dyn_path = nothing
    if settings.include_convection
        a3mste, a3mste_path = _open_required_geosfp_physics(root, date, "A3mstE", settings.Nc, layout)
        a3dyn, a3dyn_path = _open_required_geosfp_physics(root, date, "A3dyn", settings.Nc, layout)
    end

    if layout === :cubed_sphere
        return GEOSFPCSPhysicsFallback(a1, a3mste, a3dyn,
                                       a1_path, a3mste_path, a3dyn_path)
    end

    mesh = source_grid(settings; FT = Float64)
    target_lons = ntuple(p -> panel_cell_center_lonlat(mesh, p)[1], CS_PANEL_COUNT)
    target_lats = ntuple(p -> panel_cell_center_lonlat(mesh, p)[2], CS_PANEL_COUNT)
    coord_ds = settings.include_surface ? a1 : a3mste
    lons, lats = _geosfp_latlon_axes(coord_ds)
    return GEOSFPLatLonPhysicsFallback(a1, a3mste, a3dyn,
                                       a1_path, a3mste_path, a3dyn_path,
                                       lons, lats, target_lons, target_lats)
end

close_geosfp_physics!(::GEOSFPNoPhysics) = nothing

function close_geosfp_physics!(physics::Union{GEOSFPCSPhysicsFallback, GEOSFPLatLonPhysicsFallback})
    physics.a1     === nothing || close(physics.a1)
    physics.a3mste === nothing || close(physics.a3mste)
    physics.a3dyn  === nothing || close(physics.a3dyn)
    return nothing
end

# ---------------------------------------------------------------------------
# Day handles.
# ---------------------------------------------------------------------------

"""
    GEOSDayHandles

Open NCDataset handles for one UTC day's GEOS collections plus the resolved
level orientation and the hybrid coordinate (loaded once for endpoint-DELP
reconstruction). The orchestrator opens these once at the start of
`process_day` and closes them at the end.
"""
mutable struct GEOSDayHandles{V <: HybridSigmaPressure}
    ctm_a1      :: NCDataset
    ctm_i1      :: NCDataset
    next_ctm_i1 :: Union{Nothing, NCDataset}
    a1          :: Union{Nothing, NCDataset}
    a3dyn       :: Union{Nothing, NCDataset}
    a3mste      :: Union{Nothing, NCDataset}
    i3          :: Union{Nothing, NCDataset}
    # Adjacent-day A3dyn (OMEGA) + I3 (QV) handles for the `:omega_consistent`
    # closure's PCHIP time interpolation. The 3-hourly OMEGA/QV nodes do not span
    # the CTM window valid times at the day edges (A3dyn node 1 = 01:30, last node
    # 22:30; I3 last node 21:00), so the early/late windows bracket ACROSS midnight
    # into the previous/next day's nodes instead of constant-extrapolating to the
    # nearest same-day node. `nothing` ⇒ that edge falls back to constant-extrap
    # (e.g. first/last day of the archive). Opened only when these collections are.
    prev_a3dyn  :: Union{Nothing, NCDataset}
    next_a3dyn  :: Union{Nothing, NCDataset}
    prev_i3     :: Union{Nothing, NCDataset}
    next_i3     :: Union{Nothing, NCDataset}
    orientation :: Symbol                          # :bottom_up or :top_down
    vc          :: V                               # hybrid sigma-pressure (top-down)
end

mutable struct GEOSFPNativeDayHandles{V <: HybridSigmaPressure, P <: AbstractGEOSFPPhysicsFallback, C <: NCDataset}
    ctm         :: Vector{C}
    next_ctm    :: Union{Nothing, C}
    physics     :: P
    orientation :: Symbol
    vc          :: V
end

"""
    open_geos_day(settings, date; next_day_handle=true, adjacent_omega=false)
        -> GEOSDayHandles

Open per-collection NCDataset handles for `date`. When `next_day_handle` is
`true` and the next-day CTM_I1 file exists, opens it too so the last window
of `date` has its right endpoint available.

`adjacent_omega` (default `false`) opens the prev/next-day A3dyn (OMEGA) + I3
(QV) handles that the `:omega_consistent` cm closure needs to PCHIP-interpolate
across midnight. Every other closure (incl. the validated `:endpoint_balanced`
default and the GCHP VDIFF path) never reads them, so the default leaves all
four handles `nothing` and opens nothing extra.
"""
function open_geos_day(settings::GEOSSettings, date::Date;
                       next_day_handle::Bool=true,
                       adjacent_omega::Bool=false)
    ctm_a1 = NCDataset(geos_collection_path(settings, date, "CTM_A1"), "r")
    ctm_i1 = NCDataset(geos_collection_path(settings, date, "CTM_I1"), "r")

    next_ctm_i1 = nothing
    if next_day_handle
        try
            next_ctm_i1 = NCDataset(geos_collection_path(settings, date + Day(1), "CTM_I1"), "r")
        catch
            # Last day of the available archive — no next-day endpoint.
        end
    end

    need_surface = settings.include_surface || settings.include_vdiff_fields
    need_a3dyn = settings.include_convection || settings.include_vdiff_fields
    a1     = need_surface ? NCDataset(geos_collection_path(settings, date, "A1"), "r") : nothing
    a3dyn  = nothing
    a3mste = nothing
    if need_a3dyn
        a3dyn  = NCDataset(geos_collection_path(settings, date, "A3dyn"),  "r")
    end
    if settings.include_convection
        a3mste = NCDataset(geos_collection_path(settings, date, "A3mstE"), "r")
    end
    i3 = settings.include_vdiff_fields ? NCDataset(geos_collection_path(settings, date, "I3"), "r") : nothing

    # Adjacent-day A3dyn (OMEGA) + I3 (QV) for the `:omega_consistent` PCHIP
    # cross-midnight bracket. Open both prev and next so day-edge windows
    # interpolate instead of constant-extrapolating. Missing files (archive
    # edges) leave the handle `nothing` and fall back to constant-extrapolation.
    prev_a3dyn = nothing; next_a3dyn = nothing
    prev_i3 = nothing;    next_i3 = nothing
    if adjacent_omega
        prev_a3dyn = _try_open_geos_collection(settings, date - Day(1), "A3dyn")
        next_a3dyn = _try_open_geos_collection(settings, date + Day(1), "A3dyn")
        prev_i3    = _try_open_geos_collection(settings, date - Day(1), "I3")
        next_i3    = _try_open_geos_collection(settings, date + Day(1), "I3")
    end

    orientation = settings.level_orientation === :auto ?
                  detect_level_orientation(ctm_a1) :
                  settings.level_orientation

    vc = load_hybrid_coefficients(expand_data_path(settings.coefficients_file))

    return GEOSDayHandles(ctm_a1, ctm_i1, next_ctm_i1, a1, a3dyn, a3mste, i3,
                          prev_a3dyn, next_a3dyn, prev_i3, next_i3, orientation, vc)
end

"Open a GEOS collection NetCDF for `date`, returning `nothing` if absent."
function _try_open_geos_collection(settings::GEOSSettings, date::Date,
                                   collection::AbstractString)
    try
        return NCDataset(geos_collection_path(settings, date, collection), "r")
    catch
        return nothing
    end
end

function open_geosfp_native_day(settings::GEOSSettings{:geosfp}, date::Date;
                                next_day_handle::Bool = true)
    settings.include_vdiff_fields &&
        throw(ArgumentError("GEOS-FP GCHP VDIFF binary fields are not wired yet; " *
                            "use GEOS-IT C180 or add the required native/fallback T/U/V/QV collections."))
    ctm = [NCDataset(geosfp_native_hourly_ctm_path(settings, date, h), "r")
           for h in 0:23]
    next_ctm = nothing
    if next_day_handle
        try
            next_ctm = NCDataset(geosfp_native_hourly_ctm_path(settings, date + Day(1), 0), "r")
        catch
            # Last day of available archive.
        end
    end
    orientation = settings.level_orientation === :auto ?
                  detect_level_orientation(first(ctm)) :
                  settings.level_orientation
    vc = load_hybrid_coefficients(expand_data_path(settings.coefficients_file))
    physics = _open_geosfp_physics_fallback(settings, date)
    return GEOSFPNativeDayHandles(ctm, next_ctm, physics, orientation, vc)
end

"""Close all handles. Idempotent."""
function close_geos_day!(handles::GEOSDayHandles)
    close(handles.ctm_a1)
    close(handles.ctm_i1)
    handles.next_ctm_i1 === nothing || close(handles.next_ctm_i1)
    handles.a1          === nothing || close(handles.a1)
    handles.a3dyn       === nothing || close(handles.a3dyn)
    handles.a3mste      === nothing || close(handles.a3mste)
    handles.i3          === nothing || close(handles.i3)
    handles.prev_a3dyn  === nothing || close(handles.prev_a3dyn)
    handles.next_a3dyn  === nothing || close(handles.next_a3dyn)
    handles.prev_i3     === nothing || close(handles.prev_i3)
    handles.next_i3     === nothing || close(handles.next_i3)
    return nothing
end

function close_geosfp_native_day!(handles::GEOSFPNativeDayHandles)
    for ds in handles.ctm
        close(ds)
    end
    handles.next_ctm === nothing || close(handles.next_ctm)
    close_geosfp_physics!(handles.physics)
    return nothing
end

# ---------------------------------------------------------------------------
# Level-orientation auto-detection.
# DELP[k=1] dominates DELP[k=Nz] when k=1 is the surface — bottom-to-top.
# ---------------------------------------------------------------------------

"""
    detect_level_orientation(ctm_a1::NCDataset) -> Symbol

Return `:bottom_up` if k=1 is the surface (mass-thicker) and `:top_down`
if k=1 is TOA. Heuristic is unambiguous: surface DELP is O(1000 Pa),
TOA DELP is O(1 Pa).
"""
function detect_level_orientation(ctm_a1::NCDataset)
    delp = ctm_a1["DELP"]
    Nz = size(delp, 4)
    delp_top    = mean(skipmissing(delp[:, :, :, 1, 1]))
    delp_bottom = mean(skipmissing(delp[:, :, :, Nz, 1]))
    return delp_top > delp_bottom ? :bottom_up : :top_down
end
