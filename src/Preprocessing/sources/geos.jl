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
#     `compare_era5_geosit_met.jl` (`am_moist = MFXC / (g·dt_dyn) / (1−qv)`).
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

# ---------------------------------------------------------------------------
# Per-panel array slicers.
# ---------------------------------------------------------------------------

"""Read one window of a 3D field as `NTuple{6, Array{FT,3}}`, level-flipped if needed."""
function _read_panels_3d(var, win_idx::Int, orientation::Symbol; FT::Type)
    raw = Array{FT}(var[:, :, :, :, win_idx])     # (Nc, Nc, 6, Nz)
    if orientation === :bottom_up
        return ntuple(p -> reverse(raw[:, :, p, :]; dims=3), 6)
    else
        return ntuple(p -> Array(raw[:, :, p, :]), 6)
    end
end

"""Read one window of a 2D field as `NTuple{6, Matrix{FT}}`."""
function _read_panels_2d(var, win_idx::Int; FT::Type)
    raw = Array{FT}(var[:, :, :, win_idx])        # (Nc, Nc, 6)
    return ntuple(p -> Array(raw[:, :, p]), 6)
end

const GEOS_SURFACE_VAR_CANDIDATES = Dict(
    :pblh  => ("PBLH", "pblh", "ZPBL", "zpbl"),
    :ustar => ("USTAR", "ustar", "UST", "ust"),
    :hflux => ("HFLUX", "hflux", "SH", "sshf", "surface_sensible_heat_flux"),
    :t2m   => ("T2M", "t2m", "T2MDEW", "2t", "2m_temperature"),
)

const GEOS_CONVECTION_VAR_CANDIDATES = Dict(
    :cmfmc  => ("CMFMC", "cmfmc", "conv_mass_flux"),
    :dtrain => ("DTRAIN", "dtrain"),
)

_dim_norm(x) = replace(lowercase(String(x)), "_" => "", "-" => "")

function _find_dim(dims, candidates)
    wanted = Set(_dim_norm.(candidates))
    return findfirst(d -> _dim_norm(d) in wanted, dims)
end

function _find_var_name(ds::NCDataset, candidates)
    wanted = Set(lowercase.(String.(candidates)))
    for name in keys(ds)
        lowercase(String(name)) in wanted && return String(name)
    end
    throw(ArgumentError("NetCDF file is missing required variable; tried " *
                        join(String.(candidates), ", ")))
end

function _geosfp_axis(ds::NCDataset, candidates)
    for name in candidates
        haskey(ds, name) && return Float64.(ds[name][:])
    end
    throw(ArgumentError("GEOS-FP lat-lon physics file is missing coordinate axis; tried " *
                        join(candidates, ", ")))
end

function _centered_lons(lons::AbstractVector{<:Real})
    out = [mod(Float64(lon) + 180.0, 360.0) - 180.0 for lon in lons]
    # Keep the field roll in sync with `_normalize_lon_to_centered`: when the
    # source was 0..360, the roll by Nx/2 produces sorted centered longitudes.
    if !issorted(out)
        out = circshift(out, length(out) ÷ 2)
    end
    return out
end

function _geosfp_latlon_axes(ds::NCDataset)
    lons = _geosfp_axis(ds, ("lon", "longitude", "Xdim", "x"))
    lats = _geosfp_axis(ds, ("lat", "latitude", "Ydim", "y"))
    lons = _centered_lons(lons)
    lats = lats[1] > lats[end] ? reverse(lats) : lats
    return lons, lats
end

function _geos_dim_length(ds::NCDataset, candidates)
    dim_name = nothing
    wanted = Set(_dim_norm.(candidates))
    for name in keys(ds.dim)
        if _dim_norm(name) in wanted
            dim_name = String(name)
            break
        end
    end
    dim_name === nothing && return nothing
    dim = ds.dim[dim_name]
    return dim isa Integer ? Int(dim) : length(dim)
end

function _time_index_for_geosfp_physics(ds::NCDataset, win_idx::Int)
    ntime = _geos_dim_length(ds, ("time",))
    ntime === nothing && return 1
    ntime <= 1 && return 1
    if ntime == 24
        return win_idx
    elseif ntime == 8
        return _a3_index_for_window(win_idx)
    end
    idx = min(win_idx, ntime)
    return idx
end

function _read_latlon_slice(ds::NCDataset, var_name::String,
                            win_idx::Int, ::Val{Rank},
                            ::Type{FT}) where {Rank, FT}
    v = ds[var_name]
    dims = dimnames(v)
    lon_dim = _find_dim(dims, ("lon", "longitude", "xdim", "x"))
    lat_dim = _find_dim(dims, ("lat", "latitude", "ydim", "y"))
    lev_dim = Rank == 3 ? _find_dim(dims, ("lev", "level", "ilev", "edge", "interface")) : nothing
    time_dim = _find_dim(dims, ("time",))
    lon_dim === nothing && throw(ArgumentError("$(var_name) lacks longitude dimension"))
    lat_dim === nothing && throw(ArgumentError("$(var_name) lacks latitude dimension"))
    Rank == 3 && lev_dim === nothing && throw(ArgumentError("$(var_name) lacks level dimension"))

    time_idx = _time_index_for_geosfp_physics(ds, win_idx)
    idx = ntuple(d -> d == lon_dim || d == lat_dim || d == lev_dim ? Colon() :
                      d == time_dim ? time_idx : 1,
                 length(dims))
    raw = Array(v[idx...])
    kept = [dims[d] for d in eachindex(dims)
            if d == lon_dim || d == lat_dim || d == lev_dim]
    perm = if Rank == 2
        [findfirst(==(dims[lon_dim]), kept), findfirst(==(dims[lat_dim]), kept)]
    else
        [findfirst(==(dims[lon_dim]), kept),
         findfirst(==(dims[lat_dim]), kept),
         findfirst(==(dims[lev_dim]), kept)]
    end
    field = perm == collect(1:Rank) ? raw : permutedims(raw, Tuple(perm))

    if haskey(ds, "lat")
        lats_raw = ds["lat"][:]
    elseif haskey(ds, "latitude")
        lats_raw = ds["latitude"][:]
    else
        lats_raw = Float64[]
    end
    if !isempty(lats_raw) && length(lats_raw) == size(field, 2) && lats_raw[1] > lats_raw[end]
        field = Rank == 2 ? field[:, end:-1:1] : field[:, end:-1:1, :]
    end

    if haskey(ds, "lon")
        lons_raw = ds["lon"][:]
    elseif haskey(ds, "longitude")
        lons_raw = ds["longitude"][:]
    else
        lons_raw = Float64[]
    end
    if !isempty(lons_raw)
        if Rank == 2
            tmp = reshape(field, size(field, 1), size(field, 2), 1)
            field = @view _normalize_lon_to_centered(tmp, lons_raw)[:, :, 1]
        else
            field = _normalize_lon_to_centered(field, lons_raw)
        end
    end
    return Array{FT}(field)
end

@inline _wrap_lon180(lon::Real) = mod(Float64(lon) + 180.0, 360.0) - 180.0

function _interp_regular_ll(src::AbstractMatrix{FT}, lons, lats,
                            lon::Real, lat::Real) where FT
    Nx, Ny = size(src)
    λ = _wrap_lon180(lon)
    Δλ = length(lons) > 1 ? lons[2] - lons[1] : 360.0
    Δφ = length(lats) > 1 ? lats[2] - lats[1] : 180.0
    x = (λ - lons[1]) / Δλ + 1.0
    x < 1.0 && (x += Nx)
    x >= Nx + 1 && (x -= Nx)
    y = clamp((Float64(lat) - lats[1]) / Δφ + 1.0, 1.0, Ny)
    i0 = clamp(floor(Int, x), 1, Nx)
    j0 = clamp(floor(Int, y), 1, Ny - 1)
    i1 = i0 == Nx ? 1 : i0 + 1
    j1 = min(j0 + 1, Ny)
    wi = x - floor(x)
    wj = y - j0
    return (1 - wi) * (1 - wj) * src[i0, j0] +
           wi       * (1 - wj) * src[i1, j0] +
           (1 - wi) * wj       * src[i0, j1] +
           wi       * wj       * src[i1, j1]
end

function _interp_regular_ll(src::AbstractArray{FT, 3}, lons, lats,
                            lon::Real, lat::Real, k::Int) where FT
    return _interp_regular_ll(@view(src[:, :, k]), lons, lats, lon, lat)
end

function _interpolate_ll_to_panels!(dst::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                    src::AbstractMatrix{FT},
                                    physics::GEOSFPLatLonPhysicsFallback) where FT
    @inbounds for p in 1:CS_PANEL_COUNT
        out = dst[p]
        tlon = physics.target_lons[p]
        tlat = physics.target_lats[p]
        for j in axes(out, 2), i in axes(out, 1)
            out[i, j] = _interp_regular_ll(src, physics.lons, physics.lats,
                                           tlon[i, j], tlat[i, j])
        end
    end
    return dst
end

function _interpolate_ll_to_panels!(dst::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                    src::AbstractArray{FT, 3},
                                    physics::GEOSFPLatLonPhysicsFallback) where FT
    @inbounds for p in 1:CS_PANEL_COUNT
        out = dst[p]
        tlon = physics.target_lons[p]
        tlat = physics.target_lats[p]
        for k in axes(out, 3), j in axes(out, 2), i in axes(out, 1)
            out[i, j, k] = _interp_regular_ll(src, physics.lons, physics.lats,
                                              tlon[i, j], tlat[i, j], k)
        end
    end
    return dst
end

function _hflux_to_upward_wm2(raw, ds::NCDataset, var_name::String, ::Type{FT}) where FT
    units = lowercase(String(get(ds[var_name].attrib, "units", "")))
    lname = lowercase(var_name)
    if occursin("j", units) || lname in ("sshf", "surface_sensible_heat_flux")
        return FT.(-raw ./ 3600)
    end
    return FT.(raw)
end

function _validate_geos_surface_panels!(surface, path::String, win_idx::Int)
    for field in (surface.pblh, surface.ustar, surface.hflux, surface.t2m)
        all(p -> all(isfinite, p), field) ||
            error("GEOS surface fallback contains non-finite values in $(path) window $(win_idx)")
    end
    minimum(minimum, surface.pblh) > 0 ||
        error("GEOS surface PBLH must be positive in $(path) window $(win_idx)")
    minimum(minimum, surface.ustar) >= 0 ||
        error("GEOS surface USTAR must be nonnegative in $(path) window $(win_idx)")
    minimum(minimum, surface.t2m) > 150 && maximum(maximum, surface.t2m) < 350 ||
        error("GEOS surface T2M is out of range in $(path) window $(win_idx)")
    maximum(p -> maximum(abs, p), surface.hflux) < 5000 ||
        error("GEOS surface HFLUX magnitude is out of range in $(path) window $(win_idx)")
    return nothing
end

function _validate_geos_convection_panels!(field, name::String, path::String, win_idx::Int)
    field === nothing && return nothing
    all(p -> all(isfinite, p), field) ||
        error("GEOS convection $(name) contains non-finite values in $(path) window $(win_idx)")
    return nothing
end

function _validate_geos_vdiff_panels!(vdiff, win_idx::Int)
    for name in (:u, :v, :t, :qv)
        field = getfield(vdiff, name)
        all(p -> all(isfinite, p), field) ||
            error("GEOS GCHP VDIFF $(name) contains non-finite values in window $(win_idx)")
    end
    maximum(p -> maximum(abs, p), vdiff.u) < 500 ||
        error("GEOS GCHP VDIFF U wind magnitude is out of range in window $(win_idx)")
    maximum(p -> maximum(abs, p), vdiff.v) < 500 ||
        error("GEOS GCHP VDIFF V wind magnitude is out of range in window $(win_idx)")
    minimum(minimum, vdiff.t) > 100 && maximum(maximum, vdiff.t) < 400 ||
        error("GEOS GCHP VDIFF temperature is out of range in window $(win_idx)")
    minimum(minimum, vdiff.qv) >= 0 && maximum(maximum, vdiff.qv) < 0.2 ||
        error("GEOS GCHP VDIFF specific humidity is out of range in window $(win_idx)")
    return nothing
end

"""
    _ps_pa_factor(var) -> FT scaling

GEOS-IT CTM_I1 stores PS in hPa; GEOS-FP native CTM stores PS in Pa. Read
the `units` attribute (case-insensitive) and return the multiplier needed
to land in Pa. Errors loudly on unrecognized units to prevent silent
100x errors.
"""
function _ps_pa_factor(var; FT::Type)
    units = lowercase(strip(get(var.attrib, "units", "")))
    if units == "pa"
        return FT(1)
    elseif units == "hpa" || units == "mbar" || units == "millibars"
        return FT(100)
    else
        error("Unrecognized PS units `$(units)` on GEOS NetCDF variable; expected " *
              "Pa or hPa/mbar")
    end
end

# ---------------------------------------------------------------------------
# Endpoint dry-basis reconstruction.
# Given PS_total (Pa) and QV (kg/kg) at one hour, plus the hybrid
# coordinate, build (PS_dry, DELP_dry) consistent with Σ DELP_dry = PS_dry.
# ---------------------------------------------------------------------------

"""
    endpoint_dry_mass!(delp_dry, ps_dry, ps_total, qv, vc) -> (delp_dry, ps_dry)

Reconstruct dry DELP and dry PS at one endpoint hour from the moist PS and
QV provided by GEOS, using the hybrid coordinate `vc`. The output is on
top-down level convention (k=1 = TOA).

Algorithm:

    DELP_full[k] = ΔA[k] + ΔB[k] * PS_total
    DELP_dry[k]  = (1 - QV[k]) * DELP_full[k]
    PS_dry       = Σ DELP_dry[k]

In-place: writes into `delp_dry` and `ps_dry`.
"""
function endpoint_dry_mass!(delp_dry::AbstractArray{FT,3},
                            ps_dry::AbstractMatrix{FT},
                            ps_total::AbstractMatrix{FT},
                            qv::AbstractArray{FT,3},
                            vc::HybridSigmaPressure) where {FT}
    Nx, Ny, Nz = size(qv)
    @assert size(delp_dry) == size(qv)
    @assert size(ps_dry)   == size(ps_total) == (Nx, Ny)
    A = vc.A
    B = vc.B
    @assert length(A) == length(B) == Nz + 1

    @inbounds for j in 1:Ny, i in 1:Nx
        ps_total_ij = ps_total[i, j]
        ps_dry_ij = zero(FT)
        @simd for k in 1:Nz
            ΔA = FT(A[k+1] - A[k])
            ΔB = FT(B[k+1] - B[k])
            delp_full = ΔA + ΔB * ps_total_ij
            delp_dry_k = (1 - qv[i, j, k]) * delp_full
            delp_dry[i, j, k] = delp_dry_k
            ps_dry_ij += delp_dry_k
        end
        ps_dry[i, j] = ps_dry_ij
    end
    return delp_dry, ps_dry
end

"""Allocate (delp_dry, ps_dry) for one panel and run `endpoint_dry_mass!`."""
function endpoint_dry_mass(ps_total::AbstractMatrix{FT}, qv::AbstractArray{FT,3},
                           vc::HybridSigmaPressure) where {FT}
    delp_dry = similar(qv)
    ps_dry   = similar(ps_total)
    endpoint_dry_mass!(delp_dry, ps_dry, ps_total, qv, vc)
    return delp_dry, ps_dry
end

# ---------------------------------------------------------------------------
# Mass-flux scaling.
#
# MFXC and MFYC in GEOS-IT and GEOS-FP CTM files are ALREADY dry mass fluxes
# (per the in-tree diagnostic `compare_era5_geosit_met.jl` and GMAO docs:
# "GEOS am_moist = MFXC / (g × dt_dyn) / (1 − qv)"; getting MOIST from
# native MFXC requires *dividing* by `(1 − qv)`, so multiplying by it would
# double-dry). The reader therefore only divides by `mass_flux_dt` to convert
# the dynamics-step accumulated quantity to a rate-like quantity; no humidity
# correction is applied here.
# ---------------------------------------------------------------------------

function _scale_flux!(am::AbstractArray{FT,3},
                      mfxc_raw::AbstractArray{FT,3},
                      inv_dt::FT) where {FT}
    @inbounds @simd for i in eachindex(am)
        am[i] = mfxc_raw[i] * inv_dt
    end
    return am
end

# ---------------------------------------------------------------------------
# AbstractMetSettings interface.
# ---------------------------------------------------------------------------

windows_per_day(::GEOSSettings, ::Date) = 24

# Per-source output-filename override: GEOS gets a clearer prefix.
_native_output_filename(::AbstractGEOSSettings, date::Date, FT::Type) =
    "geos_transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"

# `has_convection` reports the capability that downstream code can rely on.
# When `settings.include_convection` is `true`, the reader populates
# `RawWindow.cmfmc` (from A3mstE) and `RawWindow.dtrain` (from A3dyn) per
# the 3-hourly hold-constant binding (window 1–3 → t=1, 4–6 → t=2, …);
# the orchestrator threads them into the v4 binary's `:cmfmc` / `:dtrain`
# payload sections, and the runtime `CMFMCConvection` operator consumes
# them via `ConvectionForcing` (GCHP RAS / Grell-Freitas, dry-basis).
has_convection(s::GEOSSettings) = s.include_convection
has_surface(s::GEOSSettings) = s.include_surface || s.include_vdiff_fields
has_vdiff_fields(s::GEOSSettings) = s.include_vdiff_fields

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
