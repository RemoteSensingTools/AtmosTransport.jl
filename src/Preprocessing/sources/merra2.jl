# ===========================================================================
# Native MERRA-2 NetCDF reader for the wind-derived → CS preprocessor.
#
# MERRA-2 reproduces the validated GEOS-Chem CO₂ transport input path: derive
# horizontal mass fluxes from MERRA-2 WINDS (U/V) plus a Cameron-Smith
# column pressure-fix (the Poisson balance), instead of GEOS native
# cubed-sphere MFXC. This is purely additive — the GEOS-native and ERA5 paths
# are untouched.
#
# Data lives on a regular 0.5° × 0.625° latitude-longitude grid (576 × 361),
# 72 hybrid sigma-pressure levels (the GEOS-5 L72 coordinate, SAME as GEOS-FP),
# 3-hourly (8 windows/day). Two archives are supported (`MERRA2Archive`):
#
#   NASAArchive — the GES DISC files, split by collection:
#     M2I3NVASM  inst3_3d_asm_Nv  3-hr INSTANTANEOUS  PS, QV (mass endpoints)
#     M2T3NVASM  tavg3_3d_asm_Nv  3-hr TIME-AVERAGE   U, V (advecting winds)
#
#   GEOSChemArchive — the GEOS-Chem-processed files that GEOS-Chem/GCHP read
#     (s3://gcgrid/GEOS_0.5x0.625/MERRA2/YYYY/MM/MERRA2.YYYYMMDD.*.05x0625.nc4):
#     I3      3-hr INSTANTANEOUS  PS, QV, T            (= inst3_3d_asm_Nv)
#     A3dyn   3-hr TIME-AVERAGE   U, V, DTRAIN         (centred 01:30, 04:30, …)
#     A3mstE  3-hr TIME-AVERAGE   CMFMC on 73 edges    (centred 01:30, 04:30, …)
#     A3mstC  3-hr TIME-AVERAGE   DQRCU (convective rain production → cloud base)
#     A1      1-hr TIME-AVERAGE   PBLH, USTAR, HFLUX, EFLUX, T2M (centred HH:30)
#     Same values as the GES DISC files with the level axis reversed
#     (verified 2021-12-04: U to 2e-5 m/s, PS and QV exactly). Only this
#     layout carries the convection and boundary-layer fields.
#
# Critical conventions baked into this reader:
#
#   * The GES DISC files store levels TOP-DOWN (lev=1 ≈ TOA), SAME as the GEOS
#     L72 coefficient table; the GEOS-Chem files store them SURFACE-FIRST.
#     Neither carries DELP, so the order is detected per file from inst3 QV
#     (≈7e-3 kg/kg at the surface, ≈2e-6 at the top) when the day is opened,
#     and every reader returns top-down arrays.
#
#   * PS is in Pa, U/V in m/s, QV in kg/kg. lat is ascending (S→N), lon is
#     ascending (W→E, -180..179.375°) — both already match the project's
#     `LatLonMesh(longitude=(-180,180), latitude=(-90,90))` convention, so
#     no spatial reorientation is applied.
#
#   * NCDatasets returns dimensions REVERSED from the CDL header, so a CDL
#     `QV(time, lev, lat, lon)` is read as `ds["QV"][lon, lat, lev, time]`.
#     The window readers slice the trailing `time` axis at the 1-based window
#     index and return plain `(lon, lat[, lev])` arrays.
#
#   * Windowing contract (mirrors the GEOS-IT sliding window): window `win`
#     (1..8) endpoint dry mass uses inst3 slice `win` PS/QV; the advecting
#     winds use tavg3 slice `win` (the time AVERAGE over [3(win-1), 3win]Z).
#     The final window's right endpoint is next-day inst3 slice 1.
#     Physics (GEOS-Chem layout): CMFMC and DTRAIN are A3 slice `win` (the
#     same 3-hour average as the winds); PBLH, USTAR, HFLUX and T2M are the
#     mean of the three hourly A1 records 3(win-1)+1 .. 3win; the VDIFF
#     temperature is I3 slice `win` (window start, like QV).
# ===========================================================================

"""
    NASAArchive()
    GEOSChemArchive()

MERRA-2 file layouts. `NASAArchive`: the GES DISC collections under
`{root_dir}/M2{I,T}3NVASM/YYYY/MM/` (PS, QV, U, V only). `GEOSChemArchive`: the
GEOS-Chem-processed files `{root_dir}/YYYY/MM/MERRA2.YYYYMMDD.{tag}.05x0625.nc4`,
which also carry the convection and boundary-layer fields.
"""
abstract type MERRA2Archive end
struct NASAArchive <: MERRA2Archive end
struct GEOSChemArchive <: MERRA2Archive end

merra2_archive_name(::NASAArchive) = "nasa"
merra2_archive_name(::GEOSChemArchive) = "geoschem"

"""
    merra2_archive(name) -> MERRA2Archive

The archive for the `[preprocessing] layout` key: `"nasa"` or `"geoschem"`.
"""
function merra2_archive(name::AbstractString)
    name == "nasa" && return NASAArchive()
    name == "geoschem" && return GEOSChemArchive()
    throw(ArgumentError("MERRA-2 layout must be \"nasa\" or \"geoschem\"; got $(repr(name))"))
end

# Which fields an archive carries.
_has_inst3_winds(::NASAArchive) = true
_has_inst3_winds(::GEOSChemArchive) = false          # I3 holds PS, QV, T only
_has_physics_fields(::NASAArchive) = false
_has_physics_fields(::GEOSChemArchive) = true

"""
    TopDown()
    SurfaceFirst()

Level order of a MERRA-2 file: the GES DISC files are top-down (k = 1 at the
model top, like the L72 coefficient table), the GEOS-Chem files surface-first.
"""
abstract type MERRA2LevelOrder end
struct TopDown <: MERRA2LevelOrder end
struct SurfaceFirst <: MERRA2LevelOrder end

_to_top_down!(field, ::TopDown) = field
_to_top_down!(field, ::SurfaceFirst) = reverse!(field; dims = 3)

"""
    MERRA2Settings <: AbstractMetSettings

Settings for the MERRA-2 wind-derived cubed-sphere preprocessor.

- `root_dir` — directory holding the MERRA-2 collection trees
  (`{root_dir}/M2{I,T}3NVASM/{YYYY}/{MM}/...nc4`).
- `coefficients_file` — hybrid σ-pressure coefficient TOML (the GEOS L72
  table; MERRA-2 shares the GEOS-5 L72 coordinate).
- `winds_collection` — `:tavg3` (time-averaged U/V, the default and the
  GEOS-Chem-faithful choice) or `:inst3` (instantaneous U/V from the inst3
  collection, no separate tavg3 file needed; `NASAArchive` only).
- `archive` — `NASAArchive()` or `GEOSChemArchive()` (see [`MERRA2Archive`](@ref)).
- `include_surface` — write PBLH, USTAR, HFLUX, T2M (`GEOSChemArchive` only).
- `include_convection` — write dry CMFMC and DTRAIN (`GEOSChemArchive` only).
- `include_vdiff_fields` — write the Holtslag-Boville VDIFF fields U, V, T,
  QV and the latent heat flux EFLUX (implies the surface fields;
  `GEOSChemArchive` only).
- `include_convective_cloud_base` — write GCHP's convective cloud base, the
  lowest layer with DQRCU > 0 (A3mstC; needs `include_convection`).
"""
Base.@kwdef struct MERRA2Settings{A <: MERRA2Archive} <: AbstractMetSettings
    root_dir              :: String
    coefficients_file     :: String = "config/geos_L72_coefficients.toml"
    winds_collection      :: Symbol = :tavg3
    archive               :: A      = NASAArchive()
    include_surface       :: Bool   = false
    include_convection    :: Bool   = false
    include_vdiff_fields  :: Bool   = false
    include_convective_cloud_base :: Bool = false
    column_balance_weights :: Symbol = :mass   # a key of COLUMN_WEIGHT_KINDS (cs_poisson_balance.jl)
    face_lengths   :: Symbol = :cell_centerline   # face length in the flux: :cell_centerline or :edge
    flux_thickness :: Symbol = :moist             # Δp in the flux: :moist (from moist ps) or :dry_mass
    face_fluxes    :: Symbol = :panel_average     # :panel_average (panel components) or :vector (edge lengths always)
    face_interpolation :: Symbol = :linear        # :linear, :cubic or :fv3 (cubic + FV3's along-face filter; vector only)
    wind_regrid    :: Symbol = :scalar            # :scalar (u, v separately) or :cartesian (as a vector)
end

const MERRA2_NATIVE_LEVEL_COUNT = 72
const MERRA2_NX = 576
const MERRA2_NY = 361
const MERRA2_VALID_WINDS_COLLECTIONS = (:tavg3, :inst3)
# Layer thickness in the face fluxes (MERRA-2 only; the other flux-construction
# options are shared, see `cs_transport_helpers.jl`).
const MERRA2_FLUX_THICKNESS = (:moist, :dry_mass)


"""
    validate_merra2_settings(settings) -> settings

Reject unknown wind collections and requests the archive cannot serve (only
the GEOS-Chem files carry CMFMC, DTRAIN, DQRCU and A1).
"""
function validate_merra2_settings(s::MERRA2Settings)
    s.winds_collection in MERRA2_VALID_WINDS_COLLECTIONS ||
        throw(ArgumentError("MERRA-2 winds_collection must be one of " *
                            join(MERRA2_VALID_WINDS_COLLECTIONS, ", ") *
                            "; got :$(s.winds_collection)"))
    s.winds_collection === :inst3 && !_has_inst3_winds(s.archive) &&
        throw(ArgumentError("the GEOS-Chem MERRA-2 archive has no instantaneous winds " *
                            "(I3 holds PS, QV, T); use winds_collection=:tavg3 (A3dyn)"))
    _validate_flux_construction(s, "MERRA-2")
    s.flux_thickness in MERRA2_FLUX_THICKNESS || throw(ArgumentError(
        "MERRA-2 flux_thickness must be one of $(join(MERRA2_FLUX_THICKNESS, ", ")); got :$(s.flux_thickness)"))
    has_cmfmc_cloud_base(s) && !has_convection(s) &&
        throw(ArgumentError("MERRA-2 include_convective_cloud_base needs include_convection"))
    (has_surface(s) || has_convection(s)) && !_has_physics_fields(s.archive) &&
        throw(ArgumentError("MERRA-2 surface/convection/VDIFF fields need the GEOS-Chem " *
                            "archive (layout = \"geoschem\"); the GES DISC reader covers " *
                            "inst3/tavg3 asm_Nv only"))
    return s
end

# ---------------------------------------------------------------------------
# On-disk layout.
#
# inst3 (PS/QV endpoints): collection `inst3_3d_asm_Nv`, dataset dir M2I3NVASM.
# tavg3 (U/V winds):       collection `tavg3_3d_asm_Nv`, dataset dir M2T3NVASM.
# Stream code by year: 100 (1980-91), 200 (92-00), 300 (01-10), 400 (11-99).
# ---------------------------------------------------------------------------

const _MERRA2_COLLECTIONS = (
    inst3 = (dir = "M2I3NVASM", name = "inst3_3d_asm_Nv"),
    tavg3 = (dir = "M2T3NVASM", name = "tavg3_3d_asm_Nv"),
)

# GEOS-Chem archive: role → collection tag in `MERRA2.YYYYMMDD.{tag}.05x0625.nc4`.
const _MERRA2_GEOSCHEM_COLLECTIONS = (inst3 = "I3", tavg3 = "A3dyn",
                                      a3mste = "A3mstE", a3mstc = "A3mstC", a1 = "A1")

"""
    merra2_stream_code(date) -> String

MERRA-2 production stream code for `date` (same year→stream map the download
path uses): 100 (1980-91), 200 (92-00), 300 (01-10), 400 (2011 onward).
"""
function merra2_stream_code(date::Date)
    yr = year(date)
    yr <= 1991 && return "100"
    yr <= 2000 && return "200"
    yr <= 2010 && return "300"
    return "400"
end

"""
    merra2_path(settings, date, collection) -> String

Resolve the on-disk path for one MERRA-2 `collection` on `date`:

- `NASAArchive` (`:inst3`, `:tavg3`):
  `{root_dir}/{dir}/{YYYY}/{MM}/MERRA2_{stream}.{name}.{YYYYMMDD}.nc4`
- `GEOSChemArchive` (`:inst3` → I3, `:tavg3` → A3dyn, `:a3mste`, `:a3mstc`, `:a1`):
  `{root_dir}/{YYYY}/{MM}/MERRA2.{YYYYMMDD}.{tag}.05x0625.nc4`

Existence is not checked here — `open_day` decides whether a missing file is
fatal or merely "no next-day endpoint available".
"""
merra2_path(settings::MERRA2Settings, date::Date, collection::Symbol) =
    merra2_path(settings.archive, settings.root_dir, date, collection)

function _merra2_collection(table::NamedTuple, collection::Symbol)
    hasproperty(table, collection) ||
        throw(ArgumentError("unknown MERRA-2 collection $(collection) for this archive; " *
                            "expected one of $(propertynames(table))"))
    return getproperty(table, collection)
end

function merra2_path(::GEOSChemArchive, root, date::Date, collection::Symbol)
    tag = _merra2_collection(_MERRA2_GEOSCHEM_COLLECTIONS, collection)
    return joinpath(root, Dates.format(date, "yyyy"), Dates.format(date, "mm"),
                    "MERRA2.$(Dates.format(date, "yyyymmdd")).$(tag).05x0625.nc4")
end

function merra2_path(::NASAArchive, root, date::Date, collection::Symbol)
    coll = _merra2_collection(_MERRA2_COLLECTIONS, collection)
    fname = "MERRA2_$(merra2_stream_code(date)).$(coll.name).$(Dates.format(date, "yyyymmdd")).nc4"
    return joinpath(root, coll.dir, Dates.format(date, "yyyy"), Dates.format(date, "mm"), fname)
end

# ---------------------------------------------------------------------------
# Day-handle container.
# ---------------------------------------------------------------------------

"""
    MERRA2DayHandles

Open `NCDataset` handles for one UTC day:

- `inst3` — PS/QV instantaneous endpoints (always open).
- `tavg3` — U/V time-averaged winds; `nothing` when
  `settings.winds_collection == :inst3` (winds then come from `inst3`).
- `next_inst3` — next day's inst3 dataset, for the final window's right
  endpoint look-ahead; `nothing` on the archive boundary or when
  `next_day_handle = false`.
- `a3mste`, `a3mstc`, `a1` — GEOS-Chem A3mstE (CMFMC), A3mstC (DQRCU) and
  A1 (PBL surface fields); open only when the corresponding output is
  requested.
- `level_order`, `next_level_order` — `TopDown()` or `SurfaceFirst()`,
  detected from the day's and the next day's inst3 QV; the day's order applies
  to all of that day's collections (one archive, one convention).
"""
mutable struct MERRA2DayHandles
    settings           :: MERRA2Settings
    date               :: Date
    inst3              :: NCDataset
    tavg3              :: Union{Nothing, NCDataset}
    next_inst3         :: Union{Nothing, NCDataset}
    a3mste             :: Union{Nothing, NCDataset}
    a3mstc             :: Union{Nothing, NCDataset}
    a1                 :: Union{Nothing, NCDataset}
    level_order        :: MERRA2LevelOrder
    next_level_order   :: MERRA2LevelOrder
end

"""
    open_merra2_day(settings, date; next_day_handle=true) -> MERRA2DayHandles

Open the day's inst3 (and tavg3, when `winds_collection==:tavg3`) datasets,
asserting both are on disk. When `next_day_handle=true` and the next-day
inst3 file exists, opens it for the final-window endpoint look-ahead.
"""
function open_merra2_day(settings::MERRA2Settings, date::Date;
                         next_day_handle::Bool = true)
    validate_merra2_settings(settings)
    opened = NCDataset[]             # closed again if any later step fails
    function open_required(collection, day = date)
        path = merra2_path(settings, day, collection)
        isfile(path) || error("MERRA-2 $(collection) file not found: $path")
        return push!(opened, NCDataset(path, "r"))[end]
    end
    try
        inst3 = open_required(:inst3)
        tavg3 = settings.winds_collection === :tavg3 ? open_required(:tavg3) : nothing
        a3mste = has_convection(settings) ? open_required(:a3mste) : nothing
        a3mstc = has_cmfmc_cloud_base(settings) ? open_required(:a3mstc) : nothing
        a1 = has_surface(settings) ? open_required(:a1) : nothing
        a1 === nothing || size(a1["PBLH"], 3) == 24 ||
            error("MERRA-2 A1 for $(date) has $(size(a1["PBLH"], 3)) hourly records, expected 24")

        # A missing next-day file is the archive boundary (the writer warns and
        # falls back to a zero-tendency final window); an unreadable one is an
        # error, not a boundary.
        next_path = merra2_path(settings, date + Day(1), :inst3)
        next_inst3 = next_day_handle && isfile(next_path) ?
            open_required(:inst3, date + Day(1)) : nothing

        level_order = detect_merra2_level_order(inst3, merra2_path(settings, date, :inst3))
        next_level_order = next_inst3 === nothing ? level_order :
            detect_merra2_level_order(next_inst3, next_path)

        return MERRA2DayHandles(settings, date, inst3, tavg3, next_inst3, a3mste, a3mstc, a1,
                                level_order, next_level_order)
    catch
        foreach(close, opened)
        rethrow()
    end
end

"""
    close_merra2_day!(handles)

Close every open `NCDataset`. Idempotent — safe to call from a `finally`.
"""
function close_merra2_day!(handles::MERRA2DayHandles)
    close(handles.inst3)
    handles.tavg3      === nothing || close(handles.tavg3)
    handles.next_inst3 === nothing || close(handles.next_inst3)
    handles.a3mste     === nothing || close(handles.a3mste)
    handles.a3mstc     === nothing || close(handles.a3mstc)
    handles.a1         === nothing || close(handles.a1)
    return nothing
end

# ---------------------------------------------------------------------------
# AbstractMetSettings trait hooks.
# ---------------------------------------------------------------------------

open_day(settings::MERRA2Settings, date::Date; next_day_handle::Bool = true) =
    open_merra2_day(settings, date; next_day_handle = next_day_handle)

close_day!(handles::MERRA2DayHandles) = close_merra2_day!(handles)

windows_per_day(::MERRA2Settings, ::Date) = 8

"""
    source_grid(settings::MERRA2Settings; FT=Float64) -> LatLonMesh

The native MERRA-2 source mesh (576 × 361, -180..180 lon, -90..90 lat). The
preprocessor builds its own regridder against the *target* mesh radius; this
descriptor uses `R_EARTH` and is provided for the canonical trait surface.
"""
source_grid(::MERRA2Settings; FT::Type{<:AbstractFloat} = Float64) =
    LatLonMesh(; FT = FT, Nx = MERRA2_NX, Ny = MERRA2_NY,
                longitude = (-180, 180), latitude = (-90, 90),
                radius = FT(R_EARTH))

# VDIFF output needs the PBL surface fields too (the Holtslag-Boville closure
# reads PBLH/USTAR/HFLUX/T2M), so it implies the surface sections.
has_surface(s::MERRA2Settings)      = s.include_surface || s.include_vdiff_fields
has_convection(s::MERRA2Settings)   = s.include_convection
has_vdiff_fields(s::MERRA2Settings) = s.include_vdiff_fields
# GCHP's non-local PBL scheme needs the surface moisture flux next to VDIFF.
has_pbl_eflux(s::MERRA2Settings) = has_vdiff_fields(s)
has_cmfmc_cloud_base(s::MERRA2Settings) = s.include_convective_cloud_base

# ---------------------------------------------------------------------------
# Per-window field readers.
#
# NCDatasets reverses the CDL dim order, so the variables are indexed as:
#   PS(lon, lat, time)        → ds["PS"][:, :, win]      → (Nx, Ny)
#   QV/U/V(lon, lat, lev, time) → ds["..."][:, :, :, win] → (Nx, Ny, Nz)
# The level axis is reversed for files detected as surface-first so every
# reader returns top-down arrays; no lat/lon reorientation (both already
# ascending, matching LatLonMesh).
# ---------------------------------------------------------------------------

"""
    read_merra2_ps_slice(ds, win; FT) -> Matrix{FT}  (Nx, Ny)

Read the surface pressure (Pa) at 1-based time-slice `win` from an inst3
NCDataset, as an `(Nx, Ny) = (lon, lat)` matrix.
"""
function read_merra2_ps_slice(ds::NCDataset, win::Integer; FT::Type{<:AbstractFloat})
    ps = Array{FT}(ds["PS"][:, :, Int(win)])
    size(ps) == (MERRA2_NX, MERRA2_NY) ||
        throw(DimensionMismatch("MERRA-2 PS slice $(size(ps)) ≠ ($(MERRA2_NX), $(MERRA2_NY))"))
    return ps
end


"""
    read_merra2_3d_slice(ds, name, win, Nz, order; FT) -> Array{FT,3}  (Nx, Ny, Nz)

Read a 3D field at 1-based time-slice `win` as a top-down `(lon, lat, lev)`
array. `Nz` is the file's level count (72 for layer fields, 73 for CMFMC
edges); `order` is the file's [`MERRA2LevelOrder`](@ref).
"""
function read_merra2_3d_slice(ds::NCDataset, name::AbstractString, win::Integer, Nz::Integer,
                              order::MERRA2LevelOrder; FT::Type{<:AbstractFloat})
    f = Array{FT}(ds[name][:, :, :, Int(win)])
    size(f) == (MERRA2_NX, MERRA2_NY, Int(Nz)) ||
        throw(DimensionMismatch("MERRA-2 $(name) slice $(size(f)) ≠ " *
                                "($(MERRA2_NX), $(MERRA2_NY), $(Nz))"))
    return _to_top_down!(f, order)
end

"""
    detect_merra2_level_order(inst3, path) -> TopDown() or SurfaceFirst()

Level order of an inst3/I3 file from its first-slice QV: the global-mean
specific humidity is ≈7e-3 kg/kg in the surface layer and ≈2e-6 at the top,
so the moist end is the surface. Errors unless one end is >100× the other.
Neither archive carries DELP or pressure metadata to decide from.
"""
function detect_merra2_level_order(inst3::NCDataset, path::AbstractString)
    qv = inst3["QV"]
    nz = size(qv, 3)
    layer_mean(k) = sum(Float64, qv[:, :, k, 1]) / (size(qv, 1) * size(qv, 2))
    q_lev1, q_levn = layer_mean(1), layer_mean(nz)
    q_lev1 > 100 * q_levn && return SurfaceFirst()
    q_levn > 100 * q_lev1 && return TopDown()
    error("cannot detect the level order of $path: mean QV is $q_lev1 in level 1 " *
          "and $q_levn in level $nz (expected one end >100× the other)")
end

"""
    read_merra2_window_fields(handles, win, Nz; FT)
        -> (; ps, qv, u, v)

Read one window's worth of native MERRA-2 LL fields for 1-based window index
`win` (1..8):

  - `ps` (Pa) and `qv` (kg/kg) from the **inst3** handle slice `win`
    (instantaneous mass endpoint),
  - `u`, `v` (m/s) from the **tavg3** handle slice `win` (3-hr time-average
    advecting winds), or from inst3 when `winds_collection==:inst3`.

All arrays are `(lon, lat[, lev])`, top-down, no reorientation.
"""
function read_merra2_window_fields(handles::MERRA2DayHandles, win::Integer,
                                   Nz::Integer; FT::Type{<:AbstractFloat})
    nw = windows_per_day(handles.settings, handles.date)
    1 <= win <= nw || throw(ArgumentError("window $win out of range 1..$nw"))
    order = handles.level_order

    ps = read_merra2_ps_slice(handles.inst3, win; FT = FT)
    qv = read_merra2_3d_slice(handles.inst3, "QV", win, Nz, order; FT = FT)

    winds_ds = handles.settings.winds_collection === :tavg3 ?
        (handles.tavg3 === nothing ?
            error("MERRA-2 winds_collection=:tavg3 but tavg3 handle is nothing") :
            handles.tavg3) :
        handles.inst3
    u = read_merra2_3d_slice(winds_ds, "U", win, Nz, order; FT = FT)
    v = read_merra2_3d_slice(winds_ds, "V", win, Nz, order; FT = FT)

    return (; ps = ps, qv = qv, u = u, v = v)
end

"""
    read_merra2_physics_window(handles, win, Nz; FT) -> NamedTuple

Native moist-basis physics fields for window `win`, top-down, from the
GEOS-Chem archive. Keys present only for the requested outputs:

  - `cmfmc` (Nx, Ny, Nz+1) kg m⁻² s⁻¹ and `dtrain` (Nx, Ny, Nz) kg m⁻² s⁻¹:
    A3mstE / A3dyn slice `win` (A3mstE labels CMFMC "kg m-2 s-2", a typo).
  - `dqrcu` (Nx, Ny, Nz) kg kg⁻¹ s⁻¹: A3mstC slice `win`, convective rain
    production (only its sign is used, to locate the cloud base).
  - `pblh` m, `ustar` m s⁻¹, `hflux` and `eflux` W m⁻² (upward positive),
    `t2m` K, each (Nx, Ny, 3): the hourly A1 records 3(win-1)+1 .. 3win (hour
    averages centred 30 min into each hour of the window); `eflux` only with
    VDIFF fields.
  - `t` (Nx, Ny, Nz) K: I3 slice `win`, the window-start state (as QV).
"""
function read_merra2_physics_window(handles::MERRA2DayHandles, win::Integer,
                                    Nz::Integer; FT::Type{<:AbstractFloat})
    s = handles.settings
    nw = windows_per_day(s, handles.date)
    1 <= win <= nw || throw(ArgumentError("window $win out of range 1..$nw"))
    order = handles.level_order
    out = (;)
    if has_convection(s)
        cmfmc  = read_merra2_3d_slice(handles.a3mste, "CMFMC", win, Nz + 1, order; FT = FT)
        dtrain = read_merra2_3d_slice(handles.tavg3, "DTRAIN", win, Nz, order; FT = FT)
        out = merge(out, (; cmfmc, dtrain))
    end
    if has_cmfmc_cloud_base(s)
        out = merge(out, (; dqrcu = read_merra2_3d_slice(handles.a3mstc, "DQRCU", win, Nz, order; FT = FT)))
    end
    if has_surface(s)
        hours = (3 * (Int(win) - 1) + 1):(3 * Int(win))
        function hourly(name)
            x = Array{FT}(handles.a1[name][:, :, hours])
            size(x) == (MERRA2_NX, MERRA2_NY, 3) ||
                throw(DimensionMismatch("MERRA-2 A1 $(name) $(size(x)) ≠ ($(MERRA2_NX), $(MERRA2_NY), 3)"))
            return x
        end
        out = merge(out, (; pblh = hourly("PBLH"), ustar = hourly("USTAR"),
                            hflux = hourly("HFLUX"), t2m = hourly("T2M")))
        has_pbl_eflux(s) && (out = merge(out, (; eflux = hourly("EFLUX"))))
    end
    if has_vdiff_fields(s)
        out = merge(out, (; t = read_merra2_3d_slice(handles.inst3, "T", win, Nz, order; FT = FT)))
    end
    return out
end

"""
    read_merra2_next_day_endpoint(handles, Nz; FT) -> (; ps, qv[, t])

Read the next day's inst3 slice-1 PS/QV (and T when VDIFF fields are
requested) — the right endpoint for the final window of `handles.date`.
Returns `nothing` when no next-day inst3 handle is open (archive boundary).
"""
function read_merra2_next_day_endpoint(handles::MERRA2DayHandles, Nz::Integer;
                                       FT::Type{<:AbstractFloat})
    handles.next_inst3 === nothing && return nothing
    ps = read_merra2_ps_slice(handles.next_inst3, 1; FT = FT)
    order = handles.next_level_order
    qv = read_merra2_3d_slice(handles.next_inst3, "QV", 1, Nz, order; FT = FT)
    has_vdiff_fields(handles.settings) || return (; ps = ps, qv = qv)
    t = read_merra2_3d_slice(handles.next_inst3, "T", 1, Nz, order; FT = FT)
    return (; ps = ps, qv = qv, t = t)
end
