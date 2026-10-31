"""
    InitialConditionIO

Single owner of initial-condition I/O, vertical remap, and
topology-dispatched VMR builders for the unified runtime.

## Public API (exported via `Models` → `AtmosTransport`)

- [`build_initial_mixing_ratio`](@ref) — topology-dispatched builder
  returning **dry VMR** on interior cells. Accepts `kind = uniform |
  latitude_step | gaussian_blob | file | netcdf | file_field |
  catrine_co2` for LL/RG meshes, plus `bl_enhanced` on LL. CS supports
  those shared kinds plus `pressure_layer` and signed `cs_native` fields.
- [`pack_initial_tracer_mass`](@ref) — basis-aware VMR → conservative model storage
  conversion. Dispatches on `mass_basis::AbstractMassBasis`:
  - `DryBasis` (the default runtime basis): `rm = vmr .* air_mass`.
  - `MoistBasis`: `rm = vmr .* air_mass .* (1 .- qv)`, because IC VMRs are
    dry and `air_mass` is moist; `qv` must be supplied.
- [`FileInitialConditionSource`](@ref) — container for a loaded IC
  NetCDF (3D VMR + optional hybrid coefficients).

## Contents

- LL/RG and CS `build_initial_mixing_ratio` for the kinds above. CS
  `pressure_layer` selects a level per column from the stored surface pressure
  and normalizes its dry VMR to the configured global molecule count.
- Basis-aware `pack_initial_tracer_mass` (DryBasis + MoistBasis); CS
  output is halo-padded with the halo zeroed.
- File conventions come from `GriddedIO`: coordinates (`find_coordinate`),
  orientation (`latlon_orientation`, `orient`), fill values (`read_raw` with
  `MaskedAsZero`), the source mesh of LL→CS conservative regridding
  (`source_latlon_mesh(lon, lat; radius)`, using `SnappedFaces`), time axes
  (`decode_time`) and flux units (`parse_flux_units`).
- Surface-flux loaders and builders (`initial_conditions/surface_flux*.jl`):
  static `FileSurfaceFluxField` and time-varying series, and LL/RG/CS
  `build_surface_flux_source` methods. CS `build_surface_flux_source`
  conservatively LL→CS-regrids the flux (the regridder's `dst_areas` ×
  regridded density already yields kg/s per cell) and unpacks to
  `NTuple{6, Matrix{FT}}`, or returns a `TimeVaryingSurfaceFluxSource` when
  `time_varying = true` — satisfying the per-cell rate contract of
  `SurfaceFluxSource` (`src/Operators/SurfaceFlux/sources.jl`).

Shared inventory operations have public aliases; the explicitly shared
internal runner seams are listed in this folder's README. Legacy qualified
helper names remain reachable.
"""
module InitialConditionIO

using ...Grids: bilinear_bracket, uniform_periodic_bilinear_bracket

using ...Grids: cell_areas

using NCDatasets
using Dates
using ...ConfigChecks: config_bool as _config_bool

import ...expand_data_path
using ...ConfigChecks: key_suggestion
using ...Parameters: AVOGADRO, EARTH_RADIUS, SPECIES_MOLAR_MASS, STANDARD_PRESSURE, DRY_AIR_MOLAR_MASS,
                     DEFAULT_BACKGROUND_VMR, DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG
using ...State: AbstractMassBasis, DryBasis, MoistBasis,
               AbstractSurfacePressureBasis, TotalSurfacePressure,
               DrySurfacePressure
using ...Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
                nrings, ring_longitudes, cell_index, cell_area,
                gravity, panel_cell_center_lonlat
# Regridding supplies LL→CS conservative weights; Grids owns panel unpacking.
using ...Regridding: build_regridder, apply_regridder!, default_regrid_cache_dir, source_coverage
using ...Grids: unpack_flat_to_panels_3d!, unpack_flat_to_panels_2d!,
                       CS_PANEL_COUNT
using ...Operators.SurfaceFlux: SurfaceFluxSource, TimeVaryingSurfaceFluxSource,
                               StepwiseFlux, flux_temporal_scheme
# The conventions of native gridded input files (coordinates, orientation,
# fill values, source meshes, time axes, flux units) and the `cs_native` flux
# file (reader, cell-area gate) are owned by GriddedIO.
using ...GriddedIO: hybrid_surface_first as _hybrid_surface_first,
                    validate_ap_units as _validate_ap_units,
                    VERTICAL_ORDERS as _VERTICAL_ORDERS,
                    parse_vertical_order as _parse_vertical_order,
                    PRESSURE_UNITS as _PRESSURE_UNITS,
                    HEIGHT_UNITS as _HEIGHT_UNITS,
                    file_vertical_order as _file_vertical_order
using ...GriddedIO: find_coordinate, LongitudeAxis, LatitudeAxis, TimeAxis, VerticalAxis,
                    latlon_orientation, orient, orient_coordinates, read_raw, MaskedAsZero,
                    source_latlon_mesh, decode_time, decode_time_bounds, decode_month_keys,
                    decode_month_seconds, decode_slice_intervals, known_slice_intervals,
                    stamp_intervals, interval, MonthKey,
                    parse_flux_units,
                    AbstractTimeBasis, PerSquareMetre, PerCell, PerMonth, TimeInterval,
                    FluxUnits, kg_per_unit, period_seconds, seconds_since,
                    mass_rate_factor, cs_native_flux_info, read_cs_native_flux!,
                    check_latlon_cell_area
# Grid accessors used by both LL/RG and CS surface-flux builders
using ...Grids: AbstractHorizontalMesh, nx, ny, ncells

using Printf: @sprintf

# ---------------------------------------------------------------------------
# Longitude-wrap helpers
# ---------------------------------------------------------------------------

# NOTE: the source arrays (Catrine, GridFED) may be in [-180, 180)
# convention; these wrap both to a common [0, 360) before looking up.
@inline wrapped_longitude_distance(lon, lon0) = abs(mod(lon - lon0 + 180, 360) - 180)
@inline wrapped_longitude_360(lon) = mod(lon, 360)

# ---------------------------------------------------------------------------
# Config-kind resolvers
# ---------------------------------------------------------------------------

abstract type AbstractInitialConditionSpec end

struct UniformInitialConditionSpec <: AbstractInitialConditionSpec end
struct LatitudeStepInitialConditionSpec <: AbstractInitialConditionSpec end
struct BoundaryLayerEnhancedInitialConditionSpec <: AbstractInitialConditionSpec end
struct GaussianBlobInitialConditionSpec <: AbstractInitialConditionSpec end
struct FileInitialConditionSpec <: AbstractInitialConditionSpec end
struct CatrineCO2InitialConditionSpec <: AbstractInitialConditionSpec end
struct PressureLayerInitialConditionSpec <: AbstractInitialConditionSpec end
struct CSNativeInitialConditionSpec <: AbstractInitialConditionSpec end

# One alias table for typed parsing, topology builders and runner key checks.
const INITIAL_CONDITION_KINDS = Dict(
    "uniform" => UniformInitialConditionSpec,
    "latitude_step" => LatitudeStepInitialConditionSpec,
    "lat_step" => LatitudeStepInitialConditionSpec,
    "hemisphere_step" => LatitudeStepInitialConditionSpec,
    "bl_enhanced" => BoundaryLayerEnhancedInitialConditionSpec,
    "gaussian_blob" => GaussianBlobInitialConditionSpec,
    "file" => FileInitialConditionSpec,
    "netcdf" => FileInitialConditionSpec,
    "file_field" => FileInitialConditionSpec,
    "catrine_co2" => CatrineCO2InitialConditionSpec,
    "pressure_layer" => PressureLayerInitialConditionSpec,
    "cs_native" => CSNativeInitialConditionSpec)
export INITIAL_CONDITION_KINDS

"Parse an initial-condition kind once at the runtime configuration boundary."
function initial_condition_spec(cfg::AbstractDict)
    kind = lowercase(String(get(cfg, "kind", "uniform")))
    spec_type = get(INITIAL_CONDITION_KINDS, kind, nothing)
    spec_type === nothing || return spec_type()
    throw(ArgumentError("unsupported init.kind=$(repr(kind))"))
end

@inline _init_kind(cfg) = Symbol(lowercase(String(get(cfg, "kind", "uniform"))))
@inline _is_file_init_kind(kind::Symbol) =
    get(INITIAL_CONDITION_KINDS, String(kind), nothing) in
        (FileInitialConditionSpec, CatrineCO2InitialConditionSpec)
@inline _is_latitude_step_kind(kind::Symbol) =
    get(INITIAL_CONDITION_KINDS, String(kind), nothing) === LatitudeStepInitialConditionSpec

function _cfg_get_alias(cfg, key::String, alias::String, default)
    haskey(cfg, key) && return cfg[key]
    haskey(cfg, alias) && return cfg[alias]
    return default
end

function _latitude_step_values(cfg, ::Type{FT}) where FT
    south = FT(_cfg_get_alias(cfg, "south_value", "south",
                              get(cfg, "background", DEFAULT_BACKGROUND_VMR)))
    north = FT(_cfg_get_alias(cfg, "north_value", "north", 4.4e-4))
    split = FT(get(cfg, "split_lat_deg", 0.0))
    return (; south, north, split)
end

@inline function _latitude_step_value(lat, vals)
    return typeof(vals.split)(lat) >= vals.split ? vals.north : vals.south
end

# ---------------------------------------------------------------------------
# FileInitialConditionSource struct
# ---------------------------------------------------------------------------

"""
    FileInitialConditionSource{FT}

Container for a file-based initial condition (e.g. Catrine startCO2).

# Fields
- `raw`            — 3D mixing ratio field `(nlon_src, nlat_src, nlevel_src)`.
                     With hybrid coefficients, surface first (`k = 1` is the
                     surface, decided from the interface pressures, the order
                     of `_remap_hybrid_profile!`); without them,
                     top first like the model levels.
- `qv`             — optional source specific humidity on the same grid and
                     level order as `raw`. When absent, remapping assumes a
                     unit source dry fraction (`q_v = 0`).
- `lon`, `lat`     — source coordinate vectors [degrees]. May be in [-180,180)
                     or [0,360); the bilinear sampler wraps to [0,360)
                     internally via `wrapped_longitude_360`.
- `ap`, `bp`       — hybrid half-level coefficients `(nlevel_src + 1)`:
                     `p_half[k] = ap[k] + bp[k] × ps_total`. The model
                     column's total surface pressure anchors the source coordinate;
                     a file surface-pressure field is neither needed nor read.
                     Units: `ap` [Pa], `bp` [dimensionless].
- `needs_vinterp`  — `true` for files with hybrid coefficients, which are
                     always conservatively remapped in pressure
                     (`_remap_hybrid_profile!`); files without them are copied
                     level by level and must have the target's level count.
"""
struct FileInitialConditionSource{FT}
    raw           :: Array{FT, 3}
    qv            :: Union{Nothing, Array{FT, 3}}
    lon           :: Vector{Float64}
    lat           :: Vector{Float64}
    ap            :: Vector{Float64}
    bp            :: Vector{Float64}
    needs_vinterp :: Bool
end

# ---------------------------------------------------------------------------
# Bracket search + bilinear interpolation
# ---------------------------------------------------------------------------

# Longitude and latitude centres [degrees] of a field whose leading Julia
# dimensions are (lon, lat): the coordinates `GriddedIO.find_coordinate` finds
# by name, each as long as its dimension of the field.
function _latlon_centres(ds, field_var)
    lon = find_coordinate(ds, LongitudeAxis(); length = size(field_var, 1))
    lat = find_coordinate(ds, LatitudeAxis(); length = size(field_var, 2))
    return Float64.(ds[lon][:]), Float64.(ds[lat][:])
end

"""
    _horizontal_interp_weights(lon, lat, lon_src, lat_src)

Compute bilinear interpolation indices and weights for a target point
`(lon, lat)` on source arrays with coordinates `lon_src` [degrees, periodic]
and `lat_src` [degrees, non-periodic clamped].

Returns `(ilo, ihi, jlo, jhi, w00, w10, w01, w11)` where `w00` through
`w11` are the four bilinear weights for cells `(ilo, jlo)`, `(ihi, jlo)`,
`(ilo, jhi)`, `(ihi, jhi)` respectively.

**Longitude convention**: `lon` is first wrapped to [0, 360) via
`wrapped_longitude_360` before the periodic bracket. This handles both
`[-180, 180)` and `[0, 360)` input conventions transparently.
"""
function _horizontal_interp_weights(lon::Real, lat::Real, lon_src::Vector{Float64}, lat_src::Vector{Float64})
    lon_m = wrapped_longitude_360(Float64(lon))  # → [0, 360)
    ilo, wx = uniform_periodic_bilinear_bracket(lon_m, lon_src)
    ihi = ilo == length(lon_src) ? 1 : ilo + 1   # periodic wrap at last cell
    jlo, wy = bilinear_bracket(Float64(lat), lat_src)
    jhi = min(jlo + 1, length(lat_src))           # clamp at poles (no wrap)
    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    return ilo, ihi, jlo, jhi, w00, w10, w01, w11
end

function _sample_bilinear_profile!(dest::AbstractVector{FT},
                                   raw::Array{FT, 3},
                                   lon_src::Vector{Float64},
                                   lat_src::Vector{Float64},
                                   lon::Real,
                                   lat::Real) where FT
    ilo, ihi, jlo, jhi, w00, w10, w01, w11 = _horizontal_interp_weights(lon, lat, lon_src, lat_src)
    @inbounds for k in eachindex(dest)
        dest[k] = FT(w00 * raw[ilo, jlo, k] +
                     w10 * raw[ihi, jlo, k] +
                     w01 * raw[ilo, jhi, k] +
                     w11 * raw[ihi, jhi, k])
    end
    return nothing
end

function _sample_bilinear_scalar(raw::AbstractMatrix{T},
                                 lon_src::Vector{Float64},
                                 lat_src::Vector{Float64},
                                 lon::Real,
                                 lat::Real) where T
    ilo, ihi, jlo, jhi, w00, w10, w01, w11 =
        _horizontal_interp_weights(lon, lat, lon_src, lat_src)
    return w00 * raw[ilo, jlo] +
           w10 * raw[ihi, jlo] +
           w01 * raw[ilo, jhi] +
           w11 * raw[ihi, jhi]
end

# ---------------------------------------------------------------------------
# IC config resolver + NetCDF loader
# ---------------------------------------------------------------------------

function _resolve_file_init(cfg, kind::Symbol)
    default_file, default_variable = if kind === :catrine_co2
        ("\$ATMOSTRANSPORT_DATA_ROOT/catrine/InitialConditions/startCO2_202112010000.nc", "CO2")
    else
        ("", "")
    end
    file = expand_data_path(String(get(cfg, "file", default_file)))
    variable = String(get(cfg, "variable", default_variable))
    isempty(file) && throw(ArgumentError("file-based init.kind=$(kind) requires init.file"))
    isempty(variable) && throw(ArgumentError("file-based init.kind=$(kind) requires init.variable"))
    time_index = Int(get(cfg, "time_index", 1))
    return file, variable, time_index
end

function _load_file_initial_condition_source(cfg, ::Type{FT}, Nz_target::Integer) where FT
    kind = _init_kind(cfg)
    file, variable, time_index = _resolve_file_init(cfg, kind)
    isfile(file) || throw(ArgumentError("initial-condition file not found: $file"))

    ds = NCDataset(file)
    try
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))
        raw_var = ds[variable]
        ndims(raw_var) in (3, 4) || throw(ArgumentError(
            "variable '$variable' must be 3D or 4D, got ndims=$(ndims(raw_var))"))
        lon_src, lat_src = _latlon_centres(ds, raw_var)
        lev_var = find_coordinate(ds, VerticalAxis(); length = size(raw_var, 3))
        lev_src = Float64.(ds[lev_var][:])

        # Fill values become 0 (GriddedIO.MaskedAsZero), read straight into FT.
        raw = ndims(raw_var) == 4 ?
              read_raw(FT, ds, variable, MaskedAsZero(), :, :, :, time_index) :
              read_raw(FT, ds, variable, MaskedAsZero())

        # Source humidity changes the conservative carrier and is therefore
        # opt-in. Similar-looking variables must not silently change remapping.
        humidity_name = if haskey(cfg, "humidity_variable")
            name = String(cfg["humidity_variable"])
            haskey(ds, name) || throw(ArgumentError(
                "humidity variable '$name' not found in $file"))
            name
        else
            nothing
        end
        source_qv = if humidity_name === nothing
            nothing
        else
            qv_var = ds[humidity_name]
            ndims(qv_var) == ndims(raw_var) && size(qv_var)[1:3] == size(raw_var)[1:3] ||
                throw(DimensionMismatch(
                    "humidity variable '$humidity_name' must share the IC field's " *
                    "longitude, latitude, and level dimensions"))
            ndims(qv_var) == 4 ?
                read_raw(FT, ds, humidity_name, MaskedAsZero(), :, :, :, time_index) :
                read_raw(FT, ds, humidity_name, MaskedAsZero())
        end

        haskey(ds, "ap") == haskey(ds, "bp") || throw(ArgumentError(
            "$(file) must provide both hybrid interface coefficients ap and bp, or neither"))
        has_hybrid = haskey(ds, "ap")
        ap = has_hybrid ? Float64.(ds["ap"][:]) : Float64[]
        bp = has_hybrid ? Float64.(ds["bp"][:]) : Float64[]
        has_hybrid && _validate_ap_units(ds["ap"], ap, file)

        # Latitude ascending, longitude from the first value ≥ 0 (GriddedIO).
        orientation = latlon_orientation(lon_src, lat_src)
        raw = orient(raw, orientation)
        source_qv === nothing || (source_qv = orient(source_qv, orientation))
        lon_src, lat_src = orient_coordinates(lon_src, lat_src, orientation)

        # Vertical order. Hybrid files are oriented by their interface
        # pressures (surface first, as the conservative remap expects) and
        # always remapped; the values of the level coordinate do not matter.
        # Files without hybrid coefficients are copied level by level onto the
        # top-first model levels, so they must say which end is the top.
        if has_hybrid
            surface_first = _hybrid_surface_first(ap, bp)
            haskey(cfg, "vertical_order") &&
                _parse_vertical_order(cfg["vertical_order"]) !== (surface_first ? :surface_first : :toa_first) &&
                throw(ArgumentError("init.vertical_order = $(repr(cfg["vertical_order"])) contradicts " *
                                    "the hybrid coefficients of $(file)"))
            if !surface_first
                raw = raw[:, :, end:-1:1]
                source_qv === nothing || (source_qv = source_qv[:, :, end:-1:1])
                ap = reverse(ap)
                bp = reverse(bp)
            end
        else
            if _file_vertical_order(cfg, ds[lev_var], lev_src, file) === :surface_first
                raw = raw[:, :, end:-1:1]
                source_qv === nothing || (source_qv = source_qv[:, :, end:-1:1])
            end
            size(raw, 3) == Nz_target || throw(ArgumentError(
                "$(file) has $(size(raw, 3)) levels and no hybrid coefficients (ap, bp) " *
                "to remap them onto the $(Nz_target) model levels"))
        end
        if source_qv !== nothing
            all(q -> isfinite(q) && 0 <= q < 1, source_qv) || throw(ArgumentError(
                "humidity variable '$humidity_name' must contain finite specific humidity in [0, 1)"))
        end
        return FileInitialConditionSource{FT}(raw, source_qv, lon_src, lat_src, ap, bp, has_hybrid)
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# Conservative hybrid-coordinate remapping
# ---------------------------------------------------------------------------

"""
    _remap_hybrid_profile!(dest, src_q, src_qv, ap, bp, A_tgt, B_tgt,
                           ps_total, src_p, cumulative, dry_cumulative)

Conservatively remap source dry-VMR layer means `src_q[1:Nsrc]` onto
`dest[1:Nz]`. The source is surface first and the target is top first; both
hybrid coordinates use the target column's **total** surface pressure:

```math
p^s_{i+1/2}=a^s_{i+1/2}+b^s_{i+1/2}p_{s,total}, \\qquad
p^t_{k+1/2}=A^t_{k+1/2}+B^t_{k+1/2}p_{s,total}.
```

Because the values are dry VMRs, the conserved carrier is source dry air. With
`d^s(p)=1-q^s_v(p)`, each target mean is the source-dry-air-weighted average.

```math
q^t_k = \\frac{\\int_{p^t_{k-1/2}}^{p^t_{k+1/2}}
                         q^s(p)d^s(p)\\,dp}
                 {\\int_{p^t_{k-1/2}}^{p^t_{k+1/2}} d^s(p)\\,dp}.
```

When the IC file supplies `src_qv`, `cumulative` stores `∫q^s d^s dp` and
`dry_cumulative` stores `∫d^s dp`; otherwise `d^s=1`. Dividing those source
integrals preserves a uniform VMR even when source and target humidity differ
and avoids target-cell-area dependence. Source VMR and humidity end layers are
extended outside source coverage. The three scratch vectors have length
`length(src_q) + 1`; the implementation is allocation-free and
`O(Nsource + Ntarget)` after validation.
"""
function _remap_hybrid_profile!(dest::AbstractVector{FT},
                                src_q::AbstractVector{FT},
                                src_qv::Union{Nothing, AbstractVector},
                                ap::AbstractVector{<:Real},
                                bp::AbstractVector{<:Real},
                                A_tgt::AbstractVector{<:Real},
                                B_tgt::AbstractVector{<:Real},
                                ps_total::Real,
                                src_p::AbstractVector{Float64},
                                cumulative::AbstractVector{Float64},
                                dry_cumulative::AbstractVector{Float64}) where FT
    Nsrc = length(src_q)
    Nz = length(dest)
    Nsrc > 0 || throw(ArgumentError("source profile must contain at least one layer"))
    length(A_tgt) == Nz + 1 || throw(DimensionMismatch(
        "A_tgt has length $(length(A_tgt)), expected Nz+1 = $(Nz + 1)"))
    length(B_tgt) == Nz + 1 || throw(DimensionMismatch(
        "B_tgt has length $(length(B_tgt)), expected Nz+1 = $(Nz + 1)"))
    length(ap) == Nsrc + 1 || throw(DimensionMismatch(
        "ap has length $(length(ap)), expected Nsrc+1 = $(Nsrc + 1)"))
    length(bp) == Nsrc + 1 || throw(DimensionMismatch(
        "bp has length $(length(bp)), expected Nsrc+1 = $(Nsrc + 1)"))
    src_qv === nothing || length(src_qv) == Nsrc || throw(DimensionMismatch(
        "src_qv has length $(length(src_qv)), expected Nsrc = $Nsrc"))
    length(src_p) == Nsrc + 1 || throw(DimensionMismatch(
        "src_p scratch has length $(length(src_p)), expected Nsrc+1 = $(Nsrc + 1)"))
    length(cumulative) == Nsrc + 1 || throw(DimensionMismatch(
        "cumulative scratch has length $(length(cumulative)), expected Nsrc+1 = $(Nsrc + 1)"))
    length(dry_cumulative) == Nsrc + 1 || throw(DimensionMismatch(
        "dry_cumulative scratch has length $(length(dry_cumulative)), expected Nsrc+1 = $(Nsrc + 1)"))

    ps = Float64(ps_total)
    isfinite(ps) || throw(ArgumentError(
        "target total surface pressure must be finite; got $ps_total"))

    @inbounds for k in eachindex(src_p)
        src_p[k] = Float64(ap[k]) + Float64(bp[k]) * ps
    end
    @inbounds for k in 1:Nsrc
        src_p[k] > src_p[k + 1] || throw(ArgumentError(
            "source hybrid interfaces must be strictly surface-to-TOA decreasing; " *
            "interfaces $k and $(k + 1) are $(src_p[k]) and $(src_p[k + 1]) Pa"))
    end

    p_lo = Float64(A_tgt[1]) + Float64(B_tgt[1]) * ps
    @inbounds for k in 1:Nz
        p_hi = Float64(A_tgt[k + 1]) + Float64(B_tgt[k + 1]) * ps
        p_hi > p_lo || throw(ArgumentError(
            "target hybrid interfaces must be strictly TOA-to-surface increasing; " *
            "interfaces $k and $(k + 1) are $p_lo and $p_hi Pa"))
        p_lo = p_hi
    end

    # Integral from the source top interface to each source interface. The
    # source arrays run surface -> TOA, hence the reverse fill.
    cumulative[Nsrc + 1] = 0.0
    dry_cumulative[Nsrc + 1] = 0.0
    @inbounds for s in Nsrc:-1:1
        dry_fraction = 1.0
        if src_qv !== nothing
            qv_s = Float64(src_qv[s])
            isfinite(qv_s) && 0 <= qv_s < 1 || throw(ArgumentError(
                "source specific humidity src_qv[$s]=$(src_qv[s]) must lie in [0, 1)"))
            dry_fraction = 1.0 - qv_s
        end
        dp = src_p[s] - src_p[s + 1]
        cumulative[s] = cumulative[s + 1] + Float64(src_q[s]) * dry_fraction * dp
        dry_cumulative[s] = dry_cumulative[s + 1] + dry_fraction * dp
    end

    # Target interfaces are increasing, so the source-layer cursor moves only
    # surfaceward. End-layer extension follows from using s=Nsrc below the
    # source top and s=1 above the source surface.
    s = Nsrc
    p_lo = Float64(A_tgt[1]) + Float64(B_tgt[1]) * ps
    while s > 1 && p_lo > src_p[s]
        s -= 1
    end
    dry_lo = src_qv === nothing ? 1.0 : 1.0 - Float64(src_qv[s])
    integral_lo = cumulative[s + 1] + Float64(src_q[s]) * dry_lo * (p_lo - src_p[s + 1])
    dry_integral_lo = dry_cumulative[s + 1] + dry_lo * (p_lo - src_p[s + 1])

    @inbounds for k in 1:Nz
        p_hi = Float64(A_tgt[k + 1]) + Float64(B_tgt[k + 1]) * ps
        while s > 1 && p_hi > src_p[s]
            s -= 1
        end
        dry_hi = src_qv === nothing ? 1.0 : 1.0 - Float64(src_qv[s])
        integral_hi = cumulative[s + 1] + Float64(src_q[s]) * dry_hi * (p_hi - src_p[s + 1])
        dry_integral_hi = dry_cumulative[s + 1] + dry_hi * (p_hi - src_p[s + 1])
        tracer_integral = integral_hi - integral_lo
        source_dry_integral = dry_integral_hi - dry_integral_lo
        source_dry_integral > 0 || throw(ArgumentError(
            "source dry-air pressure integral is non-positive in target layer $k"))
        dest[k] = FT(tracer_integral / source_dry_integral)
        p_lo = p_hi
        integral_lo = integral_hi
        dry_integral_lo = dry_integral_hi
    end
    return nothing
end

"""
    _total_surface_pressure(pressure_basis, ps_stored, A, B, qv)

Recover the total pressure that places both source and target hybrid
interfaces. For `TotalSurfacePressure` this is the stored value. For
`DrySurfacePressure` with target humidity, solve the linear column equation

```math
p_{s,dry}=\\sum_k (1-q_{v,k})
    [\\Delta A_k + \\Delta B_k p_{s,total}].
```

If a legacy dry-pressure binary contains no humidity payload (neither endpoint
humidity nor VDIFF `qv`), total pressure is unobservable. To keep those runs
working, interface placement explicitly approximates `ps_total` by the stored
`ps_dry` and emits one warning. This usually understates total pressure by
about 0.5--2 % in humid columns. The remap still takes its target dry-layer
carrier from the binary air masses. Current binaries record the stored
pressure meaning explicitly in `surface_pressure_basis`.
"""
@inline _total_surface_pressure(::TotalSurfacePressure, ps_stored::Real,
                                _A, _B, _qv) = Float64(ps_stored)

_warn_surface_pressure_approximation(::AbstractSurfacePressureBasis, _qv) = nothing

function _warn_surface_pressure_approximation(::DrySurfacePressure, ::Nothing)
    @warn "dry-basis transport binary stores dry surface pressure but carries no humidity payload; approximating total surface pressure as dry ps for hybrid-interface placement (typically 0.5-2% low in humid columns)" maxlog=1
    return nothing
end

function _total_surface_pressure(::DrySurfacePressure, ps_dry::Real,
                                 A_tgt, B_tgt, qv::Nothing)
    # Total pressure is not observable from a dry-pressure payload alone.
    # In the absence of any binary humidity, use qv=0, hence ps_total=ps_dry.
    return Float64(ps_dry)
end

function _total_surface_pressure(::DrySurfacePressure, ps_dry::Real,
                                 A_tgt, B_tgt, qv::AbstractVector)
    Nz = length(qv)
    length(A_tgt) == Nz + 1 && length(B_tgt) == Nz + 1 ||
        throw(DimensionMismatch("target humidity length must match the hybrid layers"))
    offset = 0.0
    slope = 0.0
    @inbounds for k in 1:Nz
        dry_fraction = 1.0 - Float64(qv[k])
        0 < dry_fraction <= 1 || throw(ArgumentError(
            "target specific humidity qv[$k]=$(qv[k]) must lie in [0, 1)"))
        offset += dry_fraction * (Float64(A_tgt[k + 1]) - Float64(A_tgt[k]))
        slope += dry_fraction * (Float64(B_tgt[k + 1]) - Float64(B_tgt[k]))
    end
    slope > 0 || throw(ArgumentError(
        "cannot recover total surface pressure: humidity-weighted hybrid B thickness is $slope"))
    ps_total = (Float64(ps_dry) - offset) / slope
    isfinite(ps_total) && ps_total > 0 || throw(ArgumentError(
        "recovered total surface pressure is invalid: $ps_total Pa"))
    return ps_total
end

# Level-by-level copy of a top-first source profile (the loader checks the counts).
function _copy_profile!(dest::AbstractVector{FT}, src_q::AbstractVector{FT}) where FT
    length(dest) == length(src_q) || throw(DimensionMismatch(
        "source profile has $(length(src_q)) levels, the model $(length(dest))"))
    copyto!(dest, src_q)
    return nothing
end

# ---------------------------------------------------------------------------
# Topology-dispatched VMR builder
#
# `build_initial_mixing_ratio` returns **dry VMR** on interior cells.
# Shapes:
#   LL: (Nx, Ny, Nz)
#   RG: (ncells, Nz)
#   CS: NTuple{6, Array{FT, 3}} of (Nc, Nc, Nz)
# ---------------------------------------------------------------------------

function _gaussian_blob_params(cfg, background::FT) where {FT}
    lon0 = FT(get(cfg, "lon0_deg", 0.0))
    lat0 = FT(get(cfg, "lat0_deg", 0.0))
    sigma_lon = FT(get(cfg, "sigma_lon_deg", DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG))
    sigma_lat = FT(get(cfg, "sigma_lat_deg", DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG))
    amplitude = FT(get(cfg, "amplitude", background))
    return lon0, lat0, sigma_lon, sigma_lat, amplitude
end

@inline function _gaussian_blob_value(lon, lat, background::FT, lon0, lat0,
                                      sigma_lon, sigma_lat, amplitude) where {FT}
    dlon = wrapped_longitude_distance(lon, lon0)
    dlat = lat - lat0
    return background + amplitude * exp(-FT(0.5) * ((dlon / sigma_lon)^2 + (dlat / sigma_lat)^2))
end

"""
    uniform_background(cfg, FT = Float64)

Return the runtime's uniform initial VMR, converted to `FT`.
"""
uniform_background(cfg, ::Type{FT} = Float64) where {FT} =
    FT(get(cfg, "background", DEFAULT_BACKGROUND_VMR))


"""
    build_initial_mixing_ratio(air_mass, mesh, cfg)
    build_initial_mixing_ratio(air_mass, grid::AtmosGrid, cfg;
        surface_pressure=nothing, pressure_basis=TotalSurfacePressure(),
        qv=nothing)

Construct the initial dry-air volume mixing ratio described by `cfg` on the
horizontal topology and vertical layout of `air_mass`.

Bare lat-lon and reduced-Gaussian meshes support the analytic `uniform`,
`latitude_step`, and `gaussian_blob` modes; `bl_enhanced` is lat-lon only.
Passing an `AtmosGrid` additionally enables file-backed modes (`file`,
`netcdf`, `file_field`, and `catrine_co2`) with topology-aware horizontal
mapping and conservative hybrid-coordinate vertical remapping. Cubed-sphere construction
requires an `AtmosGrid` and also supports `pressure_layer` and `cs_native`.
File-backed and pressure-layer construction require the transport window's
`surface_pressure` where indicated by the selected mode. Each public call owns
fresh output arrays. The driven cubed-sphere runner separately reuses private
interior buffers between analytic initializers before copying each tracer into
its independent packed state slot.

For file-backed hybrid remapping, `pressure_basis` states whether the binary's
stored surface pressure is total or dry, and `qv` is the binary's
window-start specific humidity when available. Source humidity participates
only when `init.humidity_variable` is configured explicitly. In that case each target value is the source-dry-air
mean `∫q(1-qv)dp / ∫(1-qv)dp`; without it the mean is `∫qdp / ∫dp`.
For legacy dry-pressure binaries without any
humidity payload, total pressure cannot be reconstructed: the builder keeps
the stored dry pressure for interface placement and warns once that it is
typically 0.5--2 % below total pressure in humid columns.

Returns an array shaped like `air_mass` for lat-lon and reduced-Gaussian grids,
or an `NTuple{6}` of interior `(Nc, Nc, Nz)` arrays for cubed-sphere grids.
"""
function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::LatLonMesh{FT}, cfg) where FT
    kind = _init_kind(cfg)
    background = uniform_background(cfg, FT)
    if kind === :uniform
        return fill(background, size(air_mass))
    elseif _is_latitude_step_kind(kind)
        vals = _latitude_step_values(cfg, FT)
        q = Array{FT}(undef, size(air_mass))
        for j in axes(q, 2)
            value = _latitude_step_value(mesh.φᶜ[j], vals)
            @views q[:, j, :] .= value
        end
        return q
    elseif kind === :bl_enhanced
        # Flat background + enhancement in the lowest `n_layers` model levels
        # (k = Nz-n_layers+1:Nz, since k=1=TOA, k=Nz=surface). Layer-based
        # threshold follows terrain naturally.
        n_layers = Int(get(cfg, "n_layers", 3))
        enhancement = FT(get(cfg, "enhancement", 1.0e-4))
        Nz = size(air_mass, 3)
        n_layers >= 1 && n_layers <= Nz ||
            throw(ArgumentError("init.kind=bl_enhanced: n_layers=$(n_layers) must satisfy 1 ≤ n_layers ≤ Nz=$(Nz)"))
        q = fill(background, size(air_mass))
        @views q[:, :, (Nz - n_layers + 1):Nz] .+= enhancement
        return q
    elseif kind === :gaussian_blob
        lon0, lat0, sigma_lon, sigma_lat, amplitude = _gaussian_blob_params(cfg, background)
        q = Array{FT}(undef, size(air_mass))
        for k in axes(q, 3), j in axes(q, 2), i in axes(q, 1)
            q[i, j, k] = _gaussian_blob_value(mesh.λᶜ[i], mesh.φᶜ[j], background,
                                             lon0, lat0, sigma_lon, sigma_lat, amplitude)
        end
        return q
    else
        throw(ArgumentError("unsupported init.kind=$(kind) for LatLonMesh"))
    end
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT}, mesh::ReducedGaussianMesh{FT}, cfg) where FT
    kind = _init_kind(cfg)
    background = uniform_background(cfg, FT)
    if kind === :uniform
        return fill(background, size(air_mass))
    elseif _is_latitude_step_kind(kind)
        vals = _latitude_step_values(cfg, FT)
        q = Array{FT}(undef, size(air_mass))
        for j in 1:nrings(mesh)
            value = _latitude_step_value(mesh.latitudes[j], vals)
            for i in 1:mesh.nlon_per_ring[j]
                c = cell_index(mesh, i, j)
                @views q[c, :] .= value
            end
        end
        return q
    elseif kind === :gaussian_blob
        lon0, lat0, sigma_lon, sigma_lat, amplitude = _gaussian_blob_params(cfg, background)
        q = Array{FT}(undef, size(air_mass))
        for j in 1:nrings(mesh)
            lats = mesh.latitudes[j]
            lons = ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = cell_index(mesh, i, j)
                value = _gaussian_blob_value(lons[i], lats, background,
                                              lon0, lat0, sigma_lon, sigma_lat, amplitude)
                @views q[c, :] .= value
            end
        end
        return q
    else
        throw(ArgumentError("unsupported init.kind=$(kind) for ReducedGaussianMesh"))
    end
end

# Setup scratch shared by the four file-IC column loops. The remapper's
# pressure/cumulative arrays deliberately remain Float64 for both precisions.
struct HybridRemapScratch{FT}
    q::Vector{FT}
    qv::Vector{FT}
    pressure::Vector{Float64}
    cumulative::Vector{Float64}
    dry_cumulative::Vector{Float64}
end

function HybridRemapScratch(::Type{FT}, nlevel, ninterface) where {FT}
    q = Vector{FT}(undef, nlevel)
    qv = similar(q)
    pressure = Vector{Float64}(undef, ninterface)
    return HybridRemapScratch(q, qv, pressure, similar(pressure), similar(pressure))
end

function _file_ic_column!(dest, scratch::HybridRemapScratch, source_qv,
                          needs_vinterp, ap, bp, pressure_basis, ps_stored,
                          A_tgt, B_tgt, target_qv)
    if needs_vinterp
        ps_total = _total_surface_pressure(
            pressure_basis, ps_stored, A_tgt, B_tgt, target_qv)
        _remap_hybrid_profile!(dest, scratch.q, source_qv, ap, bp,
                               A_tgt, B_tgt, ps_total, scratch.pressure,
                               scratch.cumulative, scratch.dry_cumulative)
    else
        _copy_profile!(dest, scratch.q)
    end
    return dest
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT},
                                    grid::AtmosGrid{<:LatLonMesh},
                                    cfg;
                                    surface_pressure::Union{Nothing, AbstractMatrix} = nothing,
                                    pressure_basis::AbstractSurfacePressureBasis = TotalSurfacePressure(),
                                    qv::Union{Nothing, AbstractArray} = nothing) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 3))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    scratch = HybridRemapScratch(FT, size(source.raw, 3), length(source.ap))
    src_q, src_qv = scratch.q, scratch.qv
    A_tgt = grid.vertical.A
    B_tgt = grid.vertical.B
    surface_pressure === nothing && throw(ArgumentError(
        "build_initial_mixing_ratio(::AtmosGrid{<:LatLonMesh}, ...) with " *
        "vertical-interp init kind=$(kind) requires `surface_pressure` (the " *
        "binary's stored ps) so target half-level pressures can be computed " *
        "exactly from the grid's hybrid coefficients. Pass " *
        "`window.surface_pressure` from `load_transport_window` to avoid " *
        "the area-mismatch artifact (2026-04-24)."))
    size(surface_pressure) == size(air_mass)[1:2] || throw(DimensionMismatch(
        "surface_pressure size $(size(surface_pressure)) must match air_mass " *
        "horizontal extent $(size(air_mass)[1:2])"))
    qv === nothing || size(qv) == size(air_mass) || throw(DimensionMismatch(
        "qv size $(size(qv)) must match air_mass size $(size(air_mass))"))
    source.needs_vinterp && _warn_surface_pressure_approximation(pressure_basis, qv)

    for j in axes(q, 2)
        lat = mesh.φᶜ[j]
        for i in axes(q, 1)
            lon = mesh.λᶜ[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            source.qv === nothing || _sample_bilinear_profile!(
                src_qv, source.qv, source.lon, source.lat, lon, lat)
            target_qv = !source.needs_vinterp || qv === nothing ? nothing : @view(qv[i, j, :])
            source_qv = source.qv === nothing ? nothing : src_qv
            _file_ic_column!(@view(q[i, j, :]), scratch, source_qv,
                             source.needs_vinterp, source.ap, source.bp,
                             pressure_basis, source.needs_vinterp ? surface_pressure[i, j] : nothing,
                             A_tgt, B_tgt, target_qv)
        end
    end

    return q
end

function build_initial_mixing_ratio(air_mass::AbstractArray{FT},
                                    grid::AtmosGrid{<:ReducedGaussianMesh},
                                    cfg;
                                    surface_pressure::Union{Nothing, AbstractVector} = nothing,
                                    pressure_basis::AbstractSurfacePressureBasis = TotalSurfacePressure(),
                                    qv::Union{Nothing, AbstractArray} = nothing) where FT
    kind = _init_kind(cfg)
    _is_file_init_kind(kind) || return build_initial_mixing_ratio(air_mass, grid.horizontal, cfg)

    source = _load_file_initial_condition_source(cfg, FT, size(air_mass, 2))
    mesh = grid.horizontal
    q = Array{FT}(undef, size(air_mass))
    scratch = HybridRemapScratch(FT, size(source.raw, 3), length(source.ap))
    src_q, src_qv = scratch.q, scratch.qv
    A_tgt = grid.vertical.A
    B_tgt = grid.vertical.B
    surface_pressure === nothing && throw(ArgumentError(
        "build_initial_mixing_ratio(::AtmosGrid{<:ReducedGaussianMesh}, ...) " *
        "with vertical-interp init kind=$(kind) requires `surface_pressure` " *
        "(the binary's stored ps, length = ncells). Pass " *
        "`window.surface_pressure` from `load_transport_window`."))
    length(surface_pressure) == size(air_mass, 1) || throw(DimensionMismatch(
        "surface_pressure length $(length(surface_pressure)) must match " *
        "air_mass cells $(size(air_mass, 1))"))
    qv === nothing || size(qv) == size(air_mass) || throw(DimensionMismatch(
        "qv size $(size(qv)) must match air_mass size $(size(air_mass))"))
    source.needs_vinterp && _warn_surface_pressure_approximation(pressure_basis, qv)

    for j in 1:nrings(mesh)
        lat = mesh.latitudes[j]
        lons = ring_longitudes(mesh, j)
        for i in eachindex(lons)
            c = cell_index(mesh, i, j)
            lon = lons[i]
            _sample_bilinear_profile!(src_q, source.raw, source.lon, source.lat, lon, lat)
            source.qv === nothing || _sample_bilinear_profile!(
                src_qv, source.qv, source.lon, source.lat, lon, lat)
            target_qv = !source.needs_vinterp || qv === nothing ? nothing : @view(qv[c, :])
            source_qv = source.qv === nothing ? nothing : src_qv
            _file_ic_column!(@view(q[c, :]), scratch, source_qv,
                             source.needs_vinterp, source.ap, source.bp,
                             pressure_basis, source.needs_vinterp ? surface_pressure[c] : nothing,
                             A_tgt, B_tgt, target_qv)
        end
    end

    return q
end

# ---------------------------------------------------------------------------
# pack_initial_tracer_mass — basis-aware VMR → tracer-mass conversion
#
# Rule: IC VMRs are dry.
# - DryBasis:   air_mass == m_dry   → rm = vmr .* air_mass
# - MoistBasis: air_mass == m_moist → rm = vmr .* air_mass .* (1 .- qv)
# ---------------------------------------------------------------------------

"""
    pack_initial_tracer_mass(grid, air_mass, vmr_dry; mass_basis::AbstractMassBasis,
                                                      qv = nothing)

Convert dry volume mixing ratio `vmr_dry` to conservative model storage
matching the binary's mass-basis contract. On `DryBasis` that storage is
`vmr_dry × dry_air_mass`; it is not physical kg species. Returns new storage
with the same shape as `air_mass`. On cubed-sphere grids this is a six-tuple
of halo-padded panels, with the halos set to zero.

## Dispatch

- `mass_basis::DryBasis` (the default runtime basis) — `air_mass` is
  `m_dry`. Result: `vmr_dry .* air_mass`. `qv` is ignored.
- `mass_basis::MoistBasis` — `air_mass` is `m_moist`, so the dry VMR is
  converted with `1 - qv`. Result: `vmr_dry .* air_mass .* (1 .- qv)`. `qv`
  must be supplied from the first transport window; missing `qv` errors.

CS dispatch handles per-panel halo packing.

## Arguments

- `grid`           — `AtmosGrid` with a lat-lon, reduced-Gaussian, or cubed-sphere mesh.
- `air_mass`       — storage-shaped air mass from the transport window;
                     a six-tuple of halo-padded 3D panels on cubed-sphere grids.
- `vmr_dry`        — dry volume mixing ratio. On cubed-sphere grids, a six-tuple
                     of interior 3D panels; otherwise the same shape as `air_mass`.
- `mass_basis`     — `DryBasis()` or `MoistBasis()`.
- `qv`             — specific humidity, same shape as `air_mass`;
                     required iff `mass_basis isa MoistBasis`.
"""
function pack_initial_tracer_mass(grid::AtmosGrid, air_mass, vmr_dry;
                                  mass_basis::AbstractMassBasis,
                                  qv = nothing)
    return _pack_tracer_mass(grid, air_mass, vmr_dry, mass_basis, qv)
end

function _pack_tracer_mass(::AtmosGrid{<:Union{LatLonMesh,ReducedGaussianMesh}}, air_mass, vmr_dry, ::DryBasis, qv)
    return vmr_dry .* air_mass
end


function _pack_tracer_mass(::AtmosGrid{<:Union{LatLonMesh,ReducedGaussianMesh}}, air_mass, vmr_dry, ::MoistBasis, qv)
    qv === nothing && throw(ArgumentError(
        "pack_initial_tracer_mass on MoistBasis requires qv (specific humidity) " *
        "from the first transport window to convert dry VMR to moist storage; got qv=nothing."))
    size(qv) == size(air_mass) || throw(DimensionMismatch(
        "qv shape $(size(qv)) must match air_mass shape $(size(air_mass))"))
    return vmr_dry .* air_mass .* (1 .- qv)
end


include("cubed_sphere.jl")
include("surface_flux.jl")

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export uniform_background
export FileInitialConditionSource
export AbstractInitialConditionSpec, UniformInitialConditionSpec
export LatitudeStepInitialConditionSpec, BoundaryLayerEnhancedInitialConditionSpec
export GaussianBlobInitialConditionSpec, FileInitialConditionSpec
export CatrineCO2InitialConditionSpec, PressureLayerInitialConditionSpec
export CSNativeInitialConditionSpec, initial_condition_spec
export build_initial_mixing_ratio
export pack_initial_tracer_mass
export FileSurfaceFluxField, TimeVaryingFileSurfaceFluxField
export build_surface_flux_source, build_surface_flux_sources
export surface_flux_species

end # module InitialConditionIO
