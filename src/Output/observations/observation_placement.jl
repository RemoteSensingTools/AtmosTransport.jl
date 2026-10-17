# ---------------------------------------------------------------------------
# Vertical placement of observation requests.
#
# A record states its height in one of three ways: a height above the ground
# (`intake_height`, or `altitude - elevation`), an altitude above sea level
# without its ground elevation (ObsPack aircraft-insitu and AirCore records),
# or nothing (satellite columns, station tables without heights); ObsPack
# records may also give their pressure. The model ground differs from the true
# ground, so a record is placed by its height above the ground, measured up
# from the model surface with the model's own hypsometric layer heights; an
# altitude is first turned into a height above the ground with the record's
# elevation or a DEM at the point (never with the model surface height: that
# would place it by absolute altitude). The vertical coordinates a run uses,
# height above ground and pressure, are tried in the configured order
# (`vertical_coordinates`; by default height first, then pressure); a record
# that none of them place is refused or skipped and counted. Pressure
# placement compares the record's (total) pressure with the total interface
# pressures of the column (`AbstractPressureBasis`). Placement is resolved
# once per request when the sampler is built; the layer is chosen at every
# sample from that sample's column (`placement_layer`).
#
# The static fields involved (model-grid surface height, a point DEM) come
# from external datasets read by GriddedIO, which builds the concrete types
# below; this file only defines the interfaces and the placement rules.
# ---------------------------------------------------------------------------

# -- model surface height ----------------------------------------------------

"""
    AbstractModelSurfaceHeight

Surface height of the model grid (m above sea level): the ground the model's
layer heights are measured from. [`NoModelSurfaceHeight`](@ref) (default) or
[`ModelSurfaceHeight`](@ref), read from a per-grid dataset by
`GriddedIO.read_model_surface_height`. Query it with
[`model_surface_height`](@ref).
"""
abstract type AbstractModelSurfaceHeight end

"No model surface height configured; [`model_surface_height`](@ref) returns `NaN`."
struct NoModelSurfaceHeight <: AbstractModelSurfaceHeight end

"""
    ModelSurfaceHeight(field, path)

Per-cell surface height of the model grid (m above sea level), in the mesh's
native layout: an `(Nx, Ny)` matrix on a lat-lon mesh, a vector over the
cells on a reduced Gaussian mesh, a 6-tuple of `(Nc, Nc)` panels on the cubed
sphere. `path` is the per-grid dataset it came from (written to the
observation files as provenance).
"""
struct ModelSurfaceHeight{A} <: AbstractModelSurfaceHeight
    field::A
    path::String
end

"""
    model_surface_height(surface, cell::CellLocation) -> Float64

Model surface height (m above sea level) of the cell containing an
observation; `NaN` for [`NoModelSurfaceHeight`](@ref).
"""
model_surface_height(::NoModelSurfaceHeight, ::CellLocation) = NaN
model_surface_height(h::ModelSurfaceHeight{<:AbstractMatrix}, c::CellLocation) = Float64(h.field[c.i, c.j])
model_surface_height(h::ModelSurfaceHeight{<:AbstractVector}, c::CellLocation) = Float64(h.field[c.cell])
model_surface_height(h::ModelSurfaceHeight{<:NTuple{6, AbstractMatrix}}, c::CellLocation) =
    Float64(h.field[c.panel][c.i, c.j])

"Throw unless the surface-height layout matches `mesh` (the sampler checks it once)."
check_surface_height_layout(::NoModelSurfaceHeight, mesh) = nothing
check_surface_height_layout(h::ModelSurfaceHeight{<:AbstractMatrix}, mesh::LatLonMesh) =
    _check_layout(size(h.field) == (mesh.Nx, mesh.Ny), h, mesh)
check_surface_height_layout(h::ModelSurfaceHeight{<:AbstractVector}, mesh::ReducedGaussianMesh) =
    _check_layout(length(h.field) == ncells(mesh), h, mesh)
check_surface_height_layout(h::ModelSurfaceHeight{<:NTuple{6, AbstractMatrix}}, mesh::CubedSphereMesh) =
    _check_layout(all(p -> size(p) == (mesh.Nc, mesh.Nc), h.field), h, mesh)
check_surface_height_layout(h::ModelSurfaceHeight, mesh) = _check_layout(false, h, mesh)

function _check_layout(ok::Bool, h::ModelSurfaceHeight, mesh)
    ok || throw(DimensionMismatch(
        "model surface height from $(h.path) ($(typeof(h.field))) does not match the $(summary(mesh))"))
    return nothing
end

# -- point ground elevation (DEM) --------------------------------------------

"""
    AbstractPointElevation

Ground elevation (m above sea level) at a point, from a high-resolution
digital elevation model: [`NoPointElevation`](@ref) (default) or
`GriddedIO.LatLonPointElevation` (`GriddedIO.read_point_elevation`). Query
it with [`point_elevation`](@ref).
"""
abstract type AbstractPointElevation end

"No DEM configured; [`point_elevation`](@ref) returns `NaN`."
struct NoPointElevation <: AbstractPointElevation end

"""
    point_elevation(dem, lon, lat) -> Float64

Ground elevation (m above sea level) of `dem` at the point, `NaN` where the
DEM has no value (outside its coverage, fill, or [`NoPointElevation`](@ref)).
"""
point_elevation(::NoPointElevation, lon::Real, lat::Real) = NaN

"""
    ObservationTerrain(; model_surface = NoModelSurfaceHeight(), dem = NoPointElevation())

The static ground fields observation placement may use: the model-grid
surface height ([`AbstractModelSurfaceHeight`](@ref)) and a point DEM
([`AbstractPointElevation`](@ref)). The runner builds it from
`[output.observations]` through GriddedIO and passes it to
[`build_observation_sampler`](@ref).
"""
struct ObservationTerrain{S <: AbstractModelSurfaceHeight, D <: AbstractPointElevation}
    model_surface::S
    dem::D
end
ObservationTerrain(; model_surface::AbstractModelSurfaceHeight = NoModelSurfaceHeight(),
                     dem::AbstractPointElevation = NoPointElevation()) = ObservationTerrain(model_surface, dem)

"""
    terrain_source(field) -> String

Where a terrain field came from, written to the observation files: the file
path, `""` for [`NoModelSurfaceHeight`](@ref) / [`NoPointElevation`](@ref).
Concrete DEM types add a method.
"""
terrain_source(::NoModelSurfaceHeight) = ""
terrain_source(h::ModelSurfaceHeight) = h.path
terrain_source(::NoPointElevation) = ""
terrain_source(field) = string(typeof(field))

# -- configuration -----------------------------------------------------------

"What the sampler does with requests it cannot place in the column."
abstract type AbstractUnplacedPolicy end
"`unplaced_records = \"error\"` (default): building the sampler fails, naming the requests."
struct ErrorOnUnplaced <: AbstractUnplacedPolicy end
"`unplaced_records = \"skip\"`: such requests are dropped and counted (`n_unplaced_*` attributes)."
struct SkipUnplaced <: AbstractUnplacedPolicy end
unplaced_policy_label(::ErrorOnUnplaced) = :error
unplaced_policy_label(::SkipUnplaced) = :skip
const _UNPLACED_POLICIES = (error = ErrorOnUnplaced(), skip = SkipUnplaced())

"How records that carry an altitude are placed once their height above the ground is known."
abstract type AbstractAltitudePlacement end
"Every record at its height above the ground, measured up from the model surface (default)."
struct AboveGroundPlacement <: AbstractAltitudePlacement end
"""
    AbsoluteAltitudeAbove(min_height_agl_m)

`absolute_altitude_above_agl_m = h`: records at least `h` m above the ground
(free-tropospheric aircraft and AirCore samples) are placed by their pressure
when the record has one, else by their altitude above sea level against the
model's geopotential heights (model surface height plus layer heights). The
altitude is the record's `altitude`, else its ground elevation (the record's,
else the DEM's) plus its `intake_height`; a record with neither keeps its
height above the ground. Records below `h` keep [`AboveGroundPlacement`](@ref).
Needs a model surface height (`surface_height_file`).
"""
struct AbsoluteAltitudeAbove <: AbstractAltitudePlacement
    min_height_agl_m::Float64
    function AbsoluteAltitudeAbove(h::Real)
        isfinite(h) && h >= 0 || throw(ArgumentError(
            "absolute_altitude_above_agl_m must be a finite number >= 0; got $(h)"))
        return new(Float64(h))
    end
end

"""
    AbstractVerticalCoordinate

A vertical coordinate that can place a record in the model column:
[`HeightAboveGroundCoordinate`](@ref) or [`PressureCoordinate`](@ref).
[`VerticalPlacementSpec`](@ref) lists the coordinates a run uses in
preference order (`vertical_coordinates`); [`resolve_placement`](@ref) places
each record by the first one it can.
"""
abstract type AbstractVerticalCoordinate end

"""
`"height_above_ground"`: the record's height ``h`` above the true ground, the
`intake_height`, else `altitude − elevation`, else `altitude` minus the DEM
ground elevation at the point, measured up from the model surface with the
model's hypsometric layer heights (`vertical_placement` codes
`record_height` and `altitude_minus_dem`).
"""
struct HeightAboveGroundCoordinate <: AbstractVerticalCoordinate end

"""
`"pressure"`: the record's pressure (ObsPack `pressure`, a total pressure)
against the column's interface pressures, built as the
[`AbstractPressureBasis`](@ref) says (`vertical_placement` code `pressure`).
"""
struct PressureCoordinate <: AbstractVerticalCoordinate end

const _VERTICAL_COORDINATES = (height_above_ground = HeightAboveGroundCoordinate(),
                               pressure = PressureCoordinate())
"Height above the true ground first, then pressure: `vertical_coordinates = [\"height_above_ground\", \"pressure\"]`."
const DEFAULT_VERTICAL_COORDINATES = (HeightAboveGroundCoordinate(), PressureCoordinate())
vertical_coordinate_label(::HeightAboveGroundCoordinate) = :height_above_ground
vertical_coordinate_label(::PressureCoordinate) = :pressure
_uses_coordinate(coordinates::Tuple, ::Type{C}) where {C <: AbstractVerticalCoordinate} =
    any(c -> c isa C, coordinates)

"""
    AbstractPressureBasis

The interface pressures a pressure-placed record is compared with
(`pressure_basis`): [`TotalFromHumidity`](@ref) (default) or
[`DryInterfaces`](@ref). A record's pressure is the total (ambient)
pressure, dry air plus water vapour. The air mass of a dry-basis run gives
dry interface pressures, ``p_{k+1/2} = p_{k-1/2} + g\\,m_k/A`` (the weight of
the dry air above), which are lower than the total pressure at the same
interface by the weight of the water vapour above it: about 0.5 % of the
surface pressure for 50 kg m⁻² of precipitable water.
"""
abstract type AbstractPressureBasis end

"""
`pressure_basis = "total"` (default): compare a record's pressure with the
column's total interface pressures. Specific humidity ``q_k`` (kg of water
vapour per kg of moist air) makes the moist air mass of layer ``k``
``m_k/(1 - q_k)`` for a dry air mass ``m_k`` (kg), so on a dry-basis run the
hydrostatic total pressures are

```math
p_{1/2} = p_\\mathrm{top}, \\qquad
p_{k+1/2} = p_{k-1/2} + \\frac{g\\,m_k}{A\\,(1 - q_k)},
```

with ``A`` the cell area and ``p_\\mathrm{top}`` the pure-pressure model-top
interface `A_ifc[1]`, a total pressure in the met model's hybrid coordinate
(level 1 is the top). This inverts the preprocessing's dry conversion
``Δp_\\mathrm{dry} = (1 - q)\\,Δp``. On merged or remapped levels the binary
carries a dry-mass-weighted ``q``, so the inverse is not exact there; on the
golden `pre_ll72` binary (ERA5 L137 merged to 34 levels) the rebuilt total
surface pressure differs from the stored one by −0.012 Pa on average and
0.125 Pa at most, where the dry interfaces are 169 Pa low on average and
700 Pa at worst. The humidity is
the binary's layer-mean specific humidity at the sampled boundary:
`qv_start` / `qv_end` of lat-lon binaries preprocessed with
`output.include_qv = true`; on cubed-sphere binaries with the GCHP VDIFF
payload the window-start VDIFF `qv` of the next window (the run's last window
end takes the ended window's, one window old). Reduced-Gaussian binaries, and
cubed-sphere binaries without the VDIFF payload, carry no humidity. On a
moist-basis run the air mass already includes the vapour and its interfaces
are total pressures without humidity. A dry-basis run whose binaries do not
all carry humidity cannot build them: building a sampler that places any
record by pressure then fails unless `pressure_basis = "dry"`.
"""
struct TotalFromHumidity <: AbstractPressureBasis end

"""
`pressure_basis = "dry"`: compare a record's pressure with the dry interface
pressures ``p_{k+1/2} = p_{k-1/2} + g\\,m_k/A`` of a dry-basis run, for runs
whose binaries carry no humidity (the sampler's only rule before issue #20).
The dry interfaces lie at lower pressure than the total ones, so a record can
be placed one or more layers lower in the column than its total pressure
says, most likely near the surface where the vapour weight above an
interface is largest and the layers are thinnest. Not available on a moist-basis run.
"""
struct DryInterfaces <: AbstractPressureBasis end

const _PRESSURE_BASES = (total = TotalFromHumidity(), dry = DryInterfaces())
pressure_basis_label(::TotalFromHumidity) = :total
pressure_basis_label(::DryInterfaces) = :dry

"""
    VerticalPlacementSpec(; surface_height_file = nothing, dem_file = nothing, dem_variable = "z",
                            unplaced = ErrorOnUnplaced(), altitude_placement = AboveGroundPlacement(),
                            coordinates = DEFAULT_VERTICAL_COORDINATES,
                            pressure_basis = TotalFromHumidity())

The vertical-placement settings of `[output.observations]`: the per-grid
model surface height dataset, an optional point DEM (file and variable), the
[`AbstractUnplacedPolicy`](@ref), the [`AbstractAltitudePlacement`](@ref),
the [`AbstractVerticalCoordinate`](@ref)s in preference order (a non-empty
tuple without repeats; by default height above ground, then pressure), and
the [`AbstractPressureBasis`](@ref) of pressure placement.
"""
struct VerticalPlacementSpec{U <: AbstractUnplacedPolicy, A <: AbstractAltitudePlacement,
                             C <: Tuple{Vararg{AbstractVerticalCoordinate}}, P <: AbstractPressureBasis}
    surface_height_file::Union{Nothing, String}
    dem_file::Union{Nothing, String}
    dem_variable::String
    unplaced::U
    altitude_placement::A
    coordinates::C
    pressure_basis::P
    function VerticalPlacementSpec(surface_height_file::Union{Nothing, String}, dem_file::Union{Nothing, String},
                                   dem_variable::String, unplaced::U, altitude_placement::A, coordinates::C,
                                   pressure_basis::P) where {U <: AbstractUnplacedPolicy,
                                                             A <: AbstractAltitudePlacement,
                                                             C <: Tuple{Vararg{AbstractVerticalCoordinate}},
                                                             P <: AbstractPressureBasis}
        isempty(coordinates) && throw(ArgumentError("vertical placement needs at least one vertical coordinate"))
        allunique(coordinates) || throw(ArgumentError(
            "vertical coordinates must not repeat; got $(join(vertical_coordinate_label.(coordinates), ", "))"))
        return new{U, A, C, P}(surface_height_file, dem_file, dem_variable, unplaced, altitude_placement,
                               coordinates, pressure_basis)
    end
end
VerticalPlacementSpec(; surface_height_file::Union{Nothing, AbstractString} = nothing,
                        dem_file::Union{Nothing, AbstractString} = nothing,
                        dem_variable::AbstractString = "z",
                        unplaced::AbstractUnplacedPolicy = ErrorOnUnplaced(),
                        altitude_placement::AbstractAltitudePlacement = AboveGroundPlacement(),
                        coordinates::Tuple{Vararg{AbstractVerticalCoordinate}} = DEFAULT_VERTICAL_COORDINATES,
                        pressure_basis::AbstractPressureBasis = TotalFromHumidity()) =
    VerticalPlacementSpec(surface_height_file === nothing ? nothing : String(surface_height_file),
                          dem_file === nothing ? nothing : String(dem_file), String(dem_variable),
                          unplaced, altitude_placement, coordinates, pressure_basis)

# The spec names files the runner loads through GriddedIO; a sampler built
# without them would silently place fewer records, or none by absolute altitude.
# `label` names the config table (`"[output.observations]"`).
function check_terrain(spec::VerticalPlacementSpec, terrain::ObservationTerrain, label::AbstractString)
    spec.surface_height_file === nothing || !(terrain.model_surface isa NoModelSurfaceHeight) ||
        throw(ArgumentError("$(label).surface_height_file is set but no model surface " *
                            "height was loaded (load it with GriddedIO.read_model_surface_height)"))
    spec.dem_file === nothing || !(terrain.dem isa NoPointElevation) ||
        throw(ArgumentError("$(label).dem_file is set but no DEM was loaded " *
                            "(load it with GriddedIO.read_point_elevation)"))
    _check_altitude_option(spec.altitude_placement, terrain, label)
    return nothing
end
_check_altitude_option(::AboveGroundPlacement, terrain, label) = nothing
_check_altitude_option(::AbsoluteAltitudeAbove, terrain, label) = terrain.model_surface isa NoModelSurfaceHeight &&
    throw(ArgumentError("$(label).absolute_altitude_above_agl_m needs the model surface height " *
                        "($(label).surface_height_file)"))

# -- records without any vertical coordinate ---------------------------------

"""
    AbstractMissingHeightRule

What a source's records mean when they give no vertical coordinate at all
(no `intake_height`, `altitude` or `pressure`), chosen by the source type
([`missing_height_rule`](@ref)): [`LowestLayerWithoutHeight`](@ref) or
[`UnplacedWithoutHeight`](@ref).
"""
abstract type AbstractMissingHeightRule end

"""
The record is a column or surface product: it is sampled in the lowest layer
(`vertical_placement = lowest_layer`). OCO-2 soundings (their value is the
column; the intake value is the surface layer) and table rows without height
columns (surface points).
"""
struct LowestLayerWithoutHeight <: AbstractMissingHeightRule end

"""
The record should state a height and does not: it cannot be placed
(`unplaced_records`). ObsPack records, where every platform carries a height.
"""
struct UnplacedWithoutHeight <: AbstractMissingHeightRule end

"""
    missing_height_rule(source) -> AbstractMissingHeightRule

How records of `source` without any vertical coordinate are placed:
[`LowestLayerWithoutHeight`](@ref) for `OCO2LiteSource` and `TableSource`,
[`UnplacedWithoutHeight`](@ref) for `ObsPackSource`.
"""
missing_height_rule(::OCO2LiteSource) = LowestLayerWithoutHeight()
missing_height_rule(::TableSource) = LowestLayerWithoutHeight()
missing_height_rule(::ObsPackSource) = UnplacedWithoutHeight()

# -- placement ---------------------------------------------------------------

"""
Integer `vertical_placement` codes of the observation files, the method that
placed each request (see `vertical_placement_codes_description`):

| Code | Name | Layer chosen by |
|---|---|---|
| 0 | `record_height` | the record's height above ground (`intake_height`, or `altitude − elevation`) |
| 1 | `altitude_minus_dem` | `altitude` − DEM ground elevation at the point |
| 2 | `pressure` | the record's pressure against the interface pressures of the `pressure_basis` (total by default) |
| 3 | `absolute_altitude` | `altitude` against model surface height + layer heights (`absolute_altitude_above_agl_m`) |
| 4 | `lowest_layer` | no vertical coordinate, for sources whose records are column or surface products |
"""
const VERTICAL_PLACEMENT_CODES = (record_height = Int8(0), altitude_minus_dem = Int8(1), pressure = Int8(2),
                                  absolute_altitude = Int8(3), lowest_layer = Int8(4))
const _VERTICAL_PLACEMENT_TEXT = (
    record_height = "height above ground from the record (intake_height, or altitude - elevation)",
    altitude_minus_dem = "altitude - DEM ground elevation",
    pressure = "record pressure vs model interface pressures (total, or dry: see pressure_basis)",
    absolute_altitude = "altitude above sea level vs model surface height + layer heights",
    lowest_layer = "no vertical coordinate (column or surface product): lowest model layer")
vertical_placement_codes_description() =
    join(("$(VERTICAL_PLACEMENT_CODES[k]) = $(k): $(_VERTICAL_PLACEMENT_TEXT[k])"
          for k in keys(VERTICAL_PLACEMENT_CODES)), "; ")
const _VERTICAL_PLACEMENT_MEANINGS = join(string.(keys(VERTICAL_PLACEMENT_CODES)), " ")

"""
    VerticalPlacement(method, height_agl_m, pressure_pa, model_surface_m)

How one request is placed in the column, resolved once by
[`resolve_placement`](@ref): `method` is a
[`VERTICAL_PLACEMENT_CODES`](@ref) code; `height_agl_m` the height above the
model surface that selects the layer (NaN for pressure and lowest-layer
placement); `pressure_pa` the pressure that selects it (pressure placement
only); `model_surface_m` the model surface height of the cell (NaN without a
surface-height dataset).
"""
struct VerticalPlacement
    method::Int8
    height_agl_m::Float64
    pressure_pa::Float64
    model_surface_m::Float64
end

_record_pressure(r::SoundingRequest) = r.pressure_pa
_record_pressure(::SiteRequest) = NaN
# Altitude above sea level: stated, else the ground elevation (the record's,
# else the DEM's at the point) plus the intake height; NaN when neither exists.
function _record_altitude(r::Union{SoundingRequest, SiteRequest}, dem::AbstractPointElevation)
    isfinite(r.altitude_m) && return r.altitude_m
    isfinite(r.intake_height_m) || return NaN
    isfinite(r.elevation_m) && return r.elevation_m + r.intake_height_m
    return point_elevation(dem, r.lon, r.lat) + r.intake_height_m
end

"""
    resolve_placement(spec::VerticalPlacementSpec, missing_height, request, cell, terrain)
        -> Union{VerticalPlacement, Nothing}

Place a [`SoundingRequest`](@ref) or [`SiteRequest`](@ref) in the column of
its containing `cell` by the first of `spec.coordinates`
([`AbstractVerticalCoordinate`](@ref), in preference order) that places it:

- [`HeightAboveGroundCoordinate`](@ref): the record's height above ground
  ``h`` (`intake_height`, or `altitude − elevation` with the record's or
  station's elevation; `record_height`), else ``h`` = `altitude` minus the
  DEM ground elevation at the point (`altitude_minus_dem`). The record is
  placed ``h`` above the model surface.
- [`PressureCoordinate`](@ref): the record's `pressure` (`pressure`), when it
  is `valid_pressure` (finite and positive).

A record that none of the coordinates place is unplaced (`nothing`) when it
states any vertical coordinate (an `intake_height`, an `altitude`, a valid
`pressure`), e.g. an `altitude` without a ground elevation (no record
`elevation`, no DEM value), or a coordinate the run does not use. One without
any vertical coordinate goes to the lowest layer (`lowest_layer`) under
[`LowestLayerWithoutHeight`](@ref) and is unplaced under
[`UnplacedWithoutHeight`](@ref) (`missing_height`, from
[`missing_height_rule`](@ref) of the record's source). The model surface
height is never used as the true ground.

`spec.altitude_placement = AbsoluteAltitudeAbove(h₀)` then re-places records
placed by a height ``h ≥ h_0`` by their pressure (when `pressure` is one of
the coordinates), else by absolute altitude: the stated `altitude`, else the
ground elevation (the record's, else the DEM value at the point) plus
`intake_height`; a record with neither keeps its height above the ground.
"""
function resolve_placement(spec::VerticalPlacementSpec, missing_height::AbstractMissingHeightRule,
                           r::Union{SoundingRequest, SiteRequest}, cell::CellLocation, terrain::ObservationTerrain)
    zs = model_surface_height(terrain.model_surface, cell)
    placement = _first_placement(spec.coordinates, r, zs, terrain.dem)
    if placement === nothing
        # A record that states a coordinate is never sampled in the lowest layer for lack of a usable one.
        _has_vertical_coordinate(r) && return nothing
        return _without_height(missing_height, zs)
    end
    return _apply_altitude_option(spec.altitude_placement, spec.coordinates, placement, r, zs, terrain.dem)
end

# The first coordinate, in preference order, that places the record (recursion
# over the tuple keeps the dispatch static).
_first_placement(::Tuple{}, r, zs::Float64, dem::AbstractPointElevation) = nothing
function _first_placement(coordinates::Tuple, r, zs::Float64, dem::AbstractPointElevation)
    placement = _place_by(first(coordinates), r, zs, dem)
    return placement === nothing ? _first_placement(Base.tail(coordinates), r, zs, dem) : placement
end

function _place_by(::HeightAboveGroundCoordinate, r, zs::Float64, dem::AbstractPointElevation)
    codes = VERTICAL_PLACEMENT_CODES
    isfinite(r.intake_height_m) && return VerticalPlacement(codes.record_height, r.intake_height_m, NaN, zs)
    isfinite(r.altitude_m) || return nothing
    # A station's elevation can come from the file attributes while its rows
    # give only an altitude; readers already resolved row-level elevations.
    isfinite(r.elevation_m) && return VerticalPlacement(codes.record_height, r.altitude_m - r.elevation_m, NaN, zs)
    ground = point_elevation(dem, r.lon, r.lat)
    isfinite(ground) && return VerticalPlacement(codes.altitude_minus_dem, r.altitude_m - ground, NaN, zs)
    # An altitude without any ground elevation: never placed by the model ground.
    return nothing
end

function _place_by(::PressureCoordinate, r, zs::Float64, ::AbstractPointElevation)
    pressure = _record_pressure(r)
    valid_pressure(pressure) || return nothing
    return VerticalPlacement(VERTICAL_PLACEMENT_CODES.pressure, NaN, pressure, zs)
end

_has_vertical_coordinate(r) =
    isfinite(r.intake_height_m) || isfinite(r.altitude_m) || valid_pressure(_record_pressure(r))

_without_height(::LowestLayerWithoutHeight, zs::Float64) =
    VerticalPlacement(VERTICAL_PLACEMENT_CODES.lowest_layer, NaN, NaN, zs)
_without_height(::UnplacedWithoutHeight, ::Float64) = nothing

_apply_altitude_option(::AboveGroundPlacement, coordinates, placement::VerticalPlacement, r, zs, dem) = placement
function _apply_altitude_option(option::AbsoluteAltitudeAbove, coordinates::Tuple, placement::VerticalPlacement,
                                r, zs::Float64, dem::AbstractPointElevation)
    codes = VERTICAL_PLACEMENT_CODES
    # Only placements by a height above the ground are re-placed.
    placement.method in (codes.record_height, codes.altitude_minus_dem) || return placement
    placement.height_agl_m >= option.min_height_agl_m || return placement
    if _uses_coordinate(coordinates, PressureCoordinate)
        by_pressure = _place_by(PressureCoordinate(), r, zs, dem)
        by_pressure === nothing || return by_pressure
    end
    altitude = _record_altitude(r, dem)
    # Without an altitude (an intake height with neither elevation nor DEM
    # value) the record stays at its height above the ground.
    isfinite(altitude) && isfinite(zs) || return placement
    return VerticalPlacement(codes.absolute_altitude, altitude - zs, NaN, zs)
end

# A record placed by a height above the ground lower than this (m) is below the
# true ground: a wrong elevation, DEM or altitude (an undeclared fill). It is
# still sampled, in the lowest layer, and reported. 100 m exceeds the
# disagreement of a 1-arc-minute DEM with aircraft altitudes on the ground (in
# the OCO-2 MIP v5.0.1 ObsPack, 2022, none of the aircraft records is more
# than 100 m below ETOPO 2022).
const BELOW_GROUND_TOLERANCE_M = 100.0

"""
    below_ground(placement::VerticalPlacement) -> Bool

Whether a placement by height above the ground (`record_height`,
`altitude_minus_dem`) puts the record more than
`BELOW_GROUND_TOLERANCE_M` (100 m) below the ground. Such records are sampled
in the lowest layer; the sampler counts them and warns.
"""
below_ground(p::VerticalPlacement) =
    p.method in (VERTICAL_PLACEMENT_CODES.record_height, VERTICAL_PLACEMENT_CODES.altitude_minus_dem) &&
    p.height_agl_m < -BELOW_GROUND_TOLERANCE_M

"""
    pressure_layer_index(p_half, p) -> Int

Layer `k` with `p_half[k] <= p < p_half[k+1]` (k = 1 top). A pressure at or
above the surface interface selects the lowest layer; one below the column
top selects layer 1. `p` must be finite and positive
(`valid_pressure`).
"""
function pressure_layer_index(p_half::AbstractVector, p::Real)
    valid_pressure(p) || throw(ArgumentError("a pressure placement needs a finite, positive pressure; got $(p)"))
    nlevel = length(p_half) - 1
    @inbounds for k in nlevel:-1:1
        p >= p_half[k] && return k
    end
    return 1
end

"""
    placement_layer(placement, z_half, p_half) -> Int

Model layer (k = 1 top) of a [`VerticalPlacement`](@ref) in one sampled
column: by height above the model surface (`intake_layer_index` on the
hypsometric interface heights `z_half`), by pressure (`pressure_layer_index`
on the interface pressures `p_half`), or the lowest layer.
"""
function placement_layer(placement::VerticalPlacement, z_half::AbstractVector, p_half::AbstractVector)
    codes = VERTICAL_PLACEMENT_CODES
    placement.method == codes.lowest_layer && return length(z_half) - 1
    placement.method == codes.pressure && return pressure_layer_index(p_half, placement.pressure_pa)
    return intake_layer_index(z_half, placement.height_agl_m)
end

# -- pressure basis -----------------------------------------------------------

"""
    placement_pressures!(p_half, basis, mass_basis, air_column, humidity_column, area, g, p_top) -> p_half

Interface pressures (Pa, k = 1 top) a pressure-placed record is compared
with, for a column of air mass per cell (kg) of `mass_basis`:
[`TotalFromHumidity`](@ref) builds the total pressures from the dry air mass
and the specific humidity `humidity_column` (`total_interface_pressures!`) on
a dry-basis run and takes the moist air mass as it is on a moist-basis run;
[`DryInterfaces`](@ref) gives the dry interfaces of a dry-basis run
(`interface_pressures!`). `humidity_column` is ignored where no humidity is
needed.
"""
placement_pressures!(p_half, ::TotalFromHumidity, ::DryBasis, air_column, humidity_column, area, g, p_top) =
    total_interface_pressures!(p_half, air_column, humidity_column, area, g, p_top)
placement_pressures!(p_half, ::TotalFromHumidity, ::MoistBasis, air_column, humidity_column, area, g, p_top) =
    interface_pressures!(p_half, air_column, area, g, p_top)
placement_pressures!(p_half, ::DryInterfaces, ::DryBasis, air_column, humidity_column, area, g, p_top) =
    interface_pressures!(p_half, air_column, area, g, p_top)

"""
    check_pressure_basis(basis, mass_basis, has_humidity, n_pressure, label) -> Bool

Check, when the requests are placed, that `basis` can build the interface
pressures of `n_pressure` pressure-placed requests on a run of `mass_basis`
whose binaries carry specific humidity when `has_humidity`, and return
whether sampling needs that humidity (only [`TotalFromHumidity`](@ref) on a
dry-basis run with pressure-placed requests does). Throws for `DryInterfaces`
on a moist-basis run, and for `TotalFromHumidity` with pressure-placed
requests on a dry-basis run without humidity; `label` names the config table
(`"[output.observations]"`).
"""
check_pressure_basis(::DryInterfaces, ::MoistBasis, has_humidity::Bool, n_pressure::Integer,
                     label::AbstractString) = throw(ArgumentError(
    "$(label).pressure_basis = \"dry\" needs a dry-basis run: this run's air mass is moist, so its " *
    "interface pressures already are total pressures; omit pressure_basis"))
check_pressure_basis(::DryInterfaces, ::DryBasis, has_humidity::Bool, n_pressure::Integer, label::AbstractString) =
    false
check_pressure_basis(::TotalFromHumidity, ::MoistBasis, has_humidity::Bool, n_pressure::Integer,
                     label::AbstractString) = false
function check_pressure_basis(::TotalFromHumidity, ::DryBasis, has_humidity::Bool, n_pressure::Integer,
                              label::AbstractString)
    n_pressure == 0 && return false
    has_humidity || throw(ArgumentError(
        "$(n_pressure) requests are placed by pressure. Record pressures are total pressures, and " *
        "$(label).pressure_basis = \"total\" (the default) builds the total interface pressures from the " *
        "dry air mass and the specific humidity, but the run's binaries do not all carry humidity: lat-lon " *
        "binaries need qv_start/qv_end (preprocessed with output.include_qv = true), cubed-sphere binaries " *
        "the GCHP VDIFF payload (include_vdiff_fields = true), and reduced-Gaussian binaries carry none. " *
        "Set $(label).pressure_basis = \"dry\" to compare them with the " *
        "dry interface pressures instead (they lie at lower pressure, so a record can land one or more " *
        "layers too low), or leave \"pressure\" out of $(label).vertical_coordinates"))
    return true
end

"""
    PlacementColumn(nlevel)

Scratch for [`column_layer!`](@ref) on a column of `nlevel` layers:
`p_half` receives the interface pressures a pressure placement is compared
with, and `humidity` holds the column's specific humidity (k = 1 at the
top), which the caller fills when the pressure basis needs it.
"""
struct PlacementColumn
    p_half::Vector{Float64}
    humidity::Vector{Float64}
end
PlacementColumn(nlevel::Integer) = PlacementColumn(Vector{Float64}(undef, nlevel + 1), Vector{Float64}(undef, nlevel))

"""
    column_layer!(column::PlacementColumn, placement, z_half, air_column, area, g, p_top, basis, mass_basis) -> Int

Model layer (k = 1 top) of a [`VerticalPlacement`](@ref) in one column, the
rule shared by the forward sampler and any other caller: by height above the
model surface on the hypsometric interface heights `z_half`, or the lowest
layer (as [`placement_layer`](@ref)); a pressure placement against the
interface pressures of the [`AbstractPressureBasis`](@ref) `basis`, built
into `column.p_half` from `air_column` (kg per cell, of `mass_basis`), the
cell `area` (m²), gravity `g`, the model-top pressure `p_top` and, for
[`TotalFromHumidity`](@ref) on a dry basis, `column.humidity`.
"""
function column_layer!(column::PlacementColumn, placement::VerticalPlacement, z_half::AbstractVector,
                       air_column::AbstractVector, area::Real, g::Real, p_top::Real,
                       basis::AbstractPressureBasis, mass_basis::AbstractMassBasis)
    placement.method == VERTICAL_PLACEMENT_CODES.pressure || return placement_layer(placement, z_half, column.p_half)
    placement_pressures!(column.p_half, basis, mass_basis, air_column, column.humidity, area, g, p_top)
    return pressure_layer_index(column.p_half, placement.pressure_pa)
end
