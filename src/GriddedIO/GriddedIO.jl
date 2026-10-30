"""
    GriddedIO

The one owner of the conventions of native gridded input files (surface-flux
inventories, land fractions, initial-condition fields): how their coordinates
are found, which values are fill, how a latitude-longitude field is oriented,
which cell faces its coordinates imply, how its time axis is decoded, how a
compressed input is staged, and how its flux units are read. It also owns the
schema of the surface-flux files the model writes and the runtime reads
(`cs_native` and lat-lon), with their one writer and reader.

Every rule here was measured on the files the model reads (the TRENDY v14 S3
models, GFED5, FLUXCOM-X, CAMS/LMDZ, GridFED, EDGAR); the measurements are in
`docs/memos/2026-10-10_python_flux_port_p0_facts.md`. The rules are typed:
the caller names an axis ([`LongitudeAxis`](@ref)), a fill policy
([`MaskedAsZero`](@ref)), a time decoder ([`CFDecoder`](@ref)), and the
methods dispatch on it.

## Contents

- Coordinates (`coordinates.jl`): [`find_coordinate`](@ref) by variable name
  only, [`latlon_orientation`](@ref) / [`orient`](@ref) /
  [`orient_coordinates`](@ref), and [`read_raw`](@ref) with the fill policies
  [`FillValueOnly`](@ref), [`MaskedAsNaN`](@ref) and [`MaskedAsZero`](@ref).
- Source meshes (`source_meshes.jl`): [`source_latlon_mesh`](@ref), the cell
  faces of a file's latitude-longitude grid, by the face rule
  [`SnappedFaces`](@ref) (on `Grids.LatLonMesh(λᶠ, φᶠ; radius)`).
- Time axes (`time_axes.jl`): [`decode_time`](@ref) with the decoders
  [`CFDecoder`](@ref), [`MonthsSince`](@ref), [`DecimalYears`](@ref) and
  [`AuxiliaryTimeVariable`](@ref), chosen from the raw `units` string before
  any CFTime call, and [`decode_month_keys`](@ref), the months of an axis in its
  own calendar; [`MonthKey`](@ref), [`TimeInterval`](@ref),
  [`month_keys`](@ref), [`interval`](@ref), [`seconds_since`](@ref),
  [`decode_month_seconds`](@ref), [`check_last_month`](@ref); the interval
  each slice stands for, [`decode_slice_intervals`](@ref) (one file),
  [`known_slice_intervals`](@ref) and [`stamp_intervals`](@ref) (a series of
  files).
- Staging (`staging.jl`): [`stage_input`](@ref) decompresses `.nc.gz` and
  single-member `.nc.tar.gz` inputs into a cache directory.
- Flux units (`flux_units.jl`): [`parse_flux_units`](@ref) →
  [`FluxUnits`](@ref), [`mass_rate_factor`](@ref), [`kg_per_unit`](@ref),
  [`period_seconds`](@ref), [`canonical_units_string`](@ref).
- The `cs_native` flux file (`cs_native_flux.jl`), the runtime contract:
  [`CS_NATIVE_FLUX_DIMS`](@ref), the writer [`open_cs_native_writer`](@ref) /
  [`write_slices!`](@ref) / [`publish!`](@ref) / [`abandon!`](@ref), the
  streaming reader [`cs_native_flux_info`](@ref) / [`read_cs_native_flux!`](@ref)
  (the runtime's loader, panel by panel into the caller's arrays), the
  whole-file reader [`read_cs_native_flux`](@ref) and the cell-area gate
  [`check_cell_area`](@ref).
- Lat-lon flux files for `kind = "file"` (`latlon_flux.jl`):
  [`write_latlon_flux`](@ref).
- Static ground fields for observation placement (`surface_height.jl`): the
  per-grid model surface height dataset, written by
  [`write_model_surface_height`](@ref) and read by
  [`read_model_surface_height`](@ref) into `Observations.ModelSurfaceHeight`, and a
  lat-lon DEM for point lookups, [`read_point_elevation`](@ref) →
  [`LatLonPointElevation`](@ref) (bilinear `Observations.point_elevation`).

Molar masses and the year length come from `Parameters`
(`SPECIES_MOLAR_MASS`, `CARBON_MOLAR_MASS`, `DAYS_PER_JULIAN_YEAR`).

`src/AtmosTransport.jl` loads this module after `Regridding` and before
`Preprocessing`; it imports `Parameters`, `Grids`, `Output` (the shared
cubed-sphere geometry and provenance blocks) and `AtomicFiles`. See
`src/GriddedIO/README.md` for the file map and invariants.
"""
module GriddedIO

using ..Grids: cell_centers, CS_PANEL_COUNT, half_cell_bilinear_bracket, longitude_bilinear_bracket

using ..Grids: cell_areas, float64_twin

using ..Grids: lonlat_chord_distance

using Dates
using NCDatasets
using NCDatasets: CFTime
using SHA: sha1

using ..Parameters: SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS, DAYS_PER_JULIAN_YEAR,
                    SECONDS_PER_DAY, SPECIES_CARBON_ATOMS, SPECIES_LABELS
using ..Grids: LatLonMesh, CubedSphereMesh, ReducedGaussianMesh, cell_area, ncells,
               panel_cell_center_lonlat
import ..Output
import ..Observations
using ..Output: define_cs_horizontal_geometry!, define_provenance_attributes!
using ..Observations: ModelSurfaceHeight, AbstractPointElevation
using ..AtomicFiles: atomic_publish, open_atomic_staging, commit_atomic_staging,
                     abandon_atomic_staging

# coordinates.jl
export AbstractAxis, LongitudeAxis, LatitudeAxis, TimeAxis, VerticalAxis,
       find_coordinate
export LatLonOrientation, latlon_orientation, orient, orient_coordinates
export AbstractFillPolicy, FillValueOnly, MaskedAsNaN, MaskedAsZero, read_raw
# source_meshes.jl
export AbstractFaceRule, SnappedFaces, source_latlon_mesh
# time_axes.jl
export TimeInterval, MonthKey, month_keys, interval, seconds_since, check_last_month
export AbstractTimeDecoder, CFDecoder, MonthsSince, DecimalYears, AuxiliaryTimeVariable,
       AbstractMonthStamp, MonthStart, MidMonth, MonthEnd, DEFAULT_TIME_DECODERS, decode_time,
       decode_time_bounds, decode_slice_intervals, known_slice_intervals, stamp_intervals,
       STAMP_SPACING_TOLERANCE, decode_month_keys, decode_month_seconds
# staging.jl
export stage_input
# flux_units.jl
export AbstractSubstance, TargetSpecies, Carbon, Species
export AbstractAreaBasis, PerSquareMetre, PerCell
export AbstractTimeBasis, PerSecond, PerDay, PerMonth, PerYear
export FluxUnits, parse_flux_units, kg_per_unit, period_seconds, mass_rate_factor,
       canonical_units_string
# cs_native_flux.jl
export CS_NATIVE_FLUX_DIMS, CSNativeFluxWriter, open_cs_native_writer, write_slices!, publish!,
       abandon!, cs_native_flux_info, read_cs_native_flux!, read_cs_native_flux, check_cell_area
# latlon_flux.jl
export write_latlon_flux, check_latlon_cell_area
# surface_height.jl
export write_model_surface_height, read_model_surface_height, LatLonPointElevation, read_point_elevation

export hybrid_surface_first, validate_ap_units, VERTICAL_ORDERS, parse_vertical_order, PRESSURE_UNITS, HEIGHT_UNITS, file_vertical_order, infer_year_from_path, infer_year_from_time_units

include("initial_condition_conventions.jl")
include("coordinates.jl")
include("source_meshes.jl")
include("time_axes.jl")
include("staging.jl")
include("flux_units.jl")
include("cs_native_flux.jl")
include("latlon_flux.jl")
include("surface_height.jl")

end # module GriddedIO
