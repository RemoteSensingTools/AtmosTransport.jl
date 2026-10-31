"""
    Output

Topology-aware diagnostic output for AtmosTransport runtime products.

This module owns the public NetCDF output contract. Runtime code should capture
model state with [`capture_snapshot`](@ref) and write files with
[`write_snapshot_netcdf`](@ref) instead of defining ad-hoc NetCDF layouts in
runner code.

The writer is intentionally topology-dispatched:

- [`LatLonMesh`](@ref) writes regular CF lon/lat coordinates.
- [`ReducedGaussianMesh`](@ref) writes native cell-indexed diagnostics plus a
  legacy lon/lat raster view for current debug plots.
- [`CubedSphereMesh`](@ref) writes panel-native diagnostics with GEOS-style
  `lons`, `lats`, `corner_lons`, `corner_lats`, `cell_area`, and
  `cubed_sphere` metadata so Panoply and downstream tools have enough geometry
  context to render C-grid snapshots.

New topologies should add methods for the small internal schema/diagnostic
functions in this folder; they should not special-case the runner.

Observation sampling lives in `AtmosTransport.Observations`.
"""
module Output

using ..Parameters: EARTH_RADIUS
using ..Grids: HybridSigmaPressure, cell_centers, parse_panel_convention, parse_cs_definition,
               panel_convention_tag
using ..State: basis_symbol

using Dates
using HDF5_jll
using JSON3
using NCDatasets
using Printf
using KernelAbstractions: @kernel, @index, get_backend, synchronize, CPU as KA_CPU
using ..Architectures: CPU, compensated_total, _total_accumulator_type

using ..ConfigChecks: config_bool, config_int, config_float, parse_choice, parse_float_type

# netcdf-c is not thread-safe, even across files. Daily snapshot files are
# written on a spawned task while observation and single-file snapshot appends
# happen on the main thread, so every runtime NetCDF write takes this lock.
const _NETCDF_IO_LOCK = ReentrantLock()

"""
    with_netcdf_lock(f)

Run `f()` holding the process-wide NetCDF lock. Every runtime NetCDF write
(snapshot files, the snapshot stream, observation streams) goes through it,
because netcdf-c is not thread-safe and daily snapshot files are written on a
background task.
"""
with_netcdf_lock(f) = lock(f, _NETCDF_IO_LOCK)

import ..expand_data_path
import ..source_revision
using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
               nx, ny, nrings, ring_longitudes, cell_index, ncells,
               cell_area, panel_cell_center_lonlat, panel_cell_corner_lonlat,
               cs_definition, coordinate_law, center_law, longitude_offset_deg,
               cs_definition_tag, coordinate_law_tag, center_law_tag
using ..State: mass_basis, tracer_names, get_tracer

export AbstractSnapshotFrame, SnapshotFrame, SelectedSnapshotFrame, SnapshotWriteOptions
export AbstractOutputSchedule, AbstractOutputPartition
export AbstractOutputFormat, NetCDFOutput, BinaryMmapOutput, output_format_label
export AbstractLayerSelection, FullLayerSelection, SelectedLayerSelection
export NoLayerSelection, TracerOutputFields, OutputFieldSpec
export ExplicitSnapshotSchedule, IntervalSnapshotSchedule
export SingleOutputFile, DailyOutputFiles, RuntimeOutputSpec, RuntimeOutputMetadata
export runtime_output_spec, snapshot_hours, output_enabled, output_path
export set_physics_attributes!, set_time_origin!
export output_fields, output_field_spec, output_path_for_day
export tracer_fields, layer_selection_label
export read_snapshot_binary
export capture_snapshot, write_snapshot_netcdf, write_snapshot_binary
export define_product_attributes!, define_cs_horizontal_geometry!, define_provenance_attributes!
export column_mean_mixing_ratio, layer_mass_per_area
export OUTPUT_PATH_KEYS, OUTPUT_KEYS, OUTPUT_FIELDS_KEYS, TRACER_OUTPUT_FIELDS_KEYS,
       OUTPUT_COLUMN_MASS_KEYS, OUTPUT_PER_TRACER_KEYS, OUTPUT_FORMATS,
       SNAPSHOT_SCHEDULE_KEYS, SNAPSHOT_INTERVAL_KEYS, schedule_has_times, check_output_targets
export with_netcdf_lock

include("snapshots.jl")
include("runtime_output.jl")
include("diagnostics.jl")
include("snapshot_totals.jl")
include("selected_snapshots.jl")
include("netcdf_schema.jl")
include("netcdf_writer.jl")
include("netcdf_stream.jl")
include("binary_writer.jl")
include("binary_reader.jl")

end # module Output
