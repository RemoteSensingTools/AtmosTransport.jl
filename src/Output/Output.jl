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
"""
module Output

using Dates
using HDF5_jll
using JSON3
using NCDatasets
using Printf
using Statistics: median!
using TOML
using KernelAbstractions: @kernel, @index, get_backend, synchronize, CPU as KA_CPU

function _config_bool(value, path::AbstractString)
    value isa Bool || throw(ArgumentError("$(path) must be true or false; got $(repr(value))"))
    return value
end

# netcdf-c is not thread-safe, even across files. Daily snapshot files are
# written on a spawned task while observation and single-file snapshot appends
# happen on the main thread, so every runtime NetCDF write takes this lock.
const _NETCDF_IO_LOCK = ReentrantLock()

import ..expand_data_path
using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               nx, ny, nrings, ring_longitudes, cell_index, ncells,
               cell_area, panel_cell_center_lonlat, panel_cell_corner_lonlat,
               cs_definition, coordinate_law, center_law, longitude_offset_deg,
               cs_definition_tag, coordinate_law_tag, center_law_tag,
               lonlat_to_panel_xy, gravity
using ..State: DryBasis, MoistBasis, mass_basis, tracer_names, get_tracer,
               CellState, CubedSphereState, tracer_index

export AbstractSnapshotFrame, SnapshotFrame, SelectedSnapshotFrame, SnapshotWriteOptions
export AbstractOutputSchedule, AbstractOutputPartition
export AbstractLayerSelection, FullLayerSelection, SelectedLayerSelection
export NoLayerSelection, TracerOutputFields, OutputFieldSpec
export ExplicitSnapshotSchedule, IntervalSnapshotSchedule
export SingleOutputFile, DailyOutputFiles, RuntimeOutputSpec
export runtime_output_spec, snapshot_hours, output_enabled, output_path
export output_fields, output_field_spec, output_path_for_day
export tracer_fields, layer_selection, layer_selection_label, air_mass_layer_selection
export capture_snapshot, write_snapshot_netcdf, write_snapshot_binary
export column_mean_mixing_ratio, layer_mass_per_area, column_mass_per_area
export AbstractObservationSource, OCO2LiteSource, ObsPackSource, TableSource
export AbstractObservationMode, SoundingMode, SiteMode
export AbstractSiteGrouping, SiteCodeGrouping, LocationGrouping
export AbstractTableFormat, AutoTableFormat, CSVTableFormat, TOMLTableFormat, NetCDFTableFormat
export AbstractObservationTimeInterpolation, LinearWindowInterpolation, NearestWindowSampling
export AbstractObservationOutput, NoObservationOutput, ObservationOutputSpec
export observation_output_spec, observations_enabled, observation_output_path
export OBSERVATION_RUNTIME_UNAVAILABLE_MESSAGE
export SoundingRequest, SiteRequest, ObservationSet
export read_observation_requests, build_observation_set, expand_observation_paths
export CellLocation, AbstractCellLocator, LatLonCellLocator, ReducedGaussianCellLocator
export CubedSphereCellLocator, cell_locator, locate, ncolumns, isvalid_lonlat
export ObservationGatherBuffers, gather_columns!, gather_field!
export interface_pressures!, layer_heights_agl!, intake_layer_index
export column_mean_vmr, mixing_ratio_profile!
export AbstractLayerTemperature, ConstantLayerTemperature, SurfaceLapseTemperature
export ProfileLayerTemperature
export AbstractObservationSampler, NoObservationSampler, build_observation_sampler
export observe_window_boundary!, begin_observation_day!, finish_observations!

include("snapshots.jl")
include("runtime_output.jl")
include("diagnostics.jl")
include("snapshot_totals.jl")
include("selected_snapshots.jl")
include("netcdf_schema.jl")
include("netcdf_writer.jl")
include("netcdf_stream.jl")
include("binary_writer.jl")
include("observations/observation_sources.jl")
include("observations/observation_requests.jl")
include("observations/observation_readers.jl")
include("observations/observation_output_spec.jl")
include("observations/cell_locator.jl")
include("observations/observation_gather.jl")
include("observations/observation_netcdf.jl")
include("observations/observation_sampler.jl")

end # module Output
