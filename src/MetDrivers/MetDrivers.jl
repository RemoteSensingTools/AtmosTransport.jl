"""
    MetDrivers

Meteorological data adapters for the basis-explicit transport architecture.

Provides:
- abstract driver types with capability traits;
- topology-generic transport-binary readers; and
- ERA5 native reduced-Gaussian GRIB geometry helpers.
"""
module MetDrivers

using Adapt
using Printf: @sprintf
using ..Architectures: array_adapter_for
using ..State
using ..Grids
using ..Parameters: EARTH_RADIUS
import ..source_revision

include("AbstractMetDriver.jl")
include("ConvectionForcing.jl")
include("SurfaceForcing.jl")
include("TransportBinary.jl")
include("ReplayContinuity.jl")
include("transport_binary/window.jl")
include("transport_binary/driver.jl")
include("transport_binary/cubed_sphere_driver.jl")
include("ERA5/ERA5.jl")
using .ERA5

export TRANSPORT_BINARY_FORMAT_VERSION
export TransportBinaryReader, TransportBinaryHeader, write_transport_binary
export AbstractTransportBinaryGeometry, LatLonBinaryGeometry
export ReducedGaussianBinaryGeometry, CubedSphereBinaryGeometry, binary_geometry
export TransportBinaryContract, canonical_window_constant_contract,
       validate_transport_contract!, validate_cs_writer_contract!
export StreamingTransportBinaryWriter
export open_streaming_transport_binary, write_streaming_window!,
       close_streaming_transport_binary!, set_streaming_steps_per_window_schedule!,
       set_transport_header_steps_per_window_schedule!
export TransportBinaryDriver, TransportWindow
export StructuredFluxDeltas, FaceIndexedFluxDeltas, CubedSphereFluxDeltas
export load_window!, load_flux_delta_window!
export load_tm5_convection_window!, has_tm5_convection
export load_qv_pair_window!, load_grid, load_transport_window, load_transport_window!
export driver_grid, air_mass_basis, has_humidity_endpoints
export interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!
export load_surface_window!
export ConvectionForcing, has_convection_forcing
export copy_convection_forcing!, allocate_convection_forcing_like
export PBLSurfaceForcing, has_pbl_surface_forcing
export window_count, has_qv_endpoints, has_flux_delta, has_cmfmc
export total_windows, window_dt, steps_per_window, steps_per_window_schedule
export binary_capabilities, inspect_binary
export has_surface, has_vdiff_fields, has_pbl_eflux, has_cmfmc_cloud_base
export mass_basis, grid_type, horizontal_topology, A_ifc, B_ifc
export uses_binary_substep_contract
export source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics
export flux_application_seconds, flux_storage_substep_scale
export diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!
export diagnose_cm_from_continuity_ka!
export recompute_cm_from_dm_target!, recompute_faceindexed_cm_from_dm_target!
export verify_window_continuity, verify_window_continuity_ll, verify_window_continuity_rg,
       verify_window_continuity_cs
export ERA5ReducedGaussianGeometry
export read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh

# Cubed-sphere geometry helpers
export mesh_convention, mesh_definition

end # module MetDrivers
