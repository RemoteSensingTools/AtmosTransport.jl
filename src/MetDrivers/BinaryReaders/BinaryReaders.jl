"""Memory-mapped transport readers and LL/RG/CS runtime drivers."""
module BinaryReaders
using Adapt
using Printf: @sprintf
using ...Architectures: array_adapter_for, AbstractPointOp, Fused, select_panel,
                       pin_host_memory!, copy_arrays!, host_memory_model,
                       host_mapped_vector, DiscreteHostMemory, UnifiedHostMemory
import ...Architectures: index_space, apply_point!, bound_context, launch!
using KernelAbstractions: get_backend
using ...State
using ...Grids
using ...Parameters: SURFACE_PRESSURE_BASIS_TOLERANCE, EARTH_RADIUS, STANDARD_GRAVITY,
                    GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
                    DEFAULT_MET_INTERVAL_SECONDS, GMAO_CS_LONGITUDE_OFFSET_DEG,
                    REPLAY_TOLERANCE_FLOAT64, REPLAY_TOLERANCE_FLOAT32,
                    TM5_CONVECTION_MASS_BASIS_KEY, TM5_CONVECTION_SOURCE_KEY,
                    TM5_EC2TM_CONVENTION_KEY, TM5_EC2TM_CONVENTION

using Mmap, JSON3
using ...Architectures: CPU
using ..BinaryFormat
using ..BinaryFormat: CS_OPTIONAL_2D_SECTIONS, GCHP_VDIFF_PAYLOAD_SECTIONS, PBL_SURFACE_PAYLOAD_SECTIONS,
                      dry_mass_pin_enabled, has_variable_step_schedule, parse_transport_header,
                      print_capability_rows, read_transport_header_json, required_advection_sections,
                      steps_per_window_summary, transport_disk_float_type, transport_geometry_summary,
                      transport_is_faceindexed, transport_is_structured, transport_qv_summary,
                      transport_semantics_summary, transport_surface_pressure_basis_label
using ..MetDrivers: AbstractMassFluxMetDriver, AbstractMetDriver, ConvectionForcing,
                    FaceIndexedReplayLayout, PBLSurfaceForcing, ReplayScratch,
                    StructuredDirectionalReplayLayout, faceindexed_replay_layout, replay_tolerance,
                    structured_replay_layout, verify_window_continuity, verify_window_continuity_cs,
                    flux_storage_substep_scale, run_replay_gate
import ..MetDrivers: air_mass_basis, load_transport_window, same_window_layout, surface_pressure_basis,
                     flux_kind, steps_per_window, steps_per_window_schedule, total_windows, window_dt,
                     supports_moisture, supports_native_vertical_flux, supports_convection,
                     supports_diffusion, driver_grid, flux_interpolation_mode,
                     uses_binary_substep_contract
import ..BinaryFormat: binary_geometry, grid_type, horizontal_topology, has_flux_delta,
                       binary_capabilities, binary_dry_mass_target, inspect_binary
import ...State: mass_basis
include("reader.jl")
include("cubed_sphere_reader.jl")
include("driver.jl")
include("cubed_sphere_driver.jl")
include("inspect.jl")

export TransportBinaryDriver, TransportBinaryReader, air_mass_basis, air_mass_sampling,
       binary_capabilities, binary_dry_mass_target, binary_geometry, binary_mass_basis, copy_fluxes!,
       delta_semantics, driver_grid, expected_air_mass!, flux_kind, flux_sampling, grid_type, has_cmfmc,
       has_cmfmc_cloud_base, has_flux_delta, has_pbl_eflux, has_qv_endpoints, has_surface,
       has_tm5_convection, has_vdiff_fields, horizontal_topology, humidity_sampling, inspect_binary,
       interpolate_fluxes!, interpolate_qv!, load_flux_delta_window!, load_grid, load_qv_pair_window!,
       load_surface_window!, load_tm5_convection_window!, load_transport_window, load_transport_window!,
       load_window!, mesh_convention, mesh_definition, source_flux_sampling, steps_per_window,
       steps_per_window_schedule, surface_pressure_basis, tm5_convection_mass_basis, total_windows,
       uses_binary_substep_contract, window_count, window_dt
export AbstractRawWindowStaging, RawWindowStaging, MappedWindowStaging
export section_view, infer_surface_pressure_basis, load_vdiff_qv_window!, raw_window_staging,
       read_window_payload!, unpack_window_payload!, same_window_layout, release_payload!
end
