"""Version-4 transport metadata, payload layout, writers, windows, and inspection reports."""
module BinaryFormat
using Adapt
using Printf: @sprintf
using ...Grids
using ...State
using ...Parameters
import ...source_revision
import ..MetDrivers: surface_pressure_basis, flux_kind, steps_per_window, steps_per_window_schedule

using ...Grids: pack_panels!

using ...Grids: parse_panel_convention, panel_convention_tag, parse_coordinate_law, parse_center_law,
                default_cs_definition, cs_geometry_tags

using ...State: basis_symbol
using JSON3
using Printf: @printf
using Base.Threads
using ...Architectures: CPU
using ...AtomicFiles: atomic_publish, open_atomic_staging, commit_atomic_staging,
                     abandon_atomic_staging
import ...State: mass_basis
using ...State: AbstractSurfacePressureBasis, TotalSurfacePressure,
               DrySurfacePressure


include("header.jl")
include("contract.jl")
include("payload_sections.jl")
include("window.jl")
include("writer.jl")
include("streaming_writer.jl")
include("cubed_sphere.jl")
include("inspect.jl")

export fidelity_tag, FLUX_FIDELITY_SAMPLE_SEED
export poisson_balance_target, write_padded_header!
export dry_mass_pin_enabled, is_certificate_key, FLUX_CERTIFICATE_KEY_PREFIXES, COLUMN_CLOSURE_RULE,
       FLUX_FIDELITY_SCHEMA
export TM5_CONVECTION_PAYLOAD_SECTIONS, FLUX_DELTA_PAYLOAD_SECTIONS
export section_shape, section_elements, open_streaming_cs_transport_binary, write_streaming_cs_window!,
       abandon_streaming_transport_binary!
export AbstractColumnClosureHeader, AbstractFidelityStatus, AbstractFluxFidelityKind,
       AbstractTransportBinaryGeometry, ColumnClosureHeader, CubedSphereBinaryGeometry,
       CubedSphereFluxDeltas, FaceIndexedFluxDeltas, FidelityFailed, FidelityNotApplicable,
       FidelityPassed, FidelityReason, FidelityUnavailable, FluxFidelityHeader, LatLonBinaryGeometry,
       NativePassthroughFidelity, NoColumnClosureHeader, NoFluxFidelityKind, OmegaFidelityHeader,
       ReducedGaussianBinaryGeometry, StreamingTransportBinaryWriter, StructuredFluxDeltas,
       TRANSPORT_BINARY_FORMAT_VERSION, TransportBinaryContract, TransportBinaryHeader, TransportWindow,
       WindRederivedFidelity, binary_capabilities, binary_dry_mass_target, binary_geometry,
       canonical_window_constant_contract, close_streaming_transport_binary!, grid_type,
       has_flux_delta, has_humidity_endpoints, horizontal_topology,
       inspect_binary, mass_basis, open_streaming_transport_binary,
       set_streaming_steps_per_window_schedule!, set_transport_header_steps_per_window_schedule!,
       surface_pressure_basis, validate_cs_writer_contract!, validate_transport_contract!,
       write_streaming_window!, write_transport_binary
export CS_OPTIONAL_2D_SECTIONS, GCHP_VDIFF_PAYLOAD_SECTIONS, PBL_SURFACE_PAYLOAD_SECTIONS,
       has_variable_step_schedule, parse_transport_header, print_capability_rows,
       read_transport_header_json, required_advection_sections, steps_per_window_summary,
       transport_disk_float_type, transport_geometry_summary, transport_is_faceindexed,
       transport_is_structured, transport_qv_summary, transport_semantics_summary,
       transport_surface_pressure_basis_label
export parse_flux_fidelity_header, parse_omega_fidelity_header, parse_column_closure_header,
       parse_fidelity_status, parse_flux_fidelity_kind
end
