# Transport-binary format assembly: typed metadata, contracts, readers,
# writers, and inspection share the `MetDrivers` namespace.
#
# Include order is dependency-driven (types must be defined before the methods
# that dispatch on them at include time):
#   header  →  contract  →  reader  →  cubed-sphere reader specializations
#           →  payload sections  →  writer  →  streaming writer
#           →  cubed-sphere writer  →  inspect
# Function bodies may call across files freely; only type and constant
# definitions constrain include order.

using Mmap
using JSON3
using Printf: @printf
using Base.Threads
using ..Architectures: CPU
using ..AtomicFiles: atomic_publish, open_atomic_staging, commit_atomic_staging,
                     abandon_atomic_staging
import ..State: mass_basis
using ..State: AbstractSurfacePressureBasis, TotalSurfacePressure,
               DrySurfacePressure

include("transport_binary/header.jl")
include("transport_binary/contract.jl")
include("transport_binary/reader.jl")
include("transport_binary/cubed_sphere_reader.jl")
include("transport_binary/payload_sections.jl")
include("transport_binary/writer.jl")
include("transport_binary/streaming_writer.jl")
include("transport_binary/cubed_sphere.jl")
include("transport_binary/inspect.jl")

export AbstractTransportBinaryGeometry, LatLonBinaryGeometry
export ReducedGaussianBinaryGeometry, CubedSphereBinaryGeometry
export TransportBinaryHeader, TransportBinaryReader, binary_geometry
export grid_type, horizontal_topology, load_grid, load_qv_pair_window!, load_flux_delta_window!
export has_qv_endpoints, has_flux_delta, has_surface, write_transport_binary
export source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics
export surface_pressure_basis
