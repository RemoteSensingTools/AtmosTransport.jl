using Test
using AtmosTransport

const DOCUMENTED_CORE_API = (
    :TransportModel,
    :CubedSphereState,
    :StructuredFaceFluxState,
    :FaceIndexedFluxState,
    :CubedSphereFaceFluxState,
    :CMFMCConvection,
    :TM5Convection,
    :TransportBinaryReader,
    :TransportBinaryHeader,
    :CubedSphereBinaryGeometry,
    :TransportBinaryDriver,
    :TransportWindow,
    :TransportTracerSpec,
    :write_transport_binary,
    :binary_mass_basis,
    :build_runtime_physics_recipe,
    :validate_runtime_physics_recipe,
    :run!,
    :run_window!,
)

const DOCUMENTED_MET_DRIVER_API = (
    :TransportBinaryContract,
    :StreamingTransportBinaryWriter,
    :canonical_window_constant_contract,
    :validate_transport_contract!,
    :load_window!,
    :load_qv_pair_window!,
    :load_flux_delta_window!,
    :load_transport_window,
    :binary_capabilities,
    :inspect_binary,
    :binary_mass_basis,
    :has_qv_endpoints,
    :air_mass_basis,
    :driver_grid,
)

function has_attached_doc(module_::Module, name::Symbol)
    isdefined(module_, name) || return false
    binding = Base.Docs.Binding(module_, name)
    return haskey(Base.Docs.meta(binding.mod), binding)
end

@testset "core public API has attached docstrings" begin
    for name in DOCUMENTED_CORE_API
        @test has_attached_doc(AtmosTransport, name)
    end
end

@testset "met-driver API has attached docstrings" begin
    for name in DOCUMENTED_MET_DRIVER_API
        @test has_attached_doc(AtmosTransport.MetDrivers, name)
    end
end


@testset "shared output geometry API has attached docstrings" begin
    for name in (:define_cs_horizontal_geometry!, :define_provenance_attributes!)
        @test has_attached_doc(AtmosTransport.Output, name)
    end
end

@testset "State basis conversion API has attached docstrings" begin
    @test has_attached_doc(AtmosTransport.State, :basis_symbol)
end
