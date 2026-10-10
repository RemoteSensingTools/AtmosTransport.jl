# The editor schema (schemas/atmos_transport_run.schema.json) and the runtime
# parsers agree: the schema's choices are the parsers' canonical choices, and
# every key the parsers read (the known-key tables of `validate_config`) is a
# schema property, and the other way round. The schema offers canonical
# spellings only; where a parser also accepts aliases (output format and split,
# layer selection, temporal scheme, backend), the test checks that every schema
# value parses and that the schema values reach every outcome.
using Test
using AtmosTransport
using JSON3
using InteractiveUtils: subtypes

const RunnerKeys = AtmosTransport.Models.DrivenRunner
const Specs = AtmosTransport.Models
const SCHEMA = JSON3.read(read(joinpath(@__DIR__, "..", "..", "schemas",
                                        "atmos_transport_run.schema.json"), String))

enum_of(node) = Set(string.(node.enum))
property_keys(node) = Set(String.(keys(node.properties)))

@testset "schema choices equal the parsers' choices" begin
    p = SCHEMA.properties
    d = SCHEMA.definitions
    @test enum_of(p.advection.properties.scheme) == Set(string.(Specs._ADVECTION_SCHEMES))
    @test Set(p.advection.properties.ppm_order.enum) == Set(Specs._LINROOD_PPM_ORDERS)
    @test enum_of(p.advection.properties.vertical) ==
          union(Set(keys(Specs._VERTICAL_RECONSTRUCTIONS)), Set(Specs._LINROOD_VERTICALS))
    @test enum_of(p.advection.properties.limiter) == Set(keys(Specs._PPM_LIMITERS))
    @test enum_of(p.diffusion.properties.kind) == Set(string.(keys(Specs._DIFFUSION_KINDS)))
    @test enum_of(p.convection.properties.kind) == Set(string.(Specs._CONVECTION_KINDS))
    @test enum_of(p.convection.properties.cloud_base) == Set(string.(keys(Specs._CLOUD_BASE_RULES)))
    @test enum_of(p.chemistry.properties.kind) == Set(string.(Specs._CHEMISTRY_KINDS))
    @test enum_of(p.run.properties.air_mass_reset_mode) == Set(string.(Specs._AIR_MASS_RESET_MODES))
    @test enum_of(p.run.properties.physics_cadence) == Set(string.(Specs._PHYSICS_CADENCES))
    @test enum_of(d.initial_condition.properties.kind) == Set(keys(RunnerKeys._INIT_KINDS))
    @test enum_of(d.initial_condition.properties.vertical_order) ==
          Set(string.(AtmosTransport.Models.InitialConditionIO._CS_NATIVE_VERTICAL_ORDERS))
    # Named kinds are offered; any other string is a generic file source.
    @test enum_of(d.surface_flux.properties.kind.anyOf[1]) == Set(RunnerKeys._SURFACE_FLUX_KINDS)
    @test d.surface_flux.properties.kind.anyOf[2].type == "string"
    @test enum_of(d.surface_flux.properties.regridding) ==
          Set(string.(AtmosTransport.Models.InitialConditionIO._SURFACE_FLUX_REGRIDDINGS))

    # Choices with aliases: each schema value parses, and together they reach
    # every outcome.
    schemes = [AtmosTransport.Operators.SurfaceFlux.flux_temporal_scheme(s)
               for s in enum_of(d.surface_flux.properties.temporal_scheme)]
    @test Set(typeof.(schemes)) ==
          Set(subtypes(AtmosTransport.Operators.SurfaceFlux.AbstractFluxTemporalScheme))
    @test Set(AtmosTransport.Output._parse_output_format(f)
              for f in enum_of(d.output.properties.format)) == Set([:netcdf, :binary_mmap])
    @test Set(typeof(AtmosTransport.Output._parse_layer_selection(l, "x"))
              for l in enum_of(d.layer_selection)) ==
          Set(subtypes(AtmosTransport.Output.AbstractLayerSelection))
    @test Set(c.const for c in d.output.properties.split.oneOf) == Set(["single", "daily"])
    @test Set(typeof(AtmosTransport.Output._output_partition(Dict("split" => c.const)))
              for c in d.output.properties.split.oneOf) ==
          Set(subtypes(AtmosTransport.Output.AbstractOutputPartition))
    for backend in enum_of(p.architecture.properties.backend)
        @test AtmosTransport.Architectures._architecture_symbol(backend) === Symbol(backend)
    end
    for float_type in enum_of(p.numerics.properties.float_type)
        @test RunnerKeys._cfg_float_type(Dict("numerics" => Dict("float_type" => float_type))) <:
              AbstractFloat
    end
end

@testset "schema properties equal the known-key tables" begin
    p = SCHEMA.properties
    d = SCHEMA.definitions
    input_keys = union(property_keys(p.input), (property_keys(b) for b in p.input.oneOf)...)
    @test input_keys == Set(RunnerKeys._INPUT_KEYS)
    @test property_keys(d.staging) == Set(RunnerKeys._INPUT_STAGING_KEYS)
    @test property_keys(p.architecture) == Set(RunnerKeys._ARCHITECTURE_KEYS)
    @test property_keys(p.numerics) == Set(RunnerKeys._NUMERICS_KEYS)
    @test property_keys(p.run) == Set(RunnerKeys._RUN_KEYS)
    @test property_keys(p.advection) == Set(RunnerKeys._ADVECTION_KEYS)
    @test property_keys(p.diffusion) == Set(RunnerKeys._DIFFUSION_KEYS)
    @test property_keys(p.convection) == Set(RunnerKeys._CONVECTION_KEYS)
    @test property_keys(p.chemistry) == Set(RunnerKeys._CHEMISTRY_KEYS)
    @test property_keys(d.initial_condition) == Set(RunnerKeys._INIT_KEYS)
    @test property_keys(d.surface_flux) == Set(RunnerKeys._SURFACE_FLUX_KEYS)
    @test property_keys(d.output) == Set(RunnerKeys._OUTPUT_KEYS)
    @test property_keys(d.output_fields) == Set(RunnerKeys._OUTPUT_FIELDS_KEYS)
    @test property_keys(d.tracer_output_fields) == Set(RunnerKeys._TRACER_OUTPUT_FIELDS_KEYS)
    @test Set(String.(keys(p))) == Set(RunnerKeys._TOP_LEVEL_TABLES)
    # Flat [tracers.<name>] keys are accepted for old configs but not offered.
    @test property_keys(d.tracer) == Set(["init", "surface_flux"])
end
