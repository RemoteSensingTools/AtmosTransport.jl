# Repo health: editor keys match runtime tables and reject unknown keys.
using Test
using AtmosTransport
using JSON3

const RunnerKeys = AtmosTransport.Models.DrivenRunner
const SCHEMA = JSON3.read(read(joinpath(@__DIR__, "..", "..", "schemas",
                                        "atmos_transport_run.schema.json"), String))

property_keys(node) = Set(String.(keys(node.properties)))

@testset "schema properties equal the known-key tables" begin
    p = SCHEMA.properties
    d = SCHEMA.definitions
    # The [input] branches (list or folder) only say which keys are required.
    @test property_keys(p.input) == Set(RunnerKeys._INPUT_KEYS)
    @test all(b -> !haskey(b, :properties), p.input.oneOf)
    @test property_keys(d.staging) == Set(RunnerKeys._INPUT_STAGING_KEYS)
    @test property_keys(p.architecture) == Set(RunnerKeys._ARCHITECTURE_KEYS)
    @test property_keys(p.numerics) == Set(RunnerKeys._NUMERICS_KEYS)
    # [run] also lists the legacy location of the advection keys.
    @test property_keys(p.run) == Set((RunnerKeys._RUN_KEYS..., RunnerKeys._ADVECTION_KEYS...))
    @test property_keys(p.advection) ==
          Set((RunnerKeys._ADVECTION_KEYS..., RunnerKeys._ADVECTION_ONLY_KEYS...))
    @test property_keys(p.diffusion) == Set(RunnerKeys._DIFFUSION_KEYS)
    @test property_keys(p.convection) == Set(RunnerKeys._CONVECTION_KEYS)
    @test property_keys(p.chemistry) == Set(RunnerKeys._CHEMISTRY_KEYS)
    @test property_keys(p.logging) == Set(RunnerKeys._LOGGING_KEYS)
    @test property_keys(p.checks) == Set(RunnerKeys._CHECKS_KEYS)
    @test property_keys(d.initial_condition) == Set(RunnerKeys._INIT_KEYS)
    @test property_keys(d.surface_flux) == Set(RunnerKeys._SURFACE_FLUX_KEYS)
    @test property_keys(d.output) == Set(RunnerKeys.OUTPUT_KEYS)
    @test property_keys(d.output_fields) == Set(RunnerKeys.OUTPUT_FIELDS_KEYS)
    @test property_keys(d.tracer_output_fields) == Set(RunnerKeys.TRACER_OUTPUT_FIELDS_KEYS)
    @test Set(String.(keys(p))) == Set(RunnerKeys._TOP_LEVEL_TABLES)
    # Flat [tracers.<name>] keys are accepted for old configs but not offered.
    @test property_keys(d.tracer) == Set(["init", "surface_flux"])
    # Every editor table rejects keys absent from its declared properties.
    function check_closed(node)
        if node isa AbstractVector
            foreach(check_closed,node)
        elseif node isa AbstractDict
            if get(node,:type,nothing) == "object" && haskey(node,:properties)
                @test get(node,:additionalProperties,nothing) === false
            end
            foreach(check_closed,values(node))
        end
    end
    check_closed(SCHEMA)
end
