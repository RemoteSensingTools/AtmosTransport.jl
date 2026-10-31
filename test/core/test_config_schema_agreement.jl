# The editor schema (schemas/atmos_transport_run.schema.json) and the runtime
# parsers agree: the schema's choices are the parsers' canonical choices, and
# every key the parsers read (the known-key tables of `validate_config`) is a
# schema property, and the other way round. The schema offers canonical
# spellings only; where a parser also accepts aliases (temporal scheme), the
# test checks that every schema value parses and that the schema values reach
# every outcome.
using Test
using AtmosTransport
using JSON3

const RunnerKeys = AtmosTransport.Models.DrivenRunner
const Specs = AtmosTransport.Models
const SCHEMA = JSON3.read(read(joinpath(@__DIR__, "..", "..", "schemas",
                                        "atmos_transport_run.schema.json"), String))

enum_of(node) = Set(string.(node.enum))
# The concrete subtypes of `T` defined in module `mod` (InteractiveUtils is not
# a test dependency).
concrete_subtypes(T, mod) =
    Set(t for t in (getfield(mod, n) for n in names(mod; all = true) if isdefined(mod, n))
        if t isa DataType && t !== T && t <: T && !isabstracttype(t))
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
    @test enum_of(p.logging.properties.level) == Set(keys(AtmosTransport.LogLevels.LOG_LEVELS))
    @test all(k -> RunnerKeys.run_logging(Dict("logging" => Dict("progress" => k))) isa
                   RunnerKeys.RunLogging, enum_of(p.logging.properties.progress))
    @test enum_of(d.initial_condition.properties.kind) == Set(keys(RunnerKeys._INIT_KINDS))
    @test enum_of(d.initial_condition.properties.vertical_order) ==
          Set(string.(AtmosTransport.Models.InitialConditionIO._VERTICAL_ORDERS))
    @test enum_of(d.surface_flux.properties.kind) ==
          Set(string.(AtmosTransport.Models.InitialConditionIO._SURFACE_FLUX_KINDS))
    @test enum_of(d.surface_flux.properties.regridding) ==
          Set(string.(AtmosTransport.Models.InitialConditionIO._SURFACE_FLUX_REGRIDDINGS))
    @test enum_of(d.surface_flux.properties.species) ==
          Set(string.(propertynames(AtmosTransport.Parameters.SPECIES_MOLAR_MASS)))
    @test all(s -> AtmosTransport.Models.InitialConditionIO.surface_flux_species(
                       :tracer, Dict("species" => s)) === Symbol(s),
              enum_of(d.surface_flux.properties.species))

    # Choices with aliases: each schema value parses, and together they reach
    # every outcome.
    schemes = [AtmosTransport.Operators.SurfaceFlux.flux_temporal_scheme(s)
               for s in enum_of(d.surface_flux.properties.temporal_scheme)]
    @test Set(typeof.(schemes)) ==
          concrete_subtypes(AtmosTransport.Operators.SurfaceFlux.AbstractFluxTemporalScheme,
                            AtmosTransport.Operators.SurfaceFlux)
    architecture_requests = [AtmosTransport.Architectures._architecture_request(backend)
                             for backend in enum_of(p.architecture.properties.backend)]
    @test Set(typeof.(architecture_requests)) ==
          concrete_subtypes(AtmosTransport.Architectures.AbstractArchitectureRequest,
                            AtmosTransport.Architectures)
    for float_type in enum_of(p.numerics.properties.float_type)
        @test RunnerKeys._cfg_float_type(Dict("numerics" => Dict("float_type" => float_type))) <:
              AbstractFloat
    end
end

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
end

# Every key the editor offers carries a description (shown on hover), either
# its own or that of the definition it refers to; every table rejects keys it
# does not list, as `validate_config` does.
@testset "schema keys are described and tables are closed" begin
    described(node) = haskey(node, :description) ||
        (haskey(node, Symbol("\$ref")) &&
         haskey(SCHEMA.definitions, Symbol(last(split(node[Symbol("\$ref")], "/")))) &&
         haskey(SCHEMA.definitions[Symbol(last(split(node[Symbol("\$ref")], "/")))], :description))
    undescribed = String[]
    open_tables = String[]
    # Tables are the object schemas with keys; conditional fragments inside
    # oneOf/allOf/if (which also use `properties`) only constrain values.
    function walk(node, path)
        if node isa AbstractVector
            foreach(((i, child),) -> walk(child, "$(path)[$i]"), enumerate(node))
            return
        end
        node isa AbstractDict || return
        if get(node, :type, nothing) == "object" && haskey(node, :properties)
            haskey(node, :additionalProperties) && node.additionalProperties === false ||
                push!(open_tables, path)
            for (key, child) in pairs(node.properties)
                described(child) || push!(undescribed, "$(path).$(key)")
            end
        end
        foreach(((key, child),) -> walk(child, "$(path).$(key)"), pairs(node))
    end
    walk(SCHEMA, "schema")
    @test isempty(undescribed)
    @test isempty(open_tables)
end

