# The editor schemas must list exactly the choices the parser accepts, and each
# choice's description must name the Julia type it becomes. A new source kind,
# mode, grouping, table format, or time interpolation fails here until the
# schema documents it.
using Test, AtmosTransport, JSON3, TOML, Dates
const O = AtmosTransport.Output
const ROOT = joinpath(@__DIR__, "..", "..")
const RUN = JSON3.read(read(joinpath(ROOT, "schemas", "atmos_transport_run.schema.json"), String))
const SITES = JSON3.read(read(joinpath(ROOT, "schemas", "observation_sites.schema.json"), String))

consts(entry) = Set(String(c.const) for c in entry.oneOf)
function type_named(entry, value, typename)
    for c in entry.oneOf
        String(c.const) == value && return occursin(String(typename), String(c.description))
    end
    return false
end

@testset "observation schema matches the parser's choice tables" begin
    out = RUN.definitions.observation_output
    src = RUN.definitions.observation_source
    @test Set(String.(keys(out.properties))) == Set(O._OBSERVATION_OUTPUT_KEYS)
    @test out.additionalProperties == false && src.additionalProperties == false

    choice_tables = (
        (src.properties.kind, O._SOURCE_TYPES, T -> nameof(T)),
        (src.properties.mode, O._OBSERVATION_MODES, v -> nameof(typeof(v))),
        (src.properties.site_grouping, O._SITE_GROUPINGS, v -> nameof(typeof(v))),
        (src.properties.format, O._TABLE_FORMATS, v -> nameof(typeof(v))),
        (out.properties.time_interpolation, O._TIME_INTERPOLATIONS, v -> nameof(typeof(v))),
        (src.properties.quality_filter, O._QUALITY_FILTERS, T -> nameof(T)),
    )
    for (entry, table, typename) in choice_tables
        @test consts(entry) == Set(String.(keys(table)))
        for (key, value) in pairs(table)
            @test type_named(entry, String(key), typename(value))
        end
    end

    # Every source key appears in the schema, and each kind's rule forbids the
    # other kinds' specific keys.
    common = Set(["kind", "path", "mode"])
    specific = Dict(String(k) => setdiff(Set(O.source_keys(T)), common) for (k, T) in pairs(O._SOURCE_TYPES))
    all_specific = union(values(specific)...)
    @test Set(String.(keys(src.properties))) == union(common, all_specific)
    is_kind_rule(r) = haskey(r["if"], :properties) && keys(r["if"].properties) == Set([:kind])
    rules = Dict(String(r["if"].properties.kind.const) => r.then for r in src.allOf if is_kind_rule(r))
    @test Set(keys(rules)) == Set(String.(keys(O._SOURCE_TYPES)))
    for (kind, then) in rules
        forbidden = Set(String(only(a.required)) for a in then.not.anyOf)
        @test forbidden == setdiff(all_specific, specific[kind])
    end
    # Quality keys: the schema rule for each filter forbids exactly the keys the
    # parser rejects for it.
    quality_keys = Set(O._QUALITY_FILTER_KEYS)
    for (name, T) in pairs(O._QUALITY_FILTERS)
        cfg = Dict{String, Any}("kind" => "oco2_lite", "path" => "x.nc", "quality_filter" => String(name))
        T === O.QualityFlagValues && (cfg["quality_flag_values"] = [1])
        @test O.observation_source_from_cfg(cfg, "s").quality_filter isa T
        for key in setdiff(quality_keys, Set(O.quality_filter_keys(T)))
            bad = merge(cfg, Dict{String, Any}(key => key == "quality_flag_values" ? [1] :
                                                       key == "quality_flag_max" ? 0 : "flag"))
            @test_throws ArgumentError O.observation_source_from_cfg(bad, "s")
        end
    end

    split = RUN.definitions.output.properties.split
    @test consts(split) == Set(["single", "daily"])
    @test type_named(split, "single", :SingleOutputFile) && type_named(split, "daily", :DailyOutputFiles)
end

@testset "site-table schema documents every column and schedule" begin
    site = SITES.definitions.site
    event = SITES.definitions.point_event
    # Both row types reject unknown keys and list every alias the reader accepts.
    @test site.additionalProperties == false && event.additionalProperties == false
    schedule_keys = (O._TABLE_START_KEYS..., O._TABLE_END_KEYS..., O._TABLE_TIMES_KEYS...)
    @test Set(String.(keys(site.properties))) ==
          Set(k for k in O._TABLE_KNOWN_KEYS if !(k in O._TABLE_TIME_KEYS))
    @test Set(String.(keys(event.properties))) ==
          Set(k for k in O._TABLE_KNOWN_KEYS if !(k in schedule_keys))
    for typename in (:EveryWindow, :TimeRange, :TimeList)
        @test occursin(String(typename), String(site.description))
    end
    # The shipped example parses into the schedules its comments promise.
    demo = joinpath(ROOT, "config", "examples", "observation_sites_demo.toml")
    _, sites = O.read_observation_requests(O.TableSource(demo, O.SiteMode(), O.AutoTableFormat()), 1,
                                           DateTime(2021, 12, 2), [Date(2021, 12, 2)])
    by_id = Dict(s.id => s for s in sites)
    @test by_id["mlo"].schedule isa O.EveryWindow
    @test by_id["lef_396m"].schedule isa O.TimeRange
    @test by_id["lef_396m"].intake_height_m == 396.0
    @test by_id["brw_flask"].schedule isa O.TimeList
end
