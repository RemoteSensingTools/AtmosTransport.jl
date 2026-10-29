#!/usr/bin/env julia
# Active config matrix contract.
#
# The active config tree is part of the source/topology dispatch surface. Files
# under `likely_legacy/` or `completed_experiments/` may preserve historical
# schemas, but active configs must point at the canonical preprocessing/runtime
# entrypoints, carry enough TOML structure for the unified drivers, and cite
# only scripts that exist.

using Test
using TOML
using AtmosTransport

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function _active_tomls(rel::AbstractString; skip_completed::Bool = false)
    root = joinpath(REPO_ROOT, rel)
    paths = String[]
    for (dir, _, files) in walkdir(root)
        occursin("likely_legacy", dir) && continue
        skip_completed && occursin("completed_experiments", dir) && continue
        for file in files
            endswith(file, ".toml") || continue
            push!(paths, relpath(joinpath(dir, file), REPO_ROOT))
        end
    end
    return sort(paths)
end

const SCRIPT_REFERENCE = r"scripts/[A-Za-z0-9_./*?{},-]+"

function _citation_files()
    paths = String[]
    for rel_root in ("src", "config", "docs/src")
        root = joinpath(REPO_ROOT, rel_root)
        for (dir, dirs, files) in walkdir(root)
            if rel_root == "config"
                filter!(d -> d != "likely_legacy" && d != "completed_experiments", dirs)
            end
            append!(paths, (relpath(joinpath(dir, file), REPO_ROOT) for file in files))
        end
    end
    return sort(paths)
end

function _brace_expansions(pattern::AbstractString)
    m = match(r"\{([^{}]+)\}", pattern)
    m === nothing && return [String(pattern)]
    first = m.offset
    last = first + ncodeunits(m.match) - 1
    prefix = first == firstindex(pattern) ? "" : pattern[firstindex(pattern):prevind(pattern, first)]
    suffix = last == lastindex(pattern) ? "" : pattern[nextind(pattern, last):lastindex(pattern)]
    return reduce(vcat, (_brace_expansions(prefix * choice * suffix)
                         for choice in split(only(m.captures), ',')))
end

function _glob_regex(pattern::AbstractString)
    escaped = replace(pattern, "\\" => "\\\\", "." => "\\.", "+" => "\\+",
                      "(" => "\\(", ")" => "\\)", "^" => "\\^", "\$" => "\\\$")
    escaped = replace(escaped, "**" => "__DOUBLE_STAR__")
    escaped = replace(escaped, "*" => "[^/]*", "?" => "[^/]")
    escaped = replace(escaped, "__DOUBLE_STAR__" => ".*")
    return Regex("^" * escaped * "\$")
end

function _script_reference_exists(reference::AbstractString, candidates)
    !occursin(r"[*?{]", reference) && return ispath(joinpath(REPO_ROOT, reference))
    return any(pattern -> any(candidate -> occursin(_glob_regex(pattern), candidate), candidates),
               _brace_expansions(reference))
end

_has_canonical_input(input) =
    input isa AbstractDict &&
    (haskey(input, "binary_paths") ||
     (haskey(input, "folder") &&
      haskey(input, "start_date") &&
      haskey(input, "end_date")))

function _missing_keys(cfg, required)
    missing = String[]
    for (section, key) in required
        if !haskey(cfg, section) || !haskey(cfg[section], key)
            push!(missing, "$(section).$(key)")
        end
    end
    return missing
end

# Every shipped `[numerics] float_type` goes through the strict production parser.
@testset "shipped float_type values parse" begin
    bad = String[]
    n = 0
    for rel in vcat(_active_tomls("config"), _active_tomls("test/golden/configs"))
        cfg = TOML.parsefile(joinpath(REPO_ROOT, rel))
        numerics = get(cfg, "numerics", nothing)
        numerics isa AbstractDict && haskey(numerics, "float_type") || continue
        n += 1
        try
            AtmosTransport.ConfigChecks.parse_float_type(numerics["float_type"],
                                                         "[numerics] float_type")
        catch err
            err isa ArgumentError || rethrow()
            push!(bad, "$(rel): $(err.msg)")
        end
    end
    @test n > 250
    @test isempty(bad)
end

@testset "active preprocessing configs use unified source/topology schema" begin
    bad = String[]
    for rel in _active_tomls("config/preprocessing")
        cfg = TOML.parsefile(joinpath(REPO_ROOT, rel))
        source = get(cfg, "source", Dict{String, Any}())
        input = get(cfg, "input", Dict{String, Any}())
        grid = get(cfg, "grid", Dict{String, Any}())
        if haskey(source, "toml")
            missing = _missing_keys(cfg, (
                ("source", "root_dir"),
                ("output", "directory"),
                ("grid", "type"),
                ("numerics", "dt_met_seconds"),
            ))
            isempty(missing) || push!(bad, "$(rel): missing $(join(missing, ", "))")
            haskey(grid, "type") && lowercase(String(grid["type"])) == "cubed_sphere" ||
                push!(bad, "$(rel): native-source configs must declare grid.type = cubed_sphere")
        elseif haskey(input, "spectral_dir")
            missing = _missing_keys(cfg, (
                ("input", "coefficients"),
                ("output", "directory"),
                ("grid", "type"),
                ("grid", "level_top"),
                ("grid", "level_bot"),
                ("grid", "merge_min_thickness_Pa"),
                ("numerics", "dt"),
                ("numerics", "met_interval"),
            ))
            isempty(missing) || push!(bad, "$(rel): missing $(join(missing, ", "))")
            haskey(grid, "type") &&
                lowercase(String(grid["type"])) in ("latlon", "reduced_gaussian",
                                                    "synthetic_reduced_gaussian",
                                                    "era5_native_reduced_gaussian",
                                                    "cubed_sphere") ||
                push!(bad, "$(rel): unsupported or missing spectral grid.type")
        else
            push!(bad, "$(rel): must declare either [source].toml or [input].spectral_dir")
        end
    end
    @test isempty(bad)
end

@testset "active run configs use canonical runtime input schema" begin
    bad = String[]
    for rel in _active_tomls("config/runs"; skip_completed = true)
        cfg = TOML.parsefile(joinpath(REPO_ROOT, rel))
        _has_canonical_input(get(cfg, "input", nothing)) ||
            push!(bad, "$(rel): missing canonical [input] binary_paths or folder/start_date/end_date")
    end
    @test isempty(bad)
end

# Source, documentation, and active configuration citations are executable
# contracts. Historical configs are excluded by their explicit
# likely_legacy/completed_experiments classification. References may name a
# file, directory, or glob/brace pattern.
@testset "maintained files cite existing scripts" begin
    candidates = String[]
    for (dir, dirs, files) in walkdir(joinpath(REPO_ROOT, "scripts"))
        push!(candidates, relpath(dir, REPO_ROOT) * "/")
        append!(candidates, (relpath(joinpath(dir, file), REPO_ROOT) for file in files))
    end
    bad = String[]
    checked = 0
    for rel in _citation_files()
        for (line_number, line) in enumerate(eachline(joinpath(REPO_ROOT, rel)))
            for m in eachmatch(SCRIPT_REFERENCE, line)
                reference = rstrip(c -> c in ('.', ',', ';', ':'), m.match)
                checked += 1
                _script_reference_exists(reference, candidates) ||
                    push!(bad, "$(rel):$(line_number): missing $(reference)")
            end
        end
    end
    @test checked > 100
    @test isempty(bad)
end

# Every surface flux of an active run config (and of the golden-output cases)
# has a species: `surface_flux.species` or a tracer name that implies one. The
# replaced key `molar_mass_kg_mol` appears nowhere.
@testset "active run configs give every surface flux a species" begin
    species = AtmosTransport.Models.InitialConditionIO.surface_flux_species
    bad = String[]
    n = 0
    for rel in vcat(_active_tomls("config/runs"), _active_tomls("test/golden/configs"))
        cfg = TOML.parsefile(joinpath(REPO_ROOT, rel))
        tracers = get(cfg, "tracers", nothing)
        tracers isa AbstractDict || continue
        for (name, tracer) in tracers
            tracer isa AbstractDict || continue
            sf = get(tracer, "surface_flux", nothing)
            sf isa AbstractDict || continue
            lowercase(string(get(sf, "kind", "none"))) == "none" && continue
            n += 1
            try
                species(Symbol(name), sf)
            catch err
                err isa ArgumentError || rethrow()
                push!(bad, "$(rel) [tracers.$(name).surface_flux]: $(err.msg)")
            end
        end
    end
    @test n > 400
    @test isempty(bad)
end
