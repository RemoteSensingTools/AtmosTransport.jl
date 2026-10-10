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

const STALE_ACTIVE_SCRIPT_REFS = (
    "scripts/run.jl",
    "scripts/run_transport_binary.jl",
    "scripts/run_cs_driven.jl",
    "scripts/deprecated/",              # retired runner shims
    "scripts/completed_experiments/",   # dissolved; scripts retired or under heritage/
    "scripts/preprocess_spectral_massflux.jl",
    "scripts/preprocessing/preprocess_spectral_massflux.jl",
    "scripts/preprocessing/preprocess_geos_transport_binary.jl",
    "scripts/preprocessing/preprocess_era5_reduced_gaussian_transport_binary_v2.jl",
    "scripts/preprocessing/preprocess_spectral_v4_binary.jl",
)

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
        text = read(joinpath(REPO_ROOT, rel), String)
        stale = [ref for ref in STALE_ACTIVE_SCRIPT_REFS if occursin(ref, text)]
        isempty(stale) || push!(bad, "$(rel): stale script refs $(join(stale, ", "))")
    end
    @test isempty(bad)
end

# Script paths cited in active configs (recipes, provenance, comparison
# commands) must resolve, so moving or retiring a script also updates the
# configs that name it.
@testset "active configs cite existing scripts" begin
    script_ref = r"scripts/[A-Za-z0-9_./-]+\.(?:jl|py|sh)"
    bad = String[]
    for rel in vcat(_active_tomls("config/preprocessing"),
                    _active_tomls("config/runs"; skip_completed = true))
        for line in eachline(joinpath(REPO_ROOT, rel))
            for m in eachmatch(script_ref, line)
                isfile(joinpath(REPO_ROOT, m.match)) ||
                    push!(bad, "$(rel): missing $(m.match)")
            end
        end
    end
    @test isempty(bad)
end
