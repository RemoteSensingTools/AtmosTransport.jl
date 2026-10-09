#!/usr/bin/env julia
"""
Module README freshness check (plan 21 Phase 6D).

For each tracked module directory, verifies that its README.md
mentions every `.jl` file in that directory. Catches the most common
README decay mode: adding a `.jl` file without updating the File Map
section.

Hard gate: if a README exists and misses a file, the test fails.
Module directories without a README are skipped (not every directory
needs one).

The check is intentionally narrow: it asks "is this file mentioned
anywhere in the README" (as a substring), not "is it in a specific
section." Structural conventions vary across submodules; the
invariant that matters is mere presence.

## Adding new directories

Append to `README_DIRS` when adding a new module-level directory
that should have a File Map.

## Bypass

Files can be excluded by adding them to `EXCLUDE_FILES`. Use
sparingly — the default expectation is that every `.jl` file
is either documented in README.md or explicitly excluded with
a comment explaining why.
"""

using Test

const README_DIRS = [
    # Root operators
    "src/Operators",
    # Operator submodules
    "src/Operators/Advection",
    "src/Operators/Diffusion",
    "src/Operators/SurfaceFlux",
    "src/Operators/Convection",
    "src/Operators/Chemistry",
    # State and its sub-tree
    "src/State",
    "src/State/Fields",
    # Models and drivers
    "src/Models",
    "src/MetDrivers",
    "src/MetDrivers/ERA5",
    # Grids
    "src/Grids",
    # Remaining modules and subfolders with READMEs
    "src/Adjoints", "src/Diagnostics", "src/Downloads", "src/Footprint",
    "src/Inversion", "src/MetDrivers/transport_binary",
    "src/Models/initial_conditions", "src/Models/runner", "src/Output",
    "src/Output/observations", "src/Parameters", "src/Preprocessing",
    "src/Preprocessing/sources", "src/Preprocessing/transport_binary",
    "src/Quantities", "src/Regridding", "src/Tape", "src/Visualization",
]

# Per-directory exclusions — keep this empty by default. Populate only
# when a specific .jl file has a documented reason not to appear in
# the README (e.g. compatibility shim, auto-generated).
const EXCLUDE_FILES = Dict{String, Vector{String}}(
    # "src/Operators" => ["some_shim.jl"],
)

function _check_readme(dir::AbstractString)
    root = joinpath(@__DIR__, "..", "..")
    full_dir = joinpath(root, dir)
    readme_path = joinpath(full_dir, "README.md")

    if !isfile(readme_path)
        @warn "README missing for $dir — skipping freshness check"
        return
    end

    readme_text = read(readme_path, String)
    all_jl = filter(f -> endswith(f, ".jl"), readdir(full_dir))
    excluded = get(EXCLUDE_FILES, dir, String[])
    files_to_check = filter(f -> !(f in excluded), all_jl)

    for file in files_to_check
        if !occursin(file, readme_text)
            @test begin
                @error """
                $(dir)/README.md does not mention $(file).
                Update the File Map section of $(readme_path),
                or add $(file) to EXCLUDE_FILES in test/test_readme_current.jl
                with a comment explaining why.
                """
                false
            end
        else
            @test true
        end
    end
end

@testset "Module README freshness" begin
    for dir in README_DIRS
        @testset "$dir" begin
            _check_readme(dir)
        end
    end
end
