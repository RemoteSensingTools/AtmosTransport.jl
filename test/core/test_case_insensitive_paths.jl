# No two files or folders of the package may differ only by letter case: such a pair cannot be
# checked out on the default file systems of macOS and Windows, and the General registry
# refuses to merge a version that has one under `src/`.
using Test

"""
    case_collisions(root) -> Vector{Tuple{String, String}}

Return the pairs of entries below `root` that share a folder and differ only by letter case.
"""
function case_collisions(root::AbstractString)
    found = Tuple{String, String}[]
    for (dir, folders, files) in walkdir(root)
        seen = Dict{String, String}()
        for name in vcat(folders, files)
            key = lowercase(name)
            haskey(seen, key) && push!(found, (joinpath(dir, seen[key]), joinpath(dir, name)))
            seen[key] = name
        end
        filter!(f -> f != ".git", folders)          # do not descend into the repository data
    end
    return found
end

@testset "no paths that differ only by letter case" begin
    package_root = normpath(joinpath(@__DIR__, "..", ".."))
    for sub in ("src", "ext", "test", "scripts", "config", "docs/src", "examples", "benchmarking")
        root = joinpath(package_root, sub)
        isdir(root) || continue
        @test isempty(case_collisions(root))
    end
    # The function finds a collision when there is one.
    mktempdir() do dir
        mkpath(joinpath(dir, "a"))
        write(joinpath(dir, "a", "Reader.jl"), "")
        # On a case-insensitive file system the second file cannot exist; the check is then moot.
        if !isfile(joinpath(dir, "a", "reader.jl"))
            write(joinpath(dir, "a", "reader.jl"), "")
            @test length(case_collisions(dir)) == 1
        end
    end
end
