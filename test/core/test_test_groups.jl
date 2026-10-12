# test/groups.toml: every test entry of the default tiers (core/ files and
# regridding/runtests.jl) is in exactly one topic, every quick entry exists, a
# topic maps at least one file of every src/ folder, and every path glob still
# matches a file. Also the changed-file mapping that `--changed` uses.
using Test

include(joinpath(@__DIR__, "..", "helpers", "test_groups.jl"))
using .TestGroups

const TEST_ROOT = normpath(joinpath(@__DIR__, ".."))  # ends in a slash
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

@testset "test groups cover the suite" begin
    groups = load_groups(joinpath(TEST_ROOT, "groups.toml"))
    problems = validate_groups(groups, TEST_ROOT, REPO_ROOT)
    isempty(problems) || @info "test/groups.toml problems:\n  " * join(problems, "\n  ")
    @test isempty(problems)
    @test !isempty(groups.quick)
    @test "core/test_test_groups.jl" in groups.quick
    # The quick set keeps the package health gates.
    @test "core/test_aqua.jl" in groups.quick && "core/test_jet.jl" in groups.quick
end

@testset "globs" begin
    g = TestGroups.glob_regex
    @test occursin(g("src/Grids/**"), "src/Grids/LatLonMesh.jl")
    @test occursin(g("src/Grids/**"), "src/Grids/sub/deep.jl")
    @test !occursin(g("src/Grids/*.jl"), "src/Grids/sub/deep.jl")
    @test occursin(g("**/README.md"), "README.md")
    @test occursin(g("**/README.md"), "src/Operators/README.md")
    @test occursin(g("src/x/{era5,geos}*.jl"), "src/x/geos_cs_omega.jl")
    @test !occursin(g("src/x/{era5,geos}*.jl"), "src/x/merra2.jl")
    @test !occursin(g("src/a.jl"), "src/abjl")
end

@testset "changed files select topics, entries and users" begin
    groups = TestGroups.Groups(
        [TestGroups.Topic("alpha", "", ["core/test_a.jl"], ["src/Alpha/**"]),
         TestGroups.Topic("beta", "", ["core/test_b.jl", "regridding/runtests.jl"], ["src/Beta.jl"])],
        ["core/test_a.jl"])
    mktempdir() do root
        mkpath(joinpath(root, "core")); mkpath(joinpath(root, "regridding"))
        write(joinpath(root, "core", "test_a.jl"), "include(\"../helpers/shared.jl\")\n")
        write(joinpath(root, "core", "test_b.jl"), "using Test\n")
        write(joinpath(root, "core", "test_c.jl"), "using Test\n")
        write(joinpath(root, "regridding", "runtests.jl"), "using Test\n")

        sel = select_changed(groups, root, ["src/Alpha/x.jl", "test/core/test_c.jl"])
        @test sel.topics == Set(["alpha"]) && sel.files == Set(["core/test_c.jl"]) && !sel.full

        sel = select_changed(groups, root, ["test/helpers/shared.jl", "test/regridding/test_x.jl"])
        @test isempty(sel.topics)
        @test sel.files == Set(["core/test_a.jl", "regridding/runtests.jl"])

        sel = select_changed(groups, root, ["src/Unmapped.jl", "docs/notes.txt"])
        @test sel.full
        @test any(occursin("src/Unmapped.jl", n) for n in sel.notes)
        @test any(occursin("docs/notes.txt", n) for n in sel.notes)

        @test_throws ErrorException topic_files(groups, ["gamma"])
        @test topic_files(groups, ["alpha", "beta"]) ==
              Set(["core/test_a.jl", "core/test_b.jl", "regridding/runtests.jl"])
    end
end
