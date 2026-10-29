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
    e = TestGroups.expand_braces
    @test e("src/x/{a,b}.jl") == ["src/x/a.jl", "src/x/b.jl"]
    @test e("{a,b}/{c,d}") == ["a/c", "a/d", "b/c", "b/d"]
    @test e("src/plain.jl") == ["src/plain.jl"]
end

@testset "the cell locator and the sampler select the inversion topic" begin
    # Placed 4D-Var observations share the forward sampler's locator and layer
    # rule, so changing them must run the parity tests.
    groups = load_groups(joinpath(TEST_ROOT, "groups.toml"))
    for f in ("src/Observations/Sampling/cell_locator.jl",
              "src/Observations/Sampling/observation_sampler.jl",
              "src/Observations/Sampling/observation_placement.jl")
        @test "inversion" in TestGroups.topics_for_path(groups, f)
    end
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

@testset "runtime composition and public entry points remain reachable" begin
    groups = load_groups(joinpath(TEST_ROOT, "groups.toml"))
    for path in ("src/Operators/StepPlan/StepPlan.jl",
                 "src/Models/TransportModel.jl", "src/Models/driven_stepping.jl")
        selection = select_changed(groups, TEST_ROOT, [path])
        @test "surface_flux_chemistry" in selection.topics
        @test "runner" in selection.topics
    end
    for path in ("src/Models/DrivenRunner.jl", "src/Models/InputStaging.jl",
                 "src/Models/RuntimePhysicsRecipe.jl", "src/MetDrivers/transport_binary/reader.jl",
                 "src/Preprocessing/transport_binary/entrypoint.jl",
                 "scripts/run_transport.jl",
                 "scripts/preprocessing/preprocess_transport_binary.jl",
                 "examples/generate_synthetic_quickstart.jl")
        selection = select_changed(groups, TEST_ROOT, [path])
        files = topic_files(groups, selection.topics)
        @test "core/test_public_config_and_cli.jl" in files
        @test !selection.full
    end
    @test "core/test_structure.jl" in groups.quick
    @test "health" in TestGroups.topics_for_path(groups, "src/LAYERS.toml")
    @test "health" in TestGroups.topics_for_path(groups, "test/structure/clone_allowlist.toml")
    for path in ("src/Downloads/Downloads.jl", "src/Extras/Downloads/Downloads.jl",
                 "src/Extras/Analysis/skill.jl", "src/Extras/Visualization/Visualization.jl",
                 "scripts/downloads/download_data.jl")
        selection = select_changed(groups, TEST_ROOT, [path])
        @test selection.topics == Set(["extras"])
        @test isempty(topic_files(groups, selection.topics)) && !selection.full
    end
    for path in ("src/SectionTimer.jl", "src/Diagnostics/SectionTimer.jl")
        @test "runner" in TestGroups.topics_for_path(groups, path)
    end
    for path in ("src/Diagnostics/Diagnostics.jl", "src/Diagnostics/budget_closure.jl")
        selection = select_changed(groups, TEST_ROOT, [path])
        @test "diagnostics" in selection.topics && !selection.full
        @test "core/test_analysis_budget_closure.jl" in topic_files(groups, selection.topics)
    end
end
