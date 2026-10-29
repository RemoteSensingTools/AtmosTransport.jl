# The suite runner (`test/runtests.jl`) runs every file even when some fail,
# fails at the end, and stops at once on an interrupt.
using Test

function run_fake_suite(files::Dict{String, String})
    dir = mktempdir()
    cp(joinpath(@__DIR__, "..", "runtests.jl"), joinpath(dir, "runtests.jl"))
    mkpath(joinpath(dir, "core"))
    for name in ("test_aqua.jl", "test_jet.jl")
        write(joinpath(dir, "core", name), "using Test\n@test true\n")
    end
    for (name, body) in files
        write(joinpath(dir, "core", name), body)
    end
    cmd = `$(Base.julia_cmd()) --startup-file=no $(joinpath(dir, "runtests.jl")) --tiers=core`
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    return proc.exitcode, String(take!(out))
end

@testset "test runner keeps going and fails at the end" begin
    code, log = run_fake_suite(Dict(
        "test_a.jl" => "using Test\n@testset \"a\" begin @test 1 == 2 end\n",
        "test_b.jl" => "error(\"boom\")\n",
        "test_c.jl" => "using Test\nprintln(\"MARKER test_c ran\")\n@test true\n"))
    @test code == 1
    @test occursin("MARKER test_c ran", log)
    @test occursin("Test file failed: core/test_a.jl", log)
    @test occursin("Test file failed: core/test_b.jl", log)
    @test occursin("2 of 5 test files failed", log)
    @test occursin("Slowest 5 of 5 test files", log)

    code, log = run_fake_suite(Dict(
        "test_a.jl" => "using Test\n@test true\n",
        "test_b.jl" => "throw(InterruptException())\n",
        "test_c.jl" => "using Test\nprintln(\"MARKER test_c ran\")\n@test true\n"))
    @test code != 0
    @test !occursin("MARKER test_c ran", log)
    @test !occursin("Test file failed", log)
end

# A throwaway checkout with a read-only Git stub, the runner, the groups helper,
# a groups file and trivial test files. Selector tests supply changed paths;
# they never stage files, create commits or change refs.
function fake_grouped_repo()
    repo = mktempdir()
    test = joinpath(repo, "test")
    mkpath(joinpath(test, "core")); mkpath(joinpath(test, "regridding"))
    mkpath(joinpath(test, "helpers")); mkpath(joinpath(repo, "src", "Alpha"))
    cp(joinpath(@__DIR__, "..", "runtests.jl"), joinpath(test, "runtests.jl"))
    cp(joinpath(@__DIR__, "..", "helpers", "test_groups.jl"), joinpath(test, "helpers", "test_groups.jl"))
    write(joinpath(test, "groups.toml"), """
        quick = ["core/test_a.jl", "core/test_jet.jl", "core/test_aqua.jl"]
        [topics.health]
        files = ["core/test_aqua.jl", "core/test_jet.jl"]
        paths = ["test/runtests.jl"]
        [topics.alpha]
        files = ["core/test_a.jl", "core/test_c.jl"]
        paths = ["src/Alpha/**"]
        [topics.beta]
        files = ["core/test_b.jl", "regridding/runtests.jl"]
        paths = ["src/Beta.jl"]
        """)
    for name in ("test_aqua.jl", "test_jet.jl", "test_a.jl", "test_b.jl", "test_c.jl")
        write(joinpath(test, "core", name), "using Test\nprintln(\"MARKER $name ran\")\n@test true\n")
    end
    write(joinpath(test, "regridding", "runtests.jl"), "using Test\n@test true\n")
    write(joinpath(repo, "src", "Alpha", "x.jl"), "x = 1\n")
    mkpath(joinpath(repo, "bin"))
    # TestGroups must still call its normal subprocess reader. Only the three
    # read-only commands used by changed_paths are supported in this fixture.
    stub = joinpath(repo, "bin", "git")
    write(stub, raw"""#!/bin/sh
        [ "$1" = "-C" ] || exit 2
        case "$3:$4:$5:$6" in
            diff:--name-only:--no-renames:HEAD...HEAD) ;;
            diff:--name-only:--no-renames:HEAD) cat "$2/fixture-changed-paths" ;;
            ls-files:--others:--exclude-standard:) ;;
            *) exit 2 ;;
        esac
        """)
    chmod(stub, 0o755)
    write(joinpath(repo, "fixture-changed-paths"), "")
    return repo
end

function run_runner(repo, args...)
    cmd = `$(Base.julia_cmd()) --startup-file=no $(joinpath(repo, "test", "runtests.jl")) $args`
    cmd = addenv(cmd, "PATH" => joinpath(repo, "bin") * ":" * ENV["PATH"])
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    return proc.exitcode, String(take!(out))
end

listed(log) = filter(l -> startswith(l, "core/") || startswith(l, "regridding/"), split(log, '\n'))

@testset "test runner selectors" begin
    repo = fake_grouped_repo()
    code, log = run_runner(repo, "--quick", "--list")
    @test code == 0
    # Health gates first, then the rest in tier order.
    @test listed(log) == ["core/test_aqua.jl", "core/test_jet.jl", "core/test_a.jl"]

    code, log = run_runner(repo, "--topics=beta", "--list")
    @test listed(log) == ["core/test_b.jl", "regridding/runtests.jl"]

    code, log = run_runner(repo, "--list")
    @test length(listed(log)) == 6            # no flags: all of core and regridding

    code, log = run_runner(repo, "--topics=gamma", "--list")
    @test code != 0 && occursin("Unknown test topic 'gamma'", log)
    code, log = run_runner(repo, "--quik")
    @test code != 0 && occursin("Unknown argument '--quik'", log)

    # Changes: a modified src file maps through its topic; a modified test
    # file selects itself.
    write(joinpath(repo, "src", "Alpha", "x.jl"), "x = 2\n")
    write(joinpath(repo, "fixture-changed-paths"), "src/Alpha/x.jl\n")
    code, log = run_runner(repo, "--changed=HEAD", "--list")
    @test listed(log) == ["core/test_a.jl", "core/test_c.jl"]
    write(joinpath(repo, "test", "core", "test_b.jl"), "using Test\n@test true\n")
    write(joinpath(repo, "fixture-changed-paths"), "src/Alpha/x.jl\ntest/core/test_b.jl\n")
    code, log = run_runner(repo, "--changed=HEAD", "--quick", "--list")
    @test listed(log) == ["core/test_aqua.jl", "core/test_jet.jl", "core/test_a.jl",
                          "core/test_b.jl", "core/test_c.jl"]

    # An unmapped src file runs the full default tiers.
    write(joinpath(repo, "src", "Unmapped.jl"), "y = 1\n")
    write(joinpath(repo, "fixture-changed-paths"), "src/Alpha/x.jl\nsrc/Unmapped.jl\n")
    code, log = run_runner(repo, "--changed=HEAD", "--list")
    @test length(listed(log)) == 6
    @test occursin("src/Unmapped.jl maps to no test topic", log)

    # A moved file selects the topics of its old and its new path.
    renamed = fake_grouped_repo()
    mv(joinpath(renamed, "src", "Alpha", "x.jl"), joinpath(renamed, "src", "Beta.jl"))
    write(joinpath(renamed, "fixture-changed-paths"), "src/Alpha/x.jl\nsrc/Beta.jl\n")
    code, log = run_runner(renamed, "--changed=HEAD", "--list")
    @test listed(log) == ["core/test_a.jl", "core/test_b.jl", "core/test_c.jl",
                          "regridding/runtests.jl"]

    # Selected files run; the others do not.
    code, log = run_runner(repo, "--topics=alpha")
    @test code == 0
    @test occursin("MARKER test_a.jl ran", log) && occursin("MARKER test_c.jl ran", log)
    @test !occursin("MARKER test_aqua.jl ran", log)
    @test occursin("Test suite complete: 2 files passed", log)
end
