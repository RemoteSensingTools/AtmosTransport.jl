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
