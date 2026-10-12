# Every call in the maintained Julia files (src, ext, test, scripts,
# benchmarking, examples, docs/literate, docs/scripts) passes the keywords that
# all candidate methods of src/ require: scripts/checks/check_keyword_calls.jl.
# A signature change that adds a required keyword (for example
# `surface_pressure_basis` of the transport-binary writers) fails here until
# every caller, including the untested ones, passes it.
using Test

include(joinpath(@__DIR__, "..", "..", "scripts", "checks", "check_keyword_calls.jl"))

@testset "calls pass every required keyword" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    findings, nfiles, nfunctions = KeywordCallCheck.check_repository(root)
    for f in findings
        @error "$(relpath(f.file, root)):$(f.line): `$(f.name)` call omits required keyword(s) " *
               join(f.missing_keywords, ", ")
    end
    @test isempty(findings)
    @test nfiles > 300
    @test nfunctions > 50
end

@testset "the check finds an omitted keyword" begin
    mktempdir() do root
        mkpath(joinpath(root, "src")); mkpath(joinpath(root, "test"))
        write(joinpath(root, "src", "a.jl"), """
            writer(path, n; FT = Float64, mass_basis::Symbol, surface_pressure_basis) = path
            writer(path) = writer(path, 1; mass_basis = :dry, surface_pressure_basis = 1)
            Base.@kwdef struct Cfg
                a::Int
                b::Int = 2
            end
            Base.@kwdef mutable struct State
                const c::Int
                d::Int = 0
            end
            """)
        write(joinpath(root, "test", "b.jl"), """
            writer("p", 2; mass_basis = :dry)
            writer("p", 2, mass_basis = :dry, surface_pressure_basis = 1)
            writer("p")
            writer("p", 2; kw...)
            @test_throws UndefKeywordError writer("p", 2)
            Cfg(; b = 3)
            Cfg(1, 2)
            State(; d = 1)
            """)
        findings, _, _ = KeywordCallCheck.check_repository(root)
        @test [(f.line, f.name, f.missing_keywords) for f in findings] ==
              [(1, :writer, [:surface_pressure_basis]), (6, :Cfg, [:a]), (8, :State, [:c])]
    end
end
