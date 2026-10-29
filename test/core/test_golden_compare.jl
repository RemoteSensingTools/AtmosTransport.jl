# No golden case is run: only tiny files written in temporary directories.
using Test, NCDatasets, JSON3, TOML

module GoldenCompareHarness
    include(joinpath(@__DIR__, "..", "golden", "run_goldens.jl"))
end
const GC = GoldenCompareHarness

function tiny_netcdf(path, bits; attr = Int32(4), dim = "cell", provenance = false)
    NCDataset(path, "c") do ds
        defDim(ds, dim, length(bits))
        v = defVar(ds, "tracer", Float64, (dim,); fillvalue = nothing)
        v.var[:] = reinterpret(Float64, bits)
        v.attrib["contract"] = attr
        provenance && (ds.attrib["history"] = "different provenance")
    end
end

function tiny_binary(path, bits; extra = Dict{String,Any}())
    header = Dict{String,Any}("header_bytes" => 512, "float_type" => "Float64",
                              "contract" => Dict("key" => 3))
    merge!(header, extra)
    json = JSON3.write(header)
    @assert sizeof(json) < 512
    open(path, "w") do io
        write(io, json)
        write(io, zeros(UInt8, 512 - sizeof(json)))
        write(io, bits)
    end
end

@testset "golden comparison protects stored bits and header contracts" begin
    mktempdir() do dir
        ref, new = joinpath(dir, "ref"), joinpath(dir, "new")
        mkpath(ref); mkpath(new)
        # Quiet NaN with a noncanonical payload, positive and negative zero.
        bits = UInt64[0x7ff8000000000042, 0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000]
        a, b = joinpath(ref, "tiny.nc"), joinpath(new, "tiny.nc")
        tiny_netcdf(a, bits)
        tiny_netcdf(b, bits; provenance = true)
        NCDataset(a) do ds
            @test reinterpret(UInt64, Array(ds["tracer"].var)) == bits
        end
        @test isempty(GC.compare_netcdf(a, b))
        for (index, changed) in ((1, 0x7ff8000000000043), (2, 0x8000000000000000),
                                 (4, 0x3ff0000000000001))
            different = copy(bits); different[index] = changed
            tiny_netcdf(b, different)
            @test any(s -> occursin("1 of 4 values differ", s), GC.compare_netcdf(a, b))
        end
        tiny_netcdf(b, bits; attr = Float64(4))
        @test any(s -> occursin("attribute contract", s), GC.compare_netcdf(a, b))
        tiny_netcdf(b, bits; dim = "other")
        @test any(s -> occursin("dimensions", s), GC.compare_netcdf(a, b))
        tiny_netcdf(b, bits)
        @test isempty(GC.compare_case(ref, new))

        a, b = joinpath(ref, "tiny.bin"), joinpath(new, "tiny.bin")
        tiny_binary(a, bits; extra = Dict("git_commit" => "reference"))
        tiny_binary(b, bits; extra = Dict("creation_time" => "new", "contract" =>
                                         Dict("key" => 3, "git_commit" => "nested provenance")))
        @test isempty(GC.compare_binary(a, b)) # provenance key set may differ, even nested
        tiny_binary(b, bits; extra = Dict("new_key" => 1))
        @test any(s -> occursin("header keys in one file only", s), GC.compare_binary(a, b))
        tiny_binary(b, bits; extra = Dict("contract" => Dict("different_key" => 3)))
        @test any(s -> occursin("header contract", s), GC.compare_binary(a, b))
        for (index, changed) in ((1, 0x7ff8000000000043), (2, 0x8000000000000000),
                                 (4, 0x3ff0000000000001))
            different = copy(bits); different[index] = changed
            tiny_binary(b, different)
            @test any(s -> occursin("payload: 1 of 4 values differ", s), GC.compare_binary(a, b))
        end
        @test !isempty(GC.compare_case(ref, new))

        # Exercise the actual report function, so a differing file cannot be
        # silently called identical by the CLI. No worker/case is launched.
        for folder in (ref, new)
            open(joinpath(folder, "status.toml"), "w") do io
                TOML.print(io, Dict("ok" => true, "wall_seconds" => 0.0))
            end
        end
        ref_tree, new_tree = joinpath(dir, "ref_tree"), joinpath(dir, "new_tree")
        mkpath(ref_tree); mkpath(new_tree)
        mv(ref, joinpath(ref_tree, "tiny")); mv(new, joinpath(new_tree, "tiny"))
        output = joinpath(dir, "report.txt")
        result = open(output, "w") do io
            redirect_stdout(io) do
                GC.compare_cases([Dict("name" => "tiny")], ref_tree, new_tree)
            end
        end
        @test !result
        @test occursin("DIFFERS", read(output, String))
    end
end
