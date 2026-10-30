#!/usr/bin/env julia
#
# Every preprocessing path that applies a Poisson balance uses the column
# balance and records `horizontal_balance = "column"`. GEOS closures that keep
# the native fluxes record `"none"`. The per-layer balance left no vertical mass
# flux (issue #11) and was removed with its `[numerics] balance_mode` key.
# `column_balance_weights` chooses how the correction is spread over the levels.

using Test
using Dates
using AtmosTransport
using .AtmosTransport.Preprocessing: HORIZONTAL_BALANCE_TAG, reject_removed_balance_mode,
    resolve_balance_settings, process_day, build_target_geometry,
    ERA5N320Settings, MERRA2Settings

const REPO = pkgdir(AtmosTransport)



@testset "removed writer keyword is not swallowed by native-source adapters" begin
    mktempdir() do tmp
        grid = build_target_geometry(Val(:cubed_sphere), Dict{String, Any}(
            "Nc" => 1, "staging_nlon" => 4, "staging_nlat" => 3,
            "regridder_cache_dir" => tmp), Float64)
        date = Date(2021, 12, 1)
        for settings in (ERA5N320Settings(root_dir = tmp),
                         MERRA2Settings(root_dir = tmp))
            err = try
                process_day(date, grid, settings, nothing;
                            out_path = joinpath(tmp, "unused.bin"),
                            horizontal_balance = :column)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test contains(err.msg, "horizontal_balance is removed")
        end
    end
end





@testset "LL → CS regrid script options" begin
    mod = Module(:RegridScriptBalanceTest)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Base.include(mod, joinpath(REPO, "scripts", "preprocessing", "regrid_ll_transport_binary_to_cs.jl"))
    parse = getfield(mod, :_parse_args)
    mktemp() do input, _
        base = ["--input", input, "--output", input * ".cs", "--Nc", "8"]
        @test !hasproperty(parse(base), :balance_mode)
        @test_throws ErrorException parse(vcat(base, ["--balance-mode", "column"]))   # removed
        @test parse(base).column_balance_weights == "hybrid_mass"
        @test parse(vcat(base, ["--column-balance-weights", "Hybrid_B"])).column_balance_weights == "hybrid_b"
        @test_throws ErrorException parse(vcat(base, ["--column-balance-weights", "pressure"]))
    end
end
