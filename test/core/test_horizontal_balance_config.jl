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

@testset "[numerics] balance_mode is removed" begin
    @test HORIZONTAL_BALANCE_TAG == "column"
    @test reject_removed_balance_mode(Dict{String, Any}()) === nothing
    for key in ("balance_mode", "geos_balance_mode"), value in ("column", "per_layer")
        err = try reject_removed_balance_mode(Dict{String, Any}(key => value)); nothing catch e; e end
        @test err isa ArgumentError && contains(err.msg, "[numerics].$key is removed")
        # ERA5 spectral configs reach the same check.
        @test_throws ArgumentError resolve_balance_settings(
            Dict{String, Any}("numerics" => Dict{String, Any}(key => value)))
    end
    # `hybrid_mass` is the default on every target (owner decision OF-03).
    for type in ("latlon", "reduced_gaussian", "synthetic_reduced_gaussian", "cubed_sphere")
        @test resolve_balance_settings(Dict{String, Any}(
            "grid" => Dict{String, Any}("type" => type))).column_balance_weights === :hybrid_mass
    end
    @test resolve_balance_settings(Dict{String, Any}()).column_balance_weights === :hybrid_mass
    # Native configs: rejected before the met source is read.
    err = try
        process_day(Dict{String, Any}("source" => Dict{String, Any}("toml" => "missing.toml"),
                                      "numerics" => Dict{String, Any}("balance_mode" => "column")))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && contains(err.msg, "balance_mode is removed")
end

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

@testset "column balance weights" begin
    P = AtmosTransport.Preprocessing
    @test P.resolve_column_balance_weights(Dict{String, Any}()) === :hybrid_mass
    @test ERA5N320Settings(root_dir = "/tmp").column_balance_weights === :hybrid_mass
    @test MERRA2Settings(root_dir = "/tmp").column_balance_weights === :hybrid_mass
    @test P.resolve_column_balance_weights(Dict("column_balance_weights" => "Hybrid_B")) === :hybrid_b
    @test_throws ArgumentError P.resolve_column_balance_weights(Dict("column_balance_weights" => "pressure"))
    @test_throws ArgumentError P.resolve_column_balance_weights(Dict("column_balance_weights" => 1))
    # Headers record non-default weights only (default headers stay unchanged).
    @test P._with_column_weights_record(Dict{String, Any}(), :mass) == Dict{String, Any}()
    @test P._with_column_weights_record(Dict{String, Any}(), :hybrid_b) ==
          Dict{String, Any}("column_balance_weights" => "hybrid_b")
    # GEOS: the weights apply to the closures that balance the column; the
    # pressure-fixer closures keep the native fluxes, so any weights are
    # accepted and ignored there.
    check = P._check_geos_column_weights
    @test check(:mass, :pressure_fixer) === nothing
    @test check(:hybrid_b, :endpoint_balanced) === nothing
    @test check(:hybrid_b, :moisture_filtered) === nothing
    @test check(:hybrid_mass, :omega_regularized) === nothing
    @test check(:hybrid_b, :pressure_fixer) === nothing
    @test check(:hybrid_mass, :pfix_corrected) === nothing
    @test_throws ArgumentError check(:pressure, :endpoint_balanced)
    # Native configs take the weights from the met source, not [numerics].
    err = try
        process_day(Dict{String, Any}("source" => Dict{String, Any}("toml" => "missing.toml"),
                                      "numerics" => Dict{String, Any}("column_balance_weights" => "hybrid_b")))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && contains(err.msg, "met-source [preprocessing]")
    # Every spectral target accepts every weighting; an unknown name is
    # rejected by the entry point before any file is read (the paths below do
    # not exist).
    spectral_cfg = Dict{String, Any}(
        "input" => Dict{String, Any}("spectral_dir" => "/nonexistent/spectral",
                                     "coefficients" => "/nonexistent/coefficients.toml",
                                     "thermo_dir" => "/nonexistent/thermo"),
        "output" => Dict{String, Any}("directory" => "/nonexistent/out"),
        "grid" => Dict{String, Any}("type" => "latlon", "nlon" => 8, "nlat" => 5,
                                    "merge_min_thickness_Pa" => 0.0),
        "numerics" => Dict{String, Any}("dt" => 900.0, "met_interval" => 3600.0,
                                        "column_balance_weights" => "pressure"))
    err = try process_day(spectral_cfg; day_override = "2021-12-01"); nothing catch e; e end
    @test err isa ArgumentError && contains(err.msg, "column_balance_weights must be one of")
    for type in ("latlon", "cubed_sphere")
        grid = build_target_geometry(Dict{String, Any}("type" => type, "nlon" => 8, "nlat" => 5,
                                                       "Nc" => 2), Float64)
        @test P._check_spectral_column_weights(grid, (; column_balance_weights = :hybrid_b)) === :hybrid_b
        @test P._check_spectral_column_weights(grid, NamedTuple()) === :hybrid_mass
    end
end

@testset "GEOS header records the balance its closure applies" begin
    geos_balance = AtmosTransport.Preprocessing._geos_effective_balance
    @test geos_balance(:endpoint_balanced) == "column"
    @test geos_balance(:moisture_filtered) == "column"
    @test geos_balance(:omega_regularized) == "column"
    @test geos_balance(:pressure_fixer) == "none"
    @test geos_balance(:pfix_corrected) == "none"
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
