#!/usr/bin/env julia
#
# `[numerics] balance_mode` selects the horizontal mass-flux balance of the
# preprocessors (column or per layer). The older key `geos_balance_mode` and,
# where it used to apply, the deprecated environment switch still work; every
# writer receives the typed mode and records it in the header.

using Test
using Dates
using AtmosTransport
using .AtmosTransport.Preprocessing: resolve_horizontal_balance, effective_horizontal_balance,
    ColumnBalance, LayerBalance, balance_tag, load_met_settings, build_target_geometry,
    process_day

const REPO = pkgdir(AtmosTransport)
const ENV_SWITCH = "ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE"

@testset "[numerics] balance_mode" begin
    @test resolve_horizontal_balance(Dict{String, Any}()) === nothing   # path default
    @test resolve_horizontal_balance(Dict("balance_mode" => "column")) === ColumnBalance()
    @test resolve_horizontal_balance(Dict("balance_mode" => "Per_Layer")) === LayerBalance()
    @test resolve_horizontal_balance(Dict("geos_balance_mode" => "per_layer")) === LayerBalance()
    @test resolve_horizontal_balance(Dict("balance_mode" => "column",
                                          "geos_balance_mode" => "column_poisson")) === ColumnBalance()
    @test_throws ErrorException resolve_horizontal_balance(Dict("balance_mode" => "column",
                                                                "geos_balance_mode" => "per_layer"))
    @test_throws ErrorException resolve_horizontal_balance(Dict("balance_mode" => "spectral"))
    @test balance_tag(ColumnBalance()) == "column"
    @test balance_tag(LayerBalance()) == "per_layer"
end

@testset "effective balance and the deprecated environment switch" begin
    withenv(ENV_SWITCH => nothing) do
        @test effective_horizontal_balance(nothing, ColumnBalance()) === ColumnBalance()
        @test effective_horizontal_balance(nothing, LayerBalance()) === LayerBalance()
        @test effective_horizontal_balance(LayerBalance(), ColumnBalance()) === LayerBalance()
    end
    withenv(ENV_SWITCH => "1") do
        @test (@test_logs (:warn, r"deprecated") effective_horizontal_balance(nothing, ColumnBalance())) ===
              LayerBalance()
        @test effective_horizontal_balance(LayerBalance(), ColumnBalance()) === LayerBalance()
        @test (@test_logs (:warn, r"ignored") effective_horizontal_balance(ColumnBalance(), ColumnBalance())) ===
              ColumnBalance()                                   # explicit configuration wins
        # GEOS never read the switch
        @test effective_horizontal_balance(nothing, ColumnBalance(); env = false) === ColumnBalance()
        @test effective_horizontal_balance(ColumnBalance(), ColumnBalance(); env = false) === ColumnBalance()
    end
end

@testset "GEOS header records the balance its closure applies" begin
    geos_balance = AtmosTransport.Preprocessing._geos_effective_balance
    @test geos_balance(:endpoint_balanced, :per_layer) == "per_layer"
    @test geos_balance(:endpoint_balanced, :column) == "column"
    @test geos_balance(:pressure_fixer, :per_layer) == "none"
    @test geos_balance(:pfix_corrected, :column) == "none"
    @test geos_balance(:moisture_filtered, :per_layer) == "column"
    @test geos_balance(:omega_regularized, :per_layer) == "column"
end

# The typed mode reaches the MERRA-2 and ERA5 N320 writers (their `process_day`
# used to swallow `balance_mode`): a per-layer balance with hybrid column weights
# is refused by the writer itself, before any input file is opened.
@testset "MERRA-2 and ERA5 N320 writers receive the balance mode" begin
    mktempdir() do tmp
        grid = build_target_geometry(Val(:cubed_sphere), Dict{String, Any}(
            "type" => "cubed_sphere", "Nc" => 4, "panel_convention" => "geos_native",
            "definition" => "gmao_equal_distance"), Float64)
        cases = (
            (joinpath(REPO, "config", "met_sources", "merra2_geoschem_hm_gchp.toml"), (Nz = 72,)),
            (joinpath(REPO, "config", "met_sources", "era5_n320_arco_diffusion_li.toml"),
             (Nz_native = 137, plan = nothing)),
        )
        for (toml, vertical) in cases
            settings = load_met_settings(toml; root_dir = tmp)
            @test settings.column_balance_weights !== :mass
            run(balance) = withenv(ENV_SWITCH => nothing) do
                process_day(Date(2021, 12, 1), grid, settings, vertical;
                            out_path = joinpath(tmp, "out.bin"), horizontal_balance = balance)
            end
            err = try run(LayerBalance()); nothing catch e; e end
            @test err isa ArgumentError && occursin("balance_mode = \"per_layer\"", err.msg)
            # The column default passes this check and fails later, on the missing inputs.
            err = try run(nothing); nothing catch e; e end
            @test !(err isa ArgumentError && occursin("balance_mode", err.msg))
        end
    end
end

@testset "LL → CS regrid script option --balance-mode" begin
    mod = Module(:RegridScriptBalanceTest)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    withenv("ATMOSTR_NO_AUTO_THREADS" => "1") do
        Base.include(mod, joinpath(REPO, "scripts", "preprocessing", "regrid_ll_transport_binary_to_cs.jl"))
    end
    parse = getfield(mod, :_parse_args)
    mktemp() do input, _
        base = ["--input", input, "--output", input * ".cs", "--Nc", "8"]
        @test parse(base).balance_mode === nothing
        @test parse(vcat(base, ["--balance-mode", "per_layer"])).balance_mode == "per_layer"
        @test_throws ErrorException parse(vcat(base, ["--balance-mode", "spectral"]))
    end
end
