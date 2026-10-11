#!/usr/bin/env julia
#
# `[numerics] balance_mode` selects the horizontal mass-flux balance of the
# preprocessors (column or per layer); every writer receives the typed mode and
# records it in the header. `column_balance_weights` chooses how the column
# balance spreads its correction over the levels.

using Test
using Dates
using AtmosTransport
using .AtmosTransport.Preprocessing: resolve_horizontal_balance, effective_horizontal_balance,
    ColumnBalance, LayerBalance, balance_tag, load_met_settings, build_target_geometry,
    process_day

const REPO = pkgdir(AtmosTransport)

@testset "[numerics] balance_mode" begin
    @test resolve_horizontal_balance(Dict{String, Any}()) === nothing   # path default
    @test resolve_horizontal_balance(Dict("balance_mode" => "column")) === ColumnBalance()
    @test resolve_horizontal_balance(Dict("balance_mode" => "Per_Layer")) === LayerBalance()
    # The old key name and the old value aliases are gone.
    err = try resolve_horizontal_balance(Dict("geos_balance_mode" => "column")); nothing catch e; e end
    @test err isa ArgumentError && contains(err.msg, "renamed balance_mode")
    for raw in ("column_poisson", "layer", "layer_local", "global", "spectral", 1)
        @test_throws ArgumentError resolve_horizontal_balance(Dict("balance_mode" => raw))
    end
    @test balance_tag(ColumnBalance()) == "column"
    @test balance_tag(LayerBalance()) == "per_layer"
end

@testset "column balance weights" begin
    P = AtmosTransport.Preprocessing
    @test P.resolve_column_balance_weights(Dict{String, Any}()) === :mass
    @test P.resolve_column_balance_weights(Dict("column_balance_weights" => "Hybrid_B")) === :hybrid_b
    @test_throws ArgumentError P.resolve_column_balance_weights(Dict("column_balance_weights" => "pressure"))
    @test_throws ArgumentError P.resolve_column_balance_weights(Dict("column_balance_weights" => 1))
    # A per-layer balance has no column correction to spread.
    err = try P._reject_layer_balance_weights(LayerBalance(), :hybrid_b); nothing catch e; e end
    @test err isa ArgumentError && contains(err.msg, "balance_mode = \"per_layer\"")
    @test P._reject_layer_balance_weights(LayerBalance(), :mass) === nothing
    @test P._reject_layer_balance_weights(ColumnBalance(), :hybrid_mass) === nothing
    # Headers record non-default weights only (default headers stay unchanged).
    @test P._with_column_weights_record(Dict{String, Any}(), :mass) == Dict{String, Any}()
    @test P._with_column_weights_record(Dict{String, Any}(), :hybrid_b) ==
          Dict{String, Any}("column_balance_weights" => "hybrid_b")
    # GEOS: the weights need a closure that balances the column.
    check = P._check_geos_column_weights
    @test check(:mass, :pressure_fixer, :column) === nothing
    @test check(:hybrid_b, :endpoint_balanced, :column) === nothing
    @test check(:hybrid_b, :moisture_filtered, :per_layer) === nothing   # always balances the column
    @test check(:hybrid_mass, :omega_regularized, :column) === nothing
    @test_throws ArgumentError check(:hybrid_b, :pressure_fixer, :column)
    @test_throws ArgumentError check(:hybrid_mass, :pfix_corrected, :column)
    err = try check(:hybrid_b, :endpoint_balanced, :per_layer); nothing catch e; e end
    @test err isa ArgumentError && contains(err.msg, "balance_mode = \"per_layer\"")
    @test_throws ArgumentError check(:pressure, :endpoint_balanced, :column)
    # Native configs take the weights from the met source, not [numerics].
    err = try
        process_day(Dict{String, Any}("source" => Dict{String, Any}("toml" => "missing.toml"),
                                      "numerics" => Dict{String, Any}("column_balance_weights" => "hybrid_b")))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && contains(err.msg, "met-source [preprocessing]")
    # Lat-lon and reduced-Gaussian spectral targets keep mass weights.
    @test_throws ArgumentError P._require_mass_column_weights((; column_balance_weights = :hybrid_b), "x")
    @test P._require_mass_column_weights(NamedTuple(), "x") === nothing
    # The spectral entry point checks the target before reading any file (the
    # paths below do not exist).
    spectral_cfg = Dict{String, Any}(
        "input" => Dict{String, Any}("spectral_dir" => "/nonexistent/spectral",
                                     "coefficients" => "/nonexistent/coefficients.toml",
                                     "thermo_dir" => "/nonexistent/thermo"),
        "output" => Dict{String, Any}("directory" => "/nonexistent/out"),
        "grid" => Dict{String, Any}("type" => "latlon", "nlon" => 8, "nlat" => 5,
                                    "merge_min_thickness_Pa" => 0.0),
        "numerics" => Dict{String, Any}("dt" => 900.0, "met_interval" => 3600.0,
                                        "column_balance_weights" => "hybrid_b"))
    err = try process_day(spectral_cfg; day_override = "2021-12-01"); nothing catch e; e end
    @test err isa ArgumentError && contains(err.msg, "cubed-sphere targets only")
end

@testset "effective balance" begin
    @test effective_horizontal_balance(nothing, ColumnBalance()) === ColumnBalance()
    @test effective_horizontal_balance(nothing, LayerBalance()) === LayerBalance()
    @test effective_horizontal_balance(LayerBalance(), ColumnBalance()) === LayerBalance()
    # The removed environment switch no longer changes the balance.
    withenv("ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE" => "1") do
        @test effective_horizontal_balance(nothing, ColumnBalance()) === ColumnBalance()
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
            run(balance) = process_day(Date(2021, 12, 1), grid, settings, vertical;
                                       out_path = joinpath(tmp, "out.bin"),
                                       horizontal_balance = balance)
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
    Base.include(mod, joinpath(REPO, "scripts", "preprocessing", "regrid_ll_transport_binary_to_cs.jl"))
    parse = getfield(mod, :_parse_args)
    mktemp() do input, _
        base = ["--input", input, "--output", input * ".cs", "--Nc", "8"]
        @test parse(base).balance_mode === nothing
        @test parse(vcat(base, ["--balance-mode", "per_layer"])).balance_mode == "per_layer"
        @test_throws ErrorException parse(vcat(base, ["--balance-mode", "spectral"]))
        @test parse(base).column_balance_weights == "mass"
        @test parse(vcat(base, ["--column-balance-weights", "Hybrid_B"])).column_balance_weights == "hybrid_b"
        @test_throws ErrorException parse(vcat(base, ["--column-balance-weights", "pressure"]))
    end
end
