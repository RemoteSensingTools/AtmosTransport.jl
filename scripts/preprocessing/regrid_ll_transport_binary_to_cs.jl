#!/usr/bin/env julia
# ===========================================================================
# Regrid an LL transport binary to a cubed-sphere transport binary.
#
# Compatibility CLI over `Preprocessing.regrid_transport_binary` (plan 40
# Commit 3). The concrete implementation currently dispatches to the LL → CS
# method; new binary regrid pairs should extend `regrid_transport_binary`
# rather than adding another topology-specific script.
# All of the real work — conservative regrid, wind recovery, panel-local
# rotation, flux reconstruction, per-level mass consistency, CS Poisson
# balance, cm diagnosis, streaming v4 write — lives in
# `src/Preprocessing/binary_pipeline.jl:1545`.
#
# Timestep metadata is read from the source binary header. `--steps-per-window`
# can override only the output substep count: winds are recovered with source
# scaling and CS face fluxes are reconstructed with output scaling. There is no
# `--met-interval` or `--dt` flag.
#
# Usage:
#   julia -t16 --project=. \
#       scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \
#       --input  <path/to/ll.bin> \
#       --output <path/to/cs.bin> \
#       --Nc 48
#       [--float-type Float32|Float64]   # default Float64 (matches LL source)
#       [--mass-basis dry|moist]         # default: match source header
#       [--convention gnomonic|geos_native]
#       [--definition equiangular_gnomonic|gmao_equal_distance]
#       [--steps-per-window 12]          # override source's substep count
#       [--balance-mode column|per_layer] # Poisson balance (default column)
#                                         # (smaller per-substep flux; needed
#                                         # for high-res CS output that
#                                         # otherwise fails the positivity gate)
#
# Regridder weights are auto-cached by ConservativeRegridding.jl at
# `~/.cache/AtmosTransport/cr_regridding/regridder_<hash>.jld2` keyed
# on source/target grids. Use `--cache-dir` for hermetic tests or scratch
# runs; otherwise the default cache is used. First run builds and persists;
# subsequent runs hit the cache (~6s vs the rebuild time).
# ===========================================================================

using Logging
using Dates

using AtmosTransport
using AtmosTransport.Preprocessing: regrid_transport_binary, build_target_geometry

const USAGE = """
Usage: julia --project=. scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \\
           --input <ll.bin> --output <cs.bin> --Nc <int>
           [--float-type Float32|Float64] [--mass-basis dry|moist]
           [--convention gnomonic|geos_native]
           [--definition equiangular_gnomonic|gmao_equal_distance]
           [--cache-dir <dir>]
           [--steps-per-window <int>] [--allow-positivity-violation]
           [--balance-mode column|per_layer]
"""

function _parse_args(argv)
    input = nothing
    output = nothing
    Nc = 0
    float_type = "Float64"
    mass_basis = nothing     # nothing = match source header
    convention = "gnomonic"
    definition = nothing
    cache_dir = nothing
    steps_per_window = nothing  # nothing = match source header
    require_substep_positivity = true
    balance_mode = nothing      # nothing = column (the preprocessing default)

    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--input" && i + 1 <= length(argv)
            input = expanduser(argv[i + 1]); i += 2
        elseif arg == "--output" && i + 1 <= length(argv)
            output = expanduser(argv[i + 1]); i += 2
        elseif arg == "--Nc" && i + 1 <= length(argv)
            Nc = parse(Int, argv[i + 1]); i += 2
        elseif arg == "--float-type" && i + 1 <= length(argv)
            float_type = argv[i + 1]; i += 2
        elseif arg == "--mass-basis" && i + 1 <= length(argv)
            mass_basis = argv[i + 1]; i += 2
        elseif arg == "--convention" && i + 1 <= length(argv)
            convention = argv[i + 1]; i += 2
        elseif arg == "--definition" && i + 1 <= length(argv)
            definition = argv[i + 1]; i += 2
        elseif arg == "--cache-dir" && i + 1 <= length(argv)
            cache_dir = expanduser(argv[i + 1]); i += 2
        elseif arg == "--steps-per-window" && i + 1 <= length(argv)
            steps_per_window = parse(Int, argv[i + 1]); i += 2
        elseif arg == "--balance-mode" && i + 1 <= length(argv)
            balance_mode = lowercase(argv[i + 1]); i += 2
        elseif arg == "--allow-positivity-violation"
            require_substep_positivity = false; i += 1
        elseif arg in ("-h", "--help")
            println(USAGE); exit(0)
        else
            error("Unknown argument `$(arg)`.\n$(USAGE)")
        end
    end

    input  === nothing && error("--input required.\n$(USAGE)")
    output === nothing && error("--output required.\n$(USAGE)")
    Nc > 0 ||               error("--Nc required (positive integer).\n$(USAGE)")
    isfile(input) ||        error("Input binary not found: $(input)")
    float_type in ("Float32", "Float64") ||
        error("--float-type must be Float32 or Float64, got $(float_type)")
    mass_basis === nothing || mass_basis in ("dry", "moist") ||
        error("--mass-basis must be dry or moist, got $(mass_basis)")
    norm_convention = lowercase(convention)
    norm_convention in ("gnomonic", "geos_native") ||
        error("--convention must be gnomonic or geos_native, got $(convention)")
    convention = norm_convention
    if definition !== nothing
        norm_definition = lowercase(definition)
        norm_definition in ("equiangular_gnomonic", "gmao_equal_distance") ||
            error("--definition must be equiangular_gnomonic or gmao_equal_distance, got $(definition)")
        definition = norm_definition
    end

    steps_per_window === nothing || steps_per_window >= 1 ||
        error("--steps-per-window must be ≥ 1, got $(steps_per_window)")
    balance_mode === nothing || balance_mode in ("column", "per_layer") ||
        error("--balance-mode must be column or per_layer, got $(balance_mode)")

    return (; input, output, Nc, float_type, mass_basis, convention, definition,
              cache_dir,
              steps_per_window, require_substep_positivity, balance_mode)
end

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    opts = _parse_args(ARGS)

    FT = opts.float_type == "Float32" ? Float32 : Float64

    cfg_grid = Dict{String, Any}(
        "Nc" => opts.Nc,
        "panel_convention" => opts.convention,
    )
    opts.definition === nothing || (cfg_grid["definition"] = opts.definition)
    opts.cache_dir === nothing || (cfg_grid["regridder_cache_dir"] = opts.cache_dir)
    cs_grid = build_target_geometry(Val(:cubed_sphere), cfg_grid, FT)

    basis_sym = opts.mass_basis === nothing ? nothing : Symbol(opts.mass_basis)

    @info "LL → CS transport-binary regrid"
    @info "  input:      $(opts.input)"
    @info "  output:     $(opts.output)"
    @info "  target:     C$(opts.Nc) $(opts.convention) CS" *
          (opts.definition === nothing ? "" : " ($(opts.definition))")
    @info "  float type: $(opts.float_type)"
    @info "  mass_basis: $(basis_sym === nothing ? "(match source)" : basis_sym)"

    opts.steps_per_window === nothing ||
        @info "  steps_per_window override: $(opts.steps_per_window)"

    regrid_transport_binary(opts.input, cs_grid, opts.output;
                            FT         = FT,
                            mass_basis = basis_sym,
                            steps_per_window = opts.steps_per_window,
                            require_substep_positivity = opts.require_substep_positivity,
                            horizontal_balance = opts.balance_mode === nothing ? nothing :
                                AtmosTransport.Preprocessing.resolve_horizontal_balance(
                                    Dict("balance_mode" => opts.balance_mode)))

    return opts.output
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
