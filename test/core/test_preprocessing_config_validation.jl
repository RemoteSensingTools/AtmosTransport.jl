#!/usr/bin/env julia
# Every shipped preprocessing config passes the entry point's config-level
# checks (`validate_preprocessing_config`, the same plan `process_day(cfg)`
# resolves before it opens any input). A default that a shipped config cannot
# honor — e.g. the `hybrid_mass` column weights inherited by the GEOS
# pressure-fixer configs — fails here instead of at the first preprocessing day.

using Test
using TOML
using Logging
using AtmosTransport

const Pre = AtmosTransport.Preprocessing
const CONFIG_ROOT = joinpath(pkgdir(AtmosTransport), "config", "preprocessing")

function _shipped_preprocessing_configs()
    paths = String[]
    for (dir, _, files) in walkdir(CONFIG_ROOT)
        occursin("likely_legacy", dir) && continue
        append!(paths, joinpath(dir, f) for f in files if endswith(f, ".toml"))
    end
    return sort(paths)
end

# Files outside the repository that validation reads: the reduced-Gaussian
# grid takes its geometry from a GRIB file in the user's data tree.
const EXTERNAL_VALIDATION_INPUTS = (("grid", "geometry_source_grib"),)

function _missing_external_inputs(cfg)
    missing = String[]
    for (table, key) in EXTERNAL_VALIDATION_INPUTS
        haskey(get(cfg, table, Dict()), key) || continue
        path = Pre.expand_data_path(String(cfg[table][key]))
        isfile(path) || push!(missing, "[$table].$key = $path")
    end
    return missing
end

@testset "shipped preprocessing configs validate" begin
    paths = _shipped_preprocessing_configs()
    @test length(paths) >= 51
    skipped = String[]
    # Shipped configs name their coefficient tables relative to the package
    # root; validation must not depend on the working directory (`Pkg.test`
    # runs in `test/`).
    cd(mktempdir()) do
        for path in paths
            cfg = TOML.parsefile(path)
            absent = _missing_external_inputs(cfg)
            if !isempty(absent)
                @info "Skipping $(basename(path)): its external input is absent " *
                      "($(join(absent, "; ")))."
                push!(skipped, basename(path))
                @test_skip false
                continue
            end
            plan = with_logger(NullLogger()) do
                Pre.validate_preprocessing_config(cfg)
            end
            @test plan.vertical.Nz >= 1
        end
    end
    @info "Validated $(length(paths) - length(skipped)) of $(length(paths)) shipped " *
          "preprocessing configs; skipped $(length(skipped)) whose external inputs are " *
          "absent" * (isempty(skipped) ? "." : ": $(join(skipped, ", ")).")
    @test length(paths) - length(skipped) >= 51
end

@testset "coefficient tables resolve independent of the working directory" begin
    shipped = joinpath(pkgdir(AtmosTransport), "config", "era5_L137_coefficients.toml")
    cd(mktempdir()) do
        @test Pre.resolve_coefficients_path("config/era5_L137_coefficients.toml") ==
              normpath(shipped)
        # A table in the working directory takes precedence over the shipped one.
        mkpath("config")
        cp(shipped, joinpath("config", "era5_L137_coefficients.toml"))
        # An identical copy is used silently; a differing one is used with a warning.
        @test (@test_logs Pre.resolve_coefficients_path("config/era5_L137_coefficients.toml")) ==
              "config/era5_L137_coefficients.toml"
        write(joinpath("config", "era5_L137_coefficients.toml"),
              read(shipped, String) * "\n# local edit\n")
        @test (@test_logs (:warn, r"differs from the shipped table") Pre.resolve_coefficients_path(
                   "config/era5_L137_coefficients.toml")) == "config/era5_L137_coefficients.toml"
        @test Pre.n_levels(Pre.load_hybrid_coefficients("config/geos_L72_coefficients.toml")) == 72
        err = try
            Pre.resolve_coefficients_path("config/no_such_coefficients.toml")
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin(joinpath(pwd(), "config", "no_such_coefficients.toml"), err)
        @test occursin(joinpath(pkgdir(AtmosTransport), "config", "no_such_coefficients.toml"), err)
        @test_throws ErrorException Pre.resolve_coefficients_path(joinpath(pwd(), "absent.toml"))
    end
end

@testset "GEOS closures without a column balance accept any weights" begin
    pressure_fixer = filter(_shipped_preprocessing_configs()) do path
        closure = get(get(TOML.parsefile(path), "numerics", Dict()), "geos_cm_closure", "")
        closure in ("pressure_fixer", "pfix_corrected")
    end
    @test length(pressure_fixer) == 3
    for path in pressure_fixer
        plan = with_logger(NullLogger()) do
            Pre.validate_preprocessing_config(TOML.parsefile(path))
        end
        # The met source's default weights reach the non-balancing closure.
        @test plan.settings.column_balance_weights === :hybrid_mass
        @test Pre._geos_effective_balance(plan.cm_closure) == "none"
    end
    for closure in (:pressure_fixer, :pfix_corrected), kind in (:mass, :hybrid_b, :hybrid_mass)
        @test (@test_logs min_level = Logging.Warn Pre._check_geos_column_weights(kind, closure)) === nothing
        header = Pre._with_geos_column_weights_record(Dict{String, Any}(), kind, closure)
        @test !haskey(header, "column_balance_weights")
    end
    @test Pre._with_geos_column_weights_record(Dict{String, Any}(), :hybrid_mass,
                                               :endpoint_balanced)["column_balance_weights"] ==
          "hybrid_mass"
    @test_throws ArgumentError Pre._check_geos_column_weights(:pressure, :pressure_fixer)
end
