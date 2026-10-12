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

@testset "shipped preprocessing configs validate" begin
    paths = _shipped_preprocessing_configs()
    @test length(paths) >= 60
    for path in paths
        cfg = TOML.parsefile(path)
        plan = with_logger(NullLogger()) do
            Pre.validate_preprocessing_config(cfg)
        end
        @test plan.vertical.Nz >= 1
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
