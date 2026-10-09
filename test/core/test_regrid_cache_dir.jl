# Runtime surface-flux regridding weights are cached in
# `~/.cache/AtmosTransport/cr_regridding` unless ATMOSTR_REGRID_CACHE_DIR names
# another directory (the golden-output harness uses it to recompute weights).

using Test
using AtmosTransport

const ICIO = AtmosTransport.Models.InitialConditionIO

@testset "runtime regridding cache directory" begin
    default = joinpath(homedir(), ".cache", "AtmosTransport", "cr_regridding")
    withenv("ATMOSTR_REGRID_CACHE_DIR" => nothing) do
        @test ICIO._regrid_cache_dir() == default
    end
    withenv("ATMOSTR_REGRID_CACHE_DIR" => "") do                  # empty means unset
        @test ICIO._regrid_cache_dir() == default
    end
    withenv("ATMOSTR_REGRID_CACHE_DIR" => "~/golden/_regrid_cache") do
        @test ICIO._regrid_cache_dir() == joinpath(homedir(), "golden", "_regrid_cache")
    end
end
