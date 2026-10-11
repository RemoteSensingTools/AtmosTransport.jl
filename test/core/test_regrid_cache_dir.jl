# The shared regridding weight cache (runtime surface fluxes and builders) is
# `~/.cache/AtmosTransport/cr_regridding` unless ATMOSTR_REGRID_CACHE_DIR names
# another directory (the golden-output harness uses it to recompute weights).

using Test
using AtmosTransport
using AtmosTransport.Regridding: default_regrid_cache_dir

@testset "shared regridding cache directory" begin
    default = joinpath(homedir(), ".cache", "AtmosTransport", "cr_regridding")
    withenv("ATMOSTR_REGRID_CACHE_DIR" => nothing) do
        @test default_regrid_cache_dir() == default
    end
    withenv("ATMOSTR_REGRID_CACHE_DIR" => "") do                  # empty means unset
        @test default_regrid_cache_dir() == default
    end
    withenv("ATMOSTR_REGRID_CACHE_DIR" => "~/golden/_regrid_cache") do
        @test default_regrid_cache_dir() == joinpath(homedir(), "golden", "_regrid_cache")
    end
    # The private Models copy is gone: one definition, in Regridding.
    @test !isdefined(AtmosTransport.Models.InitialConditionIO, :_regrid_cache_dir)
end
