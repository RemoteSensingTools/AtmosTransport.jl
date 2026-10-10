#!/usr/bin/env julia
#
# Top-level test runner for the offline regridding glue
# (`AtmosTransport.Regridding`). Invoked either directly
# (`julia --project=test test/regridding/runtests.jl`) or included
# from `test/runtests.jl`.

using Test

using AtmosTransport
using .AtmosTransport.Grids: GEOSNativePanelConvention, GnomonicPanelConvention
using .AtmosTransport.Regridding

@testset "Regridding (Tier 4)" begin
    include("test_cubed_sphere_corners.jl")
    include("test_conservation.jl")
    include("test_transpose.jl")
    include("test_serialization.jl")
    include("test_reduced_gaussian_stub.jl")
end
