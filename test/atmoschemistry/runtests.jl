using Pkg

Pkg.instantiate()

include("test_benchmark_records.jl")
include(joinpath(@__DIR__, "..", "core",
                 "test_atmoschemistry_extension.jl"))
include(joinpath(@__DIR__, "..", "diagnostic",
                 "test_atmoschemistry_gpu_extension.jl"))
