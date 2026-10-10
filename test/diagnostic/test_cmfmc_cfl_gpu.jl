# Opt-in CUDA diagnostic: the CMFMC CFL scan runs on the GPU and returns the host value exactly.
# ATMOSTR_RUN_CMFMC_CFL_GPU_TESTS=1 ATMOSTR_CMFMC_CFL_GPU_NAME=L40S \
# CUDA_VISIBLE_DEVICES=<authorized device> julia --project=. test/diagnostic/test_cmfmc_cfl_gpu.jl
# The checks are `check_cmfmc_cfl_on_device(adapt; float_types)` in test/helpers/cmfmc_cfl.jl;
# this file runs them on CUDA only. Another backend calls the same function with its array
# constructor and supported float types, e.g. `check_cmfmc_cfl_on_device(MtlArray;
# float_types = (Float32,))` after `using Metal` (done on an Apple M5 Pro, 2026-10-09: exact).
using Test
if get(ENV, "ATMOSTR_RUN_CMFMC_CFL_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in CMFMC CFL GPU tests"
else
    using CUDA
    expected_device = get(ENV, "ATMOSTR_CMFMC_CFL_GPU_NAME", "A100")
    isempty(expected_device) && error("ATMOSTR_CMFMC_CFL_GPU_NAME must name the authorized GPU")
    @assert occursin(expected_device, CUDA.name(CUDA.device())) "Wrong device for CMFMC CFL GPU tests"
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__, "..", "helpers", "cmfmc_cfl.jl"))
    @testset "GPU CMFMC CFL scan matches the host scan exactly" begin
        check_cmfmc_cfl_on_device(CuArray; float_types = (Float32, Float64))
    end
end
