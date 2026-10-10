# Opt-in, CUDA: the fused GPU halo fill (one launch for all panel edges, one for
# all corners) writes exactly what the reference host loops write. This file
# covers CUDA only; `check_halo_fill`'s `cases` keyword selects the Float32-only
# cases a Metal run needs.
# ATMOSTR_RUN_CS_HALO_GPU_TESTS=1 ATMOSTR_CS_HALO_GPU_NAME=L40S \
# CUDA_VISIBLE_DEVICES=<authorized device> julia --project=. test/diagnostic/test_cs_halo_fill_gpu.jl
using Test
if get(ENV, "ATMOSTR_RUN_CS_HALO_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in CS halo fill GPU tests"
else
    using CUDA
    expected_device = get(ENV, "ATMOSTR_CS_HALO_GPU_NAME", "A100")
    isempty(expected_device) && error("ATMOSTR_CS_HALO_GPU_NAME must name the authorized GPU")
    @assert occursin(expected_device, CUDA.name(CUDA.device())) "Wrong device for CS halo GPU tests"
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__, "..", "helpers", "cs_halo_fill.jl"))
    @testset "GPU halo fill matches the reference" begin
        results = check_halo_fill(library_fill!, library_corners!, CuArray)
        @test length(results) == 48
        @test all(results)
    end
end
