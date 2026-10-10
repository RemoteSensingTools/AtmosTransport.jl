# Opt-in: point operations launched on CUDA (one operation, a fused launch of
# operations with different index extents, a dependent sequence) write exactly
# what the host loops write. The checks live in test/helpers/point_ops.jl, so
# other backends can run them too.
# ATMOSTR_RUN_POINT_OPS_GPU_TESTS=1 ATMOSTR_POINT_OPS_GPU_NAME=L40S \
# CUDA_VISIBLE_DEVICES=<authorized device> julia --project=. test/diagnostic/test_point_ops_gpu.jl
using Test
if get(ENV, "ATMOSTR_RUN_POINT_OPS_GPU_TESTS", "0") != "1"
    @info "Skipping opt-in point operation GPU tests"
else
    using CUDA
    expected_device = get(ENV, "ATMOSTR_POINT_OPS_GPU_NAME", "A100")
    isempty(expected_device) && error("ATMOSTR_POINT_OPS_GPU_NAME must name the authorized GPU")
    @assert occursin(expected_device, CUDA.name(CUDA.device())) "Wrong device for point operation GPU tests"
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__, "..", "helpers", "point_ops.jl"))
    @testset "CUDA point operations" begin
        # One fused kernel on CUDA (separate launches are 3-9x slower there).
        @test PointArch.fusion_policy(CUDA.CUDABackend()) === PointArch.FuseLaunches()
        check_point_ops_on_device(CuArray)
    end
end
