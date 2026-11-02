# Opt-in GPU integration test. Default A100; explicitly authorized V100 runs:
# CUDA_VISIBLE_DEVICES=0 ATMOSTR_RUN_CS_MULTIFILE_GPU_TESTS=1 \
# ATMOSTR_MULTIFILE_GPU_NAME=V100 julia --project=. test/diagnostic/test_cs_multifile_gpu.jl
using Test
if get(ENV,"ATMOSTR_RUN_CS_MULTIFILE_GPU_TESTS","0") != "1"
    @info "Skipping opt-in CS multifile GPU tests"
else
    using CUDA
    expected_device = get(ENV,"ATMOSTR_MULTIFILE_GPU_NAME","A100")
    isempty(expected_device) && error("ATMOSTR_MULTIFILE_GPU_NAME must name the authorized GPU")
    @assert occursin(expected_device,CUDA.name(CUDA.device())) "Wrong device for CS multifile GPU tests"
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__,"..","fixtures","cs_multifile.jl"))
    using .CSDriverHandoffFixtures
    @testset "CS multifile GPU physics and CPU reference" begin
        mktempdir() do dir
            paths = joinpath.(dir,["combined.bin","first.bin","second.bin"])
            for (path,strengths) in zip(paths,([1.0,1.0,8.0,8.0],[1.0,1.0],[8.0,8.0]))
                cs_handoff_fixture(path,strengths;FT=Float32)
            end
            for (adv,conv) in (("upwind","cmfmc"),("ppm","cmfmc_matrix"),("linrood","tm5"))
                reference = cs_handoff_run([paths[1]],adv,conv;FT=Float32)
                continuous = cs_handoff_run([paths[1]],adv,conv;FT=Float32,use_gpu=true)
                split = cs_handoff_run(paths[2:3],adv,conv;FT=Float32,use_gpu=true)
                @test split.state.tracers_raw[1] isa CuArray
                for p in 1:6
                    actual = Array(split.state.tracers_raw[p])
                    @test Array(split.state.air_mass[p]) == Array(continuous.state.air_mass[p])
                    same_run = Array(continuous.state.tracers_raw[p])
                    max_abs = maximum(abs.(actual .- same_run))
                    max_rel = max_abs / max(maximum(abs,same_run),eps(Float32))
                    @info "CS file-boundary comparison" adv conv panel=p max_abs max_rel
                    # Existing GPU file handoff rounds at Float32 precision:
                    # baseline max-relative differences reach 1.74e-7 on V100.
                    @test max_rel <= 4eps(Float32)
                    @test actual ≈ reference.state.tracers_raw[p] rtol=5e-5
                    @test all(isfinite,actual)
                end
            end
        end
    end
end
