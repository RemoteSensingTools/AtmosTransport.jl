#!/usr/bin/env julia

using Test

using AtmosTransport

const Arch = AtmosTransport.Architectures

struct MockCUDAArray end
struct MockMetalArray end

@testset "execution architecture selection" begin
    cpu = Arch.architecture_from_config(Dict("backend" => "cpu"))
    @test cpu isa Arch.CPU
    @test !Arch.is_gpu(cpu)
    @test Arch.array_adapter(cpu) === Array
    @test Arch.architecture_label(cpu) == "CPU"
    @test Arch.array_adapter_for(zeros(Float32, 2, 2)) === Array

    # A working-tree CLI may include AtmosTransport after loading CUDA or
    # Metal. In that mode package extensions cannot attach to the local module,
    # so adapter discovery must also recognize arrays through loaded runtimes.
    loaded = Dict{Base.PkgId, Any}(
        Arch._RUNTIME_PACKAGE_IDS.CUDA => (CuArray = MockCUDAArray,),
        Arch._RUNTIME_PACKAGE_IDS.Metal => (MtlArray = MockMetalArray,),
    )
    @test Arch._loaded_gpu_adapter(MockCUDAArray(), loaded) === MockCUDAArray
    @test Arch._loaded_gpu_adapter(MockMetalArray(), loaded) === MockMetalArray
    @test Arch._loaded_gpu_adapter(zeros(Float32, 2), loaded) === Array

    @test Arch.architecture_from_config(Dict("use_gpu" => false)) isa Arch.CPU
    @test Arch.architecture_from_config(Dict("backend" => "cuda")) isa Arch.GPU{:cuda}
    @test Arch.GPU("metal") isa Arch.GPU{:metal}
    @test Arch.backend_name(Arch.GPU(:cuda)) === :cuda
    @test Arch.is_gpu(Arch.GPU(:metal))
    @test_throws MethodError Arch.GPU()
    @test_throws ArgumentError Arch.GPU(:rocm)
    @test_throws ArgumentError Arch.architecture_from_config(Dict("use_gpu" => true,
                                                                  "backend" => "cpu"))
    @test_throws ArgumentError Arch.architecture_from_config(Dict("backend" => "rocm"))

    architecture_source = read(joinpath(@__DIR__, "..", "..", "src", "Architectures.jl"), String)
    driven_source = join((read(joinpath(@__DIR__, "..", "..", "src", "Models", f), String)
                          for f in ("DrivenSimulation.jl", "driven_window_state.jl",
                                    "driven_physics_refresh.jl", "driven_stepping.jl")))
    @test !occursin(r"\bisdefined\(Main|\bgetproperty\(Main|Core\.eval\(Main", architecture_source)
    @test !occursin(r"\bisdefined\(Main|\bgetproperty\(Main|Core\.eval\(Main", driven_source)
end

@testset "Metal requires Float32" begin
    metal = Arch.architecture_from_config(Dict("backend" => "metal"))
    @test metal isa Arch.GPU{:metal}
    @test Arch.assert_float_type!(metal, Float32) === nothing
    @test_throws ArgumentError Arch.assert_float_type!(metal, Float64)
end

@testset "runtime kernels avoid hard Float64 accumulation" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    files = [
        "src/MetDrivers/ERA5/VerticalClosure.jl",
        "src/Operators/Convection/cmfmc_kernels.jl",
    ]
    forbidden = r"Float64\(|zero\(Float64\)|::Float64"
    for file in files
        src = read(joinpath(repo, file), String)
        @test !occursin(forbidden, src)
    end
end

@testset "DrivenRunner resolves one concrete architecture" begin
    cfg = Dict("architecture" => Dict("backend" => "cpu"),
               "numerics" => Dict("float_type" => "Float64"))
    arch = AtmosTransport.Models.DrivenRunner._cfg_architecture(cfg)
    @test arch isa Arch.CPU
    @test Arch.array_adapter(arch) === Array
    @test Arch.architecture_label(arch) == "CPU"
    @test Arch.synchronize_architecture!(arch) === nothing

    runner_source = read(joinpath(@__DIR__, "..", "..", "src", "Models",
                                  "DrivenRunner.jl"), String)
    @test occursin("Base.invokelatest(_run_driven_simulation, cfg, arch)",
                   runner_source)
end

@testset "canonical CLI activates package extensions" begin
    cli_source = read(joinpath(@__DIR__, "..", "..", "scripts",
                               "run_transport.jl"), String)
    @test occursin(r"(?m)^using AtmosTransport$", cli_source)
    @test !occursin(r"include\(.*src.*AtmosTransport\.jl", cli_source)
end
