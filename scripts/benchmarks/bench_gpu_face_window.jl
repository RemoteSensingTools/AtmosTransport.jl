# Warmed full 00–01 UTC production window, including native L137 integration,
# humidity, pin, balance, dry closure/replay, positivity and binary write.
using AtmosTransport, Dates, TOML, LinearAlgebra, FFTW
const REPO = normpath(joinpath(@__DIR__,"../.."))
const PP = AtmosTransport.Preprocessing
include(joinpath(REPO,"scripts/validation/gpu_face_memory.jl"))
BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

function install_validation_window(name)
    folder, mod, assignment = name == "ll" ?
        ("LatLon",PP.LatLonTarget,"nt = 24") :
        ("CubedSphere",PP.CubedSphereTarget,"nt, FT, nz = 24,")
    path = joinpath(REPO,"src/Preprocessing/Targets",folder,"tm5_spectral_day.jl")
    source = read(path,String)
    count(assignment,source) == 1 || error("review one-window adapter: assignment changed")
    occursin("i <= nt",source) || error("review one-window adapter: endpoint changed")
    source = replace(source,assignment=>replace(assignment,"24"=>"1"),"i <= nt"=>"i <= 24")
    Base.include_string(mod,source,path)
end
include(joinpath(REPO,"scripts/preprocessing/preprocess_transport_binary.jl"))

function benchmark_face_window(backend,name,out)
    name in ("ll","cs") || error("target must be ll or cs")
    backend in ("cpu","gpu") || error("backend must be cpu or gpu")
    if backend == "gpu"
        CUDA.functional() || error("CUDA unavailable")
        CUDA.allowscalar(false)
        CUDA.math_mode!(CUDA.DEFAULT_MATH)
        PP.Spectral.spectral_gpu_precision() === Float32 || error("requires Float32 GPU policy")
    end
    install_validation_window(name)
    mkpath(out)
    backend == "gpu" && start_face_device_memory(joinpath(out,"device_memory.csv"))
    cfg = TOML.parsefile(joinpath(REPO,"test/golden/configs",name == "ll" ? "pre_ll72_tm5.toml" : "pre_c24_tm5.toml"))
    cfg["input"]["spectral_dir"] = expanduser("~/data/AtmosTransport/met/era5/spectral_hourly")
    cfg["input"]["thermo_dir"] = expanduser("~/data/AtmosTransport/met/era5/physics")
    cfg["input"]["coefficients"] = joinpath(REPO,"config/era5_L137_coefficients.toml")
    cfg["preprocessing"]["spectral_synthesis"] = backend
    cfg["preprocessing"]["spectral_synthesis_float"] = backend == "gpu" ? "Float32" : "Float64"
    cfg["grid"]["merge_min_thickness_Pa"] = 100.0
    if name == "ll"
        cfg["grid"]["nlon"],cfg["grid"]["nlat"] = 360,180
    else
        cfg["grid"]["Nc"] = 90
        cfg["grid"]["regridder_cache_dir"] = joinpath(out,"overlap_cache")
    end
    open(joinpath(out,"timings.tsv"),"w") do io
        println(io,"phase\tbackend\ttarget\tseconds\tallocated_bytes\tpeak_host_rss\tjulia_threads\tblas_threads\tfftw_threads")
        measured_reps = parse(Int,get(ENV,"GPUQUAD_MEASURED_REPS","2"))
        0 <= measured_reps <= 2 || error("GPUQUAD_MEASURED_REPS must be 0, 1 or 2")
        for rep in 0:measured_reps
            phase = rep == 0 ? "warmup" : "measured_$rep"
            cfg["output"]["directory"] = joinpath(out,phase)
            path = joinpath(out,"$phase.toml")
            open(f->TOML.print(f,cfg),path,"w")
            GC.gc()
            backend == "gpu" && CUDA.synchronize()
            elapsed = @timed begin
                result = Base.invokelatest(main,[path,"--day","2021-12-01"])
                backend == "gpu" && CUDA.synchronize()
                result
            end
            println(io,join((phase,backend,name,elapsed.time,elapsed.bytes,Sys.maxrss(),Threads.nthreads(),BLAS.get_num_threads(),FFTW.get_num_threads()),'\t'))
            flush(io)
            @info "complete face window" backend name phase seconds=elapsed.time
        end
    end
    backend == "gpu" && record_face_device_memory(joinpath(out,"pool_peak.toml"))
end
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("usage: cpu|gpu ll|cs OUTPUT")
    ARGS[1] == "gpu" && (@eval using CUDA)
    benchmark_face_window(ARGS[1],ARGS[2],abspath(ARGS[3]))
end
