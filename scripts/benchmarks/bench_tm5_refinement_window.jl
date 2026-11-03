# CPU-only 00–01 UTC LL window, through production balance/closure/write gates.
# From the repo root:
# TMPDIR=/temp2/cfranken/tmp CUDA_VISIBLE_DEVICES= JULIA_NUM_THREADS=16 \
# OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=. \
#   scripts/benchmarks/bench_tm5_refinement_window.jl CONFIG DAY OUTPUT
# CONFIG supplies real spectral/humidity inputs and target/vertical settings.
# Warms both complete arms, then measures A B A B (two reps per arm).
# A fourth argument selects one isolated arm for process peak RSS instead.
# TSV excludes package startup; measured rows include decode/planning/write.
ENV["GIT_OPTIONAL_LOCKS"] = "0"
using AtmosTransport, Dates, TOML, LinearAlgebra, FFTW

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)
const REPO = normpath(joinpath(@__DIR__, "../.."))
const PP = AtmosTransport.Preprocessing

# Shorten only the driver's window loop. The endpoint branch must still read
# hour 01 of this day, rather than treating it as next-day midnight.
function install_one_window_driver(windows=1)
    path = joinpath(REPO, "src/Preprocessing/Targets/LatLon/tm5_spectral_day.jl")
    source = read(path, String)
    occursin("nt = 24", source) && occursin("if i <= nt", source) ||
        error("LL driver changed: review the one-window adapter")
    1 <= windows <= 23 || error("benchmark windows must be in 1:23")
    source = replace(source, "nt = 24" => "nt = $windows", "if i <= nt" => "if i <= 24")
    Base.include_string(PP.LatLonTarget, source, path)
end

include(joinpath(REPO, "scripts/preprocessing/preprocess_transport_binary.jl"))

function benchmark_refinement(config_path, day, output, isolated="")
    cfg = TOML.parsefile(config_path)
    get(cfg["grid"], "type", "latlon") == "latlon" || error("requires LL target")
    table = get!(cfg, "preprocessing", Dict{String,Any}())
    table["spectral_flux_construction"] = "tm5_spectral"
    table["spectral_synthesis"] = "cpu"
    cfg["input"]["coefficients"] = abspath(cfg["input"]["coefficients"])
    mkpath(output)
    rules = ("tm5_trapezoid", "gauss_fourier")
    isempty(isolated) || isolated in rules || error("isolated arm must be one of $rules")
    schedule = isempty(isolated) ? [("warmup", r, 0) for r in rules] : Tuple{String,String,Int}[]
    if isempty(isolated)
        append!(schedule, [("measured", r, rep) for rep in 1:2 for r in rules])
    else
        push!(schedule, ("isolated", isolated, 1))
    end
    open(joinpath(output, "timings.tsv"), "w") do io
        println(io, "phase\trule\trepetition\twall_seconds\tallocated_bytes\tpeak_rss_bytes\trss_scope\tjulia_threads\tblas_threads\tfftw_global_threads\tfield_product_fft_threads\tband_workers")
        for (phase, rule, rep) in schedule
            table["quadrature_refinement_steps"] = 2
            table["face_quadrature"] = rule
            label = "$(phase)_$(rule)_$(rep)"
            cfg["output"]["directory"] = joinpath(abspath(output), label)
            path = joinpath(output, "$label.toml")
            open(f -> TOML.print(f, cfg), path, "w")
            GC.gc()
            timing = @timed Base.invokelatest(main, [path, "--day", day])
            seconds = timing.time
            scope = isempty(isolated) ? "process_lifetime" : "isolated_arm"
            println(io, phase, '\t', rule, '\t', rep, '\t', seconds, '\t', timing.bytes, '\t', Sys.maxrss(), '\t', scope, '\t', Threads.nthreads(), '\t',
                BLAS.get_num_threads(), '\t', FFTW.get_num_threads(), '\t',
                rule == "gauss_fourier" ? PP.Spectral.cpu_band_fft_threads() : FFTW.get_num_threads(), '\t',
                rule == "gauss_fourier" ? PP.Spectral.cpu_band_workers() : 1)
            flush(io)
            @info "Quadrature window complete" phase rule rep seconds
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (3, 4) || error("usage: CONFIG DAY OUTPUT [ISOLATED_RULE]")
    install_one_window_driver()
    benchmark_refinement(ARGS...)
end
