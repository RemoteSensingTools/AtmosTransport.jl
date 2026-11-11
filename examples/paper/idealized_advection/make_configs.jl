#!/usr/bin/env julia
#
# Write one run config per grid × flow × scheme × precision of `cases.toml`:
#
#   julia --project=. examples/paper/idealized_advection/make_configs.jl [--workdir DIR] [--cases FILE]
#
# Reads <workdir>/inputs/<grid>_<flow>/case.toml (written by generate.jl) and writes
# <workdir>/runs/<grid>_<flow>_<scheme>_<f32|f64>/config.toml plus <workdir>/runs/runs.txt,
# the list of configs for the driver. The work directory defaults to
# $IDEALIZED_ADVECTION_WORKDIR.

using ArgParse
using TOML

function parse_arguments()
    s = ArgParseSettings(description = "Write run configs for the idealized advection cases.")
    @add_arg_table! s begin
        "--workdir"
            default = get(ENV, "IDEALIZED_ADVECTION_WORKDIR", "")
        "--cases"
            default = joinpath(@__DIR__, "cases.toml")
        "--gpu-precisions"
            help = "comma-separated precisions that run on a CUDA GPU (e.g. Float32; default " *
                   "\$IDEALIZED_ADVECTION_GPU_PRECISIONS, else none); a `gpu_precisions` entry of a " *
                   "grid in cases.toml overrides it for that grid"
            default = get(ENV, "IDEALIZED_ADVECTION_GPU_PRECISIONS", "")
    end
    return parse_args(s)
end

precision_tag(p) = p == "Float32" ? "f32" : "f64"

function tracer_init(topology, ic_file, name)
    kind = topology == "cubed_sphere" ? "cs_native" : "file"
    return Dict{String, Any}("init" => Dict{String, Any}("kind" => kind, "file" => ic_file, "variable" => name,
                                                         "vertical_order" => "toa_first"))
end

function run_config(case, scheme, precision, outdir, matrix; use_gpu::Bool)
    hours = collect(0:matrix["output_cadence_hours"]:Int(case["period_hours"]))
    advection = Dict{String, Any}(scheme)
    cfg = Dict{String, Any}(
        "input" => Dict{String, Any}("binary_paths" => case["binaries"], "start_date" => case["start_date"],
                                     "validate_replay" => true),
        "architecture" => use_gpu ? Dict{String, Any}("use_gpu" => true, "backend" => "cuda") :
                                    Dict{String, Any}("use_gpu" => false, "backend" => "cpu"),
        "numerics" => Dict{String, Any}("float_type" => precision),
        "advection" => advection,
        "diffusion" => Dict{String, Any}("kind" => "none"),
        "convection" => Dict{String, Any}("kind" => "none"),
        "tracers" => Dict{String, Any}(name => tracer_init(case["topology"], case["initial_conditions"], name)
                                       for name in case["tracers"]),
        "output" => Dict{String, Any}(
            "path" => joinpath(outdir, "output.nc"), "hours" => hours,
            "fields" => Dict{String, Any}("layers" => "selected", "levels" => matrix["output_levels"],
                                          "column_mean" => false, "column_mass_per_area" => false,
                                          "air_mass_layers" => "selected", "air_mass" => true,
                                          "air_mass_per_area" => false, "column_air_mass_per_area" => false)),
        "logging" => Dict{String, Any}("progress" => "lines", "progress_interval_windows" => 24))
    return cfg
end

function main()
    args = parse_arguments()
    workdir = args["workdir"]
    isempty(workdir) && error("set --workdir or IDEALIZED_ADVECTION_WORKDIR")
    matrix = TOML.parsefile(args["cases"])
    configs = String[]
    gpu_precisions = filter(!isempty, strip.(split(args["gpu-precisions"], ",")))
    for grid in matrix["grid"], flow in grid["flows"]
        tag = get(grid, "tag", "")
        name = isempty(tag) ? "$(grid["label"])_$(flow)" : "$(grid["label"])_$(flow)_$(tag)"
        case_file = joinpath(workdir, "inputs", name, "case.toml")
        isfile(case_file) || (@warn "missing $case_file (run generate.jl first); skipped"; continue)
        case = TOML.parsefile(case_file)
        for scheme_name in grid["schemes"], precision in get(grid, "precisions", matrix["precisions"])
            run_id = "$(case["case"])_$(scheme_name)_$(precision_tag(precision))"
            outdir = mkpath(joinpath(workdir, "runs", run_id))
            use_gpu = precision in get(grid, "gpu_precisions", gpu_precisions)
            cfg = run_config(case, matrix["schemes"][scheme_name], precision, outdir, matrix; use_gpu)
            path = joinpath(outdir, "config.toml")
            open(io -> TOML.print(io, cfg), path, "w")
            run_meta = Dict("run_id" => run_id, "case" => case["case"], "grid" => case["grid"],
                            "flow" => case["flow"], "scheme" => scheme_name, "precision" => precision,
                            "tag" => tag, "courant" => get(case, "courant", 0.5),
                            "windows_per_day" => get(case, "windows_per_day", 24), "case_file" => case_file,
                            "device" => use_gpu ? "gpu" : "cpu")
            open(io -> TOML.print(io, run_meta), joinpath(outdir, "run.toml"), "w")
            push!(configs, path)
        end
    end
    list = joinpath(workdir, "runs", "runs.txt")
    mkpath(dirname(list))
    open(io -> foreach(c -> println(io, c), configs), list, "w")
    println("wrote $(length(configs)) configs; list in $list")
end

main()
