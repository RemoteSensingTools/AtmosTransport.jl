#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# scripts/run_transport.jl — unified driven transport runner (plan 40 Commit 6c).
#
# ONE CLI. Reads a TOML config, opens the first transport binary,
# dispatches on its `grid_type` header field (`:latlon`,
# `:reduced_gaussian`, `:cubed_sphere`), and runs the loop. All
# topology-specific logic (IC pipeline, surface flux, snapshot output,
# GPU residency check, capability validation) lives in
# `src/Models/DrivenRunner.jl` and dispatches on mesh type via
# multiple dispatch.
#
# The TOML `[input]` block accepts either shape:
#   [input]
#   binary_paths = [ "a.bin", "b.bin" ]
# OR
#   [input]
#   folder       = "~/data/.../"
#   start_date   = "YYYY-MM-DD"
#   end_date     = "YYYY-MM-DD"
#   file_pattern = "<prefix>{YYYYMMDD}<suffix>"   # optional
#
# Usage:
#   julia --project=. scripts/run_transport.jl <config.toml>
#
# From Julia, include this file and call `main([config_path])`: it runs the
# config in the current session and throws on error (the golden harness,
# test/golden/golden_worker.jl, runs several configs in one session this way).
# The thread re-exec and the GPU preload below happen only when the file is
# run as a script.
# ---------------------------------------------------------------------------

using Logging
using TOML

# Default to 2 Julia threads. Per the GPU-profiling work on 2026-05-28, going
# from --threads=1 → --threads=2 cut the 3-day C180 full-physics wall by 34 %
# (130 s → 86 s): NetCDF/PCIe write overlaps GPU compute on the background
# thread, and host-side launch dispatch pipelines across threads. Users who
# want a different count can set JULIA_NUM_THREADS or pass --threads N
# explicitly; setting ATMOSTR_NO_AUTO_THREADS=1 disables the re-exec.
if (abspath(PROGRAM_FILE) == @__FILE__) &&
   Threads.nthreads() == 1 &&
   get(ENV, "JULIA_NUM_THREADS", "") == "" &&
   get(ENV, "ATMOSTR_NO_AUTO_THREADS", "") != "1"
    @info "Re-executing with --threads=2 (set ATMOSTR_NO_AUTO_THREADS=1 to disable)"
    julia = Base.julia_cmd()
    base = filter(a -> !startswith(String(a), "--threads") &&
                       !startswith(String(a), "--project"), julia.exec)
    # `Base.julia_cmd()` does NOT carry `--project`, so the re-exec would land in
    # the default environment and fail to find the project deps (e.g.
    # KernelAbstractions). Propagate the active project explicitly.
    proj = Base.active_project()
    proj_flag = proj === nothing ? String[] : ["--project=$(dirname(proj))"]
    new_cmd = Cmd([base; proj_flag; "--threads=2"; PROGRAM_FILE; ARGS...])
    run(new_cmd; wait = true)
    exit(0)
end

# Preload the GPU backend BEFORE `AtmosTransport` is imported so the
# CLI stack compiles in a single world age. The library entry point also has a
# one-time world-age trampoline for callers that load a backend on demand, but
# resolving it here keeps CLI startup and optional profiling straightforward.
# Inspecting the config costs about 1 ms.
if (abspath(PROGRAM_FILE) == @__FILE__) && !isempty(ARGS)
    _cfg_path = expanduser(ARGS[1])
    if isfile(_cfg_path)
        _cfg = try TOML.parsefile(_cfg_path) catch; nothing end
        if _cfg !== nothing
            _arch = get(_cfg, "architecture", Dict{String, Any}())
            _arch isa AbstractDict || throw(ArgumentError(
                "[architecture] must be a TOML table; got $(typeof(_arch))."))
            _use_gpu_raw = get(_arch, "use_gpu", false)
            _use_gpu_raw isa Bool || throw(ArgumentError(
                "[architecture].use_gpu must be true or false; got $(repr(_use_gpu_raw))"))
            _use_gpu = _use_gpu_raw
            _backend = lowercase(String(get(_arch, "backend",
                                            _use_gpu ? "auto" : "cpu")))
            _backend = replace(_backend, '-' => '_', ' ' => '_')
            _gpu_requested = _use_gpu ||
                             _backend in ("auto", "gpu", "cuda", "nvidia",
                                          "metal", "apple", "apple_metal")
            if _gpu_requested
                if _backend in ("metal", "apple", "apple_metal") ||
                   (_backend in ("auto", "gpu") && Sys.isapple())
                    @info "Preloading Metal (GPU backend)"
                    using Metal
                elseif _backend in ("cuda", "nvidia") ||
                       _backend in ("auto", "gpu")
                    @info "Preloading CUDA (GPU backend)"
                    using CUDA
                end
            end
        end
    end
end

using AtmosTransport

function _print_help(io::IO = stdout)
    println(io, "Usage:")
    println(io, "  julia --project=. scripts/run_transport.jl <config.toml>")
    println(io)
    println(io, "Canonical runtime entry point for AtmosTransport run TOMLs.")
    println(io, "Configs may use ~/..., \$ATMOSTRANSPORT_DATA_ROOT/..., or other")
    println(io, "environment-variable paths.")
    println(io)
    println(io, "Example:")
    println(io, "  julia --project=. examples/generate_synthetic_quickstart.jl")
    println(io, "  julia --project=. scripts/run_transport.jl config/examples/minimal_template.toml")
    return nothing
end

function _parse_cli(args::Vector{String})
    if args == ["-h"] || args == ["--help"]
        return nothing
    end
    isempty(args) && throw(ArgumentError("missing required <config.toml>"))
    length(args) == 1 || throw(ArgumentError(
        "expected exactly one <config.toml>; got $(length(args)) arguments"))
    startswith(args[1], "-") && throw(ArgumentError("unknown option: $(args[1])"))
    cfg_path = expanduser(args[1])
    isfile(cfg_path) || throw(ArgumentError("Config not found: $cfg_path"))
    return cfg_path
end

"""
    main(args = ARGS)

Run the config named in `args` (or print the help for `-h`/`--help`) with
console logging to the current `stderr`; return the terminal model and throw
on error. The logger is scoped to the call, so a session that runs several
configs keeps its own.
"""
function main(args::Vector{String} = ARGS)
    cfg_path = _parse_cli(args)
    if cfg_path === nothing
        _print_help(stdout)
        return nothing
    end
    cfg = TOML.parsefile(cfg_path)
    return with_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false)) do
        _run_with_optional_profiling(cfg)
    end
end

# GPU profiling brackets, gated on ATMOSTR_PROFILE_MODE.
#   "full"   — enable CUDA profiling for the entire run.
#   "window" — spawn an async timer that calls CUDA.Profile.start() after
#              ATMOSTR_PROFILE_WARMUP_SEC and CUDA.Profile.stop() +
#              process exit after ATMOSTR_PROFILE_DUR_SEC. Used with
#              `nsys profile -c cudaProfilerApi --capture-range-end=stop`.
#   ""       — unchanged.
# Whether a run is profiled depends on its config's [architecture] alone, not
# on whether CUDA happens to be loaded: a Julia session that includes this file
# (no preload) loads CUDA here on its first profiled GPU run.
const _CUDA_PKGID = Base.PkgId(
    Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")

_loaded_cuda_module() = get(Base.loaded_modules, _CUDA_PKGID, nothing)

"""
    _configured_backend(cfg) -> Symbol

The backend the config's `[architecture]` table selects (`:cpu`, `:cuda` or
`:metal`), as the runner resolves it; `backend = "auto"` (or `use_gpu = true`
without a backend) detects the GPU, which loads its package.
"""
_configured_backend(cfg) = AtmosTransport.Architectures.backend_name(
    AtmosTransport.Architectures.architecture_from_config(get(cfg, "architecture", Dict{String, Any}())))

"""
    _profile_action(mode, backend) -> Symbol

What `ATMOSTR_PROFILE_MODE = mode` does for a run on `backend` (from
`_configured_backend`): `:none` (no profile requested), `:full` or `:window`
(a CUDA profile), `:needs_cuda` (a profile requested for a run that does not
use CUDA) or `:unknown` (an unknown mode).
"""
function _profile_action(mode::AbstractString, backend::Symbol)
    mode = lowercase(mode)
    isempty(mode) && return :none
    mode in ("full", "window") || return :unknown
    return backend === :cuda ? Symbol(mode) : :needs_cuda
end

function _profile_full_run(CUDA, cfg)
    run = () -> run_driven_simulation(cfg)
    if CUDA.Profile.detect_cupti()
        @info "CUDA session is already externally profiled; using external collection"
        return CUDA.Profile.profile_externally(run)
    end
    return CUDA.Profile.profile_internally(run)
end

function _profile_window(CUDA, warmup, dur)
    sleep(warmup)
    @info "Profile window: CUDA.Profile.start()"
    CUDA.Profile.start()
    sleep(dur)
    @info "Profile window: CUDA.Profile.stop() and process exit"
    CUDA.Profile.stop()
    exit(0)
end

function _run_with_optional_profiling(cfg)
    mode = get(ENV, "ATMOSTR_PROFILE_MODE", "")
    isempty(mode) && return run_driven_simulation(cfg)
    backend = _configured_backend(cfg)
    action = _profile_action(mode, backend)
    if action === :unknown
        @warn "Unknown ATMOSTR_PROFILE_MODE=$(mode); running without profile bracket"
        return run_driven_simulation(cfg)
    elseif action === :needs_cuda
        @warn "ATMOSTR_PROFILE_MODE=$(mode) profiles CUDA runs, but [architecture] selects " *
              "$(backend); running without profile bracket"
        return run_driven_simulation(cfg)
    end
    AtmosTransport.Architectures.ensure_runtime!(GPU(:cuda))   # loads CUDA if nothing preloaded it
    CUDA = _loaded_cuda_module()
    # CUDA may be newer than the world this call started in: call into it in the latest world.
    if action === :full
        @info "ATMOSTR_PROFILE_MODE=full → CUDA activity profile of full run"
        return Base.invokelatest(_profile_full_run, CUDA, cfg)
    end
    warmup = parse(Float64, get(ENV, "ATMOSTR_PROFILE_WARMUP_SEC", "120"))
    dur    = parse(Float64, get(ENV, "ATMOSTR_PROFILE_DUR_SEC", "60"))
    @info "ATMOSTR_PROFILE_MODE=window → start after $(warmup)s, stop after +$(dur)s, then exit"
    Threads.@spawn Base.invokelatest(_profile_window, CUDA, warmup, dur)
    return run_driven_simulation(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
