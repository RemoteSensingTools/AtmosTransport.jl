#!/usr/bin/env julia
# Compare two current preprocessor runs on one TOML config for byte stability.

using Dates
using JSON3
using Logging
using Printf
using TOML

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing

const DEFAULT_IGNORED_HEADER_KEYS = Set{Symbol}([:creation_time])
const COMPARE_CHUNK_BYTES = 64 * 1024 * 1024

function _usage()
    return """
    Usage:
      julia --project=. scripts/diagnostics/compare_preprocessors.jl CONFIG.toml --day YYYY-MM-DD [options]

    Options:
      --start YYYY-MM-DD       Start date for native-source ranges.
      --end YYYY-MM-DD         End date for native-source ranges.
      --workdir DIR            Directory for baseline/candidate outputs.
      --keep                   Keep the generated comparison directory.
      --ignore-header-key KEY  Ignore an additional top-level JSON header key.
      --warn-only-positivity   Set [numerics].require_substep_positivity=false
                               for both runs.

    The script runs the config twice with separate output directories:
    baseline  -> first current-code run
    candidate -> second current-code run

    Historical legacy-vs-unified cutover comparisons must be run from the
    relevant pre-cutover commit. After Plan 41 P4, the TOML-level unified
    switch no longer exists in production code.

    It reports exact byte equality when possible. If bytes differ, it parses
    the fixed JSON header, ignores volatile header keys such as creation_time,
    and requires both normalized header equality and payload byte equality.
    """
end

function _parse_args(args)
    isempty(args) && error(_usage())
    cfg_path = abspath(expanduser(args[1]))
    isfile(cfg_path) || error("Config not found: $cfg_path")

    day = nothing
    start_date = nothing
    end_date = nothing
    workdir = nothing
    keep = false
    warn_only_positivity = false
    ignored = copy(DEFAULT_IGNORED_HEADER_KEYS)

    i = 2
    while i <= length(args)
        arg = args[i]
        if arg == "--day" && i + 1 <= length(args)
            day = args[i + 1]
            i += 2
        elseif arg == "--start" && i + 1 <= length(args)
            start_date = args[i + 1]
            i += 2
        elseif arg == "--end" && i + 1 <= length(args)
            end_date = args[i + 1]
            i += 2
        elseif arg == "--workdir" && i + 1 <= length(args)
            workdir = abspath(expanduser(args[i + 1]))
            i += 2
        elseif arg == "--ignore-header-key" && i + 1 <= length(args)
            push!(ignored, Symbol(args[i + 1]))
            i += 2
        elseif arg == "--keep"
            keep = true
            i += 1
        elseif arg == "--warn-only-positivity"
            warn_only_positivity = true
            i += 1
        elseif arg == "--help" || arg == "-h"
            println(_usage())
            exit(0)
        else
            error("Unknown or incomplete argument `$arg`.\n$(_usage())")
        end
    end

    day === nothing && (start_date === nothing || end_date === nothing) &&
        error("Provide either `--day YYYY-MM-DD` or `--start/--end`.")

    return (cfg_path = cfg_path,
            day = day,
            start_date = start_date,
            end_date = end_date,
            workdir = workdir,
            keep = keep,
            warn_only_positivity = warn_only_positivity,
            ignored = ignored)
end

function _ensure_table!(cfg::Dict{String, Any}, key::String)
    table = get!(cfg, key) do
        Dict{String, Any}()
    end
    table isa Dict || error("`[$key]` must be a TOML table.")
    return table
end

function _prepared_cfg(base_cfg,
                       out_dir::AbstractString,
                       warn_only_positivity::Bool)
    cfg = deepcopy(base_cfg)
    output = _ensure_table!(cfg, "output")
    output["directory"] = String(out_dir)
    if warn_only_positivity
        numerics = _ensure_table!(cfg, "numerics")
        numerics["require_substep_positivity"] = false
    end
    return cfg
end

function _run_preprocessor!(cfg, parsed)
    return process_day(cfg;
                       day_override = parsed.day,
                       start_date = parsed.start_date,
                       end_date = parsed.end_date)
end

function _collect_files(root::AbstractString)
    isdir(root) || error("Output directory not created: $root")
    rels = String[]
    for (dir, _dirs, files) in walkdir(root)
        for file in files
            path = joinpath(dir, file)
            isfile(path) || continue
            push!(rels, relpath(path, root))
        end
    end
    sort!(rels)
    isempty(rels) && error("No output files found under $root")
    return rels
end

function _json_to_julia(x)
    if x isa JSON3.Object
        return Dict{Symbol, Any}(Symbol(k) => _json_to_julia(v)
                                 for (k, v) in pairs(x))
    elseif x isa JSON3.Array
        return [_json_to_julia(v) for v in x]
    else
        return x
    end
end

function _read_header(path::AbstractString)
    prefix = UInt8[]
    result = nothing
    open(path, "r") do io
        while !eof(io) && result === nothing
            append!(prefix, read(io, min(COMPARE_CHUNK_BYTES, 1024 * 1024)))
            nul = findfirst(==(0x00), prefix)
            if nul !== nothing
                header = _json_to_julia(JSON3.read(String(prefix[1:nul - 1])))
                header_bytes = Int(get(header, :header_bytes,
                                       AtmosTransport.Preprocessing.HEADER_SIZE))
                header_bytes <= filesize(path) ||
                    error("Header declares header_bytes=$header_bytes, " *
                          "but file has only $(filesize(path)) bytes: $path")
                result = (header, header_bytes)
            end
            length(prefix) <= 4 * 1024 * 1024 ||
                error("Could not find NUL-padded JSON header boundary in first 4 MiB: $path")
        end
    end
    result === nothing &&
        error("Could not find NUL-padded JSON header boundary in empty/truncated file: $path")
    return result
end

function _files_equal(a::AbstractString,
                      b::AbstractString;
                      offset_a::Integer = 0,
                      offset_b::Integer = 0,
                      chunk_bytes::Integer = COMPARE_CHUNK_BYTES)
    size_a = filesize(a) - Int(offset_a)
    size_b = filesize(b) - Int(offset_b)
    (size_a >= 0 && size_b >= 0) || return false
    size_a == size_b || return false
    equal = true
    open(a, "r") do io_a
        open(b, "r") do io_b
            seek(io_a, Int(offset_a))
            seek(io_b, Int(offset_b))
            remaining = size_a
            while remaining > 0
                n = min(Int(chunk_bytes), remaining)
                if read(io_a, n) != read(io_b, n)
                    equal = false
                    break
                end
                remaining -= n
            end
        end
    end
    return equal
end

function _read_header_and_payload(bytes::Vector{UInt8})
    nul = findfirst(==(0x00), bytes)
    nul === nothing && error("Could not find NUL-padded JSON header boundary.")
    header = _json_to_julia(JSON3.read(String(bytes[1:nul - 1])))
    header_bytes = Int(get(header, :header_bytes,
                           AtmosTransport.Preprocessing.HEADER_SIZE))
    header_bytes <= length(bytes) ||
        error("Header declares header_bytes=$header_bytes, but file has only $(length(bytes)) bytes.")
    return header, @view(bytes[header_bytes + 1:end])
end

function _normalize_header!(header::Dict{Symbol, Any}, ignored::Set{Symbol})
    for key in ignored
        delete!(header, key)
    end
    return header
end

function _changed_header_keys(a::Dict{Symbol, Any}, b::Dict{Symbol, Any})
    keys_all = collect(union(keys(a), keys(b)))
    sort!(keys_all; by=String)
    return [k for k in keys_all if get(a, k, nothing) != get(b, k, nothing)]
end

function _compare_file_pair(baseline_path::AbstractString,
                            candidate_path::AbstractString,
                            ignored::Set{Symbol})
    baseline_size = filesize(baseline_path)
    candidate_size = filesize(candidate_path)
    if baseline_size == candidate_size && _files_equal(baseline_path, candidate_path)
        return (status = :exact,
                baseline_size = baseline_size,
                candidate_size = candidate_size,
                changed_header_keys = Symbol[])
    end

    baseline_header, baseline_header_bytes = _read_header(baseline_path)
    candidate_header, candidate_header_bytes = _read_header(candidate_path)
    _normalize_header!(baseline_header, ignored)
    _normalize_header!(candidate_header, ignored)

    headers_equal = baseline_header == candidate_header
    payload_equal = _files_equal(baseline_path, candidate_path;
                                 offset_a = baseline_header_bytes,
                                 offset_b = candidate_header_bytes)
    status = headers_equal && payload_equal ? :header_normalized :
             payload_equal ? :header_mismatch :
             :payload_mismatch
    return (status = status,
            baseline_size = baseline_size,
            candidate_size = candidate_size,
            changed_header_keys = headers_equal ? Symbol[] :
                _changed_header_keys(baseline_header, candidate_header))
end

function _compare_outputs(baseline_dir::AbstractString,
                          candidate_dir::AbstractString,
                          ignored::Set{Symbol})
    baseline_files = _collect_files(baseline_dir)
    candidate_files = _collect_files(candidate_dir)
    baseline_files == candidate_files || error(
        "Output file sets differ.\nbaseline:  $(baseline_files)\ncandidate: $(candidate_files)")

    results = Pair{String, Any}[]
    ok = true
    for rel in baseline_files
        result = _compare_file_pair(joinpath(baseline_dir, rel),
                                    joinpath(candidate_dir, rel),
                                    ignored)
        push!(results, rel => result)
        ok &= result.status in (:exact, :header_normalized)
    end
    return ok, results
end

function main(args = ARGS)
    parsed = _parse_args(args)
    base_cfg = TOML.parsefile(parsed.cfg_path)
    workdir = parsed.workdir === nothing ?
        mktempdir(; prefix = "preprocessor_compare_") :
        parsed.workdir
    baseline_dir = joinpath(workdir, "baseline")
    candidate_dir = joinpath(workdir, "candidate")

    generated_workdir = parsed.workdir === nothing
    keep = parsed.keep || !generated_workdir
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited = false)
    global_logger(AtmosTransport.Preprocessing.FlushingLogger(base_logger))

    try
        rm(baseline_dir; recursive = true, force = true)
        rm(candidate_dir; recursive = true, force = true)
        mkpath(baseline_dir)
        mkpath(candidate_dir)

        @info "Running baseline preprocessor" config = parsed.cfg_path output = baseline_dir
        _run_preprocessor!(_prepared_cfg(base_cfg, baseline_dir,
                                         parsed.warn_only_positivity), parsed)

        @info "Running candidate preprocessor" config = parsed.cfg_path output = candidate_dir
        _run_preprocessor!(_prepared_cfg(base_cfg, candidate_dir,
                                         parsed.warn_only_positivity), parsed)

        ok, results = _compare_outputs(baseline_dir, candidate_dir, parsed.ignored)
        println()
        println("Preprocessor reproducibility comparison: ", ok ? "PASS" : "FAIL")
        println("workdir: ", workdir)
        for (rel, result) in results
            changed = isempty(result.changed_header_keys) ? "" :
                " changed_header_keys=$(join(String.(result.changed_header_keys), ","))"
            println(@sprintf("  %-36s %-18s baseline=%d candidate=%d%s",
                             rel, String(result.status),
                             result.baseline_size, result.candidate_size, changed))
        end
        ok || exit(1)
    finally
        if !keep
            rm(workdir; recursive = true, force = true)
        else
            @info "Kept comparison outputs" workdir
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
