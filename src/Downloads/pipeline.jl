# ===========================================================================
# Download pipeline — top-level entry point
#
# Mirrors the Preprocessing pattern:
#   parse config → build source/protocol → iterate dates → execute tasks
# ===========================================================================

# Dates and Printf are available from the parent module scope (Downloads.jl)

"""
    download_data!(cfg::Dict{String,Any}; start_date=nothing, end_date=nothing,
                   dry_run=false, verify_only=false)

Main entry point. Parses TOML config and executes downloads.

- `dry_run`: report what would be downloaded without executing
- `verify_only`: check existing files against expected sizes
"""
function download_data!(cfg::Dict{String, Any};
                        start_date::Union{Date, Nothing}=nothing,
                        end_date::Union{Date, Nothing}=nothing,
                        dry_run::Bool=false,
                        verify_only::Bool=false)
    settings = download_logging(cfg)
    return with_log_level(settings.level) do
        try
            result = _download_data!(cfg; start_date, end_date, dry_run, verify_only,
                                     reporter=settings.progress)
            _finish_progress!(settings.progress)
            result
        catch
            _cancel_progress!(settings.progress)
            rethrow()
        end
    end
end

function _download_data!(cfg::Dict{String, Any};
                         start_date::Union{Date, Nothing}=nothing,
                         end_date::Union{Date, Nothing}=nothing,
                         dry_run::Bool=false,
                         verify_only::Bool=false,
                         reporter::AbstractDownloadProgressReporter=DownloadProgressLines())
    config = parse_download_config(cfg)
    dry_run || verify_only || protocol_can_download(config.protocol) || throw(ArgumentError(
        "$(nameof(typeof(config.protocol))) downloads are not implemented; run this recipe " *
        "with --dry-run or --verify, and stage the files with NASA Earthdata credentials " *
        "(docs/src/config/data_sources.md)."))

    # Override dates from CLI if provided
    sched = if !isnothing(start_date) || !isnothing(end_date)
        ScheduleConfig(
            something(start_date, config.schedule.start_date),
            something(end_date, config.schedule.end_date),
            config.schedule.chunk,
            config.schedule.max_concurrent,
            config.schedule.max_retries,
            config.schedule.retry_wait_seconds,
            config.schedule.skip_existing,
        )
    else
        config.schedule
    end

    _log_download_start(config, sched, dry_run, verify_only)

    # Group dates by chunk strategy
    date_groups = _group_dates(sched)

    out_dir = canonical_output_dir(config.output)
    mkpath(out_dir)

    n_total = length(date_groups)
    n_ok = 0
    n_skip = 0
    n_fail = 0
    total_bytes = 0

    for (i, group) in enumerate(date_groups)
        label = _group_label(group, sched.chunk)
        _report_progress!(reporter, "[$i/$n_total] $label")

        tasks = build_tasks(config.source, config.protocol, group,
                            config.output, config.requests)

        if dry_run
            _report_dry_run(reporter, tasks)
            n_ok += length(tasks)
            continue
        end

        if verify_only
            for task in tasks
                status, expected = _existing_task_status(task, config.protocol)
                if status === :verified
                    sz = filesize(task.dest_path)
                    _report_progress!(reporter,
                                      "  ✓ $(basename(task.dest_path)) ($(sz ÷ 1_000_000) MB)")
                    n_ok += 1
                    total_bytes += sz
                elseif status === :unverifiable
                    @warn "  ? UNVERIFIABLE (remote size unavailable): $(task.dest_path)"
                    n_fail += 1
                elseif status === :corrupt
                    @warn "  ✗ SIZE MISMATCH: $(task.dest_path) " *
                          "($(filesize(task.dest_path)) bytes, expected $(expected))"
                    n_fail += 1
                else
                    @warn "  ✗ MISSING: $(task.dest_path)"
                    n_fail += 1
                end
            end
            continue
        end

        for task in tasks
            if sched.skip_existing
                status, _ = _existing_task_status(task, config.protocol)
                if status === :verified
                    _report_progress!(reporter,
                                      "  Skip (verified): $(basename(task.dest_path))")
                    n_skip += 1
                    total_bytes += filesize(task.dest_path)
                    continue
                elseif status === :unverifiable
                    @warn "  Skip (legacy file has no verification metadata): $(basename(task.dest_path)). " *
                          "A future successful download will create a checksum sidecar."
                    n_skip += 1
                    total_bytes += filesize(task.dest_path)
                    continue
                end
            end

            success = execute!(task, config.protocol;
                               max_retries=sched.max_retries,
                               retry_wait=sched.retry_wait_seconds,
                               reporter)
            if success
                _write_download_manifest(task, config.protocol)
                n_ok += 1
                total_bytes += isfile(task.dest_path) ? filesize(task.dest_path) : 0
            else
                n_fail += 1
            end
        end
    end

    _log_download_summary(n_ok, n_skip, n_fail, total_bytes, dry_run, verify_only)
    return nothing
end

_task_remote_url(task::DownloadTask, ::HTTPProtocol) = task.source_url
_task_remote_url(task::DownloadTask, proto::S3Protocol) =
    proto.no_sign_request ? "https://$(proto.bucket).s3.amazonaws.com/$(task.source_url)" : nothing
_task_remote_url(::DownloadTask, ::AbstractDownloadProtocol) = nothing

function _existing_task_status(task::DownloadTask, protocol::AbstractDownloadProtocol)
    isfile(task.dest_path) || return (:missing, 0)
    filesize(task.dest_path) > 0 || return (:corrupt, 1)
    manifest_path = _download_manifest_path(task)
    isfile(manifest_path) && return _verify_download_manifest(task, protocol)
    url = _task_remote_url(task, protocol)
    if url !== nothing
        expected = _get_content_length(url)
        expected > 0 && return (
            filesize(task.dest_path) == expected ? :verified : :corrupt, expected)
    end
    return _verify_download_manifest(task, protocol)
end

_download_manifest_path(task::DownloadTask) = task.dest_path * ".download.toml"

_download_protocol_identity(::CDSProtocol) = (kind="cds",)
_download_protocol_identity(::MARSProtocol) = (kind="mars",)
_download_protocol_identity(protocol::HTTPProtocol) =
    (kind="http", base_url=protocol.base_url)
_download_protocol_identity(protocol::S3Protocol) =
    (kind="s3", bucket=protocol.bucket, prefix=protocol.prefix)
_download_protocol_identity(protocol::OPeNDAPProtocol) =
    (kind="opendap", base_url=protocol.base_url)
_download_protocol_identity(protocol::GCSProtocol) =
    (kind="gcs", bucket_base=protocol.bucket_base)

function _download_task_identity(task::DownloadTask, protocol::AbstractDownloadProtocol)
    request_items = sort!(collect(task.request); by=first)
    return repr((protocol=_download_protocol_identity(protocol),
                 source_url=task.source_url, request=request_items))
end

function _file_sha256(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _write_download_manifest(task::DownloadTask,
                                  protocol::AbstractDownloadProtocol)
    manifest_path = _download_manifest_path(task)
    staging = manifest_path * ".tmp"
    manifest = Dict{String, Any}(
        "format_version" => 1,
        "task_identity" => _download_task_identity(task, protocol),
        "size_bytes" => filesize(task.dest_path),
        "sha256" => _file_sha256(task.dest_path),
    )
    rm(staging; force=true)
    try
        open(staging, "w") do io
            TOML.print(io, manifest)
        end
        mv(staging, manifest_path; force=true)
    catch
        rm(staging; force=true)
        rethrow()
    end
    return manifest_path
end

function _verify_download_manifest(task::DownloadTask,
                                   protocol::AbstractDownloadProtocol)
    manifest_path = _download_manifest_path(task)
    isfile(manifest_path) || return (:unverifiable, 0)
    manifest = try
        TOML.parsefile(manifest_path)
    catch
        return (:corrupt, 0)
    end
    get(manifest, "format_version", nothing) == 1 || return (:corrupt, 0)
    get(manifest, "task_identity", nothing) == _download_task_identity(task, protocol) ||
        return (:corrupt, 0)
    expected_size = get(manifest, "size_bytes", nothing)
    expected_size isa Integer || return (:corrupt, 0)
    filesize(task.dest_path) == expected_size || return (:corrupt, Int(expected_size))
    expected_sha = get(manifest, "sha256", nothing)
    expected_sha isa String || return (:corrupt, Int(expected_size))
    return _file_sha256(task.dest_path) == expected_sha ?
           (:verified, Int(expected_size)) : (:corrupt, Int(expected_size))
end

# ---------------------------------------------------------------------------
# Date grouping by chunk strategy
# ---------------------------------------------------------------------------

"""
    _group_dates(sched::ScheduleConfig) -> Vector{Vector{Date}}

Group the date range by chunk strategy.
"""
function _group_dates(sched::ScheduleConfig)
    all_dates = collect(sched.start_date:Day(1):sched.end_date)

    if sched.chunk == :monthly
        # Group by (year, month) — dates are sorted, so consecutive grouping works
        groups = Vector{Date}[]
        for d in all_dates
            if isempty(groups) || (year(d), month(d)) != (year(groups[end][1]), month(groups[end][1]))
                push!(groups, Date[])
            end
            push!(groups[end], d)
        end
        return groups
    elseif sched.chunk in (:daily, :per_file)
        return [[d] for d in all_dates]
    else
        error("Unknown chunk strategy: $(sched.chunk). Use: monthly, daily, per_file")
    end
end

function _group_label(group::Vector{Date}, chunk::Symbol)
    if chunk == :monthly
        return Dates.format(group[1], "yyyy-mm") * " ($(length(group)) days)"
    else
        return Dates.format(group[1], "yyyy-mm-dd")
    end
end

# ---------------------------------------------------------------------------
# Task building — dispatches on source type
# ---------------------------------------------------------------------------

"""
    build_tasks(source, protocol, dates, output, requests) -> Vector{DownloadTask}

Build download tasks for a date group. Dispatches on source type.
"""
function build_tasks(source::AbstractDownloadSource,
                     protocol::AbstractDownloadProtocol,
                     dates::Vector{Date},
                     output::OutputConfig,
                     requests::Vector)
    error("build_tasks not implemented for $(typeof(source)) × $(typeof(protocol)). " *
          "Load the appropriate source module.")
end

# ---------------------------------------------------------------------------
# Task execution — dispatches on protocol type
# ---------------------------------------------------------------------------

"""
    execute!(task::DownloadTask, protocol; max_retries=3, retry_wait=30,
             reporter=DownloadProgressLines()) -> Bool

Execute a single download task. Dispatches on protocol type.
"""
function execute!(task::DownloadTask, protocol::AbstractDownloadProtocol;
                  max_retries::Int=3, retry_wait::Int=30,
                  reporter::AbstractDownloadProgressReporter=DownloadProgressLines())
    error("execute! not implemented for $(typeof(protocol)). " *
          "Load the appropriate protocol module.")
end

const _SUBPROCESS_READ_BYTES = 8 * 1024
const _SUBPROCESS_TAIL_BYTES = 64 * 1024

mutable struct _BoundedByteTail
    bytes::Vector{UInt8}
    limit::Int
end

_BoundedByteTail(limit::Int) = _BoundedByteTail(UInt8[], limit)

function _append_tail!(tail::_BoundedByteTail, chunk::AbstractVector{UInt8})
    isempty(chunk) && return tail
    if length(chunk) >= tail.limit
        empty!(tail.bytes)
        append!(tail.bytes, @view chunk[(end - tail.limit + 1):end])
        return tail
    end

    overflow = length(tail.bytes) + length(chunk) - tail.limit
    overflow > 0 && deleteat!(tail.bytes, 1:overflow)
    append!(tail.bytes, chunk)
    return tail
end

_tail_text(tail::_BoundedByteTail) = strip(String(copy(tail.bytes)))

function _report_subprocess_text!(reporter::AbstractDownloadProgressReporter,
                                  bytes::Vector{UInt8})
    text = strip(String(copy(bytes)))
    isempty(text) || _report_progress!(reporter, text)
    empty!(bytes)
    return nothing
end

function _append_progress!(reporter::AbstractDownloadProgressReporter,
                           pending::Vector{UInt8}, chunk::AbstractVector{UInt8})
    for byte in chunk
        if byte == UInt8('\n') || byte == UInt8('\r')
            _report_subprocess_text!(reporter, pending)
        else
            push!(pending, byte)
            # A subprocess can emit an arbitrarily long unterminated line.
            # Forward fixed-size fragments rather than buffering that line.
            length(pending) == _SUBPROCESS_READ_BYTES &&
                _report_subprocess_text!(reporter, pending)
        end
    end
    return nothing
end

function _drain_subprocess_pipe(pipe::Pipe,
                                reporter::AbstractDownloadProgressReporter)
    tail = _BoundedByteTail(_SUBPROCESS_TAIL_BYTES)
    pending = UInt8[]
    sizehint!(pending, _SUBPROCESS_READ_BYTES)
    while !eof(pipe)
        navailable = bytesavailable(pipe)
        navailable == 0 && continue
        chunk = read(pipe, min(navailable, _SUBPROCESS_READ_BYTES))
        _append_tail!(tail, chunk)
        _append_progress!(reporter, pending, chunk)
    end
    _report_subprocess_text!(reporter, pending)
    return _tail_text(tail)
end

struct _SubprocessFailure <: Exception
    label::String
    cause::ProcessFailedException
    stdout_tail::String
    stderr_tail::String
end

function Base.showerror(io::IO, failure::_SubprocessFailure)
    print(io, failure.label, " subprocess failed: ")
    showerror(io, failure.cause)
    isempty(failure.stdout_tail) ||
        print(io, "\nstdout tail:\n", failure.stdout_tail)
    isempty(failure.stderr_tail) ||
        print(io, "\nstderr tail:\n", failure.stderr_tail)
end

function _run_command(cmd::Cmd, reporter::AbstractDownloadProgressReporter;
                      label::AbstractString)
    stdout_pipe = Pipe()
    stderr_pipe = Pipe()
    process = run(pipeline(ignorestatus(cmd); stdout=stdout_pipe, stderr=stderr_pipe);
                  wait=false)
    # The parent must close its copies of the write ends so the drain tasks see
    # EOF as soon as the child exits.
    close(stdout_pipe.in)
    close(stderr_pipe.in)
    stdout_task = @async _drain_subprocess_pipe(stdout_pipe, reporter)
    stderr_task = @async _drain_subprocess_pipe(stderr_pipe, reporter)

    wait(process)
    stdout_tail = fetch(stdout_task)
    stderr_tail = fetch(stderr_task)
    success(process) || throw(_SubprocessFailure(
        String(label), ProcessFailedException(process), stdout_tail, stderr_tail))
    return nothing
end

# ---------------------------------------------------------------------------
# Retry scaffold (shared by S3, CDS, MARS)
# ---------------------------------------------------------------------------

"""
    _with_retries(f, label, dest; max_retries, retry_wait) -> Bool

Execute `f()` up to `max_retries` times. `f()` should return `true` on
success. On failure, cleans up `dest` and waits before retrying.
"""
function _with_retries(f::Function, label::String, dest::String;
                       max_retries::Int, retry_wait::Int)
    last_failure = nothing
    last_backtrace = nothing
    for attempt in 1:max_retries
        try
            f(attempt) && return true
            last_failure = nothing
            last_backtrace = nothing
        catch e
            last_failure = e
            last_backtrace = catch_backtrace()
            @warn "  Attempt $attempt failed: $e"
            isfile(dest) && rm(dest; force=true)
            attempt < max_retries && sleep(retry_wait)
        end
    end
    if last_failure isa _SubprocessFailure
        @error("  Failed after $max_retries attempts: $label",
               exception=(last_failure, last_backtrace),
               stdout_tail=last_failure.stdout_tail,
               stderr_tail=last_failure.stderr_tail)
    elseif last_failure !== nothing
        @error("  Failed after $max_retries attempts: $label",
               exception=(last_failure, last_backtrace))
    else
        @error "  Failed after $max_retries attempts: $label"
    end
    return false
end

# OPeNDAP protocol execution (Phase 4 — requires NCDatasets remote access)
function execute!(task::DownloadTask, proto::OPeNDAPProtocol;
                  max_retries::Int=3, retry_wait::Int=30,
                  reporter::AbstractDownloadProgressReporter=DownloadProgressLines())
    # TODO: implement OPeNDAP subset download via NCDatasets remote read
    # (`download_data!` refuses OPeNDAP recipes before any task runs).
    error("OPeNDAP downloads are not implemented (see `protocol_can_download`).")
end

# HTTP protocol execution
function execute!(task::DownloadTask, ::HTTPProtocol;
                  max_retries::Int=3, retry_wait::Int=30,
                  reporter::AbstractDownloadProgressReporter=DownloadProgressLines())
    mkpath(dirname(task.dest_path))
    return verified_download(task.source_url, task.dest_path;
                             max_retries=max_retries, reporter)
end

# S3 protocol execution — uses public HTTPS URL for no_sign_request buckets,
# falls back to aws CLI for authenticated buckets
function execute!(task::DownloadTask, proto::S3Protocol;
                  max_retries::Int=3, retry_wait::Int=30,
                  reporter::AbstractDownloadProgressReporter=DownloadProgressLines())
    mkpath(dirname(task.dest_path))
    if proto.no_sign_request
        # Public bucket: use HTTPS URL directly (no aws CLI needed)
        https_url = "https://$(proto.bucket).s3.amazonaws.com/$(task.source_url)"
        return verified_download(https_url, task.dest_path;
                                 max_retries=max_retries, reporter)
    else
        s3_url = "s3://$(proto.bucket)/$(task.source_url)"
        staging = task.dest_path * ".part"
        rm(staging; force=true)
        success = _with_retries(basename(task.dest_path), staging;
                             max_retries, retry_wait) do attempt
            _report_progress!(reporter,
                              "  Downloading $(basename(task.dest_path)) (attempt $attempt)...")
            _run_command(`aws s3 cp $s3_url $staging`, reporter; label="AWS S3")
            isfile(staging) && filesize(staging) > 0
        end
        success && mv(staging, task.dest_path; force=true)
        return success
    end
end

# CDS protocol execution
function execute!(task::DownloadTask, proto::CDSProtocol;
                  max_retries::Int=3, retry_wait::Int=30,
                  reporter::AbstractDownloadProgressReporter=DownloadProgressLines())
    mkpath(dirname(task.dest_path))
    dataset = get(task.request, "dataset", "reanalysis-era5-complete")
    request = Dict{String,Any}(k => v for (k, v) in task.request if k != "dataset")
    staging = task.dest_path * ".part"
    rm(staging; force=true)
    success = _with_retries(task.name, staging; max_retries, retry_wait) do attempt
        script = build_cds_retrieve_script(dataset, request, staging)
        _report_progress!(reporter,
                          "  CDS retrieve: $(task.name) (attempt $attempt)...")
        run_python(script, proto.python_env; label=task.name, reporter)
        isfile(staging) && filesize(staging) > 0
    end
    success && mv(staging, task.dest_path; force=true)
    return success
end

# MARS protocol execution (with CDS fallback)
function execute!(task::DownloadTask, proto::MARSProtocol;
                  max_retries::Int=3, retry_wait::Int=30,
                  reporter::AbstractDownloadProgressReporter=DownloadProgressLines())
    mkpath(dirname(task.dest_path))
    request = Dict{String,Any}(k => v for (k, v) in task.request if k != "dataset")
    staging = task.dest_path * ".part"
    rm(staging; force=true)
    success = _with_retries(task.name, staging; max_retries, retry_wait) do attempt
        script = build_mars_retrieve_script(request, staging)
        _report_progress!(reporter,
                          "  MARS retrieve: $(task.name) (attempt $attempt)...")
        run_python(script, proto.python_env; label=task.name, reporter)
        isfile(staging) && filesize(staging) > 0
    end
    success && mv(staging, task.dest_path; force=true)
    if !success && proto.fallback_to_cds
        @warn "  MARS failed, falling back to CDS for $(task.name)"
        return execute!(task, CDSProtocol(proto.python_env);
                        max_retries=max_retries, retry_wait=retry_wait, reporter)
    end
    return success
end

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

function _log_download_start(config::DownloadConfig, sched::ScheduleConfig,
                             dry_run::Bool, verify_only::Bool)
    mode = dry_run ? "dry-run" : verify_only ? "verify" : "download"
    out_dir = canonical_output_dir(config.output)
    n_days = Dates.value(sched.end_date - sched.start_date) + 1
    requests = [(name=get(req, "name", "unnamed"),
                 description=get(req, "description", "")) for req in config.requests]
    @info("AtmosTransport download started",
          mode=mode, source=source_name(config.source),
          start_date=sched.start_date, end_date=sched.end_date, days=n_days,
          chunk=sched.chunk, output=out_dir, requests=requests)
    return nothing
end

function _report_dry_run(reporter::AbstractDownloadProgressReporter,
                         tasks::Vector{DownloadTask})
    for task in tasks
        est = if task.estimated_size_mb >= 1000
            " (~$(@sprintf("%.0f", task.estimated_size_mb / 1000)) GB)"
        elseif task.estimated_size_mb > 0
            " (~$(@sprintf("%.0f", task.estimated_size_mb)) MB)"
        else
            ""
        end
        _report_progress!(reporter, "  → $(task.dest_path)$est")
    end
    return nothing
end

function _log_download_summary(n_ok, n_skip, n_fail, total_bytes, dry_run, verify_only)
    action = dry_run ? "Would download" : verify_only ? "Verified" : "Downloaded"
    @info("AtmosTransport download finished",
          action=action, succeeded=n_ok, skipped=n_skip, total_bytes=total_bytes)
    n_fail > 0 && @warn "AtmosTransport download failures" failed=n_fail
    return nothing
end

_report_progress!(::NoDownloadProgress, message::AbstractString) = nothing
function _report_progress!(::DownloadProgressLines, message::AbstractString)
    @info message
    return nothing
end
function _report_progress!(reporter::DownloadProgressBar, message::AbstractString)
    next!(reporter.progress; showvalues=[(:status, message)])
    return nothing
end

_finish_progress!(::Union{DownloadProgressLines, NoDownloadProgress}) = nothing
function _finish_progress!(reporter::DownloadProgressBar)
    finish!(reporter.progress)
    return nothing
end

_cancel_progress!(::Union{DownloadProgressLines, NoDownloadProgress}) = nothing
function _cancel_progress!(reporter::DownloadProgressBar)
    cancel(reporter.progress, "Download failed"; color=:red)
    return nothing
end
