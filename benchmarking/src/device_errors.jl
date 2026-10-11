# Device errors that do not reach Julia as exceptions. Base only, so the test
# suite checks it without a GPU.
#
# Metal.jl 1.9 (benchmarking/Manifest.toml resolves 1.9.3) commits every
# kernel launch in its own command buffer and reports a failed buffer only
# from the buffer's completion handler, which prints
#
#     ERROR: Failed to submit command buffer: <localized description>
#
# with `Core.println` (src/compiler/execution.jl); `Metal.synchronize` does
# not rethrow it, so a kernel that never ran would otherwise leave an "ok"
# record. While a Metal case runs, the harness therefore captures the
# process's standard output (`capture_stdout`), waits at a completion barrier
# until every handler of the case's command buffers has run
# (`completion_barrier`), and classifies these lines
# (`command_buffer_outcome`). When every failed buffer ran out of memory
# (MTLCommandBufferErrorOutOfMemory, described as "Insufficient Memory
# (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"), the case is
# `skipped_memory`; any other failure makes it `failed`.

"Line prefix of Metal.jl's report of a failed command buffer."
const COMMAND_BUFFER_FAILURE_PREFIX = "ERROR: Failed to submit command buffer:"

"""
    command_buffer_failures(text) -> Vector{String}

Descriptions of the failed Metal command buffers reported in `text`.
"""
function command_buffer_failures(text::AbstractString)
    failures = String[]
    for line in eachline(IOBuffer(text))
        range = findfirst(COMMAND_BUFFER_FAILURE_PREFIX, line)
        range === nothing || push!(failures, String(strip(line[(last(range) + 1):end])))
    end
    return failures
end

"Whether a command-buffer error description reports MTLCommandBufferErrorOutOfMemory."
is_out_of_memory_description(description::AbstractString) =
    occursin(r"OutOfMemory|Insufficient Memory|out of memory"i, description)

"""
    command_buffer_outcome(failures) -> (status, message)

`("ok", "")` without failed command buffers; `("skipped_memory", reason)`
when every failure is out of memory; otherwise `("failed", message)`. The
message lists each distinct description with its count.
"""
function command_buffer_outcome(failures::AbstractVector{<:AbstractString})
    isempty(failures) && return ("ok", "")
    descriptions = unique(failures)
    listed = join(("$(count(==(d), failures)) x $(d)" for d in descriptions), "; ")
    summary = "$(length(failures)) Metal command buffer(s) failed: $(listed)"
    all(is_out_of_memory_description, failures) &&
        return ("skipped_memory", "ran out of device memory: " * summary)
    return ("failed", summary)
end

"""
    completion_barrier(commit_sentinel; timeout, poll = 0.001)

Wait until the device's completion handlers of all work committed so far
have run. `commit_sentinel(signal)` must commit a sentinel command buffer on
the queue that ran the work, with a completion handler that calls
`signal()`. A queue completes its command buffers in commit order, and Metal
runs a queue's completion handlers one at a time in that order, so when the
sentinel's handler runs, every earlier handler has returned (and printed).
`signal` only sets an atomic flag: completion handlers run on a Metal thread
that must not switch Julia tasks. Throws when the signal does not arrive
within `timeout` seconds.
"""
function completion_barrier(commit_sentinel; timeout::Real, poll::Real = 0.001)
    done = Threads.Atomic{Bool}(false)
    commit_sentinel(() -> (done[] = true; nothing))
    deadline = time() + timeout
    while !done[]
        time() > deadline && error(
            "device completion handlers did not finish within $(timeout) s; the case's " *
            "command-buffer errors cannot be checked")
        sleep(poll)
    end
    return nothing
end

"An exception `capture_stdout` caught, with its backtrace."
struct CaughtError
    exception::Any
    backtrace::Vector{Any}
end

"""
    metal_case_decision(outcome, status) -> Symbol

What a Metal case's result decides, given `outcome` (the case's result or a
`CaughtError`) and `status`, the verdict of its captured command-buffer
reports (`command_buffer_outcome`): `:rethrow`, `:skipped_memory`, `:failed`
or `:ok`. A thrown error (from the case, or from a completion barrier that
never signalled) outranks the reports, because without the barrier they may be
incomplete; only an out-of-memory exception next to out-of-memory reports
keeps the case `:skipped_memory`.
"""
function metal_case_decision(outcome, status::AbstractString)
    if outcome isa CaughtError
        status == "skipped_memory" && is_out_of_memory(outcome.exception) && return :skipped_memory
        return :rethrow
    end
    status == "skipped_memory" && return :skipped_memory
    status == "failed" && return :failed
    return :ok
end

"""
    capture_stdout(f) -> (outcome, text)

Run `f()` with the process's standard output redirected to a temporary
file, including `Core.stdout`, to which Metal.jl's completion handlers print.
Returns `f`'s result, or a `CaughtError` when it threw (an interrupt is
rethrown), together with the captured text, which is also forwarded to the
original standard output. A file rather than a pipe: completion handlers
write from a Metal thread while the main task may wait in a synchronize, and
a pipe nobody reads could fill and stall them.
"""
function capture_stdout(f)
    original = stdout
    path, io = mktemp()
    outcome = try
        redirect_stdout(io)
        try
            f()
        finally
            redirect_stdout(original)
            close(io)
        end
    catch err
        err isa InterruptException && (rm(path; force = true); rethrow())
        CaughtError(err, catch_backtrace())
    end
    text = read(path, String)
    rm(path; force = true)
    print(original, text)
    return outcome, text
end
