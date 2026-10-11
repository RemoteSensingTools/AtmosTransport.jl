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
# process's standard output (`capture_stdout`) and classifies these lines
# (`command_buffer_outcome`). An out-of-memory buffer
# (MTLCommandBufferErrorOutOfMemory, described as "Insufficient Memory
# (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)") makes the case
# `skipped_memory`; any other failed buffer makes it `failed`.

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
when one of them ran out of memory; otherwise `("failed", message)`.
"""
function command_buffer_outcome(failures::AbstractVector{<:AbstractString})
    isempty(failures) && return ("ok", "")
    summary = "$(length(failures)) Metal command buffer(s) failed, first: $(first(failures))"
    any(is_out_of_memory_description, failures) &&
        return ("skipped_memory", "ran out of device memory: " * summary)
    return ("failed", summary)
end

"An exception `capture_stdout` caught, with its backtrace."
struct CaughtError
    exception::Any
    backtrace::Vector{Any}
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
