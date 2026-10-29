module SectionTimer

# Hand-rolled host-side section timer. Off by default; at run-start
# `maybe_enable_from_env!` turns timing on when `ATMOSTR_TIMERS` is `1`,
# `true`, `on` or `yes` (case-insensitive). Set `ATMOSTR_ALLOC_TIMERS`
# the same way to also collect CPU allocation bytes for the same sections. Measures wall-clock at named host call boundaries;
# relies on the existing `synchronize(backend)` calls inside operator
# applys to make host time ≈ GPU time per section.
# Per-phase decomposition inside a single kernel launch (build / LU / solve
# inside `_tm5_solve_column!`) is not host-timeable; that lives in a
# separate CPU microbenchmark.

using Printf

# NVTX integration is OPTIONAL — provided by the `AtmosTransportNVTXExt`
# package extension, which loads only when the user has NVTX in their
# environment. These untyped fallbacks are no-ops; the extension adds
# more-specific methods (`_nvtx_start(::AbstractString)`,
# `_nvtx_end(::NVTX.RangeId)`) that win by dispatch when NVTX is present.
# Keeping the fallbacks at `::Any` avoids the "method overwriting during
# precompilation" error that triggers when fallback and extension share an
# identical signature.
_nvtx_start(_) = nothing
_nvtx_end(_) = nothing

const _ENABLED = Threads.Atomic{Bool}(false)
const _ALLOC_ENABLED = Threads.Atomic{Bool}(false)
const _NVTX_ENABLED = Threads.Atomic{Bool}(false)
const _EPOCH = Threads.Atomic{UInt}(0)
const _TIMINGS = Dict{Symbol, Vector{Float64}}()
const _ALLOCATIONS = Dict{Symbol, Vector{Int64}}()
const _WALL_T0 = Ref{Float64}(0.0)
const _WALL_TOTAL = Ref{Float64}(0.0)
const _LOCK = ReentrantLock()

"""
    enable!(; timing=true, allocations=false, nvtx=false)
Reset and turn on. `timing` controls host-side wall-clock accumulation;
`nvtx` emits NVTX ranges for nsys/Nsight Compute timelines (off by
default — under `ATMOSTR_NVTX=1` it is set independently of timing).
"""
function enable!(; timing::Bool = true, allocations::Bool = false, nvtx::Bool = false)
    _ENABLED[] = false
    _ALLOC_ENABLED[] = false
    _NVTX_ENABLED[] = false
    lock(_LOCK) do
        _EPOCH[] = _EPOCH[] + UInt(1)
        empty!(_TIMINGS)
        empty!(_ALLOCATIONS)
        _WALL_T0[] = time_ns() / 1e9
        _WALL_TOTAL[] = 0.0
        _ALLOC_ENABLED[] = allocations
        _NVTX_ENABLED[] = nvtx
        _ENABLED[] = timing
    end
    return nothing
end

"""
    disable!()
Stop accumulating; existing samples remain in `_TIMINGS` until `enable!`
is called again.
"""
function disable!()
    _ENABLED[] = false
    _ALLOC_ENABLED[] = false
    _NVTX_ENABLED[] = false
    lock(_LOCK) do
        _EPOCH[] = _EPOCH[] + UInt(1)
        _WALL_TOTAL[] = (time_ns() / 1e9) - _WALL_T0[]
    end
    return nothing
end

is_enabled() = _ENABLED[]

@inline function record_sample!(section::Symbol, ns::Float64, bytes::Int64 = Int64(0))
    epoch = _EPOCH[]
    record_allocations = _ALLOC_ENABLED[]
    return _record_sample!(section, ns, bytes, epoch, record_allocations)
end

@inline function _record_sample!(section::Symbol, ns::Float64, bytes::Int64,
                                 epoch::UInt, record_allocations::Bool)
    lock(_LOCK) do
        (_ENABLED[] && _EPOCH[] == epoch) || return nothing
        samples = get!(() -> Float64[], _TIMINGS, section)
        push!(samples, ns)
        if record_allocations
            allocs = get!(() -> Int64[], _ALLOCATIONS, section)
            push!(allocs, bytes)
        end
    end
    return nothing
end

function _timer_snapshot()
    return lock(_LOCK) do
        timings = Dict(section => copy(samples) for (section, samples) in _TIMINGS)
        allocations = Dict(section => copy(samples) for (section, samples) in _ALLOCATIONS)
        wall_t0 = _WALL_T0[]
        wall_total = _WALL_TOTAL[]
        (; timings, allocations, wall_t0, wall_total)
    end
end

"""
    @section name expr
Time `expr` and accumulate the elapsed nanoseconds under `name` (a
`Symbol`). When `ATMOSTR_NVTX=1`, also emits an NVTX range labeled with
`name`. When everything is off the macro just executes `expr` with
zero overhead beyond a single atomic flag load.
"""
macro section(name, expr)
    nvtx_label = name isa QuoteNode ? String(name.value) : "section"
    timed = quote
        local _timer_epoch = _EPOCH[]
        if _ENABLED[]
            local _t0 = time_ns()
            local _result
            local _record_allocations = _ALLOC_ENABLED[]
            local _bytes = if _record_allocations
                Int64(Base.@allocated begin
                    _result = $(esc(expr))
                end)
            else
                _result = $(esc(expr))
                Int64(0)
            end
            _record_sample!($(esc(name)), Float64(time_ns() - _t0), _bytes,
                            _timer_epoch, _record_allocations)
            _result
        else
            $(esc(expr))
        end
    end
    quote
        if _NVTX_ENABLED[]
            local _nvtx_h = _nvtx_start($nvtx_label)
            try
                $timed
            finally
                _nvtx_end(_nvtx_h)
            end
        else
            $timed
        end
    end
end

@inline function _time_section_inner(f, name::Symbol)
    epoch = _EPOCH[]
    _ENABLED[] || return f()
    t0 = time_ns()
    record_allocations = _ALLOC_ENABLED[]
    if record_allocations
        local result
        bytes = Int64(Base.@allocated (result = f()))
        _record_sample!(name, Float64(time_ns() - t0), bytes,
                        epoch, record_allocations)
        return result
    else
        result = f()
        _record_sample!(name, Float64(time_ns() - t0), Int64(0),
                        epoch, record_allocations)
        return result
    end
end

"""
    time_section(f, name::Symbol)
Function-form equivalent of `@section`. Use when the timed region is
a do-block or already a closure.
"""
@inline function time_section(f, name::Symbol)
    # Branch on NVTX once, delegate timing+recording to the type-stable inner
    # `_time_section_inner` (defined above). The inner avoids `result = nothing`
    # pre-init so callers receive a concrete `T`, not `Union{Nothing, T}` —
    # important for hot wrappers like the snapshot writer or `_load_window`.
    if _NVTX_ENABLED[]
        h = _nvtx_start(String(name))
        try
            return _time_section_inner(f, name)
        finally
            _nvtx_end(h)
        end
    else
        return _time_section_inner(f, name)
    end
end

function _summary_row(samples::Vector{Float64})
    n = length(samples)
    n == 0 && return (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_s = sum(samples) / 1e9
    mean_ms = sum(samples) / n / 1e6
    sorted = sort(samples)
    p50_ms = sorted[max(1, ceil(Int, 0.50 * n))] / 1e6
    p95_ms = sorted[max(1, ceil(Int, 0.95 * n))] / 1e6
    max_ms = sorted[end] / 1e6
    return (n, total_s, mean_ms, p50_ms, p95_ms, max_ms)
end

"""
    report(io = stderr)
Print a per-section summary table. Columns: section, n_calls, total_s,
mean_ms, p50_ms, p95_ms, max_ms, frac%, plus alloc_MiB and mean_KiB when
allocation samples exist. `frac%` is over the sum of section totals, not over
wall-clock; the header line reports wall-clock time and the time covered by
sections separately.
"""
function report(io::IO = stderr)
    snapshot = _timer_snapshot()
    timings = snapshot.timings
    allocations = snapshot.allocations
    isempty(timings) && (println(io, "[SectionTimer] no samples"); return)
    section_total_ns = sum(sum(v) for v in values(timings); init=0.0)
    section_total_alloc = sum(sum(v) for v in values(allocations); init=Int64(0))
    wall_s = snapshot.wall_total > 0 ? snapshot.wall_total :
             (time_ns() / 1e9 - snapshot.wall_t0)
    @printf(io, "[SectionTimer] wall=%.2fs  covered=%.2fs (%.1f%%)\n",
            wall_s, section_total_ns / 1e9,
            wall_s > 0 ? 100 * (section_total_ns / 1e9) / wall_s : 0.0)
    if !isempty(allocations)
        @printf(io, "[SectionTimer] allocated=%.3f MiB\n",
                section_total_alloc / 2.0^20)
        @printf(io, "%-30s %8s %10s %10s %10s %10s %10s %8s %12s %12s\n",
                "section", "n_calls", "total_s", "mean_ms", "p50_ms",
                "p95_ms", "max_ms", "frac%", "alloc_MiB", "mean_KiB")
    else
        @printf(io, "%-30s %8s %10s %10s %10s %10s %10s %8s\n",
                "section", "n_calls", "total_s", "mean_ms", "p50_ms",
                "p95_ms", "max_ms", "frac%")
    end
    for (sec, samples) in sort(collect(timings); by = p -> -sum(p.second))
        n, total_s, mean_ms, p50_ms, p95_ms, max_ms = _summary_row(samples)
        frac = 100 * (sum(samples) / max(section_total_ns, eps()))
        if !isempty(allocations)
            alloc = sum(get(allocations, sec, Int64[]))
            mean_kib = n == 0 ? 0.0 : alloc / n / 2.0^10
            @printf(io, "%-30s %8d %10.3f %10.3f %10.3f %10.3f %10.3f %8.2f %12.3f %12.3f\n",
                    String(sec), n, total_s, mean_ms, p50_ms, p95_ms, max_ms, frac,
                    alloc / 2.0^20, mean_kib)
        else
            @printf(io, "%-30s %8d %10.3f %10.3f %10.3f %10.3f %10.3f %8.2f\n",
                    String(sec), n, total_s, mean_ms, p50_ms, p95_ms, max_ms, frac)
        end
    end
    return nothing
end

"""
    write_csv(path)
Emit the same summary as `report` to a CSV at `path`. Header:
`section,n_calls,total_s,mean_ms,p50_ms,p95_ms,max_ms,fraction_of_total,allocated_bytes,mean_alloc_bytes`.
`fraction_of_total` is a fraction (0–1), not a percentage; the allocation
columns are 0 unless allocation timing was enabled.
Returns the path on success, or `nothing` if there are no samples.
"""
function write_csv(path::AbstractString)
    snapshot = _timer_snapshot()
    timings = snapshot.timings
    allocations = snapshot.allocations
    isempty(timings) && return nothing
    section_total_ns = sum(sum(v) for v in values(timings); init=0.0)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "section,n_calls,total_s,mean_ms,p50_ms,p95_ms,max_ms,fraction_of_total,allocated_bytes,mean_alloc_bytes")
        for (sec, samples) in sort(collect(timings); by = p -> -sum(p.second))
            n, total_s, mean_ms, p50_ms, p95_ms, max_ms = _summary_row(samples)
            frac = sum(samples) / max(section_total_ns, eps())
            alloc = sum(get(allocations, sec, Int64[]))
            mean_alloc = n == 0 ? 0.0 : alloc / n
            @printf(io, "%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.3f\n",
                    String(sec), n, total_s, mean_ms, p50_ms, p95_ms, max_ms, frac,
                    alloc, mean_alloc)
        end
    end
    return path
end

"""
    maybe_enable_from_env!()
Inspect `ENV["ATMOSTR_TIMERS"]`, `ENV["ATMOSTR_NVTX"]`,
`ENV["ATMOSTR_ALLOC_TIMERS"]` at call time. `"1"` / `"true"` / `"on"` /
`"yes"` switches each axis on independently; anything else (or unset)
is a no-op for that axis. Returns `true` if anything was enabled.
"""
function maybe_enable_from_env!()
    truthy = s -> lowercase(s) in ("1", "true", "on", "yes")
    timing_on = truthy(get(ENV, "ATMOSTR_TIMERS", ""))
    nvtx_on   = truthy(get(ENV, "ATMOSTR_NVTX", ""))
    alloc_on  = truthy(get(ENV, "ATMOSTR_ALLOC_TIMERS", ""))
    if timing_on || nvtx_on
        enable!(timing = timing_on, allocations = alloc_on, nvtx = nvtx_on)
        return true
    end
    return false
end

export @section, time_section, record_sample!, enable!, disable!, is_enabled,
       report, write_csv, maybe_enable_from_env!

end # module SectionTimer
