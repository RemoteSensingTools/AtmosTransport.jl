# Resolve once, reserve before step 1. Size entries retain array templates,
# never copies of state. The binary contract has one palindrome per step;
# other schedules (and Lin-Rood) use the recorder's cached counting replay.
# `window` is the met window of every sized step (MetWindowCheckpoint's
# segments).
struct _TapeSizing
    layouts::Vector{Vector{Any}}
    pending::Vector{Bool}
    boundary_flush::Vector{Bool}
    panels::Any
    prefixes::Dict{Any, Tuple{Any, Vector{Int}}}
    window::Vector{Int}
end

function _TapeSizing(layouts, pending, panels, window::AbstractVector{<:Integer})
    length(window) == length(layouts) || throw(DimensionMismatch(
        "tape sizing of $(length(layouts)) steps given met windows of $(length(window)) steps"))
    prefixes = Dict{Any, Tuple{Any, Vector{Int}}}()
    n = length(layouts)
    boundary_flush = falses(n)
    for (s, layout) in enumerate(layouts), entry in layout
        # A flush can belong to convection from the preceding step. A
        # recorder starting at this step has no such record to flush.
        length(entry) > 3 && entry[4] < s && (boundary_flush[s] = true)
        key = _layout_key(entry)
        _, counts = get!(prefixes, key) do
            (entry, zeros(Int, n + 1))
        end
        counts[s + 1] += 1
    end
    for (_, counts) in values(prefixes)
        cumsum!(counts, counts)
    end
    return _TapeSizing(layouts, pending, collect(boundary_flush), panels, prefixes, collect(Int, window))
end

function _closed_tape_size(replay::_CSReplay, n::Int)
    panels = _replay_m(replay)
    layouts = [Any[] for _ in 1:n]
    pending = falses(n)
    waiting = false
    convection_step = 0
    for s in 1:n
        layout = layouts[s]
        for stage in _stages_of_step(replay.plan, replay.steps, s)
            if stage isa TransportStage
                append!(layout, fill((panels, eltype(first(panels))), 6))
                _append_codes!(layout, replay.advection, panels)
                waiting = false
            elseif stage isa ConvectionStage && !(replay.convection isa NoConvection)
                waiting && push!(layout, (panels, eltype(first(panels)), true, convection_step))
                waiting = true
                convection_step = s
            elseif stage isa AirMassResetStage
                if stage.mode === :preserve_vmr
                    append!(layout, fill((panels, eltype(first(panels))), 2))
                elseif waiting
                    push!(layout, (panels, eltype(first(panels)), true, convection_step))
                end
                waiting = false
            end
        end
        pending[s] = waiting
    end
    return _TapeSizing(layouts, collect(pending), panels, replay.steps.window[1:n])
end
_append_codes!(_layout, ::CSAdjointLinearScheme, _panels) = nothing
_append_codes!(layout, ::CSAdjointNonlinearScheme, panels) =
    (append!(layout, fill((panels, UInt8), 6)); nothing)

function _size_tape!(runtime::_TapeRuntime, replay::_CSReplay, n::Int)
    runtime.sizing === nothing || return runtime.sizing
    if replay.meteo.binary_substeps && cs_advection_style(replay.advection) isa CSSplitSweepStyle
        return _closed_tape_size(replay, n)
    end
    return _counted_tape_size!(runtime, replay, n, replay.source, cs_advection_style(replay.advection))
end

_counted_tape_size!(runtime::_TapeRuntime, replay::_CSReplay, n::Int, _source, _style) =
    _every_step_tape_size!(runtime, replay, n)

# Every step counted from step 1, from the replay's initial state (a counting
# run keeps no met, so a streamed source's windows are released as it goes).
function _every_step_tape_size!(runtime::_TapeRuntime, replay::_CSReplay, n::Int)
    initial = _replay_snapshot(replay)
    counter = _CSCountingRecorder(StateTape(), replay.advection)
    try
        _replay_steps!(replay, 1:n; recorder = counter)
    finally
        _release_replay_windows!(replay)
        _replay_restore!(replay, initial)
        _release_snapshot!(initial)
    end
    _store_tape_count!(runtime, counter)
    return _TapeSizing(counter.layouts, counter.pending_slot_bytes .> 0, _replay_m(replay),
                       replay.steps.window[1:n])
end

# Lin–Rood with a streamed source: count one window, not the period. A
# Lin–Rood step has a fixed record structure (one horizontal operator per
# half palindrome, no CFL subcycles) and stages every convection air mass at
# its call (no record waits for a later step), so a step's records follow
# from the stages it runs: its class, whether it is in window 1 and whether
# it is its window's first or last step. Window 1 and the longest later window
# are counted from the initial state with a fresh counter, and every step
# takes the layout of the counted step of its class (with its convection step
# index `entry[4]` moved by the same offset). Should a step stage more pool
# slots than its class did, recording throws `TapeReservationExceeded`; the
# Lin–Rood horizontal records, outside the pool, only enter the device-peak
# estimate. A run with a class the two windows do not have (a one-step
# window, a run of window 1 only ending inside it) counts every step.
function _counted_tape_size!(runtime::_TapeRuntime, replay::_CSReplay, n::Int,
                             ::AbstractStreamedCSWindowSource, ::CSLinRoodStyle)
    sizing = _class_tape_size(replay, n)
    return sizing === nothing ? _every_step_tape_size!(runtime, replay, n) : sizing
end

_step_class(steps::_ResolvedSchedule, s::Int) = (steps.window[s] == 1, steps.first[s], steps.last[s])

function _class_tape_size(replay::_CSReplay, n::Int)
    steps = replay.steps
    window = view(steps.window, 1:n)
    ranges = UnitRange{Int}[]
    first_window = findall(==(1), window)
    push!(ranges, first(first_window):last(first_window))
    later = [findall(==(w), window) for w in 2:window[n]]
    if !isempty(later)
        longest = argmax(length, later)
        push!(ranges, first(longest):last(longest))
    end
    covered = Set(_step_class(steps, s) for range in ranges for s in range)
    all(s -> _step_class(steps, s) in covered, 1:n) || return nothing
    counted = Dict{NTuple{3, Bool}, Tuple{Int, Vector{Any}, Bool}}()
    initial = _replay_snapshot(replay)
    try
        for range in ranges
            _replay_restore!(replay, initial)
            counter = _CSCountingRecorder(StateTape(), replay.advection)
            _replay_steps!(replay, range; recorder = counter)
            for (k, s) in enumerate(range)
                get!(counted, _step_class(steps, s)) do
                    (s, s <= length(counter.layouts) ? counter.layouts[s] : Any[],
                     counter.pending_slot_bytes[k] > 0)
                end
            end
        end
    finally
        _release_replay_windows!(replay)
        _replay_restore!(replay, initial)
        _release_snapshot!(initial)
    end
    layouts = Vector{Vector{Any}}(undef, n)
    pending = falses(n)
    for s in 1:n
        r, layout, waiting = counted[_step_class(steps, s)]
        layouts[s] = Any[length(entry) > 3 ? (entry[1], entry[2], entry[3], entry[4] + s - r) : entry
                         for entry in layout]
        pending[s] = waiting
    end
    return _TapeSizing(layouts, collect(pending), _replay_m(replay), steps.window[1:n])
end

function _segment_counts(size::_TapeSizing, steps::AbstractUnitRange)
    counts = Dict{Any, Tuple{Any, Int}}()
    isempty(steps) && return counts
    for (key, (entry, through)) in size.prefixes
        count = through[last(steps) + 1] - through[first(steps)]
        count > 0 && (counts[key] = (entry, count))
    end
    if size.boundary_flush[first(steps)]
        key = _layout_key((size.panels, eltype(first(size.panels))))
        entry, count = counts[key]
        count == 1 ? delete!(counts, key) : (counts[key] = (entry, count - 1))
    end
    if size.pending[last(steps)]
        entry = (size.panels, eltype(first(size.panels)))
        key = _layout_key(entry)
        counts[key] = (entry, get(counts, key, (entry, 0))[2] + 1)
    end
    return counts
end
_counts_layout(counts) = reduce(vcat, [fill(entry, n) for (entry, n) in values(counts)]; init = Any[])
_counts_bytes(counts) = sum(v -> _layout_bytes(v[1]) * v[2], values(counts); init = 0)
_segment_layout(size::_TapeSizing, steps) = _counts_layout(_segment_counts(size, steps))
_layout_total(layout) = sum(_layout_bytes, layout; init = 0)

function _window_counts(size::_TapeSizing, windows)
    bank = Dict{Any, Tuple{Any, Int}}()
    for steps in windows, (key, (entry, n)) in _segment_counts(size, steps)
        bank[key] = (entry, max(n, get(bank, key, (entry, 0))[2]))
    end
    return bank
end

_head_steps(s::StrideCheckpoint, n) = max(0, n - s.resident_tail)
_head_windows(s::StrideCheckpoint, n) =
    [checkpoint_window_range(s, w, n) for w in
        1:(checkpoint_window_count(s, n) - (s.resident_tail > 0))]

# MetWindowCheckpoint re-records every taped step (no resident tail); its
# segments over steps `1:n` follow the met window of each step.
_head_steps(::MetWindowCheckpoint, n) = n
_head_windows(s::MetWindowCheckpoint, step_windows::AbstractVector{<:Integer}, n) =
    _met_window_segments(s, view(step_windows, 1:n))

# Measured device costs estimate relative time for the log. Candidate order
# follows the storage tiers; host and disk times are not extrapolated.
const _AUTO_TAPE_COST = (linear = (full = 7.36, forward = 4.25),
                         limited = (full = 8.54, forward = 4.46))
_tape_cost(::CSAdjointLinearScheme) = _AUTO_TAPE_COST.linear
_tape_cost(_scheme) = _AUTO_TAPE_COST.limited
_tape_cost(::CSAdjointLinRoodScheme) = nothing
_expected_tape_time(::FullCheckpoint, _n, _cost) = 1.0
_expected_tape_time(s::_SegmentedCheckpoint, n, cost) =
    cost === nothing ? missing : 1 + (_head_steps(s, n) / n) * cost.forward / cost.full
_expected_tape_time(tape::AdjointTape, n, cost) =
    tape.storage isa DeviceCSTapeStorage &&
    (tape.checkpoint isa FullCheckpoint || tape.checkpoint.snapshots isa DeviceCSTapeStorage) ?
        _expected_tape_time(tape.checkpoint, n, cost) : missing

# All reverse starts remain resident (Jacobian walks each window for all of
# them). Reserve headroom for lambda, workspaces, footprints and host read /
# staging buffers beyond the tape and snapshots themselves.
function _reverse_budget_bytes(replay, n, nstarts)
    S = _bytes_per_panel_tuple(_replay_m(replay))
    surface = 6 * replay.mesh.Nc^2 * sizeof(eltype(first(_replay_m(replay))))
    return nstarts * (8S + (_has_surface_flux(replay) ? n * surface : 0)) + 4S
end

function _auto_memory_budgets(backend, free, capacity, reverse_bytes, request, host)
    available = max(0, Int(free) - _tape_device_margin(backend, Int(capacity)) - reverse_bytes)
    device = something(request.budget_bytes, available)
    return device, backend isa KACPU ? device : host
end

function _tape_budgets(replay, n, nstarts, request)
    template = first(_replay_m(replay))
    backend = get_backend(template)
    if request isa AdjointTape
        # Explicit choices are checked by allocation, not Auto's heuristic.
        device, host = typemax(Int), typemax(Int)
    else
        free, capacity = _device_tape_memory(template)
        device, host = _auto_memory_budgets(backend, free, capacity,
            _reverse_budget_bytes(replay, n, nstarts), request, _host_tape_budget())
    end
    # CPU device slabs and host snapshots share the same physical memory.
    dir = request isa AutoTape ? request.scratch_dir : nothing
    request.storage isa MmapCSTapeStorage && (dir = request.storage.dir)
    disk = typemax(Int)
    if dir !== nothing
        parent = abspath(dir)
        while !ispath(parent)
            parent = dirname(parent)
        end
        disk = Int(diskstat(parent).available)
    end
    return (; device, host, disk, shared = get_backend(template) isa KACPU)
end

function _candidate_counts(tape::AdjointTape, size::_TapeSizing, n)
    s = tape.checkpoint
    if s isa MetWindowCheckpoint
        windows = _head_windows(s, size.window, n)
        return (_window_counts(size, windows), Dict{Any, Tuple{Any, Int}}(), length(windows))
    end
    if s isa FullCheckpoint
        return (_segment_counts(size, 1:n), Dict{Any, Tuple{Any, Int}}(), 0)
    end
    head = _head_steps(s, n)
    return (_window_counts(size, _head_windows(s, n)),
            _segment_counts(size, (head + 1):n), cld(head, s.K))
end

function _candidate_layouts(tape, size, n)
    head, tail, nsnaps = _candidate_counts(tape, size, n)
    return _counts_layout(head), _counts_layout(tail), nsnaps
end

_tape_storage_tier(::DeviceCSTapeStorage) = :device
_tape_storage_tier(::PinnedHostCSTapeStorage) = :host
_tape_storage_tier(::MmapCSTapeStorage) = :disk

function _candidate_memory(tape, size, n, snapshot_bytes)
    head, tail, nsnaps = _candidate_counts(tape, size, n)
    headbytes, tailbytes = _counts_bytes(head), _counts_bytes(tail)
    bytes = headbytes + tailbytes
    tier = _tape_storage_tier(tape.storage)
    device = tier === :device ? bytes : 0
    host = tier === :host ? bytes : 0
    disk = tier === :disk ? bytes : 0
    if tape.checkpoint isa _SegmentedCheckpoint
        if tape.checkpoint.snapshots isa DeviceCSTapeStorage
            device += nsnaps * snapshot_bytes
        else
            # Persistent operator state keeps its backend types; only rm/m
            # (the dominant payload) move to pinned host storage.
            statebytes = 2 * _bytes_per_panel_tuple(size.panels)
            host += nsnaps * statebytes
            device += nsnaps * max(0, snapshot_bytes - statebytes)
        end
    end
    return (; device, host, disk, headbytes, tailbytes, nsnaps)
end

function _fits_tape(tape, size, n, snapbytes, budgets)
    mem = _candidate_memory(tape, size, n, snapbytes)
    mem.device <= budgets.device && mem.host <= budgets.host || return false
    mem.disk <= floor(Int, get(budgets, :disk, typemax(Int)) / 1.1) || return false
    budgets.shared && mem.device + mem.host > budgets.host && return false
    storage = tape.storage
    storage isa PinnedHostCSTapeStorage && mem.headbytes + mem.tailbytes > something(storage.max_bytes, typemax(Int)) && return false
    if tape.checkpoint isa _SegmentedCheckpoint
        cap = tape.checkpoint.snapshots
        cap isa PinnedHostCSTapeStorage && mem.nsnaps * 2 * _bytes_per_panel_tuple(size.panels) > something(cap.max_bytes, typemax(Int)) && return false
    end
    return true
end

_tape_content(t::AdjointTape) = t.content
_tape_content(t::AutoTape) = something(t.content, StateTape())

function _auto_candidates(request::AutoTape, replay, size, n, snapbytes, budgets)
    content = _tape_content(request)
    storage = something(request.storage, DeviceCSTapeStorage())
    candidates = AdjointTape[]
    host_candidates = AdjointTape[]
    constraint = request.checkpoint
    if constraint isa AbstractCheckpointSchedule
        push!(candidates, AdjointTape(content, storage, constraint))
    else
        constraint === nothing && push!(candidates, AdjointTape(content, storage, FullCheckpoint()))
        if content isa StateTape
            strides = constraint isa StrideSearch && constraint.K !== nothing ? (constraint.K,) : 1:n
            stores = constraint isa StrideSearch && constraint.snapshots !== nothing ?
                (constraint.snapshots,) : (DeviceCSTapeStorage(), PinnedHostCSTapeStorage())
            for snapshots in stores
                best = nothing
                smallest = nothing
                smallestbytes = typemax(Int)
                for K in strides
                    tailconstraint = constraint isa StrideSearch ? constraint.resident_tail : nothing
                    base = AdjointTape(content, storage, StrideCheckpoint(K; snapshots,
                        resident_tail = something(tailconstraint, 0)))
                    devicebudgets = snapshots isa PinnedHostCSTapeStorage ?
                        (; device = budgets.device, host = typemax(Int), disk = get(budgets, :disk, typemax(Int)), shared = false) : budgets
                    _fits_tape(base, size, n, snapbytes, devicebudgets) || continue
                    base_mem = _candidate_memory(base, size, n, snapbytes)
                    if _fits_tape(base, size, n, snapbytes, budgets) && base_mem.device < smallestbytes
                        smallest = base
                        smallestbytes = base_mem.device
                    end
                    lo, hi = tailconstraint === nothing ? (0, n - 1) : (tailconstraint, tailconstraint)
                    while lo < hi
                        mid = cld(lo + hi, 2)
                        tape = AdjointTape(content, storage, StrideCheckpoint(K; snapshots, resident_tail = mid))
                        if _fits_tape(tape, size, n, snapbytes, devicebudgets)
                            lo = mid
                        else
                            hi = mid - 1
                        end
                    end
                    tape = AdjointTape(content, storage, StrideCheckpoint(K; snapshots, resident_tail = lo))
                    _fits_tape(tape, size, n, snapbytes, budgets) || continue
                    if best === nothing || tape.checkpoint.resident_tail > best.checkpoint.resident_tail ||
                       (tape.checkpoint.resident_tail == best.checkpoint.resident_tail &&
                        checkpoint_window_count(tape.checkpoint, n) < checkpoint_window_count(best.checkpoint, n))
                        best = tape
                    end
                end
                choices = snapshots isa PinnedHostCSTapeStorage ? host_candidates : candidates
                best === nothing || push!(choices, best)
                smallest === nothing || push!(choices, smallest)
            end
        end
    end
    append!(candidates, host_candidates)
    # Disk is opt-in, after device heads with pinned-host snapshots.
    if request.storage === nothing && request.scratch_dir !== nothing &&
       !(replay.advection isa CSAdjointLinRoodScheme)
        diskrequest = AutoTape(; content, storage = MmapCSTapeStorage(request.scratch_dir),
                              checkpoint = constraint, budget_bytes = request.budget_bytes)
        append!(candidates, _auto_candidates(diskrequest, replay, size, n, snapbytes, budgets))
    end
    return candidates
end

# Array bytes in a snapshot include persistent operator state, not just rm/m.
_snapshot_bytes(::Nothing) = 0
_snapshot_bytes(x::AbstractArray) = sizeof(eltype(x)) * length(x)
_snapshot_bytes(x::Union{Tuple, NamedTuple}) = sum(_snapshot_bytes, x; init = 0)
_snapshot_bytes(_x) = 0

function _reserve_candidate!(runtime, tape, replay, size, n)
    head, tail, nsnaps = _candidate_layouts(tape, size, n)
    runtime.tape = tape
    try
        runtime.pool = _reserve_tape_pool(tape.storage, filter(_layout_pooled, head))
        _reserve_tape_reads!(runtime.pool, head)
        isempty(tail) || (runtime.tail_pool = _reserve_tape_pool(tape.storage, filter(_layout_pooled, tail)))
        runtime.tail_pool === nothing || _reserve_tape_reads!(runtime.tail_pool, tail)
        if nsnaps > 0
            snapstorage = tape.checkpoint.snapshots
            panels = _replay_m(replay)
            rm = replay.state.tracers_raw
            layout = repeat(Any[(rm, eltype(first(rm))), (panels, eltype(first(panels)))], nsnaps)
            runtime.snapshot_pool = _reserve_tape_pool(snapstorage, layout)
            for _ in 1:nsnaps
                push!(runtime.snapshots, _reserved_snapshot(replay, runtime.snapshot_pool))
            end
        end
    catch
        _release_tape_runtime!(runtime)
        rethrow()
    end
    return nothing
end

function _prepare_tape!(runtime::_TapeRuntime, replay::_CSReplay, n::Int, nstarts::Int = 1)
    runtime.sizing === nothing || return runtime.tape
    request = runtime.request
    _require_tape_supported(replay.advection, request, replay.source)
    size = _size_tape!(runtime, replay, n)
    opstate = _snapshot_operator_state(replay)
    snapbytes = 2 * _bytes_per_panel_tuple(_replay_m(replay)) + _snapshot_bytes(opstate)
    _release_snapshot_state!(opstate)
    budgets = _tape_budgets(replay, n, nstarts, request)
    candidates = request isa AutoTape ? _auto_candidates(request, replay, size, n, snapbytes, budgets) : [request]
    failures = String[]
    for tape in candidates
        if request isa AutoTape && !_fits_tape(tape, size, n, snapbytes, budgets)
            mem = _candidate_memory(tape, size, n, snapbytes)
            push!(failures, "$(tape): needs device=$(mem.device), host=$(mem.host), disk=$(mem.disk); budgets=$(budgets)")
            continue
        end
        try
            _reserve_candidate!(runtime, tape, replay, size, n)
        catch err
            _is_reservation_failure(err) || rethrow()
            push!(failures, "$(tape): " * sprint(showerror, err))
            continue
        end
        runtime.sizing = size
        mem = _candidate_memory(tape, size, n, snapbytes)
        checkpoint = tape.checkpoint
        @info("adjoint tape", storage = tape.storage, checkpoint,
            K = (checkpoint isa StrideCheckpoint ? checkpoint.K : 0),
            resident_tail = (checkpoint isa StrideCheckpoint ? checkpoint.resident_tail :
                             checkpoint isa FullCheckpoint ? n : 0),
            content = tape.content, estimate_GB = _layout_total(_segment_layout(size, 1:n)) / 1e9,
            budget_GB = budgets.device / 1e9, budgets, rejected_candidates = failures,
            reserved_GB = (mem.device + mem.host + mem.disk) / 1e9,
            expected_rel_time = _expected_tape_time(tape, n, _tape_cost(replay.advection)))
        return tape
    end
    throw(TapeReservationError(_layout_total(_segment_layout(size, 1:n)), budgets.device,
        "no tape candidate could be reserved; budgets=$(budgets); " * join(failures, "; ")))
end

_begin_taped_run(replay::_CSReplay, runtime::_TapeRuntime, n::Int; nstarts::Int = 1) =
    _begin_taped_run(_prepare_tape!(runtime, replay, n, nstarts).checkpoint, replay, runtime, n)

_require_tape_supported(scheme, tape::AutoTape, source = nothing) = begin
    tape.storage === nothing || _require_tape_storage(scheme, tape.storage)
    content = _tape_content(tape)
    _require_tape_content(content, scheme)
    tape.checkpoint === nothing || _check_tape_schedule(content, tape.checkpoint)
    _require_window_tape(source, tape)
    nothing
end

function _reserved_snapshot(replay, pool)
    rm = _allocate_tape_slot(pool, replay.state.tracers_raw).panels
    m = _allocate_tape_slot(pool, _replay_m(replay)).panels
    return (; rm, m, _snapshot_operator_state(replay)...)
end

function _snapshot_operator_state(replay)
    return (emissions = operator_state(replay.emissions),
        convection = convection_checkpoint_state(replay.convection,
            replay.convection_workspace, replay.state, replay.mesh),
        diffusion = operator_state(replay.diffusion_workspace))
end
_release_snapshot_state!(x::AbstractArray) = _free_tape_array!(x)
_release_snapshot_state!(x::Union{Tuple, NamedTuple}) = (foreach(_release_snapshot_state!, x); nothing)
_release_snapshot_state!(_x) = nothing

function _update_snapshot_state!(dest::AbstractArray, src::AbstractArray)
    if isempty(src)
        fill!(dest, zero(eltype(dest)))
    else
        size(dest) == size(src) || throw(TapeReservationExceeded(sizeof(dest), sizeof(src)))
        copyto!(dest, src)
    end
    return dest
end
_update_snapshot_state!(dest::NamedTuple, src::NamedTuple) = map(_update_snapshot_state!, dest, src)
_update_snapshot_state!(dest::Tuple, src::Tuple) = map(_update_snapshot_state!, dest, src)
_update_snapshot_state!(_dest, src) = src

_snapshot_sync!(::DeviceCSTapeStorage, _template) = nothing
_snapshot_sync!(::PinnedHostCSTapeStorage, template) = _tape_synchronize(template)

function _save_snapshot!(snapshot, replay, storage)
    template = first(_replay_m(replay))
    _snapshot_sync!(storage, template)
    foreach(copyto!, snapshot.rm, replay.state.tracers_raw)
    foreach(copyto!, snapshot.m, _replay_m(replay))
    _snapshot_sync!(storage, template)
    state = (emissions = operator_state(replay.emissions),
             convection = operator_state(replay.convection_workspace),
             diffusion = operator_state(replay.diffusion_workspace))
    try
        return (; rm = snapshot.rm, m = snapshot.m,
                 emissions = _update_snapshot_state!(snapshot.emissions, state.emissions),
                 convection = _update_snapshot_state!(snapshot.convection, state.convection),
                 diffusion = _update_snapshot_state!(snapshot.diffusion, state.diffusion))
    finally
        _release_snapshot_state!(state)
    end
end
