# Resolve once, reserve before step 1. Size entries retain array templates,
# never copies of state. The binary contract has one palindrome per step;
# other schedules (and Lin-Rood) use the recorder's cached counting replay.
struct _TapeSizing
    layouts::Vector{Vector{Any}}
    pending::Vector{Bool}
    panels::Any
    prefixes::Dict{Any, Tuple{Any, Vector{Int}}}
end

function _TapeSizing(layouts, pending, panels)
    prefixes = Dict{Any, Tuple{Any, Vector{Int}}}()
    n = length(layouts)
    for (s, layout) in enumerate(layouts), entry in layout
        key = _layout_key(entry)
        _, counts = get!(prefixes, key) do
            (entry, zeros(Int, n + 1))
        end
        counts[s + 1] += 1
    end
    for (_, counts) in values(prefixes)
        cumsum!(counts, counts)
    end
    return _TapeSizing(layouts, pending, panels, prefixes)
end

function _closed_tape_size(replay::_CSReplay, n::Int)
    panels = _replay_m(replay)
    layouts = [Any[] for _ in 1:n]
    pending = falses(n)
    waiting = false
    for s in 1:n
        layout = layouts[s]
        for stage in _stages_of_step(replay.plan, replay.steps, s)
            if stage isa TransportStage
                append!(layout, fill((panels, eltype(first(panels))), 6))
                _append_codes!(layout, replay.advection, panels)
                waiting = false
            elseif stage isa ConvectionStage && !(replay.convection isa NoConvection)
                waiting && push!(layout, (panels, eltype(first(panels))))
                waiting = true
            elseif stage isa AirMassResetStage
                if stage.mode === :preserve_vmr
                    append!(layout, fill((panels, eltype(first(panels))), 2))
                elseif waiting
                    push!(layout, (panels, eltype(first(panels))))
                end
                waiting = false
            end
        end
        pending[s] = waiting
    end
    return _TapeSizing(layouts, collect(pending), panels)
end
_append_codes!(_layout, ::CSAdjointLinearScheme, _panels) = nothing
_append_codes!(layout, ::CSAdjointNonlinearScheme, panels) =
    (append!(layout, fill((panels, UInt8), 6)); nothing)

function _size_tape!(runtime::_TapeRuntime, replay::_CSReplay, n::Int)
    runtime.sizing === nothing || return runtime.sizing
    if replay.meteo.binary_substeps && cs_advection_style(replay.advection) isa CSSplitSweepStyle
        return _closed_tape_size(replay, n)
    end
    initial = _replay_snapshot(replay)
    counter = _CSCountingRecorder(StateTape(), replay.advection)
    try
        _replay_steps!(replay, 1:n; recorder = counter)
    finally
        _replay_restore!(replay, initial)
        _release_snapshot!(initial)
    end
    _store_tape_count!(runtime, counter)
    return _TapeSizing(counter.layouts, counter.pending_slot_bytes .> 0, _replay_m(replay))
end

function _segment_counts(size::_TapeSizing, steps::AbstractUnitRange)
    counts = Dict{Any, Tuple{Any, Int}}()
    isempty(steps) && return counts
    for (key, (entry, through)) in size.prefixes
        count = through[last(steps) + 1] - through[first(steps)]
        count > 0 && (counts[key] = (entry, count))
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

# Measured device costs estimate relative time for the log. Candidate order
# follows the storage tiers; host and disk times are not extrapolated.
const _AUTO_TAPE_COST = (linear = (full = 7.36, forward = 4.25),
                         limited = (full = 8.54, forward = 4.46))
_tape_cost(::CSAdjointLinearScheme) = _AUTO_TAPE_COST.linear
_tape_cost(_scheme) = _AUTO_TAPE_COST.limited
_tape_cost(::CSAdjointLinRoodScheme) = nothing
_expected_tape_time(::FullCheckpoint, _n, _cost) = 1.0
_expected_tape_time(s::StrideCheckpoint, n, cost) =
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
    if tape.checkpoint isa StrideCheckpoint
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
    if tape.checkpoint isa StrideCheckpoint
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
    if constraint isa Union{FullCheckpoint, StrideCheckpoint}
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
    _require_tape_supported(replay.advection, request)
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
            resident_tail = (checkpoint isa StrideCheckpoint ? checkpoint.resident_tail : n),
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

_require_tape_supported(scheme, tape::AutoTape) = begin
    tape.storage === nothing || _require_tape_storage(scheme, tape.storage)
    content = _tape_content(tape)
    _require_tape_content(content, scheme)
    tape.checkpoint === nothing || _check_tape_schedule(content, tape.checkpoint)
    nothing
end

function _reserved_snapshot(replay, pool)
    rm = _allocate_tape_slot(pool, replay.state.tracers_raw).panels
    m = _allocate_tape_slot(pool, _replay_m(replay)).panels
    return (; rm, m, _snapshot_operator_state(replay)...)
end

# Convection ledger carries are sized on first use. Reserve their eventual
# Nc×Nc×1 shape now; the initial empty carry is equivalent to zero columns.
function _reserve_snapshot_state(x::AbstractArray, replay)
    if isempty(x) && ndims(x) == 3
        return zeros_like(x, eltype(x), replay.mesh.Nc, replay.mesh.Nc, 1)
    end
    return x
end
_reserve_snapshot_state(x::Union{Tuple, NamedTuple}, replay) = map(v -> _reserve_snapshot_state(v, replay), x)
_reserve_snapshot_state(x, _replay) = x
function _snapshot_operator_state(replay)
    return _reserve_snapshot_state((emissions = operator_state(replay.emissions),
        convection = operator_state(replay.convection_workspace),
        diffusion = operator_state(replay.diffusion_workspace)), replay)
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
