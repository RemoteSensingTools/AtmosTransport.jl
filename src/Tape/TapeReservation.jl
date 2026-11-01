# Budgets and reservation mechanics; the replay-specific size model and
# candidate search live in Adjoints.Footprint.AutoTape.

# Linux cgroup limits apply at every ancestor. MemAvailable and inactive_file
# include cache the kernel can reclaim; it is not live tape memory.
function _host_tape_memory(; meminfo = "/proc/meminfo", cgroup = "/proc/self/cgroup",
                           cgroup_root = "/sys/fs/cgroup")
    available, limit = Int(Sys.free_memory()), Int(Sys.total_memory())
    if isfile(meminfo)
        for line in eachline(meminfo)
            fields = split(line)
            fields[1] == "MemAvailable:" && (available = parse(Int, fields[2]) * 1024)
            fields[1] == "MemTotal:" && (limit = parse(Int, fields[2]) * 1024)
        end
    end
    group = ""
    if isfile(cgroup)
        for line in eachline(cgroup)
            startswith(line, "0::") && (group = lstrip(line[4:end], '/'))
        end
    end
    root = abspath(cgroup_root)
    path = normpath(joinpath(root, group))
    while path == root || startswith(path, root * "/")
        capfile, usedfile = joinpath(path, "memory.max"), joinpath(path, "memory.current")
        if isfile(capfile) && isfile(usedfile)
            cap = tryparse(Int, strip(read(capfile, String)))
            if cap !== nothing
                used = parse(Int, strip(read(usedfile, String)))
                reclaimable = 0
                statfile = joinpath(path, "memory.stat")
                if isfile(statfile)
                    for line in eachline(statfile)
                        fields = split(line)
                        fields[1] == "inactive_file" && (reclaimable = parse(Int, fields[2]))
                    end
                end
                limit = min(limit, cap)
                available = min(available, max(0, cap - max(0, used - reclaimable)))
            end
        end
        path == root && break
        path = dirname(path)
    end
    return available, limit
end

function _host_tape_budget(; kwargs...)
    available, limit = _host_tape_memory(; kwargs...)
    return max(0, available - max(cld(limit, 10), 8 << 30))
end

# Only Auto selection queries device memory and applies a margin. Explicit
# choices attempt their reservation, including backends without a query hook.
_device_tape_memory(a::AbstractArray) = _device_tape_memory(get_backend(a), a)
_device_tape_memory(backend, _a) = throw(TapeReservationError(0, 0,
    "no tape memory query for backend $(typeof(backend))"))
_device_tape_memory(::KATapeCPU, _a) = _host_tape_memory()
_tape_device_margin(::KATapeCPU, capacity) = cld(capacity, 10)
_tape_device_margin(_backend, capacity) = max(cld(capacity, 10), 4 << 30)
_is_reservation_failure(_err) = false
_is_reservation_failure(::Union{OutOfMemoryError, TapeReservationError}) = true

# A layout entry is (source panels, element type). It retains templates only;
# no state data is copied by sizing. Device reservations allocate all slabs
# before recording; host reservations register one exact-size arena chunk.
_layout_key(entry) = (entry[2], map(size, entry[1]), length(entry) == 2 || entry[3])
_layout_pooled(entry) = length(entry) == 2 || entry[3]
_layout_bytes((panels, T)) = Int(_tape_slot_bytes(T, panels))

function _reserve_tape_pool(storage::DeviceCSTapeStorage, layout)
    pool = _DeviceSlabPool(Any[], 0, 1, false, Dict{Any, _ReservedSlabs}())
    try
        for (panels, T) in layout
            slab = _reserved_panels(panels, T)
            push!(pool.slabs, slab)
            bank = get!(pool.banks, (T, map(size, panels))) do
                _ReservedSlabs(Any[], 0)
            end
            push!(bank.slabs, slab)
        end
        return pool
    catch
        _release_tape_pool!(pool)
        rethrow()
    end
end

function _reserve_tape_pool(storage::PinnedHostCSTapeStorage, layout)
    nbytes = sum(_layout_bytes, layout; init = 0)
    budget = min(first(_host_tape_memory()), something(storage.max_bytes, typemax(Int)))
    nbytes <= budget || throw(TapeReservationError(nbytes, budget, "pinned host reservation exceeds headroom"))
    pool = _PinnedHostArena(_HostTapeChunk[], 0, 0, 1, false, nothing,
                            Dict{CSTapeReadCacheKey, CSPanelCache}())
    finalizer(_release_tape_pool!, pool)
    try
        if !isempty(layout)
            _register_tape_read_backend!(pool, first(layout)[1])
            _reserve_host_chunk!(pool, Int64(nbytes))
        end
        return pool
    catch
        _release_tape_pool!(pool)
        rethrow()
    end
end

_reserve_tape_pool(storage::AbstractCSTapeStorage, layout) =
    _open_tape_pool(storage, sum(_layout_bytes, layout; init = 0))

function _reserved_panels(panels::NTuple{6}, ::Type{T}) where T
    allocated = Any[]
    try
        for panel in panels
            a = similar(panel, T)
            push!(allocated, a)
            # Fault in host pages while failure is still a reserve failure.
            _touch_tape_reservation!(a)
        end
        return ntuple(p -> allocated[p], 6)
    catch
        foreach(_free_tape_array!, allocated)
        rethrow()
    end
end
_touch_tape_reservation!(a::Array) = (fill!(a, zero(eltype(a))); nothing)
_touch_tape_reservation!(::AbstractArray) = nothing

# Read caches are part of the reservation too. The limiter read cache also
# serves as the recording scratch: recording finishes before the reverse.
_reserve_tape_reads!(_pool, _layout) = nothing
function _reserve_tape_reads!(pool::_AbstractHostTapePool, layout)
    isempty(layout) && return nothing
    _register_tape_read_backend!(pool, first(layout)[1])
    for (panels, T) in layout
        roles = T === UInt8 ? (LimiterCodeRead(),) : (AirMassRead(), TracerMassRead())
        for role in roles
            key = (role, T, map(size, panels))
            haskey(pool.read_caches, key) && continue
            pool.read_caches[key] = _reserved_panels(panels, T)
        end
    end
    return nothing
end

_tape_code_scratch!(_pool, _panels) = nothing
function _tape_code_scratch!(pool::_AbstractHostTapePool, panels)
    key = (LimiterCodeRead(), UInt8, map(size, panels))
    return get!(pool.read_caches, key) do
        ntuple(p -> similar(pool.read_template, UInt8, size(panels[p])), 6)
    end
end

_holds_tape_bytes(pool::_DeviceSlabPool, nbytes::Integer) =
    !pool.released && nbytes <= sum(_tape_slot_bytes, pool.slabs; init = 0)
_holds_tape_bytes(pool::_PinnedHostArena, nbytes::Integer) =
    !pool.released && nbytes <= _arena_bytes(pool)

function _reserve_host_chunk!(pool::_PinnedHostArena, capacity::Int64)
    chunk = _new_host_chunk(capacity, pool.read_template)
    push!(pool.chunks, chunk)
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), chunk.base, 0, capacity)
    return nothing
end
