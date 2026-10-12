# ===========================================================================
# Rolling NVMe input staging for the per-day transport binary loop.
#
# Transport binaries (~15 GB/day) live on NFS-mounted NAS. Cold, strided
# per-window reads over NFS make the window prefetch IO-bound — measured on a
# C180 GEOS 6-day run: prefetch_fetch_wait 96 s (NAS) vs 5 s (local NVMe), wall
# 265 s vs 160 s (−40 %). For multi-month / multi-year runs the dataset (> RAM)
# cannot be page-cached, so every new day is a cold NFS read mid-run.
#
# This layer keeps a small ROLLING window of upcoming days copied onto fast
# local NVMe and evicts days already processed, bounding NVMe use to
# (lookahead + 1 + keep_behind) day-files regardless of run length. The copy
# runs on a background task (same pattern as the async daily write) so it
# overlaps GPU transport; the main loop blocks only if a needed day is not yet
# staged. On any copy failure it transparently falls back to the original NAS
# path, so staging never breaks a run.
#
# OPT-IN: default disabled ⇒ `staged_path_for!` returns the original NAS path,
# bit-identical to a non-staged run.
# ===========================================================================

module InputStaging

using ..Models: _config_bool
using ...AtomicFiles: atomic_publish
using FileWatching: Pidfile
import TOML

export InputStager, staged_path_for!, cleanup_staging!

"""
    InputStager

Rolling local-disk stager for the daily transport binary list. Construct from
the resolved NAS `binary_paths` and the `[input.staging]` config sub-table;
call [`staged_path_for!`](@ref) at each day's driver-open site and
[`cleanup_staging!`](@ref) at run end.

All mutable bookkeeping (`staged`, `tasks`, `failed`) is touched only by the
main loop thread — background tasks are pure (copy + return the path or throw),
so there is no shared-state race.
"""
mutable struct InputStager
    enabled         :: Bool
    nas_paths       :: Vector{String}
    stage_dir       :: String
    lookahead       :: Int            # days ahead to keep staged
    keep_behind     :: Int            # processed days to retain before evicting
    cleanup_on_exit :: Bool
    staged          :: Dict{Int, String}   # idx → ready staged path
    tasks           :: Dict{Int, Task}     # idx → in-flight copy task
    failed          :: Set{Int}            # idx whose copy failed → use NAS path
    directory_lock  :: Union{Nothing,Pidfile.LockMonitor}
    closed          :: Bool
end

"""
    InputStager(binary_paths, staging_cfg)

Parse the `[input.staging]` sub-table and build the stager. Keys (all optional):

- `enabled`          : Bool   (default `false` ⇒ no staging, NAS paths used)
- `dir`              : String (REQUIRED when enabled) — local NVMe directory
- `lookahead_days`   : Int    (default `2`)
- `keep_behind_days` : Int    (default `0`)
- `cleanup_on_exit`  : Bool   (default `true`)

One run owns the staging directory at a time. If ownership or directory creation
fails, use the original source paths. Retained copies are reused only when their
recorded source metadata agrees; forcing files must stay immutable during a run.
"""
function InputStager(binary_paths::Vector{String}, staging_cfg::AbstractDict)
    enabled = _config_bool(staging_cfg, "enabled", false, "[input.staging].enabled")
    if !enabled
        return InputStager(false, binary_paths, "", 0, 0, false,
                           Dict{Int, String}(), Dict{Int, Task}(), Set{Int}(), nothing, false)
    end
    dir = get(staging_cfg, "dir", "")
    isempty(dir) && throw(ArgumentError(
        "[input.staging] enabled=true requires `dir` (a local NVMe directory)"))
    dir = expanduser(String(dir))
    lookahead   = Int(get(staging_cfg, "lookahead_days", 2))
    keep_behind = Int(get(staging_cfg, "keep_behind_days", 0))
    lookahead >= 0   || throw(ArgumentError("[input.staging] lookahead_days must be ≥ 0"))
    keep_behind >= 0 || throw(ArgumentError("[input.staging] keep_behind_days must be ≥ 0"))
    cleanup = _config_bool(staging_cfg, "cleanup_on_exit", true,
                           "[input.staging].cleanup_on_exit")
    # A shared target cannot safely serve two rolling eviction policies.
    # Keep transport available via NAS if another run owns this directory.
    lock_error = nothing
    directory_lock = try
        mkpath(dir)
        Pidfile.trymkpidlock(joinpath(dir, ".atmostransport-staging.pid"); stale_age=300)
    catch err
        lock_error = err
        false
    end
    if directory_lock === false
        @warn "Input staging directory is unavailable or in use; using NAS paths" dir exception=lock_error
        return InputStager(false,binary_paths,dir,0,0,false,
                           Dict{Int,String}(),Dict{Int,Task}(),Set{Int}(),nothing,false)
    end
    _check_staging_capacity(dir, binary_paths, lookahead + 1 + keep_behind)
    @info "Input staging enabled" dir lookahead_days=lookahead keep_behind_days=keep_behind max_staged_days=(lookahead + 1 + keep_behind)
    return InputStager(true, binary_paths, dir, lookahead, keep_behind, cleanup,
                       Dict{Int, String}(), Dict{Int, Task}(), Set{Int}(), directory_lock, false)
end

# Warn (do not fail) if the NVMe target may be too small for the rolling window.
function _check_staging_capacity(dir::AbstractString, paths::Vector{String}, max_days::Int)
    day_bytes = try filesize(first(paths)) catch; 0 end
    day_bytes == 0 && return nothing
    need = day_bytes * max_days
    free = try
        # statvfs is not in Base; the `avail` column of df gives free bytes.
        parse(Int, last(split(readchomp(`df -B1 --output=avail $dir`), '\n')))
    catch
        -1
    end
    if free >= 0 && free < need
        @warn "Input staging dir may be too small for the rolling window" dir need_GiB = round(need / 2^30, digits = 1) free_GiB = round(free / 2^30, digits = 1)
    end
    return nothing
end

# Staged filename = <hash-of-source-path>_<basename>. The hash prefix keeps the
# filenames distinct in practice when an explicit `binary_paths` list contains files
# from different directories that share a basename. Metadata also checks the
# full source path before reuse. The hash is stable within a Julia version, so a leftover staged file can be
# reused after checking its recorded source metadata rather than re-copied.
_staged_basename(src::AbstractString) =
    string(string(hash(src); base = 16), "_", basename(src))
_staged_path(mgr::InputStager, idx::Int) =
    joinpath(mgr.stage_dir, _staged_basename(mgr.nas_paths[idx]))

_staged_metadata_path(dst::AbstractString) = dst * ".source.toml"
function _source_stamp(src::AbstractString)
    info = stat(src)
    isfile(info) || throw(ArgumentError("staging source is not a regular file: $src"))
    return Dict{String,Any}("version"=>1,"source"=>abspath(src),"size"=>info.size,
                            "mtime"=>info.mtime,"ctime"=>info.ctime,
                            "inode"=>string(info.inode))
end

function _can_reuse_staged(src::AbstractString, dst::AbstractString)
    isfile(dst) && isfile(src) && isfile(_staged_metadata_path(dst)) || return false
    try
        stamp = _source_stamp(src)
        return filesize(dst) == stamp["size"] &&
               TOML.parsefile(_staged_metadata_path(dst)) == stamp
    catch
        return false
    end
end

function _inflight_copy(mgr::InputStager, src::AbstractString)
    for (idx,task) in mgr.tasks
        mgr.nas_paths[idx] == src && return task
    end
    return nothing
end

# Background copy NAS → NVMe, verified by size. Returns the staged path, or
# throws (caught by the caller, which falls back to the NAS path).
#
# Uses O_DIRECT (`dd iflag=direct oflag=direct`) when available: a buffered `cp`
# double-buffers through the page cache and contends with the running transport's
# own NVMe I/O (measured ~0.78 GB/s in-run, so the copy can't keep ahead of the
# day loop); O_DIRECT streams NAS→NVMe without polluting the cache (~1.38 GB/s).
# Falls back to `cp` if `dd` is unavailable or errors (e.g. O_DIRECT unsupported
# on the target filesystem).
function _stage_copy(src::AbstractString, dst::AbstractString)
    stamp = _source_stamp(src)
    atomic_publish(dst) do tmp
        ok = try
            run(pipeline(`dd if=$src of=$tmp bs=8M iflag=direct oflag=direct status=none`;
                         stderr=devnull))
            filesize(tmp) == stamp["size"]
        catch
            false
        end
        ok || cp(src, tmp; force = true)   # fallback: buffered copy
        filesize(tmp) == stamp["size"] || throw(ErrorException(
            "staged copy size mismatch for $src"))
        _source_stamp(src) == stamp || throw(ErrorException(
            "staging source changed during copy: $src"))
    end
    atomic_publish(_staged_metadata_path(dst)) do tmp
        open(tmp, "w") do io
            TOML.print(io, stamp)
        end
    end
    return dst
end

# Spawn copy tasks for days [idx, idx+lookahead] that are neither staged nor in flight.
function _spawn_ahead!(mgr::InputStager, idx::Int)
    n = length(mgr.nas_paths)
    for j in idx:min(idx + mgr.lookahead, n)
        (haskey(mgr.staged, j) || haskey(mgr.tasks, j) || j in mgr.failed) && continue
        src = mgr.nas_paths[j]
        dst = _staged_path(mgr, j)
        # Size alone cannot detect a rewritten source with the same byte count.
        if _can_reuse_staged(src,dst)
            mgr.staged[j] = dst
            continue
        end
        # Explicit binary lists may repeat a source for a replay/spin-up run.
        # Share its copy task instead of racing on the same temporary path.
        task = _inflight_copy(mgr,src)
        mgr.tasks[j] = task === nothing ? Threads.@spawn(_stage_copy(src,dst)) : task
    end
    return nothing
end

# Evict staged day-files strictly older than (idx - keep_behind).
function _evict_behind!(mgr::InputStager, idx::Int)
    cutoff = idx - mgr.keep_behind
    for j in collect(keys(mgr.staged))
        if j < cutoff
            path = mgr.staged[j]
            delete!(mgr.staged, j)
            # A repeated source can still belong to a current or future day.
            referenced = path in values(mgr.staged) ||
                         any(k -> _staged_path(mgr,k) == path, keys(mgr.tasks))
            if !referenced
                rm(path;force=true)
                rm(_staged_metadata_path(path);force=true)
            end
        end
    end
    return nothing
end

"""
    staged_path_for!(mgr, idx) -> String

Return the path the day-`idx` driver should open. With staging enabled this
ensures day `idx` is on local NVMe (blocking only if its copy is still in
flight), kicks off async copies for the look-ahead window, and evicts processed
days. Falls back to the NAS path if the copy failed. With staging disabled it
returns the original NAS path unchanged.
"""
function staged_path_for!(mgr::InputStager, idx::Int)
    mgr.closed && throw(ArgumentError("input stager has already been closed"))
    mgr.enabled || return mgr.nas_paths[idx]
    _spawn_ahead!(mgr, idx)
    if haskey(mgr.tasks, idx)
        task = mgr.tasks[idx]
        delete!(mgr.tasks, idx)
        try
            mgr.staged[idx] = fetch(task)
        catch err
            @warn "Input staging failed; falling back to NAS for this day" day = idx exception = err
            push!(mgr.failed, idx)
        end
    end
    _evict_behind!(mgr, idx)
    return get(mgr.staged, idx, mgr.nas_paths[idx])
end

"""
    cleanup_staging!(mgr)

Wait for any in-flight copies and remove all remaining staged files (when
`cleanup_on_exit`). Safe to call unconditionally at run end.
"""
function cleanup_staging!(mgr::InputStager)
    (mgr.enabled && !mgr.closed) || return nothing
    try
        for (j, task) in mgr.tasks
            try
                mgr.staged[j] = fetch(task)
            catch
                # Copy failure already removes its own partial files.
            end
        end
        if mgr.cleanup_on_exit
            for p in values(mgr.staged)
                rm(p;force=true)
                rm(_staged_metadata_path(p);force=true)
            end
        end
    finally
        empty!(mgr.tasks)
        empty!(mgr.staged)
        mgr.closed = true
        mgr.directory_lock === nothing || close(mgr.directory_lock)
        mgr.directory_lock = nothing
    end
    return nothing
end

end # module InputStaging
