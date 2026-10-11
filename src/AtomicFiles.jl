"""
    AtomicFiles

Publish a file only when it is complete. A writer fills a *staging file* next
to the target and renames it onto the target when it has finished; on an error
the staging file is removed and a previous target is left untouched. A reader
of the target therefore sees the previous file or the complete new one, never
a partial file.

Staging file contract (one implementation for every writer in the package):

- **Name.** `"<abspath(path)>.tmp-<pid>-<n>"`, with `n` a random `UInt32`
  from libc's generator (`Libc.rand`), so the task-local `Random` stream of a
  seeded run does not depend on whether a file was written. The file is a
  sibling of the target, so the final rename stays on one filesystem and never
  copies.
- **Creation.** `open(2)` with `O_CREAT | O_EXCL`: no other process can own
  the name (an existing name is retried with a new `n`), and nothing is
  truncated or followed.
- **Mode.** Created with `0o666`, so the published file gets
  `0o666 & ~umask`, exactly like a file created with `open(path, "w")`.
  `mktemp` is never used on POSIX systems: it creates mode `0o600`, and the
  rename would publish an owner-only file. (On Windows, where POSIX modes do
  not apply, the staging file comes from `mktemp` in the target's directory.)
- **Publication.** One replacing rename of the staging file onto the target,
  on every Julia version: libuv's `uv_fs_rename` (through the public
  `Base.rename` on Julia ≥ 1.12, through `jl_fs_rename`, the call it wraps,
  on 1.10 and 1.11), which is
  `rename(2)` on POSIX systems and `MoveFileExW` with
  `MOVEFILE_REPLACE_EXISTING` on Windows. On POSIX systems the target is
  replaced atomically: the path never disappears, and a reader opens either
  the previous file or the complete new one. If the rename fails, the staging
  file is removed and the previous target is left as it was. Base's
  `mv(tmp, path; force = true)` is not used: it deletes the target and renames
  again when a rename fails (before renaming at all on Julia 1.10). Nor is
  Julia 1.10's `Base.Filesystem.rename`, which copies and deletes when the
  rename fails. A cross-device rename (`EXDEV`) is therefore an error, never
  a copy; since the staging file is a sibling of the target it does not occur
  in practice. An existing *directory* at `path` is refused.
- **Durability.** No `fsync`: the guarantee is towards concurrent readers, not
  across a power failure.

Entry points: [`atomic_publish`](@ref) for writers that take a path (JLD2,
NetCDF, `open(path, "w")`), and [`open_atomic_staging`](@ref),
[`commit_atomic_staging`](@ref) and [`abandon_atomic_staging`](@ref) for
writers that keep the stream open across calls (the streaming transport-binary
writers).

Uses Base only, so it loads before `Quantities`; `MetDrivers` (transport
binaries), `Regridding` (regridder caches) and `Preprocessing` (spectral
caches) use it.
"""
module AtomicFiles

export atomic_publish, open_atomic_staging, commit_atomic_staging, abandon_atomic_staging

# Attempts at a unique staging name before giving up; a collision needs another
# writer of the same target with the same pid and the same 32-bit draw.
const MAX_STAGING_ATTEMPTS = 100

_staging_name(target::AbstractString) =
    string(abspath(target), ".tmp-", getpid(), "-", Libc.rand(UInt32))

function _check_target(target::AbstractString)
    isdir(target) && throw(ArgumentError(
        "cannot publish a file over the directory $(repr(target))"))
    parent = dirname(abspath(target))
    isdir(parent) || throw(ArgumentError(
        "parent directory of $(repr(target)) does not exist: $(parent)"))
    return nothing
end

"""
    open_atomic_staging(path) -> (tmp::String, io::IOStream)

Create the staging file for `path` (see [`AtomicFiles`](@ref) for its name,
`O_EXCL` creation and mode `0o666 & ~umask`) and return its name with an
`IOStream` open for reading and writing. The descriptor returned by `open(2)`
becomes `io`; the staging path is never reopened.

The caller writes through `io`, closes it, and then calls
[`commit_atomic_staging`](@ref)`(tmp, path)`, or
[`abandon_atomic_staging`](@ref)`(tmp)` if anything failed. Throws an
`ArgumentError` if the parent directory of `path` does not exist or `path` is
a directory.
"""
function open_atomic_staging(path::AbstractString)
    target = String(path)
    _check_target(target)
    Sys.iswindows() && return mktemp(dirname(abspath(target)); cleanup = false)
    flags = Base.Filesystem.JL_O_CREAT | Base.Filesystem.JL_O_EXCL |
            Base.Filesystem.JL_O_RDWR | Base.Filesystem.JL_O_CLOEXEC
    for _ in 1:MAX_STAGING_ATTEMPTS
        tmp = _staging_name(target)
        fd = ccall(:open, Cint, (Cstring, Cint, Cuint...), tmp, flags, 0o666)
        if fd >= 0
            try
                return tmp, fdio(tmp, fd, true)
            catch
                ccall(:close, Cint, (Cint,), fd)
                rm(tmp; force = true)
                rethrow()
            end
        end
        errno = Libc.errno()
        errno == Libc.EEXIST || throw(SystemError("creating staging file $(tmp)", errno))
    end
    throw(ErrorException(
        "could not create a unique staging file for $(target) in $(MAX_STAGING_ATTEMPTS) attempts"))
end

"""
    commit_atomic_staging(tmp, path) -> String

Publish the closed staging file `tmp` (from [`open_atomic_staging`](@ref)) as
`path` by one replacing rename (see [`AtomicFiles`](@ref)), so a previous file
at `path` stays readable until the new one takes its place. If `path` has
become a directory (`ArgumentError`) or the rename fails (`Base.IOError`, also
for a cross-device rename), `tmp` is removed, a previous file at `path` is
left as it was, and the error is thrown. Returns `path`.
"""
function commit_atomic_staging(tmp::AbstractString, path::AbstractString)
    target = String(path)
    try
        isdir(target) && throw(ArgumentError(
            "cannot publish a file over the directory $(repr(target))"))
        _replacing_rename(String(tmp), target)
    catch
        abandon_atomic_staging(tmp)
        rethrow()
    end
    return target
end

# One replacing rename and nothing else: rename(2) on POSIX systems,
# MoveFileExW(MOVEFILE_REPLACE_EXISTING) on Windows. A failure throws an
# `IOError` carrying the libuv error code and never deletes or copies anything.
function _replacing_rename(tmp::String, target::String)
    err = _uv_rename(tmp, target)
    err == Base.UV_EXDEV && Base.uv_error(
        "rename($(repr(tmp)), $(repr(target))): the staging file and the target " *
        "are on different filesystems, and AtomicFiles does not copy", err)
    Base.uv_error("rename($(repr(tmp)), $(repr(target)))", err)   # throws only if err < 0
    return nothing
end

# The libuv rename as an error code (0, or a negative libuv error). Julia 1.12
# makes it public as `Base.rename`; on 1.10 and 1.11 `Base.Filesystem.rename`
# copies when the rename fails, so the same libuv call is made directly there.
@static if VERSION >= v"1.12"
    function _uv_rename(tmp::String, target::String)
        try
            Base.rename(tmp, target)
        catch err
            err isa Base.IOError || rethrow()
            return Int32(err.code)
        end
        return Int32(0)
    end
else
    _uv_rename(tmp::String, target::String) =
        ccall(:jl_fs_rename, Int32, (Cstring, Cstring), tmp, target)
end

"""
    abandon_atomic_staging(tmp) -> Nothing

Remove the staging file `tmp` after a failed write; a missing file is not an
error. The caller closes its stream first. The target is not touched.
"""
function abandon_atomic_staging(tmp::AbstractString)
    rm(tmp; force = true)
    return nothing
end

"""
    atomic_publish(f, path) -> String

Write `path` atomically: `f(tmp)` writes the complete file at the staging path
`tmp` (any writer that takes a path, e.g. `open(tmp, "w")`, `JLD2.jldsave` or
`NCDataset(tmp, "c")`), which is then renamed onto `path`. If `f` throws, `tmp`
is removed, `path` is left as it was, and the error is rethrown. The staging
file is created by [`open_atomic_staging`](@ref) (`O_EXCL`, mode
`0o666 & ~umask`) and closed before `f` runs; `f` may truncate or replace it.
Returns `path`.

```julia
atomic_publish(path) do tmp
    open(io -> write(io, payload), tmp, "w")
end
```
"""
function atomic_publish(f, path::AbstractString)
    target = String(path)
    tmp, io = open_atomic_staging(target)
    try
        close(io)
        f(tmp)
    catch
        abandon_atomic_staging(tmp)
        rethrow()
    end
    return commit_atomic_staging(tmp, target)
end

end # module AtomicFiles
