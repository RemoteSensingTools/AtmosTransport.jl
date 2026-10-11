# ---------------------------------------------------------------------------
# Staging of compressed inputs
#
# Some TRENDY v14 submissions are compressed: SDGVM and VISIT as `.nc.gz`,
# iMAPLE as `.nc.tar.gz` with exactly one member per archive; all decompress
# to NETCDF3_CLASSIC (P0 facts memo, Section 4). They are decompressed once
# into a caller-chosen staging directory, keyed by the source file's identity,
# and published atomically. The system `gzip` and `tar` do the work; no Julia
# compression package is a dependency.
# ---------------------------------------------------------------------------

abstract type AbstractCompression end
struct Uncompressed <: AbstractCompression end
struct GzipCompressed <: AbstractCompression end
struct TarGzipCompressed <: AbstractCompression end

function _compression(path::AbstractString)
    name = lowercase(basename(path))
    (endswith(name, ".tar.gz") || endswith(name, ".tgz")) && return TarGzipCompressed()
    endswith(name, ".gz") && return GzipCompressed()
    return Uncompressed()
end

function _required_program(name::AbstractString, path::AbstractString)
    exe = Sys.which(name)
    exe === nothing && throw(ErrorException(
        "staging $(path) needs the program `$(name)`, which is not on the PATH"))
    return exe
end

# Cache key: SHA-1 of the source's absolute path, size and modification time.
# A changed source gets a new key (and a new staged file).
function _staging_key(path::AbstractString)
    st = stat(path)
    return bytes2hex(sha1(string(abspath(path), '\0', st.size, '\0', repr(st.mtime))))
end

# Name of the decompressed file inside the archive (or of the `.gz` file).
_staged_basename(::GzipCompressed, path) = String(chop(basename(path); tail = length(".gz")))

function _staged_basename(::TarGzipCompressed, path)
    tar = _required_program("tar", path)
    members = filter(m -> !isempty(m) && !endswith(m, "/"), readlines(`$tar -tzf $path`))
    length(members) == 1 || throw(ArgumentError(
        "$(path) must hold exactly one file to be staged; it holds $(length(members))" *
        (isempty(members) ? "" : ": $(join(members, ", "))")))
    return only(members)
end

_decompress_command(::GzipCompressed, path, member) = `$(_required_program("gzip", path)) -dc -- $path`
_decompress_command(::TarGzipCompressed, path, member) =
    `$(_required_program("tar", path)) -xzOf $path -- $member`

# NetCDF classic ("CDF\x01", "CDF\x02", "CDF\x05") or NetCDF-4/HDF5 ("\x89HDF").
function _check_netcdf_magic(file::AbstractString, source::AbstractString)
    magic = open(io -> read(io, 4), file)
    ok = (length(magic) == 4 && magic[1:3] == UInt8['C', 'D', 'F'] && magic[4] in (0x01, 0x02, 0x05)) ||
         magic == UInt8[0x89, 'H', 'D', 'F']
    ok || throw(ArgumentError("decompressing $(source) did not give a NetCDF file"))
    return nothing
end

"""
    stage_input(path; staging_dir) -> String

A path at which the input file `path` can be opened as NetCDF:

- an uncompressed file (any name not ending in `.gz`, `.tar.gz` or `.tgz`) is
  returned as it is;
- `*.gz` is decompressed with `gzip -dc`;
- `*.tar.gz` / `*.tgz` must hold exactly one file, extracted with
  `tar -xzOf`.

The decompressed file is written to `staging_dir` (created if needed; there is
no default, in particular not `/tmp`) as `<key>-<name>`, where `key` is the
SHA-1 of the source's absolute path, size and modification time, so a staged
file is reused until its source changes. It is published with
`AtomicFiles.atomic_publish` only after it decompressed completely and starts
with a NetCDF signature, so a concurrent or interrupted stager never leaves a
partial file. `gzip` and `tar` must be on the `PATH`. Staged files are not
removed; the staging directory is the caller's.
"""
function stage_input(path::AbstractString; staging_dir::AbstractString)
    kind = _compression(path)
    kind isa Uncompressed && return String(path)
    isfile(path) || throw(ArgumentError("input file not found: $(path)"))
    member = _staged_basename(kind, path)
    mkpath(staging_dir)
    target = joinpath(staging_dir, string(_staging_key(path), '-', basename(member)))
    isfile(target) && return target
    atomic_publish(target) do tmp
        run(pipeline(_decompress_command(kind, path, member); stdout = tmp))
        _check_netcdf_magic(tmp, path)
    end
    return target
end
