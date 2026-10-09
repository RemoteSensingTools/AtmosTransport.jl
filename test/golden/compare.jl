# =============================================================================
# compare.jl — exact comparison of two golden output trees
# =============================================================================
#
# A refactor must reproduce every golden output bit for bit. Files are matched
# by relative path inside a case directory and compared by type:
#
#   * NetCDF (*.nc): every variable's element type, dimensions and stored values
#     (floats bit for bit: a NaN payload or the sign of a zero counts), and every
#     attribute with its type, except the provenance attributes (time, commit,
#     host, ...).
#   * Transport binaries (*.bin): the JSON header without its provenance keys,
#     and every payload bit. A payload difference is summarised as the number
#     of differing values and the largest absolute difference.
#   * Anything else: byte for byte, except the harness's own bookkeeping at the
#     top of the case directory and directories whose name starts with "_"
#     (caches such as regridding weights).
#
# The comparison reads the files directly (NCDatasets, JSON3, Mmap), not
# through AtmosTransport, so a refactor of the readers cannot hide a change.
# =============================================================================

using NCDatasets, JSON3, Mmap, Printf

# Attributes and header keys that record when, where and by which commit a file
# was written; they differ between two runs of identical code.
const PROVENANCE_ATTRIBUTES = Set(["creation_date", "framework_commit", "framework_dirty",
                                   "runtime", "hostname", "user", "history"])
const PROVENANCE_KEYS = Set(["git_commit", "git_dirty", "creation_time", "script_path",
                             "script_mtime_unix", "generation_fingerprint", "dirty_nonce"])
const BOOKKEEPING_FILES = Set(["config.toml", "log.txt", "status.toml"])

"""
    compare_case(ref_dir, new_dir) -> Vector{String}

Differences between the outputs of one golden case in `ref_dir` and `new_dir`;
empty if every file is identical.
"""
function compare_case(ref_dir::AbstractString, new_dir::AbstractString)
    compared(path) = !(path in BOOKKEEPING_FILES) && !any(startswith("_"), splitpath(dirname(path)))
    files(dir) = Set(filter(compared, [relpath(joinpath(root, f), dir) for (root, _, fs) in walkdir(dir) for f in fs]))
    ref, new = files(ref_dir), files(new_dir)
    isempty(ref) && isempty(new) && return ["no outputs to compare"]
    diffs = ["only in reference: $f" for f in sort!(collect(setdiff(ref, new)))]
    append!(diffs, ["only in new: $f" for f in sort!(collect(setdiff(new, ref)))])
    for f in sort!(collect(intersect(ref, new)))
        a, b = joinpath(ref_dir, f), joinpath(new_dir, f)
        found = endswith(f, ".nc")  ? compare_netcdf(a, b) :
                endswith(f, ".bin") ? compare_binary(a, b) :
                read(a) == read(b)  ? String[] : ["contents differ"]
        append!(diffs, ["$f: $d" for d in found])
    end
    return diffs
end

shorten(x; n = 120) = (s = repr(x); length(s) > n ? first(s, n) * "…" : s)

# Bitwise equality of floats (NaN payloads and signed zeros count), `isequal` otherwise.
same_bits(x::T, y::T) where {T <: Base.IEEEFloat} = reinterpret(Base.uinttype(T), x) == reinterpret(Base.uinttype(T), y)
same_bits(x, y) = isequal(x, y)
same_bits(p::AbstractArray{T}, q::AbstractArray{T}) where {T <: Base.IEEEFloat} =
    reinterpret(Base.uinttype(T), p) == reinterpret(Base.uinttype(T), q)
same_bits(p::AbstractArray, q::AbstractArray) = isequal(p, q)

# --- NetCDF -------------------------------------------------------------------

function compare_netcdf(a::AbstractString, b::AbstractString)
    diffs = String[]
    NCDataset(a) do da
        NCDataset(b) do db
            compare_attributes!(diffs, "global", da.attrib, db.attrib)
            va, vb = Set(keys(da)), Set(keys(db))
            va == vb || push!(diffs, "variables in one file only: $(sort!(collect(symdiff(va, vb))))")
            for name in sort!(collect(intersect(va, vb)))
                x, y = da[name].var, db[name].var                 # stored values: no fill or scale handling
                compare_attributes!(diffs, name, x.attrib, y.attrib)
                eltype(x) == eltype(y) || (push!(diffs, "$name: element type $(eltype(x)) vs $(eltype(y))"); continue)
                dimnames(x) == dimnames(y) && size(x) == size(y) ||
                    (push!(diffs, "$name: dimensions $(dimnames(x)) $(size(x)) vs $(dimnames(y)) $(size(y))"); continue)
                p, q = Array(x), Array(y)
                same_bits(p, q) || push!(diffs, describe_difference(name, p, q))
            end
        end
    end
    return diffs
end

function compare_attributes!(diffs, owner, a, b)
    ka = setdiff(Set(keys(a)), PROVENANCE_ATTRIBUTES)
    kb = setdiff(Set(keys(b)), PROVENANCE_ATTRIBUTES)
    ka == kb || push!(diffs, "$owner attributes in one file only: $(sort!(collect(symdiff(ka, kb))))")
    for k in sort!(collect(intersect(ka, kb)))
        typeof(a[k]) == typeof(b[k]) && isequal(a[k], b[k]) ||
            push!(diffs, "$owner attribute $k: $(shorten(a[k])) vs $(shorten(b[k]))")
    end
    return diffs
end

"""
    describe_difference(name, ref, new) -> String

How many values differ and by how much: the largest absolute difference and
the largest magnitude of the reference, over finite values. (A differing NaN
payload or sign of zero counts as a differing value with |Δ| = 0.)
"""
function describe_difference(name, ref::AbstractArray{<:Real}, new::AbstractArray{<:Real})
    differ, Δ, scale = 0, 0.0, 0.0
    for i in eachindex(ref, new)
        x, y = ref[i], new[i]
        same_bits(x, y) && continue
        differ += 1
        isfinite(x) && isfinite(y) && (Δ = max(Δ, abs(Float64(x) - Float64(y))))
    end
    for x in ref
        isfinite(x) && (scale = max(scale, abs(Float64(x))))
    end
    return @sprintf("%s: %d of %d values differ, max |Δ| = %.3e (max |ref| = %.3e)", name, differ, length(ref), Δ, scale)
end
describe_difference(name, ref, new) = "$name: values differ"

# --- transport binaries ---------------------------------------------------------

"""
    binary_header(bytes) -> Dict{String, Any}

The JSON header of a transport binary: the bytes up to the first NUL.
"""
function binary_header(bytes::AbstractVector{UInt8})
    stop = something(findfirst(iszero, bytes), length(bytes) + 1) - 1
    return JSON3.read(String(bytes[1:stop]), Dict{String, Any})
end

# The header without provenance keys, at any depth.
strip_provenance(d::AbstractDict) =
    Dict{String, Any}(k => strip_provenance(v) for (k, v) in d if !(k in PROVENANCE_KEYS))
strip_provenance(v::AbstractVector) = map(strip_provenance, v)
strip_provenance(x) = x

const PAYLOAD_TYPES = Dict("Float32" => (Float32, UInt32), "Float64" => (Float64, UInt64))

function compare_binary(a::AbstractString, b::AbstractString)
    diffs = String[]
    open(a) do fa
        open(b) do fb
            ha, hb = binary_header(Mmap.mmap(fa)), binary_header(Mmap.mmap(fb))
            sa, sb = strip_provenance(ha), strip_provenance(hb)
            ka, kb = Set(keys(sa)), Set(keys(sb))
            ka == kb || push!(diffs, "header keys in one file only: $(sort!(collect(symdiff(ka, kb))))")
            for k in sort!(collect(intersect(ka, kb)))
                isequal(sa[k], sb[k]) || push!(diffs, "header $k: $(shorten(sa[k])) vs $(shorten(sb[k]))")
            end
            filesize(fa) == filesize(fb) || (push!(diffs, "file sizes $(filesize(fa)) vs $(filesize(fb))"); return)
            # Payload: everything after the padded header, mapped in its float type
            # (bit patterns for the equality test, floats to describe a difference).
            offset, (T, U) = ha["header_bytes"], PAYLOAD_TYPES[ha["float_type"]]
            n, rest = divrem(filesize(fa) - offset, sizeof(T))
            rest == 0 || (read(fa) == read(fb) || push!(diffs, "payload differs"); return)
            Mmap.mmap(fa, Vector{U}, n, offset) == Mmap.mmap(fb, Vector{U}, n, offset) ||
                push!(diffs, describe_difference("payload", Mmap.mmap(fa, Vector{T}, n, offset),
                                                 Mmap.mmap(fb, Vector{T}, n, offset)))
        end
    end
    return diffs
end

"""
    nonfinite_values(dir) -> Dict{String, Int}

Number of non-finite stored values per NetCDF variable under `dir` (outside the
`_` caches); a reference with NaNs would compare equal to any other NaNs.
"""
function nonfinite_values(dir)
    found = Dict{String, Int}()
    for (root, _, fs) in walkdir(dir), f in fs
        endswith(f, ".nc") && !any(startswith("_"), splitpath(relpath(root, dir))) || continue
        NCDataset(joinpath(root, f)) do ds
            for (name, v) in ds
                eltype(v.var) <: AbstractFloat || continue
                n = count(!isfinite, Array(v.var))
                n > 0 && (found["$(relpath(joinpath(root, f), dir)):$name"] = n)
            end
        end
    end
    return found
end
