# ---------------------------------------------------------------------------
# Weight construction, caching, and serialization
#
# - build_regridder          : identity → nested → CR.jl Regridder (SHA-1 cache
#                              key + JLD2 cache for the last)
# - default_regrid_cache_dir : the shared weight cache of runtime and builders
# - save_regridder           : JLD2 round-trip, published atomically
# - load_regridder           : JLD2 round-trip
# - apply_regridder!         : n-D array wrapper around CR.jl regrid!
# ---------------------------------------------------------------------------

# --- cache key ----------------------------------------------------------------

"""
    _hash_mesh!(io, mesh)

Serialize a mesh's structural fingerprint to `io` so that SHA-1 of the buffer
can be used as a cache key. We purposely omit fields that do not affect the
regridding geometry (halo widths, precomputed metric terms, etc).
"""
function _hash_mesh!(io::IO, m::LatLonMesh)
    write(io, "LatLonMesh\0")
    write(io, Int64(m.Nx), Int64(m.Ny))
    write(io, Float64.(m.λᶠ))
    write(io, Float64.(m.φᶠ))
    write(io, Float64(m.radius))
    return io
end

function _hash_mesh!(io::IO, m::CubedSphereMesh)
    write(io, "CubedSphereMesh\0")
    write(io, Int64(m.Nc))
    write(io, String(cs_definition_tag(cs_definition(m))) * "\0")
    write(io, coordinate_law_tag(coordinate_law(m)) * "\0")
    write(io, center_law_tag(center_law(m)) * "\0")
    write(io, String(nameof(typeof(m.convention))) * "\0")
    write(io, Float64(longitude_offset_deg(cs_definition(m))))
    write(io, Float64(m.radius))
    return io
end

function _hash_mesh!(io::IO, m::ReducedGaussianMesh)
    write(io, "ReducedGaussianMesh\0")
    write(io, Int64.(m.nlon_per_ring))
    write(io, Float64.(m.latitudes))
    write(io, Float64.(m.lat_faces))
    write(io, Float64(m.radius))
    return io
end

# v3: cubed-sphere corners in Float64 for every mesh precision; v2 caches
# could hold weights from Float32 corners under a precision-free key.
# Nested cubed-sphere pairs return a `NestedCubedSphereRegrid` before the cache
# is consulted, so nothing is cached for them; the cached content of every
# other pair is unchanged, so the version stays 3 (v3 files of nested pairs are
# simply no longer read).
const _REGRIDDER_CACHE_VERSION = 3

"""
    default_regrid_cache_dir() -> String

Directory of the shared conservative-regridding weight cache: the value of
`ATMOSTR_REGRID_CACHE_DIR` if set and not empty (the golden-output harness
points it into each case, so weights are recomputed), else
`~/.cache/AtmosTransport/cr_regridding`, with `~` expanded. Read at each call,
so the precompiled package carries no home directory. Runtime surface fluxes
pass it as `cache_dir` to [`build_regridder`](@ref).
"""
function default_regrid_cache_dir()
    dir = get(ENV, "ATMOSTR_REGRID_CACHE_DIR", "")
    return expanduser(isempty(dir) ? joinpath("~", ".cache", "AtmosTransport", "cr_regridding") : dir)
end

function _regridder_cache_key(src, dst; normalize::Bool)
    io = IOBuffer()
    write(io, "atmostransport_regridder_v$(_REGRIDDER_CACHE_VERSION)\0")
    write(io, normalize ? 0x01 : 0x00)
    _hash_mesh!(io, src)
    write(io, "|\0")
    _hash_mesh!(io, dst)
    return bytes2hex(SHA.sha1(take!(io)))
end

# --- build_regridder ----------------------------------------------------------

"""
    build_regridder(src_mesh, dst_mesh; normalize=false, cache_dir=nothing, kwargs...)
        -> IdentityRegrid | NestedCubedSphereRegrid | Regridder

Construct the regridder that maps source-mesh fields to destination-mesh
fields, choosing in this order:

1. [`IdentityRegrid`](@ref) when [`meshes_equivalent`](@ref)`(src, dst)`;
2. [`NestedCubedSphereRegrid`](@ref) when [`meshes_nested`](@ref)`(src, dst)`
   (cubed spheres of one definition whose `Nc` differ by an integer factor):
   exact block remapping, no polygon intersection;
3. otherwise a `ConservativeRegridding.Regridder` built by true spherical
   polygon intersection.

The first two take no weights from or to disk and ignore `normalize` and
`kwargs`. If `cache_dir` is supplied, a conservative `Regridder` is cached on
disk using a SHA-1 fingerprint of the `(src_mesh, dst_mesh)` pair (see
[`default_regrid_cache_dir`](@ref) for the shared default). Subsequent calls
with identical meshes load the cached weights instead of recomputing them.
Custom construction `kwargs` cannot be combined with `cache_dir`, because
arbitrary operators do not have a stable, portable cache fingerprint.

`normalize=false` matches the convention used by `xESMF`: the sparse
intersections matrix contains raw area values (m²), and the per-cell
normalization by `dst_areas` happens inside `regrid!`. Pass `normalize=true`
to scale `intersections`, `src_areas`, and `dst_areas` by `max(intersections)`
— useful for display but not for weight-level comparisons.

Any extra `kwargs` are forwarded to `ConservativeRegridding.Regridder`.
"""
function build_regridder(src, dst;
                         normalize::Bool = false,
                         cache_dir::Union{Nothing, AbstractString} = nothing,
                         kwargs...)
    cache_dir !== nothing && !isempty(kwargs) && throw(ArgumentError(
        "cache_dir cannot be combined with custom regridder keyword arguments; " *
        "omit cache_dir or call build_regridder without extra kwargs"))
    ir = identity_regrid_or_nothing(src, dst)
    ir === nothing || return ir
    nr = nested_regrid_or_nothing(src, dst)
    nr === nothing || return nr
    if cache_dir !== nothing
        mkpath(cache_dir)
        key = _regridder_cache_key(src, dst; normalize)
        path = joinpath(cache_dir, "regridder_" * key * ".jld2")
        if isfile(path)
            @info "Loading cached regridder" path
            return load_regridder(path)
        end
        @info "Building regridder (will cache to disk)" path
        r = ConservativeRegridding.Regridder(dst, src; normalize, kwargs...)
        save_regridder(path, r)
        return r
    end
    return ConservativeRegridding.Regridder(dst, src; normalize, kwargs...)
end

# --- save / load via JLD2 -----------------------------------------------------

"""
    save_regridder(path, regridder) -> path

Write the regridder's intersection matrix and area vectors to `path` as a
JLD2 file. Temporary work arrays are not serialized (they're rebuilt on
load). The file is published with `AtomicFiles.atomic_publish`: a reader sees
the previous file or the complete new one, and the file gets mode
`0o666 & ~umask` like any file the user creates, so a shared cache stays
readable by the group.
"""
function save_regridder(path::AbstractString, r::Regridder)
    mkpath(dirname(abspath(path)))
    return atomic_publish(path) do tmp
        JLD2.jldsave(tmp;
            intersections = r.intersections,
            dst_areas     = r.dst_areas,
            src_areas     = r.src_areas,
            format_version = _REGRIDDER_CACHE_VERSION,
        )
    end
end

"""
    load_regridder(path) -> Regridder

Inverse of [`save_regridder`](@ref). Rebuilds the temporary work arrays from
the area vectors.
"""
function load_regridder(path::AbstractString)
    data = JLD2.load(path)
    haskey(data, "format_version") || error("not a regridder JLD2 file: $path")
    v = data["format_version"]
    v == _REGRIDDER_CACHE_VERSION ||
        error("regridder cache format version mismatch: expected $(_REGRIDDER_CACHE_VERSION), got $v")

    intersections = data["intersections"]
    dst_areas = data["dst_areas"]
    src_areas = data["src_areas"]
    dst_temp = zeros(eltype(dst_areas), length(dst_areas))
    src_temp = zeros(eltype(src_areas), length(src_areas))
    return Regridder(intersections, dst_areas, src_areas, dst_temp, src_temp)
end

# --- apply_regridder! ---------------------------------------------------------

"""
    apply_regridder!(dst, regridder, src) -> dst

Apply a regridder to field data. Thin wrapper over
`ConservativeRegridding.regrid!` that handles n-dimensional arrays whose
**first dimension is the flattened horizontal index** with any number of
trailing dimensions (levels, time windows, tracers, etc).

`dst` and `src` may be:
- plain `Vector`s of length `n_dst` / `n_src`, or
- multi-dimensional arrays with a leading horizontal axis of length
  `n_dst` / `n_src` and matching trailing shape.

For the multi-dim case the horizontal regridding is applied column-by-column
using CR.jl's temporary work arrays — no extra allocations per column.

The field is treated as intensive (a value per area or per point). Per-cell
totals go through `apply_regridder!(dst, r, src, ExtensiveCellField())`
(`quantity_regrid.jl`). `IdentityRegrid` and `NestedCubedSphereRegrid` have
their own three-argument methods with the same contract.
"""
function apply_regridder!(dst::AbstractVector, r::Regridder, src::AbstractVector)
    return ConservativeRegridding.regrid!(dst, r, src)
end

# `x` as a matrix of horizontal columns (leading length `n`, checked by the
# caller). A matrix is used as it is: `reshape` would allocate a new array
# header on every call (Julia ≥ 1.11).
_flat_columns(x::AbstractMatrix, n::Int) = x
_flat_columns(x::AbstractArray, n::Int) = reshape(x, n, :)

function apply_regridder!(dst::AbstractArray, r::Regridder, src::AbstractArray)
    n_dst = length(r.dst_areas)
    n_src = length(r.src_areas)
    size(dst, 1) == n_dst ||
        error("dst leading dim $(size(dst,1)) ≠ regridder n_dst $n_dst")
    size(src, 1) == n_src ||
        error("src leading dim $(size(src,1)) ≠ regridder n_src $n_src")

    dst_flat = _flat_columns(dst, n_dst)
    src_flat = _flat_columns(src, n_src)
    size(dst_flat, 2) == size(src_flat, 2) ||
        error("trailing shape mismatch: dst $(size(dst_flat,2)) ≠ src $(size(src_flat,2))")

    @inbounds for col in axes(dst_flat, 2)
        ConservativeRegridding.regrid!(
            view(dst_flat, :, col), r, view(src_flat, :, col),
        )
    end
    return dst
end
