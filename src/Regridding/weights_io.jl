# ---------------------------------------------------------------------------
# Weight construction, caching, and serialization
#
# - build_regridder      : CR.jl Regridder + SHA-1 cache key + JLD2 cache
# - save_regridder       : JLD2 round-trip
# - load_regridder       : JLD2 round-trip
# - save_esmf_weights    : ESMF offline-weights NetCDF (S, row, col, frac_*)
# - apply_regridder!     : n-D array wrapper around CR.jl regrid!
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
const _REGRIDDER_CACHE_VERSION = 3

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
    build_regridder(src_mesh, dst_mesh; normalize=false, cache_dir=nothing, kwargs...) -> Regridder

Construct a `ConservativeRegridding.Regridder` that maps source-mesh fields to
destination-mesh fields by true spherical polygon intersection.

If `cache_dir` is supplied, the regridder is cached on disk using a SHA-1
fingerprint of the `(src_mesh, dst_mesh)` pair. Subsequent calls with
identical meshes load the cached weights instead of recomputing them.
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
load).
"""
function save_regridder(path::AbstractString, r::Regridder)
    mkpath(dirname(path))
    temp_path, temp_io = mktemp(dirname(path))
    close(temp_io)
    try
        JLD2.jldsave(temp_path;
            intersections = r.intersections,
            dst_areas     = r.dst_areas,
            src_areas     = r.src_areas,
            format_version = _REGRIDDER_CACHE_VERSION,
        )
        mv(temp_path, path; force = true)
    finally
        isfile(temp_path) && rm(temp_path; force = true)
    end
    return path
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

# --- ESMF NetCDF weights ------------------------------------------------------

"""
    save_esmf_weights(path, regridder;
                      src_shape=nothing, dst_shape=nothing,
                      src_grid_name="source", dst_grid_name="destination") -> path

Export the regridder weights to an ESMF offline-weights NetCDF file.

Format (ESMF convention):

| variable | shape   | meaning                                                         |
|----------|---------|-----------------------------------------------------------------|
| `S`      | `(n_s,)`| weight value `S[k] = intersections[row[k], col[k]] / dst_areas[row[k]]` |
| `row`    | `(n_s,)`| destination cell index (1-indexed)                              |
| `col`    | `(n_s,)`| source cell index (1-indexed)                                   |
| `frac_a` | `(n_a,)`| fraction of source cell covered by the destination grid         |
| `frac_b` | `(n_b,)`| fraction of destination cell covered by the source grid         |
| `area_a` | `(n_a,)`| source cell areas (mesh units, typically m²)                    |
| `area_b` | `(n_b,)`| destination cell areas                                          |

This format is what `xESMF` / `ESMF_RegridWeightGen` / GCHP's tile-file
loader all consume, which lets the CR.jl output be diffed against xESMF
fixtures at weight level and handed off to ESMF-based tooling.

Optional `src_shape` / `dst_shape` tuples (e.g. `(Nx, Ny)`,
`(Nc, Nc, 6)`) are stored as global attributes for provenance.
"""
function save_esmf_weights(path::AbstractString, r::Regridder;
                           src_shape::Union{Nothing, Tuple} = nothing,
                           dst_shape::Union{Nothing, Tuple} = nothing,
                           src_grid_name::AbstractString = "source",
                           dst_grid_name::AbstractString = "destination")
    A         = r.intersections
    src_areas = r.src_areas
    dst_areas = r.dst_areas

    n_a = length(src_areas)   # source cells
    n_b = length(dst_areas)   # destination cells
    size(A) == (n_b, n_a) ||
        error("intersections matrix shape $(size(A)) does not match (n_b, n_a) = $((n_b, n_a))")

    # Walk the sparse matrix column-by-column (CSC natural order).
    rows = rowvals(A)
    vals = nonzeros(A)
    n_s  = length(vals)

    row_out = Vector{Int64}(undef, n_s)
    col_out = Vector{Int64}(undef, n_s)
    S_out   = Vector{Float64}(undef, n_s)
    covered_src = zeros(Float64, n_a)
    covered_dst = zeros(Float64, n_b)

    idx = 0
    for col in 1:n_a
        for ptr in nzrange(A, col)
            idx += 1
            d = rows[ptr]
            a = Float64(vals[ptr])
            row_out[idx] = d
            col_out[idx] = col
            S_out[idx]   = a / Float64(dst_areas[d])
            covered_src[col] += a
            covered_dst[d]   += a
        end
    end

    frac_a = covered_src ./ Float64.(src_areas)
    frac_b = covered_dst ./ Float64.(dst_areas)

    mkpath(dirname(path))
    NCDataset(path, "c") do ds
        ds.dim["n_s"] = n_s
        ds.dim["n_a"] = n_a
        ds.dim["n_b"] = n_b

        defVar(ds, "S",      S_out,                   ("n_s",); attrib = ["long_name" => "weight value"])
        defVar(ds, "row",    row_out,                 ("n_s",); attrib = ["long_name" => "destination cell index (1-based)"])
        defVar(ds, "col",    col_out,                 ("n_s",); attrib = ["long_name" => "source cell index (1-based)"])
        defVar(ds, "frac_a", frac_a,                  ("n_a",); attrib = ["long_name" => "source cell fraction covered by destination grid"])
        defVar(ds, "frac_b", frac_b,                  ("n_b",); attrib = ["long_name" => "destination cell fraction covered by source grid"])
        defVar(ds, "area_a", Float64.(src_areas),     ("n_a",); attrib = ["long_name" => "source cell areas", "units" => "m^2"])
        defVar(ds, "area_b", Float64.(dst_areas),     ("n_b",); attrib = ["long_name" => "destination cell areas", "units" => "m^2"])

        ds.attrib["title"]             = "ConservativeRegridding.jl weights (ESMF format)"
        ds.attrib["created_by"]        = "AtmosTransport.Regridding.save_esmf_weights"
        ds.attrib["created_at"]        = string(now())
        ds.attrib["source_grid"]       = String(src_grid_name)
        ds.attrib["destination_grid"]  = String(dst_grid_name)
        ds.attrib["normalization"]     = "destarea"
        ds.attrib["map_method"]        = "Conservative remapping"
        if src_shape !== nothing
            ds.attrib["source_grid_shape"] = collect(Int64.(src_shape))
        end
        if dst_shape !== nothing
            ds.attrib["destination_grid_shape"] = collect(Int64.(dst_shape))
        end
    end
    return path
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
"""
function apply_regridder!(dst::AbstractVector, r::Regridder, src::AbstractVector)
    return ConservativeRegridding.regrid!(dst, r, src)
end

function apply_regridder!(dst::AbstractArray, r::Regridder, src::AbstractArray)
    n_dst = length(r.dst_areas)
    n_src = length(r.src_areas)
    size(dst, 1) == n_dst ||
        error("dst leading dim $(size(dst,1)) ≠ regridder n_dst $n_dst")
    size(src, 1) == n_src ||
        error("src leading dim $(size(src,1)) ≠ regridder n_src $n_src")

    dst_flat = reshape(dst, n_dst, :)
    src_flat = reshape(src, n_src, :)
    size(dst_flat, 2) == size(src_flat, 2) ||
        error("trailing shape mismatch: dst $(size(dst_flat,2)) ≠ src $(size(src_flat,2))")

    @inbounds for col in axes(dst_flat, 2)
        ConservativeRegridding.regrid!(
            view(dst_flat, :, col), r, view(src_flat, :, col),
        )
    end
    return dst
end
