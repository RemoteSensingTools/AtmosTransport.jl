# ===========================================================================
# Identity-passthrough "regridder" for source-mesh == target-mesh paths.
#
# When the orchestrator constructs source and target meshes that are
# structurally equivalent (e.g. GEOS-IT C180 native source and CS C180
# preprocessing target), the conservative regridder is a costly identity
# matrix. `IdentityRegrid` short-circuits that with a `copyto!` — type-stable,
# zero allocation, no `if` branch in the hot path.
# ===========================================================================

"""
    IdentityRegrid{M}

Passthrough sentinel returned by `build_regridder` when source and target
meshes are structurally equivalent (per `meshes_equivalent`). Carries the
shared mesh and its cell areas [m²] as `src_areas` and `dst_areas`, in the
flat cell order of `apply_regridder!`, so callers that convert between
densities and cell totals use it as they use a conservative `Regridder`. It
has no intersection matrix: it cannot be saved as weights, and `normalize`
does not scale its areas.
"""
struct IdentityRegrid{M}
    mesh      :: M
    src_areas :: Vector{Float64}
    dst_areas :: Vector{Float64}
end

function IdentityRegrid(mesh)
    areas = _flat_cell_areas(mesh)
    return IdentityRegrid(mesh, areas, areas)
end

# Cell areas in the flat order of the regridders: lat-lon `i` fastest, cubed
# sphere panel by panel, reduced Gaussian by cell index.
_flat_cell_areas(m::LatLonMesh) = vec([Float64(cell_area(m, i, j)) for i in 1:m.Nx, j in 1:m.Ny])
_flat_cell_areas(m::CubedSphereMesh) = repeat(vec(Float64.(m.cell_areas)), 6)
_flat_cell_areas(m::ReducedGaussianMesh) = [Float64(cell_area(m, c)) for c in 1:ncells(m)]

# ---------------------------------------------------------------------------
# Structural equivalence: same shape ⇒ identity passthrough is safe.
# Default fallback returns false — different mesh types are never equivalent.
# Concrete subtype methods compare the geometry-defining fields, mirroring
# what `_hash_mesh!` already serializes for the cache key.
# ---------------------------------------------------------------------------

"""
    meshes_equivalent(src, dst) -> Bool

True when the two meshes describe the same horizontal grid up to
floating-point equality of the geometry-defining fields. Used by
`build_regridder` to skip building a conservative regridder when the
source and target are the same grid.
"""
meshes_equivalent(::Any, ::Any) = false

meshes_equivalent(a::LatLonMesh, b::LatLonMesh) =
    a.Nx == b.Nx && a.Ny == b.Ny &&
    a.λᶠ == b.λᶠ && a.φᶠ == b.φᶠ &&
    a.radius == b.radius

meshes_equivalent(a::CubedSphereMesh, b::CubedSphereMesh) =
    a.Nc == b.Nc &&
    cs_definition_tag(cs_definition(a)) === cs_definition_tag(cs_definition(b)) &&
    typeof(coordinate_law(a)) === typeof(coordinate_law(b)) &&
    typeof(center_law(a)) === typeof(center_law(b)) &&
    typeof(a.convention) === typeof(b.convention) &&
    longitude_offset_deg(cs_definition(a)) == longitude_offset_deg(cs_definition(b)) &&
    a.radius == b.radius

meshes_equivalent(a::ReducedGaussianMesh, b::ReducedGaussianMesh) =
    a.nlon_per_ring == b.nlon_per_ring &&
    a.latitudes     == b.latitudes &&
    a.lat_faces     == b.lat_faces &&
    a.radius        == b.radius

"""
    identity_regrid_or_nothing(src, dst) -> Union{IdentityRegrid, Nothing}

Helper used inside `build_regridder` to early-out when meshes are
equivalent. Returns `nothing` when a real regridder must be built.
"""
identity_regrid_or_nothing(src, dst) =
    meshes_equivalent(src, dst) ? IdentityRegrid(src) : nothing

# ---------------------------------------------------------------------------
# apply_regridder! extension — passthrough is `copyto!`.
# ---------------------------------------------------------------------------

apply_regridder!(dst::AbstractArray, ::IdentityRegrid, src::AbstractArray) =
    copyto!(dst, src)
