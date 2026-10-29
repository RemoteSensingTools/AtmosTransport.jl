# ---------------------------------------------------------------------------
# Geometry operations — the universal face/cell API
#
# Every horizontal mesh must implement these functions. Structured meshes
# provide efficient (i,j)-indexed specializations; unstructured meshes use
# explicit connectivity tables.
# ---------------------------------------------------------------------------

# ---- Cell queries ----

"""
    ncells(mesh::AbstractHorizontalMesh) -> Int

Total number of horizontal cells in the mesh.
"""
function ncells end
_missing_mesh_method(mesh::AbstractHorizontalMesh, name::Symbol) =
    throw(ArgumentError("`$(name)` must be implemented for horizontal mesh " *
                        "`$(typeof(mesh))`."))
ncells(mesh::AbstractHorizontalMesh) = _missing_mesh_method(mesh, :ncells)

"""
    cell_area(mesh::AbstractHorizontalMesh, c) -> FT

Area [m²] of cell `c`.

`c` can be:
- A flat integer index (column-major: `c = i + (j-1)*Nx` for structured)
- A tuple `(i, j)` for structured meshes

Both forms must be implemented by all concrete mesh types.
"""
function cell_area end
cell_area(mesh::AbstractHorizontalMesh, _c) = _missing_mesh_method(mesh, :cell_area)

"""
    cell_faces(mesh::AbstractHorizontalMesh, c) -> indices

Indices of faces bounding cell `c`. For structured meshes this
returns the 4 (or 6 for hex) face indices; for unstructured meshes
it uses the CSR adjacency.
"""
function cell_faces end
cell_faces(mesh::AbstractHorizontalMesh, _c) = _missing_mesh_method(mesh, :cell_faces)

# ---- Face queries ----

"""
    nfaces(mesh::AbstractHorizontalMesh) -> Int

Total number of horizontal faces in the mesh.
"""
function nfaces end
nfaces(mesh::AbstractHorizontalMesh) = _missing_mesh_method(mesh, :nfaces)

"""
    face_length(mesh::AbstractHorizontalMesh, f) -> FT

Length [m] of face `f` (or area of the face cross-section for 3D).
"""
function face_length end
face_length(mesh::AbstractHorizontalMesh, _f) = _missing_mesh_method(mesh, :face_length)

"""
    face_normal(mesh::AbstractHorizontalMesh, f) -> (nx, ny)

Unit normal of face `f` in **logical coordinates** (not geographic).

For structured meshes, components are in the `(i, j)` index directions:
- X-faces → `(1, 0)` (positive = increasing `i` = eastward on LatLon)
- Y-faces → `(0, 1)` (positive = increasing `j` = northward on LatLon)

For unstructured meshes, implementations should return components in
a **local tangent-plane frame** defined per face (e.g., east/north or
panel-local). The frame convention MUST be documented by each mesh type
so that flux-signing is unambiguous.

The transport operators multiply `face_normal` by `face_length` and
the face flux to compute the signed contribution to cell convergence.
Consistent normal orientation between `face_cells` (left → right = positive)
and `face_normal` is required.
"""
function face_normal end
face_normal(mesh::AbstractHorizontalMesh, _f) = _missing_mesh_method(mesh, :face_normal)

"""
    face_cells(mesh::AbstractHorizontalMesh, f) -> (left, right)

The two cells sharing face `f`. Convention: flux from left to right
is positive. Boundary faces use `0` or a sentinel for the exterior cell.
"""
function face_cells end
face_cells(mesh::AbstractHorizontalMesh, _f) = _missing_mesh_method(mesh, :face_cells)

# ---- Structured mesh convenience ----

"""
    nx(mesh::AbstractStructuredMesh) -> Int
    ny(mesh::AbstractStructuredMesh) -> Int

Logical grid dimensions for structured meshes.
"""
function nx end
function ny end
nx(mesh::AbstractStructuredMesh) = _missing_mesh_method(mesh, :nx)
ny(mesh::AbstractStructuredMesh) = _missing_mesh_method(mesh, :ny)

export ncells, nfaces, cell_area, cell_faces
export face_length, face_normal, face_cells
export nx, ny

"""Materialize face-left and face-right cell connectivity as Int32 vectors."""
function face_connectivity(mesh::AbstractHorizontalMesh)
    left = Vector{Int32}(undef, nfaces(mesh))
    right = Vector{Int32}(undef, nfaces(mesh))
    @inbounds for f in eachindex(left)
        l, r = face_cells(mesh, f)
        left[f] = Int32(l)
        right[f] = Int32(r)
    end
    return left, right
end

export face_connectivity
