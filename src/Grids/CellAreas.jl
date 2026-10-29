"""Flat Float64 cell areas in native/regridder order, preserving mesh-rounded values."""
cell_areas(m::LatLonMesh) = vec([Float64(cell_area(m, i, j)) for i in 1:m.Nx, j in 1:m.Ny])
cell_areas(m::CubedSphereMesh) = repeat(vec(Float64.(m.cell_areas)), CS_PANEL_COUNT)
cell_areas(m::ReducedGaussianMesh) = [Float64(cell_area(m, c)) for c in 1:ncells(m)]

"""Float64 cubed-sphere mesh with the same definition, radius and resolution, without halos."""
float64_twin(mesh::CubedSphereMesh) =
    CubedSphereMesh(; FT = Float64, Nc = mesh.Nc, Hp = 0, definition = cs_definition(mesh),
                    radius = Float64(mesh.radius))

"""Per-panel cell areas recomputed in Float64 when the mesh uses another precision."""
float64_cell_areas(m::CubedSphereMesh{Float64}) = m.cell_areas
float64_cell_areas(m::CubedSphereMesh) = float64_twin(m).cell_areas

"""Recomputed Float64 areas in panel-major flat order (distinct from stored-precision `cell_areas`)."""
flat_float64_cell_areas(m::CubedSphereMesh) = repeat(vec(float64_cell_areas(m)), CS_PANEL_COUNT)

"""Total cubed-sphere area, using Float64 accumulation of stored panel areas."""
cs_total_area(mesh::CubedSphereMesh) = cs_total_area(mesh.cell_areas)
cs_total_area(areas::AbstractMatrix) = CS_PANEL_COUNT * sum(Float64, areas)

export cell_areas, float64_twin, float64_cell_areas, flat_float64_cell_areas, cs_total_area

"""
    cell_area_matrix(mesh) -> Matrix

Expand the latitude-band area vector from `LatLonMesh` into a full `(Nx, Ny)`
matrix so later kernels can index cell area directly by `(i, j)`.
"""
function cell_area_matrix(mesh::LatLonMesh{FT}) where FT
    area_by_lat = cell_areas_by_latitude(mesh)
    area = Array{FT}(undef, mesh.Nx, mesh.Ny)
    for j in 1:mesh.Ny
        @views area[:, j] .= area_by_lat[j]
    end
    return area
end

export cell_area_matrix
