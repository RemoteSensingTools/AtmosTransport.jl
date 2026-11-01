_copy_panel_tuple(panels) = ntuple(p -> copy(panels[p]), 6)

function _zero_surface_rates(mesh::CubedSphereMesh, prototype::AbstractArray{FT}) where {FT}
    return ntuple(_ -> zeros_like(prototype, FT, mesh.Nc, mesh.Nc), 6)
end
