_copy_panel_tuple(panels) = ntuple(p -> copy(panels[p]), 6)

function _zero_panel_tuple_like(panels)
    return ntuple(p -> begin
        a = similar(panels[p])
        fill!(a, zero(eltype(a)))
        a
    end, 6)
end

function _zero_surface_rates(mesh::CubedSphereMesh, prototype::AbstractArray{FT}) where {FT}
    return ntuple(_ -> begin
        a = similar(prototype, FT, mesh.Nc, mesh.Nc)
        fill!(a, zero(FT))
        a
    end, 6)
end
