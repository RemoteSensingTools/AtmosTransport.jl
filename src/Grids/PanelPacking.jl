"""Flat cell range occupied by panel `p` in column-major panel storage."""
@inline cs_panel_flat_range(p::Int, Nc::Int) = (p - 1) * Nc * Nc + 1 : p * Nc * Nc

"""
    unpack_flat_to_panels_3d!(panels, flat, Nc, Nz)

Unpack a flat `(6Nc², Nz)` matrix into 6 panel arrays `(Nc, Nc, Nz)`.
"""
function unpack_flat_to_panels_3d!(panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                    flat::AbstractMatrix, Nc::Int, Nz::Int) where FT
    size(flat) == (CS_PANEL_COUNT * Nc * Nc, Nz) || throw(DimensionMismatch("flat panel matrix shape"))
    all(p -> size(p) == (Nc, Nc, Nz), panels) || throw(DimensionMismatch("panel shape"))
    for p in 1:CS_PANEL_COUNT
        r = cs_panel_flat_range(p, Nc)
        for k in 1:Nz
            @inbounds for (linear, flat_idx) in enumerate(r)
                j, i = fldmod1(linear, Nc)  # (div, mod) = (j, i) for column-major (j-1)*Nc+i
                panels[p][i, j, k] = flat[flat_idx, k]
            end
        end
    end
    return panels
end

"""
    unpack_flat_to_panels_2d!(panels, flat, Nc)

Unpack a flat `(6Nc²,)` vector into 6 panel arrays `(Nc, Nc)`.
"""
function unpack_flat_to_panels_2d!(panels::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                    flat::AbstractVector, Nc::Int) where FT
    length(flat) == CS_PANEL_COUNT * Nc * Nc || throw(DimensionMismatch("flat panel vector length"))
    all(p -> size(p) == (Nc, Nc), panels) || throw(DimensionMismatch("panel shape"))
    for p in 1:CS_PANEL_COUNT
        r = cs_panel_flat_range(p, Nc)
        @inbounds for (linear, flat_idx) in enumerate(r)
            j, i = fldmod1(linear, Nc)  # (div, mod) = (j, i) for column-major (j-1)*Nc+i
            panels[p][i, j] = flat[flat_idx]
        end
    end
    return panels
end


export cs_panel_flat_range, unpack_flat_to_panels_2d!, unpack_flat_to_panels_3d!

"""
    pack_panels!(dest, offset, panels) -> next_offset

Copy panel arrays consecutively into a flat vector, converting to its element
 type. `offset` is the number of elements before the first panel.
"""
function pack_panels!(dest::AbstractVector{FT}, offset::Int, panels::Tuple) where FT
    o = offset
    for panel_data in panels
        n = length(panel_data)
        @inbounds for idx in 1:n
            dest[o + idx] = convert(FT, panel_data[idx])
        end
        o += n
    end
    return o
end

export pack_panels!
