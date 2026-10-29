"""Map `(edge, depth, along)` to the interior (i, j) that provides source data."""
@inline function edge_interior_ij(q_e::Integer, d::Integer, s::Integer, Nc::Integer, Hp::Integer)
    if     q_e == EDGE_NORTH  # read from top rows going inward
        return (Hp + s, Hp + Nc + 1 - d)
    elseif q_e == EDGE_SOUTH  # read from bottom rows going inward
        return (Hp + s, Hp + d)
    elseif q_e == EDGE_EAST   # read from right columns going inward
        return (Hp + Nc + 1 - d, Hp + s)
    else  # EDGE_WEST         # read from left columns going inward
        return (Hp + d, Hp + s)
    end
end

"""Map `(edge, depth, along)` to the halo (i, j) that receives data."""
@inline function edge_halo_ij(e::Integer, d::Integer, s::Integer, Nc::Integer, Hp::Integer)
    if     e == EDGE_NORTH  # above interior top
        return (Hp + s, Hp + Nc + d)
    elseif e == EDGE_SOUTH  # below interior bottom
        return (Hp + s, Hp + 1 - d)
    elseif e == EDGE_EAST   # right of interior
        return (Hp + Nc + d, Hp + s)
    else  # EDGE_WEST       # left of interior
        return (Hp + 1 - d, Hp + s)
    end
end

"""Source index for an FV3 rotated corner halo cell; accepts host and kernel integer types."""
@inline function corner_source_ij(i_dst, j_dst, Nc, Hp, N, dir)
    if dir == 1
        if i_dst <= Hp && j_dst <= Hp
            return (j_dst, 2 * Hp + 1 - i_dst)
        elseif i_dst > Hp + Nc && j_dst <= Hp
            return (N + 1 - j_dst, i_dst - Nc)
        elseif i_dst > Hp + Nc && j_dst > Hp + Nc
            return (j_dst, 2 * (Nc + Hp) + 1 - i_dst)
        else
            return (N + 1 - j_dst, i_dst + Nc)
        end
    else
        if i_dst <= Hp && j_dst <= Hp
            return (2 * Hp + 1 - j_dst, i_dst)
        elseif i_dst > Hp + Nc && j_dst <= Hp
            return (Nc + j_dst, N + 1 - i_dst)
        elseif i_dst > Hp + Nc && j_dst > Hp + Nc
            return (2 * (Nc + Hp) + 1 - j_dst, i_dst)
        else
            return (j_dst - Nc, N + 1 - i_dst)
        end
    end
end

"""
    cs_edge_cell(edge, s, Nc) -> (i, j)

Return the cell indices `(i, j)` at position `s` along edge `edge`.
Edge 1=north, 2=south, 3=east, 4=west.
"""
@inline function cs_edge_cell(edge::Integer, s::Integer, Nc::Integer)
    if edge == EDGE_NORTH       # j = Nc, i = s
        return (s, Nc)
    elseif edge == EDGE_SOUTH   # j = 1, i = s
        return (s, 1)
    elseif edge == EDGE_EAST    # i = Nc, j = s
        return (Nc, s)
    else                        # i = 1, j = s
        return (1, s)
    end
end

"""
    cs_edge_face_location(edge, s, Nc) -> (dir, i, j)

Return the per-panel am/bm index for the boundary face at position `s` along
edge `edge`.

| Edge    | Face array   | Index             |
|---------|-------------|-------------------|
| north   | bm[s, Nc+1] | dir=2, i=s, j=Nc+1 |
| south   | bm[s, 1]    | dir=2, i=s, j=1     |
| east    | am[Nc+1, s] | dir=1, i=Nc+1, j=s  |
| west    | am[1, s]    | dir=1, i=1, j=s      |
"""
@inline function cs_edge_face_location(edge::Integer, s::Integer, Nc::Integer)
    if edge == EDGE_NORTH       # north boundary
        return (2, s, Nc + 1)
    elseif edge == EDGE_SOUTH   # south boundary
        return (2, s, 1)
    elseif edge == EDGE_EAST    # east boundary
        return (1, Nc + 1, s)
    else                        # west boundary
        return (1, 1, s)
    end
end

"""
    cs_flat_index(i, j, p, Nc) -> Int

Map panel-local cell `(i, j)` on panel `p` to a global cell index in `1:6Nc²`.
Column-major within each panel: cell `(i, j, p)` → `(p-1)*Nc² + (j-1)*Nc + i`.
"""
@inline cs_flat_index(i::Integer, j::Integer, p::Integer, Nc::Integer) =
    (p - 1) * Nc * Nc + (j - 1) * Nc + i

"""Decode a flat column-major cubed-sphere cell index as `(i, j, panel)`."""
@inline function cs_panel_ij(c, Nc)
    p, r = divrem(c - 1, Nc * Nc)
    j, i = divrem(r, Nc)
    return i + 1, j + 1, p + 1
end

"""Local flux axis of a panel edge (1 for x, 2 for y)."""
@inline cs_seam_axis(edge) = edge in (EDGE_EAST, EDGE_WEST) ? 1 : 2
"""Signed outward flux orientation for an edge, in the requested scalar type."""
@inline cs_seam_sign(edge, ::Type{FT}) where FT =
    edge in (EDGE_EAST, EDGE_NORTH) ? one(FT) : -one(FT)
"""Boundary flux-face `(i, j)` at along-edge position `s`."""
@inline function cs_seam_face_index(edge, s, Nc)
    _, i, j = cs_edge_face_location(edge, s, Nc)
    return i, j
end

"""Return physical seam `n` in panel-major, edge-major canonical order."""
function cs_seam(conn::PanelConnectivity, n::Int)
    seam = 0
    for p in 1:6, edge in 1:4
        p < conn.neighbors[p][edge].panel || continue
        (seam += 1) == n && return (p, edge)
    end
    throw(ArgumentError("cubed-sphere connectivity has $seam physical seams, not 12"))
end

export edge_interior_ij, edge_halo_ij, corner_source_ij
export cs_edge_cell, cs_edge_face_location, cs_flat_index, cs_panel_ij
export cs_seam_axis, cs_seam_sign, cs_seam_face_index, cs_seam

"""The twelve physical seams, ordered by canonical panel then edge."""
canonical_seams(conn::PanelConnectivity) = ntuple(n -> cs_seam(conn, n), 12)

"""Sign of a neighbor flux mirror relative to the canonical edge flux."""
@inline seam_mirror_sign(edge, other_edge, ::Type{FT}) where FT =
    cs_seam_sign(edge, FT) == cs_seam_sign(other_edge, FT) ? -one(FT) : one(FT)

export canonical_seams, seam_mirror_sign

"""Return seam number, side, reversal and canonical edge for a panel edge."""
function cs_panel_edge_seam(conn::PanelConnectivity, p::Int, edge::Int)
    q = conn.neighbors[p][edge].panel
    lower, lower_edge = p < q ? (p, edge) : (q, reciprocal_edge(conn, p, edge))
    seam = 0
    for r in 1:6, e in 1:4
        r < conn.neighbors[r][e].panel || continue
        seam += 1
        (r, e) == (lower, lower_edge) && break
    end
    return seam, p > q, conn.neighbors[lower][lower_edge].orientation != 0, lower_edge
end

export cs_panel_edge_seam
