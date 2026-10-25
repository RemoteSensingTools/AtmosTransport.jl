# ---------------------------------------------------------------------------
# Adjoint of CS halo exchange
#
# Reverse-mode of `fill_panel_halos!` (and `copy_corners!` when dir != 0).
# The kernels here scatter halo cell contributions back to the interior
# cells of the neighbouring panels they were copied from, applying the
# same edge/corner orientation logic as the forward pass.
# ---------------------------------------------------------------------------

@inline function _edge_interior_ij(q_e, d, s, Nc, Hp)
    if q_e == EDGE_NORTH
        return (Hp + s, Hp + Nc + 1 - d)
    elseif q_e == EDGE_SOUTH
        return (Hp + s, Hp + d)
    elseif q_e == EDGE_EAST
        return (Hp + Nc + 1 - d, Hp + s)
    else
        return (Hp + d, Hp + s)
    end
end

@inline function _edge_halo_ij(e, d, s, Nc, Hp)
    if e == EDGE_NORTH
        return (Hp + s, Hp + Nc + d)
    elseif e == EDGE_SOUTH
        return (Hp + s, Hp + 1 - d)
    elseif e == EDGE_EAST
        return (Hp + Nc + d, Hp + s)
    else
        return (Hp + 1 - d, Hp + s)
    end
end

@inline function _corner_source_ij(i_dst, j_dst, Nc, Hp, N, dir)
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

@kernel function _adjoint_corner_halo_kernel!(lambda, Nc, Hp, N, dir)
    di, dj, k = @index(Global, NTuple)
    @inbounds begin
        i_sw = Hp + 1 - di;  j_sw = Hp + 1 - dj
        i_se = Hp + Nc + di; j_se = Hp + 1 - dj
        i_ne = Hp + Nc + di; j_ne = Hp + Nc + dj
        i_nw = Hp + 1 - di;  j_nw = Hp + Nc + dj

        si, sj = _corner_source_ij(i_sw, j_sw, Nc, Hp, N, dir)
        val = lambda[i_sw, j_sw, k]
        @atomic lambda[si, sj, k] += val
        lambda[i_sw, j_sw, k] = zero(val)

        si, sj = _corner_source_ij(i_se, j_se, Nc, Hp, N, dir)
        val = lambda[i_se, j_se, k]
        @atomic lambda[si, sj, k] += val
        lambda[i_se, j_se, k] = zero(val)

        si, sj = _corner_source_ij(i_ne, j_ne, Nc, Hp, N, dir)
        val = lambda[i_ne, j_ne, k]
        @atomic lambda[si, sj, k] += val
        lambda[i_ne, j_ne, k] = zero(val)

        si, sj = _corner_source_ij(i_nw, j_nw, Nc, Hp, N, dir)
        val = lambda[i_nw, j_nw, k]
        @atomic lambda[si, sj, k] += val
        lambda[i_nw, j_nw, k] = zero(val)
    end
end

@kernel function _adjoint_edge_halo_kernel!(dst, src, e, q_e, flip, Nc, Hp)
    s, d, k = @index(Global, NTuple)
    @inbounds begin
        s_src = flip ? (Nc + 1 - s) : s
        i_src, j_src = _edge_interior_ij(q_e, d, s_src, Nc, Hp)
        i_dst, j_dst = _edge_halo_ij(e, d, s, Nc, Hp)
        val = dst[i_dst, j_dst, k]
        @atomic src[i_src, j_src, k] += val
        dst[i_dst, j_dst, k] = zero(val)
    end
end

function _adjoint_fill_panel_halos!(lambda_panels::NTuple{6},
                                    mesh::CubedSphereMesh; dir::Int=0)
    Nc, Hp = mesh.Nc, mesh.Hp
    Hp == 0 && return nothing
    if dir in (1, 2)
        N = Nc + 2 * Hp
        @inbounds for p in 1:6
            q = lambda_panels[p]
            backend = get_backend(q)
            kernel! = _adjoint_corner_halo_kernel!(backend, 256)
            kernel!(q, Int32(Nc), Int32(Hp), Int32(N), Int32(dir);
                    ndrange=(Hp, Hp, size(q, 3)))
        end
    end

    conn = mesh.connectivity
    @inbounds for p in 1:6
        for e in 1:4
            nb = conn.neighbors[p][e]
            q_e = reciprocal_edge(conn, p, e)
            dst = lambda_panels[p]
            src = lambda_panels[nb.panel]
            backend = get_backend(dst)
            kernel! = _adjoint_edge_halo_kernel!(backend, 256)
            kernel!(dst, src, Int32(e), Int32(q_e), nb.orientation >= 2,
                    Int32(Nc), Int32(Hp);
                    ndrange=(Nc, Hp, size(dst, 3)))
        end
    end
    return nothing
end
