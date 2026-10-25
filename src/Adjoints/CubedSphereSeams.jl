# Reverse the paired seam exchange while meteorology remains prescribed.
# Capture both output-cell seeds before the per-panel reverse overwrites them.

@kernel function _cs_cache_seam_seed_kernel!(seeds, @Const(lambda_a), @Const(lambda_b),
                                            edge, other_edge, reversed, seam, Nc, Hp)
    s, k = @index(Global, NTuple)
    t = reversed ? Nc + 1 - s : s
    i, j = _cs_seam_face_index(edge, s, Nc)
    u, v = _cs_seam_face_index(other_edge, t, Nc)
    i = Hp + min(i, Nc); j = Hp + min(j, Nc)
    u = Hp + min(u, Nc); v = Hp + min(v, Nc)
    @inbounds seeds[s, k, seam] = _cs_seam_sign(edge, eltype(lambda_a)) *
        (lambda_b[u, v, k] - lambda_a[i, j, k])
end

function _cache_cs_seam_seeds!(seeds, lambda::NTuple{6}, mesh::CubedSphereMesh,
                               ::Val{D}) where D
    Nc, Hp, Nz = mesh.Nc, mesh.Hp, size(lambda[1], 3)
    size(seeds) == (Nc, Nz, 12) ||
        throw(DimensionMismatch("CS seam adjoint cache must match Nc=$Nc, Nz=$Nz and 12 physical edges"))
    backend = get_backend(lambda[1])
    kernel! = _cs_cache_seam_seed_kernel!(backend, 64)
    seam = 0
    for p in 1:6, edge in 1:4
        neighbor = mesh.connectivity.neighbors[p][edge]
        q = neighbor.panel
        p < q || continue
        seam += 1
        _cs_seam_axis(edge) == D || continue
        other = reciprocal_edge(mesh.connectivity, p, edge)
        kernel!(seeds, lambda[p], lambda[q], edge, other, neighbor.orientation != 0,
                seam, Nc, Hp; ndrange=(Nc, Nz))
    end
    synchronize(backend)
    return nothing
end

Base.@propagate_inbounds function _add_cs_seam_adjoint!(lambda, m, rm, i, j, k, F, seed, scheme, N, ::Val{1})
    if rm === nothing
        _add_x_face_adjoint!(lambda, m, Int32(i), j, k, F, seed, scheme, _HaloPaddedAxis())
    else
        _add_x_face_adjoint!(lambda, m, rm, Int32(i), j, k, F, seed, scheme, _HaloPaddedAxis())
    end
end
Base.@propagate_inbounds function _add_cs_seam_adjoint!(lambda, m, rm, i, j, k, F, seed, scheme, N, ::Val{2})
    if rm === nothing
        _add_y_face_adjoint!(lambda, m, i, Int32(j), k, F, seed, scheme, Int32(N))
    else
        _add_y_face_adjoint!(lambda, m, rm, i, Int32(j), k, F, seed, scheme, Int32(N))
    end
end

@kernel function _cs_seam_adjoint_kernel!(lambda, @Const(m), @Const(rm), @Const(flux),
                                         @Const(seeds), scheme, direction, edge,
                                         seam, Nc, Hp, scale)
    s, k = @index(Global, NTuple)
    i, j = _cs_seam_face_index(edge, s, Nc)
    i += Hp; j += Hp
    @inbounds _add_cs_seam_adjoint!(lambda, m, rm, i, j, k,
        scale * flux[i, j, k], seeds[s, k, seam], scheme, Nc + 2Hp, direction)
end

function _apply_cs_seam_adjoint!(lambda::NTuple{6}, m::NTuple{6}, rm,
                                 flux::NTuple{6}, seeds, mesh::CubedSphereMesh,
                                 scheme, direction::Val{D}, scale) where D
    Nc, Hp, Nz = mesh.Nc, mesh.Hp, size(lambda[1], 3)
    backend = get_backend(lambda[1])
    kernel! = _cs_seam_adjoint_kernel!(backend, 64)
    seam = 0
    for p in 1:6, edge in 1:4
        p < mesh.connectivity.neighbors[p][edge].panel || continue
        seam += 1
        _cs_seam_axis(edge) == D || continue
        rm_panel = rm === nothing ? nothing : rm[p]
        kernel!(lambda[p], m[p], rm_panel, flux[p], seeds, scheme, direction,
                edge, seam, Nc, Hp, eltype(lambda[p])(scale); ndrange=(Nc, Nz))
    end
    synchronize(backend)
    return nothing
end
