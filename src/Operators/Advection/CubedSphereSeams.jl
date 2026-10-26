# A directional group contains its panel-interior faces and the physical seams
# whose lower-numbered panel owns an edge in that direction. A rotated seam
# transfers to both panels in the SAME group, even when its neighbor's local
# edge lies on the other axis. Cache each transfer before any in-place sweep.

struct _CSInteriorFlux{FT, A <: AbstractArray{FT, 3}, D} <: AbstractArray{FT, 3}
    data::A
    first_face::Int
    last_face::Int
end

_CSInteriorFlux(a::AbstractArray{FT, 3}, mesh::CubedSphereMesh, ::Val{D}) where {FT, D} =
    _CSInteriorFlux{FT, typeof(a), D}(a, mesh.Hp + 1, mesh.Hp + mesh.Nc + 1)
Base.size(a::_CSInteriorFlux) = size(a.data)
Base.parent(a::_CSInteriorFlux) = a.data
Base.IndexStyle(::Type{<:_CSInteriorFlux}) = IndexCartesian()
@inline function Base.getindex(a::_CSInteriorFlux{FT, A, D}, i::Integer, j::Integer,
                              k::Integer) where {FT, A, D}
    face = D == 1 ? i : j
    (face == a.first_face || face == a.last_face) && return zero(FT)
    @inbounds return a.data[i, j, k]
end
function Adapt.adapt_structure(to, a::_CSInteriorFlux{FT, A, D}) where {FT, A, D}
    data = Adapt.adapt(to, a.data)
    return _CSInteriorFlux{FT, typeof(data), D}(data, a.first_face, a.last_face)
end

@inline _cs_seam_axis(edge) = edge in (EDGE_EAST, EDGE_WEST) ? 1 : 2
@inline _cs_seam_sign(edge, ::Type{FT}) where FT =
    edge in (EDGE_EAST, EDGE_NORTH) ? one(FT) : -one(FT)
@inline function _cs_seam_face_index(edge, s, Nc)
    edge == EDGE_NORTH && return (s, Nc + 1)
    edge == EDGE_SOUTH && return (s, 1)
    edge == EDGE_EAST && return (Nc + 1, s)
    return (1, s)
end
@inline _cs_seam_tracer(rm::AbstractArray{FT, 3}, t) where FT = rm
@inline _cs_seam_tracer(rm::AbstractArray{FT, 4}, t) where FT = TracerView(rm, Int32(t))
@inline function _cs_add_seam_tracer!(rm::AbstractArray{FT, 3}, i, j, k, t, amount) where FT
    @inbounds rm[i, j, k] += amount
end
@inline function _cs_add_seam_tracer!(rm::AbstractArray{FT, 4}, i, j, k, t, amount) where FT
    @inbounds rm[i, j, k, t] += amount
end

@inline _cs_seam_tracer_flux(::Val{1}, i, j, k, rm, m, F, scheme, N) =
    _xface_tracer_flux(Int32(i), j, k, rm, m, F, scheme, Int32(N))
@inline _cs_seam_tracer_flux(::Val{2}, i, j, k, rm, m, F, scheme, N) =
    _yface_tracer_flux(i, Int32(j), k, rm, m, F, scheme, Int32(N))
@inline function _cs_seam_tracer_flux(::Val{1}, i, j, k, rm, m, F, ::UpwindScheme, N)
    donor = F >= zero(F) ? i - 1 : i
    @inbounds return _gamma_clamped_x_flux(F, m[donor, j, k], rm[donor, j, k])
end
@inline function _cs_seam_tracer_flux(::Val{2}, i, j, k, rm, m, F, ::UpwindScheme, N)
    donor = F >= zero(F) ? j - 1 : j
    @inbounds return _gamma_clamped_x_flux(F, m[i, donor, k], rm[i, donor, k])
end

"""
    SeamTransferCache{D}(p, edge, seam, active)

Cache the transfer through physical seam `seam`, edge `edge` of the
lower-numbered panel `p`, for directional group `D` (1 = X, 2 = Y): at
along-edge index `s` and level `k`, the signed air-mass flux and every tracer's
face flux, read before any panel of the group is updated. A seam of the other
group is inactive (an empty index space), so the twelve seams of a cube form
one `Fused` launch for either group.
"""
struct SeamTransferCache{D} <: AbstractPointOp
    p::Int
    edge::Int
    seam::Int
    active::Bool
end

index_space(op::SeamTransferCache, ctx) = (op.active ? ctx.Nc : 0, size(ctx.cache, 2))

bound_context(op::SeamTransferCache, ctx) =
    merge(ctx, (rm = ctx.panels_rm[op.p], m = ctx.panels_m[op.p], flux = ctx.panels_flux[op.p]))
@inline _seam_panel(op::SeamTransferCache, ctx, ::Val{:rm}) =
    hasproperty(ctx, :rm) ? ctx.rm : select_panel(ctx.panels_rm, op.p)
@inline _seam_panel(op::SeamTransferCache, ctx, ::Val{:m}) =
    hasproperty(ctx, :m) ? ctx.m : select_panel(ctx.panels_m, op.p)
@inline _seam_panel(op::SeamTransferCache, ctx, ::Val{:flux}) =
    hasproperty(ctx, :flux) ? ctx.flux : select_panel(ctx.panels_flux, op.p)

@inline function apply_point!(op::SeamTransferCache{D}, ctx, s, k) where D
    (; cache, scheme, Nc, Hp, Nt, scale) = ctx
    rm = _seam_panel(op, ctx, Val(:rm))
    m = _seam_panel(op, ctx, Val(:m))
    flux = _seam_panel(op, ctx, Val(:flux))
    i, j = _cs_seam_face_index(op.edge, s, Nc)
    i += Hp; j += Hp
    @inbounds begin
        F = scale * flux[i, j, k]
        sign = _cs_seam_sign(op.edge, eltype(m))
        cache[s, k, Nt + 1, op.seam] = sign * F
        for t in 1:Nt
            tracer = _cs_seam_tracer(rm, t)
            cache[s, k, t, op.seam] = sign * _cs_seam_tracer_flux(
                Val(D), i, j, k, tracer, m, F, scheme, Nc + 2Hp)
        end
    end
    return nothing
end

# Physical seam `n` of a cube in seam-cache order: the `n`-th edge `edge` of a
# panel `p` whose neighbor has the higher panel number (panels, then edges, in
# increasing order, as `_apply_cs_seams!` counts them).
function _cs_seam(conn::PanelConnectivity, n::Int)
    seam = 0
    for p in 1:6, edge in 1:4
        p < conn.neighbors[p][edge].panel || continue
        (seam += 1) == n && return (p, edge)
    end
    throw(ArgumentError("cubed-sphere connectivity has $seam physical seams, not 12"))
end

function _seam_cache_ops(conn::PanelConnectivity, ::Val{D}) where D
    return Fused(ntuple(Val(12)) do n
        p, edge = _cs_seam(conn, n)
        SeamTransferCache{D}(p, edge, n, _cs_seam_axis(edge) == D)
    end)
end

"""
    SeamTransferApply{D}(p, q, edge, other_edge, reversed, seam, active,
                         partner_seam, partner_s, partner_on_q)

Apply the cached transfer of physical seam `seam` (directional group `D`) at
along-edge index `s` and level `k`: remove it from the edge cell of the
lower-numbered panel `p` (side A) and add it to the matching edge cell of `q`
(side B). An inactive seam has an empty index space, as in
[`SeamTransferCache`](@ref).

The seams of a group write disjoint cells except at panel corners, where the
two edges of one panel meeting at a corner may both belong to the group (with
the GEOS-native connectivity four corner cells per group). Each such cell is
written only by the operation of the later seam, which applies the earlier
seam's contribution and then its own, so the cell sees the additions in seam
order, as with one launch per seam. The endpoint tuples are indexed
(A at `s = 1`, A at `s = Nc`, B at `s = 1`, B at `s = Nc`); `partner_seam` is 0
where the corner cell receives no other contribution (everywhere with
`Nc = 1`, where the operations run one after the other), and `partner_s` and
`partner_on_q` locate the other contribution in the cache.
"""
struct SeamTransferApply{D} <: AbstractPointOp
    p::Int
    q::Int
    edge::Int
    other_edge::Int
    reversed::Bool
    seam::Int
    active::Bool
    partner_seam::NTuple{4, Int}
    partner_s::NTuple{4, Int}
    partner_on_q::NTuple{4, Bool}
end

index_space(op::SeamTransferApply, ctx) = (op.active ? ctx.Nc : 0, size(ctx.cache, 2))

bound_context(op::SeamTransferApply, ctx) =
    merge(ctx, (rm_a = ctx.panels_rm[op.p], m_a = ctx.panels_m[op.p],
                rm_b = ctx.panels_rm[op.q], m_b = ctx.panels_m[op.q]))
@inline _seam_side_arrays(op::SeamTransferApply, ctx, ::Val{:a}) =
    hasproperty(ctx, :rm_a) ? (ctx.rm_a, ctx.m_a) :
        (select_panel(ctx.panels_rm, op.p), select_panel(ctx.panels_m, op.p))
@inline _seam_side_arrays(op::SeamTransferApply, ctx, ::Val{:b}) =
    hasproperty(ctx, :rm_b) ? (ctx.rm_b, ctx.m_b) :
        (select_panel(ctx.panels_rm, op.q), select_panel(ctx.panels_m, op.q))

@inline _seam_endpoint(t::NTuple{4}, n) = n == 1 ? t[1] : n == 2 ? t[2] : n == 3 ? t[3] : t[4]

# Add the cached transfer (air mass, then every tracer) of seam `seam` at index
# `s` to one cell: subtracted on side A, added on side B.
@inline function _cs_add_seam_transfer!(rm, m, i, j, k, cache, s, seam, Nt, on_q::Bool)
    @inbounds begin
        air = cache[s, k, Nt + 1, seam]
        if on_q
            m[i, j, k] += air
        else
            m[i, j, k] -= air
        end
        for tracer in 1:Nt
            amount = cache[s, k, tracer, seam]
            _cs_add_seam_tracer!(rm, i, j, k, tracer, on_q ? amount : -amount)
        end
    end
    return nothing
end

@inline function _cs_apply_seam_side!(rm, m, i, j, k, cache, s, Nt, op, on_q, endpoint)
    if endpoint != 0
        partner = _seam_endpoint(op.partner_seam, endpoint)
        partner > op.seam && return nothing     # the later seam writes this corner
        partner > 0 && _cs_add_seam_transfer!(rm, m, i, j, k, cache,
            _seam_endpoint(op.partner_s, endpoint), partner, Nt,
            _seam_endpoint(op.partner_on_q, endpoint))
    end
    _cs_add_seam_transfer!(rm, m, i, j, k, cache, s, op.seam, Nt, on_q)
    return nothing
end

@inline function apply_point!(op::SeamTransferApply, ctx, s, k)
    (; cache, Nc, Hp, Nt) = ctx
    rm_a, m_a = _seam_side_arrays(op, ctx, Val(:a))
    rm_b, m_b = _seam_side_arrays(op, ctx, Val(:b))
    t = op.reversed ? Nc + 1 - s : s
    # Face Nc + 1 belongs to cell Nc.
    i, j = _cs_seam_face_index(op.edge, s, Nc)
    u, v = _cs_seam_face_index(op.other_edge, t, Nc)
    i = Hp + min(i, Nc); j = Hp + min(j, Nc)
    u = Hp + min(u, Nc); v = Hp + min(v, Nc)
    end_a, end_b = s == 1 ? (1, 3) : s == Nc ? (2, 4) : (0, 0)
    _cs_apply_seam_side!(rm_a, m_a, i, j, k, cache, s, Nt, op, false, end_a)
    _cs_apply_seam_side!(rm_b, m_b, u, v, k, cache, s, Nt, op, true, end_b)
    return nothing
end

# The seam number of panel `p`'s edge `edge`, whether `p` is the seam's side B,
# and whether the seam reverses the along-edge index.
function _cs_panel_edge_seam(conn::PanelConnectivity, p::Int, edge::Int)
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

# The other contribution of group `D` to the corner cell at along-edge cell
# index `along` (1 or Nc) of panel `panel`'s edge `edge`, as (seam, cache index,
# side B), or `(0, 0, false)`: the corner's other edge belongs to the group when
# its seam's lower panel owns it on axis `D`.
function _cs_seam_corner_partner(conn::PanelConnectivity, ::Val{D}, panel, edge,
                                 along, Nc) where D
    i, j = min.(_cs_seam_face_index(edge, along, Nc), Nc)
    other, other_along = _cs_seam_axis(edge) == 2 ? (i == 1 ? EDGE_WEST : EDGE_EAST, j) :
                                                    (j == 1 ? EDGE_SOUTH : EDGE_NORTH, i)
    seam, on_q, reversed, lower_edge = _cs_panel_edge_seam(conn, panel, other)
    _cs_seam_axis(lower_edge) == D || return (0, 0, false)
    return (seam, on_q && reversed ? Nc + 1 - other_along : other_along, on_q)
end

# With `Nc = 1` all four edges of a panel end in its one cell; the operations
# then carry no partners and `_apply_cs_seams!` runs them one after the other.
function _seam_apply_ops(conn::PanelConnectivity, ::Val{D}, Nc::Int) where D
    return Fused(ntuple(Val(12)) do n
        p, edge = _cs_seam(conn, n)
        q = conn.neighbors[p][edge].panel
        other_edge = reciprocal_edge(conn, p, edge)
        reversed = conn.neighbors[p][edge].orientation != 0
        active = _cs_seam_axis(edge) == D
        partners = ntuple(Val(4)) do e
            on_q, s = e > 2, isodd(e) ? 1 : Nc
            (active && Nc >= 2) || return (0, 0, false)
            along = on_q && reversed ? Nc + 1 - s : s
            on_q ? _cs_seam_corner_partner(conn, Val(D), q, other_edge, along, Nc) :
                   _cs_seam_corner_partner(conn, Val(D), p, edge, along, Nc)
        end
        SeamTransferApply{D}(p, q, edge, other_edge, reversed, n, active,
                             map(first, partners), map(x -> x[2], partners), map(last, partners))
    end)
end

function _cache_cs_seams!(cache, panels_rm::NTuple{6}, panels_m::NTuple{6},
                          panels_flux::NTuple{6}, mesh::CubedSphereMesh,
                          scheme::AbstractAdvectionScheme, direction::Val{D}, scale) where D
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz, Nt = size(panels_m[1], 3), size(panels_rm[1], 4)
    (size(cache, 1), size(cache, 2), size(cache, 4)) == (Nc, Nz, 12) ||
        throw(DimensionMismatch("CS seam cache must match Nc=$Nc, Nz=$Nz and 12 physical edges"))
    size(cache, 3) >= Nt + 1 || throw(DimensionMismatch("CS seam cache needs $Nt tracer slots plus air mass"))
    # One launch for the group's seams. The sweeps and the seam application
    # that read the cache follow on the same device queue, so no host
    # synchronization is needed (host loops run synchronously).
    ctx = (; panels_rm, panels_m, panels_flux, cache, scheme, Nc, Hp, Nt,
           scale = eltype(panels_m[1])(scale))
    launch!(_seam_cache_ops(mesh.connectivity, direction), ctx, get_backend(panels_rm[1]);
            workgroup = 64, sync = false)
    return nothing
end

function _apply_cs_seams!(panels_rm::NTuple{6}, panels_m::NTuple{6}, cache,
                          mesh::CubedSphereMesh, direction::Val{D}) where D
    Nc, Hp, Nt = mesh.Nc, mesh.Hp, size(panels_rm[1], 4)
    # One launch for the group's seams; corner cells shared by two seams are
    # written in seam order by one operation (see `SeamTransferApply`). Later
    # work on the device queue (the halo exchange after the sweep) sees these
    # cells in queue order; on the host the loops complete here.
    ctx = (; panels_rm, panels_m, cache, Nc, Hp, Nt)
    ops = _seam_apply_ops(mesh.connectivity, direction, Nc)
    backend = get_backend(panels_rm[1])
    if Nc >= 2
        launch!(ops, ctx, backend; workgroup = 64, sync = false)
    else
        launch!(Sequence(ops.ops), ctx, backend; workgroup = 64, sync = false)
    end
    return nothing
end

function _sweep_cs_horizontal!(panels_rm::NTuple{6}, panels_m::NTuple{6},
                                panels_flux::NTuple{6}, mesh::CubedSphereMesh,
                                scheme::AbstractAdvectionScheme, workspace,
                                direction::Val{D}; flux_scale=one(eltype(panels_m[1]))) where D
    _cache_cs_seams!(workspace.seam_flux, panels_rm, panels_m, panels_flux,
                     mesh, scheme, direction, flux_scale)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz, Nt = size(panels_m[1], 3), size(panels_rm[1], 4)
    for p in 1:6
        flux = _CSInteriorFlux(panels_flux[p], mesh, direction)
        if ndims(panels_rm[p]) == 3
            sweep! = D == 1 ? _sweep_x_panel! : _sweep_y_panel!
            sweep!(panels_rm[p], panels_m[p], flux, scheme, workspace.rm_A,
                   workspace.m_A, Nc, Hp, Nz; flux_scale)
        else
            sweep! = D == 1 ? _sweep_x_panel_mt! : _sweep_y_panel_mt!
            sweep!(panels_rm[p], panels_m[p], flux, scheme, workspace.rm_4d_A,
                   workspace.m_A, Nc, Hp, Nz, Nt; flux_scale)
        end
    end
    _apply_cs_seams!(panels_rm, panels_m, workspace.seam_flux, mesh, direction)
    return nothing
end
