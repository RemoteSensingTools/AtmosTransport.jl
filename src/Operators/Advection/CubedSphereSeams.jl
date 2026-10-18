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

@kernel function _cs_apply_seam_kernel!(rm_a, m_a, rm_b, m_b, @Const(cache),
                                       edge, other_edge, reversed, seam, Nc, Hp, Nt)
    s, k = @index(Global, NTuple)
    t = reversed ? Nc + 1 - s : s
    i, j = _cs_seam_face_index(edge, s, Nc)
    u, v = _cs_seam_face_index(other_edge, t, Nc)
    i = Hp + min(i, Nc); j = Hp + min(j, Nc)
    u = Hp + min(u, Nc); v = Hp + min(v, Nc)
    @inbounds begin
        air = cache[s, k, Nt + 1, seam]
        m_a[i, j, k] -= air
        m_b[u, v, k] += air
        for tracer in 1:Nt
            amount = cache[s, k, tracer, seam]
            _cs_add_seam_tracer!(rm_a, i, j, k, tracer, -amount)
            _cs_add_seam_tracer!(rm_b, u, v, k, tracer, amount)
        end
    end
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
                          mesh::CubedSphereMesh, ::Val{D}) where D
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz, Nt = size(panels_m[1], 3), size(panels_rm[1], 4)
    backend = get_backend(panels_rm[1])
    kernel! = _cs_apply_seam_kernel!(backend, 64)
    seam = 0
    for p in 1:6, edge in 1:4
        neighbor = mesh.connectivity.neighbors[p][edge]
        q = neighbor.panel
        p < q || continue
        seam += 1
        _cs_seam_axis(edge) == D || continue
        other_edge = reciprocal_edge(mesh.connectivity, p, edge)
        # Contacts meeting at a corner write the same cell. Keep their launches
        # ordered on the backend stream; threads within one contact are disjoint.
        kernel!(panels_rm[p], panels_m[p], panels_rm[q], panels_m[q], cache,
                edge, other_edge, neighbor.orientation != 0, seam, Nc, Hp, Nt;
                ndrange=(Nc, Nz))
    end
    # Later work on the device queue (the halo exchange after the sweep) sees
    # these cells in queue order; the CPU backend completes each launch here.
    backend isa KA_CPU && synchronize(backend)
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
