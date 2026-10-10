# ---------------------------------------------------------------------------
# Cubed-sphere panel halo exchange for src
#
# Fills halo regions of six-panel fields, NTuple{6} of 3-D `(x, y, z)` or packed
# 4-D `(x, y, z, tracer)` arrays of any array type, with extended horizontal
# dimensions (Nc + 2*Hp) × (Nc + 2*Hp) per panel. Interior data lives at
# indices [Hp+1:Hp+Nc, Hp+1:Hp+Nc, ...].
#
# Edge convention:
#   1 = North (top,    j = Nc)
#   2 = South (bottom, j = 1)
#   3 = East  (right,  i = Nc)
#   4 = West  (left,   i = 1)
#
# Orientation codes (from PanelConnectivity):
#   0 = aligned  → along-edge same direction
#   2 = reversed → along-edge reversed
#
# References:
#   Putman & Lin (2007) — FV3 cubed-sphere grid connectivity
#   Martin et al. (2022, GMD) — GCHP panel exchange
# ---------------------------------------------------------------------------

using KernelAbstractions: get_backend
using ...Architectures: AbstractPointOp, Fused, Sequence, select_panel
import ...Architectures: index_space, apply_point!, bound_context, launch!

# ---------------------------------------------------------------------------
# Edge indexing helpers
# ---------------------------------------------------------------------------

"""Map `(edge, depth, along)` to the interior (i, j) that provides source data."""
@inline function _edge_interior_ij(q_e::Int, d::Int, s::Int, Nc::Int, Hp::Int)
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
@inline function _edge_halo_ij(e::Int, d::Int, s::Int, Nc::Int, Hp::Int)
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

# ---------------------------------------------------------------------------
# Corner fill — FV3 tp_core.F90 rotation formulas
# ---------------------------------------------------------------------------

"""
    _set_corner_cells!(q, di, dj, k, Nc, Hp, N, dir)

Fill all 4 corner cells at offset `(di, dj)` within each `Hp × Hp` corner block.

`dir = 1` for X-sweep (rotates Y-edge halos into corners).
`dir = 2` for Y-sweep (rotates X-edge halos into corners).

Formulas translated from FV3 `tp_core.F90` via `src/Grids/halo_exchange.jl`.
"""
@inline function _set_corner_cells!(q, di::Int, dj::Int, k::Int,
                                     Nc::Int, Hp::Int, N::Int, dir::Int)
    oi_sw = Hp + 1 - di;  oj_sw = Hp + 1 - dj
    oi_se = Hp + Nc + di; oj_se = Hp + 1 - dj
    oi_ne = Hp + Nc + di; oj_ne = Hp + Nc + dj
    oi_nw = Hp + 1 - di;  oj_nw = Hp + Nc + dj

    @inbounds if dir == 1
        # X-direction: rotate Y-edge halos into corners
        q[oi_sw, oj_sw, k] = q[oj_sw, 2*Hp + 1 - oi_sw, k]
        q[oi_se, oj_se, k] = q[N + 1 - oj_se, oi_se - Nc, k]
        q[oi_ne, oj_ne, k] = q[oj_ne, 2*(Nc + Hp) + 1 - oi_ne, k]
        q[oi_nw, oj_nw, k] = q[N + 1 - oj_nw, oi_nw + Nc, k]
    else
        # Y-direction: rotate X-edge halos into corners
        q[oi_sw, oj_sw, k] = q[2*Hp + 1 - oj_sw, oi_sw, k]
        q[oi_se, oj_se, k] = q[Nc + oj_se, N + 1 - oi_se, k]
        q[oi_ne, oj_ne, k] = q[2*(Nc + Hp) + 1 - oj_ne, oi_ne, k]
        q[oi_nw, oj_nw, k] = q[oj_nw - Nc, N + 1 - oi_nw, k]
    end
    return nothing
end

@inline function _set_corner_cells!(q::AbstractArray{<:Any, 4}, di::Int, dj::Int,
                                     k::Int, t::Int, Nc::Int, Hp::Int,
                                     N::Int, dir::Int)
    oi_sw = Hp + 1 - di;  oj_sw = Hp + 1 - dj
    oi_se = Hp + Nc + di; oj_se = Hp + 1 - dj
    oi_ne = Hp + Nc + di; oj_ne = Hp + Nc + dj
    oi_nw = Hp + 1 - di;  oj_nw = Hp + Nc + dj

    @inbounds if dir == 1
        q[oi_sw, oj_sw, k, t] = q[oj_sw, 2*Hp + 1 - oi_sw, k, t]
        q[oi_se, oj_se, k, t] = q[N + 1 - oj_se, oi_se - Nc, k, t]
        q[oi_ne, oj_ne, k, t] = q[oj_ne, 2*(Nc + Hp) + 1 - oi_ne, k, t]
        q[oi_nw, oj_nw, k, t] = q[N + 1 - oj_nw, oi_nw + Nc, k, t]
    else
        q[oi_sw, oj_sw, k, t] = q[2*Hp + 1 - oj_sw, oi_sw, k, t]
        q[oi_se, oj_se, k, t] = q[Nc + oj_se, N + 1 - oi_se, k, t]
        q[oi_ne, oj_ne, k, t] = q[2*(Nc + Hp) + 1 - oj_ne, oi_ne, k, t]
        q[oi_nw, oj_nw, k, t] = q[oj_nw - Nc, N + 1 - oi_nw, k, t]
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Halo fill as point operations (src/Architectures.jl)
#
# Each operation's body is written once. `launch!` runs it as loops on the host
# and, on a device, as the backend's `fusion_policy` says: one kernel for all 24
# panel edges and one for all 6 panels' corners by default (CUDA), or one launch
# per edge and per panel corner with its panel arrays bound (Metal); either way
# one synchronization follows. Edge fills write only halo cells and read only
# interior cells, so they are independent; corner fills read the edge halos,
# so they follow in a `Sequence`.
# ---------------------------------------------------------------------------

# Operations are built on the host, where their panel and edge numbers are
# checked: inside a kernel `select_panel` maps every panel number outside 1:5 to
# panel 6, and the edge index helpers map every edge number outside 1:3 to West.
_check_halo_panel(p::Int) =
    p in 1:6 || throw(ArgumentError("cubed-sphere panel number must be in 1:6; got $p"))
_check_halo_edge(e::Int) =
    e in 1:4 || throw(ArgumentError("cubed-sphere panel edge number must be in 1:4; got $e"))

"""
Fill the halo of edge `e` of panel `p` from the interior of panel `neighbor`,
whose matching edge is `reciprocal`; `flip` reverses the along-edge index.
"""
struct EdgeHaloFill <: AbstractPointOp
    p::Int
    e::Int
    neighbor::Int
    reciprocal::Int
    flip::Bool
    function EdgeHaloFill(p::Int, e::Int, neighbor::Int, reciprocal::Int, flip::Bool)
        _check_halo_panel(p); _check_halo_panel(neighbor)
        _check_halo_edge(e);  _check_halo_edge(reciprocal)
        return new(p, e, neighbor, reciprocal, flip)
    end
end

function EdgeHaloFill(conn::PanelConnectivity, p::Int, e::Int)
    _check_halo_panel(p); _check_halo_edge(e)
    nb = conn.neighbors[p][e]
    return EdgeHaloFill(p, e, nb.panel, reciprocal_edge(conn, p, e), nb.orientation >= 2)
end

# Indices: along-edge `s`, depth `d`, then the field's trailing dimensions.
index_space(::EdgeHaloFill, ctx) =
    (ctx.Nc, ctx.Hp, Base.tail(Base.tail(size(ctx.panels[1])))...)

# Run by itself (host loops, or a separate launch under `SeparateLaunches`) the two
# panel arrays are looked up once per edge (`bound_context`); in the
# fused device launch they are selected per work item.
bound_context(op::EdgeHaloFill, ctx) = merge(ctx, (dst = ctx.panels[op.p], src = ctx.panels[op.neighbor]))
@inline _edge_dst(op::EdgeHaloFill, ctx) = hasproperty(ctx, :dst) ? ctx.dst : select_panel(ctx.panels, op.p)
@inline _edge_src(op::EdgeHaloFill, ctx) = hasproperty(ctx, :src) ? ctx.src : select_panel(ctx.panels, op.neighbor)

@inline function apply_point!(op::EdgeHaloFill, ctx, s, d, K...)
    (; Nc, Hp) = ctx
    s_src = op.flip ? (Nc + 1 - s) : s
    i_src, j_src = _edge_interior_ij(op.reciprocal, d, s_src, Nc, Hp)
    i_dst, j_dst = _edge_halo_ij(op.e, d, s, Nc, Hp)
    @inbounds _edge_dst(op, ctx)[i_dst, j_dst, K...] = _edge_src(op, ctx)[i_src, j_src, K...]
    return nothing
end

"""Rotate the corner halo cells of panel `p` for sweep direction `dir` (1 = X, 2 = Y)."""
struct CornerHaloFill <: AbstractPointOp
    p::Int
    dir::Int
    function CornerHaloFill(p::Int, dir::Int)
        _check_halo_panel(p)
        dir in (1, 2) || throw(ArgumentError("corner fill sweep direction must be 1 (X) or 2 (Y); got $dir"))
        return new(p, dir)
    end
end

# Indices: corner offsets `di`, `dj`, then the field's trailing dimensions.
index_space(::CornerHaloFill, ctx) =
    (ctx.Hp, ctx.Hp, Base.tail(Base.tail(size(ctx.panels[1])))...)

bound_context(op::CornerHaloFill, ctx) = merge(ctx, (q = ctx.panels[op.p],))
@inline _corner_panel(op::CornerHaloFill, ctx) = hasproperty(ctx, :q) ? ctx.q : select_panel(ctx.panels, op.p)

@inline function apply_point!(op::CornerHaloFill, ctx, di, dj, K...)
    (; Nc, Hp) = ctx
    _set_corner_cells!(_corner_panel(op, ctx), di, dj, K..., Nc, Hp, Nc + 2 * Hp, op.dir)
    return nothing
end

_edge_halo_fills(conn::PanelConnectivity) =
    Fused(ntuple(slot -> EdgeHaloFill(conn, (slot - 1) ÷ 4 + 1, (slot - 1) % 4 + 1), Val(24)))

_corner_halo_fills(dir::Int) = Fused(ntuple(p -> CornerHaloFill(p, dir), 6))

function _halo_context(panels::NTuple{6}, mesh::CubedSphereMesh)
    Nc, Hp = mesh.Nc, mesh.Hp
    Hp <= Nc || throw(ArgumentError(
        "cubed-sphere halo width Hp = $Hp exceeds the panel size Nc = $Nc"))
    size(panels[1], 1) == size(panels[1], 2) == Nc + 2Hp &&
        all(q -> size(q) == size(panels[1]), panels) || throw(DimensionMismatch(
        "cubed-sphere halo fill expects six panels of $(Nc + 2Hp) × $(Nc + 2Hp) cells " *
        "(Nc = $Nc, Hp = $Hp) and equal trailing dimensions; got $(map(size, panels))"))
    return (; panels, Nc, Hp)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    fill_panel_halos!(panels::NTuple{6}, mesh::CubedSphereMesh; dir=0)

Fill halo regions of a 6-panel cubed-sphere field by copying interior data
from neighboring panels with correct edge-to-edge orientation mapping.

Each `panels[p]` must be `(Nc + 2Hp) × (Nc + 2Hp) × ...` (3-D fields or packed
`(x, y, z, tracer)` storage) with interior at `[Hp+1:Hp+Nc, Hp+1:Hp+Nc, ...]`.

If `dir` is 1 or 2, corner fill is performed for the given sweep direction
(1=X, 2=Y) using the FV3 tp_core rotation formulas.

On a GPU with the default fusion policy (CUDA) this is one launch for all panel
edges, a second for the corners when `dir` is 1 or 2, and one synchronization;
on Metal the 24 edges and 6 corner blocks are launched one by one with their
panel arrays bound, then synchronized once.
"""
function fill_panel_halos!(panels::NTuple{6, A},
                           mesh::CubedSphereMesh;
                           dir::Int = 0) where {A <: AbstractArray}
    ctx = _halo_context(panels, mesh)
    mesh.Hp == 0 && return nothing
    edges = _edge_halo_fills(mesh.connectivity)
    backend = get_backend(panels[1])
    if dir in (1, 2)
        launch!(Sequence(edges, _corner_halo_fills(dir)), ctx, backend)
    else
        launch!(edges, ctx, backend)
    end
    return nothing
end

"""
    copy_corners!(panels, mesh, dir)

Standalone corner fill for 6-panel fields. Rotates corner halo cells
according to sweep direction `dir` (1 = X-sweep, 2 = Y-sweep); with `Hp > 0`
any other `dir` throws an `ArgumentError`.
"""
function copy_corners!(panels::NTuple{6}, mesh::CubedSphereMesh, dir::Int)
    ctx = _halo_context(panels, mesh)
    mesh.Hp == 0 && return nothing
    launch!(_corner_halo_fills(dir), ctx, get_backend(panels[1]))
    return nothing
end

export fill_panel_halos!, copy_corners!
