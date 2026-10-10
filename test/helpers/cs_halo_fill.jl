using AtmosTransport, Random, KernelAbstractions
using AtmosTransport.Grids: GnomonicPanelConvention, GEOSNativePanelConvention, reciprocal_edge
const HaloAdv = AtmosTransport.Operators.Advection
const HaloArch = AtmosTransport.Architectures

# ---------------------------------------------------------------------------
# Reference: a frozen, independent transcription of the pre-point-operation
# host loops (src/Operators/Advection/HaloExchange.jl at f6437e89), with the
# 3-D and 4-D loops combined over CartesianIndices of the trailing dimensions,
# so the library is checked against code it does not share.
# ---------------------------------------------------------------------------
function _ref_interior_ij(q_e, d, s, Nc, Hp)
    q_e == 1 && return (Hp + s, Hp + Nc + 1 - d)   # north
    q_e == 2 && return (Hp + s, Hp + d)            # south
    q_e == 3 && return (Hp + Nc + 1 - d, Hp + s)   # east
    return (Hp + d, Hp + s)                        # west
end
function _ref_halo_ij(e, d, s, Nc, Hp)
    e == 1 && return (Hp + s, Hp + Nc + d)
    e == 2 && return (Hp + s, Hp + 1 - d)
    e == 3 && return (Hp + Nc + d, Hp + s)
    return (Hp + 1 - d, Hp + s)
end
function _ref_corner!(q, di, dj, K, Nc, Hp, N, dir)
    oi_sw = Hp + 1 - di;  oj_sw = Hp + 1 - dj
    oi_se = Hp + Nc + di; oj_se = Hp + 1 - dj
    oi_ne = Hp + Nc + di; oj_ne = Hp + Nc + dj
    oi_nw = Hp + 1 - di;  oj_nw = Hp + Nc + dj
    if dir == 1
        q[oi_sw, oj_sw, K] = q[oj_sw, 2*Hp + 1 - oi_sw, K]
        q[oi_se, oj_se, K] = q[N + 1 - oj_se, oi_se - Nc, K]
        q[oi_ne, oj_ne, K] = q[oj_ne, 2*(Nc + Hp) + 1 - oi_ne, K]
        q[oi_nw, oj_nw, K] = q[N + 1 - oj_nw, oi_nw + Nc, K]
    else
        q[oi_sw, oj_sw, K] = q[2*Hp + 1 - oj_sw, oi_sw, K]
        q[oi_se, oj_se, K] = q[Nc + oj_se, N + 1 - oi_se, K]
        q[oi_ne, oj_ne, K] = q[2*(Nc + Hp) + 1 - oj_ne, oi_ne, K]
        q[oi_nw, oj_nw, K] = q[oj_nw - Nc, N + 1 - oi_nw, K]
    end
end
function reference_corners!(panels, mesh, dir)
    Nc, Hp = mesh.Nc, mesh.Hp; N = Nc + 2Hp
    for q in panels, K in CartesianIndices(axes(q)[3:end]), dj in 1:Hp, di in 1:Hp
        _ref_corner!(q, di, dj, K, Nc, Hp, N, dir)
    end
    return panels
end
function reference_fill_panel_halos!(panels, mesh; dir = 0)
    Nc, Hp, conn = mesh.Nc, mesh.Hp, mesh.connectivity
    for p in 1:6, e in 1:4
        nb = conn.neighbors[p][e]
        q_e = reciprocal_edge(conn, p, e)
        dst, src = panels[p], panels[nb.panel]
        for K in CartesianIndices(axes(dst)[3:end]), d in 1:Hp, s in 1:Nc
            s_src = nb.orientation >= 2 ? Nc + 1 - s : s
            dst[_ref_halo_ij(e, d, s, Nc, Hp)..., K] = src[_ref_interior_ij(q_e, d, s_src, Nc, Hp)..., K]
        end
    end
    dir in (1, 2) && reference_corners!(panels, mesh, dir)
    return panels
end

# Distinct values everywhere (halos start as garbage) so any wrong source or
# destination index shows up.
halo_fixture(FT, N, dims...; seed) =
    (rng = MersenneTwister(seed); ntuple(_ -> rand(rng, FT, N, N, dims...), 6))

# (float type, array rank) cases: rank 3 is `(x, y, z)`, rank 4 is packed
# `(x, y, z, tracer)` storage. A backend without Float64 (Metal) needs a
# Float32-only list, e.g. `cases = ((Float32, 3), (Float32, 4))`.
const HALO_FILL_CASES = ((Float32, 3), (Float64, 3), (Float32, 4))

# Run `fill!(panels, mesh, dir)` / `corners!(panels, mesh, dir)` on `adapt`ed
# copies and compare with the reference. One Bool per (convention, Hp, case,
# dir) combination: 2 × 2 × length(cases) × (3 + 1), 48 for the default cases.
function check_halo_fill(fill!, corners!, adapt; cases = HALO_FILL_CASES)
    results = Bool[]
    for conv in (GnomonicPanelConvention(), GEOSNativePanelConvention()), Hp in (1, 3)
        Nc, Nz, Nt = 12, 5, 3
        mesh = CubedSphereMesh(; FT = Float64, Nc, Hp, convention = conv)
        N = Nc + 2Hp
        for (FT, rank) in cases
            dims = rank == 3 ? (Nz,) : rank == 4 ? (Nz, Nt) :
                   throw(ArgumentError("halo fill test arrays have rank 3 or 4; got $rank"))
            for dir in (0, 1, 2)
                host = halo_fixture(FT, N, dims...; seed = 7 + dir)
                dev = map(adapt, host)
                reference_fill_panel_halos!(host, mesh; dir)
                fill!(dev, mesh, dir)
                push!(results, all(p -> Array(dev[p]) == host[p], 1:6))
            end
            host = halo_fixture(FT, N, dims...; seed = 11)
            dev = map(adapt, host)
            reference_corners!(host, mesh, 2)
            corners!(dev, mesh, 2)
            push!(results, all(p -> Array(dev[p]) == host[p], 1:6))
        end
    end
    return results
end

library_fill!(p, mesh, dir) = HaloAdv.fill_panel_halos!(p, mesh; dir)
library_corners!(p, mesh, dir) = HaloAdv.copy_corners!(p, mesh, dir)

# The device kernel path on KernelAbstractions' CPU backend (the library runs a
# host loop there), so CI exercises the fused-kernel slot dispatch.
function _kernel_launch!(op, ctx)
    backend = KernelAbstractions.CPU()
    HaloArch._point_op_kernel!(backend, 64)(op, ctx; ndrange = HaloArch.index_space(op, ctx))
    KernelAbstractions.synchronize(backend)
end
function kernel_fill!(p, mesh, dir)
    ctx = HaloAdv._halo_context(p, mesh)
    _kernel_launch!(HaloAdv._edge_halo_fills(mesh.connectivity), ctx)
    dir in (1, 2) && _kernel_launch!(HaloAdv._corner_halo_fills(dir), ctx)
end
kernel_corners!(p, mesh, dir) =
    _kernel_launch!(HaloAdv._corner_halo_fills(dir), HaloAdv._halo_context(p, mesh))
