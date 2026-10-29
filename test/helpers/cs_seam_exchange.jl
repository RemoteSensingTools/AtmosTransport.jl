using AtmosTransport, Random
using AtmosTransport.Operators: MonotoneLimiter
const CSSeamAdv = AtmosTransport.Operators.Advection
const CSSeamGrids = AtmosTransport.Grids

function cs_seam_fixture(FT, Nc, Nz, Nt, convention)
    Hp = 3
    mesh = CubedSphereMesh(; FT, Nc, Hp, convention)
    N = Nc + 2Hp
    rng = MersenneTwister(1729)
    mass = ntuple(_ -> FT(10) .+ rand(rng, FT, N, N, Nz), 6)
    tracer = ntuple(6) do p
        a = zeros(FT, N, N, Nz, Nt)
        for t in 1:Nt, k in 1:Nz, j in 1:Nc, i in 1:Nc
            q = t == 1 ? FT(0.4) : FT(0.4sin(0.3i + 0.2j + 0.1k + p + t))
            a[Hp+i, Hp+j, k, t] = mass[p][Hp+i, Hp+j, k] * q
        end
        a
    end
    am = ntuple(_ -> FT(0.3) .* randn(rng, FT, N + 1, N, Nz), 6)
    bm = ntuple(_ -> FT(0.3) .* randn(rng, FT, N, N + 1, Nz), 6)
    x = map(a -> view(a, Hp+1:Hp+Nc+1, Hp+1:Hp+Nc, :), am)
    y = map(a -> view(a, Hp+1:Hp+Nc, Hp+1:Hp+Nc+1, :), bm)
    raw_x, raw_y = map(copy, x), map(copy, y)
    AtmosTransport.Preprocessing.sync_all_cs_boundary_mirrors!(raw_x, raw_y, mesh.connectivity, Nc, Nz)
    foreach(copyto!, x, raw_x)
    foreach(copyto!, y, raw_y)
    CSSeamAdv.fill_panel_halos!(mass, mesh)
    CSSeamAdv.fill_panel_halos!(tracer, mesh)
    return mesh, mass, tracer, am, bm
end

# Independent scalar loops: reconstruction uses the established panel helpers,
# while physical face/cell indexing comes from the preprocessing contact map.
function reference_cs_group(rm, m, am, bm, mesh, scheme, direction, scale)
    Nc, Hp, Nz = mesh.Nc, mesh.Hp, size(m[1], 3)
    N = Int32(Nc + 2Hp)
    out, air = map(copy, rm), map(copy, m)
    for p in 1:6, k in 1:Nz, jj in 1:Nc, ii in 1:Nc
        i, j = Hp + ii, Hp + jj
        if direction == 1
            left = ii == 1 ? zero(scale) : scale * am[p][i, j, k]
            right = ii == Nc ? zero(scale) : scale * am[p][i+1, j, k]
            fleft = CSSeamAdv._xface_tracer_flux(Int32(i), j, k, rm[p], m[p], left, scheme, N)
            fright = CSSeamAdv._xface_tracer_flux(Int32(i+1), j, k, rm[p], m[p], right, scheme, N)
        else
            left = jj == 1 ? zero(scale) : scale * bm[p][i, j, k]
            right = jj == Nc ? zero(scale) : scale * bm[p][i, j+1, k]
            fleft = CSSeamAdv._yface_tracer_flux(i, Int32(j), k, rm[p], m[p], left, scheme, N)
            fright = CSSeamAdv._yface_tracer_flux(i, Int32(j+1), k, rm[p], m[p], right, scheme, N)
        end
        out[p][i, j, k] = rm[p][i, j, k] + fleft - fright
        air[p][i, j, k] = m[p][i, j, k] + left - right
    end
    location = AtmosTransport.Grids.cs_edge_face_location
    for p in 1:6, edge in 1:4
        neighbor = mesh.connectivity.neighbors[p][edge]
        q = neighbor.panel
        p < q || continue
        other = CSSeamGrids.reciprocal_edge(mesh.connectivity, p, edge)
        axis, _, _ = location(edge, 1, Nc)
        axis == direction || continue
        sign = edge in (CSSeamGrids.EDGE_EAST, CSSeamGrids.EDGE_NORTH) ? one(scale) : -one(scale)
        for k in 1:Nz, s in 1:Nc
            t = neighbor.orientation == 0 ? s : Nc + 1 - s
            _, i, j = location(edge, s, Nc)
            _, u, v = location(other, t, Nc)
            a, b = Hp + min(i, Nc), Hp + min(j, Nc)
            c, d = Hp + min(u, Nc), Hp + min(v, Nc)
            flux = scale * (axis == 1 ? am[p][Hp+i, Hp+j, k] : bm[p][Hp+i, Hp+j, k])
            transfer = axis == 1 ?
                CSSeamAdv._xface_tracer_flux(Int32(Hp+i), Hp+j, k, rm[p], m[p], flux, scheme, N) :
                CSSeamAdv._yface_tracer_flux(Hp+i, Int32(Hp+j), k, rm[p], m[p], flux, scheme, N)
            out[p][a, b, k] -= sign * transfer
            out[q][c, d, k] += sign * transfer
            air[p][a, b, k] -= sign * flux
            air[q][c, d, k] += sign * flux
        end
    end
    return out, air
end

cs_seam_interior(a, mesh) = view(a, mesh.Hp+1:mesh.Hp+mesh.Nc,
                               mesh.Hp+1:mesh.Hp+mesh.Nc, :)
cs_seam_maxdiff(a, b, mesh) = maximum(maximum(abs, cs_seam_interior(a[p], mesh) .-
                                             cs_seam_interior(b[p], mesh)) for p in 1:6)
