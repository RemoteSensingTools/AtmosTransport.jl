using Test, Random, LinearAlgebra
using AtmosTransport

const SeamAdvection = AtmosTransport.Operators.Advection
const SeamGrids = AtmosTransport.Grids

function seam_face_fixture(FT, Nc, Nz, rng)
    return ntuple(4) do field
        shape = field <= 2 ? (Nc + 1, Nc, Nz) : (Nc, Nc + 1, Nz)
        ntuple(_ -> FT.(rand(rng, -2048:2048, shape)), 6)
    end
end

seam_copy(fields) = map(panels -> map(copy, panels), fields)
seam_dot(a, b) = sum(dot(Float64.(a[f][p]), Float64.(b[f][p]))
                       for f in 1:4, p in 1:6)

function independent_seam_mean(fields, mesh)
    expected = seam_copy(fields)
    Nc = mesh.Nc
    # Use the preprocessing contact-map indexing as an independent reference
    # for the runtime exchange. Every expected value reads original fields.
    location = AtmosTransport.Grids.cs_edge_face_location
    for p in 1:6, edge in 1:4
        neighbor = mesh.connectivity.neighbors[p][edge]
        q = neighbor.panel
        other_edge = SeamGrids.reciprocal_edge(mesh.connectivity, p, edge)
        for s in 1:Nc, k in axes(fields[1][1], 3)
            t = neighbor.orientation == 0 ? s : Nc + 1 - s
            axis, i, j = location(edge, s, Nc)
            other_axis, u, v = location(other_edge, t, Nc)
            f = axis == 1 ? 1 : 3
            g = other_axis == 1 ? 1 : 3
            common = (fields[f][p][i,j,k] + fields[f+1][p][i,j,k] +
                      fields[g][q][u,v,k] + fields[g+1][q][u,v,k]) / 4
            expected[f][p][i,j,k] = common
            expected[f+1][p][i,j,k] = common
        end
    end
    return expected
end

@testset "Lin-Rood shared faces preserve interiors and transpose" begin
    for FT in (Float32, Float64), convention in
            (SeamGrids.GnomonicPanelConvention(), SeamGrids.GEOSNativePanelConvention())
        mesh = CubedSphereMesh(; FT, Nc=5, Hp=3, convention)
        rng = MersenneTwister(2917)
        x = seam_face_fixture(FT, mesh.Nc, 3, rng)
        y = seam_face_fixture(FT, mesh.Nc, 3, rng)
        px, py = seam_copy(x), seam_copy(y)
        SeamAdvection._share_lr_seam_faces!(px..., mesh)
        SeamAdvection._share_lr_seam_faces!(py..., mesh)
        @test px == independent_seam_mean(x, mesh)
        @test seam_dot(px, y) == seam_dot(x, py)
        repeated = seam_copy(px)
        SeamAdvection._share_lr_seam_faces!(repeated..., mesh)
        @test repeated == px
    end
end
