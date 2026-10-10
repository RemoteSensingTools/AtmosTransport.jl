# The cubed-sphere halo exchange against a frozen transcription of the
# pre-point-operation host loops, on the library's host path and on the fused
# device kernel run through KernelAbstractions' CPU backend.
# test/diagnostic/test_cs_halo_fill_gpu.jl repeats the check on CUDA.
using Test
include(joinpath(@__DIR__, "..", "helpers", "cs_halo_fill.jl"))

@testset "Halo exchange matches the reference" begin
    host = check_halo_fill(library_fill!, library_corners!, copy)
    @test length(host) == 48 && all(host)
    kernel = check_halo_fill(kernel_fill!, kernel_corners!, copy)
    @test length(kernel) == 48 && all(kernel)
    # The 24 edge operations build into one concrete type (no run-time dispatch).
    conn = CubedSphereMesh(; Nc = 4).connectivity
    @test only(Base.return_types(HaloAdv._edge_halo_fills, Tuple{typeof(conn)})) ===
          HaloArch.Fused{NTuple{24, HaloAdv.EdgeHaloFill}}
end

@testset "Halo exchange rejects a halo deeper than the panel" begin
    mesh = CubedSphereMesh(; Nc = 2, Hp = 3)
    panels = ntuple(_ -> zeros(8, 8, 2), 6)
    @test_throws ArgumentError HaloAdv.fill_panel_halos!(panels, mesh)
    mesh = CubedSphereMesh(; Nc = 4, Hp = 2)
    @test_throws DimensionMismatch HaloAdv.fill_panel_halos!(ntuple(_ -> zeros(9, 8, 2), 6), mesh)
    # Five valid 8 × 8 × 2 panels and one that differs in a horizontal or only
    # in a trailing dimension.
    wide = ntuple(p -> p == 6 ? zeros(8, 9, 2) : zeros(8, 8, 2), 6)
    deep = ntuple(p -> p == 3 ? zeros(8, 8, 3) : zeros(8, 8, 2), 6)
    @test_throws DimensionMismatch HaloAdv.fill_panel_halos!(wide, mesh)
    @test_throws DimensionMismatch HaloAdv.fill_panel_halos!(deep, mesh)
    @test_throws DimensionMismatch HaloAdv.copy_corners!(deep, mesh, 1)
    # Without halos there is nothing to fill, but malformed panels are still rejected.
    flat = CubedSphereMesh(; Nc = 4, Hp = 0)
    @test HaloAdv.fill_panel_halos!(ntuple(_ -> zeros(4, 4, 2), 6), flat) === nothing
    @test_throws DimensionMismatch HaloAdv.fill_panel_halos!(ntuple(_ -> zeros(5, 4, 2), 6), flat)
    @test_throws DimensionMismatch HaloAdv.copy_corners!(ntuple(_ -> zeros(5, 4, 2), 6), flat, 1)
end

@testset "Halo operations reject invalid panel and edge numbers" begin
    Edge, Corner = HaloAdv.EdgeHaloFill, HaloAdv.CornerHaloFill
    @test isbitstype(Edge) && isbitstype(Corner)
    @test Edge(1, 1, 3, 4, true) isa Edge
    @test Corner(6, 2) isa Corner
    # (p, e, neighbor, reciprocal) with one number out of range.
    for args in ((0, 1, 3, 4), (7, 1, 3, 4), (1, 0, 3, 4), (1, 5, 3, 4),
                 (1, 1, 0, 4), (1, 1, 7, 4), (1, 1, 3, 0), (1, 1, 3, 5))
        @test_throws ArgumentError Edge(args..., false)
    end
    conn = CubedSphereMesh(; Nc = 4).connectivity
    @test_throws ArgumentError Edge(conn, 7, 1)
    @test_throws ArgumentError Edge(conn, 1, 5)
    for (p, dir) in ((0, 1), (7, 1), (1, 0), (1, 3))
        @test_throws ArgumentError Corner(p, dir)
    end
    mesh = CubedSphereMesh(; Nc = 4, Hp = 2)
    @test_throws ArgumentError HaloAdv.copy_corners!(ntuple(_ -> zeros(8, 8, 2), 6), mesh, 0)
end
