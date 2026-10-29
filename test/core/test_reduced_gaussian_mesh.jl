#!/usr/bin/env julia

using Test

using AtmosTransport
using .AtmosTransport.Grids: FaceIndexedFluxTopology, boundary_face_count,
                             boundary_face_range, face_cells, face_length,
                             face_normal, flux_topology, nboundaries,
                             ring_cell_count, ring_longitudes


@testset "ReducedGaussianMesh geometry" begin
    latitudes = [-75.0, -25.0, 25.0, 75.0]
    nlon_per_ring = [4, 8, 8, 4]
    mesh = ReducedGaussianMesh(latitudes, nlon_per_ring; FT=Float64)

    @test flux_topology(mesh) isa FaceIndexedFluxTopology
    @test nrings(mesh) == 4
    @test ring_cell_count(mesh, 1) == 4
    @test ring_cell_count(mesh, 2) == 8
    @test ring_longitudes(mesh, 1) == [45.0, 135.0, 225.0, 315.0]

    @test ncells(mesh) == 24
    @test nfaces(mesh) == 56

    total_area = sum(cell_area(mesh, c) for c in 1:ncells(mesh))
    earth_area = 4π * mesh.radius^2
    @test total_area ≈ earth_area rtol=1e-12

    @test face_normal(mesh, 1) == (1.0, 0.0)
    @test face_normal(mesh, ncells(mesh) + 1) == (0.0, 1.0)

    west_left, west_right = face_cells(mesh, 1)
    @test west_left == cell_index(mesh, 4, 1)
    @test west_right == cell_index(mesh, 1, 1)

    south_left, south_right = face_cells(mesh, ncells(mesh) + 1)
    @test south_left == 0
    @test south_right == cell_index(mesh, 1, 1)

    @test nboundaries(mesh) == 5
    @test boundary_face_count(mesh, 2) == 8

    first_interior_meridional_face = first(boundary_face_range(mesh, 2))
    south_cell, north_cell = face_cells(mesh, first_interior_meridional_face)
    @test south_cell == cell_index(mesh, 1, 1)
    @test north_cell == cell_index(mesh, 1, 2)

    polar_cell_faces = cell_faces(mesh, cell_index(mesh, 1, 1))
    mid_cell_faces = cell_faces(mesh, cell_index(mesh, 3, 2))
    @test length(polar_cell_faces) == 5
    @test length(mid_cell_faces) == 4

    @test face_length(mesh, ncells(mesh) + 1) ≈ 0.0 atol=1e-12
    @test face_length(mesh, first_interior_meridional_face) > 0.0
end
