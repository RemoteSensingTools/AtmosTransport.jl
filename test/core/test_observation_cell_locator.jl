using Test, AtmosTransport, Random
using AtmosTransport.Grids: CubedSphereMesh, GnomonicPanelConvention, GEOSNativePanelConvention,
                            panel_cell_center_lonlat, panel_cell_corner_lonlat, cell_index,
                            ring_longitudes
using AtmosTransport.Output: cell_locator, locate, ncolumns, isvalid_lonlat, CellLocation,
                             LatLonCellLocator, ReducedGaussianCellLocator, CubedSphereCellLocator

# Independent lat-lon reference: linear scan over the faces.
function brute_force_latlon(mesh, lon, lat)
    west = Float64(mesh.λᶠ[1])
    λ = mod(Float64(lon) - west, 360.0) + west
    i = max(1, min(mesh.Nx, count(f -> Float64(f) <= λ, mesh.λᶠ)))
    j = max(1, min(mesh.Ny, count(f -> Float64(f) <= Float64(lat), mesh.φᶠ)))
    return (i, j)
end

unit_xyz(lon, lat) = (cosd(lat) * cosd(lon), cosd(lat) * sind(lon), sind(lat))
cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# Independent cubed-sphere reference: the point must lie inside the spherical
# quadrilateral spanned by the located cell's corners (gnomonic cell edges are
# great circles, so each edge is a plane through the origin).
function inside_cs_cell(corner_lons, corner_lats, i, j, lon, lat)
    p = unit_xyz(lon, lat)
    quad = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
    signs = Float64[]
    for q in 1:4
        a = quad[q]; b = quad[mod1(q + 1, 4)]
        ca = unit_xyz(corner_lons[a...], corner_lats[a...])
        cb = unit_xyz(corner_lons[b...], corner_lats[b...])
        push!(signs, dot3(cross3(ca, cb), p))
    end
    return all(>=(-1e-12), signs) || all(<=(1e-12), signs)
end

@testset "isvalid_lonlat rejects fill values and non-finite input" begin
    @test isvalid_lonlat(10.0, 20.0)
    @test isvalid_lonlat(550, -89)
    @test !isvalid_lonlat(-999999.0, 10.0)
    @test !isvalid_lonlat(10.0, -999999.0)
    @test !isvalid_lonlat(NaN, 0.0)
    @test !isvalid_lonlat(0.0, Inf)
    @test !isvalid_lonlat(0.0, 90.5)
end

@testset "lat-lon locator: global mesh, (-180, 180) convention" begin
    mesh = LatLonMesh(; FT = Float64, Nx = 8, Ny = 4)
    loc = cell_locator(mesh)
    @test loc isa LatLonCellLocator
    @test ncolumns(loc) == 32
    @test @inferred(Union{Nothing, CellLocation}, locate(loc, 10.0, 20.0)) isa CellLocation
    for j in 1:mesh.Ny, i in 1:mesh.Nx
        c = locate(loc, mesh.λᶜ[i], mesh.φᶜ[j])
        @test (c.i, c.j, c.panel) == (i, j, 1)
        @test c.cell == i + (j - 1) * mesh.Nx
        @test c.column == c.cell
        @test c.lon == mesh.λᶜ[i]
        @test c.lat == mesh.φᶜ[j]
        @test c.area == cell_area(mesh, i, j)
    end
    # Any longitude convention maps to the same cell.
    base = locate(loc, -170.0, 10.0)
    for lon in (190.0, 550.0, -530.0)
        @test locate(loc, lon, 10.0) == base
    end
    # The seam: -180 and 180 are the western face of cell 1; 360 and -360 are 0°.
    for lon in (-180.0, 180.0)
        @test locate(loc, lon, 0.0).i == 1
    end
    for lon in (360.0, -360.0)
        @test locate(loc, lon, 0.0) == locate(loc, 0.0, 0.0)
    end
    @test locate(loc, 0.0, 0.0).i == 5
    # A point exactly on a face belongs to the cell to its east / north.
    @test locate(loc, mesh.λᶠ[3], 0.0).i == 3
    @test locate(loc, 0.0, mesh.φᶠ[2]).j == 2
    # Poles clamp into the outermost rows.
    @test locate(loc, 0.0, -90.0).j == 1
    @test locate(loc, 0.0, 90.0).j == mesh.Ny
    @test locate(loc, 10, 20) == locate(loc, 10.0, 20.0)
    @test_throws ArgumentError locate(loc, 0.0, 91.0)
    @test_throws ArgumentError locate(loc, NaN, 0.0)
    @test_throws ArgumentError locate(loc, -999999.0, 10.0)
    @test_throws ArgumentError cell_locator(mesh; halo_width = 1)
    # Random points against the brute-force face scan.
    rng = MersenneTwister(7)
    for _ in 1:2000
        lon = rand(rng) * 1080 - 540
        lat = rand(rng) * 180 - 90
        c = locate(loc, lon, lat)
        @test (c.i, c.j) == brute_force_latlon(mesh, lon, lat)
    end
end

@testset "lat-lon locator: Float32 meshes with inexact cell widths stay global" begin
    for Nx in (200, 3600)
        mesh = LatLonMesh(; FT = Float32, Nx = Nx, Ny = 10)
        loc = cell_locator(mesh)
        @test locate(loc, 190.0, 0.0) !== nothing
        @test locate(loc, 190.0, 0.0) == locate(loc, -170.0, 0.0)
        @test locate(loc, -180.5, 0.0) !== nothing
        @test locate(loc, 180.0, 0.0).i == 1
        @test locate(loc, 179.9999, 0.0).i == Nx
    end
    mesh = LatLonMesh(; FT = Float32, Nx = 12, Ny = 6, longitude = (0, 360))
    loc = cell_locator(mesh)
    @test locate(loc, -10.0, 0.0) == locate(loc, 350.0, 0.0)
    @test locate(loc, 350.0, 0.0).i == 12
    @test locate(loc, 0.0, 0.0).i == 1
    for j in 1:mesh.Ny, i in 1:mesh.Nx
        c = locate(loc, mesh.λᶜ[i], mesh.φᶜ[j])
        @test (c.i, c.j) == (i, j)
    end
end

@testset "lat-lon locator: regional meshes wrap the convention, then test coverage" begin
    regional = LatLonMesh(; FT = Float64, Nx = 9, Ny = 3, longitude = (0, 90), latitude = (0, 45))
    rloc = cell_locator(regional)
    @test locate(rloc, 45.0, 20.0).i == 5
    @test locate(rloc, 90.0, 45.0) == CellLocation(1, 9, 3, 27, 27, regional.λᶜ[9], regional.φᶜ[3],
                                                   cell_area(regional, 9, 3))
    @test locate(rloc, 95.0, 20.0) === nothing
    @test locate(rloc, -5.0, 20.0) === nothing
    @test locate(rloc, 45.0, 50.0) === nothing
    @test locate(rloc, 45.0, -1.0) === nothing
    @test locate(rloc, -300.0, 20.0) == locate(rloc, 60.0, 20.0)
    @test @inferred(Union{Nothing, CellLocation}, locate(rloc, 95.0, 20.0)) === nothing

    west = LatLonMesh(; FT = Float64, Nx = 70, Ny = 30, longitude = (-130, -60), latitude = (20, 50))
    wloc = cell_locator(west)
    @test locate(wloc, 250.0, 35.0) == locate(wloc, -110.0, 35.0)
    @test locate(wloc, -110.0, 35.0).i == 21
    @test locate(wloc, -59.0, 35.0) === nothing
    @test locate(wloc, -131.0, 35.0) === nothing
end

@testset "reduced-Gaussian locator" begin
    for FT in (Float64, Float32)
        mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 8]; FT = FT)
        loc = cell_locator(mesh)
        @test loc isa ReducedGaussianCellLocator
        @test ncolumns(loc) == ncells(mesh)
        @test @inferred(locate(loc, 10.0, 20.0)) isa CellLocation
        # Ring 1 (south, 4 cells of 90°): longitude bins in [0, 360).
        @test locate(loc, 0.0, -45.0).i == 1
        @test locate(loc, 89.9, -45.0).i == 1
        @test locate(loc, 90.0, -45.0).i == 2
        @test locate(loc, 359.9, -45.0).i == 4
        @test locate(loc, -0.1, -45.0).i == 4
        @test locate(loc, -170.0, -45.0) == locate(loc, 190.0, -45.0)
        # Ring split at the midpoint latitude 0; poles clamp into the outer rings.
        @test locate(loc, 10.0, -0.1).j == 1
        @test locate(loc, 10.0, 0.0).j == 2
        @test locate(loc, 10.0, -90.0).j == 1
        @test locate(loc, 10.0, 90.0).j == 2
        for j in 1:nrings(mesh)
            for (i, lon) in enumerate(ring_longitudes(mesh, j))
                c = locate(loc, lon, mesh.latitudes[j])
                @test (c.i, c.j, c.panel) == (i, j, 1)
                @test c.cell == cell_index(mesh, i, j)
                @test c.column == c.cell
                @test c.lon ≈ lon
                @test c.lat == mesh.latitudes[j]
                @test c.area == cell_area(mesh, c.cell)
            end
        end
        @test_throws ArgumentError cell_locator(mesh; halo_width = 2)
        @test_throws ArgumentError locate(loc, -999.0, 10.0)
    end
end

@testset "cubed-sphere locator: both conventions, halo-aware columns" begin
    rng = MersenneTwister(11)
    for (convention, Hp) in ((GnomonicPanelConvention(), 3), (GEOSNativePanelConvention(), 2))
        mesh = CubedSphereMesh(; FT = Float64, Nc = 6, Hp = Hp, convention = convention)
        loc = cell_locator(mesh)
        @test loc isa CubedSphereCellLocator
        @test loc.halo_width == Hp
        @test @inferred(locate(loc, 10.0, 20.0)) isa CellLocation
        Np = mesh.Nc + 2Hp
        @test ncolumns(loc) == Np^2
        slab = LinearIndices((Np, Np))
        corners = ntuple(p -> panel_cell_corner_lonlat(mesh, p), 6)
        for p in 1:6
            lons, lats = panel_cell_center_lonlat(mesh, p)
            for j in 1:mesh.Nc, i in 1:mesh.Nc
                c = locate(loc, lons[i, j], lats[i, j])
                @test (c.panel, c.i, c.j) == (p, i, j)
                @test c.cell == i + (j - 1) * mesh.Nc
                @test c.column == slab[Hp + i, Hp + j]
                @test c.lon == lons[i, j]
                @test c.lat == lats[i, j]
                @test c.area == cell_area(mesh, i, j)
            end
            # Panel corners and edges stay inside the index range.
            clons, clats = corners[p]
            for j in axes(clons, 2), i in axes(clons, 1)
                c = locate(loc, clons[i, j], clats[i, j])
                @test 1 <= c.panel <= 6 && 1 <= c.i <= mesh.Nc && 1 <= c.j <= mesh.Nc
            end
        end
        # Random points: the located cell's corner quadrilateral must contain the point.
        for _ in 1:1500
            lon = rand(rng) * 720 - 360
            lat = rand(rng) * 180 - 90
            c = locate(loc, lon, lat)
            @test inside_cs_cell(corners[c.panel]..., c.i, c.j, lon, lat)
        end
        # Longitude convention independence and the poles.
        @test locate(loc, -170.0, 20.0) == locate(loc, 190.0, 20.0)
        for (lon, lat) in ((123.0, 90.0), (-45.0, -90.0))
            c = locate(loc, lon, lat)
            @test inside_cs_cell(corners[c.panel]..., c.i, c.j, lon, lat)
        end
        # An explicit halo override changes only the storage column.
        unpadded = cell_locator(mesh; halo_width = 0)
        a = locate(loc, 10.0, 10.0)
        b = locate(unpadded, 10.0, 10.0)
        @test (a.panel, a.i, a.j, a.cell) == (b.panel, b.i, b.j, b.cell)
        @test b.column == a.i + (a.j - 1) * mesh.Nc
        @test_throws ArgumentError cell_locator(mesh; halo_width = -1)
        @test_throws ArgumentError locate(loc, -999999.0, 10.0)
    end
    f32 = CubedSphereMesh(; FT = Float32, Nc = 4, Hp = 1)
    floc = cell_locator(f32)
    for p in 1:6
        lons, lats = panel_cell_center_lonlat(f32, p)
        for j in 1:4, i in 1:4
            c = locate(floc, lons[i, j], lats[i, j])
            @test (c.panel, c.i, c.j) == (p, i, j)
        end
    end
end
