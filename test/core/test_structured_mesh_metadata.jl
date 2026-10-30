#!/usr/bin/env julia

using Test

using AtmosTransport
using .AtmosTransport.Grids:
    GnomonicPanelConvention, GEOSNativePanelConvention,
    panel_count, panel_convention, cs_definition, cs_definition_tag,
    coordinate_law, coordinate_law_tag, center_law, center_law_tag,
    longitude_offset_deg, panel_labels, panel_cell_center_lonlat,
    panel_cell_corner_lonlat, panel_cell_local_tangent_basis,
    EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST, reciprocal_edge

_lonlat_to_xyz(lon, lat) = (
    cosd(lat) * cosd(lon),
    cosd(lat) * sind(lon),
    sind(lat),
)

function _edge_segment_midpoint(lons, lats, edge::Int, s::Int)
    Nc = size(lons, 1) - 1
    a, b = if edge == EDGE_NORTH
        ((s, Nc + 1), (s + 1, Nc + 1))
    elseif edge == EDGE_SOUTH
        ((s, 1), (s + 1, 1))
    elseif edge == EDGE_EAST
        ((Nc + 1, s), (Nc + 1, s + 1))
    else
        ((1, s), (1, s + 1))
    end
    ax, ay, az = _lonlat_to_xyz(lons[a...], lats[a...])
    bx, by, bz = _lonlat_to_xyz(lons[b...], lats[b...])
    mx, my, mz = ax + bx, ay + by, az + bz
    n = sqrt(mx^2 + my^2 + mz^2)
    return (mx / n, my / n, mz / n)
end

_dist3(a, b) = sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2 + (a[3] - b[3])^2)

@testset "LatLonMesh style and validation" begin
    mesh = LatLonMesh(; FT=Float64, size=(720, 360), longitude=(-180, 180), latitude=(-90, 90))

    @test nx(mesh) == 720
    @test ny(mesh) == 360
    @test ncells(mesh) == 720 * 360
    @test nfaces(mesh) == (721 * 360) + (720 * 361)
    @test summary(mesh) == "720×360 LatLonMesh{Float64}"

    shown = sprint(show, mesh)
    @test occursin("longitude:", shown)
    @test occursin("latitude:", shown)
    @test occursin("Δλ=0.5", shown)
    @test occursin("Δφ=0.5", shown)

    @test_throws ArgumentError LatLonMesh(; FT=Float64, size=(0, 360))
    @test_throws ArgumentError LatLonMesh(; FT=Float64, size=(720, 360), latitude=(-95, 90))
    @test_throws ArgumentError LatLonMesh(; FT=Float64, Nx=720, Ny=360, size=(360, 180))
end

@testset "CubedSphereMesh conventions" begin
    default_mesh = CubedSphereMesh(; FT=Float64, Nc=180)
    gnomonic_mesh = CubedSphereMesh(; FT=Float64, Nc=180, convention=GnomonicPanelConvention())
    geos_mesh = CubedSphereMesh(; FT=Float64, Nc=180, convention=GEOSNativePanelConvention())

    @test panel_count(default_mesh) == 6
    @test nx(default_mesh) == 180
    @test ny(default_mesh) == 180
    @test ncells(default_mesh) == 6 * 180^2
    @test nfaces(default_mesh) == 6 * 2 * 180 * 181

    @test panel_convention(default_mesh) isa GnomonicPanelConvention
    @test panel_convention(geos_mesh) isa GEOSNativePanelConvention
    @test panel_convention(gnomonic_mesh) isa GnomonicPanelConvention
    @test cs_definition_tag(cs_definition(default_mesh)) === :equiangular_gnomonic
    @test cs_definition_tag(cs_definition(geos_mesh)) === :gmao_equal_distance
    @test coordinate_law_tag(coordinate_law(geos_mesh)) == "gmao_equal_distance_gnomonic"
    @test center_law_tag(center_law(geos_mesh)) == "four_corner_normalized"
    @test longitude_offset_deg(cs_definition(geos_mesh)) == -10.0
    @test panel_labels(default_mesh) == (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)
    @test panel_labels(geos_mesh) == (:equatorial_1, :equatorial_2, :north_pole, :equatorial_4, :equatorial_5, :south_pole)
    @test panel_labels(gnomonic_mesh) == (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)

    @test occursin("C180 CubedSphereMesh", summary(default_mesh))
    shown = sprint(show, geos_mesh)
    @test occursin("equatorial_1", shown)
    @test occursin("north_pole", shown)
    @test occursin("gmao_equal_distance_gnomonic", shown)
    @test occursin("four_corner_normalized", shown)

    lons, lats = panel_cell_center_lonlat(geos_mesh, 1)
    @test isapprox(lons[90, 140], 349.7229398200612; atol=1e-12)
    @test isapprox(lats[90, 140], 26.46802143378195; atol=1e-12)
    @test isapprox(lats[90, 2] - lats[90, 1], 0.41761077556816417; atol=1e-12)
    @test isapprox(lats[90, 91] - lats[90, 90], 0.5541113082381415; atol=1e-12)

    @test_throws ArgumentError CubedSphereMesh(; FT=Float64, Nc=0)
end

@testset "CubedSphereMesh connectivity matches corner geometry" begin
    for convention in (GnomonicPanelConvention(), GEOSNativePanelConvention())
        @testset "$(nameof(typeof(convention)))" begin
            Nc = 6
            mesh = CubedSphereMesh(; FT=Float64, Nc=Nc, convention=convention)
            corners = [panel_cell_corner_lonlat(mesh, p) for p in 1:6]

            for p in 1:6
                for edge in (EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST)
                    neighbor = mesh.connectivity.neighbors[p][edge]
                    q = neighbor.panel
                    reciprocal = reciprocal_edge(mesh.connectivity, p, edge)
                    for s in 1:Nc
                        t = neighbor.orientation == 0 ? s : Nc + 1 - s
                        p_mid = _edge_segment_midpoint(corners[p]..., edge, s)
                        q_mid = _edge_segment_midpoint(corners[q]..., reciprocal, t)
                        @test _dist3(p_mid, q_mid) < 1e-12
                    end
                end
            end
        end
    end
end

@testset "CubedSphereMesh local tangent basis" begin
    mesh = CubedSphereMesh(; FT=Float64, Nc=8, convention=GEOSNativePanelConvention())
    for p in 1:6
        x_east, x_north, y_east, y_north = panel_cell_local_tangent_basis(mesh, p)
        @test maximum(abs, x_east.^2 .+ x_north.^2 .- 1) < 1e-10
        @test maximum(abs, y_east.^2 .+ y_north.^2 .- 1) < 1e-10
    end

    # GEOS-native panels 4/5 match file order: local Xdim runs mostly
    # southward and local Ydim runs eastward.
    for p in (4, 5)
        x_east, x_north, y_east, y_north = panel_cell_local_tangent_basis(mesh, p)
        @test abs(sum(x_east) / length(x_east)) < 1e-10
        @test sum(x_north) / length(x_north) < -0.5
        @test sum(y_east) / length(y_east) > 0.5
        @test abs(sum(y_north) / length(y_north)) < 1e-10
    end
end


@testset "Shared CS geometry tags and panel storage" begin
    G = AtmosTransport.Grids
    for (conv, tag, definition, law, center, offset) in (
        (G.GnomonicPanelConvention(), "gnomonic", "equiangular_gnomonic", "equiangular_gnomonic", "angular_midpoint", 0.0),
        (G.GEOSNativePanelConvention(), "geos_native", "gmao_equal_distance", "gmao_equal_distance_gnomonic", "four_corner_normalized", -10.0))
        @test G.parse_panel_convention(uppercase(tag)) === conv
        @test G.parse_panel_convention(Symbol(uppercase(tag))) === conv
        @test G.panel_convention_tag(conv) == tag
        def = G.default_cs_definition(conv)
        tags = G.cs_geometry_tags(def)
        @test tags == (cs_definition=definition, cs_coordinate_law=law,
                       cs_center_law=center, longitude_offset_deg=offset, panel_convention=tag)
        @test typeof(G.parse_coordinate_law(uppercase(law))) == typeof(G.coordinate_law(def))
        @test typeof(G.parse_center_law(Symbol(uppercase(center)))) == typeof(G.center_law(def))
        @test G.cs_geometry_tags(G.parse_cs_definition(uppercase(definition), conv)) == tags
    end
    for parse in (G.parse_panel_convention, G.parse_coordinate_law, G.parse_center_law)
        @test_throws ArgumentError parse(:unknown)
    end
    @test_throws ArgumentError G.parse_cs_definition(:unknown, G.GnomonicPanelConvention())

    # Unique cell/level values expose panel permutations, transposes and stride mistakes.
    Nc, Nz = 3, 2
    flat = reshape(Float64.(1:(G.CS_PANEL_COUNT * Nc^2 * Nz)), :, Nz)
    panels = ntuple(_ -> zeros(Float32, Nc, Nc, Nz), G.CS_PANEL_COUNT)
    @test G.unpack_flat_to_panels_3d!(panels, flat, Nc, Nz) === panels
    for p in 1:G.CS_PANEL_COUNT, j in 1:Nc, i in 1:Nc, k in 1:Nz
        c = G.cs_flat_index(i, j, p, Nc)
        @test G.cs_panel_ij(c, Nc) == (i, j, p)
        @test panels[p][i, j, k] === Float32(flat[c, k])
    end
    two_d = ntuple(_ -> zeros(Float32, Nc, Nc), G.CS_PANEL_COUNT)
    G.unpack_flat_to_panels_2d!(two_d, flat[:, 1], Nc)
    packed = zeros(Float64, G.CS_PANEL_COUNT * Nc^2)
    @test G.pack_panels!(packed, 0, two_d) == length(packed)
    @test packed == flat[:, 1]
    @test_throws DimensionMismatch G.unpack_flat_to_panels_2d!(two_d, packed[2:end], Nc)
    @test_throws DimensionMismatch G.unpack_flat_to_panels_3d!(panels, flat[:, 1:1], Nc, Nz)
    # Integer widths used by adjoint kernels must not require a narrowing conversion.
    for edge in 1:4, s in 1:Nc, d in 1:2
        args = (edge, d, s, Nc, 2)
        @test G.edge_interior_ij(Int32.(args)...) == G.edge_interior_ij(args...)
        @test G.edge_halo_ij(Int32.(args)...) == G.edge_halo_ij(args...)
    end
end

@testset "Cubed-sphere total cell area" begin
    for FT in (Float32, Float64), Nc in (8, 24, 48)
        mesh = CubedSphereMesh(; FT, Nc)
        expected = FT(4) * FT(π) * mesh.radius^2
        @test abs(6sum(mesh.cell_areas) - expected) / expected < (FT == Float32 ? 1e-5 : 1e-12)
    end
end
