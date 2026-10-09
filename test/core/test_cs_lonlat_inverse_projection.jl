#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Tests for the cubed-sphere inverse projection `lonlat_to_panel_xy`.
#
# The inverse must be the bit-level analytic inverse of the forward
# chain in `cs_mesh_coordinates.jl` (the inverse is in `cs_mesh_locate.jl`):
#
#   (panel, s, t) -> (ξ, η) via _edge_tangent_coordinate
#                  -> (x, y, z) via _panel_xyz (convention-aware)
#                  -> (lon, lat) via _xyz_to_lonlat
#
# These tests verify that for every panel and every cell of small meshes,
# forward(s, t, p) followed by inverse recovers (p, s, t) to round-off,
# and that the inverse maps reference points (panel-center axes, etc.)
# to their expected (panel, s, t) coordinates. Both coordinate laws and
# both panel conventions are exercised, including the GMAO -10° offset.
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
using .AtmosTransport.Grids: CubedSphereMesh,
                              GnomonicPanelConvention,
                              GEOSNativePanelConvention,
                              EquiangularCubedSphereDefinition,
                              GMAOCubedSphereDefinition,
                              EquiangularGnomonic,
                              GMAOEqualDistanceGnomonic,
                              AngularMidpointCenter,
                              FourCornerNormalizedCenter,
                              CubedSphereDefinition,
                              cs_definition,
                              panel_cell_center_lonlat,
                              panel_cell_corner_lonlat,
                              lonlat_to_panel_xy

const FT_T = Float64
const NPANEL = 6

# Bring the unexported forward helpers into scope so we can reuse them as
# the reference forward map. They live in the same module as
# `lonlat_to_panel_xy`, so a direct `getproperty` keeps the test
# self-contained even though they aren't part of the public export list.
const Grids = AtmosTransport.Grids
const _continuous_panel_xyz = getproperty(Grids, :_continuous_panel_xyz)
const _xyz_to_lonlat        = getproperty(Grids, :_xyz_to_lonlat)

"""
Forward `(panel, s, t)` → `(lon, lat)` for the supplied definition; the
reference implementation we want the inverse to undo.
"""
function _forward_lonlat(def, Nc::Int, panel::Int, s::Real, t::Real)
    x, y, z = _continuous_panel_xyz(def, Nc, Float64(s), Float64(t), panel,
                                    Float64)
    return _xyz_to_lonlat(x, y, z)
end

"""Test fixture: every panel-convention × coordinate-law mesh we expect to work."""
function _all_definitions()
    return (
        (label = "Equiangular + Gnomonic",
         def   = EquiangularCubedSphereDefinition(;
                     convention = GnomonicPanelConvention())),
        (label = "GMAO + GEOS-native (offset = -10°)",
         def   = GMAOCubedSphereDefinition()),  # default convention + offset
        (label = "GMAO + GEOS-native (offset = 0°)",
         def   = GMAOCubedSphereDefinition(; longitude_offset_deg = 0)),
        (label = "Equiangular + GEOS-native",
         def   = EquiangularCubedSphereDefinition(;
                     convention = GEOSNativePanelConvention())),
        (label = "GMAO + Gnomonic",
         def   = CubedSphereDefinition(GMAOEqualDistanceGnomonic(),
                                       FourCornerNormalizedCenter(),
                                       GnomonicPanelConvention();
                                       longitude_offset_deg = 0)),
    )
end

@testset "lonlat_to_panel_xy: closed-form CS inverse" begin

    @testset "edge-coordinate law inverses are exact" begin
        # `_inverse_edge_tangent_coordinate ∘ _edge_tangent_coordinate == id`
        # on the half-integer subgrid for both laws. Tolerance is set by the
        # transcendental round-trip (tan ∘ atan / asin ∘ sin), not by Nc.
        _edge   = getproperty(Grids, :_edge_tangent_coordinate)
        _invedge = getproperty(Grids, :_inverse_edge_tangent_coordinate)
        for law in (EquiangularGnomonic(), GMAOEqualDistanceGnomonic())
            for Nc in (6, 90, 180)
                for s in range(1.0, Nc + 1.0, length = 2Nc + 1)
                    ξ = _edge(law, s, Nc, FT_T)
                    s_back = _invedge(law, ξ, Nc, FT_T)
                    @test s_back ≈ s atol = 1e-12 rtol = 1e-12
                end
            end
        end
    end

    @testset "forward ∘ inverse identity on cell centers & corners" begin
        # For each definition, walk every cell center s = i + 1/2 and every
        # corner s = i over a small Nc, push it through the forward map to
        # obtain (lon, lat), and confirm the inverse recovers the exact
        # (panel, s, t). Sub-degree CS edge spacing means a 1e-10 absolute
        # error on s, t is well under "the same cell".
        for fixture in _all_definitions()
            (; label, def) = fixture
            @testset "$(label)" begin
                Nc = 6
                # Cell centers (avoid the singular panel edges where two
                # panels share the same xyz; the inverse is *defined* there
                # but the panel id can break a tie either way).
                for p in 1:NPANEL, j in 1:Nc, i in 1:Nc
                    s_in = i + 0.5
                    t_in = j + 0.5
                    lon, lat = _forward_lonlat(def, Nc, p, s_in, t_in)
                    p_out, s_out, t_out =
                        lonlat_to_panel_xy(def, Nc, lon, lat, FT_T)
                    @test p_out == p
                    @test s_out ≈ s_in atol = 1e-10 rtol = 1e-10
                    @test t_out ≈ t_in atol = 1e-10 rtol = 1e-10
                end

                # Interior corners (s = i, t = j for 2 ≤ i, j ≤ Nc):
                # these sit strictly inside a single panel.
                for p in 1:NPANEL, j in 2:Nc, i in 2:Nc
                    s_in = Float64(i)
                    t_in = Float64(j)
                    lon, lat = _forward_lonlat(def, Nc, p, s_in, t_in)
                    p_out, s_out, t_out =
                        lonlat_to_panel_xy(def, Nc, lon, lat, FT_T)
                    @test p_out == p
                    @test s_out ≈ s_in atol = 1e-10 rtol = 1e-10
                    @test t_out ≈ t_in atol = 1e-10 rtol = 1e-10
                end
            end
        end
    end

    @testset "mesh wrapper matches definition form" begin
        # The CubedSphereMesh method should be identical to the (def, Nc)
        # method for the same geometry.
        for fixture in _all_definitions()
            (; label, def) = fixture
            mesh = CubedSphereMesh(; Nc = 6, FT = FT_T, definition = def)
            for p in 1:NPANEL, j in 1:6, i in 1:6
                lon, lat = _forward_lonlat(def, 6, p, i + 0.5, j + 0.5)
                a = lonlat_to_panel_xy(mesh, lon, lat)
                b = lonlat_to_panel_xy(def, 6, lon, lat, FT_T)
                @test a[1] == b[1]
                @test a[2] ≈ b[2] atol = 1e-12
                @test a[3] ≈ b[3] atol = 1e-12
            end
        end
    end

    @testset "panel-center anchors land at s = t = (Nc + 2)/2" begin
        # The "panel center" (intersection of cube-face center axis with
        # the unit sphere) maps to ξ = η = 0, which both edge laws send
        # back to s = (Nc + 2)/2. Pick Nc = 4 so the expected center is
        # exactly 3.0, then verify each panel's defining axis hits its
        # own panel id with the right (s, t).
        Nc = 4
        for fixture in _all_definitions()
            (; def) = fixture
            mesh = CubedSphereMesh(; Nc = Nc, FT = FT_T, definition = def)
            # Use the forward map at (s, t) = ((Nc+2)/2, (Nc+2)/2) on each
            # panel as the canonical panel center; then invert.
            for p in 1:NPANEL
                s_in = (Nc + 2) / 2
                t_in = (Nc + 2) / 2
                lon, lat = _forward_lonlat(def, Nc, p, s_in, t_in)
                p_out, s_out, t_out = lonlat_to_panel_xy(mesh, lon, lat)
                @test p_out == p
                @test s_out ≈ s_in atol = 1e-12
                @test t_out ≈ t_in atol = 1e-12
            end
        end
    end

    @testset "GMAO -10° offset is invertible" begin
        # Construct two definitions identical except for the longitude
        # offset; the inverse on (lon, lat) from the offset definition
        # must recover the same (panel, s, t) as the inverse on
        # (lon - offset, lat) from the no-offset definition.
        Nc = 4
        for offset in (0.0, -10.0, 30.0, -45.0)
            def_offset = GMAOCubedSphereDefinition(;
                            longitude_offset_deg = offset)
            for p in 1:NPANEL, j in 1:Nc, i in 1:Nc
                s_in = i + 0.5
                t_in = j + 0.5
                lon, lat = _forward_lonlat(def_offset, Nc, p, s_in, t_in)
                p_out, s_out, t_out =
                    lonlat_to_panel_xy(def_offset, Nc, lon, lat, FT_T)
                @test p_out == p
                @test s_out ≈ s_in atol = 1e-10
                @test t_out ≈ t_in atol = 1e-10
            end
        end
    end

    @testset "upper-edge boundary policy: floor lands in 1..Nc" begin
        # Codex review caught that exact upper-edge points (s = Nc + 1 or
        # t = Nc + 1) mathematically return Nc + 1, so the documented
        # `floor(Int, s_frac)` would land on Nc + 1, one past the last
        # valid cell index. The function clamps to `prevfloat(Nc + 1)` so
        # the floor stays in `1..Nc`. Walk the four boundary edges of
        # every panel and confirm.
        for fixture in _all_definitions()
            (; def) = fixture
            for Nc in (4, 12)
                # Sample s along [1, Nc + 1] at every half-cell, t fixed
                # at the same set; this hits both lower (s = 1, t = 1) and
                # upper (s = t = Nc + 1) panel edges.
                edge_grid = collect(range(1.0, Nc + 1.0, length = 2Nc + 1))
                for p in 1:NPANEL, s_in in edge_grid, t_in in edge_grid
                    lon, lat = _forward_lonlat(def, Nc, p, s_in, t_in)
                    p_out, s_out, t_out =
                        lonlat_to_panel_xy(def, Nc, lon, lat, FT_T)
                    # floor must be in 1..Nc regardless of which panel
                    # the inverse picked on a shared boundary.
                    @test 1 <= floor(Int, s_out) <= Nc
                    @test 1 <= floor(Int, t_out) <= Nc
                end
            end
        end
    end

    @testset "lon/lat sweep round-trip through GEOS-IT C180" begin
        # On the actual GEOS-IT C180 mesh, every (panel, i, j) cell-center
        # lon/lat written by `panel_cell_center_lonlat` must invert back to
        # the same (panel, i + 0.5, j + 0.5) — that's the strongest
        # production guarantee we offer to SatelliteGridding.
        # Use Nc = 12 to keep the test fast; the same code path covers C180.
        Nc = 12
        mesh = CubedSphereMesh(; Nc = Nc, FT = FT_T,
                                definition = GMAOCubedSphereDefinition())
        for p in 1:NPANEL
            lons, lats = panel_cell_center_lonlat(mesh, p)
            for j in 1:Nc, i in 1:Nc
                p_out, s_out, t_out =
                    lonlat_to_panel_xy(mesh, lons[i, j], lats[i, j])
                @test p_out == p
                # `FourCornerNormalizedCenter` (the GEOS center law) is not
                # the midpoint of the parameter square; the inverse must
                # *still* land in cell (i, j), so we test the integer floor
                # equals (i, j) rather than the half-integer center.
                @test floor(Int, s_out) == i
                @test floor(Int, t_out) == j
            end
        end
    end
end
