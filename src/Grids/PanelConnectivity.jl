"""Number of panels in a cubed sphere."""
const CS_PANEL_COUNT = 6
export CS_PANEL_COUNT

# ---------------------------------------------------------------------------
# Cubed-sphere panel connectivity for src
#
# GEOS-FP native cubed-sphere panel numbering (from nf dimension):
#   Panel 1: equatorial, centered ~350°E  (Africa/Atlantic)
#   Panel 2: equatorial, centered ~80°E   (Asia/Indian Ocean)
#   Panel 3: north polar cap              (lat > 35°N)
#   Panel 4: equatorial, centered ~170°E  (Pacific, rotated 90° CW)
#   Panel 5: equatorial, centered ~260°E  (Americas, rotated 90° CW)
#   Panel 6: south polar cap              (lat < 35°S)
#
# Panels 1 & 2 have local axes X=east, Y=north (standard orientation).
# Panels 4 & 5 have local axes X=south, Y=east in native GEOS arrays.
# Panels 3 & 6 are polar caps with curvilinear local axes.
#
# Each panel has 4 local-index edges: +Y (1), -Y (2), +X (3), -X (4).
# Historical names use north/south/east/west constants, but for rotated or
# Y-reversed panels these names are local-index directions, not necessarily
# geographic directions.
# `orientation` encodes the along-edge direction when transferring halo data:
#   0 = aligned (s on dst maps to s on src)
#   2 = reversed (s on dst maps to Nc+1-s on src)
#
# Ported from src/Grids/panel_connectivity.jl — verified against
# GEOS-FP C720 file corner coordinates and NetCDF metadata.
#
# References:
#   Putman & Lin (2007) — FV3 cubed-sphere grid
#   Martin et al. (2022, GMD) — GCHP v13
# ---------------------------------------------------------------------------

"""
    PanelEdge

A connection from one panel edge to a neighbor panel's edge.

# Fields
- `panel :: Int` — neighbor panel index (1-6)
- `orientation :: Int` — along-edge direction code (0=aligned, 2=reversed)
"""
struct PanelEdge
    panel       :: Int
    orientation :: Int
end

"""
    PanelConnectivity

Complete edge-to-edge connectivity for a 6-panel cubed sphere.

`neighbors[p][e]` gives the `PanelEdge` for panel `p`, edge `e`,
where edges are numbered by local index side:
1=+Y, 2=-Y, 3=+X, 4=-X.
"""
struct PanelConnectivity
    neighbors :: NTuple{6, NTuple{4, PanelEdge}}
end

const EDGE_NORTH = 1  # local +Y edge (j = Nc)
const EDGE_SOUTH = 2  # local -Y edge (j = 1)
const EDGE_EAST  = 3  # local +X edge (i = Nc)
const EDGE_WEST  = 4  # local -X edge (i = 1)

"""
    default_panel_connectivity() -> PanelConnectivity

Return the GEOS-FP native cubed-sphere panel connectivity.

Edge-to-edge connections (Panel p local edge → Panel q local edge):
  P1 +Y→P3 -X(rev)   P1 -Y→P6 +Y(aln)   P1 +X→P2 -X(aln)   P1 -X→P5 +Y(rev)
  P2 +Y→P3 -Y(aln)   P2 -Y→P6 +X(rev)   P2 +X→P4 -Y(rev)   P2 -X→P1 +X(aln)
  P3 +Y→P5 -X(rev)   P3 -Y→P2 +Y(aln)   P3 +X→P4 -X(aln)   P3 -X→P1 +Y(rev)
  P4 +Y→P5 -Y(aln)   P4 -Y→P2 +X(rev)   P4 +X→P6 -Y(rev)   P4 -X→P3 +X(aln)
  P5 +Y→P1 -X(rev)   P5 -Y→P4 +Y(aln)   P5 +X→P6 -X(aln)   P5 -X→P3 +Y(rev)
  P6 +Y→P1 -Y(aln)   P6 -Y→P4 +X(rev)   P6 +X→P2 -Y(rev)   P6 -X→P5 +X(aln)

The table is generated from the same GEOS-native corner geometry as
`panel_cell_corner_lonlat(mesh, p)`, so balance, treeify, and NetCDF output
share one convention contract.
"""
function default_panel_connectivity()
    return PanelConnectivity((
        # Panel 1 (equatorial ~350°E)
        (PanelEdge(3, 2), PanelEdge(6, 0), PanelEdge(2, 0), PanelEdge(5, 2)),
        # Panel 2 (equatorial ~80°E)
        (PanelEdge(3, 0), PanelEdge(6, 2), PanelEdge(4, 2), PanelEdge(1, 0)),
        # Panel 3 (north polar cap)
        (PanelEdge(5, 2), PanelEdge(2, 0), PanelEdge(4, 0), PanelEdge(1, 2)),
        # Panel 4 (equatorial ~170°E, GEOS Ydim reversed)
        (PanelEdge(5, 0), PanelEdge(2, 2), PanelEdge(6, 2), PanelEdge(3, 0)),
        # Panel 5 (equatorial ~260°E, GEOS Ydim reversed)
        (PanelEdge(1, 2), PanelEdge(4, 0), PanelEdge(6, 0), PanelEdge(3, 2)),
        # Panel 6 (south polar cap)
        (PanelEdge(1, 0), PanelEdge(4, 2), PanelEdge(2, 2), PanelEdge(5, 0)),
    ))
end

"""
    gnomonic_panel_connectivity() -> PanelConnectivity

Return the classical gnomonic cubed-sphere panel connectivity.

Panels 1-4 are the equatorial belt (+x, +y, -x, -y faces); panel 5 is the
north polar cap (+z face); panel 6 is the south polar cap (-z face).

This matches the gnomonic ordering used by the ERA5-CS preprocessor and all
CS transport binaries written by `open_streaming_cs_transport_binary`.

Edge-to-edge connections (P→Q with orientation 0=aligned, 2=reversed):
  P1 N→P5_S(0)  P1 S→P6_N(0)  P1 E→P2_W(0)  P1 W→P4_E(0)
  P2 N→P5_E(0)  P2 S→P6_E(2)  P2 E→P3_W(0)  P2 W→P1_E(0)
  P3 N→P5_N(2)  P3 S→P6_S(2)  P3 E→P4_W(0)  P3 W→P2_E(0)
  P4 N→P5_W(2)  P4 S→P6_W(0)  P4 E→P1_W(0)  P4 W→P3_E(0)
  P5 N→P3_N(2)  P5 S→P1_N(0)  P5 E→P2_N(0)  P5 W→P4_N(2)
  P6 N→P1_S(0)  P6 S→P3_S(2)  P6 E→P2_S(2)  P6 W→P4_S(0)
"""
function gnomonic_panel_connectivity()
    return PanelConnectivity((
        # Panel 1 (+x equatorial face)
        (PanelEdge(5, 0), PanelEdge(6, 0), PanelEdge(2, 0), PanelEdge(4, 0)),
        # Panel 2 (+y equatorial face)
        (PanelEdge(5, 0), PanelEdge(6, 2), PanelEdge(3, 0), PanelEdge(1, 0)),
        # Panel 3 (-x equatorial face)
        (PanelEdge(5, 2), PanelEdge(6, 2), PanelEdge(4, 0), PanelEdge(2, 0)),
        # Panel 4 (-y equatorial face)
        (PanelEdge(5, 2), PanelEdge(6, 0), PanelEdge(1, 0), PanelEdge(3, 0)),
        # Panel 5 (north polar cap, +z face)
        (PanelEdge(3, 2), PanelEdge(1, 0), PanelEdge(2, 0), PanelEdge(4, 2)),
        # Panel 6 (south polar cap, -z face)
        (PanelEdge(1, 0), PanelEdge(3, 2), PanelEdge(2, 2), PanelEdge(4, 0)),
    ))
end

"""
    reciprocal_edge(conn, p, e) -> Int

Find which edge of the neighbor panel connects back to panel `p` edge `e`.
"""
function reciprocal_edge(conn::PanelConnectivity, p::Int, e::Int)
    q = conn.neighbors[p][e].panel
    for eq in 1:4
        conn.neighbors[q][eq].panel == p && return eq
    end
    error("Broken panel connectivity: no reciprocal edge for P$p edge $e → P$q")
end

export PanelEdge, PanelConnectivity
export default_panel_connectivity, gnomonic_panel_connectivity
export reciprocal_edge
export EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
