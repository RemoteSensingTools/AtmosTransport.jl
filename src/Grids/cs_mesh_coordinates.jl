# CubedSphereMesh forward geometry: panel → xyz/lon-lat projection, cell centers and corners,
# face edge lengths and local tangent bases (the mesh constructor uses these).
# Split from CubedSphereMesh.jl (refactor phase 4); included by Grids.jl in this order.

# ---------------------------------------------------------------------------
# Visualization helpers — lat/lon coordinates for cell centers and corners
# ---------------------------------------------------------------------------

"""Convert Cartesian `(x, y, z)` on the unit sphere to `(lon, lat)` in degrees, lon in [0, 360)."""
@inline function _xyz_to_lonlat(x, y, z)
    lon = atand(y, x)
    lat = asind(z / sqrt(x^2 + y^2 + z^2))
    lon < 0 && (lon += 360)
    return lon, lat
end

@inline _dot3(ax, ay, az, bx, by, bz) = ax * bx + ay * by + az * bz

@inline function _normalize3(x, y, z)
    n = sqrt(x^2 + y^2 + z^2)
    invn = inv(max(n, eps(typeof(n))))
    return (x * invn, y * invn, z * invn)
end

@inline function _rotate_z_lon_offset(x::FT, y::FT, z::FT, offset_deg::Real) where FT
    θ = FT(deg2rad(offset_deg))
    c = cos(θ)
    s = sin(θ)
    return (c * x - s * y, s * x + c * y, z)
end

@inline function _edge_tangent_coordinate(::EquiangularGnomonic,
                                          s::Real, Nc::Int,
                                          ::Type{FT}) where FT
    α = -FT(π) / 4 + (FT(s) - one(FT)) * (FT(π) / (2 * FT(Nc)))
    return tan(α)
end

@inline function _edge_tangent_coordinate(::GMAOEqualDistanceGnomonic,
                                          s::Real, Nc::Int,
                                          ::Type{FT}) where FT
    r = inv(sqrt(FT(3)))
    α0 = asin(r)
    β = -(α0 / FT(Nc)) * (FT(Nc) + FT(2) - FT(2) * FT(s))
    b = tan(β) * cos(α0)
    return b / r
end

@inline function _panel_xyz(::GnomonicPanelConvention, ξ::FT, η::FT, panel::Int) where FT
    return _gnomonic_xyz(ξ, η, panel)
end

"""
    _panel_xyz(::GEOSNativePanelConvention, ξ, η, panel)

GEOS-FP/GEOS-IT native cubed-sphere coordinates for arrays exposed by
NCDatasets as `(Xdim, Ydim, nf, ...)`.

GEOS native files use panel order `1, 2, north, 4, 5, south`. Panels 4 and 5
are stored with a quarter-turn relative to the mathematical gnomonic panel
order. Panel 3 (north pole) is stored with a quarter-turn so its lower-left
corner starts at ~35°E after the GMAO `-10°` longitude shift.

This method applies panel orientation only. The longitude shift is applied by
`_panel_xyz(definition, ξ, η, panel)`.
"""
@inline function _panel_xyz(::GEOSNativePanelConvention, ξ::FT, η::FT, panel::Int) where FT
    # Panels 4 and 5 are 90° CW rotated relative to gnomonic (the file-axis
    # `i` is the *latitude* index, `j` is the longitude index — opposite of
    # the gnomonic convention). The earlier `(ξ, -η)` Y-flip produced
    # ~89° lon disagreement at off-diagonal corners against the actual
    # `lons`, `lats` arrays inside GEOS-IT C180 NetCDFs (validated against
    # GEOSIT.20211202.A3dyn.C180.nc on 2026-04-29). The 90° CW rotation
    # `(η, -ξ)` reproduces those arrays at every off-diagonal cell.
    ξg, ηg, gpanel = if panel == 1
        (ξ, η, 1)
    elseif panel == 2
        (ξ, η, 2)
    elseif panel == 3
        (-η, ξ, 5)
    elseif panel == 4
        (η, -ξ, 3)
    elseif panel == 5
        (η, -ξ, 4)
    elseif panel == 6
        (ξ, η, 6)
    else
        throw(ArgumentError("invalid GEOS native panel id $panel"))
    end
    return _gnomonic_xyz(ξg, ηg, gpanel)
end

@inline function _panel_xyz(def::CubedSphereDefinition, ξ::FT, η::FT,
                           panel::Int) where FT
    x, y, z = _panel_xyz(panel_convention(def), ξ, η, panel)
    offset = longitude_offset_deg(def)
    return iszero(offset) ? (x, y, z) : _rotate_z_lon_offset(x, y, z, offset)
end

@inline function _continuous_panel_xyz(def::CubedSphereDefinition, Nc::Int,
                                       s::Real, t::Real, panel::Int,
                                       ::Type{FT}) where FT
    law = coordinate_law(def)
    ξ = _edge_tangent_coordinate(law, s, Nc, FT)
    η = _edge_tangent_coordinate(law, t, Nc, FT)
    return _panel_xyz(def, ξ, η, panel)
end

"""
    cs_corner_xyz(mesh::CubedSphereMesh, i, j, panel) -> (x, y, z)

Unit vector of corner `(i, j)` (`1 ≤ i, j ≤ Nc + 1`) of `panel`, in Float64;
corner `(i, j)` is the lower-left corner of cell `(i, j)`.
"""
cs_corner_xyz(mesh::CubedSphereMesh, i::Integer, j::Integer, panel::Integer) =
    _corner_xyz(mesh.definition, mesh.Nc, i, j, Int(panel), Float64)

"""
    cs_face_edge_lengths(mesh::CubedSphereMesh) -> (Lx, Ly)

Great-circle lengths of the cell faces of a panel (every panel has the same
geometry in its local indices). `Lx[i, j]` is the face between cells `i − 1`
and `i` of row `j` (`(Nc + 1) × Nc`); `Ly[i, j]` the face between cells
`j − 1` and `j` of column `i` (`Nc × (Nc + 1)`). Evaluated in Float64 from
the cell corners and rounded once. Unlike `mesh.Δx`/`mesh.Δy` (centerline
widths of a cell), these are the lengths through which a face flux passes.
"""
function cs_face_edge_lengths(mesh::CubedSphereMesh{FT}) where FT
    Nc, def, R = mesh.Nc, mesh.definition, Float64(mesh.radius)
    corner(i, j) = _corner_xyz(def, Nc, i, j, 1, Float64)
    Lx = [FT(R * spherical_distance(corner(i, j), corner(i, j + 1))) for i in 1:Nc+1, j in 1:Nc]
    Ly = [FT(R * spherical_distance(corner(i, j), corner(i + 1, j))) for i in 1:Nc, j in 1:Nc+1]
    return Lx, Ly
end

@inline function _corner_xyz(def::CubedSphereDefinition, Nc::Int,
                             i::Integer, j::Integer, panel::Int,
                             ::Type{FT}) where FT
    return _continuous_panel_xyz(def, Nc, i, j, panel, FT)
end

@inline function _cell_center_xyz(def::CubedSphereDefinition,
                                  ::AngularMidpointCenter,
                                  Nc::Int, i::Integer, j::Integer,
                                  panel::Int, ::Type{FT}) where FT
    return _continuous_panel_xyz(def, Nc, FT(i) + FT(0.5), FT(j) + FT(0.5),
                                 panel, FT)
end

@inline function _cell_center_xyz(def::CubedSphereDefinition,
                                  ::FourCornerNormalizedCenter,
                                  Nc::Int, i::Integer, j::Integer,
                                  panel::Int, ::Type{FT}) where FT
    v1 = _corner_xyz(def, Nc, i,     j,     panel, FT)
    v2 = _corner_xyz(def, Nc, i + 1, j,     panel, FT)
    v3 = _corner_xyz(def, Nc, i + 1, j + 1, panel, FT)
    v4 = _corner_xyz(def, Nc, i,     j + 1, panel, FT)
    return _normalize3(v1[1] + v2[1] + v3[1] + v4[1],
                       v1[2] + v2[2] + v3[2] + v4[2],
                       v1[3] + v2[3] + v3[3] + v4[3])
end

@inline function _cell_center_xyz(def::CubedSphereDefinition, Nc::Int,
                                  i::Integer, j::Integer, panel::Int,
                                  ::Type{FT}) where FT
    return _cell_center_xyz(def, center_law(def), Nc, i, j, panel, FT)
end

"""
    panel_cell_center_lonlat(Nc, panel, FT) -> (lons, lats)
    panel_cell_center_lonlat(mesh::CubedSphereMesh, panel) -> (lons, lats)

Return `(Nc, Nc)` arrays of cell-center longitudes and latitudes in degrees
for the given `panel`.

The `Nc` method is the classical gnomonic convention. The mesh method honors
`panel_convention(mesh)`, including GEOS-native panel ordering and orientation.
"""
function panel_cell_center_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat})
    return _panel_cell_center_lonlat(Nc, panel, FT, EquiangularCubedSphereDefinition())
end

function panel_cell_center_lonlat(mesh::CubedSphereMesh{FT}, panel::Int) where FT
    return _panel_cell_center_lonlat(mesh.Nc, panel, FT, cs_definition(mesh))
end

function _panel_cell_center_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat},
                                   convention::AbstractCubedSpherePanelConvention)
    return _panel_cell_center_lonlat(Nc, panel, FT, _default_cs_definition(convention))
end

function _panel_cell_center_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat},
                                   def::CubedSphereDefinition)
    lons = zeros(FT, Nc, Nc)
    lats = zeros(FT, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        x, y, z = _cell_center_xyz(def, Nc, i, j, panel, FT)
        lons[i, j], lats[i, j] = _xyz_to_lonlat(x, y, z)
    end
    return lons, lats
end

"""
    panel_cell_corner_lonlat(Nc, panel, FT) -> (lons, lats)
    panel_cell_corner_lonlat(mesh::CubedSphereMesh, panel[, T = eltype(mesh)]) -> (lons, lats)

Return `(Nc+1, Nc+1)` arrays of cell-corner longitudes and latitudes in
degrees. The mesh method honors `panel_convention(mesh)` and evaluates the
corners in `T`; regridding geometry passes `Float64` whatever the mesh precision.
"""
function panel_cell_corner_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat})
    return _panel_cell_corner_lonlat(Nc, panel, FT, EquiangularCubedSphereDefinition())
end

function panel_cell_corner_lonlat(mesh::CubedSphereMesh{FT}, panel::Int,
                                  ::Type{T} = FT) where {FT, T <: AbstractFloat}
    return _panel_cell_corner_lonlat(mesh.Nc, panel, T, cs_definition(mesh))
end

function _panel_cell_corner_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat},
                                   convention::AbstractCubedSpherePanelConvention)
    return _panel_cell_corner_lonlat(Nc, panel, FT, _default_cs_definition(convention))
end

function _panel_cell_corner_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat},
                                   def::CubedSphereDefinition)
    lons = zeros(FT, Nc + 1, Nc + 1)
    lats = zeros(FT, Nc + 1, Nc + 1)
    for j in 1:(Nc + 1), i in 1:(Nc + 1)
        x, y, z = _corner_xyz(def, Nc, i, j, panel, FT)
        lons[i, j], lats[i, j] = _xyz_to_lonlat(x, y, z)
    end
    return lons, lats
end

@inline function _panel_unit_xyz(def::CubedSphereDefinition, Nc::Int,
                                s::Float64, t::Float64, panel::Int)
    x, y, z = _continuous_panel_xyz(def, Nc, s, t, panel, Float64)
    return _normalize3(x, y, z)
end

@inline function _east_north_basis(x::Float64, y::Float64, z::Float64)
    rxy = hypot(x, y)
    if rxy > 1e-14
        east = (-y / rxy, x / rxy, 0.0)
    else
        # Longitude is singular at the pole. Pick a stable tangent basis; no
        # production C-grid has a cell center exactly at the pole, but odd Nc
        # synthetic tests can.
        east = (0.0, 1.0, 0.0)
    end
    north = (-z * east[2], z * east[1], x * east[2] - y * east[1])
    north = _normalize3(north...)
    return east, north
end

"""
    panel_cell_local_tangent_basis(mesh, panel)
        -> (x_east, x_north, y_east, y_north)

Return four `(Nc, Nc)` matrices describing the local panel-coordinate unit
vectors at cell centers in geographic `(east, north)` components.

For each cell, `(x_east[i,j], x_north[i,j])` is the unit vector for increasing
local X (`i`) and `(y_east[i,j], y_north[i,j])` is the unit vector for
increasing local Y (`j`). The helper honors `panel_convention(mesh)`, including
GEOS-native Y-reversed panels, and is the geometry contract used by
preprocessing wind rotation.
"""
function panel_cell_local_tangent_basis(mesh::CubedSphereMesh{FT}, panel::Int) where FT
    Nc = mesh.Nc
    def = cs_definition(mesh)
    h = 1.0e-6

    x_east  = zeros(FT, Nc, Nc)
    x_north = zeros(FT, Nc, Nc)
    y_east  = zeros(FT, Nc, Nc)
    y_north = zeros(FT, Nc, Nc)

    for j in 1:Nc, i in 1:Nc
        s = i + 0.5
        t = j + 0.5

        x0, y0, z0 = _cell_center_xyz(def, Nc, i, j, panel, Float64)
        xp, yp, zp = _panel_unit_xyz(def, Nc, s + h, t, panel)
        xm, ym, zm = _panel_unit_xyz(def, Nc, s - h, t, panel)
        xq, yq, zq = _panel_unit_xyz(def, Nc, s, t + h, panel)
        xr, yr, zr = _panel_unit_xyz(def, Nc, s, t - h, panel)

        dx = ((xp - xm) / (2h), (yp - ym) / (2h), (zp - zm) / (2h))
        dy = ((xq - xr) / (2h), (yq - yr) / (2h), (zq - zr) / (2h))

        # Remove radial roundoff before normalizing to tangent unit vectors.
        rx = _dot3(dx[1], dx[2], dx[3], x0, y0, z0)
        ry = _dot3(dy[1], dy[2], dy[3], x0, y0, z0)
        ex = _normalize3(dx[1] - rx * x0, dx[2] - rx * y0, dx[3] - rx * z0)
        ey = _normalize3(dy[1] - ry * x0, dy[2] - ry * y0, dy[3] - ry * z0)

        east, north = _east_north_basis(x0, y0, z0)

        x_east[i, j]  = FT(_dot3(ex[1], ex[2], ex[3], east[1], east[2], east[3]))
        x_north[i, j] = FT(_dot3(ex[1], ex[2], ex[3], north[1], north[2], north[3]))
        y_east[i, j]  = FT(_dot3(ey[1], ey[2], ey[3], east[1], east[2], east[3]))
        y_north[i, j] = FT(_dot3(ey[1], ey[2], ey[3], north[1], north[2], north[3]))
    end

    return (x_east, x_north, y_east, y_north)
end
