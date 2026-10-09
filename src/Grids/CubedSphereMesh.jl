# ---------------------------------------------------------------------------
# CubedSphereMesh — structured cubed-sphere grid for src
#
# Multiple-dispatch cubed-sphere geometry with computed cell areas, edge
# lengths, and panel connectivity. The coordinate law, center law, panel
# ordering, and longitude offset are explicit because "cubed sphere" is not a
# single geometry: GEOS-IT/GEOS-FP use the GMAO equal-distance gnomonic grid,
# while older synthetic targets used an equiangular gnomonic construction.
#
# Geometry is computed at construction time from the gnomonic projection
# using CubedSphere.SphericalGeometry for exact spherical area/distance.
#
# References:
#   Putman & Lin (2007) — FV3 cubed-sphere grid
#   Martin et al. (2022, GMD) — GCHP v13
# ---------------------------------------------------------------------------

using CubedSphere.SphericalGeometry: spherical_area_quadrilateral, spherical_distance

abstract type AbstractCubedSphereCoordinateLaw end
abstract type AbstractCubedSphereCenterLaw end
abstract type AbstractCubedSpherePanelConvention end
abstract type AbstractCubedSphereDefinition end

"""
    EquiangularGnomonic <: AbstractCubedSphereCoordinateLaw

Classical equiangular gnomonic face coordinate law.

For edge index `s = 1, ..., Nc + 1`, the local angle is

```math
α_s = -π/4 + (s - 1) π / (2Nc),
```

and the tangent-plane coordinate used by the gnomonic projection is

```math
ξ_s = tan(α_s).
```

This law is useful for controlled synthetic targets and legacy binaries. It is
not the native GEOS-IT/GEOS-FP cubed-sphere coordinate law.
"""
struct EquiangularGnomonic <: AbstractCubedSphereCoordinateLaw end

raw"""
    GMAOEqualDistanceGnomonic <: AbstractCubedSphereCoordinateLaw

GMAO/GEOS equal-distance gnomonic edge coordinate law used by GEOS-IT C180 and
GEOS-FP C720 cubed-sphere products.

This is `grid_type = 0` in the GEOS/FV and ESMF cubed-sphere code path. Let

```math
r = 1 / \sqrt{3}, \qquad α_0 = \sin^{-1}(r).
```

For edge index `s = 1, ..., Nc + 1`, GEOS computes

```math
β_s = -\frac{α_0}{Nc}(Nc + 2 - 2s),
\qquad
b_s = \tan(β_s)\cos(α_0).
```

The unrotated panel-1 cube point is proportional to

```math
(r, b_i, b_j).
```

Equivalently, the tangent-plane coordinates passed to the common gnomonic
panel map are `ξ_i = b_i / r`, `η_j = b_j / r`. This non-uniform edge law is
what produces the GEOS C180 panel-1 meridional center spacing of roughly
0.42° at the panel edges and 0.55° near the panel center.
"""
struct GMAOEqualDistanceGnomonic <: AbstractCubedSphereCoordinateLaw end

"""
    AngularMidpointCenter <: AbstractCubedSphereCenterLaw

Cell-center law that evaluates the continuous face coordinate map at the
midpoint of the logical cell:

```math
x_c(i,j) = x(i + 1/2, j + 1/2).
```

This matches the historical synthetic/equiangular `CubedSphereMesh` behavior.
It is not how GEOS writes native C180/C720 center coordinates.
"""
struct AngularMidpointCenter <: AbstractCubedSphereCenterLaw end

raw"""
    FourCornerNormalizedCenter <: AbstractCubedSphereCenterLaw

GEOS/FV/ESMF `cell_center2` center law.

Given the four unit corner vectors of cell `(i,j)`,
`v₁ = v(i,j)`, `v₂ = v(i+1,j)`, `v₃ = v(i+1,j+1)`,
`v₄ = v(i,j+1)`, the center is

```math
v_c = \frac{v_1 + v_2 + v_3 + v_4}
           {\|v_1 + v_2 + v_3 + v_4\|}.
```

GEOS-IT/GEOS-FP `lons`/`lats` arrays are written from this normalized
four-corner sum after the GEOS cubed-sphere corners have been generated.
"""
struct FourCornerNormalizedCenter <: AbstractCubedSphereCenterLaw end

"""
    CubedSphereDefinition(coordinate_law, center_law, panel_convention;
                          longitude_offset_deg=0, tag=:custom)

Complete horizontal cubed-sphere definition.

The definition is the geometry contract consumed by `CubedSphereMesh` and all
derived helpers. It deliberately separates:

- `coordinate_law`: how logical face edges are placed on the gnomonic cube,
- `center_law`: how cell centers are derived from the face geometry,
- `panel_convention`: how the six physical faces are ordered/oriented in file
  arrays and halo connectivity,
- `longitude_offset_deg`: a final rigid z-axis rotation applied to all panels.

For native GMAO files use [`GMAOCubedSphereDefinition`](@ref), which combines
`GMAOEqualDistanceGnomonic`, `FourCornerNormalizedCenter`,
`GEOSNativePanelConvention`, and the GEOS `-10°` longitude shift.
"""
struct CubedSphereDefinition{L <: AbstractCubedSphereCoordinateLaw,
                             C <: AbstractCubedSphereCenterLaw,
                             P <: AbstractCubedSpherePanelConvention} <: AbstractCubedSphereDefinition
    coordinate_law       :: L
    center_law           :: C
    panel_convention     :: P
    longitude_offset_deg :: Float64
    tag                  :: Symbol
end

function CubedSphereDefinition(law::L, center::C, convention::P;
                               longitude_offset_deg::Real = 0,
                               tag::Symbol = :custom) where {
                               L <: AbstractCubedSphereCoordinateLaw,
                               C <: AbstractCubedSphereCenterLaw,
                               P <: AbstractCubedSpherePanelConvention}
    return CubedSphereDefinition{L, C, P}(
        law, center, convention, Float64(longitude_offset_deg), tag)
end

"""
    GnomonicPanelConvention <: AbstractCubedSpherePanelConvention

Classical gnomonic panel numbering:
1-4 are equatorial panels, 5 is the north-pole panel, 6 is the south-pole panel.
"""
struct GnomonicPanelConvention <: AbstractCubedSpherePanelConvention end

"""
    GEOSNativePanelConvention <: AbstractCubedSpherePanelConvention

Panel numbering and orientation used by native GEOS-FP / GEOS-IT cubed-sphere
files: panels 1-2 are equatorial, 3 is north-pole, 4-5 are equatorial, and 6
is south-pole.

This type describes panel storage/order only. The GEOS `-10°` longitude shift
and GMAO equal-distance coordinate law live in `GMAOCubedSphereDefinition`.
"""
struct GEOSNativePanelConvention <: AbstractCubedSpherePanelConvention end

"""
    EquiangularCubedSphereDefinition(; convention=GnomonicPanelConvention(),
                                      longitude_offset_deg=0)

Legacy synthetic/equiangular cubed-sphere definition:
`EquiangularGnomonic` corners plus `AngularMidpointCenter` centers.
"""
EquiangularCubedSphereDefinition(;
    convention::AbstractCubedSpherePanelConvention = GnomonicPanelConvention(),
    longitude_offset_deg::Real = 0,
) = CubedSphereDefinition(EquiangularGnomonic(), AngularMidpointCenter(),
                           convention;
                           longitude_offset_deg = longitude_offset_deg,
                           tag = :equiangular_gnomonic)

"""
    GMAOCubedSphereDefinition(; convention=GEOSNativePanelConvention(),
                               longitude_offset_deg=-10)

Native GMAO/GEOS cubed-sphere definition used by GEOS-IT C180 and GEOS-FP
C720 files:

- `GMAOEqualDistanceGnomonic` edge/corner law,
- `FourCornerNormalizedCenter` (`cell_center2`) center law,
- `GEOSNativePanelConvention` by default,
- final `-10°` longitude shift away from the Japan corner.
"""
GMAOCubedSphereDefinition(;
    convention::AbstractCubedSpherePanelConvention = GEOSNativePanelConvention(),
    longitude_offset_deg::Real = -10,
) = CubedSphereDefinition(GMAOEqualDistanceGnomonic(),
                           FourCornerNormalizedCenter(),
                           convention;
                           longitude_offset_deg = longitude_offset_deg,
                           tag = :gmao_equal_distance)

"""
    GEOSIT_C180(; FT=Float64, Hp=1, radius=EARTH_RADIUS)
    GEOSFP_C720(; FT=Float64, Hp=1, radius=EARTH_RADIUS)

Convenience constructors for the two native GMAO cubed-sphere targets used for
GEOS-IT and GEOS-FP comparisons.
"""
GEOSIT_C180(; kwargs...) = CubedSphereMesh(; Nc = 180, definition = GMAOCubedSphereDefinition(), kwargs...)
GEOSFP_C720(; kwargs...) = CubedSphereMesh(; Nc = 720, definition = GMAOCubedSphereDefinition(), kwargs...)

coordinate_law(def::CubedSphereDefinition) = def.coordinate_law
center_law(def::CubedSphereDefinition) = def.center_law
panel_convention(def::CubedSphereDefinition) = def.panel_convention
longitude_offset_deg(def::CubedSphereDefinition) = def.longitude_offset_deg
cs_definition_tag(def::CubedSphereDefinition) = def.tag

coordinate_law_tag(::EquiangularGnomonic) = "equiangular_gnomonic"
coordinate_law_tag(::GMAOEqualDistanceGnomonic) = "gmao_equal_distance_gnomonic"
center_law_tag(::AngularMidpointCenter) = "angular_midpoint"
center_law_tag(::FourCornerNormalizedCenter) = "four_corner_normalized"

function _default_cs_definition(convention::GnomonicPanelConvention)
    return EquiangularCubedSphereDefinition(; convention)
end

function _default_cs_definition(convention::GEOSNativePanelConvention)
    return GMAOCubedSphereDefinition(; convention)
end

"""
    panel_connectivity_for(convention) -> PanelConnectivity

Return the `PanelConnectivity` that matches the given panel-numbering convention.
Dispatches to `gnomonic_panel_connectivity()` or `default_panel_connectivity()`
so that the mesh and all downstream code automatically use the right edge table.
"""
panel_connectivity_for(::GnomonicPanelConvention)   = gnomonic_panel_connectivity()
panel_connectivity_for(::GEOSNativePanelConvention) = default_panel_connectivity()

# ---------------------------------------------------------------------------
# Gnomonic projection
# ---------------------------------------------------------------------------

"""
    _gnomonic_xyz(ξ, η, panel) -> (x, y, z)

Map local tangent-plane coordinates `(ξ, η)` to Cartesian `(x, y, z)` on the
unit sphere via the gnomonic (central) projection for the given `panel`.

The gnomonic projection maps from a tangent plane touching the sphere at the
panel center to the sphere surface via a straight line through the sphere's
center. In angular coordinates, `α` and `β` are the local east and north
angles from the panel center, both in `[-π/4, π/4]`. The tangent-plane
coordinates are `ξ = tan(α)` and `η = tan(β)`.

The normalisation factor `d = 1 / √(1 + ξ² + η²)` projects back onto the
unit sphere (ensures `x² + y² + z² = 1`).

## Panel orientation (gnomonic convention)

| Panel | Center face | (x, y, z) at ξ=η=0 | Local ξ → global | Local η → global |
|-------|------------|---------------------|-----------------|-----------------|
| 1     | +x face    | (1, 0, 0)           | ξ → +y          | η → +z          |
| 2     | +y face    | (0, 1, 0)           | ξ → −x          | η → +z          |
| 3     | −x face    | (−1, 0, 0)          | ξ → −y          | η → +z          |
| 4     | −y face    | (0, −1, 0)          | ξ → +x          | η → +z          |
| 5     | +z (N pole)| (0, 0, 1)           | ξ → +y          | η → −x          |
| 6     | −z (S pole)| (0, 0, −1)          | ξ → +y          | η → +x          |

Panels 1-4 are the equatorial belt; panels 5, 6 are the polar caps.
"""
@inline function _gnomonic_xyz(ξ::FT, η::FT, panel::Int) where FT
    d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)   # gnomonic normalisation
    if     panel == 1;  return ( d,  ξ*d,  η*d)   # +x face
    elseif panel == 2;  return (-ξ*d,  d,  η*d)   # +y face
    elseif panel == 3;  return (-d, -ξ*d,  η*d)   # −x face
    elseif panel == 4;  return ( ξ*d, -d,  η*d)   # −y face
    elseif panel == 5;  return (-η*d,  ξ*d,  d)   # +z (north pole)
    else;               return ( η*d,  ξ*d, -d)   # −z (south pole)
    end
end

# ---------------------------------------------------------------------------
# CubedSphereMesh
# ---------------------------------------------------------------------------

"""
    CubedSphereMesh{FT, C, D} <: AbstractStructuredMesh

Structured cubed-sphere mesh with `Nc` cells per panel edge and 6 panels.

Cell areas and edge lengths are computed once at construction time from the
definition's corner geometry. All 6 panels share identical metrics by symmetry,
so only one set is stored.

# Fields
- `Nc :: Int` — cells per panel edge
- `Hp :: Int` — halo padding width (1 for slopes, 3 for PPM)
- `radius :: FT` — planet radius [m]
- `definition :: D` — coordinate law + center law + panel convention contract
- `convention :: C` — panel numbering convention
- `connectivity :: PanelConnectivity` — edge-to-edge panel connectivity
- `cell_areas :: Matrix{FT}` — `(Nc, Nc)` cell areas [m²] (one panel, all identical)
- `Δx :: Matrix{FT}` — `(Nc, Nc)` x-direction cell widths [m]
- `Δy :: Matrix{FT}` — `(Nc, Nc)` y-direction cell widths [m]
"""
struct CubedSphereMesh{FT <: AbstractFloat,
                       C <: AbstractCubedSpherePanelConvention,
                       D <: AbstractCubedSphereDefinition} <: AbstractStructuredMesh
    Nc           :: Int
    Hp           :: Int
    radius       :: FT
    definition   :: D
    convention   :: C
    connectivity :: PanelConnectivity
    cell_areas   :: Matrix{FT}
    Δx           :: Matrix{FT}
    Δy           :: Matrix{FT}
end

Base.eltype(::CubedSphereMesh{FT}) where FT = FT
Base.eltype(::Type{<:CubedSphereMesh{FT}}) where FT = FT

"""Number of panels (always 6 for a cubed sphere)."""
@inline panel_count(::CubedSphereMesh) = 6

"""Return the panel-numbering convention struct (Gnomonic or GEOSNative)."""
@inline panel_convention(m::CubedSphereMesh) = m.convention

"""Return the full cubed-sphere geometry definition carried by `mesh`."""
@inline cs_definition(m::CubedSphereMesh) = m.definition

"""Return the coordinate law carried by `mesh.definition`."""
@inline coordinate_law(m::CubedSphereMesh) = coordinate_law(cs_definition(m))

"""Return the center law carried by `mesh.definition`."""
@inline center_law(m::CubedSphereMesh) = center_law(cs_definition(m))

"""
    panel_labels(convention) -> NTuple{6, Symbol}

Return symbolic labels for each panel under the given convention.

- **Gnomonic**: `(:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)`
  — panels 1-4 are equatorial (centred on ±x, ±y axes), 5 and 6 are polar.
- **GEOS native**: `(:equatorial_1, ..., :north_pole, ..., :south_pole)`
  — panels 1-2 and 4-5 are equatorial, 3 is north pole, 6 is south pole.
  This matches the file-panel ordering in GEOS-FP/GEOS-IT NetCDF variables.
"""
panel_labels(::GnomonicPanelConvention) =
    (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)
panel_labels(::GEOSNativePanelConvention) =
    (:equatorial_1, :equatorial_2, :north_pole, :equatorial_4, :equatorial_5, :south_pole)
panel_labels(m::CubedSphereMesh) = panel_labels(panel_convention(m))

"""Cells per panel edge (same in both x and y by construction)."""
nx(m::CubedSphereMesh) = m.Nc
ny(m::CubedSphereMesh) = m.Nc

"""Total number of cells across all 6 panels: `6 × Nc²`."""
ncells(m::CubedSphereMesh) = panel_count(m) * m.Nc^2

"""Total number of faces across all 6 panels: `6 × 2 × Nc × (Nc + 1)` (x + y faces per panel)."""
nfaces(m::CubedSphereMesh) = panel_count(m) * 2 * m.Nc * (m.Nc + 1)

"""Cell area [m²] at local panel indices `(i, j)`. Same on all 6 panels by gnomonic symmetry."""
cell_area(m::CubedSphereMesh, i::Integer, j::Integer) = m.cell_areas[i, j]

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

function _validate_cubed_sphere_size(Nc::Integer)
    Nc > 0 || throw(ArgumentError("Nc must be positive"))
    return Int(Nc)
end

"""
    CubedSphereMesh(; Nc, FT=Float64, Hp=1, radius=EARTH_RADIUS,
                      convention=nothing, definition=nothing)

Construct a cubed-sphere mesh with `Nc` cells per panel edge.

`definition` is the authoritative geometry contract. If omitted, the
constructor chooses a definition from `convention`:

- `GnomonicPanelConvention()` -> `EquiangularCubedSphereDefinition()`
- `GEOSNativePanelConvention()` -> `GMAOCubedSphereDefinition()`

This preserves legacy synthetic CS behavior for default meshes while making
GEOS-IT/GEOS-FP targets use the actual GMAO equal-distance geometry whenever
the GEOS-native panel convention is requested.

# Keyword arguments
- `Nc :: Int` — cells per panel edge (e.g. 48, 90, 180, 720)
- `FT :: Type` — floating-point type (default `Float64`)
- `Hp :: Int` — halo padding width (default 1 for slopes; use 3 for PPM)
- `radius` — planet radius [m] (default Earth)
- `convention` — panel numbering convention used when `definition` is omitted
- `definition` — full [`CubedSphereDefinition`](@ref); overrides `convention`
"""
function CubedSphereMesh(; FT::Type{<:AbstractFloat} = Float64,
                           Nc::Int,
                           Hp::Int = 1,
                           radius = FT(EARTH_RADIUS),
                           convention = nothing,
                           definition = nothing)
    Nc = _validate_cubed_sphere_size(Nc)
    Hp >= 0 || throw(ArgumentError("Hp must be non-negative, got $Hp"))

    R = FT(radius)
    conv = convention === nothing ? GnomonicPanelConvention() : convention
    conv isa AbstractCubedSpherePanelConvention ||
        throw(ArgumentError("convention must be an AbstractCubedSpherePanelConvention, got $(typeof(conv))"))

    def = definition === nothing ? _default_cs_definition(conv) : definition
    def isa AbstractCubedSphereDefinition ||
        throw(ArgumentError("definition must be an AbstractCubedSphereDefinition, got $(typeof(def))"))
    if definition !== nothing && convention !== nothing &&
       typeof(panel_convention(def)) !== typeof(conv)
        throw(ArgumentError("definition panel convention $(typeof(panel_convention(def))) " *
                            "does not match convention $(typeof(conv)); pass only definition " *
                            "or build a matching CubedSphereDefinition"))
    end

    conv = panel_convention(def)
    conn = panel_connectivity_for(conv)

    areas = zeros(FT, Nc, Nc)
    dx    = zeros(FT, Nc, Nc)
    dy    = zeros(FT, Nc, Nc)

    # Compute geometry for panel 1 — all panels are identical by symmetry.
    # Corners, areas and lengths are evaluated in Float64 and rounded once:
    # Girard's theorem in Float32 errs by up to ~1e-4 per C90 cell, which
    # biases every area-weighted emission total.
    p = 1
    R64 = Float64(R)                   # the stored radius, evaluated in Float64
    for j in 1:Nc, i in 1:Nc
        v1 = _corner_xyz(def, Nc, i,     j,     p, Float64)
        v2 = _corner_xyz(def, Nc, i + 1, j,     p, Float64)
        v3 = _corner_xyz(def, Nc, i + 1, j + 1, p, Float64)
        v4 = _corner_xyz(def, Nc, i,     j + 1, p, Float64)

        # Cell area: exact spherical quadrilateral area via Girard's theorem.
        Ω = spherical_area_quadrilateral(v1, v2, v3, v4)
        areas[i, j] = FT(R64^2 * Ω)

        # Δx/Δy are centerline great-circle distances between opposing edge
        # midpoints. They are used as per-cell metric lengths by the flux
        # reconstruction code.
        mid_w = _normalize3(v1[1] + v4[1], v1[2] + v4[2], v1[3] + v4[3])
        mid_e = _normalize3(v2[1] + v3[1], v2[2] + v3[2], v2[3] + v3[3])
        dx[i, j] = FT(R64 * spherical_distance(mid_w, mid_e))

        mid_s = _normalize3(v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3])
        mid_n = _normalize3(v4[1] + v3[1], v4[2] + v3[2], v4[3] + v3[3])
        dy[i, j] = FT(R64 * spherical_distance(mid_s, mid_n))
    end

    return CubedSphereMesh{FT, typeof(conv), typeof(def)}(
        Nc, Hp, R, def, conv, conn, areas, dx, dy)
end

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function Base.summary(m::CubedSphereMesh)
    FT = eltype(m)
    conv = nameof(typeof(panel_convention(m)))
    def = cs_definition_tag(cs_definition(m))
    return string("C", m.Nc, " CubedSphereMesh{", FT, ", ", conv, ", ", def, "}")
end

function Base.show(io::IO, m::CubedSphereMesh)
    print(io, summary(m), "\n",
          "├── panels:    ", panel_count(m), " with ", m.Nc, "×", m.Nc, " cells each\n",
          "├── halo:      ", m.Hp, " cells\n",
          "├── ordering:  ", join(string.(panel_labels(m)), ", "), "\n",
          "├── coord law: ", coordinate_law_tag(coordinate_law(m)), "\n",
          "├── center:    ", center_law_tag(center_law(m)), "\n",
          "├── Δx range:  ", round(minimum(m.Δx)/1e3, digits=2), "–",
                             round(maximum(m.Δx)/1e3, digits=2), " km\n",
          "└── radius:    ", m.radius, " m")
end

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

# ---------------------------------------------------------------------------
# Inverse projection: (lon, lat) → (panel, s_frac, t_frac)
# ---------------------------------------------------------------------------
#
# Mirrors the forward `(s, t, panel) → (ξ, η) → unit xyz → (lon, lat)` chain
# end-to-end so callers (e.g. SatelliteGridding bin/scatter onto the native
# CS grid) can ask "which cell of which panel does this observation fall
# into?" without spinning up a nearest-neighbour search.
#
# Convention: `s_frac` and `t_frac` are continuous edge indices on
# `[1, Nc + 1]`, matching the forward helpers. Cell `(i, j)` of a panel
# spans `s ∈ [i, i + 1]` and `t ∈ [j, j + 1]`; so `floor(Int, s_frac)` is
# the integer cell index. Points outside `[1, Nc + 1]² × {1..6}` cannot
# arise from a valid lat/lon on the sphere — every direction on the unit
# sphere falls inside exactly one panel.

"""
    _gnomonic_inverse(x, y, z) -> (ξg, ηg, gpanel)

Inverse of [`_gnomonic_xyz`](@ref): given a unit-sphere Cartesian point
`(x, y, z)`, return the local tangent-plane coordinates `(ξg, ηg)` and the
gnomonic panel id `gpanel ∈ 1..6` whose central axis is closest to the
point. Ties on a panel edge resolve to the panel listed first in the
[+x, +y, +z] dominance order; the returned `(ξg, ηg)` then sit on the panel
edge and downstream callers should treat boundary indices defensively.
"""
@inline function _gnomonic_inverse(x::FT, y::FT, z::FT) where {FT}
    ax = abs(x); ay = abs(y); az = abs(z)
    if ax >= ay && ax >= az
        # ±x dominant: panel 1 if x > 0, panel 3 if x < 0.
        if x > 0
            # Forward 1: (d, ξd, ηd) ⇒ ξ = y/x, η = z/x.
            return (y / x, z / x, 1)
        else
            # Forward 3: (-d, -ξd, ηd) ⇒ d = -x, ξ = y/x, η = -z/x.
            return (y / x, -z / x, 3)
        end
    elseif ay >= az
        if y > 0
            # Forward 2: (-ξd, d, ηd) ⇒ ξ = -x/y, η = z/y.
            return (-x / y, z / y, 2)
        else
            # Forward 4: (ξd, -d, ηd) ⇒ d = -y, ξ = -x/y, η = -z/y.
            return (-x / y, -z / y, 4)
        end
    else
        if z > 0
            # Forward 5: (-ηd, ξd, d) ⇒ ξ = y/z, η = -x/z.
            return (y / z, -x / z, 5)
        else
            # Forward 6: (ηd, ξd, -d) ⇒ d = -z, ξ = -y/z, η = -x/z.
            return (-y / z, -x / z, 6)
        end
    end
end

"""
    _undo_panel_convention(convention, ξg, ηg, gpanel) -> (ξ, η, panel)

Undo the panel-id and (ξ, η) re-mapping that `_panel_xyz(convention, ...)`
applied *before* dispatching to [`_gnomonic_xyz`](@ref). The
`GnomonicPanelConvention` method is the identity; the
`GEOSNativePanelConvention` method reverses the 90° rotations on the
poles (`gpanel 5 ← panel 3`) and the equatorial-belt swaps
(`gpanel 3 ← panel 4`, `gpanel 4 ← panel 5`).
"""
@inline function _undo_panel_convention(::GnomonicPanelConvention,
                                        ξg::FT, ηg::FT, gpanel::Int) where {FT}
    return (ξg, ηg, gpanel)
end

@inline function _undo_panel_convention(::GEOSNativePanelConvention,
                                        ξg::FT, ηg::FT, gpanel::Int) where {FT}
    # Forward map applied by `_panel_xyz(::GEOSNativePanelConvention, ξ, η, panel)`:
    #     panel 3 → gpanel 5 with (ξg, ηg) = (-η, ξ)   ⇒ inverse: (ξ, η) = (ηg, -ξg)
    #     panel 4 → gpanel 3 with (ξg, ηg) = ( η, -ξ)  ⇒ inverse: (ξ, η) = (-ηg, ξg)
    #     panel 5 → gpanel 4 with (ξg, ηg) = ( η, -ξ)  ⇒ inverse: (ξ, η) = (-ηg, ξg)
    # gpanels 1, 2, 6 are passed through unchanged.
    if gpanel == 1
        return (ξg, ηg, 1)
    elseif gpanel == 2
        return (ξg, ηg, 2)
    elseif gpanel == 5
        return (ηg, -ξg, 3)
    elseif gpanel == 3
        return (-ηg, ξg, 4)
    elseif gpanel == 4
        return (-ηg, ξg, 5)
    else
        return (ξg, ηg, 6)
    end
end

raw"""
    _inverse_edge_tangent_coordinate(law, ξ, Nc, FT) -> s

Inverse of [`_edge_tangent_coordinate`](@ref): map a tangent-plane
coordinate `ξ ∈ [-1, 1]` back to the continuous edge index
`s ∈ [1, Nc + 1]`.

EquiangularGnomonic: forward `ξ = tan(α)` with `α = -π/4 + (s - 1)π/(2Nc)`
⇒ inverse `s = 1 + (α + π/4) · 2Nc/π` for `α = atan(ξ)`.

GMAOEqualDistanceGnomonic: forward `ξ = b/r`, `b = tan(β) cos(α₀)`,
`β = -(α₀/Nc)(Nc + 2 - 2s)` with `r = 1/√3, α₀ = asin(r)` ⇒ inverse
`s = (Nc + 2)/2 + β · Nc/(2 α₀)` for `β = atan(b / cos(α₀))`, `b = ξ · r`.
Closed-form, no Newton iteration.
"""
@inline function _inverse_edge_tangent_coordinate(::EquiangularGnomonic,
                                                  ξ::Real, Nc::Int,
                                                  ::Type{FT}) where {FT}
    α = atan(FT(ξ))
    return one(FT) + (α + FT(π) / 4) * (FT(2 * Nc) / FT(π))
end

@inline function _inverse_edge_tangent_coordinate(::GMAOEqualDistanceGnomonic,
                                                  ξ::Real, Nc::Int,
                                                  ::Type{FT}) where {FT}
    r = inv(sqrt(FT(3)))
    α0 = asin(r)
    b = FT(ξ) * r
    β = atan(b / cos(α0))
    return FT(Nc + 2) / FT(2) + β * FT(Nc) / (FT(2) * α0)
end

"""
    lonlat_to_panel_xy(mesh::CubedSphereMesh, lon_deg, lat_deg)
        -> (panel::Int, s_frac, t_frac)
    lonlat_to_panel_xy(def::CubedSphereDefinition, Nc, lon_deg, lat_deg, FT)
        -> (panel::Int, s_frac, t_frac)

Closed-form inverse of the cubed-sphere forward map for a single
`(longitude, latitude)` in degrees. Returns the GEOS-native panel id
`panel ∈ 1..6` and continuous edge indices `s_frac`, `t_frac` on the
panel's `[1, Nc + 1)²` parameter square (half-open at the upper edge).

The integer cell index for downstream gridders is `floor(Int, s_frac)`
along x and `floor(Int, t_frac)` along y, and is guaranteed to lie in
`1..Nc` for every finite `(lon, lat)`. Exact upper-panel-edge points
(which mathematically return `Nc + 1`) are nudged to
`prevfloat(FT(Nc + 1))` so the floor lands on the adjacent in-panel
cell — the standard half-open-cell convention that prevents
`BoundsError` in callers and avoids dropping observations along the
12 panel boundaries. Round-off slop below `1.0` is similarly raised to
`one(FT)`. The whole sphere is covered.

Honors `panel_convention(mesh)` (Gnomonic vs GEOS-native), the
mesh's `coordinate_law` (EquiangularGnomonic vs GMAOEqualDistanceGnomonic),
and the definition's `longitude_offset_deg` (e.g. the GMAO `-10°` shift).
"""
@inline function lonlat_to_panel_xy(def::CubedSphereDefinition, Nc::Int,
                                    lon_deg::Real, lat_deg::Real,
                                    ::Type{FT}) where {FT}
    # lonlat → unit xyz
    coslat = cosd(FT(lat_deg))
    x = coslat * cosd(FT(lon_deg))
    y = coslat * sind(FT(lon_deg))
    z = sind(FT(lat_deg))

    # Undo the rigid z-axis rotation (e.g. GEOS -10°) applied at the end of
    # the forward map. `_rotate_z_lon_offset` rotates by `+offset`; the
    # inverse rotates by `-offset`.
    offset = longitude_offset_deg(def)
    if !iszero(offset)
        x, y, z = _rotate_z_lon_offset(x, y, z, -offset)
    end

    # Inverse gnomonic: pick dominant axis ⇒ gnomonic panel + local (ξg, ηg).
    ξg, ηg, gpanel = _gnomonic_inverse(x, y, z)

    # Undo any panel-convention re-mapping (Gnomonic is identity; GEOS-native
    # un-rotates panels 3, 4, 5 and re-labels per the file ordering).
    ξ, η, panel = _undo_panel_convention(panel_convention(def), ξg, ηg, gpanel)

    # Invert the edge coordinate law to recover continuous indices.
    law = coordinate_law(def)
    s = _inverse_edge_tangent_coordinate(law, ξ, Nc, FT)
    t = _inverse_edge_tangent_coordinate(law, η, Nc, FT)

    # Clamp to [1, prevfloat(Nc + 1)]: forward(s = Nc + 1) lands on the upper
    # panel edge and the inverse mathematically returns exactly Nc + 1, which
    # `floor(Int, …)` maps to Nc + 1 — one past the last valid cell. Nudging
    # to `prevfloat(Nc + 1)` makes the documented `floor(Int, s_frac)` land in
    # `Nc` along that edge (the half-open-cell convention every structured
    # gridder uses). Round-off slop below 1 is similarly raised so the floor
    # stays at 1. The clamp loses at most one ULP of sub-cell position
    # information, which is well below SatelliteGridding's oversampling
    # granularity (~1 / n_oversample of a cell).
    s_max = prevfloat(FT(Nc + 1))
    t_max = prevfloat(FT(Nc + 1))
    s = clamp(s, one(FT), s_max)
    t = clamp(t, one(FT), t_max)
    return (panel, s, t)
end

@inline function lonlat_to_panel_xy(mesh::CubedSphereMesh{FT},
                                    lon_deg::Real, lat_deg::Real) where {FT}
    return lonlat_to_panel_xy(cs_definition(mesh), mesh.Nc, lon_deg, lat_deg, FT)
end

export AbstractCubedSphereCoordinateLaw, AbstractCubedSphereCenterLaw
export AbstractCubedSpherePanelConvention, AbstractCubedSphereDefinition
export EquiangularGnomonic, GMAOEqualDistanceGnomonic
export AngularMidpointCenter, FourCornerNormalizedCenter
export CubedSphereDefinition, EquiangularCubedSphereDefinition, GMAOCubedSphereDefinition
export GEOSIT_C180, GEOSFP_C720
export GnomonicPanelConvention, GEOSNativePanelConvention, panel_connectivity_for
export CubedSphereMesh, panel_count, panel_convention, panel_labels, cs_definition
export cs_face_edge_lengths, cs_corner_xyz
export coordinate_law, center_law, longitude_offset_deg, cs_definition_tag
export coordinate_law_tag, center_law_tag
export panel_cell_center_lonlat, panel_cell_corner_lonlat
export panel_cell_local_tangent_basis
export lonlat_to_panel_xy
