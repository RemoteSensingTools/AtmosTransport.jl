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
