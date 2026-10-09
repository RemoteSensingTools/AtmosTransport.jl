# CubedSphereMesh: inverse projection (lon, lat) → (panel, s, t) for locating points.
# Split from CubedSphereMesh.jl (refactor phase 4); included by Grids.jl in this order.

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
