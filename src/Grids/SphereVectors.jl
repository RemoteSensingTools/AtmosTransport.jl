"""Unit Cartesian vector for longitude/latitude in degrees, retaining input precision."""
@inline lonlat_to_unit_xyz(lon, lat) = (cosd(lat) * cosd(lon), cosd(lat) * sind(lon), sind(lat))

"""Three-component dot product, preserving left-to-right arithmetic."""
@inline dot3(ax, ay, az, bx, by, bz) = ax * bx + ay * by + az * bz
@inline dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
"""Three-component cross product."""
@inline cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

export lonlat_to_unit_xyz, dot3, cross3

"""Unit Cartesian vector from longitude/latitude in radians (cos/sin arithmetic)."""
@inline unit_xyz_from_radians(lon, lat) =
    (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))

export unit_xyz_from_radians

"""Normalize a vector by its norm, with no epsilon floor (preprocessing policy)."""
@inline unit3(a) = a ./ sqrt(dot3(a, a))
"""Angle between unit vectors from their chord, accurate at small angles."""
@inline arc3(a, b) = 2 * asin(min(1.0, sqrt(dot3(a .- b, a .- b)) / 2))

"""Interpolate along a great-circle arc with angle `θ`."""
@inline slerp3(a, b, θ, t) = (sin((1 - t) * θ) .* a .+ sin(t * θ) .* b) ./ sin(θ)

export unit3, arc3, slerp3

"""Chord distance between degree longitude/latitude pairs on the unit sphere."""
lonlat_chord_distance(lon1, lat1, lon2, lat2) =
    sqrt(sum(abs2, lonlat_to_unit_xyz(lon1, lat1) .- lonlat_to_unit_xyz(lon2, lat2)))

export lonlat_chord_distance
