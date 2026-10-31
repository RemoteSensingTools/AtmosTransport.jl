# ---------------------------------------------------------------------------
# NativeGRIBGeometry -- reduced Gaussian geometry helpers for ERA5 / IFS GRIB
#
# ERA5 complete/native GRIB mixes spectral (`gridType = sh`) and reduced
# Gaussian (`gridType = reduced_gg`) products. Humidity and many physics
# fields are already on the native reduced Gaussian grid. This file reads the
# reduced-Gaussian geometry directly from GRIB metadata so src can build a
# transport mesh without interpolating through a regular lat-lon grid.
# ---------------------------------------------------------------------------

using GRIB

"""
    ERA5ReducedGaussianGeometry{FT}

Reduced-Gaussian geometry read from a native ERA5 / IFS GRIB message.

# Fields
- `latitudes` -- ring-center latitudes [degrees], ordered south -> north
- `nlon_per_ring` -- number of longitude cells on each ring (`pl`)
- `grid_type` -- GRIB grid type, expected to be `"reduced_gg"`
- `gaussian_number` -- reduced-Gaussian resolution parameter `N`
- `longitude_first` -- longitude of first stored grid point [degrees]
- `longitude_last` -- longitude of last stored grid point [degrees]
- `j_scans_positively` -- raw GRIB scanning-direction flag
- `j_points_are_consecutive` -- raw GRIB storage-order flag
"""
struct ERA5ReducedGaussianGeometry{FT <: AbstractFloat}
    latitudes               :: Vector{FT}
    nlon_per_ring           :: Vector{Int}
    grid_type               :: String
    gaussian_number         :: Int
    longitude_first         :: FT
    longitude_last          :: FT
    j_scans_positively      :: Bool
    j_points_are_consecutive:: Bool
end

Base.eltype(::ERA5ReducedGaussianGeometry{FT}) where FT = FT
Base.eltype(::Type{<:ERA5ReducedGaussianGeometry{FT}}) where FT = FT

function Base.summary(geometry::ERA5ReducedGaussianGeometry)
    FT = eltype(geometry)
    return string(length(geometry.latitudes), "-ring ERA5ReducedGaussianGeometry{", FT, "} ",
                  "[gridType=", geometry.grid_type, ", N=", geometry.gaussian_number, "]")
end

function Base.show(io::IO, geometry::ERA5ReducedGaussianGeometry)
    print(io, summary(geometry), "\n",
          "├── latitude:  [", first(geometry.latitudes), ", ", last(geometry.latitudes), "] degrees\n",
          "├── nlon/ring: min=", minimum(geometry.nlon_per_ring), ", max=", maximum(geometry.nlon_per_ring), "\n",
          "└── storage:   jScansPositively=", geometry.j_scans_positively,
          ", jPointsAreConsecutive=", geometry.j_points_are_consecutive)
end

@inline _grib_string(msg, key::AbstractString) = String(msg[key])
@inline _grib_int(msg, key::AbstractString) = Int(msg[key])
@inline _grib_bool(msg, key::AbstractString) = msg[key] != 0

function _grib_date(msg)
    return try
        _grib_int(msg, "date")
    catch
        _grib_int(msg, "dataDate")
    end
end

function _grib_time(msg)
    return try
        _grib_int(msg, "time")
    catch
        _grib_int(msg, "dataTime")
    end
end

function _matches_reduced_gaussian_message(msg;
                                           param_id::Union{Nothing, Integer} = nothing,
                                           level::Union{Nothing, Integer} = nothing,
                                           date::Union{Nothing, Integer} = nothing,
                                           time::Union{Nothing, Integer} = nothing)
    _grib_string(msg, "gridType") == "reduced_gg" || return false
    param_id === nothing || _grib_int(msg, "paramId") == param_id || return false
    level === nothing || _grib_int(msg, "level") == level || return false
    date === nothing || _grib_date(msg) == date || return false
    time === nothing || _grib_time(msg) == time || return false

    return true
end

function _extract_ring_latitudes(latitudes_flat::AbstractVector{FT},
                                 nlon_per_ring::AbstractVector{Int}) where FT <: AbstractFloat
    expected_points = sum(nlon_per_ring)
    length(latitudes_flat) == expected_points ||
        error("Reduced Gaussian geometry mismatch: sum(pl) = $expected_points but latitudes has $(length(latitudes_flat)) entries")

    latitudes = Vector{FT}(undef, length(nlon_per_ring))
    tol = FT(1e-8)
    offset = 1

    for j in eachindex(nlon_per_ring)
        ring_nlon = nlon_per_ring[j]
        ring_last = offset + ring_nlon - 1
        ring_lats = @view latitudes_flat[offset:ring_last]
        latitudes[j] = ring_lats[1]
        maximum(abs.(ring_lats .- latitudes[j])) <= tol ||
            error("Unsupported reduced Gaussian storage layout: latitude varies within ring $j")
        offset = ring_last + 1
    end

    if latitudes[1] > latitudes[end]
        reverse!(latitudes)
        reverse!(nlon_per_ring)
    end
    all(diff(latitudes) .> zero(FT)) ||
        error("Reduced Gaussian ring latitudes must be strictly increasing south -> north")

    return latitudes, nlon_per_ring
end

function _geometry_from_reduced_gaussian_message(msg, ::Type{FT}) where FT <: AbstractFloat
    grid_type = _grib_string(msg, "gridType")
    grid_type == "reduced_gg" ||
        error("Expected reduced_gg GRIB message, got $(repr(grid_type))")

    nlon_per_ring = Int.(msg["pl"])
    isempty(nlon_per_ring) && error("Reduced Gaussian GRIB message has empty pl array")

    latitudes_flat = FT.(vec(msg["latitudes"]))
    latitudes, nlon_per_ring = _extract_ring_latitudes(latitudes_flat, nlon_per_ring)

    return ERA5ReducedGaussianGeometry{FT}(
        latitudes,
        nlon_per_ring,
        grid_type,
        _grib_int(msg, "N"),
        FT(msg["longitudeOfFirstGridPointInDegrees"]),
        FT(msg["longitudeOfLastGridPointInDegrees"]),
        _grib_bool(msg, "jScansPositively"),
        _grib_bool(msg, "jPointsAreConsecutive"),
    )
end

"""
    read_era5_reduced_gaussian_geometry(path; FT=Float64, param_id=nothing,
                                        level=nothing, date=nothing, time=nothing)

Read reduced-Gaussian ring geometry from the first matching native GRIB
message in `path`.

This is intended for ERA5 / IFS native fields already archived on the reduced
Gaussian grid, for example `q`, `o3`, and many cloud/physics variables.
Spectral files (`gridType = sh`) are intentionally rejected.
"""
function read_era5_reduced_gaussian_geometry(path::AbstractString;
                                             FT::Type{<:AbstractFloat} = Float64,
                                             param_id::Union{Nothing, Integer} = nothing,
                                             level::Union{Nothing, Integer} = nothing,
                                             date::Union{Nothing, Integer} = nothing,
                                             time::Union{Nothing, Integer} = nothing)
    isfile(path) || error("GRIB file not found: $path")

    gf = GribFile(path)
    geometry = nothing
    seen_grid_types = String[]

    try
        for msg in gf
            grid_type = _grib_string(msg, "gridType")
            grid_type in seen_grid_types || push!(seen_grid_types, grid_type)
            _matches_reduced_gaussian_message(msg;
                                              param_id=param_id,
                                              level=level,
                                              date=date,
                                              time=time) || continue
            geometry = _geometry_from_reduced_gaussian_message(msg, FT)
            break
        end
    finally
        destroy(gf)
    end

    geometry !== nothing && return geometry

    filter_bits = String[]
    param_id !== nothing && push!(filter_bits, "paramId=$(param_id)")
    level !== nothing && push!(filter_bits, "level=$(level)")
    date !== nothing && push!(filter_bits, "date=$(date)")
    time !== nothing && push!(filter_bits, "time=$(time)")
    filter_desc = isempty(filter_bits) ? "without additional filters" : join(filter_bits, ", ")
    grid_desc = isempty(seen_grid_types) ? "none" : join(seen_grid_types, ", ")

    error("No reduced_gg message found in $path ($filter_desc). Seen grid types: $grid_desc")
end

"""
    read_era5_reduced_gaussian_mesh(path; FT=Float64, radius=EARTH_RADIUS, kwargs...)

Convenience wrapper that reads native reduced-Gaussian GRIB geometry and
returns a `ReducedGaussianMesh` ready for `src` transport geometry.
"""
function read_era5_reduced_gaussian_mesh(path::AbstractString;
                                         FT::Type{<:AbstractFloat} = Float64,
                                         radius = FT(EARTH_RADIUS),
                                         kwargs...)
    geom = read_era5_reduced_gaussian_geometry(path; FT=FT, kwargs...)
    return ReducedGaussianMesh(geom.latitudes, geom.nlon_per_ring; FT=FT, radius=radius)
end

export ERA5ReducedGaussianGeometry
export read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh
