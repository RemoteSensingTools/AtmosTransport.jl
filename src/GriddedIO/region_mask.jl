# ---------------------------------------------------------------------------
# Integer region masks (TransCom regions, PFT maps, combined region × PFT codes)
#
# A region mask assigns one integer code to each cell of a surface-flux grid.
# The runtime splits a flux into one tracer per selected code
# (`[tagged_fluxes]`), so the mask must be on the flux's own grid: a lat-lon
# mask `(lon, lat)` with the flux file's coordinates, or, for `cs_native`
# fluxes, a `(Xdim, Ydim, nf)` mask on the run's cube. Codes are arbitrary
# integers (no assumption of small or contiguous ids); fill values never count
# as a code.
# ---------------------------------------------------------------------------

"""
    AbstractRegionMask

An integer region mask read by [`read_region_mask`](@ref):
[`LatLonRegionMask`](@ref) or [`CubedSphereRegionMask`](@ref). Both carry the
codes `codes` (`Int64`), the cells that hold a code `valid` (fill values and
NaN are not valid), and the `file` and `variable` they were read from.
"""
abstract type AbstractRegionMask end

"""
    LatLonRegionMask

A lat-lon region mask `(Nx, Ny)` in the runtime orientation
([`latlon_orientation`](@ref): latitude ascending, longitudes from the first
value ≥ 0, modulo 360°), with its oriented centre coordinates `lon`, `lat`.
"""
struct LatLonRegionMask <: AbstractRegionMask
    codes    :: Matrix{Int64}
    valid    :: BitMatrix
    lon      :: Vector{Float64}
    lat      :: Vector{Float64}
    file     :: String
    variable :: String
end

"""
    CubedSphereRegionMask

A cubed-sphere region mask `(Nc, Nc, 6)` in the `cs_native` layout (Julia
dimensions `("Xdim", "Ydim", "nf")`, NetCDF `(nf, Ydim, Xdim)`), panels in the
file's (the run's) panel convention.
"""
struct CubedSphereRegionMask <: AbstractRegionMask
    codes    :: Array{Int64, 3}
    valid    :: BitArray{3}
    file     :: String
    variable :: String
end

"""Julia dimension names of a cubed-sphere region mask variable."""
const CS_REGION_MASK_DIMS = ("Xdim", "Ydim", "nf")

# Two lat-lon grids are the same when their oriented centres agree to this many
# degrees: far below any grid spacing, above the rounding of Float32 coordinates
# (≈ 2e-5° at 360°).
const _REGION_MASK_COORD_TOL_DEG = 1e-4

# Stored values (as NCDatasets decodes them, `missing` for fill) → Int64 codes
# and the cells that hold one. Integer values are converted exactly (no Float64
# round trip); float values must be integers (NaN: no code).
_region_code(x::Integer) = Int64(x)        # InexactError beyond Int64
function _region_code(x::AbstractFloat)
    isnan(x) && return nothing
    isfinite(x) && isinteger(x) && abs(x) <= maxintfloat(Float64) || return x
    return Int64(x)
end
_region_code(x) = x

function _region_codes(data::AbstractArray, file, variable)
    codes = zeros(Int64, size(data))
    valid = falses(size(data))
    for i in eachindex(data)
        x = data[i]
        x === missing && continue
        c = _region_code(x)
        c === nothing && continue
        c isa Int64 || throw(ArgumentError(
            "region mask '$(variable)' in $(file) holds the non-integer value $(x); a region " *
            "mask stores one integer code per cell (fill values mark cells without a code)"))
        codes[i] = c
        valid[i] = true
    end
    return codes, valid
end

# The mask's values with `_FillValue`/`missing_value` as `missing`, in their
# stored type (the `MaskedAsNaN` view of `read_raw`, without the Float64 copy).
_region_values(ds, variable) = Array(_cf_view(ds, variable, MaskedAsNaN()))

"""
    read_region_mask(file, variable) -> AbstractRegionMask

Read the integer region mask `variable` of the NetCDF `file`. A variable with
the Julia dimensions `CS_REGION_MASK_DIMS` (`Xdim`, `Ydim`, `nf`; six panels)
is a [`CubedSphereRegionMask`](@ref) (its `lons`/`lats` cell centres are
checked against the run's cube by [`check_region_mask_grid`](@ref)); any other
two-dimensional variable is a lat-lon mask whose coordinates
[`find_coordinate`](@ref) finds by name and whose dimensions must be those of
the coordinates (`(lon, lat)`, or `(lat, lon)`, which is transposed), oriented
as the runtime orients flux fields ([`LatLonRegionMask`](@ref)). Values may be
stored as any integer type (converted exactly to Int64) or float type (each an
integer up to 2^53 in magnitude). `_FillValue` and `missing_value` cells (and
NaN) hold no code.
"""
function read_region_mask(file::AbstractString, variable::AbstractString)
    isfile(file) || throw(ArgumentError("region mask file not found: $(file)"))
    return NCDataset(file) do ds
        haskey(ds, variable) || throw(ArgumentError(
            "region mask variable '$(variable)' not found in $(file)"))
        v = ds[variable]
        dims = Tuple(String.(dimnames(v)))
        if dims == CS_REGION_MASK_DIMS
            size(v, 1) == size(v, 2) && size(v, 3) == CS_PANEL_COUNT || throw(DimensionMismatch(
                "cubed-sphere region mask '$(variable)' in $(file) has shape $(size(v)); " *
                "expected (Nc, Nc, $(CS_PANEL_COUNT))"))
            codes, valid = _region_codes(_region_values(ds, variable), file, variable)
            return CubedSphereRegionMask(codes, valid, String(file), String(variable))
        end
        ndims(v) == 2 || throw(ArgumentError(
            "region mask '$(variable)' in $(file) has dimensions $(dims); expected a lat-lon " *
            "field (lon, lat) or a cubed-sphere field $(CS_REGION_MASK_DIMS)"))
        # Coordinates by name; the field's axes by the coordinates' dimensions.
        lon_name = find_coordinate(ds, LongitudeAxis())
        lat_name = find_coordinate(ds, LatitudeAxis())
        lon_dim = String(only(dimnames(ds[lon_name])))
        lat_dim = String(only(dimnames(ds[lat_name])))
        data = _region_values(ds, variable)
        if dims == (lat_dim, lon_dim)
            data = permutedims(data)
        elseif dims != (lon_dim, lat_dim)
            throw(ArgumentError(
                "region mask '$(variable)' in $(file) has dimensions $(dims); expected the " *
                "coordinate dimensions ($(repr(lon_dim)), $(repr(lat_dim))) in either order"))
        end
        lon, lat = Float64.(ds[lon_name][:]), Float64.(ds[lat_name][:])
        o = latlon_orientation(lon, lat)
        codes, valid = _region_codes(orient(data, o), file, variable)
        lon, lat = orient_coordinates(lon, lat, o)
        return LatLonRegionMask(codes, valid, lon, lat, String(file), String(variable))
    end
end

"""
    region_mask_values(mask) -> Vector{Int64}

The distinct codes of `mask`, ascending (fill cells excluded).
"""
region_mask_values(mask::AbstractRegionMask) = sort!(unique(mask.codes[mask.valid]))

"""
    region_indicator(mask, value) -> BitArray

`true` in the cells of `mask` whose code is `value` (never in fill cells).
"""
region_indicator(mask::AbstractRegionMask, value::Integer) =
    mask.valid .& (mask.codes .== Int64(value))

function _grid_label(lon::AbstractVector, lat::AbstractVector)
    span(x) = isempty(x) ? "empty" : "$(first(x))…$(last(x))"
    return "$(length(lon))×$(length(lat)) (lon $(span(lon)), lat $(span(lat)))"
end

"""
    check_region_mask_grid(mask::LatLonRegionMask, lon, lat; source)
    check_region_mask_grid(mask::CubedSphereRegionMask, mesh::CubedSphereMesh; source)

Throw an `ArgumentError` unless `mask` is on the grid of the flux it splits:
the flux's oriented lat-lon centres `lon`, `lat` (same sizes, every centre
within 1e-4°), or the run's cube `mesh` (same `Nc`; the mask file's
`lons`/`lats` cell centres, required, within a unit-sphere chord of 1e-7 of the
mesh's, panel by panel; a `cell_area` variable must match the mesh,
[`check_cell_area`](@ref)). The message
names both grids; `source` names the flux.
"""
function check_region_mask_grid(mask::LatLonRegionMask, lon::AbstractVector, lat::AbstractVector;
                                source::AbstractString = "surface flux")
    same = length(mask.lon) == length(lon) && length(mask.lat) == length(lat) &&
           all(abs.(mask.lon .- lon) .<= _REGION_MASK_COORD_TOL_DEG) &&
           all(abs.(mask.lat .- lat) .<= _REGION_MASK_COORD_TOL_DEG)
    same || throw(ArgumentError(
        "region mask '$(mask.variable)' in $(mask.file) is on the lat-lon grid " *
        "$(_grid_label(mask.lon, mask.lat)), but $(source) is on " *
        "$(_grid_label(lon, lat)); the mask must be on the flux's own grid (same " *
        "coordinates after orientation, within $(_REGION_MASK_COORD_TOL_DEG)°)"))
    return nothing
end

check_region_mask_grid(mask::CubedSphereRegionMask, lon::AbstractVector, lat::AbstractVector;
                       source::AbstractString = "surface flux") = throw(ArgumentError(
    "region mask '$(mask.variable)' in $(mask.file) is a C$(size(mask.codes, 1)) cubed-sphere " *
    "mask, but $(source) is on the lat-lon grid $(_grid_label(lon, lat)); the mask must be " *
    "on the flux's own grid"))

function check_region_mask_grid(mask::CubedSphereRegionMask, mesh::CubedSphereMesh;
                                source::AbstractString = "surface flux")
    Nc = size(mask.codes, 1)
    Nc == mesh.Nc || throw(ArgumentError(
        "region mask '$(mask.variable)' in $(mask.file) is a C$(Nc) cubed-sphere mask, but " *
        "$(source) is on the run's C$(mesh.Nc) cube; the mask must be on the flux's own grid"))
    # The panel layout: the stored cell centres must be the cube's (as for the
    # model surface-height files), and a stored `cell_area` its areas.
    NCDataset(mask.file) do ds
        haskey(ds, "lons") && haskey(ds, "lats") || throw(ArgumentError(
            "cubed-sphere region mask $(mask.file) has no 'lons'/'lats' cell centres " *
            "(Xdim, Ydim, nf); the panel layout of the mask cannot be checked against $(source)"))
        file_lon = read_raw(ds, "lons", FillValueOnly())
        file_lat = read_raw(ds, "lats", FillValueOnly())
        twin = float64_twin(mesh)
        for p in 1:CS_PANEL_COUNT
            lon, lat = panel_cell_center_lonlat(twin, p)
            _check_mesh_centres(view(file_lon, :, :, p), view(file_lat, :, :, p), lon, lat,
                                mask.file, "region mask cubed-sphere panel $(p)", 1e-7)
        end
        check_cell_area(ds, mesh; file = mask.file)
    end
    return nothing
end

check_region_mask_grid(mask::LatLonRegionMask, mesh::CubedSphereMesh;
                       source::AbstractString = "surface flux") = throw(ArgumentError(
    "region mask '$(mask.variable)' in $(mask.file) is on the lat-lon grid " *
    "$(_grid_label(mask.lon, mask.lat)), but $(source) is on the run's C$(mesh.Nc) cube " *
    "(cs_native); the mask must be a cubed-sphere mask $(CS_REGION_MASK_DIMS) on that cube"))
