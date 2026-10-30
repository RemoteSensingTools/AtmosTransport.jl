# ---------------------------------------------------------------------------
# Static ground fields for observation placement
#
# Two external datasets, never carried in transport binaries:
#
# * the per-grid model surface height (m above sea level), one file per model
#   grid: the model ground, the reference of absolute-altitude placement and
#   a per-row diagnostic (never a stand-in for the true ground). Its schema is
#   defined here, with its one writer and reader;
#   `scripts/preprocessing/build_model_surface_height.jl` produces it from a
#   met source's surface geopotential;
# * a high-resolution lat-lon digital elevation model (DEM), the true ground
#   of records that give an altitude above sea level without their elevation,
#   sampled at points by bilinear interpolation.
#
# Both are read into the `Output` placement interfaces
# (`Observations.ModelSurfaceHeight`, `Observations.AbstractPointElevation`).
# ---------------------------------------------------------------------------

const _SURFACE_HEIGHT_VARIABLE = "surface_height"
const _METRE_UNITS = ("m", "meter", "meters", "metre", "metres")
# Stored cell centres must match the runtime mesh centres to this chord
# distance on the unit sphere (1e-7 ≈ 6e-6°), whatever the longitude
# convention, widened to the rounding of a Float32 lat-lon or reduced-Gaussian
# mesh (8 eps ≈ 5e-5°). Two different grids differ by far more. Cubed-sphere
# centres are compared with those of a Float64 twin of the mesh instead: a
# Float32 mesh computes them in Float32 (chord 2.7e-6 from Float64 at C90).
_centre_tolerance(mesh) = max(1e-7, 8 * eps(eltype(mesh)))

_is_metres(units) = units isa AbstractString && lowercase(strip(units)) in _METRE_UNITS

function _check_metre_units(var, name::AbstractString, path::AbstractString)
    units = get(var.attrib, "units", nothing)
    _is_metres(units) || throw(ArgumentError(
        "variable '$(name)' in $(path) has units $(repr(units)); expected metres " *
        "($(join(repr.(_METRE_UNITS), ", "))): heights must be in m above sea level"))
    return nothing
end

function _check_mesh_centres(file_lon, file_lat, mesh_lon, mesh_lat, path::AbstractString, what::AbstractString,
                        tolerance::Float64)
    size(file_lon) == size(mesh_lon) && size(file_lat) == size(mesh_lat) || throw(DimensionMismatch(
        "$(what) centres in $(path) have shape $(size(file_lon)); the mesh needs $(size(mesh_lon))"))
    worst = maximum(lonlat_chord_distance(file_lon[k], file_lat[k], mesh_lon[k], mesh_lat[k])
                    for k in eachindex(file_lon, file_lat, mesh_lon, mesh_lat); init = 0.0)
    worst <= tolerance || throw(ArgumentError(
        "$(what) centres in $(path) are up to $(round(rad2deg(worst); sigdigits = 3))° from the " *
        "runtime mesh centres: the file was written for another grid"))
    return nothing
end

function _check_finite_heights(values, path::AbstractString)
    n = count(!isfinite, values)
    n == 0 || throw(ArgumentError(
        "$(path): $(n) model surface heights are fill or non-finite; the model ground must be complete"))
    return nothing
end

# -- writer -------------------------------------------------------------------

_surface_height_attributes() = Dict{String, Any}(
    "units" => "m", "standard_name" => "surface_altitude",
    "long_name" => "model-grid surface height above sea level")

_surface_height_shape(mesh::LatLonMesh) = (mesh.Nx, mesh.Ny)
_surface_height_shape(mesh::ReducedGaussianMesh) = (ncells(mesh),)
_surface_height_shape(mesh::CubedSphereMesh) = (mesh.Nc, mesh.Nc, 6)

function _define_surface_height!(ds, mesh::LatLonMesh, field)
    _define_latlon_flux_axes!(ds, mesh)
    defVar(ds, _SURFACE_HEIGHT_VARIABLE, Float64, ("lon", "lat");
           attrib = merge(_surface_height_attributes(), Dict{String, Any}("cell_measures" => "area: cell_area")))[:, :] = field
    return nothing
end

function _define_surface_height!(ds, mesh::CubedSphereMesh, field)
    define_cs_horizontal_geometry!(ds, mesh)
    defVar(ds, _SURFACE_HEIGHT_VARIABLE, Float64, ("Xdim", "Ydim", "nf");
           attrib = merge(_surface_height_attributes(),
                          Dict{String, Any}("grid_mapping" => "cubed_sphere",
                                            "coordinates" => "lons lats")))[:, :, :] = field
    return nothing
end

function _define_surface_height!(ds, mesh::ReducedGaussianMesh, field)
    lon, lat = cell_centers(mesh; FT=Float64)
    defDim(ds, "cell", ncells(mesh))
    defVar(ds, "longitude", Float64, ("cell",);
           attrib = Dict("units" => "degrees_east", "standard_name" => "longitude"))[:] = lon
    defVar(ds, "latitude", Float64, ("cell",);
           attrib = Dict("units" => "degrees_north", "standard_name" => "latitude"))[:] = lat
    defVar(ds, "cell_area", Float64, ("cell",);
           attrib = Dict("units" => "m2", "standard_name" => "cell_area"))[:] =
        cell_areas(mesh)
    defVar(ds, _SURFACE_HEIGHT_VARIABLE, Float64, ("cell",); attrib = _surface_height_attributes())[:] = field
    return nothing
end

"""
    write_model_surface_height(path, mesh, field; attributes = Dict()) -> String

Write the per-grid model surface height dataset atomically
(`AtomicFiles.atomic_publish`) and return `path`. `field` holds the surface
height in m above sea level, finite everywhere, in the mesh's layout:
`(Nx, Ny)` on a `LatLonMesh`, `(ncells,)` on a `ReducedGaussianMesh`,
`(Nc, Nc, 6)` on a `CubedSphereMesh` (panels in the mesh's convention).

The file holds the Float64 variable `surface_height` (`units = "m"`,
`standard_name = "surface_altitude"`) and the mesh geometry: lat-lon `lon`,
`lat`, bounds and `cell_area`; cubed-sphere `Xdim`/`Ydim`/`nf`, `lons`/`lats`,
corners, `cell_area` and the `cubed_sphere` mapping
(`Output.define_cs_horizontal_geometry!`); reduced Gaussian `cell` with
`longitude`, `latitude` and `cell_area`. Global attributes: `Conventions`,
`title`, the provenance block, then `attributes` (for example the source file
and the regridding). [`read_model_surface_height`](@ref) reads it back.
"""
function write_model_surface_height(path::AbstractString, mesh, field::AbstractArray{<:Real};
                                    attributes = Dict{String, Any}())
    size(field) == _surface_height_shape(mesh) || throw(DimensionMismatch(
        "surface height has size $(size(field)); the $(summary(mesh)) needs $(_surface_height_shape(mesh))"))
    _check_all_finite(field, _SURFACE_HEIGHT_VARIABLE)
    values = Float64.(field)
    extra = _string_dict(attributes)
    return atomic_publish(String(path)) do tmp
        NCDataset(tmp, "c") do ds
            ds.attrib["Conventions"] = "CF-1.8"
            ds.attrib["title"] = "AtmosTransport model-grid surface height"
            define_provenance_attributes!(ds)
            for (k, v) in extra
                ds.attrib[k] = v
            end
            _define_surface_height!(ds, mesh, values)
        end
    end
end

# -- reader -------------------------------------------------------------------

function _surface_height_variable(ds, path::AbstractString)
    haskey(ds, _SURFACE_HEIGHT_VARIABLE) || throw(ArgumentError(
        "$(path) has no variable '$(_SURFACE_HEIGHT_VARIABLE)'; write it with write_model_surface_height " *
        "(scripts/preprocessing/build_model_surface_height.jl)"))
    var = ds[_SURFACE_HEIGHT_VARIABLE]
    _check_metre_units(var, _SURFACE_HEIGHT_VARIABLE, path)
    return var
end

function _check_dimnames(var, expected::Tuple, path::AbstractString)
    names = Tuple(String.(NCDatasets.dimnames(var)))
    names == expected || throw(DimensionMismatch(
        "'$(_SURFACE_HEIGHT_VARIABLE)' in $(path) has dimensions $(names); expected $(expected)"))
    return nothing
end

function _read_surface_height(ds, path::String, mesh::LatLonMesh)
    var = _surface_height_variable(ds, path)
    lon_name = find_coordinate(ds, LongitudeAxis(); length = mesh.Nx)
    lat_name = find_coordinate(ds, LatitudeAxis(); length = mesh.Ny)
    _check_dimnames(var, (lon_name, lat_name), path)
    file_lon = read_raw(ds, lon_name, FillValueOnly())
    file_lat = read_raw(ds, lat_name, FillValueOnly())
    mesh_lon = Float64.(mesh.λᶜ)
    mesh_lat = Float64.(mesh.φᶜ)
    _check_mesh_centres([file_lon[i] for i in 1:mesh.Nx, j in 1:mesh.Ny], [file_lat[j] for i in 1:mesh.Nx, j in 1:mesh.Ny],
                   [mesh_lon[i] for i in 1:mesh.Nx, j in 1:mesh.Ny], [mesh_lat[j] for i in 1:mesh.Nx, j in 1:mesh.Ny],
                   path, "lat-lon", _centre_tolerance(mesh))
    field = read_raw(ds, _SURFACE_HEIGHT_VARIABLE, MaskedAsNaN())
    _check_finite_heights(field, path)
    return Matrix{Float64}(field)
end

function _read_surface_height(ds, path::String, mesh::CubedSphereMesh)
    var = _surface_height_variable(ds, path)
    _check_dimnames(var, ("Xdim", "Ydim", "nf"), path)
    size(var) == (mesh.Nc, mesh.Nc, 6) || throw(DimensionMismatch(
        "'$(_SURFACE_HEIGHT_VARIABLE)' in $(path) has size $(size(var)); the C$(mesh.Nc) mesh needs " *
        "$((mesh.Nc, mesh.Nc, 6))"))
    haskey(ds, "lons") && haskey(ds, "lats") || throw(ArgumentError(
        "$(path) has no 'lons'/'lats' cell centres; the panel layout cannot be checked"))
    file_lon = read_raw(ds, "lons", FillValueOnly())
    file_lat = read_raw(ds, "lats", FillValueOnly())
    twin = float64_twin(mesh)
    for p in 1:6
        lon, lat = panel_cell_center_lonlat(twin, p)
        _check_mesh_centres(view(file_lon, :, :, p), view(file_lat, :, :, p), lon, lat,
                       path, "cubed-sphere panel $(p)", 1e-7)
    end
    check_cell_area(ds, mesh; file = path)
    field = read_raw(ds, _SURFACE_HEIGHT_VARIABLE, MaskedAsNaN())
    _check_finite_heights(field, path)
    return ntuple(p -> Matrix{Float64}(field[:, :, p]), 6)
end

function _read_surface_height(ds, path::String, mesh::ReducedGaussianMesh)
    var = _surface_height_variable(ds, path)
    _check_dimnames(var, ("cell",), path)
    length(var) == ncells(mesh) || throw(DimensionMismatch(
        "'$(_SURFACE_HEIGHT_VARIABLE)' in $(path) has $(length(var)) cells; the mesh has $(ncells(mesh))"))
    mesh_lon, mesh_lat = cell_centers(mesh; FT=Float64)
    _check_mesh_centres(read_raw(ds, "longitude", FillValueOnly()), read_raw(ds, "latitude", FillValueOnly()),
                   mesh_lon, mesh_lat, path, "reduced-Gaussian", _centre_tolerance(mesh))
    field = read_raw(ds, _SURFACE_HEIGHT_VARIABLE, MaskedAsNaN())
    _check_finite_heights(field, path)
    return Vector{Float64}(field)
end

"""
    read_model_surface_height(path, mesh) -> Observations.ModelSurfaceHeight

Read the per-grid model surface height dataset written by
[`write_model_surface_height`](@ref) for `mesh` (`LatLonMesh`,
`ReducedGaussianMesh` or `CubedSphereMesh`). The variable `surface_height`
must be in metres; its stored cell centres must be the mesh's (unit-sphere
chord within 1e-7, about 6e-6°, of the Float64 centres on the cubed sphere;
within 8 eps of a Float32 lat-lon or reduced-Gaussian mesh), a stored
cubed-sphere `cell_area` must pass
[`check_cell_area`](@ref), and every value must be finite. Returns the field
in the mesh's native layout (a matrix, a vector of cells, or six panels) as
`Observations.ModelSurfaceHeight`, which the observation sampler queries per cell.
"""
function read_model_surface_height(path::AbstractString, mesh)
    file = String(path)
    isfile(file) || throw(ArgumentError("model surface height file not found: $(file)"))
    field = NCDataset(ds -> _read_surface_height(ds, file, mesh), file, "r")
    return ModelSurfaceHeight(field, file)
end

# -- point DEM ----------------------------------------------------------------

"""
    LatLonPointElevation

A latitude-longitude digital elevation model read by
[`read_point_elevation`](@ref): ascending longitudes `lon` (any
convention, spanning less than 360°) and latitudes `lat`, the ground
elevation `values[i, j]` (m above sea level, Float32, `NaN` where the file has
fill), whether the longitudes wrap around the globe (`periodic`), and the
file `path`. `Observations.point_elevation(dem, lon, lat)` interpolates it
bilinearly between the four surrounding centres.
"""
struct LatLonPointElevation <: AbstractPointElevation
    lon::Vector{Float64}
    lat::Vector{Float64}
    values::Matrix{Float32}
    periodic::Bool
    path::String
end

Observations.terrain_source(dem::LatLonPointElevation) = dem.path

_strictly_increasing(x) = all(k -> x[k + 1] > x[k], 1:(length(x) - 1))

"""
    read_point_elevation(path; variable = "z") -> LatLonPointElevation

Read a lat-lon DEM for point lookups. The file holds 1-D longitude and
latitude centre coordinates (found by name, [`find_coordinate`](@ref)) and a
2-D variable `variable` on those two coordinate dimensions, with the ground
elevation in metres above sea level; `units` must say metres. The axes are
identified by the variable's dimension names: `(lon, lat)` in Julia order
(CF/NetCDF order `(lat, lon)`, the usual layout) or the transpose, which is
read transposed back; any other dimensions are an error. Fill values
(`_FillValue`, `missing_value`) become `NaN`. Latitudes may be stored in
either order (they are flipped to ascending with [`orient`](@ref));
longitudes must be strictly ascending in any convention (−180…180, 0…360, or
a regional window). A gridline-registered global DEM, whose first and last
longitudes are the same meridian 360° apart, has its last column dropped.
The DEM must give the land or water surface: over the ocean the ground is
the sea surface (0 m), so a bathymetric product (ETOPO) must be clipped at 0
over the sea first; a value below −450 m (deeper than the Dead Sea shore) is
rejected as bathymetry or an undeclared fill value, and a value above
9000 m (higher than Mount Everest) as an undeclared fill value.

The whole variable is read into memory as Float32: about 0.9 GB for a global
1-arc-minute grid (21600 × 10800).
"""
function read_point_elevation(path::AbstractString; variable::AbstractString = "z")
    file = String(path)
    isfile(file) || throw(ArgumentError("DEM file not found: $(file)"))
    return NCDataset(file, "r") do ds
        name = String(variable)
        haskey(ds, name) || throw(ArgumentError("DEM $(file) has no variable '$(name)' (set dem_variable)"))
        var = ds[name]
        ndims(var) == 2 || throw(ArgumentError(
            "DEM variable '$(name)' in $(file) has $(ndims(var)) dimensions; expected 2 (lon, lat)"))
        _check_metre_units(var, name, file)
        lon_name = find_coordinate(ds, LongitudeAxis())
        lat_name = find_coordinate(ds, LatitudeAxis())
        # The axes are told apart by the variable's dimension names, never by
        # their lengths: a square DEM would otherwise be read transposed.
        transposed = _dem_axis_order(var, lon_name, lat_name, name, file)
        lon = read_raw(ds, lon_name, FillValueOnly())
        lat = read_raw(ds, lat_name, FillValueOnly())
        all(isfinite, lon) && all(isfinite, lat) || throw(ArgumentError("DEM $(file) has fill coordinates"))
        o = LatLonOrientation(length(lat) > 1 && lat[1] > lat[end], 0, false)
        lon_o, lat_o = orient_coordinates(lon, lat, o)
        length(lon_o) >= 2 && length(lat_o) >= 2 && _strictly_increasing(lon_o) && _strictly_increasing(lat_o) ||
            throw(ArgumentError("DEM $(file) needs at least two strictly monotone longitudes and latitudes"))
        raw = read_raw(Float32, ds, name, MaskedAsNaN())
        values = orient(transposed ? permutedims(raw) : raw, o)
        lon_o, values = _drop_seam_column(lon_o, values, file)
        span = lon_o[end] - lon_o[1]
        spacing = span / (length(lon_o) - 1)
        span < 360 || throw(ArgumentError("DEM $(file) longitudes span $(span)°; expected less than 360°"))
        periodic = isapprox(span + spacing, 360.0; atol = 1e-3 * spacing)
        _check_land_surface(values, name, file)
        LatLonPointElevation(lon_o, lat_o, values, periodic, file)
    end
end

# Whether the DEM variable is stored `(lat, lon)` in Julia order (CF/NetCDF
# `(lon, lat)`); `(lon, lat)` is the usual layout. Any other dimension names
# are an error.
function _dem_axis_order(var, lon_name::String, lat_name::String, name::String, path::String)
    dims = Tuple(String.(NCDatasets.dimnames(var)))
    dims == (lon_name, lat_name) && return false
    dims == (lat_name, lon_name) && return true
    throw(DimensionMismatch(
        "DEM variable '$(name)' in $(path) has dimensions $(dims); expected the coordinate dimensions " *
        "($(repr(lon_name)), $(repr(lat_name))) in either order"))
end

# A gridline-registered global DEM (ETOPO, GMTED "g" grids) holds the
# antimeridian twice, as its first and last longitude 360° apart; the last
# column repeats the first and is dropped, leaving a periodic grid.
function _drop_seam_column(lon::Vector{Float64}, values::AbstractMatrix, path::String)
    n = length(lon)
    n >= 3 || return lon, values
    spacing = (lon[n] - lon[1]) / (n - 1)
    isapprox(lon[n] - lon[1], 360.0; atol = 1e-3 * spacing) || return lon, values
    @debug "DEM $(path): longitudes $(lon[1])…$(lon[n]) repeat the antimeridian; the last column is dropped"
    return lon[1:(n - 1)], values[1:(n - 1), :]
end

# The lowest land surface is the Dead Sea shore, about −430 m; a DEM going
# deeper holds bathymetry (ETOPO, GEBCO) or an undeclared fill value, either of
# which would put ocean and coastal records kilometres above the ground. The
# highest is Mount Everest, 8849 m; a DEM going higher holds an undeclared fill
# value (32767, 9999), which would put records kilometres below the ground.
const _LOWEST_LAND_SURFACE_M = Observations.LOWEST_GROUND_ELEVATION_M
const _HIGHEST_LAND_SURFACE_M = 9000.0

function _check_land_surface(values, name::AbstractString, path::AbstractString)
    lowest = minimum(v -> isfinite(v) ? Float64(v) : Inf, values; init = Inf)
    lowest >= _LOWEST_LAND_SURFACE_M || throw(ArgumentError(
        "DEM variable '$(name)' in $(path) goes down to $(lowest) m, below the lowest land surface " *
        "($(_LOWEST_LAND_SURFACE_M) m): it holds bathymetry or an undeclared fill value. The DEM must give " *
        "the ground or sea surface; clip a bathymetric product at 0 m over the sea"))
    highest = maximum(v -> isfinite(v) ? Float64(v) : -Inf, values; init = -Inf)
    highest <= _HIGHEST_LAND_SURFACE_M || throw(ArgumentError(
        "DEM variable '$(name)' in $(path) goes up to $(highest) m, above the highest land surface " *
        "($(_HIGHEST_LAND_SURFACE_M) m): it holds an undeclared fill value. Declare it as _FillValue " *
        "or missing_value"))
    return nothing
end

"""
    point_elevation(dem::LatLonPointElevation, lon, lat) -> Float64

Bilinear interpolation of the DEM between the four centres around the point,
with `NaN` corners left out and the weights of the others renormalised; `NaN`
when all four are `NaN` or the point is more than half a spacing outside the
DEM.
"""
function Observations.point_elevation(dem::LatLonPointElevation, lon::Real, lat::Real)
    bx = longitude_bilinear_bracket(dem.lon, Float64(lon), dem.periodic)
    by = half_cell_bilinear_bracket(dem.lat, Float64(lat))
    (bx === nothing || by === nothing) && return NaN
    i0, i1, wx = bx
    j0, j1, wy = by
    total = 0.0
    weight = 0.0
    for (i, a) in ((i0, 1 - wx), (i1, wx)), (j, b) in ((j0, 1 - wy), (j1, wy))
        v = dem.values[i, j]
        w = a * b
        if isfinite(v) && w > 0
            total += w * Float64(v)
            weight += w
        end
    end
    weight > 0 && return total / weight
    # Exactly on a centre whose value is fill, or every corner fill.
    return NaN
end
