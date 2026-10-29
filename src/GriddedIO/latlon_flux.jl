# ---------------------------------------------------------------------------
# Latitude-longitude surface-flux files for `surface_flux.kind = "file"`
#
# The flux products that stay on a latitude-longitude grid (ECCO-Darwin ocean
# CO2) are CF files the runtime's generic loader reads: 1-D `lon` and `lat`
# centre coordinates (found by name), a static Float32 density variable
# `(lon, lat)` in kg of the species per m² per second, positive into the
# atmosphere, without fill values. The runtime regrids such a file
# conservatively onto its own grid. It reads one static field per tracer
# (`kind = "file"` is not a time-varying kind), so the writer writes no time
# axis.
# ---------------------------------------------------------------------------

# Coordinates and geometry of a lat-lon flux file: `lon`/`lat` centres with
# their `lon_bnds`/`lat_bnds` faces (for CF tools; readers take faces from the
# centres, never from bounds) and the mesh's own `cell_area`.
function _define_latlon_flux_axes!(ds, mesh::LatLonMesh)
    Nx, Ny = mesh.Nx, mesh.Ny
    defDim(ds, "lon", Nx)
    defDim(ds, "lat", Ny)
    haskey(ds.dim, "nv") || defDim(ds, "nv", 2)
    defVar(ds, "lon", Float64, ("lon",);
           attrib = Dict("units" => "degrees_east", "standard_name" => "longitude",
                         "long_name" => "cell-centre longitude", "axis" => "X",
                         "bounds" => "lon_bnds"))[:] = Float64.(mesh.λᶜ)
    defVar(ds, "lat", Float64, ("lat",);
           attrib = Dict("units" => "degrees_north", "standard_name" => "latitude",
                         "long_name" => "cell-centre latitude", "axis" => "Y",
                         "bounds" => "lat_bnds"))[:] = Float64.(mesh.φᶜ)
    λᶠ = Float64.(mesh.λᶠ)
    φᶠ = Float64.(mesh.φᶠ)
    defVar(ds, "lon_bnds", Float64, ("nv", "lon");
           attrib = Dict("units" => "degrees_east"))[:, :] = permutedims(hcat(λᶠ[1:Nx], λᶠ[2:(Nx + 1)]))
    defVar(ds, "lat_bnds", Float64, ("nv", "lat");
           attrib = Dict("units" => "degrees_north"))[:, :] = permutedims(hcat(φᶠ[1:Ny], φᶠ[2:(Ny + 1)]))
    area = reshape(cell_areas(mesh), Nx, Ny)
    defVar(ds, "cell_area", Float64, ("lon", "lat");
           attrib = Dict("units" => "m2", "standard_name" => "cell_area",
                         "long_name" => "horizontal cell area"))[:, :] = area
    return nothing
end

# The runtime reads the written centres back (Float64), orients them
# (`latlon_orientation`, `orient_coordinates`: latitude ascending, longitudes
# from the first value ≥ 0, modulo 360°) and derives snapped faces from those
# coordinates. A mesh is writable only if that reproduces its faces bit for
# bit: each oriented cell keeps the faces of its mesh cell, shifted by the same
# multiple of 360° as its centre. Otherwise the runtime would regrid the file
# on other cells and change the emitted mass.

# The mesh's longitude faces in the runtime's cell order, or `nothing` when the
# oriented cells are not contiguous. Oriented cell i is mesh cell `order[i]`,
# its centre moved by a multiple of 360° (`orient_coordinates`); its faces move
# by the same multiple.
function _oriented_longitude_faces(mesh::LatLonMesh, o::LatLonOrientation, λ_oriented)
    order = _longitude_order(o, mesh.Nx)
    λᶜ, λᶠ = Float64.(mesh.λᶜ), Float64.(mesh.λᶠ)
    shift = [360 * round(Int, (λ_oriented[i] - λᶜ[order[i]]) / 360) for i in 1:mesh.Nx]
    west = [λᶠ[order[i]] + shift[i] for i in 1:mesh.Nx]
    east = [λᶠ[order[i] + 1] + shift[i] for i in 1:mesh.Nx]
    all(i -> east[i] === west[i + 1], 1:(mesh.Nx - 1)) || return nothing
    return push!(west, east[end])
end

function _first_face_difference(name, mine::Vector{Float64}, runtime::Vector{Float64})
    k = findfirst(i -> mine[i] !== runtime[i], eachindex(mine, runtime))
    return k === nothing ? nothing :
           "$(name) face $(k) is $(mine[k])° in the mesh and $(runtime[k])° in the runtime"
end

# Why the runtime would not rebuild the faces of `mesh` from its written
# centres, or `nothing` when it rebuilds them bit for bit.
function _runtime_face_mismatch(mesh::LatLonMesh)
    min(mesh.Nx, mesh.Ny) >= 2 || return "the runtime needs at least two centres per axis"
    λᶜ, φᶜ = Float64.(mesh.λᶜ), Float64.(mesh.φᶜ)      # what the file stores
    o = latlon_orientation(λᶜ, φᶜ)                     # a LatLonMesh stores φ ascending
    λ, φ = orient_coordinates(λᶜ, φᶜ, o)
    runtime = source_latlon_mesh(λ, φ; radius = mesh.radius)
    mine_λᶠ = _oriented_longitude_faces(mesh, o, λ)
    mine_λᶠ === nothing &&
        return "its longitude cells are not contiguous once oriented (a regional axis across 0°)"
    return something(_first_face_difference("longitude", mine_λᶠ, Float64.(runtime.λᶠ)),
                     _first_face_difference("latitude", Float64.(mesh.φᶠ), Float64.(runtime.φᶠ)),
                     Some(nothing))
end

function _check_runtime_reconstructs_faces(mesh::LatLonMesh)
    why = _runtime_face_mismatch(mesh)
    why === nothing || throw(ArgumentError(
        "write_latlon_flux: the runtime loader (surface_flux.kind = \"file\") rebuilds the " *
        "cell faces from the written centres (oriented, then GriddedIO.SnappedFaces) and " *
        "would regrid this $(mesh.Nx)×$(mesh.Ny) mesh on other cells: $(why). Only meshes " *
        "whose faces it reconstructs exactly can be written."))
    return nothing
end

# One flux variable of the lat-lon writer: `name => data` or
# `name => (data, attributes)` (dispatch on the value, so a vector mixing both
# forms, whose element type is `Pair{String, Any}`, works).
_latlon_variable(spec::Pair) = _latlon_variable(String(first(spec)), last(spec))
_latlon_variable(name::String, data::AbstractArray) = (name, data, Dict{String, Any}())
_latlon_variable(name::String, spec::Tuple{AbstractArray, Any}) = (name, spec[1], _string_dict(spec[2]))
_latlon_variable(name::String, spec) = throw(ArgumentError(
    "flux variable $(repr(name)) must be given as `name => data` or `name => (data, attributes)`"))

"""
    write_latlon_flux(path, mesh::LatLonMesh, variables; species::Symbol,
                      attributes = Dict(), deflate = 1, config_text = nothing) -> String

Write a static latitude-longitude surface-flux file for
`surface_flux.kind = "file"` atomically (`AtomicFiles.atomic_publish`) and
return `path`.

- **Static only**: each variable is one field `(lon, lat)`, with no time
  dimension. The runtime's `kind = "file"` loader reads one static field per
  tracer and is not a time-varying kind, so a series would never be read as
  one; time-varying fluxes on the cubed sphere go to `cs_native` files
  ([`open_cs_native_writer`](@ref)).
- `variables` lists the fields as `name => data` or `name => (data,
  attributes)`; `data` is Float32 kg of `species` per m² per second, positive
  into the atmosphere, size `(Nx, Ny)`, Julia order `(lon, lat)`. Every value
  must be finite; the file stores no `_FillValue`.
- **The mesh must be one the runtime rebuilds exactly.** The runtime takes the
  cell faces from the written centres (oriented, then [`SnappedFaces`](@ref)),
  never from `lon_bnds`/`lat_bnds`. A mesh whose faces that rule does not
  reproduce bit for bit is refused with an `ArgumentError`: its file would be
  regridded on other cells.
- Coordinates: `lon`, `lat` (Float64 cell centres `mesh.λᶜ`, `mesh.φᶜ`) with
  `lon_bnds`, `lat_bnds` (the mesh faces) and `cell_area` `(lon, lat)` from
  `cell_area(mesh, i, j)` [m²].
- Each field: `units = canonical_units_string(species)`,
  `sign_convention = "positive into the atmosphere"`,
  `cell_measures = "area: cell_area"`; chunks `(Nx, Ny)`, deflate level
  `deflate` with shuffle.
- Global attributes as for the `cs_native` writer: `Conventions`, `species`,
  provenance, `config_text`/`config_sha1` when given, then `attributes`.
"""
function write_latlon_flux(path::AbstractString, mesh::LatLonMesh, variables;
                           species::Symbol, attributes = Dict{String, Any}(), deflate::Integer = 1,
                           config_text::Union{Nothing, AbstractString} = nothing)
    0 <= deflate <= 9 || throw(ArgumentError("deflate must be in 0:9; got $(deflate)"))
    units = canonical_units_string(species)
    specs = [_latlon_variable(v) for v in variables]
    isempty(specs) && throw(ArgumentError("a flux file needs at least one variable"))
    names = first.(specs)
    allunique(names) || throw(ArgumentError("flux variable names must be unique; got $(names)"))
    shape = (mesh.Nx, mesh.Ny)
    for (name, data, _) in specs
        name in ("lon", "lat", "lon_bnds", "lat_bnds", "cell_area") &&
            throw(ArgumentError("the variable name $(repr(name)) is taken by the lat-lon geometry"))
        eltype(data) === Float32 || throw(ArgumentError(
            "flux variable $(repr(name)) must be Float32 (round once, before writing); got $(eltype(data))"))
        size(data) == shape || throw(DimensionMismatch(
            "flux variable $(repr(name)) has size $(size(data)); expected $(shape) = (lon, lat): " *
            "lat-lon flux files are static (kind = \"file\" reads one field), with no time dimension"))
        _check_all_finite(data, name)
    end
    _check_runtime_reconstructs_faces(mesh)
    global_attributes = _string_dict(attributes)
    owned = ("cell_measures" => "area: cell_area",)
    return atomic_publish(String(path)) do tmp
        NCDataset(tmp, "c") do ds
            _define_latlon_flux_axes!(ds, mesh)
            _define_flux_globals!(ds, species, global_attributes, config_text)
            for (name, data, var_attributes) in specs
                v = defVar(ds, name, Float32, ("lon", "lat");
                           chunksizes = [mesh.Nx, mesh.Ny], deflatelevel = Int(deflate), shuffle = true,
                           attrib = _flux_variable_attributes(units, var_attributes, owned))
                v.var[:, :] = data
            end
        end
    end
end

"""
    check_latlon_cell_area(area, mesh_area, units; file, rtol = 1e-2, flux = nothing) -> area

The cell-area gate of a lat-lon flux file whose `cell_area` or `area` variable
sets the inventory's total (`area` [m²], oriented like the flux). Its `units`
must be square metres (`"m2"`, `"m^2"`, `"m**2"`; an empty string, for an
absent attribute, is accepted), and it must agree with `mesh_area`, the cell
areas of the mesh that the file's coordinates define on the runtime sphere, as
`isapprox(area, mesh_area; rtol)`: the norm-based relative difference
‖a − A‖ ≤ `rtol` max(‖a‖, ‖A‖) (Frobenius norm). The default 1 % passes any
Earth-radius convention (6.371 versus 6.378137 × 10⁶ m is 0.22 % in area) and
the files' own cell definitions, while areas in km² or a land fraction, which
would scale the emitted total, fail. A size mismatch is a `DimensionMismatch`.

With `flux` (the flux slice, oriented like `area`), every cell whose flux is
nonzero must have a finite, positive area: the norm test alone passes a few
masked (zero) rows, such as polar rows, and their emission would be dropped
without a word (per-cell units give a cell without area no flux; per-m² units
give it no rate).
"""
function check_latlon_cell_area(area::AbstractMatrix, mesh_area::AbstractMatrix,
                                units::AbstractString; file::AbstractString,
                                rtol::Real = 1e-2, flux::Union{Nothing, AbstractMatrix} = nothing)
    u = lowercase(replace(strip(units), " " => ""))
    isempty(u) || u in ("m2", "m^2", "m**2") || throw(ArgumentError(
        "cell areas in $(file) have units $(repr(units)); expected m2"))
    size(area) == size(mesh_area) || throw(DimensionMismatch(
        "cell areas in $(file) have size $(size(area)); the flux grid has $(size(mesh_area))"))
    if flux !== nothing
        size(flux) == size(area) || throw(DimensionMismatch(
            "cell areas in $(file) have size $(size(area)); the flux has $(size(flux))"))
        bad = findall(i -> flux[i] != 0 && !(isfinite(area[i]) && area[i] > 0), eachindex(area))
        isempty(bad) || throw(ArgumentError(
            "cell areas in $(file) are zero, negative or not finite in $(length(bad)) cell(s) " *
            "with a nonzero flux (first: cell $(Tuple(CartesianIndices(area)[first(bad)])), " *
            "area $(area[first(bad)])); their emission would be dropped. Expected square " *
            "metres of every cell"))
    end
    isapprox(area, mesh_area; rtol) || throw(ArgumentError(
        "cell areas in $(file) differ from the areas of the grid its coordinates define by " *
        "$(round(sqrt(sum(abs2, area .- mesh_area) / sum(abs2, mesh_area)); sigdigits = 3)) " *
        "(relative norm); " *
        "expected square metres of the same cells"))
    return area
end
