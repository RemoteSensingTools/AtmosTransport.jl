# ---------------------------------------------------------------------------
# The `cs_native` surface-flux file: the runtime contract, defined once
#
# A `cs_native` file holds flux densities that are already on the runtime's
# cubed sphere, so the runtime (`surface_flux.kind = "cs_native"`) reads them
# without regridding:
#
#   - each series variable has NetCDF dimensions (time, nf, Ydim, Xdim), Julia
#     (Xdim, Ydim, nf, time) = `CS_NATIVE_FLUX_DIMS`, panels in the GEOS file
#     convention of the mesh;
#   - values are kg of a species (or of carbon) per m² per second, positive into
#     the atmosphere; the runtime multiplies them by its own mesh's cell areas;
#   - slices are start-stamped: slice k holds from its stamp until the next
#     (the runtime's `StepwiseFlux`), and `time_bnds` records each interval;
#   - no value is fill or non-finite.
#
# The writer produces such files (the surface-flux builder; the CATRINE protocol
# script); the reader is the runtime's loader and the builder's readback.
# ---------------------------------------------------------------------------

"""
    CS_NATIVE_FLUX_DIMS

Dimension names of a `cs_native` flux series in Julia order,
`("Xdim", "Ydim", "nf", "time")`; in NetCDF (CF) order `(time, nf, Ydim, Xdim)`.
`Xdim` and `Ydim` index the cells of a panel and `nf` the six panels, in the
panel convention of the mesh (`Output.define_cs_horizontal_geometry!`).
"""
const CS_NATIVE_FLUX_DIMS = ("Xdim", "Ydim", "nf", "time")

# Faces of the cube.

# Relative tolerance of today's norm-based cell-area gate (`check_cell_area`).
const _CELL_AREA_RTOL = 1e-4

# Sign of every flux the writers store: the runtime adds the flux to the tracer.
const _SIGN_CONVENTION = "positive into the atmosphere"

# Attributes the writers own on a flux variable: the units come from the
# species, the sign is the runtime's, and no value is fill or packed.
const _WRITER_VARIABLE_ATTRIBUTES = ("units", "sign_convention", "grid_mapping", "coordinates",
                                     "cell_measures", "_FillValue", "missing_value",
                                     "scale_factor", "add_offset", "valid_min", "valid_max",
                                     "valid_range")

# ---------------------------------------------------------------------------
# Shared pieces of the two flux writers
# ---------------------------------------------------------------------------

_string_dict(attributes) = Dict{String, Any}(String(k) => v for (k, v) in pairs(attributes))

# One flux variable of a writer: `name` or `name => attributes`.
_variable_spec(name::Union{AbstractString, Symbol}) = (String(name), Dict{String, Any}())
_variable_spec(spec::Pair) = (String(first(spec)), _string_dict(last(spec)))

function _variable_specs(variables)
    specs = [_variable_spec(v) for v in variables]
    isempty(specs) && throw(ArgumentError("a flux file needs at least one variable"))
    names = first.(specs)
    allunique(names) || throw(ArgumentError("flux variable names must be unique; got $(names)"))
    return specs
end

function _flux_variable_attributes(units::AbstractString, attributes::Dict{String, Any}, owned)
    for key in keys(attributes)
        key in _WRITER_VARIABLE_ATTRIBUTES && throw(ArgumentError(
            "the variable attribute $(repr(key)) is set by the writer (units from the species, " *
            "sign $(repr(_SIGN_CONVENTION)), no fill values or packing)"))
    end
    out = Dict{String, Any}("units" => units, "sign_convention" => _SIGN_CONVENTION)
    for (key, value) in owned
        out[key] = value
    end
    return merge!(out, attributes)
end

# Global attributes of a flux file: CF version, species, the provenance block
# (`Output.define_provenance_attributes!`), the build configuration and its
# SHA-1 when given, then the caller's attributes, which may not replace any of
# these (nor the geometry tags defined before).
function _define_flux_globals!(ds, species::Symbol, attributes::Dict{String, Any}, config_text)
    ds.attrib["Conventions"] = "CF-1.8"
    ds.attrib["species"] = String(species)
    if config_text !== nothing
        ds.attrib["config_text"] = String(config_text)
        ds.attrib["config_sha1"] = bytes2hex(sha1(String(config_text)))
    end
    define_provenance_attributes!(ds)
    for (key, value) in attributes
        haskey(ds.attrib, key) && throw(ArgumentError(
            "the global attribute $(repr(key)) is set by the writer"))
        ds.attrib[key] = value
    end
    return nothing
end

function _check_all_finite(data::AbstractArray, name::AbstractString)
    i = findfirst(x -> !isfinite(x), data)
    i === nothing || throw(ArgumentError(
        "flux variable $(repr(name)) has the non-finite value $(data[i]) at index " *
        "$(Tuple(CartesianIndices(data)[i])); flux files hold no fill or non-finite values"))
    return nothing
end

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

"""
    CSNativeFluxWriter

A `cs_native` flux file being written, returned by
[`open_cs_native_writer`](@ref). Slices go in with [`write_slices!`](@ref);
[`publish!`](@ref) completes the file and renames it onto its path, and
[`abandon!`](@ref) discards it. Until then only a staging file next to the
target exists (`AtomicFiles`), so readers of the path see the previous file
(or none), never a partial one.
"""
mutable struct CSNativeFluxWriter{M <: CubedSphereMesh, D}
    path      :: String
    staging   :: String
    ds        :: D
    mesh      :: M
    intervals :: Vector{TimeInterval}
    written   :: Dict{String, BitVector}
    isopen    :: Bool
end

Base.isopen(w::CSNativeFluxWriter) = w.isopen

function Base.show(io::IO, w::CSNativeFluxWriter)
    print(io, "CSNativeFluxWriter(", repr(w.path), ", C", w.mesh.Nc, ", ",
          length(w.intervals), " slices, ", join(sort!(collect(keys(w.written))), ", "),
          w.isopen ? "" : "; closed", ")")
end

"""
    open_cs_native_writer(path, mesh::CubedSphereMesh, intervals::Vector{TimeInterval},
                          variables; species::Symbol, chunk_time::Integer, deflate = 1,
                          attributes = Dict(), config_text = nothing) -> CSNativeFluxWriter
    open_cs_native_writer(f, path, mesh, intervals, variables; kwargs...) -> String

Open a `cs_native` surface-flux file for `mesh` with one slice per interval.
`variables` lists the flux series, each a name or `name => attributes` (for
example `"GPP_CO2_FLUX" => Dict("long_name" => "gross primary production")`).
The file is written to a staging file next to `path` and appears at `path`
only when [`publish!`](@ref) succeeds; a relative `path` is taken relative to
the working directory at opening. The `do`-block form calls `f(writer)`, then
`publish!`, and returns the absolute path; if `f` throws, the file is
abandoned.

The file (NetCDF-4):

- **Geometry** from `Output.define_cs_horizontal_geometry!(ds, mesh)`:
  dimensions `Xdim`, `Ydim`, `Xcorner`, `Ycorner`, `nf`; cell centres
  `lons`/`lats`, corners, `cell_area = mesh.cell_areas` (Float64, m²) and the
  `cubed_sphere` grid mapping with `semi_major_axis = mesh.radius`.
- **Time**: the `intervals` must be contiguous (`stop` of one is `start` of the
  next). `time` holds each interval's start (Float64,
  `"hours since <first day> 00:00:00"`, `calendar = "proleptic_gregorian"`),
  `time_bnds` `(nv, time)` its start and stop. Slices are start-stamped, so the
  runtime's stepwise scheme holds each slice for exactly its interval.
- **Series**: one Float32 variable per entry of `variables`, dimensions
  [`CS_NATIVE_FLUX_DIMS`](@ref), `units = canonical_units_string(species)`
  (`"kg CO2 m-2 s-1"` for `:co2`), `sign_convention = "positive into the
  atmosphere"`, `grid_mapping = "cubed_sphere"`, `coordinates = "lons lats"`,
  and no `_FillValue`. Chunks are `(Nc, Nc, 1, chunk_time)` (Julia order;
  `chunk_time` is clamped to the number of slices), deflate level `deflate`
  with shuffle: `chunk_time = 1` for daily files and `24` for hourly files
  (the P0 read benchmarks, design Section 2.4).
- **Global attributes**: `Conventions = "CF-1.8"`, `species`, the provenance
  block of `Output.define_provenance_attributes!` (its `creation_date` is the
  publication time), `config_text` and `config_sha1` (SHA-1 of the text) when
  `config_text` is given, then `attributes`. A caller attribute may not replace
  one the writer sets.

`species` must be a key of `SPECIES_MOLAR_MASS`.
"""
function open_cs_native_writer(path::AbstractString, mesh::CubedSphereMesh,
                               intervals::AbstractVector{TimeInterval}, variables;
                               species::Symbol, chunk_time::Integer, deflate::Integer = 1,
                               attributes = Dict{String, Any}(),
                               config_text::Union{Nothing, AbstractString} = nothing)
    _check_contiguous(intervals)
    chunk_time >= 1 || throw(ArgumentError("chunk_time must be at least 1; got $(chunk_time)"))
    0 <= deflate <= 9 || throw(ArgumentError("deflate must be in 0:9; got $(deflate)"))
    units = canonical_units_string(species)
    specs = _variable_specs(variables)
    global_attributes = _string_dict(attributes)
    target = abspath(String(path))      # a long build may change the working directory
    staging, io = open_atomic_staging(target)
    close(io)
    ds = nothing
    try
        ds = NCDataset(staging, "c")
        define_cs_horizontal_geometry!(ds, mesh)
        _define_interval_time_axis!(ds, intervals)
        _define_flux_globals!(ds, species, global_attributes, config_text)
        Nc = mesh.Nc
        chunks = [Nc, Nc, 1, min(Int(chunk_time), length(intervals))]
        owned = ("grid_mapping" => "cubed_sphere", "coordinates" => "lons lats")
        for (name, var_attributes) in specs
            haskey(ds, name) && throw(ArgumentError(
                "the variable name $(repr(name)) is taken by the cs_native geometry or time axis"))
            defVar(ds, name, Float32, CS_NATIVE_FLUX_DIMS;
                   chunksizes = chunks, deflatelevel = Int(deflate), shuffle = true,
                   attrib = _flux_variable_attributes(units, var_attributes, owned))
        end
    catch
        ds === nothing || close(ds)
        abandon_atomic_staging(staging)
        rethrow()
    end
    written = Dict{String, BitVector}(name => falses(length(intervals)) for (name, _) in specs)
    return CSNativeFluxWriter(target, staging, ds, mesh, collect(intervals), written, true)
end

function open_cs_native_writer(f::Function, path::AbstractString, mesh::CubedSphereMesh,
                               intervals::AbstractVector{TimeInterval}, variables; kwargs...)
    w = open_cs_native_writer(path, mesh, intervals, variables; kwargs...)
    try
        f(w)
    catch
        isopen(w) && abandon!(w)
        rethrow()
    end
    return publish!(w)
end

_check_open(w::CSNativeFluxWriter) = isopen(w) || throw(ArgumentError(
    "the cs_native writer for $(w.path) is closed (published or abandoned)"))

"""
    write_slices!(w::CSNativeFluxWriter, name, k::AbstractUnitRange,
                  density::AbstractArray{Float32, 4}) -> w

Write the densities `density`, size `(Nc, Nc, 6, length(k))`, into slices `k`
of the series `name` (kg of the species per m² per second, Julia order
`(Xdim, Ydim, nf, time)`). Every value must be finite (an `ArgumentError`
names the first that is not); the file stores no fill values. Slices may be
written in any order and rewritten; [`publish!`](@ref) requires every slice of
every series.
"""
function write_slices!(w::CSNativeFluxWriter, name::AbstractString, k::AbstractUnitRange{<:Integer},
                       density::AbstractArray{Float32, 4})
    _check_open(w)
    haskey(w.written, name) || throw(ArgumentError(
        "$(repr(name)) is not a series of $(w.path); the series are " *
        "$(join(sort!(collect(keys(w.written))), ", "))"))
    n = length(w.intervals)
    !isempty(k) && first(k) >= 1 && last(k) <= n || throw(ArgumentError(
        "slices $(k) are outside 1:$(n) of $(w.path)"))
    Nc = w.mesh.Nc
    size(density) == (Nc, Nc, CS_PANEL_COUNT, length(k)) || throw(DimensionMismatch(
        "densities for slices $(k) of $(repr(name)) have size $(size(density)); expected " *
        "$((Nc, Nc, CS_PANEL_COUNT, length(k))) = (Xdim, Ydim, nf, time)"))
    _check_all_finite(density, name)
    w.ds[name].var[:, :, :, k] = density
    w.written[name][k] .= true
    return w
end

"""
    publish!(w::CSNativeFluxWriter) -> String

Complete the file and publish it at its path by one atomic rename
(`AtomicFiles.commit_atomic_staging`); return the absolute path. Every slice of every
series must have been written; otherwise, and on any other error, the staging
file is removed (the writer is abandoned) and the error is thrown. The
provenance `creation_date` is set to the publication time.
"""
function publish!(w::CSNativeFluxWriter)
    _check_open(w)
    unwritten = [(name, findall(!, mask)) for (name, mask) in w.written if !all(mask)]
    if !isempty(unwritten)
        abandon!(w)
        name, slices = first(sort!(unwritten; by = first))
        throw(ArgumentError(
            "cs_native file $(w.path) is incomplete: series $(repr(name)) misses " *
            "$(length(slices)) slice(s), the first $(first(slices)); the file was abandoned"))
    end
    try
        define_provenance_attributes!(w.ds)
        close(w.ds)
    catch
        abandon!(w)
        rethrow()
    end
    w.isopen = false
    return commit_atomic_staging(w.staging, w.path)
end

"""
    abandon!(w::CSNativeFluxWriter) -> nothing

Close the writer and remove its staging file; the path keeps its previous
file, if any. Abandoning a closed writer does nothing.
"""
function abandon!(w::CSNativeFluxWriter)
    w.isopen || return nothing
    w.isopen = false
    try
        isopen(w.ds) && close(w.ds)
    finally
        abandon_atomic_staging(w.staging)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

# The units contract of a cs_native series: kg (not g, not molar) of the species
# or of carbon, per m², per second. The runtime has always rejected anything
# else, so that a file can never be read with a silently wrong factor.
_is_cs_native_units(u::FluxUnits{<:AbstractSubstance, PerSquareMetre, PerSecond}) =
    !u.molar && u.amount_per_unit == 1.0
_is_cs_native_units(::FluxUnits) = false

const _CS_NATIVE_UNITS_HINT = "expected kg (of the species) m-2 s-1 or kgC m-2 s-1; convert molar, " *
                              "gram, per-cell or missing units first"

function _cs_native_flux_units(text::AbstractString, path::AbstractString)
    u = try
        parse_flux_units(text)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("unsupported surface-flux units $(repr(text)) in $(path): " *
                            "$(err.msg); $(_CS_NATIVE_UNITS_HINT)"))
    end
    _is_cs_native_units(u) || throw(ArgumentError(
        "unsupported surface-flux units $(repr(text)) in $(path); $(_CS_NATIVE_UNITS_HINT)"))
    return u
end

_slice_range(::Colon, times::AbstractVector) = 1:length(times)
function _slice_range(r::AbstractUnitRange{<:Integer}, times::AbstractVector)
    isempty(r) && throw(ArgumentError("the slice range to read is empty"))
    first(r) >= 1 && last(r) <= length(times) || throw(ArgumentError(
        "slices $(r) are outside 1:$(length(times))"))
    return Int(first(r)):Int(last(r))
end
_slice_range(selector::Function, times::AbstractVector) = _slice_range(selector(times), times)

function _existing_flux_file(path::AbstractString)
    file = String(path)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $(file)"))
    return file
end

# The layout of a cs_native series: dimension names `CS_NATIVE_FLUX_DIMS`,
# Xdim = Ydim = mesh.Nc, six panels, at least one slice, floating-point values.
# Returns the CF variable.
function _cs_native_variable(ds, path::String, name::String, mesh::CubedSphereMesh)
    haskey(ds, name) || throw(ArgumentError("variable '$(name)' not found in $(path)"))
    v = ds[name]
    dims = Tuple(String.(dimnames(v)))
    dims == CS_NATIVE_FLUX_DIMS || throw(ArgumentError(
        "native cubed-sphere surface-flux variable '$(name)' in $(path) has dimensions " *
        "$(dims); expected $(CS_NATIVE_FLUX_DIMS) (Julia order; NetCDF (time, nf, Ydim, Xdim))"))
    Nc = mesh.Nc
    size(v, 1) == Nc && size(v, 2) == Nc || throw(DimensionMismatch(
        "native cubed-sphere flux has horizontal shape $(size(v, 1))x$(size(v, 2)); " *
        "transport grid is C$(Nc)"))
    size(v, 3) == CS_PANEL_COUNT || throw(DimensionMismatch(
        "native cubed-sphere flux has nf=$(size(v, 3)); expected $(CS_PANEL_COUNT)"))
    size(v, 4) > 0 || throw(ArgumentError("native cubed-sphere flux has no time slices"))
    T = nonmissingtype(eltype(v))
    T <: AbstractFloat || throw(ArgumentError(
        "native cubed-sphere flux '$(name)' in $(path) is stored as $(T); expected floating point"))
    return v
end

function _cs_native_flux_info(ds, path::String, name::String, mesh::CubedSphereMesh, slices)
    v = _cs_native_variable(ds, path, name, mesh)
    ntime = size(v, 4)
    times = decode_time(ds, find_coordinate(ds, TimeAxis(); length = ntime))
    all(k -> times[k] < times[k + 1], 1:(ntime - 1)) || throw(ArgumentError(
        "native cubed-sphere surface-flux times must be strictly increasing in $(path)"))
    units = _cs_native_flux_units(string(get(v.attrib, "units", "")), path)
    check_cell_area(ds, mesh; file = path)
    kr = _slice_range(slices, times)
    return (; slices = kr, times = times[kr], units)
end

_has_missing(data::AbstractArray{T}) where T = Missing <: T && any(ismissing, data)

# Slices `kr` of the CF variable `v`, panel by panel (the chunk layout), into
# `dest[p]`, each value converted once to the destination element type. One
# panel-slab read buffer (two when the variable declares fill values or packing)
# is the only allocation. Fill values, and values that are not finite once
# converted, are errors.
function _read_cs_native_panels!(dest::NTuple{6, AbstractArray{<:AbstractFloat, 3}}, v,
                                 kr::UnitRange{Int}, name::String, path::String)
    shape = (size(v, 1), size(v, 2), length(kr))
    for (p, d) in enumerate(dest)
        size(d) == shape || throw(DimensionMismatch(
            "destination panel $(p) for slices $(kr) of '$(name)' has size $(size(d)); " *
            "expected $(shape) = (Xdim, Ydim, slice)"))
    end
    buffer = Array{eltype(parent(v))}(undef, shape)
    data = eltype(v) === eltype(buffer) ? buffer : Array{eltype(v)}(undef, shape)
    for (p, d) in enumerate(dest)
        NCDatasets.load!(v, data, buffer, :, :, p, kr)
        _has_missing(data) && throw(ArgumentError(
            "native cubed-sphere flux has fill values on panel $(p) of $(path)"))
        d .= data
        i = findfirst(x -> !isfinite(x), d)
        i === nothing && continue
        x, y, t = Tuple(CartesianIndices(d)[i])
        throw(ArgumentError(
            "native cubed-sphere flux '$(name)' contains non-finite values in $(path) " *
            "(first at $((x, y, p, kr[t])), Julia order (Xdim, Ydim, nf, time) of the file, " *
            "read as $(eltype(d)))"))
    end
    return dest
end

"""
    cs_native_flux_info(path_or_ds, variable; mesh::CubedSphereMesh, slices = :)
        -> (; slices::UnitRange{Int}, times::Vector{DateTime}, units::FluxUnits)

The checks of [`read_cs_native_flux`](@ref) that need no flux values (layout,
time axis, units, cell-area gate) on the `cs_native` series `variable` of a
file path or an open NetCDF dataset, and the slices `slices` selects (`:`, a
range of slice indices, or a function of the decoded times returning one):
their indices, times and the parsed units. A caller allocates its arrays from
`length(slices)` and streams the values in with [`read_cs_native_flux!`](@ref).
"""
function cs_native_flux_info(path::AbstractString, variable::AbstractString;
                             mesh::CubedSphereMesh, slices = Colon())
    file = _existing_flux_file(path)
    return NCDataset(ds -> _cs_native_flux_info(ds, file, String(variable), mesh, slices), file)
end

cs_native_flux_info(ds, variable::AbstractString; mesh::CubedSphereMesh, slices = Colon()) =
    _cs_native_flux_info(ds, _dataset_path(ds), String(variable), mesh, slices)

"""
    read_cs_native_flux!(dest_panels, path_or_ds, variable, slices = :; mesh::CubedSphereMesh)
        -> (; slices::UnitRange{Int}, times::Vector{DateTime}, units::FluxUnits)

Read slices `slices` of the `cs_native` series `variable` (of a file path or an
open NetCDF dataset) panel by panel into the caller's arrays: `dest_panels[p]`,
size `(Nc, Nc, length(slices))`, receives panel `p`, each stored value
converted once to its element type (`FT(x)`, as `FT.(ds[variable][:, :, p,
slices])` does). Beyond the checks it allocates one panel-slab read buffer
(two when the variable declares fill values or packing), so a caller that fills
its final arrays holds the series once, never the whole cube twice. This is the
runtime's loader (`surface_flux.kind = "cs_native"`).

Every check of [`read_cs_native_flux`](@ref) applies ([`cs_native_flux_info`](@ref)
runs first); a value that is not finite once converted (a Float64 value beyond
the Float32 range read into Float32) is an error too. `slices` is any selection
`cs_native_flux_info` takes. Returns the slices read, their times and the
parsed units; the caller converts with [`mass_rate_factor`](@ref)`(units,
species)` and the mesh cell areas.
"""
function read_cs_native_flux!(dest_panels::NTuple{6, AbstractArray{<:AbstractFloat, 3}},
                              path::AbstractString, variable::AbstractString, slices = Colon();
                              mesh::CubedSphereMesh)
    file = _existing_flux_file(path)
    return NCDataset(ds -> _read_cs_native_flux!(dest_panels, ds, file, String(variable), mesh, slices),
                     file)
end

read_cs_native_flux!(dest_panels::NTuple{6, AbstractArray{<:AbstractFloat, 3}}, ds,
                     variable::AbstractString, slices = Colon(); mesh::CubedSphereMesh) =
    _read_cs_native_flux!(dest_panels, ds, _dataset_path(ds), String(variable), mesh, slices)

function _read_cs_native_flux!(dest_panels, ds, path::String, name::String, mesh::CubedSphereMesh,
                               slices)
    info = _cs_native_flux_info(ds, path, name, mesh, slices)
    _read_cs_native_panels!(dest_panels, ds[name], info.slices, name, path)
    return info
end

"""
    read_cs_native_flux(path, variable; mesh::CubedSphereMesh, slices = :)
        -> (; density, times::Vector{DateTime}, units::FluxUnits)

Read the `cs_native` flux series `variable` of the file `path` for the runtime
mesh `mesh` into one array, the whole-file form of [`read_cs_native_flux!`](@ref)
(the runtime streams panel by panel into its own arrays instead). The
surface-flux builder uses it for its readback.

Checks, each an error when it fails:

- the variable has dimensions [`CS_NATIVE_FLUX_DIMS`](@ref) with
  `Xdim = Ydim = mesh.Nc`, six panels and at least one slice;
- its time axis (found by [`find_coordinate`](@ref)) decodes with
  [`decode_time`](@ref) and the default decoders, and is strictly increasing;
- its `units` parse ([`parse_flux_units`](@ref)) as kg of a species or of
  carbon per m² per second (`kg CO2 m-2 s-1`, `kg m-2 s-1`, `kgC m-2 s-1`);
  gram, molar, per-cell, other time units and missing units are refused;
- a stored `cell_area` passes [`check_cell_area`](@ref);
- no value of the slices read is fill (`_FillValue` or `missing_value`) or
  non-finite.

`slices` selects the slices to read: `:` (all), a range of slice indices, or a
function of the decoded times (`Vector{DateTime}`) returning such a range, so a
caller can select by time without opening the file twice.

Returns `density` (Julia order `(Xdim, Ydim, nf, slice)`, in the stored
floating-point type: Float32 for files of [`open_cs_native_writer`](@ref)),
the `times` of the slices read, and the parsed `units`. The caller turns
densities into kg of its species per second per cell with
[`mass_rate_factor`](@ref)`(units, species)` and the mesh cell areas.
"""
function read_cs_native_flux(path::AbstractString, variable::AbstractString;
                             mesh::CubedSphereMesh, slices = Colon())
    file = _existing_flux_file(path)
    name = String(variable)
    return NCDataset(file) do ds
        info = _cs_native_flux_info(ds, file, name, mesh, slices)
        v = ds[name]
        density = Array{nonmissingtype(eltype(v))}(undef, mesh.Nc, mesh.Nc, CS_PANEL_COUNT,
                                                   length(info.slices))
        _read_cs_native_panels!(ntuple(p -> view(density, :, :, p, :), CS_PANEL_COUNT), v,
                                info.slices, name, file)
        (; density, times = info.times, units = info.units)
    end
end

"""
    check_cell_area(ds, mesh::CubedSphereMesh; file = path of ds) -> nothing

The cell-area gate of a `cs_native` file. The runtime multiplies the stored
densities by `mesh.cell_areas`; a file prepared on another cube geometry (an
equiangular instead of the GMAO cube, another radius) has other cell areas,
and its emitted mass would be silently wrong. When the dataset `ds` has a
`cell_area` variable, shape `(Nc, Nc)` or `(Nc, Nc, 6)`, each panel `a` must
satisfy `isapprox(a, mesh.cell_areas; rtol = 1e-4)`: the norm-based relative
difference ‖a − A‖ ≤ 10⁻⁴ max(‖a‖, ‖A‖) (Frobenius norm over the panel).
Fill values in `cell_area` fail the gate; a file without `cell_area` passes.

This is the gate the runtime has applied since the `cs_native` kind exists;
legacy files written with Float32-era geometry (per-cell area differences up
to 1.75e-4 on C90 polar panels) pass it.
"""
function check_cell_area(ds, mesh::CubedSphereMesh; file::AbstractString = _dataset_path(ds))
    # TODO(P11): make the gate element-wise as the design decides (Section 2.4 and
    # 6.1 item 2): max |a_file / a_mesh - 1| above 1e-3 is an error, above 1e-9 a
    # warning, and a stored `semi_major_axis` must equal mesh.radius (rtol 1e-12).
    # P2 keeps the runtime's norm-based gate unchanged.
    haskey(ds, "cell_area") || return nothing
    file_area = read_raw(ds, "cell_area", MaskedAsNaN())
    mesh_area = Float64.(mesh.cell_areas)
    Nc = mesh.Nc
    panels = if size(file_area) == (Nc, Nc)
        (file_area,)
    elseif size(file_area) == (Nc, Nc, CS_PANEL_COUNT)
        ntuple(p -> view(file_area, :, :, p), CS_PANEL_COUNT)
    else
        throw(DimensionMismatch("cell_area in $(file) has shape $(size(file_area)); " *
                                "expected ($(Nc), $(Nc)) or ($(Nc), $(Nc), $(CS_PANEL_COUNT))"))
    end
    for (p, a) in enumerate(panels)
        isapprox(a, mesh_area; rtol = _CELL_AREA_RTOL) || throw(ArgumentError(
            "cell_area in $(file) (panel $(p)) differs from the runtime C$(Nc) mesh areas; " *
            "the flux was prepared on another cube geometry"))
    end
    return nothing
end
