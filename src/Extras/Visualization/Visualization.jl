"""
    Visualization

Topology-aware snapshot visualization support.

This module owns the lightweight data layer only: snapshot discovery, field
views, unit scaling, and topology-aware rasterization for quick maps. Makie
plotting lives in `AtmosTransportMakieExt` so the model/runtime does not pay a
plotting load cost unless a Makie backend is explicitly loaded.
"""
module Visualization

using ..ConfigChecks: parse_choice

using ..Grids: parse_panel_convention, parse_coordinate_law, parse_center_law

using Dates
using NCDatasets
using Printf
using Statistics: quantile

using ..Grids: LatLonMesh, CubedSphereMesh,
               AbstractCubedSpherePanelConvention,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               AbstractCubedSphereDefinition, CubedSphereDefinition, default_cs_definition,
               EquiangularGnomonic, GMAOEqualDistanceGnomonic,
               AngularMidpointCenter, FourCornerNormalizedCenter
using ..Regridding: build_regridder, apply_regridder!
using ..Parameters: IFS_EARTH_RADIUS
using ..GriddedIO: decode_time, CFDecoder

abstract type AbstractSnapshotTopology end

abstract type AbstractFieldTransform end

struct ColumnMean <: AbstractFieldTransform end
struct ColumnSum <: AbstractFieldTransform end
struct LevelSlice <: AbstractFieldTransform end
struct SurfaceSlice <: AbstractFieldTransform end

abstract type AbstractDisplayUnit end

struct NativeUnit <: AbstractDisplayUnit end
struct PPMUnit <: AbstractDisplayUnit end

"Select every available snapshot frame."
struct AllFrames end

field_transform_label(::ColumnMean) = :column_mean
field_transform_label(::ColumnSum) = :column_sum
field_transform_label(::LevelSlice) = :level_slice
field_transform_label(::SurfaceSlice) = :surface_slice

display_unit_label(::NativeUnit) = :native
display_unit_label(::PPMUnit) = :ppm

_parse_field_transform(transform::AbstractFieldTransform) = transform
_parse_field_transform(transform) =
    parse_choice(transform, (column_mean = ColumnMean(), column_sum = ColumnSum(),
                            level_slice = LevelSlice(), surface_slice = SurfaceSlice()),
                 "visualization transform"; case_sensitive = true, allow_symbol = true)

_parse_display_unit(unit::AbstractDisplayUnit) = unit
_parse_display_unit(unit) =
    parse_choice(unit, (native = NativeUnit(), ppm = PPMUnit()),
                 "visualization unit"; case_sensitive = true, allow_symbol = true)

_require_field_transform(transform::AbstractFieldTransform) = transform
function _require_field_transform(transform)
    throw(ArgumentError(
        "fieldview transform must be an AbstractFieldTransform such as ColumnMean(); " *
        "got $(repr(transform)). PlotSpec parses canonical Symbol/String selectors."))
end

_require_display_unit(unit::AbstractDisplayUnit) = unit
function _require_display_unit(unit)
    throw(ArgumentError(
        "fieldview unit must be an AbstractDisplayUnit such as NativeUnit() or PPMUnit(); " *
        "got $(repr(unit)). PlotSpec parses canonical Symbol/String selectors."))
end

"""
    LatLonSnapshotTopology(lons, lats; grid_type=:latlon)

Regular lon-lat snapshot geometry. Values are stored as `(lon, lat)`.
"""
struct LatLonSnapshotTopology <: AbstractSnapshotTopology
    lons::Vector{Float64}
    lats::Vector{Float64}
    grid_type::Symbol
end

"""
    ReducedGaussianSnapshotTopology(lons, lats, nrings, regridding)

Reduced-Gaussian snapshot geometry after the current writer's ring-aware
nearest-neighbor projection onto a regular lon-lat diagnostic grid.
"""
struct ReducedGaussianSnapshotTopology <: AbstractSnapshotTopology
    lons::Vector{Float64}
    lats::Vector{Float64}
    nrings::Int
    regridding::String
end

"""
    CubedSphereSnapshotTopology(Nc, nlevel, panel_convention[, definition])

Cubed-sphere snapshot geometry. Values are stored as `(Xdim, Ydim, nf)` for a
single horizontal field, with `nf == 6`. `definition` is the mesh the snapshot
was written on (coordinate and center laws, panel convention, longitude
offset); it defaults to the panel convention's standard definition.
"""
struct CubedSphereSnapshotTopology{P <: AbstractCubedSpherePanelConvention,
                                   D <: AbstractCubedSphereDefinition} <: AbstractSnapshotTopology
    Nc::Int
    nlevel::Int
    panel_convention::P
    definition::D
end

CubedSphereSnapshotTopology(Nc, nlevel,
                            panel_convention::AbstractCubedSpherePanelConvention) =
    CubedSphereSnapshotTopology(Nc, nlevel, panel_convention,
                                default_cs_definition(panel_convention))

"""
    SnapshotDataset

Metadata for an AtmosTransport NetCDF snapshot file.

The object intentionally does not hold an open NetCDF handle. Field reads open
the file briefly and close it immediately, which is robust for scripts,
animations, and long debugging sessions.
"""
struct SnapshotDataset{T <: AbstractSnapshotTopology, TT}
    path::String
    topology::T
    times::Vector{TT}
    hours::Vector{Float64}
    variables::Vector{Symbol}
    attributes::Dict{String, Any}
end

"""
    HorizontalField

A topology-native horizontal field view from one snapshot time.

For LL/RG snapshots `values` is a `(lon, lat)` matrix. For CS snapshots it is a
`(Xdim, Ydim, nf)` array until rasterized or plotted natively.
"""
struct HorizontalField{T <: AbstractSnapshotTopology, A, F <: AbstractFieldTransform, TT}
    topology::T
    values::A
    name::Symbol
    units::String
    time::TT
    time_index::Int
    transform::F
    level::Union{Nothing, Int}
    source_path::String
end

"""
    RasterField

Regular lon-lat representation used by fast debug maps and movies.
"""
struct RasterField{F <: AbstractFieldTransform, TT}
    lons::Vector{Float64}
    lats::Vector{Float64}
    values::Matrix{Float64}
    name::Symbol
    units::String
    time::TT
    transform::F
    source_topology::Symbol
end

"""
    SnapshotRegridCache()

Cache expensive topology-to-raster geometry, especially CS→LL conservative
regridders, across frames in snapshot grids and movies.
"""
struct SnapshotRegridCache
    entries::Dict{Any, Any}
end

SnapshotRegridCache() = SnapshotRegridCache(Dict{Any, Any}())

"""
    PlotSpec(variable; transform=ColumnMean(), level=nothing, title=nothing, unit=NativeUnit())

One panel specification for multi-panel plots and movies.
"""
struct PlotSpec{F <: AbstractFieldTransform, U <: AbstractDisplayUnit}
    variable::Symbol
    transform::F
    level::Union{Nothing, Int}
    title::Union{Nothing, String}
    unit::U
end

PlotSpec(variable::Symbol; transform=ColumnMean(),
         level::Union{Nothing, Int}=nothing,
         title::Union{Nothing, String}=nothing,
         unit=NativeUnit()) =
    PlotSpec(variable, _parse_field_transform(transform), level, title,
             _parse_display_unit(unit))

PlotSpec(variable::AbstractString; kwargs...) = PlotSpec(Symbol(variable); kwargs...)

_is_cs_snapshot(ds) = haskey(ds.dim, "Xdim") && haskey(ds.dim, "nf")
_is_lonlat_snapshot(ds) = haskey(ds.dim, "lon") && haskey(ds.dim, "lat")

function _read_attributes(ds)
    attrs = Dict{String, Any}()
    for key in keys(ds.attrib)
        attrs[String(key)] = ds.attrib[key]
    end
    return attrs
end

function _read_times(ds)
    haskey(ds, "time") || return (Float64[], Float64[])
    var = ds["time"]
    hours = Float64.(collect(var.var[:]))
    units = strip(String(get(var.attrib, "units", "")))
    if occursin(r"^hours\s+since\s+"i, units)
        return DateTime.(decode_time(ds, "time", (CFDecoder(),))), hours
    end
    # Legacy snapshots did not encode their origin. Keep their elapsed-hour API
    # so old files remain inspectable, while current CF snapshots expose dates.
    return hours, hours
end

function _snapshot_variables(ds, topology::AbstractSnapshotTopology)
    names = Symbol[]
    if topology isa CubedSphereSnapshotTopology
        for key in keys(ds)
            key in ("time", "nf", "air_mass", "air_mass_per_area") && continue
            v = ds[key]
            ndims(v) == 5 && push!(names, Symbol(key))
            endswith(String(key), "_column_mean") &&
                push!(names, Symbol(chop(String(key); tail=length("_column_mean"))))
        end
    else
        for key in keys(ds)
            endswith(String(key), "_column_mean") || continue
            base = first(split(String(key), "_column_mean"))
            push!(names, Symbol(base))
        end
    end
    return sort!(unique(names))
end

"""
    open_snapshot(path) -> SnapshotDataset

Read snapshot metadata and infer its topology.

Supported formats are the current AtmosTransport LL/RG column-mean snapshots
and CS panel snapshots written by `run_driven_simulation`.
"""
function open_snapshot(path::AbstractString)
    expanded = expanduser(String(path))
    isfile(expanded) || throw(ArgumentError("snapshot file not found: $(expanded)"))

    NCDataset(expanded, "r") do ds
        attrs = _read_attributes(ds)
        times, hours = _read_times(ds)
        topology = if _is_cs_snapshot(ds)
            Nc = haskey(attrs, "Nc") ? Int(attrs["Nc"]) : length(ds.dim["Xdim"])
            nlevel = haskey(ds, "lev") ? size(ds["lev"], 1) : size(ds["air_mass"], 4)
            conv = parse_panel_convention(get(attrs, "panel_convention", "gnomonic"))
            CubedSphereSnapshotTopology(Nc, nlevel, conv, _cs_snapshot_definition(attrs, conv))
        elseif _is_lonlat_snapshot(ds)
            lons = Float64.(collect(ds["lon"][:]))
            lats = Float64.(collect(ds["lat"][:]))
            grid_type = Symbol(get(attrs, "grid_type", "latlon"))
            if occursin("reduced_gaussian", String(grid_type))
                nr = haskey(attrs, "nrings") ? Int(attrs["nrings"]) : length(lats)
                rg = String(get(attrs, "regridding", "unknown"))
                ReducedGaussianSnapshotTopology(lons, lats, nr, rg)
            else
                LatLonSnapshotTopology(lons, lats, grid_type)
            end
        else
            throw(ArgumentError("$(expanded) is not a recognized AtmosTransport snapshot"))
        end
        variables = _snapshot_variables(ds, topology)
        return SnapshotDataset(expanded, topology, times, hours, variables, attrs)
    end
end

available_variables(snapshot::SnapshotDataset) = snapshot.variables
snapshot_times(snapshot::SnapshotDataset) = snapshot.times
snapshot_topology(snapshot::SnapshotDataset) = snapshot.topology

function _resolve_time_index(snapshot::SnapshotDataset, time)
    times = snapshot.times
    isempty(times) && return Int(time)
    if time isa Integer
        1 <= time <= length(times) ||
            throw(ArgumentError("time index $(time) outside 1:$(length(times))"))
        return Int(time)
    elseif time isa Real
        _, idx = findmin(abs.(snapshot.hours .- Float64(time)))
        return idx
    elseif time isa Dates.TimeType && eltype(times) <: Dates.TimeType
        _, idx = findmin(abs.(Dates.value.(times .- time)))
        return idx
    else
        throw(ArgumentError("time must be an integer index, numeric snapshot hour, or decoded date"))
    end
end

frame_indices(snapshot::SnapshotDataset, ::AllFrames=AllFrames()) =
    collect(eachindex(snapshot.times))
frame_indices(snapshot::SnapshotDataset, times::AbstractVector) =
    [_resolve_time_index(snapshot, t) for t in times]
frame_indices(snapshot::SnapshotDataset, times::AbstractRange) =
    [_resolve_time_index(snapshot, t) for t in collect(times)]
frame_indices(snapshot::SnapshotDataset, time::Union{Real, Dates.TimeType}) =
    [_resolve_time_index(snapshot, time)]
function frame_indices(::SnapshotDataset, times)
    throw(ArgumentError(
        "frame selection must be AllFrames(), a numeric time/index, range, or vector; got $(repr(times))"))
end

_scale_units!(values, units::String, ::NativeUnit) = (values, units)
function _scale_units!(values, ::String, ::PPMUnit)
    values .*= 1.0e6
    return values, "ppm"
end

function _field_key(::SnapshotDataset, variable::Symbol, ::ColumnMean)
    return "$(variable)_column_mean"
end
function _field_key(snapshot::SnapshotDataset, ::Symbol, transform::AbstractFieldTransform)
    throw(ArgumentError(
        "$(typeof(snapshot.topology)) only has column-mean variables in this snapshot format; " *
        "got transform=$(field_transform_label(transform))"))
end

function _cs_column_mean(vmr, air)
    Nc1, Nc2, nf, Nz = size(vmr)
    out = zeros(Float64, Nc1, Nc2, nf)
    @inbounds for p in 1:nf, j in 1:Nc2, i in 1:Nc1
        num = 0.0
        den = 0.0
        for k in 1:Nz
            m = Float64(air[i, j, p, k])
            num += Float64(vmr[i, j, p, k]) * m
            den += m
        end
        out[i, j, p] = den > 0 ? num / den : NaN
    end
    return out
end

function _cs_column_sum(values)
    Nc1, Nc2, nf, Nz = size(values)
    out = zeros(Float64, Nc1, Nc2, nf)
    @inbounds for p in 1:nf, j in 1:Nc2, i in 1:Nc1
        acc = 0.0
        for k in 1:Nz
            acc += Float64(values[i, j, p, k])
        end
        out[i, j, p] = acc
    end
    return out
end

function _read_lonlat_field(snapshot::SnapshotDataset, ds, variable::Symbol,
                            transform::AbstractFieldTransform, ti::Int,
                            unit::AbstractDisplayUnit)
    key = _field_key(snapshot, variable, transform)
    haskey(ds, key) || throw(ArgumentError("$(basename(snapshot.path)) has no variable `$(key)`"))
    values = Array{Float64}(ds[key][:, :, ti])
    units = String(get(ds[key].attrib, "units", ""))
    values, units = _scale_units!(values, units, unit)
    return HorizontalField(snapshot.topology, values, variable, units,
                           snapshot.times[ti], ti, transform, nothing, snapshot.path)
end

function _stored_cs_column_mean(snapshot::SnapshotDataset, ds, variable::Symbol,
                                transform::ColumnMean, ti::Int,
                                unit::AbstractDisplayUnit)
    name = String(variable)
    column_name = "$(name)_column_mean"
    haskey(ds, column_name) || return nothing
    values = Array{Float64}(ds[column_name][:, :, :, ti])
    units = String(get(ds[column_name].attrib, "units", "mol mol-1"))
    values, units = _scale_units!(values, units, unit)
    return HorizontalField(snapshot.topology, values, variable, units,
                           snapshot.times[ti], ti, transform, nothing, snapshot.path)
end
_stored_cs_column_mean(::SnapshotDataset, ds, ::Symbol,
                       ::AbstractFieldTransform, ::Int, ::AbstractDisplayUnit) = nothing

function _transform_cs_field(snapshot::SnapshotDataset, ds, raw, levels, selected::Bool,
                             ::ColumnMean, ::Union{Nothing, Int}, ti::Int)
    selected &&
        throw(ArgumentError("CS column_mean requires full levels or a stored column mean"))
    haskey(ds, "air_mass") ||
        throw(ArgumentError("CS column_mean requires `air_mass` in $(basename(snapshot.path))"))
    return _cs_column_mean(raw, Array{Float64}(ds["air_mass"][:, :, :, :, ti]))
end

_transform_cs_field(::SnapshotDataset, ds, raw, levels, selected::Bool,
                    ::ColumnSum, ::Union{Nothing, Int}, ti::Int) =
    _cs_column_sum(raw)

function _transform_cs_field(::SnapshotDataset, ds, raw, levels, selected::Bool,
                             ::LevelSlice, level::Union{Nothing, Int}, ti::Int)
    lev = level === nothing ? throw(ArgumentError("level_slice requires `level`")) : level
    index = findfirst(==(lev), levels)
    index === nothing && throw(ArgumentError("model level $(lev) was not written"))
    return Array{Float64}(raw[:, :, :, index])
end

function _transform_cs_field(snapshot::SnapshotDataset, ds, raw, levels, selected::Bool,
                             ::SurfaceSlice, level::Union{Nothing, Int}, ti::Int)
    lev = level === nothing ? snapshot.topology.nlevel : level
    index = findfirst(==(lev), levels)
    index === nothing && throw(ArgumentError("model level $(lev) was not written"))
    return Array{Float64}(raw[:, :, :, index])
end

_default_cs_units(::ColumnMean) = "mol mol-1"
_default_cs_units(::ColumnSum) = ""
_default_cs_units(::LevelSlice) = "mol mol-1"
_default_cs_units(::SurfaceSlice) = "mol mol-1"

function _read_cs_field(snapshot::SnapshotDataset, ds, variable::Symbol,
                        transform::AbstractFieldTransform, ti::Int,
                        level::Union{Nothing, Int}, unit::AbstractDisplayUnit)
    stored = _stored_cs_column_mean(snapshot, ds, variable, transform, ti, unit)
    stored === nothing || return stored
    name = String(variable)
    haskey(ds, name) || throw(ArgumentError("$(basename(snapshot.path)) has no CS variable `$(name)`"))
    raw = Array{Float64}(ds[name][:, :, :, :, ti])
    selected = "lev_selected" in dimnames(ds[name])
    levels = selected ? Int.(ds["lev_selected"][:]) : collect(1:size(raw, 4))
    values = _transform_cs_field(snapshot, ds, raw, levels, selected,
                                 transform, level, ti)
    units = String(get(ds[name].attrib, "units", _default_cs_units(transform)))
    values, units = _scale_units!(values, units, unit)
    return HorizontalField(snapshot.topology, values, variable, units,
                           snapshot.times[ti], ti, transform, level, snapshot.path)
end

"""
    fieldview(snapshot, variable; transform=ColumnMean(), time=1, level=nothing, unit=NativeUnit())

Load one topology-native horizontal field from a snapshot file.

`time` may be a one-based frame index or a snapshot hour. For CS snapshots,
`transform` may be `ColumnMean()`, `ColumnSum()`, `LevelSlice()`, or
`SurfaceSlice()`. Stored CS column means support column-only output; level
selection uses original model indices and rejects levels that were not written.
LL/RG snapshots currently support `ColumnMean()` only.
"""
function fieldview(snapshot::SnapshotDataset, variable::Union{Symbol, AbstractString};
                   transform=ColumnMean(),
                   time=1,
                   level::Union{Nothing, Int}=nothing,
                   unit=NativeUnit())
    var = Symbol(variable)
    parsed_transform = _require_field_transform(transform)
    parsed_unit = _require_display_unit(unit)
    ti = _resolve_time_index(snapshot, time)
    NCDataset(snapshot.path, "r") do ds
        if snapshot.topology isa CubedSphereSnapshotTopology
            return _read_cs_field(snapshot, ds, var, parsed_transform, ti, level, parsed_unit)
        else
            return _read_lonlat_field(snapshot, ds, var, parsed_transform, ti, parsed_unit)
        end
    end
end

function _source_topology_symbol(::LatLonSnapshotTopology)
    :latlon
end

function _source_topology_symbol(::ReducedGaussianSnapshotTopology)
    :reduced_gaussian
end

function _source_topology_symbol(::CubedSphereSnapshotTopology)
    :cubed_sphere
end

function _target_lonlat_mesh(resolution::Tuple{Int, Int})
    Nx, Ny = resolution
    mesh = LatLonMesh(; FT=Float64, Nx=Nx, Ny=Ny,
                      longitude=(-180.0, 180.0),
                      latitude=(-90.0, 90.0),
                      radius=IFS_EARTH_RADIUS)
    return mesh, Float64.(mesh.λᶜ), Float64.(mesh.φᶜ)
end

# The mesh definition recorded by the snapshot writer (`define_cs_horizontal_geometry!`);
# files written before those attributes get the panel convention's default.
function _cs_snapshot_definition(attrs,
                                 convention::AbstractCubedSpherePanelConvention)
    haskey(attrs, "cs_coordinate_law") || return default_cs_definition(convention)
    law = parse_coordinate_law(attrs["cs_coordinate_law"])
    center = parse_center_law(get(attrs, "cs_center_law", ""))
    return CubedSphereDefinition(law, center, convention;
                                 longitude_offset_deg = Float64(attrs["longitude_of_central_meridian"]),
                                 tag = Symbol(get(attrs, "cs_definition", "custom")))
end

function _cs_to_ll_cache!(cache::SnapshotRegridCache,
                          topology::CubedSphereSnapshotTopology,
                          resolution::Tuple{Int, Int})
    key = (:cs_to_ll, topology.Nc, resolution, topology.definition)
    if !haskey(cache.entries, key)
        ll_mesh, lons, lats = _target_lonlat_mesh(resolution)
        cs_mesh = CubedSphereMesh(; FT=Float64,
                                  Nc=topology.Nc,
                                  radius=IFS_EARTH_RADIUS,
                                  definition=topology.definition)
        regridder = build_regridder(cs_mesh, ll_mesh; normalize=false)
        cache.entries[key] = (; regridder, lons, lats)
    end
    return cache.entries[key]
end

"""
    as_raster(field; resolution=(360, 181), cache=SnapshotRegridCache())

Return a regular lon-lat `RasterField` for fast plotting.

LL and current RG diagnostic snapshots are already raster-like. CS fields are
conservatively regridded to lon-lat and reuse `cache` across frames.
"""
function as_raster(field::HorizontalField;
                   resolution::Tuple{Int, Int}=(360, 181),
                   cache::SnapshotRegridCache=SnapshotRegridCache())
    topology = field.topology
    if topology isa LatLonSnapshotTopology
        return RasterField(topology.lons, topology.lats, Matrix{Float64}(field.values),
                           field.name, field.units, field.time, field.transform,
                           _source_topology_symbol(topology))
    elseif topology isa ReducedGaussianSnapshotTopology
        return RasterField(topology.lons, topology.lats, Matrix{Float64}(field.values),
                           field.name, field.units, field.time, field.transform,
                           _source_topology_symbol(topology))
    elseif topology isa CubedSphereSnapshotTopology
        geom = _cs_to_ll_cache!(cache, topology, resolution)
        Nx, Ny = resolution
        dst = zeros(Float64, Nx * Ny)
        apply_regridder!(dst, geom.regridder, vec(field.values))
        return RasterField(geom.lons, geom.lats, reshape(dst, Nx, Ny),
                           field.name, field.units, field.time, field.transform,
                           _source_topology_symbol(topology))
    else
        throw(ArgumentError("unsupported visualization topology: $(typeof(topology))"))
    end
end

"""
    robust_colorrange(fields; trim=(0.01, 0.99))

Compute a robust color range from one or more fields by ignoring non-finite
values and clipping to quantiles.
"""
function robust_colorrange(fields; trim::Tuple{<:Real, <:Real}=(0.01, 0.99))
    arrays = fields isa AbstractVector ? fields : [fields]
    vals = Float64[]
    for item in arrays
        values = item isa RasterField || item isa HorizontalField ? item.values : item
        append!(vals, Float64[x for x in values if isfinite(x)])
    end
    isempty(vals) && return (0.0, 1.0)
    lo = quantile(vals, Float64(trim[1]))
    hi = quantile(vals, Float64(trim[2]))
    lo == hi && (hi = lo + max(abs(lo), 1.0) * eps(Float64))
    return (lo, hi)
end

function _makie_missing()
    throw(ArgumentError(
        "Makie plotting is not loaded. Load a backend first, e.g. " *
        "`using CairoMakie; using AtmosTransport`, then call the visualization API."))
end

"""
    mapplot(field; kwargs...) -> Figure

Render one topology-aware [`HorizontalField`](@ref) as a lon-lat map. Load a
Makie backend such as CairoMakie before calling this function. Cubed-sphere
fields are conservatively rasterized and may reuse a [`SnapshotRegridCache`](@ref).
"""
function mapplot(args...; kwargs...)
    _makie_missing()
end

"""
    mapplot!(axis, field; kwargs...)

Render a topology-aware field into an existing Makie axis and return the plot
object. A Makie backend must be loaded.
"""
function mapplot!(args...; kwargs...)
    _makie_missing()
end

"""
    snapshot_grid(snapshot, variable; kwargs...) -> Figure

Render one snapshot variable at several output times in a shared figure. A
Makie backend must be loaded.
"""
function snapshot_grid(args...; kwargs...)
    _makie_missing()
end

"""
    movie(snapshot, variable, output_path; kwargs...) -> String

Render one snapshot variable to an animation and return `output_path`. A Makie
backend must be loaded.
"""
function movie(args...; kwargs...)
    _makie_missing()
end

"""
    movie_grid(snapshot, specs, output_path; kwargs...) -> String

Render several [`PlotSpec`](@ref) panels to one animation and return
`output_path`. A Makie backend must be loaded.
"""
function movie_grid(args...; kwargs...)
    _makie_missing()
end

"""
    catrine_map_curtains(at_path, gc_dir; kwargs...) -> NamedTuple

Create the CATRINE AtmosTransport/GEOS-Chem column-map and pressure-curtain
comparison product. Output is controlled by the `out_dir` keyword. This
specialized visualization requires a Makie backend.
"""
function catrine_map_curtains(args...; kwargs...)
    _makie_missing()
end

"""
    catrine_map_curtains_3way(at_geos_path, at_era5_path, gc_dir; kwargs...) -> NamedTuple

Compare GEOS-Chem with AtmosTransport runs driven by GEOS-IT and ERA5 in one
CATRINE column-map and pressure-curtain product. Output is controlled by the
`out_dir` keyword. This specialized visualization requires a Makie backend.
"""
function catrine_map_curtains_3way(args...; kwargs...)
    _makie_missing()
end

export AbstractSnapshotTopology
export LatLonSnapshotTopology, ReducedGaussianSnapshotTopology, CubedSphereSnapshotTopology
export AbstractFieldTransform, ColumnMean, ColumnSum, LevelSlice, SurfaceSlice
export AbstractDisplayUnit, NativeUnit, PPMUnit, AllFrames
export field_transform_label, display_unit_label
export SnapshotDataset, HorizontalField, RasterField, SnapshotRegridCache, PlotSpec
export open_snapshot, available_variables, snapshot_times, snapshot_topology
export fieldview, frame_indices, as_raster, robust_colorrange
export mapplot, mapplot!, snapshot_grid, movie, movie_grid
export catrine_map_curtains, catrine_map_curtains_3way

end # module Visualization
