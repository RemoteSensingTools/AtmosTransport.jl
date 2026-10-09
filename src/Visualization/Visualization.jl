"""
    Visualization

Topology-aware snapshot visualization support.

This module owns the lightweight data layer only: snapshot discovery, field
views, unit scaling, and topology-aware rasterization for quick maps. Makie
plotting lives in `AtmosTransportMakieExt` so the model/runtime does not pay a
plotting load cost unless a Makie backend is explicitly loaded.
"""
module Visualization

using NCDatasets
using Printf
using Statistics: quantile

using ..Grids: LatLonMesh, CubedSphereMesh,
               GnomonicPanelConvention, GEOSNativePanelConvention
using ..Regridding: build_regridder, apply_regridder!
using ..Parameters: IFS_EARTH_RADIUS


abstract type AbstractSnapshotTopology end

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
    CubedSphereSnapshotTopology(Nc, nlevel, panel_convention)

Cubed-sphere snapshot geometry. Values are stored as `(Xdim, Ydim, nf)` for a
single horizontal field, with `nf == 6`.
"""
struct CubedSphereSnapshotTopology <: AbstractSnapshotTopology
    Nc::Int
    nlevel::Int
    panel_convention::Symbol
end

"""
    SnapshotDataset

Metadata for an AtmosTransport NetCDF snapshot file.

The object intentionally does not hold an open NetCDF handle. Field reads open
the file briefly and close it immediately, which is robust for scripts,
animations, and long debugging sessions.
"""
struct SnapshotDataset{T <: AbstractSnapshotTopology}
    path::String
    topology::T
    times::Vector{Float64}
    variables::Vector{Symbol}
    attributes::Dict{String, Any}
end

"""
    HorizontalField

A topology-native horizontal field view from one snapshot time.

For LL/RG snapshots `values` is a `(lon, lat)` matrix. For CS snapshots it is a
`(Xdim, Ydim, nf)` array until rasterized or plotted natively.
"""
struct HorizontalField{T <: AbstractSnapshotTopology, A}
    topology::T
    values::A
    name::Symbol
    units::String
    time::Float64
    time_index::Int
    transform::Symbol
    level::Union{Nothing, Int}
    source_path::String
end

"""
    RasterField

Regular lon-lat representation used by fast debug maps and movies.
"""
struct RasterField
    lons::Vector{Float64}
    lats::Vector{Float64}
    values::Matrix{Float64}
    name::Symbol
    units::String
    time::Float64
    transform::Symbol
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
    PlotSpec(variable; transform=:column_mean, level=nothing, title=nothing, unit=:native)

One panel specification for multi-panel plots and movies.
"""
struct PlotSpec
    variable::Symbol
    transform::Symbol
    level::Union{Nothing, Int}
    title::Union{Nothing, String}
    unit::Symbol
end

PlotSpec(variable::Symbol; transform::Symbol=:column_mean,
         level::Union{Nothing, Int}=nothing,
         title::Union{Nothing, String}=nothing,
         unit::Symbol=:native) =
    PlotSpec(variable, transform, level, title, unit)

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
    haskey(ds, "time") || return Float64[]
    # Snapshots write time as Float64 with units "hours since YYYY-MM-DD".
    # NCDatasets auto-converts that to a Vector{DateTime} via CF-time
    # decoding. Use `.var[:]` to bypass and read the raw Float64s, falling
    # back to DateTime→Float64 hours via the time origin attribute when a
    # legacy file stores actual DateTime values.
    var = ds["time"]
    raw = var.var[:]
    if eltype(raw) <: AbstractFloat
        return Float64.(collect(raw))
    end
    return Float64.(collect(raw))
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
        times = _read_times(ds)
        topology = if _is_cs_snapshot(ds)
            Nc = haskey(attrs, "Nc") ? Int(attrs["Nc"]) : length(ds.dim["Xdim"])
            nlevel = haskey(ds, "lev") ? size(ds["lev"], 1) : size(ds["air_mass"], 4)
            conv = Symbol(get(attrs, "panel_convention", "gnomonic"))
            CubedSphereSnapshotTopology(Nc, nlevel, conv)
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
        return SnapshotDataset(expanded, topology, times, variables, attrs)
    end
end

available_variables(snapshot::SnapshotDataset) = snapshot.variables
snapshot_times(snapshot::SnapshotDataset) = snapshot.times
snapshot_topology(snapshot::SnapshotDataset) = snapshot.topology

function _resolve_time_index(times::AbstractVector{<:Real}, time)
    isempty(times) && return Int(time)
    if time isa Integer
        1 <= time <= length(times) ||
            throw(ArgumentError("time index $(time) outside 1:$(length(times))"))
        return Int(time)
    elseif time isa Real
        _, idx = findmin(abs.(Float64.(times) .- Float64(time)))
        return idx
    else
        throw(ArgumentError("time must be an integer index or numeric snapshot hour"))
    end
end

function frame_indices(snapshot::SnapshotDataset, times=:all)
    if times === :all
        return collect(eachindex(snapshot.times))
    elseif times isa AbstractVector
        return [_resolve_time_index(snapshot.times, t) for t in times]
    elseif times isa AbstractRange
        return [_resolve_time_index(snapshot.times, t) for t in collect(times)]
    else
        return [_resolve_time_index(snapshot.times, times)]
    end
end

function _scale_units!(values, units::String, unit::Symbol)
    unit === :native && return values, units
    if unit === :ppm
        values .*= 1.0e6
        return values, "ppm"
    end
    throw(ArgumentError("unsupported visualization unit: $(unit)"))
end

function _field_key(snapshot::SnapshotDataset, variable::Symbol, transform::Symbol)
    transform === :column_mean ||
        throw(ArgumentError("$(typeof(snapshot.topology)) only has column-mean variables in this snapshot format; got transform=$(transform)"))
    return "$(variable)_column_mean"
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
                            transform::Symbol, ti::Int, unit::Symbol)
    key = _field_key(snapshot, variable, transform)
    haskey(ds, key) || throw(ArgumentError("$(basename(snapshot.path)) has no variable `$(key)`"))
    values = Array{Float64}(ds[key][:, :, ti])
    units = String(get(ds[key].attrib, "units", ""))
    values, units = _scale_units!(values, units, unit)
    return HorizontalField(snapshot.topology, values, variable, units,
                           snapshot.times[ti], ti, transform, nothing, snapshot.path)
end

function _read_cs_field(snapshot::SnapshotDataset, ds, variable::Symbol,
                        transform::Symbol, ti::Int,
                        level::Union{Nothing, Int}, unit::Symbol)
    name = String(variable)
    column_name = "$(name)_column_mean"
    if transform === :column_mean && haskey(ds, column_name)
        values = Array{Float64}(ds[column_name][:, :, :, ti])
        units = String(get(ds[column_name].attrib, "units", "mol mol-1"))
        values, units = _scale_units!(values, units, unit)
        return HorizontalField(snapshot.topology, values, variable, units,
                               snapshot.times[ti], ti, transform, nothing, snapshot.path)
    end
    haskey(ds, name) || throw(ArgumentError("$(basename(snapshot.path)) has no CS variable `$(name)`"))
    raw = Array{Float64}(ds[name][:, :, :, :, ti])
    selected = "lev_selected" in dimnames(ds[name])
    levels = selected ? Int.(ds["lev_selected"][:]) : collect(1:size(raw, 4))
    values = if transform === :column_mean
        selected && throw(ArgumentError("CS column_mean requires full levels or a stored column mean"))
        haskey(ds, "air_mass") ||
            throw(ArgumentError("CS column_mean requires `air_mass` in $(basename(snapshot.path))"))
        air = Array{Float64}(ds["air_mass"][:, :, :, :, ti])
        _cs_column_mean(raw, air)
    elseif transform === :column_sum
        _cs_column_sum(raw)
    elseif transform === :level_slice
        lev = level === nothing ? throw(ArgumentError("level_slice requires `level`")) : level
        index = findfirst(==(lev), levels)
        index === nothing && throw(ArgumentError("model level $(lev) was not written"))
        Array{Float64}(raw[:, :, :, index])
    elseif transform === :surface_slice
        lev = level === nothing ? snapshot.topology.nlevel : level
        index = findfirst(==(lev), levels)
        index === nothing && throw(ArgumentError("model level $(lev) was not written"))
        Array{Float64}(raw[:, :, :, index])
    else
        throw(ArgumentError("unsupported CS transform: $(transform)"))
    end
    units = String(get(ds[name].attrib, "units", transform in (:column_mean, :level_slice, :surface_slice) ? "mol mol-1" : ""))
    values, units = _scale_units!(values, units, unit)
    return HorizontalField(snapshot.topology, values, variable, units,
                           snapshot.times[ti], ti, transform, level, snapshot.path)
end

"""
    fieldview(snapshot, variable; transform=:column_mean, time=1, level=nothing, unit=:native)

Load one topology-native horizontal field from a snapshot file.

`time` may be a one-based frame index or a snapshot hour. For CS snapshots,
`transform` may be `:column_mean`, `:column_sum`, `:level_slice`, or
`:surface_slice`. Stored CS column means support column-only output; level
selection uses original model indices and rejects levels that were not written.
LL/RG snapshots currently carry column means only.
"""
function fieldview(snapshot::SnapshotDataset, variable::Union{Symbol, AbstractString};
                   transform::Symbol=:column_mean,
                   time=1,
                   level::Union{Nothing, Int}=nothing,
                   unit::Symbol=:native)
    var = Symbol(variable)
    ti = _resolve_time_index(snapshot.times, time)
    NCDataset(snapshot.path, "r") do ds
        if snapshot.topology isa CubedSphereSnapshotTopology
            return _read_cs_field(snapshot, ds, var, transform, ti, level, unit)
        else
            return _read_lonlat_field(snapshot, ds, var, transform, ti, unit)
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

function _cs_panel_convention(sym::Symbol)
    sym === :gnomonic && return GnomonicPanelConvention()
    sym === :geos_native && return GEOSNativePanelConvention()
    throw(ArgumentError("unsupported CS panel_convention=$(sym); expected :gnomonic or :geos_native"))
end

function _cs_to_ll_cache!(cache::SnapshotRegridCache,
                          topology::CubedSphereSnapshotTopology,
                          resolution::Tuple{Int, Int})
    key = (:cs_to_ll, topology.Nc, resolution, topology.panel_convention)
    if !haskey(cache.entries, key)
        ll_mesh, lons, lats = _target_lonlat_mesh(resolution)
        convention = _cs_panel_convention(topology.panel_convention)
        cs_mesh = CubedSphereMesh(; FT=Float64,
                                  Nc=topology.Nc,
                                  radius=IFS_EARTH_RADIUS,
                                  convention=convention)
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
export SnapshotDataset, HorizontalField, RasterField, SnapshotRegridCache, PlotSpec
export open_snapshot, available_variables, snapshot_times, snapshot_topology
export fieldview, frame_indices, as_raster, robust_colorrange
export mapplot, mapplot!, snapshot_grid, movie, movie_grid
export catrine_map_curtains, catrine_map_curtains_3way

end # module Visualization
