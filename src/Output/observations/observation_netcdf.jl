# ---------------------------------------------------------------------------
# Append-only NetCDF sinks for observation sampling: one file for point
# events (`_soundings`, unlimited `obs`), one for station series (`_sites`,
# fixed `site`, unlimited `time`).
#
# Both follow the snapshot stream contract: every append is flushed with
# `sync`, a `completed_*` attribute records the last successful append, a
# failed append poisons the stream, and `close` is idempotent. Writes are
# queued and flushed whenever the shared NetCDF lock is free, so the
# transport loop never waits for a background daily snapshot write; `close`
# drains the queue. A file is created on the first flush, so a run with no
# matches still leaves a valid, empty file.
# ---------------------------------------------------------------------------

const _OBSERVATION_CONTRACT = "AtmosTransport observations v1"
const _OBSERVATION_TIME_UNITS = "seconds since 1970-01-01 00:00:00"
const _OBSERVATION_CHUNK = 4096

"""
    AbstractObservationStream

Append-only NetCDF sink with the snapshot-stream contract. Concrete streams
have `path`, `dataset`, `count`, `closed`, `failed`, and `queue` fields;
writes are queued closures `op(stream)` that run under the shared NetCDF lock.
"""
abstract type AbstractObservationStream end

function _check_open(stream::AbstractObservationStream, what::AbstractString)
    stream.failed && throw(ArgumentError("cannot append $(what) after a NetCDF write failure"))
    stream.closed && throw(ArgumentError("cannot append $(what) to a closed stream"))
    return nothing
end

# Run queued writes. Without `block`, give up if the lock is busy (another
# task is writing a snapshot file); the rows stay queued for the next call.
function _drain!(stream::AbstractObservationStream; block::Bool)
    isempty(stream.queue) && return true
    if block
        lock(_NETCDF_IO_LOCK)
    else
        trylock(_NETCDF_IO_LOCK) || return false
    end
    try
        while !isempty(stream.queue)
            popfirst!(stream.queue)(stream)
        end
    catch
        # Poison: drop the queue, close what is open, preserve the error.
        stream.failed = true
        empty!(stream.queue)
        ds = stream.dataset
        stream.dataset = nothing
        stream.closed = true
        try
            ds === nothing || close(ds)
        catch
        end
        rethrow()
    finally
        unlock(_NETCDF_IO_LOCK)
    end
    return true
end

function _submit!(stream::AbstractObservationStream, op, what::AbstractString)
    _check_open(stream, what)
    push!(stream.queue, op)
    _drain!(stream; block = false)
    return nothing
end

"Number of queued writes not yet flushed to disk."
pending_writes(stream::AbstractObservationStream) = length(stream.queue)

function Base.close(stream::AbstractObservationStream)
    stream.closed && return nothing
    try
        _drain!(stream; block = true)
    finally
        ds = stream.dataset
        stream.dataset = nothing
        stream.closed = true
        ds === nothing || with_netcdf_lock(() -> close(ds))
    end
    return nothing
end

# Close only if that does not wait for the lock; used for retired daily files.
function _try_close!(stream::AbstractObservationStream)
    stream.closed && return true
    trylock(_NETCDF_IO_LOCK) || return false
    try
        close(stream)
    finally
        unlock(_NETCDF_IO_LOCK)
    end
    return true
end

"Queue summary attributes (written at day rotation and at the end of a run)."
function write_summary_attributes!(stream::AbstractObservationStream, attributes::AbstractDict)
    (stream.closed || stream.failed) && return nothing
    values = Dict{String, Any}(String(k) => v for (k, v) in pairs(attributes))
    _submit!(stream, st -> begin
                 for (key, value) in values
                     st.dataset.attrib[key] = value
                 end
                 NCDatasets.sync(st.dataset)
             end, "summary attributes")
    return nothing
end

# Create the file and its schema on the first flush.
function _create_op(define!)
    return stream -> begin
        _ensure_parent_dir(stream.path)
        ds = _create_netcdf_dataset(stream.path)
        try
            define!(ds)
            NCDatasets.sync(ds)
        catch
            close(ds)
            rethrow()
        end
        stream.dataset = ds
    end
end

"Variable-name suffix and long-name wording for the state's mass basis."
_basis_suffix(::DryBasis) = "_dry"
_basis_suffix(::MoistBasis) = ""
_basis_word(::DryBasis) = "dry "
_basis_word(::MoistBasis) = ""

# Streams take the basis as a type (`DryBasis()`) or, like the snapshot
# writers, as `:dry` / `:moist`.
const _MASS_BASES = (dry = DryBasis(), moist = MoistBasis())
_as_mass_basis(basis::AbstractMassBasis) = basis
_as_mass_basis(basis::Symbol) = haskey(_MASS_BASES, basis) ? _MASS_BASES[basis] :
    throw(ArgumentError("mass_basis must be :dry or :moist; got $(repr(basis))"))

_has_panel(::CubedSphereMesh) = true
_has_panel(_) = false

function _observation_common_attributes!(ds, mesh, mass_basis::AbstractMassBasis, origin::DateTime,
                                         attributes::AbstractDict)
    ds.attrib["Conventions"] = "CF-1.8"
    ds.attrib["title"] = "AtmosTransport observation sampling"
    ds.attrib["source"] = "AtmosTransport.jl"
    ds.attrib["institution"] = get(ENV, "ATMOSTR_INSTITUTION", "Caltech / Frankenberg group")
    ds.attrib["grid"] = summary(mesh)
    ds.attrib["grid_type"] = _grid_type_string(mesh)
    ds.attrib["mass_basis"] = String(_basis_symbol(mass_basis))
    ds.attrib["output_contract"] = _OBSERVATION_CONTRACT
    ds.attrib["run_time_origin"] = Dates.format(origin, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
    ds.attrib["horizontal_sampling"] = "containing_cell"
    ds.attrib["vertical_ordering"] = "lev[1] is the top of the atmosphere; lev[end] is the surface"
    ds.attrib["pressure_reconstruction"] =
        "p_half[1] = A_ifc[1]; p_half[k+1] = p_half[k] + g * air_mass[k] / cell_area " *
        "(" * _basis_word(mass_basis) * "air)"
    ds.attrib["height_method_codes"] = height_method_codes_description()
    _define_provenance_attributes!(ds)
    for (key, value) in pairs(attributes)
        ds.attrib[String(key)] = value
    end
    return nothing
end

# Floats carry the snapshot writer's fill sentinel as both `_FillValue` and
# `missing_value`; strings cannot be compressed or chunked.
function _def_obs_var(ds, name::AbstractString, ::Type{T}, dims; deflate_level::Int,
                      attrib = Dict{String, Any}(), chunks = nothing) where {T}
    compress = deflate_level > 0 && T !== String
    if T <: AbstractFloat
        attrib = merge(Dict{String, Any}("missing_value" => _payload_fill_value(T)), attrib)
        return defVar(ds, name, T, dims; attrib, fillvalue = _payload_fill_value(T),
                      deflatelevel = compress ? deflate_level : 0, shuffle = compress,
                      chunksizes = chunks)
    end
    return defVar(ds, name, T, dims; attrib, deflatelevel = compress ? deflate_level : 0,
                  shuffle = compress, chunksizes = T === String ? nothing : chunks)
end

_time_attrib(long_name) = Dict{String, Any}("units" => _OBSERVATION_TIME_UNITS,
                                             "calendar" => "proleptic_gregorian",
                                             "long_name" => long_name)
_attrib(units, long_name) = Dict{String, Any}("units" => units, "long_name" => long_name)
_attrib(long_name) = Dict{String, Any}("long_name" => long_name)

# Every variable name a sampler run may create, so tracer names can be
# checked before any file is opened.
const _OBSERVATION_FIXED_NAMES = (
    "id", "source", "time", "latitude", "longitude", "elevation", "intake_height",
    "cell_lat", "cell_lon", "cell_index", "cell_i", "cell_j", "cell_panel", "cell_area",
    "sample_time_prev", "sample_time_next", "interp_weight", "interp_flag",
    "intake_level", "intake_layer_bottom_agl", "intake_layer_top_agl", "height_method",
    "site_id", "schedule_start", "schedule_end", "lev", "ilev", "obs", "site",
    "ps", "ps_dry", "p_half", "p_half_dry", "air_mass_per_area", "air_mass_per_area_dry")
const _OBSERVATION_TRACER_SUFFIXES = ("", "_column_mean", "_intake", "_surface")

"Throw if tracer names would collide with each other or with built-in variables."
function check_observation_tracer_names(names::AbstractVector{Symbol})
    taken = Set{String}(_OBSERVATION_FIXED_NAMES)
    for name in names, suffix in _OBSERVATION_TRACER_SUFFIXES
        var = String(name) * suffix
        var in taken && throw(ArgumentError(
            "observation output variable $(repr(var)) for tracer $(name) collides with another " *
            "output variable; rename the tracer or drop it from [output.observations].tracers"))
        push!(taken, var)
    end
    return nothing
end

# -- point events (soundings) ----------------------------------------------

"""
    SoundingNetCDFStream(path, mesh, nlevel, tracer_names; mass_basis, origin, deflate_level, attributes)

Append-only point-event sink. Each appended batch writes one row per event:
identity, time, location, containing cell, bracketing sample times and
weight, interface pressures, per-layer air mass per area, each tracer's
mixing-ratio profile and column mean, and the intake-layer value.
"""
mutable struct SoundingNetCDFStream <: AbstractObservationStream
    path::String
    dataset::Union{Nothing, NCDataset}
    count::Int
    nlevel::Int
    tracer_names::Vector{Symbol}
    mass_basis::AbstractMassBasis
    has_panel::Bool
    closed::Bool
    failed::Bool
    queue::Vector{Any}
end

"""
    SoundingBatch(; ids, sources, times, latitudes, longitudes, cells, sample_time_prev,
                  sample_time_next, weights, flags, p_half, air_mass_per_area, tracers,
                  column_means, elevations, intake_heights, intake_levels, layer_bottom,
                  layer_top, height_methods, intake_values)

Columns of one emitted batch of `n` events. Profiles are `(nlevel, n)`,
`p_half` is `(nlevel + 1, n)`, tracers `(nlevel, ntracer, n)`, column means
and intake values `(ntracer, n)`. Times are seconds since 1970-01-01 UTC.
"""
struct SoundingBatch
    ids::Vector{String}
    sources::Vector{Int32}
    times::Vector{Float64}
    latitudes::Vector{Float64}
    longitudes::Vector{Float64}
    cells::Vector{CellLocation}
    sample_time_prev::Vector{Float64}
    sample_time_next::Vector{Float64}
    weights::Vector{Float32}
    flags::Vector{Int8}
    p_half::Matrix{Float64}
    air_mass_per_area::Matrix{Float64}
    tracers::Array{Float64, 3}
    column_means::Matrix{Float64}
    elevations::Vector{Float64}
    intake_heights::Vector{Float64}
    intake_levels::Vector{Int32}
    layer_bottom::Vector{Float64}
    layer_top::Vector{Float64}
    height_methods::Vector{Int8}
    intake_values::Matrix{Float64}

    function SoundingBatch(; ids, sources, times, latitudes, longitudes, cells, sample_time_prev,
                           sample_time_next, weights, flags, p_half, air_mass_per_area, tracers,
                           column_means, elevations, intake_heights, intake_levels, layer_bottom,
                           layer_top, height_methods, intake_values)
        n = length(ids)
        for (name, v) in (("sources", sources), ("times", times), ("latitudes", latitudes),
                          ("longitudes", longitudes), ("cells", cells),
                          ("sample_time_prev", sample_time_prev), ("sample_time_next", sample_time_next),
                          ("weights", weights), ("flags", flags), ("elevations", elevations),
                          ("intake_heights", intake_heights), ("intake_levels", intake_levels),
                          ("layer_bottom", layer_bottom), ("layer_top", layer_top),
                          ("height_methods", height_methods))
            length(v) == n || throw(DimensionMismatch("SoundingBatch.$(name) has $(length(v)) rows; expected $(n)"))
        end
        nlevel = size(air_mass_per_area, 1)
        ntracer = size(tracers, 2)
        size(p_half) == (nlevel + 1, n) || throw(DimensionMismatch("SoundingBatch.p_half must be (nlevel + 1, n)"))
        size(air_mass_per_area, 2) == n || throw(DimensionMismatch("SoundingBatch.air_mass_per_area must be (nlevel, n)"))
        size(tracers, 1) == nlevel && size(tracers, 3) == n ||
            throw(DimensionMismatch("SoundingBatch.tracers must be (nlevel, ntracer, n)"))
        size(column_means) == (ntracer, n) || throw(DimensionMismatch("SoundingBatch.column_means must be (ntracer, n)"))
        size(intake_values) == (ntracer, n) || throw(DimensionMismatch("SoundingBatch.intake_values must be (ntracer, n)"))
        return new(ids, sources, times, latitudes, longitudes, cells, sample_time_prev, sample_time_next,
                   weights, flags, p_half, air_mass_per_area, tracers, column_means, elevations,
                   intake_heights, intake_levels, layer_bottom, layer_top, height_methods, intake_values)
    end
end

Base.length(batch::SoundingBatch) = length(batch.ids)

function SoundingNetCDFStream(path::AbstractString, mesh, nlevel::Integer,
                              tracer_names::AbstractVector{Symbol};
                              mass_basis::Union{Symbol, AbstractMassBasis}, origin::DateTime,
                              deflate_level::Integer = 0,
                              attributes::AbstractDict = Dict{String, Any}())
    nlevel >= 1 || throw(ArgumentError("sounding stream needs at least one level"))
    isempty(tracer_names) && throw(ArgumentError("sounding stream needs at least one tracer"))
    stream = SoundingNetCDFStream(String(path), nothing, 0, Int(nlevel), collect(tracer_names),
                                  _as_mass_basis(mass_basis), _has_panel(mesh), false, false, Any[])
    attrs = copy(attributes)
    _submit!(stream, _create_op(ds -> _define_sounding_schema!(ds, stream, mesh, origin,
                                                              Int(deflate_level), attrs)), "schema")
    return stream
end

const _INTERP_FLAG_MEANINGS = join(string.(keys(INTERP_FLAG_CODES)), " ")

function _define_sounding_schema!(ds, stream::SoundingNetCDFStream, mesh, origin::DateTime,
                                  dl::Int, attributes)
    nlevel = stream.nlevel
    suffix = _basis_suffix(stream.mass_basis)
    word = _basis_word(stream.mass_basis)
    units = _tracer_units(_basis_symbol(stream.mass_basis))
    defDim(ds, "obs", Inf)
    defDim(ds, "lev", nlevel)
    defDim(ds, "ilev", nlevel + 1)
    _observation_common_attributes!(ds, mesh, stream.mass_basis, origin, attributes)
    ds.attrib["completed_soundings"] = 0
    ds.attrib["tracers"] = join(String.(stream.tracer_names), " ")
    c1 = (_OBSERVATION_CHUNK,)
    obs(name, T, attrib) = _def_obs_var(ds, name, T, ("obs",); deflate_level = dl, chunks = c1, attrib)
    _def_obs_var(ds, "id", String, ("obs",); deflate_level = 0,
                 attrib = _attrib("observation identifier (sounding_id, obspack_id, or table/site id)"))
    obs("source", Int32, _attrib("1-based index into [[output.observations.sources]]"))
    obs("time", Float64, _time_attrib("observation time"))
    obs("latitude", Float64, Dict{String, Any}("units" => "degrees_north", "standard_name" => "latitude"))
    obs("longitude", Float64, Dict{String, Any}("units" => "degrees_east", "standard_name" => "longitude"))
    obs("elevation", Float64, _attrib("m", "surface elevation above sea level from the source (NaN unknown)"))
    obs("intake_height", Float64, _attrib("m", "intake height above ground (NaN: lowest model layer)"))
    obs("cell_lat", Float64, _attrib("degrees_north", "containing cell centre latitude"))
    obs("cell_lon", Float64, _attrib("degrees_east", "containing cell centre longitude (mesh convention)"))
    obs("cell_index", Int32, _attrib("native flat cell index within the panel / mesh"))
    obs("cell_i", Int32, _attrib("cell index along x / longitude"))
    obs("cell_j", Int32, _attrib("cell index along y / latitude"))
    stream.has_panel && obs("cell_panel", Int32, _attrib("cubed-sphere panel (file convention)"))
    obs("cell_area", Float64, _attrib("m2", "containing cell area"))
    obs("sample_time_prev", Float64, _time_attrib("met-window end before the observation (first sample)"))
    obs("sample_time_next", Float64, _time_attrib("met-window end after the observation (second sample)"))
    obs("interp_weight", Float32, _attrib("weight of sample_time_next in the linear blend; NaN for nearest-window rows"))
    obs("interp_flag", Int8, Dict{String, Any}(
        "long_name" => "0 = bracketed by two window ends, 1 = one-sided (single sample), 2 = nearest window end",
        "flag_values" => collect(Int8, values(INTERP_FLAG_CODES)), "flag_meanings" => _INTERP_FLAG_MEANINGS))
    obs("ps" * suffix, Float32, _attrib("Pa", word * "surface pressure (p_half[end])"))
    obs("intake_level", Int32, _attrib("1-based model layer containing the intake (nlevel = surface layer)"))
    obs("intake_layer_bottom_agl", Float32, _attrib("m", "bottom of the intake layer above the model surface"))
    obs("intake_layer_top_agl", Float32, _attrib("m", "top of the intake layer above the model surface"))
    obs("height_method", Int8, _attrib("temperature source for layer heights (see height_method_codes)"))
    _def_obs_var(ds, "p_half" * suffix, Float32, ("ilev", "obs"); deflate_level = dl,
                 chunks = (nlevel + 1, _OBSERVATION_CHUNK), attrib = _attrib("Pa", word * "interface pressure, top down"))
    _def_obs_var(ds, "air_mass_per_area" * suffix, Float32, ("lev", "obs"); deflate_level = dl,
                 chunks = (nlevel, _OBSERVATION_CHUNK), attrib = _attrib("kg m-2", word * "air mass per unit area per layer"))
    for name in stream.tracer_names
        _def_obs_var(ds, String(name), Float32, ("lev", "obs"); deflate_level = dl,
                     chunks = (nlevel, _OBSERVATION_CHUNK), attrib = _attrib(units, "$(name) mixing ratio profile"))
        obs(String(name) * "_column_mean", Float32, _attrib(units, "$(name) air-mass-weighted column mean mixing ratio"))
        obs(String(name) * "_intake", Float32, _attrib(units, "$(name) mixing ratio in the intake layer"))
    end
    return nothing
end

"""
    append_soundings!(stream, batch) -> count

Queue `batch` after the rows already reserved, then flush if the NetCDF lock
is free; each flushed batch is synced before `completed_soundings` is
published. Returns the new reserved row count.
"""
function append_soundings!(stream::SoundingNetCDFStream, batch::SoundingBatch)
    _check_open(stream, "soundings")
    n = length(batch)
    n == 0 && return stream.count
    size(batch.p_half, 1) == stream.nlevel + 1 || throw(DimensionMismatch(
        "batch has $(size(batch.p_half, 1) - 1) levels; the stream has $(stream.nlevel)"))
    size(batch.tracers, 2) == length(stream.tracer_names) || throw(DimensionMismatch(
        "batch has $(size(batch.tracers, 2)) tracers; the stream has $(length(stream.tracer_names))"))
    rows = stream.count + 1:stream.count + n
    stream.count += n
    _submit!(stream, st -> _write_soundings!(st, batch, rows), "soundings")
    return stream.count
end

function _write_soundings!(stream::SoundingNetCDFStream, batch::SoundingBatch, rows::UnitRange{Int})
    ds = stream.dataset
    suffix = _basis_suffix(stream.mass_basis)
    cells = batch.cells
    ds["id"][rows] = batch.ids
    ds["source"][rows] = batch.sources
    ds["time"][rows] = batch.times
    ds["latitude"][rows] = batch.latitudes
    ds["longitude"][rows] = batch.longitudes
    ds["elevation"][rows] = batch.elevations
    ds["intake_height"][rows] = batch.intake_heights
    ds["cell_lat"][rows] = [c.lat for c in cells]
    ds["cell_lon"][rows] = [c.lon for c in cells]
    ds["cell_index"][rows] = Int32[c.cell for c in cells]
    ds["cell_i"][rows] = Int32[c.i for c in cells]
    ds["cell_j"][rows] = Int32[c.j for c in cells]
    stream.has_panel && (ds["cell_panel"][rows] = Int32[c.panel for c in cells])
    ds["cell_area"][rows] = [c.area for c in cells]
    ds["sample_time_prev"][rows] = batch.sample_time_prev
    ds["sample_time_next"][rows] = batch.sample_time_next
    ds["interp_weight"][rows] = batch.weights
    ds["interp_flag"][rows] = batch.flags
    ds["ps" * suffix][rows] = Float32.(view(batch.p_half, stream.nlevel + 1, :))
    ds["intake_level"][rows] = batch.intake_levels
    ds["intake_layer_bottom_agl"][rows] = Float32.(batch.layer_bottom)
    ds["intake_layer_top_agl"][rows] = Float32.(batch.layer_top)
    ds["height_method"][rows] = batch.height_methods
    ds["p_half" * suffix][:, rows] = Float32.(batch.p_half)
    ds["air_mass_per_area" * suffix][:, rows] = Float32.(batch.air_mass_per_area)
    for (t, name) in enumerate(stream.tracer_names)
        ds[String(name)][:, rows] = Float32.(view(batch.tracers, :, t, :))
        ds[String(name) * "_column_mean"][rows] = Float32.(view(batch.column_means, t, :))
        ds[String(name) * "_intake"][rows] = Float32.(view(batch.intake_values, t, :))
    end
    NCDatasets.sync(ds)
    ds.attrib["completed_soundings"] = last(rows)
    NCDatasets.sync(ds)
    return nothing
end

# -- station series (sites) -------------------------------------------------

"""
    SiteNetCDFStream(path, mesh, nlevel, tracer_names, sites, cells; mass_basis, origin, deflate_level, write_profiles, attributes)

Append-only station sink: one record per met-window end with, per site, the
mixing ratio in the layer containing the intake height (`<tracer>_intake`),
the lowest-layer value (`<tracer>_surface`), the chosen layer and its bounds,
surface pressure, and optionally the full profiles (`<tracer>`, `p_half`,
air mass). Sites outside their [`TimeRange`](@ref) are written as NaN with
`intake_level = 0`.
"""
mutable struct SiteNetCDFStream <: AbstractObservationStream
    path::String
    dataset::Union{Nothing, NCDataset}
    count::Int
    nlevel::Int
    nsite::Int
    tracer_names::Vector{Symbol}
    mass_basis::AbstractMassBasis
    write_profiles::Bool
    closed::Bool
    failed::Bool
    queue::Vector{Any}
end

"""
    SiteRecord(time, nsite, nlevel, ntracer; profiles)

One time record for every site, allocated uninitialised for the sampler to
fill. `values` (intake layer) and `surface_values` are `(nsite, ntracer)`;
with `profiles = true` the record also carries `p_half (nlevel + 1, nsite)`,
`air_mass_per_area (nlevel, nsite)` and `profiles (nlevel, nsite, ntracer)`,
otherwise those arrays are empty.
"""
struct SiteRecord
    time::Float64
    ps::Vector{Float64}
    intake_level::Vector{Int32}
    layer_bottom::Vector{Float64}
    layer_top::Vector{Float64}
    height_method::Vector{Int8}
    values::Matrix{Float64}
    surface_values::Matrix{Float64}
    p_half::Matrix{Float64}
    air_mass_per_area::Matrix{Float64}
    profiles::Array{Float64, 3}
end

function SiteRecord(time::Real, nsite::Integer, nlevel::Integer, ntracer::Integer; profiles::Bool)
    return SiteRecord(Float64(time),
                      Vector{Float64}(undef, nsite), Vector{Int32}(undef, nsite),
                      Vector{Float64}(undef, nsite), Vector{Float64}(undef, nsite),
                      Vector{Int8}(undef, nsite),
                      Matrix{Float64}(undef, nsite, ntracer), Matrix{Float64}(undef, nsite, ntracer),
                      profiles ? Matrix{Float64}(undef, nlevel + 1, nsite) : Matrix{Float64}(undef, 0, 0),
                      profiles ? Matrix{Float64}(undef, nlevel, nsite) : Matrix{Float64}(undef, 0, 0),
                      profiles ? Array{Float64, 3}(undef, nlevel, nsite, ntracer) : Array{Float64, 3}(undef, 0, 0, 0))
end

function SiteNetCDFStream(path::AbstractString, mesh, nlevel::Integer,
                          tracer_names::AbstractVector{Symbol},
                          sites::AbstractVector{SiteRequest}, cells::AbstractVector{CellLocation};
                          mass_basis::Union{Symbol, AbstractMassBasis}, origin::DateTime,
                          deflate_level::Integer = 0,
                          write_profiles::Bool = false,
                          attributes::AbstractDict = Dict{String, Any}())
    nlevel >= 1 || throw(ArgumentError("site stream needs at least one level"))
    isempty(tracer_names) && throw(ArgumentError("site stream needs at least one tracer"))
    length(sites) == length(cells) || throw(DimensionMismatch("one cell per site is required"))
    stream = SiteNetCDFStream(String(path), nothing, 0, Int(nlevel), length(sites),
                              collect(tracer_names), _as_mass_basis(mass_basis), write_profiles,
                              false, false, Any[])
    attrs = copy(attributes)
    site_list = collect(sites)
    cell_list = collect(cells)
    _submit!(stream, _create_op(ds -> _define_site_schema!(ds, stream, mesh, site_list, cell_list,
                                                          origin, Int(deflate_level), attrs)), "schema")
    return stream
end

_schedule_bounds(::AbstractSiteSchedule) = (NaN, NaN)
_schedule_bounds(range::TimeRange) = (range.start_seconds, range.stop_seconds)

function _define_site_schema!(ds, stream::SiteNetCDFStream, mesh, sites, cells, origin::DateTime,
                              dl::Int, attributes)
    nlevel, nsite = stream.nlevel, stream.nsite
    suffix = _basis_suffix(stream.mass_basis)
    word = _basis_word(stream.mass_basis)
    units = _tracer_units(_basis_symbol(stream.mass_basis))
    defDim(ds, "site", nsite)
    defDim(ds, "time", Inf)
    if stream.write_profiles
        defDim(ds, "lev", nlevel)
        defDim(ds, "ilev", nlevel + 1)
    end
    _observation_common_attributes!(ds, mesh, stream.mass_basis, origin, attributes)
    ds.attrib["completed_times"] = 0
    ds.attrib["tracers"] = join(String.(stream.tracer_names), " ")
    # Static site metadata.
    defVar(ds, "site_id", String, ("site",))[:] = [s.id for s in sites]
    defVar(ds, "source", Int32, ("site",); attrib = _attrib("1-based index into [[output.observations.sources]]"))[:] =
        Int32[s.source for s in sites]
    static(name, values, attrib) = (_def_obs_var(ds, name, Float64, ("site",); deflate_level = 0, attrib)[:] = values)
    static("latitude", [s.lat for s in sites], Dict{String, Any}("units" => "degrees_north", "standard_name" => "latitude"))
    static("longitude", [s.lon for s in sites], Dict{String, Any}("units" => "degrees_east", "standard_name" => "longitude"))
    static("elevation", [s.elevation_m for s in sites], _attrib("m", "site surface elevation above sea level (NaN unknown)"))
    static("intake_height", [s.intake_height_m for s in sites], _attrib("m", "intake height above ground (NaN: lowest layer)"))
    origin_unix = datetime2unix(origin)
    static("schedule_start", [origin_unix + _schedule_bounds(s.schedule)[1] for s in sites],
           _time_attrib("start of the site's sampling range (NaN: every window end)"))
    static("schedule_end", [origin_unix + _schedule_bounds(s.schedule)[2] for s in sites],
           _time_attrib("end of the site's sampling range (NaN: every window end)"))
    static("cell_lat", [c.lat for c in cells], _attrib("degrees_north", "containing cell centre latitude"))
    static("cell_lon", [c.lon for c in cells], _attrib("degrees_east", "containing cell centre longitude"))
    static("cell_area", [c.area for c in cells], _attrib("m2", "containing cell area"))
    defVar(ds, "cell_index", Int32, ("site",))[:] = Int32[c.cell for c in cells]
    defVar(ds, "cell_i", Int32, ("site",))[:] = Int32[c.i for c in cells]
    defVar(ds, "cell_j", Int32, ("site",))[:] = Int32[c.j for c in cells]
    _has_panel(mesh) && (defVar(ds, "cell_panel", Int32, ("site",))[:] = Int32[c.panel for c in cells])
    # Per-record variables.
    c2 = (max(nsite, 1), 1024)
    c3(n) = (n, max(nsite, 1), 64)
    rec(name, T, attrib) = _def_obs_var(ds, name, T, ("site", "time"); deflate_level = dl, chunks = c2, attrib)
    _def_obs_var(ds, "time", Float64, ("time",); deflate_level = dl, chunks = (1024,), attrib = _time_attrib("met-window end"))
    rec("ps" * suffix, Float32, _attrib("Pa", word * "surface pressure"))
    rec("intake_level", Int32, _attrib("1-based model layer containing the intake (nlevel = surface layer; 0 = not sampled)"))
    rec("intake_layer_bottom_agl", Float32, _attrib("m", "bottom of the intake layer above the model surface"))
    rec("intake_layer_top_agl", Float32, _attrib("m", "top of the intake layer above the model surface"))
    rec("height_method", Int8, _attrib("temperature source for layer heights (see height_method_codes)"))
    for name in stream.tracer_names
        rec(String(name) * "_intake", Float32, _attrib(units, "$(name) mixing ratio in the intake layer"))
        rec(String(name) * "_surface", Float32, _attrib(units, "$(name) mixing ratio in the lowest model layer"))
        stream.write_profiles && _def_obs_var(ds, String(name), Float32, ("lev", "site", "time");
                                              deflate_level = dl, chunks = c3(nlevel),
                                              attrib = _attrib(units, "$(name) mixing ratio profile"))
    end
    if stream.write_profiles
        _def_obs_var(ds, "p_half" * suffix, Float32, ("ilev", "site", "time"); deflate_level = dl,
                     chunks = c3(nlevel + 1), attrib = _attrib("Pa", word * "interface pressure, top down"))
        _def_obs_var(ds, "air_mass_per_area" * suffix, Float32, ("lev", "site", "time"); deflate_level = dl,
                     chunks = c3(nlevel), attrib = _attrib("kg m-2", word * "air mass per unit area per layer"))
    end
    return nothing
end

"""
    append_site_record!(stream, record) -> count

Queue one met-window-end record for every site, then flush if the NetCDF
lock is free; each flushed record is synced before `completed_times` is
published. Returns the new reserved record count.
"""
function append_site_record!(stream::SiteNetCDFStream, record::SiteRecord)
    _check_open(stream, "site record")
    nsite, nlevel = stream.nsite, stream.nlevel
    length(record.ps) == nsite || throw(DimensionMismatch("record has $(length(record.ps)) sites; stream has $(nsite)"))
    size(record.values) == (nsite, length(stream.tracer_names)) || throw(DimensionMismatch("values must be (nsite, ntracer)"))
    if stream.write_profiles
        size(record.profiles) == (nlevel, nsite, length(stream.tracer_names)) ||
            throw(DimensionMismatch("profiles must be (nlevel, nsite, ntracer)"))
        size(record.p_half) == (nlevel + 1, nsite) || throw(DimensionMismatch("p_half must be (nlevel+1, nsite)"))
    end
    stream.count += 1
    r = stream.count
    _submit!(stream, st -> _write_site_record!(st, record, r), "site record")
    return r
end

function _write_site_record!(stream::SiteNetCDFStream, record::SiteRecord, r::Int)
    ds = stream.dataset
    suffix = _basis_suffix(stream.mass_basis)
    ds["time"][r] = record.time
    ds["ps" * suffix][:, r] = Float32.(record.ps)
    ds["intake_level"][:, r] = record.intake_level
    ds["intake_layer_bottom_agl"][:, r] = Float32.(record.layer_bottom)
    ds["intake_layer_top_agl"][:, r] = Float32.(record.layer_top)
    ds["height_method"][:, r] = record.height_method
    for (t, name) in enumerate(stream.tracer_names)
        ds[String(name) * "_intake"][:, r] = Float32.(view(record.values, :, t))
        ds[String(name) * "_surface"][:, r] = Float32.(view(record.surface_values, :, t))
        stream.write_profiles && (ds[String(name)][:, :, r] = Float32.(view(record.profiles, :, :, t)))
    end
    if stream.write_profiles
        ds["p_half" * suffix][:, :, r] = Float32.(record.p_half)
        ds["air_mass_per_area" * suffix][:, :, r] = Float32.(record.air_mass_per_area)
    end
    NCDatasets.sync(ds)
    ds.attrib["completed_times"] = r
    NCDatasets.sync(ds)
    return nothing
end
