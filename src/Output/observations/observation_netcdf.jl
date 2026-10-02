# ---------------------------------------------------------------------------
# Append-only NetCDF sinks for observation sampling: one file for soundings
# (unlimited `obs`), one for sites (fixed `site`, unlimited `time`).
#
# Both follow the snapshot stream contract: every append is flushed with
# `sync`, a `completed_*` attribute records the last successful append, a
# failed append poisons the stream, and `close` is idempotent. Files are
# created when the stream is opened so a run with no matches still leaves a
# valid, empty file.
# ---------------------------------------------------------------------------

const _OBSERVATION_CONTRACT = "AtmosTransport observations v1"
const _OBSERVATION_TIME_UNITS = "seconds since 1970-01-01 00:00:00"
const _OBSERVATION_CHUNK = 4096

"""
    AbstractObservationStream

Append-only NetCDF sink with the snapshot-stream contract: `dataset`, `count`,
`closed`, `failed` fields; `close` is idempotent and a failed append poisons
the stream.
"""
abstract type AbstractObservationStream end

function _check_open(stream::AbstractObservationStream, what::AbstractString)
    stream.failed && throw(ArgumentError("cannot append $(what) after a NetCDF write failure"))
    stream.closed && throw(ArgumentError("cannot append $(what) to a closed stream"))
    return nothing
end

function _poison!(stream::AbstractObservationStream)
    stream.failed = true
    try
        close(stream)
    catch
        # Preserve the original write error when cleanup also fails.
    end
    return nothing
end

function Base.close(stream::AbstractObservationStream)
    stream.closed && return nothing
    try
        ds = stream.dataset
        ds === nothing || lock(_NETCDF_IO_LOCK) do
            close(ds)
        end
    finally
        stream.dataset = nothing
        stream.closed = true
    end
    return nothing
end

# Summary attributes written once at the end of a successful run.
function write_summary_attributes!(stream::AbstractObservationStream, attributes::AbstractDict)
    (stream.closed || stream.failed || stream.dataset === nothing) && return nothing
    lock(_NETCDF_IO_LOCK) do
        ds = stream.dataset
        for (key, value) in pairs(attributes)
            ds.attrib[String(key)] = value
        end
        NCDatasets.sync(ds)
    end
    return nothing
end

"Variable-name suffix and long-name wording for the state's mass basis."
_basis_suffix(mass_basis::Symbol) = mass_basis === :dry ? "_dry" : ""
_basis_word(mass_basis::Symbol) = mass_basis === :dry ? "dry " : ""

function _observation_common_attributes!(ds, mesh, mass_basis::Symbol, origin::DateTime,
                                         attributes::AbstractDict)
    ds.attrib["Conventions"] = "CF-1.8"
    ds.attrib["title"] = "AtmosTransport observation sampling"
    ds.attrib["source"] = "AtmosTransport.jl"
    ds.attrib["institution"] = get(ENV, "ATMOSTR_INSTITUTION", "Caltech / Frankenberg group")
    ds.attrib["grid"] = summary(mesh)
    ds.attrib["grid_type"] = _grid_type_string(mesh)
    ds.attrib["mass_basis"] = String(mass_basis)
    ds.attrib["output_contract"] = _OBSERVATION_CONTRACT
    ds.attrib["run_time_origin"] = Dates.format(origin, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
    ds.attrib["horizontal_sampling"] = "containing_cell"
    ds.attrib["vertical_ordering"] = "lev[1] is the top of the atmosphere; lev[end] is the surface"
    ds.attrib["pressure_reconstruction"] =
        "p_half[1] = A_ifc[1]; p_half[k+1] = p_half[k] + g * air_mass[k] / cell_area " *
        "(" * _basis_word(mass_basis) * "air)"
    ds.attrib["creation_date"] = _iso8601_utc_now()
    ds.attrib["framework"] = "AtmosTransport.jl"
    ds.attrib["framework_commit"] = _git_commit_sha()
    ds.attrib["framework_dirty"] = _git_dirty_flag()
    ds.attrib["runtime"] = _runtime_environment_string()
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

# -- soundings --------------------------------------------------------------

"""
    SoundingNetCDFStream(path, mesh, nlevel, tracer_names; mass_basis, origin, deflate_level, attributes)

Append-only sounding sink. Each appended batch writes one row per sounding:
identity, time, location, containing cell, bracketing sample times and
weight, interface pressures, per-layer air mass per area, and each tracer's
mixing-ratio profile and column mean.
"""
mutable struct SoundingNetCDFStream <: AbstractObservationStream
    path::String
    dataset::Union{Nothing, NCDataset}
    count::Int
    nlevel::Int
    tracer_names::Vector{Symbol}
    mass_basis::Symbol
    has_panel::Bool
    closed::Bool
    failed::Bool
end

"""
    SoundingBatch

Columns of one emitted batch, `n` soundings. Profiles are `(nlevel, n)`,
`p_half` is `(nlevel + 1, n)`, tracers are `(nlevel, ntracer, n)` and column
means `(ntracer, n)`. Times are seconds since 1970-01-01 UTC.
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

    function SoundingBatch(ids, sources, times, latitudes, longitudes, cells, sample_time_prev,
                           sample_time_next, weights, flags, p_half, air_mass_per_area, tracers,
                           column_means)
        n = length(ids)
        for (name, v) in (("sources", sources), ("times", times), ("latitudes", latitudes),
                          ("longitudes", longitudes), ("cells", cells),
                          ("sample_time_prev", sample_time_prev), ("sample_time_next", sample_time_next),
                          ("weights", weights), ("flags", flags))
            length(v) == n || throw(DimensionMismatch("SoundingBatch.$(name) has $(length(v)) rows; expected $(n)"))
        end
        nlevel = size(air_mass_per_area, 1)
        size(p_half) == (nlevel + 1, n) || throw(DimensionMismatch("SoundingBatch.p_half must be (nlevel + 1, n)"))
        size(air_mass_per_area, 2) == n || throw(DimensionMismatch("SoundingBatch.air_mass_per_area must be (nlevel, n)"))
        size(tracers, 1) == nlevel && size(tracers, 3) == n ||
            throw(DimensionMismatch("SoundingBatch.tracers must be (nlevel, ntracer, n)"))
        size(column_means) == (size(tracers, 2), n) ||
            throw(DimensionMismatch("SoundingBatch.column_means must be (ntracer, n)"))
        return new(ids, sources, times, latitudes, longitudes, cells, sample_time_prev, sample_time_next,
                   weights, flags, p_half, air_mass_per_area, tracers, column_means)
    end
end

Base.length(batch::SoundingBatch) = length(batch.ids)

function SoundingNetCDFStream(path::AbstractString, mesh, nlevel::Integer,
                              tracer_names::AbstractVector{Symbol};
                              mass_basis::Symbol, origin::DateTime, deflate_level::Integer = 0,
                              attributes::AbstractDict = Dict{String, Any}())
    nlevel >= 1 || throw(ArgumentError("sounding stream needs at least one level"))
    isempty(tracer_names) && throw(ArgumentError("sounding stream needs at least one tracer"))
    has_panel = mesh isa CubedSphereMesh
    stream = SoundingNetCDFStream(String(path), nothing, 0, Int(nlevel), collect(tracer_names),
                                  mass_basis, has_panel, false, false)
    lock(_NETCDF_IO_LOCK) do
        _ensure_parent_dir(stream.path)
        ds = _create_netcdf_dataset(stream.path)
        try
            _define_sounding_schema!(ds, stream, mesh, origin, Int(deflate_level), attributes)
            NCDatasets.sync(ds)
        catch
            close(ds)
            rethrow()
        end
        stream.dataset = ds
    end
    return stream
end

function _define_sounding_schema!(ds, stream::SoundingNetCDFStream, mesh, origin::DateTime,
                                  deflate_level::Int, attributes)
    nlevel = stream.nlevel
    suffix = _basis_suffix(stream.mass_basis)
    word = _basis_word(stream.mass_basis)
    defDim(ds, "obs", Inf)
    defDim(ds, "lev", nlevel)
    defDim(ds, "ilev", nlevel + 1)
    _observation_common_attributes!(ds, mesh, stream.mass_basis, origin, attributes)
    ds.attrib["completed_soundings"] = 0
    ds.attrib["tracers"] = join(String.(stream.tracer_names), " ")
    dl = deflate_level
    chunk1 = (_OBSERVATION_CHUNK,)
    _def_obs_var(ds, "id", String, ("obs",); deflate_level = 0,
                 attrib = Dict{String, Any}("long_name" => "observation identifier (sounding_id, obspack_id, or table id)"))
    _def_obs_var(ds, "source", Int32, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("long_name" => "1-based index into [[output.observations.sources]]"))
    _def_obs_var(ds, "time", Float64, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = _time_attrib("observation time"))
    _def_obs_var(ds, "latitude", Float64, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("units" => "degrees_north", "standard_name" => "latitude"))
    _def_obs_var(ds, "longitude", Float64, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("units" => "degrees_east", "standard_name" => "longitude"))
    _def_obs_var(ds, "cell_lat", Float64, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("units" => "degrees_north", "long_name" => "containing cell centre latitude"))
    _def_obs_var(ds, "cell_lon", Float64, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("units" => "degrees_east", "long_name" => "containing cell centre longitude (mesh convention)"))
    _def_obs_var(ds, "cell_index", Int32, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("long_name" => "native flat cell index within the panel / mesh"))
    _def_obs_var(ds, "cell_i", Int32, ("obs",); deflate_level = dl, chunks = chunk1)
    _def_obs_var(ds, "cell_j", Int32, ("obs",); deflate_level = dl, chunks = chunk1)
    stream.has_panel && _def_obs_var(ds, "cell_panel", Int32, ("obs",); deflate_level = dl, chunks = chunk1,
                                     attrib = Dict{String, Any}("long_name" => "cubed-sphere panel (file convention)"))
    _def_obs_var(ds, "cell_area", Float64, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("units" => "m2"))
    _def_obs_var(ds, "sample_time_prev", Float64, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = _time_attrib("met-window end before the observation (first sample)"))
    _def_obs_var(ds, "sample_time_next", Float64, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = _time_attrib("met-window end after the observation (second sample)"))
    _def_obs_var(ds, "interp_weight", Float32, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("long_name" => "weight of sample_time_next in the linear blend; NaN for nearest-window sampling"))
    _def_obs_var(ds, "interp_flag", Int8, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("long_name" => "0 = bracketed by two window ends, 1 = one-sided (single sample)",
                                            "flag_values" => Int8[0, 1], "flag_meanings" => "bracketed one_sided"))
    _def_obs_var(ds, "ps" * suffix, Float32, ("obs",); deflate_level = dl, chunks = chunk1,
                 attrib = Dict{String, Any}("units" => "Pa", "long_name" => word * "surface pressure (p_half[end])"))
    _def_obs_var(ds, "p_half" * suffix, Float32, ("ilev", "obs"); deflate_level = dl,
                 chunks = (nlevel + 1, _OBSERVATION_CHUNK),
                 attrib = Dict{String, Any}("units" => "Pa", "long_name" => word * "interface pressure, top down"))
    _def_obs_var(ds, "air_mass_per_area" * suffix, Float32, ("lev", "obs"); deflate_level = dl,
                 chunks = (nlevel, _OBSERVATION_CHUNK),
                 attrib = Dict{String, Any}("units" => "kg m-2", "long_name" => word * "air mass per unit area per layer"))
    for name in stream.tracer_names
        _def_obs_var(ds, String(name), Float32, ("lev", "obs"); deflate_level = dl,
                     chunks = (nlevel, _OBSERVATION_CHUNK),
                     attrib = Dict{String, Any}("units" => _tracer_units(stream.mass_basis),
                                                "long_name" => "$(name) mixing ratio profile"))
        _def_obs_var(ds, String(name) * "_column_mean", Float32, ("obs",); deflate_level = dl, chunks = chunk1,
                     attrib = Dict{String, Any}("units" => _tracer_units(stream.mass_basis),
                                                "long_name" => "$(name) air-mass-weighted column mean mixing ratio"))
    end
    return nothing
end

"""
    append_soundings!(stream, batch) -> count

Write `batch` after the rows already in the file, flush, then publish
`completed_soundings`. Returns the new row count.
"""
function append_soundings!(stream::SoundingNetCDFStream, batch::SoundingBatch)
    _check_open(stream, "soundings")
    n = length(batch)
    n == 0 && return stream.count
    size(batch.p_half, 1) == stream.nlevel + 1 || throw(DimensionMismatch(
        "batch has $(size(batch.p_half, 1) - 1) levels; the stream has $(stream.nlevel)"))
    size(batch.tracers, 2) == length(stream.tracer_names) || throw(DimensionMismatch(
        "batch has $(size(batch.tracers, 2)) tracers; the stream has $(length(stream.tracer_names))"))
    suffix = _basis_suffix(stream.mass_basis)
    rows = stream.count + 1:stream.count + n
    lock(_NETCDF_IO_LOCK) do
        try
            ds = stream.dataset
            ds["id"][rows] = batch.ids
            ds["source"][rows] = batch.sources
            ds["time"][rows] = batch.times
            ds["latitude"][rows] = batch.latitudes
            ds["longitude"][rows] = batch.longitudes
            ds["cell_lat"][rows] = [c.lat for c in batch.cells]
            ds["cell_lon"][rows] = [c.lon for c in batch.cells]
            ds["cell_index"][rows] = Int32[c.cell for c in batch.cells]
            ds["cell_i"][rows] = Int32[c.i for c in batch.cells]
            ds["cell_j"][rows] = Int32[c.j for c in batch.cells]
            stream.has_panel && (ds["cell_panel"][rows] = Int32[c.panel for c in batch.cells])
            ds["cell_area"][rows] = [c.area for c in batch.cells]
            ds["sample_time_prev"][rows] = batch.sample_time_prev
            ds["sample_time_next"][rows] = batch.sample_time_next
            ds["interp_weight"][rows] = batch.weights
            ds["interp_flag"][rows] = batch.flags
            ds["ps" * suffix][rows] = Float32.(view(batch.p_half, stream.nlevel + 1, :))
            ds["p_half" * suffix][:, rows] = Float32.(batch.p_half)
            ds["air_mass_per_area" * suffix][:, rows] = Float32.(batch.air_mass_per_area)
            for (t, name) in enumerate(stream.tracer_names)
                ds[String(name)][:, rows] = Float32.(view(batch.tracers, :, t, :))
                ds[String(name) * "_column_mean"][rows] = Float32.(view(batch.column_means, t, :))
            end
            NCDatasets.sync(ds)
            ds.attrib["completed_soundings"] = stream.count + n
            NCDatasets.sync(ds)
        catch
            _poison!(stream)
            rethrow()
        end
    end
    stream.count += n
    return stream.count
end

# -- sites ------------------------------------------------------------------

"""
    SiteNetCDFStream(path, mesh, nlevel, tracer_names, sites, cells; mass_basis, origin, deflate_level, write_profiles, attributes)

Append-only site sink: one record per met-window end with, per site, the
mixing ratio in the layer containing the intake height, the lowest-layer
value, the chosen layer and its bounds, surface pressure, and optionally
the full profiles.
"""
mutable struct SiteNetCDFStream <: AbstractObservationStream
    path::String
    dataset::Union{Nothing, NCDataset}
    count::Int
    nlevel::Int
    nsite::Int
    tracer_names::Vector{Symbol}
    mass_basis::Symbol
    write_profiles::Bool
    closed::Bool
    failed::Bool
end

"""
    SiteRecord(time, nsite, nlevel, ntracer; profiles)

One time record for every site, allocated uninitialised for the sampler to
fill. `values` and `surface_values` are `(nsite, ntracer)`; with
`profiles = true` the record also carries `p_half (nlevel + 1, nsite)`,
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
                          mass_basis::Symbol, origin::DateTime, deflate_level::Integer = 0,
                          write_profiles::Bool = false,
                          attributes::AbstractDict = Dict{String, Any}())
    nlevel >= 1 || throw(ArgumentError("site stream needs at least one level"))
    isempty(tracer_names) && throw(ArgumentError("site stream needs at least one tracer"))
    length(sites) == length(cells) || throw(DimensionMismatch("one cell per site is required"))
    stream = SiteNetCDFStream(String(path), nothing, 0, Int(nlevel), length(sites),
                              collect(tracer_names), mass_basis, write_profiles, false, false)
    lock(_NETCDF_IO_LOCK) do
        _ensure_parent_dir(stream.path)
        ds = _create_netcdf_dataset(stream.path)
        try
            _define_site_schema!(ds, stream, mesh, sites, cells, origin, Int(deflate_level), attributes)
            NCDatasets.sync(ds)
        catch
            close(ds)
            rethrow()
        end
        stream.dataset = ds
    end
    return stream
end

function _define_site_schema!(ds, stream::SiteNetCDFStream, mesh, sites, cells, origin::DateTime,
                              deflate_level::Int, attributes)
    nlevel, nsite = stream.nlevel, stream.nsite
    suffix = _basis_suffix(stream.mass_basis)
    word = _basis_word(stream.mass_basis)
    defDim(ds, "site", nsite)
    defDim(ds, "time", Inf)
    if stream.write_profiles
        defDim(ds, "lev", nlevel)
        defDim(ds, "ilev", nlevel + 1)
    end
    _observation_common_attributes!(ds, mesh, stream.mass_basis, origin, attributes)
    ds.attrib["completed_times"] = 0
    ds.attrib["tracers"] = join(String.(stream.tracer_names), " ")
    ds.attrib["height_method_codes"] = height_method_codes_description()
    dl = deflate_level
    # Static site metadata.
    defVar(ds, "site_id", String, ("site",))[:] = [s.id for s in sites]
    defVar(ds, "source", Int32, ("site",);
           attrib = Dict{String, Any}("long_name" => "1-based index into [[output.observations.sources]]"))[:] = Int32[s.source for s in sites]
    _static(name, values, attrib) = (_def_obs_var(ds, name, Float64, ("site",); deflate_level = 0, attrib = attrib)[:] = values)
    _static("latitude", [s.lat for s in sites], Dict{String, Any}("units" => "degrees_north", "standard_name" => "latitude"))
    _static("longitude", [s.lon for s in sites], Dict{String, Any}("units" => "degrees_east", "standard_name" => "longitude"))
    _static("elevation", [s.elevation_m for s in sites], Dict{String, Any}("units" => "m", "long_name" => "site surface elevation above sea level (NaN unknown)"))
    _static("intake_height", [s.intake_height_m for s in sites], Dict{String, Any}("units" => "m", "long_name" => "intake height above ground (NaN: lowest layer)"))
    _static("cell_lat", [c.lat for c in cells], Dict{String, Any}("units" => "degrees_north"))
    _static("cell_lon", [c.lon for c in cells], Dict{String, Any}("units" => "degrees_east"))
    _static("cell_area", [c.area for c in cells], Dict{String, Any}("units" => "m2"))
    defVar(ds, "cell_index", Int32, ("site",))[:] = Int32[c.cell for c in cells]
    defVar(ds, "cell_i", Int32, ("site",))[:] = Int32[c.i for c in cells]
    defVar(ds, "cell_j", Int32, ("site",))[:] = Int32[c.j for c in cells]
    mesh isa CubedSphereMesh && (defVar(ds, "cell_panel", Int32, ("site",))[:] = Int32[c.panel for c in cells])
    # Per-record variables.
    chunk2 = (max(nsite, 1), 1024)
    _def_obs_var(ds, "time", Float64, ("time",); deflate_level = dl, chunks = (1024,),
                 attrib = _time_attrib("met-window end"))
    _def_obs_var(ds, "ps" * suffix, Float32, ("site", "time"); deflate_level = dl, chunks = chunk2,
                 attrib = Dict{String, Any}("units" => "Pa", "long_name" => word * "surface pressure"))
    _def_obs_var(ds, "intake_level", Int32, ("site", "time"); deflate_level = dl, chunks = chunk2,
                 attrib = Dict{String, Any}("long_name" => "1-based model layer containing the intake (nlevel = surface layer)"))
    _def_obs_var(ds, "intake_layer_bottom_agl", Float32, ("site", "time"); deflate_level = dl, chunks = chunk2,
                 attrib = Dict{String, Any}("units" => "m", "long_name" => "bottom of the intake layer above ground"))
    _def_obs_var(ds, "intake_layer_top_agl", Float32, ("site", "time"); deflate_level = dl, chunks = chunk2,
                 attrib = Dict{String, Any}("units" => "m", "long_name" => "top of the intake layer above ground"))
    _def_obs_var(ds, "height_method", Int8, ("site", "time"); deflate_level = dl, chunks = chunk2,
                 attrib = Dict{String, Any}("long_name" => "temperature source for layer heights (see height_method_codes)"))
    for name in stream.tracer_names
        _def_obs_var(ds, String(name), Float32, ("site", "time"); deflate_level = dl, chunks = chunk2,
                     attrib = Dict{String, Any}("units" => _tracer_units(stream.mass_basis),
                                                "long_name" => "$(name) mixing ratio in the intake layer"))
        _def_obs_var(ds, String(name) * "_surface", Float32, ("site", "time"); deflate_level = dl, chunks = chunk2,
                     attrib = Dict{String, Any}("units" => _tracer_units(stream.mass_basis),
                                                "long_name" => "$(name) mixing ratio in the lowest model layer"))
        stream.write_profiles && _def_obs_var(ds, String(name) * "_profile", Float32, ("lev", "site", "time");
                                              deflate_level = dl, chunks = (nlevel, max(nsite, 1), 64),
                                              attrib = Dict{String, Any}("units" => _tracer_units(stream.mass_basis),
                                                                         "long_name" => "$(name) mixing ratio profile"))
    end
    if stream.write_profiles
        _def_obs_var(ds, "p_half" * suffix, Float32, ("ilev", "site", "time"); deflate_level = dl,
                     chunks = (nlevel + 1, max(nsite, 1), 64),
                     attrib = Dict{String, Any}("units" => "Pa", "long_name" => word * "interface pressure, top down"))
        _def_obs_var(ds, "air_mass_per_area" * suffix, Float32, ("lev", "site", "time"); deflate_level = dl,
                     chunks = (nlevel, max(nsite, 1), 64),
                     attrib = Dict{String, Any}("units" => "kg m-2", "long_name" => word * "air mass per unit area per layer"))
    end
    return nothing
end

"""
    append_site_record!(stream, record) -> count

Write one met-window-end record for every site, flush, then publish
`completed_times`.
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
    suffix = _basis_suffix(stream.mass_basis)
    r = stream.count + 1
    lock(_NETCDF_IO_LOCK) do
        try
            ds = stream.dataset
            ds["time"][r] = record.time
            ds["ps" * suffix][:, r] = Float32.(record.ps)
            ds["intake_level"][:, r] = record.intake_level
            ds["intake_layer_bottom_agl"][:, r] = Float32.(record.layer_bottom)
            ds["intake_layer_top_agl"][:, r] = Float32.(record.layer_top)
            ds["height_method"][:, r] = record.height_method
            for (t, name) in enumerate(stream.tracer_names)
                ds[String(name)][:, r] = Float32.(view(record.values, :, t))
                ds[String(name) * "_surface"][:, r] = Float32.(view(record.surface_values, :, t))
                stream.write_profiles && (ds[String(name) * "_profile"][:, :, r] = Float32.(view(record.profiles, :, :, t)))
            end
            if stream.write_profiles
                ds["p_half" * suffix][:, :, r] = Float32.(record.p_half)
                ds["air_mass_per_area" * suffix][:, :, r] = Float32.(record.air_mass_per_area)
            end
            NCDatasets.sync(ds)
            ds.attrib["completed_times"] = r
            NCDatasets.sync(ds)
        catch
            _poison!(stream)
            rethrow()
        end
    end
    stream.count = r
    return r
end

