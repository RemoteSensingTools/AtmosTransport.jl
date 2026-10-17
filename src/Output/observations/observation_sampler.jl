# ---------------------------------------------------------------------------
# Runtime sampler: gathers model columns at met-window ends and writes the
# sounding and site files. `NoObservationSampler` is the default that keeps
# every runner path identical to a run without `[output.observations]`.
# Runners call the interface once per met window, never per cell.
#
# Time semantics (linear interpolation): at window end `t_k` the sampler
# gathers every point event in [t_{k-1}, t_k + Δt_next). Events in
# [t_{k-1}, t_k) receive their second sample and are emitted as the linear
# blend of the two window-end states (masses are blended, then divided);
# events in [t_k, t_k + Δt_next) are held with their first sample. The first
# call at t = 0 on the initial state seeds the first bracket; events exactly
# at the final window end are emitted from that last sample at the finish.
# Station series are written at every window end their schedule allows.
# ---------------------------------------------------------------------------

abstract type AbstractObservationSampler end

"No-op sampler installed when observation output is absent or disabled."
struct NoObservationSampler <: AbstractObservationSampler end

"Whether the runner must visit every met-window end for this sampler."
samples_observations(::NoObservationSampler) = false
samples_observations(::AbstractObservationSampler) = true

"Soundings sampled at the previous window end, waiting for their second sample."
mutable struct PendingSoundings
    range::UnitRange{Int}        # indices into the sampler's time-sorted soundings
    t_prev::Float64              # window end (seconds) at which they were sampled
    air::Matrix{Float64}         # (nlevel, n) in request order
    tracers::Array{Float64, 3}   # (nlevel, ntracer, n) in request order
    temperature::Union{Nothing, Matrix{Float64}}   # (nlevel, n) layer temperature, if the run has one
    humidity::Union{Nothing, Matrix{Float64}}      # (nlevel, n) specific humidity, when pressure placement needs it
end

"Running totals reported in the summary attributes and the end-of-run log."
Base.@kwdef mutable struct ObservationCounters
    emitted::Int = 0
    dropped_before_start::Int = 0
    dropped_after_end::Int = 0
    one_sided::Int = 0
    unlocated_soundings::Int = 0
    unlocated_sites::Int = 0
    unplaced_soundings::Int = 0
    unplaced_sites::Int = 0
    site_records::Int = 0
end
Base.copy(c::ObservationCounters) =
    ObservationCounters(c.emitted, c.dropped_before_start, c.dropped_after_end, c.one_sided,
                        c.unlocated_soundings, c.unlocated_sites, c.unplaced_soundings,
                        c.unplaced_sites, c.site_records)

"""
    GatherPlan(cells, npanel)

Gather order for a batch of cell locations: columns grouped by panel (the
cubed sphere needs one contiguous run per panel), the per-panel ranges, and
`position[i]`, the gather slot of request `i`.
"""
struct GatherPlan
    columns::Vector{Int32}
    panel_ranges::Vector{UnitRange{Int}}
    position::Vector{Int}
end

# `column` selects the slab index: the halo-aware `c.column` for state arrays,
# or `c.cell` for interior-only fields. Grouping and positions do not depend on it.
function GatherPlan(cells::AbstractVector{CellLocation}, npanel::Int; column = c -> c.column)
    counts = zeros(Int, npanel)
    for c in cells
        counts[c.panel] += 1
    end
    starts = cumsum(counts) .- counts
    cursor = copy(starts)
    position = Vector{Int}(undef, length(cells))
    columns = Vector{Int32}(undef, length(cells))
    for (i, c) in enumerate(cells)
        cursor[c.panel] += 1
        position[i] = cursor[c.panel]
        columns[cursor[c.panel]] = Int32(column(c))
    end
    ranges = [starts[p] + 1:starts[p] + counts[p] for p in 1:npanel]
    return GatherPlan(columns, ranges, position)
end

"""
    ObservationSampler

Runtime state of observation sampling: the located and vertically placed
point events and sites, whether pressure placement needs the humidity,
device gather buffers, the held first samples of events awaiting their second
sample, the open (and retired daily) output streams, and counters. Built by
[`build_observation_sampler`](@ref).
"""
mutable struct ObservationSampler{M, L <: AbstractCellLocator,
                                  TI <: AbstractObservationTimeInterpolation,
                                  B <: ObservationGatherBuffers} <: AbstractObservationSampler
    spec::ObservationOutputSpec
    terrain_sources::NamedTuple{(:surface_height_file, :dem_file), Tuple{String, String}}
    origin::DateTime
    mesh::M
    locator::L
    time_interpolation::TI
    tracer_names::Vector{Symbol}
    nlevel::Int
    p_top::Float64
    gravity::Float64
    mass_basis::AbstractMassBasis
    npanel::Int
    soundings::Vector{SoundingRequest}      # located, time-sorted
    sounding_times::Vector{Float64}
    sounding_cells::Vector{CellLocation}
    sounding_placements::Vector{VerticalPlacement}
    sites::Vector{SiteRequest}
    site_cells::Vector{CellLocation}
    site_placements::Vector{VerticalPlacement}
    needs_humidity::Bool                    # pressure-placed requests on a dry basis: humidity at every sample
    buffers::B
    temperature_buffers::Union{Nothing, B}
    humidity_buffers::Union{Nothing, B}
    cursor::Int                             # first sounding not yet gathered as "new"
    pending::Union{Nothing, PendingSoundings}
    last_boundary::Float64                  # NaN before the first call
    soundings_stream::Union{Nothing, SoundingNetCDFStream}
    sites_stream::Union{Nothing, SiteNetCDFStream}
    retired::Vector{AbstractObservationStream}   # previous days' files still flushing
    opened_paths::Set{String}
    started::Bool
    counters::ObservationCounters
    counters_at_open::ObservationCounters   # totals when the current files were opened
    finished::Bool
end

_npanel(::CubedSphereMesh) = 6
_npanel(_) = 1

_reference_array(a::AbstractArray) = a
_reference_array(a::NTuple{6}) = a[1]

function _resolve_tracer_slots(state, tracers)
    available = collect(tracer_names(state))
    names = tracers === nothing ? available : tracers
    slots = Int[]
    for name in names
        idx = tracer_index(state, name)
        idx === nothing && throw(ArgumentError(
            "[output.observations].tracers names $(name), which is not a model tracer; " *
            "available: $(join(String.(available), ", "))"))
        push!(slots, idx)
    end
    return collect(names), slots
end

"""
    build_observation_sampler(spec, state, grid; origin, window_seconds, halo_width,
                              terrain = ObservationTerrain(), has_humidity = false)

Read every source for the days covering `window_seconds` (seconds after
`origin`, half-open), locate the requests on `grid.horizontal`, place each in
the column ([`resolve_placement`](@ref) with the static ground fields of
`terrain`, see [`ObservationTerrain`](@ref)), and prepare the device gather
buffers. Requests that cannot be placed fail the build or are skipped and
counted, as `spec.placement.unplaced` says. `has_humidity` says whether the
run's binaries carry specific humidity: on a dry-basis run, pressure-placed
requests need it for the total interface pressures of
[`TotalFromHumidity`](@ref) (the default `spec.placement.pressure_basis`),
and the build fails without it unless the basis is
[`DryInterfaces`](@ref); every window end must then pass `humidity` to
[`observe_window_boundary!`](@ref). Output files are opened by
`begin_observation_day!`. `NoObservationOutput` yields
`NoObservationSampler()`.
"""
build_observation_sampler(::NoObservationOutput, args...; kwargs...) = NoObservationSampler()

function build_observation_sampler(spec::ObservationOutputSpec, state, grid;
                                   origin::DateTime, window_seconds::Tuple{Real, Real},
                                   halo_width::Integer, terrain::ObservationTerrain = ObservationTerrain(),
                                   has_humidity::Bool = false)
    mesh = grid.horizontal
    vertical = grid.vertical
    iszero(vertical.B[1]) || throw(ArgumentError(
        "observation sampling needs a pure-pressure top interface (B[1] == 0); got B[1] = $(vertical.B[1])"))
    check_terrain(spec.placement, terrain, _OUTPUT_OBSERVATIONS_LABEL)
    check_surface_height_layout(terrain.model_surface, mesh)
    names, slots = _resolve_tracer_slots(state, spec.tracers)
    check_observation_tracer_names(names)
    reference = _reference_array(state.air_mass)
    nlevel = size(reference, ndims(reference))
    locator = cell_locator(mesh; halo_width)
    set = build_observation_set(spec.sources, origin, window_seconds)

    counters = ObservationCounters()
    examples = String[]   # a few unplaced requests for the message
    soundings, cells, placements = _locate_and_place(set.soundings, locator, spec, terrain, counters, examples)
    sites, site_cells, site_placements = _locate_and_place(set.sites, locator, spec, terrain, counters, examples)
    _handle_unplaced(spec.placement, counters, examples)
    basis = mass_basis(state)
    n_pressure = _n_pressure_placed(placements) + _n_pressure_placed(site_placements)
    needs_humidity = check_pressure_basis(spec.placement.pressure_basis, basis, has_humidity, n_pressure,
                                          _OUTPUT_OBSERVATIONS_LABEL)
    times = [r.time_seconds for r in soundings]
    buffers = ObservationGatherBuffers(reference, nlevel, slots;
                                       capacity = max(1024, length(sites) + 1))
    @info "observation sampling: $(length(soundings)) point events and $(length(sites)) station series located " *
          "($(counters.unlocated_soundings + counters.unlocated_sites) outside the mesh, " *
          "$(set.dropped_outside_window) outside the run window, $(set.skipped_invalid) source rows skipped)"
    _log_placements(placements, site_placements, counters)
    n_pressure > 0 && @info "observation pressure placement: $(n_pressure) requests against " *
                            _pressure_basis_text(spec.placement.pressure_basis, basis)
    _warn_below_ground(soundings, placements, sites, site_placements)
    isempty(soundings) && isempty(sites) &&
        @warn "observation sampling has nothing to sample; no observation files will be written"
    # Only the provenance of the terrain is kept: placements are resolved, and a
    # global DEM (about 0.9 GB) must not live as long as the sampler.
    terrain_sources = (surface_height_file = terrain_source(terrain.model_surface),
                       dem_file = terrain_source(terrain.dem))
    return ObservationSampler(spec, terrain_sources, origin, mesh, locator, spec.time_interpolation, names, nlevel,
                              Float64(vertical.A[1]), Float64(gravity(grid)),
                              basis, _npanel(mesh), soundings, times, cells, placements,
                              sites, site_cells, site_placements, needs_humidity,
                              buffers, nothing, nothing, 1, nothing, NaN,
                              nothing, nothing, AbstractObservationStream[], Set{String}(), false,
                              counters, ObservationCounters(), false)
end

# The config table named in the sampler's messages.
const _OUTPUT_OBSERVATIONS_LABEL = "[output.observations]"

"""
    observation_needs_humidity(sampler) -> Bool

Whether [`observe_window_boundary!`](@ref) needs the specific humidity: the
sampler places requests by total pressure on a dry-basis run.
"""
observation_needs_humidity(::NoObservationSampler) = false
observation_needs_humidity(s::ObservationSampler) = s.needs_humidity

_n_pressure_placed(placements) = count(p -> p.method == VERTICAL_PLACEMENT_CODES.pressure, placements)
_pressure_basis_text(::TotalFromHumidity, ::DryBasis) =
    "total interface pressures from the dry air mass and the specific humidity"
_pressure_basis_text(::TotalFromHumidity, ::MoistBasis) = "the (total) interface pressures of the moist air mass"
_pressure_basis_text(::DryInterfaces, ::DryBasis) =
    "dry interface pressures (pressure_basis = \"dry\"; record pressures are total pressures)"

_count_unplaced!(c::ObservationCounters, ::SoundingRequest) = (c.unplaced_soundings += 1)
_count_unplaced!(c::ObservationCounters, ::SiteRequest) = (c.unplaced_sites += 1)
_count_unlocated!(c::ObservationCounters, ::SoundingRequest) = (c.unlocated_soundings += 1)
_count_unlocated!(c::ObservationCounters, ::SiteRequest) = (c.unlocated_sites += 1)

function _locate_and_place(requests::Vector{R}, locator, spec::ObservationOutputSpec,
                           terrain::ObservationTerrain, counters::ObservationCounters,
                           examples::Vector{String}) where {R}
    kept = R[]
    cells = CellLocation[]
    placements = VerticalPlacement[]
    for request in requests
        cell = locate(locator, request.lon, request.lat)
        cell === nothing && (_count_unlocated!(counters, request); continue)
        rule = missing_height_rule(spec.sources[request.source])
        placement = resolve_placement(spec.placement, rule, request, cell, terrain)
        if placement === nothing
            _count_unplaced!(counters, request)
            length(examples) < 5 && push!(examples, _unplaced_example(request))
            continue
        end
        push!(kept, request)
        push!(cells, cell)
        push!(placements, placement)
    end
    return kept, cells, placements
end

_n_unplaced(c::ObservationCounters) = c.unplaced_soundings + c.unplaced_sites

_fmt_height(x) = isfinite(x) ? string(round(x; digits = 1), " m") : "none"
_unplaced_pressure(r::SoundingRequest) = r.pressure_pa
_unplaced_pressure(::SiteRequest) = NaN
_unplaced_example(r::Union{SoundingRequest, SiteRequest}) =
    "source $(r.source) id $(repr(r.id)) at lon $(r.lon), lat $(r.lat): altitude $(_fmt_height(r.altitude_m)), " *
    "elevation $(_fmt_height(r.elevation_m)), intake_height $(_fmt_height(r.intake_height_m)), " *
    "pressure $(isfinite(_unplaced_pressure(r)) ? string(_unplaced_pressure(r), " Pa") : "none")"

_coordinate_text(::HeightAboveGroundCoordinate) =
    "height above the ground (an altitude needs a record elevation or a dem_file value at the point)"
_coordinate_text(::PressureCoordinate) = "valid pressure"

function _unplaced_message(c::ObservationCounters, examples, coordinates::Tuple)
    labels = join(("\"$(vertical_coordinate_label(x))\"" for x in coordinates), ", ")
    return "$(c.unplaced_soundings) point events and $(c.unplaced_sites) station series give none of the " *
           "vertical coordinates in use (vertical_coordinates = [$(labels)]): no " *
           join(_coordinate_text.(coordinates), " and no ") * ". They give an altitude above sea level " *
           "without a ground elevation, only a coordinate the run does not use, or no vertical coordinate " *
           "at all (ObsPack). The model surface height is not used as the true ground. " *
           "First $(length(examples)): $(join(examples, "; "))"
end

# What could place them: a DEM (only with the height coordinate), another
# coordinate, or skipping; `label` names the config table.
function _unplaced_remedy(coordinates::Tuple, label::AbstractString)
    dem = _uses_coordinate(coordinates, HeightAboveGroundCoordinate) ?
          "Set $(label).dem_file to a DEM covering these points (the ground elevation, m above sea level), add" :
          "Add"
    return "$(dem) the coordinate the records give to $(label).vertical_coordinates, or set " *
           "$(label).unplaced_records = \"skip\" to drop them (counted in the n_unplaced_* file attributes)."
end

# `label` names the config table and `context` the caller in the messages; the
# inversion's `bind_to_mesh` shares this policy.
_handle_unplaced(spec::VerticalPlacementSpec, c::ObservationCounters, examples;
                 label::AbstractString = _OUTPUT_OBSERVATIONS_LABEL, context::AbstractString = "observation sampling") =
    _n_unplaced(c) == 0 ? nothing : _handle_unplaced_nonzero(spec.unplaced, c, examples, spec.coordinates, label, context)
_handle_unplaced_nonzero(::ErrorOnUnplaced, c::ObservationCounters, examples, coordinates, label, context) =
    throw(ArgumentError("$(context) cannot place every request in the column. " *
                        _unplaced_message(c, examples, coordinates) * " " * _unplaced_remedy(coordinates, label)))
_handle_unplaced_nonzero(::SkipUnplaced, c::ObservationCounters, examples, coordinates, label, context) =
    @warn "$(context) skips requests it cannot place (unplaced_records = \"skip\"). " *
          _unplaced_message(c, examples, coordinates)

function _log_placements(placements, site_placements, c::ObservationCounters)
    counts = [count(p -> p.method == code, placements) + count(p -> p.method == code, site_placements)
              for code in values(VERTICAL_PLACEMENT_CODES)]
    summary = join(("$(n) $(name)" for (name, n) in zip(keys(VERTICAL_PLACEMENT_CODES), counts) if n > 0), ", ")
    @info "observation vertical placement: " * (isempty(summary) ? "nothing to place" : summary) *
          (_n_unplaced(c) > 0 ? "; $(_n_unplaced(c)) unplaced" : "")
    return nothing
end

# Records placed more than BELOW_GROUND_TOLERANCE_M below the ground are kept
# (sampled in the lowest layer, `placement_height_agl` records the height) but
# reported, with the first few. Returns their number. `context` names the
# caller (the inversion's `bind_to_mesh` shares the check).
function _warn_below_ground(soundings, placements, sites, site_placements;
                            context::AbstractString = "observation sampling")
    rows = [(r, p) for (r, p) in Iterators.flatten((zip(soundings, placements), zip(sites, site_placements)))
            if below_ground(p)]
    isempty(rows) && return 0
    examples = join(("source $(r.source) id $(repr(r.id)) at lon $(r.lon), lat $(r.lat): " *
                     "$(round(p.height_agl_m; digits = 1)) m above the ground (altitude " *
                     "$(_fmt_height(r.altitude_m)), elevation $(_fmt_height(r.elevation_m)), intake_height " *
                     "$(_fmt_height(r.intake_height_m)))" for (r, p) in first(rows, 5)), "; ")
    @warn "$(context): $(length(rows)) requests are more than $(BELOW_GROUND_TOLERANCE_M) m below the " *
          "ground (a wrong elevation, DEM value, or altitude); they are sampled in the lowest layer and " *
          "their height is written as placement_height_agl. First $(min(5, length(rows))): $(examples)"
    return length(rows)
end

# -- output files -----------------------------------------------------------

function _source_descriptions(spec::ObservationOutputSpec)
    return JSON3.write([Dict("index" => i, "kind" => String(source_kind(src)),
                             "mode" => String(mode_label(source_mode(src))),
                             "path" => source_path_template(src)) for (i, src) in enumerate(spec.sources)])
end

function _stream_attributes(s::ObservationSampler)
    return Dict{String, Any}(
        "time_interpolation" => String(time_interpolation_label(s.time_interpolation)),
        "sources" => _source_descriptions(s.spec),
        "layer_height_temperature_kelvin" => s.spec.layer_height_temperature_kelvin,
        "p_top" => s.p_top,
        "gravity" => s.gravity,
        "unplaced_records" => String(unplaced_policy_label(s.spec.placement.unplaced)),
        "surface_height_file" => s.terrain_sources.surface_height_file,
        "dem_file" => s.terrain_sources.dem_file,
        "absolute_altitude_above_agl_m" => _altitude_option_attribute(s.spec.placement.altitude_placement),
        "vertical_coordinates" => join(String.(vertical_coordinate_label.(s.spec.placement.coordinates)), " "),
        "pressure_basis" => String(pressure_basis_label(s.spec.placement.pressure_basis)))
end

_current_streams(s::ObservationSampler) =
    AbstractObservationStream[st for st in (s.soundings_stream, s.sites_stream) if st !== nothing]

# Hand the current day's files to the retired list; they flush and close
# whenever the NetCDF lock is free, so a day switch never waits for the
# background snapshot write.
function _retire_streams!(s::ObservationSampler)
    append!(s.retired, _current_streams(s))
    s.soundings_stream = nothing
    s.sites_stream = nothing
    _reap_retired!(s)
    return nothing
end

# Streams leave the retired list only once closed; a close that throws leaves
# the list consistent (the failing stream is poisoned and dropped first).
function _reap_retired!(s::ObservationSampler)
    i = 1
    while i <= length(s.retired)
        stream = s.retired[i]
        closed = try
            _try_close!(stream)
        catch
            deleteat!(s.retired, i)
            rethrow()
        end
        closed ? deleteat!(s.retired, i) : (i += 1)
    end
    return nothing
end

# Close every stream; each close is attempted even if an earlier one throws.
function _close_streams!(s::ObservationSampler)
    streams = vcat(_current_streams(s), s.retired)
    s.soundings_stream = nothing
    s.sites_stream = nothing
    empty!(s.retired)
    errors = Any[]
    for st in streams
        try
            close(st)
        catch err
            push!(errors, err)
        end
    end
    isempty(errors) || throw(length(errors) == 1 ? only(errors) : CompositeException(errors))
    return nothing
end

# NaN: every record is placed above the ground.
_altitude_option_attribute(::AboveGroundPlacement) = NaN
_altitude_option_attribute(option::AbsoluteAltitudeAbove) = option.min_height_agl_m

# Counts attributable to the currently open files (all of them for a single
# file); request-set properties (unlocated, unplaced) are run-level and repeated.
function _write_file_summaries!(s::ObservationSampler)
    c, c0 = s.counters, s.counters_at_open
    unlocated = Dict{String, Any}("n_unlocated_soundings" => c.unlocated_soundings,
                                  "n_unlocated_sites" => c.unlocated_sites,
                                  "n_unplaced_soundings" => c.unplaced_soundings,
                                  "n_unplaced_sites" => c.unplaced_sites)
    s.soundings_stream === nothing || write_summary_attributes!(s.soundings_stream, merge(unlocated,
        Dict{String, Any}("n_emitted" => c.emitted - c0.emitted,
                          "n_before_start" => c.dropped_before_start - c0.dropped_before_start,
                          "n_after_end" => c.dropped_after_end - c0.dropped_after_end,
                          "n_one_sided" => c.one_sided - c0.one_sided)))
    s.sites_stream === nothing || write_summary_attributes!(s.sites_stream, merge(unlocated,
        Dict{String, Any}("n_records" => c.site_records - c0.site_records)))
    return nothing
end

function _claim_path!(s::ObservationSampler, path::AbstractString)
    path in s.opened_paths && throw(ArgumentError(
        "observation output $(path) would be written twice in this run; use a {date}, " *
        "{YYYYMMDD}, or {day} token in [output.observations].path for daily files"))
    push!(s.opened_paths, path)
    return path
end

# Each file exists only when it has requests: a soundings file for point
# events, a sites file for station series.
function _open_streams!(s::ObservationSampler, date_label::AbstractString, day_index::Integer)
    s.counters_at_open = copy(s.counters)
    attributes = _stream_attributes(s)
    if !isempty(s.soundings)
        path = _claim_path!(s, observation_output_path(s.spec, SoundingMode(), date_label, day_index))
        s.soundings_stream = SoundingNetCDFStream(path, s.mesh, s.nlevel, s.tracer_names;
            mass_basis = s.mass_basis, origin = s.origin, deflate_level = s.spec.deflate_level, attributes)
    end
    if !isempty(s.sites)
        path = _claim_path!(s, observation_output_path(s.spec, SiteMode(), date_label, day_index))
        s.sites_stream = SiteNetCDFStream(path, s.mesh, s.nlevel, s.tracer_names, s.sites, s.site_cells,
                                          s.site_placements;
            mass_basis = s.mass_basis, origin = s.origin, deflate_level = s.spec.deflate_level,
            write_profiles = s.spec.write_profile_for_sites, attributes)
    end
    return nothing
end

"""
    begin_observation_day!(sampler, date_label, day_index)

Called by the runner before the first window boundary of each input binary.
Single-file runs open their files on the first call; daily runs close the
previous day's files and open the next pair. Rows are written to the files
open at emission time. A binary's last window ends at the next day's 00:00,
which is still observed before the next binary opens: with linear
interpolation a sounding lands in the file of its own day, and each day's
sites file ends with the 00:00 sample of the next day (the first file also
holds the initial state at t = 0). With `nearest_window`, soundings in the
first half window after 00:00 are emitted at that boundary and land in the
previous day's file.
"""
begin_observation_day!(::NoObservationSampler, ::AbstractString, ::Integer) = nothing

function begin_observation_day!(s::ObservationSampler, date_label::AbstractString, day_index::Integer)
    s.finished && throw(ArgumentError("observation sampler already finished"))
    _begin_day!(s.spec.partition, s, date_label, day_index)
    return nothing
end

function _begin_day!(::SingleOutputFile, s::ObservationSampler, date_label, day_index)
    s.started || _open_streams!(s, date_label, day_index)
    s.started = true
    return nothing
end

function _begin_day!(::DailyOutputFiles, s::ObservationSampler, date_label, day_index)
    _write_file_summaries!(s)
    _retire_streams!(s)
    _open_streams!(s, date_label, day_index)
    s.started = true
    return nothing
end

# -- window boundaries ------------------------------------------------------

"""
    observe_window_boundary!(sampler, state, time_seconds; next_window_seconds, temperature = nothing,
                             humidity = nothing)

Sample `state` at a met-window end `time_seconds` after the run origin.
`next_window_seconds` is the length of the window that starts now (it sets
which soundings receive their first sample). `temperature`, when given, is
a per-layer temperature field (K, k = 1 at the top) used for site layer
heights, either with the air-mass layout or, on the cubed sphere, as
interior-only panels; otherwise the configured constant applies.
`humidity` is the specific humidity (kg kg⁻¹ of moist air, k = 1 at the top)
at this window end, in the same layouts; a sampler that places requests by
total pressure on a dry-basis run ([`TotalFromHumidity`](@ref)) requires it
and otherwise ignores it. Call once at `t = 0` on the initial state and
after every window.
"""
observe_window_boundary!(::NoObservationSampler, state, time_seconds::Real; kwargs...) = nothing

function observe_window_boundary!(s::ObservationSampler, state, time_seconds::Real;
                                  next_window_seconds::Real, temperature = nothing, humidity = nothing)
    s.finished && throw(ArgumentError("observation sampler already finished"))
    s.started || throw(ArgumentError(
        "begin_observation_day! must run before the first window boundary"))
    t = Float64(time_seconds)
    dt_next = Float64(next_window_seconds)
    isfinite(dt_next) && dt_next > 0 || throw(ArgumentError(
        "next_window_seconds must be positive; got $(next_window_seconds)"))
    isnan(s.last_boundary) || t > s.last_boundary || throw(ArgumentError(
        "window boundaries must increase; got $(t) after $(s.last_boundary)"))
    fields = (; temperature, humidity = _sampled_humidity(s, state, humidity))
    _observe!(s.time_interpolation, s, state, t, dt_next, fields)
    s.last_boundary = t
    _reap_retired!(s)
    return nothing
end

# The humidity field the sampler gathers: `nothing` unless pressure placement
# needs it, which then requires it at every window end.
function _sampled_humidity(s::ObservationSampler, state, humidity)
    s.needs_humidity || return nothing
    humidity === nothing && throw(ArgumentError(
        "observation sampling places requests by total pressure on a dry-basis run and needs the specific " *
        "humidity at every window end, but none was passed (humidity = nothing)"))
    _check_field_layout(state.air_mass, humidity, "humidity")
    return humidity
end

# A field with the air-mass layout must match it; cubed-sphere panels are
# checked by `_field_plan` (with or without the halo).
_check_field_layout(air::AbstractArray, field::AbstractArray, name) = size(field) == size(air) ||
    throw(DimensionMismatch("$(name) has size $(size(field)); the air mass has size $(size(air))"))
_check_field_layout(air::NTuple{6}, field::NTuple{6, <:AbstractArray}, name) = nothing
_check_field_layout(air, field, name) = throw(ArgumentError(
    "$(name) must have the air-mass layout ($(typeof(air))); got $(typeof(field))"))

# Gather the point events `range` plus every site, in one launch.
function _gather!(s::ObservationSampler, state, range::UnitRange{Int}, fields::NamedTuple)
    cells = vcat(view(s.sounding_cells, range), s.site_cells)
    plan = GatherPlan(cells, s.npanel)
    gather_columns!(s.buffers, state, plan.columns, plan.panel_ranges)
    _gather_temperature!(s, cells, plan, fields.temperature)
    _gather_humidity!(s, cells, plan, fields.humidity)
    return plan
end

_field_buffers(s::ObservationSampler) =
    ObservationGatherBuffers(s.buffers.air, s.nlevel, Int[]; capacity = s.buffers.capacity)

function _gather_layer_field!(buffers::ObservationGatherBuffers, s::ObservationSampler, cells, plan::GatherPlan,
                              field)
    fplan = _field_plan(s, cells, plan, field)
    gather_field!(buffers, field, fplan.columns, fplan.panel_ranges)
    return nothing
end

_gather_temperature!(s::ObservationSampler, cells, plan::GatherPlan, ::Nothing) = nothing
function _gather_temperature!(s::ObservationSampler, cells, plan::GatherPlan, temperature)
    s.temperature_buffers === nothing && (s.temperature_buffers = _field_buffers(s))
    return _gather_layer_field!(s.temperature_buffers, s, cells, plan, temperature)
end

_gather_humidity!(s::ObservationSampler, cells, plan::GatherPlan, ::Nothing) = nothing
function _gather_humidity!(s::ObservationSampler, cells, plan::GatherPlan, humidity)
    s.humidity_buffers === nothing && (s.humidity_buffers = _field_buffers(s))
    return _gather_layer_field!(s.humidity_buffers, s, cells, plan, humidity)
end

# Lat-lon and reduced-Gaussian fields share the state layout.
_field_plan(s::ObservationSampler, cells, plan::GatherPlan, ::AbstractArray) = plan
function _field_plan(s::ObservationSampler, cells, plan::GatherPlan, field::NTuple{6, <:AbstractArray})
    n1 = size(field[1], 1)
    Nc = s.mesh.Nc
    n1 == Nc + 2 * s.locator.halo_width && return plan
    n1 == Nc && return GatherPlan(cells, s.npanel; column = c -> c.cell)
    throw(DimensionMismatch("field panels are $(n1) wide; expected $(Nc) or $(Nc + 2 * s.locator.halo_width)"))
end

function _observe!(::LinearWindowInterpolation, s::ObservationSampler, state, t::Float64,
                   dt_next::Float64, fields::NamedTuple)
    times = s.sounding_times
    mid = searchsortedfirst(times, t)
    hi_next = searchsortedfirst(times, t + dt_next)
    pending = s.pending
    if isnan(s.last_boundary)
        # The very first window end: soundings before `t` cannot be bracketed.
        s.counters.dropped_before_start += max(mid - s.cursor, 0)
        lo = max(mid, s.cursor)
    else
        # Everything from the held batch (or, after an empty window, from the
        # cursor) up to `t` is due now; rows without a held sample go one-sided.
        lo = pending === nothing ? s.cursor : first(pending.range)
    end
    hi_next = max(hi_next, lo)
    range = lo:hi_next - 1
    plan = _gather!(s, state, range, fields)
    nsounding = length(range)
    if mid > lo
        _emit_bracketed!(s, pending, lo, mid, t, plan, fields.temperature)
    end
    s.pending = mid < hi_next ? _hold!(s, mid, hi_next, t, plan, lo, fields.temperature) : nothing
    s.cursor = hi_next
    _emit_sites!(s, t, plan, nsounding, fields.temperature)
    return nothing
end

function _observe!(::NearestWindowSampling, s::ObservationSampler, state, t::Float64,
                   dt_next::Float64, fields::NamedTuple)
    times = s.sounding_times
    lo = s.cursor
    if isnan(s.last_boundary)
        mid = searchsortedfirst(times, t)
        s.counters.dropped_before_start += max(mid - lo, 0)
        lo = max(mid, lo)
    end
    hi = max(searchsortedfirst(times, t + dt_next / 2), lo)
    range = lo:hi - 1
    plan = _gather!(s, state, range, fields)
    nsounding = length(range)
    if nsounding > 0
        times_now = fill(t, nsounding)
        batch = _sounding_batch(s, range, times_now, times_now, fill(NaN32, nsounding),
                                fill(interp_flag(NearestWindowSampling()), nsounding),
                                (i, k) -> s.buffers.air_host[k, plan.position[i]],
                                (i, k, tr) -> s.buffers.tracers_host[k, tr, plan.position[i]],
                                i -> _row_layer_temperature(s, plan.position[i], fields.temperature),
                                (i, k) -> _gathered_humidity(s, k, plan.position[i]))
        append_soundings!(s.soundings_stream, batch)
        s.counters.emitted += nsounding
    end
    s.cursor = hi
    _emit_sites!(s, t, plan, nsounding, fields.temperature)
    return nothing
end

# Soundings lo:mid-1 have their second sample in the current gather; those
# inside `pending.range` are blended, the rest (only after a change of window
# length between binaries, or after a window without soundings) are emitted
# one-sided from the current state.
function _emit_bracketed!(s::ObservationSampler, pending::Union{Nothing, PendingSoundings},
                          lo::Int, mid::Int, t::Float64, plan::GatherPlan, temperature)
    n = mid - lo
    times = s.sounding_times
    t_prev = pending === nothing ? t : pending.t_prev
    last_held = pending === nothing ? lo - 1 : last(pending.range)
    pending_lo = pending === nothing ? lo : first(pending.range)
    span = t - t_prev
    bracketed = [idx <= last_held for idx in lo:mid - 1]
    w = [bracketed[i] ? clamp((times[lo + i - 1] - t_prev) / span, 0.0, 1.0) : 1.0 for i in 1:n]
    weights = Float32.(w)
    flags = [interp_flag(LinearWindowInterpolation(), b) for b in bracketed]
    prev = [b ? t_prev : t for b in bracketed]
    next = fill(t, n)
    air = s.buffers.air_host
    tracers = s.buffers.tracers_host
    blend_air = (i, k) -> begin
        current = Float64(air[k, plan.position[i]])
        bracketed[i] ? (1 - w[i]) * pending.air[k, lo - pending_lo + i] + w[i] * current : current
    end
    blend_tracer = (i, k, tr) -> begin
        current = Float64(tracers[k, tr, plan.position[i]])
        bracketed[i] ? (1 - w[i]) * pending.tracers[k, tr, lo - pending_lo + i] + w[i] * current : current
    end
    # The humidity is blended like the masses (it only places pressure-placed rows).
    blend_humidity = (i, k) -> begin
        current = _gathered_humidity(s, k, plan.position[i])
        bracketed[i] ? (1 - w[i]) * pending.humidity[k, lo - pending_lo + i] + w[i] * current : current
    end
    layer_temperature = i -> _row_layer_temperature(s, plan.position[i], temperature)
    batch = _sounding_batch(s, lo:mid - 1, prev, next, weights, flags, blend_air, blend_tracer,
                            layer_temperature, blend_humidity)
    append_soundings!(s.soundings_stream, batch)
    s.counters.emitted += n
    s.counters.one_sided += count(!, bracketed)
    return nothing
end

# Copy the first samples of soundings mid:hi-1 out of the gather buffers. The
# layer temperature is kept too: events exactly at the final window end are
# emitted from this sample alone (`_emit_held!`); and the humidity, which is
# blended like the masses when pressure placement needs it.
function _hold!(s::ObservationSampler, mid::Int, hi::Int, t::Float64, plan::GatherPlan, lo::Int,
                temperature)
    n = hi - mid
    slots = [plan.position[idx - lo + 1] for idx in mid:hi - 1]
    air = Matrix{Float64}(undef, s.nlevel, n)
    tracers = Array{Float64, 3}(undef, s.nlevel, length(s.tracer_names), n)
    for (i, slot) in enumerate(slots)
        @inbounds for k in 1:s.nlevel
            air[k, i] = Float64(s.buffers.air_host[k, slot])
            for tr in 1:length(s.tracer_names)
                tracers[k, tr, i] = Float64(s.buffers.tracers_host[k, tr, slot])
            end
        end
    end
    humidity = s.needs_humidity ? Float64.(s.humidity_buffers.air_host[:, slots]) : nothing
    return PendingSoundings(mid:hi - 1, t, air, tracers, _held_temperature(s, slots, temperature), humidity)
end

# Specific humidity of layer `k` in gather slot `slot`; NaN when pressure
# placement needs none (it is then never read).
_gathered_humidity(s::ObservationSampler, k::Int, slot::Int) =
    s.needs_humidity ? Float64(s.humidity_buffers.air_host[k, slot]) : NaN

_held_temperature(s::ObservationSampler, slots, ::Nothing) = nothing
_held_temperature(s::ObservationSampler, slots, temperature) =
    Float64.(s.temperature_buffers.air_host[:, slots])

_held_layer_temperature(s::ObservationSampler, ::Nothing, i::Int) =
    ConstantLayerTemperature(s.spec.layer_height_temperature_kelvin)
_held_layer_temperature(s::ObservationSampler, temperature::Matrix{Float64}, i::Int) =
    ProfileLayerTemperature(view(temperature, :, i))

# Assemble the output rows for point events `range`; `air_at(i, k)` and
# `tracer_at(i, k, tr)` return the (already blended) masses of the i-th row,
# `temperature_at(i)` its `AbstractLayerTemperature` for intake heights, and
# `humidity_at(i, k)` its specific humidity (read only for pressure placement).
function _sounding_batch(s::ObservationSampler, range::UnitRange{Int}, prev::Vector{Float64},
                         next::Vector{Float64}, weights::Vector{Float32}, flags::Vector{Int8},
                         air_at, tracer_at, temperature_at, humidity_at)
    n = length(range)
    nlevel = s.nlevel
    ntracer = length(s.tracer_names)
    origin_unix = datetime2unix(s.origin)
    p_half = Matrix{Float64}(undef, nlevel + 1, n)
    per_area = Matrix{Float64}(undef, nlevel, n)
    vmr = Array{Float64, 3}(undef, nlevel, ntracer, n)
    means = Matrix{Float64}(undef, ntracer, n)
    intake_values = Matrix{Float64}(undef, ntracer, n)
    levels = Vector{Int32}(undef, n)
    bottoms = Vector{Float64}(undef, n)
    tops = Vector{Float64}(undef, n)
    methods = Vector{Int8}(undef, n)
    air_col = Vector{Float64}(undef, nlevel)
    tracer_col = Vector{Float64}(undef, nlevel)
    z_half = Vector{Float64}(undef, nlevel + 1)
    column = PlacementColumn(nlevel)
    cells = Vector{CellLocation}(undef, n)
    requests = view(s.soundings, range)
    for (i, idx) in enumerate(range)
        cell = s.sounding_cells[idx]
        cells[i] = cell
        @inbounds for k in 1:nlevel
            air_col[k] = air_at(i, k)
            per_area[k, i] = air_col[k] / cell.area
        end
        layer_temperature = temperature_at(i)
        level = _sample_layer!(column, view(p_half, :, i), z_half, s, s.sounding_placements[idx], air_col,
                               cell.area, layer_temperature, k -> humidity_at(i, k))
        levels[i] = Int32(level)
        bottoms[i] = z_half[level + 1]
        tops[i] = z_half[level]
        methods[i] = height_method_code(layer_temperature)
        for tr in 1:ntracer
            @inbounds for k in 1:nlevel
                tracer_col[k] = tracer_at(i, k, tr)
            end
            profile = view(vmr, :, tr, i)
            mixing_ratio_profile!(profile, air_col, tracer_col)
            means[tr, i] = column_mean_vmr(air_col, tracer_col)
            intake_values[tr, i] = profile[level]
        end
    end
    return SoundingBatch(; ids = [r.id for r in requests], sources = Int32[r.source for r in requests],
                         times = [origin_unix + r.time_seconds for r in requests],
                         latitudes = [r.lat for r in requests], longitudes = [r.lon for r in requests],
                         cells, sample_time_prev = prev .+ origin_unix, sample_time_next = next .+ origin_unix,
                         weights, flags, p_half, air_mass_per_area = per_area, tracers = vmr,
                         column_means = means, elevations = [r.elevation_m for r in requests],
                         intake_heights = [r.intake_height_m for r in requests],
                         altitudes = [r.altitude_m for r in requests], pressures = [r.pressure_pa for r in requests],
                         placements = s.sounding_placements[range], intake_levels = levels,
                         layer_bottom = bottoms, layer_top = tops, height_methods = methods, intake_values)
end

# Layer temperature for heights: the gathered per-layer field when the run
# passes one, otherwise the configured constant.
_row_layer_temperature(s::ObservationSampler, slot::Int, ::Nothing) =
    ConstantLayerTemperature(s.spec.layer_height_temperature_kelvin)
_row_layer_temperature(s::ObservationSampler, slot::Int, temperature) =
    ProfileLayerTemperature(view(s.temperature_buffers.air_host, :, slot))

# Layer of `placement` in one sampled column (`sample_column_layer!`, which
# also fills the interface pressures `p_half` and heights `z_half`); the
# humidity column is filled only for a pressure placement that needs it.
function _sample_layer!(column::PlacementColumn, p_half::AbstractVector{Float64}, z_half::AbstractVector{Float64},
                        s::ObservationSampler, placement::VerticalPlacement, air_col::AbstractVector, area::Real,
                        layer_temperature::AbstractLayerTemperature, humidity_at)
    if s.needs_humidity && placement.method == VERTICAL_PLACEMENT_CODES.pressure
        @inbounds for k in eachindex(column.humidity)
            column.humidity[k] = humidity_at(k)
        end
    end
    return sample_column_layer!(column, p_half, z_half, placement, air_col, area, s.gravity, s.p_top,
                                layer_temperature, s.spec.placement.pressure_basis, s.mass_basis)
end

function _emit_sites!(s::ObservationSampler, t::Float64, plan::GatherPlan, nsounding::Int, temperature)
    stream = s.sites_stream
    stream === nothing && return nothing
    nsite = length(s.sites)
    nlevel = s.nlevel
    ntracer = length(s.tracer_names)
    profiles = s.spec.write_profile_for_sites
    record = SiteRecord(datetime2unix(s.origin) + t, nsite, nlevel, ntracer; profiles)
    p_half = Vector{Float64}(undef, nlevel + 1)
    z_half = Vector{Float64}(undef, nlevel + 1)
    air_col = Vector{Float64}(undef, nlevel)
    tracer_col = Vector{Float64}(undef, nlevel)
    vmr = Vector{Float64}(undef, nlevel)
    column = PlacementColumn(nlevel)
    for (i, site) in enumerate(s.sites)
        if !site_active(site.schedule, t)
            _blank_site!(record, i, profiles)
            continue
        end
        slot = plan.position[nsounding + i]
        cell = s.site_cells[i]
        @inbounds for k in 1:nlevel
            air_col[k] = Float64(s.buffers.air_host[k, slot])
        end
        layer_temperature = _row_layer_temperature(s, slot, temperature)
        level = _sample_layer!(column, p_half, z_half, s, s.site_placements[i], air_col, cell.area,
                               layer_temperature, k -> _gathered_humidity(s, k, slot))
        record.ps[i] = p_half[nlevel + 1]
        record.intake_level[i] = Int32(level)
        record.layer_bottom[i] = z_half[level + 1]
        record.layer_top[i] = z_half[level]
        record.height_method[i] = height_method_code(layer_temperature)
        for tr in 1:ntracer
            @inbounds for k in 1:nlevel
                tracer_col[k] = Float64(s.buffers.tracers_host[k, tr, slot])
            end
            mixing_ratio_profile!(vmr, air_col, tracer_col)
            record.values[i, tr] = vmr[level]
            record.surface_values[i, tr] = vmr[nlevel]
            profiles && (record.profiles[:, i, tr] .= vmr)
        end
        if profiles
            record.p_half[:, i] .= p_half
            record.air_mass_per_area[:, i] .= air_col ./ cell.area
        end
    end
    append_site_record!(stream, record)
    s.counters.site_records += 1
    return nothing
end

# A site outside its time range: NaN values, layer 0.
function _blank_site!(record::SiteRecord, i::Int, profiles::Bool)
    record.ps[i] = NaN
    record.intake_level[i] = 0
    record.layer_bottom[i] = NaN
    record.layer_top[i] = NaN
    record.height_method[i] = HEIGHT_METHOD_NOT_SAMPLED
    record.values[i, :] .= NaN
    record.surface_values[i, :] .= NaN
    if profiles
        record.profiles[:, i, :] .= NaN
        record.p_half[:, i] .= NaN
        record.air_mass_per_area[:, i] .= NaN
    end
    return nothing
end

# -- end of run -------------------------------------------------------------

"""
    finish_observations!(sampler)

Emit events exactly at the final window end, drop events that never received
their second sample, write the summary attributes, and warn about anything
that was not sampled. Idempotent.
"""
finish_observations!(::NoObservationSampler) = nothing

function finish_observations!(s::ObservationSampler)
    s.finished && return nothing
    pending = s.pending
    s.pending = nothing
    held = 0
    if pending !== nothing
        # Events exactly at the last window end have their complete sample.
        at_end = searchsortedlast(view(s.sounding_times, pending.range), pending.t_prev)
        at_end > 0 && _emit_held!(s, pending, at_end)
        held = length(pending.range) - at_end
    end
    s.counters.dropped_after_end += held + max(length(s.sounding_times) - s.cursor + 1, 0)
    c = s.counters
    _write_file_summaries!(s)
    if c.dropped_before_start + c.dropped_after_end > 0
        @warn "observation sampling dropped $(c.dropped_before_start) soundings before the first " *
              "window end and $(c.dropped_after_end) after the last one"
    end
    @info "observation sampling wrote $(c.emitted) soundings ($(c.one_sided) one-sided) and " *
          "$(c.site_records) site records"
    s.finished = true
    return nothing
end

function _emit_held!(s::ObservationSampler, pending::PendingSoundings, n::Int)
    t = pending.t_prev
    times_now = fill(t, n)
    batch = _sounding_batch(s, first(pending.range):first(pending.range) + n - 1, times_now, times_now,
                            zeros(Float32, n), fill(interp_flag(LinearWindowInterpolation(), true), n),
                            (i, k) -> pending.air[k, i], (i, k, tr) -> pending.tracers[k, tr, i],
                            i -> _held_layer_temperature(s, pending.temperature, i),
                            (i, k) -> pending.humidity[k, i])
    append_soundings!(s.soundings_stream, batch)
    s.counters.emitted += n
    return nothing
end

function Base.close(s::ObservationSampler)
    _close_streams!(s)
    return nothing
end
Base.close(::NoObservationSampler) = nothing
