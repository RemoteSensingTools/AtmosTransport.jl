# Observation sampling glue. The runner parses `[output.observations]`, builds
# the sampler once the model state exists, and hands ownership to
# `RunSnapshotOutput` so the sampler is closed on every exit path together
# with the snapshot stream. Sampling happens at met-window ends only.
#
# Clock contract: the run origin (t = 0) is the start of window 1 of the first
# input binary, i.e. `[input].start_date` at 00:00 UTC, or `start_time` when
# the inputs are an explicit `binary_paths` list. A single-file run starting at
# `start_window > 1` begins `(start_window - 1)` windows after the origin.

const _OBSERVATION_ORIGIN_MESSAGE =
    "[output.observations] needs an absolute run origin: set [input].start_date " *
    "or [output.observations].start_time"

# Run origin (t = 0): `[input].start_date` at 00:00 UTC, else `start_time`.
function _observation_time_origin(spec::ObservationOutputSpec, cfg)
    reference = _run_reference_time(cfg)
    if reference !== nothing
        spec.start_time === nothing || spec.start_time == reference || throw(ArgumentError(
            "[output.observations].start_time = $(spec.start_time) disagrees with " *
            "[input].start_date = $(Date(reference)); the run origin is start_date at 00:00 UTC"))
        return reference
    end
    spec.start_time === nothing && throw(ArgumentError(_OBSERVATION_ORIGIN_MESSAGE))
    return spec.start_time
end

# `validate_config` preflight.
_check_observation_time_origin(::NoObservationOutput, cfg) = nothing
_check_observation_time_origin(spec::ObservationOutputSpec, cfg) =
    (_observation_time_origin(spec, cfg); nothing)

# Sampled tracers must exist in `[tracers]` (when the config uses that table).
_check_observation_tracers(::NoObservationOutput, cfg) = nothing
function _check_observation_tracers(spec::ObservationOutputSpec, cfg)
    spec.tracers === nothing && return nothing
    tracers_cfg = get(cfg, "tracers", nothing)
    tracers_cfg isa AbstractDict || return nothing
    defined = Set(Symbol(String(k)) for k in keys(tracers_cfg))
    missing_names = [t for t in spec.tracers if !(t in defined)]
    isempty(missing_names) || throw(ArgumentError(
        "[output.observations].tracers names $(join(String.(missing_names), ", ")), which " *
        "[tracers] does not define; defined: $(join(sort!(String.(collect(defined))), ", "))"))
    # Averaging-kernel columns are written for sampled tracers only.
    kernel_tracers = unique!(reduce(vcat, (averaging_kernel_tracers(s) for s in spec.sources); init = Symbol[]))
    unsampled = [t for t in kernel_tracers if !(t in spec.tracers)]
    isempty(unsampled) || throw(ArgumentError(
        "[output.observations].sources ask for averaging-kernel columns of $(join(String.(unsampled), ", ")), " *
        "which [output.observations].tracers does not sample"))
    check_observation_tracer_names(spec.tracers; averaging_kernel_tracers = kernel_tracers)
    return nothing
end

# The placement datasets must exist; their grid is checked when they are read.
_check_observation_placement_files(::NoObservationOutput) = nothing
function _check_observation_placement_files(spec::ObservationOutputSpec)
    for (key, path) in (("surface_height_file", spec.placement.surface_height_file),
                        ("dem_file", spec.placement.dem_file))
        path === nothing || isfile(path) || throw(ArgumentError(
            "[output.observations].$(key) = $(repr(path)) does not exist"))
    end
    return nothing
end

# Static ground fields of `[output.observations]`, read through GriddedIO on
# the run's horizontal mesh (never from transport binaries).
function _observation_terrain(placement::VerticalPlacementSpec, mesh)
    model_surface = placement.surface_height_file === nothing ? NoModelSurfaceHeight() :
                    read_model_surface_height(placement.surface_height_file, mesh)
    dem = placement.dem_file === nothing ? NoPointElevation() :
          read_point_elevation(placement.dem_file; variable = placement.dem_variable)
    return ObservationTerrain(; model_surface, dem)
end

function _binary_label_date(path::AbstractString)
    label = _binary_date_label(path)
    isempty(label) && return nothing
    date = tryparse(Date, label, dateformat"yyyymmdd")
    date === nothing && @warn "observation sampling: 8-digit token $(label) in $(basename(path)) is not a date; ignoring it"
    return date
end

# The labelled day of the first binary must be the origin's day, otherwise
# every sounding time would be shifted silently.
function _check_origin_against_binaries(origin::DateTime, binary_paths)
    isempty(binary_paths) && return nothing
    first_day = _binary_label_date(first(binary_paths))
    first_day === nothing || first_day == Date(origin) || throw(ArgumentError(
        "observation run origin $(origin) is not on the first binary's date $(first_day) " *
        "($(basename(first(binary_paths))))"))
    Time(origin) == Time(0) ||
        @warn "observation run origin $(origin) is not 00:00 UTC; it must be the start of window 1 of the first binary"
    return nothing
end

# `has_humidity`: whether the run's binaries carry the specific humidity that
# `_window_humidity` hands the sampler (total pressures for pressure placement).
function _install_observation_sampler!(output::RunSnapshotOutput, output_cfg::AbstractDict,
                                       partition::AbstractOutputPartition, state, grid;
                                       cfg, binary_paths, halo_width::Integer, has_humidity::Bool,
                                       offset_seconds::Real = 0.0, span_seconds::Real = Inf)
    spec = observation_output_spec(output_cfg; partition)
    sampler = _build_sampler(spec, state, grid; cfg, binary_paths, halo_width, has_humidity,
                             offset_seconds, span_seconds)
    output.observations = sampler
    return sampler
end

_build_sampler(::NoObservationOutput, state, grid; kwargs...) = NoObservationSampler()
function _build_sampler(spec::ObservationOutputSpec, state, grid; cfg, binary_paths, halo_width,
                        has_humidity, offset_seconds, span_seconds)
    origin = _observation_time_origin(spec, cfg)
    _check_origin_against_binaries(origin, binary_paths)
    for mode in (SoundingMode(), SiteMode())
        _check_unique_day_paths(spec.partition,
            (label, idx) -> observation_output_path(spec, mode, label, idx), binary_paths,
            "[output.observations].path")
    end
    t0 = Float64(offset_seconds)
    window_seconds = (t0, t0 + Float64(span_seconds))
    @info "observation sampling window: $(origin + Millisecond(round(Int, 1000 * window_seconds[1]))) " *
          "to $(origin + Millisecond(round(Int, 1000 * window_seconds[2]))) UTC"
    terrain = _observation_terrain(spec.placement, grid.horizontal)
    return build_observation_sampler(spec, state, grid; origin, window_seconds, halo_width, terrain, has_humidity)
end

# Where the sampler's boundary humidity comes from (`_boundary_humidity`).
abstract type _AbstractObservationHumidity end
struct _WindowFieldHumidity <: _AbstractObservationHumidity end
struct _NextWindowVDIFFHumidity{H, D} <: _AbstractObservationHumidity
    binary_paths::Vector{String}
    host::H      # NTuple of host panels (Nc, Nc, Nz), read from the binary
    field::D     # the same panels on the state's backend, handed to the sampler
end

# Per-binary entry: switch daily files, and on the first binary sample the
# initial state at the start of the run.
_begin_observation_binary!(timer, ::NoObservationSampler, sim, path, idx::Integer, seconds::Real; kwargs...) =
    nothing
function _begin_observation_binary!(timer, sampler::AbstractObservationSampler, sim, path,
                                    idx::Integer, seconds::Real;
                                    humidity::_AbstractObservationHumidity = _WindowFieldHumidity())
    begin_observation_day!(sampler, _binary_date_label(path), idx)
    idx == 1 && _observe_window_end!(timer, sampler, sim, seconds; run_start = true, humidity, binary_index = idx)
    return nothing
end

# GCHP VDIFF layer temperature (cubed-sphere binaries only, interior panels,
# k = 1 at the top) of the window that just ended; otherwise the sampler's
# constant applies. Virtual-temperature effects are neglected.
_window_temperature(sim) = _vdiff_temperature(sim.window.vdiff)
_vdiff_temperature(::Nothing) = nothing
_vdiff_temperature(vdiff) = vdiff.t

# Specific humidity (kg kg⁻¹ of moist air, layer means, k = 1 at the top) at
# a sampled boundary, for total interface pressures, by where it comes from:
#
# - `_WindowFieldHumidity`: the window's own fields, `qv_start` at the run
#   start and `qv_end` at the window's end (lat-lon binaries written with
#   `include_qv`; the reduced-Gaussian preprocessor writes none).
# - `_NextWindowVDIFFHumidity` (cubed sphere): the GCHP VDIFF `qv` is a
#   window-start value, so the humidity at the end of window w is the `qv` of
#   window w + 1 (window 1 of the next binary after a binary's last window),
#   read from the binary. The run start takes the first window's own `qv`.
#
# `sim.window` is the window that just ended (the next one loads on the next step).
_window_humidity(window, run_start::Bool) =
    window.qv_start !== nothing ? (run_start ? window.qv_start : window.qv_end) : _vdiff_humidity(window.vdiff)
_vdiff_humidity(::Nothing) = nothing
_vdiff_humidity(vdiff) = vdiff.qv

# The cubed-sphere source when the sampler needs humidity (else nothing is read).
function _cs_observation_humidity(sampler, state, binary_paths)
    observation_needs_humidity(sampler) || return _WindowFieldHumidity()
    reference = first(state.air_mass)
    Nc = size(reference, 1) - 2 * state.halo_width
    dims = (Nc, Nc, size(reference, 3))
    host = ntuple(_ -> Array{eltype(reference)}(undef, dims), length(state.air_mass))
    field = ntuple(_ -> similar(reference, dims), length(state.air_mass))
    return _NextWindowVDIFFHumidity(collect(String, binary_paths), host, field)
end

_boundary_humidity(::_WindowFieldHumidity, sim, run_start::Bool, binary_index::Integer) =
    _window_humidity(sim.window, run_start)
function _boundary_humidity(h::_NextWindowVDIFFHumidity, sim, run_start::Bool, binary_index::Integer)
    run_start && return _vdiff_humidity(sim.window.vdiff)
    window = sim.current_window_index
    if window < total_windows(sim.driver)
        load_vdiff_qv_window!(h.host, sim.driver.reader, window + 1)
    elseif binary_index < length(h.binary_paths)
        reader = TransportBinaryReader(h.binary_paths[binary_index + 1]; FT = eltype(first(h.host)))
        try
            load_vdiff_qv_window!(h.host, reader, 1)
        finally
            close(reader)
        end
    else
        # HACK: the run's last window end has no next window; it takes the ended
        # window's start humidity, one window old. On the golden MERRA-2 C90 L72
        # binary that puts the rebuilt interfaces up to 23 Pa (99th percentile
        # 11 Pa) from the met model's total interfaces, against <= 0.23 Pa with
        # the next window's humidity.
        # TODO: carry a window-end humidity in cubed-sphere binaries (a VDIFF qv
        # endpoint) so the final window end is exact too.
        return _vdiff_humidity(sim.window.vdiff)
    end
    foreach(copyto!, h.field, h.host)
    return h.field
end

# Whether a binary (its `binary_capabilities`) carries the humidity
# `_window_humidity` reads: `qv_start`/`qv_end`, or on the cubed sphere the
# GCHP VDIFF payload (loaded only when complete, which the reader enforces).
_binary_observation_humidity(caps) =
    caps.humidity || (caps.grid_type === :cubed_sphere && :vdiff_qv in caps.payload_sections)

# `has_humidity` of the sampler: every binary must carry humidity, checked
# from the headers before transport (`binary_humidity`, one flag per binary),
# so a later binary without it cannot stop the run midway; a direct runner
# call without the flags falls back to the first binary.
_run_observation_humidity(::Nothing, binary_paths, first_binary::Bool) = first_binary
function _run_observation_humidity(binary_humidity::AbstractVector{Bool}, binary_paths, first_binary::Bool)
    length(binary_humidity) == length(binary_paths) || throw(DimensionMismatch(
        "$(length(binary_humidity)) humidity flags for $(length(binary_paths)) binaries"))
    return all(binary_humidity)
end

_observe_window_end!(timer, ::NoObservationSampler, sim, seconds::Real; kwargs...) = nothing
function _observe_window_end!(timer, sampler::AbstractObservationSampler, sim, seconds::Real;
                              run_start::Bool = false,
                              humidity::_AbstractObservationHumidity = _WindowFieldHumidity(),
                              binary_index::Integer = 1)
    temperature = _window_temperature(sim)
    humidity = observation_needs_humidity(sampler) ? _boundary_humidity(humidity, sim, run_start, binary_index) :
               nothing
    next_window_seconds = Float64(window_dt(sim.driver))
    timed_io_write!(timer, () -> observe_window_boundary!(sampler, sim.model.state, Float64(seconds);
                                                          next_window_seconds, temperature, humidity))
    return nothing
end
