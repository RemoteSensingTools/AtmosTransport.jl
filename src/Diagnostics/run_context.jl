# ===========================================================================
# The pieces of a finished run that its diagnostics need, built by the
# runtime's own code from the run config.
#
# A post-run diagnostic must see the run exactly as the runtime saw it: the
# same tracer specifications, the same horizontal grid (whose cell areas the
# surface-flux loader multiplies by), the same float type, the same run start
# (the origin of the snapshot `time` axis and of the flux slice times), and
# the same output file names. `RunContext` gathers them with the functions
# `run_driven_simulation` itself calls, so no diagnostic re-derives them.
# ===========================================================================

"""
One met window of a run: binary index, start (s since the run start), length
(s), number of advection steps, and the length of one step (s) as the runtime
computes it in the run's float type, `FT(length) / FT(steps)`.
"""
const RunWindow = @NamedTuple{binary::Int, start::Float64, length::Float64, steps::Int, step::Float64}

"""
    RunContext(config_path::AbstractString)
    RunContext(config::AbstractDict; config_path = "")

The parts of a transport run that post-run diagnostics need, built with the
runtime's own code from the run config (a TOML path or the parsed `Dict`):

- `config`: the parsed run config;
- `grid`: the grid of the first transport binary, opened on the CPU in the
  run's float type (`[numerics].float_type`), as the runner builds it;
- `binary_paths`: the transport binaries, in run order (`[input]`);
- `tracer_specs`: the tracer specifications the runner parses
  (`[tracers.<name>]`), sorted by name; without `[tracers]`, the legacy single
  tracer `[run].tracer_name` (default `CO2`) of a lat-lon or reduced-Gaussian
  run, initialised from `[init]` and without a surface flux;
- `reference_time`: the run start, `[input].start_date` at 00 UTC. Snapshot
  `time` values are hours since it, and flux slice times are seconds since it;
- `windows`: every met window the run steps through, in order, as
  `(; binary, start, length, steps, step)`: the index of its binary, its start
  in seconds since the run start, its length (the header's window length in
  the run's float type), its number of advection steps (the binary's
  `steps_per_window` schedule) and the step length the runtime applies,
  `FT(length) / FT(steps)` in the run's float type (in a Float32 run seven
  steps of a 3600 s window add up to 3599.99994 s). The windows are those of
  `[run].start_window` through `[run].stop_window` of every binary, as the
  runner selects them;
- `run_seconds`: the run length, computed as the runner computes it from
  every binary's window layout;
- `air_mass`: the total air mass of window 1 of the first binary, in kg (dry
  air for the default dry-basis binaries): the air mass the runner builds the
  initial tracer state on, also when `[run].start_window` is larger;
- `decaying_tracers`: tracers with a `[chemistry]` sink (half-lives of
  `kind = "decay"`).

`floattype(ctx)` returns the run's float type. Building a context reads the
header of every binary (about 40 ms each) and the first window of the first.
The run must have `[input].start_date`, and its window range and tracers must
be ones the runner accepts (`1 ≤ start_window ≤ stop_window ≤` the window
count of each binary; `start_window = 1` and every window of every binary for
a run of several binaries; on a cubed-sphere grid `start_window = 1` and a
`[tracers]` table).
"""
struct RunContext{FT <: AbstractFloat, G}
    config           :: Dict{String, Any}
    config_path      :: String
    grid             :: G
    binary_paths     :: Vector{String}
    tracer_specs     :: Vector{TransportTracerSpec}
    reference_time   :: DateTime
    windows          :: Vector{RunWindow}
    run_seconds      :: Float64
    air_mass         :: Float64
    decaying_tracers :: Vector{Symbol}
end

RunContext(config_path::AbstractString) =
    RunContext(TOML.parsefile(config_path); config_path = abspath(config_path))

function RunContext(config::AbstractDict; config_path::AbstractString = "")
    cfg = Dict{String, Any}(String(k) => v for (k, v) in pairs(config))
    FT = _cfg_float_type(cfg)
    input_cfg = get(cfg, "input", nothing)
    input_cfg isa AbstractDict || throw(ArgumentError("the run config has no [input] table"))
    binary_paths = expand_binary_paths(input_cfg)
    isempty(binary_paths) && throw(ArgumentError("[input] resolved to no transport binary"))
    reference_time = _run_reference_time(cfg)
    reference_time === nothing && throw(ArgumentError(
        "the run config has no [input].start_date; snapshot times and flux slice times " *
        "are counted from it"))

    run_cfg = get(cfg, "run", Dict{String, Any}())
    start_window = _run_window_index(run_cfg, "start_window", 1)     # integers, as the runner
    stop_window = get(run_cfg, "stop_window", nothing) === nothing ? nothing :
                  _run_window_index(run_cfg, "stop_window", nothing)
    layouts = [_binary_window_layout(path) for path in binary_paths]
    _check_window_ranges(layouts, start_window, stop_window)
    runtime_layout = _runtime_window_layout([(l.window_seconds, l.nwindow) for l in layouts], FT)
    run_seconds = 3600 * _layout_run_hours(runtime_layout; start_window, stop_window_override = stop_window)
    windows = _run_windows(layouts, runtime_layout, start_window, stop_window, FT)

    driver = TransportBinaryDriver(first(binary_paths); FT = FT, arch = CPU(), Hp = 1)
    try
        _run_time_origin(cfg, driver, first(binary_paths))   # start_date agrees with the binary
        grid = driver_grid(driver)
        specs = _run_tracer_specs(cfg, grid.horizontal)
        grid.horizontal isa CubedSphereMesh && start_window != 1 && throw(ArgumentError(
            "cubed-sphere runs require [run].start_window = 1, as the runner does"))
        # The runner packs the initial tracer state on window 1 (`_make_structured_model`;
        # cubed-sphere runs require start_window = 1), whatever `[run].start_window` is.
        air_mass = _total_air_mass(load_transport_window(driver, 1).air_mass, grid.horizontal)
        decaying = _sink_tracers(chemistry_spec(get(cfg, "chemistry", Dict{String, Any}())))
        return RunContext{FT, typeof(grid)}(cfg, String(config_path), grid, binary_paths,
                                            collect(TransportTracerSpec, specs), reference_time,
                                            windows, run_seconds, air_mass, decaying)
    finally
        close(driver)
    end
end

# The window ranges the runner accepts: `1 ≤ start ≤ stop ≤ nwindow` in every
# binary (`DrivenSimulation`), and the whole of every binary in a run of several
# (`_check_multifile_window_count`).
function _check_window_ranges(layouts, start_window::Int, stop_window)
    for layout in layouts
        _check_multifile_window_count(layout.nwindow, start_window, stop_window, length(layouts))
        stop = stop_window === nothing ? layout.nwindow : Int(stop_window)
        1 <= start_window <= stop <= layout.nwindow || throw(ArgumentError(
            "invalid window range: start_window=$(start_window), stop_window=$(stop), " *
            "total_windows=$(layout.nwindow)"))
    end
    return nothing
end

# Window length, window count and steps per window from a binary's header, as
# the runner reads them (`binary_capabilities`, `steps_per_window_schedule`).
function _binary_window_layout(path::AbstractString)
    reader = TransportBinaryReader(String(path); FT = Float64)
    try
        caps = binary_capabilities(reader)
        return (; window_seconds = Float64(caps.window_seconds), nwindow = Int(caps.nwindow),
                steps = Int.(steps_per_window_schedule(reader)))
    finally
        close(reader)
    end
end

# The windows the runner steps through: windows `start_window:stop` of every
# binary, with `stop` the binary's last window or `[run].stop_window`. The step
# is the runtime's `Δt = FT(window_dt) / FT(steps)` (`DrivenSimulation`).
function _run_windows(layouts, runtime_layout, start_window::Int, stop_window, ::Type{FT}) where {FT}
    windows = RunWindow[]
    clock = 0.0
    for (b, (layout, (len, nwindow))) in enumerate(zip(layouts, runtime_layout))
        stop = stop_window === nothing ? nwindow : min(Int(stop_window), nwindow)
        for w in start_window:stop
            steps = layout.steps[w]
            push!(windows, (; binary = b, start = clock, length = len, steps,
                            step = Float64(FT(len) / FT(steps))))
            clock += len
        end
    end
    return windows
end

Grids.floattype(::RunContext{FT}) where {FT} = FT

function Base.show(io::IO, ctx::RunContext)
    print(io, "RunContext(", floattype(ctx), ", ", length(ctx.binary_paths), " binaries, ",
          length(ctx.tracer_specs), " tracers, start ", ctx.reference_time, ", ",
          ctx.run_seconds / 3600, " h)")
end

# Total air mass of one window in kg, summed in Float64 over the interior
# cells: cubed-sphere panels carry a halo of width `(size - Nc) ÷ 2`.
_total_air_mass(m::Tuple, mesh::CubedSphereMesh) = sum(panel -> _interior_air_mass(panel, mesh), m)
function _interior_air_mass(panel::AbstractArray{<:Real, 3}, mesh::CubedSphereMesh)
    h = (size(panel, 1) - mesh.Nc) ÷ 2
    return sum(Float64, @view panel_interior(panel, h)[1:mesh.Nc, 1:mesh.Nc, :])
end
_total_air_mass(m::AbstractArray{<:Real}, mesh) = sum(Float64, m)

# Tracers with a global chemistry sink.
_sink_tracers(::NoChemistrySpec) = Symbol[]
_sink_tracers(spec::DecayChemistrySpec) = collect(Symbol, keys(spec.half_lives))

_decays(ctx::RunContext, tracer::Symbol) =
    any(name -> lowercase(String(name)) == lowercase(String(tracer)), ctx.decaying_tracers)

_output_spec(ctx::RunContext) =
    runtime_output_spec(get(ctx.config, "output", Dict{String, Any}()), floattype(ctx);
                        default_cap_hours = ctx.run_seconds / 3600)

"""
    run_snapshot_paths(ctx::RunContext) -> Vector{String}

The snapshot files the run writes, in time order, named as the runner names
them from `[output].path`: one file for `split = "single"`; for
`split = "daily"` one file per transport binary that takes a snapshot (the
binary's `YYYYMMDD` date fills `{YYYYMMDD}`, `{date}` or `{day}`). A snapshot
at a window end belongs to that window's binary, the one at hour 0 to the
first. Only NetCDF output can be read back; an ATMSNAP
(`format = "binary_mmap"`) or disabled output is an error.
"""
function run_snapshot_paths(ctx::RunContext)
    spec = _output_spec(ctx)
    output_enabled(spec) || throw(ArgumentError(
        "the run writes no snapshots ([output] is disabled, has no path or no snapshot times)"))
    return _run_snapshot_paths(ctx, spec, spec.format)
end

_run_snapshot_paths(ctx::RunContext, spec, ::NetCDFOutput) =
    _partition_paths(ctx, spec, spec.partition)
function _run_snapshot_paths(::RunContext, spec, format::AbstractOutputFormat)
    throw(ArgumentError(
        "the run writes $(output_format_label(format)) snapshots; the diagnostics read NetCDF snapshots"))
end

_partition_paths(ctx, spec, ::SingleOutputFile) = [output_path(spec)]
function _partition_paths(ctx, spec, ::DailyOutputFiles)
    owners = sort!(unique(_snapshot_binary(ctx, h) for h in _scheduled_snapshot_hours(ctx)))
    return [output_path_for_day(spec, _binary_date_label(ctx.binary_paths[b]), b) for b in owners]
end

# The binary during which the runner takes the snapshot at hour `h`: the one
# whose window ends there (hour 0 is taken before the first window).
function _snapshot_binary(ctx::RunContext, h::Real)
    h <= _HOUR_MATCH && return first(ctx.windows).binary
    k = findfirst(w -> (w.start + w.length) / 3600 >= h - _HOUR_MATCH, ctx.windows)
    k === nothing && throw(ArgumentError("snapshot hour $(h) is after the end of the run"))
    return ctx.windows[k].binary
end

# Snapshot hours the runner writes: the scheduled hours up to the end of the run.
_scheduled_snapshot_hours(ctx::RunContext) =
    filter(h -> h <= ctx.run_seconds / 3600 + _HOUR_MATCH, snapshot_hours(_output_spec(ctx)))

# Hours closer than this (1e-6 h, 3.6 ms) are the same snapshot time, as in the
# runner's own schedule checks.
const _HOUR_MATCH = 1e-6
