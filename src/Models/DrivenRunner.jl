"""
    DrivenRunner

Library-level entry point for the driven transport runtime.

The canonical CLI, `scripts/run_transport.jl`, is a thin wrapper over
`run_driven_simulation(cfg)`. The library function handles LL/RG and CS
runtime flows with dispatch driven by the first binary's header
(`inspect_binary(first_path).grid_type`).

## Ownership boundary

- **Binary header** (`grid_type`, `mass_basis`, `payload_sections`,
  panel convention) — authoritative for topology and capability.
  Accessed here via `binary_capabilities(driver.reader)` and
  `air_mass_basis(driver)`.
- **TOML `[input]`** — either an explicit `binary_paths = [...]`
  list (Shape A) or a `folder + start_date + end_date
  (+ file_pattern)` block (Shape B). `expand_binary_paths` preserves explicit
  list order and sorts folder selections by date, checking gaps and duplicates.
- **TOML physics** (`[advection]` / `[diffusion]` / `[convection]`)
  — validated against binary capabilities by
  `validate_runtime_physics_recipe` / `build_runtime_physics_recipe`
  before the loop.
- **TOML `[tracers.*]`** — tracer specs consumed via
  `build_initial_mixing_ratio` + `pack_initial_tracer_mass`
  (basis-aware per `feedback_vmr_to_mass_basis_aware`) and
  `build_surface_flux_sources`.

## Where the physics actually runs

`DrivenRunner` is the orchestration layer, not the kernel layer. It opens
transport binaries, builds initial state, chooses output cadence, and installs
the TOML-selected operators on a `TransportModel`. The live physics call chain
is:

1. `build_runtime_physics_recipe` reads `[advection]`, `[diffusion]`,
   `[convection]`, `[chemistry]`, and tracer surface-flux settings and returns
   the operator objects.
2. `_make_structured_model` or the CS model constructor installs those
   operators on `TransportModel(state, fluxes, grid, recipe.advection; ...)`.
3. `DrivenSimulation(model, driver; ...)` loads the current met window and
   wraps chemistry and surface sources into the model with `with_chemistry`
   and `with_emissions`.
4. The runtime loop below calls `run_window!(sim)` for LL/RG or `step!(sim)`
   for CS. Those functions live in `driven_stepping.jl`.
5. `DrivenSimulation.step!` refreshes time-varying forcing from the driver,
   then calls `TransportModel.step!` or, for binary-scheduled substeps,
   `transport_step!` bracketed by exponential-decay half-steps, plus
   end-of-window convection (and unsplit non-decay chemistry).
6. `TransportModel.jl` is where advection, surface emissions, diffusion,
   convection, and chemistry are applied to the state.

So, when following a run as a scientist: start here to understand data and
configuration flow, then jump to `DrivenSimulation.step!` and
`TransportModel.step!` to see the actual physics ordering.

## GPU residency (feedback_verify_gpu_runs_on_gpu)

When `[architecture].use_gpu = true` or `backend` selects a GPU, the runner
asserts that `state.air_mass` lives on the selected backend after model
construction and prints a `[gpu verified] …` line. A silent CPU fallback
aborts the run with a precise error message.
"""
module DrivenRunner

using Adapt
using ..Models: _config_bool, _spec_int, Callback, NaNChecker, WindowInterval,
                AbstractInitialConditionSpec, initial_condition_spec
using Dates: Dates, Date, DateTime, Day, Millisecond, Time, @dateformat_str
using Printf: @sprintf, @printf
using Logging
using ...LogLevels: log_level, with_log_level
using ProgressMeter: Progress, next!, finish!, update!

import ...expand_data_path
using ...SectionTimer
using ...ConfigChecks: unknown_key_messages, parse_float_type
using ...Parameters: GLOBAL_MEAN_DRY_SURFACE_PRESSURE
using ..State: AbstractMassBasis, DryBasis, MoistBasis, CellState,
                CubedSphereState, total_air_mass, total_mass, tracer_names,
                tracer_index, get_tracer
using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, nlevels
using ..Operators: LinRoodPPMScheme, PPMScheme, CW84Limiter, FV3ScalarProfile, SlopesScheme, UpwindScheme,
                  ImplicitVerticalDiffusion, TimeVaryingSurfaceFluxSource,
                  total_emission_rate,
                  uses_diffusive_surface_flux_boundary,
                  AbstractConvection,
                  NoConvection, TM5Convection, CMFMCConvection,
                  CMFMCMatrixConvection
using ..Architectures: AbstractArchitecture, architecture_from_config,
                       is_gpu, ensure_runtime!,
                       array_adapter, architecture_label, device_name,
                       backend_name, synchronize_architecture!,
                       assert_residency!, assert_float_type!
using ..MetDrivers: AbstractMetDriver, TransportBinaryDriver,
                     load_transport_window, driver_grid, air_mass_basis,
                     total_windows, window_dt, binary_capabilities,
                     inspect_binary, steps_per_window,
                     steps_per_window_schedule, release_payload!, same_window_layout,
                     surface_pressure_basis, supports_moisture,
                     TransportBinaryReader, load_vdiff_qv_window!
using ..InitialConditionIO: _regridding_method, _surface_flux_kind, surface_flux_species,
                            _hold_outside_coverage, RUN_SPAN_CLOCK_SLACK_SECONDS
using ..InitialConditionIO: build_initial_mixing_ratio, _build_cs_initial_mixing_ratio,
                             pack_initial_tracer_mass, _cs_pack_interior_into_halo!,
                             build_surface_flux_sources,
                             TotalSurfacePressure
using ...GriddedIO: read_model_surface_height, read_point_elevation
using ..BinaryPathExpander: expand_binary_paths
using ..InputStaging: InputStager, staged_path_for!, cleanup_staging!
using ..Output: AbstractSnapshotFrame, NetCDFSnapshotStream, append_snapshot!, SnapshotFrame,
                AbstractOutputPartition, SingleOutputFile, DailyOutputFiles,
                AbstractOutputFormat, NetCDFOutput, BinaryMmapOutput,
                RuntimeOutputSpec, runtime_output_spec, snapshot_hours,
                set_physics_attributes!, set_time_origin!,
                output_enabled, output_path, output_path_for_day,
                capture_snapshot, write_snapshot_netcdf, write_snapshot_binary,
                NoObservationOutput, ObservationOutputSpec, observation_output_spec,
                AbstractObservationSampler, NoObservationSampler, build_observation_sampler,
                samples_observations, begin_observation_day!, observe_window_boundary!,
                finish_observations!, check_observation_tracer_names, observation_needs_humidity,
                averaging_kernel_tracers,
                observation_output_path, SoundingMode, SiteMode,
                VerticalPlacementSpec, ObservationTerrain, NoModelSurfaceHeight, NoPointElevation,
                _SNAPSHOT_SCHEDULE_KEYS, _SNAPSHOT_INTERVAL_KEYS, _schedule_has_times
# TransportModel + DrivenSimulation live alongside us in the Models module;
# reach up to the parent and pull them in.
using ..Models: TransportModel
import ..Models: DrivenSimulation, run_window!, run!, step!, allocate_face_fluxes,
                  _finish_window_prefetch!
# Physics-recipe helpers: `build_runtime_physics_recipe` /
# `validate_runtime_physics_recipe` are defined in `RuntimePhysicsRecipe.jl`
# (loaded before us in Models). Pull them in so we don't have to stutter
# through `Main.AtmosTransport.*`.
using ..Models: build_runtime_physics_recipe, validate_runtime_physics_recipe,
                 build_runtime_advection, configured_halo_width,
                 CubedSphereRuntimeRecipeStyle, runtime_physics_attributes
# Config-only parsers, run by `validate_config` before any binary is opened.
using ..Models: advection_spec, diffusion_spec, convection_spec, chemistry_spec,
                 _ADVECTION_SPEC_KEYS,
                 _advection_section, _diffusion_section, _convection_section,
                 _chemistry_section, _normalize_air_mass_reset_mode,
                 _resolve_physics_cadence, binary_cfl_check

export run_driven_simulation, validate_config, TransportTracerSpec

include("runner/progress.jl")
include("runner/logging.jl")
include("runner/configuration.jl")
include("runner/config_keys.jl")
include("runner/summary.jl")
include("runner/resources.jl")
include("runner/output.jl")
include("runner/observations.jl")
include("runner/model_setup.jl")

# ===========================================================================
# run_driven_simulation — top-level entry
# ===========================================================================

function _validate_input_binary_expectations(caps, input_cfg::AbstractDict,
                                             path::AbstractString)
    if haskey(input_cfg, "expected_nlevel")
        expected = Int(input_cfg["expected_nlevel"])
        caps.nlevel == expected || throw(ArgumentError(
            "[input].expected_nlevel=$expected but $(basename(path)) has " *
            "nlevel=$(caps.nlevel). This usually means the run config is " *
            "pointing at an older preprocessing product."))
    end
    if haskey(input_cfg, "required_preprocessor_contract")
        required = String(input_cfg["required_preprocessor_contract"])
        actual = caps.preprocessor_contract
        actual == required || throw(ArgumentError(
            "[input].required_preprocessor_contract=$(repr(required)) but " *
            "$(basename(path)) declares $(repr(actual))."))
    end
    if _config_bool(input_cfg, "require_adaptive_substeps", false,
                    "[input].require_adaptive_substeps")
        caps.adaptive_substeps === true || throw(ArgumentError(
            "[input].require_adaptive_substeps=true but $(basename(path)) " *
            "does not declare adaptive_substeps=true."))
    end
    return nothing
end

function _log_binary_summary(path::AbstractString, caps)
    schedule = caps.variable_step_schedule ?
        "adaptive" : string(caps.steps_per_window)
    fields = "[" * join(String.(sort(collect(caps.payload_sections))), ",") * "]"
    @info "[binary] $(path) grid=$(caps.grid_type) levels=$(caps.nlevel) " *
          "basis=$(caps.mass_basis) steps/window=$(schedule) fields=$(fields)"
    return nothing
end

"""
    run_driven_simulation(cfg::AbstractDict) -> TransportModel

Canonical high-level entry point for scientists running an AtmosTransport TOML.
Use this unless you are writing a custom time loop. It resolves `[input]` to a
sorted binary list, prints a one-line summary for each binary, dispatches on
the first binary's `grid_type`, validates physics-vs-capability, runs the loop,
optionally writes topology-native snapshots, and returns the terminal
`TransportModel`.

This function does not call the advection/diffusion/convection kernels
directly. The handoff to physics happens inside the structured loop at
`run_window!(sim)` and inside the CS loop at `step!(sim)`. Both routes enter
`DrivenSimulation.step!`, which refreshes forcing and then calls
`TransportModel.step!` / `transport_step!` / `convection_chemistry_step!`.
"""
function run_driven_simulation(cfg::AbstractDict)
    # The whole run, validation included, logs at `[logging] level`. An invalid
    # level is reported by `validate_config`, which then runs at the default.
    level = try
        log_level(cfg)
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
    return with_log_level(level) do
        ok, errors = validate_config(cfg)
        ok || throw(ArgumentError(
            "Invalid AtmosTransport run config:\n  - " * join(errors, "\n  - ")))
        arch = _cfg_architecture(cfg)
        ensure_runtime!(arch)
        # Loading an optional GPU package adds its array and Adapt methods in a new
        # Julia world. Enter the complete run implementation through `invokelatest`
        # once, after that load, so every subsequent GPU operation sees those
        # methods. This single startup trampoline keeps the hot loop on normal
        # dispatch while making the library entry point safe without CLI preloading.
        Base.invokelatest(_run_driven_simulation, cfg, arch)
    end
end

function _run_driven_simulation(cfg::AbstractDict, arch::AbstractArchitecture)
    input_cfg = get(cfg, "input", Dict{String, Any}())
    binary_paths = expand_binary_paths(input_cfg)
    isempty(binary_paths) &&
        throw(ArgumentError("[input] resolved to an empty binary list"))
    # Section timing instrumentation, off unless ATMOSTR_TIMERS is 1/true/on/yes.
    # Enabled here so every section accumulator covers the whole driven
    # loop including snapshot capture / write.
    timers_on = SectionTimer.maybe_enable_from_env!()
    result = try
        _run_driven_simulation_inputs(cfg, input_cfg, binary_paths, arch)
    finally
        # Header inspection and resource construction can fail before stepping.
        timers_on && SectionTimer.disable!()
    end
    if timers_on
        SectionTimer.report(stderr)
        output_cfg = get(cfg, "output", Dict{String, Any}())
        snapshot_file = expand_data_path(String(get(output_cfg, "path",
                                                    get(output_cfg, "snapshot_file",
                                                        get(output_cfg, "filename", "")))))
        if !isempty(snapshot_file)
            csv_path = replace(snapshot_file, r"\.nc$" => "") * ".timings.csv"
            written = SectionTimer.write_csv(csv_path)
            written !== nothing && @info "Section timings → $(written)"
        end
    end
    return result
end

function _run_driven_simulation_inputs(cfg, input_cfg, binary_paths, arch::AbstractArchitecture)
    # Dispatch on the first binary's grid_type — the ownership boundary
    # (binary header owns topology, TOML owns physics kinds). Opening the
    # reader in `inspect_binary` runs the header-contract and file-size
    # checks; the cm-continuity replay gate runs later, when
    # `TransportBinaryDriver` opens each binary.
    binary_caps = [(path = path, caps = inspect_binary(path; io = devnull))
                   for path in binary_paths]
    for item in binary_caps
        _log_binary_summary(item.path, item.caps)
    end
    _check_run_surface_pressure_bases(binary_caps)
    _check_run_dry_mass_targets(binary_caps)
    caps = first(binary_caps).caps
    _validate_input_binary_expectations(caps, input_cfg, first(binary_paths))
    # (window length, window count) of every binary, for the snapshot schedule.
    window_layout = [(item.caps.window_seconds, item.caps.nwindow) for item in binary_caps]
    # Whether each binary carries the specific humidity observation sampling reads.
    binary_humidity = [_binary_observation_humidity(item.caps) for item in binary_caps]
    run_cfg = get(cfg, "run", Dict{String, Any}())
    _check_window_range(window_layout;
                        start_window = Int(get(run_cfg, "start_window", 1)),
                        stop_window_override = get(run_cfg, "stop_window", nothing))
    # Rolling NVMe input staging (opt-in via [input.staging]; default off ⇒
    # `staged_path_for!` returns the NAS path, bit-identical to today). Created
    # here and torn down in `finally` so staged multi-GB files are always
    # cleaned up, even if the run throws partway through.
    stager = InputStager(binary_paths, get(input_cfg, "staging", Dict{String, Any}()))
    input_resources = RunInputResources()
    output_resources = RunSnapshotOutput()
    return try
        _with_run_resource(output_resources) do
            _with_run_resource(input_resources) do
                _run_driven_simulation_for(Val(caps.grid_type), binary_paths, cfg,
                                           stager, arch, output_resources, input_resources,
                                           window_layout, binary_humidity)
            end
        end
    finally
        cleanup_staging!(stager)
    end
end

function _check_run_surface_pressure_bases(binary_caps)
    isempty(binary_caps) && return nothing
    reference = first(binary_caps)
    basis_ref = reference.caps.surface_pressure_basis
    for item in Iterators.drop(binary_caps, 1)
        typeof(item.caps.surface_pressure_basis) === typeof(basis_ref) && continue
        throw(ArgumentError(
            "transport binaries in one run must share one surface-pressure basis: " *
            "$(basename(reference.path)) stores $(nameof(typeof(basis_ref))), but " *
            "$(basename(item.path)) stores " *
            "$(nameof(typeof(item.caps.surface_pressure_basis))). Regenerate the " *
            "mismatched day so hybrid-coordinate consumers, including diffusion, " *
            "interpret surface pressure consistently."))
    end
    return basis_ref
end

function _check_run_dry_mass_targets(binary_caps)
    isempty(binary_caps) && return nothing
    explicit = nothing
    explicit_path = nothing
    for item in binary_caps
        state = item.caps.dry_mass_pinned
        state === nothing && continue
        if explicit === nothing
            explicit = state
            explicit_path = item.path
        elseif state != explicit
            throw(ArgumentError(
                "transport binaries in one run cannot mix explicitly pinned and " *
                "unpinned dry-mass states: $(basename(explicit_path)) has " *
                "dry_mass_pinned=$(explicit), but $(basename(item.path)) has " *
                "dry_mass_pinned=$(state). Regenerate the mismatched binary with " *
                "one consistent mass-fix configuration."))
        end
    end

    comparable = filter(binary_caps) do item
        item.caps.dry_mass_pinned !== false &&
            item.caps.dry_mass_target_ps_pa !== nothing
    end
    isempty(comparable) && return nothing
    reference = first(comparable)
    target_ref = Float64(reference.caps.dry_mass_target_ps_pa)
    for item in Iterators.drop(comparable, 1)
        target = Float64(item.caps.dry_mass_target_ps_pa)
        isapprox(target, target_ref; atol = 0.1, rtol = 1e-7) && continue
        throw(ArgumentError(
            "transport binaries in one run must share one dry-mass target: " *
            "$(basename(reference.path)) has $(target_ref) Pa " *
            "($(reference.caps.dry_mass_target_source)), but " *
            "$(basename(item.path)) has $(target) Pa " *
            "($(item.caps.dry_mass_target_source)). Regenerate the mismatched binary " *
            "with GLOBAL_MEAN_DRY_SURFACE_PRESSURE=$(GLOBAL_MEAN_DRY_SURFACE_PRESSURE) Pa."))
    end
    return target_ref
end

_run_driven_simulation_for(::Val{:latlon}, binary_paths::Vector{String}, cfg,
                           stager::InputStager, arch::AbstractArchitecture,
                           output_resources::RunSnapshotOutput, input_resources::RunInputResources,
                           window_layout, binary_humidity) =
    _run_driven_simulation_structured(binary_paths, cfg, stager, arch, output_resources,
                                      input_resources, window_layout, binary_humidity)
_run_driven_simulation_for(::Val{:reduced_gaussian}, binary_paths::Vector{String}, cfg,
                           stager::InputStager, arch::AbstractArchitecture,
                           output_resources::RunSnapshotOutput, input_resources::RunInputResources,
                           window_layout, binary_humidity) =
    _run_driven_simulation_structured(binary_paths, cfg, stager, arch, output_resources,
                                      input_resources, window_layout, binary_humidity)
_run_driven_simulation_for(::Val{:cubed_sphere}, binary_paths::Vector{String}, cfg,
                           stager::InputStager, arch::AbstractArchitecture,
                           output_resources::RunSnapshotOutput, input_resources::RunInputResources,
                           window_layout, binary_humidity) =
    _run_driven_simulation_cs(binary_paths, cfg, stager, arch, output_resources,
                              input_resources, window_layout, binary_humidity)
function _run_driven_simulation_for(::Val{grid_type}, _binary_paths::Vector{String},
                                    _cfg, _stager::InputStager, _arch, _output_resources::RunSnapshotOutput, _input_resources::RunInputResources,
                                    _window_layout, _binary_humidity) where grid_type
    throw(ArgumentError("Unsupported transport-binary grid_type=$(grid_type)."))
end

# Parse the run start as a `DateTime` (at 00:00) from `[input].start_date`,
# used as the `reference_time` for time-varying surface fluxes so their slice
# times align to the simulation clock. Returns `nothing` if no start_date is
# present (`validate_config` then rejects time-varying surface fluxes). The
# runners check it against the first binary's dates (`_run_time_origin`)
# before any forcing is built, so a present start_date is the run origin.
function _run_reference_time(cfg)
    input_cfg = get(cfg, "input", nothing)
    input_cfg isa AbstractDict || return nothing
    haskey(input_cfg, "start_date") || return nothing
    return DateTime(Date(String(input_cfg["start_date"])))
end

function _header_date(driver)
    raw = driver.reader.header.raw_header
    haskey(raw, "date") || return nothing
    value = raw["date"]
    try
        return Date(String(value))
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("invalid transport-binary `date` header $(repr(value))"))
    end
end

# The forcing day in a binary's file name: the first `YYYYMMDD` or
# `YYYY-MM-DD` token delimited by non-digits, so a hash prefix, a longer
# numeric identifier or a partly dashed token (`2021-1201`) is never read as a
# date. Canonical daily binaries predate the `date` header but carry their day
# in the name (e.g. `transport_20211202.bin`). A token that is not a calendar
# date (`20211301`) does not date the file.
function _filename_date(path::AbstractString)
    stamp = match(r"(?:^|\D)(\d{8}|\d{4}-\d{2}-\d{2})(?=\D|$)", basename(path))
    stamp === nothing && return nothing
    token = something(stamp[1])
    date = tryparse(Date, replace(token, "-" => ""), dateformat"yyyymmdd")
    date === nothing && @warn "the token $(token) in $(basename(path)) is not a calendar " *
                              "date; the file name does not date the run"
    return date
end

# The run's absolute origin, 00:00 UTC of the day of window 1 of the first
# binary, from every date that names it: `[input].start_date`, the first
# binary's `date` header and the date in its original (unstaged) file name.
# All present dates must agree: a disagreement would shift the output times
# and the time-varying surface fluxes, so it is an error. `nothing` when no
# date is present. Both runners call this once, right after opening the first
# binary, whether or not the run writes output; `RunContext` does the same.
function _run_time_origin(cfg, driver, original_binary_path::AbstractString)
    dates = Tuple{String, Date}[]
    start = _run_reference_time(cfg)
    start === nothing || push!(dates, ("[input].start_date", Date(start)))
    header_day = _header_date(driver)
    header_day === nothing ||
        push!(dates, ("the first transport binary's `date` header", header_day))
    filename_day = _filename_date(original_binary_path)
    filename_day === nothing ||
        push!(dates, ("the date in the first transport binary's file name " *
                      basename(original_binary_path), filename_day))
    isempty(dates) && return nothing
    label, day = first(dates)
    for (other_label, other_day) in dates[2:end]
        other_day == day || throw(ArgumentError(
            "$(label) = $(day) disagrees with $(other_label) = $(other_day)"))
    end
    return DateTime(day)
end

# CF time origin of the output: the run origin plus the windows skipped by
# `[run].start_window`. Output needs an absolute origin; a run without output
# does not.
function _runtime_output_origin(origin::Union{DateTime, Nothing}, driver;
                                start_window::Integer = 1)
    origin === nothing && throw(ArgumentError(
        "runtime output needs [input].start_date, a transport binary `date` " *
        "header, or a YYYYMMDD date in the binary filename to write the CF time origin"))
    offset_ms = round(Int64, (start_window - 1) * Float64(window_dt(driver)) * 1000)
    return origin + Millisecond(offset_ms)
end

# Run interval `(0, t_end)` in seconds since the run start; time-varying flux
# files are read only where the run can use them, and must cover it. `t_end`
# is the end of the run's windows (`run_hours`, from every binary's header)
# plus `RUN_SPAN_CLOCK_SLACK_SECONDS` (1 s) for rounding in the accumulated
# Float64 run clock, which the coverage check does not ask the files to cover;
# `_check_flux_run_span` guards against a binary that ends later anyway. `[input] end_date` adds nothing: with `folder`
# every day through it has a binary, and with `binary_paths` it is not read.
# Without `start_date` (`_flux_run_span` returns `nothing`) there are no
# time-varying fluxes to check: `validate_config` requires the date for them.
_run_time_span(run_hours::Real) = (0.0, 3600 * Float64(run_hours) + RUN_SPAN_CLOCK_SLACK_SECONDS)

_flux_run_span(cfg, run_hours::Real) =
    _run_reference_time(cfg) === nothing ? nothing : _run_time_span(run_hours)

# A binary whose windows end after the loaded flux interval would silently hold
# the last loaded slice; fail before it runs.
function _check_flux_run_span(run_span, binary_end_s::Real, surface_sources, path)
    run_span === nothing && return nothing
    any(s -> s isa TimeVaryingSurfaceFluxSource, surface_sources) || return nothing
    binary_end_s <= run_span[2] || throw(ArgumentError(
        "$(basename(path)) runs to $(binary_end_s) s after the run start, past the " *
        "$(run_span[2]) s for which time-varying surface fluxes were loaded " *
        "(the binaries' windows)."))
    return nothing
end

function _run_driven_simulation_structured(binary_paths::Vector{String}, cfg,
                                           stager::InputStager,
                                           arch::AbstractArchitecture,
                                           output_resources::RunSnapshotOutput,
                                           input_resources::RunInputResources,
                                           window_layout = nothing, binary_humidity = nothing)
    FT = _cfg_float_type(cfg)
    assert_float_type!(arch, FT)
    run_cfg = get(cfg, "run", Dict{String, Any}())
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)
    haskey(run_cfg, "reset_air_mass_each_window") &&
        throw(ArgumentError("run.reset_air_mass_each_window was replaced by " *
                            "run.air_mass_reset_mode = \"none\", " *
                            "\"preserve_vmr\", or \"preserve_tracer_mass\""))
    air_mass_reset_mode = get(run_cfg, "air_mass_reset_mode", "preserve_tracer_mass")
    physics_cadence = get(run_cfg, "physics_cadence", nothing)
    prefetch_windows = _config_bool(run_cfg, "prefetch_windows", true, "[run].prefetch_windows")
    callbacks = run_checks(cfg)

    tracer_specs = _structured_tracer_specs(cfg)

    # `stager` (rolling NVMe input staging) is created + torn down by the caller
    # `run_driven_simulation`; here we just route driver opens through it.
    # Open first driver, build recipe, validate capability, build model
    validate_replay = _config_bool(get(cfg, "input", Dict{String, Any}()), "validate_replay",
                                   false, "[input].validate_replay")
    first_driver = input_resources.driver = TransportBinaryDriver(staged_path_for!(stager, 1);
                                          FT = FT,
                                          arch = arch, validate_replay)
    _check_multifile_window_range(first_driver, start_window, stop_window_override,
                                  length(binary_paths))
    run_origin = _run_time_origin(cfg, first_driver, first(binary_paths))
    # Windows of every binary: the run length and the window ends of the schedule.
    layout = _runtime_window_layout(something(window_layout,
                                              _uniform_window_layout(first_driver, binary_paths)), FT)
    run_hours = _layout_run_hours(layout; start_window, stop_window_override)
    output_cfg = get(cfg, "output", Dict{String, Any}())
    output_spec = runtime_output_spec(output_cfg, FT; default_cap_hours = run_hours)
    snapshot_schedule_hours = snapshot_hours(output_spec)
    do_snapshots = output_enabled(output_spec)
    do_snapshots && set_time_origin!(output_spec,
        _runtime_output_origin(run_origin, first_driver; start_window))
    _check_snapshot_day_paths(output_spec, binary_paths)
    _check_output_targets(output_spec, output_cfg)
    _check_snapshot_schedule(output_spec,
                             _window_end_hours(layout; start_window, stop_window_override))
    do_snapshots && _validate_structured_output_format(output_spec.format)
    recipe = build_runtime_physics_recipe(cfg, first_driver, FT)
    _validate_capability_match(first_driver, recipe)

    # Build the stateful physics object. The recipe selected from TOML is
    # installed on `TransportModel` here, but no physics is applied yet; that
    # starts when `run_window!(sim)` or `run!(sim)` calls
    # `DrivenSimulation.step!` below.
    model = _make_structured_model(first_driver;
                                    FT = FT, recipe = recipe,
                                    tracer_specs = tracer_specs, arch = arch,
                                    start_window = start_window)
    _assert_gpu_residency!(model.state, arch)

    grid_of_first = driver_grid(first_driver)
    sampler = _install_observation_sampler!(output_resources, output_cfg, output_spec.partition,
                                            model.state, grid_of_first;
                                            cfg, binary_paths, halo_width = 0,
                                            has_humidity = _run_observation_humidity(binary_humidity, binary_paths,
                                                                                     supports_moisture(first_driver)),
                                            offset_seconds = (start_window - 1) * Float64(window_dt(first_driver)),
                                            span_seconds = 3600 * run_hours)
    # Observation sampling needs the per-window loop even without snapshots.
    # Progress lines need the windows as they finish, so step window by window.
    do_windows = do_snapshots || samples_observations(sampler) ||
                 run_logging(cfg).progress isa ProgressLines
    flux_run_span = _flux_run_span(cfg, run_hours)
    surface_sources = build_surface_flux_sources(grid_of_first, tracer_specs, FT;
                                                 reference_time = _run_reference_time(cfg),
                                                 run_span = flux_run_span)
    set_physics_attributes!(output_spec,
        runtime_physics_attributes(cfg, first_driver, surface_sources,
                                   air_mass_basis(first_driver), binary_paths))
    m0 = total_air_mass(model.state)
    tracer_masses0 = Dict(name => total_mass(model.state, name)
                          for name in tracer_names(model.state))
    source_tracers = Set(source.tracer_name for source in surface_sources)

    _log_runtime_summary(topology = :Structured,
                         mesh_label = summary(grid_of_first.horizontal),
                         levels = nlevels(grid_of_first),
                         halo_width = 0,
                         backend = architecture_label(arch),
                         FT = FT,
                         recipe = recipe,
                         driver = first_driver,
                         tracers = tracer_names(model.state),
                         binary_count = length(binary_paths),
                         snapshot_file = _output_display_path(output_spec))
    for source in surface_sources
        @info @sprintf("Surface source %s total model-storage rate: %.12e kg_air_equiv/s",
                       String(source.tracer_name), total_emission_rate(source, 0.0))
    end

    stream = _single_netcdf_stream(output_resources, output_spec, model.grid;
                                    mass_basis=air_mass_basis(first_driver))
    snapshots = AbstractSnapshotFrame[]
    day_snapshots = AbstractSnapshotFrame[]
    snapshot_count = Ref(0)
    snap_idx = 1
    total_elapsed_hours = 0.0

    # The run's window count, from every binary's layout.
    timer = RunProgressTimer(length(_window_end_hours(layout; start_window, stop_window_override));
                             reporter = run_logging(cfg).progress)
    # The timer holds process-wide compile timing until `stop_compile_timing!`;
    # release it on every exit path, including a run that throws.
    try
        function capture_structured!(hour_total)
            timed_io_write!(timer, () -> begin
                frame = capture_snapshot(model; time_hours = hour_total,
                                         fields=_snapshot_capture_fields(output_spec))
                _record_snapshot!(stream, output_spec.partition, snapshots,
                                      day_snapshots, frame)
            end)
            snapshot_count[] += 1
            return nothing
        end

        if do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
           _snapshot_due(0.0, snapshot_schedule_hours[snap_idx],
                         Float64(window_dt(first_driver)) / 3600)
            capture_structured!(0.0)
            set_progress_status!(timer;
                                 detail = @sprintf("snapshot %d at t=%.0fh",
                                                   snap_idx, 0.0),
                                 redraw = true)
            snap_idx += 1
        end

        run_time_seconds = 0.0   # accumulated across binaries (sim clock origin)
        # Observation times are absolute: a single-file run starting at a later
        # window begins `start_window - 1` windows after the run origin.
        observation_offset_seconds = (start_window - 1) * Float64(window_dt(first_driver))
        for (idx, path) in enumerate(binary_paths)
            # `path` (NAS) is kept for date labels/logging; the driver opens the
            # staged local copy when staging is enabled (idx==1 already staged above).
            driver = idx == 1 ? first_driver :
                     timed_io_read!(timer,
                         () -> TransportBinaryDriver(staged_path_for!(stager, idx);
                                                     FT = FT, arch = arch, validate_replay))
            input_resources.driver = driver
            validate_runtime_physics_recipe(recipe, driver)
            _check_multifile_window_range(driver, start_window, stop_window_override,
                                          length(binary_paths))
            stop_window = stop_window_override === nothing ?
                          total_windows(driver) : Int(stop_window_override)
            _check_flux_run_span(flux_run_span,
                                 run_time_seconds +
                                 (stop_window - start_window + 1) * Float64(window_dt(driver)),
                                 surface_sources, path)
            initialize_air_mass = idx == 1
            sim = timed_io_read!(timer,
                () -> DrivenSimulation(model, driver;
                                        start_window = start_window,
                                        stop_window = stop_window,
                                        initialize_air_mass = initialize_air_mass,
                                        air_mass_reset_mode = air_mass_reset_mode,
                                        physics_cadence = physics_cadence,
                                        prefetch_windows = prefetch_windows,
                                        callbacks = callbacks,
                                        surface_sources = surface_sources,
                                        chemistry = recipe.chemistry,
                                        # seconds since RUN start — see the CS loop
                                        # note: per-binary clock restarts replay
                                        # day-1 time-varying fluxes
                                        start_time = run_time_seconds))
            input_resources.simulation = sim
            model = sim.model
            if !initialize_air_mass
                boundary_rel = maximum(abs.(model.state.air_mass .- sim.window.air_mass)) /
                               max(maximum(abs.(sim.window.air_mass)), eps(FT))
                set_progress_status!(timer;
                                     detail = @sprintf("boundary air-mass mismatch before %s: %.3e",
                                                       basename(path), boundary_rel),
                                     redraw = true)
            end
            window_hours = Float64(window_dt(driver)) / 3600.0
            n_windows = stop_window - start_window + 1
            day_start_seconds = run_time_seconds + observation_offset_seconds
            windows_done = 0
            _begin_observation_binary!(timer, sampler, sim, path, idx, day_start_seconds)
            set_progress_status!(timer;
                                 status = @sprintf("running %s with %s on %s (%d windows)",
                                                   basename(path),
                                                   nameof(typeof(recipe.advection)),
                                                   summary(driver_grid(driver).horizontal),
                                                   n_windows),
                                 detail = "loading first window",
                                 redraw = true)
            synchronize_architecture!(arch)
            t0 = time()

            if do_windows
                for _ in 1:n_windows
                    # Physics handoff for LL/RG: `run_window!` advances through all
                    # substeps in the current met window. Each substep refreshes
                    # forcing in `DrivenSimulation.step!` and then calls the
                    # operator order documented in `TransportModel.step!`.
                    timed_transport!(timer, () -> run_window!(sim))
                    tick_window!(timer;
                                 status = @sprintf("%s window %d/%d  steps/window=%d",
                                                   basename(path),
                                                   sim.current_window_index,
                                                   stop_window,
                                                   sim.steps_per_window),
                                 detail = @sprintf("snapshots=%d  output=%s",
                                                   snapshot_count[],
                                                   _output_basename(output_spec)))
                    total_elapsed_hours += window_hours
                    windows_done += 1
                    _observe_window_end!(timer, sampler, sim,
                                         day_start_seconds + windows_done * Float64(window_dt(driver)))
                    while do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
                          _snapshot_due(total_elapsed_hours, snapshot_schedule_hours[snap_idx],
                                        window_hours)
                        capture_structured!(total_elapsed_hours)
                        set_progress_status!(timer;
                                             detail = @sprintf("snapshot %d at t=%.0fh  output=%s",
                                                               snap_idx,
                                                               total_elapsed_hours,
                                                               _output_basename(output_spec)),
                                             redraw = true)
                        snap_idx += 1
                    end
                end
            else
                # Same physics path as above, but without per-window snapshot
                # interrupts. `run!` repeatedly calls `DrivenSimulation.step!`.
                timed_transport!(timer, () -> run!(sim))
                total_elapsed_hours += n_windows * window_hours
                # `run!` doesn't tick per window; advance the bar to the
                # binary's window count in one shot.
                for local_win in start_window:stop_window
                    tick_window!(timer;
                                 status = @sprintf("%s window %d/%d  steps/window=%d",
                                                   basename(path), local_win,
                                                   stop_window,
                                                   sim.steps_per_window_schedule[local_win]),
                                 detail = "batch window accounting after run!()")
                end
            end

            run_time_seconds += (stop_window - start_window + 1) * Float64(window_dt(driver))
            synchronize_architecture!(arch)
            set_progress_status!(timer;
                                 status = @sprintf("finished %s", basename(path)),
                                 detail = @sprintf("file wall %.2fs", time() - t0),
                                 redraw = true)
            if do_snapshots && output_spec.partition isa DailyOutputFiles &&
               output_enabled(output_spec) && !isempty(day_snapshots)
                out_path = _output_path_for_partition(output_spec, output_spec.partition,
                                                       _binary_date_label(path), idx)
                grid_ref = driver_grid(first_driver)
                mb = air_mass_basis(first_driver)
                _start_daily_output!(output_resources, output_spec, out_path,
                                      day_snapshots, grid_ref, mb)
                set_progress_status!(timer;
                                     detail = @sprintf("async write %s", basename(out_path)),
                                     redraw = true)
            end
            close(input_resources)
        end

        finish_observations!(sampler)
        # Drain the last in-flight async daily write before the final flush / mass
        # accounting, so the run never returns with a write still pending.
        _wait_pending_output!(output_resources)

        if do_snapshots
            # `air_mass_basis(driver)` already returns the Symbol and has been
            # validated to match `model.state`'s basis by
            # `_check_basis_compatibility` before any step!.
            _flush_single_output!(output_spec.partition, timer, output_spec,
                                  snapshots, driver_grid(first_driver);
                                  mass_basis = air_mass_basis(first_driver))
        end

        summarize_progress!(timer)
    finally
        stop_compile_timing!(timer)
    end

    m1 = total_air_mass(model.state)
    @info @sprintf("Final air-mass change vs initial state:  %.3e", (m1 - m0) / m0)
    for name in tracer_names(model.state)
        rm0 = Float64(tracer_masses0[name])
        rm1 = total_mass(model.state, name)
        if name in source_tracers
            @info @sprintf("Final model storage for %s (with source): %.12e carrier-air kg",
                           String(name), rm1)
        elseif abs(rm0) > eps(Float64)
            @info @sprintf("Final tracer-storage drift for %s:      %.3e",
                           String(name), (rm1 - rm0) / rm0)
        else
            @info @sprintf("Final model storage for %s:             %.12e carrier-air kg",
                           String(name), rm1)
        end
    end
    return model
end

# ===========================================================================
# CS runner
# ===========================================================================

function _run_driven_simulation_cs(binary_paths::Vector{String}, cfg,
                                   stager::InputStager,
                                   arch::AbstractArchitecture,
                                   output_resources::RunSnapshotOutput,
                                   input_resources::RunInputResources,
                                   window_layout = nothing, binary_humidity = nothing)
    FT   = _cfg_float_type(cfg)
    assert_float_type!(arch, FT)

    run_cfg = get(cfg, "run", Dict{String, Any}())
    Int(get(run_cfg, "start_window", 1)) == 1 || throw(ArgumentError(
        "Cubed-sphere driven runs currently require [run] start_window=1."))
    advection = build_runtime_advection(cfg, CubedSphereRuntimeRecipeStyle())
    Hp = configured_halo_width(cfg, advection)
    stop_window_override = get(run_cfg, "stop_window", nothing)
    haskey(run_cfg, "reset_air_mass_each_window") &&
        throw(ArgumentError("run.reset_air_mass_each_window was replaced by " *
                            "run.air_mass_reset_mode = \"none\", " *
                            "\"preserve_vmr\", or \"preserve_tracer_mass\""))
    air_mass_reset_mode = get(run_cfg, "air_mass_reset_mode", "preserve_tracer_mass")
    physics_cadence = get(run_cfg, "physics_cadence", nothing)
    prefetch_windows = _config_bool(run_cfg, "prefetch_windows", true, "[run].prefetch_windows")
    callbacks = run_checks(cfg)

    tracers_cfg = get(cfg, "tracers", Dict{String, Any}())
    isempty(tracers_cfg) && error("[tracers] must define at least one tracer")
    # Use the same tracer-spec parser as the LL/RG runner so
    # `[tracers.*.surface_flux]` blocks are picked up. Plain inline-Dict
    # parsing (the previous implementation) silently dropped surface_flux
    # configs and produced zero fossil emissions on CS.
    tracer_specs = _parse_tracer_specs(cfg)
    tracer_specs === nothing &&
        error("[tracers] section is malformed; expected per-tracer subsections")
    tracer_init = Dict(spec.name => spec.init_cfg for spec in tracer_specs)

    # `stager` (rolling NVMe input staging) is created + torn down by the caller
    # `run_driven_simulation`; here we just route driver opens through it.
    # First driver + model (reuses air_mass from window 1)
    validate_replay = _config_bool(get(cfg, "input", Dict{String, Any}()), "validate_replay",
                                   false, "[input].validate_replay")
    driver1 = input_resources.driver = TransportBinaryDriver(staged_path_for!(stager, 1);
                                    FT = FT, arch = arch, Hp = Hp, validate_replay)
    run_origin = _run_time_origin(cfg, driver1, first(binary_paths))
    layout = _runtime_window_layout(something(window_layout,
                                              _uniform_window_layout(driver1, binary_paths)), FT)
    run_hours = _layout_run_hours(layout; stop_window_override)
    output_cfg = get(cfg, "output", Dict{String, Any}())
    output_spec = runtime_output_spec(output_cfg, FT; default_cap_hours = run_hours)
    snapshot_schedule_hours = snapshot_hours(output_spec)
    do_snapshots = output_enabled(output_spec)
    do_snapshots && set_time_origin!(output_spec, _runtime_output_origin(run_origin, driver1))
    _check_snapshot_day_paths(output_spec, binary_paths)
    _check_output_targets(output_spec, output_cfg)
    _check_snapshot_schedule(output_spec, _window_end_hours(layout; stop_window_override))
    if stop_window_override !== nothing && length(binary_paths) > 1 &&
       Int(stop_window_override) < total_windows(driver1)
        throw(ArgumentError(
            "[run] stop_window=$(stop_window_override) with multiple cubed-sphere " *
            "daily binaries would carry state from a partial day into the next " *
            "day's 00Z forcing. Use one binary for partial-window debugging, or " *
            "omit stop_window for a continuous multi-day run."))
    end
    recipe  = build_runtime_physics_recipe(cfg, driver1, FT; halo_width = Hp)
    _validate_capability_match(driver1, recipe)

    grid    = driver_grid(driver1)
    mesh    = grid.horizontal
    window1 = load_transport_window(driver1, 1)
    air_mass = window1.air_mass
    Nz = size(air_mass[1], 3)

    # Honor the binary's mass_basis — `DrivenSimulation._check_basis_compatibility`
    # compares `mass_basis(model.state/fluxes)` against `air_mass_basis(driver)`
    # and throws if they diverge. Hardcoding `DryBasis` would trip a
    # runtime ArgumentError on any moist-basis CS binary.
    basis_sym = air_mass_basis(driver1)
    BasisT    = basis_sym === :dry   ? DryBasis   :
                basis_sym === :moist ? MoistBasis :
                error("CS binary has unsupported mass_basis $(basis_sym); expected :dry or :moist")

    # CS tracers flow through the unified IC pipeline.
    # DryBasis is the default runtime basis; MoistBasis requires qv from
    # window 1 to convert the dry IC VMR, which CS windows do not carry —
    # so moist binaries error explicitly here rather than producing
    # silently wrong tracer mass.
    basis_sym === :moist &&
        error("CS driven runner does not yet support moist-basis binaries: " *
              "`pack_initial_tracer_mass` needs qv, which canonical CS v4 " *
              "payloads do not contain. Regenerate the binary on dry basis " *
              "(`regrid_ll_transport_binary_to_cs.jl --mass-basis dry`), " *
              "or extend the CS window + this runner to thread qv.")

    qv_initial = _initial_condition_qv(window1)
    pressure_basis = surface_pressure_basis(driver1)
    state = _initialize_cs_dry_state(
        grid, air_mass, tracer_init;
        surface_pressure = window1.surface_pressure, pressure_basis, qv = qv_initial)
    fluxes = _allocate_cs_runner_fluxes(mesh, Nz, FT, BasisT)
    # Workspace constructors allocate with `similar(state.air_mass[1])`.
    # Move prognostic storage first so scratch is created on the target backend,
    # avoiding temporary CPU workspaces and copying their unused contents.
    adaptor = array_adapter(arch)
    if adaptor !== Array
        state = Base.invokelatest(Adapt.adapt, adaptor, state)
        fluxes = Base.invokelatest(Adapt.adapt, adaptor, fluxes)
    end

    # Build the CS physics object. The recipe-selected operators are installed
    # on the model here; the kernels start running later in the `step!(sim)`
    # loop after `DrivenSimulation` has loaded/refreshed each forcing window.
    model = TransportModel(state, fluxes, grid, recipe.advection;
                            binary_cfl_check = recipe.binary_cfl_check,
                            diffusion  = recipe.diffusion,
                            convection = recipe.convection)
    # Adapt remaining operator and geometry metadata. Device state and scratch
    # already live on the selected backend and are retained by Adapt.
    if adaptor !== Array
        model  = Base.invokelatest(Adapt.adapt, adaptor, model)
        state  = model.state                           # rebind post-adapt
        fluxes = model.fluxes
    end
    _assert_gpu_residency!(model.state, arch)
    sampler = _install_observation_sampler!(output_resources, output_cfg, output_spec.partition,
                                            model.state, grid;
                                            cfg, binary_paths, halo_width = Hp,
                                            has_humidity = _run_observation_humidity(binary_humidity, binary_paths,
                                                                                     _window_humidity(window1, true) !== nothing),
                                            span_seconds = 3600 * run_hours)
    observation_humidity = _cs_observation_humidity(sampler, model.state, binary_paths)
    observation_clock_seconds = 0.0   # exact Float64 run clock for observation times

    # Build surface-flux sources from the parsed tracer specs and log per-source
    # mass rates. Matches the LL/RG path; `DrivenSimulation`'s constructor
    # adapts these to the model backend (CPU Array or GPU array) via
    # `_adapt_sources_to_model_backend`, so no manual adapt step here.
    flux_run_span = _flux_run_span(cfg, run_hours)
    surface_sources = build_surface_flux_sources(grid, tracer_specs, FT;
                                                 reference_time = _run_reference_time(cfg),
                                                 run_span = flux_run_span)
    set_physics_attributes!(output_spec,
        runtime_physics_attributes(cfg, driver1, surface_sources,
                                   air_mass_basis(driver1), binary_paths))
    source_tracers = Set(source.tracer_name for source in surface_sources)
    for source in surface_sources
        # Global storage rate at the run start, by dispatch on the source type.
        @info @sprintf("Surface source %s total model-storage rate: %.12e kg_air_equiv/s",
                       String(source.tracer_name), total_emission_rate(source, 0.0))
    end

    _log_runtime_summary(topology = :CubedSphere,
                         mesh_label = @sprintf("C%d", mesh.Nc),
                         levels = Nz,
                         halo_width = Hp,
                         backend = architecture_label(arch),
                         FT = FT,
                         recipe = recipe,
                         driver = driver1,
                         tracers = keys(tracer_init),
                         binary_count = length(binary_paths),
                         snapshot_file = _output_display_path(output_spec))

    # NetCDF captures selected fields; ATMSNAP retains full native state.
    # The run owns the stream and any background daily write.
    stream = _single_netcdf_stream(output_resources, output_spec, model.grid;
                                    mass_basis=air_mass_basis(driver1))
    snapshots = AbstractSnapshotFrame[]
    day_snapshots = AbstractSnapshotFrame[]
    snapshot_count = Ref(0)

    # Progress + IO/Transport timer over the run's windows, from every binary's layout.
    timer = RunProgressTimer(length(_window_end_hours(layout; stop_window_override));
                             reporter = run_logging(cfg).progress)
    # The timer holds process-wide compile timing until `stop_compile_timing!`;
    # release it on every exit path, including a run that throws.
    try
        function capture_cs!(hour_total)
            timed_io_write!(timer, () -> begin
                frame = capture_snapshot(model; time_hours = hour_total,
                                         halo_width = Hp,
                                         fields=_snapshot_capture_fields(output_spec))
                _record_snapshot!(stream, output_spec.partition, snapshots,
                                      day_snapshots, frame)
            end)
            snapshot_count[] += 1
            return nothing
        end

        snap_idx = 1
        total_hour = 0.0
        if do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
           _snapshot_due(0.0, snapshot_schedule_hours[snap_idx], Float64(window_dt(driver1)) / 3600)
            capture_cs!(0.0)
            set_progress_status!(timer;
                                 detail = @sprintf("snapshot %d at t=%.1fh",
                                                   snap_idx, 0.0),
                                 redraw = true)
            snap_idx += 1
        end

        t0 = time()
        # The previous binary's simulation: the next one refills its window
        # buffers when the layouts agree (`reuse_buffers_of`).
        previous_sim = nothing
        for (driver_idx, path) in enumerate(binary_paths)
            # `path` (NAS) kept for labels/logging; driver opens the staged local
            # copy when staging is enabled (driver_idx==1 already staged above).
            driver = driver_idx == 1 ? driver1 :
                     timed_io_read!(timer,
                         () -> TransportBinaryDriver(staged_path_for!(stager, driver_idx);
                                                      FT = FT, arch = arch, Hp = Hp, validate_replay))
            input_resources.driver = driver
            validate_runtime_physics_recipe(recipe, driver; halo_width = Hp)
            _check_multifile_window_range(driver, 1, stop_window_override,
                                          length(binary_paths))
            stop_window = stop_window_override === nothing ?
                          total_windows(driver) :
                          Int(stop_window_override)
            window_hours = window_dt(driver) / 3600.0
            _check_flux_run_span(flux_run_span,
                                 total_hour * 3600.0 + stop_window * Float64(window_dt(driver)),
                                 surface_sources, path)

            # Keep state, flux arrays, and numerical workspaces across files.
            # The new simulation refreshes forcing, diffusion geometry, and caches.
            initialize_air_mass = driver_idx == 1
            # Hand over the previous binary's window buffers only when they fit;
            # otherwise drop them first, so the new ones need not coexist.
            reusable_sim = previous_sim !== nothing &&
                           same_window_layout(previous_sim.driver, driver) ? previous_sim : nothing
            previous_sim = nothing
            sim = timed_io_read!(timer,
                () -> DrivenSimulation(model, driver;
                                        start_window = 1, stop_window = stop_window,
                                        initialize_air_mass = initialize_air_mass,
                                        air_mass_reset_mode = air_mass_reset_mode,
                                        physics_cadence = physics_cadence,
                                        prefetch_windows = prefetch_windows,
                                        callbacks = callbacks,
                                        surface_sources = surface_sources,
                                        # accumulated run time: time-varying surface
                                        # sources index emission slices in seconds
                                        # since RUN start; restarting at 0 per day
                                        # replays day-1 fluxes (the +1 Pg/month
                                        # co2_natural surplus).
                                        chemistry = recipe.chemistry,
                                        reuse_buffers_of = reusable_sim,
                                        start_time = total_hour * 3600.0))
            # `DrivenSimulation` may wrap `model` with a surface-flux operator;
            # keep snapshots and the return value aligned with the stepped model.
            input_resources.simulation = sim
            model = sim.model
            day_start_seconds = observation_clock_seconds
            windows_done = 0
            _begin_observation_binary!(timer, sampler, sim, path, driver_idx, day_start_seconds;
                                       humidity = observation_humidity)

            day_t0 = time()
            set_progress_status!(timer;
                                 status = @sprintf("running %s (%d windows)",
                                                   basename(path), stop_window),
                                 detail = @sprintf("schedule max=%d current=%d",
                                                   maximum(sim.steps_per_window_schedule),
                                                   sim.steps_per_window),
                                 redraw = true)
            while sim.iteration < sim.final_iteration
                # Physics handoff for CS: one `step!(sim)` is one runtime substep.
                # `DrivenSimulation.step!` refreshes window forcing, then delegates
                # to `TransportModel.step!` or to the binary-scheduled
                # the binary-scheduled window composition. See
                # `driven_stepping.jl` for the exact operator order.
                timed_transport!(timer, () -> step!(sim))
                if sim.iteration == sim.current_window_end_iteration
                    tick_window!(timer;
                                 status = @sprintf("%s window %d/%d  steps/window=%d",
                                                   basename(path),
                                                   sim.current_window_index,
                                                   stop_window,
                                                   sim.steps_per_window),
                                 detail = @sprintf("snapshots=%d  output=%s",
                                                   snapshot_count[],
                                                   _output_basename(output_spec)))
                    total_hour += window_hours
                    windows_done += 1
                    _observe_window_end!(timer, sampler, sim,
                                         day_start_seconds + windows_done * Float64(window_dt(driver));
                                         humidity = observation_humidity, binary_index = driver_idx)
                    while do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
                          _snapshot_due(total_hour, snapshot_schedule_hours[snap_idx], window_hours)
                        capture_cs!(total_hour)
                        set_progress_status!(timer;
                                             detail = @sprintf("snapshot %d at t=%.1fh  output=%s",
                                                               snap_idx,
                                                               total_hour,
                                                               _output_basename(output_spec)),
                                             redraw = true)
                        snap_idx += 1
                    end
                end
            end
            set_progress_status!(timer;
                                 status = @sprintf("finished %s", basename(path)),
                                 detail = @sprintf("file wall %.1fs", time() - day_t0),
                                 redraw = true)
            observation_clock_seconds += windows_done * Float64(window_dt(driver))
            if do_snapshots && output_spec.partition isa DailyOutputFiles &&
               output_enabled(output_spec) && !isempty(day_snapshots)
                out_path = _output_path_for_partition(output_spec, output_spec.partition,
                                                       _binary_date_label(path), driver_idx)
                grid_ref = grid
                mb = BasisT === DryBasis ? :dry : :moist
                _start_daily_output!(output_resources, output_spec, out_path,
                                      day_snapshots, grid_ref, mb)
                set_progress_status!(timer;
                                     detail = @sprintf("async write %s", basename(out_path)),
                                     redraw = true)
            end
            close(input_resources)
            previous_sim = sim
        end

        finish_observations!(sampler)
        # Drain the last in-flight async daily write before the final mass accounting,
        # so the run never returns with a write still pending.
        _wait_pending_output!(output_resources)

        @info @sprintf("Done: %.1fs  (%d snapshots, final t=%.1fh)",
                       time() - t0, snapshot_count[], total_hour)

        for name in keys(tracer_init)
            rm1 = total_mass(state, name)
            if name in source_tracers
                @info @sprintf("  %s total mass (with source): %.6e kg", name, rm1)
            else
                @info @sprintf("  %s total mass:               %.6e kg", name, rm1)
            end
        end

        if do_snapshots
            # BasisT was bound at model construction (dry by default on CS);
            # reuse it so the NetCDF records the same basis the `air_mass`
            # arrays were stored under.
            _flush_single_output!(output_spec.partition, timer, output_spec,
                                  snapshots, grid;
                                  mass_basis = BasisT === DryBasis ? :dry : :moist)
        end
        summarize_progress!(timer)
    finally
        stop_compile_timing!(timer)
    end
    return model
end

end # module DrivenRunner
