"""
    DrivenRunner

Library-level entry point for the driven transport runtime.

The canonical CLI, `scripts/run_transport.jl`, is a thin wrapper over
`run_driven_simulation(cfg)`. The library function handles LL/RG and CS
runtime flows with dispatch driven by the first binary's header
(`inspect_binary(first_path).grid_type`). Historical runner names live under
`scripts/deprecated/` only for reference.

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
   for CS. Those functions live in `DrivenSimulation.jl`.
5. `DrivenSimulation.step!` refreshes time-varying forcing from the driver,
   then calls `TransportModel.step!` or, for binary-scheduled substeps,
   `transport_step!` plus an end-of-window `convection_chemistry_step!`.
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
using ..Models: _config_bool
using Dates: Date, DateTime
using Printf: @sprintf, @printf
using Logging
using ProgressMeter: Progress, next!, finish!, update!

import ...expand_data_path
using ...SectionTimer
using ..State: AbstractMassBasis, DryBasis, MoistBasis, CellState,
                CubedSphereState, total_air_mass, total_mass, tracer_names,
                tracer_index, get_tracer
using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, nlevels
using ..Operators: LinRoodPPMScheme, PPMScheme, SlopesScheme, UpwindScheme,
                  ImplicitVerticalDiffusion,
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
                     steps_per_window_schedule, release_payload!
using ..InitialConditionIO: build_initial_mixing_ratio, _build_cs_initial_mixing_ratio,
                             pack_initial_tracer_mass, _cs_pack_interior_into_halo!,
                             build_surface_flux_sources
using ..BinaryPathExpander: expand_binary_paths
using ..InputStaging: InputStager, staged_path_for!, cleanup_staging!
using ..Output: AbstractSnapshotFrame, NetCDFSnapshotStream, append_snapshot!, SnapshotFrame,
                AbstractOutputPartition, SingleOutputFile, DailyOutputFiles,
                RuntimeOutputSpec, runtime_output_spec, snapshot_hours,
                output_enabled, output_path, output_path_for_day,
                capture_snapshot, write_snapshot_netcdf, write_snapshot_binary,
                NoObservationOutput, ObservationOutputSpec, observation_output_spec,
                OBSERVATION_RUNTIME_UNAVAILABLE_MESSAGE, _NETCDF_IO_LOCK,
                AbstractObservationSampler, NoObservationSampler, build_observation_sampler
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
                 CubedSphereRuntimeRecipeStyle

export run_driven_simulation, validate_config, TransportTracerSpec

include("runner/progress.jl")
include("runner/configuration.jl")
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
    return Base.invokelatest(_run_driven_simulation, cfg, arch)
end

function _run_driven_simulation(cfg::AbstractDict, arch::AbstractArchitecture)
    input_cfg = get(cfg, "input", Dict{String, Any}())
    binary_paths = expand_binary_paths(input_cfg)
    isempty(binary_paths) &&
        throw(ArgumentError("[input] resolved to an empty binary list"))
    # Section timing instrumentation, off unless ATMOSTR_TIMERS=1.
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
    # (binary header owns topology, TOML owns physics kinds). The
    # capability probe also runs the load-time
    # gates (stale-binary, cm-continuity) as a side effect of opening
    # the reader in `inspect_binary`.
    binary_caps = [(path = path, caps = inspect_binary(path; io = devnull))
                   for path in binary_paths]
    for item in binary_caps
        _log_binary_summary(item.path, item.caps)
    end
    caps = first(binary_caps).caps
    _validate_input_binary_expectations(caps, input_cfg, first(binary_paths))
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
                                           stager, arch, output_resources, input_resources)
            end
        end
    finally
        cleanup_staging!(stager)
    end
end

_run_driven_simulation_for(::Val{:latlon}, binary_paths::Vector{String}, cfg,
                           stager::InputStager, arch::AbstractArchitecture,
                           output_resources::RunSnapshotOutput, input_resources::RunInputResources) =
    _run_driven_simulation_structured(binary_paths, cfg, stager, arch, output_resources, input_resources)
_run_driven_simulation_for(::Val{:reduced_gaussian}, binary_paths::Vector{String}, cfg,
                           stager::InputStager, arch::AbstractArchitecture,
                           output_resources::RunSnapshotOutput, input_resources::RunInputResources) =
    _run_driven_simulation_structured(binary_paths, cfg, stager, arch, output_resources, input_resources)
_run_driven_simulation_for(::Val{:cubed_sphere}, binary_paths::Vector{String}, cfg,
                           stager::InputStager, arch::AbstractArchitecture,
                           output_resources::RunSnapshotOutput, input_resources::RunInputResources) =
    _run_driven_simulation_cs(binary_paths, cfg, stager, arch, output_resources, input_resources)
function _run_driven_simulation_for(::Val{grid_type}, _binary_paths::Vector{String},
                                    _cfg, _stager::InputStager, _arch, _output_resources::RunSnapshotOutput, _input_resources::RunInputResources) where grid_type
    throw(ArgumentError("Unsupported transport-binary grid_type=$(grid_type)."))
end

# Parse the run start as a `DateTime` (at 00:00) from `[input].start_date`,
# used as the `reference_time` for time-varying surface fluxes so their slice
# times align to the simulation clock. Returns `nothing` if no start_date is
# present (the time-varying loader then assumes the file origin == run start).
function _run_reference_time(cfg)
    input_cfg = get(cfg, "input", nothing)
    input_cfg isa AbstractDict || return nothing
    haskey(input_cfg, "start_date") || return nothing
    return DateTime(Date(String(input_cfg["start_date"])))
end

# Reduce a surface source's per-cell rate to a scalar total for logging,
# handling both static (`cell_mass_rate`) and time-varying
# (`cell_mass_rate_series`, summed over the first slice) sources.
function _surface_source_total_rate(source)
    if hasproperty(source, :cell_mass_rate)
        r = source.cell_mass_rate
        return r isa Tuple ? Float64(sum(sum, r)) : Float64(sum(r))
    else
        series = source.cell_mass_rate_series   # NTuple{6} of (Nc,Nc,ntime)
        return Float64(sum(p -> sum(@view p[:, :, 1]), series))
    end
end

function _run_driven_simulation_structured(binary_paths::Vector{String}, cfg,
                                           stager::InputStager,
                                           arch::AbstractArchitecture,
                                           output_resources::RunSnapshotOutput,
                                           input_resources::RunInputResources)
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

    init_cfg = get(cfg, "init", Dict{String, Any}())
    tracer_specs = something(_parse_tracer_specs(cfg),
                             (TransportTracerSpec(Symbol(get(run_cfg, "tracer_name", "CO2")),
                                                  _copy_cfg_dict(init_cfg),
                                                  Dict{String, Any}()),))

    # `stager` (rolling NVMe input staging) is created + torn down by the caller
    # `run_driven_simulation`; here we just route driver opens through it.
    # Open first driver, build recipe, validate capability, build model
    first_driver = input_resources.driver = TransportBinaryDriver(staged_path_for!(stager, 1);
                                          FT = FT,
                                          arch = arch)
    _check_multifile_window_range(first_driver, start_window, stop_window_override,
                                  length(binary_paths))
    output_cfg = get(cfg, "output", Dict{String, Any}())
    output_spec = runtime_output_spec(output_cfg, FT;
                                      default_cap_hours = _output_default_cap_hours(
                                          first_driver, length(binary_paths);
                                          start_window = start_window,
                                          stop_window_override = stop_window_override))
    snapshot_schedule_hours = snapshot_hours(output_spec)
    do_snapshots = output_enabled(output_spec)
    _install_observation_sampler!(output_resources, output_cfg, output_spec.partition)
    recipe = build_runtime_physics_recipe(cfg, first_driver, FT)
    _validate_capability_match(first_driver, recipe)

    # Build the stateful physics object. The recipe selected from TOML is
    # installed on `TransportModel` here, but no physics is applied yet; that
    # starts when `run_window!(sim)` or `run!(sim)` calls
    # `DrivenSimulation.step!` below.
    model = _make_structured_model(first_driver;
                                    FT = FT, recipe = recipe,
                                    tracer_specs = tracer_specs, arch = arch)
    _assert_gpu_residency!(model.state, arch)

    grid_of_first = driver_grid(first_driver)
    surface_sources = build_surface_flux_sources(grid_of_first, tracer_specs, FT;
                                                 reference_time = _run_reference_time(cfg))
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
                       String(source.tracer_name),
                       _surface_source_total_rate(source))
    end

    stream = _single_netcdf_stream(output_resources, output_spec, model.grid;
                                    mass_basis=air_mass_basis(first_driver))
    snapshots = AbstractSnapshotFrame[]
    day_snapshots = AbstractSnapshotFrame[]
    snapshot_count = Ref(0)
    snap_idx = 1
    total_elapsed_hours = 0.0

    # Estimate total windows for the progress bar. Each daily binary has
    # the same window count for a homogeneous run; multiplying gives a
    # close-enough total. Use min(stop_window_override, ...) when set.
    per_binary = stop_window_override === nothing ?
                 total_windows(first_driver) - start_window + 1 :
                 Int(stop_window_override) - start_window + 1
    timer = RunProgressTimer(per_binary * length(binary_paths))

    function capture_structured!(hour_total)
        timed_io_write!(timer, () -> begin
            frame = capture_snapshot(model; time_hours = hour_total,
                                     fields=output_spec.format === :netcdf ? output_spec.fields : nothing)
            _record_snapshot!(stream, output_spec.partition, snapshots,
                                  day_snapshots, frame)
        end)
        snapshot_count[] += 1
        return nothing
    end

    if do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
       abs(snapshot_schedule_hours[snap_idx]) < 0.5
        capture_structured!(0.0)
        set_progress_status!(timer;
                             detail = @sprintf("snapshot %d at t=%.0fh",
                                               snap_idx, 0.0),
                             redraw = true)
        snap_idx += 1
    end

    run_time_seconds = 0.0   # accumulated across binaries (sim clock origin)
    for (idx, path) in enumerate(binary_paths)
        # `path` (NAS) is kept for date labels/logging; the driver opens the
        # staged local copy when staging is enabled (idx==1 already staged above).
        driver = idx == 1 ? first_driver :
                 timed_io_read!(timer,
                     () -> TransportBinaryDriver(staged_path_for!(stager, idx);
                                                 FT = FT, arch = arch))
        input_resources.driver = driver
        validate_runtime_physics_recipe(recipe, driver)
        _check_multifile_window_range(driver, start_window, stop_window_override,
                                      length(binary_paths))
        stop_window = stop_window_override === nothing ?
                      total_windows(driver) : Int(stop_window_override)
        initialize_air_mass = idx == 1
        sim = timed_io_read!(timer,
            () -> DrivenSimulation(model, driver;
                                    start_window = start_window,
                                    stop_window = stop_window,
                                    initialize_air_mass = initialize_air_mass,
                                    air_mass_reset_mode = air_mass_reset_mode,
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

        if do_snapshots
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
                while snap_idx <= length(snapshot_schedule_hours) &&
                      abs(total_elapsed_hours - snapshot_schedule_hours[snap_idx]) < 0.5
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

    m1 = total_air_mass(model.state)
    @info @sprintf("Final air-mass change vs initial state:  %.3e", (m1 - m0) / m0)
    for name in tracer_names(model.state)
        rm0 = Float64(tracer_masses0[name])
        rm1 = Float64(total_mass(model.state, name))
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
                                   input_resources::RunInputResources)
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
    driver1 = input_resources.driver = TransportBinaryDriver(staged_path_for!(stager, 1);
                                    FT = FT, arch = arch, Hp = Hp)
    output_cfg = get(cfg, "output", Dict{String, Any}())
    output_spec = runtime_output_spec(output_cfg, FT;
                                      default_cap_hours = _output_default_cap_hours(
                                          driver1, length(binary_paths);
                                          stop_window_override = stop_window_override))
    snapshot_schedule_hours = snapshot_hours(output_spec)
    do_snapshots = output_enabled(output_spec)
    _install_observation_sampler!(output_resources, output_cfg, output_spec.partition)
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
    # DryBasis is the default per invariant 14; MoistBasis
    # requires qv from window1 (feedback_vmr_to_mass_basis_aware), which
    # CS windows do not carry today — so moist binaries error explicitly
    # here rather than producing silently wrong tracer mass.
    basis_sym === :moist &&
        error("CS driven runner does not yet support moist-basis binaries: " *
              "`pack_initial_tracer_mass` needs qv, which canonical CS v4 " *
              "payloads do not contain. Regenerate the binary on dry basis " *
              "(`regrid_ll_transport_binary_to_cs.jl --mass-basis dry`), " *
              "or extend the CS window + this runner to thread qv.")

    state = _initialize_cs_dry_state(grid, air_mass, tracer_init;
                                     surface_pressure = window1.surface_pressure)
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

    # Build surface-flux sources from the parsed tracer specs and log per-source
    # mass rates. Matches the LL/RG path; `DrivenSimulation`'s constructor
    # adapts these to the model backend (CPU Array or GPU array) via
    # `_adapt_sources_to_model_backend`, so no manual adapt step here.
    surface_sources = build_surface_flux_sources(grid, tracer_specs, FT;
                                                 reference_time = _run_reference_time(cfg))
    source_tracers = Set(source.tracer_name for source in surface_sources)
    for source in surface_sources
        # Reduce the topology-shaped per-cell rate to a scalar for the log;
        # `_surface_source_total_rate` handles static (2D / 6-tuple) and
        # time-varying (first-slice) sources alike.
        @info @sprintf("Surface source %s total model-storage rate: %.12e kg_air_equiv/s",
                       String(source.tracer_name), _surface_source_total_rate(source))
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

    # Progress + IO/Transport timer. Estimate total windows from the
    # first driver; closes to the truth on homogeneous daily runs.
    per_binary_estimate = stop_window_override === nothing ?
                          total_windows(driver1) :
                          min(Int(stop_window_override), total_windows(driver1))
    timer = RunProgressTimer(per_binary_estimate * length(binary_paths))

    function capture_cs!(hour_total)
        timed_io_write!(timer, () -> begin
            frame = capture_snapshot(model; time_hours = hour_total,
                                     halo_width = Hp,
                                     fields=output_spec.format === :netcdf ? output_spec.fields : nothing)
            _record_snapshot!(stream, output_spec.partition, snapshots,
                                  day_snapshots, frame)
        end)
        snapshot_count[] += 1
        return nothing
    end

    snap_idx = 1
    total_hour = 0.0
    if do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
       abs(snapshot_schedule_hours[snap_idx]) < 0.5
        capture_cs!(0.0)
        set_progress_status!(timer;
                             detail = @sprintf("snapshot %d at t=%.1fh",
                                               snap_idx, 0.0),
                             redraw = true)
        snap_idx += 1
    end

    t0 = time()
    for (driver_idx, path) in enumerate(binary_paths)
        # `path` (NAS) kept for labels/logging; driver opens the staged local
        # copy when staging is enabled (driver_idx==1 already staged above).
        driver = driver_idx == 1 ? driver1 :
                 timed_io_read!(timer,
                     () -> TransportBinaryDriver(staged_path_for!(stager, driver_idx);
                                                  FT = FT, arch = arch, Hp = Hp))
        input_resources.driver = driver
        validate_runtime_physics_recipe(recipe, driver; halo_width = Hp)
        _check_multifile_window_range(driver, 1, stop_window_override,
                                      length(binary_paths))
        stop_window = stop_window_override === nothing ?
                      total_windows(driver) :
                      min(Int(stop_window_override), total_windows(driver))
        window_hours = window_dt(driver) / 3600.0

        # Keep state, flux arrays, and numerical workspaces across files.
        # The new simulation refreshes forcing, diffusion geometry, and caches.
        initialize_air_mass = driver_idx == 1
        sim = timed_io_read!(timer,
            () -> DrivenSimulation(model, driver;
                                    start_window = 1, stop_window = stop_window,
                                    initialize_air_mass = initialize_air_mass,
                                    air_mass_reset_mode = air_mass_reset_mode,
                                    surface_sources = surface_sources,
                                    # accumulated run time: time-varying surface
                                    # sources index emission slices in seconds
                                    # since RUN start; restarting at 0 per day
                                    # replays day-1 fluxes (the +1 Pg/month
                                    # co2_natural surplus). (`callbacks` /
                                    # reference-cadence is an anomaly-ref feature
                                    # not on this branch, so omitted here.)
                                    chemistry = recipe.chemistry,
                                    start_time = total_hour * 3600.0))
        # `DrivenSimulation` may wrap `model` with a surface-flux operator;
        # keep snapshots and the return value aligned with the stepped model.
        input_resources.simulation = sim
        model = sim.model

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
            # `transport_step!` + end-of-window `convection_chemistry_step!`
            # split. See those functions for the actual operator order.
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
                while do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
                      abs(total_hour - snapshot_schedule_hours[snap_idx]) < 0.5
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
    end

    # Drain the last in-flight async daily write before the final mass accounting,
    # so the run never returns with a write still pending.
    _wait_pending_output!(output_resources)

    @info @sprintf("Done: %.1fs  (%d snapshots, final t=%.1fh)",
                   time() - t0, snapshot_count[], total_hour)

    for name in keys(tracer_init)
        rm1 = Float64(total_mass(state, name))
        if name in source_tracers
            @info @sprintf("  %s total mass (with source): %.6e kg", name, rm1)
        else
            @info @sprintf("  %s total mass:               %.6e kg", name, rm1)
        end
    end

    if do_snapshots
        # BasisT was bound at model construction (dry by default on CS per
        # invariant 14); reuse it so the NetCDF records the same basis the
        # `air_mass` arrays were stored under.
        _flush_single_output!(output_spec.partition, timer, output_spec,
                              snapshots, grid;
                              mass_basis = BasisT === DryBasis ? :dry : :moist)
    end
    summarize_progress!(timer)
    return model
end

end # module DrivenRunner
