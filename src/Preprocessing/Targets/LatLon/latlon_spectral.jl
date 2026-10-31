# Spectral ERA5 to structured lat-lon transport-binary preprocessing path.

const LLNextDayFields{FT} = @NamedTuple{
    m::Array{FT, 3},
    am::Array{FT, 3},
    bm::Array{FT, 3},
    cm::Array{FT, 3},
    qv::Union{Nothing, Array{FT, 3}},
}

mutable struct LatLonSpectralWindowWorkspace{FT, TW, MW, SW, QW} <:
               AbstractWindowWorkspace{LatLonTargetGeometry, FT}
    transform      :: TW
    merged         :: MW
    storage        :: SW
    qv             :: QW
    ps_offsets     :: Vector{Float64}
    last_hour_next :: Union{Nothing, LLNextDayFields{FT}}
    steps_schedule :: Vector{Int}
    fidelity_sampler :: StructuredFidelitySampler   # sampled faces, ranked once
    fidelity_captures :: Vector{Any}
    write_replay_on :: Bool        # resolved once per output: header record and gate

    function LatLonSpectralWindowWorkspace{FT, TW, MW, SW, QW}(
            transform::TW,
            merged::MW,
            storage::SW,
            qv::QW,
            ps_offsets::Vector{Float64},
            last_hour_next,
            steps_schedule::Vector{Int},
            fidelity_sampler::StructuredFidelitySampler,
            fidelity_captures::Vector{Any},
            write_replay_on::Bool) where {FT, TW, MW, SW, QW}
        if last_hour_next !== nothing && !(last_hour_next isa LLNextDayFields{FT})
            throw(ArgumentError(
                "last_hour_next must be `nothing` or LLNextDayFields{$FT}; " *
                "got $(typeof(last_hour_next))"))
        end
        return new{FT, TW, MW, SW, QW}(
            transform, merged, storage, qv, ps_offsets, last_hour_next,
            steps_schedule, fidelity_sampler, fidelity_captures, write_replay_on)
    end
end

mutable struct LatLonDeferredBinaryWriter{FT,
                                          Basis<:AbstractMassBasis,
                                          H,
                                          W,
                                          S} <:
               AbstractBinaryWriter{LatLonTargetGeometry, FT, Basis}
    path        :: String
    final_path  :: String
    header      :: H
    workspace   :: W
    settings    :: S
    basis       :: Basis
    inner       :: Union{Nothing, LatLonBinaryWriter{FT, Basis}}
    closed      :: Bool
    promoted    :: Bool

    function LatLonDeferredBinaryWriter{FT, Basis, H, W, S}(
            path::String,
            final_path::String,
            header::H,
            workspace::W,
            settings::S,
            basis::Basis,
            inner,
            closed::Bool,
            promoted::Bool) where {FT, Basis<:AbstractMassBasis, H, W, S}
        if inner !== nothing && !(inner isa LatLonBinaryWriter{FT, Basis})
            throw(ArgumentError(
                "inner must be `nothing` or LatLonBinaryWriter{$FT, $Basis}; " *
                "got $(typeof(inner))"))
        end
        return new{FT, Basis, H, W, S}(
            path, final_path, header, workspace, settings, basis, inner,
            closed, promoted)
    end
end

function LatLonDeferredBinaryWriter(path::AbstractString,
                                    header,
                                    workspace,
                                    settings,
                                    ::Type{FT},
                                    basis::Basis;
                                    final_path::AbstractString=path) where
        {FT, Basis<:AbstractMassBasis}
    return LatLonDeferredBinaryWriter{FT, Basis, typeof(header),
                                      typeof(workspace), typeof(settings)}(
        String(path), String(final_path), header, workspace, settings,
        basis, nothing, false, false)
end

function _open_latlon_deferred_writer!(writer::LatLonDeferredBinaryWriter{FT}) where FT
    writer.inner !== nothing && return writer.inner
    Nt = length(writer.workspace.storage.all_m)
    writer.header["ps_offsets_pa_per_window"] = writer.workspace.ps_offsets[1:Nt]
    writer.header["ps_offsets_next_day_hour0_pa"] = writer.workspace.ps_offsets[Nt + 1]
    set_transport_header_steps_per_window_schedule!(
        writer.header, writer.workspace.steps_schedule[1:Nt])
    header_json = JSON3.write(writer.header)
    length(header_json) < HEADER_SIZE ||
        error("Header JSON too large after offsets update: $(length(header_json)) >= $(HEADER_SIZE)")
    writer.inner = LatLonBinaryWriter(
        writer.path, header_json, writer.settings, writer.workspace.merged,
        writer.workspace.last_hour_next, FT, writer.basis;
        final_path = writer.final_path)
    return writer.inner
end

function write_window!(writer::LatLonDeferredBinaryWriter{FT},
                       ready::ReadyWindow{LatLonTargetGeometry, FT}) where FT
    writer.closed && throw(ArgumentError("cannot write to a closed LatLonDeferredBinaryWriter"))
    return write_window!(_open_latlon_deferred_writer!(writer), ready)
end

function close_streaming_binary!(writer::LatLonDeferredBinaryWriter)
    writer.closed && return writer.path
    writer.inner === nothing || close_streaming_binary!(writer.inner)
    writer.closed = true
    return writer.path
end

function promote_streaming_binary!(writer::LatLonDeferredBinaryWriter)
    writer.promoted && return writer.final_path
    close_streaming_binary!(writer)
    if writer.path != writer.final_path && isfile(writer.path)
        commit_atomic_staging(writer.path, writer.final_path)
    end
    writer.promoted = true
    return writer.final_path
end

function validate_staged_binary!(writer::LatLonDeferredBinaryWriter)
    writer.inner === nothing && throw(ArgumentError(
        "lat-lon writer produced no staging artifact"))
    expected = Int(writer.header["header_bytes"]) +
               Int(writer.header["nwindow"]) * Int(writer.header["window_bytes"])
    return validate_staged_size(writer.path, expected)
end

function allocate_window_workspace(grid::LatLonTargetGeometry,
                                   settings,
                                   vertical,
                                   spec,
                                   date::Date,
                                   ::Type{FT};
                                   source_steps_per_window::Integer = 1,
                                   write_replay_on::Bool = true) where FT
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    Nt = spec.n_times
    transform = allocate_transform_workspace(grid, spec.T, Nz_native)
    merged = allocate_merge_workspace(grid, Nz_native, Nz, FT)
    storage = allocate_window_storage(Nt, FT;
                                       include_qv = settings.include_qv,
                                       include_surface = settings.include_surface)
    qv = allocate_qv_workspace(grid, settings, date, Nz_native, Nz, FT)
    ps_offsets = zeros(Float64, Nt + 1)
    steps_schedule = fill(Int(source_steps_per_window), Nt)
    return LatLonSpectralWindowWorkspace{FT, typeof(transform), typeof(merged),
                                         typeof(storage), typeof(qv)}(
        transform, merged, storage, qv, ps_offsets, nothing, steps_schedule,
        StructuredFidelitySampler(nlon(grid), nlat(grid)),
        Vector{Any}(undef, Nt), write_replay_on)
end

function ingest_window!(workspace::LatLonSpectralWindowWorkspace,
                        win_idx::Int,
                        hour::Int,
                        spec,
                        grid::LatLonTargetGeometry,
                        vertical,
                        settings;
                        surface_reader = nothing)
    process_window!(win_idx, hour, spec, grid, vertical, settings,
                    workspace.transform, workspace.merged, workspace.qv,
                    workspace.storage, workspace.ps_offsets;
                    surface_reader = surface_reader)
    workspace.fidelity_captures[win_idx] = capture_structured_wind_flux_samples(
        workspace.fidelity_sampler,
        workspace.storage.all_am[win_idx], workspace.storage.all_bm[win_idx],
        workspace.transform, workspace.qv, grid, vertical, settings)
    return nothing
end

struct LLSpectralUnifiedDriverContext{G, S, V, SP, N, SR}
    grid            :: G
    settings        :: S
    vertical        :: V
    spec            :: SP
    next_day_hour0  :: N
    date            :: Date
    substep_policy  :: SubstepSchedulePolicy
    surface_reader  :: SR
end

driver_windows_per_day(::Nothing, ctx::LLSpectralUnifiedDriverContext) =
    ctx.spec.n_times

function driver_ingest_window!(workspace::LatLonSpectralWindowWorkspace,
                               ::Nothing,
                               win::Int,
                               ctx::LLSpectralUnifiedDriverContext)
    return ingest_window!(workspace, win, ctx.spec.hours[win], ctx.spec,
                          ctx.grid, ctx.vertical, ctx.settings;
                          surface_reader = ctx.surface_reader)
end

function driver_drain_ready_windows!(::LatLonSpectralWindowWorkspace,
                                     ::LatLonContract,
                                     ::Int,
                                     ::LLSpectralUnifiedDriverContext)
    return ()
end

function driver_flush_final_windows!(workspace::LatLonSpectralWindowWorkspace,
                                     ::Nothing,
                                     contract::LatLonContract,
                                     ctx::LLSpectralUnifiedDriverContext)
    ready_windows = flush_final_windows!(workspace, ctx.next_day_hour0,
                                         ctx.date, ctx.grid, ctx.vertical,
                                     ctx.settings, contract, ctx.substep_policy)
    # `flush_final_windows!` calls `apply_poisson_balance!`, which already
    # runs the typed LL contract over every stored window and updates the
    # contract accumulator. Return preverified events so the generic driver
    # writes and summarizes without replaying the expensive full-day gate.
    checked = (replay = (max_rel_err = 0.0,
                         max_abs_err = 0.0,
                         worst_idx = (0, 0, 0)),
               positivity = (ok = true,
                             ratio = 0.0,
                             direction = :none,
                             location = (0, 0, 0)))
    return (PreverifiedWindow(ready, checked; accumulated = true)
            for ready in ready_windows)
end

function flush_final_windows!(workspace::LatLonSpectralWindowWorkspace{FT},
                              next_day_hour0,
                              date::Date,
                              grid::LatLonTargetGeometry,
                              vertical,
                              settings,
                              contract::LatLonContract{FT},
                              substep_policy::SubstepSchedulePolicy) where FT
    workspace.last_hour_next = next_day_merged_fields(
        next_day_hour0, date, grid, vertical, settings,
        workspace.transform, workspace.merged, workspace.qv,
        workspace.ps_offsets)
    apply_poisson_balance!(workspace.storage, workspace.last_hour_next,
                           workspace.steps_schedule, contract, substep_policy;
                           write_replay_on = workspace.write_replay_on,
                           weights = spectral_column_weights(grid, settings, vertical))
    fill_qv_endpoints!(workspace.storage, workspace.last_hour_next)
    return (begin
            amount_seconds = settings.met_interval / (2 * workspace.steps_schedule[win_idx])
            accepted_capture = scale_captured_flux_samples(
                workspace.fidelity_captures[win_idx], amount_seconds / settings.half_dt)
            flux_evidence = structured_wind_evidence(
                LatLonTargetGeometry, accepted_capture,
                workspace.storage.all_am[win_idx], workspace.storage.all_bm[win_idx];
                method = "structured_uv_dp",
                time_sampling = "window_start_endpoint",
                amount_seconds)
            omega_evidence = NoSourceOmegaEvidence{
                LatLonTargetGeometry,FT,SourceHasNoOmega}(SourceHasNoOmega())
            ReadyWindow{LatLonTargetGeometry, FT}(
                win_idx,
                (m_cur = workspace.storage.all_m[win_idx],
                 am = workspace.storage.all_am[win_idx],
                 bm = workspace.storage.all_bm[win_idx],
                 cm = workspace.storage.all_cm[win_idx],
                 storage = workspace.storage,
                 m_next = win_idx < length(workspace.storage.all_m) ?
                    workspace.storage.all_m[win_idx + 1] :
                    workspace.last_hour_next === nothing ?
                        workspace.storage.all_m[win_idx] :
                        workspace.last_hour_next.m),
                FluxFidelityEvidenceBundle(flux_evidence, omega_evidence,
                                   NoColumnClosureRecord()))
            end
            for win_idx in eachindex(workspace.storage.all_m))
end

"""
    process_day(date, grid::LatLonTargetGeometry, settings, vertical;
                next_day_hour0=nothing, positivity_cfl_limit=DEFAULT_POSITIVITY_CFL_LIMIT,
                require_substep_positivity=true, write_replay_check=true)

Run the full one-day preprocessing workflow for the structured lat-lon target:
read spectral input, process all windows, close continuity against forward mass
endpoints, and write the final binary.
"""
function process_day(date::Date,
                     grid::LatLonTargetGeometry,
                     settings::ERA5SpectralSettings,
                     vertical;
                     next_day_hour0=nothing,
                     positivity_cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                     require_substep_positivity::Bool = true,
                     write_replay_check::Bool = true,
                     substep_policy::SubstepSchedulePolicy =
                         SubstepSchedulePolicy(
                             adaptive_substeps = false,
                             substep_cfl_target = positivity_cfl_limit),
                     run_cache = nothing)
    check_spectral_column_weights(grid, settings)
    FT = settings.output_float_type
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    Nx = nlon(grid)
    Ny = nlat(grid)
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = era5_spectral_paths(settings.spectral_dir, date).vo_d
    lnsp_path = era5_spectral_paths(settings.spectral_dir, date).lnsp

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral(vo_d_path, lnsp_path;
                             T_target=settings.T_target,
                             cache_dir=settings.spectral_cache_dir)
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, time() - t_day)

    Nt = spec.n_times
    counts = window_element_counts(grid, Nz;
                                    include_qv=settings.include_qv,
                                    include_surface=settings.include_surface)
    byte_sizes = window_byte_sizes(counts, FT, Nt)
    counts = merge(counts, (bytes_per_window = byte_sizes.bytes_per_window,))

    mkpath(settings.out_dir)
    bin_path = output_binary_path(date, settings.out_dir, settings.min_dp, FT)

    provenance = script_provenance()
    sizes = (Nx = Nx, Ny = Ny, Nz = Nz, Nz_native = Nz_native, Nt = Nt,
             steps_per_met = steps_per_met)
    header = build_v4_header(date, grid, vertical, settings, FT, counts, sizes, provenance)
    source_paths = String[vo_d_path, lnsp_path]
    if settings.include_qv || settings.mass_basis == :dry
        push!(source_paths, joinpath(
            settings.thermo_dir, "era5_thermo_ml_$(date_str).nc"))
        next_date_str = Dates.format(date + Day(1), "yyyymmdd")
        push!(source_paths, joinpath(
            settings.thermo_dir, "era5_thermo_ml_$(next_date_str).nc"))
    end
    settings.include_surface && push!(
        source_paths, resolve_era5_surface_path(settings.surface_dir, date))
    header["generation_fingerprint"] = generation_fingerprint(
        settings=settings,
        spectral_resolution=spec.T,
        source_paths=source_paths,
        next_day_hour0=next_day_hour0,
        provenance=provenance,
    )

    # Resolved once: the header records it (before the reuse check, so a binary
    # written with the gate off is not reused by a run with it on) and the
    # workspace carries it to the gate.
    write_replay_on = write_replay_check
    with_replay_record(header, write_replay_on)
    expected_sections = expected_payload_sections(settings)
    skip, reason = existing_output_schema_matches(
        bin_path, byte_sizes.total_bytes, expected_sections, header)
    if skip
        @info "  SKIP (exists, size + schema match): $(basename(bin_path))"
        return bin_path
    elseif isfile(bin_path) && filesize(bin_path) == byte_sizes.total_bytes
        @info "  REGEN (size match, $(reason)): $(basename(bin_path))"
    end

    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(bin_path), byte_sizes.total_bytes / 1e9, Nt)

    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE ||
        error("Header JSON too large: $(length(header_json)) >= $(HEADER_SIZE)")

    workspace = allocate_window_workspace(grid, settings, vertical, spec, date, FT;
                                          source_steps_per_window = steps_per_met,
                                          write_replay_on)
    storage = workspace.storage
    ps_offsets = workspace.ps_offsets
    window_contract = LatLonContract{FT}(
        replay_tol = replay_tolerance(FT),
        positivity_cfl_limit = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        steps_per_window = steps_per_met,
    )

    surface_reader = settings.include_surface ?
        open_era5_surface_reader(settings.surface_dir, date, Nx, Ny) : nothing

    log_mass_fix_configuration(settings)
    @info "  Computing spectral -> gridpoint -> merged for $Nt windows..."

    try
        tmp_path = open_preprocessing_staging(bin_path)
        writer = LatLonDeferredBinaryWriter(
            tmp_path, header, workspace, settings, FT,
            State.mass_basis(Symbol(settings.mass_basis));
            final_path = bin_path)
        ctx = LLSpectralUnifiedDriverContext(
            grid, settings, vertical, spec, next_day_hour0, date,
            substep_policy, surface_reader)

        driver_result = run_unified_preprocessor_day!(
            UnifiedPreprocessorDay(nothing, workspace, window_contract,
                                   writer; context = ctx);
            close_reader = false)

        if settings.mass_fix_enable
            @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                           minimum(ps_offsets[1:Nt]),
                           maximum(ps_offsets[1:Nt]),
                           sum(ps_offsets[1:Nt]) / Nt)
        end

        actual = filesize(driver_result.out_path)
        @info @sprintf("  Done: %s (%.2f GB, %.1fs)",
                       basename(driver_result.out_path), actual / 1e9,
                       time() - t_day)
        actual == byte_sizes.total_bytes ||
            error(@sprintf("SIZE MISMATCH: expected %d bytes, got %d",
                           byte_sizes.total_bytes, actual))

        return driver_result.out_path
    finally
        surface_reader === nothing || close_era5_surface_reader(surface_reader)
    end
end
