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

    function LatLonSpectralWindowWorkspace{FT, TW, MW, SW, QW}(
            transform::TW,
            merged::MW,
            storage::SW,
            qv::QW,
            ps_offsets::Vector{Float64},
            last_hour_next,
            steps_schedule::Vector{Int}) where {FT, TW, MW, SW, QW}
        if last_hour_next !== nothing && !(last_hour_next isa LLNextDayFields{FT})
            throw(ArgumentError(
                "last_hour_next must be `nothing` or LLNextDayFields{$FT}; " *
                "got $(typeof(last_hour_next))"))
        end
        return new{FT, TW, MW, SW, QW}(
            transform, merged, storage, qv, ps_offsets, last_hour_next,
            steps_schedule)
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
        mv(writer.path, writer.final_path; force = true)
    end
    writer.promoted = true
    return writer.final_path
end

function quarantine_streaming_binary!(writer::LatLonDeferredBinaryWriter)
    writer.promoted && return writer.path
    close_streaming_binary!(writer)
    isfile(writer.path) && rm(writer.path; force = true)
    return writer.path
end

function validate_staged_binary!(writer::LatLonDeferredBinaryWriter)
    writer.inner === nothing && throw(ArgumentError(
        "lat-lon writer produced no staging artifact"))
    expected = Int(writer.header["header_bytes"]) +
               Int(writer.header["nwindow"]) * Int(writer.header["window_bytes"])
    return _validate_staged_size(writer.path, expected)
end

function allocate_window_workspace(grid::LatLonTargetGeometry,
                                   settings,
                                   vertical,
                                   spec,
                                   date::Date,
                                   ::Type{FT};
                                   cache = nothing,
                                   source_steps_per_window::Integer = 1) where FT
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    Nt = spec.n_times
    transform = allocate_transform_workspace(grid, spec.T, Nz_native)
    merged = allocate_merge_workspace(grid, Nz_native, Nz, FT)
    storage = allocate_window_storage(Nt, FT;
                                       include_qv = settings.include_qv,
                                       tm5_convection = settings.tm5_convection_enable,
                                       include_surface = settings.include_surface)
    qv = allocate_qv_workspace(grid, settings, date, Nz_native, Nz, FT)
    ps_offsets = zeros(Float64, Nt + 1)
    steps_schedule = fill(Int(source_steps_per_window), Nt)
    return LatLonSpectralWindowWorkspace{FT, typeof(transform), typeof(merged),
                                         typeof(storage), typeof(qv)}(
        transform, merged, storage, qv, ps_offsets, nothing, steps_schedule)
end

function ingest_window!(workspace::LatLonSpectralWindowWorkspace,
                        win_idx::Int,
                        hour::Int,
                        spec,
                        grid::LatLonTargetGeometry,
                        vertical,
                        settings;
                        physics_reader = nothing,
                        tm5_ws = nothing,
                        tm5_source_ps = nothing,
                        tm5_stats = nothing,
                        surface_reader = nothing)
    process_window!(win_idx, hour, spec, grid, vertical, settings,
                    workspace.transform, workspace.merged, workspace.qv,
                    workspace.storage, workspace.ps_offsets;
                    physics_reader = physics_reader,
                    tm5_ws = tm5_ws,
                    tm5_source_ps = tm5_source_ps,
                    tm5_stats = tm5_stats,
                    surface_reader = surface_reader)
    return nothing
end

drain_ready_windows!(::LatLonSpectralWindowWorkspace) = ()

struct LLSpectralUnifiedDriverContext{G, S, V, SP, N, PR, TW, TPS, TS, SR}
    grid            :: G
    settings        :: S
    vertical        :: V
    spec            :: SP
    next_day_hour0  :: N
    date            :: Date
    substep_policy  :: SubstepSchedulePolicy
    physics_reader  :: PR
    tm5_ws          :: TW
    tm5_source_ps   :: TPS
    tm5_stats       :: TS
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
                          physics_reader = ctx.physics_reader,
                          tm5_ws = ctx.tm5_ws,
                          tm5_source_ps = ctx.tm5_source_ps,
                          tm5_stats = ctx.tm5_stats,
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
                           balance = get(settings, :horizontal_balance, nothing))
    fill_qv_endpoints!(workspace.storage, workspace.last_hour_next)
    return (ReadyWindow{LatLonTargetGeometry, FT}(
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
                        workspace.last_hour_next.m))
            for win_idx in eachindex(workspace.storage.all_m))
end

function _tm5_ll_source_grid_from_physics(reader, ::Type{FT}) where FT <: AbstractFloat
    h = reader.header
    return build_target_geometry(Val(:latlon),
        Dict{String, Any}("type" => "latlon",
                          "nlon" => h.Nlon,
                          "nlat" => h.Nlat),
        FT)
end

@inline _tm5_ll_same_shape(source::LatLonTargetGeometry,
                           target::LatLonTargetGeometry) =
    nlon(source) == nlon(target) && nlat(source) == nlat(target)

function _tm5_ll_regridder_cache_key(source::LatLonTargetGeometry,
                                     target::LatLonTargetGeometry)
    return Symbol("tm5_ll_regridder_",
                  nlon(source), "x", nlat(source),
                  "_to_", nlon(target), "x", nlat(target))
end

function _get_or_build_tm5_ll_regridder!(cache,
                                         source::LatLonTargetGeometry,
                                         target::LatLonTargetGeometry)
    _tm5_ll_same_shape(source, target) && return nothing
    key = _tm5_ll_regridder_cache_key(source, target)
    cached = cache === nothing ? nothing : get(cache, key, nothing)
    cached !== nothing && return cached

    t0 = time()
    regridder = build_regridder(source.mesh, target.mesh; normalize = false)
    cache === nothing || (cache[key] = regridder)
    @info @sprintf("  TM5 LL regridder: %dx%d -> %dx%d  nnz=%d (%.1fs)",
                   nlon(source), nlat(source), nlon(target), nlat(target),
                   length(regridder.intersections.nzval), time() - t0)
    return regridder
end

function _install_tm5_regrid_header_metadata!(header,
                                              source::LatLonTargetGeometry,
                                              target::LatLonTargetGeometry,
                                              regridder)
    source_matches_target = regridder === nothing
    header["tm5_source_grid_type"] = "latlon"
    header["tm5_source_nlon"] = nlon(source)
    header["tm5_source_nlat"] = nlat(source)
    header["tm5_target_nlon"] = nlon(target)
    header["tm5_target_nlat"] = nlat(target)
    header["tm5_regrid_method"] = source_matches_target ? "identity" : "conservative"
    header["tm5_ps_source"] = source_matches_target ?
        "spectral_target_grid" : "spectral_source_grid_with_target_mass_fix_offset"
    return header
end

"""
    process_day(date, grid::LatLonTargetGeometry, settings, vertical;
                next_day_hour0=nothing, positivity_cfl_limit=0.95,
                require_substep_positivity=true)

Run the full one-day preprocessing workflow for the structured lat-lon target:
read spectral input, process all windows, close continuity against forward mass
endpoints, and write the final binary.
"""
function process_day(date::Date,
                     grid::LatLonTargetGeometry,
                     settings::ERA5SpectralSettings,
                     vertical;
                     next_day_hour0=nothing,
                     positivity_cfl_limit::Real = 0.95,
                     require_substep_positivity::Bool = true,
                     substep_policy::SubstepSchedulePolicy =
                         SubstepSchedulePolicy(
                             adaptive_substeps = false,
                             substep_cfl_target = positivity_cfl_limit),
                     run_cache = nothing)
    FT = settings.output_float_type
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    Nx = nlon(grid)
    Ny = nlat(grid)
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

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
                                    tm5_convection=settings.tm5_convection_enable,
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
    if settings.tm5_convection_enable
        push!(source_paths, joinpath(
            settings.tm5_physics_bin_dir, string(year(date)),
            "era5_physics_$(date_str).bin"))
    end
    settings.include_surface && push!(
        source_paths, _resolve_era5_surface_path(settings.surface_dir, date))
    header["generation_fingerprint"] = generation_fingerprint(
        settings=settings,
        spectral_resolution=spec.T,
        source_paths=source_paths,
        next_day_hour0=next_day_hour0,
        provenance=provenance,
    )

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
                                          cache = run_cache,
                                          source_steps_per_window = steps_per_met)
    storage = workspace.storage
    ps_offsets = workspace.ps_offsets
    window_contract = LatLonContract{FT}(
        replay_tol = replay_tolerance(FT),
        positivity_cfl_limit = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        steps_per_window = steps_per_met,
    )

    # TM5 convection setup. Native LL keeps the identity fast path; lower-
    # resolution LL targets synthesize source-grid PS and conservatively
    # regrid merged TM5 fields onto the target payload grid.
    physics_reader = nothing
    tm5_ws         = nothing
    tm5_source_ps  = nothing
    tm5_stats      = nothing
    if settings.tm5_convection_enable
        physics_reader = open_era5_physics_binary(settings.tm5_physics_bin_dir, date)
        Nlon_src = physics_reader.header.Nlon
        Nlat_src = physics_reader.header.Nlat
        tm5_source_grid = _tm5_ll_source_grid_from_physics(physics_reader, Float64)
        tm5_regridder = _get_or_build_tm5_ll_regridder!(run_cache, tm5_source_grid, grid)
        _install_tm5_regrid_header_metadata!(header, tm5_source_grid, grid, tm5_regridder)
        if tm5_regridder !== nothing
            tm5_source_ps = allocate_tm5_source_pressure_workspace(tm5_source_grid, spec.T)
            @info @sprintf("  TM5 source-grid PS: spectral %dx%d -> physics %dx%d; fields regridded to target %dx%d",
                           Nx, Ny, Nlon_src, Nlat_src, Nx, Ny)
        end
        tm5_ws    = allocate_tm5_workspace(Nlon_src, Nlat_src, Nz_native, Nz, FT;
                                            regridder = tm5_regridder,
                                            target_nlon = Nx,
                                            target_nlat = Ny,
                                            physics_eltype = Float32)
        tm5_stats = TM5CleanupStats()
        header_json = JSON3.write(header)
        length(header_json) < HEADER_SIZE ||
            error("Header JSON too large after TM5 metadata: $(length(header_json)) >= $(HEADER_SIZE)")
    end

    surface_reader = settings.include_surface ?
        open_era5_surface_reader(settings.surface_dir, date, Nx, Ny) : nothing

    log_mass_fix_configuration(settings)
    @info "  Computing spectral -> gridpoint -> merged for $Nt windows..."

    try
        tmp_path = bin_path * ".tmp"
        rm(tmp_path; force = true)
        writer = LatLonDeferredBinaryWriter(
            tmp_path, header, workspace, settings, FT,
            mass_basis_from_symbol(Symbol(settings.mass_basis));
            final_path = bin_path)
        ctx = LLSpectralUnifiedDriverContext(
            grid, settings, vertical, spec, next_day_hour0, date,
            substep_policy,
            physics_reader, tm5_ws, tm5_source_ps, tm5_stats, surface_reader)

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
        tm5_stats === nothing || log_tm5_cleanup_stats(tm5_stats, date_str)

        actual = filesize(driver_result.out_path)
        @info @sprintf("  Done: %s (%.2f GB, %.1fs)",
                       basename(driver_result.out_path), actual / 1e9,
                       time() - t_day)
        actual == byte_sizes.total_bytes ||
            error(@sprintf("SIZE MISMATCH: expected %d bytes, got %d",
                           byte_sizes.total_bytes, actual))

        last_merged = (m = storage.all_m[Nt],
                       am = storage.all_am[Nt],
                       bm = storage.all_bm[Nt])
        return driver_result.out_path, last_merged
    finally
        physics_reader === nothing || close_era5_physics_binary(physics_reader)
        surface_reader === nothing || close_era5_surface_reader(surface_reader)
    end
end
