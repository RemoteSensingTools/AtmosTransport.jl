# TM5 S3–S6: direct spectral stages join the existing LL storage, column
# balance, explicit-dm closure, replay/positivity gates and binary writer.

process_day(date::Date,grid::LatLonTargetGeometry,settings::ERA5SpectralSettings,vertical;kwargs...) =
    process_day(spectral_construction(settings),date,grid,settings,vertical;kwargs...)

function process_window!(::TM5SpectralFluxes,ws::SpectralIntegralWorkspace,
                         H,vo,d,T,q_native,endpoint_plan,settings)
    vertical = endpoint_plan.vertical
    return construct_tm5_endpoint(ws,H,view(vo,:,:,vertical.level_range),
        view(d,:,:,vertical.level_range),T,q_native,endpoint_plan,settings)
end

function process_day(c::TM5SpectralFluxes,date::Date,grid::LatLonTargetGeometry,
                     settings::ERA5SpectralSettings,vertical;
                     next_day_hour0=nothing,
                     positivity_cfl_limit::Real=DEFAULT_POSITIVITY_CFL_LIMIT,
                     require_substep_positivity::Bool=true,write_replay_check::Bool=true,
                     substep_policy=SubstepSchedulePolicy(adaptive_substeps=false,
                         substep_cfl_target=positivity_cfl_limit),run_cache=nothing)
    validate_spectral_construction(c,grid,settings)
    validate_spectral_vertical(c,vertical)
    endpoint_plan = TM5EndpointPlan(vertical)
    next_day_hour0 === nothing && error("tm5_spectral requires next-day 00 UTC; no zero-tendency fallback")
    paths=era5_spectral_paths(settings.spectral_dir,date)
    return with_spectral_hours(paths; cache_dir=settings.spectral_cache_dir) do read_hour
        first_spec = read_hour(0)
        T = first_spec.T
        next_day_hour0.T == T || error("spectral truncation changed at next-day endpoint")
        nt=24; FT=settings.output_float_type
        nx=nlon(grid); ny=nlat(grid); nz=vertical.Nz
        steps=exact_steps_per_window(settings.met_interval,settings.dt)
        qv=allocate_qv_workspace(grid,settings,date,vertical.Nz_native,nz,Float64)
        ws=SpectralIntegralWorkspace(grid,T,c;synthesis=get(settings,:spectral_synthesis,GPUSynthesis()),
            precision=get(settings,:spectral_synthesis_float,AutoSynthesisPrecision()),batch=min(8,vertical.Nz_native))
        function endpoint(i)
            if i <= nt
                read_window_qv!(qv,i,nx,ny,vertical.Nz_native)
                h = i == 1 ? first_spec : read_hour(i-1)
            else
                read_next_day_qv!(qv,date,settings,nx,ny,vertical.Nz_native)
                h = next_day_hour0
            end
            h.T == T || error("spectral truncation changed within day")
            return process_window!(c,ws,h.lnsp,h.vo,h.d,T,qv.qv_native,endpoint_plan,settings)
        end
        a = endpoint(1)
        first_spec = nothing
        offsets = [a.offset]
        solver = LLPoissonWorkspace(nx,ny)
        dry_solver = LLPoissonWorkspace(nx,ny)
        weights = spectral_column_weights(grid,settings,vertical)
        contract = LatLonContract{FT}(replay_tol=replay_tolerance(FT),
            positivity_cfl_limit=positivity_cfl_limit,
            require_substep_positivity=require_substep_positivity,steps_per_window=steps)
        counts = window_element_counts(grid,nz;
            include_qv=settings.include_qv,include_surface=settings.include_surface)
        bytes = window_byte_sizes(counts,FT,nt)
        counts = merge(counts,(;bytes_per_window=bytes.bytes_per_window))
        sizes = (;Nx=nx,Ny=ny,Nz=nz,Nz_native=vertical.Nz_native,Nt=nt,steps_per_met=steps)
        header = build_v4_header(date,grid,vertical,settings,FT,counts,sizes,script_provenance())
        merge!(header,tm5_construction_metadata(c,T),spectral_synthesis_metadata(ws))
        header["source_flux_sampling"] = "window_mean"
        with_replay_record(header,write_replay_check)
        mkpath(settings.out_dir)
        path = output_binary_path(date,settings.out_dir,settings.min_dp,FT)
        surface_reader = settings.include_surface ?
            open_era5_surface_reader(settings.surface_dir,date,nx,ny) : nothing
        writer = nothing
        pending = nothing
        schedule = Int[]
        try
            for i in 1:nt
                b = endpoint(i+1)
                push!(offsets,b.offset)
                window = balance_window_layers!(c,a,b,grid,solver,
                    settings.met_interval,ws.quadrature.areas)
                am,bm = directional_fluxes(window.u,window.v,grid)
                storage = allocate_window_storage(1,FT;
                    include_qv=settings.include_qv,include_surface=settings.include_surface)
                storage.all_m[1] = reshape(FT.(window.m),nx,ny,nz)
                storage.all_am[1] = FT.(am.*settings.half_dt)
                storage.all_bm[1] = FT.(bm.*settings.half_dt)
                storage.all_cm[1] = zeros(FT,nx,ny,nz+1)
                storage.all_ps[1] = reshape(FT.(a.ps),nx,ny)
                if settings.include_qv
                    storage.all_qv_start[1] = reshape(FT.(a.q),nx,ny,nz)
                end
                last = (;m=reshape(FT.(window.m_next),nx,ny,nz),
                    am=storage.all_am[1],bm=storage.all_bm[1],cm=storage.all_cm[1],
                    qv=settings.include_qv ? reshape(FT.(b.q),nx,ny,nz) : nothing)
                function rebuild_rates(_,n)
                    seconds = flux_application_seconds(settings.met_interval,n,:substep_mass_amount)
                    storage.all_am[1] .= FT.(am.*seconds)
                    storage.all_bm[1] .= FT.(bm.*seconds)
                end
                local_schedule = [steps]
                apply_poisson_balance!(storage,last,local_schedule,contract,substep_policy;
                    write_replay_on=write_replay_check,weights,rebuild_rates,
                    poisson_workspace=dry_solver,window_offset=i-1)
                push!(schedule,only(local_schedule))
                fill_qv_endpoints!(storage,last)
                surface_reader === nothing || _store_window_surface_fields!(storage,1,surface_reader,FT;source_window=i)
                @info "TM5 moist layer balance" window=i diagnostics=map(
                    x->(;x.pre,x.post,x.passes,x.max_layer_residual),window.diagnostics)
                if writer === nothing
                    writer = LatLonBinaryWriter(open_preprocessing_staging(path),JSON3.write(header),settings,
                        allocate_merge_workspace(grid,vertical.Nz_native,nz,FT),last,FT,
                        State.mass_basis(:dry);final_path=path)
                else
                    # Window-constant fluxes retain the historical delta payload.
                    # One completed lookahead window suffices to write those deltas.
                    writer.last_hour_next = (;m=storage.all_m[1],am=storage.all_am[1],
                        bm=storage.all_bm[1],cm=storage.all_cm[1],
                        qv=settings.include_qv ? storage.all_qv_start[1] : nothing)
                    evidence = tm5_flux_evidence(LatLonTargetGeometry,FT)
                    write_window!(writer,ReadyWindow{LatLonTargetGeometry,FT}(
                        i-1,(;storage=pending,storage_index=1),evidence))
                end
                pending = storage
                writer.last_hour_next = last
                a = b
            end
            evidence = tm5_flux_evidence(LatLonTargetGeometry,FT)
            write_window!(writer,ReadyWindow{LatLonTargetGeometry,FT}(
                nt,(;storage=pending,storage_index=1),evidence))
            writer.header["ps_offsets_pa_per_window"] = offsets[1:nt]
            writer.header["ps_offsets_next_day_hour0_pa"] = offsets[end]
            set_transport_header_steps_per_window_schedule!(writer.header,schedule)
            set_contract_steps_schedule!(contract,schedule)
            summarize_status!(contract)
            close_streaming_binary!(writer)
            validate_staged_size(writer.path,bytes.total_bytes)
            promote_streaming_binary!(writer)
        catch
            writer === nothing || quarantine_streaming_binary!(writer)
            rethrow()
        finally
            surface_reader === nothing || close_era5_surface_reader(surface_reader)
        end
        return path
    end # bounded hourly reader
end
