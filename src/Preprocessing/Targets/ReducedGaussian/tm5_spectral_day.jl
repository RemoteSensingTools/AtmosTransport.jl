process_day(date::Date,grid::ReducedGaussianTargetGeometry,settings::ERA5SpectralSettings,vertical;kwargs...) =
    process_day(spectral_construction(settings),date,grid,settings,vertical;kwargs...)

synthesize_and_merge_window!(c::TM5SpectralFluxes,ws::SpectralIntegralWorkspace,args...) =
    process_window!(c,ws,args...)

function process_day(c::TM5SpectralFluxes,date::Date,grid::ReducedGaussianTargetGeometry,
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
    settings_include_surface(settings) && throw(ArgumentError("RG spectral preprocessing does not support surface payloads"))
    paths=era5_spectral_paths(settings.spectral_dir,date)
    return with_spectral_hours(paths; cache_dir=settings.spectral_cache_dir) do read_hour
        first_spec = read_hour(0)
        T = first_spec.T
        next_day_hour0.T == T || error("spectral truncation changed at next-day endpoint")
        nt=24
        FT=settings.output_float_type
        nz=vertical.Nz
        nc=ncells(grid.mesh)
        source_steps=exact_steps_per_window(settings.met_interval,settings.dt)
        qv=allocate_reduced_qv_workspace(grid,vertical.Nz_native,era5_thermo_path(settings.thermo_dir,date);settings)
        qv === nothing && error("tm5_spectral requires native humidity")
        ws=SpectralIntegralWorkspace(grid,T,c;synthesis=get(settings,:spectral_synthesis,GPUSynthesis()),
            precision=get(settings,:spectral_synthesis_float,AutoSynthesisPrecision()),batch=min(8,vertical.Nz_native))
        function endpoint(i)
            if i<=nt
                load_rg_qv!(qv,era5_thermo_path(settings.thermo_dir,date),i,vertical.Nz_native)
                h = i == 1 ? first_spec : read_hour(i-1)
                h.T == T || error("spectral truncation changed within day")
                return synthesize_and_merge_window!(c,ws,h.lnsp,h.vo,h.d,T,qv.qv_cell,endpoint_plan,settings)
            end
            qv.qv_ll .= read_qv_from_thermo(era5_thermo_path(settings.thermo_dir,date+Day(1)),1,
                qv.Nx_ll,qv.Ny_ll,vertical.Nz_native;
                FT=Float64)
            _interpolate_ll_to_rg!(qv)
            return synthesize_and_merge_window!(c,ws,next_day_hour0.lnsp,next_day_hour0.vo,
                next_day_hour0.d,T,qv.qv_cell,endpoint_plan,settings)
        end
        left,right=face_connectivity(grid.mesh)
        L=build_compressed_laplacian(left,right,nc)
        factor=DirectRingPoisson(L) # reused for all levels, endpoints, windows and dry repairs
        scratch=(;psi=zeros(nc),rhs=zeros(nc),r=zeros(nc),p=zeros(nc),Ap=zeros(nc),z=zeros(nc))
        weights=spectral_column_weights(grid,settings,vertical)
        contract=ReducedGaussianContract{FT}(replay_tol=replay_tolerance(FT),
            positivity_cfl_limit=positivity_cfl_limit,require_substep_positivity=require_substep_positivity,
            steps_per_window=source_steps,face_left=left,face_right=right)
        transport_grid=AtmosGrid(grid.mesh,vertical.merged_vc,CPU();FT=FT,radius=grid.mesh.radius)
        metadata=spectral_synthesis_metadata(ws)
        metadata["preprocessor"]="preprocess_transport_binary.jl"
        metadata["source_type"]="era5_spectral"
        metadata["horizontal_balance"]=HORIZONTAL_BALANCE_TAG
        metadata["mass_fix_enabled"]=settings.mass_fix_enable
        metadata["mass_fix_target_ps_dry_pa"]=settings.target_ps_dry_pa
        with_replay_record(metadata,write_replay_check)
        with_column_weights_record(metadata,check_spectral_column_weights(grid,settings))
        mkpath(settings.out_dir)
        path=output_binary_path(date,settings.out_dir,settings.min_dp,FT)
        writer=nothing
        schedule=Int[]
        offsets=Float64[]
        a=endpoint(1)
        push!(offsets,a.offset)
        first_spec = nothing
        try
            for i in 1:nt
                b=endpoint(i+1)
                push!(offsets,b.offset)
                rate=balance_window_layers!(c,a,b,grid,factor,settings.met_interval,ws.quadrature.areas)
                raw=vcat(rate.u,rate.v)
                m=FT.(rate.m)
                m_next=FT.(rate.m_next)
                steps=initial_substeps(substep_policy,source_steps)
                flux=zeros(FT,size(raw))
                cm=zeros(FT,nc,nz+1)
                dm=zeros(nc,nz)
                div=zeros(nc,nz)
                while true
                    # Every retry starts from pristine dry rates, never corrected amounts.
                    flux64=raw.*flux_application_seconds(settings.met_interval,steps,:substep_mass_amount)
                    balance_compressed_column_fluxes!(flux64,Float64.(m),Float64.(m_next),
                        left,right,L,steps,scratch;
                        weights,direct_solver=factor)
                    flux .= FT.(flux64)
                    dm .= (Float64.(m_next).-Float64.(m))./(2steps)
                    recompute_cm_from_dm_target!(faceindexed_replay_layout(left,right),div,cm,
                        m,dm,flux;
                        residual_weight=ColumnResidualWeight(weights))
                    cm[:,1] .= zero(FT)
                    cm[:,end] .= zero(FT)
                    contract.steps_per_window=steps
                    payload=(;m_cur=m,hflux=flux,cm,m_next)
                    diag=write_replay_check ? verify_window!(payload,contract,i) :
                        (;replay=nothing,positivity=verify_substep_positivity_rg!(m,flux,cm,left,right;
                            cfl_limit=positivity_cfl_limit))
                    next=next_substeps(substep_policy,steps,diag.positivity.ratio)
                    if next==steps
                        update_accumulator!(contract,diag.positivity,i)
                        @info "TM5 RG window closure" window=i steps replay=diag.replay moist=map(x->x.post,rate.diagnostics)
                        break
                    end
                    steps=next
                end
                push!(schedule,steps)
                base=(;m,hflux=flux,cm,ps=FT.(a.ps))
                window=settings.include_qv ? merge(base,(qv_start=FT.(a.q),qv_end=FT.(b.q))) : base
                if writer === nothing
                    writer=open_streaming_transport_binary(path,transport_grid,nt,window;
                        FT,dt_met_seconds=settings.met_interval,half_dt_seconds=settings.half_dt,
                        steps_per_window=source_steps,source_flux_sampling=:window_mean,
                        flux_sampling=:window_constant,humidity_sampling=settings.include_qv ? :window_endpoints : :none,
                        delta_semantics=:none,mass_basis=:dry,surface_pressure_basis=TotalSurfacePressure(),
                        flux_construction=TM5SpectralFluxesHeader(1,T,c.refinement_steps,c.vod2uv_radius),
                        extra_header=metadata)
                end
                write_streaming_window!(writer,window)
                a=b
            end
            writer.header["ps_offsets_pa_per_window"]=offsets[1:end-1]
            writer.header["ps_offsets_next_day_hour0_pa"]=offsets[end]
            set_streaming_steps_per_window_schedule!(writer,schedule)
            summarize_status!(contract)
            close_streaming_transport_binary!(writer)
        catch
            writer === nothing || abandon_streaming_transport_binary!(writer)
            rethrow()
        end
        return path
    end # bounded hourly reader
end


function tm5_horizontal_convergence(u,v,grid::ReducedGaussianTargetGeometry)
    left,right=face_connectivity(grid.mesh)
    flux=vcat(u,v)
    out=zeros(ncells(grid.mesh),size(u,2))
    for k in axes(flux,2), f in eachindex(left)
        l=left[f]
        r=right[f]
        l>0 && (out[l,k]-=flux[f,k])
        r>0 && (out[r,k]+=flux[f,k])
    end
    return out
end

function balance_tm5_layers!(u,v,target,grid::ReducedGaussianTargetGeometry,solver::DirectRingPoisson)
    left,right=face_connectivity(grid.mesh)
    flux=vcat(u,v)
    for k in axes(flux,2)
        balance_compressed_level!(view(flux,:,k),-view(target,:,k),left,right,solver)
    end
    u .= view(flux,1:size(u,1),:)
    v .= view(flux,size(u,1)+1:size(flux,1),:)
    return nothing
end

function dry_target_fluxes!(u,v,q,grid::ReducedGaussianTargetGeometry)
    left,right=face_connectivity(grid.mesh)
    flux=vcat(u,v)
    for k in axes(flux,2),f in eachindex(left)
        l=left[f]
        r=right[f]
        if l>0 && r>0
            flux[f,k] *= 1-(q[l,k]+q[r,k])/2
        else
            flux[f,k]=0
        end
    end
    u .= view(flux,1:size(u,1),:)
    v .= view(flux,size(u,1)+1:size(flux,1),:)
    return nothing
end
