# Direct target adapters use flat canonical face rates until final storage.
function tm5_cs_directional_fluxes(u, grid::CubedSphereTargetGeometry)
    ft, n, nz = grid.face_table, grid.Nc, size(u, 2)
    am = ntuple(_ -> zeros(eltype(u), n+1, n, nz), 6)
    bm = ntuple(_ -> zeros(eltype(u), n, n+1, nz), 6)
    for k in 1:nz, f in 1:ft.nf
        p, i, j = ft.face_panel[f], ft.face_idx_i[f], ft.face_idx_j[f]
        out = ft.face_dir[f] == 1 ? am[p] : bm[p]
        out[i, j, k] = u[f, k]
    end
    sync_cs_mirrors!(am, bm, ft, nz)
    return am, bm
end

function tm5_horizontal_convergence(u, v, grid::CubedSphereTargetGeometry)
    ft = grid.face_table
    out = zeros(ft.nc, size(u, 2))
    # Same outward-divergence accumulation as cs_face_divergence!, negated
    # after summation to preserve its canonical arithmetic.
    am, bm = tm5_cs_directional_fluxes(u, grid)
    for k in axes(u, 2)
        cs_face_divergence!(view(out, :, k), am, bm, ft, k)
    end
    out .*= -1
    return out
end

function balance_tm5_layers!(u, v, target, grid::CubedSphereTargetGeometry, solver::DirectCSPoisson)
    am, bm = tm5_cs_directional_fluxes(u, grid)
    ft = grid.face_table
    scratch = CSPoissonScratch(ft.nc)
    outward = _ArrayTarget(-target)
    for k in axes(u, 2)
        balance_cs_level!(k, am, bm, outward, ft, solver, scratch)
        for f in 1:ft.nf
            p, i, j = ft.face_panel[f], ft.face_idx_i[f], ft.face_idx_j[f]
            u[f, k] = ft.face_dir[f] == 1 ? am[p][i, j, k] : bm[p][i, j, k]
        end
    end
    return nothing
end

function dry_target_fluxes!(u, v, q, grid::CubedSphereTargetGeometry)
    ft = grid.face_table
    for k in axes(u, 2), f in 1:ft.nf
        humidity = clamp((q[ft.face_left[f], k]+q[ft.face_right[f], k])/2,
            zero(eltype(q)), eltype(q)(MAX_SPECIFIC_HUMIDITY))
        u[f, k] *= 1-humidity
    end
    return nothing
end

process_day(date::Date, grid::CubedSphereTargetGeometry, settings::ERA5SpectralSettings, vertical; kwargs...) =
    process_day(spectral_construction(settings), date, grid, settings, vertical; kwargs...)

"""Prepare geometry once; refined samples are never reconstructed from LL fluxes."""
function tm5_cs_integral_workspace(grid::CubedSphereTargetGeometry, T, c::TM5SpectralFluxes; cache_dir=nothing, kwargs...)
    q = CSSpectralQuadrature(grid, T, c)
    faces = line_integral_face_matrix(q.mesh, grid.mesh, grid.face_table; samples_per_face=q.samples_per_face)
    overlap = build_regridder(q.mesh, grid.mesh; cache_dir)
    return CSSpectralIntegralWorkspace(q, T, faces, overlap; kwargs...)
end

"""Reuse synthesis/geometry and one graph factor across the serial run's days."""
function tm5_cs_run_artifacts(grid::CubedSphereTargetGeometry, T, c::TM5SpectralFluxes,
                              settings, vertical, cache)
    synthesis = get(settings,:spectral_synthesis,GPUSynthesis())
    precision = get(settings,:spectral_synthesis_float,AutoSynthesisPrecision())
    batch = min(8,vertical.Nz_native)
    cache_identity = (typeof(grid.mesh),grid.Nc,cs_geometry_tags(grid.mesh),grid.mesh.radius,T,
        c.refinement_steps,c.vod2uv_radius,typeof(synthesis),precision,batch)
    key = Symbol("tm5_cs_artifacts_",string(hash(cache_identity)))
    return cached_or_build!(cache,key) do
        integrals = tm5_cs_integral_workspace(grid,T,c;cache_dir=grid.cache_dir,
            synthesis,precision,batch)
        (;workspace=integrals,solver=DirectCSPoisson(grid.face_table))
    end
end

function tm5_cs_quadrature_metadata(ws::CSSpectralIntegralWorkspace)
    q = ws.quadrature
    return merge(spectral_synthesis_metadata(ws), Dict{String,Any}(
        "cs_face_quadrature"=>"great_circle_midpoint_bilinear_v1",
        "samples_per_face"=>q.samples_per_face,
        "refined_nlon"=>q.mesh.Nx, "refined_nlat"=>q.mesh.Ny,
        "cs_cell_quadrature"=>"conservative_refined_box_mean_v1"))
end

function tm5_cs_cell_panels(values::AbstractMatrix, grid, ::Type{FT}) where FT
    n, nz = grid.Nc, size(values, 2)
    panels = ntuple(_ -> zeros(FT, n, n, nz), 6)
    unpack_flat_to_panels_3d!(panels, values, n, nz)
    return panels
end

"""Close pristine dry rates against stored endpoints, retrying only amount stages."""
function close_tm5_cs_window(rate, ps, grid, settings, vertical, contract, policy, weights;
                             write_replay_check::Bool=true, window_index::Int=1)
    FT, n, nz = settings.output_float_type, grid.Nc, vertical.Nz
    m = tm5_cs_cell_panels(rate.m, grid, FT)
    m_next = tm5_cs_cell_panels(rate.m_next, grid, FT)
    ps_panels = ntuple(p -> reshape(FT.(ps[(p-1)*n*n+1:p*n*n]), n, n), 6)
    cm = ntuple(_ -> zeros(FT,n,n,nz+1), 6)
    dm = ntuple(_ -> zeros(FT,n,n,nz), 6)
    ft = grid.face_table
    degree, scratch = cs_cell_face_degree(ft), CSPoissonScratch(ft.nc)
    steps = initial_substeps(policy, exact_steps_per_window(settings.met_interval, settings.dt))
    while true
        seconds = flux_application_seconds(settings.met_interval, steps, :substep_mass_amount)
        am, bm = tm5_cs_directional_fluxes(FT.(rate.u.*seconds), grid)
        balance = balance_cs_column_mass_fluxes!(am,bm,m,m_next,ft,degree,steps,scratch;
            weights, fixed_global_mass=settings.mass_fix_enable && settings.mass_basis === :dry)
        fill_cs_window_mass_tendency!(dm,m,m_next,steps)
        closure = diagnose_balanced_cs_cm!(cm,am,bm,dm,m,n,nz,balance,"TM5 direct CS window",weights)
        # Residual booking can leave a rounded surface remainder. The direct
        # TM5 route enforces the closed physical boundary before replay.
        for panel in cm
            panel[:, :, end] .= zero(FT)
        end
        contract.steps_per_window = steps
        diag = verify_window!((; m_cur=m,am,bm,cm,m_next),contract,window_index;
            write_replay_on=write_replay_check)
        next = next_substeps(policy,steps,diag.positivity.ratio)
        if next == steps
            update_accumulator!(contract,diag.positivity,window_index)
            delta = ntuple(p -> m_next[p].-m[p],6)
            return (;payload=(;m,am,bm,cm,ps=ps_panels,dm=delta),steps,diag,
                closure=column_closure_record(balance,closure))
        end
        steps = next
    end
end

"""Stream retained-spectrum TM5 endpoints directly to a CS target and v5 binary."""
function process_day(c::TM5SpectralFluxes, date::Date, grid::CubedSphereTargetGeometry,
                     settings::ERA5SpectralSettings, vertical;
                     next_day_hour0=nothing,
                     positivity_cfl_limit::Real=DEFAULT_POSITIVITY_CFL_LIMIT,
                     require_substep_positivity::Bool=true, write_replay_check::Bool=true,
                     substep_policy=SubstepSchedulePolicy(adaptive_substeps=false,
                         substep_cfl_target=positivity_cfl_limit), run_cache=nothing)
    validate_spectral_construction(c,grid,settings)
    validate_spectral_vertical(c,vertical)
    validate_tm5_pbl_target(grid,settings,vertical)
    next_day_hour0 === nothing && error("tm5_spectral requires next-day 00 UTC; no zero-tendency fallback")
    settings.include_surface && throw(ArgumentError("direct TM5 CS surface payloads require include_tm5_pbl=true instead of include_surface"))
    paths = era5_spectral_paths(settings.spectral_dir,date)
    return with_spectral_hours(paths;cache_dir=settings.spectral_cache_dir) do read_hour
        first_spec = read_hour(0)
        T = first_spec.T
        next_day_hour0.T == T || error("spectral truncation changed at next-day endpoint")
        nt, FT, nz = 24, settings.output_float_type, vertical.Nz
        plan = TM5EndpointPlan(vertical)
        artifacts, reused = tm5_cs_run_artifacts(grid,T,c,settings,vertical,run_cache)
        ws, factor = artifacts.workspace, artifacts.solver
        @info "TM5 CS run artifacts" reused
        lons,lats = tm5_target_coordinates(grid)
        function endpoint(i)
            day, record = i <= nt ? (date,i) : (date+Dates.Day(1),1)
            thermo_path = era5_thermo_path(settings.thermo_dir,day)
            sx,sy = read_thermo_coordinates(thermo_path)
            humidity_weights = LatLonBilinearWeights(sx,sy,lons,lats)
            qll = read_qv_from_thermo(thermo_path,record,length(sx),length(sy),vertical.Nz_native;FT=Float64)
            q = zeros(grid.face_table.nc,vertical.Nz_native)
            interpolate_ll_qv!(q,qll,humidity_weights)
            h = i == 1 ? first_spec : i <= nt ? read_hour(i-1) : next_day_hour0
            h.T == T || error("spectral truncation changed within day")
            return construct_tm5_endpoint(ws,h.lnsp,view(h.vo,:,:,vertical.level_range),
                view(h.d,:,:,vertical.level_range),T,q,plan,settings)
        end
        weights = spectral_column_weights(grid,settings,vertical)
        steps = exact_steps_per_window(settings.met_interval,settings.dt)
        contract = CubedSphereContract{FT}(;replay_tol=replay_tolerance(FT),
            positivity_cfl_limit,require_substep_positivity,steps_per_window=steps)
        metadata = tm5_cs_quadrature_metadata(ws)
        merge!(metadata,Dict{String,Any}("preprocessor"=>"preprocess_transport_binary.jl",
            "source_type"=>"era5_spectral", "target_type"=>"cubed_sphere",
            "horizontal_balance"=>HORIZONTAL_BALANCE_TAG, "poisson_balanced"=>true,
            "vertical_mapping_method"=>String(vertical_mapping_method(vertical)),
            "merge_map"=>vertical.merge_map, "source_flux_sampling"=>"window_mean",
            "mass_fix_enabled"=>settings.mass_fix_enable,
            "mass_fix_target_ps_dry_pa"=>settings.target_ps_dry_pa,
            "adaptive_substeps"=>substep_policy.adaptive_substeps,
            "runtime_substep_contract"=>"binary_schedule"))
        with_replay_record(metadata,write_replay_check)
        with_column_weights_record(metadata,get(settings,:column_balance_weights,:hybrid_mass))
        mkpath(settings.out_dir)
        path = output_binary_path(date,settings.out_dir,settings.min_dp,FT)
        staging = open_preprocessing_staging(path)
        writer = nothing
        inner = nothing
        pbl_cache = get(settings,:include_tm5_pbl,false) ? TM5PBLInputCache() : nothing
        schedule, offsets = Int[], Float64[]
        try
            a = endpoint(1)
            first_spec = nothing
            push!(offsets,a.offset)
            inner = open_streaming_cs_transport_binary(staging,grid.Nc,CS_PANEL_COUNT,nz,nt,vertical.merged_vc;
                FT,dt_met_seconds=settings.met_interval,half_dt_seconds=settings.half_dt,
                steps_per_window=steps,include_flux_delta=true,source_flux_sampling=:window_mean,
                include_tm5_pbl=pbl_cache !== nothing,mass_basis=:dry,surface_pressure_basis=TotalSurfacePressure(),
                planet_radius=grid.mesh.radius,cs_geometry_tags(grid.mesh)...,
                flux_construction=TM5SpectralFluxesHeader(1,T,c.refinement_steps,c.vod2uv_radius),extra_header=metadata)
            writer = CubedSphereBinaryWriter(inner,State.mass_basis(:dry);
                Nc=grid.Nc,npanel=CS_PANEL_COUNT,final_path=path)
            for i in 1:nt
                b = endpoint(i+1)
                push!(offsets,b.offset)
                rates = balance_window_layers!(c,a,b,grid,factor,settings.met_interval,ws.quadrature.areas)
                closed = close_tm5_cs_window(rates,a.ps,grid,settings,vertical,contract,substep_policy,weights;
                    write_replay_check, window_index=i)
                payload = closed.payload
                pbl_cache === nothing || (payload=merge(payload,(;tm5_pbl=_cs_tm5_pbl_inputs(
                    date,i,settings,vertical,payload.ps,grid,pbl_cache))))
                push!(schedule,closed.steps)
                @info "TM5 CS window closure" window=i steps=closed.steps replay=closed.diag.replay moist=map(x->x.post,rates.diagnostics)
                evidence = tm5_flux_evidence(CubedSphereTargetGeometry,FT)
                write_window!(writer,ReadyWindow{CubedSphereTargetGeometry,FT}(i,payload,
                    FluxFidelityEvidenceBundle(evidence.flux,evidence.omega,closed.closure)))
                a = b
            end
            inner.header["ps_offsets_pa_per_window"] = offsets[1:end-1]
            inner.header["ps_offsets_next_day_hour0_pa"] = offsets[end]
            set_streaming_steps_per_window_schedule!(inner,schedule)
            summarize_status!(contract)
            promote_streaming_binary!(writer)
        catch
            inner === nothing || abandon_streaming_transport_binary!(inner)
            isfile(staging) && rm(staging;force=true)
            rethrow()
        finally
            pbl_cache === nothing || close(pbl_cache)
        end
        return path
    end
end
