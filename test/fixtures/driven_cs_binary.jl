function write_driven_cs_binary(path::AbstractString;
                                FT::Type{<:AbstractFloat} = Float64,
                                Nc::Int = 4,
                                Nz::Int = 2,
                                window_mass_scales::Tuple{Vararg{Real}} = (1, 1),
                                window_dm_panels = nothing,
                                surface_windows = nothing,
                                vdiff_windows = nothing,
                                convection_windows = nothing,
                                dtrain_windows = nothing,
                                tm5_windows = nothing,
                                steps_per_window::Int = 2,
                                steps_per_window_by_window = nothing)
    window_dm_panels !== nothing &&
        length(window_dm_panels) == length(window_mass_scales) ||
        window_dm_panels === nothing || throw(ArgumentError("window_dm_panels length must match window_mass_scales"))
    surface_windows !== nothing &&
        length(surface_windows) == length(window_mass_scales) ||
        surface_windows === nothing || throw(ArgumentError("surface_windows length must match window_mass_scales"))
    vdiff_windows !== nothing &&
        length(vdiff_windows) == length(window_mass_scales) ||
        vdiff_windows === nothing || throw(ArgumentError("vdiff_windows length must match window_mass_scales"))
    convection_windows !== nothing &&
        length(convection_windows) == length(window_mass_scales) ||
        convection_windows === nothing || throw(ArgumentError("convection_windows length must match window_mass_scales"))
    dtrain_windows !== nothing &&
        length(dtrain_windows) == length(window_mass_scales) ||
        dtrain_windows === nothing || throw(ArgumentError("dtrain_windows length must match window_mass_scales"))
    tm5_windows !== nothing &&
        length(tm5_windows) == length(window_mass_scales) ||
        tm5_windows === nothing || throw(ArgumentError("tm5_windows length must match window_mass_scales"))
    vc = if Nz == 2
        AtmosTransport.HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    elseif Nz == 5
        AtmosTransport.HybridSigmaPressure(FT[0, 100, 300, 600, 1000, 2000],
                                           FT[0, 0, 0.1, 0.3, 0.7, 1])
    else
        AtmosTransport.HybridSigmaPressure(
            FT.(collect(range(0, stop = 2000, length = Nz + 1))),
            FT.(collect(range(0, stop = 1, length = Nz + 1))))
    end
    writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
        path, Nc, 6, Nz, length(window_mass_scales), vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
        FT = FT,
        dt_met_seconds = 3600.0,
        steps_per_window = steps_per_window,
        include_flux_delta = window_dm_panels !== nothing,
        mass_basis = :dry,
        surface_pressure_basis = TotalSurfacePressure(),
        include_surface = surface_windows !== nothing,
        include_cmfmc = convection_windows !== nothing,
        include_dtrain = dtrain_windows !== nothing,
        include_tm5conv = tm5_windows !== nothing,
        include_gchp_vdiff = vdiff_windows !== nothing,
    )
    steps_per_window_by_window !== nothing &&
        AtmosTransport.MetDrivers.set_streaming_steps_per_window_schedule!(
            writer, steps_per_window_by_window)

    for (win, scale) in enumerate(window_mass_scales)
        window = (
            m  = ntuple(_ -> fill(FT(scale), Nc, Nc, Nz), 6),
            am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6),
            bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6),
            cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6),
            ps = ntuple(_ -> fill(FT(90000), Nc, Nc), 6),
        )
        if convection_windows !== nothing
            window = merge(window, (cmfmc = convection_windows[win],))
        end
        if surface_windows !== nothing
            window = merge(window, (surface = surface_windows[win],))
        end
        if vdiff_windows !== nothing
            window = merge(window, (vdiff = vdiff_windows[win],))
        end
        if dtrain_windows !== nothing
            window = merge(window, (dtrain = dtrain_windows[win],))
        end
        if tm5_windows !== nothing
            window = merge(window, (tm5_fields = tm5_windows[win],))
        end
        if window_dm_panels !== nothing
            window = merge(window, (dm = window_dm_panels[win],))
        end
        AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, 6)
    end

    AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)
    return nothing
end
