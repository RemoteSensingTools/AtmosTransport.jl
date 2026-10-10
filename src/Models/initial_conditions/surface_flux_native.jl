"""
    _build_native_timevarying_cs_surface_flux_source(mesh, tracer_name, cfg,
                                                      FT, reference_time)

Load an already aligned GEOS-native cubed-sphere flux-density series with
Julia/NCDatasets shape `(Nc, Nc, 6, ntime)`, corresponding to NetCDF dimensions
`(time, nf, Ydim, Xdim)`. The input is kg species m⁻² s⁻¹. It is multiplied by
the runtime mesh's exact cell areas and converted to the model storage basis.
No horizontal interpolation is performed.
"""
function _build_native_timevarying_cs_surface_flux_source(
        mesh::CubedSphereMesh, tracer_name::Symbol, cfg, ::Type{FT},
        reference_time::Union{DateTime, Nothing}; run_span = nothing) where FT
    file, variable, _ = _resolve_surface_flux_file(cfg, :cs_native)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))

    ds = NCDataset(file)
    try
        time_var = _ic_find_coord(ds, ["time", "t"])
        isnothing(time_var) && throw(ArgumentError(
            "could not find time coordinate in native cubed-sphere flux $file"))
        haskey(ds, variable) || throw(ArgumentError(
            "variable '$variable' not found in $file"))

        raw_var = ds[variable]
        ndims(raw_var) == 4 || throw(ArgumentError(
            "native cubed-sphere surface-flux variable '$variable' must be 4D " *
            "(Xdim,Ydim,nf,time in Julia), got ndims=$(ndims(raw_var))"))
        Nc = mesh.Nc
        size(raw_var, 1) == Nc && size(raw_var, 2) == Nc || throw(DimensionMismatch(
            "native cubed-sphere flux has horizontal shape " *
            "$(size(raw_var, 1))x$(size(raw_var, 2)); transport grid is C$(Nc)"))
        size(raw_var, 3) == CS_PANEL_COUNT || throw(DimensionMismatch(
            "native cubed-sphere flux has nf=$(size(raw_var, 3)); expected $CS_PANEL_COUNT"))
        ntime = size(raw_var, 4)
        ntime > 0 || throw(ArgumentError("native cubed-sphere flux has no time slices"))

        _require_reference_time(reference_time)
        time_units = String(get(ds[time_var].attrib, "units", ""))
        times_all = _surface_flux_times_seconds(ds[time_var][:], time_units, reference_time)
        issorted(times_all) || throw(ArgumentError(
            "native cubed-sphere surface-flux times must be ascending"))
        # Only the slices the run can use (a contiguous range of the ascending axis).
        kr = if run_span === nothing
            1:ntime
        else
            t_lo, t_hi = _needed_slice_window(times_all, run_span)
            searchsortedfirst(times_all, t_lo):searchsortedlast(times_all, t_hi)
        end
        times_sec = times_all[kr]

        units_norm = _normalize_units_string(get(raw_var.attrib, "units", ""))
        species_scale = FT(_native_flux_species_scale(units_norm, file))
        scale = species_scale * FT(get(cfg, "scale", 1.0)) *
                FT(_surface_flux_storage_scale(tracer_name, cfg))

        _check_native_flux_cell_area(ds, mesh, file)
        area = reshape(FT.(mesh.cell_areas), Nc, Nc, 1)
        panels_series = ntuple(p -> begin
            panel = raw_var[:, :, p, kr]
            any(ismissing, panel) && throw(ArgumentError(
                "native cubed-sphere flux has fill values on panel $p of $file"))
            density = FT.(panel)
            all(isfinite, density) || throw(ArgumentError(
                "native cubed-sphere flux contains non-finite values on panel $p"))
            density .* area .* scale
        end, CS_PANEL_COUNT)

        # Hourly flux fields represent interval means and are held constant
        # over their stamped hour unless the config explicitly requests a
        # different reconstruction.
        scheme = flux_temporal_scheme(String(get(cfg, "temporal_scheme", "stepwise")))
        return TimeVaryingSurfaceFluxSource(
            tracer_name, panels_series, times_sec, scheme)
    finally
        close(ds)
    end
end

# kg m⁻² s⁻¹ of the tracer species, or of carbon (converted to CO2 by 44/12);
# molar, gram and missing units are rejected (`_per_second_flux_scale`).
_native_flux_species_scale(units_norm::AbstractString, file) = _per_second_flux_scale(units_norm, file)

# A flux file built on another cube geometry (for example equiangular vs the
# GMAO definition) has different cell areas; when the file records them,
# they must match the runtime mesh or the emitted mass is silently wrong.
function _check_native_flux_cell_area(ds, mesh::CubedSphereMesh, file)
    haskey(ds, "cell_area") || return nothing
    file_area = Float64.(nomissing(Array(ds["cell_area"]), NaN))
    mesh_area = Float64.(mesh.cell_areas)
    Nc = mesh.Nc
    panels = if size(file_area) == (Nc, Nc)
        (file_area,)
    elseif size(file_area) == (Nc, Nc, CS_PANEL_COUNT)
        ntuple(p -> view(file_area, :, :, p), CS_PANEL_COUNT)
    else
        throw(DimensionMismatch("cell_area in $file has shape $(size(file_area)); " *
                                "expected ($Nc, $Nc) or ($Nc, $Nc, $CS_PANEL_COUNT)"))
    end
    for (p, a) in enumerate(panels)
        isapprox(a, mesh_area; rtol = 1e-4) || throw(ArgumentError(
            "cell_area in $file (panel $p) differs from the runtime C$(Nc) mesh areas; " *
            "the flux was prepared on another cube geometry"))
    end
    return nothing
end
