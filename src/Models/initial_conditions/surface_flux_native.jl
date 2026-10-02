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
        reference_time::Union{DateTime, Nothing}) where FT
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

        units_norm = _normalize_units_string(get(raw_var.attrib, "units", ""))
        species_scale = if units_norm in ("kgcm-2s-1", "kgc/m2/s", "kgcm2s-1")
            FT(44.0 / 12.0)
        elseif isempty(units_norm) || occursin("/s", units_norm) || occursin("s-1", units_norm)
            one(FT)
        else
            throw(ArgumentError(
                "native cubed-sphere surface flux has unsupported units '$units_norm' " *
                "in $file; expected a per-area, per-second mass flux"))
        end
        scale = species_scale * FT(get(cfg, "scale", 1.0)) *
                FT(_surface_flux_storage_scale(tracer_name, cfg))

        area = reshape(FT.(mesh.cell_areas), Nc, Nc, 1)
        panels_series = ntuple(p -> begin
            density = FT.(nomissing(raw_var[:, :, p, :], zero(FT)))
            all(isfinite, density) || throw(ArgumentError(
                "native cubed-sphere flux contains non-finite values on panel $p"))
            density .* area .* scale
        end, CS_PANEL_COUNT)

        time_units = String(get(ds[time_var].attrib, "units", ""))
        times_sec = _surface_flux_times_seconds(ds[time_var][:], time_units, reference_time)
        issorted(times_sec) || throw(ArgumentError(
            "native cubed-sphere surface-flux times must be ascending"))

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

