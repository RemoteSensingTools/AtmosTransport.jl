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
        species_scale = FT(_native_flux_species_scale(units_norm, file))
        scale = species_scale * FT(get(cfg, "scale", 1.0)) *
                FT(_surface_flux_storage_scale(tracer_name, cfg))

        _check_native_flux_cell_area(ds, mesh, file)
        area = reshape(FT.(mesh.cell_areas), Nc, Nc, 1)
        panels_series = ntuple(p -> begin
            panel = raw_var[:, :, p, :]
            any(ismissing, panel) && throw(ArgumentError(
                "native cubed-sphere flux has fill values on panel $p of $file"))
            density = FT.(panel)
            all(isfinite, density) || throw(ArgumentError(
                "native cubed-sphere flux contains non-finite values on panel $p"))
            density .* area .* scale
        end, CS_PANEL_COUNT)

        reference_time === nothing && @warn(
            "native cubed-sphere surface flux: no reference_time supplied; assuming " *
            "the file's time origin equals the run start (first slice → t=0).")
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

# kg m⁻² s⁻¹ of the tracer species (`kg CO2 m-2 s-1`, `kg m-2 s-1`), or of
# carbon (`kgC m-2 s-1`, converted to CO2 by 44/12). Molar or gram units are
# rejected rather than silently mis-scaled; missing units are assumed to be
# kg species m⁻² s⁻¹ with a warning.
const _NATIVE_FLUX_UNITS_RE = r"^kg([a-z0-9]*)(m-2s-1|/m2/s|m2s-1|/m2s)$"

function _native_flux_species_scale(units_norm::AbstractString, file)
    if isempty(units_norm)
        @warn "native cubed-sphere surface flux in $file has no units; assuming kg species m-2 s-1"
        return 1.0
    end
    m = match(_NATIVE_FLUX_UNITS_RE, units_norm)
    m === nothing && throw(ArgumentError(
        "native cubed-sphere surface flux has unsupported units '$units_norm' in $file; " *
        "expected kg species m-2 s-1 or kgC m-2 s-1"))
    return m.captures[1] == "c" ? 44.0 / 12.0 : 1.0
end

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
