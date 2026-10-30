"""
    _build_native_timevarying_cs_surface_flux_source(mesh, tracer_name, cfg,
                                                      FT, reference_time; run_span = nothing)

Build a `TimeVaryingSurfaceFluxSource` from a `cs_native` flux series
(`surface_flux.kind = "cs_native"`): densities already on the runtime cube,
checked by `GriddedIO.cs_native_flux_info` (layout
`GriddedIO.CS_NATIVE_FLUX_DIMS`, units kg species or kgC m⁻² s⁻¹, cell-area
gate, strictly increasing times) and read panel by panel by
`GriddedIO.read_cs_native_flux!` straight into the source's `FT` arrays (no
fill or non-finite values), so the series is held once. The runtime adds three
things:

1. slice times in seconds since the run start `reference_time`
   (`GriddedIO.seconds_since`); with `run_span = (t_start, t_end)` the slice
   intervals (`GriddedIO.decode_slice_intervals`: `time_bnds`, else the
   stamps' spacing) must cover the run unless `hold_outside_coverage = true`
   (`_check_flux_time_coverage`), and only the slices `_needed_slice_window`
   keeps are read;
2. per-cell rates, in place: density × `mesh.cell_areas` × the units factor
   (`GriddedIO.mass_rate_factor` to the tracer's species,
   [`surface_flux_species`](@ref): 1 for kg of that species or unnamed kg,
   M(CO2)/M(C) for kgC; kg of another species is an error) × `scale` × the
   storage factor `DRY_AIR_MOLAR_MASS / M_species`;
3. the temporal scheme (`temporal_scheme`, default `"stepwise"`: each slice
   holds from its stamp until the next).

No horizontal interpolation is performed.
"""
function _build_native_timevarying_cs_surface_flux_source(
        mesh::CubedSphereMesh, tracer_name::Symbol, cfg, ::Type{FT},
        reference_time::Union{DateTime, Nothing}; run_span = nothing) where FT
    file, variable, _ = _resolve_surface_flux_file(cfg, :cs_native)
    _require_reference_time(reference_time)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))
    # Only the slices the run can use (a contiguous range of the ascending axis).
    slices = run_span === nothing ? Colon() :
             times -> _native_flux_slices(seconds_since(times, reference_time), run_span)
    Nc = mesh.Nc
    panels_series, flux = NCDataset(file) do ds
        info = cs_native_flux_info(ds, variable; mesh, slices)
        if run_span !== nothing && !_hold_outside_coverage(cfg)
            # Time is the last Julia dimension of `CS_NATIVE_FLUX_DIMS`.
            time_var = find_coordinate(ds, TimeAxis(); length = size(ds[variable], 4))
            label = "surface flux of tracer $(tracer_name) (kind = cs_native; $(basename(file)))"
            intervals = _in_coverage_context(() -> decode_slice_intervals(ds, time_var), label)
            _check_flux_time_coverage(_interval_seconds(intervals, reference_time)...,
                                      run_span, reference_time; source = label)
        end
        # The source's own arrays, filled panel by panel with FT densities.
        panels = ntuple(_ -> Array{FT}(undef, Nc, Nc, length(info.slices)), CS_PANEL_COUNT)
        read_cs_native_flux!(panels, ds, variable, info.slices; mesh)
        panels, info
    end
    times_sec = seconds_since(flux.times, reference_time)

    species_scale = FT(mass_rate_factor(flux.units, surface_flux_species(tracer_name, cfg)))
    scale = species_scale * FT(get(cfg, "scale", 1.0)) *
            FT(_surface_flux_storage_scale(tracer_name, cfg))
    area = reshape(FT.(mesh.cell_areas), Nc, Nc, 1)
    for panel in panels_series
        panel .= panel .* area .* scale         # (density × area) × scale, in FT
    end

    # Hourly flux fields represent interval means and are held constant
    # over their stamped hour unless the config explicitly requests a
    # different reconstruction.
    scheme = flux_temporal_scheme(String(get(cfg, "temporal_scheme", "stepwise")))
    return TimeVaryingSurfaceFluxSource(tracer_name, panels_series, times_sec, scheme)
end

# Slice indices of the ascending `times_sec` (seconds since the run start) that
# can affect the run `run_span`.
function _native_flux_slices(times_sec::AbstractVector{<:Real}, run_span)
    t_lo, t_hi = _needed_slice_window(times_sec, run_span)
    return searchsortedfirst(times_sec, t_lo):searchsortedlast(times_sec, t_hi)
end
