# ===========================================================================
# Region-tagged surface fluxes (`[tagged_fluxes.<group>]`)
#
# One flux dataset and an integer region mask on the flux's own grid are split
# into one tracer per selected mask value. The runner expands each group into
# ordinary `[tracers.<name>]` entries whose surface-flux table is the group's
# `flux` table plus the selection under `TAGGED_REGION_KEY`
# (`DrivenRunner._expand_tagged_fluxes`). `build_surface_flux_sources` hands
# the tracers of one group to `_build_tagged_surface_flux_sources`, which
# reads the flux once, multiplies it by each value's 0/1 indicator on the
# source grid, and builds an ordinary `SurfaceFluxSource` or
# `TimeVaryingSurfaceFluxSource` per tracer with the untagged builders' code.
# ===========================================================================

"""
    TaggedFluxRegion(group, mask_file, mask_variable, value)

The region selection of one tracer of a `[tagged_fluxes.<group>]` table: the
cells of the mask `mask_variable` in `mask_file` whose code is `value`. The
runner stores it under `TAGGED_REGION_KEY` in the tracer's surface-flux
table; it is not a TOML key.
"""
struct TaggedFluxRegion
    group         :: String
    mask_file     :: String
    mask_variable :: String
    value         :: Int64
end

"""
    TAGGED_REGION_KEY

The surface-flux table key that carries a tracer's [`TaggedFluxRegion`](@ref).
Its value is a Julia object, which TOML cannot write, so only the expansion of
`[tagged_fluxes]` sets it.
"""
const TAGGED_REGION_KEY = "_tagged_region"

"""
    tagged_flux_region(surface_flux_cfg) -> Union{TaggedFluxRegion, Nothing}

The region selection of a tracer's surface-flux table, or `nothing` for an
untagged flux.
"""
tagged_flux_region(cfg) = (r = get(cfg, TAGGED_REGION_KEY, nothing); r isa TaggedFluxRegion ? r : nothing)

const _SECONDS_PER_JULIAN_YEAR = DAYS_PER_JULIAN_YEAR * SECONDS_PER_DAY

"""
    tagged_flux_species(group, names, flux_cfg) -> Symbol

The species of the tracers `names` of `[tagged_fluxes.<group>]` with the flux
table `flux_cfg` ([`surface_flux_species`](@ref) of each). The flux is read,
and its units converted, once for the group, so all must resolve to one
species; otherwise, or when a name implies none and `flux.species` is unset,
an `ArgumentError` names the group.
"""
function tagged_flux_species(group, names, flux_cfg)
    label = "[tagged_fluxes.$(group)]"
    species = try
        unique(surface_flux_species(Symbol(name), flux_cfg) for name in names)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("$(label): set flux.species (tagged tracer names imply no species; " *
                            "known species: $(_known_species_list()))"))
    end
    length(species) == 1 || throw(ArgumentError(
        "$(label): its tracers resolve to different species ($(join(species, ", "))); " *
        "set flux.species"))
    return only(species)
end

"""
    _build_tagged_surface_flux_sources(grid, specs, FT; reference_time, run_span) -> Vector

Sources of the tracers `specs` of one `[tagged_fluxes]` group, in the order of
`specs` (`nothing` for a group whose flux kind is `none`). The flux is read
once; the mask (`GriddedIO.read_region_mask`) must be on its grid
(`GriddedIO.check_region_mask_grid`), and each tracer's field is the flux
times the 0/1 indicator of its value, regridded with one shared regridder.
The setup log gives each tracer's emission and the group total next to the
flux over the selected values and the whole flux, all on the source grid.
"""
function _build_tagged_surface_flux_sources(grid, specs, ::Type{FT};
                                            reference_time = nothing,
                                            run_span = nothing) where FT
    regions = [tagged_flux_region(spec.surface_flux_cfg) for spec in specs]
    group = first(regions).group
    cfg = first(specs).surface_flux_cfg
    kind = _surface_flux_kind(cfg)
    kind === :none && return Any[nothing for _ in specs]
    species = tagged_flux_species(group, [spec.name for spec in specs], cfg)
    mask = read_region_mask(first(regions).mask_file, first(regions).mask_variable)
    # The expansion checked the values against the mask it read; the file may
    # have changed since, and a vanished value would silently emit nothing.
    present = Set(region_mask_values(mask))
    absent = [r.value for r in regions if !(r.value in present)]
    isempty(absent) || throw(ArgumentError(
        "[tagged_fluxes.$(group)]: mask '$(mask.variable)' in $(mask.file) no longer holds " *
        "$(join(absent, ", "))"))
    label = "[tagged_fluxes.$(group)] flux"
    built = _tagged_sources(grid.horizontal, kind, specs, regions, mask, cfg, species, FT;
                            reference_time, run_span, label)
    _log_tagged_group(group, species, specs, built)
    return Any[s for s in built.sources]
end

# Each method returns `(; sources, totals, selected_total, file_total)`: the
# sources in the order of `specs`, their physical rates [kg species s⁻¹], and
# on the source grid the flux over the selected values and the whole flux.

# Static fields, every topology: one load, one regridder.
function _tagged_sources(mesh, kind, specs, regions, mask, cfg, species, ::Type{FT};
                         reference_time, run_span, label) where FT
    _surface_flux_time_varying(cfg) &&
        return _tagged_timevarying_sources(mesh, kind, specs, regions, mask, cfg, species, FT;
                                           reference_time, run_span, label)
    # The untagged builder's checks (cs_native, time-varying on LL/RG).
    kind === :cs_native && throw(ArgumentError(
        "surface_flux kind = \"cs_native\" requires time_varying = true" *
        (mesh isa CubedSphereMesh ? "" : " and a cubed-sphere run")))
    mesh isa CubedSphereMesh && !(_regridding_method(cfg) isa ConservativeFluxRegridding) &&
        @warn "CS surface-flux: `regridding = \"$(cfg["regridding"])\"` requested; CS supports conservative only — forcing conservative."
    field = _load_file_surface_flux_field(cfg, FT; species, radius = mesh.radius)
    check_region_mask_grid(mask, field.lon, field.lat; source = label)
    conservative = mesh isa CubedSphereMesh || _regridding_method(cfg) isa ConservativeFluxRegridding
    regridder = conservative ? _build_surface_flux_regridder(field.lon, field.lat, mesh) : nothing
    sources = Any[]
    selected_total = 0.0
    for (spec, region) in zip(specs, regions)
        raw = field.raw .* region_indicator(mask, region.value)
        total = sum(Float64.(raw) .* field.cell_area)
        selected_total += total
        masked = FileSurfaceFluxField{FT}(raw, field.lon, field.lat, field.cell_area, total)
        push!(sources, _static_surface_flux_source(mesh, spec.name, spec.surface_flux_cfg,
                                                   masked, FT; regridder))
    end
    totals = [total_emission_rate(s, 0.0) / _surface_flux_storage_scale(spec.name, spec.surface_flux_cfg)
              for (s, spec) in zip(sources, specs)]
    return (; sources, totals, selected_total, file_total = field.native_total_mass_rate)
end

function _tagged_timevarying_sources(mesh, kind, specs, regions, mask, cfg, species, ::Type{FT};
                                     reference_time, run_span, label) where FT
    mesh isa CubedSphereMesh || throw(ArgumentError(
        "time-varying surface flux is supported on cubed-sphere grids only"))
    _surface_flux_supports_time_varying(kind) || throw(ArgumentError(
        "time-varying surface flux not supported for kind=$(kind); " *
        "supported: :lmdz_co2, :gridfed_fossil_co2, :cs_native"))
    kind === :cs_native &&
        return _tagged_native_sources(mesh, specs, regions, mask, cfg, FT;
                                      reference_time, run_span, label)
    # The grid gate before the series is read, on every listed file (the
    # loader requires them to share one grid, but reads only those of the run).
    files, variable = _resolve_timevarying_surface_flux_files(cfg, kind, reference_time)
    for file in files
        check_region_mask_grid(mask, _oriented_flux_centres(file, variable)...;
                               source = "$(label) ($(basename(file)))")
    end
    field = _load_timevarying_surface_flux_field(cfg, FT, reference_time; species, run_span,
                                                 source = label)
    check_region_mask_grid(mask, field.lon, field.lat; source = label)
    regridder = _build_surface_flux_regridder(field.lon, field.lat, mesh)
    area = reshape(regridder.src_areas, size(field.raw_series, 1), size(field.raw_series, 2))
    ntime = size(field.raw_series, 3)
    slice_mean(weight) = sum(t -> sum(Float64.(view(field.raw_series, :, :, t)) .* weight .* area),
                             1:ntime) / ntime
    sources = [_timevarying_cs_surface_flux_source(mesh, spec.name, spec.surface_flux_cfg, field,
                                                   regridder, FT;
                                                   weight = region_indicator(mask, region.value))
               for (spec, region) in zip(specs, regions)]
    selected = sum(region -> slice_mean(region_indicator(mask, region.value)), regions)
    return (; sources, totals = _series_mean_totals(sources, specs), selected_total = selected,
            file_total = slice_mean(1.0))
end

# `cs_native`: the flux is already on the run's cube, so is the mask; the
# untagged builder reads the series once, and each tracer is that series times
# its indicator.
function _tagged_native_sources(mesh::CubedSphereMesh, specs, regions, mask, cfg, ::Type{FT};
                                reference_time, run_span, label) where FT
    check_region_mask_grid(mask, mesh; source = label)
    base = _build_native_timevarying_cs_surface_flux_source(
        mesh, first(specs).name, cfg, FT, reference_time; run_span)
    sources = map(zip(specs, regions)) do (spec, region)
        indicator = region_indicator(mask, region.value)
        series = ntuple(p -> base.cell_mass_rate_series[p] .* view(indicator, :, :, p),
                        CS_PANEL_COUNT)
        TimeVaryingSurfaceFluxSource(spec.name, series, copy(base.times), base.scheme)
    end
    # The flux over the selected values, from the untagged series and the
    # union of the selected cells.
    selected = falses(size(mask.codes))
    for region in regions
        selected .|= region_indicator(mask, region.value)
    end
    union = TimeVaryingSurfaceFluxSource(first(specs).name,
        ntuple(p -> base.cell_mass_rate_series[p] .* view(selected, :, :, p), CS_PANEL_COUNT),
        copy(base.times), base.scheme)
    return (; sources, totals = _series_mean_totals(sources, specs),
            selected_total = only(_series_mean_totals((union,), specs[1:1])),
            file_total = only(_series_mean_totals((base,), specs[1:1])))
end

# Oriented lon/lat centres of the flux variable `variable` in `file`, read
# without its data.
function _oriented_flux_centres(file, variable)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))
    return NCDataset(file) do ds
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))
        lon, lat = _latlon_centres(ds, ds[variable])
        orient_coordinates(lon, lat, latlon_orientation(lon, lat))
    end
end

# Physical kg species s⁻¹ of each source, the mean over its slices.
_series_mean_totals(sources, specs) =
    [sum(t -> total_emission_rate(s, t), s.times) / length(s.times) /
     _surface_flux_storage_scale(spec.name, spec.surface_flux_cfg)
     for (s, spec) in zip(sources, specs)]

function _log_tagged_group(group, species, specs, built)
    (; sources, totals, selected_total, file_total) = built
    pg(x) = x * _SECONDS_PER_JULIAN_YEAR / 1e12
    time_varying = any(s -> s isa TimeVaryingSurfaceFluxSource, sources)
    basis = time_varying ? "mean over the loaded slices" : "static"
    for (spec, total) in zip(specs, totals)
        @info @sprintf("Tagged flux %s / %s: %.6e kg %s/s (%.6g Pg %s/yr, %s)",
                       group, String(spec.name), total, species, pg(total), species, basis)
    end
    tagged = sum(totals)
    # An absolute residual: emission and uptake can cancel in a net total.
    @info @sprintf("Tagged flux %s: %d tracers, sum on the model grid %.6e kg %s/s; on the flux grid: selected values %.6e kg/s (residual %.3e kg/s), whole flux %.6e kg/s (%s)",
                   group, length(specs), tagged, species, selected_total, tagged - selected_total,
                   file_total, basis)
    return nothing
end
