# ===========================================================================
# Surface-flux loader + builders
#
# Owns the file-based surface-flux path: NetCDF load + LL→{LL,RG,CS}
# conservative regrid + area integration. Every builder returns
# `SurfaceFluxSource` with cell_mass_rate in model-storage units per cell per
# second. File inventories are physical kg species/s; the builders convert
# them to dry-air-equivalent storage for the dry-VMR transport state.
#
# Contents:
#   - static fields: `FileSurfaceFluxField`, loaded by
#     `_load_file_surface_flux_field`;
#   - time-varying series: `TimeVaryingFileSurfaceFluxField`, loaded by
#     `_load_timevarying_surface_flux_field` (CS runs only);
#   - builders: `build_surface_flux_source` (LL, RG, CS) and
#     `build_surface_flux_sources`;
#   - the species of a tracer's flux: `surface_flux_species`.
# The file conventions (coordinates, orientation, fill values, time axes, flux
# units, source meshes) come from `GriddedIO`. Regridding helpers live in
# `surface_flux_regridding.jl`; the GEOS-native CS series loader lives in
# `surface_flux_native.jl`.
# ===========================================================================

# Tracer names that imply the species of their surface flux, so that
# `surface_flux.species` may be omitted.
const _TRACER_NAME_SPECIES = (co2 = :co2, co2_natural = :co2, co2_fossil = :co2,
                              fossil_co2 = :co2, sf6 = :sf6, rn222 = :rn222)

_known_species_list() = join(("\"$(s)\"" for s in propertynames(SPECIES_MOLAR_MASS)), ", ")

"""
    surface_flux_species(tracer_name::Symbol, cfg) -> Symbol

The species whose kilograms the surface flux `cfg` of tracer `tracer_name`
counts: `cfg["species"]` (`"co2"`, `"sf6"` or `"rn222"`, the entries of
`Parameters.SPECIES_MOLAR_MASS`), or, when the key is absent, the species the
tracer name implies (`co2`, `co2_natural`, `co2_fossil`, `fossil_co2` → `:co2`;
`sf6` → `:sf6`; `rn222` → `:rn222`). An unknown species, or a tracer with
another name and no `species`, is an `ArgumentError` that names the known
species.

The species sets the conversion of physical kg species s⁻¹ to the model's
storage units, `DRY_AIR_MOLAR_MASS / SPECIES_MOLAR_MASS[species]` (storage is
dry VMR × dry-air mass). It also decides how the flux units convert
(`GriddedIO.kg_per_unit`): kg of another named species is an error, and kg of
carbon is converted to the species by `M(species) / M(C)`.
"""
function surface_flux_species(tracer_name::Symbol, cfg)
    haskey(cfg, "molar_mass_kg_mol") && throw(ArgumentError(
        "surface flux of tracer $(tracer_name): molar_mass_kg_mol was replaced by " *
        "species = $(_known_species_list()); the molar mass comes from SPECIES_MOLAR_MASS"))
    if haskey(cfg, "species")
        raw = cfg["species"]
        species = raw isa AbstractString ? Symbol(raw) : nothing
        species !== nothing && hasproperty(SPECIES_MOLAR_MASS, species) && return species
        throw(ArgumentError(
            "surface flux of tracer $(tracer_name): species = $(repr(raw)) is not a known " *
            "species; known species: $(_known_species_list())"))
    end
    hasproperty(_TRACER_NAME_SPECIES, tracer_name) &&
        return getproperty(_TRACER_NAME_SPECIES, tracer_name)
    throw(ArgumentError(
        "surface flux for tracer $(tracer_name) is in physical kg species/s, but the " *
        "transport state stores dry VMR times dry-air mass. Set surface_flux.species to " *
        "one of $(_known_species_list()) so the source can be converted to model storage " *
        "units (only tracers named $(join(propertynames(_TRACER_NAME_SPECIES), ", ")) " *
        "may omit it)."))
end

# The inventory year in a file name: a standalone 19xx/20xx not preceded by a
# version marker (`GCP-GridFEDv2024.0_2022.short.nc` is 2022). Several
# distinct candidates are ambiguous and must be resolved with `year`.
function _infer_year_from_path(path::AbstractString)
    name = basename(path)
    years = Int[]
    for m in eachmatch(r"(?<!\d)(?:19|20)\d{2}(?!\d)", name)
        m.offset > 1 && name[prevind(name, m.offset)] in ('v', 'V') && continue
        push!(years, parse(Int, m.match))
    end
    unique!(years)
    isempty(years) && return nothing
    length(years) == 1 || throw(ArgumentError(
        "surface-flux file name $(repr(name)) contains several years $(years); " *
        "set surface_flux.year"))
    return only(years)
end

function _infer_year_from_time_units(ds)
    haskey(ds, "time") || return nothing
    units = String(get(ds["time"].attrib, "units", ""))
    m = match(r"((?:19|20)\d{2})-\d{1,2}-\d{1,2}", units)
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function _surface_flux_year(cfg, file::AbstractString, ds)
    haskey(cfg, "year") && return Int(cfg["year"])
    year = _infer_year_from_time_units(ds)
    year === nothing || return year
    year = _infer_year_from_path(file)
    year === nothing || return year
    throw(ArgumentError(
        "per-month flux units (gridfed_fossil_co2: kgCO2/month/m2) need a calendar " *
        "year to convert monthly totals to per-second rates. Set surface_flux.year " *
        "or use a filename containing YYYY."))
end

"""
    FileSurfaceFluxField{FT}

2-D flux density `raw` `(Nx_src, Ny_src)` loaded from a NetCDF emission file,
in kg species m⁻² s⁻¹ after `_load_file_surface_flux_field` has normalised the
native units, with the source cell areas `cell_area` [m²]: the file's
`cell_area`/`area` variable, else the areas of the source mesh
(`_lonlat_cell_areas_m2`, latitude-circle cells on the destination sphere).
`native_total_mass_rate = Σ raw × cell_area` [kg s⁻¹] is the file's own total,
which the conservative regrid of the per-cell rate `raw × cell_area` preserves
and the bilinear path renormalises to.
"""
struct FileSurfaceFluxField{FT}
    raw                    :: Array{FT, 2}
    lon                    :: Vector{Float64}
    lat                    :: Vector{Float64}
    cell_area              :: Matrix{Float64}
    native_total_mass_rate :: Float64
end

"""
    TimeVaryingFileSurfaceFluxField{FT}

A stack of 2-D flux-density slices `(Nx_src, Ny_src, ntime)` loaded from
a NetCDF emission file, keeping every time slice (no monthly averaging).
Units are kg/m²/s after `_load_timevarying_surface_flux_field` has
normalised the native units, identically per slice. `times_sec` holds
the slice times in seconds since the run `reference_time`, ascending.
"""
struct TimeVaryingFileSurfaceFluxField{FT}
    raw_series :: Array{FT, 3}
    lon        :: Vector{Float64}
    lat        :: Vector{Float64}
    times_sec  :: Vector{Float64}
end

# Surface-flux sources; `file` is a generic NetCDF source.
const _SURFACE_FLUX_KINDS = (:none, :file, :cs_native, :lmdz_co2, :gridfed_fossil_co2,
                             :edgar_sf6, :zhang_rn222)

function _surface_flux_kind(cfg)
    kind = Symbol(lowercase(String(get(cfg, "kind", "none"))))
    if kind === :lmdz_co2 && haskey(cfg, "month") &&
       !_config_bool(cfg, "time_varying", false, "surface_flux.time_varying")
        throw(ArgumentError(
            "surface_flux.month is not supported for a static lmdz_co2 source: " *
            "it averages all time slices in the file. Remove month or set time_varying = true."))
    end
    if !(kind in _SURFACE_FLUX_KINDS)
        suggestion = key_suggestion(String(kind), string.(_SURFACE_FLUX_KINDS))
        throw(ArgumentError(
            "surface_flux.kind = \"$(kind)\" is not a known source " *
            "($(join(_SURFACE_FLUX_KINDS, ", "))" *
            (suggestion === nothing ? "" : "; did you mean `$(suggestion)`?") *
            "); use kind = \"file\" for a generic NetCDF file"))
    end
    return kind
end

function _surface_flux_storage_scale(tracer_name::Symbol, cfg)
    # Surface inventories are physical kg species/s. The prognostic tracer
    # storage is dry VMR * dry-air mass, so source rates must be converted to
    # dry-air-equivalent storage before being applied by the surface kernels.
    species = surface_flux_species(tracer_name, cfg)
    return DRY_AIR_MOLAR_MASS / getproperty(SPECIES_MOLAR_MASS, species)
end

# Derive per-cell area `(Nx, Ny)` on a lat/lon grid from the cell faces implied
# by its file coordinate vectors (`GriddedIO.SnappedFaces`).
# Used for per-cell units (EDGAR Tonnes) when the source file does not carry
# a `cell_area` or `area` variable.
function _lonlat_cell_areas_m2(lon::AbstractVector, lat::AbstractVector; radius::Real = EARTH_RADIUS)
    Nx, Ny = length(lon), length(lat)
    mesh = source_latlon_mesh(lon, lat; radius)
    areas = reshape(cell_areas(mesh), Nx, Ny)
    return lat[2] > lat[1] ? areas : areas[:, end:-1:1]
end

function _resolve_surface_flux_file(cfg, kind::Symbol)
    default_file, default_variable = if kind === :gridfed_fossil_co2
        ("\$ATMOSTRANSPORT_DATA_ROOT/catrine/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc", "TOTAL")
    elseif kind === :edgar_sf6
        ("\$ATMOSTRANSPORT_DATA_ROOT/catrine/Emissions/edgar_v8/v8.0_FT2022_GHG_SF6_2022_TOTALS_emi.nc", "emissions")
    elseif kind === :zhang_rn222
        ("\$ATMOSTRANSPORT_DATA_ROOT/catrine/Emissions/ZHANG_Rn222/Rn222_Emis_Zhang_Liu_et_al_05x05_mass.nc", "rnemis")
    elseif kind === :lmdz_co2
        # Default points at the Dec 2021 CAMS monthly file. Set
        # `surface_flux.file` for another month, or `files` / `file_pattern`
        # for a time-varying multi-month series.
        ("\$ATMOSTRANSPORT_DATA_ROOT/catrine/Emissions/LMDZ_fluxes/z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc",
         "flux_apos")
    else
        ("", "")
    end
    file = expand_data_path(String(get(cfg, "file", default_file)))
    variable = String(get(cfg, "variable", default_variable))
    isempty(file) && throw(ArgumentError("surface_flux.kind=$(kind) requires surface_flux.file"))
    isempty(variable) && throw(ArgumentError("surface_flux.kind=$(kind) requires surface_flux.variable"))

    default_time_index = kind === :gridfed_fossil_co2 ? Int(get(cfg, "month", 0)) :
                          kind === :zhang_rn222       ? Int(get(cfg, "month", 1)) :
                          1
    time_index = Int(get(cfg, "time_index", default_time_index))
    if kind === :gridfed_fossil_co2 && time_index < 1
        if _config_bool(cfg, "time_varying", false, "surface-flux time_varying")
            time_index = 1  # unused by the time-varying twelve-month loader
        else
            throw(ArgumentError("surface_flux.kind=gridfed_fossil_co2 requires surface_flux.time_index or surface_flux.month"))
        end
    end
    time_index < 1 && throw(ArgumentError("surface_flux.time_index must be >= 1"))
    return file, variable, time_index
end

# `time_index` is deliberately positional. In contrast, an explicitly named
# `month` is a semantic selector whenever the file has a time coordinate.
# This keeps partial-year and out-of-order inventories from silently reading a
# different month while preserving the coordinate-less twelve-slice GridFED
# layout.
function _static_surface_flux_time_index(cfg, ds, raw_var,
                                         fallback::Integer)
    haskey(cfg, "time_index") && return Int(cfg["time_index"])
    haskey(cfg, "month") || return Int(fallback)
    month = Int(cfg["month"])
    1 <= month <= 12 || throw(ArgumentError("surface_flux.month must be in 1:12"))
    ndims(raw_var) == 3 || return Int(fallback)
    time_var = find_coordinate(ds, TimeAxis(); length = size(raw_var, 3), required = false)
    time_var === nothing && return Int(fallback)
    keys = decode_month_keys(ds, time_var)
    matches = findall(key -> key.month == month, keys)
    length(matches) == 1 || throw(ArgumentError(
        "surface_flux.month = $month matches $(length(matches)) slices on time coordinate " *
        "'$time_var'; expected exactly one"))
    return only(matches)
end

"""Resolve one or more regular-grid files for a time-varying surface source."""
function _resolve_timevarying_surface_flux_files(
        cfg, kind::Symbol, reference_time::Union{DateTime, Nothing})
    _file, variable, _time_index = _resolve_surface_flux_file(cfg, kind)

    files::Vector{String} = if haskey(cfg, "files")
        values = cfg["files"]
        values isa AbstractVector || throw(ArgumentError(
            "time-varying surface_flux.files must be an array of paths"))
        length(values) > 1 && haskey(cfg, "year") && throw(ArgumentError(
            "surface_flux.year applies one calendar year to every file; with several " *
            "`files`, omit it so each file's year comes from its name or time axis"))
        [expand_data_path(String(v)) for v in values]
    elseif haskey(cfg, "file_pattern")
        pattern = String(cfg["file_pattern"])
        occursin("{YYYYMM}", pattern) || throw(ArgumentError(
            "time-varying surface_flux.file_pattern must contain {YYYYMM}"))
        source_year = haskey(cfg, "year") ? Int(cfg["year"]) :
                      reference_time === nothing ? nothing : Dates.year(reference_time)
        source_year === nothing && throw(ArgumentError(
            "time-varying surface_flux.file_pattern requires surface_flux.year " *
            "when no run reference time is available"))
        [expand_data_path(replace(pattern, "{YYYYMM}" =>
                                  @sprintf("%04d%02d", source_year, month)))
         for month in 1:12]
    else
        [_file]
    end

    isempty(files) && throw(ArgumentError(
        "time-varying surface flux requires at least one input file"))
    missing_files = filter(file -> !isfile(file), files)
    isempty(missing_files) || throw(ArgumentError(
        "time-varying surface-flux file(s) not found: " * join(missing_files, ", ")))
    return files, variable
end

# Time-varying fluxes place their slices on the run clock, which needs the run's
# first day (`[input].start_date`).
_require_reference_time(reference_time) = reference_time === nothing && throw(ArgumentError(
    "time-varying surface fluxes need the run start: set [input].start_date"))

# ---------------------------------------------------------------------------
# Flux units → kg species m⁻² s⁻¹
#
# The units string of the flux variable is parsed by `GriddedIO.parse_flux_units`
# (one grammar for every input: kg, g, tonnes or mol; of the species, of carbon
# or unnamed; per m² or per cell; per second, day, calendar month or Julian
# year). A slice is converted to a density of the tracer's species,
#
#     F [kg species m⁻² s⁻¹] = v × k / T            (per m²)
#     F [kg species m⁻² s⁻¹] = v × k / (A × T)      (per cell, A the cell area)
#
# with k = `kg_per_unit(units, species)` (1 for kg of the species or unnamed kg;
# M(CO2)/M(C) for kg of carbon; 1e-3 for grams; M(species) per mol) and
# T = `period_seconds` (1 s, 86400 s, the length of the calendar month the
# slice covers, or DAYS_PER_JULIAN_YEAR × 86400 s). k and T are applied as two
# operations, a multiplication then a division, so that per-second units
# (T = 1) and per-month units (k = 1) give the same Float bits as the former
# per-unit branches (`raw .*= k`, `raw ./= T`).
# ---------------------------------------------------------------------------

# Run `f`, naming the variable and file in any ArgumentError it throws.
function _in_flux_file_context(f, variable, file)
    try
        return f()
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("surface-flux variable '$(variable)' in $(file): $(err.msg)"))
    end
end

# A molar flux that names no substance (`mmol m-2 s-1`) is of the tracer's own
# species; a substance named in the string (`mol C m-2 s-1`) wins.
function _flux_units(raw_var, variable, file, species::Symbol)
    units = String(string(get(raw_var.attrib, "units", "")))
    return _in_flux_file_context(variable, file) do
        parse_flux_units(units; molar_default = species)
    end
end

# Per-slice denominators for per-month units. Coordinate-backed files derive
# them from the time coordinate's CF bounds and calendar. The ordinal rule is
# deliberately limited to the legacy GridFED layout: twelve slices on a time
# dimension for which no coordinate variable exists.
_slice_periods(::AbstractTimeBasis, cfg, file, ds, raw_var, kind) = nothing
function _slice_periods(::PerMonth, cfg, file, ds, raw_var, kind::Symbol)
    ntime = ndims(raw_var) == 3 ? size(raw_var, 3) : 1
    time_var = find_coordinate(ds, TimeAxis(); length = ntime, required = false)
    if time_var !== nothing
        periods = decode_month_seconds(ds, time_var)
        length(periods) == ntime || throw(DimensionMismatch(
            "time coordinate '$time_var' has $(length(periods)) values; " *
            "surface-flux variable has $ntime slices"))
        return periods
    end
    kind === :gridfed_fossil_co2 || throw(ArgumentError(
        "per-month surface-flux units need a CF time coordinate; only the " *
        "coordinate-less GridFED format may use ordinal months"))
    ntime == 12 || throw(ArgumentError(
        "coordinate-less GridFED per-month source must contain 12 monthly slices, " *
        "got $ntime in $file"))
    year = _surface_flux_year(cfg, file, ds)
    return Float64[Dates.daysinmonth(year, month) * 86400 for month in 1:12]
end

_unit_seconds(units, ::Nothing) = period_seconds(units)
_unit_seconds(units, iv::TimeInterval) = period_seconds(units, iv)
_unit_seconds(units, seconds::Real) = Float64(seconds)

# One slice `raw` (in place) from `units` to kg `species` m⁻² s⁻¹; `iv` is the
# slice's time interval (per-month units) and `area` the source cell areas
# [m²] (per-cell units).
function _to_flux_density!(raw::AbstractArray{FT}, units::FluxUnits{<:Any, PerSquareMetre},
                           species::Symbol, iv, area) where FT
    k = FT(kg_per_unit(units, species))
    T = FT(_unit_seconds(units, iv))
    raw .= raw .* k ./ T
    return raw
end

function _to_flux_density!(raw::AbstractMatrix{FT}, units::FluxUnits{<:Any, PerCell},
                           species::Symbol, iv, area::AbstractMatrix) where FT
    # Per-cell amounts (EDGAR: tonnes per cell and year), divided by the cell
    # area in Float64; cells without area get no flux.
    k = kg_per_unit(units, species)
    T = _unit_seconds(units, iv)
    @inbounds for j in axes(raw, 2), i in axes(raw, 1)
        a = area[i, j]
        raw[i, j] = a > 0 ? FT(k * Float64(raw[i, j]) / (a * T)) : zero(FT)
    end
    return raw
end

_to_flux_density!(raw, units::FluxUnits{<:Any, PerCell}, species, iv, ::Nothing) =
    throw(ArgumentError(
        "per-cell flux units $(repr(units.text)) need the source cell areas; time-varying " *
        "sources read densities (per m²)"))

function _load_file_surface_flux_field(cfg, ::Type{FT}; species::Symbol,
                                       radius::Real = EARTH_RADIUS) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    file, variable, time_index = _resolve_surface_flux_file(cfg, kind)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))

    ds = NCDataset(file)
    try
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))
        raw_var = ds[variable]
        ndims(raw_var) in (2, 3) || throw(ArgumentError(
            "surface-flux variable '$variable' must be 2D or 3D, got ndims=$(ndims(raw_var))"))
        lon_src, lat_src = _latlon_centres(ds, raw_var)
        time_index = _static_surface_flux_time_index(cfg, ds, raw_var, time_index)
        ndims(raw_var) == 2 || 1 <= time_index <= size(raw_var, 3) || throw(ArgumentError(
            "surface_flux time index $time_index is outside 1:$(size(raw_var, 3))"))

        # Fill values become 0 (GriddedIO.MaskedAsZero).
        raw = if ndims(raw_var) == 3
            if kind === :lmdz_co2
                # CAMS LMDZ files store 3-hourly fluxes (`time = 248`
                # for a 31-day month). The static path carries a single
                # 2D field, so average over the time axis to get the
                # monthly-mean rate. Sub-monthly variability uses the
                # `time_varying = true` path (cubed-sphere runs only).
                ntime = size(raw_var, 3)
                acc = zeros(Float64, size(raw_var, 1), size(raw_var, 2))
                for t in 1:ntime
                    acc .+= read_raw(ds, variable, MaskedAsZero(), :, :, t)
                end
                acc ./= ntime
                FT.(acc)
            else
                read_raw(FT, ds, variable, MaskedAsZero(), :, :, time_index)
            end
        else
            read_raw(FT, ds, variable, MaskedAsZero())
        end

        area_name = haskey(ds, "cell_area") ? "cell_area" : haskey(ds, "area") ? "area" : nothing
        file_area = area_name === nothing ? nothing : read_raw(ds, area_name, MaskedAsZero())

        # Latitude ascending, longitude from the first value ≥ 0 (GriddedIO).
        orientation = latlon_orientation(lon_src, lat_src)
        raw = orient(raw, orientation)
        file_area === nothing || (file_area = orient(file_area, orientation))
        lon_src, lat_src = orient_coordinates(lon_src, lat_src, orientation)
        # The source cell areas: the file's (in m² of the same cells,
        # `GriddedIO.check_latlon_cell_area`), else the band areas of the source
        # mesh on the destination mesh's sphere.
        mesh_area = _lonlat_cell_areas_m2(lon_src, lat_src; radius)
        size(mesh_area) == size(raw) || throw(DimensionMismatch(
            "the coordinates of $(file) define a $(size(mesh_area)) grid; the flux has $(size(raw))"))
        cell_area_src = file_area === nothing ? mesh_area :
            check_latlon_cell_area(file_area, mesh_area,
                                   String(string(get(ds[area_name].attrib, "units", "")));
                                   file = "$(file) ('$(area_name)')", flux = raw)

        # Units → kg species m⁻² s⁻¹. Coordinate-backed per-month
        # values use that slice's CF bounds/calendar; only coordinate-less
        # GridFED uses `time_index` as the ordinal month. Per-cell values
        # (EDGAR) are divided by the source cell areas, so the per-cell
        # totals are the source's total whatever the destination radius.
        units = _flux_units(raw_var, variable, file, species)
        _in_flux_file_context(variable, file) do
            periods = _slice_periods(units.time, cfg, file, ds, raw_var, kind)
            iv = periods === nothing ? nothing : periods[time_index]
            area = units.area isa PerCell ? cell_area_src : nothing
            _to_flux_density!(raw, units, species, iv, area)
        end

        raw .*= FT(get(cfg, "scale", 1.0))
        native_total_mass_rate = sum(Float64.(raw) .* cell_area_src)
        return FileSurfaceFluxField{FT}(raw, lon_src, lat_src, cell_area_src,
                                        native_total_mass_rate)
    finally
        close(ds)
    end
end

# One file of a time-varying series: all slices, or the file's slice indices
# `slices` (no monthly averaging), the static loader's reorientation and unit
# conversion (to kg `species` m⁻² s⁻¹), and slice times in seconds since
# `reference_time`.
function _load_single_timevarying_surface_flux_field(
        cfg, ::Type{FT}, reference_time::Union{DateTime, Nothing},
        file::AbstractString, variable::AbstractString;
        species::Symbol,
        slices::Union{Nothing, AbstractVector{<:Integer}} = nothing) where FT
    kind = _surface_flux_kind(cfg)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))

    ds = NCDataset(file)
    try
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))
        raw_var = ds[variable]
        ndims(raw_var) == 3 || throw(ArgumentError(
            "time-varying surface-flux variable '$variable' must be 3D (lon,lat,time), got ndims=$(ndims(raw_var))"))
        lon_src, lat_src = _latlon_centres(ds, raw_var)
        Nx, Ny, ntime = size(raw_var)
        keep = slices === nothing ? (1:ntime) : slices
        raw = Array{FT, 3}(undef, Nx, Ny, length(keep))
        for (out, t) in enumerate(keep)      # fill values become 0
            raw[:, :, out] .= read_raw(FT, ds, variable, MaskedAsZero(), :, :, t)
        end

        # Latitude ascending, longitude from the first value ≥ 0 (GriddedIO).
        orientation = latlon_orientation(lon_src, lat_src)
        raw = orient(raw, orientation)
        lon_src, lat_src = orient_coordinates(lon_src, lat_src, orientation)

        # Per-slice unit conversion (as the static loader). Coordinate-backed
        # per-month values use their CF time bounds/calendar; only legacy
        # coordinate-less GridFED uses slice k as month k.
        units = _flux_units(raw_var, variable, file, species)
        periods = _slice_periods(units.time, cfg, file, ds, raw_var, kind)
        _in_flux_file_context(variable, file) do
            for (out, t) in enumerate(keep)
                iv = periods === nothing ? nothing : periods[t]
                _to_flux_density!(view(raw, :, :, out), units, species, iv, nothing)
            end
        end
        raw .*= FT(get(cfg, "scale", 1.0))

        time_var = find_coordinate(ds, TimeAxis(); length = ntime, required = false)
        times_sec = _timevarying_slice_seconds(time_var, ds, cfg, kind, file, reference_time)[keep]

        # --- emission temporal-stamp convention (CAMS / LMDZ natural CO2) ---
        # The CAMS file (`flux_apos`, 3-hourly, "hours since 2021-12-01") uses
        # INTERVAL-START stamps: HEMCO/GEOS-Chem holds slice k (stamp k·Δ)
        # PIECEWISE-CONSTANT over [k·Δ, (k+1)·Δ) — verified against GC's
        # EmisCO2_Total, which at output stamp T equals the CAMS slice at T−Δ
        # (e.g. GC's 03:00z emission = the 00:00 CAMS slice). The faithful
        # match is therefore `temporal_scheme = "stepwise"` (StepwiseFlux holds
        # the largest knot ≤ t, i.e. v_k over [k·Δ, (k+1)·Δ)), with the knots
        # left UNSHIFTED. Do NOT add a +Δ time shift here: a shift only realigns
        # the point values at the knots, while the integrated/transported
        # emission still depends on the scheme (a "conservative" linear-blend
        # scheme would average adjacent slices regardless of any shift). Keeping
        # the raw stamps + StepwiseFlux reproduces GC's step exactly.

        return TimeVaryingFileSurfaceFluxField{FT}(raw, lon_src, lat_src, times_sec)
    finally
        close(ds)
    end
end

# Slice times (seconds since `reference_time`) from the file's time axis,
# decoded by `GriddedIO.decode_time` (the axis is classified by its units
# before any CFTime call), or, for GridFED files without one, the first
# instant of each month of the source year.
function _timevarying_slice_seconds(time_var::AbstractString, ds, cfg, kind::Symbol, file,
                                    reference_time)
    _require_reference_time(reference_time)
    if _surface_flux_temporal_scheme(cfg) isa StepwiseFlux
        bounds = decode_time_bounds(ds, time_var)
        bounds === nothing || return seconds_since(vec(bounds[1, :]), reference_time)
    end
    return seconds_since(decode_time(ds, time_var), reference_time)
end


function _surface_flux_temporal_scheme(cfg)
    default = _surface_flux_kind(cfg) in (:lmdz_co2, :gridfed_fossil_co2) ?
              "stepwise" : "linear"
    return flux_temporal_scheme(String(get(cfg, "temporal_scheme", default)))
end
function _timevarying_slice_seconds(::Nothing, ds, cfg, kind::Symbol, file, reference_time)
    kind === :gridfed_fossil_co2 || throw(ArgumentError(
        "no time coordinate (variable time or t) in $file; only gridfed_fossil_co2 files, " *
        "twelve monthly slices, may omit it"))
    _require_reference_time(reference_time)
    source_year = _surface_flux_year(cfg, file, ds)
    return seconds_since(DateTime[DateTime(source_year, month, 1) for month in 1:12], reference_time)
end

# Slice times of one file, read from its time axis without its data.
function _timevarying_file_seconds(cfg, reference_time, file::AbstractString,
                                   variable::AbstractString)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))
    return NCDataset(file) do ds
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))
        ntime = size(ds[variable], ndims(ds[variable]))
        time_var = find_coordinate(ds, TimeAxis(); length = ntime, required = false)
        _timevarying_slice_seconds(time_var, ds, cfg, _surface_flux_kind(cfg), file, reference_time)
    end
end

# ---------------------------------------------------------------------------
# Time coverage of a time-varying source
#
# Outside its slices a time-varying source holds the first or last slice
# (`StepwiseFlux`, `_time_interp_bracket`). For a file of the wrong period
# (a December file in a January run) that silently emits one 3-hourly value
# for weeks, so the run period must lie inside the intervals the slices stand
# for (`_timevarying_source_intervals`, over all files of the source together)
# unless the source opts in to the hold with `hold_outside_coverage = true`
# (e.g. a climatology).
# ---------------------------------------------------------------------------

"""
    RUN_SPAN_CLOCK_SLACK_SECONDS

The runner's run span (`DrivenRunner._run_time_span`) ends this long after the
end of its last met window, slack for the rounding of its accumulated Float64
clock. The time-coverage check of a time-varying surface flux does not ask the
files to cover it.
"""
const RUN_SPAN_CLOCK_SLACK_SECONDS = 1.0

# Opt-in: a run may extend past the time coverage of the source's files, and
# the source holds its first or last slice there.
_hold_outside_coverage(cfg) =
    _config_bool(cfg, "hold_outside_coverage", false, "surface-flux hold_outside_coverage")

"""
    _timevarying_source_intervals(cfg, files, variable) -> Vector{TimeInterval}

The interval each slice of a time-varying source stands for, over all its
`files` together: the intervals a file's time axis defines
(`GriddedIO.known_slice_intervals`: CF bounds, `months since`), the months of
the source year for GridFED files without a time axis, and, for the stamps
without bounds, `GriddedIO.stamp_intervals` of the stamps of **all** files
merged and sorted. So daily files with one 00Z stamp each are a daily series,
a gap between or inside files is a gap, and only the very last stamp of the
series is extended (by the series' step). One stamp in all, which may be a
day, a month or a year, is an `ArgumentError`.
"""
function _timevarying_source_intervals(cfg, files, variable::AbstractString)
    intervals = TimeInterval[]
    stamps = DateTime[]
    for file in files
        NCDataset(file) do ds
            ntime = size(ds[variable], ndims(ds[variable]))
            time_var = find_coordinate(ds, TimeAxis(); length = ntime, required = false)
            if time_var === nothing  # GridFED (`_timevarying_file_seconds` refuses others)
                year = _surface_flux_year(cfg, file, ds)
                append!(intervals, [interval(MonthKey(year, month)) for month in 1:12])
            else
                known = known_slice_intervals(ds, time_var)
                known === nothing ? append!(stamps, decode_time(ds, time_var)) :
                                    append!(intervals, known)
            end
        end
    end
    isempty(stamps) || append!(intervals, stamp_intervals(
        sort(stamps); source = "the $(length(stamps)) time stamp(s) without CF bounds of its files"))
    return intervals
end

# Run `f`, which reads the slice intervals of a source's files; an
# ArgumentError (e.g. one stamp without bounds) names the source and the remedies.
function _in_coverage_context(f, source::AbstractString)
    try
        return f()
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError(
            "$(source): the time coverage of its files is unknown ($(err.msg)). Give the " *
            "time axis CF bounds, or set hold_outside_coverage = true to skip the coverage check."))
    end
end

_interval_seconds(intervals, reference_time::DateTime) =
    (Float64[seconds_since(iv.start, reference_time) for iv in intervals],
     Float64[seconds_since(iv.stop, reference_time) for iv in intervals])

# A short name of a source's files for messages.
_files_label(files) = length(files) <= 2 ? join(basename.(files), ", ") :
    "$(basename(first(files))) … $(basename(last(files))) ($(length(files)) files)"

"""
    _check_flux_time_coverage(starts, stops, run_span, reference_time; source)

Throw an `ArgumentError` unless the slices with intervals
`[starts[k], stops[k])` (seconds since `reference_time`) cover the run
`run_span = (t_start, t_end)` without a gap, up to the clock slack
`RUN_SPAN_CLOCK_SLACK_SECONDS` at its end. The message names the first time
range no slice covers, the files' coverage, and the two remedies: files that
cover the run, or `hold_outside_coverage = true`. `source` names the source.
"""
function _check_flux_time_coverage(starts::AbstractVector{<:Real}, stops::AbstractVector{<:Real},
                                   run_span, reference_time::DateTime; source::AbstractString)
    isempty(starts) && throw(ArgumentError("$(source): the time-varying flux has no time slices"))
    t0 = Float64(run_span[1])
    t1 = max(t0, Float64(run_span[2]) - RUN_SPAN_CLOCK_SLACK_SECONDS)
    covered = t0                      # [t0, covered) is covered
    for k in sortperm(starts)
        (covered >= t1 || starts[k] > covered) && break
        covered = max(covered, Float64(stops[k]))
    end
    covered >= t1 && return nothing
    later = filter(>(covered), starts)
    gap_end = isempty(later) ? t1 : min(t1, minimum(later))
    at(s) = reference_time + Millisecond(round(Int64, 1000 * s))
    throw(ArgumentError(
        "$(source): no time slice covers $(at(covered)) to $(at(gap_end)), inside the run " *
        "($(at(t0)) to $(at(t1))); the slices span $(at(minimum(starts))) to " *
        "$(at(maximum(stops))). " *
        "Outside its slices a time-varying source would hold its first or last slice. List " *
        "files that cover the run (`files`, or `file_pattern` with `year`), or set " *
        "hold_outside_coverage = true to hold the end slices on purpose (e.g. a climatology)."))
end

"""
    _needed_slice_window(times, run_span) -> (t_lo, t_hi)

Slice times that bound the run `run_span = (t_start, t_end)` (seconds since
the run start): the latest slice at or before `t_start` and the earliest at or
after `t_end` (the first or last slice when none exists). Slices outside
`[t_lo, t_hi]` cannot affect the run under any temporal scheme, which only
uses the slices bracketing a time.
"""
function _needed_slice_window(times::AbstractVector{<:Real}, run_span::Tuple{<:Real, <:Real})
    isempty(times) && throw(ArgumentError("time-varying surface flux has no time slices"))
    t0, t1 = run_span
    t0 <= t1 || throw(ArgumentError("run span must be ordered; got $(run_span)"))
    sorted = sort(times)
    i_lo = searchsortedlast(sorted, t0)
    i_hi = searchsortedfirst(sorted, t1)
    return sorted[max(i_lo, 1)], sorted[min(i_hi, length(sorted))]
end

"""
    _load_timevarying_surface_flux_field(cfg, FT, reference_time; species, run_span = nothing)

Load a regular lon/lat `(lon, lat, time)` surface-flux series from one file
(`file`), an explicit list (`files = [...]`, chronological), or twelve monthly
files (`file_pattern` with `{YYYYMM}`, for `year` or the run-start year).
Slices from all files are concatenated and sorted by time; timestamps must be
unique and every file must share one lon/lat grid. Units are converted per
slice to kg `species` m⁻² s⁻¹ (GridFED monthly totals use each month's length);
time axes are decoded by `GriddedIO.decode_time`.
A `reference_time` (the run start) is required. With `run_span = (t_start, t_end)`
(seconds since the run start) and a `reference_time`, every file's time axis
is read but flux data only for the slices in
`_needed_slice_window(times, run_span)`, which are returned; otherwise every
slice is read. With a `run_span`, the files must cover it
(`_check_flux_time_coverage` of `_timevarying_source_intervals`, the slice
intervals of all files together) unless `hold_outside_coverage = true`;
`source` names the source in that error.
"""
function _load_timevarying_surface_flux_field(cfg, ::Type{FT},
                                              reference_time::Union{DateTime, Nothing};
                                              species::Symbol, run_span = nothing,
                                              source::AbstractString = "time-varying surface flux") where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    _require_reference_time(reference_time)
    all_files, variable = _resolve_timevarying_surface_flux_files(cfg, kind, reference_time)
    files, slices = if run_span === nothing
        all_files, fill(nothing, length(all_files))
    else
        file_times = [_timevarying_file_seconds(cfg, reference_time, f, variable) for f in all_files]
        if !_hold_outside_coverage(cfg)
            label = "$(source) (kind = $(kind); $(_files_label(all_files)))"
            intervals = _in_coverage_context(label) do
                _timevarying_source_intervals(cfg, all_files, variable)
            end
            _check_flux_time_coverage(_interval_seconds(intervals, reference_time)...,
                                      run_span, reference_time; source = label)
        end
        t_lo, t_hi = _needed_slice_window(reduce(vcat, file_times), run_span)
        keep = [findall(t -> t_lo <= t <= t_hi, ts) for ts in file_times]
        used = findall(!isempty, keep)
        all_files[used], keep[used]
    end
    fields = [_load_single_timevarying_surface_flux_field(
                  cfg, FT, reference_time, file, variable; species, slices = s)
              for (file, s) in zip(files, slices)]

    lon = first(fields).lon
    lat = first(fields).lat
    for (file, field) in zip(files, fields)
        field.lon == lon || throw(ArgumentError(
            "time-varying surface-flux longitude grid differs in $file"))
        field.lat == lat || throw(ArgumentError(
            "time-varying surface-flux latitude grid differs in $file"))
    end

    raw = length(fields) == 1 ? first(fields).raw_series :
          cat((field.raw_series for field in fields)...; dims = 3)
    times_sec = reduce(vcat, (field.times_sec for field in fields))
    perm = sortperm(times_sec)
    times_sec = times_sec[perm]
    raw = raw[:, :, perm]
    issorted(times_sec; lt = <=) || throw(ArgumentError(
        "time-varying surface-flux timestamps must be unique and strictly increasing"))

    @info "Loaded time-varying surface flux" kind files=length(files) listed_files=length(all_files) slices=length(times_sec) first_time_seconds=first(times_sec) last_time_seconds=last(times_sec)
    return TimeVaryingFileSurfaceFluxField{FT}(raw, lon, lat, times_sec)
end

# Only the cubed-sphere builder takes a run span (time-varying sources are CS only).
_build_surface_flux_source_spanned(grid::AtmosGrid{<:CubedSphereMesh}, name, cfg, ::Type{FT},
                                   reference_time, run_span) where FT =
    build_surface_flux_source(grid, name, cfg, FT; reference_time, run_span)
_build_surface_flux_source_spanned(grid, name, cfg, ::Type{FT}, reference_time, _run_span) where FT =
    build_surface_flux_source(grid, name, cfg, FT; reference_time)

include("surface_flux_regridding.jl")
include("surface_flux_native.jl")

# ---------------------------------------------------------------------------
# build_surface_flux_source — LL / RG / CS
# ---------------------------------------------------------------------------

# Opt-in flag for the time-varying surface-flux path (default false → one
# static field: the configured time slice, or the time mean for `lmdz_co2`).
@inline _surface_flux_time_varying(cfg) =
    _config_bool(cfg, "time_varying", false, "surface-flux time_varying")

# LL series are regridded; cs_native already matches the runtime mesh.
@inline _surface_flux_supports_time_varying(kind::Symbol) =
    kind in (:lmdz_co2, :gridfed_fossil_co2, :cs_native)

function build_surface_flux_source(grid::AtmosGrid{<:LatLonMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT};
                                   reference_time::Union{DateTime, Nothing} = nothing) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    kind === :cs_native && throw(ArgumentError(
        "surface_flux kind = \"cs_native\" is only available on cubed-sphere runs"))
    _surface_flux_time_varying(cfg) && throw(ArgumentError(
        "time-varying surface flux is supported on cubed-sphere grids only"))

    source = _load_file_surface_flux_field(cfg, FT; species = surface_flux_species(tracer_name, cfg),
                                           radius = grid.horizontal.radius)
    mesh = grid.horizontal
    rate = reshape(_surface_flux_rate(_regridding_method(cfg), source, mesh, FT;
                                      label = "surface flux of tracer $(tracer_name)"),
                   nx(mesh), ny(mesh))
    rate .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    return SurfaceFluxSource(tracer_name, rate)
end

function build_surface_flux_source(grid::AtmosGrid{<:ReducedGaussianMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT};
                                   reference_time::Union{DateTime, Nothing} = nothing) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    kind === :cs_native && throw(ArgumentError(
        "surface_flux kind = \"cs_native\" is only available on cubed-sphere runs"))
    _surface_flux_time_varying(cfg) && throw(ArgumentError(
        "time-varying surface flux is supported on cubed-sphere grids only"))

    source = _load_file_surface_flux_field(cfg, FT; species = surface_flux_species(tracer_name, cfg),
                                           radius = grid.horizontal.radius)
    rate = _surface_flux_rate(_regridding_method(cfg), source, grid.horizontal, FT;
                              label = "surface flux of tracer $(tracer_name)")
    rate .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    return SurfaceFluxSource(tracer_name, rate)
end

"""
    build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh},
                              tracer_name, cfg, ::Type{FT})

CS surface-flux builder. Conservatively LL→CS regrids the per-cell rate
(flux density × source cell area, kg species/s; `ConservativeFluxRegridding`)
onto the 6 CS panels as an extensive field, so the total is the file's own,
then converts that physical rate to the model storage basis. Returns a
`SurfaceFluxSource` whose `cell_mass_rate` is an `NTuple{6, Matrix{FT}}`
of interior-only `(Nc, Nc)` panels.

`cfg` must set `kind` (non-`none`); any of the file-based surface-flux
kinds `_load_file_surface_flux_field` understands work
(`gridfed_fossil_co2` or user-supplied `file` + `variable`).
Conservative regridding is enforced — CS bilinear is not supported.

For `kind = "cs_native"` and `time_varying = true`, an aligned NetCDF
`(time,nf,Ydim,Xdim)` series is loaded directly, multiplied by native cell
areas, and converted to storage units. No regridding is applied.

If `cfg["time_varying"] = true` and the kind supports a 3-D
(lon,lat,time) series (`:lmdz_co2`, `:gridfed_fossil_co2`), the builder keeps every
time slice, builds the LL→CS regridder ONCE, applies it per slice, and
returns a [`TimeVaryingSurfaceFluxSource`](@ref) whose
`cell_mass_rate_series` is an `NTuple{6}` of `(Nc, Nc, ntime)` panels
plus a `times` vector (seconds since `reference_time`). The default
(`time_varying` absent/false) path returns a static `SurfaceFluxSource`.
"""
function build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT};
                                   reference_time::Union{DateTime, Nothing} = nothing,
                                   run_span = nothing) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    kind === :cs_native && !_surface_flux_time_varying(cfg) && throw(ArgumentError(
        "surface_flux kind = \"cs_native\" requires time_varying = true"))

    _regridding_method(cfg) isa ConservativeFluxRegridding ||
        @warn "CS surface-flux: `regridding = \"$(cfg["regridding"])\"` requested; CS supports conservative only — forcing conservative."

    mesh = grid.horizontal
    Nc = mesh.Nc

    if _surface_flux_time_varying(cfg)
        _surface_flux_supports_time_varying(kind) || throw(ArgumentError(
            "time-varying surface flux not supported for kind=$(kind); " *
            "supported: :lmdz_co2, :gridfed_fossil_co2, :cs_native"))
        kind === :cs_native && return _build_native_timevarying_cs_surface_flux_source(
            mesh, tracer_name, cfg, FT, reference_time; run_span)
        return _build_timevarying_cs_surface_flux_source(mesh, tracer_name, cfg, FT, reference_time;
                                                         run_span)
    end

    source = _load_file_surface_flux_field(cfg, FT; species = surface_flux_species(tracer_name, cfg),
                                           radius = grid.horizontal.radius)

    # _conservative_surface_flux_rate already returns kg/s per cell
    # (regridder.dst_areas × regridded flux density), so the panel unpack
    # only needs to reshape the flat `6*Nc^2` vector into 6 × (Nc, Nc).
    rate_flat = _conservative_surface_flux_rate(source, mesh, FT)
    rate_flat .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    length(rate_flat) == CS_PANEL_COUNT * Nc * Nc || throw(DimensionMismatch(
        "CS surface-flux conservative regrid returned $(length(rate_flat)) cells; expected $(CS_PANEL_COUNT * Nc * Nc)"))

    panels = ntuple(_ -> Matrix{FT}(undef, Nc, Nc), CS_PANEL_COUNT)
    unpack_flat_to_panels_2d!(panels, rate_flat, Nc)

    return SurfaceFluxSource(tracer_name, panels)
end

# Build the time-varying CS source: one regridder, applied per slice into
# stacked `(Nc, Nc, ntime)` panel series.
function _build_timevarying_cs_surface_flux_source(mesh, tracer_name::Symbol, cfg,
                                                   ::Type{FT},
                                                   reference_time::Union{DateTime, Nothing};
                                                   run_span = nothing) where FT
    Nc = mesh.Nc
    field = _load_timevarying_surface_flux_field(cfg, FT, reference_time;
                                                 species = surface_flux_species(tracer_name, cfg),
                                                 run_span,
                                                 source = "surface flux of tracer $(tracer_name)")
    ntime = length(field.times_sec)
    storage_scale = FT(_surface_flux_storage_scale(tracer_name, cfg))

    regridder = _build_surface_flux_regridder(field.lon, field.lat, mesh)
    panels_series = ntuple(_ -> Array{FT, 3}(undef, Nc, Nc, ntime), CS_PANEL_COUNT)

    slice_panels = ntuple(_ -> Matrix{FT}(undef, Nc, Nc), CS_PANEL_COUNT)
    @inbounds for t in 1:ntime
        # Only report mass conservation on the first slice to avoid log spam.
        rate_flat = _apply_surface_flux_regridder(regridder, @view(field.raw_series[:, :, t]),
                                                  FT; report = (t == 1))
        rate_flat .*= storage_scale
        length(rate_flat) == CS_PANEL_COUNT * Nc * Nc || throw(DimensionMismatch(
            "CS time-varying surface-flux regrid returned $(length(rate_flat)) cells; expected $(CS_PANEL_COUNT * Nc * Nc)"))
        unpack_flat_to_panels_2d!(slice_panels, rate_flat, Nc)
        for p in 1:CS_PANEL_COUNT
            panels_series[p][:, :, t] .= slice_panels[p]
        end
    end

    # Kind-aware default: lmdz_co2 (CAMS) is held PIECEWISE-CONSTANT in 3-hourly
    # blocks by HEMCO/GEOS-Chem (verified: GC's EmisCO2_Total at stamp T equals
    # the CAMS slice at T−Δ), so it MUST default to "stepwise" for GC parity — a
    # linear/interp default smears the diurnal cycle (anomaly corr 0.91 vs 0.998).
    # GridFED monthly totals are likewise held constant through each month.
    # Other kinds keep the generic "linear" default.
    scheme = _surface_flux_temporal_scheme(cfg)
    return TimeVaryingSurfaceFluxSource(tracer_name, panels_series, field.times_sec, scheme)
end

"""
    build_surface_flux_sources(grid, tracer_specs, ::Type{FT}; reference_time=nothing, run_span=nothing)

Build surface-flux source instances for every tracer spec that requests
one. Returns a tuple (possibly empty) suitable for the
`surface_sources = (…,)` kwarg on `DrivenSimulation`.

`reference_time` (the run start `DateTime`) is threaded to each
per-tracer builder so the time-varying CS path can align its slice times
to the simulation clock. `run_span = (t_start, t_end)` (seconds since the run
start) lets the time-varying CS path read only the slices the run can use.
Both are ignored by static sources.
"""
function build_surface_flux_sources(grid, tracer_specs, ::Type{FT};
                                    reference_time::Union{DateTime, Nothing} = nothing,
                                    run_span = nothing) where FT
    sources = Any[]
    for spec in tracer_specs
        source = _build_surface_flux_source_spanned(grid, spec.name, spec.surface_flux_cfg, FT,
                                                    reference_time, run_span)
        source === nothing || push!(sources, source)
    end
    return Tuple(sources)
end
