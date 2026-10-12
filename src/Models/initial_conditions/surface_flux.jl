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

2-D flux density `(Nx_src, Ny_src)` loaded from a NetCDF emission file.
Units are kg/m²/s after `_load_file_surface_flux_field` has normalised
the native units. `native_total_mass_rate` is the pre-regrid global
integral, preserved for the bilinear path's renormalisation (the
conservative path does not need it).
"""
struct FileSurfaceFluxField{FT}
    raw                    :: Array{FT, 2}
    lon                    :: Vector{Float64}
    lat                    :: Vector{Float64}
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

# Derive per-cell area `(Nx, Ny)` on a regular lat/lon grid from the
# coordinate vectors. Uses the spherical-cap formula
# `R² · Δlon · |sin(φ + Δlat/2) - sin(φ - Δlat/2)|`.
# Used for per-cell units (EDGAR Tonnes) when the source file does not carry
# a `cell_area` or `area` variable.
# TODO(P2b): areas of the source mesh (`Grids.cell_area`), with the
# pole-centred faces; that changes the EDGAR band areas in the last bits.
function _lonlat_cell_areas_m2(lon::AbstractVector, lat::AbstractVector; radius::Real = EARTH_RADIUS)
    Nx, Ny = length(lon), length(lat)
    R = Float64(radius)
    # Cell width in radians, assuming uniform spacing. Take it from the full
    # span: coordinates stored in Float32 make the first difference err by
    # ~1e-4 relative, a bias of every cell area.
    dlon = Nx > 1 ? deg2rad(abs(lon[end] - lon[1]) / (Nx - 1)) : deg2rad(360.0 / Nx)
    dlat_half = Ny > 1 ? deg2rad(abs(lat[end] - lat[1]) / (Ny - 1)) / 2 : deg2rad(180.0 / Ny) / 2
    out = Array{Float64, 2}(undef, Nx, Ny)
    @inbounds for j in 1:Ny
        ϕ = deg2rad(lat[j])
        band = R * R * dlon * abs(sin(ϕ + dlat_half) - sin(ϕ - dlat_half))
        for i in 1:Nx
            out[i, j] = band
        end
    end
    return out
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

        cell_area_src = haskey(ds, "cell_area") ? read_raw(ds, "cell_area", MaskedAsZero()) :
                        haskey(ds, "area")      ? read_raw(ds, "area", MaskedAsZero()) :
                        nothing

        # Latitude ascending, longitude from the first value ≥ 0 (GriddedIO).
        orientation = latlon_orientation(lon_src, lat_src)
        raw = orient(raw, orientation)
        cell_area_src === nothing || (cell_area_src = orient(cell_area_src, orientation))
        lon_src, lat_src = orient_coordinates(lon_src, lat_src, orientation)

        # Units → kg species m⁻² s⁻¹. Coordinate-backed per-month
        # values use that slice's CF bounds/calendar; only coordinate-less
        # GridFED uses `time_index` as the ordinal month. Per-cell values
        # (EDGAR) are divided by the file's cell areas or, without them, the
        # band areas of the grid on the destination mesh's sphere, so the
        # per-cell totals survive regridding whatever its radius.
        units = _flux_units(raw_var, variable, file, species)
        _in_flux_file_context(variable, file) do
            periods = _slice_periods(units.time, cfg, file, ds, raw_var, kind)
            iv = periods === nothing ? nothing : periods[time_index]
            area = !(units.area isa PerCell) ? nothing :
                   cell_area_src !== nothing ? cell_area_src :
                   _lonlat_cell_areas_m2(lon_src, lat_src; radius)
            _to_flux_density!(raw, units, species, iv, area)
        end

        raw .*= FT(get(cfg, "scale", 1.0))
        native_total_mass_rate = cell_area_src === nothing ? NaN : sum(Float64.(raw) .* cell_area_src)
        return FileSurfaceFluxField{FT}(raw, lon_src, lat_src, native_total_mass_rate)
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

        # Require ascending; sort consistently if needed.
        if !issorted(times_sec)
            perm = sortperm(times_sec)
            times_sec = times_sec[perm]
            raw = raw[:, :, perm]
        end

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
    return seconds_since(decode_time(ds, time_var), reference_time)
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
slice is read.
"""
function _load_timevarying_surface_flux_field(cfg, ::Type{FT},
                                              reference_time::Union{DateTime, Nothing};
                                              species::Symbol, run_span = nothing) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    _require_reference_time(reference_time)
    all_files, variable = _resolve_timevarying_surface_flux_files(cfg, kind, reference_time)
    files, slices = if run_span === nothing
        all_files, fill(nothing, length(all_files))
    else
        file_times = [_timevarying_file_seconds(cfg, reference_time, f, variable) for f in all_files]
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
    method = _regridding_method(cfg)
    mesh = grid.horizontal

    if method === :conservative
        rate_flat = _conservative_surface_flux_rate(source, mesh, FT)
        rate_flat .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
        rate = reshape(rate_flat, nx(mesh), ny(mesh))
    else
        # Legacy bilinear sampling
        rate = Array{FT}(undef, nx(mesh), ny(mesh))
        for j in axes(rate, 2)
            area = Float64(cell_area(mesh, 1, j))
            lat = mesh.φᶜ[j]
            for i in axes(rate, 1)
                lon = mesh.λᶜ[i]
                flux_density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lon, lat)
                rate[i, j] = FT(flux_density * area)
            end
        end
        _renormalize_surface_flux_rate!(rate, source)
        rate .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    end

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
    method = _regridding_method(cfg)
    mesh = grid.horizontal

    if method === :conservative
        rate = _conservative_surface_flux_rate(source, mesh, FT)
        rate .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    else
        # Legacy bilinear sampling
        rate = Array{FT}(undef, ncells(mesh))
        for j in 1:nrings(mesh)
            lat = mesh.latitudes[j]
            lons = ring_longitudes(mesh, j)
            for i in eachindex(lons)
                c = cell_index(mesh, i, j)
                flux_density = _sample_bilinear_scalar(source.raw, source.lon, source.lat, lons[i], lat)
                rate[c] = FT(flux_density * Float64(cell_area(mesh, c)))
            end
        end
        _renormalize_surface_flux_rate!(rate, source)
        rate .*= FT(_surface_flux_storage_scale(tracer_name, cfg))
    end

    return SurfaceFluxSource(tracer_name, rate)
end

"""
    build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh},
                              tracer_name, cfg, ::Type{FT})

CS surface-flux builder. Conservatively LL→CS
regrids the 2-D flux density (kg/m²/s) onto the 6 CS panel cell
centres, multiplies by each panel's cell area to yield per-cell kg
species/s, then converts that physical rate to the model storage basis.
Returns a
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

    method = _regridding_method(cfg, "conservative")
    haskey(cfg, "regridding") && method !== :conservative &&
        @warn "CS surface-flux: `regridding = \"$(method)\"` requested; CS supports conservative only — forcing conservative."

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
                                                 run_span)
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
    default_scheme = _surface_flux_kind(cfg) in (:lmdz_co2, :gridfed_fossil_co2) ?
                     "stepwise" : "linear"
    scheme = flux_temporal_scheme(String(get(cfg, "temporal_scheme", default_scheme)))
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
