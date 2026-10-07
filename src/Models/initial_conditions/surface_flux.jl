# ===========================================================================
# Surface-flux loader + builders
#
# Owns the file-based surface-flux path: NetCDF load + LL→{LL,RG,CS}
# conservative regrid + area integration. Every builder returns
# `SurfaceFluxSource` with cell_mass_rate in model-storage units per cell per
# second. File inventories are physical kg species/s; the builders convert
# them to dry-air-equivalent storage for the dry-VMR transport state.
#
# Hoisted verbatim (modulo renames for dependency consolidation) from the
# historical LL/RG runner:
#   FileSurfaceFluxField, SECONDS_PER_MONTH, _surface_flux_kind,
#   _resolve_surface_flux_file, _normalize_units_string,
#   _load_file_surface_flux_field, _renormalize_surface_flux_rate!,
#   _REGRID_CACHE_DIR, _conservative_surface_flux_rate,
#   _regridding_method, build_surface_flux_source (LL + RG),
#   build_surface_flux_sources.
#
# `_build_emission_source_mesh` is dropped in favour of the shared
# `_build_source_latlon_mesh` introduced for the IC path.
# ===========================================================================

const SECONDS_PER_MONTH = 365.25 * 86400 / 12
const _DAYS_PER_MONTH_COMMON = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
const _DRY_AIR_MOLAR_MASS_KG_MOL = 28.96546e-3
const _KNOWN_TRACER_MOLAR_MASS_KG_MOL = Dict{Symbol, Float64}(
    :co2         => 44.0095e-3,
    :co2_natural => 44.0095e-3,
    :co2_fossil  => 44.0095e-3,
    :fossil_co2  => 44.0095e-3,
    :sf6         => 146.055e-3,
    :rn222       => 222.0e-3,
)

_is_leap_year(year::Integer) =
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0

function _days_in_month(year::Integer, month::Integer)
    1 <= month <= 12 || throw(ArgumentError("month must be in 1:12, got $month"))
    month == 2 && _is_leap_year(year) && return 29
    return _DAYS_PER_MONTH_COMMON[month]
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
        "surface_flux.kind=gridfed_fossil_co2 with kgCO2/month/m2 units " *
        "requires a calendar year to convert monthly totals to per-second " *
        "rates. Set surface_flux.year or use a filename containing YYYY."))
end

function _gridfed_month_seconds(cfg, file::AbstractString, ds, month::Integer)
    year = _surface_flux_year(cfg, file, ds)
    return Float64(_days_in_month(year, month)) * 86400.0
end

const _REGRID_CACHE_DIR = expanduser("~/.cache/AtmosTransport/cr_regridding")

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

@inline _surface_flux_kind(cfg) = Symbol(lowercase(String(get(cfg, "kind", "none"))))

function _tracer_molar_mass_kg_mol(tracer_name::Symbol, cfg)
    if haskey(cfg, "molar_mass_kg_mol")
        molar_mass = Float64(cfg["molar_mass_kg_mol"])
        molar_mass > 0 || throw(ArgumentError(
            "surface_flux.molar_mass_kg_mol for $(tracer_name) must be positive"))
        return molar_mass
    end
    haskey(_KNOWN_TRACER_MOLAR_MASS_KG_MOL, tracer_name) &&
        return _KNOWN_TRACER_MOLAR_MASS_KG_MOL[tracer_name]
    throw(ArgumentError(
        "surface flux for tracer $(tracer_name) is in physical kg species/s, " *
        "but the transport state stores dry VMR times dry-air mass. Set " *
        "surface_flux.molar_mass_kg_mol so the source can be converted to " *
        "model storage units."))
end

function _surface_flux_storage_scale(tracer_name::Symbol, cfg)
    # Surface inventories are physical kg species/s. The prognostic tracer
    # storage is dry VMR * dry-air mass, so source rates must be converted to
    # dry-air-equivalent storage before being applied by the surface kernels.
    return _DRY_AIR_MOLAR_MASS_KG_MOL / _tracer_molar_mass_kg_mol(tracer_name, cfg)
end

# Derive per-cell area `(Nx, Ny)` on a regular lat/lon grid from the
# coordinate vectors. Uses the spherical-cap formula
# `R² · Δlon · |sin(φ + Δlat/2) - sin(φ - Δlat/2)|` with R = 6.371e6 m.
# Used by the EDGAR-Tonnes branch when the source file does not carry
# a `cell_area` or `area` variable.
function _lonlat_cell_areas_m2(lon::AbstractVector, lat::AbstractVector)
    Nx, Ny = length(lon), length(lat)
    R = 6.371e6
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

function _normalize_units_string(units)
    units_str = String(units)
    return lowercase(replace(strip(units_str), " " => "", "^" => "", "²" => "2"))
end

function _load_file_surface_flux_field(cfg, ::Type{FT}) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    file, variable, time_index = _resolve_surface_flux_file(cfg, kind)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))

    ds = NCDataset(file)
    try
        lon_var = _ic_find_coord(ds, ["lon", "longitude", "x"])
        lat_var = _ic_find_coord(ds, ["lat", "latitude", "y"])
        isnothing(lon_var) && throw(ArgumentError("could not find longitude coordinate in $file"))
        isnothing(lat_var) && throw(ArgumentError("could not find latitude coordinate in $file"))
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))

        lon_src = Float64.(ds[lon_var][:])
        lat_src = Float64.(ds[lat_var][:])

        raw_var = ds[variable]
        raw = if ndims(raw_var) == 3
            if kind === :lmdz_co2
                # CAMS LMDZ files store 3-hourly fluxes (`time = 248`
                # for a 31-day month). For a one-month forward run we
                # use the monthly mean: average over the time axis so
                # the surface-flux pipeline (which carries a single
                # 2D field) sees a representative constant rate.
                # Sub-monthly variability is a follow-up.
                ntime = size(raw_var, 3)
                acc = zeros(Float64, size(raw_var, 1), size(raw_var, 2))
                @inbounds for t in 1:ntime
                    acc .+= Float64.(nomissing(raw_var[:, :, t], 0.0))
                end
                acc ./= ntime
                FT.(acc)
            else
                FT.(nomissing(raw_var[:, :, time_index], zero(FT)))
            end
        elseif ndims(raw_var) == 2
            FT.(nomissing(raw_var[:, :], zero(FT)))
        else
            throw(ArgumentError("surface-flux variable '$variable' must be 2D or 3D, got ndims=$(ndims(raw_var))"))
        end

        cell_area_src = if haskey(ds, "cell_area")
            Float64.(nomissing(ds["cell_area"][:, :], 0.0))
        elseif haskey(ds, "area")
            Float64.(nomissing(ds["area"][:, :], 0.0))
        else
            nothing
        end

        if length(lat_src) > 1 && lat_src[1] > lat_src[end]
            raw = raw[:, end:-1:1]
            lat_src = reverse(lat_src)
            cell_area_src === nothing || (cell_area_src = cell_area_src[:, end:-1:1])
        end

        if minimum(lon_src) < 0
            split = findfirst(>=(0), lon_src)
            if split !== nothing
                idx = vcat(split:length(lon_src), 1:split-1)
                lon_src = mod.(lon_src[idx], 360.0)
                raw = raw[idx, :]
                cell_area_src === nothing || (cell_area_src = cell_area_src[idx, :])
            end
        end

        units_norm = _normalize_units_string(get(raw_var.attrib, "units", ""))
        if kind === :gridfed_fossil_co2 || units_norm == "kgco2/month/m2"
            month_seconds = _gridfed_month_seconds(cfg, file, ds, time_index)
            raw ./= FT(month_seconds)
        elseif kind === :edgar_sf6 || units_norm == "tonnes"
            # EDGAR v8 stores per-cell annual mass (tonnes). Convert to
            # per-area per-second flux: kg/m²/s = (1000 · tonnes) /
            # (cell_area · seconds_per_year). cell_area must be present
            # in the file OR derivable from the lat/lon grid.
            cell_area_for_norm = cell_area_src
            if cell_area_for_norm === nothing
                cell_area_for_norm = _lonlat_cell_areas_m2(lon_src, lat_src)
            end
            seconds_per_year = 365.25 * 86400
            @inbounds for j in 1:size(raw, 2), i in 1:size(raw, 1)
                a = cell_area_for_norm[i, j]
                raw[i, j] = a > 0 ? FT(1000 * Float64(raw[i, j]) /
                                       (a * seconds_per_year)) : zero(FT)
            end
        elseif kind === :lmdz_co2 || units_norm in ("kgcm-2s-1", "kgc/m2/s", "kgcm2s-1")
            # CAMS / LMDZ flux is reported in kg of CARBON per m² per s.
            # Multiply by 44/12 = M(CO2)/M(C) to convert to kg of CO2.
            raw .*= FT(44.0 / 12.0)
        elseif !(isempty(units_norm) || occursin("/s", units_norm) || occursin("s-1", units_norm))
            throw(ArgumentError("unsupported surface-flux units '$units_norm' in $file; expected kgCO2/month/m2, Tonnes, kgC/m2/s, or per-second flux units"))
        end

        raw .*= FT(get(cfg, "scale", 1.0))
        native_total_mass_rate = cell_area_src === nothing ? NaN : sum(Float64.(raw) .* cell_area_src)
        return FileSurfaceFluxField{FT}(raw, lon_src, lat_src, native_total_mass_rate)
    finally
        close(ds)
    end
end

# Parse the slice times of a surface-flux file into seconds since
# `reference_time`. NCDatasets usually decodes CF time to DateTime/CFTime;
# we handle both the decoded (date-like) and raw-numeric ("hours since …")
# cases. Returns a `Vector{Float64}` (ascending after the caller sorts).
function _surface_flux_times_seconds(time_vals, units::AbstractString,
                                     reference_time::Union{DateTime, Nothing})
    n = length(time_vals)
    times_sec = Vector{Float64}(undef, n)

    sample = first(time_vals)
    # CFTime values from NCDatasets are <: Dates.TimeType in practice; the
    # `DateTime(x)` constructor accepts them. Numeric (Real) values mean the
    # variable was NOT CF-decoded, so we parse the `units` origin ourselves.
    if sample isa Real
        # units like "hours since 2021-12-01 00:00:00"
        m = match(r"(\w+)\s+since\s+(.+)", strip(String(units)))
        m === nothing && throw(ArgumentError(
            "time-varying surface flux: cannot parse numeric time units '$(units)'"))
        unit_word = lowercase(m.captures[1])
        per_unit_sec = unit_word in ("second", "seconds", "sec", "s") ? 1.0 :
                       unit_word in ("minute", "minutes", "min")       ? 60.0 :
                       unit_word in ("hour", "hours", "hr", "h")        ? 3600.0 :
                       unit_word in ("day", "days", "d")                ? 86400.0 :
                       throw(ArgumentError("time-varying surface flux: unsupported time unit '$(unit_word)'"))
        origin = _parse_cf_time_origin(strip(m.captures[2]))
        ref = reference_time === nothing ? origin : reference_time
        ref_offset_sec = Dates.value(origin - ref) / 1000.0   # ms → s
        for k in 1:n
            times_sec[k] = ref_offset_sec + Float64(time_vals[k]) * per_unit_sec
        end
    else
        # Date-like (DateTime / CFTime). Convert each to a DateTime and diff.
        if reference_time === nothing
            ref = DateTime(sample)   # assume file origin == run start (first slice)
        else
            ref = reference_time
        end
        for k in 1:n
            times_sec[k] = Dates.value(DateTime(time_vals[k]) - ref) / 1000.0
        end
    end
    return times_sec
end

# Parse a CF time-origin string ("2021-12-01 00:00:00" / "2021-12-01T00:00:00"
# / "2021-12-01") into a DateTime.
function _parse_cf_time_origin(s::AbstractString)
    ss = replace(strip(String(s)), "T" => " ")
    for fmt in (dateformat"y-m-d H:M:S", dateformat"y-m-d H:M", dateformat"y-m-d")
        try
            return DateTime(ss, fmt)
        catch
        end
    end
    throw(ArgumentError("time-varying surface flux: cannot parse time origin '$(s)'"))
end

# One file of a time-varying series: all slices (no monthly averaging), the
# static loader's lat-flip / lon-roll reorientation and unit conversion, and
# slice times in seconds since `reference_time`.
function _load_single_timevarying_surface_flux_field(
        cfg, ::Type{FT}, reference_time::Union{DateTime, Nothing},
        file::AbstractString, variable::AbstractString) where FT
    kind = _surface_flux_kind(cfg)
    isfile(file) || throw(ArgumentError("surface-flux file not found: $file"))

    ds = NCDataset(file)
    try
        lon_var = _ic_find_coord(ds, ["lon", "longitude", "x"])
        lat_var = _ic_find_coord(ds, ["lat", "latitude", "y"])
        time_var = _ic_find_coord(ds, ["time", "t"])
        isnothing(lon_var) && throw(ArgumentError("could not find longitude coordinate in $file"))
        isnothing(lat_var) && throw(ArgumentError("could not find latitude coordinate in $file"))
        haskey(ds, variable) || throw(ArgumentError("variable '$variable' not found in $file"))

        lon_src = Float64.(ds[lon_var][:])
        lat_src = Float64.(ds[lat_var][:])

        raw_var = ds[variable]
        ndims(raw_var) == 3 || throw(ArgumentError(
            "time-varying surface-flux variable '$variable' must be 3D (lon,lat,time), got ndims=$(ndims(raw_var))"))
        Nx, Ny, ntime = size(raw_var)
        raw = Array{FT, 3}(undef, Nx, Ny, ntime)
        @inbounds for t in 1:ntime
            raw[:, :, t] .= FT.(nomissing(raw_var[:, :, t], zero(FT)))
        end

        # --- reorientation (identical to the static loader, per slice) ---
        if length(lat_src) > 1 && lat_src[1] > lat_src[end]
            raw = raw[:, end:-1:1, :]
            lat_src = reverse(lat_src)
        end
        if minimum(lon_src) < 0
            split = findfirst(>=(0), lon_src)
            if split !== nothing
                idx = vcat(split:length(lon_src), 1:split-1)
                lon_src = mod.(lon_src[idx], 360.0)
                raw = raw[idx, :, :]
            end
        end

        # --- per-slice unit conversion (mirrors the static loader) ---
        units_norm = _normalize_units_string(get(raw_var.attrib, "units", ""))
        if kind === :gridfed_fossil_co2 || units_norm == "kgco2/month/m2"
            ntime == 12 || throw(ArgumentError(
                "time-varying GridFED source must contain 12 monthly slices, got $ntime in $file"))
            source_year = _surface_flux_year(cfg, file, ds)
            @inbounds for month in 1:12
                raw[:, :, month] ./= FT(_days_in_month(source_year, month) * 86400.0)
            end
        elseif kind === :lmdz_co2 || units_norm in ("kgcm-2s-1", "kgc/m2/s", "kgcm2s-1")
            raw .*= FT(44.0 / 12.0)   # kgC → kgCO2
        elseif !(isempty(units_norm) || occursin("/s", units_norm) || occursin("s-1", units_norm))
            throw(ArgumentError(
                "time-varying surface flux: unsupported units '$units_norm' in $file; " *
                "expected kgC/m2/s or per-second flux units"))
        end
        raw .*= FT(get(cfg, "scale", 1.0))

        times_sec = _timevarying_slice_seconds(time_var, ds, cfg, kind, file, reference_time)

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

# Slice times (seconds since `reference_time`) from the file's time axis, or,
# for GridFED files without one, the first instant of each month.
_timevarying_slice_seconds(time_var::AbstractString, ds, cfg, kind::Symbol, file, reference_time) =
    _surface_flux_times_seconds(ds[time_var][:], String(get(ds[time_var].attrib, "units", "")),
                                reference_time)
function _timevarying_slice_seconds(::Nothing, ds, cfg, kind::Symbol, file, reference_time)
    kind === :gridfed_fossil_co2 || throw(ArgumentError("could not find time coordinate in $file"))
    source_year = _surface_flux_year(cfg, file, ds)
    ref = reference_time === nothing ? DateTime(source_year, 1, 1) : reference_time
    return [Dates.value(DateTime(source_year, month, 1) - ref) / 1000.0 for month in 1:12]
end

"""
    _load_timevarying_surface_flux_field(cfg, FT, reference_time)

Load a regular lon/lat `(lon, lat, time)` surface-flux series from one file
(`file`), an explicit list (`files = [...]`, chronological), or twelve monthly
files (`file_pattern` with `{YYYYMM}`, for `year` or the run-start year).
Slices from all files are concatenated and sorted by time; timestamps must be
unique and every file must share one lon/lat grid. Units are converted per
slice to kg species m⁻² s⁻¹ (GridFED monthly totals use each month's length).
When `reference_time === nothing`, each file's own time origin is taken as
the run start and a warning is emitted.
"""
function _load_timevarying_surface_flux_field(cfg, ::Type{FT},
                                              reference_time::Union{DateTime, Nothing}) where FT
    kind = _surface_flux_kind(cfg)
    kind === :none && return nothing
    reference_time === nothing && @warn(
        "time-varying surface flux: no reference_time supplied; assuming the file's " *
        "time origin equals the run start (first slice → t=0).")
    files, variable = _resolve_timevarying_surface_flux_files(cfg, kind, reference_time)
    fields = [_load_single_timevarying_surface_flux_field(
                  cfg, FT, reference_time, file, variable) for file in files]

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

    @info "Loaded time-varying surface flux" kind files=length(files) slices=length(times_sec) first_time_seconds=first(times_sec) last_time_seconds=last(times_sec)
    return TimeVaryingFileSurfaceFluxField{FT}(raw, lon, lat, times_sec)
end

include("surface_flux_regridding.jl")
include("surface_flux_native.jl")

# ---------------------------------------------------------------------------
# build_surface_flux_source — LL / RG / CS
# ---------------------------------------------------------------------------

# Opt-in flag for the time-varying surface-flux path (default false →
# byte-identical static monthly-mean behavior).
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
        "time-varying surface flux is CS-only (LatLon support is a follow-up)"))

    source = _load_file_surface_flux_field(cfg, FT)
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
        "time-varying surface flux is CS-only (ReducedGaussian support is a follow-up)"))

    source = _load_file_surface_flux_field(cfg, FT)
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
(`time_varying` absent/false) path is byte-identical to before.
"""
function build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh},
                                   tracer_name::Symbol, cfg, ::Type{FT};
                                   reference_time::Union{DateTime, Nothing} = nothing) where FT
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
            mesh, tracer_name, cfg, FT, reference_time)
        return _build_timevarying_cs_surface_flux_source(mesh, tracer_name, cfg, FT, reference_time)
    end

    source = _load_file_surface_flux_field(cfg, FT)

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
                                                   reference_time::Union{DateTime, Nothing}) where FT
    Nc = mesh.Nc
    field = _load_timevarying_surface_flux_field(cfg, FT, reference_time)
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
    build_surface_flux_sources(grid, tracer_specs, ::Type{FT}; reference_time=nothing)

Build surface-flux source instances for every tracer spec that requests
one. Returns a tuple (possibly empty) suitable for the
`surface_sources = (…,)` kwarg on `DrivenSimulation`.

`reference_time` (the run start `DateTime`) is threaded to each
per-tracer builder so the time-varying CS path can align its slice times
to the simulation clock. It is ignored by static sources.
"""
function build_surface_flux_sources(grid, tracer_specs, ::Type{FT};
                                    reference_time::Union{DateTime, Nothing} = nothing) where FT
    sources = Any[]
    for spec in tracer_specs
        source = build_surface_flux_source(grid, spec.name, spec.surface_flux_cfg, FT;
                                           reference_time = reference_time)
        source === nothing || push!(sources, source)
    end
    return Tuple(sources)
end
