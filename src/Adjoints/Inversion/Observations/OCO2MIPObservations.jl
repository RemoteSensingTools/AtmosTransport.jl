# ---------------------------------------------------------------------------
# OCO-2 / OCO-3 v11 MIP 10-second files → observation schema v2.
#
# The MIP observation files (OCO2_b11.2_10sec_GOOD_r2/r3.nc4,
# OCO3_b11_10sec_GOOD_r2.nc4) hold one record per 10-second span: the
# inverse-variance-weighted mean of the good soundings, with its uncertainty,
# a model error, the data type (land nadir, land glint, ocean glint, ...), the
# MIP assimilation flag, and the retrieval on 20 levels (sigma levels times the
# retrieved surface pressure, pressure weights, normalized averaging kernel,
# prior). Each record becomes a `kind = "column"` record with a retrieval
# block (`Observations.read_oco2_retrieval_columns`): the levels as given
# (`HatLevelProjection`, the default) or TM5-4DVar's 19 layers
# (`tm5_layer_column`). Every record of the time range is kept; those outside
# the experiment's data types and assimilation flags get a nonzero flag, so
# that `bind_to_mesh(...; flagged_records = "cosample")` co-samples them, as
# the MIP protocol asks.
# ---------------------------------------------------------------------------

# Data type codes of the MIP files (`data_type`, also the last digit of the
# sounding id) and their labels; 0 is OCO-3's snapshot area maps (SAM).
const OCO2_MIP_DATA_TYPES = (SAM = 0, LN = 1, LG = 2, LT = 3, LTR = 4, ON = 5, OG = 6, OT = 7, OTR = 8, MIX = 9)

function _oco2_mip_data_type_codes(data_types)
    codes = Int[]
    for t in data_types
        key = Symbol(uppercase(String(t)))
        haskey(OCO2_MIP_DATA_TYPES, key) || throw(ArgumentError(
            "OCO-2 MIP data type $(repr(t)) is not one of $(join(String.(keys(OCO2_MIP_DATA_TYPES)), ", "))"))
        push!(codes, OCO2_MIP_DATA_TYPES[key])
    end
    return codes
end

_oco2_mip_data_type_label(code::Integer) =
    something(findfirst(==(Int(code)), OCO2_MIP_DATA_TYPES), Symbol("type", code))

# Record times from `date (epoch_dimension = 7, sounding_id)` (year, month,
# day, hour, minute, second, millisecond), truncated to the second as TM5-4DVar
# reads them (`datetime(*date[:6])`; its comment notes invalid `time` values in
# one release).
function _oco2_mip_dates(ds, label)
    raw = Array(nc_variable(ds, ("date",), label).var)
    size(raw, 1) >= 6 || throw(ArgumentError("$(label): date has $(size(raw, 1)) components; expected 7"))
    return [ntuple(c -> Int16(raw[c, k]), 6) for k in axes(raw, 2)]
end

# A record variable over the block `cols`, as Float64.
_oco2_mip_column(ds, name, cols, label) = raw_observation_block(nc_variable(ds, (name,), label), cols, label)

function _oco2_mip_unit_scale(ds, name, label)
    var = nc_variable(ds, (name,), label)
    return mole_fraction_unit_scale(get(var.attrib, "units", nothing), "$(label) $(name)")
end

# The instrument label: the first token of the `ShortName` attribute
# (`OCO2_v11.2_XCO2_10sec_avg`), else of the file name.
function _oco2_mip_instrument(ds, path)
    name = get(ds.attrib, "ShortName", basename(path))
    return String(first(split(String(name), '_')))
end

"""
    read_oco2_mip_observations(path; time_range = nothing, data_types = ("LN", "LG", "OG"),
                               assimilate_flags = (1,), model_sigma = :file,
                               mapping = Observations.HatLevelProjection(), instrument = nothing,
                               tracer = "CO2") -> CSObservationSet

The records of an OCO-2 or OCO-3 v11 MIP 10-second file (read-only) as an
observation set of schema v2 ([`CSObservationRecord`](@ref)), one per
record whose time (`date`, truncated to the second) lies in
`time_range = (start, stop)` (`DateTime`s, half-open; `nothing` = all):

| record field | from the MIP file |
|---|---|
| `id` | `sounding_id` |
| `date_components` | `date[1:6]` (UTC) |
| `lat`, `lon` | `latitude`, `longitude` |
| `kind` | `"column"` |
| `elevation_m` | `altitude` (m above sea level) |
| `time_window_s` | 0 (the model step containing the time) |
| `value`, `value_sigma` | `xco2_2019_scale`, `xco2_uncertainty` (ppm → mol/mol) |
| `model_sigma` | `model_error` (`model_sigma = :file`), or a constant in ppm, or 0 |
| `flag` | 0 when `assimilate_flag ∈ assimilate_flags` and the data type is in `data_types`; else `1 + assimilate_flag` |
| `instrument_type` | `"<instrument>_<data type>"`, e.g. `"OCO2_OG"` |
| `tracer` | `tracer` |
| `retrieval` | the 20 levels (`sigma_levels × psurf`, hPa → Pa), `pressure_weight`, `xco2_averaging_kernel`, `co2_profile_apriori` (ppm → mol/mol), surface first, and the prior column `xco2_apriori` (ppm → mol/mol; `Observations.read_oco2_retrieval_columns`) |

`data_types` name the experiment's data types (`LN` land nadir, `LG` land
glint, `LT` land target, `LTR` land transition, `ON` ocean nadir, `OG` ocean
glint, `OT`, `OTR`, `MIX` mixed land/water, `SAM` OCO-3 snapshot area maps;
the MIP assimilates LN, LG and OG). `assimilate_flags` are the MIP `assimilate_flag` values to assimilate
(1; TM5-4DVar's MIP setup also takes the withheld 2). Every record of the
range is kept: flag 1 marks `assimilate_flag = 0` (sparse, target,
transition, mixed scenes; evaluation only), 2 an assimilable record whose
data type is outside `data_types`, 3 a withheld record (`assimilate_flag =
2`), unless `assimilate_flags` selects them. Bind with
`flagged_records = "cosample"` to co-sample them.

`mapping` is `Observations.HatLevelProjection()` (the levels as given, the
default) or `Observations.LayerRebin()` (TM5-4DVar's 19 layers,
`Observations.tm5_layer_column`; the CLI's `--mapping tm5`).
`instrument` defaults to the first token of the file's `ShortName`
(`"OCO2"`, `"OCO3"`). Sounding ids are unique within one file; OCO-2 and
OCO-3 ids can coincide, so give each instrument its own set.
"""
function read_oco2_mip_observations(path::AbstractString; time_range = nothing, kwargs...)
    isfile(path) || throw(ArgumentError("OCO-2 MIP file not found: $(repr(path))"))
    return NCDatasets.NCDataset(path, "r") do ds
        label = "OCO-2 MIP file $(repr(path))"
        dates = _oco2_mip_dates(ds, label)
        rows = _oco2_mip_rows(dates, time_range)
        CSObservationSet(_oco2_mip_records(ds, path, rows, dates, label; kwargs...);
                         time_origin = "1970-01-01 00:00:00")
    end
end

_oco2_mip_mapping_text(::HatLevelProjection) = "hat: the 20 levels as given (retrieval_mapping = hat_levels)"
_oco2_mip_mapping_text(::LayerRebin) = "tm5: TM5-4DVar's 19 layers (retrieval_mapping = layer_rebin)"

function _oco2_mip_rows(dates, time_range)
    time_range === nothing && return collect(eachindex(dates))
    start, stop = time_range
    start < stop || throw(ArgumentError("time_range must be (start, stop) with start < stop; got $(time_range)"))
    return findall(dc -> start <= _record_datetime(dc) < stop, dates)
end

function _oco2_mip_records(ds, path, rows::AbstractVector{<:Integer}, dates, label;
                           data_types = ("LN", "LG", "OG"), assimilate_flags = (1,), model_sigma = :file,
                           mapping::AbstractRetrievalMapping = HatLevelProjection(), instrument = nothing,
                           tracer::AbstractString = "CO2")
    isempty(rows) && return CSObservationRecord[]
    types = _oco2_mip_data_type_codes(data_types)
    flags_in = Int.(collect(assimilate_flags))
    all(f -> f in (0, 1, 2), flags_in) || throw(ArgumentError(
        "assimilate_flags must be MIP assimilate_flag values (0, 1, 2); got $(assimilate_flags)"))
    model_sigma === :file || model_sigma isa Real && isfinite(model_sigma) && model_sigma >= 0 ||
        throw(ArgumentError("model_sigma must be :file (the MIP model_error) or a constant >= 0 in ppm; got " *
                            "$(repr(model_sigma))"))
    name = instrument === nothing ? _oco2_mip_instrument(ds, path) : String(instrument)
    lo, hi = extrema(rows)
    cols = lo:hi
    column(var) = _oco2_mip_column(ds, var, cols, label)
    ids = Array(nc_variable(ds, ("sounding_id",), label).var[cols])
    lats, lons, elevation = column("latitude"), column("longitude"), column("altitude")
    value = column("xco2_2019_scale") .* _oco2_mip_unit_scale(ds, "xco2_2019_scale", label)
    value_sigma = column("xco2_uncertainty") .* _oco2_mip_unit_scale(ds, "xco2_uncertainty", label)
    model = model_sigma === :file ?
        column("model_error") .* _oco2_mip_unit_scale(ds, "model_error", label) :
        fill(Float64(model_sigma) * 1e-6, length(cols))
    data_type = column("data_type")
    assimilate_flag = column("assimilate_flag")
    retrievals = read_oco2_retrieval_columns(ds, rows; label)
    return map(enumerate(rows)) do (n, row)
        c = row - lo + 1
        code = Int(data_type[c])
        af = Int(assimilate_flag[c])
        selected = af in flags_in && code in types
        CSObservationRecord(; id = ids[c], date_components = dates[row], lat = lats[c], lon = lons[c],
                            kind = ColumnObservationKind(), elevation_m = elevation[c], time_window_s = 0,
                            flag = selected ? 0 : 1 + af, value = value[c], value_sigma = value_sigma[c],
                            model_sigma = model[c], instrument_type = "$(name)_$(_oco2_mip_data_type_label(code))",
                            tracer, retrieval = oco2_level_form(mapping, retrievals[n]))
    end
end

const _OCO2_MIP_PARTITIONS = (day = ("{YYYYMMDD}", Dates.DateFormat("yyyymmdd"), Dates.Date),
                              month = ("{YYYYMM}", Dates.DateFormat("yyyymm"),
                                       t -> Dates.firstdayofmonth(Dates.Date(t))))

"""
    convert_oco2_mip_observations(path, output; partition = :day, time_range = nothing,
                                  data_types = ("LN", "LG", "OG"), assimilate_flags = (1,),
                                  model_sigma = :file, mapping = Observations.HatLevelProjection(),
                                  instrument = nothing, tracer = "CO2") -> Vector{String}

Convert an OCO-2 or OCO-3 v11 MIP 10-second file to observation files of
schema v2 ([`write_observations`](@ref)), one per day (`partition = :day`,
`output` holds `{YYYYMMDD}`) or per month (`:month`, `{YYYYMM}`), for the
records of `time_range` (all records without one); days or months without
records get no file. The records are those of
[`read_oco2_mip_observations`](@ref) with the same options; the files record
the source file, the options and the mapping as root attributes. Returns the
paths written, in time order.
"""
function convert_oco2_mip_observations(path::AbstractString, output::AbstractString; partition::Symbol = :day,
                                       time_range = nothing, data_types = ("LN", "LG", "OG"),
                                       assimilate_flags = (1,), model_sigma = :file,
                                       mapping::AbstractRetrievalMapping = HatLevelProjection(),
                                       instrument = nothing, tracer::AbstractString = "CO2")
    haskey(_OCO2_MIP_PARTITIONS, partition) || throw(ArgumentError(
        "partition must be :day or :month; got $(repr(partition))"))
    token, format, key = _OCO2_MIP_PARTITIONS[partition]
    occursin(token, output) || throw(ArgumentError(
        "the output path of partition = :$(partition) must contain $(token); got $(repr(output))"))
    isfile(path) || throw(ArgumentError("OCO-2 MIP file not found: $(repr(path))"))
    options = (; data_types, assimilate_flags, model_sigma, mapping, instrument, tracer)
    attributes = Dict{String, Any}(
        "source_file" => basename(path),
        "source_converter" => "AtmosTransport.convert_oco2_mip_observations",
        "oco2_mip_data_types" => join(uppercase.(String.(collect(data_types))), " "),
        "oco2_mip_assimilate_flags" => join(string.(collect(assimilate_flags)), " "),
        "oco2_mip_model_sigma" => model_sigma === :file ? "model_error" : "$(model_sigma) ppm",
        "oco2_mip_mapping" => _oco2_mip_mapping_text(mapping),
        "oco2_mip_flag_convention" => "flag = 0: assimilated in this experiment; otherwise 1 + the MIP " *
                                      "assimilate_flag (1: evaluation-only scenes, 2: assimilable but outside " *
                                      "the experiment's data types, 3: withheld for evaluation)")
    return NCDatasets.NCDataset(path, "r") do ds
        label = "OCO-2 MIP file $(repr(path))"
        dates = _oco2_mip_dates(ds, label)
        rows = _oco2_mip_rows(dates, time_range)
        groups = Dict{Dates.Date, Vector{Int}}()
        for row in rows
            push!(get!(groups, key(_record_datetime(dates[row])), Int[]), row)
        end
        written = String[]
        for day in sort!(collect(keys(groups)))
            target = replace(output, token => Dates.format(day, format))
            records = _oco2_mip_records(ds, path, groups[day], dates, label; options...)
            write_observations(target, CSObservationSet(records; time_origin = "1970-01-01 00:00:00"); attributes)
            push!(written, target)
        end
        written
    end
end
