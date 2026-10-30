"""Sources for observation columns and sampling."""
module Sources

using Dates
using NCDatasets
using Printf
using Statistics: median!
using TOML
using ...ConfigChecks: config_bool as _config_bool, check_known_keys as _check_known_keys
import ...expand_data_path
using ..Retrievals

export AbstractAveragingKernelOutput, AbstractObservationMode, AbstractObservationSource,
       AbstractQualityFilter, AbstractSiteGrouping, AbstractSiteSchedule, AbstractTableFormat,
       AutoTableFormat, AveragingKernelColumns, CSVTableFormat, EveryWindow,
       LOWEST_GROUND_ELEVATION_M, LocationGrouping, NetCDFTableFormat, NoAveragingKernels,
       NoQualityFilter, OCO2LiteSource, ObsPackSource, ObservationSet, QualityFlagFilter,
       QualityFlagValues, SiteCodeGrouping, SiteMode, SiteRequest, SoundingMode, SoundingRequest,
       TOMLTableFormat, TableSource, TimeList, TimeRange, averaging_kernel_tracers,
       build_observation_set, expand_observation_paths, isvalid_lonlat, mode_label, nc_variable,
       observation_source_from_cfg, parse_iso_utc, quality_filter_keys, raw_observation_block,
       read_observation_requests, read_oco2_retrieval_columns, record_height_above_ground,
       schedule_label, site_active, source_keys, source_kind, source_mode, source_path_template,
       valid_elevation, valid_intake_height, valid_pressure, _is_plain_int

include("observation_sources.jl")
include("observation_requests.jl")
include("observation_readers.jl")

export pressure_unit_scale

end # module Sources
