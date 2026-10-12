#!/usr/bin/env julia
# ===========================================================================
# compare_cs_flux_files.jl: compare two `cs_native` surface-flux files
#
# The production-level (L2) comparator of the surface-flux port. It compares a
# reference and a candidate file variable by variable, in the layout the
# runtime reads (`kind = "cs_native"`): Julia dimensions (Xdim, Ydim, nf, time),
# flux density per m², panels in the order of the target mesh's convention
# (GEOS order for every current file). Variables without a time dimension,
# (Xdim, Ydim, nf), such as `land_fraction` or `cell_area`, are compared cell
# by cell when named with `--var`. A compared variable with any other dimension
# names or order is an error (bands and categories would be taken at the wrong
# cells).
#
# Cell areas come from the target mesh, as at run time: either the transport
# binary's mesh (`--binary`, read with `MetDrivers.TransportBinaryReader` and
# `load_grid(...; Hp = 0).horizontal`), or a GEOS-native GMAO cubed sphere
# built from `--nc` and `--radius` (default `Parameters.EARTH_RADIUS`). A
# `cell_area` variable stored in a file is never used for the totals; its
# largest element-wise deviation from the mesh areas is reported. When a file
# stores cell centres `lats`/`lons`, they must be (Xdim, Ydim, nf) fields
# within a tenth of the nominal cell width (90°/Nc) of the mesh cell centres,
# else the comparison stops with an error: the file is on another grid or panel
# convention. The largest angle is reported. A file with only one of `lats` and
# `lons` is an error; a file with neither cannot be checked for its panel
# convention. The same rules hold for the mask file.
#
# Variables: without `--var`, every (Xdim, Ydim, nf, time) series of either file
# is compared. A series in only one file fails the comparison, unless
# `--common-variables-only` (for example a v1 product against a v3 product with
# more variables), which compares the series of both files and lists the others.
# A name that is a series in one file and has other dimensions in the other is
# an error. `--common-variables-only` goes with the default selection only.
#
# Slices are matched as intervals [start, end): the decoded stamp of the `time`
# coordinate (dimension `time`), whatever the raw encoding (units, origin), and
# the end of the slice by the rule of the totals below (time-bounds upper
# bound, else the next stamp; for the last slice without bounds one month or one
# spacing later; unknown for a single slice without bounds, which only matches
# another unknown end). Every metric is computed on the matched slices. By
# default any difference between the two time axes fails the comparison: a
# stamp in only one file, or a common stamp whose slices end at different times
# (equal stamps with unequal time bounds integrate to different masses). With
# `--common-times` it does not (for example a short build against a long one):
# the slices both files share with the same start and end are compared, as long
# as there is at least one. A comparison of variables without a time dimension
# only reads no time axis.
#
# Reference scale (`--reference-scale NAME=FACTOR` or `NAME=PRESET`): every
# stored value a of the reference variable NAME becomes T(Float64(a) × FACTOR),
# T the reference's stored type, before any metric; that is the value the
# reference would hold had it been written with the factor. Fill values are
# detected on the stored values. Presets (`REFERENCE_SCALE_PRESETS`) are
# computed from `AtmosTransport.Parameters`:
#   carbon_44_12_to_iupac = (SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS) / (44/12)
#       = 0.99932484..., for files written with 44/12 before commit 71a08395.
#
# Units: every compared (Xdim, Ydim, nf, time) series must carry a non-empty
# `units` attribute in both files, else the comparison stops with an error
# (the runtime rejects a flux without units); (Xdim, Ydim, nf) fields such as
# masks may be unitless. The `units` attributes of the two variables must be
# equal strings; otherwise the comparison stops with an error, unless
# `--allow-unit-mismatch` (for example when a reference scale converts the
# units).
#
# Time bounds: slices are start-stamped. When the `time` coordinate names a
# `bounds` variable, it must exist as (2, ntime) stamps (Julia order, `time`
# last) whose lower bounds equal the stamps and where each upper bound equals
# the next lower bound (no gap, no overlap); anything else, such as
# centre-stamped intervals, is an error.
#
# Reported per variable (a = reference, b = candidate):
#   * global totals Σ density × mesh cell area per slice (rate), and per
#     calendar month and year (mass). Slice k holds from its stamp t_k to t_k+1
#     (the runtime's stepwise scheme); the last slice ends at its time-bounds
#     upper bound, else one month later for month-start stamps, else one
#     spacing later. Month totals split a slice that crosses a month boundary.
#     Relative difference (b - a) / |a| (0 when both are 0, ±Inf when only a
#     is 0); a month or year is `complete` when both files cover all of it.
#     Not computed for variables without a time dimension.
#   * per-cell relative differences |b - a| / max(|a|, |b|) pooled over cells
#     and slices: p50, p99 (quantile type 7) and max. The value lies in [0, 2]:
#     1 when only one of a, b is 0, above 1 when their signs differ. Cells
#     where both are 0 carry no information and are left out. Quantiles are
#     taken over the Float32-rounded differences; the max is exact.
#     Cells are split into a partition, taken in this order:
#       coastal  - with a mask: the land class (mask > --mask-threshold,
#                  default 0) differs from that of an edge neighbour, panel
#                  edges included (`Preprocessing.build_cs_global_face_table`);
#       polar    - the remaining cells of the polar panels (3 and 6 in GEOS
#                  order; `panel_labels` of the mesh convention);
#       interior - all other cells;
#     plus `all`. Without a mask there are no coastal cells.
#   * 60 equal-area latitude bands (`Analysis.EqualAreaBands`), uniform in sin(latitude) of the mesh cell
#     centres: duration-weighted time-mean rate per band, and the largest
#     per-slice relative difference per band (time-dependent variables only).
#   * Float32 ulp distance between a and b (values stored in another type are
#     rounded to Float32 first; +0 and -0 are 0 ulp apart) and the fraction of
#     values whose stored bits are equal, per category and overall. Bits are
#     only compared between values of the same stored type: when the two
#     variables are stored in different types (Float32 against Float64), the
#     bitwise fraction is not available and a `bitwise_fraction` check fails.
#   * non-finite values and stored fill values (`_FillValue`, `missing_value`),
#     which the runtime rejects; any of them fails the comparison. A value is a
#     fill value as NCDatasets decides it: any NaN when the fill value is NaN,
#     else == (so a NaN stored under a NaN fill value counts as both).
#
# Coastal mask (`--mask-var NAME [--mask-file PATH] [--mask-threshold X]`): a
# cs_native-layout variable, (Xdim, Ydim, nf) or (Xdim, Ydim, nf, time) with one
# slice, read from the reference file or from PATH. When PATH stores
# `lats`/`lons`, their largest angle from the mesh cell centres must stay below
# a tenth of the nominal cell width (90°/Nc), else the mask is rejected as being
# on another grid or panel convention. The coastal class only depends on where
# the land class changes, so a mask that is 1 on ocean and 0 on land gives the
# same coastal cells. Recommended sources (measured, with the command, in the
# parity harness README `scripts/validation/flux_port_parity/README.md`):
#   C90: `land_fraction` of the SIF v3 product
#        /temp1/cfranken/sif_gpp_iav/fluxes/sif_gpp_daily_co2flux_c90_2018_2025_v3.nc
#        (vegetated land fraction of the FLUXCOM-X pilot), --mask-threshold 0.05,
#        the land-cell rule of the SIF products: 16,343 land cells, 5180 coastal
#        cells. Antarctica is land; Greenland and Iceland are not.
#   C30: no land-fraction file exists at C30. The closest is
#        `nearest_land_fallback` of
#        /temp1/cfranken/trendy_transcom_c30_20260908/transcom_weights_c30.nc
#        (1 where no 0.25° sample of a TransCom-11 land region falls in the
#        cell), --mask-threshold 0: 1935 land cells, 903 coastal cells. It has
#        no land south of 60°S and none in Greenland, so those coasts are not
#        coastal under this mask.
#   C180: no land-fraction file was found.
#
# Not here yet: TransCom-11 region totals (design §4.1 L2). They come with the
# region rasters of step P7 (`Preprocessing.SurfaceFluxes` regions).
#
# Usage:
#   julia --project=. scripts/validation/compare_cs_flux_files.jl \
#       --reference <python_output.nc> --candidate <julia_output.nc> \
#       --binary <C90 transport binary> \
#       [--var FIRE_CO2_FLUX] [--var NAME_IN_REFERENCE:NAME_IN_CANDIDATE] [--var land_fraction] \
#       [--reference-scale FIRE_CO2_FLUX=carbon_44_12_to_iupac] [--allow-unit-mismatch] \
#       [--mask-var land_fraction] [--mask-file <file with the mask>] [--mask-threshold 0.05] \
#       [--common-times] [--common-variables-only] [--block-slices <n>] \
#       [--tol year_total=1e-9] [--tol cell_p99.interior=1e-6] [--tol bitwise_fraction=0.999] \
#       [--json report.json] [--summary summary.txt]
#
# Each `--tol` applies to every compared variable. Every tolerance, from the
# command line, the `tolerances` argument or the parity harness, goes through
# `validate_tolerance`: finite and non-negative, `bitwise_fraction` within
# [0, 1], `ulp_max` a whole number (an infinite or negative tolerance would make
# its gate pass whatever the discrepancy). A check whose sample is empty passes
# and is reported as `n/a`: a category without cells, a
# relative-difference quantile where every value is zero in both files, or a
# totals or band metric of a variable without a time dimension. A check that
# cannot be computed for any other reason fails (for example month totals of
# one slice without time bounds, or the bitwise fraction of variables stored in
# different types). The text summary goes to stdout; the JSON report holds
# every number above. The exit status is 0 when all checks pass and 1
# otherwise. Memory: one block of slices per file (`--block-slices`; by default
# a multiple of the time chunk of about 256 MiB) plus 4 bytes per differing
# value for the quantiles (one Float32 vector per category, grown by `push!`
# and sorted in place; the spare capacity and the copy made while a vector
# grows can briefly add up to about twice that). The quantiles over all cells
# are selected from the three sorted category vectors, without a merged copy.
# ===========================================================================

using Dates
using Logging
using Printf
using JSON3
using NCDatasets
using AtmosTransport
using AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention, panel_labels,
                            panel_cell_center_lonlat
using AtmosTransport.Parameters: EARTH_RADIUS, SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS
using AtmosTransport.MetDrivers: TransportBinaryReader, load_grid
using AtmosTransport.Analysis: EqualAreaBands, band_index
using AtmosTransport.Preprocessing: build_cs_global_face_table
using AtmosTransport.AtomicFiles: atomic_publish
# The runtime's `cs_native` series layout (Julia order), defined once in GriddedIO.
using AtmosTransport.GriddedIO: CS_NATIVE_FLUX_DIMS

# (Xdim, Ydim, nf) fields of a cs_native file, such as `land_fraction`.
const CS_NATIVE_STATIC_DIMS = CS_NATIVE_FLUX_DIMS[1:3]
const CS_PANELS = 6
const N_SIN_LAT_BANDS = 60
const POLAR_PANEL_LABELS = (:north_pole, :south_pole)

# A mask whose stored cell centres are further than this fraction of the
# nominal cell width (90°/Nc) from the mesh centres is on another grid.
const CENTRE_DEVIATION_LIMIT_CELLS = 0.1

# Cell partition, in order of precedence (codes index `CATEGORY_NAMES`).
const INTERIOR, COASTAL, POLAR = UInt8(1), UInt8(2), UInt8(3)
const CATEGORY_NAMES = ("interior", "coastal", "polar")
const CATEGORY_SUFFIXES = ("all", CATEGORY_NAMES...)

# Float32 ulp-distance histogram buckets (inclusive ranges).
const ULP_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 16), (17, 256), (257, typemax(Int64)))
_bucket_label((lo, hi)) = hi == typemax(Int64) ? ">=$(lo)" : lo == hi ? "$(lo)" : "$(lo)-$(hi)"

const DEFAULT_BLOCK_BYTES = 256 * 2^20

# The C -> CO2 factor of every Python flux file written before commit 71a08395
# (`44.0 / 12.0`). A legacy encoding reproduced to rescale old files, not a
# physical constant; the physical ratio is SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS.
const LEGACY_CO2_PER_CARBON = 44 / 12

# Named reference scales: factor and its definition.
const REFERENCE_SCALE_PRESETS = (
    carbon_44_12_to_iupac = (factor = (SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS) / LEGACY_CO2_PER_CARBON,
                             definition = "(SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS) / (44/12): " *
                                          "a kg CO2 value converted from carbon with 44/12, rewritten with M(CO2)/M(C)"),
)

const REFERENCE_SCALE_RULE = "each stored reference value a becomes T(Float64(a) * factor), T the reference's " *
                             "stored type, before every metric; fill values are detected on the stored values"

# metric => (bound, per-category, meaning); `:upper` passes when value <= tolerance,
# `:lower` when value >= tolerance.
const TOLERANCE_METRICS = (
    slice_total      = (:upper, false, "max |(b - a)/a| of the per-slice global totals"),
    month_total      = (:upper, false, "max |(b - a)/a| of the calendar-month totals"),
    year_total       = (:upper, false, "max |(b - a)/a| of the calendar-year totals"),
    band             = (:upper, false, "max over the 60 bands of |b - a| / max |a| (time-mean band rates)"),
    cell_p50         = (:upper, true,  "median per-cell relative difference"),
    cell_p99         = (:upper, true,  "99th percentile of the per-cell relative difference"),
    cell_max         = (:upper, true,  "largest per-cell relative difference"),
    ulp_max          = (:upper, true,  "largest Float32 ulp distance"),
    bitwise_fraction = (:lower, true,  "fraction of bitwise-equal values (passes when >= tolerance)"),
)
const TOTAL_METRICS = ("slice_total", "month_total", "year_total", "band")
const RELATIVE_METRICS = ("cell_p50", "cell_p99", "cell_max")

const USAGE = """
Usage: julia --project=. scripts/validation/compare_cs_flux_files.jl \\
           --reference <a.nc> --candidate <b.nc>
           (--binary <transport.bin> | --nc <Nc> [--radius <m>])
           [--var NAME | --var NAME_A:NAME_B]...
           [--reference-scale NAME=FACTOR | --reference-scale NAME=PRESET]...
           [--allow-unit-mismatch]
           [--mask-var <variable> [--mask-file <path>] [--mask-threshold <x>]]
           [--common-times] [--common-variables-only] [--block-slices <n>]
           [--tol <metric>=<value>]...
           [--json <report.json>] [--summary <summary.txt>]

Variables: (Xdim, Ydim, nf, time) series and (Xdim, Ydim, nf) fields; without
--var every series of either file, and a series in only one file fails unless
--common-variables-only.
Reference-scale presets:
$(join(("  " * rpad(string(k), 24) * @sprintf("%.17g", v.factor) * "  " * v.definition
        for (k, v) in pairs(REFERENCE_SCALE_PRESETS)), "\n"))
Tolerance metrics (applied to every compared variable; the ones marked * take an
optional category suffix .all (default), .interior, .coastal or .polar):
$(join(("  " * rpad(string(k) * (v[2] ? "*" : ""), 18) * v[3] for (k, v) in pairs(TOLERANCE_METRICS)), "\n"))
Tolerances are finite and non-negative; bitwise_fraction lies in [0, 1] and
ulp_max is a whole number. A check whose sample is empty passes as n/a. Exit
status: 0 when every check passes, 1 otherwise.
"""

# ---------------------------------------------------------------------------
# Command line and option values
# ---------------------------------------------------------------------------

# Raw `metric[.category]` => number of one `--tol`; checked by `validate_tolerances`.
function _parse_tolerance(spec::AbstractString)
    parts = split(spec, '='; limit = 2)
    length(parts) == 2 || throw(ArgumentError("--tol expects <metric>=<value>, got `$(spec)`"))
    value = tryparse(Float64, strip(parts[2]))
    value === nothing && throw(ArgumentError("--tol $(strip(parts[1])): `$(parts[2])` is not a number"))
    return String(strip(parts[1])) => value
end

"""
    validate_tolerance(key, value; context = "") -> (canonical_key => Float64)

The one check of a tolerance, shared by the command line, `compare_cs_flux_files`
and the parity harness. `key` is a metric of `TOLERANCE_METRICS`, with a
category suffix for the metrics that take one (canonical form `metric` or
`metric.category`, default category `all`). `value` is a real number (not a
Bool), finite and non-negative; `bitwise_fraction` lies in [0, 1]; `ulp_max`
counts ulps and is a whole number. `context` is appended to error messages.
"""
function validate_tolerance(key::AbstractString, value; context::AbstractString = "")
    canonical = _tolerance_key(strip(key))
    metric = first(split(canonical, '.'))
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "tolerance `$(canonical)`$(context) must be a number, got $(repr(value))"))
    x = Float64(value)
    isfinite(x) && x >= 0 || throw(ArgumentError(
        "tolerance `$(canonical)`$(context) must be finite and non-negative, got $(value)"))
    metric == "bitwise_fraction" && x > 1 && throw(ArgumentError(
        "tolerance `$(canonical)`$(context) is a fraction and must lie in [0, 1], got $(value)"))
    metric == "ulp_max" && !isinteger(x) && throw(ArgumentError(
        "tolerance `$(canonical)`$(context) counts ulps and must be a whole number, got $(value)"))
    return canonical => x
end

"""
    validate_tolerances(pairs; context = "") -> Vector{Pair{String, Float64}}

`validate_tolerance` of every `key => value` pair, in order; a metric given
twice (after making keys canonical) is an error.
"""
function validate_tolerances(pairs; context::AbstractString = "")
    out = Pair{String, Float64}[]
    for (key, value) in pairs
        tol = validate_tolerance(String(key), value; context)
        any(t -> t.first == tol.first, out) &&
            throw(ArgumentError("tolerance `$(tol.first)` given twice$(context)"))
        push!(out, tol)
    end
    return out
end

# Canonical key: `metric` or `metric.category` (category metrics default to `.all`).
function _tolerance_key(raw::AbstractString)
    name, category = occursin('.', raw) ? split(raw, '.'; limit = 2) : (raw, nothing)
    sym = Symbol(name)
    haskey(TOLERANCE_METRICS, sym) || throw(ArgumentError(
        "unknown tolerance metric `$(name)`; known: $(join(keys(TOLERANCE_METRICS), ", "))"))
    per_category = TOLERANCE_METRICS[sym][2]
    if category === nothing
        return per_category ? "$(name).all" : String(name)
    end
    per_category || throw(ArgumentError("tolerance metric `$(name)` takes no category suffix"))
    category in CATEGORY_SUFFIXES || throw(ArgumentError(
        "unknown cell category `$(category)`; known: $(join(CATEGORY_SUFFIXES, ", "))"))
    return "$(name).$(category)"
end

function _variable_pair(spec::AbstractString)
    parts = split(spec, ':')
    length(parts) in (1, 2) && all(!isempty, parts) ||
        throw(ArgumentError("--var expects NAME or NAME_IN_REFERENCE:NAME_IN_CANDIDATE, got `$(spec)`"))
    return String(first(parts)) => String(last(parts))
end
_variable_pair(p::Pair) = String(p.first) => String(p.second)

"""
    resolve_reference_scale(value) -> (; factor, preset, definition)

A reference scale from a number or from a name in `REFERENCE_SCALE_PRESETS`
(a string holding a number is read as that number). The factor must be finite
and non-zero.
"""
function resolve_reference_scale(value::Real)
    factor = Float64(value)
    isfinite(factor) && factor != 0 || throw(ArgumentError(
        "a reference scale must be a finite non-zero number, got $(value)"))
    return (; factor, preset = nothing, definition = nothing)
end
function resolve_reference_scale(value::AbstractString)
    name = Symbol(strip(value))
    if haskey(REFERENCE_SCALE_PRESETS, name)
        preset = REFERENCE_SCALE_PRESETS[name]
        return (; factor = preset.factor, preset = String(name), definition = preset.definition)
    end
    factor = tryparse(Float64, strip(value))
    factor === nothing && throw(ArgumentError(
        "unknown reference scale `$(value)`: give a number or one of the presets " *
        "$(join(keys(REFERENCE_SCALE_PRESETS), ", "))"))
    return resolve_reference_scale(factor)
end
resolve_reference_scale(value::NamedTuple{(:factor, :preset, :definition)}) =
    (resolve_reference_scale(value.factor); value)
resolve_reference_scale(value::Bool) =
    throw(ArgumentError("a reference scale must be a number or a preset name, got $(value)"))

function _parse_reference_scale(spec::AbstractString)
    parts = split(spec, '='; limit = 2)
    length(parts) == 2 && !isempty(strip(parts[1])) || throw(ArgumentError(
        "--reference-scale expects NAME=FACTOR or NAME=PRESET, got `$(spec)`"))
    return String(strip(parts[1])) => resolve_reference_scale(parts[2])
end

# name in the reference => resolved scale; `specs` holds name => value pairs
# (a vector, a Dict or a NamedTuple).
function _reference_scales(specs)
    scales = Dict{String, NamedTuple}()
    for (name, value) in (specs isa NamedTuple ? pairs(specs) : specs)
        key = String(name)
        haskey(scales, key) && throw(ArgumentError("reference scale for `$(key)` given twice"))
        scales[key] = resolve_reference_scale(value)
    end
    return scales
end

function _parse_args(argv)
    reference = candidate = binary = mask_var = mask_file = json = summary = nothing
    Nc = radius = block_slices = mask_threshold = nothing
    common_times = common_variables_only = allow_unit_mismatch = false
    variables = Pair{String, String}[]
    raw_tolerances = Pair{String, Float64}[]
    scales = Pair{String, Any}[]

    i = 1
    function value()
        i + 1 <= length(argv) || throw(ArgumentError("missing value for $(argv[i])\n$(USAGE)"))
        return argv[i + 1]
    end
    function number(T)
        x = tryparse(T, value())
        x === nothing && throw(ArgumentError("$(argv[i]) expects a $(T), got `$(value())`"))
        return x
    end
    while i <= length(argv)
        arg = argv[i]
        if arg == "--reference"
            reference = expanduser(value()); i += 2
        elseif arg == "--candidate"
            candidate = expanduser(value()); i += 2
        elseif arg == "--binary"
            binary = expanduser(value()); i += 2
        elseif arg == "--nc"
            Nc = number(Int); i += 2
        elseif arg == "--radius"
            radius = number(Float64); i += 2
        elseif arg == "--var"
            push!(variables, _variable_pair(value())); i += 2
        elseif arg == "--reference-scale"
            scale = _parse_reference_scale(value())
            any(s -> s.first == scale.first, scales) &&
                throw(ArgumentError("reference scale for `$(scale.first)` given twice"))
            push!(scales, scale); i += 2
        elseif arg == "--allow-unit-mismatch"
            allow_unit_mismatch = true; i += 1
        elseif arg == "--mask-var"
            mask_var = value(); i += 2
        elseif arg == "--mask-file"
            mask_file = expanduser(value()); i += 2
        elseif arg == "--mask-threshold"
            mask_threshold = number(Float64); i += 2
        elseif arg == "--common-times"
            common_times = true; i += 1
        elseif arg == "--common-variables-only"
            common_variables_only = true; i += 1
        elseif arg == "--block-slices"
            block_slices = number(Int); i += 2
        elseif arg == "--tol"
            push!(raw_tolerances, _parse_tolerance(value())); i += 2
        elseif arg == "--json"
            json = expanduser(value()); i += 2
        elseif arg == "--summary"
            summary = expanduser(value()); i += 2
        elseif arg in ("-h", "--help")
            println(USAGE); exit(0)
        else
            throw(ArgumentError("unknown argument `$(arg)`\n$(USAGE)"))
        end
    end

    tolerances = validate_tolerances(raw_tolerances; context = " (--tol)")
    reference === nothing && throw(ArgumentError("--reference is required\n$(USAGE)"))
    candidate === nothing && throw(ArgumentError("--candidate is required\n$(USAGE)"))
    isfile(reference) || throw(ArgumentError("reference file not found: $(reference)"))
    isfile(candidate) || throw(ArgumentError("candidate file not found: $(candidate)"))
    (binary === nothing) == (Nc === nothing) &&
        throw(ArgumentError("give the target mesh by exactly one of --binary or --nc\n$(USAGE)"))
    binary === nothing || isfile(binary) || throw(ArgumentError("transport binary not found: $(binary)"))
    binary !== nothing && radius !== nothing &&
        throw(ArgumentError("--radius goes with --nc; a binary carries its own radius"))
    Nc === nothing || Nc >= 1 || throw(ArgumentError("--nc must be positive, got $(Nc)"))
    radius === nothing || (isfinite(radius) && radius > 0) ||
        throw(ArgumentError("--radius must be a positive length in m, got $(radius)"))
    block_slices === nothing || block_slices >= 1 ||
        throw(ArgumentError("--block-slices must be positive, got $(block_slices)"))
    mask_var === nothing && (mask_file !== nothing || mask_threshold !== nothing) &&
        throw(ArgumentError("--mask-file and --mask-threshold need --mask-var"))
    mask_threshold === nothing || isfinite(mask_threshold) ||
        throw(ArgumentError("--mask-threshold must be finite, got $(mask_threshold)"))
    mask_file === nothing || isfile(mask_file) || throw(ArgumentError("mask file not found: $(mask_file)"))
    mask_var === nothing && any(t -> endswith(t.first, ".coastal"), tolerances) &&
        throw(ArgumentError("coastal tolerances need --mask-var"))
    common_variables_only && !isempty(variables) && throw(ArgumentError(
        "--common-variables-only goes with the default variable selection; with --var the selection is explicit"))

    return (; reference, candidate, binary, Nc, radius, variables, reference_scale = scales,
              allow_unit_mismatch, mask_var, mask_file,
              mask_threshold = something(mask_threshold, 0.0), common_times, common_variables_only,
              block_slices, tolerances, json, summary)
end

# ---------------------------------------------------------------------------
# Target mesh, cell categories and latitude bands
# ---------------------------------------------------------------------------

"""
    load_target_mesh(; binary = nothing, Nc = nothing, radius = nothing) -> (mesh, info)

The Float64 cubed-sphere mesh whose `cell_areas` weight the totals: the mesh of
a transport binary (as the runtime builds it), or a GEOS-native GMAO mesh of
`Nc` cells per panel edge on a sphere of `radius` (default `EARTH_RADIUS`).
"""
function load_target_mesh(; binary = nothing, Nc = nothing, radius = nothing)
    if binary !== nothing
        reader = TransportBinaryReader(String(binary); FT = Float64)
        mesh = try
            load_grid(reader; FT = Float64, Hp = 0).horizontal
        finally
            close(reader)
        end
        mesh isa CubedSphereMesh || throw(ArgumentError(
            "$(binary) is not a cubed-sphere transport binary (mesh $(typeof(mesh)))"))
        return mesh, (; source = "binary", path = abspath(binary))
    end
    Nc === nothing && throw(ArgumentError("load_target_mesh needs `binary` or `Nc`"))
    R = radius === nothing ? EARTH_RADIUS : Float64(radius)
    mesh = CubedSphereMesh(; Nc, FT = Float64, Hp = 0, convention = GEOSNativePanelConvention(),
                           radius = R)
    return mesh, (; source = "constructed", path = nothing)
end

"""
    cell_categories(mesh; mask = nothing, land_threshold = 0.0) -> (codes, polar_panels)

`(Nc, Nc, 6)` codes of the cell partition (`COASTAL` before `POLAR` before
`INTERIOR`). A cell is coastal when its land class (`mask > land_threshold`)
differs from that of a cell sharing one of its edges, across panel edges too.
"""
function cell_categories(mesh::CubedSphereMesh; mask = nothing, land_threshold::Real = 0.0)
    Nc = mesh.Nc
    codes = fill(INTERIOR, Nc, Nc, CS_PANELS)
    labels = panel_labels(mesh)
    polar_panels = [p for p in 1:CS_PANELS if labels[p] in POLAR_PANEL_LABELS]
    for p in polar_panels
        codes[:, :, p] .= POLAR
    end
    if mask !== nothing
        size(mask) == (Nc, Nc, CS_PANELS) || throw(DimensionMismatch(
            "mask has shape $(size(mask)); expected ($(Nc), $(Nc), $(CS_PANELS))"))
        land = vec(mask .> land_threshold)
        # Face-table cell index (p-1)Nc² + (j-1)Nc + i is the linear index of (i, j, p).
        faces = build_cs_global_face_table(Nc, mesh.connectivity)
        for f in 1:faces.nf
            l, r = faces.face_left[f], faces.face_right[f]
            if land[l] != land[r]
                codes[l] = COASTAL
                codes[r] = COASTAL
            end
        end
    end
    return codes, polar_panels
end

"""
    sin_latitude_bands(mesh, nbands = 60) -> Array{Int, 3}

Band index of every cell centre in `Analysis.EqualAreaBands(nbands)`: band `b`
holds the centres with `s_{b-1} <= sin(latitude) < s_b`, `s_k = (k / nbands) * 2 - 1`
(equal-area bands, band 1 at the south pole).
"""
function sin_latitude_bands(mesh::CubedSphereMesh, nbands::Int = N_SIN_LAT_BANDS)
    Nc = mesh.Nc
    bands = EqualAreaBands(nbands)
    band = Array{Int}(undef, Nc, Nc, CS_PANELS)
    for p in 1:CS_PANELS
        _, lat = panel_cell_center_lonlat(mesh, p)
        band[:, :, p] .= band_index.(Ref(bands), lat)
    end
    return band
end

_unit_vector(lon, lat) = (cosd(lat) * cosd(lon), cosd(lat) * sind(lon), sind(lat))

# Angle in degrees between two points given in degrees (chord formula, exact
# for small angles).
function _angle_deg(lon1, lat1, lon2, lat2)
    u, v = _unit_vector(lon1, lat1), _unit_vector(lon2, lat2)
    chord = sqrt((u[1] - v[1])^2 + (u[2] - v[2])^2 + (u[3] - v[3])^2)
    return 2 * asind(min(1.0, chord / 2))
end

centre_deviation_limit_deg(Nc::Integer) = CENTRE_DEVIATION_LIMIT_CELLS * 90 / Nc

"""
    centre_deviation_deg(ds, mesh) -> Union{Float64, Nothing}

Largest angle (degrees) between the `lats`/`lons` cell centres stored in `ds`
and the cell centres of `mesh`; `nothing` when `ds` stores neither. Only one of
the two is an `ArgumentError`; stored centres in another layout than
(Xdim, Ydim, nf) of the mesh are a `DimensionMismatch`.
"""
function centre_deviation_deg(ds, mesh::CubedSphereMesh)
    has_lats, has_lons = haskey(ds, "lats"), haskey(ds, "lons")
    has_lats || has_lons || return nothing
    has_lats && has_lons || throw(ArgumentError(
        "$(NCDatasets.path(ds)) stores `$(has_lats ? "lats" : "lons")` without `$(has_lats ? "lons" : "lats")`: " *
        "the cell centres cannot be checked against the mesh"))
    Nc = mesh.Nc
    for name in ("lats", "lons")
        v = ds[name]
        Tuple(dimnames(v)) == CS_NATIVE_STATIC_DIMS && size(v) == (Nc, Nc, CS_PANELS) ||
            throw(DimensionMismatch(
                "cell centres `$(name)` in $(NCDatasets.path(ds)) have dimensions $(dimnames(v)) and size " *
                "$(size(v)); expected the cs_native layout $(CS_NATIVE_STATIC_DIMS) of C$(Nc) " *
                "($(Nc), $(Nc), $(CS_PANELS))"))
    end
    lat = Float64.(Array(ds["lats"].var))
    lon = Float64.(Array(ds["lons"].var))
    deviation = 0.0
    for p in 1:CS_PANELS
        mlon, mlat = panel_cell_center_lonlat(mesh, p)
        for j in 1:Nc, i in 1:Nc
            deviation = max(deviation, _angle_deg(lon[i, j, p], lat[i, j, p], mlon[i, j], mlat[i, j]))
        end
    end
    return deviation
end

# ---------------------------------------------------------------------------
# Time axes
# ---------------------------------------------------------------------------

_seconds(p::Period) = Millisecond(p) / Millisecond(Second(1))
_stamp(t::DateTime) = Dates.format(t, dateformat"yyyy-mm-ddTHH:MM:SS")
_month_label((y, m)) = @sprintf("%04d-%02d", y, m)
_is_month_start(t::DateTime) = t == firstdayofmonth(t)

"""
    read_time_axis(ds, path) -> NamedTuple

Decoded stamps of the `time` coordinate (dimension `time`) and the end of every
slice. Slices are start-stamped: slice k holds from its stamp to the next
stamp, the last one for one month (month-start stamps) or one spacing. A
declared `bounds` variable must agree with that: lower bounds equal to the
stamps, each upper bound equal to the next lower bound; it then gives the end
of the last slice. Other bounds (centre-stamped, gaps, overlaps) are an error.
"""
function read_time_axis(ds, path)
    haskey(ds, "time") || throw(ArgumentError("no time coordinate `time` in $(path)"))
    var = ds["time"]
    Tuple(dimnames(var)) == ("time",) || throw(DimensionMismatch(
        "the `time` coordinate of $(path) has dimensions $(dimnames(var)); expected (\"time\",)"))
    raw = var[:]
    all(t -> t isa DateTime, raw) || throw(ArgumentError(
        "time axis of $(path) does not decode to Gregorian DateTime stamps " *
        "(units $(repr(get(var.attrib, "units", ""))), calendar $(repr(get(var.attrib, "calendar", ""))))"))
    times = DateTime.(raw)
    isempty(times) && throw(ArgumentError("time axis of $(path) is empty"))
    all(k -> times[k] < times[k + 1], 1:length(times) - 1) ||
        throw(ArgumentError("time stamps of $(path) are not strictly ascending"))

    bounds_name = get(var.attrib, "bounds", nothing)
    ends, end_rule = bounds_name === nothing ? _infer_slice_ends(times) :
                     (_read_time_bounds(ds, String(bounds_name), times, path), "time bounds `$(bounds_name)`")
    return (; name = "time", units = String(get(var.attrib, "units", "")),
              calendar = String(get(var.attrib, "calendar", "")), times, ends, end_rule)
end

# Upper bounds of start-stamped slices, after checking the bounds variable
# against the stamps (see `read_time_axis`).
function _read_time_bounds(ds, name, times::Vector{DateTime}, path)
    haskey(ds, name) || throw(ArgumentError(
        "the time axis of $(path) declares bounds `$(name)`, which is not in the file"))
    bv = ds[name]
    n = length(times)
    ndims(bv) == 2 && size(bv) == (2, n) && dimnames(bv)[2] == "time" || throw(DimensionMismatch(
        "time bounds `$(name)` of $(path) have dimensions $(dimnames(bv)) and size $(size(bv)); " *
        "expected (2, $(n)) with `time` last (Julia order)"))
    bounds = bv[:, :]
    all(t -> t isa DateTime, bounds) || throw(ArgumentError(
        "time bounds `$(name)` of $(path) do not decode to DateTime stamps"))
    lower, upper = DateTime.(bounds[1, :]), DateTime.(bounds[2, :])
    k = findfirst(k -> lower[k] != times[k], 1:n)
    k === nothing || throw(ArgumentError(
        "time bounds `$(name)` of $(path) do not start at the stamps (slice $(k): stamp $(_stamp(times[k])), " *
        "lower bound $(_stamp(lower[k]))); slices are start-stamped, so the lower bounds must equal the stamps"))
    k = findfirst(k -> upper[k] <= lower[k], 1:n)
    k === nothing || throw(ArgumentError(
        "time bounds `$(name)` of $(path) do not end after their start (slice $(k): " *
        "$(_stamp(lower[k])) .. $(_stamp(upper[k])))"))
    k = findfirst(k -> upper[k] != lower[k + 1], 1:n - 1)
    k === nothing || throw(ArgumentError(
        "time bounds `$(name)` of $(path) $(upper[k] < lower[k + 1] ? "leave a gap" : "overlap") between " *
        "slices $(k) and $(k + 1) (upper bound $(_stamp(upper[k])), next lower bound $(_stamp(lower[k + 1])))"))
    return upper
end

function _infer_slice_ends(times::Vector{DateTime})
    n = length(times)
    n >= 2 || return nothing, "unknown: one slice without time bounds"
    ends = similar(times)
    ends[1:n - 1] .= times[2:n]
    if all(_is_month_start, times)
        step = 12 * (year(times[n]) - year(times[n - 1])) + month(times[n]) - month(times[n - 1])
        ends[n] = times[n] + Month(step)
        return ends, "next stamp; last slice $(step) month(s) (month-start stamps)"
    end
    ends[n] = times[n] + (times[n] - times[n - 1])
    return ends, "next stamp; last slice one spacing"
end

# (year, month) => seconds of [t0, t1) in that calendar month.
function _month_overlaps(t0::DateTime, t1::DateTime)
    out = Pair{Tuple{Int, Int}, Float64}[]
    cur = t0
    while cur < t1
        nxt = min(t1, firstdayofmonth(cur) + Month(1))
        push!(out, (year(cur), month(cur)) => _seconds(nxt - cur))
        cur = nxt
    end
    return out
end

_month_seconds((y, m)) = _seconds(Date(y, m, 1) + Month(1) - Date(y, m, 1))
_year_seconds(y) = _seconds(Date(y + 1, 1, 1) - Date(y, 1, 1))

# End of slice k (the rule of the totals), `nothing` when unknown (one slice
# without time bounds).
_slice_end(axis, k::Int) = axis.ends === nothing ? nothing : axis.ends[k]
_stamp_or_unknown(t) = t === nothing ? "unknown" : _stamp(t)

"""
    _match_intervals(axis_a, axis_b) -> (matched, end_differs)

Slices are intervals [stamp, end). `matched`: ascending stamps present in both
axes whose slices also end at the same time (an unknown end only matches an
unknown end); `end_differs`: the common stamps whose ends differ, as
`(; start, reference_end, candidate_end)`.
"""
function _match_intervals(axis_a, axis_b)
    pos_b = Dict(t => k for (k, t) in enumerate(axis_b.times))
    matched = DateTime[]
    end_differs = NamedTuple{(:start, :reference_end, :candidate_end), NTuple{3, String}}[]
    for (ka, t) in enumerate(axis_a.times)
        kb = get(pos_b, t, nothing)
        kb === nothing && continue
        ea, eb = _slice_end(axis_a, ka), _slice_end(axis_b, kb)
        if ea == eb
            push!(matched, t)
        else
            push!(end_differs, (; start = _stamp(t), reference_end = _stamp_or_unknown(ea),
                                  candidate_end = _stamp_or_unknown(eb)))
        end
    end
    return matched, end_differs
end

# ---------------------------------------------------------------------------
# Per-element statistics
# ---------------------------------------------------------------------------

mutable struct CategoryStats
    n            :: Int                 # values compared (cells × slices)
    n_nonfinite  :: Int
    n_both_zero  :: Int
    n_one_zero   :: Int
    n_equal      :: Int                 # nonzero and numerically equal
    rel          :: Vector{Float32}     # nonzero relative differences (>= 0)
    max_rel      :: Float64
    argmax       :: NTuple{4, Int}      # (i, j, panel, compared slice or 0)
    argmax_ab    :: NTuple{2, Float64}
    n_bitwise    :: Int
    ulp_max      :: Int64
    ulp_hist     :: Vector{Int}
end

CategoryStats() = CategoryStats(0, 0, 0, 0, 0, Float32[], 0.0, (0, 0, 0, 0), (NaN, NaN),
                                0, 0, zeros(Int, length(ULP_BUCKETS)))

# Distance in representable Float32 values; +0 and -0 coincide.
@inline function _ordered(x::Float32)
    i = Int64(reinterpret(Int32, x))
    return i >= 0 ? i : Int64(typemin(Int32)) - i
end
@inline _ulp_distance(a::Float32, b::Float32) = abs(_ordered(a) - _ordered(b))

# Stored bits are only compared within one stored type: values stored in
# different types never count as bitwise equal (and the fraction is then
# reported as unavailable, see `_category_summary`).
@inline _bits_equal(a::T, b::T) where T <: AbstractFloat = reinterpret(Unsigned, a) == reinterpret(Unsigned, b)
@inline _bits_equal(::AbstractFloat, ::AbstractFloat) = false

# The reference value as stored had it been written with the scale factor.
@inline _scaled(x::T, factor::Float64) where T <: AbstractFloat = T(Float64(x) * factor)

@inline function _accumulate!(s::CategoryStats, a, b, loc::NTuple{4, Int})
    s.n += 1
    if !(isfinite(a) && isfinite(b))
        s.n_nonfinite += 1
        return s
    end
    _bits_equal(a, b) && (s.n_bitwise += 1)
    d = _ulp_distance(Float32(a), Float32(b))
    d > s.ulp_max && (s.ulp_max = d)
    for (q, (lo, hi)) in enumerate(ULP_BUCKETS)
        if lo <= d <= hi
            s.ulp_hist[q] += 1
            break
        end
    end
    a64, b64 = Float64(a), Float64(b)
    m = max(abs(a64), abs(b64))
    if m == 0
        s.n_both_zero += 1
        return s
    end
    r = abs(b64 - a64) / m
    (a64 == 0 || b64 == 0) && (s.n_one_zero += 1)
    if r == 0
        s.n_equal += 1
    else
        push!(s.rel, Float32(r))
    end
    if r > s.max_rel
        s.max_rel = r
        s.argmax = loc
        s.argmax_ab = (a64, b64)
    end
    return s
end

# Counts, maximum and ulp statistics of several categories; the relative
# differences are not copied (`rel` stays empty, see `_cell_summaries`).
function _merge_counts(stats::AbstractVector{CategoryStats})
    out = CategoryStats()
    for s in stats
        out.n += s.n; out.n_nonfinite += s.n_nonfinite
        out.n_both_zero += s.n_both_zero; out.n_one_zero += s.n_one_zero; out.n_equal += s.n_equal
        if s.max_rel > out.max_rel
            out.max_rel, out.argmax, out.argmax_ab = s.max_rel, s.argmax, s.argmax_ab
        end
        out.n_bitwise += s.n_bitwise
        out.ulp_max = max(out.ulp_max, s.ulp_max)
        out.ulp_hist .+= s.ulp_hist
    end
    return out
end

"""
    kth_smallest(sorted_vectors, k) -> Float32

The `k`-th smallest value (1-based) of the union of ascending vectors of
non-negative Float32 values, without merging them: a bisection on the bit
patterns (which order non-negative Float32 values as their values) for the
smallest value with at least `k` values at or below it.
"""
function kth_smallest(sorted_vectors::AbstractVector{<:AbstractVector{Float32}}, k::Integer)
    total = sum(length, sorted_vectors; init = 0)
    1 <= k <= total || throw(BoundsError(1:total, k))
    nonempty = [v for v in sorted_vectors if !isempty(v)]
    length(nonempty) == 1 && return nonempty[1][k]
    lo, hi = UInt32(0), maximum(v -> reinterpret(UInt32, last(v)), nonempty)
    while lo < hi
        mid = lo + (hi - lo) >> 1
        at_or_below = sum(v -> searchsortedlast(v, reinterpret(Float32, mid)), nonempty)
        at_or_below >= k ? (hi = mid) : (lo = mid + UInt32(1))
    end
    return reinterpret(Float32, lo)
end

# Quantile (type 7, as `Statistics.quantile`) of the sample made of `nzero`
# zeros and the values of the ascending vectors `sorted_vectors`; `nothing`
# for an empty sample.
function _quantile_with_zeros(sorted_vectors::AbstractVector{<:AbstractVector{Float32}},
                              nzero::Integer, p::Real)
    N = nzero + sum(length, sorted_vectors; init = 0)
    N == 0 && return nothing
    value(k) = k <= nzero ? 0.0 : Float64(kth_smallest(sorted_vectors, k - nzero))
    h = (N - 1) * p + 1
    lo = floor(Int, h)
    hi = min(lo + 1, N)
    return value(lo) + (h - lo) * (value(hi) - value(lo))
end

# Summary of the union of the categories `parts`, whose `rel` vectors are
# sorted. `same_type`: both variables have the same stored type, so stored
# bits can be compared.
function _category_summary(parts::AbstractVector{CategoryStats}, times::Vector{DateTime}, same_type::Bool)
    s = _merge_counts(parts)
    rels = [part.rel for part in parts]
    finite = s.n - s.n_nonfinite
    q(p) = _quantile_with_zeros(rels, s.n_equal, p)
    worst = s.argmax[3] == 0 ? nothing :
            (; i = s.argmax[1], j = s.argmax[2], panel = s.argmax[3],
               time = s.argmax[4] == 0 ? nothing : _stamp(times[s.argmax[4]]),
               reference = s.argmax_ab[1], candidate = s.argmax_ab[2])
    return (; n_values = s.n, n_nonfinite = s.n_nonfinite, n_both_zero = s.n_both_zero,
              n_one_zero = s.n_one_zero, n_sample = finite - s.n_both_zero,
              p50 = q(0.5), p99 = q(0.99),
              max = finite - s.n_both_zero > 0 ? s.max_rel : nothing, worst,
              bitwise_equal_fraction = finite > 0 && same_type ? s.n_bitwise / finite : nothing,
              ulp_max = finite > 0 ? s.ulp_max : nothing,
              within_1ulp_fraction = finite > 0 ? (s.ulp_hist[1] + s.ulp_hist[2]) / finite : nothing,
              ulp_histogram = [(; ulps = _bucket_label(b), count = c) for (b, c) in zip(ULP_BUCKETS, s.ulp_hist)])
end

# (all, per category indexed by code): each category's relative differences
# are sorted in place (QuickSort needs no scratch buffer); the quantiles over
# all cells are selected from the sorted category vectors.
function _cell_summaries(stats::AbstractVector{CategoryStats}, times::Vector{DateTime}, same_type::Bool)
    foreach(s -> sort!(s.rel; alg = QuickSort), stats)
    per_category = [_category_summary([s], times, same_type) for s in stats]
    return _category_summary(stats, times, same_type), per_category
end

# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_units(v) = String(strip(string(get(v.attrib, "units", ""))))

# :series for (Xdim, Ydim, nf, time) variables, :static for (Xdim, Ydim, nf);
# any other dimension names or order is an error.
function _variable_kind(ds, name, path)
    haskey(ds, name) || throw(ArgumentError("variable `$(name)` not found in $(path)"))
    dims = Tuple(dimnames(ds[name]))
    dims == CS_NATIVE_FLUX_DIMS && return :series
    dims == CS_NATIVE_STATIC_DIMS && return :static
    throw(DimensionMismatch("`$(name)` in $(path) has dimensions $(dims); expected the cs_native layout " *
                            "$(CS_NATIVE_FLUX_DIMS) or $(CS_NATIVE_STATIC_DIMS) (Julia order)"))
end

function _variable_kind(ds_a, ds_b, pair::Pair{String, String}, path_a, path_b)
    ka = _variable_kind(ds_a, pair.first, path_a)
    kb = _variable_kind(ds_b, pair.second, path_b)
    ka == kb || throw(DimensionMismatch(
        "`$(pair.first)` in $(path_a) is a $(ka) variable but `$(pair.second)` in $(path_b) is a $(kb) variable"))
    return ka
end

function _check_storage(v, name, path)
    (haskey(v.attrib, "scale_factor") || haskey(v.attrib, "add_offset")) && throw(ArgumentError(
        "`$(name)` in $(path) is packed (scale_factor/add_offset); cs_native fields are stored unpacked"))
    eltype(v.var) <: AbstractFloat || throw(ArgumentError(
        "`$(name)` in $(path) is stored as $(eltype(v.var)); expected floating point"))
    return v
end

function _check_flux_variable(ds, name, path, Nc, ntime)
    _variable_kind(ds, name, path) === :series || throw(DimensionMismatch(
        "`$(name)` in $(path) has dimensions $(dimnames(ds[name])); expected $(CS_NATIVE_FLUX_DIMS)"))
    v = ds[name]
    size(v) == (Nc, Nc, CS_PANELS, ntime) || throw(DimensionMismatch(
        "`$(name)` in $(path) has size $(size(v)); expected ($(Nc), $(Nc), $(CS_PANELS), $(ntime)) " *
        "= (Xdim, Ydim, nf, time) for the C$(Nc) mesh"))
    return _check_storage(v, name, path)
end

function _check_static_variable(ds, name, path, Nc)
    _variable_kind(ds, name, path) === :static || throw(DimensionMismatch(
        "`$(name)` in $(path) has dimensions $(dimnames(ds[name])); expected $(CS_NATIVE_STATIC_DIMS)"))
    v = ds[name]
    size(v) == (Nc, Nc, CS_PANELS) || throw(DimensionMismatch(
        "`$(name)` in $(path) has size $(size(v)); expected ($(Nc), $(Nc), $(CS_PANELS)) " *
        "= (Xdim, Ydim, nf) for the C$(Nc) mesh"))
    return _check_storage(v, name, path)
end

function _fill_values(v)
    T = eltype(v.var)
    vals = T[]
    for key in ("_FillValue", "missing_value")
        haskey(v.attrib, key) || continue
        for x in v.attrib[key]
            push!(vals, T(x))
        end
    end
    return vals
end

# Whether `x` is one of the stored fill values, by the rule of NCDatasets
# (`CommonDataModel.isfillvalue`): a NaN fill value matches any NaN, any other
# fill value matches by ==.
@inline _is_fill_value(x, fill) = isnan(fill) ? isnan(x) : x == fill
@inline _is_fill(x, fills) = any(f -> _is_fill_value(x, f), fills)

function _time_chunk(v)
    storage, chunks = NCDatasets.chunking(v)
    return storage === :chunked ? Int(chunks[4]) : 1
end

function _block_length(va, vb, Nc, n, override)
    override === nothing || return clamp(override, 1, max(n, 1))
    chunk = max(_time_chunk(va), _time_chunk(vb))
    slice_bytes = Nc * Nc * CS_PANELS * max(sizeof(eltype(va.var)), sizeof(eltype(vb.var)))
    per_budget = max(1, DEFAULT_BLOCK_BYTES ÷ slice_bytes)
    return clamp(chunk * max(1, per_budget ÷ chunk), 1, max(n, 1))
end

# Raw stored values of the ascending slice indices `idx`.
function _read_slices(v, idx::AbstractVector{Int})
    lo, hi = first(idx), last(idx)
    hi - lo + 1 == length(idx) && return v.var[:, :, :, lo:hi]
    out = Array{eltype(v.var)}(undef, size(v, 1), size(v, 2), size(v, 3), length(idx))
    for (n, k) in enumerate(idx)
        out[:, :, :, n] = v.var[:, :, :, k]
    end
    return out
end

function _file_area_deviation(ds, mesh)
    haskey(ds, "cell_area") || return nothing
    a = Float64.(Array(ds["cell_area"].var))
    Nc = mesh.Nc
    size(a) in ((Nc, Nc), (Nc, Nc, CS_PANELS)) || return (; shape = collect(size(a)), max_rel_diff = nothing)
    ref = Float64.(mesh.cell_areas)
    dev = maximum(abs(a[i, j, p] / ref[i, j] - 1) for p in 1:size(a, 3), j in 1:Nc, i in 1:Nc)
    return (; shape = collect(size(a)), max_rel_diff = dev)
end

# Largest angle of the stored `lats`/`lons` from the mesh centres (`nothing`
# without stored centres); beyond `centre_deviation_limit_deg` the file is on
# another grid or panel convention, an error.
function _file_centre_deviation(ds, path, mesh; what = "file")
    deviation = centre_deviation_deg(ds, mesh)
    limit = centre_deviation_limit_deg(mesh.Nc)
    deviation === nothing || deviation <= limit || throw(ArgumentError(
        "the cell centres (`lats`/`lons`) of $(what) $(path) are up to $(deviation)° from the C$(mesh.Nc) " *
        "mesh centres (limit $(limit)°): the $(what) is on another grid or panel convention"))
    return deviation
end

"""
    read_mask(path, name, mesh) -> (mask, centre_deviation_deg)

The coastal-class mask: variable `name` of `path` in the cs_native layout,
(Xdim, Ydim, nf) or (Xdim, Ydim, nf, time) with one slice, as Float64. Fill,
packed or non-finite values are rejected, and so is a file that stores only
one of `lats` and `lons`, or centres more than `centre_deviation_limit_deg(Nc)`
from the mesh cell centres.
"""
function read_mask(path, name, mesh::CubedSphereMesh)
    Nc = mesh.Nc
    NCDataset(path) do ds
        haskey(ds, name) || throw(ArgumentError("mask variable `$(name)` not found in $(path)"))
        v = ds[name]
        dims = Tuple(dimnames(v))
        layout_ok = (dims == CS_NATIVE_STATIC_DIMS && size(v) == (Nc, Nc, CS_PANELS)) ||
                    (dims == CS_NATIVE_FLUX_DIMS && size(v) == (Nc, Nc, CS_PANELS, 1))
        layout_ok || throw(DimensionMismatch(
            "mask `$(name)` in $(path) has dimensions $(dims) and size $(size(v)); expected the cs_native " *
            "layout $(CS_NATIVE_STATIC_DIMS) of C$(Nc) ($(Nc), $(Nc), $(CS_PANELS)), or $(CS_NATIVE_FLUX_DIMS) with one slice"))
        (haskey(v.attrib, "scale_factor") || haskey(v.attrib, "add_offset")) && throw(ArgumentError(
            "mask `$(name)` in $(path) is packed (scale_factor/add_offset)"))
        raw = Array(v.var)
        ndims(raw) == 4 && (raw = raw[:, :, :, 1])
        fills = _fill_values(v)
        any(x -> _is_fill(x, fills), raw) && throw(ArgumentError("mask `$(name)` in $(path) has fill values"))
        mask = Float64.(raw)
        all(isfinite, mask) || throw(ArgumentError("mask `$(name)` in $(path) has non-finite values"))
        deviation = _file_centre_deviation(ds, path, mesh; what = "mask file")
        return mask, deviation
    end
end

# ---------------------------------------------------------------------------
# Comparison of one variable
# ---------------------------------------------------------------------------

_signed_rel(a, b) = a == b ? 0.0 : a == 0 ? copysign(Inf, b) : (b - a) / abs(a)
_max_abs(x) = isempty(x) ? nothing : maximum(abs, x)
_factor(scale) = scale === nothing ? 1.0 : Float64(scale.factor)

# Running sums of one variable over the compared slices.
mutable struct VariableAccumulator{TA, TB}
    stats          :: Vector{CategoryStats}   # indexed by category code
    fills_a        :: Vector{TA}              # stored fill values of each file
    fills_b        :: Vector{TB}
    n_fill_a       :: Int
    n_fill_b       :: Int
    wa             :: Array{Float64, 3}       # density × area of the current slice
    wb             :: Array{Float64, 3}
    total_a        :: Vector{Float64}         # global total per compared slice
    total_b        :: Vector{Float64}
    band_mean_a    :: Vector{Float64}         # Σ_k band total × slice weight
    band_mean_b    :: Vector{Float64}
    band_slice_max :: Vector{Float64}         # max_k |relative difference| per band
end

VariableAccumulator(fills_a::Vector{TA}, fills_b::Vector{TB}, Nc::Int, n::Int) where {TA, TB} =
    VariableAccumulator{TA, TB}([CategoryStats() for _ in CATEGORY_NAMES], fills_a, fills_b, 0, 0,
                                zeros(Nc, Nc, CS_PANELS), zeros(Nc, Nc, CS_PANELS), zeros(n), zeros(n),
                                zeros(N_SIN_LAT_BANDS), zeros(N_SIN_LAT_BANDS), zeros(N_SIN_LAT_BANDS))

function _process_block!(acc::VariableAccumulator, A::AbstractArray{<:Real, 4},
                         B::AbstractArray{<:Real, 4}, k0::Int, ctx, factor::Float64)
    Nc = size(A, 1)
    area, codes, band = ctx.area, ctx.codes, ctx.band
    nbands = length(acc.band_mean_a)
    band_a = zeros(nbands)
    band_b = zeros(nbands)
    for kk in 1:size(A, 4)
        k = k0 + kk - 1
        fill!(band_a, 0.0); fill!(band_b, 0.0)
        @inbounds for p in 1:CS_PANELS, j in 1:Nc, i in 1:Nc
            stored = A[i, j, p, kk]; b = B[i, j, p, kk]
            acc.n_fill_a += _is_fill(stored, acc.fills_a)
            acc.n_fill_b += _is_fill(b, acc.fills_b)
            a = _scaled(stored, factor)
            wa = Float64(a) * area[i, j]
            wb = Float64(b) * area[i, j]
            acc.wa[i, j, p] = wa
            acc.wb[i, j, p] = wb
            band_a[band[i, j, p]] += wa
            band_b[band[i, j, p]] += wb
            _accumulate!(acc.stats[codes[i, j, p]], a, b, (i, j, p, k))
        end
        acc.total_a[k] = sum(acc.wa)       # pairwise summation
        acc.total_b[k] = sum(acc.wb)
        @inbounds for q in 1:nbands
            acc.band_mean_a[q] += band_a[q] * ctx.weight_a[k]
            acc.band_mean_b[q] += band_b[q] * ctx.weight_b[k]
            r = abs(_signed_rel(band_a[q], band_b[q]))
            (r > acc.band_slice_max[q] || isnan(r)) && (acc.band_slice_max[q] = r)
        end
    end
    return acc
end

# Mass per calendar month (and covered seconds) of the per-slice rates.
function _month_masses(totals, overlaps)
    mass = Dict{Tuple{Int, Int}, Float64}()
    covered = Dict{Tuple{Int, Int}, Float64}()
    for (k, parts) in enumerate(overlaps), (key, seconds) in parts
        mass[key] = get(mass, key, 0.0) + totals[k] * seconds
        covered[key] = get(covered, key, 0.0) + seconds
    end
    return mass, covered
end

function _period_totals(total_a, total_b, ctx)
    (ctx.overlaps_a === nothing || ctx.overlaps_b === nothing) && return nothing, nothing
    ma, ca = _month_masses(total_a, ctx.overlaps_a)
    mb, cb = _month_masses(total_b, ctx.overlaps_b)
    months = sort!(collect(union(keys(ma), keys(mb))))
    month_rows = map(months) do key
        a, b = get(ma, key, 0.0), get(mb, key, 0.0)
        complete = get(ca, key, 0.0) == _month_seconds(key) && get(cb, key, 0.0) == _month_seconds(key)
        (; month = _month_label(key), reference = a, candidate = b, rel_diff = _signed_rel(a, b), complete)
    end
    years = sort!(unique(first.(months)))
    year_rows = map(years) do y
        a = sum((get(ma, key, 0.0) for key in months if first(key) == y); init = 0.0)
        b = sum((get(mb, key, 0.0) for key in months if first(key) == y); init = 0.0)
        cov_a = sum((get(ca, key, 0.0) for key in months if first(key) == y); init = 0.0)
        cov_b = sum((get(cb, key, 0.0) for key in months if first(key) == y); init = 0.0)
        complete = cov_a == _year_seconds(y) && cov_b == _year_seconds(y)
        (; year = y, reference = a, candidate = b, rel_diff = _signed_rel(a, b), complete)
    end
    return month_rows, year_rows
end

function _variable_header(va, vb, name_a, name_b, kind, scale)
    units_a, units_b = _units(va), _units(vb)
    ta, tb = eltype(va.var), eltype(vb.var)
    return (; kind = String(kind), reference_variable = name_a, candidate_variable = name_b,
              units = (; reference = units_a, candidate = units_b, equal = units_a == units_b),
              stored_types = (; reference = string(ta), candidate = string(tb), equal = ta == tb),
              reference_scale = scale)
end

function compare_variable(ds_a, ds_b, pair::Pair{String, String}, ctx; block_slices = nothing,
                          scale = nothing)
    name_a, name_b = pair
    Nc = ctx.Nc
    va = _check_flux_variable(ds_a, name_a, ctx.path_a, Nc, ctx.ntime_a)
    vb = _check_flux_variable(ds_b, name_b, ctx.path_b, Nc, ctx.ntime_b)
    header = _variable_header(va, vb, name_a, name_b, :series, scale)
    factor = _factor(scale)

    n = length(ctx.idx_a)
    acc = VariableAccumulator(_fill_values(va), _fill_values(vb), Nc, n)
    block = _block_length(va, vb, Nc, n, block_slices)
    @info "Comparing $(name_a) / $(name_b): $(n) slices in blocks of $(block)" *
          (scale === nothing ? "" : "; reference × $(factor)")
    for k0 in 1:block:n
        k1 = min(n, k0 + block - 1)
        A = _read_slices(va, ctx.idx_a[k0:k1])
        B = _read_slices(vb, ctx.idx_b[k0:k1])
        _process_block!(acc, A, B, k0, ctx, factor)
    end

    weight_sum_a, weight_sum_b = sum(ctx.weight_a), sum(ctx.weight_b)
    band_a = acc.band_mean_a ./ weight_sum_a
    band_b = acc.band_mean_b ./ weight_sum_b
    band_scale = maximum(abs, band_a)
    band_errors = abs.(band_b .- band_a)
    band_metric = band_scale > 0 ? maximum(band_errors) / band_scale :
                  all(iszero, band_errors) ? 0.0 : Inf
    slice_rel = _signed_rel.(acc.total_a, acc.total_b)
    worst_slice = isempty(slice_rel) ? nothing : argmax(abs.(slice_rel))
    month_rows, year_rows = _period_totals(acc.total_a, acc.total_b, ctx)

    all_summary, per_category = _cell_summaries(acc.stats, ctx.times, header.stored_types.equal)
    nonfinite = all_summary.n_nonfinite
    rate_units = "($(header.units.reference)) m2"
    return merge(header,
                 (; invalid = (; nonfinite_pairs = nonfinite, fill_values_reference = acc.n_fill_a,
                                 fill_values_candidate = acc.n_fill_b),
                    slice_totals = (; units = rate_units,
                                      max_abs_rel_diff = _max_abs(slice_rel),
                                      worst_time = worst_slice === nothing ? nothing : _stamp(ctx.times[worst_slice]),
                                      reference = acc.total_a, candidate = acc.total_b, rel_diff = slice_rel),
                    month_totals = (; units = "$(rate_units) s",
                                      max_abs_rel_diff = month_rows === nothing ? nothing : _max_abs(getfield.(month_rows, :rel_diff)),
                                      months = month_rows),
                    year_totals = (; units = "$(rate_units) s",
                                     max_abs_rel_diff = year_rows === nothing ? nothing : _max_abs(getfield.(year_rows, :rel_diff)),
                                     years = year_rows),
                    bands = (; n_bands = N_SIN_LAT_BANDS, units = rate_units,
                               definition = "time-mean rate (weighted by slice duration) per band uniform in sin(latitude), band 1 at the south pole",
                               max_abs_diff_over_max_band = band_metric,
                               n_cells = ctx.band_cells,
                               reference = band_a, candidate = band_b,
                               rel_diff = _signed_rel.(band_a, band_b),
                               max_slice_abs_rel_diff = acc.band_slice_max),
                    cells = (; definition = "|b - a| / max(|a|, |b|) over cells and slices; both-zero values left out",
                               all = all_summary,
                               interior = per_category[INTERIOR], coastal = per_category[COASTAL],
                               polar = per_category[POLAR])))
end

"""
    compare_static_variable(ds_a, ds_b, name_in_reference => name_in_candidate, ctx; scale = nothing)

Cell-by-cell comparison of an (Xdim, Ydim, nf) variable: relative differences
by category, Float32 ulp distance, bitwise equality (same stored type only),
fill and non-finite values.
"""
function compare_static_variable(ds_a, ds_b, pair::Pair{String, String}, ctx; scale = nothing)
    name_a, name_b = pair
    Nc = ctx.Nc
    va = _check_static_variable(ds_a, name_a, ctx.path_a, Nc)
    vb = _check_static_variable(ds_b, name_b, ctx.path_b, Nc)
    header = _variable_header(va, vb, name_a, name_b, :static, scale)
    factor = _factor(scale)
    A = Array(va.var)
    B = Array(vb.var)
    fills_a, fills_b = _fill_values(va), _fill_values(vb)
    n_fill_a = count(x -> _is_fill(x, fills_a), A)
    n_fill_b = count(x -> _is_fill(x, fills_b), B)
    stats = [CategoryStats() for _ in CATEGORY_NAMES]
    @inbounds for p in 1:CS_PANELS, j in 1:Nc, i in 1:Nc
        _accumulate!(stats[ctx.codes[i, j, p]], _scaled(A[i, j, p], factor), B[i, j, p], (i, j, p, 0))
    end
    all_summary, per_category = _cell_summaries(stats, DateTime[], header.stored_types.equal)
    return merge(header,
                 (; invalid = (; nonfinite_pairs = all_summary.n_nonfinite, fill_values_reference = n_fill_a,
                                 fill_values_candidate = n_fill_b),
                    slice_totals = nothing, month_totals = nothing, year_totals = nothing, bands = nothing,
                    cells = (; definition = "|b - a| / max(|a|, |b|) over cells; both-zero values left out",
                               all = all_summary,
                               interior = per_category[INTERIOR], coastal = per_category[COASTAL],
                               polar = per_category[POLAR])))
end

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

# (; value, na, unavailable): `na` explains why the check passes as n/a (an
# empty sample); `unavailable` why a requested metric could not be computed.
function _metric_value(vr, key::AbstractString)
    name, category = occursin('.', key) ? split(key, '.'; limit = 2) : (key, "all")
    if name in TOTAL_METRICS
        vr.kind == "static" && return (; value = nothing, unavailable = nothing,
            na = "not applicable: `$(vr.reference_variable)` has no time dimension")
        value = name == "slice_total" ? vr.slice_totals.max_abs_rel_diff :
                name == "month_total" ? vr.month_totals.max_abs_rel_diff :
                name == "year_total"  ? vr.year_totals.max_abs_rel_diff :
                                        vr.bands.max_abs_diff_over_max_band
        return (; value, na = nothing,
                  unavailable = value === nothing ? "slice durations unknown (one slice without time bounds)" : nothing)
    end
    cells = getproperty(vr.cells, Symbol(category))
    cells.n_values == 0 && return (; value = nothing, unavailable = nothing,
                                     na = "no cells in category `$(category)`")
    if name in RELATIVE_METRICS
        value = name == "cell_p50" ? cells.p50 : name == "cell_p99" ? cells.p99 : cells.max
        value === nothing && cells.n_nonfinite == 0 && return (; value, unavailable = nothing,
            na = "no informative values in `$(category)`: every value is zero in both files")
        return (; value, na = nothing,
                  unavailable = value === nothing ? "no finite values in `$(category)`" : nothing)
    end
    name in ("ulp_max", "bitwise_fraction") || throw(ArgumentError("unknown metric $(key)"))
    value = name == "ulp_max" ? cells.ulp_max : cells.bitwise_equal_fraction
    unavailable = value !== nothing ? nothing :
                  cells.n_values == cells.n_nonfinite ? "no finite values in `$(category)`" :
                  "stored types differ ($(vr.stored_types.reference) vs $(vr.stored_types.candidate)): " *
                  "no stored bits to compare"
    return (; value, na = nothing, unavailable)
end

function _check(metric, m, tolerance, bound)
    sign = bound === :upper ? "<=" : ">="
    if m.value === nothing && m.na !== nothing
        return (; metric, value = nothing, tolerance, bound = sign, status = "n/a", pass = true, note = m.na)
    end
    pass = m.value !== nothing && (bound === :upper ? m.value <= tolerance : m.value >= tolerance)
    return (; metric, value = m.value, tolerance, bound = sign, status = pass ? "pass" : "fail", pass,
              note = m.unavailable)
end

function _variable_checks(vr, tolerances)
    checks = Any[]
    bad = vr.invalid.nonfinite_pairs + vr.invalid.fill_values_reference + vr.invalid.fill_values_candidate
    push!(checks, (; metric = "finite", value = bad, tolerance = 0, bound = "<=",
                     status = bad == 0 ? "pass" : "fail", pass = bad == 0, note = nothing))
    for (key, tol) in tolerances
        bound = TOLERANCE_METRICS[Symbol(first(split(key, '.')))][1]
        push!(checks, _check(key, _metric_value(vr, key), tol, bound))
    end
    return checks
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    compare_cs_flux_files(reference, candidate, mesh; variables = [], reference_scale = [],
                          allow_unit_mismatch = false, mask_var = nothing, mask_file = nothing,
                          mask_threshold = 0.0, common_times = false,
                          common_variables_only = false, tolerances = [],
                          block_slices = nothing, mesh_info = nothing) -> NamedTuple

Compare two `cs_native` flux files on `mesh` (see the file header for every
metric and rule). `variables` holds names or `name_in_reference =>
name_in_candidate` pairs of (Xdim, Ydim, nf, time) or (Xdim, Ydim, nf)
variables (default: every (Xdim, Ydim, nf, time) series of either file);
`reference_scale` holds `name_in_reference => factor` pairs, a factor being a
number or a preset name of `REFERENCE_SCALE_PRESETS`; `tolerances` holds
`metric => value` pairs. Throws for variables or stored cell centres outside
the cs_native layout of `mesh`, a series without `units`, unequal `units`
(unless `allow_unit_mismatch`) and time bounds that are not start-stamped and
contiguous. The coastal class comes from `mask_var` in `mask_file` (default:
the reference file). The result's `pass` is `true` when the time axes agree (or
share stamps, with `common_times`; no time axis is read when only (Xdim, Ydim,
nf) variables are compared), no series of the default selection is in only one
file (unless `common_variables_only`), no value is non-finite or a fill value,
and every tolerance holds (or is n/a) for every variable.
"""
function compare_cs_flux_files(reference::AbstractString, candidate::AbstractString,
                               mesh::CubedSphereMesh; variables = Pair{String, String}[],
                               reference_scale = Pair{String, Any}[], allow_unit_mismatch::Bool = false,
                               mask_var = nothing, mask_file = nothing, mask_threshold::Real = 0.0,
                               common_times::Bool = false, common_variables_only::Bool = false,
                               tolerances = Pair{String, Float64}[], block_slices = nothing,
                               mesh_info = nothing)
    tolerances = validate_tolerances(tolerances)
    mask_var === nothing && mask_file !== nothing && throw(ArgumentError("mask_file needs mask_var"))
    mask_var === nothing && any(t -> endswith(t.first, ".coastal"), tolerances) &&
        throw(ArgumentError("coastal tolerances need a mask (mask_var)"))
    isfinite(mask_threshold) || throw(ArgumentError("mask_threshold must be finite, got $(mask_threshold)"))
    requested = [_variable_pair(v) for v in variables]
    common_variables_only && !isempty(requested) && throw(ArgumentError(
        "common_variables_only goes with the default variable selection; with `variables` the selection is explicit"))
    scales = _reference_scales(reference_scale)
    Nc = mesh.Nc
    mask_path = mask_var === nothing ? nothing : abspath(String(something(mask_file, reference)))
    mask_values, mask_deviation = mask_var === nothing ? (nothing, nothing) :
                                  read_mask(mask_path, String(mask_var), mesh)
    codes, polar_panels = cell_categories(mesh; mask = mask_values, land_threshold = mask_threshold)
    band = sin_latitude_bands(mesh)
    area = Float64.(mesh.cell_areas)

    NCDataset(reference) do ds_a
        NCDataset(candidate) do ds_b
            centre_a = _file_centre_deviation(ds_a, reference, mesh)
            centre_b = _file_centre_deviation(ds_b, candidate, mesh)
            variable_pairs, series_only_a, series_only_b = isempty(requested) ?
                _series_selection(ds_a, ds_b, reference, candidate) : (requested, String[], String[])
            selection_pass = common_variables_only || (isempty(series_only_a) && isempty(series_only_b))
            listed(names) = isempty(names) ? "none" : join(names, ", ")
            selection_pass || @warn "series in only one file (reference: $(listed(series_only_a)); " *
                                    "candidate: $(listed(series_only_b))); the comparison fails"
            kinds = [_variable_kind(ds_a, ds_b, pair, reference, candidate) for pair in variable_pairs]
            for name in keys(scales)
                any(p -> p.first == name, variable_pairs) || throw(ArgumentError(
                    "reference scale for `$(name)`, which is not a compared reference variable " *
                    "($(join(first.(variable_pairs), ", ")))"))
            end
            for ((name_a, name_b), kind) in zip(variable_pairs, kinds)
                ua, ub = _units(ds_a[name_a]), _units(ds_b[name_b])
                if kind === :series
                    for (u, name, path) in ((ua, name_a, reference), (ub, name_b, candidate))
                        isempty(u) && throw(ArgumentError(
                            "`$(name)` in $(path) has no `units` attribute; a flux series must declare its units"))
                    end
                end
                ua == ub && continue
                allow_unit_mismatch || throw(ArgumentError(
                    "units differ: `$(name_a)` in $(reference) is $(repr(ua)), `$(name_b)` in $(candidate) " *
                    "is $(repr(ub)); allow the comparison as stored with --allow-unit-mismatch " *
                    "(allow_unit_mismatch = true)"))
                @warn "units differ for $(name_a) / $(name_b): $(repr(ua)) vs $(repr(ub)); compared as stored" *
                      (haskey(scales, name_a) ? " × the reference scale" : "")
            end

            needs_time = any(==(:series), kinds)
            axis_a = needs_time ? read_time_axis(ds_a, reference) : nothing
            axis_b = needs_time ? read_time_axis(ds_b, candidate) : nothing
            common, end_differs = needs_time ? _match_intervals(axis_a, axis_b) : (DateTime[], [])
            only_a = needs_time ? setdiff(axis_a.times, axis_b.times) : DateTime[]
            only_b = needs_time ? setdiff(axis_b.times, axis_a.times) : DateTime[]
            identical = needs_time && axis_a.times == axis_b.times && isempty(end_differs)
            time_pass = !needs_time || identical || (common_times && !isempty(common))
            needs_time && !identical &&
                @warn "time axes differ: $(length(only_a)) stamps only in the reference, " *
                      "$(length(only_b)) only in the candidate, $(length(end_differs)) common stamps " *
                      "whose slices end at different times; comparing $(length(common)) matched slices"

            ctx = (; Nc, area, codes, band, times = common, path_a = reference, path_b = candidate)
            if needs_time
                pos_a = Dict(t => k for (k, t) in enumerate(axis_a.times))
                pos_b = Dict(t => k for (k, t) in enumerate(axis_b.times))
                idx_a = [pos_a[t] for t in common]
                idx_b = [pos_b[t] for t in common]
                durations(axis, idx) = axis.ends === nothing ? nothing :
                    [_seconds(axis.ends[k] - axis.times[k]) for k in idx]
                overlaps(axis, idx) = axis.ends === nothing ? nothing :
                    [_month_overlaps(axis.times[k], axis.ends[k]) for k in idx]
                dur_a, dur_b = durations(axis_a, idx_a), durations(axis_b, idx_b)
                ctx = merge(ctx, (; idx_a, idx_b,
                                    ntime_a = length(axis_a.times), ntime_b = length(axis_b.times),
                                    weight_a = something(dur_a, ones(length(common))),
                                    weight_b = something(dur_b, ones(length(common))),
                                    overlaps_a = overlaps(axis_a, idx_a), overlaps_b = overlaps(axis_b, idx_b),
                                    band_cells = [count(==(q), band) for q in 1:N_SIN_LAT_BANDS]))
            end

            results = Any[]
            for (pair, kind) in zip(variable_pairs, kinds)
                scale = get(scales, pair.first, nothing)
                if kind === :static
                    vr = compare_static_variable(ds_a, ds_b, pair, ctx; scale)
                elseif !isempty(common)
                    vr = compare_variable(ds_a, ds_b, pair, ctx; block_slices, scale)
                else
                    continue
                end
                checks = _variable_checks(vr, tolerances)
                push!(results, merge(vr, (; checks, pass = all(c -> c.pass, checks))))
            end
            pass = time_pass && selection_pass && !isempty(results) && all(r -> r.pass, results)
            stamps(ts) = [_stamp(t) for t in ts[1:min(end, 20)]]
            file_info(path, ds, axis, centre) = (; path = abspath(path),
                time_units = axis === nothing ? nothing : axis.units,
                calendar = axis === nothing ? nothing : axis.calendar,
                n_slices = axis === nothing ? nothing : length(axis.times),
                slice_ends = axis === nothing ? nothing : axis.end_rule,
                cell_area_vs_mesh = _file_area_deviation(ds, mesh),
                cell_centre_vs_mesh_deg = centre)
            return (; tool = "scripts/validation/compare_cs_flux_files.jl",
                      created = _stamp(now(UTC)), pass,
                      reference = file_info(reference, ds_a, axis_a, centre_a),
                      candidate = file_info(candidate, ds_b, axis_b, centre_b),
                      mesh = (; source = mesh_info === nothing ? "argument" : mesh_info.source,
                                path = mesh_info === nothing ? nothing : mesh_info.path,
                                Nc, radius_m = Float64(mesh.radius),
                                convention = string(nameof(typeof(mesh.convention))),
                                total_area_m2 = CS_PANELS * sum(area)),
                      time = needs_time ?
                             (; identical, common_times_mode = common_times, pass = time_pass,
                                n_compared = length(common),
                                first = isempty(common) ? nothing : _stamp(first(common)),
                                last = isempty(common) ? nothing : _stamp(last(common)),
                                n_only_in_reference = length(only_a), n_only_in_candidate = length(only_b),
                                only_in_reference = stamps(only_a), only_in_candidate = stamps(only_b),
                                n_end_differs = length(end_differs),
                                end_differs = end_differs[1:min(end, 20)]) :
                             nothing,
                      categories = (; rule = "coastal (mask land class differs from an edge neighbour) > polar panels > interior",
                                      mask_var = mask_var === nothing ? nothing : String(mask_var),
                                      mask_file = mask_path,
                                      mask_threshold = mask_var === nothing ? nothing : Float64(mask_threshold),
                                      mask_cell_centre_vs_mesh_deg = mask_deviation,
                                      polar_panels,
                                      n_cells = (; interior = count(==(INTERIOR), codes),
                                                   coastal = count(==(COASTAL), codes),
                                                   polar = count(==(POLAR), codes))),
                      reference_scale_rule = isempty(scales) ? nothing : REFERENCE_SCALE_RULE,
                      tolerances = [(; metric = k, tolerance = v) for (k, v) in tolerances],
                      variable_selection = (; rule = isempty(requested) ?
                                                "every (Xdim, Ydim, nf, time) series of either file" :
                                                "named variables",
                                              common_variables_only,
                                              series_only_in_reference = series_only_a,
                                              series_only_in_candidate = series_only_b,
                                              pass = selection_pass),
                      variables = results)
        end
    end
end

_is_series(ds, name) = Tuple(dimnames(ds[name])) == CS_NATIVE_FLUX_DIMS

"""
    _series_selection(ds_a, ds_b, path_a, path_b) -> (pairs, only_in_a, only_in_b)

The default variable selection: the (Xdim, Ydim, nf, time) series of both
files (in the reference's order) and the sorted names of the series of only
one file. A name that is a series in one file and has other dimensions in the
other is a `DimensionMismatch`.
"""
function _series_selection(ds_a, ds_b, path_a, path_b)
    series_a = [String(n) for n in keys(ds_a) if _is_series(ds_a, n)]
    series_b = [String(n) for n in keys(ds_b) if _is_series(ds_b, n)]
    for (names, ds, path) in ((series_a, ds_b, path_b), (series_b, ds_a, path_a)), n in names
        haskey(ds, n) && !_is_series(ds, n) && throw(DimensionMismatch(
            "`$(n)` is a $(CS_NATIVE_FLUX_DIMS) series in one file but has dimensions $(dimnames(ds[n])) " *
            "in $(path); name the variables to compare with --var"))
    end
    common = [n for n in series_a if n in series_b]
    isempty(common) && throw(ArgumentError(
        "the files share no $(CS_NATIVE_FLUX_DIMS) series; name the variables to compare with --var"))
    return [n => n for n in common], sort!(setdiff(series_a, series_b)), sort!(setdiff(series_b, series_a))
end

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_jsonable(x::AbstractFloat) = isfinite(x) ? Float64(x) : string(x)
_jsonable(x::NamedTuple) = NamedTuple{keys(x)}(map(_jsonable, values(x)))
_jsonable(x::Union{AbstractVector, Tuple}) = [_jsonable(v) for v in x]
_jsonable(x::AbstractDict) = Dict(string(k) => _jsonable(v) for (k, v) in x)
_jsonable(x::Symbol) = String(x)
_jsonable(x) = x

# Write `path` through the package's shared atomic-publication contract.
function _write_atomically(f, path::AbstractString)
    mkpath(dirname(abspath(path)))
    atomic_publish(path) do tmp
        open(f, tmp, "w")
    end
    return path
end

write_report_json(path, report) = _write_atomically(io -> (JSON3.pretty(io, JSON3.write(_jsonable(report))); println(io)), path)

_fmt(x::Nothing) = "n/a"
_fmt(x::Integer) = string(x)
_fmt(x::Real) = isfinite(x) ? @sprintf("%.3e", x) : string(x)

_scale_label(scale::Nothing) = ""
_scale_label(scale) = " (reference × " * @sprintf("%.17g", scale.factor) *
                      (scale.preset === nothing ? "" : ", preset $(scale.preset)") * ")"

"""
    check_label(check) -> String

`ok`, `FAIL` or `n/a (reason)` for one entry of a variable's `checks`.
"""
check_label(chk) = chk.status == "n/a" ? "n/a ($(chk.note))" :
                   chk.pass ? "ok" : (chk.note === nothing ? "FAIL" : "FAIL ($(chk.note))")

"""
    summary_text(report) -> String

A short human-readable summary of a `compare_cs_flux_files` report.
"""
function summary_text(report)
    io = IOBuffer()
    println(io, "compare_cs_flux_files: ", report.pass ? "PASS" : "FAIL")
    slices(f) = f.n_slices === nothing ? "" : " ($(f.n_slices) slices)"
    println(io, "  reference: ", report.reference.path, slices(report.reference))
    println(io, "  candidate: ", report.candidate.path, slices(report.candidate))
    m = report.mesh
    println(io, "  mesh: C", m.Nc, " ", m.convention, " from ", m.source,
            m.path === nothing ? "" : " $(m.path)", ", radius ", m.radius_m, " m")
    t = report.time
    if t === nothing
        println(io, "  times: none read (only variables without a time dimension)")
    else
        println(io, "  times: ", t.n_compared, " compared",
                t.identical ? ", identical" : ", $(t.n_only_in_reference) only in reference, $(t.n_only_in_candidate) only in candidate" *
                                              (t.n_end_differs == 0 ? "" : ", $(t.n_end_differs) with other slice ends"),
                t.n_compared > 0 ? " ($(t.first) .. $(t.last))" : "", t.pass ? "" : "  FAIL")
    end
    sel = report.variable_selection
    if !isempty(sel.series_only_in_reference) || !isempty(sel.series_only_in_candidate)
        println(io, "  series in one file only: reference [", join(sel.series_only_in_reference, ", "),
                "], candidate [", join(sel.series_only_in_candidate, ", "), "]",
                sel.pass ? " (not compared: common variables only)" : "  FAIL")
    end
    c = report.categories
    println(io, "  cells: interior ", c.n_cells.interior, ", coastal ", c.n_cells.coastal,
            ", polar ", c.n_cells.polar, c.mask_var === nothing ? " (no mask)" :
            " (mask $(c.mask_var) > $(c.mask_threshold) from $(c.mask_file))")
    for v in report.variables
        name = v.reference_variable == v.candidate_variable ? v.reference_variable :
               "$(v.reference_variable) / $(v.candidate_variable)"
        println(io, name, " [", v.units.reference, v.units.equal ? "" : " vs $(v.units.candidate)", "]",
                v.kind == "static" ? " (no time dimension)" : "", _scale_label(v.reference_scale), ": ",
                v.pass ? "PASS" : "FAIL")
        if v.kind == "series"
            println(io, "  totals max |rel|: slice ", _fmt(v.slice_totals.max_abs_rel_diff),
                    ", month ", _fmt(v.month_totals.max_abs_rel_diff),
                    ", year ", _fmt(v.year_totals.max_abs_rel_diff),
                    "; bands max |diff|/max band ", _fmt(v.bands.max_abs_diff_over_max_band))
        end
        println(io, "  cell rel diff p50 / p99 / max:")
        for cat in CATEGORY_SUFFIXES
            s = getproperty(v.cells, Symbol(cat))
            @printf(io, "    %-9s %s / %s / %s  (n=%d)\n", cat, _fmt(s.p50), _fmt(s.p99), _fmt(s.max), s.n_sample)
        end
        a = v.cells.all
        println(io, "  float32: bitwise ", a.bitwise_equal_fraction === nothing ? "n/a" :
                @sprintf("%.6f", a.bitwise_equal_fraction), ", <=1 ulp ",
                a.within_1ulp_fraction === nothing ? "n/a" : @sprintf("%.6f", a.within_1ulp_fraction),
                ", max ", _fmt(a.ulp_max), " ulp")
        for chk in v.checks
            println(io, "  check ", chk.metric, " = ", _fmt(chk.value), " ", chk.bound, " ",
                    _fmt(chk.tolerance), "  ", check_label(chk))
        end
    end
    isempty(report.variables) && println(io, "  no variables compared")
    return String(take!(io))
end

function main(argv = ARGS)
    opts = _parse_args(argv)
    mesh, mesh_info = load_target_mesh(; binary = opts.binary, Nc = opts.Nc, radius = opts.radius)
    report = compare_cs_flux_files(opts.reference, opts.candidate, mesh;
                                   variables = opts.variables, reference_scale = opts.reference_scale,
                                   allow_unit_mismatch = opts.allow_unit_mismatch,
                                   mask_var = opts.mask_var, mask_file = opts.mask_file,
                                   mask_threshold = opts.mask_threshold,
                                   common_times = opts.common_times,
                                   common_variables_only = opts.common_variables_only,
                                   tolerances = opts.tolerances, block_slices = opts.block_slices, mesh_info)
    text = summary_text(report)
    print(stdout, text)
    opts.json === nothing || write_report_json(opts.json, report)
    opts.summary === nothing || _write_atomically(io -> print(io, text), opts.summary)
    return report.pass ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
