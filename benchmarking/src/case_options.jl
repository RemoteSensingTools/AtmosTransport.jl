# Benchmark case matrix: built-in defaults, named presets, and command-line
# overrides. Base + TOML only, shared by the harness (`AtmosTransportBenchmarks`)
# and the lab lane (`benchmarking/lab/LabLane.jl`), which uses it to know how
# many records a preset run must produce. Include `labels.jl` first.

using TOML

"Directory holding the named presets (`<name>.toml`)."
const PRESET_DIRECTORY = normpath(joinpath(@__DIR__, "..", "presets"))

const SUPPORTED_OPERATORS = (:advection, :diffusion, :convection, :full, :io, :adjoint)
const SUPPORTED_SCHEMES = (:upwind, :slopes, :ppm, :linrood5, :linrood7)
const SUPPORTED_FLOAT_TYPES = ("Float32", "Float64")

"""
Operators whose cost does not depend on the tracer count. `adjoint` times
`cs_surface_emission_footprint`, the reverse-mode footprint of one scalar
objective of one tracer, so it runs as a single `1tr` case whatever `tracers`
lists.
"""
const SINGLE_TRACER_OPERATORS = (:adjoint,)

"""
Operators that never run the advection scheme selected by `scheme`:
`diffusion` and `convection` run without advection, `io` moves fields only,
and `adjoint` always differentiates `PPMScheme(NoLimiter())`. They run once,
under the default scheme `$(DEFAULT_SERIES_SCHEME)` (omitted from series
names), whatever `scheme` lists.
"""
const SCHEME_FREE_OPERATORS = (:diffusion, :convection, :io, :adjoint)

"""
    CaseOptions

Resolved case matrix of one harness run; `case_matrix` lists its
cases. `convection_depth` is the number of bottom levels in the TM5
convection matrix (`TM5Convection.lmax_conv`); `0` means the full column, and
positive values are clipped to each case's level count. `max_grid` maps an
operator to the largest grid size `Nc` it runs on (preset `[limits]`).
"""
struct CaseOptions
    group::String
    float_types::Vector{String}
    grids::Vector{Int}
    levels::Vector{Int}
    tracers::Vector{Int}
    operators::Vector{Symbol}
    schemes::Vector{Symbol}
    steps::Int
    warmup_steps::Int
    repeats::Int
    convection_depth::Int
    max_grid::Dict{Symbol, Int}
end

# Built-in values used when neither the command line nor a preset sets an
# option. These are the historical CLI defaults of `run_benchmarks.jl`.
const DEFAULT_CASE_OPTIONS = Dict{String, Any}(
    "group" => "Synthetic CS Sweep",
    "float_type" => ["Float32"],
    "grid" => ["C24"],
    "levels" => [32],
    "tracers" => [1],
    "operator" => ["advection"],
    "scheme" => ["ppm"],
    "steps" => 5,
    "warmup_steps" => 2,
    "repeats" => 5,
    "convection_depth" => 0,
    "limits" => Dict{String, Any}(),
)

# Keys a preset file may set, in addition to its human-readable description.
const PRESET_KEYS = Tuple(sort!(vcat(collect(keys(DEFAULT_CASE_OPTIONS)), ["description"])))

"""
    preset_file(name_or_path) -> String

Path of a preset: an existing `.toml` path is used as given, otherwise
`name` resolves to `$(basename(PRESET_DIRECTORY))/<name>.toml` next to the harness.
"""
function preset_file(name_or_path::AbstractString)
    s = String(name_or_path)
    endswith(s, ".toml") && isfile(s) && return abspath(s)
    occursin(r"^[A-Za-z0-9_-]+$", s) ||
        throw(ArgumentError("preset $(repr(s)) is neither a preset name nor an existing .toml file"))
    path = joinpath(PRESET_DIRECTORY, s * ".toml")
    isfile(path) || throw(ArgumentError(
        "unknown preset $(repr(s)); available: $(join(available_presets(), ", "))"))
    return path
end

available_presets() = isdir(PRESET_DIRECTORY) ?
    sort!([splitext(f)[1] for f in readdir(PRESET_DIRECTORY) if endswith(f, ".toml")]) :
    String[]

"""
    parse_preset(table::AbstractDict; source = "preset") -> Dict{String,Any}

Validate a parsed preset table: unknown keys are errors (a typo must not
silently fall back to a default). Returns the case-option entries.
"""
function parse_preset(table::AbstractDict; source::AbstractString = "preset")
    unknown = sort!([String(k) for k in keys(table) if !(String(k) in PRESET_KEYS)])
    isempty(unknown) || throw(ArgumentError(
        "$(source): unknown preset key(s) $(join(unknown, ", ")); allowed: $(join(PRESET_KEYS, ", "))"))
    return Dict{String, Any}(String(k) => v for (k, v) in table if String(k) != "description")
end

load_preset(name_or_path::AbstractString) =
    (path = preset_file(name_or_path); parse_preset(TOML.parsefile(path); source = path))

_option_list(v::AbstractVector) = v
_option_list(v::AbstractString) = split(v, ","; keepempty = false)
_option_list(v) = [v]

_parse_grid(x::Integer) = Int(x)
_parse_grid(x::AbstractString) = parse(Int, replace(strip(x), r"^[Cc]" => ""))
_parse_count(x::Integer) = Int(x)
_parse_count(x::AbstractString) = parse(Int, replace(strip(x), r"^[LlCc]" => ""))

function _parse_float_name(x)
    key = lowercase(strip(string(x)))
    key in ("f32", "float32") && return "Float32"
    key in ("f64", "float64") && return "Float64"
    throw(ArgumentError("unsupported float type $(x); use $(join(SUPPORTED_FLOAT_TYPES, " or "))"))
end

function _parse_member(x, allowed, what)
    sym = Symbol(lowercase(strip(string(x))))
    sym in allowed ||
        throw(ArgumentError("unsupported $(what) $(x); use one of $(join(allowed, ", "))"))
    return sym
end

function _positive(name, v::Integer; allow_zero = false)
    (allow_zero ? v >= 0 : v > 0) ||
        throw(ArgumentError("$(name) must be $(allow_zero ? "non-negative" : "positive"), got $(v)"))
    return Int(v)
end

_scalar_int(x::Integer) = Int(x)
_scalar_int(x::AbstractString) = parse(Int, strip(x))

# The synthetic fields need an interior cell corner on each panel (C2 or larger).
function _grid_size(x)
    Nc = _positive("grid", _parse_grid(x))
    Nc >= 2 || throw(ArgumentError(
        "grid C$(Nc) is too small: the synthetic benchmark fields need C2 or larger"))
    return Nc
end

# Preset `[limits]`: per operator, a table with the largest grid it runs on,
#   [limits.io]
#   max_grid = "C180"
const LIMIT_KEYS = ("max_grid",)

function _parse_limits(limits)
    limits isa AbstractDict ||
        throw(ArgumentError("limits must be a table of per-operator tables, got $(repr(limits))"))
    max_grid = Dict{Symbol, Int}()
    for (op, table) in limits
        operator = _parse_member(op, SUPPORTED_OPERATORS, "operator in limits")
        table isa AbstractDict ||
            throw(ArgumentError("limits.$(op) must be a table such as { max_grid = \"C180\" }"))
        unknown = sort!([String(k) for k in keys(table) if !(String(k) in LIMIT_KEYS)])
        isempty(unknown) || throw(ArgumentError(
            "limits.$(op): unknown key(s) $(join(unknown, ", ")); allowed: $(join(LIMIT_KEYS, ", "))"))
        haskey(table, "max_grid") && (max_grid[operator] = _grid_size(table["max_grid"]))
    end
    return max_grid
end

# The group is the first field of every series name.
function _parse_group(x)
    group = String(strip(string(x)))
    isempty(group) && throw(ArgumentError("group must not be empty"))
    occursin('/', group) &&
        throw(ArgumentError("group $(repr(group)) must not contain '/', the series separator"))
    startswith(group, BREAKDOWN_SERIES_PREFIX) &&
        throw(ArgumentError("group $(repr(group)) must not start with $(repr(BREAKDOWN_SERIES_PREFIX))"))
    return group
end

"""
    resolve_case_options(overrides, preset) -> CaseOptions

Layer the case options: an `overrides` entry that is not `nothing` (the
command line) wins over the `preset` entry, which wins over
`DEFAULT_CASE_OPTIONS`. Keys use preset spelling (`float_type`,
`warmup_steps`, ...); values may be comma-separated strings or TOML values.
"""
function resolve_case_options(overrides::AbstractDict, preset::Union{Nothing, AbstractDict})
    unknown = sort!([String(k) for k in keys(overrides) if !haskey(DEFAULT_CASE_OPTIONS, String(k))])
    isempty(unknown) || throw(ArgumentError("unknown case option(s) $(join(unknown, ", "))"))
    function pick(key)
        v = get(overrides, key, nothing)
        v === nothing || return v
        preset === nothing || (v = get(preset, key, nothing))
        return v === nothing ? DEFAULT_CASE_OPTIONS[key] : v
    end
    nonempty(name, v) = isempty(v) ? throw(ArgumentError("$(name) list is empty")) : v
    return CaseOptions(
        _parse_group(pick("group")),
        nonempty("float_type", unique!(_parse_float_name.(_option_list(pick("float_type"))))),
        nonempty("grid", [_grid_size(x) for x in _option_list(pick("grid"))]),
        nonempty("levels", [_positive("levels", _parse_count(x)) for x in _option_list(pick("levels"))]),
        nonempty("tracers", [_positive("tracers", _parse_count(x)) for x in _option_list(pick("tracers"))]),
        nonempty("operator", [_parse_member(x, SUPPORTED_OPERATORS, "operator")
                              for x in _option_list(pick("operator"))]),
        nonempty("scheme", [_parse_member(x, SUPPORTED_SCHEMES, "scheme")
                            for x in _option_list(pick("scheme"))]),
        _positive("steps", _scalar_int(pick("steps"))),
        _positive("warmup_steps", _scalar_int(pick("warmup_steps")); allow_zero = true),
        _positive("repeats", _scalar_int(pick("repeats"))),
        _positive("convection_depth", _scalar_int(pick("convection_depth")); allow_zero = true),
        _parse_limits(pick("limits")),
    )
end

"One benchmark case of a `case_matrix`."
const CaseSpec = NamedTuple{(:float_type, :grid, :levels, :tracers, :operator, :scheme),
                            Tuple{String, Int, Int, Int, Symbol, Symbol}}

"""
    case_matrix(options) -> Vector{CaseSpec}

The cases a run measures, in run order: every combination of float type,
grid, levels, tracers, operator, and scheme, except that operators in
`SINGLE_TRACER_OPERATORS` run at one tracer and operators in
`SCHEME_FREE_OPERATORS` run under the default scheme only, so no two cases
measure the same thing under different names, and that an operator with a
`max_grid` limit runs only on grids up to it. Repeated list entries give one
case.
"""
function case_matrix(o::CaseOptions)
    cases = CaseSpec[]
    default_scheme = Symbol(DEFAULT_SERIES_SCHEME)
    for float_type in o.float_types, grid in o.grids, levels in o.levels, operator in o.operators
        grid <= get(o.max_grid, operator, typemax(Int)) || continue
        tracer_counts = operator in SINGLE_TRACER_OPERATORS ? [1] : o.tracers
        schemes = operator in SCHEME_FREE_OPERATORS ? [default_scheme] : o.schemes
        for tracers in tracer_counts, scheme in schemes
            push!(cases, (; float_type, grid, levels, tracers, operator, scheme))
        end
    end
    return unique!(cases)
end

"Number of benchmark cases (records) a run with these options produces."
expected_case_count(o::CaseOptions) = length(case_matrix(o))

"""
    effective_convection_depth(convection_depth, levels) -> Int

`TM5Convection.lmax_conv` for one case: `0` (full column) stays `0`, positive
depths are clipped to the case's level count.
"""
effective_convection_depth(depth::Integer, levels::Integer) =
    depth == 0 ? 0 : min(Int(depth), Int(levels))

"""
    convection_forcing_band(levels, convection_depth) -> UnitRange{Int}

Levels that carry the synthetic updraft entrainment and detrainment: the
middle third of the active convective depth, counted with `k = 1` at the model
top. With the full column active (`convection_depth = 0` or `>= levels`) this
is the historical band `max(2, Nz÷3):max(lo, min(Nz-1, 2Nz÷3))`, so existing
series measure the same forcing.
"""
function convection_forcing_band(levels::Integer, convection_depth::Integer)
    Nz = Int(levels)
    L = effective_convection_depth(convection_depth, Nz)
    L = L == 0 ? Nz : L
    offset = Nz - L
    lo = min(max(2, L ÷ 3), L)          # min(…, L) only matters for L = 1
    hi = max(lo, min(L - 1, 2L ÷ 3))
    return (offset + lo):(offset + hi)
end
