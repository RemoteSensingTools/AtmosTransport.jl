"""
    ConfigChecks

Shared helpers for reading TOML configuration tables: strict Boolean, integer
and date values, the `[numerics] float_type` parser, and known-key checks that
suggest the intended key for a misspelled one. `Output`, `Preprocessing` and
`Models` use them, so this module is loaded right after `Architectures`.
"""
module ConfigChecks

using Dates: Date

export config_bool, config_int, config_date, parse_float_type, key_suggestion, unknown_key_messages, check_known_keys

"""
    config_bool(value, path) -> Bool
    config_bool(cfg, key, default, path) -> Bool

Return a TOML Boolean, rejecting anything else (`"true"`, `1`, ...) with an
`ArgumentError` that names the setting `path`.
"""
function config_bool(value, path::AbstractString)
    value isa Bool || throw(ArgumentError("$(path) must be true or false; got $(repr(value))"))
    return value
end

config_bool(cfg::AbstractDict, key::AbstractString, default::Bool, path::AbstractString) =
    config_bool(get(cfg, key, default), path)

"""
    config_int(value, path; min = nothing) -> Int

Read an integer setting, rejecting Booleans, values outside `Int`, and values
below `min` with an `ArgumentError` naming `path`.
"""
function config_int(value, path::AbstractString; min = nothing)
    value isa Integer && !(value isa Bool) && typemin(Int) <= value <= typemax(Int) ||
        throw(ArgumentError("$(path) must be an integer representable as Int; got $(repr(value))"))
    min === nothing || value >= min ||
        throw(ArgumentError("$(path) must be >= $(min); got $(repr(value))"))
    return Int(value)
end

"""
    config_date(value, path) -> Date

Read a TOML date or an ISO date string (`YYYY-MM-DD`). Invalid values raise an
`ArgumentError` naming `path`.
"""
function config_date(value, path::AbstractString)
    value isa Date && return value
    if value isa AbstractString
        date = tryparse(Date, value)
        date === nothing || return date
    end
    throw(ArgumentError("$(path) must be a date in YYYY-MM-DD format; got $(repr(value))"))
end

"""
    parse_float_type(raw, path) -> Type{Float32} or Type{Float64}

Parse a `float_type` setting. `"Float32"` and `"Float64"` are accepted in any
letter case; anything else (including surrounding spaces) throws an
`ArgumentError` that names the setting `path` and lists the valid values. The
runtime and both preprocessing entry paths read `[numerics] float_type` through
this function, each with its own default.
"""
function parse_float_type(raw, path::AbstractString)
    s = raw isa AbstractString ? lowercase(raw) : ""
    s == "float32" && return Float32
    s == "float64" && return Float64
    throw(ArgumentError("$(path) must be \"Float32\" or \"Float64\"; got $(repr(raw))."))
end

# Levenshtein distance between two strings, by characters.
function _edit_distance(a::AbstractString, b::AbstractString)
    s, t = collect(a), collect(b)
    previous = collect(0:length(t))
    current = similar(previous)
    for (i, c) in enumerate(s)
        current[1] = i
        for (j, d) in enumerate(t)
            current[j + 1] = min(previous[j + 1] + 1, current[j] + 1,
                                 previous[j] + (c == d ? 0 : 1))
        end
        previous, current = current, previous
    end
    return previous[end]
end

"""
    key_suggestion(key, allowed) -> Union{String, Nothing}

The allowed key a misspelled `key` most likely means: the closest one by
case-insensitive edit distance, if at most 2 (1 for keys of up to four
characters), else an allowed key that contains `key` (at least four
characters long) or is contained in it. `nothing` if none qualifies.
"""
function key_suggestion(key::AbstractString, allowed)
    k = lowercase(key)
    best, best_distance = nothing, typemax(Int)
    for candidate in allowed
        d = _edit_distance(k, lowercase(String(candidate)))
        d < best_distance && ((best, best_distance) = (String(candidate), d))
    end
    best !== nothing && best_distance <= (length(k) <= 4 ? 1 : 2) && return best
    length(k) >= 4 || return nothing
    for candidate in allowed
        c = lowercase(String(candidate))
        (occursin(k, c) || occursin(c, k)) && return String(candidate)
    end
    return nothing
end

"""
    unknown_key_messages(cfg, allowed) -> Vector{String}

One entry per key of the table `cfg` that is not in `allowed`, sorted, each
with a suggestion when one exists, e.g. ``kz_max (did you mean `Kz_max`?)``.
"""
function unknown_key_messages(cfg::AbstractDict, allowed)
    messages = String[]
    for key in sort!([String(k) for k in keys(cfg)])
        key in allowed && continue
        suggestion = key_suggestion(key, allowed)
        push!(messages, suggestion === nothing ? key : "$(key) (did you mean `$(suggestion)`?)")
    end
    return messages
end

"""
    check_known_keys(cfg, allowed, label)

Throw an `ArgumentError` listing every key of `cfg` that is not in `allowed`,
for tables where an unknown key is an error.
"""
function check_known_keys(cfg::AbstractDict, allowed, label::AbstractString)
    unknown = unknown_key_messages(cfg, allowed)
    isempty(unknown) || throw(ArgumentError(
        "Unknown $(label) option(s): $(join(unknown, ", ")). " *
        "Supported: $(join(allowed, ", "))."))
    return nothing
end

end # module ConfigChecks
