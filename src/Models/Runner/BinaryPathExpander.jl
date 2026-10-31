"""
    BinaryPathExpander

Expand an `[input]` TOML block into transport-binary file paths. Explicit
lists retain the supplied order; folder/date selection sorts by date and
checks continuity.

## Accepted TOML shapes

```toml
# Shape A: explicit list, consumed in the supplied order
[input]
binary_paths = ["...a.bin", "...b.bin"]

# Shape B: folder + inclusive date range
[input]
folder       = "~/data/.../cs_c48/.../"
start_date   = "2021-12-01"
end_date     = "2021-12-10"
file_pattern = "era5_transport_{YYYYMMDD}_merged1000Pa_float32.bin"  # optional
```

Shapes A and B are mutually exclusive.

## Folder scanning

When `folder` is supplied:

- `readdir(folder)` → candidate filenames.
- Each filename is parsed for an 8-digit `YYYYMMDD` token. If
  `file_pattern` is supplied, the `{YYYYMMDD}` placeholder marks
  where the date lives, and the surrounding text is matched
  literally (as a regex). Without `file_pattern`, any `\\d{8}` run
  anywhere in the filename is used.
- Files parseable as dates within `[start_date, end_date]` are kept;
  files outside the range or not matching the pattern are ignored.
- The resulting sorted list is verified for continuity: every date
  in the closed interval must be present. Missing days list in the
  error message.

## Returns

`Vector{String}` with home/environment-variable expansion. Explicit paths
retain their order and may remain relative to the working directory; their
dates are not checked here. Folder selection sorts chronologically by parsed
date and rejects missing or duplicate dates.
"""
module BinaryPathExpander

using Dates
using ....ConfigChecks: config_date
import ....expand_data_path

export expand_binary_paths

"""
    expand_binary_paths(input_cfg::AbstractDict) -> Vector{String}

See module docstring for the accepted TOML shapes. Throws
`ArgumentError` on mutual exclusion, missing folder, date parse
failure, or gaps in the date range.
"""
function expand_binary_paths(input_cfg::AbstractDict)
    has_explicit = haskey(input_cfg, "binary_paths")
    has_folder   = haskey(input_cfg, "folder")

    if has_explicit && has_folder
        throw(ArgumentError(
            "[input] cannot set both `binary_paths` and `folder`; pick one."))
    end

    if has_explicit
        haskey(input_cfg, "start_date") && config_date(input_cfg["start_date"], "[input].start_date")
        raw = input_cfg["binary_paths"]
        raw isa AbstractVector ||
            throw(ArgumentError("[input].binary_paths must be a list, got $(typeof(raw))"))
        return [expand_data_path(String(p)) for p in raw]
    end

    if has_folder
        return _expand_folder_range(input_cfg)
    end

    throw(ArgumentError(
        "[input] must set either `binary_paths = [...]` or " *
        "`folder + start_date + end_date`."))
end

# ---------------------------------------------------------------------------
# Folder scan + date filter + continuity check
# ---------------------------------------------------------------------------

function _expand_folder_range(input_cfg::AbstractDict)
    folder_raw = input_cfg["folder"]
    folder     = expand_data_path(String(folder_raw))
    isdir(folder) ||
        throw(ArgumentError("[input].folder does not exist: $folder"))

    haskey(input_cfg, "start_date") || throw(ArgumentError(
        "[input].folder requires start_date = \"YYYY-MM-DD\""))
    haskey(input_cfg, "end_date") || throw(ArgumentError(
        "[input].folder requires end_date = \"YYYY-MM-DD\""))

    start_date = config_date(input_cfg["start_date"], "[input].start_date")
    end_date   = config_date(input_cfg["end_date"], "[input].end_date")
    start_date <= end_date || throw(ArgumentError(
        "[input].start_date ($start_date) must be <= end_date ($end_date)"))

    file_pattern = haskey(input_cfg, "file_pattern") ?
                   String(input_cfg["file_pattern"]) : nothing
    regex = _compile_date_regex(file_pattern)

    # Scan + match
    entries = Tuple{Date, String}[]      # (date, absolute_path)
    for name in readdir(folder)
        date = _parse_date_from_name(name, regex)
        date === nothing && continue
        start_date <= date <= end_date || continue
        push!(entries, (date, joinpath(folder, name)))
    end

    # Continuity check
    expected_dates = start_date:Day(1):end_date
    present_dates  = Set(d for (d, _) in entries)
    missing_days   = [d for d in expected_dates if !(d in present_dates)]
    isempty(missing_days) || throw(ArgumentError(
        "[input].folder is missing $(length(missing_days)) date(s) in " *
        "[$start_date, $end_date]: $(missing_days)"))

    # De-dup (multiple files per date → error, as date should be unique)
    sort!(entries; by = first)
    for k in 2:length(entries)
        entries[k][1] == entries[k - 1][1] && throw(ArgumentError(
            "[input].folder has multiple files for date $(entries[k][1]): " *
            "$(basename(entries[k - 1][2])) and $(basename(entries[k][2]))"))
    end

    return [path for (_, path) in entries]
end

# ---------------------------------------------------------------------------
# Filename → Date parser
# ---------------------------------------------------------------------------

# Default: any 8-digit run in the filename.
const _DEFAULT_DATE_REGEX = r"(\d{8})"

function _compile_date_regex(file_pattern::Nothing)
    return _DEFAULT_DATE_REGEX
end

function _compile_date_regex(file_pattern::AbstractString)
    # Replace `{YYYYMMDD}` with a capture group, escape the rest as literal
    # regex. Requires exactly one `{YYYYMMDD}` placeholder.
    occursin("{YYYYMMDD}", file_pattern) || throw(ArgumentError(
        "[input].file_pattern must contain the `{YYYYMMDD}` placeholder; " *
        "got: $file_pattern"))
    count("{YYYYMMDD}", file_pattern) == 1 || throw(ArgumentError(
        "[input].file_pattern may contain `{YYYYMMDD}` at most once"))

    before, after = split(file_pattern, "{YYYYMMDD}"; limit = 2)
    return Regex("^" * _escape_regex(before) * "(\\d{8})" * _escape_regex(after) * "\$")
end

# Manual escape of regex metacharacters; keeps the pattern readable in errors.
function _escape_regex(s::AbstractString)
    return replace(s, r"[\\.^$|?*+()\[\]{}]" => s"\\\0")
end

function _parse_date_from_name(name::AbstractString, regex::Regex)
    m = match(regex, name)
    m === nothing && return nothing
    token = m.captures[1]
    try
        return Date(token, dateformat"yyyymmdd")
    catch
        return nothing
    end
end

end # module BinaryPathExpander
