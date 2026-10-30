# Native input metadata policies, moved unchanged from Models. These deliberately
# retain the existing Pa gate, order inference and inventory-year rules.
using ..Parameters: STANDARD_PRESSURE

# Whether hybrid interface coefficients run from the surface to the top
# (pressure decreasing with index), checked at standard surface pressure.
function hybrid_surface_first(ap, bp)
    p_first = ap[1] + bp[1] * STANDARD_PRESSURE
    p_last = ap[end] + bp[end] * STANDARD_PRESSURE
    p_first == p_last && throw(ArgumentError(
        "hybrid coefficients give the same pressure at both ends of the column"))
    return p_first > p_last
end

# Hybrid A coefficients are pressures. Explicit units must be Pa; without a
# units attribute, values above 2000 unambiguously indicate Pa, while an
# all-zero pure-sigma A is unit-independent. Smaller nonzero magnitudes are
# ambiguous with hPa and must not be guessed.
function validate_ap_units(ap_var, ap, file)
    raw_units = String(string(get(ap_var.attrib, "units", "")))
    units = lowercase(strip(raw_units))
    units = replace(units, r"[\s_-]+" => "")
    if !isempty(units)
        units in ("pa", "pascal", "pascals") || throw(ArgumentError(
            "hybrid coefficient ap in $(file) must use Pa; got units = " *
            repr(raw_units)))
        return nothing
    end
    scale = maximum(abs, ap; init = 0.0)
    (iszero(scale) || scale > 2000) || throw(ArgumentError(
        "hybrid coefficient ap in $(file) has no units and maximum magnitude $scale, " *
        "which is ambiguous with hPa; add units = \"Pa\""))
    return nothing
end

# `init.vertical_order` of a file: which end of its levels comes first.
const VERTICAL_ORDERS = (:surface_first, :toa_first)

function parse_vertical_order(raw)
    order = raw isa AbstractString ? Symbol(lowercase(raw)) : nothing
    order in VERTICAL_ORDERS || throw(ArgumentError(
        "init.vertical_order must be \"surface_first\" or \"toa_first\"; got $(repr(raw))"))
    return order
end

const PRESSURE_UNITS = ("pa", "hpa", "kpa", "mb", "mbar", "millibar", "millibars", "bar",
                         "pascal", "pascals", "hectopascal", "hectopascals")
const HEIGHT_UNITS = ("m", "km", "cm", "meter", "meters", "metre", "metres",
                       "kilometer", "kilometers", "kilometre", "kilometres")

# Level order of a file without hybrid coefficients: `init.vertical_order` when
# given, else from the vertical coordinate. Pressure units or CF `positive =
# "down"` mean the values grow toward the surface, height units or `positive =
# "up"` that they grow toward the top; a coordinate that says neither needs
# `vertical_order`.
function file_vertical_order(cfg, lev_var, lev_values, file)
    haskey(cfg, "vertical_order") && return parse_vertical_order(cfg["vertical_order"])
    positive = lowercase(strip(string(get(lev_var.attrib, "positive", ""))))
    units = lowercase(strip(string(get(lev_var.attrib, "units", ""))))
    grows_down = positive == "down" || (isempty(positive) && units in PRESSURE_UNITS)
    grows_up = positive == "up" || (isempty(positive) && units in HEIGHT_UNITS)
    grows_down || grows_up || throw(ArgumentError(
        "cannot tell which end of the levels of $(file) is the top: its vertical coordinate " *
        "has neither pressure or height units nor a CF positive attribute, and the file " *
        "has no hybrid coefficients; set init.vertical_order = \"toa_first\" or \"surface_first\""))
    increasing = length(lev_values) < 2 || lev_values[end] > lev_values[1]
    return increasing == grows_down ? :toa_first : :surface_first
end

function infer_year_from_path(path::AbstractString)
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

function infer_year_from_time_units(ds)
    haskey(ds, "time") || return nothing
    units = String(get(ds["time"].attrib, "units", ""))
    m = match(r"((?:19|20)\d{2})-\d{1,2}-\d{1,2}", units)
    return m === nothing ? nothing : parse(Int, m.captures[1])
end
