# ---------------------------------------------------------------------------
# Flux units: one grammar for every units string of a surface-flux input
#
# A flux unit is an amount (kg, g, tonnes, mol, mmol, µmol) of a substance
# (carbon, a named species, or the output species when the string names none),
# per area (m⁻², or per grid cell when no area is named) and per time (second,
# day, calendar month, Julian year). The strings in use (P0 facts memo,
# Section 3; design 2.4) include `kg m-2 s-1`, `kg  m-2 s-1`, `kg.m-2.s-1`,
# `kg m$^{-2}$ s$^{-1}$`, `kg m-2 s-1, per unit land area occupied by the PFT`,
# `kgC m-2 s-1`, `gC m-2 d-1`, `g C per month` (per cell), `mmol m-2 s-1`,
# `kgCO2/month/m2` (GridFED) and `Tonnes` (EDGAR, per cell and year).
# ---------------------------------------------------------------------------

"""
    AbstractSubstance

What a flux amount counts: [`TargetSpecies`](@ref), [`Carbon`](@ref) or a
named [`Species`](@ref).
"""
abstract type AbstractSubstance end

"""
    TargetSpecies()

The units name no substance (`kg m-2 s-1`): the amount is of the output species,
the runtime's reading of such files. Molar units cannot be of an unnamed
substance.
"""
struct TargetSpecies <: AbstractSubstance end

"""
    Carbon()

Amounts of carbon (`kgC`, `g C`, `mol C`). Converted to a carbon-bearing
species with one carbon atom per molecule by `M(species) / CARBON_MOLAR_MASS`,
which conserves carbon atoms (CO₂: `SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS`
≈ 3.66419).
"""
struct Carbon <: AbstractSubstance end

"""
    Species(name::Symbol)

Amounts of the species `name` (`kgCO2` → `Species(:co2)`). Names are lower
case; the molar mass of a species comes from `SPECIES_MOLAR_MASS`.
"""
struct Species <: AbstractSubstance
    name :: Symbol
    Species(name::Symbol) = new(Symbol(lowercase(String(name))))
end
Species(name::AbstractString) = Species(Symbol(name))

"""
    AbstractAreaBasis

What a flux is per: [`PerSquareMetre`](@ref) (a density) or [`PerCell`](@ref)
(a per-cell rate; the caller divides by the cell area).
"""
abstract type AbstractAreaBasis end

"Flux per square metre (`m-2`, `/m2`)."
struct PerSquareMetre <: AbstractAreaBasis end

"Flux per grid cell (no area in the units, or `cell-1`): GFED `g C per month`, EDGAR `Tonnes`."
struct PerCell <: AbstractAreaBasis end

"""
    AbstractTimeBasis

The time unit of a flux: [`PerSecond`](@ref), [`PerDay`](@ref),
[`PerMonth`](@ref) or [`PerYear`](@ref). [`period_seconds`](@ref) gives its
length.
"""
abstract type AbstractTimeBasis end

"Per second (`s-1`, `/s`)."
struct PerSecond <: AbstractTimeBasis end

"Per day of 86400 s (`d-1`, `/day`)."
struct PerDay <: AbstractTimeBasis end

"Per calendar month (`month-1`, `/month`, `per month`): the length of the month containing the interval."
struct PerMonth <: AbstractTimeBasis end

"Per Julian year of `DAYS_PER_JULIAN_YEAR` days (`yr-1`, `/year`; EDGAR `Tonnes`)."
struct PerYear <: AbstractTimeBasis end

"""
    FluxUnits{S, A, T}

Parsed flux units ([`parse_flux_units`](@ref)).

# Fields
- `text` — the units string as given
- `amount_per_unit` — kg (mass units) or mol (molar units) per stated amount
  unit: `1` for kg and mol, `1e-3` for g and mmol, `1e3` for tonnes
- `molar` — `true` for molar units (mol, mmol, µmol)
- `substance::S` — [`TargetSpecies`](@ref), [`Carbon`](@ref) or [`Species`](@ref)
- `area::A` — [`PerSquareMetre`](@ref) or [`PerCell`](@ref)
- `time::T` — [`PerSecond`](@ref), [`PerDay`](@ref), [`PerMonth`](@ref) or [`PerYear`](@ref)
"""
struct FluxUnits{S <: AbstractSubstance, A <: AbstractAreaBasis, T <: AbstractTimeBasis}
    text            :: String
    amount_per_unit :: Float64
    molar           :: Bool
    substance       :: S
    area            :: A
    time            :: T
end

_describe(::TargetSpecies) = "the output species"
_describe(::Carbon) = "carbon"
_describe(s::Species) = String(s.name)
_describe(::PerSquareMetre) = "per m²"
_describe(::PerCell) = "per cell"
_describe(::PerSecond) = "per second"
_describe(::PerDay) = "per day"
_describe(::PerMonth) = "per calendar month"
_describe(::PerYear) = "per Julian year"

function Base.show(io::IO, u::FluxUnits)
    unit = u.molar ? "mol" : "kg"
    print(io, "FluxUnits(", repr(u.text), ": ", u.amount_per_unit, " ", unit, " of ",
          _describe(u.substance), " ", _describe(u.area), " ", _describe(u.time), ")")
end

# ---- Grammar ----

# Free-text qualifiers after a comma that do not change the units. SDGVM `ra`:
# the flux is per unit land area occupied by the PFT; the land-fraction weight
# is applied by the source configuration, not by the units.
const _UNIT_ANNOTATIONS = ("per unit land area occupied by the pft",)

# Amount units: (token, amount per unit, molar), longest token first where
# tokens share a prefix.
const _AMOUNT_TOKENS = (("tonnes", 1.0e3, false), ("tonne", 1.0e3, false),
                        ("kg", 1.0, false), ("g", 1.0e-3, false),
                        ("mmol", 1.0e-3, true), ("umol", 1.0e-6, true), ("mol", 1.0, true))

# Area and time factors of the normalised string, longest token first where
# tokens share a prefix. "/m2s" is the run-together `kg/m2s` (per m² per second).
const _FACTOR_TOKENS = (
    ("/m2s", PerSquareMetre(), PerSecond()),
    ("/m2", PerSquareMetre(), nothing), ("m-2", PerSquareMetre(), nothing),
    ("cell-1", PerCell(), nothing), ("/cell", PerCell(), nothing), ("percell", PerCell(), nothing),
    ("s-1", nothing, PerSecond()), ("/s", nothing, PerSecond()),
    ("day-1", nothing, PerDay()), ("/day", nothing, PerDay()), ("perday", nothing, PerDay()),
    ("d-1", nothing, PerDay()), ("/d", nothing, PerDay()),
    ("month-1", nothing, PerMonth()), ("/month", nothing, PerMonth()),
    ("permonth", nothing, PerMonth()), ("mon-1", nothing, PerMonth()), ("/mon", nothing, PerMonth()),
    ("year-1", nothing, PerYear()), ("/year", nothing, PerYear()), ("peryear", nothing, PerYear()),
    ("yr-1", nothing, PerYear()), ("/yr", nothing, PerYear()),
)

# Lower case, annotation split off, whitespace and LaTeX/exponent markup
# removed: "kg m$^{-2}$ s$^{-1}$" → "kgm-2s-1", "kg.m-2.s-1" → "kgm-2s-1".
function _normalized_units(text::AbstractString)
    s = lowercase(strip(text))
    if occursin(',', s)
        head, tail = split(s, ','; limit = 2)
        strip(tail) in _UNIT_ANNOTATIONS || throw(ArgumentError(
            "unknown qualifier $(repr(strip(tail))) in flux units $(repr(text))"))
        s = head
    end
    s = replace(s, "µ" => "u", "μ" => "u", "²" => "2", "⁻" => "-", "¹" => "1")
    return filter(c -> !(isspace(c) || c in ('$', '{', '}', '^', '*', '.')), s)
end

# Scan `rest` as a sequence of area and time factors; `nothing` if it is not one
# or names an area or a time twice.
function _scan_factors(rest::AbstractString)
    area = nothing
    time = nothing
    pos = 1
    while pos <= lastindex(rest)
        tail = SubString(rest, pos)
        i = findfirst(t -> startswith(tail, t[1]), _FACTOR_TOKENS)
        i === nothing && return nothing
        token, a, t = _FACTOR_TOKENS[i]
        if a !== nothing
            area === nothing || return nothing
            area = a
        end
        if t !== nothing
            time === nothing || return nothing
            time = t
        end
        pos += ncodeunits(token)
    end
    return (; area, time)
end

# Species token right after the amount: "(co2)" or a bare name; candidates are
# the bare name's prefixes from longest to shortest, then no token.
function _species_candidates(rest::AbstractString)
    m = match(r"^\([a-z][a-z0-9]*\)", rest)
    m === nothing || return [m.match]
    m = match(r"^[a-z][a-z0-9]*", rest)
    m === nothing && return [""]
    return [[m.match[1:i] for i in length(m.match):-1:1]; ""]
end

_substance_from_token(token::AbstractString) =
    (name = strip(token, ['(', ')']); name == "c" ? Carbon() : Species(Symbol(name)))

_substance_from_keyword(::Nothing) = nothing
_substance_from_keyword(s::AbstractSubstance) = s
_substance_from_keyword(s::Union{Symbol, AbstractString}) =
    lowercase(String(s)) in ("c", "carbon") ? Carbon() : Species(Symbol(s))

"""
    parse_flux_units(s; substance = nothing, molar_default = nothing) -> FluxUnits

Parse the flux units string `s`. Case, whitespace, `.`/`*` separators and LaTeX
exponent markup (`m\$^{-2}\$`) are ignored. The grammar is

    <amount>[<substance>]<factors>[, <qualifier>]

- amount: `kg`, `g`, `tonnes`, `mol`, `mmol`, `µmol`/`umol`;
- substance (optional, right after the amount): `C` (carbon) or a species
  name such as `CO2`, `(CO2)`;
- factors, in any order: an area (`m-2`, `/m2`, `cell-1`) and a time (`s-1`,
  `/s`, `d-1`, `/day`, `month-1`, `/month`, `per month`, `yr-1`, `/year`);
  without an area the flux is per cell. A time is required, except for
  `Tonnes`, the EDGAR convention for tonnes per cell and year;
- qualifier: only `per unit land area occupied by the PFT` (SDGVM `ra`).

`substance` names the substance when the string does not (TRENDY files write
`kg m-2 s-1` for kg of carbon: `substance = :carbon`; ECCO-Darwin writes
`mmol m-2 s-1` of CO₂: `substance = :co2`). `:c`/`:carbon` give
[`Carbon`](@ref), other names a [`Species`](@ref). A substance named both in
the string and by the keyword must agree. Without either, mass units are of
the [`TargetSpecies`](@ref); molar units without a substance take
`molar_default` (the runtime loaders pass the tracer's species, so
`mmol m-2 s-1` is moles of that species) and throw when it is `nothing`.

Examples: `"kg m-2 s-1"`, `"kg  m-2 s-1"`, `"kg.m-2.s-1"`,
`"kg m\$^{-2}\$ s\$^{-1}\$"`, `"kgC m-2 s-1"`, `"gC m-2 d-1"`,
`"g C per month"` (per cell), `"kgCO2/month/m2"`, `"Tonnes"`,
`"mmol m-2 s-1"` (with `substance`), and the runtime's
`"kg/m2/s"`, `"kg(CO2) m^-2 s^-1"`, `"kg/m²/s"`. Anything else throws an
`ArgumentError`.
"""
function parse_flux_units(s::AbstractString; substance = nothing, molar_default = nothing)
    text = String(s)
    norm = _normalized_units(text)
    isempty(norm) && throw(ArgumentError("flux units are empty; the units of the source must be given"))
    a = findfirst(t -> startswith(norm, t[1]), _AMOUNT_TOKENS)
    a === nothing && throw(ArgumentError(
        "flux units $(repr(text)) do not start with an amount (kg, g, tonnes, mol, mmol, µmol)"))
    amount_token, amount_per_unit, molar = _AMOUNT_TOKENS[a]
    rest = norm[(ncodeunits(amount_token) + 1):end]
    parsed = nothing
    token = ""
    for candidate in _species_candidates(rest)
        factors = _scan_factors(rest[(ncodeunits(candidate) + 1):end])
        factors === nothing && continue
        factors.time === nothing && !startswith(amount_token, "tonne") && continue
        parsed = factors
        token = candidate
        break
    end
    parsed === nothing && throw(ArgumentError(
        "cannot parse flux units $(repr(text)): expected an amount, an optional substance, " *
        "an optional area (m-2) and a time (s-1, d-1, month-1, yr-1)"))
    area = something(parsed.area, PerCell())
    time = something(parsed.time, PerYear())                 # `Tonnes`: per year
    from_string = isempty(token) ? nothing : _substance_from_token(token)
    from_keyword = _substance_from_keyword(substance)
    from_string !== nothing && from_keyword !== nothing && from_string != from_keyword &&
        throw(ArgumentError("flux units $(repr(text)) name $(_describe(from_string)), but the " *
                            "source declares $(_describe(from_keyword))"))
    resolved = something(from_string, from_keyword, TargetSpecies())
    molar && resolved isa TargetSpecies && molar_default !== nothing &&
        (resolved = _substance_from_keyword(molar_default))
    molar && resolved isa TargetSpecies && throw(ArgumentError(
        "molar flux units $(repr(text)) need a substance; give the species " *
        "(parse_flux_units(s; substance = :co2))"))
    return FluxUnits(text, amount_per_unit, molar, resolved, area, time)
end

# ---- Conversion factors ----

_period_seconds(::PerSecond, iv) = 1.0
_period_seconds(::PerDay, iv) = SECONDS_PER_DAY
_period_seconds(::PerYear, iv) = DAYS_PER_JULIAN_YEAR * SECONDS_PER_DAY
_period_seconds(::PerMonth, ::Nothing) = throw(ArgumentError(
    "per-month flux units need the time interval the flux applies to"))
function _period_seconds(::PerMonth, iv::TimeInterval)
    month = interval(MonthKey(iv.start))
    iv.stop <= month.stop || throw(ArgumentError(
        "per-month flux units apply within one calendar month; the interval " *
        "[$(iv.start), $(iv.stop)) crosses into the next month"))
    return Dates.value(month.stop - month.start) / 1000.0
end

"""
    period_seconds(u::FluxUnits, iv::TimeInterval) -> Float64
    period_seconds(u::FluxUnits) -> Float64

Length in seconds of the time unit of `u`: `1` (per second), `86400` (per
day), `DAYS_PER_JULIAN_YEAR × 86400` (per year), or the length of the
calendar month that contains the interval `iv` (per month; `iv` must lie in
one month, and the form without `iv` throws).
"""
period_seconds(u::FluxUnits, iv::TimeInterval) = _period_seconds(u.time, iv)
period_seconds(u::FluxUnits) = _period_seconds(u.time, nothing)

function _molar_mass(species::Symbol)
    hasproperty(SPECIES_MOLAR_MASS, species) || throw(ArgumentError(
        "no molar mass for species $(repr(species)); SPECIES_MOLAR_MASS has " *
        "$(join(repr.(propertynames(SPECIES_MOLAR_MASS)), ", "))"))
    return Float64(getproperty(SPECIES_MOLAR_MASS, species))
end

function _carbon_to_species(species::Symbol)
    hasproperty(SPECIES_CARBON_ATOMS, species) || throw(ArgumentError(
        "carbon fluxes can be converted only to species with carbon " *
        "($(join(repr.(propertynames(SPECIES_CARBON_ATOMS)), ", "))); got $(repr(species))"))
    return getproperty(SPECIES_CARBON_ATOMS, species)
end

_species_kg(::TargetSpecies, molar::Bool, amount, species) = amount
function _species_kg(s::Species, molar::Bool, amount, species)
    s.name == species || throw(ArgumentError(
        "flux units are amounts of $(s.name), but the output species is $(species)"))
    return molar ? amount * _molar_mass(species) : amount
end
function _species_kg(::Carbon, molar::Bool, amount, species)
    atoms = _carbon_to_species(species)
    return molar ? amount * _molar_mass(species) / atoms :
                   amount * _molar_mass(species) / (atoms * CARBON_MOLAR_MASS)
end

"""
    kg_per_unit(u::FluxUnits, species::Symbol) -> Float64

Kilograms of `species` per stated amount unit of `u`:

- mass units of the [`TargetSpecies`](@ref) or of `Species(species)`:
  `amount_per_unit`;
- molar units of `Species(species)`: `amount_per_unit × M(species)`;
- [`Carbon`](@ref): `amount_per_unit × M(species) / M(C)` (mass) or
  `amount_per_unit × M(species)` (molar), for species with one carbon atom
  (CO₂). For `kgC` and `:co2` this is exactly
  `SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS`.

`M` is `SPECIES_MOLAR_MASS` and `CARBON_MOLAR_MASS`. Units of another species
than `species` throw.
"""
kg_per_unit(u::FluxUnits, species::Symbol) =
    _species_kg(u.substance, u.molar, u.amount_per_unit, Symbol(lowercase(String(species))))

"""
    mass_rate_factor(u::FluxUnits, species::Symbol, iv::TimeInterval) -> Float64
    mass_rate_factor(u::FluxUnits, species::Symbol) -> Float64

Factor that turns a value in units `u` into kg of `species` per second (per m²
or per cell, as `u.area` says):
[`kg_per_unit`](@ref)`(u, species) / `[`period_seconds`](@ref)`(u, iv)`. The
form without `iv` is for units whose time unit does not depend on the
calendar (not per month). For per-second units it is `kg_per_unit` exactly.
"""
mass_rate_factor(u::FluxUnits, species::Symbol, iv::TimeInterval) =
    kg_per_unit(u, species) / period_seconds(u, iv)
mass_rate_factor(u::FluxUnits, species::Symbol) = kg_per_unit(u, species) / period_seconds(u)

"""
    canonical_units_string(species::Symbol) -> String

The units string the flux writers use for a density of `species`:
`"kg CO2 m-2 s-1"` for `:co2` (`"kg SF6 m-2 s-1"`, `"kg Rn222 m-2 s-1"`).
[`parse_flux_units`](@ref) reads it back as kg of `Species(species)` per m² per
second. The species must be one of `SPECIES_MOLAR_MASS`.
"""
function canonical_units_string(species::Symbol)
    _molar_mass(species)                                   # known species only
    label = hasproperty(SPECIES_LABELS, species) ? getproperty(SPECIES_LABELS, species) :
                                                    uppercase(String(species))
    return "kg $(label) m-2 s-1"
end
