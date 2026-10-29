# ---------------------------------------------------------------------------
# Mass basis tags shared by cell state and flux state.
# ---------------------------------------------------------------------------

"""
    AbstractMassBasis

Supertype for mass-basis tags carried by cell and face-flux states.
"""
abstract type AbstractMassBasis end

"""
    MoistBasis <: AbstractMassBasis

Tag for total-air / moist mass.
"""
struct MoistBasis <: AbstractMassBasis end

"""
    DryBasis <: AbstractMassBasis

Tag for dry-air mass.
"""
struct DryBasis <: AbstractMassBasis end

mass_basis(::Type{B}) where {B <: AbstractMassBasis} = B()

"""Typed interpretation of a transport binary's stored surface pressure."""
abstract type AbstractSurfacePressureBasis end

"""Stored `ps` is the total (moist-air) pressure used by the hybrid coordinate."""
struct TotalSurfacePressure <: AbstractSurfacePressureBasis end

"""Stored `ps` is the column dry-air pressure implied by the dry air mass."""
struct DrySurfacePressure <: AbstractSurfacePressureBasis end

export AbstractMassBasis, MoistBasis, DryBasis, mass_basis
export AbstractSurfacePressureBasis, TotalSurfacePressure, DrySurfacePressure

"""Return the canonical `:dry` or `:moist` tag for a mass basis."""
@inline basis_symbol(::DryBasis) = :dry
@inline basis_symbol(::MoistBasis) = :moist
basis_symbol(value::Union{Symbol, AbstractString}) = basis_symbol(mass_basis(value))

"""
    mass_basis(value::Union{Symbol, AbstractString})

Parse a dry/moist tag (case insensitive). Unknown values raise `ArgumentError`.
State and flux accessors return these same basis instances.
"""
function mass_basis(value::Union{Symbol, AbstractString})
    tag = lowercase(String(value))
    tag == "dry" && return DryBasis()
    tag == "moist" && return MoistBasis()
    throw(ArgumentError("unknown mass basis $(repr(value)); expected dry or moist"))
end

export basis_symbol
