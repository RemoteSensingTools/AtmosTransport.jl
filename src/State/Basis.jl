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
