"""Replay deltas for directional lat-lon face fluxes and cell air mass."""
struct StructuredFluxDeltas{AAm, ABm, ACm, AM}
    dam :: AAm
    dbm :: ABm
    dcm :: ACm
    dm  :: AM
end

"""Replay deltas for face-indexed horizontal flux, vertical flux, and air mass."""
struct FaceIndexedFluxDeltas{AH, ACm, AM}
    dhflux :: AH
    dcm    :: ACm
    dm     :: AM
end

"""
    CubedSphereFluxDeltas

Replay delta for panel-native cubed-sphere air mass. Canonical cubed-sphere
forcing keeps face fluxes constant within a meteorological window.
"""
struct CubedSphereFluxDeltas{AM}
    dm :: AM
end

"""
    TransportWindow

A decoded version-4 forcing interval. The concrete flux and delta types encode
the horizontal topology, while optional fields describe physical capabilities:
humidity endpoints, convection, PBL surface forcing, GCHP VDIFF state, and
precomputed TM5 interface diffusion exchange `dkg` [kg s⁻¹]. A window owns no
prognostic tracer state.
"""
struct TransportWindow{Basis <: AbstractMassBasis, M, PS, F, Q, D, C, S, V, DK}
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
    convection       :: C
    surface          :: S
    vdiff            :: V
    dkg              :: DK
end

function Adapt.adapt_structure(to, deltas::StructuredFluxDeltas)
    dam = Adapt.adapt(to, deltas.dam)
    dbm = Adapt.adapt(to, deltas.dbm)
    dcm = Adapt.adapt(to, deltas.dcm)
    dm = Adapt.adapt(to, deltas.dm)
    return StructuredFluxDeltas{typeof(dam), typeof(dbm), typeof(dcm), typeof(dm)}(dam, dbm, dcm, dm)
end

function Adapt.adapt_structure(to, deltas::FaceIndexedFluxDeltas)
    dhflux = Adapt.adapt(to, deltas.dhflux)
    dcm = Adapt.adapt(to, deltas.dcm)
    dm = Adapt.adapt(to, deltas.dm)
    return FaceIndexedFluxDeltas{typeof(dhflux), typeof(dcm), typeof(dm)}(dhflux, dcm, dm)
end

function Adapt.adapt_structure(to, deltas::CubedSphereFluxDeltas)
    return CubedSphereFluxDeltas(Adapt.adapt(to, deltas.dm))
end

function Adapt.adapt_structure(to, window::TransportWindow{B}) where {B <: AbstractMassBasis}
    air_mass = Adapt.adapt(to, window.air_mass)
    surface_pressure = Adapt.adapt(to, window.surface_pressure)
    fluxes = Adapt.adapt(to, window.fluxes)
    qv_start = Adapt.adapt(to, window.qv_start)
    qv_end = Adapt.adapt(to, window.qv_end)
    deltas = Adapt.adapt(to, window.deltas)
    convection = Adapt.adapt(to, window.convection)
    surface = Adapt.adapt(to, window.surface)
    vdiff = Adapt.adapt(to, window.vdiff)
    dkg = Adapt.adapt(to, window.dkg)
    return TransportWindow{B, typeof(air_mass), typeof(surface_pressure),
                           typeof(fluxes), typeof(qv_start), typeof(deltas),
                           typeof(convection), typeof(surface), typeof(vdiff),
                           typeof(dkg)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas,
        convection, surface, vdiff, dkg)
end

mass_basis(::TransportWindow{B}) where {B} = B()
has_humidity_endpoints(window::TransportWindow) =
    window.qv_start !== nothing && window.qv_end !== nothing
has_flux_delta(window::TransportWindow) = window.deltas !== nothing
has_convection_forcing(window::TransportWindow) = window.convection !== nothing

function TransportWindow(air_mass, surface_pressure,
                         fluxes::AbstractFaceFluxState{B};
                         qv_start = nothing, qv_end = nothing,
                         deltas = nothing, convection = nothing,
                         surface = nothing, vdiff = nothing,
                         dkg = nothing) where {B <: AbstractMassBasis}
    return TransportWindow{B, typeof(air_mass), typeof(surface_pressure),
                           typeof(fluxes), typeof(qv_start), typeof(deltas),
                           typeof(convection), typeof(surface), typeof(vdiff),
                           typeof(dkg)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas,
        convection, surface, vdiff, dkg)
end
