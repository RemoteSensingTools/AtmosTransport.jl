# ---------------------------------------------------------------------------
# MetState — upstream meteorological fields (before flux construction)
#
# MetState holds raw meteorological fields for source-specific processing.
# Transport operators consume typed mass-flux windows instead.
# ---------------------------------------------------------------------------

"""
    MetState{PA <: AbstractArray, QA <: AbstractArray, M}

Container for meteorological fields upstream of the transport core.

# Fields
- `ps :: PA` — surface pressure [Pa]. Layout: `(Nx, Ny)` or `(ncells,)`.
- `q  :: QA` — specific humidity [kg/kg]. Layout is the matching horizontal
  shape plus a trailing vertical dimension.
- `metvars :: M` — additional met-specific fields (winds, omega, diffusivities, etc.)
  as a `NamedTuple`. Content depends on the met driver.

Transport operators never receive `MetState` directly; preprocessing converts
source meteorology into the current dry-basis transport-binary contract.

!!! warning "Deprecated"
    No part of AtmosTransport uses `MetState`; it will be removed in the next
    minor release.
"""
struct MetState{PA <: AbstractArray, QA <: AbstractArray, M <: NamedTuple}
    ps      :: PA
    q       :: QA
    metvars :: M

    function MetState(ps::PA, q::QA, metvars::M) where
            {PA <: AbstractArray, QA <: AbstractArray, M <: NamedTuple}
        Base.depwarn("`MetState` is deprecated: no part of AtmosTransport uses it. It will be removed in the next minor release.", :MetState)
        ndims(q) == ndims(ps) + 1 || throw(DimensionMismatch(
            "MetState q must have exactly one more dimension than ps; got ndims(ps)=$(ndims(ps)), ndims(q)=$(ndims(q))"))
        size(q)[1:ndims(ps)] == size(ps) || throw(DimensionMismatch(
            "MetState q horizontal shape $(size(q)[1:ndims(ps)]) must match ps shape $(size(ps))"))
        return new{PA, QA, M}(ps, q, metvars)
    end
end

function MetState(ps::AbstractArray, q::AbstractArray; metvars...)
    return MetState(ps, q, NamedTuple(metvars))
end

export MetState
