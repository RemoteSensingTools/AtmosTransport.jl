# Storage helpers shared by the two convection families.
# Cell-area metrics come from the mesh, which stays on the host. Copy them in
# the workspace float type into storage allocated like `reference` (the air
# mass the workspace serves), so a workspace built for device state carries
# device metrics without a later `Adapt.adapt` of the whole workspace.
@inline function _cmfmc_metric_buffer(metric::AbstractArray, reference::AbstractArray,
                                      ::Type{FT}) where FT
    source = eltype(metric) === FT ? metric : FT.(metric)
    return copyto!(similar(reference, FT, size(metric)), source)
end
@inline function _cmfmc_metric_buffer(metrics::NTuple{N}, reference::AbstractArray,
                                      ::Type{FT}) where {N, FT}
    return ntuple(i -> _cmfmc_metric_buffer(metrics[i], reference, FT), N)
end

# Column ledger carry (see `_ledger_deposit`): one value per column and
# tracer, `(Nx, Ny, Nt)` / `(ncells, Nt)` / six `(Nc, Nc, Nt)` panels. A
# workspace starts with an empty carry of its topology's storage type; the
# first call sizes it with zeros and later calls of the same tracer layout
# reuse it, so the field type stays concrete.
_empty_ledger_carry(air_mass::AbstractArray{<:Any, 3}, ::Type{FT}) where FT = similar(air_mass, FT, 0, 0, 0)
_empty_ledger_carry(air_mass::AbstractArray{<:Any, 2}, ::Type{FT}) where FT = similar(air_mass, FT, 0, 0)
_empty_ledger_carry(air_mass::NTuple{6, AbstractArray}, ::Type{FT}) where FT =
    map(m -> similar(m, FT, 0, 0, 0), air_mass)

_fitted_ledger_carry(carry::AbstractArray, dims::Dims) =
    size(carry) == dims ? carry : fill!(similar(carry, dims), zero(eltype(carry)))
_fitted_ledger_carry(carry::NTuple{6, AbstractArray}, dims::Dims) =
    all(c -> size(c) == dims, carry) ? carry :
    map(c -> fill!(similar(c, dims), zero(eltype(c))), carry)
