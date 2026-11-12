# ===========================================================================
# IFSWorkspace
# ===========================================================================

"""
    IFSWorkspace{FT, S, C, CA, L}

Per-run workspace for [`IFSConvection`](@ref).

# Fields

- `mu`, `md :: S` — updraft and downdraft interface mass fluxes
  (kg m⁻² s⁻¹), slot `k` = interface `k`, the top of layer `k`; the ground
  interface carries no flux and is not stored. Computed once per column and
  call, shared by all tracers.
- `cu`, `cd :: S` — per-tracer scratch: updraft and downdraft values at the
  interfaces, then the plume flux and the solver coefficients.

  All four are allocated like the air mass: `(Nx, Ny, Nz)`, `(ncells, Nz)`,
  or six `(Nc + 2Hp, Nc + 2Hp, Nz)` panels (only the interior is used).
- `courant :: C` — each column's largest Courant number of the last call,
  shaped like one level of the air mass and zero where never written. The
  host rejects a non-finite or too large value before transporting.
- `cell_metrics :: CA` — cell areas (m²) on the storage backend of the air
  mass; they convert air mass per cell to kg m⁻².
- `ledger_carry :: L` — persistent state of the column mass ledger, as in
  `TM5Workspace`: per column and tracer, `(Nx, Ny, Nt)` /
  `(ncells, Nt)` / six `(Nc, Nc, Nt)`. Allocated empty; the first call sizes
  it with zeros. Not stored in restarts.
"""
mutable struct IFSWorkspace{FT, S, C, CA, L}
    mu           :: S
    md           :: S
    cu           :: S
    cd           :: S
    courant      :: C
    cell_metrics :: CA
    ledger_carry :: L          # persistent; sized by `_ifs_ledger_carry!`
end

_ifs_scratch_like(air_mass::AbstractArray) = similar(air_mass)
_ifs_scratch_like(air_mass::NTuple{6, AbstractArray}) = map(similar, air_mass)

_ifs_column_like(air_mass::AbstractArray{FT, 3}) where FT =
    fill!(similar(air_mass, FT, size(air_mass, 1), size(air_mass, 2)), zero(FT))
_ifs_column_like(air_mass::AbstractArray{FT, 2}) where FT =
    fill!(similar(air_mass, FT, size(air_mass, 1)), zero(FT))
_ifs_column_like(air_mass::NTuple{6, AbstractArray}) = map(_ifs_column_like, air_mass)

_ifs_template(air_mass::AbstractArray) = air_mass
_ifs_template(air_mass::NTuple{6, AbstractArray}) = air_mass[1]

"""
    IFSWorkspace(air_mass; cell_metrics = nothing) -> IFSWorkspace

Allocate the workspace for an air-mass payload (a structured or face-indexed
array, or a cubed-sphere panel tuple). `cell_metrics` (cell areas, as returned
by the convection workspace factory) is copied onto the backend of
`air_mass`; the operator requires it.
"""
function IFSWorkspace(air_mass; cell_metrics = nothing)
    template = _ifs_template(air_mass)
    FT = eltype(template)
    mu, md, cu, cd = ntuple(_ -> _ifs_scratch_like(air_mass), 4)
    courant = _ifs_column_like(air_mass)
    metrics = cell_metrics === nothing ? nothing : _cmfmc_metric_buffer(cell_metrics, template, FT)
    carry = _empty_ledger_carry(air_mass, FT)
    return IFSWorkspace{FT, typeof(mu), typeof(courant), typeof(metrics), typeof(carry)}(
        mu, md, cu, cd, courant, metrics, carry)
end

function Adapt.adapt_structure(to, ws::IFSWorkspace{FT}) where FT
    mu, md, cu, cd = map(a -> Adapt.adapt(to, a), (ws.mu, ws.md, ws.cu, ws.cd))
    courant = Adapt.adapt(to, ws.courant)
    metrics = ws.cell_metrics === nothing ? nothing : Adapt.adapt(to, ws.cell_metrics)
    carry = Adapt.adapt(to, ws.ledger_carry)          # persistent state moves along
    return IFSWorkspace{FT, typeof(mu), typeof(courant), typeof(metrics), typeof(carry)}(
        mu, md, cu, cd, courant, metrics, carry)
end

_ifs_ledger_carry!(ws::IFSWorkspace, dims::Dims) =
    ws.ledger_carry = _fitted_ledger_carry(ws.ledger_carry, dims)
