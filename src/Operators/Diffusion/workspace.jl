"""
    DiffusionWorkspace

Preallocated storage for implicit vertical diffusion.

- `factors` stores tracer-independent Thomas superdiagonal factors, or the
  forward transfer/retention ratios for conservative Dkg mass transport.
- `layer_thickness` stores geometric layer thickness [m] used to convert
  cell-centered Kz [m² s⁻¹] into interface mass exchange [kg s⁻¹].
- `references` stores one cancellation-reducing column reference per packed
  tracer for VMR solves (`(Nx, Ny, Nt)` per cubed-sphere panel and for
  lat-lon, `(ncells, Nt)` for reduced-Gaussian). Single-tracer solves and the
  conservative Dkg mass path keep their reference in registers.
- `hybrid_coefficients` persistently stages the vertical A/B vectors on the
  workspace backend. Window refreshes reuse this storage for dry-pressure
  recovery and hydrostatic `dz`, avoiding per-panel device allocations.
- `ledger_carry` is persistent state of the conservative Dkg column mass
  ledger: per column and tracer, the part of the last solve's rounding
  residual that the largest exchanging layer could not hold (about half its
  ulp at most; see `_ledger_deposit`). Cubed-sphere workspaces hold
  `(Nx, Ny, Nt)` zeros per panel, shaped like `references`; other topologies,
  which have no Dkg path, hold `nothing`. Restarts do not store it: a
  restarted run drops about half an ulp of each column's largest layer,
  once.

The workspace owns only diffusion data. Advection and convection buffers live
in their respective workspace types.
"""
mutable struct HybridCoefficientWorkspace{A, B}
    A :: A
    B :: B
    pure_pressure :: Bool
    positive_b_coordinate :: Bool
end

HybridCoefficientWorkspace(A, B) = HybridCoefficientWorkspace(A, B, false, false)

function HybridCoefficientWorkspace(layer_thickness)
    reference = layer_thickness isa NTuple{6} ? layer_thickness[1] : layer_thickness
    FT = eltype(reference)
    ninterface = size(reference, ndims(reference)) + 1
    return HybridCoefficientWorkspace(similar(reference, FT, ninterface),
                                      similar(reference, FT, ninterface), false, false)
end

struct DiffusionWorkspace{F, D, R, C, L}
    factors         :: F
    layer_thickness :: D
    references      :: R
    hybrid_coefficients :: C
    ledger_carry    :: L
end

DiffusionWorkspace(factors, layer_thickness, references,
                   hybrid_coefficients = HybridCoefficientWorkspace(layer_thickness)) =
    DiffusionWorkspace(factors, layer_thickness, references, hybrid_coefficients,
                       _zero_ledger_carry(references))

# One carry per column and tracer, shaped like the per-tracer references of a
# cubed-sphere workspace. Only the cubed-sphere Dkg path keeps a ledger.
_zero_ledger_carry(references::NTuple{6, AbstractArray}) =
    ntuple(p -> fill!(similar(references[p]), zero(eltype(references[p]))), 6)
_zero_ledger_carry(references) = nothing

function DiffusionWorkspace(air_mass::AbstractArray{FT, N}, n_tracers::Integer = 0) where {FT, N}
    N in (2, 3) || throw(ArgumentError(
        "DiffusionWorkspace requires rank-2 or rank-3 air mass; got rank $N"))
    Nt = Int(n_tracers)
    Nt >= 0 || throw(ArgumentError("n_tracers must be nonnegative"))
    columns = size(air_mass)[1:N-1]                    # (Nx, Ny) or (ncells,)
    references = similar(air_mass, FT, columns..., Nt)
    return DiffusionWorkspace(similar(air_mass), similar(air_mass), references)
end

function DiffusionWorkspace(air_mass::NTuple{6, A}, halo_width::Integer,
                            n_tracers::Integer) where {FT, A <: AbstractArray{FT, 3}}
    Hp = Int(halo_width)
    Nt = Int(n_tracers)
    Hp >= 0 || throw(ArgumentError("halo_width must be nonnegative"))
    Nt >= 0 || throw(ArgumentError("n_tracers must be nonnegative"))
    Nxi, Nyi, Nz = size(air_mass[1])
    Nx, Ny = Nxi - 2Hp, Nyi - 2Hp
    Nx > 0 && Ny > 0 || throw(DimensionMismatch(
        "air-mass panels are too small for halo_width=$Hp"))
    all(size(panel) == (Nxi, Nyi, Nz) for panel in air_mass) ||
        throw(DimensionMismatch("all cubed-sphere air-mass panels must have the same shape"))
    factors = ntuple(_ -> similar(air_mass[1], FT, Nx, Ny, Nz), 6)
    layer_thickness = ntuple(_ -> similar(air_mass[1], FT, Nx, Ny, Nz), 6)
    references = ntuple(_ -> similar(air_mass[1], FT, Nx, Ny, Nt), 6)
    return DiffusionWorkspace(factors, layer_thickness, references)
end

DiffusionWorkspace(state::CellState) = DiffusionWorkspace(state.air_mass, ntracers(state))
DiffusionWorkspace(state::CubedSphereState) =
    DiffusionWorkspace(state.air_mass, state.halo_width, ntracers(state))

function Adapt.adapt_structure(to, workspace::DiffusionWorkspace)
    return DiffusionWorkspace(
        Adapt.adapt(to, workspace.factors),
        Adapt.adapt(to, workspace.layer_thickness),
        Adapt.adapt(to, workspace.references),
        Adapt.adapt(to, workspace.hybrid_coefficients),
        Adapt.adapt(to, workspace.ledger_carry),
    )
end

function Adapt.adapt_structure(to, coefficients::HybridCoefficientWorkspace)
    return HybridCoefficientWorkspace(Adapt.adapt(to, coefficients.A),
                                      Adapt.adapt(to, coefficients.B),
                                      coefficients.pure_pressure,
                                      coefficients.positive_b_coordinate)
end
