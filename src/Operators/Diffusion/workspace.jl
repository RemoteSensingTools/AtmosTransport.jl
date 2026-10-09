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

The workspace owns only diffusion data. Advection and convection buffers live
in their respective workspace types.
"""
struct DiffusionWorkspace{F, D, R}
    factors         :: F
    layer_thickness :: D
    references      :: R
end

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
    )
end
