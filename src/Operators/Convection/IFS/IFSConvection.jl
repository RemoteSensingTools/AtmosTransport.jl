# ---------------------------------------------------------------------------
# IFSConvection — tracer transport of the ECMWF IFS convection scheme
# (CUCTRACER + CUBIDIAG) driven by the four TM5 convection fields.
# ---------------------------------------------------------------------------

"""
    AbstractIFSTracerVariant

Plume and compensating-flow arithmetic of [`IFSConvection`](@ref):
[`IFSMonotoneUpstream`](@ref) (default) or [`IFSCuctracer`](@ref) (the
arithmetic of the IFS routine, for comparison).
"""
abstract type AbstractIFSTracerVariant end

"""
    IFSCuctracer()

The arithmetic of the IFS tracer routine `CUCTRACER` with the implicit solver
`CUBIDIAG` (`RMFSOLCT = 1`): the downdraft starts with one tenth updraft air
and nine tenths air of the layer above its first interface, the compensating
flow carries the value of the layer above in both directions, and the lowest
layer is kept from going negative by resetting the downdraft value at its top.
`[convection] variant = "cuctracer"`, for comparison with IFS.

It can return negative values for sharp profiles, and it amplifies wherever
the downdraft exceeds the updraft at an interface: there the compensating flow
is taken from downstream, and a perturbation grows by about
`1 / (1 - dt_sub (M^d - M^u) / m)` per sub-step. Plume-consistent columns with
a downdraft below cloud base or above the updraft top do this; over hundreds
of calls such columns can grow without bound.
"""
struct IFSCuctracer <: AbstractIFSTracerVariant end

"""
    IFSMonotoneUpstream()

Variant without new extrema for Courant numbers up to one: the downdraft
starts with air of the layer above its first interface only, the compensating
flow takes the value of the layer it comes from (above where the net mass flux
is upward, below where it is downward), and no lowest-layer correction is
needed. In floating point the bounds hold up to rounding.
`[convection] variant = "monotone"` (default).
"""
struct IFSMonotoneUpstream <: AbstractIFSTracerVariant end

"""
    IFSConvection(; cfl_limit = 0.8, variant = IFSMonotoneUpstream())

Convective tracer transport of the ECMWF IFS convection scheme (OpenIFS
CY48R1.1 `cuctracer.F90` and `cubidiag.F90`, no scavenging) driven by the
four TM5 convection fields of the transport binary,
`forcing.tm5_fields = (; entu, detu, entd, detd)` in kg m⁻² s⁻¹ per layer
(`k = 1` the model top), on the air-mass basis of the state, as for
[`TM5Convection`](@ref). Forward model only: there is no adjoint.

Per column, the interface fluxes are `M^u_k = max(M^u_{k+1} + entu_k - detu_k, 0)`
(ground up, closed at the top of the highest layer with `detu > 0`) and
`M^d_{k+1} = max(M^d_k + entd_k - detd_k, 0)` (top down, closed at the bottom of
the lowest layer with `detd > 0`); no flux crosses the model top or the ground. The updraft and downdraft carry
entraining-plume values; their flux `M^u C^u - M^d C^d` and the compensating
flow of the net mass flux `M^u - M^d` move the tracer, the compensating part
implicitly (`variant`).

`cfl_limit` (in `(0, 1]`) sets the sub-steps of each column:
`n = ceil(dt max_k (max(M^u_k, M^u_{k+1}) + max(M^d_k, M^d_{k+1})) / m_k / cfl_limit)`,
each of length `dt / n` with the tracer of the previous one; the air mass does
not change. IFS instead scales the mass flux down to 0.8 of its CFL value.
A column whose sub-step count would exceed 1024, whose Courant number is not
finite, with a negative field value, or with a layer without air at or below
the highest layer a flux touches, is an error.

Column tracer mass is conserved by the flux form; a column ledger over the
whole column returns the rounding residual of each sub-step in every precision
(as for [`TM5Convection`](@ref)). See the IFS convection section of the
operator-on-binaries manual page for the equations and the differences from
IFS and TM5.
"""
struct IFSConvection{FT, V <: AbstractIFSTracerVariant} <: AbstractConvection
    cfl_limit :: FT
    variant   :: V

    function IFSConvection{FT, V}(cfl_limit::FT, variant::V) where {FT, V <: AbstractIFSTracerVariant}
        (isfinite(cfl_limit) && zero(FT) < cfl_limit <= one(FT)) || throw(ArgumentError(
            "IFSConvection: `cfl_limit` must lie in (0, 1]; got $(cfl_limit)"))
        return new{FT, V}(cfl_limit, variant)
    end
end

IFSConvection(; cfl_limit::Real = 0.8, variant::AbstractIFSTracerVariant = IFSMonotoneUpstream()) =
    IFSConvection{typeof(float(cfl_limit)), typeof(variant)}(float(cfl_limit), variant)

# Largest sub-step count per column and call.
const _IFS_N_SUB_MAX = 1024

# =========================================================================
# Array-level entry: apply_convection!
# =========================================================================

function _assert_ifs_forcing(forcing::ConvectionForcing)
    forcing.tm5_fields === nothing && throw(ArgumentError(
        "IFSConvection requires `forcing.tm5_fields` " *
        "(NamedTuple with :entu, :detu, :entd, :detd) to be populated. " *
        "Use `with_convection_forcing(model, ConvectionForcing(nothing, nothing, tm5_fields))` " *
        "or ensure the driver populates `window.convection.tm5_fields`."))
    return forcing.tm5_fields
end

function _ifs_require_cell_metrics(workspace::IFSWorkspace)
    workspace.cell_metrics === nothing && throw(ArgumentError(
        "IFSConvection requires `workspace.cell_metrics` (cell areas) to convert " *
        "air mass per cell to kg/m². Build the model with `with_convection` or " *
        "construct `IFSWorkspace(air_mass; cell_metrics = ...)`."))
    return workspace.cell_metrics
end

_ifs_courant_extrema(courant::AbstractArray) = extrema(courant)
function _ifs_courant_extrema(courant::NTuple{6, AbstractArray})
    panels = map(extrema, courant)
    return minimum(first, panels), maximum(last, panels)
end

# The Courant numbers live on the device; after the flux launch has finished,
# one reduction per array checks that every column has valid fields and a
# finite, bounded sub-step count before any tracer moves. Negative values are
# the status codes of `_ifs_column_fluxes!`.
function _ifs_check_courant(courant, cfl_limit)
    lowest, worst = _ifs_courant_extrema(courant)
    lowest == _IFS_NEGATIVE_FORCING && throw(ArgumentError(
        "IFSConvection: a column has a negative entu, detu, entd or detd; the " *
        "four convection fields must be non-negative. Check the transport binary."))
    lowest == _IFS_AIRLESS_LAYER && throw(ArgumentError(
        "IFSConvection: a layer without air (air mass <= 0) lies in a convective " *
        "column, at or below the highest layer a mass flux touches. Check the air " *
        "mass and the convection fields."))
    isfinite(worst) || throw(ArgumentError(
        "IFSConvection: a column has a non-finite Courant number; the window's " *
        "entu/detu/entd/detd or the air mass of a convective column contain NaN or Inf. " *
        "Check the transport binary."))
    worst / cfl_limit <= _IFS_N_SUB_MAX || throw(ArgumentError(
        "IFSConvection: the largest column Courant number $(worst) needs more than " *
        "$(_IFS_N_SUB_MAX) sub-steps at cfl_limit = $(cfl_limit). Check that the " *
        "convection fields are in kg/m²/s on the basis of the air mass and that the " *
        "air mass is in kg per cell, or use a shorter time step."))
    return nothing
end

"""
    apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                      op::IFSConvection, dt, workspace::IFSWorkspace, grid) -> nothing

Array-level entry point: tracer mass `q_raw` (kg per cell) and air mass
`air_mass` (kg per cell) of a lat-lon, reduced-Gaussian or cubed-sphere grid.
Two launches (fluxes, then transport) and one host check of the Courant
numbers in between.
"""
function apply_convection!(q_raw::AbstractArray{FT, 4}, air_mass::AbstractArray{FT, 3},
                           forcing::ConvectionForcing, op::IFSConvection, dt::Real,
                           workspace::IFSWorkspace, grid::AtmosGrid{<:LatLonMesh}) where FT
    f = _assert_ifs_forcing(forcing)
    areas = _ifs_require_cell_metrics(workspace)
    Nx, Ny, _, Nt = size(q_raw)
    backend = get_backend(q_raw)
    dt_ft, cfl = FT(dt), FT(op.cfl_limit)
    carry = _ifs_ledger_carry!(workspace, (Nx, Ny, Nt))
    ws = workspace
    launch!(_ifs_fluxes_ll_kernel!, backend, (16, 16), (Nx, Ny),
            ws.courant, ws.mu, ws.md, air_mass, f.entu, f.detu, f.entd, f.detd, areas, dt_ft)
    _ifs_check_courant(ws.courant, cfl)
    launch!(_ifs_transport_ll_kernel!, backend, (16, 16), (Nx, Ny),
            q_raw, carry, air_mass, f.detu, f.detd, ws.mu, ws.md, ws.cu, ws.cd,
            ws.courant, areas, dt_ft, cfl, op.variant)
    return nothing
end

function apply_convection!(q_raw::AbstractArray{FT, 3}, air_mass::AbstractArray{FT, 2},
                           forcing::ConvectionForcing, op::IFSConvection, dt::Real,
                           workspace::IFSWorkspace, grid::AtmosGrid{<:ReducedGaussianMesh}) where FT
    f = _assert_ifs_forcing(forcing)
    areas = _ifs_require_cell_metrics(workspace)
    ncell, _, Nt = size(q_raw)
    backend = get_backend(q_raw)
    dt_ft, cfl = FT(dt), FT(op.cfl_limit)
    carry = _ifs_ledger_carry!(workspace, (ncell, Nt))
    ws = workspace
    launch!(_ifs_fluxes_rg_kernel!, backend, 256, ncell,
            ws.courant, ws.mu, ws.md, air_mass, f.entu, f.detu, f.entd, f.detd, areas, dt_ft)
    _ifs_check_courant(ws.courant, cfl)
    launch!(_ifs_transport_rg_kernel!, backend, 256, ncell,
            q_raw, carry, air_mass, f.detu, f.detd, ws.mu, ws.md, ws.cu, ws.cd,
            ws.courant, areas, dt_ft, cfl, op.variant)
    return nothing
end

function apply_convection!(q_raw::NTuple{6, <:AbstractArray{FT, 4}},
                           air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                           forcing::ConvectionForcing, op::IFSConvection, dt::Real,
                           workspace::IFSWorkspace, grid::AtmosGrid{<:CubedSphereMesh}) where FT
    f = _assert_ifs_forcing(forcing)
    areas = _ifs_require_cell_metrics(workspace)
    Nc, Hp = grid.horizontal.Nc, grid.horizontal.Hp
    Nt = size(q_raw[1], 4)
    backend = get_backend(q_raw[1])
    dt_ft, cfl = FT(dt), FT(op.cfl_limit)
    carry = _ifs_ledger_carry!(workspace, (Nc, Nc, Nt))
    ws = workspace
    launch!(_ifs_fluxes_cs_kernel!, backend, (16, 16), (Nc, Nc, 6),
            ws.courant, ws.mu, ws.md, air_mass, f.entu, f.detu, f.entd, f.detd, areas,
            Int(Hp), dt_ft)
    _ifs_check_courant(ws.courant, cfl)
    launch!(_ifs_transport_cs_kernel!, backend, (16, 16), (Nc, Nc, 6),
            q_raw, carry, air_mass, f.detu, f.detd, ws.mu, ws.md, ws.cu, ws.cd,
            ws.courant, areas, Int(Hp), dt_ft, cfl, op.variant)
    return nothing
end

# =========================================================================
# State-level entry: apply!
# =========================================================================

"""
    apply!(state, forcing::ConvectionForcing, grid::AtmosGrid,
           op::IFSConvection, dt::Real; workspace::IFSWorkspace) -> state

Apply [`IFSConvection`](@ref) to the tracer mass of a lat-lon,
reduced-Gaussian or cubed-sphere state.
"""
function apply!(state::CellState{B, A, Raw}, forcing::ConvectionForcing,
                grid::AtmosGrid{<:LatLonMesh}, op::IFSConvection, dt::Real;
                workspace::IFSWorkspace) where {B, A, Raw <: AbstractArray{<:Any, 4}}
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt, workspace, grid)
    return state
end

function apply!(state::CellState{B, A, Raw}, forcing::ConvectionForcing,
                grid::AtmosGrid{<:ReducedGaussianMesh}, op::IFSConvection, dt::Real;
                workspace::IFSWorkspace) where {B, A, Raw <: AbstractArray{<:Any, 3}}
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt, workspace, grid)
    return state
end

function apply!(state::CubedSphereState, forcing::ConvectionForcing,
                grid::AtmosGrid{<:CubedSphereMesh}, op::IFSConvection, dt::Real;
                workspace::IFSWorkspace)
    apply_convection!(state.tracers_raw, state.air_mass, forcing, op, dt, workspace, grid)
    return state
end
