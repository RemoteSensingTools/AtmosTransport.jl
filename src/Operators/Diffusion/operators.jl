# `AbstractDiffusion` is the global root declared in
# `src/Operators/AbstractOperators.jl`. Concrete subtypes here:
# [`NoDiffusion`](@ref), [`ImplicitVerticalDiffusion`](@ref). Non-local /
# counter-gradient variants can be added as sibling concrete types.
# Every concrete subtype implements
#
#     apply!(state::CellState, meteo, grid, op, dt; workspace)
#
# mutating `state.tracers_raw` in place and returning `state`.

"""
    AbstractSurfaceFluxCoupling

Typed policy for where configured surface fluxes enter relative to a
vertical diffusion/mixing operator.

- `SplitSurfaceFluxCoupling`: existing Strang-center composition,
  `V(dt/2) -> S(dt) -> V(dt/2)`.
- `DiffusiveSurfaceFluxBoundary`: GCHP/VDIFF-style lower-boundary
  placement, `S(dt) -> V(dt)`, so fresh surface flux is included in
  the implicit vertical mixing solve.
"""
abstract type AbstractSurfaceFluxCoupling end

"""
    SplitSurfaceFluxCoupling()

Apply surface flux at the transport-palindrome center, bracketed by two
half-step diffusion solves.
"""
struct SplitSurfaceFluxCoupling <: AbstractSurfaceFluxCoupling end

"""
    DiffusiveSurfaceFluxBoundary()

Inject surface mass immediately before one full implicit vertical diffusion
solve, so the new surface mass participates in that solve.
"""
struct DiffusiveSurfaceFluxBoundary <: AbstractSurfaceFluxCoupling end

"""
    NoDiffusion()

Identity operator — `apply!` is a no-op. Default for configurations
without active vertical mixing, and the value `strang_split_mt!` sees
when the palindrome's V position is unoccupied.
"""
struct NoDiffusion <: AbstractDiffusion end

Adapt.adapt_structure(_to, op::NoDiffusion) = op

@inline uses_diffusive_surface_flux_boundary(::NoDiffusion) = false

"""
    ImplicitVerticalDiffusion(; kz_field)

Backward-Euler vertical diffusion driven by either a cell-centered Kz field or
an exact precomputed TM5 interface `dkg` field.
Two spatial layouts are supported:

- structured: `AbstractTimeVaryingField{FT, 3}` over `(Nx, Ny, Nz)`
- face-indexed: `AbstractTimeVaryingField{FT, 2}` over `(ncells, Nz)`

Concrete examples:
- `ConstantField{FT, 3}` / `ConstantField{FT, 2}`
- `PreComputedKzField{FT, A}` wrapping 3D or 2D storage
- `CubedSphereField` wrapping one structured rank-3 field per panel

# `apply!` contract

    apply!(state, meteo, grid, op::ImplicitVerticalDiffusion, dt; workspace)

- Refreshes the Kz cache with `update_field!(op.kz_field, current_time(meteo))`;
  `meteo = nothing` uses `t = 0` for standalone calls.
- For Kz fields, reads `workspace.layer_thickness` as the current layer thicknesses [m].
  The caller is responsible for filling this array before calling
  `apply!` — typically from a hydrostatic integration of the current
  `delp` and surface temperature.
- `PrecomputedCSDkgField` bypasses Kz/geometry reconstruction and does not read
  `layer_thickness`; its interface exchange [kg s⁻¹] is already complete.
- Uses `workspace.factors` for Thomas elimination or Dkg mass-retention factors.
- Launches a topology-specific mass-flux kernel. Packed layouts factor each
  atmospheric column once and advance every tracer with those factors.

The spatial operator is linear, but Backward Euler is not a semigroup:
`V(dt)` and `V(dt/2) ∘ V(dt/2)` differ by `O(dt²)`. The surface-coupling
policy therefore selects the timestep composition explicitly.

# Fields
- `kz_field::KzF` — any `AbstractTimeVaryingField{FT, 2}` or
  `AbstractTimeVaryingField{FT, 3}` providing cell-centered Kz values
  [m²/s geometric], or a `PrecomputedCSDkgField`.
"""
struct ImplicitVerticalDiffusion{FT, KzF, SFC <: AbstractSurfaceFluxCoupling} <: AbstractDiffusion
    kz_field              :: KzF
    surface_flux_coupling :: SFC

    function ImplicitVerticalDiffusion{FT, KzF, SFC}(kz_field::KzF,
                                                     surface_flux_coupling::SFC) where {
                                                     FT, KzF, SFC <: AbstractSurfaceFluxCoupling}
        (KzF <: AbstractTimeVaryingField{FT, 2} ||
         KzF <: AbstractTimeVaryingField{FT, 3} ||
         KzF <: AbstractCubedSphereField{FT}) ||
            throw(ArgumentError("ImplicitVerticalDiffusion: kz_field must be an " *
                "AbstractTimeVaryingField{$FT, 2}, AbstractTimeVaryingField{$FT, 3}, " *
                "or AbstractCubedSphereField{$FT}; got $KzF"))
        new{FT, KzF, SFC}(kz_field, surface_flux_coupling)
    end
end

"""
    ImplicitVerticalDiffusion(; kz_field)

Keyword constructor. `FT` is inferred from `kz_field`.
"""
@inline _diffusion_field_eltype(::AbstractTimeVaryingField{FT}) where FT = FT
@inline _diffusion_field_eltype(::AbstractCubedSphereField{FT}) where FT = FT

@inline function _diffusion_field_eltype(kz_field)
    throw(ArgumentError("ImplicitVerticalDiffusion: kz_field must be an " *
        "AbstractTimeVaryingField or AbstractCubedSphereField; got $(typeof(kz_field))"))
end

function ImplicitVerticalDiffusion(; kz_field,
                                   surface_flux_coupling::AbstractSurfaceFluxCoupling =
                                       SplitSurfaceFluxCoupling())
    FT = _diffusion_field_eltype(kz_field)
    return ImplicitVerticalDiffusion{FT, typeof(kz_field),
                                     typeof(surface_flux_coupling)}(
        kz_field, surface_flux_coupling)
end

Adapt.adapt_structure(to, op::ImplicitVerticalDiffusion) =
    ImplicitVerticalDiffusion(; kz_field = Adapt.adapt(to, op.kz_field),
                              surface_flux_coupling = op.surface_flux_coupling)

@inline uses_diffusive_surface_flux_boundary(op::ImplicitVerticalDiffusion) =
    op.surface_flux_coupling isa DiffusiveSurfaceFluxBoundary

"""
    surface_flux_coupling(diffusion_op) -> AbstractSurfaceFluxCoupling

The policy that orders surface flux and vertical diffusion at the palindrome
center. `NoDiffusion` has no solve to couple to and reports the default
`SplitSurfaceFluxCoupling()`, whose two half solves are then no-ops.
"""
@inline surface_flux_coupling(::NoDiffusion) = SplitSurfaceFluxCoupling()
@inline surface_flux_coupling(op::ImplicitVerticalDiffusion) = op.surface_flux_coupling

"""
    surface_flux_center!(diffuse!, emit!, coupling, dt)

Apply surface flux and vertical diffusion for one transport step of length
`dt` at the palindrome center, in the order that `coupling` defines:

| `coupling` | sequence |
| --- | --- |
| `SplitSurfaceFluxCoupling` | `diffuse!(dt/2)`, `emit!(dt)`, `diffuse!(dt/2)` |
| `DiffusiveSurfaceFluxBoundary` | `emit!(dt)`, `diffuse!(dt)` |

`diffuse!(τ)` applies vertical diffusion over `τ`; `emit!(τ)` adds the
surface flux integrated over `τ`. Backward Euler is not a semigroup, so the
two sequences differ at `O(dt²)` (see `ImplicitVerticalDiffusion`).

This is the single definition of the center ordering. Production transport
reaches it through `palindrome_center!` (`Operators.Advection`). The
cubed-sphere adjoint forward replays call it with the same closures, and its
tape recorders with closures that record each operation before applying it;
the reverse walk visits those records backwards, so the adjoint transposes
this composition.
"""
@inline function surface_flux_center!(diffuse!::D, emit!::E, ::SplitSurfaceFluxCoupling,
                                      dt) where {D, E}
    half_dt = dt / 2
    diffuse!(half_dt)
    emit!(dt)
    diffuse!(half_dt)
    return nothing
end

@inline function surface_flux_center!(diffuse!::D, emit!::E, ::DiffusiveSurfaceFluxBoundary,
                                      dt) where {D, E}
    emit!(dt)
    diffuse!(dt)
    return nothing
end

# =========================================================================
# apply!
# =========================================================================

"""
    apply!(state::CellState, meteo, grid, op::NoDiffusion, dt; workspace=nothing)

No-op; returns `state` unchanged.
"""
function apply!(state::CellState, meteo, grid, ::NoDiffusion, dt;
                workspace = nothing)
    return state
end

function apply!(state::CubedSphereState, meteo, grid, ::NoDiffusion, dt;
                workspace = nothing)
    return state
end

"""
    apply!(state::CellState, meteo, grid, op::ImplicitVerticalDiffusion, dt;
           workspace)

Apply one Backward-Euler implicit diffusion step to every tracer in
`state.tracers_raw` using the column Kz field `op.kz_field` and the dz
stored in `workspace.layer_thickness` (caller-filled). Delegates to
[`apply_vertical_diffusion_vmr!`](@ref), which is the array-level entry
point consumed by both the structured multi-tracer palindrome and the
face-indexed reduced-Gaussian transport block.

Throws if `workspace` is not supplied or if its `layer_thickness` shape
doesn't match `state.tracers_raw`.
"""
function apply!(state::CellState, meteo, grid,
                op::ImplicitVerticalDiffusion{FT}, dt;
                workspace) where FT
    workspace === nothing && throw(ArgumentError(
        "ImplicitVerticalDiffusion.apply!: workspace is required " *
        "(factors and layer_thickness must be supplied)"))
    # LL packed + RG face-indexed now go through the mass-flux VMR
    # wrapper: pre-scale tracer_mass → VMR, solve with mass-flux
    # coefficients, post-scale VMR → tracer_mass. Preserves `Σ m·q` to
    # roundoff for inert tracers, matching the CS path's conservation
    # contract.
    apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass,
                                   op, workspace, dt, meteo)
    return state
end

function apply!(state::CubedSphereState, meteo, grid,
                op::ImplicitVerticalDiffusion{FT, KzF}, dt;
                workspace) where {FT, KzF <: AbstractCubedSphereField{FT}}
    workspace === nothing && throw(ArgumentError(
        "ImplicitVerticalDiffusion.apply!: workspace is required " *
        "(cubed-sphere diffusion needs a panel-native DiffusionWorkspace)"))
    apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass, op,
                                  workspace, dt, meteo;
                                  halo_width = state.halo_width)
    return state
end

function apply!(state::CubedSphereState, meteo, grid,
                op::ImplicitVerticalDiffusion, dt;
                workspace) 
    throw(ArgumentError(
        "CubedSphereState diffusion requires a panel-native kz_field. " *
        "Wrap six structured rank-3 fields in CubedSphereField(...) before " *
        "constructing ImplicitVerticalDiffusion."))
end

# =========================================================================
# Array-level entry points
# =========================================================================

"""
    apply_vertical_diffusion_vmr!(rm, air_mass, op, workspace, dt, meteo = nothing;
                                  [halo_width, tracer]) -> nothing

Apply one Backward-Euler diffusion step to tracer mass `rm` (in and out):
structured `(Nx, Ny, Nz, Nt)`, face-indexed `(ncells, Nz, Nt)` or
`(ncells, Nz)`, or six halo-padded cubed-sphere panels
`(Nc + 2Hp, Nc + 2Hp, Nz[, Nt])` with `halo_width = Hp`. This is the entry
point of the transport palindrome and of the cubed-sphere adjoint replays. For Kz fields
the column solve gives the new mixing ratio and the tracer mass is updated in
flux form, `rm_k += F_{k−½} − F_{k+½}`, so each column total changes only by
unbiased rounding (`diffusion_kernels.jl`). Precomputed cubed-sphere `dkg`
fields use the conservative mass solve of `conservative_dkg.jl`.

`NoDiffusion` is a no-op method, so call sites compile to nothing.
"""
function apply_vertical_diffusion_vmr! end

apply_vertical_diffusion_vmr!(q_raw::NTuple{6}, air_mass::NTuple{6},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing; halo_width = 0, tracer = nothing) = nothing
# LL/RG NoDiffusion stubs.
apply_vertical_diffusion_vmr!(q_raw::AbstractArray{<:Any, 4},
                              air_mass::AbstractArray{<:Any, 3},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing) = nothing
apply_vertical_diffusion_vmr!(q_raw::AbstractArray{<:Any, 3},
                              air_mass::AbstractArray{<:Any, 2},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing) = nothing
apply_vertical_diffusion_vmr!(q_raw::AbstractArray{<:Any, 2},
                              air_mass::AbstractArray{<:Any, 2},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing; tracer = nothing) = nothing

# ---------------------------------------------------------------------------
# Column solves: shape checks and launches
# ---------------------------------------------------------------------------

# The column arrays a solve uses: the Thomas ratios, the interface exchange
# and the elimination scratch, plus the layer thickness for Kz fields.
_column_workspace_arrays(workspace, ::KzExchange) =
    (factors = workspace.factors, exchange = workspace.exchange,
     elimination = workspace.elimination, layer_thickness = workspace.layer_thickness)
_exchange_source(op::ImplicitVerticalDiffusion, workspace) =
    KzExchange(op.kz_field, workspace.layer_thickness)
_panel_exchange_sources(op::ImplicitVerticalDiffusion{FT, <:AbstractCubedSphereField}, workspace) where FT =
    ntuple(p -> KzExchange(panel_field(op.kz_field, p), workspace.layer_thickness[p]), 6)

# Structured lat-lon, packed tracers (Nx, Ny, Nz, Nt).
function _diffuse_columns!(form, q_raw::AbstractArray{FT, 4}, air_mass::AbstractArray{FT, 3},
                           op::ImplicitVerticalDiffusion, workspace::DiffusionWorkspace,
                           dt, meteo; tracer = nothing) where FT
    Nx, Ny, Nz, Nt = size(q_raw)
    size(air_mass) == (Nx, Ny, Nz) || throw(DimensionMismatch(
        "lat-lon diffusion: air_mass shape $(size(air_mass)) does not match the tracer " *
        "spatial shape $((Nx, Ny, Nz))"))
    source = _exchange_source(op, workspace)
    _check_column_workspace(_column_workspace_arrays(workspace, source), (Nx, Ny, Nz), "(lat-lon)")
    carry, first_slot = _column_ledger_carry(form, workspace.ledger_carry, (Nx, Ny), Nt, true,
                                             tracer, "(lat-lon)")
    update_field!(op.kz_field, current_time(meteo))
    launch!(_vertical_diffusion_structured_kernel!, get_backend(q_raw), (8, 8), (Nx, Ny),
            form, q_raw, air_mass, source, workspace.exchange, workspace.factors,
            workspace.elimination, carry, FT(dt), Nz, Nt, first_slot)
    return nothing
end

# Face-indexed reduced Gaussian, packed (ncells, Nz, Nt) or single (ncells, Nz).
function _diffuse_columns!(form, q_raw::Union{AbstractArray{FT, 3}, AbstractArray{FT, 2}},
                           air_mass::AbstractArray{FT, 2}, op::ImplicitVerticalDiffusion,
                           workspace::DiffusionWorkspace, dt, meteo; tracer = nothing) where FT
    ncells, Nz = size(q_raw, 1), size(q_raw, 2)
    Nt = size(q_raw, 3)
    size(air_mass) == (ncells, Nz) || throw(DimensionMismatch(
        "reduced-Gaussian diffusion: air_mass shape $(size(air_mass)) does not match the " *
        "tracer spatial shape $((ncells, Nz))"))
    source = _exchange_source(op, workspace)
    _check_column_workspace(_column_workspace_arrays(workspace, source), (ncells, Nz),
                            "(face-indexed)")
    carry, first_slot = _column_ledger_carry(form, workspace.ledger_carry, (ncells,), Nt,
                                             ndims(q_raw) == 3, tracer, "(face-indexed)")
    update_field!(op.kz_field, current_time(meteo))
    launch!(_vertical_diffusion_face_kernel!, get_backend(q_raw), 256, ncells,
            form, q_raw, air_mass, source, workspace.exchange, workspace.factors,
            workspace.elimination, carry, FT(dt), Nz, Nt, first_slot)
    return nothing
end

# Cubed sphere: six halo-padded panels (Nc + 2Hp, Ny + 2Hp, Nz[, Nt]), one launch.
function _diffuse_panels!(form, q_raw::NTuple{6, A}, air_mass::NTuple{6},
                          op::ImplicitVerticalDiffusion, workspace::DiffusionWorkspace,
                          dt, meteo, halo_width::Integer; tracer = nothing) where {A <: AbstractArray}
    FT = eltype(A)
    Hp = Int(halo_width)
    Hp >= 0 || throw(ArgumentError("halo_width must be nonnegative; got $Hp"))
    shape = size(q_raw[1])
    all(p -> size(q_raw[p]) == shape, 2:6) || throw(DimensionMismatch(
        "cubed-sphere diffusion panels must share one shape; got $(map(size, q_raw))"))
    Nc, Ny, Nz = shape[1] - 2Hp, shape[2] - 2Hp, shape[3]
    packed = ndims(A) == 4
    Nt = packed ? shape[4] : 1
    Nc > 0 && Ny > 0 || throw(DimensionMismatch(
        "cubed-sphere panel shape $shape has no interior with halo_width = $Hp"))
    sources = _panel_exchange_sources(op, workspace)
    carry = workspace.ledger_carry
    carry_panels = ntuple(_ -> nothing, 6)
    first_slot = 1
    for p in 1:6
        size(air_mass[p]) == shape[1:3] || throw(DimensionMismatch(
            "cubed-sphere diffusion panel $p: tracer spatial shape $(shape[1:3]) does not " *
            "match air_mass shape $(size(air_mass[p]))"))
        arrays = _column_workspace_arrays(workspace, sources[p])
        arrays = map(a -> a isa NTuple{6} ? a[p] : a, arrays)
        _check_column_workspace(arrays, (Nc, Ny, Nz), "panel $p")
        _, first_slot = _column_ledger_carry(form, _ledger_panel(carry, p),
                                             (Nc, Ny), Nt, packed, tracer, "on panel $p")
    end
    carry_panels = ntuple(p -> carry isa NTuple{6} ?
                         _select_ledger_panel(carry, p, packed, tracer) : nothing, 6)
    update_field!(op.kz_field, current_time(meteo))
    launch!(_vertical_diffusion_panels_kernel!, get_backend(q_raw[1]), (8, 8, 1), (Nc, Ny, 6),
            form, q_raw, air_mass, sources, workspace.exchange, workspace.factors,
            workspace.elimination, carry_panels, FT(dt), Nz, Nt, Hp, first_slot)
    return nothing
end

# ---------------------------------------------------------------------------
# Tracer-mass solves: flux form with the column ledger for Kz fields
# ---------------------------------------------------------------------------

function apply_vertical_diffusion_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing;
                                       halo_width::Integer,
                                       tracer::Union{Nothing, Integer} = nothing) where {
                                           FT, A <: AbstractArray{FT, 3},
                                           KzF <: AbstractCubedSphereField{FT}}
    _diffuse_panels!(_TracerMassFluxForm(), q_raw, air_mass, op, workspace, dt, meteo, halo_width;
                     tracer)
    return nothing
end

function apply_vertical_diffusion_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing;
                                       halo_width::Integer) where {FT, A <: AbstractArray{FT, 4},
                                                                    KzF <: AbstractCubedSphereField{FT}}
    _diffuse_panels!(_TracerMassFluxForm(), q_raw, air_mass, op, workspace, dt, meteo, halo_width)
    return nothing
end

# Lat-lon packed tracer mass.
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 4},
                                       air_mass::AbstractArray{FT, 3},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 3}}
    size(q_raw, ndims(q_raw)) == 0 && return nothing
    _check_packed_carry(workspace, q_raw)
    _diffuse_columns!(_TracerMassFluxForm(), q_raw, air_mass, op, workspace, dt, meteo)
    return nothing
end

# Face-indexed reduced-Gaussian packed tracer mass.
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 3},
                                       air_mass::AbstractArray{FT, 2},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    size(q_raw, ndims(q_raw)) == 0 && return nothing
    _check_packed_carry(workspace, q_raw)
    _diffuse_columns!(_TracerMassFluxForm(), q_raw, air_mass, op, workspace, dt, meteo)
    return nothing
end

# Face-indexed reduced-Gaussian single-tracer mass; `tracer` selects its
# ledger carry slot (the per-tracer reduced-Gaussian transport passes it).
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 2},
                                       air_mass::AbstractArray{FT, 2},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing;
                                       tracer::Union{Nothing, Integer} = nothing) where {
                                           FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    _diffuse_columns!(_TracerMassFluxForm(), q_raw, air_mass, op, workspace, dt, meteo; tracer)
    return nothing
end

# ---------------------------------------------------------------------------
# Precomputed cubed-sphere Dkg: conservative mass solve (conservative_dkg.jl)
# ---------------------------------------------------------------------------

"""
    apply_vertical_diffusion_vmr!(rm, air_mass, op, workspace, dt, meteo;
                                  halo_width, tracer = nothing)

Cubed-sphere helper for state variables stored as tracer mass. The implicit
solver uses the current dry air mass. Precomputed Dkg uses conservative
bidiagonal factors directly on tracer mass. Kz fields solve for the mixing
ratio and update the tracer mass in flux form before advection resumes.

The Dkg solve keeps a column mass ledger whose sub-ulp remainder persists in
`workspace.ledger_carry`, one value per column and tracer (see
`_ledger_deposit`). Packed tracers (`(N, N, Nz, Nt)` panels) use carry slot
`t` for tracer `t`. A single-tracer panel tuple uses slot `tracer`, its index
in the state; with `tracer = nothing` it keeps no carry, so the call is a
function of its input alone (adjoint tapes and replays rely on this). Other
fields ignore `tracer`.
"""
function apply_vertical_diffusion_vmr!(rm::NTuple{6, A}, air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt, meteo=nothing;
                                       halo_width::Integer,
                                       tracer::Union{Nothing, Integer} = nothing) where {
                                           FT, A <: AbstractArray{FT, 3},
                                           KzF <: AbstractCSDkgField{FT}}
    _apply_cs_dkg_mass!(rm, air_mass, op, workspace, dt, meteo, halo_width, tracer)
end

function apply_vertical_diffusion_vmr!(rm::NTuple{6, A}, air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt, meteo=nothing;
                                       halo_width::Integer) where {
                                           FT, A <: AbstractArray{FT, 4},
                                           KzF <: AbstractCSDkgField{FT}}
    _apply_cs_dkg_mass!(rm, air_mass, op, workspace, dt, meteo, halo_width, nothing)
end

function _apply_cs_dkg_mass!(rm::NTuple{6}, air_mass::NTuple{6}, op,
                             workspace::DiffusionWorkspace, dt, meteo, halo_width, tracer)
    FT = eltype(rm[1])
    Hp = Int(halo_width)
    Hp >= 0 || throw(ArgumentError("halo_width must be nonnegative"))
    packed = ndims(rm[1]) == 4
    length(workspace.factors) == 6 || throw(DimensionMismatch("Dkg requires six factor panels"))
    carry = workspace.ledger_carry
    # Check every panel before changing any tracer values.
    for p in 1:6
        N, Ny, Nz, Nt = size(rm[p], 1), size(rm[p], 2), size(rm[p], 3), size(rm[p], 4)
        Nc, Nj = N - 2Hp, Ny - 2Hp
        Nc > 0 && Nj > 0 && Nz > 0 || throw(DimensionMismatch("Dkg panel $p has no physical interior"))
        size(air_mass[p]) == (N, Ny, Nz) || throw(DimensionMismatch("Dkg panel $p tracer and air shapes differ"))
        size(workspace.factors[p]) == (Nc, Nj, Nz) ||
            throw(DimensionMismatch("Dkg factor panel $p must have shape $((Nc, Nj, Nz))"))
        _column_ledger_carry(_TracerMassFluxForm(), _ledger_panel(carry, p),
                             (Nc, Nj), Nt, packed, tracer, "on panel $p")
    end
    all(p -> size(rm[p]) == size(rm[1]), 2:6) || throw(DimensionMismatch(
        "Dkg panels must share one shape; got $(map(size, rm))"))
    update_field!(op.kz_field, current_time(meteo))
    # One launch covers all six panels (the third index selects the panel), so
    # each column runs alone in a launch of 6 Nc² columns instead of Nc².
    Nc, Ny = size(rm[1], 1) - 2Hp, size(rm[1], 2) - 2Hp
    Nz, Nt = size(rm[1], 3), size(rm[1], 4)
    backend = get_backend(rm[1])
    dkg = ntuple(p -> panel_field(op.kz_field, p), 6)
    carry_panels = ntuple(p -> _select_ledger_panel(carry, p, packed, tracer), 6)
    column_tile = _cs_dkg_mass_workgroupsize(backend, FT)
    tracer_tile = _cs_dkg_tracer_workgroupsize(backend, FT)
    if packed && Nt > 1 && tracer_tile !== nothing
        # Queue the read-only tracer solves after factor construction on the
        # same backend stream; the synchronization below completes both.
        launch!(_vertical_diffusion_cs_dkg_factors_panels_kernel!, backend, column_tile,
                (Nc, Ny, 6), workspace.factors, air_mass, dkg, FT(dt), Nz, Hp; sync = false)
        launch!(_vertical_diffusion_cs_mass_dkg_tracers_panels_kernel!, backend, tracer_tile,
                (Nc, Ny, 6Nt), rm, carry_panels, air_mass, dkg, workspace.factors, FT(dt),
                Nz, Nt, Hp; sync = false)
    elseif packed
        launch!(_vertical_diffusion_cs_mass_dkg_packed_panels_kernel!, backend, column_tile,
                (Nc, Ny, 6), rm, carry_panels, air_mass, dkg, workspace.factors, FT(dt), Nz,
                Nt, Hp; sync = false)
    else
        launch!(_vertical_diffusion_cs_mass_dkg_panels_kernel!, backend, column_tile,
                (Nc, Ny, 6), rm, carry_panels, air_mass, dkg, workspace.factors, FT(dt), Nz, Hp,
                tracer === nothing ? 1 : Int(tracer); sync = false)
    end
    synchronize(backend)
    return nothing
end
