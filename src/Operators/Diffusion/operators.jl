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
- `ProfileKzField{FT}` with default rank 3 or
  `ProfileKzField(profile; spatial_rank = 2)`
- `PreComputedKzField{FT, A}` wrapping 3D or 2D storage
- `DerivedKzField` for meteorology-driven Beljaars-Viterbo on
  structured grids
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
  The Dkg mass path does not use the packed reference scratch.
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
# Lower-level apply_vertical_diffusion! — array-level entry point
# =========================================================================

"""
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt,
                              meteo = nothing) -> nothing

Low-level entry point. Applies one Backward-Euler diffusion step to a
raw tracer buffer in any of the supported layouts:

- structured packed tracers: `q_raw :: (Nx, Ny, Nz, Nt)`
- face-indexed packed tracers: `q_raw :: (ncells, Nz, Nt)`
- face-indexed single-tracer slice: `q_raw :: (ncells, Nz)`

This is the function `strang_split_mt!` calls at the palindrome
center. The face-indexed reduced-Gaussian path also
uses it at its H → V → D → V → H center slot.

`meteo` is threaded through to `update_field!(op.kz_field, t)` as the Float64
clock `t = current_time(meteo)` (`0.0` if `meteo === nothing`).
`air_mass` is mandatory because the solver conserves tracer mass, not the
geometric integral of mixing ratio.

`NoDiffusion` is a no-op: the method is `= nothing` so Julia's
dispatch reduces the call site to a dead branch when
`diffusion_op isa NoDiffusion`. This makes the palindrome
integration compile to an identity.
"""
function apply_vertical_diffusion! end
function apply_vertical_diffusion_vmr! end

apply_vertical_diffusion!(q_raw::NTuple{6}, air_mass::NTuple{6},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing; halo_width = 0) = nothing
apply_vertical_diffusion_vmr!(q_raw::NTuple{6}, air_mass::NTuple{6},
                              ::NoDiffusion, workspace, dt,
                              meteo = nothing; halo_width = 0) = nothing
# LL/RG mass-flux NoDiffusion stubs.
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 4},
                          air_mass::AbstractArray{<:Any, 3},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 3},
                          air_mass::AbstractArray{<:Any, 2},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
apply_vertical_diffusion!(q_raw::AbstractArray{<:Any, 2},
                          air_mass::AbstractArray{<:Any, 2},
                          ::NoDiffusion, workspace, dt,
                          meteo = nothing) = nothing
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
                              meteo = nothing) = nothing

@inline function _check_diffusion_workspace_shape(dz_scratch, w_scratch,
                                                  expected_shape, shape_label)
    size(dz_scratch) == size(w_scratch) ||
        throw(DimensionMismatch("w_scratch and dz_scratch sizes must match"))
    size(dz_scratch) == expected_shape || throw(DimensionMismatch(
        "workspace scratch arrays are $(size(dz_scratch)) but q_raw " *
        "$shape_label shape is $(expected_shape)"))
    return nothing
end

@inline function _check_cs_diffusion_workspace_shape(dz_scratch, w_scratch,
                                                     expected_shape, panel)
    size(dz_scratch) == size(w_scratch) ||
        throw(DimensionMismatch("cubed-sphere w_scratch and dz_scratch sizes must match on panel $panel"))
    size(dz_scratch) == expected_shape || throw(DimensionMismatch(
        "cubed-sphere workspace scratch arrays on panel $panel are $(size(dz_scratch)) " *
        "but the interior panel shape is $(expected_shape)"))
    return nothing
end

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace::DiffusionWorkspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 3},
                                                                KzF <: AbstractCSDkgField{FT}}
    w_scratch = workspace.factors
    update_field!(op.kz_field, current_time(meteo))
    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        size(w_scratch[p]) == (Nc, Ny, Nz) || throw(DimensionMismatch(
            "cubed-sphere dkg workspace panel $p has shape $(size(w_scratch[p])); expected $((Nc, Ny, Nz))"))
        panel_dkg = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_single_dkg_kernel!(backend, (8, 8))
        kernel(panel_q, panel_m, panel_dkg, w_scratch[p], FT(dt), Nz, Hp;
               ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace::DiffusionWorkspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 4},
                                                                KzF <: AbstractCSDkgField{FT}}
    w_scratch = workspace.factors
    reference_scratch = workspace.references
    length(w_scratch) == 6 && length(reference_scratch) == 6 ||
        throw(DimensionMismatch(
            "cubed-sphere dkg workspace must provide 6 factor and reference panels"))
    update_field!(op.kz_field, current_time(meteo))
    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        size(w_scratch[p]) == (Nc, Ny, Nz) || throw(DimensionMismatch(
            "cubed-sphere dkg workspace panel $p has shape $(size(w_scratch[p])); expected $((Nc, Ny, Nz))"))
        size(reference_scratch[p]) == (Nc, Ny, Nt) || throw(DimensionMismatch(
            "cubed-sphere dkg reference panel $p has shape $(size(reference_scratch[p])); expected $((Nc, Ny, Nt))"))
        panel_dkg = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_dkg_kernel!(backend, (8, 8))
        kernel(panel_q, panel_m, panel_dkg, w_scratch[p], reference_scratch[p],
               FT(dt), Nz, Nt, Hp; ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace::DiffusionWorkspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 3},
                                                                KzF <: AbstractCubedSphereField{FT}}
    hasproperty(workspace, :factors) && hasproperty(workspace, :layer_thickness) ||
        throw(ArgumentError(
            "cubed-sphere diffusion requires a workspace with panel-native " *
            "`factors` and `layer_thickness` tuples"))

    w_scratch = workspace.factors
    dz_scratch = workspace.layer_thickness
    length(w_scratch) == 6 && length(dz_scratch) == 6 ||
        throw(DimensionMismatch(
            "cubed-sphere diffusion workspace must provide 6 factor and geometry panels"))

    update_field!(op.kz_field, current_time(meteo))

    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q) == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere diffusion panel $p: q shape $(size(panel_q)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        _check_cs_diffusion_workspace_shape(dz_scratch[p], w_scratch[p], (Nc, Ny, Nz), p)
        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_single_kernel!(backend, (8, 8))
        kernel(panel_q, panel_m, panel_kz, dz_scratch[p], w_scratch[p],
               FT(dt), Nz, Hp;
               ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function apply_vertical_diffusion!(q_raw::NTuple{6, A},
                                   air_mass::NTuple{6, <:AbstractArray{FT, 3}},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace::DiffusionWorkspace, dt,
                                   meteo = nothing;
                                   halo_width::Integer) where {FT, A <: AbstractArray{FT, 4},
                                                                KzF <: AbstractCubedSphereField{FT}}
    hasproperty(workspace, :factors) && hasproperty(workspace, :layer_thickness) &&
        hasproperty(workspace, :references) ||
        throw(ArgumentError(
            "cubed-sphere diffusion requires a workspace with panel-native " *
            "`factors`, `layer_thickness`, and `references` tuples"))

    w_scratch = workspace.factors
    dz_scratch = workspace.layer_thickness
    reference_scratch = workspace.references
    length(w_scratch) == 6 && length(dz_scratch) == 6 &&
        length(reference_scratch) == 6 || throw(DimensionMismatch(
            "cubed-sphere diffusion workspace must provide 6 factor, geometry, and reference panels"))

    update_field!(op.kz_field, current_time(meteo))

    Hp = Int(halo_width)
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q)[1:3] == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere packed diffusion panel $p: q spatial shape " *
            "$(size(panel_q)[1:3]) does not match air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        _check_cs_diffusion_workspace_shape(dz_scratch[p], w_scratch[p], (Nc, Ny, Nz), p)
        size(reference_scratch[p]) == (Nc, Ny, Nt) || throw(DimensionMismatch(
            "cubed-sphere diffusion reference panel $p has shape $(size(reference_scratch[p])); expected $((Nc, Ny, Nt))"))
        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_q)
        kernel = _vertical_diffusion_cs_kernel!(backend, (8, 8))
        kernel(panel_q, panel_m, panel_kz, dz_scratch[p], w_scratch[p],
               reference_scratch[p], FT(dt), Nz, Nt, Hp;
               ndrange = (Nc, Ny))
        synchronize(backend)
    end
    return nothing
end

function _cs_scale_tracer_mass_to_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray}
    # Single barrier at the end: the six CS panels are spatially
    # independent within this scale step, so the six kernel launches
    # can stream through the backend's queue and we only need one
    # synchronize() before the caller observes the result. Saves five
    # device-host round-trips per call.
    Hp = Int(halo_width)
    backend = get_backend(q_raw[1])
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q) == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere tracer panel $p shape $(size(panel_q)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nc > 0 && Ny > 0 || throw(DimensionMismatch(
            "cubed-sphere panel $p shape $(size(panel_q)) cannot provide an " *
            "interior with halo_width=$Hp"))
        kernel = _cs_tracer_mass_to_vmr_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz))
    end
    synchronize(backend)
    return q_raw
end

function _cs_scale_tracer_mass_to_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray{<:Any, 4}}
    Hp = Int(halo_width)
    backend = get_backend(q_raw[1])
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q)[1:3] == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere packed tracer panel $p spatial shape $(size(panel_q)[1:3]) " *
            "does not match air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        Nc > 0 && Ny > 0 || throw(DimensionMismatch(
            "cubed-sphere panel $p shape $(size(panel_q)) cannot provide an " *
            "interior with halo_width=$Hp"))
        kernel = _cs_tracer_mass_to_vmr_4d_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz, Nt))
    end
    synchronize(backend)
    return q_raw
end

function _cs_scale_vmr_to_tracer_mass!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray}
    Hp = Int(halo_width)
    backend = get_backend(q_raw[1])
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q) == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere tracer panel $p shape $(size(panel_q)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nc > 0 && Ny > 0 || throw(DimensionMismatch(
            "cubed-sphere panel $p shape $(size(panel_q)) cannot provide an " *
            "interior with halo_width=$Hp"))
        kernel = _cs_vmr_to_tracer_mass_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz))
    end
    synchronize(backend)
    return q_raw
end

function _cs_scale_vmr_to_tracer_mass!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       halo_width::Integer) where {A <: AbstractArray{<:Any, 4}}
    Hp = Int(halo_width)
    backend = get_backend(q_raw[1])
    @inbounds for p in 1:6
        panel_q = q_raw[p]
        panel_m = air_mass[p]
        size(panel_q)[1:3] == size(panel_m) || throw(DimensionMismatch(
            "cubed-sphere packed tracer panel $p spatial shape $(size(panel_q)[1:3]) " *
            "does not match air_mass shape $(size(panel_m))"))
        Nc = size(panel_q, 1) - 2 * Hp
        Ny = size(panel_q, 2) - 2 * Hp
        Nz = size(panel_q, 3)
        Nt = size(panel_q, 4)
        Nc > 0 && Ny > 0 || throw(DimensionMismatch(
            "cubed-sphere panel $p shape $(size(panel_q)) cannot provide an " *
            "interior with halo_width=$Hp"))
        kernel = _cs_vmr_to_tracer_mass_4d_kernel!(backend, (8, 8, 1))
        kernel(panel_q, panel_m, Hp; ndrange = (Nc, Ny, Nz, Nt))
    end
    synchronize(backend)
    return q_raw
end

"""
    apply_vertical_diffusion_vmr!(rm, air_mass, op, workspace, dt, meteo; halo_width)

Cubed-sphere helper for state variables stored as tracer mass. The implicit
solver uses the current dry air mass. Precomputed Dkg uses conservative
bidiagonal factors directly on tracer mass. Other fields convert mass to VMR,
apply the column solve, then restore tracer mass before advection resumes.
"""
function apply_vertical_diffusion_vmr!(rm::NTuple{6, A}, air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt, meteo=nothing;
                                       halo_width::Integer) where {
                                           FT, A <: AbstractArray{FT, 3},
                                           KzF <: AbstractCSDkgField{FT}}
    _apply_cs_dkg_mass!(rm, air_mass, op, workspace, dt, meteo, halo_width)
end

function apply_vertical_diffusion_vmr!(rm::NTuple{6, A}, air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt, meteo=nothing;
                                       halo_width::Integer) where {
                                           FT, A <: AbstractArray{FT, 4},
                                           KzF <: AbstractCSDkgField{FT}}
    _apply_cs_dkg_mass!(rm, air_mass, op, workspace, dt, meteo, halo_width)
end

function _apply_cs_dkg_mass!(rm::NTuple{6}, air_mass::NTuple{6}, op,
                             workspace::DiffusionWorkspace, dt, meteo, halo_width)
    FT = eltype(rm[1])
    Hp = Int(halo_width)
    Hp >= 0 || throw(ArgumentError("halo_width must be nonnegative"))
    packed = ndims(rm[1]) == 4
    length(workspace.factors) == 6 || throw(DimensionMismatch("Dkg requires six factor panels"))
    # Check every panel before changing any tracer values.
    for p in 1:6
        N, Ny, Nz, Nt = size(rm[p], 1), size(rm[p], 2), size(rm[p], 3), size(rm[p], 4)
        Nc, Nj = N - 2Hp, Ny - 2Hp
        Nc > 0 && Nj > 0 && Nz > 0 || throw(DimensionMismatch("Dkg panel $p has no physical interior"))
        size(air_mass[p]) == (N, Ny, Nz) || throw(DimensionMismatch("Dkg panel $p tracer and air shapes differ"))
        size(workspace.factors[p]) == (Nc, Nj, Nz) ||
            throw(DimensionMismatch("Dkg factor panel $p must have shape $((Nc, Nj, Nz))"))
    end
    update_field!(op.kz_field, current_time(meteo))
    for p in 1:6
        Nc, Ny = size(rm[p], 1) - 2Hp, size(rm[p], 2) - 2Hp
        Nz, Nt = size(rm[p], 3), size(rm[p], 4)
        backend = get_backend(rm[p])
        dkg = panel_field(op.kz_field, p)
        column_tile = _cs_dkg_mass_workgroupsize(backend, FT)
        tracer_tile = _cs_dkg_tracer_workgroupsize(backend, FT)
        if packed && Nt > 1 && tracer_tile !== nothing
            factor! = _vertical_diffusion_cs_dkg_factors_kernel!(backend, column_tile)
            solve! = _vertical_diffusion_cs_mass_dkg_tracers_kernel!(backend, tracer_tile)
            # Queue the read-only tracer solves after factor construction on
            # the same backend stream. The panel synchronization completes both.
            factor!(workspace.factors[p], air_mass[p], dkg, FT(dt), Nz, Hp;
                    ndrange=(Nc, Ny))
            solve!(rm[p], air_mass[p], dkg, workspace.factors[p], FT(dt), Nz, Hp;
                   ndrange=(Nc, Ny, Nt))
        elseif packed
            kernel! = _vertical_diffusion_cs_mass_dkg_packed_kernel!(backend, column_tile)
            kernel!(rm[p], air_mass[p], dkg, workspace.factors[p], FT(dt), Nz, Nt, Hp; ndrange=(Nc, Ny))
        else
            kernel! = _vertical_diffusion_cs_mass_dkg_kernel!(backend, column_tile)
            kernel!(rm[p], air_mass[p], dkg, workspace.factors[p], FT(dt), Nz, Hp;
                    ndrange=(Nc, Ny))
        end
        synchronize(backend)
    end
    return nothing
end

function apply_vertical_diffusion_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing;
                                       halo_width::Integer) where {FT, A <: AbstractArray{FT, 3},
                                                                    KzF <: AbstractCubedSphereField{FT}}
    _cs_scale_tracer_mass_to_vmr!(q_raw, air_mass, halo_width)
    # Mass-flux kernel needs `air_mass` for coefficient construction.
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo;
                              halo_width = halo_width)
    _cs_scale_vmr_to_tracer_mass!(q_raw, air_mass, halo_width)
    return nothing
end

function apply_vertical_diffusion_vmr!(q_raw::NTuple{6, A},
                                       air_mass::NTuple{6},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing;
                                       halo_width::Integer) where {FT, A <: AbstractArray{FT, 4},
                                                                    KzF <: AbstractCubedSphereField{FT}}
    _cs_scale_tracer_mass_to_vmr!(q_raw, air_mass, halo_width)
    # Mass-flux kernel needs `air_mass` for coefficient construction.
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo;
                              halo_width = halo_width)
    _cs_scale_vmr_to_tracer_mass!(q_raw, air_mass, halo_width)
    return nothing
end

# ---------------------------------------------------------------------------
# Mass-flux LL packed + face-indexed RG paths. These methods take `air_mass`
# explicitly and dispatch to the topology-specific kernel variants.
# ---------------------------------------------------------------------------

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 4},
                                   air_mass::AbstractArray{FT, 3},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace::DiffusionWorkspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 3}}
    w_scratch  = workspace.factors
    dz_scratch = workspace.layer_thickness
    Nx, Ny, Nz, Nt = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (Nx, Ny, Nz),
                                     "spatial")
    size(air_mass) == (Nx, Ny, Nz) || throw(DimensionMismatch(
        "LL mass-flux diffusion: air_mass shape $(size(air_mass)) does not " *
        "match q_raw spatial shape $((Nx, Ny, Nz))"))
    update_field!(op.kz_field, current_time(meteo))
    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_kernel_mass_flux!(backend, (8, 8))
    kernel(q_raw, air_mass, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz, Nt;
           ndrange = (Nx, Ny))
    synchronize(backend)
    return nothing
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 3},
                                   air_mass::AbstractArray{FT, 2},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace::DiffusionWorkspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    w_scratch  = workspace.factors
    dz_scratch = workspace.layer_thickness
    ncells, Nz, Nt = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (ncells, Nz),
                                     "face-indexed")
    size(air_mass) == (ncells, Nz) || throw(DimensionMismatch(
        "RG mass-flux diffusion: air_mass shape $(size(air_mass)) does not " *
        "match q_raw spatial shape $((ncells, Nz))"))
    update_field!(op.kz_field, current_time(meteo))
    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_face_kernel_mass_flux!(backend, 256)
    kernel(q_raw, air_mass, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz, Nt;
           ndrange = ncells)
    synchronize(backend)
    return nothing
end

function apply_vertical_diffusion!(q_raw::AbstractArray{FT, 2},
                                   air_mass::AbstractArray{FT, 2},
                                   op::ImplicitVerticalDiffusion{FT, KzF},
                                   workspace::DiffusionWorkspace, dt,
                                   meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    w_scratch  = workspace.factors
    dz_scratch = workspace.layer_thickness
    ncells, Nz = size(q_raw)
    _check_diffusion_workspace_shape(dz_scratch, w_scratch, (ncells, Nz),
                                     "face-indexed")
    size(air_mass) == (ncells, Nz) || throw(DimensionMismatch(
        "RG mass-flux diffusion: air_mass shape $(size(air_mass)) does not " *
        "match q_raw shape $((ncells, Nz))"))
    update_field!(op.kz_field, current_time(meteo))
    backend = get_backend(q_raw)
    kernel = _vertical_diffusion_face_single_kernel_mass_flux!(backend, 256)
    kernel(q_raw, air_mass, op.kz_field, dz_scratch, w_scratch, FT(dt), Nz;
           ndrange = ncells)
    synchronize(backend)
    return nothing
end

# Mass-VMR scaling helpers for the LL + RG arrays.

function _ll_scale_tracer_mass_to_vmr!(q_raw::AbstractArray{FT, 4},
                                       air_mass::AbstractArray{FT, 3}) where {FT}
    Nx, Ny, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _ll_tracer_mass_to_vmr_kernel!(backend, (8, 8, 1, 1))
    kernel(q_raw, air_mass; ndrange = (Nx, Ny, Nz, Nt))
    synchronize(backend)
    return q_raw
end

function _ll_scale_vmr_to_tracer_mass!(q_raw::AbstractArray{FT, 4},
                                       air_mass::AbstractArray{FT, 3}) where {FT}
    Nx, Ny, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _ll_vmr_to_tracer_mass_kernel!(backend, (8, 8, 1, 1))
    kernel(q_raw, air_mass; ndrange = (Nx, Ny, Nz, Nt))
    synchronize(backend)
    return q_raw
end

function _face_scale_tracer_mass_to_vmr!(q_raw::AbstractArray{FT, 3},
                                         air_mass::AbstractArray{FT, 2}) where {FT}
    ncells, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _face_tracer_mass_to_vmr_kernel!(backend, (256, 1, 1))
    kernel(q_raw, air_mass; ndrange = (ncells, Nz, Nt))
    synchronize(backend)
    return q_raw
end

function _face_scale_vmr_to_tracer_mass!(q_raw::AbstractArray{FT, 3},
                                         air_mass::AbstractArray{FT, 2}) where {FT}
    ncells, Nz, Nt = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _face_vmr_to_tracer_mass_kernel!(backend, (256, 1, 1))
    kernel(q_raw, air_mass; ndrange = (ncells, Nz, Nt))
    synchronize(backend)
    return q_raw
end

function _face_scale_tracer_mass_to_vmr!(q_raw::AbstractArray{FT, 2},
                                         air_mass::AbstractArray{FT, 2}) where {FT}
    ncells, Nz = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _face_single_tracer_mass_to_vmr_kernel!(backend, (256, 1))
    kernel(q_raw, air_mass; ndrange = (ncells, Nz))
    synchronize(backend)
    return q_raw
end

function _face_scale_vmr_to_tracer_mass!(q_raw::AbstractArray{FT, 2},
                                         air_mass::AbstractArray{FT, 2}) where {FT}
    ncells, Nz = size(q_raw)
    backend = get_backend(q_raw)
    kernel = _face_single_vmr_to_tracer_mass_kernel!(backend, (256, 1))
    kernel(q_raw, air_mass; ndrange = (ncells, Nz))
    synchronize(backend)
    return q_raw
end

# LL packed VMR wrapper: pre-scale tracer_mass → VMR, mass-flux solve,
# post-scale VMR → tracer_mass.
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 4},
                                       air_mass::AbstractArray{FT, 3},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 3}}
    _ll_scale_tracer_mass_to_vmr!(q_raw, air_mass)
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo)
    _ll_scale_vmr_to_tracer_mass!(q_raw, air_mass)
    return nothing
end

# Face-indexed RG packed VMR wrapper.
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 3},
                                       air_mass::AbstractArray{FT, 2},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    _face_scale_tracer_mass_to_vmr!(q_raw, air_mass)
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo)
    _face_scale_vmr_to_tracer_mass!(q_raw, air_mass)
    return nothing
end

# Face-indexed RG single-tracer VMR wrapper.
function apply_vertical_diffusion_vmr!(q_raw::AbstractArray{FT, 2},
                                       air_mass::AbstractArray{FT, 2},
                                       op::ImplicitVerticalDiffusion{FT, KzF},
                                       workspace::DiffusionWorkspace, dt,
                                       meteo = nothing) where {FT, KzF <: AbstractTimeVaryingField{FT, 2}}
    _face_scale_tracer_mass_to_vmr!(q_raw, air_mass)
    apply_vertical_diffusion!(q_raw, air_mass, op, workspace, dt, meteo)
    _face_scale_vmr_to_tracer_mass!(q_raw, air_mass)
    return nothing
end
