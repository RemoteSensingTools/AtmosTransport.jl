# Model-facing apply! entry points of the advection operators (structured, face-indexed and cubed-sphere states).
# Split from StrangSplitting.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# apply! entry points
# =========================================================================

"""
    apply!(state, fluxes, grid, scheme, dt; workspace)

Structured-mesh advection entry point.  Delegates to [`strang_split!`](@ref).

The `dt` argument is not used inside the kernels. Callers are responsible
for preparing `fluxes` so they already represent the intended substep forcing.
"""
function apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                grid::AtmosGrid{<:LatLonMesh},
                scheme::AbstractAdvectionScheme, dt;
                workspace::AdvectionWorkspace,
                diffusion_workspace = nothing,
                cfl_limit::Real = one(eltype(state.air_mass)),
                diffusion_op::AbstractDiffusion = NoDiffusion(),
                emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                meteo = nothing) where {B <: AbstractMassBasis}
    strang_split!(state, fluxes, grid, scheme;
                  workspace = workspace, cfl_limit = cfl_limit,
                  diffusion_workspace = diffusion_workspace,
                  diffusion_op = diffusion_op,
                  emissions_op = emissions_op,
                  meteo = meteo, dt = dt)
    return nothing
end

# NoAdvection no-op apply! — LatLon. The workspace, cfl_limit, and
# meteo arguments are accepted but ignored when diffusion is also
# disabled, so callers (e.g. `transport_step!` in
# `Models/TransportModel.jl`) can dispatch without branching on the
# scheme. NoAdvection + non-trivial diffusion is supported: a single
# V(dt) step is applied via the mass-flux VMR wrapper, which is
# mass-conserving on its own and needs no Strang palindrome. Surface
# emissions remain rejected because they are Strang half-steps
# wrapping the advection block.
@inline function apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                         grid::AtmosGrid{<:LatLonMesh},
                         ::NoAdvection, dt;
                         workspace = nothing,
                         diffusion_workspace = nothing,
                         cfl_limit::Real = one(eltype(state.air_mass)),
                         diffusion_op::AbstractDiffusion = NoDiffusion(),
                         emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                         meteo = nothing) where {B <: AbstractMassBasis}
    _noadvection_reject_emissions_op(emissions_op)
    _preflight_diffusion(diffusion_op, diffusion_workspace, dt,
                         state.air_mass, ntracers(state))
    if !(diffusion_op isa NoDiffusion)
        # NoAdvection + diffusion: skip the Strang half-step structure
        # entirely and apply a single V(dt) step on the LL state. The
        # mass-flux VMR wrapper preserves Σ tracer_mass per column to
        # roundoff, so this is the natural "diffusion-only"
        # experimental setup.
        apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass,
                                       diffusion_op, diffusion_workspace, dt, meteo)
    end
    return nothing
end

# Emission rejection still stands when NoAdvection is selected — surface
# fluxes require the Strang half-step wrap around the advection step to
# preserve 2nd-order accuracy. Diffusion alone is fine (it's mass-
# conserving on its own via the VMR wrapper, no symmetric-splitting
# requirement).
@inline function _noadvection_reject_emissions_op(emissions_op::AbstractSurfaceFluxOperator)
    if !(emissions_op isa NoSurfaceFlux)
        throw(ArgumentError(
            "NoAdvection is incompatible with non-NoSurfaceFlux: surface " *
            "emissions are applied as Strang half-steps wrapping the " *
            "advection palindrome, so running them without advection " *
            "drops the symmetric splitting. Remove `[tracers.*.surface_flux]` " *
            "blocks for a convection-only or diffusion-only run."))
    end
    return nothing
end

# Validate the direct operator API before any advection sweep mutates state.
# TransportModel supplies this workspace automatically; direct callers must
# make the operator's storage ownership explicit.
@inline _preflight_diffusion(::NoDiffusion, _, _, ::AbstractArray, ::Integer) = nothing
@inline _preflight_diffusion(::NoDiffusion, _, _, ::NTuple{6}, ::Integer, ::Integer) = nothing

function _preflight_diffusion(op::AbstractDiffusion, workspace, dt,
                              air_mass::AbstractArray, n_tracers::Integer)
    workspace isa DiffusionWorkspace || throw(ArgumentError(
        "$(typeof(op)) requires `diffusion_workspace = DiffusionWorkspace(state)`; " *
        "got $(typeof(workspace))."))
    dt isa Real || throw(ArgumentError("$(typeof(op)) requires a real-valued `dt`; got $(repr(dt))."))
    expected = size(air_mass)
    size(workspace.factors) == expected || throw(DimensionMismatch(
        "diffusion factor workspace has shape $(size(workspace.factors)); expected $(expected)"))
    size(workspace.layer_thickness) == expected || throw(DimensionMismatch(
        "diffusion layer-thickness workspace has shape $(size(workspace.layer_thickness)); expected $(expected)"))
    eltype(workspace.factors) === eltype(air_mass) &&
        eltype(workspace.layer_thickness) === eltype(air_mass) || throw(ArgumentError(
            "diffusion workspace and state must share one element type"))
    typeof(get_backend(workspace.factors)) === typeof(get_backend(air_mass)) &&
        typeof(get_backend(workspace.layer_thickness)) === typeof(get_backend(air_mass)) ||
        throw(ArgumentError("diffusion workspace and state must use the same backend"))
    _packed_references(workspace, expected[1:end-1], n_tracers, eltype(air_mass), get_backend(air_mass))
    return nothing
end

function _preflight_advection_workspace(workspace::AdvectionWorkspace,
                                        tracers_raw::AbstractArray{FT, 4},
                                        air_mass::AbstractArray{FT, 3}) where FT
    tracer_shape = size(tracers_raw)
    mass_shape = size(air_mass)
    size(workspace.rm_4d_A) == tracer_shape &&
        size(workspace.rm_4d_B) == tracer_shape || throw(DimensionMismatch(
            "advection tracer workspace has shapes $(size(workspace.rm_4d_A)) and " *
            "$(size(workspace.rm_4d_B)); expected $(tracer_shape). Construct it with " *
            "`AdvectionWorkspace(state)`."))
    all(buffer -> size(buffer) == mass_shape,
        (workspace.rm_A, workspace.m_A, workspace.rm_B, workspace.m_B)) ||
        throw(DimensionMismatch(
            "advection mass workspace must match air-mass shape $(mass_shape)"))
    backend = typeof(get_backend(air_mass))
    all(buffer -> eltype(buffer) === FT && typeof(get_backend(buffer)) === backend,
        (workspace.rm_A, workspace.m_A, workspace.rm_B, workspace.m_B,
         workspace.rm_4d_A, workspace.rm_4d_B)) || throw(ArgumentError(
            "advection workspace and state must share one element type and backend"))
    return nothing
end

function _preflight_diffusion(op::AbstractDiffusion, workspace, dt,
                              air_mass::NTuple{6}, n_tracers::Integer,
                              halo_width::Integer)
    workspace isa DiffusionWorkspace || throw(ArgumentError(
        "$(typeof(op)) requires `diffusion_workspace = DiffusionWorkspace(state)`; " *
        "got $(typeof(workspace))."))
    dt isa Real || throw(ArgumentError("$(typeof(op)) requires a real-valued `dt`; got $(repr(dt))."))
    Hp = Int(halo_width)
    Nxi, Nyi, Nz = size(air_mass[1])
    expected = (Nxi - 2Hp, Nyi - 2Hp, Nz)
    reference_expected = (expected[1], expected[2], Int(n_tracers))
    workspace.factors isa NTuple{6} &&
        workspace.layer_thickness isa NTuple{6} &&
        workspace.references isa NTuple{6} || throw(DimensionMismatch(
            "cubed-sphere diffusion workspace must contain six factor, layer-thickness, and reference panels"))
    @inbounds for p in 1:6
        size(workspace.factors[p]) == expected || throw(DimensionMismatch(
            "diffusion factor workspace panel $p has shape $(size(workspace.factors[p])); expected $(expected)"))
        size(workspace.layer_thickness[p]) == expected || throw(DimensionMismatch(
            "diffusion layer-thickness workspace panel $p has shape $(size(workspace.layer_thickness[p])); expected $(expected)"))
        size(workspace.references[p]) == reference_expected || throw(DimensionMismatch(
            "diffusion reference workspace panel $p has shape $(size(workspace.references[p])); expected $(reference_expected)"))
        eltype(workspace.factors[p]) === eltype(air_mass[p]) &&
            eltype(workspace.layer_thickness[p]) === eltype(air_mass[p]) &&
            eltype(workspace.references[p]) === eltype(air_mass[p]) || throw(ArgumentError(
                "diffusion workspace panel $p and state must share one element type"))
        backend = typeof(get_backend(air_mass[p]))
        typeof(get_backend(workspace.factors[p])) === backend &&
            typeof(get_backend(workspace.layer_thickness[p])) === backend &&
            typeof(get_backend(workspace.references[p])) === backend || throw(ArgumentError(
                "diffusion workspace panel $p and state must use the same backend"))
    end
    return nothing
end

function apply!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                grid::AtmosGrid{<:CubedSphereMesh},
                scheme::AbstractAdvectionScheme, dt;
                workspace::AdvectionWorkspace) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphereMesh remains metadata-only in src; structured advection is only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
end

function apply!(state::CubedSphereState{B}, fluxes::CubedSphereFaceFluxState{B},
                grid::AtmosGrid{<:CubedSphereMesh},
                scheme::AbstractAdvectionScheme, dt;
                workspace,
                diffusion_workspace = nothing,
                cfl_limit::Real = 0.95,
                diffusion_op::AbstractDiffusion = NoDiffusion(),
                emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                meteo = nothing) where {B <: AbstractMassBasis}
    strang_split!(state, fluxes, grid, scheme;
                  workspace = workspace,
                  diffusion_workspace = diffusion_workspace,
                  cfl_limit = cfl_limit,
                  diffusion_op = diffusion_op,
                  emissions_op = emissions_op,
                  meteo = meteo,
                  dt = dt)
    return nothing
end

# NoAdvection no-op apply! — CubedSphere.  Same contract as the LL
# variant above.  The CS workspace is typed as `Any` so callers can
# pass either `nothing` (the natural choice when advection is off)
# or a leftover `CSAdvectionWorkspace` from a previous configuration
# without touching this dispatch.
@inline function apply!(state::CubedSphereState{B}, fluxes::CubedSphereFaceFluxState{B},
                         grid::AtmosGrid{<:CubedSphereMesh},
                         ::NoAdvection, dt;
                         workspace = nothing,
                         diffusion_workspace = nothing,
                         cfl_limit::Real = 0.95,
                         diffusion_op::AbstractDiffusion = NoDiffusion(),
                         emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                         meteo = nothing) where {B <: AbstractMassBasis}
    _noadvection_reject_emissions_op(emissions_op)
    _preflight_diffusion(diffusion_op, diffusion_workspace, dt,
                         state.air_mass, ntracers(state), state.halo_width)
    if !(diffusion_op isa NoDiffusion)
        # NoAdvection + diffusion on CS: single V(dt) step via the
        # mass-flux VMR wrapper. State carries `state.halo_width`.
        apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass,
                                       diffusion_op, diffusion_workspace, dt, meteo;
                                       halo_width = state.halo_width)
    end
    return nothing
end

# NoAdvection no-op apply! — face-indexed (ReducedGaussian).  Same
# contract as the LL / CS variants above.  Defined OUTSIDE the
# `@eval` loop below because that loop only generates methods
# dispatched on `AbstractConstantScheme` (the only face-indexed
# advection scheme that has kernels); `NoAdvection` is a sibling
# subtype and needs its own method.
@inline function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                         grid::AtmosGrid{<:AbstractHorizontalMesh},
                         ::NoAdvection, dt;
                         workspace = nothing,
                         diffusion_workspace = nothing,
                         cfl_limit::Real = one(eltype(state.air_mass)),
                         diffusion_op::AbstractDiffusion = NoDiffusion(),
                         emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                         meteo = nothing) where {B <: AbstractMassBasis}
    _noadvection_reject_emissions_op(emissions_op)
    _preflight_diffusion(diffusion_op, diffusion_workspace, dt,
                         state.air_mass, ntracers(state))
    if !(diffusion_op isa NoDiffusion)
        # NoAdvection + diffusion on face-indexed RG: single V(dt) step.
        apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass,
                                       diffusion_op, diffusion_workspace, dt, meteo)
    end
    return nothing
end

# Face-indexed apply! — shared tracer loop, generated for each topology.
#
# Accepts the same kwarg surface as the structured path
# (`diffusion_op`, `emissions_op`, `meteo`) because `TransportModel.step!`
# forwards them unconditionally.
#
# Reduced-Gaussian now supports both column-local center operators at the
# H → V → (D or D/2 → S → D/2) → V → H palindrome center. The advection
# sweeps remain per-tracer on face-indexed meshes, so the surface-flux
# hook uses the single-tracer `(ncells, Nz)` array-level entry point.
for (scheme_type, h_sweep, v_sweep) in (
    (:AbstractConstantScheme,    :sweep_horizontal!, :sweep_vertical!),
)
    @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                          grid::AtmosGrid{<:AbstractHorizontalMesh},
                          scheme::$scheme_type, dt;
                          workspace::AdvectionWorkspace,
                          diffusion_workspace = nothing,
                          cfl_limit::Real = one(eltype(state.air_mass)),
                          diffusion_op::AbstractDiffusion = NoDiffusion(),
                          emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                          meteo = nothing) where {B <: AbstractMassBasis}
        _preflight_diffusion(diffusion_op, diffusion_workspace, dt,
                             state.air_mass, ntracers(state))
        # The palindrome below couples emissions as V(dt/2) → S(dt) → V(dt/2) only.
        uses_diffusive_surface_flux_boundary(diffusion_op) && throw(ArgumentError(
            "DiffusiveSurfaceFluxBoundary is not implemented for reduced-Gaussian runs; " *
            "use SplitSurfaceFluxCoupling."))
        m = state.air_mass
        hflux, cm = fluxes.horizontal_flux, fluxes.cm
        cfl_limit_ft = convert(eltype(m), cfl_limit)

        n_tr = ntracers(state)
        n_tr == 0 && return nothing

        m_save = n_tr > 1 ? similar(m) : m
        if n_tr > 1
            copyto!(m_save, m)
        end

        # Face-indexed path keeps a per-tracer loop (multi-tracer fusion
        # on unstructured grids is out of scope). Slices are
        # taken as views into state.tracers_raw so the algorithm is
        # identical to the earlier NamedTuple-iterating version; only
        # the data source changed.
        raw = state.tracers_raw
        last_dim = ndims(raw)
        tracer_names = state.tracer_names
        for idx in 1:n_tr
            if idx > 1
                copyto!(m, m_save)
            end

            rm_tracer = selectdim(raw, last_dim, idx)
            tracer_name = tracer_names[idx]

            _sweep_horizontal_face_subcycled!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, cfl_limit_ft)
            _sweep_vertical_face_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
            # Route through the mass-flux VMR wrapper so the
            # Strang-coupled per-tracer diffusion step is column-mass
            # conserving (preserves Σ tracer_mass per column to roundoff).
            # `m` is the current air-mass slice `(ncells, Nz)`; `rm_tracer`
            # is the per-tracer mass slice `(ncells, Nz)`. The wrapper
            # does pre-scale → mass-flux kernel → post-scale internally.
            if emissions_op isa NoSurfaceFlux
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(rm_tracer, m, diffusion_op, diffusion_workspace, dt, meteo)
            else
                half_dt = dt / 2
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(rm_tracer, m, diffusion_op, diffusion_workspace, half_dt, meteo)
                apply_surface_flux!(rm_tracer, emissions_op, workspace, dt, meteo, grid;
                                    tracer_names = (tracer_name,))
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(rm_tracer, m, diffusion_op, diffusion_workspace, half_dt, meteo)
            end
            _sweep_vertical_face_subcycled!(rm_tracer, m, cm, scheme, workspace, cfl_limit_ft)
            _sweep_horizontal_face_subcycled!(rm_tracer, m, hflux, grid.horizontal, scheme, workspace, cfl_limit_ft)
        end

        return nothing
    end
end

# Error stubs for unsupported face-indexed scheme families.
# `kwargs...` also swallows the `diffusion_op` / `emissions_op`
# / `meteo` forwarding; the stubs still throw `ArgumentError` so no
# additional work is needed.
for (scheme_type, label) in (
    (:AbstractLinearScheme,            "linear-reconstruction"),
    (:AbstractQuadraticScheme,         "quadratic-reconstruction"),
)
    @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                          grid::AtmosGrid{<:AbstractHorizontalMesh},
                          scheme::$scheme_type, dt; kwargs...) where {B <: AbstractMassBasis}
        throw(ArgumentError("Face-connected " * $label * " schemes not yet implemented for $(typeof(grid.horizontal))"))
    end
end
