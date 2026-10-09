# ---------------------------------------------------------------------------
# Strang splitting orchestrator for structured grids
#
# Performs dimensionally-split advection using the Strang (1968) symmetric
# splitting sequence:  X → Y → Z → Z → Y → X
#
# This second-order splitting eliminates the first-order directional bias
# that would arise from a simple X → Y → Z sequence.  Each half of the
# palindrome advances the state by half a timestep in each direction,
# yielding a full timestep with O(Δt²) splitting error.
#
# The directional sweeps are in sweeps.jl, their CFL subcycling in
# subcycling.jl, the model-facing apply! methods in strang_apply.jl and the
# multi-tracer palindrome in multitracer_strang.jl.
#
# References
# ----------
# - Strang (1968), "On the construction and comparison of difference
#   schemes", SIAM J. Numer. Anal., 5:506–517.
# - Russell & Lerner (1981), "A new finite-differencing scheme for the
#   tracer transport equation", J. Appl. Meteor., 20:1483–1498.
#   → Strang splitting of slopes advection in TM5.
# ---------------------------------------------------------------------------

# =========================================================================
# Strang splitting: X → Y → Z → Z → Y → X
# =========================================================================
#
# The Strang (1968) symmetric splitting sequence achieves second-order
# accuracy in the splitting error while using directional sweeps in the
# palindrome X → Y → Z → Z → Y → X.
#
# In `src`, the kernels consume a fully prepared mass-flux state for the
# active substep.  Any time interpolation or window-to-substep conversion is
# handled upstream by the driver/runtime layer before these sweeps are called.
#
# Multi-tracer handling: each tracer is advected independently through
# the full Strang sequence.  Air mass is saved before the first tracer
# and restored before subsequent tracers, so that each tracer sees the
# same initial mass field (mass changes from one tracer's advection
# must not leak into another tracer's transport).

"""
    strang_split!(state, fluxes, grid, scheme; workspace)

Perform one full Strang-split advection step on a structured mesh.

# Splitting sequence
```
  X → Y → Z → Z → Y → X
  ─────────────────────────
  half    half   full  half    half
```

All `Nt = ntracers(state)` tracers are advanced together in a single
multi-tracer kernel launch per direction. The mass update is computed
once per cell; tracer fluxes are evaluated per-tracer inside the
kernel. The per-tracer Julia loop has been eliminated in favour of
`strang_split_mt!` on the packed `state.tracers_raw` buffer.

# Arguments
- `state::CellState` — contains `air_mass` and `tracers_raw`
- `fluxes::StructuredFaceFluxState` — mass fluxes (am, bm, cm)
- `grid::AtmosGrid{<:LatLonMesh}` — structured lat-lon grid
- `scheme` — advection scheme (`AbstractAdvectionScheme`)
- `workspace::AdvectionWorkspace` — pre-allocated double buffers;
  use `AdvectionWorkspace(state)` so the 4D ping-pong buffers are
  sized for `ntracers(state)`.
"""
function strang_split!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                       grid::AtmosGrid{<:LatLonMesh},
                       scheme::AbstractAdvectionScheme;
                       workspace::AdvectionWorkspace,
                       diffusion_workspace = nothing,
                       cfl_limit::Real = one(eltype(state.air_mass)),
                       diffusion_op::AbstractDiffusion = NoDiffusion(),
                       emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                       meteo = nothing,
                       dt::Union{Nothing, Real} = nothing) where {B <: AbstractMassBasis}
    m = state.air_mass
    am, bm, cm = fluxes.am, fluxes.bm, fluxes.cm

    if ntracers(state) == 0
        return nothing
    end

    strang_split_mt!(state.tracers_raw, m, am, bm, cm, scheme, workspace;
                     cfl_limit = cfl_limit,
                     diffusion_op = diffusion_op,
                     diffusion_workspace = diffusion_workspace,
                     emissions_op = emissions_op,
                     tracer_names = state.tracer_names,
                     meteo = meteo,
                     grid = grid,
                     dt = dt)
    return nothing
end

function strang_split!(state::CellState{B}, fluxes::StructuredFaceFluxState{B},
                       grid::AtmosGrid{<:CubedSphereMesh},
                       scheme::AbstractAdvectionScheme;
                       workspace::AdvectionWorkspace) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphereMesh remains metadata-only in src; structured advection is only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
end

@inline function _copy_cs_storage!(dest::NTuple{6}, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(dest[p], src[p])
    end
    return dest
end

@inline function _similar_cs_storage(src::NTuple{6})
    return ntuple(p -> similar(src[p]), 6)
end

function _cs_transport_step!(::CSSplitSweepStyle,
                             rm_tracer, m,
                             fluxes::CubedSphereFaceFluxState,
                             mesh::CubedSphereMesh,
                             scheme::AbstractAdvectionScheme,
                             workspace::CSAdvectionWorkspace;
                             cfl_limit::Real = 0.95,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint! = nothing)
    strang_split_cs!(rm_tracer, m, fluxes.am, fluxes.bm, fluxes.cm,
                     mesh, scheme, workspace;
                     cfl_limit = cfl_limit,
                     subcycle_count = subcycle_count,
                     midpoint! = midpoint!)
    return nothing
end

function _cs_transport_step!(::CSSplitSweepStyle,
                             rm_4d::NTuple{6, <:AbstractArray{<:Any, 4}},
                             m,
                             fluxes::CubedSphereFaceFluxState,
                             mesh::CubedSphereMesh,
                             scheme::AbstractAdvectionScheme,
                             workspace::CSAdvectionWorkspace;
                             cfl_limit::Real = 0.95,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint! = nothing)
    strang_split_cs_mt!(rm_4d, m, fluxes.am, fluxes.bm, fluxes.cm,
                        mesh, scheme, workspace;
                        cfl_limit = cfl_limit,
                        subcycle_count = subcycle_count,
                        midpoint! = midpoint!)
    return nothing
end

function _cs_transport_step!(::CSSplitSweepStyle,
                             _rm_tracer, _m,
                             _fluxes::CubedSphereFaceFluxState,
                             _mesh::CubedSphereMesh,
                             scheme::AbstractAdvectionScheme,
                             workspace;
                             kwargs...)
    throw(ArgumentError(
        "Cubed-sphere split-sweep advection with $(typeof(scheme)) requires " *
        "`CSAdvectionWorkspace`; got $(typeof(workspace))."))
end

function _cs_transport_step!(::CSLinRoodStyle,
                             rm_tracer, m,
                             fluxes::CubedSphereFaceFluxState,
                             mesh::CubedSphereMesh,
                             scheme::LinRoodPPMScheme{ORD},
                             workspace::CSLinRoodAdvectionWorkspace;
                             cfl_limit::Real = 0.95,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint! = nothing) where ORD
    _ = subcycle_count
    # The runtime driver Hp-pads the flux panels; the LinRood cross-term kernels
    # expect the interior faces only. Strip the halo here (see
    # `_cs_flux_*_interior`) so the kernels read the correct cell. `cm` stays
    # padded — the shared vertical `_sweep_z!` indexes it with the Hp offset like
    # the rest of the CS sweeps.
    Nc, Hp = mesh.Nc, mesh.Hp
    am = ntuple(p -> _cs_flux_x_interior(fluxes.am[p], Nc, Hp), 6)
    bm = ntuple(p -> _cs_flux_y_interior(fluxes.bm[p], Nc, Hp), 6)
    _strang_split_linrood_ppm_cs!(rm_tracer, m, am, bm, fluxes.cm,
                                  mesh, Val(ORD), workspace;
                                  cfl_limit = cfl_limit,
                                  midpoint! = midpoint!,
                                  vertical = scheme.vertical)
    return nothing
end

function _cs_transport_step!(::CSLinRoodStyle,
                             _rm_tracer, _m,
                             _fluxes::CubedSphereFaceFluxState,
                             _mesh::CubedSphereMesh,
                             scheme::LinRoodPPMScheme,
                             workspace;
                             kwargs...)
    throw(ArgumentError(
        "Cubed-sphere Lin-Rood advection with $(typeof(scheme)) requires " *
        "`CSLinRoodAdvectionWorkspace`; got $(typeof(workspace))."))
end

@inline _meteo_driver_for_substeps(::Nothing) = nothing
@inline function _meteo_driver_for_substeps(meteo)
    return hasproperty(meteo, :driver) ? getproperty(meteo, :driver) : meteo
end

@inline function _cs_runtime_subcycle_count(meteo)
    return uses_binary_substep_contract(_meteo_driver_for_substeps(meteo)) ? 1 : nothing
end

function strang_split!(state::CubedSphereState{B}, fluxes::CubedSphereFaceFluxState{B},
                       grid::AtmosGrid{<:CubedSphereMesh},
                       scheme::AbstractAdvectionScheme;
                       workspace,
                       diffusion_workspace = nothing,
                       cfl_limit::Real = 0.95,
                       diffusion_op::AbstractDiffusion = NoDiffusion(),
                       emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                       meteo = nothing,
                       dt::Union{Nothing, Real} = nothing) where {B <: AbstractMassBasis}
    _preflight_diffusion(diffusion_op, diffusion_workspace, dt,
                         state.air_mass, ntracers(state), state.halo_width)
    (!(diffusion_op isa NoDiffusion) || !(emissions_op isa NoSurfaceFlux)) &&
        dt === nothing && throw(ArgumentError(
            "cubed-sphere transport with diffusion or surface flux requires the step dt"))

    n_tr = ntracers(state)
    n_tr == 0 && return nothing

    fill_panel_halos!(state.air_mass, grid.horizontal; dir=1)

    m = state.air_mass
    subcycle_count = _cs_runtime_subcycle_count(meteo)
    if cs_advection_style(scheme) isa CSSplitSweepStyle
        fill_panel_halos!(state.tracers_raw, grid.horizontal; dir=1)
        midpoint! = if emissions_op isa NoSurfaceFlux
            (active_rm, active_m) -> SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
                active_rm, active_m, diffusion_op, diffusion_workspace, dt, meteo;
                halo_width = state.halo_width)
        elseif uses_diffusive_surface_flux_boundary(diffusion_op)
            (active_rm, active_m) -> begin
                apply_surface_flux!(active_rm, emissions_op, workspace, dt, meteo, grid;
                                    tracer_names = state.tracer_names,
                                    halo_width = state.halo_width,
                                    deposit = emission_deposit(diffusion_op))
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
                    active_rm, active_m, diffusion_op, diffusion_workspace, dt, meteo;
                    halo_width = state.halo_width)
            end
        else
            half_dt = dt / 2
            (active_rm, active_m) -> begin
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
                    active_rm, active_m, diffusion_op, diffusion_workspace, half_dt, meteo;
                    halo_width = state.halo_width)
                apply_surface_flux!(active_rm, emissions_op, workspace, dt, meteo, grid;
                                    tracer_names = state.tracer_names,
                                    halo_width = state.halo_width)
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
                    active_rm, active_m, diffusion_op, diffusion_workspace, half_dt, meteo;
                    halo_width = state.halo_width)
            end
        end
        _cs_transport_step!(CSSplitSweepStyle(), state.tracers_raw, m, fluxes,
                            grid.horizontal, scheme, workspace;
                            cfl_limit = cfl_limit,
                            subcycle_count = subcycle_count,
                            midpoint! = midpoint!)
        return nothing
    end

    m_save = n_tr > 1 ? _similar_cs_storage(m) : m
    if n_tr > 1
        _copy_cs_storage!(m_save, m)
    end

    tracer_names = state.tracer_names
    for idx in 1:n_tr
        if idx > 1
            _copy_cs_storage!(m, m_save)
        end

        rm_tracer = get_tracer(state, idx)
        fill_panel_halos!(rm_tracer, grid.horizontal; dir=1)
        tracer_name = tracer_names[idx]
        midpoint! = if emissions_op isa NoSurfaceFlux
            () -> SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
                rm_tracer, m, diffusion_op, diffusion_workspace, dt, meteo;
                halo_width = state.halo_width)
        elseif uses_diffusive_surface_flux_boundary(diffusion_op)
            () -> begin
                apply_surface_flux!(rm_tracer, emissions_op, workspace, dt, meteo, grid;
                                    tracer_names = (tracer_name,),
                                    halo_width = state.halo_width,
                                    deposit = emission_deposit(diffusion_op))
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
                    rm_tracer, m, diffusion_op, diffusion_workspace, dt, meteo;
                    halo_width = state.halo_width)
            end
        else
            half_dt = dt / 2
            () -> begin
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
                    rm_tracer, m, diffusion_op, diffusion_workspace, half_dt, meteo;
                    halo_width = state.halo_width)
                apply_surface_flux!(rm_tracer, emissions_op, workspace, dt, meteo, grid;
                                    tracer_names = (tracer_name,),
                                    halo_width = state.halo_width)
                SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
                    rm_tracer, m, diffusion_op, diffusion_workspace, half_dt, meteo;
                    halo_width = state.halo_width)
            end
        end
        _cs_transport_step!(cs_advection_style(scheme),
                            rm_tracer, m, fluxes, grid.horizontal, scheme, workspace;
                            cfl_limit = cfl_limit,
                            subcycle_count = subcycle_count,
                            midpoint! = midpoint!)
    end

    return nothing
end
