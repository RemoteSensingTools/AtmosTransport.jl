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

When the palindrome outflow budget needs `n > 1` subcycles, the step runs
`(X → Y → Z)^n → midpoint → (Z → Y → X)^n` with every flux divided by `n`
(see [`strang_split_mt!`](@ref)).

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
                       cfl_limit::Real = DEFAULT_CFL_LIMIT,
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

"""
    palindrome_center!(diffuse!, emit!, diffusion_op, emissions_op, dt)

Physics at the center of one transport palindrome of length `dt`. Without a
surface-flux operator (`NoSurfaceFlux`) the center is one vertical diffusion
solve, `diffuse!(dt)` (a no-op for `NoDiffusion`). With one, the diffusion
operator's coupling orders the two through `surface_flux_center!`:
`V(dt/2) → S(dt) → V(dt/2)` (`SplitSurfaceFluxCoupling`) or `S(dt) → V(dt)`
(`DiffusiveSurfaceFluxBoundary`).

`diffuse!(τ)` and `emit!(τ)` apply vertical diffusion and the surface flux
over `τ` to the caller's state. Every production palindrome (cubed-sphere,
lat-lon and reduced-Gaussian) builds its center with this function. The
cubed-sphere adjoint replays a run with a surface-flux operator (a footprint
is the derivative with respect to its emissions), so it calls
`surface_flux_center!` with the same coupling.
"""
@inline palindrome_center!(diffuse!::D, emit!::E, _diffusion_op, ::NoSurfaceFlux,
                           dt) where {D, E} = (diffuse!(dt); nothing)

@inline palindrome_center!(diffuse!::D, emit!::E, diffusion_op,
                           ::AbstractSurfaceFluxOperator, dt) where {D, E} =
    surface_flux_center!(diffuse!, emit!, surface_flux_coupling(diffusion_op), dt)

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
                             rm_4d::NTuple{6, <:AbstractArray{<:Any, 4}},
                             m,
                             fluxes::CubedSphereFaceFluxState,
                             mesh::CubedSphereMesh,
                             scheme::AbstractAdvectionScheme,
                             workspace::CSAdvectionWorkspace;
                             cfl_limit::Real = DEFAULT_CFL_LIMIT,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint! = nothing,
                             recorder::AbstractStepRecorder = NoStepRecorder())
    strang_split_cs_mt!(rm_4d, m, fluxes.am, fluxes.bm, fluxes.cm,
                        mesh, scheme, workspace;
                        cfl_limit = cfl_limit,
                        subcycle_count = subcycle_count,
                        midpoint! = midpoint!,
                        recorder = recorder)
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
                             cfl_limit::Real = DEFAULT_CFL_LIMIT,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint! = nothing,
                             recorder::AbstractStepRecorder = NoStepRecorder()) where ORD
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
                                  vertical = scheme.vertical,
                                  recorder = recorder)
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
                       cfl_limit::Real = DEFAULT_CFL_LIMIT,
                       diffusion_op::AbstractDiffusion = NoDiffusion(),
                       emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                       meteo = nothing,
                       dt::Union{Nothing, Real} = nothing,
                       recorder::AbstractStepRecorder = NoStepRecorder()) where {B <: AbstractMassBasis}
    preflight_diffusion(diffusion_op, diffusion_workspace, dt,
                         state.air_mass, ntracers(state), state.halo_width)
    (!(diffusion_op isa NoDiffusion) || !(emissions_op isa NoSurfaceFlux)) &&
        dt === nothing && throw(ArgumentError(
            "cubed-sphere transport with diffusion or surface flux requires the step dt"))

    n_tr = ntracers(state)
    n_tr == 0 && return nothing

    m = state.air_mass
    subcycle_count = _cs_runtime_subcycle_count(meteo)
    deposit = center_emission_deposit(surface_flux_coupling(diffusion_op), diffusion_op)
    if cs_advection_style(scheme) isa CSSplitSweepStyle
        # Queued only: the first sweep follows on the same device queue.
        fill_cs_step_start_halos!(state.tracers_raw, m, grid.horizontal; sync = false)
        record_halo!(recorder, 1)
        midpoint! = (active_rm, active_m) -> begin
            record_center!(recorder, active_m, diffusion_op)
            palindrome_center!(
                τ -> (record_diffusion!(recorder, diffusion_op, diffusion_workspace, τ);
                      SectionTimer.@section(:diffusion, apply_vertical_diffusion_vmr!(
                          active_rm, active_m, diffusion_op, diffusion_workspace, τ, meteo;
                          halo_width = state.halo_width))),
                τ -> (record_emission!(recorder, τ);
                      apply_surface_flux!(active_rm, emissions_op, workspace, τ, meteo, grid;
                                          tracer_names = state.tracer_names,
                                          halo_width = state.halo_width, deposit)),
                diffusion_op, emissions_op, dt)
        end
        _cs_transport_step!(CSSplitSweepStyle(), state.tracers_raw, m, fluxes,
                            grid.horizontal, scheme, workspace;
                            cfl_limit = cfl_limit,
                            subcycle_count = subcycle_count,
                            midpoint! = midpoint!,
                            recorder = recorder)
        return nothing
    end

    # Lin-Rood transports one tracer at a time from the same initial air mass.
    # Its horizontal operator (`fv_tp_2d_cs!`) refills the edge and corner
    # halos of rm and m before it reads them, so these fills are not read
    # (and a recorder gets no record of them: their output never reaches the
    # result, so their tangent is zero).
    fill_panel_halos!(m, grid.horizontal; dir=1)
    m_save = n_tr > 1 ? _similar_cs_storage(m) : m
    if n_tr > 1
        _copy_cs_storage!(m_save, m)
    end

    tracer_names = state.tracer_names
    for idx in 1:n_tr
        if idx > 1
            _copy_cs_storage!(m, m_save)
        end

        # One tracer at a time: `tracer = idx` selects its diffusion ledger carry.
        rm_tracer = get_tracer(state, idx)
        fill_panel_halos!(rm_tracer, grid.horizontal; dir=1)
        tracer_name = tracer_names[idx]
        midpoint! = () -> begin
            record_center!(recorder, m, diffusion_op)
            palindrome_center!(
                τ -> (record_diffusion!(recorder, diffusion_op, diffusion_workspace, τ);
                      SectionTimer.@section(:diffusion, apply_vertical_diffusion_vmr!(
                          rm_tracer, m, diffusion_op, diffusion_workspace, τ, meteo;
                          halo_width = state.halo_width, tracer = idx))),
                τ -> (record_emission!(recorder, τ);
                      apply_surface_flux!(rm_tracer, emissions_op, workspace, τ, meteo, grid;
                                          tracer_names = (tracer_name,),
                                          halo_width = state.halo_width, deposit)),
                diffusion_op, emissions_op, dt)
        end
        _cs_transport_step!(cs_advection_style(scheme),
                            rm_tracer, m, fluxes, grid.horizontal, scheme, workspace;
                            cfl_limit = cfl_limit,
                            subcycle_count = subcycle_count,
                            midpoint! = midpoint!,
                            recorder = recorder)
    end

    return nothing
end
