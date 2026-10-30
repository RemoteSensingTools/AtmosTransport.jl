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
