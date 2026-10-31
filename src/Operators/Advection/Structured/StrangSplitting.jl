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
                       cfl_limit::Real = DEFAULT_CFL_LIMIT,
                       midpoint::AbstractMidpointHook = NoMidpoint()) where {B <: AbstractMassBasis}
    m = state.air_mass
    am, bm, cm = fluxes.am, fluxes.bm, fluxes.cm

    if ntracers(state) == 0
        return nothing
    end

    strang_split_mt!(state.tracers_raw, m, am, bm, cm, scheme, workspace;
                     cfl_limit = cfl_limit,
                     midpoint = midpoint)
    return nothing
end
