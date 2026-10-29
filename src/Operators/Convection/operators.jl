# ---------------------------------------------------------------------------
# Convection operator hierarchy + NoConvection default.
# ---------------------------------------------------------------------------

# `AbstractConvection` is the global root declared in
# `src/Operators/AbstractOperators.jl`. Concrete subtypes here:
#
# - `NoConvection` — identity, default.
# - `CMFMCConvection` — GCHP-style RAS / Grell-Freitas transport with
#   CMFMC + optional DTRAIN, mandatory CFL sub-cycling, well-mixed
#   sub-cloud layer.
# - `TM5Convection` — TM5-style four-field matrix transport with
#   in-kernel LU solve.
#
# Every concrete subtype implements
#
#     apply!(state::CellState{B},
#            forcing::ConvectionForcing,
#            grid::AtmosGrid,
#            op,
#            dt::Real;
#            workspace) where {B <: AbstractMassBasis}
#
# mutating `state.tracers_raw` in place and returning `state`. The forcing
# arrives via `TransportModel.convection_forcing`, populated at each window's
# first substep by `DrivenSimulation._refresh_forcing!`. No `meteo` kwarg — unlike
# `ImplicitVerticalDiffusion` (which needs time to refresh Kz) or
# `SurfaceFluxOperator` (which needs time to sample `StepwiseField`
# emission rates), convection forcing IS the time information; the
# operator does not call `current_time`.

"""
    NoConvection()

Identity operator — `apply!` is a no-op. Default for configurations
without active convection. Dispatch is a compile-time dead branch in
`TransportModel.step!`, so the convection block collapses to zero
floating-point work when no operator is installed (bit-exact backward-
compatible with the no-op behavior).
"""
struct NoConvection <: AbstractConvection end

# =========================================================================
# apply!  (state-level, delegates to apply_convection!)
# =========================================================================

"""
    apply!(state, forcing::ConvectionForcing, grid::AtmosGrid,
           ::NoConvection, dt; workspace=nothing)

No-op. Accepts any `ConvectionForcing` (including the all-nothing
placeholder) and any workspace, including `nothing`. Returns `state`
unchanged.
"""
@inline function apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
                        ::NoConvection, dt::Real;
                        workspace = nothing)
    return state
end

@inline function apply!(state::CubedSphereState, forcing::ConvectionForcing,
                        grid::AtmosGrid, ::NoConvection, dt::Real;
                        workspace = nothing)
    return state
end
