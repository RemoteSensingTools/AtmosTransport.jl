# ---------------------------------------------------------------------------
# Tape records for the CS adjoint reverse loop.
#
# Each forward-recorded operation pushes one record type onto the tape;
# the reverse-loop driver in `src/Footprint/ReverseLoop.jl` dispatches on
# these record types and calls the appropriate per-physics adjoint kernel.
#
# Notes on the scheme parameter:
#
# The record type parameter `S` is intentionally unconstrained in this
# module. The `CSAdjointSupportedScheme` union that enumerates the
# concrete scheme types with adjoint kernels lives in `src/Adjoints/`
# and is enforced at the **construction call sites** (`_record_sweep!`
# in Footprint) and the **kernel dispatch** (`_adjoint_scheme_sweep!`
# in Adjoints). Keeping `_CSSweepRecord` parametric on `S::Any` here
# avoids a circular dependency between `Tape` (which would otherwise
# need to import the union from `Adjoints`) and `Adjoints` (which
# constructs these records).
#
# `_CSLinRoodHorizRecord` is defined separately in
# `src/Adjoints/LinRoodTape.jl` and is
# **not** part of the `_CSTapeOp` union below. The reverse-loop driver
# handles it via an explicit branch alongside the union dispatch.
# ---------------------------------------------------------------------------

"""
    _CSSweepRecord{FT, T, R, F3, S}

Forward-recorded per-direction advection sweep. `direction ∈ (:x, :y, :z)`,
`scheme` is the advection scheme used, `panels_m` / `panels_rm` are the
staged air-mass / tracer-mass tape slots (the latter is `nothing` for
linear schemes that don't need a tracer tape), `panels_flux` is the
per-panel face-flux tuple, `flux_scale` is the scaling applied during
the subcycle.
"""
struct _CSSweepRecord{FT, T, R, F3, S}
    direction::Symbol
    scheme::S
    panels_m::T
    panels_rm::R
    panels_flux::NTuple{6, F3}
    flux_scale::FT
end

"""
    _CSHaloRecord(dir)

Forward-recorded cross-panel halo fill (`dir ∈ (0, 1, 2)`: full / X-only / Y-only).
The reverse pass applies the adjoint of the halo fill at the same `dir`.
"""
struct _CSHaloRecord
    dir::Int
end

"""
    _CSMidpointRecord(step)

Marker record at the Strang-palindrome midpoint of model step `step`.
The reverse pass accumulates a surface footprint snapshot at this op.
"""
struct _CSMidpointRecord
    step::Int
end

"""
    _CSStepEndRecord(step)

Marker record at the end of model step `step`, after convection: the state
that runtime output and the forward observation sampler see. Every recorder
pushes one per step. The reverse pass injects the adjoint forcing of the
observations sampled after step `step` here, so the forcing enters before the
transpose of that step's convection (TM5-4DVAR adds its forcing just before the
adjoint of the operator after which the forward run sampled). Without a forcing
the record is a no-op.
"""
struct _CSStepEndRecord
    step::Int
end

"""
    _CSDiffusionRecord{FT, T, D, W}

Forward-recorded implicit vertical-diffusion application. `op` is the
diffusion operator (e.g., `ImplicitVerticalDiffusion`), `workspace` is the
operator's workspace at this step, `panels_m` is the air-mass tape slot,
`dt` is the (half-)timestep size.
"""
struct _CSDiffusionRecord{FT, T, D, W}
    op::D
    workspace::W
    panels_m::T
    dt::FT
end

"""
    _CSConvectionRecord{FT, T, C, F}

Forward-recorded convection step. `op` is the convection operator
(`CMFMCConvection` or `TM5Convection`), `forcing` is the
`ConvectionForcing` at this step, `panels_m` is the air-mass tape slot,
`dt` is the step size.
"""
struct _CSConvectionRecord{FT, T, C, F}
    op::C
    forcing::F
    panels_m::T
    dt::FT
end

const _CSTapeOp = Union{
    _CSSweepRecord,
    _CSHaloRecord,
    _CSMidpointRecord,
    _CSStepEndRecord,
    _CSDiffusionRecord,
    _CSConvectionRecord,
}
