# ---------------------------------------------------------------------------
# Tape records for the CS adjoint reverse loop.
#
# Each forward-recorded operation pushes one record type onto the tape;
# the reverse-loop driver in `src/Adjoints/Footprint/ReverseLoop.jl` dispatches on
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
# handled directly by the reverse-loop driver. The driver
# handles it via an explicit branch alongside the union dispatch.
# ---------------------------------------------------------------------------

"""
    _CSSweepRecord{FT, T, R, F3, S}

Forward-recorded per-direction advection sweep. `direction ∈ (:x, :y, :z)`,
`scheme` is the advection scheme used, `panels_m` is the tape slot of the air
mass entering the sweep, and `panels_tracer` the slot of what the scheme's
reverse reads of the tracer: `nothing` for the linear schemes, the limiter
branch codes of a monotone or CW84 PPM sweep (six `UInt8` panels, read with
`LimiterCodeRead()`), the tracer mass entering an FV3-profile sweep (read with
`TracerMassRead()`). `panels_flux` is the per-panel face-flux tuple,
`flux_scale` the scaling applied during the subcycle.
"""
struct _CSSweepRecord{FT, T, R, F3, S}
    direction::Symbol
    scheme::S
    panels_m::T
    panels_tracer::R
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
    _CSMidpointRecord(step, dt)

Marker record where the surface-flux operator of model step `step` adds `dt`
worth of emissions (the palindrome center). The reverse pass accumulates the
step's surface footprint `dt · λ` at this op.
"""
struct _CSMidpointRecord{FT}
    step::Int
    dt::FT
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
`dt` is the (half-)timestep size and `time` the clock (`current_time`) at
which the forward evaluated the operator's Kz field.
"""
struct _CSDiffusionRecord{FT, T, D, W}
    op::D
    workspace::W
    panels_m::T
    dt::FT
    time::Float64
end

"""
    _CSConvectionRecord{FT, T, C, F}

Forward-recorded convection application. `op` is the convection operator
(`CMFMCConvection`, `TM5Convection` or `CMFMCMatrixConvection`), `forcing` is
the `ConvectionForcing` it read, `panels_m` is the air-mass tape slot, `dt`
the interval it integrated over (the advection step or the met window) and
`substeps` the CFL sub-step count the forward call used (`CMFMCConvection`;
1 for the others). The convection reverse reads interior cells only, so a
split-sweep tape's record shares the slot of the next air mass staged with
the same interior: the next step's first sweep, or the old air mass of a
`:preserve_vmr` reset.
"""
struct _CSConvectionRecord{FT, T, C, F}
    op::C
    forcing::F
    panels_m::T
    dt::FT
    substeps::Int
end

"""
    _CSAirMassResetRecord(mode, panels_old_m, panels_new_m)

Forward-recorded air-mass reset to a met target. `:preserve_tracer_mass`
leaves the tracer mass unchanged (no staged air masses); `:preserve_vmr`
scales it by `new_m / old_m` and stages both air masses.
"""
struct _CSAirMassResetRecord{T}
    mode::Symbol
    panels_old_m::T
    panels_new_m::T
end
