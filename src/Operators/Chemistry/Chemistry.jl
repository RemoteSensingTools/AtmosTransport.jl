"""
    Chemistry

Source/sink operators for tracer transformations (decay, photolysis, ...).

Type hierarchy:

    AbstractChemistryOperator
    ├── NoChemistry                     — identity / inert tracers
    ├── ExponentialDecay{FT, N}         — multi-tracer first-order decay
    └── CompositeChemistry              — sequential composition

Interface (OPERATOR_COMPOSITION.md §6):

    apply!(state::CellState, meteo, grid, op::AbstractChemistryOperator, dt;
           workspace=nothing)

The operator mutates `state.tracers_raw` in place and returns `state`.
`meteo`, `grid`, and `workspace` are accepted for interface conformance
and may be `nothing` for operators that do not need them (pure decay).

Multi-tracer decay is fused into a single KernelAbstractions kernel —
see `chemistry_kernels.jl`. Tracers not listed in the operator's
`tracer_names` are left untouched.
"""
module Chemistry

using KernelAbstractions: get_backend, synchronize

using ...State: CellState, CubedSphereState
using ...State: ntracers, tracer_index, tracer_names
using ...State: AbstractTimeVaryingField, ConstantField, field_value, update_field!
using ...MetDrivers: current_time
import ..AbstractOperator, ..apply!

export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry

include("chemistry_kernels.jl")

# =========================================================================
# Type hierarchy
# =========================================================================

abstract type AbstractChemistryOperator <: AbstractOperator end

"""
    NoChemistry()

Identity operator — `apply!` is a no-op. Default for runs without active
chemistry.
"""
struct NoChemistry <: AbstractChemistryOperator end

"""
    ExponentialDecay{FT, N, R}(decay_rates, tracer_names)

Multi-tracer first-order decay: `c *= exp(-rate * dt)` applied in-place
to every selected tracer at every cell. Exact for constant rate and any
`dt`; unconditionally stable; trivially parallel.

# Fields
- `decay_rates  :: R` — an `NTuple{N, <: AbstractTimeVaryingField{FT, 0}}`
  of rate-valued fields, one per selected tracer [1/s]. `apply!` calls
  `update_field!` on each rate before launching the kernel.
- `tracer_names :: NTuple{N, Symbol}` — which tracers this operator applies to

# Construction
```julia
ExponentialDecay(; Rn222 = 330_350.4)                   # from half-lives [s]
ExponentialDecay(Float32; Rn222 = 330_350.4, Kr85 = 3.394e8)
```
The keyword constructor converts half-life `T` to decay rate
`λ = log(2) / T` (first-order exponential decay) and wraps each rate
in a `ConstantField{FT, 0}`. Future plans may pass time-varying rates
(e.g. temperature-dependent reaction rates) through the same field
interface.

Common isotopes:
- ²²²Rn: half-life = 330_350.4 s (3.8235 days) → λ ≈ 2.098e-6 s⁻¹
- ⁸⁵Kr:  half-life = 3.394e8 s (10.76 years)  → λ ≈ 2.042e-9 s⁻¹
"""
struct ExponentialDecay{FT, N, R} <: AbstractChemistryOperator
    decay_rates  :: R
    tracer_names :: NTuple{N, Symbol}

    function ExponentialDecay{FT, N, R}(rates::R, names::NTuple{N, Symbol}) where {FT, N, R}
        R <: NTuple{N, AbstractTimeVaryingField{FT, 0}} ||
            throw(ArgumentError("ExponentialDecay: decay_rates must be an " *
                "NTuple{$N, <:AbstractTimeVaryingField{$FT, 0}}, got $R"))
        return new{FT, N, R}(rates, names)
    end
end

"Keyword constructor: `ExponentialDecay(; Rn222 = half_life_seconds, ...)`."
function ExponentialDecay(FT::Type{<:AbstractFloat} = Float64; half_lives...)
    nt = NamedTuple(half_lives)
    names = keys(nt)
    N = length(names)
    N > 0 || throw(ArgumentError("ExponentialDecay requires at least one tracer half-life"))
    for name in names
        half_life = nt[name]
        half_life isa Real && isfinite(half_life) && half_life > 0 ||
            throw(ArgumentError("ExponentialDecay half-life for $(name) must be finite and positive; got $(repr(half_life))"))
    end
    rates = ntuple(i -> ConstantField{FT, 0}(FT(log(2) / nt[i])), N)
    return ExponentialDecay{FT, N, typeof(rates)}(rates, names)
end

"""
    CompositeChemistry(schemes...)
    CompositeChemistry(schemes::Tuple)

Apply multiple chemistry operators sequentially. Used when different
species need independent transformations or when different operator
types (decay + photolysis + ...) must run in a prescribed order.

```julia
chem = CompositeChemistry(
    ExponentialDecay(; Rn222 = 330_350.4),
    ExponentialDecay(; Kr85  = 3.394e8),
)
```
"""
struct CompositeChemistry{S <: Tuple} <: AbstractChemistryOperator
    schemes :: S
end

CompositeChemistry(schemes::AbstractChemistryOperator...) = CompositeChemistry(schemes)

# =========================================================================
# apply! dispatch
# =========================================================================

"""
    apply!(state::CellState, meteo, grid, op::NoChemistry, dt; workspace=nothing)

No-op — returns `state` unchanged.
"""
function apply!(state, meteo, grid, ::NoChemistry, dt;
                workspace = nothing)
    return state
end

"""
    decay_decrement(FT, rate, dt) -> FT

`expm1(-rate · dt)` evaluated in Float64 and rounded once. In Float32,
`exp(-rate · dt)` carries a relative error of up to ~3e-5 in the decayed
fraction `1 − exp(-rate · dt)` (Rn-222, dt = 300–900 s), a fixed bias of the
tracer lifetime.
"""
decay_decrement(::Type{FT}, rate, dt) where FT <: AbstractFloat =
    FT(expm1(-Float64(rate) * Float64(dt)))

# Refresh each rate for the Float64 model clock and form the decrements the
# kernel applies. `ConstantField.update_field!` ignores `t`; time-varying rate
# fields (e.g. `StepwiseField`) consume it here. Callers without a met driver
# pass `meteo = nothing`, whose `current_time` is 0.0.
_decay_decrements(op::ExponentialDecay{FT, N}, meteo, dt) where {FT, N} = ntuple(N) do n
    rate = op.decay_rates[n]
    update_field!(rate, current_time(meteo))
    decay_decrement(FT, field_value(rate, ()), dt)
end

"""
    apply!(state::CellState, meteo, grid, op::ExponentialDecay, dt; workspace=nothing)

Decay every tracer listed in `op.tracer_names` by `exp(-rate * dt)` in
place. `meteo`, `grid`, and `workspace` are unused (accepted for
interface conformance with other operators).

Throws `ArgumentError` if any name in `op.tracer_names` is not carried by
`state`.
"""
function apply!(state::CellState, meteo, grid,
                op::ExponentialDecay{FT, N}, dt;
                workspace = nothing) where {FT, N}
    N == 0 && return state

    # Resolve names → indices at call time.
    indices = ntuple(N) do n
        idx = tracer_index(state, op.tracer_names[n])
        if idx === nothing
            throw(ArgumentError("ExponentialDecay: tracer $(op.tracer_names[n]) " *
                "not present in state (tracer_names = $(tracer_names(state)))"))
        end
        Int32(idx)
    end

    decrements = _decay_decrements(op, meteo, dt)

    raw = state.tracers_raw
    backend = get_backend(raw)
    kernel! = _exp_decay_kernel!(backend, 256)

    # Launch across the spatial axes; the trailing tracer axis is handled
    # by the kernel's inner loop over `indices`.
    spatial_shape = ntuple(i -> size(raw, i), ndims(raw) - 1)
    kernel!(raw, indices, decrements, Int32(N);
            ndrange = spatial_shape)
    synchronize(backend)
    return state
end

"""
    apply!(state::CellState, meteo, grid, op::CompositeChemistry, dt; workspace=nothing)

Apply each sub-operator in order.
"""
function apply!(state::CellState, meteo, grid,
                op::CompositeChemistry, dt;
                workspace = nothing)
    for sub in op.schemes
        apply!(state, meteo, grid, sub, dt; workspace = workspace)
    end
    return state
end

# =========================================================================
# CubedSphereState dispatches
#
# CS tracer storage is an NTuple{6, Array{FT, 4}} of panel-native arrays
# with shape `(Nc + 2Hp, Nc + 2Hp, Nz, Nt)`. Chemistry is pointwise, so
# each panel can run the same rank-agnostic decay kernel as the CellState
# path; we just loop over the six panels and launch per-panel.
#
# Halo cells are decayed alongside interior cells. This is consistent:
# halos hold mirror copies of neighbor-panel interior values; applying
# the same decay uniformly preserves that mirror relationship until the
# next halo exchange refreshes them anyway.
# =========================================================================

"""
    apply!(state::CubedSphereState, meteo, grid, op::ExponentialDecay, dt;
           workspace=nothing)

Decay every tracer listed in `op.tracer_names` by `exp(-rate * dt)` in
place across all six CS panels. Halo cells are decayed alongside interior
cells (see module comment).
"""
function apply!(state::CubedSphereState, meteo, grid,
                op::ExponentialDecay{FT, N}, dt;
                workspace = nothing) where {FT, N}
    N == 0 && return state

    indices = ntuple(N) do n
        idx = tracer_index(state, op.tracer_names[n])
        if idx === nothing
            throw(ArgumentError("ExponentialDecay: tracer $(op.tracer_names[n]) " *
                "not present in state (tracer_names = $(tracer_names(state)))"))
        end
        Int32(idx)
    end

    decrements = _decay_decrements(op, meteo, dt)

    # Launch once per panel. One panel's raw has shape (Nx, Ny, Nz, Nt);
    # ndrange iterates over (Nx, Ny, Nz) and the kernel's inner loop
    # handles the tracer axis.
    raws = state.tracers_raw
    backend = get_backend(raws[1])
    kernel! = _exp_decay_kernel!(backend, 256)

    N_i32 = Int32(N)

    @inbounds for p in 1:6
        raw = raws[p]
        spatial_shape = ntuple(i -> size(raw, i), ndims(raw) - 1)
        kernel!(raw, indices, decrements, N_i32;
                ndrange = spatial_shape)
    end
    synchronize(backend)
    return state
end

"""
    apply!(state::CubedSphereState, meteo, grid, ::NoChemistry, dt; workspace=nothing)

No-op — returns `state` unchanged.
"""
function apply!(state::CubedSphereState, meteo, grid, ::NoChemistry, dt;
                workspace = nothing)
    return state
end

"""
    apply!(state::CubedSphereState, meteo, grid, op::CompositeChemistry, dt;
           workspace=nothing)

Apply each sub-operator in order.
"""
function apply!(state::CubedSphereState, meteo, grid,
                op::CompositeChemistry, dt;
                workspace = nothing)
    for sub in op.schemes
        apply!(state, meteo, grid, sub, dt; workspace = workspace)
    end
    return state
end

end # module Chemistry
