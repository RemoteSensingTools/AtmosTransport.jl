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
import ..apply!

export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry
export AtmosChemistryOperator
export AbstractChemistryForcingProvider, ConstantChemistryForcing
export CallUpdatedChemistryForcing
export CompositeChemistryPlan, CompositeChemistryWorkspace
export chemistry_workspace_plan
export chemistry_workspace_storage_bytes
export chemistry_block!

include("chemistry_kernels.jl")

# =========================================================================
# Type hierarchy
# =========================================================================

abstract type AbstractChemistryOperator end

"Memory plan for optional chemistry coupling workspaces."
function chemistry_workspace_plan end

"Measured storage owned by an allocated optional chemistry workspace."
function chemistry_workspace_storage_bytes end

"How thermodynamic, fixed-species, and photolysis forcing is refreshed."
abstract type AbstractChemistryForcingProvider end

"Forcing captured when the chemistry workspace is constructed."
struct ConstantChemistryForcing{F} <: AbstractChemistryForcingProvider
    forcing :: F
end

"Forcing evaluated once at each chemistry call from an initial schema value."
struct CallUpdatedChemistryForcing{F, U} <: AbstractChemistryForcingProvider
    initial :: F
    update  :: U
end

chemistry_forcing_provider(provider::AbstractChemistryForcingProvider) = provider
chemistry_forcing_provider(forcing) = ConstantChemistryForcing(forcing)
initial_chemistry_forcing(provider::ConstantChemistryForcing) = provider.forcing
initial_chemistry_forcing(provider::CallUpdatedChemistryForcing) = provider.initial
chemistry_forcing(provider::ConstantChemistryForcing, meteo, grid) =
    provider.forcing
chemistry_forcing(provider::CallUpdatedChemistryForcing, meteo, grid) =
    provider.update(provider.initial, meteo, grid)

"""
    AtmosChemistryOperator(model, forcing; tracer_names, workspace_policy,
                           dry_air_molar_mass=28.9647)

Typed coupling contract for the optional AtmosChemistry extension. The
compiled chemistry model determines the active species set; `tracer_names`
maps those species, in mechanism order, to packed transport tracers.

Passing a `ChemistryForcing` value gives constant forcing. Wrap forcing in
`CallUpdatedChemistryForcing(initial, update)` when meteorology or photolysis
must be evaluated once per chemistry call. Updated forcing must preserve the
initial field names and scalar/spatial layout so its preallocated backend
storage and memory plan remain valid.
"""
struct AtmosChemistryOperator{M, F <: AbstractChemistryForcingProvider,
                             Names <: Tuple, W, FT <: AbstractFloat} <:
       AbstractChemistryOperator
    model              :: M
    forcing_provider   :: F
    tracer_names       :: Names
    workspace_policy   :: W
    dry_air_molar_mass :: FT
end

function AtmosChemistryOperator(model, forcing;
                                tracer_names = nothing,
                                workspace_policy = nothing,
                                dry_air_molar_mass::Real = 28.9647)
    tracer_names === nothing && throw(ArgumentError(
        "tracer_names are required unless an optional chemistry extension " *
        "provides them from its model"))
    names = Tuple(Symbol(name) for name in tracer_names)
    length(unique(names)) == length(names) ||
        throw(ArgumentError("chemistry tracer names must be unique"))
    FT = eltype(model)
    FT <: AbstractFloat || throw(ArgumentError(
        "chemistry model must define an AbstractFloat element type"))
    isfinite(dry_air_molar_mass) && dry_air_molar_mass > 0 ||
        throw(ArgumentError(
            "dry_air_molar_mass must be finite and positive"))
    provider = chemistry_forcing_provider(forcing)
    return AtmosChemistryOperator{typeof(model), typeof(provider), typeof(names),
                                  typeof(workspace_policy), FT}(
        model, provider, names, workspace_policy, FT(dry_air_molar_mass))
end

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

"One preallocated workspace for each operator in a `CompositeChemistry`."
struct CompositeChemistryWorkspace{W <: Tuple}
    workspaces :: W
end

"Aggregate memory preflight plus one plan per `CompositeChemistry` operator."
struct CompositeChemistryPlan{P <: Tuple}
    plans              :: P
    backend_bytes      :: Int
    host_bytes         :: Int
    total_bytes        :: Int
    peak_backend_bytes :: Int
    budget_bytes       :: Int
end

chemistry_plan_accounting(::Nothing) =
    (; backend = 0, host = 0, peak_backend = 0, budget = typemax(Int))

function chemistry_plan_accounting(plan)
    throw(ArgumentError(
        "$(typeof(plan)) must implement chemistry_plan_accounting so it can " *
        "participate in CompositeChemistry memory preflight"))
end

function CompositeChemistryPlan(plans::Tuple)
    reports = map(chemistry_plan_accounting, plans)
    backend_bytes = sum(report.backend for report in reports; init = 0)
    host_bytes = sum(report.host for report in reports; init = 0)
    prefix = 0
    peak_backend_bytes = 0
    for report in reports
        peak_backend_bytes = max(
            peak_backend_bytes, prefix + report.peak_backend)
        prefix += report.backend
    end
    budget_bytes = minimum(
        (report.budget for report in reports); init = typemax(Int))
    peak_backend_bytes <= budget_bytes || throw(ArgumentError(
        "composite chemistry workspace exceeds its shared backend budget: " *
        "budget=$budget_bytes, steady=$backend_bytes, " *
        "construction_peak=$peak_backend_bytes"))
    return CompositeChemistryPlan(
        plans, backend_bytes, host_bytes, backend_bytes + host_bytes,
        peak_backend_bytes, budget_bytes)
end

chemistry_workspace_plan(
    ::Union{NoChemistry, ExponentialDecay}, state, grid) = nothing

function chemistry_workspace_plan(
        operator::CompositeChemistry, state, grid)
    plans = map(operator.schemes) do scheme
        chemistry_workspace_plan(scheme, state, grid)
    end
    return CompositeChemistryPlan(plans)
end

_workspace_storage_bytes(::Nothing) =
    (; chemistry = 0, adapter = 0, total = 0)
_workspace_storage_bytes(workspace) =
    chemistry_workspace_storage_bytes(workspace)

function chemistry_workspace_storage_bytes(
        workspace::CompositeChemistryWorkspace)
    reports = map(_workspace_storage_bytes, workspace.workspaces)
    chemistry = sum(report.chemistry for report in reports; init = 0)
    adapter = sum(report.adapter for report in reports; init = 0)
    return (; chemistry, adapter, total = chemistry + adapter,
            components = reports)
end

function _composite_workspaces(workspace::CompositeChemistryWorkspace,
                               count)
    length(workspace.workspaces) == count || throw(DimensionMismatch(
        "composite chemistry workspace count does not match its operators"))
    return workspace.workspaces
end

_composite_workspaces(::Nothing, count) = ntuple(_ -> nothing, count)

_apply_composite!(state, meteo, grid, ::Tuple{}, ::Tuple{}, dt) = state

function _apply_composite!(state, meteo, grid, schemes::Tuple,
                           workspaces::Tuple, dt)
    apply!(state, meteo, grid, first(schemes), dt;
           workspace = first(workspaces))
    return _apply_composite!(state, meteo, grid, Base.tail(schemes),
                             Base.tail(workspaces), dt)
end

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

    # Refresh rate caches for the current time, then materialize to scalars
    # for the kernel. `ConstantField.update_field!` ignores `t`; once
    # non-constant rate fields (e.g. `StepwiseField{FT, 0}` for time-varying
    # decay rates) are wired in, this is where they consume simulation time.
    # For `meteo === nothing` (test fixtures, direct TransportModel callers
    # without a met driver), fall back to `zero(FT)`; the stub at
    # `AbstractMetDriver.jl:77` already returns `0.0` so concrete drivers
    # wanting to drive time-varying fields must override.
    t = meteo === nothing ? zero(FT) : FT(current_time(meteo))
    rates = ntuple(N) do n
        r = op.decay_rates[n]
        update_field!(r, t)
        field_value(r, ())
    end

    raw = state.tracers_raw
    backend = get_backend(raw)
    kernel! = _exp_decay_kernel!(backend, 256)

    # Launch across the spatial axes; the trailing tracer axis is handled
    # by the kernel's inner loop over `indices`.
    spatial_shape = ntuple(i -> size(raw, i), ndims(raw) - 1)
    kernel!(raw, indices, rates, FT(dt), Int32(N);
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
    workspaces = _composite_workspaces(workspace, length(op.schemes))
    return _apply_composite!(
        state, meteo, grid, op.schemes, workspaces, dt)
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

    t = meteo === nothing ? zero(FT) : FT(current_time(meteo))
    rates = ntuple(N) do n
        r = op.decay_rates[n]
        update_field!(r, t)
        field_value(r, ())
    end

    # Launch once per panel. One panel's raw has shape (Nx, Ny, Nz, Nt);
    # ndrange iterates over (Nx, Ny, Nz) and the kernel's inner loop
    # handles the tracer axis.
    raws = state.tracers_raw
    backend = get_backend(raws[1])
    kernel! = _exp_decay_kernel!(backend, 256)

    dt_FT = FT(dt)
    N_i32 = Int32(N)

    @inbounds for p in 1:6
        raw = raws[p]
        spatial_shape = ntuple(i -> size(raw, i), ndims(raw) - 1)
        kernel!(raw, indices, rates, dt_FT, N_i32;
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
    workspaces = _composite_workspaces(workspace, length(op.schemes))
    return _apply_composite!(
        state, meteo, grid, op.schemes, workspaces, dt)
end

# =========================================================================
# chemistry_block! — step-level block composer
# =========================================================================

include("chemistry_block.jl")

end # module Chemistry
