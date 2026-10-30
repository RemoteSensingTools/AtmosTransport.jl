"""
    CSTimeSampling(steps, weights)
    CSTimeSampling(step)

The model states an observation samples: the states after `steps` (strictly
increasing, after convection: the state runtime output sees), averaged with
`weights` (positive, summing to 1). The simulated value is
``H = \\sum_s w_s\\, h(\\mathbf{r}_s)``, with ``h`` the objective evaluated on
the state after step ``s``. `CSTimeSampling(step)` samples one state (weight 1).

`bind_to_mesh` builds an instantaneous sampling for a record without a time
window (the step whose interval contains the record time) and, for a record
with `time_window_s > 0`, the steps whose intervals overlap the window with
weights proportional to the overlap (TM5-4DVAR's Δt/ΣΔt window mean).
"""
struct CSTimeSampling
    steps::Vector{Int}
    weights::Vector{Float64}
    function CSTimeSampling(steps::AbstractVector{<:Integer}, weights::AbstractVector{<:Real})
        n = length(steps)
        n >= 1 || throw(ArgumentError("CSTimeSampling needs at least one step"))
        length(weights) == n || throw(DimensionMismatch(
            "CSTimeSampling has $n steps but $(length(weights)) weights"))
        first(steps) >= 1 || throw(ArgumentError(
            "CSTimeSampling steps must be positive, got $(first(steps))"))
        all(>(0), diff(steps)) || throw(ArgumentError(
            "CSTimeSampling steps must be strictly increasing, got $(steps)"))
        all(w -> isfinite(w) && w > 0, weights) || throw(ArgumentError(
            "CSTimeSampling weights must be finite and positive, got $(weights)"))
        total = sum(Float64, weights)
        abs(total - 1) <= 8n * eps(Float64) || throw(ArgumentError(
            "CSTimeSampling weights must sum to 1, got $(total)"))
        return new(collect(Int, steps), collect(Float64, weights))
    end
end

CSTimeSampling(step::Integer) = CSTimeSampling([Int(step)], [1.0])

Base.:(==)(a::CSTimeSampling, b::CSTimeSampling) = a.steps == b.steps && a.weights == b.weights
Base.hash(s::CSTimeSampling, h::UInt) = hash(s.weights, hash(s.steps, hash(:CSTimeSampling, h)))

"""
    CSObservation(step, objective, value, sigma; id = 0, assimilate = true)
    CSObservation(sampling::CSTimeSampling, objective, value, sigma; id = 0, assimilate = true)

Scalar observation of the CS 4D-Var surface-flux path: `objective` sampled
as `sampling` says (one step, or a weighted window of steps; see
[`CSTimeSampling`](@ref)), with observed `value` and error standard
deviation `sigma`. `step` is the last sampled step, after which the
observation is complete (the only one for an instantaneous observation).
`id` is the identifier of the observation record it was bound from
(`bind_to_mesh`), `0` for an observation built by hand.

`assimilate = false` makes it a **co-sample**: `cs_surface_flux_4dvar`
simulates it on the forward replay and reports its simulated value, residual
and stencil (with `assimilated = false`), but it adds nothing to the cost or
the gradient, and does not extend the taped run. `bind_to_mesh` co-samples
records with a nonzero flag under `flagged_records = "cosample"`, and the
members of cell averages ([`CellAverage`](@ref)).
"""
struct CSObservation{O <: AbstractCSFootprintObjective, FT <: AbstractFloat}
    step::Int
    sampling::CSTimeSampling
    objective::O
    value::FT
    sigma::FT
    id::Int64
    assimilate::Bool
    function CSObservation{O, FT}(sampling::CSTimeSampling, objective::O, value::FT, sigma::FT,
                                  id::Int64, assimilate::Bool) where {O <: AbstractCSFootprintObjective,
                                                                      FT <: AbstractFloat}
        isfinite(value) || throw(ArgumentError("CSObservation value must be finite, got $value"))
        isfinite(sigma) || throw(ArgumentError("CSObservation sigma must be finite, got $sigma"))
        sigma > 0 || throw(ArgumentError("CSObservation sigma must be positive, got $sigma"))
        return new{O, FT}(last(sampling.steps), sampling, objective, value, sigma, id, assimilate)
    end
end

function CSObservation(sampling::CSTimeSampling, objective::AbstractCSFootprintObjective,
                       value::Real, sigma::Real; id::Integer = 0, assimilate::Bool = true)
    FT = promote_type(typeof(float(value)), typeof(float(sigma)))
    return CSObservation{typeof(objective), FT}(sampling, objective, FT(value), FT(sigma), Int64(id), assimilate)
end

function CSObservation(step::Integer, objective::AbstractCSFootprintObjective,
                       value::Real, sigma::Real; id::Integer = 0, assimilate::Bool = true)
    step > 0 || throw(ArgumentError("CSObservation step must be positive, got $step"))
    return CSObservation(CSTimeSampling(step), objective, value, sigma; id, assimilate)
end

"""
Stencil placement code of an observation whose objective is a column mean
([`CSColumnMeanObjective`](@ref)): every level of the column, no vertical
placement. The point objectives use the `Observations.VERTICAL_PLACEMENT_CODES`.
"""
const COLUMN_MEAN_PLACEMENT_CODE = Int8(-1)
"""
Stencil placement code of an observation whose objective has a fixed stencil
instead of being placed: a [`CSLayerMeanObjective`](@ref) naming its level,
or another fixed operator.
"""
const FIXED_LEVEL_PLACEMENT_CODE = Int8(-2)
"""
Stencil placement code of an observation whose objective is an
averaging-kernel column ([`CSRetrievalColumnObjective`](@ref)): every level
of the column, weighted by the retrieval's pressure weights and averaging
kernel on the model column stretched onto the retrieval column.
"""
const RETRIEVAL_COLUMN_PLACEMENT_CODE = Int8(-3)
"""
Stencil placement code of a cell average of column records
([`CSCellAverageObjective`](@ref), [`CellAverage`](@ref)): every level of
the cell, the inverse-variance-weighted mean of its members' operators.
"""
const CELL_AVERAGE_PLACEMENT_CODE = Int8(-4)

"""
    CSObservationStencil(panel, i, j, steps, weights, levels, placement_method,
                         height_agl_m, pressure_pa,
                         surface_pressure_mismatch_pa, retrieval_profile, assimilated)

Where an observation was sampled in a 4D-Var evaluation, returned per
observation in [`CS4DVarResult`](@ref)`.stencils`: the cell `(panel, i, j)`,
the sampled `steps` and their `weights` (its [`CSTimeSampling`](@ref)), the
model level sampled at each step (`levels`, k = 1 at the top, resolved from
the replayed column at the end of that step; `0` for an operator that reads
more than one level, as a column mean), the `placement_method`
(`Observations.VERTICAL_PLACEMENT_CODES` for a [`CSPlacedPointObjective`](@ref),
[`COLUMN_MEAN_PLACEMENT_CODE`](@ref),
[`RETRIEVAL_COLUMN_PLACEMENT_CODE`](@ref), [`CELL_AVERAGE_PLACEMENT_CODE`](@ref)
or [`FIXED_LEVEL_PLACEMENT_CODE`](@ref) otherwise), and the height above the
model surface (m) or the pressure (Pa) that placed it (NaN when not used).
An operator without a single cell has the cell `(0, 0, 0)`.

For an averaging-kernel column ([`CSRetrievalColumnObjective`](@ref)):
`surface_pressure_mismatch_pa`, the model minus the retrieval surface
pressure (Pa; time-weighted over the sampled steps; NaN for other
operators); `retrieval_profile`, the model's dry mixing ratios on the
retrieval layers or levels (surface first, the projected profile ``P x``,
time-weighted; empty for other operators).

`assimilated` is `false` for a co-sample (an observation built with
`assimilate = false`: a flagged record under `flagged_records = "cosample"`,
or a member of a cell average) and for an averaging-kernel column whose
surface-pressure mismatch exceeded `max_surface_pressure_mismatch_pa` under
`SkipUnplaced()`: the observation's simulated value is reported, but it is
not in the cost.
"""
struct CSObservationStencil
    panel::Int
    i::Int
    j::Int
    steps::Vector{Int}
    weights::Vector{Float64}
    levels::Vector{Int}
    placement_method::Int8
    height_agl_m::Float64
    pressure_pa::Float64
    surface_pressure_mismatch_pa::Float64
    retrieval_profile::Vector{Float64}
    assimilated::Bool
end
