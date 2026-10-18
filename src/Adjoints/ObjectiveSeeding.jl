# ---------------------------------------------------------------------------
# Footprint-objective seeding + evaluation.
#
# Defines the abstract objective type and its concrete subtypes
# (`CSLayerMeanObjective`, `CSColumnMeanObjective`, the 4D-Var observation
# objectives `CSPlacedPointObjective` and `CSRetrievalColumnObjective`, which
# need the column meteorology of the sampled state, and `CSSeedObjective`),
# plus the kernels that:
#
#   * validate an objective against mesh/Nz bounds,
#   * forward-evaluate it on a `(panels_rm, panels_m)` pair to get the
#     scalar value of the cost function,
#   * seed `lambda_panels` from the objective at final time
#     (`_seed_objective!`: zero, then add the objective's linear functional
#     from `../Footprint/StepForcing.jl`), and
#   * accumulate the per-step surface footprint
#     (`_accumulate_surface_footprint!`).
# ---------------------------------------------------------------------------

abstract type AbstractCSFootprintObjective end

"""
    CSLayerMeanObjective(panel, i, j, level)

Scalar objective equal to the final-layer mixing ratio
`rm[panel][i, j, level] / m[panel][i, j, level]` on the physical CS
interior indices. `level = Nz` is the surface layer.
"""
struct CSLayerMeanObjective <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
    level::Int
end

"""
    CSColumnMeanObjective(panel, i, j)

Scalar objective equal to the final air-mass-weighted column mean mixing
ratio at one physical CS interior cell.
"""
struct CSColumnMeanObjective <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
end

"""
    CSPlacedPointObjective(panel, i, j, area, placement, pressure_basis)

Mixing ratio `rm / m` of the model layer that contains an in-situ sample, in
the physical CS interior cell `(panel, i, j)` of area `area` (m², the cell
that contains the sample). `placement` (an `Output.VerticalPlacement`) says
how the sample is placed in the column — its height above the model surface
or its pressure — and is resolved once, by `bind_to_mesh` with
`Output.resolve_placement`, the rule of the forward observation sampler. The
layer itself is chosen every time the objective is sampled, from the replayed
air mass of that state and the [`CSObservationColumnMet`](@ref) (hypsometric
layer heights; interface pressures of `pressure_basis`), with
`Output.sample_column_layer!`. Sampling it therefore needs the column
meteorology: `cs_surface_flux_4dvar(...; column_met)`; the final-time
footprint and Jacobian entry points do not take it.
"""
struct CSPlacedPointObjective{B <: AbstractPressureBasis} <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
    area::Float64
    placement::VerticalPlacement
    pressure_basis::B
    function CSPlacedPointObjective(panel::Integer, i::Integer, j::Integer, area::Real,
                                    placement::VerticalPlacement,
                                    pressure_basis::B) where {B <: AbstractPressureBasis}
        isfinite(area) && area > 0 || throw(ArgumentError(
            "CSPlacedPointObjective area must be finite and positive, got $area"))
        placement.method == VERTICAL_PLACEMENT_CODES.lowest_layer && throw(ArgumentError(
            "an inversion point observation is placed by its height or pressure; " *
            "the lowest-layer placement of column products does not apply"))
        return new{B}(Int(panel), Int(i), Int(j), Float64(area), placement, pressure_basis)
    end
end

"""
    CSRetrievalColumnObjective(panel, i, j, area, retrieval, pressure_basis;
                               max_surface_pressure_mismatch_pa = Inf,
                               mismatch_policy = ErrorOnUnplaced())

Averaging-kernel column average of the physical CS interior cell
`(panel, i, j)` (area `area`, m²): a retrieved column (satellite or TCCON
sounding) compared with the model through the retrieval's own layers,
pressure weights, averaging kernel and prior (`retrieval`, an
`Output.RetrievalColumn`). At every sampled state the model's interface
pressures of `pressure_basis` (total pressures from the dry air mass and the
specific humidity, `TotalFromHumidity`, or the dry interfaces,
`DryInterfaces`) are stretched onto the retrieval column and the dry mixing
ratios ``x_k = r_k/\\max(m_k, ε)`` are mapped onto its layers or levels by
the projection ``P`` of the retrieval's mapping (the rebin ``R`` of TM5-4DVar
for a layer column, the hat projection for a level column;
`Output.retrieval_column_weights!`):

```math
X = \\sum_k H_k\\, x_k + \\sum_i h_i (1 - a_i)\\, x_{a,i}, \\qquad
H_k = \\sum_i h_i a_i P_{ik}.
```

`max_surface_pressure_mismatch_pa` bounds the time-weighted difference
between the model and the retrieval surface pressure; a record above it is
an error under `mismatch_policy = ErrorOnUnplaced()` and is left out of the
cost (reported, not assimilated) under `SkipUnplaced()`. Sampling it needs
the column meteorology (`cs_surface_flux_4dvar(...; column_met)`); built by
`bind_to_mesh` from a column record with a retrieval block.
"""
struct CSRetrievalColumnObjective{M <: AbstractRetrievalMapping, B <: AbstractPressureBasis,
                                  U <: AbstractUnplacedPolicy} <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
    area::Float64
    retrieval::RetrievalColumn{M}
    pressure_basis::B
    max_surface_pressure_mismatch_pa::Float64
    mismatch_policy::U
    function CSRetrievalColumnObjective(panel::Integer, i::Integer, j::Integer, area::Real,
                                        retrieval::RetrievalColumn{M}, pressure_basis::B,
                                        max_surface_pressure_mismatch_pa::Real,
                                        mismatch_policy::U) where {M <: AbstractRetrievalMapping,
                                                                   B <: AbstractPressureBasis,
                                                                   U <: AbstractUnplacedPolicy}
        isfinite(area) && area > 0 || throw(ArgumentError(
            "CSRetrievalColumnObjective area must be finite and positive, got $area"))
        max_surface_pressure_mismatch_pa > 0 || throw(ArgumentError(
            "max_surface_pressure_mismatch_pa must be positive (Inf for no limit), got " *
            "$(max_surface_pressure_mismatch_pa)"))
        return new{M, B, U}(Int(panel), Int(i), Int(j), Float64(area), retrieval, pressure_basis,
                            Float64(max_surface_pressure_mismatch_pa), mismatch_policy)
    end
end

CSRetrievalColumnObjective(panel::Integer, i::Integer, j::Integer, area::Real, retrieval::RetrievalColumn,
                           pressure_basis::AbstractPressureBasis;
                           max_surface_pressure_mismatch_pa::Real = Inf,
                           mismatch_policy::AbstractUnplacedPolicy = ErrorOnUnplaced()) =
    CSRetrievalColumnObjective(panel, i, j, area, retrieval, pressure_basis, max_surface_pressure_mismatch_pa,
                               mismatch_policy)

"""
    CSCellAverageObjective(panel, i, j, members, weights, ids;
                           measurement_sigma = NaN, model_sigma = NaN, correlation = NaN)

The weighted mean ``\\sum_n w_n h_n`` of column objectives `members` of one
physical CS interior cell `(panel, i, j)` (`CSRetrievalColumnObjective`s or
`CSColumnMeanObjective`s; `weights` positive, summing to 1; `ids` the record
ids of the members): a cell average of soundings, built by `bind_to_mesh`
with `cell_averaging = CellAverage(...)`, the weights proportional to
``1/σ_n^2``. The operators are linear, so its functional is the same
weighted sum of the members' functionals: the gradient of the averaged
observation is the weighted combination of the members' gradients. Sampled
like its members (`cs_surface_flux_4dvar(...; column_met)`). The
surface-pressure screen applies to records, not averages, so an
averaging-kernel member must have no surface-pressure limit
(`max_surface_pressure_mismatch_pa = Inf`). `measurement_sigma`,
`model_sigma` and `correlation` record how the average's error was built
(``σ_\\mathrm{meas}``, ``σ_\\mathrm{mod}``, ``ρ``; departure files report
them).
"""
struct CSCellAverageObjective <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
    members::Vector{AbstractCSFootprintObjective}
    weights::Vector{Float64}
    ids::Vector{Int64}
    measurement_sigma::Float64
    model_sigma::Float64
    correlation::Float64
    function CSCellAverageObjective(panel::Integer, i::Integer, j::Integer, members::AbstractVector,
                                    weights::AbstractVector{<:Real}, ids::AbstractVector{<:Integer};
                                    measurement_sigma::Real = NaN, model_sigma::Real = NaN,
                                    correlation::Real = NaN)
        n = length(members)
        n >= 1 && length(weights) == n && length(ids) == n || throw(ArgumentError(
            "CSCellAverageObjective needs members with one weight and one id each; got $(n), " *
            "$(length(weights)), $(length(ids))"))
        for m in members
            m isa Union{CSRetrievalColumnObjective, CSColumnMeanObjective} || throw(ArgumentError(
                "a cell average combines column objectives (CSRetrievalColumnObjective, CSColumnMeanObjective); " *
                "got $(nameof(typeof(m)))"))
            (m.panel, m.i, m.j) == (panel, i, j) || throw(ArgumentError(
                "a cell average combines objectives of one cell $((panel, i, j)); a member is in " *
                "$((m.panel, m.i, m.j))"))
            _screens_records(m) && throw(ArgumentError(
                "a cell average cannot contain an averaging-kernel column with a surface-pressure limit " *
                "(max_surface_pressure_mismatch_pa = $(m.max_surface_pressure_mismatch_pa) Pa): the screen applies " *
                "to records, and the average would assimilate a screened record"))
        end
        all(w -> isfinite(w) && w > 0, weights) && abs(sum(weights) - 1) <= 8n * eps(Float64) ||
            throw(ArgumentError("cell-average weights must be positive and sum to 1; got $(weights)"))
        return new(Int(panel), Int(i), Int(j), collect(AbstractCSFootprintObjective, members),
                   collect(Float64, weights), collect(Int64, ids), Float64(measurement_sigma), Float64(model_sigma),
                   Float64(correlation))
    end
end

_screens_records(m::CSRetrievalColumnObjective) = isfinite(m.max_surface_pressure_mismatch_pa)
_screens_records(::AbstractCSFootprintObjective) = false

"""
    CSSeedObjective()

Marker objective used when the caller supplies an explicit final adjoint
seed (`dJ/drm_final`) instead of one of the built-in scalar objectives.
"""
struct CSSeedObjective <: AbstractCSFootprintObjective end

# ---------------------------------------------------------------------------
# Objective validation + forward evaluation
# ---------------------------------------------------------------------------

function _validate_objective(obj::CSLayerMeanObjective, mesh::CubedSphereMesh, Nz::Int)
    1 <= obj.panel <= 6 || throw(ArgumentError("panel must be in 1:6, got $(obj.panel)"))
    1 <= obj.i <= mesh.Nc || throw(ArgumentError("i must be in 1:$(mesh.Nc), got $(obj.i)"))
    1 <= obj.j <= mesh.Nc || throw(ArgumentError("j must be in 1:$(mesh.Nc), got $(obj.j)"))
    1 <= obj.level <= Nz || throw(ArgumentError("level must be in 1:$Nz, got $(obj.level)"))
    return nothing
end

function _validate_objective(obj::Union{CSColumnMeanObjective, CSPlacedPointObjective,
                                       CSRetrievalColumnObjective, CSCellAverageObjective},
                             mesh::CubedSphereMesh, Nz::Int)
    1 <= obj.panel <= 6 || throw(ArgumentError("panel must be in 1:6, got $(obj.panel)"))
    1 <= obj.i <= mesh.Nc || throw(ArgumentError("i must be in 1:$(mesh.Nc), got $(obj.i)"))
    1 <= obj.j <= mesh.Nc || throw(ArgumentError("j must be in 1:$(mesh.Nc), got $(obj.j)"))
    return nothing
end

# A placed point objective chooses its layer, and an averaging-kernel column
# its weights, from the column meteorology of the sampled state, which only the
# 4D-Var path carries.
_placed_objective_needs_met() = throw(ArgumentError(
    "a CSPlacedPointObjective chooses its layer from the column meteorology of the sampled state " *
    "(CSObservationColumnMet); use it as an observation of cs_surface_flux_4dvar(...; column_met). " *
    "The footprint and Jacobian entry points take only objectives with a fixed stencil " *
    "(CSLayerMeanObjective, CSColumnMeanObjective)"))
_retrieval_objective_needs_met() = throw(ArgumentError(
    "a CSRetrievalColumnObjective builds its weights from the interface pressures of the sampled state " *
    "(CSObservationColumnMet); use it as an observation of cs_surface_flux_4dvar(...; column_met). " *
    "The footprint and Jacobian entry points take only objectives with a fixed stencil " *
    "(CSLayerMeanObjective, CSColumnMeanObjective)"))

_cell_average_needs_4dvar() = throw(ArgumentError(
    "a CSCellAverageObjective (a cell average of column records) is sampled by cs_surface_flux_4dvar(...; " *
    "column_met); the footprint and Jacobian entry points take only objectives with a fixed stencil " *
    "(CSLayerMeanObjective, CSColumnMeanObjective)"))

function _validate_objective(::CSSeedObjective, mesh::CubedSphereMesh, Nz::Int)
    throw(ArgumentError(
        "`CSSeedObjective` is reserved for explicit final adjoint seeds; " *
        "use `cs_surface_emission_footprint_from_seed(final_adjoint_rm, ...)`"))
end

# Air-mass floor of the objectives, `eps(FT)`. The layer objective's evaluation
# `rm / max(m, floor)` and its adjoint seed `1 / max(m, floor)` both read it here,
# so the seed stays the derivative of the evaluated objective.
@inline _objective_mass_floor(::Type{FT}) where {FT} = eps(FT)

@kernel function _evaluate_layer_objective_kernel!(out, @Const(rm), @Const(m),
                                                   i, j, k)
    _ = @index(Global)
    FT = eltype(rm)
    @inbounds out[1] = rm[i, j, k] / max(m[i, j, k], _objective_mass_floor(FT))
end

@kernel function _evaluate_column_objective_kernel!(out, @Const(rm), @Const(m),
                                                    i, j, Nz::Int)
    _ = @index(Global)
    FT = eltype(rm)
    num = zero(FT)
    den = zero(FT)
    @inbounds for k in 1:Nz
        num += rm[i, j, k]
        den += m[i, j, k]
    end
    @inbounds out[1] = num / max(den, _objective_mass_floor(FT))
end

_host_scalar(a) = Array(a)[1]

function evaluate_objective(obj::CSLayerMeanObjective, panels_rm, panels_m,
                            mesh::CubedSphereMesh)
    Hp = mesh.Hp
    p = obj.panel
    ii = Hp + obj.i
    jj = Hp + obj.j
    k = obj.level
    FT = eltype(panels_rm[p])
    out = similar(panels_rm[p], FT, 1)
    backend = get_backend(panels_rm[p])
    kernel! = _evaluate_layer_objective_kernel!(backend, 1)
    kernel!(out, panels_rm[p], panels_m[p], Int32(ii), Int32(jj), Int32(k);
            ndrange = 1)
    synchronize(backend)
    return _host_scalar(out)
end

function evaluate_objective(obj::CSColumnMeanObjective, panels_rm, panels_m,
                            mesh::CubedSphereMesh)
    Hp = mesh.Hp
    p = obj.panel
    ii = Hp + obj.i
    jj = Hp + obj.j
    FT = eltype(panels_rm[p])
    out = similar(panels_rm[p], FT, 1)
    backend = get_backend(panels_rm[p])
    kernel! = _evaluate_column_objective_kernel!(backend, 1)
    kernel!(out, panels_rm[p], panels_m[p], Int32(ii), Int32(jj),
            size(panels_rm[p], 3); ndrange = 1)
    synchronize(backend)
    return _host_scalar(out)
end

evaluate_objective(::CSPlacedPointObjective, panels_rm, panels_m, mesh::CubedSphereMesh) =
    _placed_objective_needs_met()
evaluate_objective(::CSRetrievalColumnObjective, panels_rm, panels_m, mesh::CubedSphereMesh) =
    _retrieval_objective_needs_met()
evaluate_objective(::CSCellAverageObjective, panels_rm, panels_m, mesh::CubedSphereMesh) =
    _cell_average_needs_4dvar()

# ---------------------------------------------------------------------------
# Adjoint seeding + per-step surface accumulator
# ---------------------------------------------------------------------------

# The final-time seed `∂J/∂rm_final` of a built-in objective: zero the adjoint
# state, then add the objective's linear functional at the final air mass
# (`_linear_functional`, `_inject_functional!` in `../Footprint/StepForcing.jl`).
# The seed values are the ones the objectives have always used, and adding them
# to zero reproduces them bit for bit.
function _seed_objective!(lambda_panels, obj::AbstractCSFootprintObjective, final_m,
                          mesh::CubedSphereMesh)
    FT = eltype(lambda_panels[1])
    @inbounds for p in 1:6
        fill!(lambda_panels[p], zero(FT))
    end
    _inject_functional!(lambda_panels, _linear_functional(obj, 1, final_m, mesh), mesh)
    return nothing
end

function _seed_objective!(lambda_panels, ::CSSeedObjective, final_m,
                          mesh::CubedSphereMesh)
    throw(ArgumentError(
        "`CSSeedObjective` is reserved for explicit final adjoint seeds; " *
        "use `cs_surface_emission_footprint_from_seed(final_adjoint_rm, ...)`"))
end

@kernel function _accumulate_surface_footprint_kernel!(footprint, @Const(lambda), dt, Hp, Nz)
    i, j = @index(Global, NTuple)
    @inbounds footprint[i, j] = dt * lambda[i + Hp, j + Hp, Nz]
end

function _accumulate_surface_footprint!(footprint, lambda_panels, dt, mesh::CubedSphereMesh)
    Hp = mesh.Hp
    Nz = size(lambda_panels[1], 3)
    @inbounds for p in 1:6
        backend = get_backend(lambda_panels[p])
        kernel! = _accumulate_surface_footprint_kernel!(backend, (16, 16))
        kernel!(footprint[p], lambda_panels[p], eltype(lambda_panels[p])(dt), Int32(Hp), Int32(Nz);
                ndrange=(mesh.Nc, mesh.Nc))
        synchronize(backend)
    end
    return nothing
end
