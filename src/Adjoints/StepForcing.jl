# ---------------------------------------------------------------------------
# Observation operators as fixed sparse linear functionals, and the adjoint
# forcing the reverse walk injects at end-of-step markers.
#
# With fixed meteorology the air mass `m` of every step is known before any
# tracer is transported, so a mixing-ratio observation sampled after the steps
# `s` of its time sampling, with weights `ω_s` summing to 1,
#
#     y_sim = h(rm) = offset + Σₑ wₑ · rm_{sₑ}[pₑ][iₑ, jₑ, kₑ],
#
# is linear in the tracer masses with weights `wₑ = ω_{sₑ} · ŵₑ`, `ŵₑ` taken
# from `m_{sₑ}` (1/max(m, eps) for a layer, 1/Σm for a column mean). Its
# gradient ∂h/∂rm_s is the weight vector of the entries of step `s`.
# `CSLinearFunctional` stores one such operator; an instantaneous observation
# has the entries of one step.
#
# The 4D-Var cost J_obs = ½ Σᵢ ((hᵢ(rm) − yᵢ)/σᵢ)² has
# ∂J_obs/∂rm_s = Σᵢ (hᵢ − yᵢ)/σᵢ² · ∂hᵢ/∂rm_s. `CSStepForcing`
# holds these scaled weights, grouped by step and panel, so one reverse walk
# can add each step's sum to the adjoint state at that step's
# `_CSStepEndRecord` (`_inject_step_forcing!`, one work item per cell, entries
# in a fixed order, no atomics). `NoStepForcing` is the no-op used by the
# single-objective footprints.
#
# Loaded into the `Adjoints` module right after `ObjectiveSeeding.jl`, whose
# `_seed_objective!` injects one functional through the same kernel.
# ---------------------------------------------------------------------------

"""
    CSLinearFunctional{FT}(steps, panel, i, j, k, weight, offset)
    CSLinearFunctional{FT}(step::Integer, panel, i, j, k, weight, offset)

Observation operator as a fixed sparse linear functional of the tracer mass
`rm_s` at the end of model steps `s` (after convection):

    h(rm) = offset + Σₑ weight[e] · rm_{steps[e]}[panel[e]][Hp + i[e], Hp + j[e], k[e]]

`steps[e]` is the step of entry `e` (the scalar-`step` form gives every entry
that step: an instantaneous observation); `i`, `j` are interior cell indices
in `1:Nc`; `k` is the level (`Nz` is the surface layer). The weights come
from the replayed air mass at each entry's step, which does not depend on the
tracer, so `h` is linear in `rm` and its gradient is the weight vector; a
window observation's entries carry its time-sampling weights. `offset` holds
terms that do not depend on the transported tracer (for example an averaging
kernel's prior term); it is zero for the built-in objectives. Built by
`_linear_functional` (one step) and `_observe_cs_forward!` (an observation's
sampled steps).
"""
struct CSLinearFunctional{FT}
    steps::Vector{Int}
    panel::Vector{Int}
    i::Vector{Int}
    j::Vector{Int}
    k::Vector{Int}
    weight::Vector{FT}
    offset::FT

    function CSLinearFunctional{FT}(steps::AbstractVector{<:Integer}, panel, i, j, k, weight,
                                    offset) where {FT}
        n = length(weight)
        length(steps) == length(panel) == length(i) == length(j) == length(k) == n ||
            throw(DimensionMismatch("CSLinearFunctional entry vectors must have equal lengths"))
        all(>=(1), steps) || throw(ArgumentError(
            "CSLinearFunctional steps must be positive, got $(steps)"))
        return new{FT}(collect(Int, steps), collect(Int, panel), collect(Int, i), collect(Int, j),
                       collect(Int, k), collect(FT, weight), FT(offset))
    end
end

function CSLinearFunctional{FT}(step::Integer, panel, i, j, k, weight, offset) where {FT}
    step >= 1 || throw(ArgumentError("CSLinearFunctional step must be positive, got $step"))
    return CSLinearFunctional{FT}(fill(Int(step), length(weight)), panel, i, j, k, weight, offset)
end

# ---------------------------------------------------------------------------
# The built-in objectives as linear functionals
# ---------------------------------------------------------------------------

"""
    _linear_functional(objective, step, panels_m, mesh) -> CSLinearFunctional

`objective` as a linear functional of the tracer mass at the end of `step`,
with weights from the air mass `panels_m` of that state. The weights are the
derivatives of the objectives' evaluations, air-mass floor included:
`1 / max(m, eps)` for `CSLayerMeanObjective` and `1 / max(Σₖ m, eps)` for
`CSColumnMeanObjective` (the column seed had no floor before; the two agree for
every column holding more than `eps` of air).
"""
function _linear_functional(obj::CSLayerMeanObjective, step::Integer, panels_m,
                            mesh::CubedSphereMesh)
    FT = eltype(panels_m[1])
    ii = mesh.Hp + obj.i
    jj = mesh.Hp + obj.j
    k = obj.level
    # Range indexing keeps the read GPU-safe (no scalar indexing).
    m = _host_scalar(panels_m[obj.panel][ii:ii, jj:jj, k:k])
    weight = one(FT) / max(m, _objective_mass_floor(FT))
    return CSLinearFunctional{FT}(step, (obj.panel,), (obj.i,), (obj.j,), (k,),
                                  (weight,), zero(FT))
end

function _linear_functional(obj::CSColumnMeanObjective, step::Integer, panels_m,
                            mesh::CubedSphereMesh)
    FT = eltype(panels_m[1])
    p = obj.panel
    ii = mesh.Hp + obj.i
    jj = mesh.Hp + obj.j
    Nz = size(panels_m[p], 3)
    denom = sum(@view panels_m[p][ii, jj, :])
    weight = one(FT) / max(FT(denom), _objective_mass_floor(FT))
    return CSLinearFunctional{FT}(step, fill(p, Nz), fill(obj.i, Nz), fill(obj.j, Nz),
                                  1:Nz, fill(weight, Nz), zero(FT))
end

_linear_functional(::CSSeedObjective, ::Integer, _panels_m, ::CubedSphereMesh) =
    throw(ArgumentError(
        "`CSSeedObjective` is reserved for explicit final adjoint seeds; " *
        "use `cs_surface_emission_footprint_from_seed(final_adjoint_rm, ...)`"))

_linear_functional(::CSPlacedPointObjective, ::Integer, _panels_m, ::CubedSphereMesh) =
    _placed_objective_needs_met()
_linear_functional(::CSRetrievalColumnObjective, ::Integer, _panels_m, ::CubedSphereMesh) =
    _retrieval_objective_needs_met()
_linear_functional(::CSCellAverageObjective, ::Integer, _panels_m, ::CubedSphereMesh) =
    _cell_average_needs_4dvar()

# ---------------------------------------------------------------------------
# Step forcing of the reverse walk
# ---------------------------------------------------------------------------

abstract type AbstractStepForcing end

"""
    NoStepForcing()

No adjoint forcing at the end-of-step markers: the reverse walk transposes the
tape from its starting adjoint state alone. The default of every footprint
path, which therefore stays bit-identical to a walk without the markers.
"""
struct NoStepForcing <: AbstractStepForcing end

"""
    CSStepForcing

Adjoint forcing that the reverse walk adds at each `_CSStepEndRecord`: the
entries of a set of `CSLinearFunctional`s, each weight multiplied by its
observation's scale (the weighted departure `(H(x) − y)/σ²` in 4D-Var).
Built by `_assemble_step_forcing`.

Entries are sorted by step, panel, level, `j`, `i`, then input order, and
entries of one cell (step, panel, `i`, `j`, `k`) form a segment. The segments
of one `(step, panel)` pair are contiguous: `first_segment[step, panel]` and
`segment_count[step, panel]` (host matrices, count 0 for no entries) locate
them. Segment `s` adds `w[segment_ptr[s] : segment_ptr[s + 1] − 1]` to the
cell `(ii[s], jj[s], kk[s])` (halo-padded indices). The index and weight
vectors live on the backend of the adjoint state.
"""
struct CSStepForcing{FT, VI <: AbstractVector{Int32}, VF <: AbstractVector{FT}} <: AbstractStepForcing
    nz::Int
    first_segment::Matrix{Int}
    segment_count::Matrix{Int}
    segment_ptr::VI
    ii::VI
    jj::VI
    kk::VI
    w::VF
end

"""
    _assemble_step_forcing(functionals, scales, mesh, Nz, prototype) -> CSStepForcing

Combine `functionals[n]` scaled by `scales[n]` into one `CSStepForcing` whose
device vectors are `similar` to `prototype` (an adjoint-state panel). Entries
of the same cell and step are added in input order.
"""
function _assemble_step_forcing(functionals::AbstractVector{<:CSLinearFunctional},
                                scales, mesh::CubedSphereMesh, Nz::Int,
                                prototype::AbstractArray{FT}) where {FT}
    length(functionals) == length(scales) || throw(DimensionMismatch(
        "$(length(functionals)) functionals but $(length(scales)) scales"))
    n = sum(f -> length(f.weight), functionals; init = 0)
    step = Vector{Int}(undef, n)
    panel = Vector{Int}(undef, n)
    ci = Vector{Int}(undef, n)
    cj = Vector{Int}(undef, n)
    ck = Vector{Int}(undef, n)
    w = Vector{FT}(undef, n)
    e = 0
    @inbounds for (f, scale) in zip(functionals, scales)
        s = FT(scale)
        for idx in eachindex(f.weight)
            e += 1
            p, i, j, k = f.panel[idx], f.i[idx], f.j[idx], f.k[idx]
            1 <= p <= 6 || throw(ArgumentError("functional panel must be in 1:6, got $p"))
            1 <= i <= mesh.Nc || throw(ArgumentError("functional i must be in 1:$(mesh.Nc), got $i"))
            1 <= j <= mesh.Nc || throw(ArgumentError("functional j must be in 1:$(mesh.Nc), got $j"))
            1 <= k <= Nz || throw(ArgumentError("functional level must be in 1:$Nz, got $k"))
            step[e] = f.steps[idx]
            panel[e] = p
            ci[e] = i
            cj[e] = j
            ck[e] = k
            # A non-finite scale (a NaN simulated value) propagates into the
            # gradient, as it did before; the optimizer's line search sees it.
            w[e] = s * FT(f.weight[idx])
        end
    end

    # A fixed order: step, panel, level, j, i, then input position. Each
    # injection work item owns one cell and adds its entries in this order, so
    # the result is deterministic without atomics.
    order = sortperm(collect(zip(step, panel, ck, cj, ci, 1:n)))
    last_step = maximum(step; init = 0)
    first_segment = zeros(Int, last_step, 6)
    segment_count = zeros(Int, last_step, 6)
    segment_ptr = Int32[]
    seg_i, seg_j, seg_k = Int32[], Int32[], Int32[]
    same_cell(a, b) = step[a] == step[b] && panel[a] == panel[b] &&
                      ck[a] == ck[b] && cj[a] == cj[b] && ci[a] == ci[b]
    @inbounds for pos in 1:n
        e = order[pos]
        pos > 1 && same_cell(order[pos - 1], e) && continue
        # A new cell: open a segment (and the first segment of a new pair).
        s, p = step[e], panel[e]
        segment_count[s, p] == 0 && (first_segment[s, p] = length(segment_ptr) + 1)
        segment_count[s, p] += 1
        push!(segment_ptr, Int32(pos))
        push!(seg_i, Int32(mesh.Hp + ci[e]))
        push!(seg_j, Int32(mesh.Hp + cj[e]))
        push!(seg_k, Int32(ck[e]))
    end
    push!(segment_ptr, Int32(n + 1))

    device(v) = (d = similar(prototype, eltype(v), length(v)); copyto!(d, v); d)
    segment_ptr_d = device(segment_ptr)
    w_d = device(w[order])
    return CSStepForcing{FT, typeof(segment_ptr_d), typeof(w_d)}(
        Nz, first_segment, segment_count, segment_ptr_d,
        device(seg_i), device(seg_j), device(seg_k), w_d)
end

# A forcing must not reference steps beyond the reverse pass (their entries
# would never be reached) and must match the adjoint state's level count.
_validate_step_forcing(::NoStepForcing, ::Int, ::Int) = nothing
function _validate_step_forcing(forcing::CSStepForcing, nsteps::Int, Nz::Int)
    forcing.nz == Nz || throw(DimensionMismatch(
        "step forcing has $(forcing.nz) levels; the adjoint state has $Nz"))
    size(forcing.first_segment, 1) <= nsteps || throw(ArgumentError(
        "step forcing references step $(size(forcing.first_segment, 1)), but the " *
        "reverse pass has $nsteps steps"))
    return nothing
end

# One work item per cell (segment): it adds the cell's entries in their fixed
# order. Segments are distinct cells, so no two work items write one cell.
@kernel function _inject_step_forcing_kernel!(lambda, @Const(segment_ptr), @Const(ii),
                                              @Const(jj), @Const(kk), @Const(w),
                                              first_segment)
    n = @index(Global, Linear)
    @inbounds begin
        s = first_segment + n - 1
        i, j, k = ii[s], jj[s], kk[s]
        acc = lambda[i, j, k]
        for e in segment_ptr[s]:(segment_ptr[s + 1] - one(eltype(segment_ptr)))
            acc += w[e]
        end
        lambda[i, j, k] = acc
    end
end

"""
    _inject_step_forcing!(lambda_panels, forcing, step, mesh)

Add the forcing entries of `step` to the adjoint state `lambda_panels` (the
reverse walk calls it at `_CSStepEndRecord(step)`). A no-op for
`NoStepForcing` and for steps without entries.
"""
_inject_step_forcing!(_lambda_panels, ::NoStepForcing, ::Int, ::CubedSphereMesh) = nothing

function _inject_step_forcing!(lambda_panels, forcing::CSStepForcing, step::Int,
                               mesh::CubedSphereMesh)
    step <= size(forcing.first_segment, 1) || return nothing
    @inbounds for p in 1:6
        count = forcing.segment_count[step, p]
        count == 0 && continue
        backend = get_backend(lambda_panels[p])
        kernel! = _inject_step_forcing_kernel!(backend, 256)
        kernel!(lambda_panels[p], forcing.segment_ptr, forcing.ii, forcing.jj, forcing.kk,
                forcing.w, forcing.first_segment[step, p]; ndrange = count)
    end
    return nothing
end

# Add one functional of one step (scale 1) to the adjoint state: the final-time
# seed of a single-objective footprint (`_seed_objective!`).
function _inject_functional!(lambda_panels, f::CSLinearFunctional{FT},
                             mesh::CubedSphereMesh) where {FT}
    step = only(unique(f.steps))
    forcing = _assemble_step_forcing([f], (one(FT),), mesh, size(lambda_panels[1], 3),
                                     lambda_panels[1])
    _inject_step_forcing!(lambda_panels, forcing, step, mesh)
    return nothing
end
