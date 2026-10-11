"""
    AbstractReplayLayout

Topology-specific description of how horizontal fluxes map to a per-cell
horizontal divergence field for replay and explicit-dm continuity closure.
"""
abstract type AbstractReplayLayout end

struct StructuredDirectionalReplayLayout <: AbstractReplayLayout end

struct FaceIndexedReplayLayout{L <: AbstractVector{<:Integer},
                               R <: AbstractVector{<:Integer}} <: AbstractReplayLayout
    face_left  :: L
    face_right :: R
end

const STRUCTURED_DIRECTIONAL_REPLAY = StructuredDirectionalReplayLayout()

@inline structured_replay_layout() = STRUCTURED_DIRECTIONAL_REPLAY

@inline faceindexed_replay_layout(face_left::AbstractVector{<:Integer},
                                  face_right::AbstractVector{<:Integer}) =
    FaceIndexedReplayLayout(face_left, face_right)

"""
    replay_tolerance(FT)
    replay_tolerance(FT, DiskFT)

Tolerance of the per-cell replay check (`verify_window_continuity`) for data of
float type `FT`: `REPLAY_TOLERANCE_FLOAT64` or `REPLAY_TOLERANCE_FLOAT32` from
`src/Parameters`. For a binary stored as `DiskFT` and loaded as `FT`, the less
precise of the two sets it, because the loaded values carry that type's rounding.
"""
replay_tolerance(::Type{Float64}) = REPLAY_TOLERANCE_FLOAT64
replay_tolerance(::Type{Float32}) = REPLAY_TOLERANCE_FLOAT32
replay_tolerance(::Type{FT}, ::Type{DiskFT}) where {FT, DiskFT} =
    max(replay_tolerance(FT), replay_tolerance(DiskFT))

"""
    ReplayScratch(dims)

Per-cell work arrays of the replay check, of shape `dims` (the air-mass shape):
`div`, the net horizontal outflow of each cell, and `mag`, the sum of the
magnitudes of its horizontal face fluxes. Allocate one per grid and reuse it
across windows.
"""
struct ReplayScratch{A <: AbstractArray{Float64}}
    div :: A
    mag :: A

    function ReplayScratch(div::A, mag::A) where {A <: AbstractArray{Float64}}
        size(div) == size(mag) ||
            error("ReplayScratch: div shape $(size(div)) != mag shape $(size(mag))")
        return new{A}(div, mag)
    end
end

ReplayScratch(dims::Dims) =
    ReplayScratch(Array{Float64}(undef, dims), Array{Float64}(undef, dims))

Base.size(scratch::ReplayScratch) = size(scratch.div)

@inline _continuity_cm_shape(sz::NTuple{N, Int}) where {N} =
    ntuple(d -> d < N ? sz[d] : sz[d] + 1, Val(N))

@inline function _horizontal_columns(arr::AbstractArray{<:Any, N}) where {N}
    N >= 2 || error("continuity kernels require at least one horizontal axis and one vertical axis")
    return CartesianIndices(ntuple(d -> axes(arr, d), Val(N - 1)))
end

@inline function _column_index(col::CartesianIndex{M}, k::Int, ::Val{N}) where {M, N}
    coords = Tuple(col)
    return CartesianIndex(ntuple(d -> d <= M ? coords[d] : k, Val(N)))
end

@inline function _check_continuity_shapes(name::AbstractString,
                                          div_h::AbstractArray{Float64, N},
                                          m::AbstractArray{<:Any, N},
                                          cm::AbstractArray{<:Any, N},
                                          target::AbstractArray{<:Any, N}) where {N}
    size(div_h) == size(m) ||
        error("$name: div_h shape $(size(div_h)) != m shape $(size(m))")
    size(target) == size(m) ||
        error("$name: target shape $(size(target)) != m shape $(size(m))")
    expected_cm = _continuity_cm_shape(size(m))
    size(cm) == expected_cm ||
        error("$name: cm shape $(size(cm)) != expected $expected_cm")
    return nothing
end

"""
    horizontal_divergence!(div_h, layout, flux_args...)

Populate `div_h` with the per-cell horizontal divergence implied by the
topology-specific flux payload.
"""
function horizontal_divergence!(div_h::AbstractArray{Float64, 3},
                                ::StructuredDirectionalReplayLayout,
                                am::AbstractArray,
                                bm::AbstractArray)
    Nx, Ny, Nz = size(div_h)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        div_h[i, j, k] = (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                         (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
    end
    return div_h
end

function horizontal_divergence!(div_h::AbstractMatrix{Float64},
                                layout::FaceIndexedReplayLayout,
                                hflux::AbstractMatrix)
    fill!(div_h, 0.0)
    @inbounds for k in axes(div_h, 2), f in eachindex(layout.face_left)
        flux = Float64(hflux[f, k])
        left  = Int(layout.face_left[f])
        right = Int(layout.face_right[f])
        left  > 0 && (div_h[left,  k] += flux)
        right > 0 && (div_h[right, k] -= flux)
    end
    return div_h
end

"""
    horizontal_flux_magnitude!(mag_h, layout, flux_args...)

Populate `mag_h` with the sum of `|F|` over each cell's horizontal faces, the
horizontal part of the scale the replay residual is measured against.
"""
function horizontal_flux_magnitude!(mag_h::AbstractArray{Float64, 3},
                                    ::StructuredDirectionalReplayLayout,
                                    am::AbstractArray,
                                    bm::AbstractArray)
    Nx, Ny, Nz = size(mag_h)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mag_h[i, j, k] = (abs(Float64(am[i + 1, j, k])) + abs(Float64(am[i, j, k]))) +
                         (abs(Float64(bm[i, j + 1, k])) + abs(Float64(bm[i, j, k])))
    end
    return mag_h
end

function horizontal_flux_magnitude!(mag_h::AbstractMatrix{Float64},
                                    layout::FaceIndexedReplayLayout,
                                    hflux::AbstractMatrix)
    fill!(mag_h, 0.0)
    @inbounds for k in axes(mag_h, 2), f in eachindex(layout.face_left)
        flux = abs(Float64(hflux[f, k]))
        left  = Int(layout.face_left[f])
        right = Int(layout.face_right[f])
        left  > 0 && (mag_h[left,  k] += flux)
        right > 0 && (mag_h[right, k] += flux)
    end
    return mag_h
end

"""
    recompute_cm_from_dm_target!(layout, div_h, cm, m, dm_target, flux_args...)

Shared explicit-dm continuity closure. The topology-specific code only builds
`div_h`; the vertical integration and residual redistribution are common.
"""
function recompute_cm_from_dm_target!(layout::AbstractReplayLayout,
                                      div_h::AbstractArray{Float64, N},
                                      cm::AbstractArray{FT, N},
                                      m::AbstractArray{<:Any, N},
                                      dm_target::AbstractArray{<:Real, N},
                                      flux_args...) where {FT, N}
    horizontal_divergence!(div_h, layout, flux_args...)
    _check_continuity_shapes("recompute_cm_from_dm_target!", div_h, m, cm, dm_target)
    fill!(cm, zero(FT))

    @inbounds for col in _horizontal_columns(m)
        acc = 0.0
        total_m = 0.0
        for k in axes(m, N)
            idx = _column_index(col, k, Val(N))
            total_m += Float64(m[idx])
            acc = acc - Float64(div_h[idx]) - Float64(dm_target[idx])
            cm[_column_index(col, k + 1, Val(N))] = FT(acc)
        end

        residual = acc
        if residual != 0.0 && total_m > 0.0
            cum_fix = 0.0
            for k in axes(m, N)
                idx = _column_index(col, k, Val(N))
                idx_next = _column_index(col, k + 1, Val(N))
                cum_fix += (Float64(m[idx]) / total_m) * residual
                cm[idx_next] = FT(Float64(cm[idx_next]) - cum_fix)
            end
        end
    end
    return nothing
end

"""
    recompute_cm_from_dm_target!(cm, am, bm, m, dm_target)

Structured-directional convenience wrapper for the shared explicit-dm
continuity closure.
"""
function recompute_cm_from_dm_target!(cm::AbstractArray{FT, 3},
                                      am::AbstractArray,
                                      bm::AbstractArray,
                                      m::AbstractArray,
                                      dm_target::AbstractArray{<:Real, 3}) where {FT}
    div_h = Array{Float64}(undef, size(m))
    return recompute_cm_from_dm_target!(structured_replay_layout(), div_h, cm, m, dm_target, am, bm)
end

"""
    recompute_faceindexed_cm_from_dm_target!(cm, hflux, face_left, face_right,
                                             div_scratch, m, dm_target)

Face-indexed convenience wrapper for the shared explicit-dm continuity closure.
"""
function recompute_faceindexed_cm_from_dm_target!(cm::AbstractMatrix{FT},
                                                  hflux::AbstractMatrix,
                                                  face_left::AbstractVector{<:Integer},
                                                  face_right::AbstractVector{<:Integer},
                                                  div_scratch::AbstractMatrix{Float64},
                                                  m::AbstractMatrix,
                                                  dm_target::AbstractMatrix{<:Real}) where {FT}
    layout = faceindexed_replay_layout(face_left, face_right)
    return recompute_cm_from_dm_target!(layout, div_scratch, cm, m, dm_target, hflux)
end

"""
    verify_window_continuity(scratch, m_cur, cm, m_next, steps_per_window)
    verify_window_continuity(layout, scratch, m_cur, cm, m_next, steps_per_window, flux_args...)

Replay one window's stored mass fluxes against the stored air-mass endpoints,
cell by cell. With `n = steps_per_window`, each cell's budget residual is

    r = m_cur − 2n (div_h + cm[k+1] − cm[k]) − m_next

(`div_h` the net horizontal outflow of one flux application) and its scale is

    s = |m_cur| + |m_next| + 2n (Σ_faces |F_h| + |cm[k]| + |cm[k+1]|),

the summed magnitudes of the budget's terms, which bounds the rounding error of
the sum. The result's `max_rel_err` is the largest `|r| / s` of any cell, with
`max_abs_err = |r|` and `worst_idx` at that cell. A non-finite residual counts
as `Inf`, so NaN or Inf in the stored data fails every tolerance. Compare
`max_rel_err` with `replay_tolerance(FT)`.

The first form assumes `scratch.div` and `scratch.mag` (a `ReplayScratch`) are
already populated; the second builds them through the topology-specific
`layout`.

This certifies that the stored fluxes carry `m_cur` to `m_next` under the
runtime's palindrome continuity. It does not test the winds themselves:
preprocessing diagnoses `cm` from the same horizontal fluxes and endpoints
just before the check; see [What the replay gate certifies](@ref).
"""
function verify_window_continuity(scratch::ReplayScratch{<:AbstractArray{Float64, N}},
                                  m_cur::AbstractArray{<:Any, N},
                                  cm::AbstractArray{<:Any, N},
                                  m_next::AbstractArray{<:Any, N},
                                  steps_per_window::Integer) where {N}
    div_h = scratch.div
    mag_h = scratch.mag
    _check_continuity_shapes("verify_window_continuity", div_h, m_cur, cm, m_next)
    two_steps = Float64(2 * Int(steps_per_window))

    max_abs = 0.0
    max_rel = 0.0
    worst_idx = ntuple(_ -> 0, Val(N))
    @inbounds for idx in CartesianIndices(m_cur)
        coords = Tuple(idx)
        idx_next = CartesianIndex(Base.setindex(coords, coords[end] + 1, N))
        m0 = Float64(m_cur[idx])
        m1 = Float64(m_next[idx])
        cm_top = Float64(cm[idx])
        cm_bot = Float64(cm[idx_next])
        net_div = Float64(div_h[idx]) + (cm_bot - cm_top)
        abs_err = abs(m0 - two_steps * net_div - m1)
        scale = abs(m0) + abs(m1) +
                two_steps * (Float64(mag_h[idx]) + abs(cm_top) + abs(cm_bot))
        rel_err = abs_err == 0 ? 0.0 : abs_err / scale
        isfinite(rel_err) || (rel_err = Inf)
        if rel_err > max_rel
            max_rel = rel_err
            max_abs = abs_err
            worst_idx = coords
        end
    end
    return (max_abs_err = max_abs, max_rel_err = max_rel, worst_idx = worst_idx)
end

function verify_window_continuity(layout::AbstractReplayLayout,
                                  scratch::ReplayScratch{<:AbstractArray{Float64, N}},
                                  m_cur::AbstractArray{<:Any, N},
                                  cm::AbstractArray{<:Any, N},
                                  m_next::AbstractArray{<:Any, N},
                                  steps_per_window::Integer,
                                  flux_args...) where {N}
    horizontal_divergence!(scratch.div, layout, flux_args...)
    horizontal_flux_magnitude!(scratch.mag, layout, flux_args...)
    return verify_window_continuity(scratch, m_cur, cm, m_next, steps_per_window)
end

"""
    verify_window_continuity_ll(m_cur, am, bm, cm, m_next, steps_per_window)

Structured-directional convenience wrapper for the shared replay kernel.
"""
function verify_window_continuity_ll(m_cur::AbstractArray{FT, 3},
                                     am::AbstractArray,
                                     bm::AbstractArray,
                                     cm::AbstractArray,
                                     m_next::AbstractArray,
                                     steps_per_window::Integer) where {FT}
    return verify_window_continuity(structured_replay_layout(), ReplayScratch(size(m_cur)),
                                    m_cur, cm, m_next, steps_per_window, am, bm)
end

"""
    verify_window_continuity_rg(m_cur, hflux, cm, m_next, face_left, face_right,
                                steps_per_window; scratch = ReplayScratch(size(m_cur)))

Face-indexed convenience wrapper for the shared replay kernel. Pass a reused
`scratch` to avoid allocating per window.
"""
function verify_window_continuity_rg(m_cur::AbstractMatrix{FT},
                                     hflux::AbstractMatrix,
                                     cm::AbstractMatrix,
                                     m_next::AbstractMatrix,
                                     face_left::AbstractVector{<:Integer},
                                     face_right::AbstractVector{<:Integer},
                                     steps_per_window::Integer;
                                     scratch::ReplayScratch{<:AbstractMatrix{Float64}} =
                                         ReplayScratch(size(m_cur))) where {FT}
    layout = faceindexed_replay_layout(face_left, face_right)
    return verify_window_continuity(layout, scratch, m_cur, cm, m_next,
                                    steps_per_window, hflux)
end

"""
    verify_window_continuity_cs(m_cur, am, bm, cm, m_next, steps_per_window)

Cubed-sphere convenience wrapper for the shared replay kernel. The caller must
provide panel-local arrays with synchronized boundary mirrors so each panel's
structured divergence matches the closed-sphere flux field.
"""
function verify_window_continuity_cs(m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                     am::NTuple{NP, <:AbstractArray},
                                     bm::NTuple{NP, <:AbstractArray},
                                     cm::NTuple{NP, <:AbstractArray},
                                     m_next::NTuple{NP, <:AbstractArray},
                                     steps_per_window::Integer) where {FT, NP}
    NP >= 1 || error("verify_window_continuity_cs requires at least one panel")
    layout = structured_replay_layout()
    scratch = ReplayScratch(size(m_cur[1]))
    worst_rel = 0.0
    worst_abs = 0.0
    worst_idx = (0, 0, 0, 0)

    for p in 1:NP
        diag = verify_window_continuity(layout, scratch, m_cur[p], cm[p], m_next[p],
                                        steps_per_window, am[p], bm[p])
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_idx = (p, diag.worst_idx...)
        end
    end

    return (max_abs_err = worst_abs, max_rel_err = worst_rel, worst_idx = worst_idx)
end

@inline function replay_summary_message(worst_rel::Real,
                                        worst_abs::Real,
                                        worst_win::Integer,
                                        worst_idx)
    return @sprintf("max per-cell |residual|/scale = %.3e  (abs=%.3e kg  win=%d  cell=%s)",
                    worst_rel, worst_abs, worst_win, worst_idx)
end

@inline function replay_failure_message(prefix::AbstractString,
                                        worst_rel::Real,
                                        tol_rel::Real,
                                        worst_win::Integer,
                                        worst_idx,
                                        worst_abs::Real)
    return "$(prefix) FAILED: " *
           @sprintf("rel=%.3e > tol=%.3e at window %d cell %s (abs=%.3e kg).",
                    worst_rel, tol_rel, worst_win, worst_idx, worst_abs) *
           " Stored fluxes do not integrate to stored m_next under palindrome continuity. " *
           "Regenerate the binary with the current preprocessor."
end

"""
    run_replay_gate(diag_for_window, Nt; tol_rel, summary_label=nothing, failure_prefix)

Run a replay gate across `Nt` consecutive windows, tracking the worst
diagnostic and applying common summary/failure formatting.
"""
function run_replay_gate(diag_for_window, Nt::Integer;
                         tol_rel::Real,
                         summary_label::Union{Nothing, AbstractString}=nothing,
                         failure_prefix::AbstractString)
    Nt >= 1 || return (worst_window = 0, worst_rel = 0.0, worst_abs = 0.0, worst_idx = nothing)

    first_diag = diag_for_window(1)
    worst_rel = first_diag.max_rel_err
    worst_abs = first_diag.max_abs_err
    worst_win = 1
    worst_idx = first_diag.worst_idx

    for win in 2:Nt
        diag = diag_for_window(win)
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_win = win
            worst_idx = diag.worst_idx
        end
    end

    summary_label !== nothing &&
        @info "$summary_label: $(replay_summary_message(worst_rel, worst_abs, worst_win, worst_idx))"

    worst_rel <= tol_rel ||
        error(replay_failure_message(failure_prefix, worst_rel, tol_rel, worst_win, worst_idx, worst_abs))

    return (worst_window = worst_win, worst_rel = worst_rel, worst_abs = worst_abs, worst_idx = worst_idx)
end
