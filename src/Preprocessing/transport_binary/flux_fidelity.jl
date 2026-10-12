# Typed write-time certification that the horizontal payload was constructed
# from the intended winds and that diagnosed vertical transport agrees with a
# source OMEGA field when one is open. Evidence stays beside ReadyWindow and is
# never part of its payload, so certification cannot alter payload bytes.

const FLUX_FIDELITY_SCHEMA = 1
const FLUX_FIDELITY_SAMPLE_SEED = UInt64(0x6a09e667f3bcc909)
const FLUX_FIDELITY_SAMPLE_RULE = "cs_stratified_faces_v1"
const OMEGA_FIDELITY_PRESSURE_BANDS =
    ("surface-700", "700-300", "300-100", "100-10", "<10")

abstract type AbstractFluxFidelityReason end
struct EvidenceNotAttached <: AbstractFluxFidelityReason end
struct SourceHasNoWinds <: AbstractFluxFidelityReason end
struct SourcePayloadOnly <: AbstractFluxFidelityReason end
struct SourceNotOpened <: AbstractFluxFidelityReason end
struct SourceHasNoOmega <: AbstractFluxFidelityReason end
struct OmegaCheckDisabled <: AbstractFluxFidelityReason end

fidelity_reason(::EvidenceNotAttached) = "evidence_not_attached"
fidelity_reason(::SourceHasNoWinds) = "source_has_no_winds"
fidelity_reason(::SourcePayloadOnly) = "source_payload_only"
fidelity_reason(::SourceNotOpened) = "source_not_opened"
fidelity_reason(::SourceHasNoOmega) = "source_has_no_omega"
fidelity_reason(::OmegaCheckDisabled) = "check_disabled"

abstract type AbstractFluxFidelityEvidence{G,FT} end
abstract type AbstractOmegaFidelityEvidence{G,FT} end

"One canonical face/level comparison; mirrors are represented by one face id."
struct FluxFidelitySample{FT}
    face::Int
    level::Int
    reference::FT
    candidate::FT
    balanced::FT
end

struct WindDerivedFluxEvidence{G,FT,M,S} <: AbstractFluxFidelityEvidence{G,FT}
    method::M
    samples::S
    samples_per_window::Int
    time_sampling::String
    amount_seconds::Float64
end

struct NativeFluxEvidence{G,FT,S} <: AbstractFluxFidelityEvidence{G,FT}
    samples::S
    samples_per_window::Int
    method::String
    amount_seconds::Float64
end

struct NoFluxFidelityEvidence{G,FT,R<:AbstractFluxFidelityReason} <:
       AbstractFluxFidelityEvidence{G,FT}
    reason::R
end

struct OmegaFidelitySample{FT}
    source_pa_s::FT
    candidate_pa_s::FT
    pressure_hpa::FT
    bottom_residual_pa_s::FT
end

struct SourceOmegaEvidence{G,FT,S} <: AbstractOmegaFidelityEvidence{G,FT}
    samples::S
    source::String
    sampling::String
    basis::String
    baseline_id::String
    nrmse_limit::Float64
end

struct NoSourceOmegaEvidence{G,FT,R<:AbstractFluxFidelityReason} <:
       AbstractOmegaFidelityEvidence{G,FT}
    reason::R
end

struct FluxFidelityEvidenceBundle{F,O}
    flux::F
    omega::O
end

function default_fidelity_evidence(::Type{G}, ::Type{FT}) where {G,FT}
    return FluxFidelityEvidenceBundle(
        NoFluxFidelityEvidence{G,FT,SourcePayloadOnly}(SourcePayloadOnly()),
        NoSourceOmegaEvidence{G,FT,SourceHasNoOmega}(SourceHasNoOmega()))
end

"Canonical sampled face identifiers, stable across processes and Julia versions."
struct FidelitySampleMap
    version::Int
    seed::UInt64
    faces::Vector{Int}
end

@inline function _splitmix64(x::UInt64)
    z = x + UInt64(0x9e3779b97f4a7c15)
    z = xor(z, z >> 30) * UInt64(0xbf58476d1ce4e5b9)
    z = xor(z, z >> 27) * UInt64(0x94d049bb133111eb)
    return xor(z, z >> 31)
end

function FidelitySampleMap(face_table::CSGlobalFaceTable;
                           version::Integer=1,
                           samples_per_window::Integer=256,
                           seed::UInt64=FLUX_FIDELITY_SAMPLE_SEED)
    version == 1 || throw(ArgumentError("unsupported fidelity sample-map version $version"))
    ns = min(max(Int(samples_per_window), 0), face_table.nf)
    ns == 0 && return FidelitySampleMap(1, seed, Int[])
    # Seed each stratum explicitly: panel seams, faces next to panel corners,
    # then panel/interior x/y faces. Canonical face ids make mirror duplication
    # impossible by construction.
    seam = Int[]; corner = Int[]; interior = Int[]
    Nc = face_table.Nc
    for f in 1:face_table.nf
        i = Int(face_table.face_idx_i[f]); j = Int(face_table.face_idx_j[f])
        if face_table.mirror_panel[f] != 0
            push!(seam, f)
        elseif min(i, j) <= 2 || max(i, j) >= Nc
            push!(corner, f)
        else
            push!(interior, f)
        end
    end
    selected = Int[]
    function take_stratum!(xs, wanted, salt)
        isempty(xs) && return
        ranked = sort(xs; by=f -> _splitmix64(xor(seed, UInt64(f), UInt64(salt))))
        append!(selected, @view ranked[1:min(wanted, length(ranked))])
    end
    take_stratum!(seam, cld(ns, 4), 1)
    take_stratum!(corner, cld(ns, 4), 2)
    take_stratum!(interior, ns - length(selected), 3)
    if length(selected) < ns
        remaining = setdiff(collect(1:face_table.nf), selected)
        sort!(remaining; by=f -> _splitmix64(xor(seed, UInt64(f), UInt64(4))))
        append!(selected, @view remaining[1:(ns - length(selected))])
    end
    sort!(selected)
    return FidelitySampleMap(1, seed, selected)
end

mutable struct FluxFidelityAccumulator
    flux_seen::Bool
    flux_unavailable_reason::Union{Nothing,String}
    flux_kind::String
    method::String
    time_sampling::String
    samples_per_window::Int
    sample_count::Int
    windows::Int
    amount_min::Float64
    amount_max::Float64
    max_abs::Float64
    sumsq_abs::Float64
    max_rel::Float64
    sumsq_rel::Float64
    max_ulp::UInt64
    abs_tol::Float64
    rel_tol::Float64
    scale_floor::Float64
    worst_window::Int
    worst_face::Int
    worst_level::Int
    balance_max_abs::Float64
    balance_max_ratio::Float64
    balance_sumsq_ratio::Float64
    balance_count::Int
    omega_seen::Bool
    omega_unavailable_reason::Union{Nothing,String}
    omega_source::String
    omega_sampling::String
    omega_basis::String
    omega_count::Int
    omega_sum_error::Float64
    omega_sumsq_error::Float64
    omega_max_abs::Float64
    omega_sum_source::Float64
    omega_sum_candidate::Float64
    omega_sum_source2::Float64
    omega_sum_candidate2::Float64
    omega_sum_cross::Float64
    omega_source_scale2::Float64
    omega_bottom_max::Float64
    omega_band_error2::Vector{Float64}
    omega_band_source2::Vector{Float64}
    omega_band_count::Vector{Int}
    omega_baseline_id::String
    omega_limit::Float64
end

function FluxFidelityAccumulator()
    return FluxFidelityAccumulator(false, nothing, "none", "", "", 0, 0, 0,
        Inf, -Inf, 0.0, 0.0, 0.0, 0.0, UInt64(0), 0.0, 0.0, 0.0,
        0, 0, 0, 0.0, 0.0, 0.0, 0,
        false, nothing, "", "", "", 0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        zeros(5), zeros(5), zeros(Int, 5), "", Inf)
end

@inline function _ordered_bits(x::Float32)
    u = reinterpret(UInt32, x)
    sign = UInt32(1) << 31
    return iszero(u & sign) ? (u | sign) : ~u
end
@inline function _ordered_bits(x::Float64)
    u = reinterpret(UInt64, x)
    sign = UInt64(1) << 63
    return iszero(u & sign) ? (u | sign) : ~u
end
@inline _ulp_distance(a::FT, b::FT) where {FT<:Union{Float32,Float64}} =
    UInt64(abs(Int128(_ordered_bits(a)) - Int128(_ordered_bits(b))))

function _verify_flux_samples!(acc::FluxFidelityAccumulator, evidence, win::Int, kind::String)
    samples = evidence.samples
    FT = isempty(samples) ? Float64 : typeof(first(samples).reference)
    rel_tol = 256 * eps(FT)
    max_scale = isempty(samples) ? 1.0 : maximum(s -> max(abs(Float64(s.reference)), abs(Float64(s.candidate))), samples)
    scale_floor = max(floatmin(FT), eps(FT) * max(max_scale, 1.0))
    abs_tol = 64 * eps(FT) * max(max_scale, 1.0)
    for s in samples
        ref = Float64(s.reference); got = Float64(s.candidate)
        ae = abs(got - ref)
        scale = max(abs(ref), abs(got), scale_floor)
        re = ae / scale
        ulp = _ulp_distance(s.reference, s.candidate)
        if ae > abs_tol && re > rel_tol
            throw(ArgumentError("flux-fidelity gate FAILED before write for window $win " *
                "face=$(s.face) level=$(s.level): reference=$ref candidate=$got " *
                "abs=$ae > $abs_tol and rel=$re > $rel_tol"))
        end
        if ae > acc.max_abs
            acc.max_abs = ae; acc.worst_window = win
            acc.worst_face = s.face; acc.worst_level = s.level
        end
        acc.sumsq_abs += ae * ae
        acc.max_rel = max(acc.max_rel, re)
        acc.sumsq_rel += re * re
        acc.max_ulp = max(acc.max_ulp, ulp)
        correction = abs(Float64(s.balanced) - ref)
        ratio = correction / scale
        acc.balance_max_abs = max(acc.balance_max_abs, correction)
        acc.balance_max_ratio = max(acc.balance_max_ratio, ratio)
        acc.balance_sumsq_ratio += ratio * ratio
        acc.balance_count += 1
    end
    acc.flux_seen = true
    acc.flux_unavailable_reason = nothing
    acc.flux_kind = kind
    acc.method = String(evidence.method)
    acc.time_sampling = hasproperty(evidence, :time_sampling) ? evidence.time_sampling : "native"
    acc.samples_per_window = max(acc.samples_per_window, evidence.samples_per_window)
    acc.sample_count += length(samples)
    acc.windows += 1
    acc.amount_min = min(acc.amount_min, evidence.amount_seconds)
    acc.amount_max = max(acc.amount_max, evidence.amount_seconds)
    acc.abs_tol = max(acc.abs_tol, abs_tol)
    acc.rel_tol = max(acc.rel_tol, rel_tol)
    acc.scale_floor = max(acc.scale_floor, scale_floor)
    return nothing
end

verify_flux_fidelity!(acc, e::WindDerivedFluxEvidence, win) =
    _verify_flux_samples!(acc, e, win, "wind_rederived")
verify_flux_fidelity!(acc, e::NativeFluxEvidence, win) =
    _verify_flux_samples!(acc, e, win, "native_passthrough")
function verify_flux_fidelity!(acc::FluxFidelityAccumulator,
                               e::NoFluxFidelityEvidence, _win)
    acc.flux_seen || (acc.flux_unavailable_reason = fidelity_reason(e.reason))
    return nothing
end

@inline function _omega_band(p_hpa)
    p_hpa >= 700 && return 1
    p_hpa >= 300 && return 2
    p_hpa >= 100 && return 3
    p_hpa >= 10 && return 4
    return 5
end

function verify_omega_fidelity!(acc::FluxFidelityAccumulator,
                                e::SourceOmegaEvidence, win)
    for s in e.samples
        src = Float64(s.source_pa_s); got = Float64(s.candidate_pa_s)
        isfinite(src) && isfinite(got) || throw(ArgumentError(
            "OMEGA-fidelity sample is non-finite in window $win"))
        err = got - src
        acc.omega_count += 1
        acc.omega_sum_error += err
        acc.omega_sumsq_error += err * err
        acc.omega_max_abs = max(acc.omega_max_abs, abs(err))
        acc.omega_sum_source += src; acc.omega_sum_candidate += got
        acc.omega_sum_source2 += src * src
        acc.omega_sum_candidate2 += got * got
        acc.omega_sum_cross += src * got
        acc.omega_source_scale2 += src * src
        acc.omega_bottom_max = max(acc.omega_bottom_max, abs(Float64(s.bottom_residual_pa_s)))
        b = _omega_band(Float64(s.pressure_hpa))
        acc.omega_band_error2[b] += err * err
        acc.omega_band_source2[b] += src * src
        acc.omega_band_count[b] += 1
    end
    acc.omega_seen = true
    acc.omega_unavailable_reason = nothing
    acc.omega_source = e.source; acc.omega_sampling = e.sampling
    acc.omega_basis = e.basis; acc.omega_baseline_id = e.baseline_id
    acc.omega_limit = min(acc.omega_limit, e.nrmse_limit)
    return nothing
end
function verify_omega_fidelity!(acc::FluxFidelityAccumulator,
                                e::NoSourceOmegaEvidence, _win)
    acc.omega_seen || (acc.omega_unavailable_reason = fidelity_reason(e.reason))
    return nothing
end

function verify_flux_fidelity!(acc::FluxFidelityAccumulator,
                               bundle::FluxFidelityEvidenceBundle, win::Int)
    verify_flux_fidelity!(acc, bundle.flux, win)
    verify_omega_fidelity!(acc, bundle.omega, win)
    return nothing
end

function _fidelity_status_for_reason(reason::Union{Nothing,String})
    reason in ("source_payload_only", "source_has_no_winds", "source_has_no_omega") &&
        return "not_applicable"
    return "unavailable"
end

function flux_fidelity_header!(header::AbstractDict, acc::FluxFidelityAccumulator)
    for key in collect(keys(header))
        (startswith(String(key), "flux_fidelity_") || startswith(String(key), "omega_fidelity_") ||
         startswith(String(key), "balance_correction_")) && delete!(header, key)
    end
    header["flux_fidelity_schema"] = FLUX_FIDELITY_SCHEMA
    header["flux_fidelity_sample_rule"] = FLUX_FIDELITY_SAMPLE_RULE
    header["flux_fidelity_sample_seed"] = string(FLUX_FIDELITY_SAMPLE_SEED)
    if acc.flux_seen
        n = acc.sample_count
        header["flux_fidelity_status"] = "passed"
        header["flux_fidelity_kind"] = acc.flux_kind
        header["flux_fidelity_samples_per_window"] = acc.samples_per_window
        header["flux_fidelity_sample_count"] = n
        header["flux_fidelity_windows"] = acc.windows
        header["flux_fidelity_method"] = acc.method
        header["flux_fidelity_time_sampling"] = acc.time_sampling
        header["flux_fidelity_amount_seconds_min"] = acc.amount_min
        header["flux_fidelity_amount_seconds_max"] = acc.amount_max
        header["flux_fidelity_max_abs_kg"] = acc.max_abs
        header["flux_fidelity_rms_abs_kg"] = n == 0 ? 0.0 : sqrt(acc.sumsq_abs / n)
        header["flux_fidelity_max_rel"] = acc.max_rel
        header["flux_fidelity_rms_rel"] = n == 0 ? 0.0 : sqrt(acc.sumsq_rel / n)
        header["flux_fidelity_max_ulp"] = acc.max_ulp
        header["flux_fidelity_abs_tol_kg"] = acc.abs_tol
        header["flux_fidelity_rel_tol"] = acc.rel_tol
        header["flux_fidelity_scale_floor_kg"] = acc.scale_floor
        header["flux_fidelity_worst_window"] = acc.worst_window
        header["flux_fidelity_worst_face"] = acc.worst_face
        header["flux_fidelity_worst_level"] = acc.worst_level
        header["balance_correction_max_abs_kg"] = acc.balance_max_abs
        header["balance_correction_max_ratio"] = acc.balance_max_ratio
        header["balance_correction_rms_ratio"] = acc.balance_count == 0 ? 0.0 :
            sqrt(acc.balance_sumsq_ratio / acc.balance_count)
    else
        reason = something(acc.flux_unavailable_reason, "evidence_not_attached")
        header["flux_fidelity_status"] = _fidelity_status_for_reason(reason)
        header["flux_fidelity_kind"] = "none"
        header["flux_fidelity_reason"] = reason
        header["flux_fidelity_samples_per_window"] = 0
        header["flux_fidelity_sample_count"] = 0
        header["flux_fidelity_windows"] = 0
    end

    if acc.omega_seen
        n = acc.omega_count
        rmse = n == 0 ? 0.0 : sqrt(acc.omega_sumsq_error / n)
        source_rms = n == 0 ? 0.0 : sqrt(acc.omega_source_scale2 / n)
        nrmse = rmse / max(source_rms, eps(Float64))
        mx = acc.omega_sum_source / max(n, 1)
        my = acc.omega_sum_candidate / max(n, 1)
        sxx = acc.omega_sum_source2 - n * mx * mx
        syy = acc.omega_sum_candidate2 - n * my * my
        sxy = acc.omega_sum_cross - n * mx * my
        slope = sxy / max(sxx, eps(Float64))
        corr = sxy / sqrt(max(sxx * syy, eps(Float64)))
        passed = nrmse <= acc.omega_limit
        header["omega_fidelity_status"] = passed ? "passed" : "failed"
        passed || (header["omega_fidelity_reason"] = "baseline_limit_exceeded")
        header["omega_fidelity_source"] = acc.omega_source
        header["omega_fidelity_sampling"] = acc.omega_sampling
        header["omega_fidelity_basis"] = acc.omega_basis
        header["omega_fidelity_sample_count"] = n
        header["omega_fidelity_bias_pa_s"] = n == 0 ? 0.0 : acc.omega_sum_error / n
        header["omega_fidelity_rmse_pa_s"] = rmse
        header["omega_fidelity_nrmse"] = nrmse
        header["omega_fidelity_max_abs_pa_s"] = acc.omega_max_abs
        header["omega_fidelity_slope"] = slope
        header["omega_fidelity_correlation"] = corr
        header["omega_fidelity_bottom_residual_pa_s"] = acc.omega_bottom_max
        header["omega_fidelity_pressure_bands_hpa"] = collect(OMEGA_FIDELITY_PRESSURE_BANDS)
        header["omega_fidelity_band_nrmse"] = [sqrt(acc.omega_band_error2[i] /
            max(acc.omega_band_count[i], 1)) / max(sqrt(acc.omega_band_source2[i] /
            max(acc.omega_band_count[i], 1)), eps(Float64)) for i in 1:5]
        header["omega_fidelity_baseline_id"] = acc.omega_baseline_id
        header["omega_fidelity_limit"] = acc.omega_limit
        header["omega_fidelity_passed"] = passed
        passed || throw(ArgumentError("aggregate OMEGA-fidelity gate FAILED: nrmse=$nrmse > " *
                                      "limit=$(acc.omega_limit) ($(acc.omega_baseline_id))"))
    else
        reason = something(acc.omega_unavailable_reason, "source_not_opened")
        header["omega_fidelity_status"] = _fidelity_status_for_reason(reason)
        header["omega_fidelity_reason"] = reason
        header["omega_fidelity_source"] = "none"
        header["omega_fidelity_sampling"] = "none"
        header["omega_fidelity_basis"] = "none"
        header["omega_fidelity_sample_count"] = 0
    end
    return header
end

# ---------------------------------------------------------------------------
# Independent sampled CS evaluators. They execute the defining scalar formula
# rather than invoking the bulk constructors that produced `raw_am/raw_bm`.
# ---------------------------------------------------------------------------

struct CapturedWindFluxSamples{FT}
    face::Vector{Int}
    level::Vector{Int}
    reference::Vector{FT}
    candidate::Vector{FT}
end

function scale_captured_flux_samples(c::CapturedWindFluxSamples{FT}, factor::Real) where FT
    return CapturedWindFluxSamples(copy(c.face), copy(c.level),
        FT.(factor .* c.reference), FT.(factor .* c.candidate))
end

function combine_captured_flux_samples(a::CapturedWindFluxSamples{FT},
                                       b::CapturedWindFluxSamples{FT},
                                       raw_am, raw_bm, ft::CSGlobalFaceTable;
                                       scale::Real=1,
                                       mean_endpoints::Bool=false) where FT
    a.face == b.face && a.level == b.level || throw(ArgumentError(
        "endpoint fidelity sample maps differ"))
    reference = similar(a.reference)
    candidate = similar(a.candidate)
    weight = mean_endpoints ? 0.5 : 0.0
    for q in eachindex(reference)
        f, k = a.face[q], a.level[q]
        reference[q] = FT(_canonical_face_value(raw_am, raw_bm, ft, f, k))
        value = mean_endpoints ? weight * (a.candidate[q] + b.candidate[q]) : a.candidate[q]
        candidate[q] = FT(scale * value)
    end
    return CapturedWindFluxSamples(copy(a.face), copy(a.level), reference, candidate)
end

function merge_captured_flux_samples(native::CapturedWindFluxSamples{FT},
                                     plan::VerticalPlan, raw_am, raw_bm,
                                     ft::CSGlobalFaceTable) where FT
    nf = length(unique(native.face))
    face = Vector{Int}(undef, nf * plan.Nz_output)
    level = Vector{Int}(undef, length(face))
    reference = Vector{FT}(undef, length(face))
    candidate = Vector{FT}(undef, length(face))
    q = 0
    # Native captures are emitted face-major, then level 1:Nz_native.
    for (face_slot, f) in enumerate(unique(native.face)), l in 1:plan.Nz_output
        q += 1; face[q] = f; level[q] = l
        reference[q] = FT(_canonical_face_value(raw_am, raw_bm, ft, f, l))
        base = (face_slot - 1) * plan.Nz_native
        candidate[q] = FT(sum(Float64(native.candidate[base + k]) for k in plan.groups[l]))
    end
    return CapturedWindFluxSamples(face, level, reference, candidate)
end

flux_fidelity_method(::PanelAverageFluxes) = "panel_average"
flux_fidelity_method(::VectorFaceFluxes) = "vector"
flux_fidelity_method(::LineIntegralFaceFluxes) = "line_integral"

"""Direct midpoint stencils for the sampled line-integral fidelity faces.

These are rebuilt from the source/target meshes and deliberately do not read
the production sparse operator `method.L`.  Repeated source-cell entries stay
uncombined, preserving the defining 16-midpoint quadrature as the independent
check rather than merely applying selected rows of the kernel under test.
"""
struct LineIntegralFidelityStencil
    faces :: Vector{Int}
    cells :: Vector{Vector{Int}}
    weights :: Vector{Vector{Float64}}
    normals :: Vector{NTuple{3,Float64}}
end

function LineIntegralFidelityStencil(method::LineIntegralFaceFluxes,
                                     sample_map::FidelitySampleMap)
    source, target, ft = method.source_mesh, method.target_mesh, method.face_table
    centre, _, _ = _cs_cell_bases(target)
    cells = Vector{Vector{Int}}(undef, length(sample_map.faces))
    weights = Vector{Vector{Float64}}(undef, length(sample_map.faces))
    normals = Vector{NTuple{3,Float64}}(undef, length(sample_map.faces))
    for (slot, f) in pairs(sample_map.faces)
        a, b = _cs_face_corners(target, ft, f)
        n = _unit3(_cross3(a, b))
        sign = _dot3(n, centre[ft.face_right[f]] .- centre[ft.face_left[f]]) >= 0 ? 1.0 : -1.0
        theta = _arc3(a, b)
        segment_weight = sign * Float64(target.radius) * theta / method.samples_per_face
        face_cells = Int[]
        face_weights = Float64[]
        sizehint!(face_cells, 4 * method.samples_per_face)
        sizehint!(face_weights, 4 * method.samples_per_face)
        for q in 1:method.samples_per_face
            x = _slerp3(a, b, theta, (q - 0.5) / method.samples_per_face)
            for (c, wc) in _point_weights(source, x)
                push!(face_cells, c)
                push!(face_weights, segment_weight * wc)
            end
        end
        cells[slot], weights[slot] = face_cells, face_weights
        normals[slot] = (Float64(n[1]), Float64(n[2]), Float64(n[3]))
    end
    return LineIntegralFidelityStencil(copy(sample_map.faces), cells, weights, normals)
end

function capture_native_flux_samples(am, bm, ft::CSGlobalFaceTable;
                                     sample_map=FidelitySampleMap(ft))
    FT = eltype(am[1]); Nz = size(am[1], 3)
    return _capture_cs_samples(am, bm, ft, sample_map, Nz,
                               (f, k) -> _canonical_face_value(am, bm, ft, f, k))
end

@inline function _canonical_face_value(am, bm, ft::CSGlobalFaceTable, f, k)
    p = Int(ft.face_panel[f]); i = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
    return ft.face_dir[f] == 1 ? am[p][i, j, k] : bm[p][i, j, k]
end

function _capture_cs_samples(raw_am, raw_bm, ft, smap, Nz, evaluator)
    FT = eltype(raw_am[1])
    n = length(smap.faces) * Nz
    face = Vector{Int}(undef, n); level = Vector{Int}(undef, n)
    reference = Vector{FT}(undef, n); candidate = Vector{FT}(undef, n)
    q = 0
    for f in smap.faces, k in 1:Nz
        q += 1; face[q] = f; level[q] = k
        reference[q] = _canonical_face_value(raw_am, raw_bm, ft, f, k)
        candidate[q] = FT(evaluator(f, k))
    end
    return CapturedWindFluxSamples(face, level, reference, candidate)
end

function capture_cs_wind_flux_samples(method::PanelAverageFluxes,
                                      raw_am, raw_bm, u, v, dp,
                                      ft::CSGlobalFaceTable, gravity, dt_factor,
                                      Nc::Int, Nz::Int;
                                      sample_map=FidelitySampleMap(ft))
    scale = Float64(dt_factor) / Float64(gravity)
    function evaluator(f, k)
        p = Int(ft.face_panel[f]); i = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
        if ft.face_dir[f] == 1
            if i == 1
                un, thick = u[p][1, j, k], dp[p][1, j, k]
            elseif i == Nc + 1
                un, thick = u[p][Nc, j, k], dp[p][Nc, j, k]
            else
                un = 0.5 * (u[p][i - 1, j, k] + u[p][i, j, k])
                thick = 0.5 * (dp[p][i - 1, j, k] + dp[p][i, j, k])
            end
            return un * thick * _xface_length(method.lengths, i, j, Nc) * scale
        else
            if j == 1
                un, thick = v[p][i, 1, k], dp[p][i, 1, k]
            elseif j == Nc + 1
                un, thick = v[p][i, Nc, k], dp[p][i, Nc, k]
            else
                un = 0.5 * (v[p][i, j - 1, k] + v[p][i, j, k])
                thick = 0.5 * (dp[p][i, j - 1, k] + dp[p][i, j, k])
            end
            return un * thick * _yface_length(method.lengths, i, j, Nc) * scale
        end
    end
    return _capture_cs_samples(raw_am, raw_bm, ft, sample_map, Nz, evaluator)
end

function capture_cs_wind_flux_samples(method::VectorFaceFluxes,
                                      raw_am, raw_bm, u, v, dp,
                                      ft::CSGlobalFaceTable, gravity, dt_factor,
                                      Nc::Int, Nz::Int;
                                      sample_map=FidelitySampleMap(ft))
    scale = Float64(dt_factor) / Float64(gravity)
    function evaluator(f, k)
        un, thick = _vector_face_value(method.geom, f, u, v, dp, k)
        filter = method.geom.filter
        if filter !== nothing
            un = sum(filter.w[f][m] *
                     first(_vector_face_value(method.geom, filter.face[f][m],
                                              u, v, dp, k)) for m in 1:5)
        end
        return un * thick * method.geom.len[f] * scale
    end
    return _capture_cs_samples(raw_am, raw_bm, ft, sample_map, Nz, evaluator)
end

function capture_cs_wind_flux_samples(method::LineIntegralFaceFluxes,
                                      raw_am, raw_bm, u::AbstractMatrix,
                                      v::AbstractMatrix, dp::AbstractMatrix,
                                      ft::CSGlobalFaceTable, gravity, dt_factor,
                                      _Nc::Int, Nz::Int;
                                      sample_map=FidelitySampleMap(ft),
                                      line_rows=nothing)
    (; sinλ, cosλ, sinφ, cosφ) = method.src_trig
    scale = Float64(dt_factor) / Float64(gravity)
    FT = eltype(raw_am[1])
    n = length(sample_map.faces) * Nz
    face = Vector{Int}(undef, n); level = Vector{Int}(undef, n)
    reference = Vector{FT}(undef, n); candidate = Vector{FT}(undef, n)
    stencil = isnothing(line_rows) ?
        LineIntegralFidelityStencil(method, sample_map) : line_rows
    stencil.faces == sample_map.faces || throw(ArgumentError(
        "cached line-integral fidelity stencil does not match the sample map"))
    # Re-evaluate the defining midpoint quadrature directly. Each face has
    # `samples_per_face * 4` source contributions and no full-grid buffer.
    Threads.@threads :static for slot in eachindex(sample_map.faces)
        f = sample_map.faces[slot]
        cells, weights, normal = (stencil.cells[slot], stencil.weights[slot],
                                  stencil.normals[slot])
        for k in 1:Nz
            q = (slot - 1) * Nz + k
            face[q] = f; level[q] = k
            reference[q] = FT(_canonical_face_value(raw_am, raw_bm, ft, f, k))
            integral = 0.0
            @inbounds for r in eachindex(cells)
                c = cells[r]
                uu = Float64(u[c, k]) * Float64(dp[c, k])
                vv = Float64(v[c, k]) * Float64(dp[c, k])
                component = normal[1] * (-sinλ[c] * uu - sinφ[c] * cosλ[c] * vv) +
                            normal[2] * ( cosλ[c] * uu - sinφ[c] * sinλ[c] * vv) +
                            normal[3] * ( cosφ[c] * vv)
                integral += weights[r] * component
            end
            candidate[q] = FT(integral * scale)
        end
    end
    return CapturedWindFluxSamples(face, level, reference, candidate)
end

function wind_derived_evidence(::Type{G}, captured::CapturedWindFluxSamples{FT},
                               balanced_am, balanced_bm, ft::CSGlobalFaceTable;
                               method::AbstractString,
                               samples_per_window::Integer,
                               time_sampling::AbstractString,
                               amount_seconds::Real) where {G,FT}
    samples = Vector{FluxFidelitySample{FT}}(undef, length(captured.face))
    for q in eachindex(samples)
        f, k = captured.face[q], captured.level[q]
        samples[q] = FluxFidelitySample(f, k, captured.reference[q], captured.candidate[q],
            FT(_canonical_face_value(balanced_am, balanced_bm, ft, f, k)))
    end
    return WindDerivedFluxEvidence{G,FT,String,typeof(samples)}(
        String(method), samples, Int(samples_per_window), String(time_sampling),
        Float64(amount_seconds))
end

function native_flux_evidence(::Type{G}, captured::CapturedWindFluxSamples{FT},
                              balanced_am, balanced_bm, ft::CSGlobalFaceTable;
                              method::AbstractString="native_scaling_mapping",
                              samples_per_window::Integer=length(unique(captured.face)),
                              amount_seconds::Real) where {G,FT}
    samples = Vector{FluxFidelitySample{FT}}(undef, length(captured.face))
    for q in eachindex(samples)
        f, k = captured.face[q], captured.level[q]
        samples[q] = FluxFidelitySample(f, k, captured.reference[q], captured.candidate[q],
            FT(_canonical_face_value(balanced_am, balanced_bm, ft, f, k)))
    end
    return NativeFluxEvidence{G,FT,typeof(samples)}(
        samples, Int(samples_per_window), String(method), Float64(amount_seconds))
end

function capture_structured_flux_samples(am::AbstractArray{FT,3},
                                         bm::AbstractArray{FT,3};
                                         candidate_am=am,
                                         candidate_bm=bm,
                                         samples_per_window::Integer=256) where FT
    na = size(am, 1) * size(am, 2)
    nb = size(bm, 1) * size(bm, 2)
    nface = na + nb
    ns = min(Int(samples_per_window), nface)
    ranked = sort(collect(1:nface); by=f -> _splitmix64(xor(
        FLUX_FIDELITY_SAMPLE_SEED, UInt64(f))))
    faces = sort(ranked[1:ns])
    Nz = size(am, 3)
    face = Int[]; level = Int[]; reference = FT[]; candidate = FT[]
    for f in faces, k in 1:Nz
        reference_value, candidate_value = if f <= na
            i, j = Tuple(CartesianIndices((axes(am, 1), axes(am, 2)))[f])
            (am[i, j, k], candidate_am[i, j, k])
        else
            i, j = Tuple(CartesianIndices((axes(bm, 1), axes(bm, 2)))[f - na])
            (bm[i, j, k], candidate_bm[i, j, k])
        end
        push!(face, f); push!(level, k)
        push!(reference, reference_value); push!(candidate, candidate_value)
    end
    return CapturedWindFluxSamples(face, level, reference, candidate)
end

function capture_structured_wind_flux_samples(am::AbstractArray{FT,3},
                                              bm::AbstractArray{FT,3},
                                              transform, qv, grid, vertical,
                                              settings;
                                              samples_per_window::Integer=256) where FT
    Nx, Ny, Nz_native = size(transform.dp)
    native_am = zeros(FT, Nx + 1, Ny, Nz_native)
    native_bm = zeros(FT, Nx, Ny + 1, Nz_native)
    R_g = Float64(grid.mesh.radius) / Float64(STANDARD_GRAVITY)
    dlon, dlat = Float64(grid.dlon), Float64(grid.dlat)
    dt_amount = Float64(settings.half_dt)
    dry = settings.mass_basis == :dry
    qfield = hasproperty(qv, :qv_native) ? qv.qv_native : nothing
    dry && qfield === nothing && throw(ArgumentError(
        "dry structured fidelity reconstruction requires native humidity"))
    dA, dB = vertical.ab.dA, vertical.ab.dB
    @inbounds for k in 1:Nz_native, j in 1:Ny, i in 1:(Nx + 1)
        il = i == 1 ? Nx : i - 1
        ir = i <= Nx ? i : 1
        ps_face = exp((log(transform.sp[il, j]) + log(transform.sp[ir, j])) / 2)
        thick = abs(Float64(dA[k]) + Float64(dB[k]) * ps_face)
        qfactor = dry ? 1 - clamp(0.5 * (Float64(qfield[il, j, k]) +
                                         Float64(qfield[ir, j, k])), 0.0, 0.999999) : 1.0
        native_am[i, j, k] = FT((j == 1 || j == Ny) ? 0.0 :
            Float64(transform.u_stag[i, j, k]) / Float64(grid.cos_lat[j]) *
            thick * R_g * dlat * dt_amount * qfactor)
    end
    @inbounds for k in 1:Nz_native, j in 1:(Ny + 1), i in 1:Nx
        if j == 1 || j == Ny + 1
            native_bm[i, j, k] = zero(FT)
        else
            ps_face = exp((log(transform.sp[i, j - 1]) + log(transform.sp[i, j])) / 2)
            thick = abs(Float64(dA[k]) + Float64(dB[k]) * ps_face)
            qfactor = dry ? 1 - clamp(0.5 * (Float64(qfield[i, j - 1, k]) +
                                             Float64(qfield[i, j, k])), 0.0, 0.999999) : 1.0
            native_bm[i, j, k] = FT(Float64(transform.v_stag[i, j, k]) *
                thick * R_g * dlon * dt_amount * qfactor)
        end
    end
    candidate_am, candidate_bm = zeros(FT, size(am)), zeros(FT, size(bm))
    if vertical_mapping_method(vertical) === :pressure_overlap
        ps_xface = similar(transform.sp, Float64, Nx + 1, Ny)
        ps_yface = similar(transform.sp, Float64, Nx, Ny + 1)
        fill_xface_surface_pressure!(ps_xface, transform.sp)
        fill_yface_surface_pressure!(ps_yface, transform.sp)
        remap_extensive_field_pressure_overlap!(candidate_am, native_am, ps_xface,
            vertical.vc_native, vertical.merged_vc)
        remap_extensive_field_pressure_overlap!(candidate_bm, native_bm, ps_yface,
            vertical.vc_native, vertical.merged_vc)
    else
        merge_cell_field!(candidate_am, native_am, vertical.merge_map)
        merge_cell_field!(candidate_bm, native_bm, vertical.merge_map)
    end
    apply_structured_pole_constraints!(candidate_am, candidate_bm)
    return capture_structured_flux_samples(am, bm;
        candidate_am, candidate_bm, samples_per_window)
end

function structured_wind_evidence(::Type{G}, captured::CapturedWindFluxSamples{FT},
                                  am, bm;
                                  method::AbstractString,
                                  time_sampling::AbstractString,
                                  amount_seconds::Real) where {G,FT}
    na = size(am, 1) * size(am, 2)
    samples = Vector{FluxFidelitySample{FT}}(undef, length(captured.face))
    for q in eachindex(samples)
        f, k = captured.face[q], captured.level[q]
        balanced = if f <= na
            i, j = Tuple(CartesianIndices((axes(am, 1), axes(am, 2)))[f])
            am[i, j, k]
        else
            i, j = Tuple(CartesianIndices((axes(bm, 1), axes(bm, 2)))[f - na])
            bm[i, j, k]
        end
        samples[q] = FluxFidelitySample(f, k, captured.reference[q],
                                        captured.candidate[q], FT(balanced))
    end
    return WindDerivedFluxEvidence{G,FT,String,typeof(samples)}(
        String(method), samples, length(unique(captured.face)),
        String(time_sampling), Float64(amount_seconds))
end

function capture_faceindexed_flux_samples(hflux::AbstractMatrix{FT};
                                          candidate_hflux=hflux,
                                          samples_per_window::Integer=256) where FT
    nf, Nz = size(hflux)
    ns = min(Int(samples_per_window), nf)
    ranked = sort(collect(1:nf); by=f -> _splitmix64(xor(
        FLUX_FIDELITY_SAMPLE_SEED, UInt64(f))))
    faces = sort(ranked[1:ns])
    face = Int[]; level = Int[]; reference = FT[]; candidate = FT[]
    for f in faces, k in 1:Nz
        push!(face, f); push!(level, k)
        push!(reference, hflux[f, k]); push!(candidate, candidate_hflux[f, k])
    end
    return CapturedWindFluxSamples(face, level, reference, candidate)
end

function faceindexed_wind_evidence(::Type{G}, captured::CapturedWindFluxSamples{FT},
                                   hflux;
                                   method::AbstractString,
                                   amount_seconds::Real) where {G,FT}
    samples = [FluxFidelitySample(captured.face[q], captured.level[q],
        captured.reference[q], captured.candidate[q],
        FT(hflux[captured.face[q], captured.level[q]])) for q in eachindex(captured.face)]
    return WindDerivedFluxEvidence{G,FT,String,typeof(samples)}(
        String(method), samples, length(unique(captured.face)),
        "window_start_endpoint", Float64(amount_seconds))
end

function source_omega_evidence_cs(::Type{G}, cm, omega, qv, ps, A, B,
                                  cell_areas, gravity, amount_seconds;
                                  source::AbstractString,
                                  sampling::AbstractString,
                                  baseline_id::AbstractString,
                                  nrmse_limit::Real,
                                  samples_per_window::Integer=256) where G
    FT = eltype(cm[1])
    Nc = size(ps[1], 1); Nz = size(omega[1], 3)
    nc = 6 * Nc * Nc
    ns = min(Int(samples_per_window), nc)
    cells = sort(sort(collect(1:nc); by=c -> _splitmix64(xor(
        FLUX_FIDELITY_SAMPLE_SEED, UInt64(c), UInt64(0x0a11ce))))[1:ns])
    samples = OmegaFidelitySample{FT}[]
    for c in cells
        panel, rem = divrem(c - 1, Nc * Nc)
        j0, i0 = divrem(rem, Nc)
        p, i, j = panel + 1, i0 + 1, j0 + 1
        area = Float64(cell_areas[i, j])
        bottom = Float64(omega[p][i, j, Nz]) *
                 (1 - Float64(qv[p][i, j, Nz]))
        for k_ifc in 2:Nz
            source_omega = 0.5 * (Float64(omega[p][i, j, k_ifc - 1]) +
                                  Float64(omega[p][i, j, k_ifc])) *
                           (1 - 0.5 * (Float64(qv[p][i, j, k_ifc - 1]) +
                                      Float64(qv[p][i, j, k_ifc])))
            candidate = Float64(gravity) * Float64(cm[p][i, j, k_ifc]) /
                        (area * Float64(amount_seconds))
            pressure = (Float64(A[k_ifc]) + Float64(B[k_ifc]) *
                        Float64(ps[p][i, j])) / 100
            push!(samples, OmegaFidelitySample(FT(source_omega), FT(candidate),
                                               FT(pressure), FT(bottom)))
        end
    end
    return SourceOmegaEvidence{G,FT,typeof(samples)}(
        samples, String(source), String(sampling), "dry_(1-q)_interface",
        String(baseline_id), Float64(nrmse_limit))
end

