# Typed write-time certification that the horizontal payload was constructed
# from the intended winds and that diagnosed vertical transport agrees with a
# source OMEGA field when one is open. Evidence stays beside ReadyWindow and is
# never part of its payload, so certification cannot alter payload bytes.
#
# Scope of the checks (documented in docs/src/concepts/binary_format.md):
# - Reference and candidate read the same input arrays (winds, pressure
#   thickness, humidity), so no check here can catch an input error
#   (registration, level order, units).
# - The panel-average and line-integral evaluators are independent of the
#   production constructors. The vector evaluator reuses the production face
#   geometry and FV3 filter (`_vector_face_value`, `geom.len`), so it catches
#   assembly, scatter and time-scaling errors but not geometry errors.
# - Native GEOS fluxes have no independent source: their certificate is
#   `not_applicable` and records only the balance correction.

const FLUX_FIDELITY_SAMPLE_SEED = UInt64(0x6a09e667f3bcc909)
const OMEGA_FIDELITY_PRESSURE_BANDS =
    ("surface-700", "700-300", "300-100", "100-10", "<10")
# Header-key prefixes of the write-time certificate (`FLUX_CERTIFICATE_KEY_PREFIXES`,
# MetDrivers): patched at close, copied verbatim by payload-preserving tools.

# The deterministic sample rule of each target topology (header
# `flux_fidelity_sample_rule`); certificates without samples record "none".
fidelity_sample_rule(::Type{CubedSphereTargetGeometry}) = "cs_stratified_faces_v1"
fidelity_sample_rule(::Type{LatLonTargetGeometry}) = "structured_hashed_faces_v1"
fidelity_sample_rule(::Type{ReducedGaussianTargetGeometry}) = "rg_hashed_latitude_lines_v1"

# ---------------------------------------------------------------------------
# Typed reasons. Each carries its status; strings appear only in the header.
# ---------------------------------------------------------------------------

abstract type AbstractFluxFidelityReason end
struct EvidenceNotAttached <: AbstractFluxFidelityReason end
struct SourceHasNoWinds <: AbstractFluxFidelityReason end
struct SourcePayloadOnly <: AbstractFluxFidelityReason end
struct SourceNotOpened <: AbstractFluxFidelityReason end
struct SourceHasNoOmega <: AbstractFluxFidelityReason end
"Native (GEOS) fluxes have no independent source to rederive them from."
struct NativeFluxesNotRederivable <: AbstractFluxFidelityReason end

fidelity_reason(::EvidenceNotAttached) = "evidence_not_attached"
fidelity_reason(::SourceHasNoWinds) = "source_has_no_winds"
fidelity_reason(::SourcePayloadOnly) = "source_payload_only"
fidelity_reason(::SourceNotOpened) = "source_not_opened"
fidelity_reason(::SourceHasNoOmega) = "source_has_no_omega"
fidelity_reason(::NativeFluxesNotRederivable) = "native_fluxes_not_rederivable"

# `not_applicable`: the check cannot exist for this source; `unavailable`: it
# could, but its input was not opened or attached.
fidelity_status_label(::Union{SourceHasNoWinds, SourcePayloadOnly, SourceHasNoOmega,
                              NativeFluxesNotRederivable}) = "not_applicable"
fidelity_status_label(::Union{EvidenceNotAttached, SourceNotOpened}) = "unavailable"

# ---------------------------------------------------------------------------
# Evidence types
# ---------------------------------------------------------------------------

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
    time_sampling::String          # the window's effective sampling
    amount_seconds::Float64
end

"One sampled native face/level: the native amount before and after the balance."
struct FluxBalanceSample{FT}
    face::Int
    level::Int
    native::FT
    balanced::FT
end

"""
Native-flux windows (GEOS): no independent reconstruction exists, so the
evidence records only how far the column balance moved the sampled native
amounts; the certificate status is `not_applicable`.
"""
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

"""
    OmegaFidelityBaseline(id; nrmse_limit, correlation_min, slope_min, slope_max)

Named, versioned acceptance gates of a path's OMEGA comparison. A certificate
passes when it has samples, every metric is finite, `NRMSE < nrmse_limit`,
correlation `> correlation_min` and `slope_min < slope < slope_max`; the
defaults are `OMEGA_FIDELITY_NRMSE_LIMIT`, `OMEGA_FIDELITY_CORRELATION_MIN`,
`OMEGA_FIDELITY_SLOPE_MIN` and `OMEGA_FIDELITY_SLOPE_MAX` (Parameters).
"""
struct OmegaFidelityBaseline
    id::String
    nrmse_limit::Float64
    correlation_min::Float64
    slope_min::Float64
    slope_max::Float64
end
OmegaFidelityBaseline(id::AbstractString;
                      nrmse_limit::Real = OMEGA_FIDELITY_NRMSE_LIMIT,
                      correlation_min::Real = OMEGA_FIDELITY_CORRELATION_MIN,
                      slope_min::Real = OMEGA_FIDELITY_SLOPE_MIN,
                      slope_max::Real = OMEGA_FIDELITY_SLOPE_MAX) =
    OmegaFidelityBaseline(String(id), Float64(nrmse_limit), Float64(correlation_min),
                          Float64(slope_min), Float64(slope_max))

# Per-path baselines. MERRA-2 wind-derived C90 measures NRMSE 0.47, r 0.886,
# slope 0.865 (2022-01 golden); GEOS OMEGA closures use the same gates, which
# no golden measures yet. v3: the slope is bounded on both sides.
const MERRA2_WIND_OMEGA_BASELINE = OmegaFidelityBaseline("merra2_wind_omega_v3")
const GEOS_NATIVE_OMEGA_BASELINE = OmegaFidelityBaseline("geos_native_omega_v3")

struct SourceOmegaEvidence{G,FT,S} <: AbstractOmegaFidelityEvidence{G,FT}
    samples::S
    source::String
    sampling::String
    basis::String
    baseline::OmegaFidelityBaseline
end

struct NoSourceOmegaEvidence{G,FT,R<:AbstractFluxFidelityReason} <:
       AbstractOmegaFidelityEvidence{G,FT}
    reason::R
end

"""
Per-window record of a cubed-sphere column balance and its `cm` diagnosis
(`balance_cs_column_mass_fluxes!`, `diagnose_cs_cm!`): the largest projected
column residual `max_c |r_c^⊥|` and its ratio to the cell's closure tolerance
`τ_c = max(closure_tol, 8 ε_FT Σ_{f∂c} Σ_k |F_{f,k}|)`, and the largest
residual `|R_c|` folded into `cm` with its ratio to the cell's tolerance.
Both ratios are at most 1 for a window that passed. The projected residual
describes the fluxes as stored (`full_window_column_closure_record` for GEOS
full-window amounts); the `cm` fields describe the diagnosis of `cm`.
"""
abstract type AbstractColumnClosureRecord end
struct ColumnClosureRecord <: AbstractColumnClosureRecord
    projected_residual_kg :: Float64
    residual_to_tolerance :: Float64
    booked_residual_kg    :: Float64
    cm_residual_to_tolerance :: Float64
end
"No cubed-sphere column balance ran (lat-lon/reduced Gaussian, payload-only, pressure fixer)."
struct NoColumnClosureRecord <: AbstractColumnClosureRecord end

column_closure_record(balance_diag, cm_diag) = ColumnClosureRecord(
    Float64(balance_diag.final_column_projected_residual),
    Float64(balance_diag.final_column_residual_ratio),
    Float64(cm_diag.max_booked_residual),
    Float64(cm_diag.max_raw_residual_ratio))

"""
    full_window_column_closure_record(record, am, bm, m, m_next, ft, steps, scratch) -> record′

Closure record of a payload stored as full-window amounts (GEOS): the balanced
per-application fluxes multiplied by `2 steps` in the storage type, a product
that rounds unless `2 steps` is a power of two. The column closure is
evaluated again on the stored `am`, `bm` in per-application units
(`recheck_cs_column_closure` with `flux_scale = 1/(2 steps)`), so the projected
residual and its ratio describe the payload as written; kg values are stated
in the stored (full-window) amounts. The `cm` fields keep the residual folded
into `cm` when it was diagnosed, per application, in stored amounts.
"""
full_window_column_closure_record(r::NoColumnClosureRecord, _am, _bm, _m, _m_next,
                                  _ft, _steps, _scratch) = r
function full_window_column_closure_record(r::ColumnClosureRecord, am, bm, m, m_next,
                                           ft::CSGlobalFaceTable, steps::Integer,
                                           scratch::CSPoissonScratch)
    amounts = 2 * Int(steps)
    stored = recheck_cs_column_closure(am, bm, m, m_next, ft, Int(steps), scratch, (;);
                                       flux_scale = 1 / amounts)
    return ColumnClosureRecord(amounts * Float64(stored.final_column_projected_residual),
                               Float64(stored.final_column_residual_ratio),
                               amounts * r.booked_residual_kg, r.cm_residual_to_tolerance)
end

"""
    rescaled_column_closure_record(record, am, bm, m, m_next, ft, balanced_steps, steps, scratch) -> record′

Closure record of a payload balanced for `balanced_steps` substeps and then
rescaled to `steps` (the adaptive output schedule of the LL → CS regrid,
`rescale_substep_amounts!`). The stored `am`, `bm`, `cm` are the balanced
amounts times `balanced_steps/steps` in the storage type, a product that rounds
unless the ratio is a power of two. The column closure is evaluated again on
the stored `am`, `bm` for `steps` substeps (`recheck_cs_column_closure`), so the
projected residual and its ratio describe the payload as written. The residual
folded into `cm` is scaled with `cm`; its ratio to the tolerance is kept from
the diagnosis (an upper bound: the tolerance floor `closure_tol` does not scale).
With `steps == balanced_steps` nothing was rescaled and `record` is returned.
"""
rescaled_column_closure_record(r::NoColumnClosureRecord, _am, _bm, _m, _m_next,
                               _ft, _balanced_steps, _steps, _scratch) = r
function rescaled_column_closure_record(r::ColumnClosureRecord, am, bm, m, m_next,
                                        ft::CSGlobalFaceTable, balanced_steps::Integer,
                                        steps::Integer, scratch::CSPoissonScratch)
    steps == balanced_steps && return r
    stored = recheck_cs_column_closure(am, bm, m, m_next, ft, Int(steps), scratch, (;))
    return ColumnClosureRecord(Float64(stored.final_column_projected_residual),
                               Float64(stored.final_column_residual_ratio),
                               r.booked_residual_kg * balanced_steps / steps,
                               r.cm_residual_to_tolerance)
end

struct FluxFidelityEvidenceBundle{F,O,C<:AbstractColumnClosureRecord}
    flux::F
    omega::O
    closure::C
end

function default_fidelity_evidence(::Type{G}, ::Type{FT}) where {G,FT}
    return FluxFidelityEvidenceBundle(
        NoFluxFidelityEvidence{G,FT,SourcePayloadOnly}(SourcePayloadOnly()),
        NoSourceOmegaEvidence{G,FT,SourceHasNoOmega}(SourceHasNoOmega()),
        NoColumnClosureRecord())
end

# ---------------------------------------------------------------------------
# Deterministic sample maps
# ---------------------------------------------------------------------------

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

"""
Sampled faces of a structured lat-lon target (`structured_hashed_faces_v1`):
canonical ids `1:na` for the `(Nx+1) × Ny` x faces, `na+1:na+nb` for the
`Nx × (Ny+1)` y faces, ranked once by the SplitMix64 hash. Built once per
workspace; the check evaluates only these faces' columns.
"""
struct StructuredFidelitySampler
    Nx::Int
    Ny::Int
    faces::Vector{Int}
end

function StructuredFidelitySampler(Nx::Integer, Ny::Integer; samples_per_window::Integer=256)
    nface = (Nx + 1) * Ny + Nx * (Ny + 1)
    ns = min(Int(samples_per_window), nface)
    ranked = sort(collect(1:nface); by=f -> _splitmix64(xor(FLUX_FIDELITY_SAMPLE_SEED, UInt64(f))))
    return StructuredFidelitySampler(Int(Nx), Int(Ny), sort(ranked[1:ns]))
end

"Sampled cubed-sphere cells of the OMEGA comparison, ranked once per workspace."
struct OmegaFidelityCells
    Nc::Int
    cells::Vector{Int}
end

function OmegaFidelityCells(Nc::Integer; samples_per_window::Integer=256)
    nc = 6 * Int(Nc) * Int(Nc)
    ns = min(Int(samples_per_window), nc)
    ranked = sort(collect(1:nc); by=c -> _splitmix64(xor(
        FLUX_FIDELITY_SAMPLE_SEED, UInt64(c), UInt64(0x0a11ce))))
    return OmegaFidelityCells(Int(Nc), sort(ranked[1:ns]))
end

# ---------------------------------------------------------------------------
# Accumulator: per-window verification and the aggregate header certificate
# ---------------------------------------------------------------------------

mutable struct FluxFidelityAccumulator
    sample_rule::String
    wind_windows::Int
    native_windows::Int
    none_windows::Int
    none_reason::Union{Nothing,AbstractFluxFidelityReason}
    method::String
    time_sampling::Vector{String}
    samples_per_window::Int
    sample_count::Int
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
    balance_sumsq_delta::Float64
    balance_sumsq_native::Float64
    balance_count::Int
    balance_scale_floor::Float64
    balance_level_delta2::Vector{Float64}
    balance_level_native2::Vector{Float64}
    balance_level_count::Vector{Int}
    omega_windows::Int
    omega_none_windows::Int
    omega_none_reason::Union{Nothing,AbstractFluxFidelityReason}
    omega_source::String
    omega_sampling::String
    omega_basis::String
    omega_baseline::Union{Nothing,OmegaFidelityBaseline}
    omega_count::Int
    omega_sum_error::Float64
    omega_sumsq_error::Float64
    omega_max_abs::Float64
    omega_sum_source::Float64
    omega_sum_candidate::Float64
    omega_sum_source2::Float64
    omega_sum_candidate2::Float64
    omega_sum_cross::Float64
    omega_bottom_max::Float64
    omega_band_error2::Vector{Float64}
    omega_band_source2::Vector{Float64}
    omega_band_count::Vector{Int}
    windows_seen::Int
    closure_windows::Int
    closure_projected_max::Float64
    closure_ratio_max::Float64
    closure_booked_max::Float64
    closure_cm_ratio_max::Float64
end

FluxFidelityAccumulator() = FluxFidelityAccumulator(
    "none", 0, 0, 0, nothing, "", String[], 0, 0, Inf, -Inf,
    0.0, 0.0, 0.0, 0.0, UInt64(0), 0.0, 0.0, 0.0, 0, 0, 0,
    0.0, 0.0, 0.0, 0, 0.0, Float64[], Float64[], Int[],
    0, 0, nothing, "", "", "", nothing, 0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    zeros(5), zeros(5), zeros(Int, 5),
    0, 0, 0.0, 0.0, 0.0, 0.0)

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

# Shared per-window bookkeeping of sampled evidence (wind or native).
function _record_sampled_window!(acc::FluxFidelityAccumulator, ::Type{G}, evidence,
                                 win, method::AbstractString,
                                 time_sampling::AbstractString) where G
    window_samples = length(evidence.samples)
    window_samples > 0 && evidence.samples_per_window > 0 || throw(ArgumentError(
        "flux-fidelity evidence of window $win has no samples"))
    isfinite(evidence.amount_seconds) && evidence.amount_seconds > 0 || throw(ArgumentError(
        "flux-fidelity evidence of window $win has amount duration $(evidence.amount_seconds) s"))
    if acc.samples_per_window == 0
        acc.samples_per_window = evidence.samples_per_window
        acc.method = String(method)
        acc.sample_rule = fidelity_sample_rule(G)
    else
        acc.samples_per_window == evidence.samples_per_window && acc.method == method ||
            throw(ArgumentError("flux-fidelity evidence of window $win changes its sample " *
                                "map or method ($(evidence.samples_per_window) samples, " *
                                "$method) from earlier windows"))
    end
    push!(acc.time_sampling, String(time_sampling))
    acc.sample_count += window_samples
    acc.amount_min = min(acc.amount_min, evidence.amount_seconds)
    acc.amount_max = max(acc.amount_max, evidence.amount_seconds)
    return nothing
end

# One sampled face/level of the column-balance correction `Δ = F_balanced − F`
# of its pre-balance amount `F` (the wind reference or the native amount):
# flux-weighted sums, globally and per level (`level` 1 = top).
@inline function _record_balance_correction!(acc::FluxFidelityAccumulator, native::Float64,
                                             balanced::Float64, level::Integer)
    correction = balanced - native
    isfinite(correction) || throw(ArgumentError("flux balance correction is not finite"))
    level >= 1 || throw(ArgumentError("flux balance sample has level $level"))
    if level > length(acc.balance_level_count)
        n = Int(level)
        append!(acc.balance_level_delta2, zeros(n - length(acc.balance_level_delta2)))
        append!(acc.balance_level_native2, zeros(n - length(acc.balance_level_native2)))
        append!(acc.balance_level_count, zeros(Int, n - length(acc.balance_level_count)))
    end
    acc.balance_max_abs = max(acc.balance_max_abs, abs(correction))
    acc.balance_sumsq_delta += correction * correction
    acc.balance_sumsq_native += native * native
    acc.balance_count += 1
    acc.balance_level_delta2[level] += correction * correction
    acc.balance_level_native2[level] += native * native
    acc.balance_level_count[level] += 1
    return nothing
end

function verify_flux_fidelity!(acc::FluxFidelityAccumulator,
                               evidence::WindDerivedFluxEvidence{G}, win) where G
    samples = evidence.samples
    _record_sampled_window!(acc, G, evidence, win, String(evidence.method),
                            evidence.time_sampling)
    FT = typeof(first(samples).reference)
    rel_tol = 256 * eps(FT)
    max_scale = maximum(s -> max(abs(Float64(s.reference)), abs(Float64(s.candidate))), samples)
    scale_floor = max(floatmin(FT), eps(FT) * max(max_scale, 1.0))
    abs_tol = 64 * eps(FT) * max(max_scale, 1.0)
    for s in samples
        ref = Float64(s.reference); got = Float64(s.candidate)
        isfinite(ref) && isfinite(got) || throw(ArgumentError(
            "flux-fidelity gate FAILED before write for window $win face=$(s.face) " *
            "level=$(s.level): non-finite reference=$ref candidate=$got"))
        ae = abs(got - ref)
        scale = max(abs(ref), abs(got), scale_floor)
        re = ae / scale
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
        acc.max_ulp = max(acc.max_ulp, _ulp_distance(s.reference, s.candidate))
        _record_balance_correction!(acc, ref, Float64(s.balanced), s.level)
    end
    acc.wind_windows += 1
    acc.abs_tol = max(acc.abs_tol, abs_tol)
    acc.rel_tol = max(acc.rel_tol, rel_tol)
    acc.scale_floor = max(acc.scale_floor, scale_floor)
    acc.balance_scale_floor = max(acc.balance_scale_floor, scale_floor)
    return nothing
end

function verify_flux_fidelity!(acc::FluxFidelityAccumulator,
                               evidence::NativeFluxEvidence{G}, win) where G
    samples = evidence.samples
    _record_sampled_window!(acc, G, evidence, win, evidence.method, "native")
    # The correction is measured against the pre-balance (native) amounts, as
    # for wind evidence against the wind reference.
    FT = typeof(first(samples).native)
    max_scale = maximum(s -> abs(Float64(s.native)), samples)
    scale_floor = max(floatmin(FT), eps(FT) * max(max_scale, 1.0))
    for s in samples
        _record_balance_correction!(acc, Float64(s.native), Float64(s.balanced), s.level)
    end
    acc.native_windows += 1
    acc.balance_scale_floor = max(acc.balance_scale_floor, scale_floor)
    return nothing
end

function verify_flux_fidelity!(acc::FluxFidelityAccumulator,
                               e::NoFluxFidelityEvidence, _win)
    acc.none_reason === nothing || typeof(acc.none_reason) === typeof(e.reason) ||
        throw(ArgumentError("windows record different flux-fidelity reasons: " *
                            "$(fidelity_reason(acc.none_reason)), $(fidelity_reason(e.reason))"))
    acc.none_reason = e.reason
    acc.none_windows += 1
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
    isempty(e.samples) && throw(ArgumentError(
        "OMEGA-fidelity evidence of window $win has no samples"))
    if acc.omega_baseline === nothing
        acc.omega_baseline = e.baseline
        acc.omega_source = e.source; acc.omega_sampling = e.sampling
        acc.omega_basis = e.basis
    else
        acc.omega_baseline == e.baseline && acc.omega_source == e.source &&
            acc.omega_sampling == e.sampling && acc.omega_basis == e.basis ||
            throw(ArgumentError("OMEGA-fidelity evidence of window $win changes its " *
                                "source, sampling, basis or baseline"))
    end
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
        acc.omega_bottom_max = max(acc.omega_bottom_max, abs(Float64(s.bottom_residual_pa_s)))
        b = _omega_band(Float64(s.pressure_hpa))
        acc.omega_band_error2[b] += err * err
        acc.omega_band_source2[b] += src * src
        acc.omega_band_count[b] += 1
    end
    acc.omega_windows += 1
    return nothing
end

function verify_omega_fidelity!(acc::FluxFidelityAccumulator,
                                e::NoSourceOmegaEvidence, _win)
    acc.omega_none_reason === nothing || typeof(acc.omega_none_reason) === typeof(e.reason) ||
        throw(ArgumentError("windows record different OMEGA-fidelity reasons"))
    acc.omega_none_reason = e.reason
    acc.omega_none_windows += 1
    return nothing
end

verify_column_closure!(::FluxFidelityAccumulator, ::NoColumnClosureRecord, _win) = nothing
function verify_column_closure!(acc::FluxFidelityAccumulator, r::ColumnClosureRecord, win)
    values = (r.projected_residual_kg, r.residual_to_tolerance,
              r.booked_residual_kg, r.cm_residual_to_tolerance)
    all(isfinite, values) && all(>=(0), values) || throw(ArgumentError(
        "column-closure record of window $win is not finite and non-negative: $(r)"))
    r.residual_to_tolerance <= 1 && r.cm_residual_to_tolerance <= 1 || throw(ArgumentError(
        "column-closure record of window $win exceeds its per-cell tolerance: $(r)"))
    acc.closure_windows += 1
    acc.closure_projected_max = max(acc.closure_projected_max, r.projected_residual_kg)
    acc.closure_ratio_max = max(acc.closure_ratio_max, r.residual_to_tolerance)
    acc.closure_booked_max = max(acc.closure_booked_max, r.booked_residual_kg)
    acc.closure_cm_ratio_max = max(acc.closure_cm_ratio_max, r.cm_residual_to_tolerance)
    return nothing
end

"""
    verify_flux_fidelity!(acc, bundle::FluxFidelityEvidenceBundle, win)

Verify one window's evidence before its payload is written: sampled wind
fluxes against their independent reconstruction (throws on a failed sample),
non-empty and finite OMEGA samples, and a column-closure record within its
per-cell tolerances. Aggregates are serialized by `flux_fidelity_header!`.
"""
function verify_flux_fidelity!(acc::FluxFidelityAccumulator,
                               bundle::FluxFidelityEvidenceBundle, win::Int)
    verify_flux_fidelity!(acc, bundle.flux, win)
    verify_omega_fidelity!(acc, bundle.omega, win)
    verify_column_closure!(acc, bundle.closure, win)
    acc.windows_seen += 1
    return nothing
end

# Column-closure keys: written when every window carried a cubed-sphere
# column-balance record, absent when none did (no such balance ran).
function _column_closure_header!(header::AbstractDict, acc::FluxFidelityAccumulator)
    acc.closure_windows == 0 && return header
    acc.closure_windows == acc.windows_seen || throw(ArgumentError(
        "column-closure records cover $(acc.closure_windows) of $(acc.windows_seen) windows"))
    haskey(header, "nwindow") && Int(header["nwindow"]) != acc.closure_windows &&
        throw(ArgumentError("column-closure records cover $(acc.closure_windows) windows; " *
                            "the header has nwindow=$(header["nwindow"])"))
    header["column_closure_rule"] = COLUMN_CLOSURE_RULE
    header["column_closure_windows"] = acc.closure_windows
    header["column_closure_projected_residual_max_kg"] = acc.closure_projected_max
    header["column_closure_residual_to_tolerance_max"] = acc.closure_ratio_max
    header["column_closure_booked_residual_max_kg"] = acc.closure_booked_max
    header["column_closure_cm_residual_to_tolerance_max"] = acc.closure_cm_ratio_max
    return header
end

# The sample count of a certificate must cover every window, every sampled
# face and every level of the header it is patched into.
function _check_certificate_coverage(header::AbstractDict, windows::Int,
                                     samples_per_window::Int, sample_count::Int)
    haskey(header, "nwindow") && Int(header["nwindow"]) != windows && throw(ArgumentError(
        "flux-fidelity evidence covers $windows windows; the header has nwindow=$(header["nwindow"])"))
    haskey(header, "nlevel") &&
        sample_count != windows * samples_per_window * Int(header["nlevel"]) &&
        throw(ArgumentError("flux-fidelity sample count $sample_count != windows " *
            "($windows) × samples per window ($samples_per_window) × nlevel ($(header["nlevel"]))"))
    return nothing
end

function _write_time_sampling!(header::AbstractDict, sampling::Vector{String})
    if all(==(first(sampling)), sampling)
        header["flux_fidelity_time_sampling"] = first(sampling)
    else
        header["flux_fidelity_time_sampling"] = "mixed"
        header["flux_fidelity_time_sampling_by_window"] = copy(sampling)
    end
    return header
end

# Flux-weighted size of the column-balance correction: `RMS(Δ)/RMS(F)` over
# every sampled face, level and window, and the largest per-level value with
# its level. RMS(F) is floored at the scale floor `F_floor`, so a level whose
# sampled pre-balance amounts are all zero still gives a finite value.
function _balance_relative_rms(delta2::Float64, native2::Float64, count::Int, floor::Float64)
    count > 0 || return 0.0
    return sqrt(delta2) / max(sqrt(native2), sqrt(count) * floor)
end

function _write_balance_correction!(header::AbstractDict, acc::FluxFidelityAccumulator)
    floor = acc.balance_scale_floor
    worst_level, worst = 0, -1.0
    for k in eachindex(acc.balance_level_count)
        r = _balance_relative_rms(acc.balance_level_delta2[k], acc.balance_level_native2[k],
                                  acc.balance_level_count[k], floor)
        r > worst && ((worst_level, worst) = (k, r))
    end
    header["balance_correction_max_abs_kg"] = acc.balance_max_abs
    header["balance_correction_rms_relative"] = _balance_relative_rms(
        acc.balance_sumsq_delta, acc.balance_sumsq_native, acc.balance_count, floor)
    header["balance_correction_max_level_rms_relative"] = worst
    header["balance_correction_max_level"] = worst_level
    return header
end

function _write_flux_certificate!(header::AbstractDict, acc::FluxFidelityAccumulator)
    categories = count(>(0), (acc.wind_windows, acc.native_windows, acc.none_windows))
    categories <= 1 || throw(ArgumentError(
        "mixed flux-fidelity evidence: $(acc.wind_windows) wind-derived, " *
        "$(acc.native_windows) native and $(acc.none_windows) unchecked windows"))
    header["flux_fidelity_sample_rule"] = acc.sample_rule
    if acc.wind_windows > 0
        n = acc.sample_count
        _check_certificate_coverage(header, acc.wind_windows, acc.samples_per_window, n)
        header["flux_fidelity_status"] = "passed"
        header["flux_fidelity_kind"] = "wind_rederived"
        header["flux_fidelity_samples_per_window"] = acc.samples_per_window
        header["flux_fidelity_sample_count"] = n
        header["flux_fidelity_windows"] = acc.wind_windows
        header["flux_fidelity_method"] = acc.method
        _write_time_sampling!(header, acc.time_sampling)
        header["flux_fidelity_amount_seconds_min"] = acc.amount_min
        header["flux_fidelity_amount_seconds_max"] = acc.amount_max
        header["flux_fidelity_max_abs_kg"] = acc.max_abs
        header["flux_fidelity_rms_abs_kg"] = sqrt(acc.sumsq_abs / n)
        header["flux_fidelity_max_rel"] = acc.max_rel
        header["flux_fidelity_rms_rel"] = sqrt(acc.sumsq_rel / n)
        header["flux_fidelity_max_ulp"] = acc.max_ulp
        header["flux_fidelity_abs_tol_kg"] = acc.abs_tol
        header["flux_fidelity_rel_tol"] = acc.rel_tol
        header["flux_fidelity_scale_floor_kg"] = acc.scale_floor
        header["flux_fidelity_worst_window"] = acc.worst_window
        header["flux_fidelity_worst_face"] = acc.worst_face
        header["flux_fidelity_worst_level"] = acc.worst_level
        _write_balance_correction!(header, acc)
    elseif acc.native_windows > 0
        n = acc.sample_count
        _check_certificate_coverage(header, acc.native_windows, acc.samples_per_window, n)
        reason = NativeFluxesNotRederivable()
        header["flux_fidelity_status"] = fidelity_status_label(reason)
        header["flux_fidelity_kind"] = "native_passthrough"
        header["flux_fidelity_reason"] = fidelity_reason(reason)
        header["flux_fidelity_samples_per_window"] = acc.samples_per_window
        header["flux_fidelity_sample_count"] = n
        header["flux_fidelity_windows"] = acc.native_windows
        header["flux_fidelity_method"] = acc.method
        _write_time_sampling!(header, acc.time_sampling)
        header["flux_fidelity_amount_seconds_min"] = acc.amount_min
        header["flux_fidelity_amount_seconds_max"] = acc.amount_max
        _write_balance_correction!(header, acc)
    else
        reason = something(acc.none_reason, EvidenceNotAttached())
        header["flux_fidelity_sample_rule"] = "none"
        header["flux_fidelity_status"] = fidelity_status_label(reason)
        header["flux_fidelity_kind"] = "none"
        header["flux_fidelity_reason"] = fidelity_reason(reason)
        header["flux_fidelity_samples_per_window"] = 0
        header["flux_fidelity_sample_count"] = 0
        header["flux_fidelity_windows"] = 0
    end
    return header
end

function _write_omega_certificate!(header::AbstractDict, acc::FluxFidelityAccumulator)
    acc.omega_windows > 0 && acc.omega_none_windows > 0 && throw(ArgumentError(
        "OMEGA-fidelity evidence covers $(acc.omega_windows) of " *
        "$(acc.omega_windows + acc.omega_none_windows) windows"))
    if acc.omega_windows == 0
        reason = something(acc.omega_none_reason, SourceNotOpened())
        header["omega_fidelity_status"] = fidelity_status_label(reason)
        header["omega_fidelity_reason"] = fidelity_reason(reason)
        header["omega_fidelity_source"] = "none"
        header["omega_fidelity_sampling"] = "none"
        header["omega_fidelity_basis"] = "none"
        header["omega_fidelity_sample_count"] = 0
        return header
    end
    baseline = acc.omega_baseline
    n = acc.omega_count
    rmse = sqrt(acc.omega_sumsq_error / n)
    source_rms = sqrt(acc.omega_sum_source2 / n)
    nrmse = rmse / max(source_rms, eps(Float64))
    mx = acc.omega_sum_source / n
    my = acc.omega_sum_candidate / n
    sxx = acc.omega_sum_source2 - n * mx * mx
    syy = acc.omega_sum_candidate2 - n * my * my
    sxy = acc.omega_sum_cross - n * mx * my
    slope = sxy / max(sxx, eps(Float64))
    correlation = sxy / sqrt(max(sxx * syy, eps(Float64)))
    band_nrmse = [sqrt(acc.omega_band_error2[i] / max(acc.omega_band_count[i], 1)) /
                  max(sqrt(acc.omega_band_source2[i] / max(acc.omega_band_count[i], 1)),
                      eps(Float64)) for i in 1:5]
    metrics = (acc.omega_sum_error / n, rmse, nrmse, acc.omega_max_abs, slope,
               correlation, acc.omega_bottom_max)
    gates = (nrmse = nrmse < baseline.nrmse_limit,
             correlation = correlation > baseline.correlation_min,
             slope = baseline.slope_min < slope < baseline.slope_max)
    passed = all(isfinite, metrics) && all(isfinite, band_nrmse) && all(values(gates))
    header["omega_fidelity_status"] = passed ? "passed" : "failed"
    passed || (header["omega_fidelity_reason"] = "baseline_gate_failed")
    header["omega_fidelity_source"] = acc.omega_source
    header["omega_fidelity_sampling"] = acc.omega_sampling
    header["omega_fidelity_basis"] = acc.omega_basis
    header["omega_fidelity_sample_count"] = n
    header["omega_fidelity_bias_pa_s"] = metrics[1]
    header["omega_fidelity_rmse_pa_s"] = rmse
    header["omega_fidelity_nrmse"] = nrmse
    header["omega_fidelity_max_abs_pa_s"] = acc.omega_max_abs
    header["omega_fidelity_slope"] = slope
    header["omega_fidelity_correlation"] = correlation
    header["omega_fidelity_bottom_residual_pa_s"] = acc.omega_bottom_max
    header["omega_fidelity_pressure_bands_hpa"] = collect(OMEGA_FIDELITY_PRESSURE_BANDS)
    header["omega_fidelity_band_nrmse"] = band_nrmse
    header["omega_fidelity_baseline_id"] = baseline.id
    header["omega_fidelity_nrmse_limit"] = baseline.nrmse_limit
    header["omega_fidelity_correlation_min"] = baseline.correlation_min
    header["omega_fidelity_slope_min"] = baseline.slope_min
    header["omega_fidelity_slope_max"] = baseline.slope_max
    header["omega_fidelity_passed"] = passed
    failed = join((String(k) for (k, ok) in pairs(gates) if !ok), ", ")
    passed || throw(ArgumentError(
        "aggregate OMEGA-fidelity gate FAILED ($(baseline.id)): nrmse=$nrmse " *
        "(limit < $(baseline.nrmse_limit)), correlation=$correlation " *
        "(> $(baseline.correlation_min)), slope=$slope (between $(baseline.slope_min) " *
        "and $(baseline.slope_max))" *
        (isempty(failed) ? "; non-finite metric" : "; failed: $failed")))
    return header
end

"""
    flux_fidelity_header!(header, acc) -> header

Replace every certificate key of `header` (`FLUX_CERTIFICATE_KEY_PREFIXES`) by
the aggregate of `acc`. Consistency is enforced at write time: one kind of flux
evidence for all windows, sample counts that cover `nwindow` windows and
`nlevel` levels when the header has them, OMEGA evidence for all windows or
none, and column-closure records for all windows or none. A failed aggregate
OMEGA gate writes its `failed` certificate and then throws, so the staged
binary is quarantined.
"""
function flux_fidelity_header!(header::AbstractDict, acc::FluxFidelityAccumulator)
    for key in collect(keys(header))
        _is_certificate_key(String(key)) && delete!(header, key)
    end
    _column_closure_header!(header, acc)
    header["flux_fidelity_schema"] = FLUX_FIDELITY_SCHEMA
    header["flux_fidelity_sample_seed"] = string(FLUX_FIDELITY_SAMPLE_SEED)
    _write_flux_certificate!(header, acc)
    _write_omega_certificate!(header, acc)
    return header
end

"""
    payload_only_fidelity_header!(header, G, FT) -> header

The certificate of a writer that only transforms an existing binary's payload
(coarsening): flux and OMEGA fidelity `not_applicable`, reason
`source_payload_only`, through the same serializer as every other writer.
"""
function payload_only_fidelity_header!(header::AbstractDict, ::Type{G}, ::Type{FT}) where {G,FT}
    acc = FluxFidelityAccumulator()
    verify_flux_fidelity!(acc, FluxFidelityEvidenceBundle(
        NoFluxFidelityEvidence{G,FT,SourcePayloadOnly}(SourcePayloadOnly()),
        NoSourceOmegaEvidence{G,FT,SourcePayloadOnly}(SourcePayloadOnly()),
        NoColumnClosureRecord()), 1)
    return flux_fidelity_header!(header, acc)
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

"Native (GEOS) amounts at the sampled faces, before the column balance."
struct CapturedNativeFluxSamples{FT}
    face::Vector{Int}
    level::Vector{Int}
    native::Vector{FT}
end

function scale_captured_flux_samples(c::CapturedWindFluxSamples{FT}, factor::Real) where FT
    return CapturedWindFluxSamples(copy(c.face), copy(c.level),
        FT.(factor .* c.reference), FT.(factor .* c.candidate))
end

scale_captured_flux_samples(c::CapturedNativeFluxSamples{FT}, factor::Real) where FT =
    CapturedNativeFluxSamples(copy(c.face), copy(c.level), FT.(factor .* c.native))

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

"""
Moist hybrid layer thickness `|ΔA_k + ΔB_k p_s|` of the line-integral source
cells, evaluated lazily at the sampled stencil cells (as the production
`line_integral_face_fluxes!` with `ps`) instead of a full source-grid array.
"""
struct HybridLayerThickness{P<:AbstractVector}
    dA::Vector{Float64}
    dB::Vector{Float64}
    ps::P
    # From interface coefficients `A`, `B` (top first) and source-cell `ps`.
    function HybridLayerThickness(A::AbstractVector, B::AbstractVector,
                                  ps::P) where {P<:AbstractVector}
        length(A) == length(B) || throw(DimensionMismatch("A and B lengths differ"))
        return new{P}([Float64(A[k + 1]) - Float64(A[k]) for k in 1:(length(A) - 1)],
                      [Float64(B[k + 1]) - Float64(B[k]) for k in 1:(length(B) - 1)], ps)
    end
end

@inline _layer_thickness(dp::AbstractMatrix, c, k) = Float64(dp[c, k])
@inline _layer_thickness(t::HybridLayerThickness, c, k) =
    abs(t.dA[k] + t.dB[k] * Float64(t.ps[c]))

"""
    capture_native_flux_samples(am, bm, ft; sample_map) -> CapturedNativeFluxSamples

The native amounts at the sampled faces before the column balance; the
certificate later compares them with the stored (balanced) amounts. Native
fluxes have no independent source, so nothing here can test their fidelity.
"""
function capture_native_flux_samples(am, bm, ft::CSGlobalFaceTable;
                                     sample_map=FidelitySampleMap(ft))
    FT = eltype(am[1]); Nz = size(am[1], 3)
    n = length(sample_map.faces) * Nz
    face = Vector{Int}(undef, n); level = Vector{Int}(undef, n)
    native = Vector{FT}(undef, n)
    q = 0
    for f in sample_map.faces, k in 1:Nz
        q += 1; face[q] = f; level[q] = k
        native[q] = _canonical_face_value(am, bm, ft, f, k)
    end
    return CapturedNativeFluxSamples(face, level, native)
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

# The vector evaluator re-evaluates the face formula, but with the production
# face geometry and FV3 filter (`_vector_face_value`, `geom.len`): it catches
# assembly, scatter and time-scaling errors, not geometry errors.
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

"""
    capture_cs_wind_flux_samples(method::LineIntegralFaceFluxes, raw_am, raw_bm,
                                 u, v, dp, ft, gravity, dt_factor, Nc, Nz;
                                 sample_map, line_rows)

Sampled line integrals from the source winds `u`, `v` (`ncell × Nz`) and layer
thickness `dp`: a source-grid matrix (dry thickness) or a
`HybridLayerThickness` evaluated lazily (moist), so no full source-grid
buffer is built per endpoint.
"""
function capture_cs_wind_flux_samples(method::LineIntegralFaceFluxes,
                                      raw_am, raw_bm, u::AbstractMatrix,
                                      v::AbstractMatrix,
                                      dp::Union{AbstractMatrix, HybridLayerThickness},
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
                thick = _layer_thickness(dp, c, k)
                uu = Float64(u[c, k]) * thick
                vv = Float64(v[c, k]) * thick
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

function native_flux_evidence(::Type{G}, captured::CapturedNativeFluxSamples{FT},
                              balanced_am, balanced_bm, ft::CSGlobalFaceTable;
                              method::AbstractString,
                              samples_per_window::Integer,
                              amount_seconds::Real) where {G,FT}
    samples = Vector{FluxBalanceSample{FT}}(undef, length(captured.face))
    for q in eachindex(samples)
        f, k = captured.face[q], captured.level[q]
        samples[q] = FluxBalanceSample(f, k, captured.native[q],
            FT(_canonical_face_value(balanced_am, balanced_bm, ft, f, k)))
    end
    return NativeFluxEvidence{G,FT,typeof(samples)}(
        samples, Int(samples_per_window), String(method), Float64(amount_seconds))
end

# ---------------------------------------------------------------------------
# Structured lat-lon: only the sampled faces' columns are rebuilt.
# ---------------------------------------------------------------------------

# Vertical map of one sampled column, as `merge_cell_field!` (merge map) or
# `remap_extensive_field_pressure_overlap!` (pressure overlap) does it.
function _map_structured_fidelity_column!(column::AbstractVector{FT}, native::AbstractVector{FT},
                                          vertical, ps_face::Float64,
                                          ::Val{:merge_map}) where FT
    fill!(column, zero(FT))
    mm = vertical.merge_map
    @inbounds for k in eachindex(mm)
        column[mm[k]] += native[k]
    end
    return column
end
_map_structured_fidelity_column!(column, native, vertical, ps_face::Float64,
                                 ::Val{:pressure_overlap}) =
    _remap_extensive_column_pressure_overlap!(column, native, ps_face,
                                              vertical.vc_native, vertical.merged_vc)

"""
    capture_structured_wind_flux_samples(sampler, am, bm, transform, qv, grid,
                                         vertical, settings)

Rebuild the sampled lat-lon face amounts from the staggered spectral winds,
face pressure thickness and (dry basis) face humidity, column by column: the
native column at the face, then the configured vertical map. The pole rows are
ordinary cells whose zonal faces carry flux (issue #58); only the meridional
faces on the poles (`bm[:, 1, :]`, `bm[:, Ny+1, :]`) are closed. Memory is
`O(Nz)`; no full-grid flux array is built.
"""
function capture_structured_wind_flux_samples(sampler::StructuredFidelitySampler,
                                              am::AbstractArray{FT,3},
                                              bm::AbstractArray{FT,3},
                                              transform, qv, grid, vertical,
                                              settings) where FT
    Nx, Ny, Nz_native = size(transform.dp)
    (Nx, Ny) == (sampler.Nx, sampler.Ny) || throw(DimensionMismatch(
        "fidelity sampler is for $(sampler.Nx)×$(sampler.Ny), the grid is $(Nx)×$(Ny)"))
    Nz = size(am, 3)
    R_g = Float64(grid.mesh.radius) / Float64(STANDARD_GRAVITY)
    dlon, dlat = Float64(grid.dlon), Float64(grid.dlat)
    dt_amount = Float64(settings.half_dt)
    dry = settings.mass_basis == :dry
    qfield = hasproperty(qv, :qv_native) ? qv.qv_native : nothing
    dry && qfield === nothing && throw(ArgumentError(
        "dry structured fidelity reconstruction requires native humidity"))
    dA, dB = vertical.ab.dA, vertical.ab.dB
    mapping = Val(vertical_mapping_method(vertical))
    sp = transform.sp
    na = (Nx + 1) * Ny
    native = Vector{FT}(undef, Nz_native)
    column = Vector{FT}(undef, Nz)
    n = length(sampler.faces) * Nz
    face = Vector{Int}(undef, n); level = Vector{Int}(undef, n)
    reference = Vector{FT}(undef, n); candidate = Vector{FT}(undef, n)
    q = 0
    @inbounds for f in sampler.faces
        if f <= na
            i, j = Tuple(CartesianIndices((Nx + 1, Ny))[f])
            il = i == 1 ? Nx : i - 1
            ir = i <= Nx ? i : 1
            ps_face = exp((log(sp[il, j]) + log(sp[ir, j])) / 2)
            for k in 1:Nz_native
                thick = abs(Float64(dA[k]) + Float64(dB[k]) * ps_face)
                qfactor = dry ? 1 - clamp(0.5 * (Float64(qfield[il, j, k]) +
                                                 Float64(qfield[ir, j, k])), 0.0, 0.999999) : 1.0
                native[k] = FT(Float64(transform.u_stag[i, j, k]) / Float64(grid.cos_lat[j]) *
                               thick * R_g * dlat * dt_amount * qfactor)
            end
            _map_structured_fidelity_column!(column, native, vertical,
                                             _xface_surface_pressure(sp, i, j), mapping)
            reference_column = view(am, i, j, :)
        else
            i, j = Tuple(CartesianIndices((Nx, Ny + 1))[f - na])
            for k in 1:Nz_native
                native[k] = if j == 1 || j == Ny + 1
                    zero(FT)
                else
                    ps_face = exp((log(sp[i, j - 1]) + log(sp[i, j])) / 2)
                    thick = abs(Float64(dA[k]) + Float64(dB[k]) * ps_face)
                    qfactor = dry ? 1 - clamp(0.5 * (Float64(qfield[i, j - 1, k]) +
                                                     Float64(qfield[i, j, k])), 0.0, 0.999999) : 1.0
                    FT(Float64(transform.v_stag[i, j, k]) * thick * R_g * dlon * dt_amount * qfactor)
                end
            end
            _map_structured_fidelity_column!(column, native, vertical,
                                             _yface_surface_pressure(sp, i, j), mapping)
            (j == 1 || j == Ny + 1) && fill!(column, zero(FT))
            reference_column = view(bm, i, j, :)
        end
        for k in 1:Nz
            q += 1
            face[q] = f; level[q] = k
            reference[q] = reference_column[k]
            candidate[q] = column[k]
        end
    end
    return CapturedWindFluxSamples(face, level, reference, candidate)
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

# ---------------------------------------------------------------------------
# Face-indexed (reduced Gaussian): the sampled candidate values are produced
# by `ReducedFidelitySampler` during synthesis (reduced_transport_helpers.jl).
# ---------------------------------------------------------------------------

function capture_faceindexed_flux_samples(faces::AbstractVector{<:Integer},
                                          hflux::AbstractMatrix{FT},
                                          candidate_samples::AbstractMatrix{FT}) where FT
    ns, Nz = length(faces), size(hflux, 2)
    size(candidate_samples) == (ns, Nz) || throw(DimensionMismatch(
        "candidate samples $(size(candidate_samples)) for $ns faces × $Nz levels"))
    n = ns * Nz
    face = Vector{Int}(undef, n); level = Vector{Int}(undef, n)
    reference = Vector{FT}(undef, n); candidate = Vector{FT}(undef, n)
    q = 0
    for (slot, f) in pairs(faces), k in 1:Nz
        q += 1
        face[q] = f; level[q] = k
        reference[q] = hflux[f, k]
        candidate[q] = candidate_samples[slot, k]
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

# ---------------------------------------------------------------------------
# Source OMEGA on the cubed sphere
# ---------------------------------------------------------------------------

"""
    source_omega_evidence_cs(G, cm, omega, qv, ps, A, B, cell_areas, gravity,
                             amount_seconds, cells::OmegaFidelityCells;
                             source, sampling, baseline)

Compare the downward interface velocity of the stored `cm`,
`ω_cm = g cm / (A_cell Δt_amount)`, with the source OMEGA averaged to the
interface and put on the dry basis with the window's humidity `qv`, at the
cached sample cells.
"""
function source_omega_evidence_cs(::Type{G}, cm, omega, qv, ps, A, B,
                                  cell_areas, gravity, amount_seconds,
                                  cells::OmegaFidelityCells;
                                  source::AbstractString,
                                  sampling::AbstractString,
                                  baseline::OmegaFidelityBaseline) where G
    FT = eltype(cm[1])
    Nc = size(ps[1], 1); Nz = size(omega[1], 3)
    Nc == cells.Nc || throw(DimensionMismatch(
        "OMEGA sample cells are for C$(cells.Nc), the fields for C$Nc"))
    samples = OmegaFidelitySample{FT}[]
    sizehint!(samples, length(cells.cells) * (Nz - 1))
    for c in cells.cells
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
        samples, String(source), String(sampling), "dry_(1-q)_interface", baseline)
end
