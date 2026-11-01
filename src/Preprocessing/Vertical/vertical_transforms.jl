# Typed native-to-output vertical transforms. Spectral pressure overlap uses
# build_vertical_setup plus the working remap in Spectral/mass_support.jl.

# ---------------------------------------------------------------------------
# Field-kind singletons. The forward driver picks the kind per field; the
# rule is fixed here so every transform implementation honors the same
# physical semantics.
# ---------------------------------------------------------------------------

"""
    AbstractFieldKind

Singleton-type tag selecting the vertical-reduction rule
`apply_vertical!` uses for one payload field. Concrete subtypes (all
zero-size singletons) below.
"""
abstract type AbstractFieldKind end

"""Center-level extensive mass (e.g. `delp`, `m`). Sum native layers within each merged group."""
struct MassField               <: AbstractFieldKind end


"""Horizontal face mass flux (e.g. `am`, `bm`). Already integrated over the layer thickness; sum within merged group."""
struct MassFluxField           <: AbstractFieldKind end


"""Convective interface flux (e.g. `cmfmc`). Select the kept interfaces."""
struct ConvectionInterfaceFlux <: AbstractFieldKind end

"""Non-convective interface mass flux (kg/s); select retained interfaces."""
struct InterfaceMassFluxField <: AbstractFieldKind end

"""Convective center tendency (e.g. `dtrain`). Extensive at the layer; sum within merged group."""
struct ConvectionTendencyField <: AbstractFieldKind end

"""Center-level intensive field (e.g. `T`, `Q`). Mass-weighted mean within merged group; weights provided positionally."""
struct IntensiveCenterField    <: AbstractFieldKind end


# ---------------------------------------------------------------------------
# Vertical-transform abstract type + concrete policies.
# ---------------------------------------------------------------------------

"""
    AbstractVerticalTransform

Typed nominal selecting how native source levels are mapped to output
levels. Concrete subtypes:

  - `IdentityVertical`        — no merge.
  - `MergeLayersThinnerThan`  — automatic local coarsening.
  - `MergeAbovePressure`      — upper-atmosphere coarsening.
  - `LevelSelection`          — echlevs-style level selection.

A concrete transform `T` is consumed by `plan_vertical(transform,
native_vc)` to produce a `VerticalPlan{FT, T}`. `apply_vertical!`
dispatches on `(plan, ::FieldKind)` to select the right per-field rule.
"""
abstract type AbstractVerticalTransform end

"""No-op identity vertical transform. `Nz_output = Nz_native`, `merge_map[k] = k`."""
struct IdentityVertical <: AbstractVerticalTransform end

"""
    MergeLayersThinnerThan(min_thickness_Pa; reference_surface_pressure_Pa = STANDARD_PRESSURE)

Typed wrapper for the `merge_thin_levels` algorithm: merge adjacent native
layers from the surface up, closing each group as soon as its thickness at the
reference surface pressure reaches `min_thickness_Pa`. The native layers left
above the last group become the top output layer, which can be thinner than
`min_thickness_Pa` (ERA5 L137 at 100 Pa: 117 layers, top layer 6.8 Pa from
native layers 1–4); every other output layer is at least that thick.
"""
Base.@kwdef struct MergeLayersThinnerThan <: AbstractVerticalTransform
    min_thickness_Pa              :: Float64
    reference_surface_pressure_Pa :: Float64 = STANDARD_PRESSURE
end

"""
    MergeAbovePressure(pressure_Pa; target_min_thickness_Pa = Inf,
                                     reference_surface_pressure_Pa = STANDARD_PRESSURE)

Upper-atmosphere coarsening: native layers whose midpoint pressure is
LOWER than `pressure_Pa` (physically ABOVE the cutoff altitude) get
greedily merged until each merged layer exceeds
`target_min_thickness_Pa`. Below the cutoff, the native grid is
preserved verbatim.

`target_min_thickness_Pa = Inf` merges every above-cutoff native layer
into one top cap. The GEOS-IT L72 use case is
`pressure_Pa = 100.0` + `target_min_thickness_Pa = 50.0` — merges the
~14 Pa mesospheric layers into ~50 Pa output layers while keeping the
troposphere and stratosphere at native resolution.
`target_min_thickness_Pa` must be positive (`Inf` allowed): the greedy
merge never advances past a zero, negative or NaN threshold.
"""
Base.@kwdef struct MergeAbovePressure <: AbstractVerticalTransform
    pressure_Pa                   :: Float64
    target_min_thickness_Pa       :: Float64 = Inf
    reference_surface_pressure_Pa :: Float64 = STANDARD_PRESSURE

    function MergeAbovePressure(pressure_Pa, target_min_thickness_Pa,
                                reference_surface_pressure_Pa)
        target_min_thickness_Pa > 0 || throw(ArgumentError(
            "MergeAbovePressure target_min_thickness_Pa must be > 0 (Inf merges " *
            "every layer above the cutoff); got $(target_min_thickness_Pa)."))
        return new(pressure_Pa, target_min_thickness_Pa, reference_surface_pressure_Pa)
    end
end

"""
    LevelSelection(echlevs)

Typed wrapper for the existing `select_levels_echlevs` algorithm.
`echlevs` is a vector of native level INTERFACE indices (0-based,
numbered from the model top as in TM5); levels between selected interfaces are summed. See
`vertical_coordinates.jl:64` and the `ECHLEVS_ML137_*` constants.
"""
struct LevelSelection <: AbstractVerticalTransform
    echlevs :: Vector{Int}
end

LevelSelection(echlevs::Tuple{Vararg{Integer}}) =
    LevelSelection(collect(Int, echlevs))

# ---------------------------------------------------------------------------
# VerticalPlan{FT, T}: the planned transform. Holds the merged hybrid
# coordinate, the per-transform mapping data, and Nz_output. The `T`
# type parameter is the originating transform type so the
# `apply_vertical!` dispatch picks the right rule.
# ---------------------------------------------------------------------------

"""
    VerticalPlan{FT, T <: AbstractVerticalTransform}

Result of `plan_vertical`. Type-parameterized by the originating
transform so the `apply_vertical!` dispatch is statically resolved.

Fields:
  - `transform`     : the originating `AbstractVerticalTransform` value.
  - `native_vc`     : the input native hybrid coordinate.
  - `merged_vc`     : the output hybrid coordinate.
  - `merge_map`     : `Vector{Int}` such that `merge_map[k_native]` is
                       the output-level index for merge-map flavors.
  - `groups`        : `Vector{UnitRange{Int}}` of native center-level
                       ranges that map to each output level (derived
                       from `merge_map` for the merge-map flavors).
  - `Nz_output`     : the output level count.
  - `Nz_native`     : the input level count (cached for cheap access).
"""
struct VerticalPlan{FT, T <: AbstractVerticalTransform}
    transform :: T
    native_vc :: HybridSigmaPressure{FT}
    merged_vc :: HybridSigmaPressure{FT}
    merge_map :: Vector{Int}
    groups    :: Vector{UnitRange{Int}}
    Nz_output :: Int
    Nz_native :: Int
end

# ---------------------------------------------------------------------------
# plan_vertical — dispatches on the concrete transform.
# ---------------------------------------------------------------------------

"""
    plan_vertical(transform::AbstractVerticalTransform,
                  native_vc::HybridSigmaPressure{FT}) → VerticalPlan{FT, typeof(transform)}

Materialize the planned vertical-coordinate mapping for one day's run.
Called once per day (or once per run if `native_vc` is invariant); the
returned plan is reused across all windows.
"""
function plan_vertical end

# Identity: merge_map = 1:Nz; groups = [k:k for k in 1:Nz]; merged_vc = native_vc.
function plan_vertical(transform::IdentityVertical, native_vc::HybridSigmaPressure{FT}) where FT
    Nz = n_levels(native_vc)
    mm = collect(1:Nz)
    groups = [k:k for k in 1:Nz]
    return VerticalPlan{FT, IdentityVertical}(
        transform, native_vc, native_vc, mm, groups, Nz, Nz)
end

# MergeLayersThinnerThan: delegate to the existing `merge_thin_levels` so
# the math is bit-exact with today's path. Derive `groups` from the
# returned `merge_map`.
function plan_vertical(transform::MergeLayersThinnerThan,
                       native_vc::HybridSigmaPressure{FT}) where FT
    merged_vc, mm = merge_thin_levels(
        native_vc;
        min_thickness_Pa = transform.min_thickness_Pa,
        p_surface = transform.reference_surface_pressure_Pa)
    Nz_out = n_levels(merged_vc)
    groups = _groups_from_merge_map(mm, Nz_out)
    return VerticalPlan{FT, MergeLayersThinnerThan}(
        transform, native_vc, merged_vc, mm, groups, Nz_out, n_levels(native_vc))
end

# MergeAbovePressure: identity below the pressure cutoff; greedy
# merge_thin_levels-style above. The native level orientation is taken
# from the hybrid coordinate's interface ordering — top-of-atmosphere is
# native level 1, surface is native level Nz (i.e. p_half[1] ≈ 0 for
# top-down convention as defined in `config/geos_L72_coefficients.toml`).
function plan_vertical(transform::MergeAbovePressure,
                       native_vc::HybridSigmaPressure{FT}) where FT
    Nz = n_levels(native_vc)
    ps_ref = FT(transform.reference_surface_pressure_Pa)
    cutoff = FT(transform.pressure_Pa)
    target_thr = FT(transform.target_min_thickness_Pa)

    # Per-level midpoint pressure and thickness at the reference surface
    # pressure. midpoint_p[k] is the geometric midpoint between half-
    # levels k and k+1.
    dp = [level_thickness(native_vc, k, ps_ref) for k in 1:Nz]
    midpoint_p = Vector{FT}(undef, Nz)
    for k in 1:Nz
        midpoint_p[k] = (pressure_at_interface(native_vc, k, ps_ref) +
                          pressure_at_interface(native_vc, k + 1, ps_ref)) / 2
    end

    # Which native levels are above the cutoff (lower pressure)? The
    # "above" set is the contiguous top of the column on top-down hybrid
    # grids. We treat any level with midpoint_p < cutoff as eligible for
    # coarsening.
    eligible = falses(Nz)
    for k in 1:Nz
        eligible[k] = midpoint_p[k] < cutoff
    end
    # If no level is eligible, this is identity.
    if !any(eligible)
        return plan_vertical(IdentityVertical(), native_vc)
    end

    # Build groups: above-cutoff native levels are greedily merged into
    # output groups whose accumulated thickness exceeds target_thr.
    # Below-cutoff native levels are 1-to-1 (passthrough).
    groups = UnitRange{Int}[]
    k = 1
    while k <= Nz
        if eligible[k]
            # Find a contiguous run of eligible levels and accumulate.
            run_start = k
            acc = zero(FT)
            run_end = k
            while run_end <= Nz && eligible[run_end] && acc < target_thr
                acc += dp[run_end]
                run_end += 1
            end
            # `run_end` overshot by one; back off if accumulator reached
            # the threshold.
            run_end_actual = run_end - 1
            push!(groups, run_start:run_end_actual)
            k = run_end_actual + 1
        else
            push!(groups, k:k)
            k += 1
        end
    end

    Nz_out = length(groups)
    mm = Vector{Int}(undef, Nz)
    for (l, range) in enumerate(groups)
        for kk in range
            mm[kk] = l
        end
    end
    merged_vc = _merge_vc_from_merge_map(native_vc, mm, Nz_out, FT)
    return VerticalPlan{FT, MergeAbovePressure}(
        transform, native_vc, merged_vc, mm, groups, Nz_out, Nz)
end

# LevelSelection: delegate to existing `select_levels_echlevs`.
function plan_vertical(transform::LevelSelection,
                       native_vc::HybridSigmaPressure{FT}) where FT
    selected_vc, mm = select_levels_echlevs(native_vc, transform.echlevs)
    Nz_out = n_levels(selected_vc)
    groups = _groups_from_merge_map(mm, Nz_out)
    return VerticalPlan{FT, LevelSelection}(
        transform, native_vc, selected_vc, mm, groups, Nz_out, n_levels(native_vc))
end

# ---------------------------------------------------------------------------
# Helpers used by the merge-map flavors above.
# ---------------------------------------------------------------------------

# Build a `merged_vc` from a `merge_map` by keeping the half-level
# coefficients at each group's boundary. `merge_map[k]` is the output
# index that native level k maps to.
function _merge_vc_from_merge_map(native_vc::HybridSigmaPressure{FT},
                                    mm::Vector{Int}, Nz_out::Int,
                                    ::Type{FT}) where FT
    Nz_native = n_levels(native_vc)
    # The kept interface indices are the boundary half-levels: the top
    # of the first group, and the bottom of each subsequent group.
    keep = Int[1]
    current_group = 1
    for k in 1:Nz_native
        if mm[k] > current_group
            push!(keep, k)
            current_group = mm[k]
        end
    end
    push!(keep, Nz_native + 1)
    @assert length(keep) == Nz_out + 1 "merge_map → keep mismatch: " *
        "expected $(Nz_out + 1) interfaces, got $(length(keep))"
    return HybridSigmaPressure(FT[native_vc.A[k] for k in keep],
                                FT[native_vc.B[k] for k in keep])
end

# Build groups (one UnitRange per output level) from a merge_map.
# Julia-style review round-1: walk `mm` once (O(Nz_native)) and track
# group boundaries, instead of the original two-loop / `findall(==(l))`
# version which was O(Nz_native × Nz_out). plan_vertical is called once
# per day at most, so the hot-path impact is small — but the single-
# pass form is also easier to read and validates contiguity in-line
# rather than as a length cross-check.
function _groups_from_merge_map(mm::Vector{Int}, Nz_out::Int)
    Nz_native = length(mm)
    groups = Vector{UnitRange{Int}}(undef, Nz_out)
    start = 0
    current_l = 0
    @inbounds for k in 1:Nz_native
        l = mm[k]
        1 ≤ l ≤ Nz_out ||
            error("merge_map[$(k)] = $(l) is out of bounds [1, $(Nz_out)].")
        if l != current_l
            # Boundary: close out previous group (if any) and validate
            # that the new group is the next contiguous one.
            if current_l != 0
                groups[current_l] = start:k - 1
            end
            l == current_l + 1 ||
                error("merge_map is non-contiguous at native level $(k): " *
                      "expected output level $(current_l + 1), got $(l).")
            current_l = l
            start = k
        end
    end
    current_l == Nz_out ||
        error("merge_map terminates at output level $(current_l) but " *
              "Nz_out = $(Nz_out); levels $(current_l + 1):$(Nz_out) are empty.")
    groups[current_l] = start:Nz_native
    return groups
end

# ---------------------------------------------------------------------------
# apply_vertical! — dispatches on (plan, FieldKind).
#
# `buf_out` is a `(N1, N2, Nz_output)` array; `buf_in` is
# `(N1, N2, Nz_native)`. For CS panel tuples, the caller iterates over
# panels and calls this per-panel. For ConvectionInterfaceFlux the shape is `(N1, N2, Nz+1)` — see the
# per-FieldKind methods below.
# ---------------------------------------------------------------------------

"""
    apply_vertical!(buf_out, buf_in, plan::VerticalPlan, kind::AbstractFieldKind, args...)

Apply the vertical transform to one window of `buf_in`, writing the
result into `buf_out`. Dispatches on the `(plan.transform, kind)`
combination:

  - Extensive center fields (`MassField`,
    `MassFluxField`, `ConvectionTendencyField`) sum native layers
    within each output group.
  - Interface fields (`ConvectionInterfaceFlux`)
    select the kept half-level interfaces.
  - `IntensiveCenterField` takes an additional positional `weights`
    argument (native mass-per-layer); produces the mass-weighted mean
    within each output group.

`buf_out` and `buf_in` must be 3D arrays with the vertical axis on
dim 3.
"""
function apply_vertical! end

# Extensive center fields: sum native layers within each output group.
# Same rule for MassField, MassFluxField, ConvectionTendencyField.
const _EXTENSIVE_CENTER_FIELDS = Union{MassField,
                                         MassFluxField, ConvectionTendencyField}

# Merge-map transforms share one implementation.
const _MERGE_MAP_TRANSFORM = Union{IdentityVertical,
                                    MergeLayersThinnerThan,
                                    MergeAbovePressure, LevelSelection}

function apply_vertical!(buf_out::AbstractArray{T, 3},
                          buf_in::AbstractArray{S, 3},
                          plan::VerticalPlan{FT, <:_MERGE_MAP_TRANSFORM},
                          ::_EXTENSIVE_CENTER_FIELDS) where {T, S, FT}
    _check_center_shapes(buf_out, buf_in, plan)
    return sum_merged_layers!(buf_out, buf_in, plan.merge_map)
end

# Interface fields: select kept half-level interfaces. The `merge_map`
# is over CENTER levels, so the kept half-level indices are the GROUP
# BOUNDARIES — derived from `groups` here for clarity.
function apply_vertical!(buf_out::AbstractArray{T, 3},
                          buf_in::AbstractArray{S, 3},
                          plan::VerticalPlan{FT, <:_MERGE_MAP_TRANSFORM},
                          ::Union{ConvectionInterfaceFlux,InterfaceMassFluxField}
                          ) where {T, S, FT}
    _check_interface_shapes(buf_out, buf_in, plan)
    # Interface indices kept: TOA (1), then the bottom half-level of each
    # top-down group; the final group's bottom is the surface (Nz_native + 1).
    Nz_out = plan.Nz_output
    # Julia-style review round-1: hoist the per-level
    # `plan.groups[l].stop` lookup out of the inner `(i, j)` loop. The
    # native-interface index for output interface `l+1` is invariant
    # across the horizontal — only its value `buf_in[i, j, …]` varies.
    # We pre-fetch the top-of-group index for each output level once
    # (Nz_out reads instead of Nx*Ny*Nz_out), then iterate `(l, j, i)`.
    @inbounds for j in axes(buf_in, 2), i in axes(buf_in, 1)
        buf_out[i, j, 1] = T(buf_in[i, j, 1])
    end
    @inbounds for l in 1:Nz_out
        top_native_of_l = plan.groups[l].stop
        for j in axes(buf_in, 2), i in axes(buf_in, 1)
            buf_out[i, j, l + 1] = T(buf_in[i, j, top_native_of_l + 1])
        end
    end
    return buf_out
end

# IntensiveCenterField: mass-weighted mean within each output group.
# `weights` is the native mass per layer (or pressure thickness; same
# rule). Must have shape (N1, N2, Nz_native).
function apply_vertical!(buf_out::AbstractArray{T, 3},
                          buf_in::AbstractArray{S, 3},
                          plan::VerticalPlan{FT, <:_MERGE_MAP_TRANSFORM},
                          ::IntensiveCenterField,
                          weights::AbstractArray{W, 3}) where {T, S, FT, W}
    _check_center_shapes(buf_out, buf_in, plan)
    size(weights) == size(buf_in) ||
        error("apply_vertical!(IntensiveCenterField, weights) — weights shape " *
              "$(size(weights)) must match buf_in shape $(size(buf_in)).")
    mm = plan.merge_map
    Nz_out = plan.Nz_output
    fill!(buf_out, zero(T))
    accum_w = zeros(W, size(buf_in, 1), size(buf_in, 2), Nz_out)
    @inbounds for k_native in 1:plan.Nz_native
        l_out = mm[k_native]
        for j in axes(buf_in, 2), i in axes(buf_in, 1)
            buf_out[i, j, l_out] += T(buf_in[i, j, k_native] * weights[i, j, k_native])
            accum_w[i, j, l_out] += weights[i, j, k_native]
        end
    end
    # Julia-style review round-1: use `floatmin(W)` rather than hardcoded
    # magic numbers. Any subnormal `w` is treated as essentially zero and
    # the division is skipped — physically correct, since `w` is a mass-
    # weight accumulator and a layer with zero accumulated mass must
    # contribute zero to the merged intensive value.
    floor_w = floatmin(W)
    @inbounds for l in 1:Nz_out, j in axes(buf_in, 2), i in axes(buf_in, 1)
        w = accum_w[i, j, l]
        if w > floor_w
            buf_out[i, j, l] /= w
        end
    end
    return buf_out
end

# Shape validators.
@inline function _check_center_shapes(buf_out::AbstractArray{<:Any, 3},
                                       buf_in::AbstractArray{<:Any, 3},
                                       plan::VerticalPlan)
    size(buf_in,  3) == plan.Nz_native ||
        error("apply_vertical! — buf_in vertical extent $(size(buf_in, 3)) " *
              "does not match plan.Nz_native = $(plan.Nz_native).")
    size(buf_out, 3) == plan.Nz_output ||
        error("apply_vertical! — buf_out vertical extent $(size(buf_out, 3)) " *
              "does not match plan.Nz_output = $(plan.Nz_output).")
    size(buf_in)[1:2] == size(buf_out)[1:2] ||
        error("apply_vertical! — horizontal shapes differ: " *
              "in $(size(buf_in)[1:2]) vs out $(size(buf_out)[1:2]).")
    return nothing
end

@inline function _check_interface_shapes(buf_out::AbstractArray{<:Any, 3},
                                          buf_in::AbstractArray{<:Any, 3},
                                          plan::VerticalPlan)
    size(buf_in,  3) == plan.Nz_native + 1 ||
        error("apply_vertical!(interface) — buf_in vertical extent " *
              "$(size(buf_in, 3)) does not match plan.Nz_native + 1 = " *
              "$(plan.Nz_native + 1).")
    size(buf_out, 3) == plan.Nz_output + 1 ||
        error("apply_vertical!(interface) — buf_out vertical extent " *
              "$(size(buf_out, 3)) does not match plan.Nz_output + 1 = " *
              "$(plan.Nz_output + 1).")
    size(buf_in)[1:2] == size(buf_out)[1:2] ||
        error("apply_vertical!(interface) — horizontal shapes differ: " *
              "in $(size(buf_in)[1:2]) vs out $(size(buf_out)[1:2]).")
    return nothing
end

"""Sum native layers in ascending order using a native-to-output layer map.

The vertical axis is last. Each input element is converted to the output
eltype before addition, matching `apply_vertical!` for mixed precision.
"""
function sum_merged_layers!(out::AbstractArray{T, N}, native::AbstractArray{S, N},
                            mm::AbstractVector{Int}) where {T, S, N}
    fill!(out, zero(T))
    out_layers = reshape(out, :, size(out, N))
    native_layers = reshape(native, :, size(native, N))
    @inbounds for k in eachindex(mm)
        l = mm[k]
        for i in axes(native_layers, 1)
            out_layers[i, l] += T(native_layers[i, k])
        end
    end
    return out
end
