# Palindrome CFL subcycle count: the static outflow budget shared by every
# split-sweep topology, and its cubed-sphere evaluation.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# Palindrome outflow budget (all split-sweep topologies)
# =========================================================================
#
# One transport step applies every direction twice: X Y Z | midpoint | Z Y X
# for structured and cubed-sphere grids, H V | midpoint | V H on face-indexed
# grids. When the stored fluxes are too large for one palindrome, the runtime
# runs the subcycled composition
#
#     (X Y Z)^n  midpoint  (Z Y X)^n        every pass with flux / n
#
# (`strang_split_mt!`, `strang_split_cs!`, `strang_split_cs_mt!`, the
# face-indexed `apply!`). Every topology takes `n` from the same budget: with
# `out_d = max(0, -F_lo) + max(0, F_hi)` the cell's outflow through its two `d`
# faces for one full-flux application,
#
#     2 (out_x + out_y + out_z) / (n m)  <=  cfl_limit
#
# i.e. two consecutive passes (one palindrome's worth of flux) remove at most
# `cfl_limit` (default `DEFAULT_CFL_LIMIT`, 0.95) of the cell's step-start mass.
# The first two passes therefore cannot empty a cell. Later passes see the
# evolved mass, so the budget is a static proxy there, measured rather than
# proven (tests in test/core/test_subcycle_composition.jl).

"""
    _outflow(F_lo, F_hi)

Mass a cell loses through its two faces of one direction, `F_lo` the lower- and
`F_hi` the higher-index face flux (positive toward higher index):
`max(0, -F_lo) + max(0, F_hi)`. Both faces count at a divergent point (Lin &
Rood 1996).
"""
@inline _outflow(F_lo, F_hi) = max(zero(F_lo), -F_lo) + max(zero(F_hi), F_hi)

"""
    _palindrome_outflow_ratio(m, out)

Outflow of one palindrome relative to the cell mass, `2 out / m`, where `out`
is the cell's outflow summed over all directions for one application of each.
Cells without positive mass contribute zero.
"""
@inline _palindrome_outflow_ratio(m, out) = ifelse(m > zero(m), (out + out) / m, zero(m))

"""
    _palindrome_subcycle_count(max_ratio, cfl_limit, max_n_sub, topology) -> Int

Number of subcycled passes `n` of a palindrome whose largest outflow ratio
(`_palindrome_outflow_ratio`) is `max_ratio`: the smallest `n` with
`max_ratio / n <= cfl_limit`. `cfl_limit = Inf` disables subcycling.
"""
function _palindrome_subcycle_count(max_ratio, cfl_limit, max_n_sub::Integer, topology)
    cfl_limit > zero(cfl_limit) || throw(ArgumentError(
        "$(topology) advection requires cfl_limit > 0, got $(cfl_limit)"))
    max_ratio <= cfl_limit && return 1
    n_sub = ceil(Int, max_ratio / cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError(
        "$(topology) palindrome subcycling exceeded max_n_sub=$(max_n_sub) " *
        "(outflow ratio $(max_ratio), cfl_limit $(cfl_limit))"))
    return n_sub
end

# =========================================================================
# Cubed-sphere subcycle count
# =========================================================================

"""Static palindrome CFL subcycle count of a cubed-sphere step from its initial mass.

The shared palindrome budget (`_palindrome_subcycle_count`) evaluated on the
interior cells of the six panels, with every flux scaled by `flux_scale`.

This is the runtime-side second line of defense for the CS Strang sequence;
binaries with the `binary_schedule` substep contract pass their own count
(see `_cs_palindrome_subcycles`). The sequence applies each direction twice,
so a per-direction CFL pilot can under-estimate cells where moderate outgoing
flux exists in several directions at once. The numerator is the one of the
preprocessor's adaptive schedule gate (which can also divide by the smaller of
the two window-end masses).
"""
function _cs_static_palindrome_subcycle_count(panels_am::NTuple{6},
                                              panels_bm::NTuple{6},
                                              panels_cm::NTuple{6},
                                              panels_m::NTuple{6},
                                              Nc::Int, Hp::Int, Nz::Int,
                                              cfl_limit::Real;
                                              flux_scale = one(eltype(panels_m[1])),
                                              max_n_sub::Int = DEFAULT_MAX_SUBCYCLES)
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    iL = Hp + 1
    iH = Hp + Nc
    max_cfl = zero(FT)
    @inbounds for p in 1:6
        m_int = view(panels_m[p], iL:iH, iL:iH, 1:Nz)
        ax_lo = view(panels_am[p], iL    :iH,     iL:iH,     1:Nz)
        ax_hi = view(panels_am[p], iL + 1:iH + 1, iL:iH,     1:Nz)
        by_lo = view(panels_bm[p], iL:iH,     iL    :iH,     1:Nz)
        by_hi = view(panels_bm[p], iL:iH,     iL + 1:iH + 1, 1:Nz)
        cz_lo = view(panels_cm[p], iL:iH, iL:iH, 1    :Nz)
        cz_hi = view(panels_cm[p], iL:iH, iL:iH, 2:Nz + 1)
        cfl_panel = mapreduce(max, m_int, ax_lo, ax_hi, by_lo, by_hi,
                              cz_lo, cz_hi; init = zero(FT)) do mi, axl, axh, byl, byh, czl, czh
            _palindrome_outflow_ratio(mi, _outflow(fs * axl, fs * axh) +
                                          _outflow(fs * byl, fs * byh) +
                                          _outflow(fs * czl, fs * czh))
        end
        max_cfl = max(max_cfl, cfl_panel)
    end
    return _palindrome_subcycle_count(max_cfl, cfl_limit, max_n_sub, "cubed-sphere")
end


# NOTE: Evolving-mass pilot functions (_cs_x/y/z_pilot_subcycle_count) were
# removed — they were dead code (`strang_split_cs!` uses the shared static
# palindrome budget above).
# The static pilot is sufficient because the gamma-clamped sweep handles CFL > 1
# safely. If evolving-mass pilots are needed in the future, see git history.

"""
    _cs_palindrome_subcycles(check, caller, subcycle_count, panels_am, panels_bm,
                             panels_cm, panels_m, Nc, Hp, Nz, cfl_limit; flux_scale)

Subcycle count of one cubed-sphere palindrome: the runtime CFL budget
(`_cs_static_palindrome_subcycle_count`) when `subcycle_count === nothing`,
else the binary's count, which `check::BinaryCFLCheck` compares with that
budget. `caller` names the operator in error messages.
"""
_cs_palindrome_subcycles(::AbstractBinaryCFLCheck, caller, ::Nothing,
                         am, bm, cm, m, Nc, Hp, Nz, cfl_limit; flux_scale) =
    SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
        am, bm, cm, m, Nc, Hp, Nz, cfl_limit; flux_scale)

function _cs_palindrome_subcycles(check::AbstractBinaryCFLCheck, caller, subcycle_count::Integer,
                                  am, bm, cm, m, Nc, Hp, Nz, cfl_limit; flux_scale)
    n = Int(subcycle_count)
    n >= 1 || throw(ArgumentError("$(caller): subcycle_count must be ≥ 1, got $(subcycle_count)"))
    _check_binary_subcycles(check, caller, n, am, bm, cm, m, Nc, Hp, Nz, cfl_limit; flux_scale)
    return n
end

_check_binary_subcycles(::NoBinaryCFLCheck, caller, n, am, bm, cm, m, Nc, Hp, Nz, cfl_limit;
                        flux_scale) = nothing

function _check_binary_subcycles(::BinaryCFLCheck, caller, n, am, bm, cm, m, Nc, Hp, Nz, cfl_limit;
                                 flux_scale)
    required = SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
        am, bm, cm, m, Nc, Hp, Nz, cfl_limit; flux_scale)
    required <= n || throw(ArgumentError(
        "$(caller): the binary's substep schedule requests subcycle_count=$n, but the " *
        "runtime CFL budget requires $required ([advection] check_binary_cfl = true). " *
        "Regenerate the binary."))
    return nothing
end
