# =========================================================================
# Cubed-sphere subcycle count
# =========================================================================

"""Static palindrome CFL subcycle count of a cubed-sphere step.

The shared palindrome budget (`_palindrome_subcycle_count`, reference mass
`min(m, m_end)`) evaluated on the interior cells of the six panels, with every
flux scaled by `flux_scale`.

This is the runtime-side second line of defense for the CS Strang sequence;
binaries with the `binary_schedule` substep contract pass their own count
(see `_cs_palindrome_subcycles`). The sequence applies each direction twice,
so a per-direction CFL pilot can under-estimate cells where moderate outgoing
flux exists in several directions at once. Numerator and reference mass are
those of the preprocessor's positivity gate (`verify_substep_positivity_cs!`),
which takes the smaller of the window's two end masses.
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
                                          _outflow(fs * czl, fs * czh),
                                      _net_inflow(fs * axl, fs * axh) +
                                          _net_inflow(fs * byl, fs * byh) +
                                          _net_inflow(fs * czl, fs * czh))
        end
        max_cfl = max(max_cfl, cfl_panel)
    end
    return _palindrome_subcycle_count(max_cfl, cfl_limit, max_n_sub, "cubed-sphere")
end


# NOTE: Evolving-mass pilot functions (_cs_x/y/z_pilot_subcycle_count) were
# removed — they were dead code (`strang_split_cs_mt!` uses the shared static
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
