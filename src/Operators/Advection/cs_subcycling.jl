# Cubed-sphere CFL subcycle count: the static palindrome outflow budget.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# CFL-based subcycle count
# =========================================================================

"""Static palindrome CFL subcycle count from initial mass.

A cell loses mass through a face when the flux points out of it: with `F_lo`,
`F_hi` its lower- and higher-index faces (positive = toward higher index),

    outgoing = max(0, −F_lo) + max(0, F_hi)

per direction, so both faces count at a divergent point (Lin & Rood 1996).

This is the runtime-side second line of defense for the CS Strang sequence.
The actual sequence applies each direction twice (`X-Y-Z-Z-Y-X`), so a
per-direction CFL pilot can under-estimate cells where moderate outgoing flux
exists in several directions at once. This budget is still a static proxy, not
an evolving-mass proof, but it is conservative with respect to the old
direction-isolated metric and uses the same palindrome-outflow numerator as the
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
                                              max_n_sub::Int = 4096)
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
        zero_FT = zero(FT)
        cfl_panel = mapreduce(max, m_int, ax_lo, ax_hi, by_lo, by_hi,
                              cz_lo, cz_hi; init = zero_FT) do mi, axl, axh, byl, byh, czl, czh
            out_x = max(zero_FT, -(fs * axl)) + max(zero_FT, fs * axh)
            out_y = max(zero_FT, -(fs * byl)) + max(zero_FT, fs * byh)
            out_z = max(zero_FT, -(fs * czl)) + max(zero_FT, fs * czh)
            outgoing_half = out_x + out_y + out_z
            outgoing = outgoing_half + outgoing_half
            ifelse(mi > zero_FT, outgoing / mi, zero_FT)
        end
        max_cfl = max(max_cfl, cfl_panel)
    end
    max_cfl <= cfl_limit && return 1
    n_sub = ceil(Int, max_cfl / cfl_limit)
    n_sub <= max_n_sub ||
        error("cubed-sphere palindrome subcycling exceeded max_n_sub=$(max_n_sub)")
    return n_sub
end


# NOTE: Evolving-mass pilot functions (_cs_x/y/z_pilot_subcycle_count) were
# removed — they were dead code (`strang_split_cs!` uses the shared static
# palindrome budget above).
# The static pilot is sufficient because the gamma-clamped sweep handles CFL > 1
# safely. If evolving-mass pilots are needed in the future, see git history.
