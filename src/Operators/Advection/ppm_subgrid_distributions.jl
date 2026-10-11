# ---------------------------------------------------------------------------
# PPM Subgrid Distribution Helpers
#
# Inline functions for computing parabolic edge values at cell faces,
# dispatched on ORD via compile-time Val{ORD} parameter.
#
# Each variant reads a 5-point stencil (q_{i-2}, q_{i-1}, q_i, q_{i+1}, q_{i+2})
# and returns left and right face values (q_L, q_R) for the parabolic profile.
#
# Reference: Putman & Lin (2007), "Finite-volume transport on various cubed-sphere
# grids"; Colella & Woodward (1984), "The piecewise parabolic method".
#
# ORD=5/7 use the 4th-order cell-edge interpolation (7/12, -1/12); ORD=6 uses the
# unlimited 5th-order upwind-biased stencil. (The earlier port from legacy
# src_legacy/Advection/ppm_subgrid_distributions.jl carried two reconstruction
# bugs — an ORD=5 Huynh call that collapsed to 2-point averaging, and ORD=6
# weights that summed to 58/60 — both corrected here.)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Safe mixing ratio (rm / m with zero guard)
# ---------------------------------------------------------------------------

"""
    _linrood_mass_floor(FT) -> FT

Air mass at or below which the Lin-Rood path treats a cell as empty, `100 eps(FT)`:
its mixing ratio is zero (`_safe_mixing_ratio`, the q-space update and
pre-advection kernels in `LinRood.jl`) and it donates no Courant fraction
(`_courant_fraction`). The adjoint kernels (`linrood_adjoint_*.jl`) take their
zero-gradient threshold from here too, so they branch exactly where the forward
does.
"""
@inline _linrood_mass_floor(::Type{FT}) where {FT} = FT(100) * eps(FT)

"""Extract mixing ratio, returning zero if the air mass is at or below `_linrood_mass_floor`."""
@inline function _safe_mixing_ratio(rm::FT, m::FT) where FT
    return m > _linrood_mass_floor(FT) ? rm / m : zero(FT)
end

# ---------------------------------------------------------------------------
# Minmod flux limiter (generic 3-argument version)
# ---------------------------------------------------------------------------

"""
    minmod_ppm(a, b, c)

Minmod limiter for PPM: returns the value with smallest magnitude if all have
the same sign, otherwise zero. Used in ORD=4 and ORD=5.
"""
@inline function minmod_ppm(a::FT, b::FT, c::FT) where FT
    return _minmod3(a, b, c)
end

# ---------------------------------------------------------------------------
# ORD=4: Optimized PPM (LR96 PPM + minmod limiter)
#
# Putman & Lin Sec. 4, "ORD=4 scheme"
# Collela & Woodward (1984), "The piecewise parabolic method"
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord4(q_imm, q_im, q_i, q_ip, q_ipp)

Compute PPM edge values (q_L, q_R) using ORD=4 (LR96 PPM + minmod).
Uses a 5-point stencil with minmod slope limiter.
"""
@inline function _ppm_edge_values_ord4(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    dq_imm = q_im - q_imm
    dq_im = q_i - q_im
    dq_i = q_ip - q_i
    dq_ip = q_ipp - q_ip

    s_im = minmod_ppm(dq_im, dq_imm, dq_i)
    s_i = minmod_ppm(dq_i, dq_ip, dq_im)

    q_L = q_i - s_im / 2
    q_R = q_i + s_i / 2

    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=5: PPM with Huynh's second constraint
#
# Putman & Lin Sec. 4, "ORD=5 scheme"
# Huynh (1996), "Schemes and constraints for advection"
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)

Compute PPM cell-edge values (q_L, q_R) for ORD=5 via the 4th-order interpolation

    q_{i-1/2} = (7/12)(q_im + q_i) - (1/12)(q_imm + q_ip)   (= q_L)
    q_{i+1/2} = (7/12)(q_i + q_ip) - (1/12)(q_im + q_ipp)   (= q_R)

(Colella & Woodward 1984; FV3 xppm/yppm `ord>=5` base; Putman & Lin 2007 Sec. 4).
Quasi-monotone (Huynh-class) limiting is applied separately by
`_apply_monotonicity` in the face kernels, so this returns the *unlimited*
4th-order edge values. Exact for constant and linear fields; 4th-order accurate
for smooth data.
"""
@inline function _ppm_edge_values_ord5(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    p1, one_twelfth = _ppm_edge_weights(FT)    # (7/12, 1/12)
    p2 = -one_twelfth
    q_L = p1 * (q_im + q_i) + p2 * (q_imm + q_ip)
    q_R = p1 * (q_i + q_ip) + p2 * (q_im + q_ipp)
    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=6: Quasi-5th order (non-monotonic, best pointwise errors)
#
# Putman & Lin Appendix B, "ORD=6 scheme"
# Suresh & Huynh (1997)
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord6(q_imm, q_im, q_i, q_ip, q_ipp)

Compute edge values (q_L, q_R) using ORD=6 (unlimited 5th-order upwind-biased,
non-monotone). Best pointwise (L∞) error at the cost of small over/undershoots.

    q_{i-1/2} = (-3 q_imm + 27 q_im + 47 q_i - 13 q_ip +  2 q_ipp) / 60   (= q_L)
    q_{i+1/2} = ( 2 q_imm - 13 q_im + 47 q_i + 27 q_ip -  3 q_ipp) / 60   (= q_R)

(Suresh & Huynh 1997; the q_R weights are the standard 5th-order upwind stencil,
q_L its mirror.) Both stencils sum to 60/60 = 1, so constant fields are preserved
exactly — the previous coefficients summed to 58/60 and did not.
"""
@inline function _ppm_edge_values_ord6(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    inv60 = one(FT) / FT(60)
    q_L = inv60 * (-FT(3) * q_imm + FT(27) * q_im + FT(47) * q_i - FT(13) * q_ip + FT(2) * q_ipp)
    q_R = inv60 * ( FT(2) * q_imm - FT(13) * q_im + FT(47) * q_i + FT(27) * q_ip - FT(3) * q_ipp)
    return (q_L, q_R)
end

# ---------------------------------------------------------------------------
# ORD=7: Special gnomonic face discontinuity treatment
#
# Putman & Lin Appendix C, Eq. 47
# Averages two one-sided second-order extrapolations at CS face boundaries.
# ---------------------------------------------------------------------------

"""
    _ppm_face_edge_value_ord7_discontinuous(q_left_0, q_left_1, q_right_0, q_right_1)

Compute edge value at a gnomonic CS face discontinuity. Averages two one-sided
second-order extrapolations (Putman & Lin Appendix C, Eq. 47).
"""
@inline function _ppm_face_edge_value_ord7_discontinuous(
    q_left_0::FT, q_left_1::FT,
    q_right_0::FT, q_right_1::FT
) where FT
    extrap_left = FT(3/2) * q_left_0 - q_left_1 / FT(2)
    extrap_right = FT(3/2) * q_right_0 - q_right_1 / FT(2)
    return (extrap_left + extrap_right) / 2
end

# ---------------------------------------------------------------------------
# Dispatcher for all ORD variants
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values(q_imm, q_im, q_i, q_ip, q_ipp, ::Val{ORD})

Dispatch to the appropriate PPM subgrid distribution for the given ORD.
Returns (q_L, q_R) edge values for a parabolic flux calculation.
"""
@inline function _ppm_edge_values(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT, ::Val{4}) where FT
    return _ppm_edge_values_ord4(q_imm, q_im, q_i, q_ip, q_ipp)
end

@inline function _ppm_edge_values(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT, ::Val{5}) where FT
    return _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)
end

@inline function _ppm_edge_values(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT, ::Val{6}) where FT
    return _ppm_edge_values_ord6(q_imm, q_im, q_i, q_ip, q_ipp)
end

@inline function _ppm_edge_values(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT, ::Val{7}) where FT
    # ORD=7 uses the same 4th-order interior reconstruction as ORD=5; the special
    # gnomonic CS face-discontinuity treatment is applied at the kernel level via
    # _apply_ord7_boundary().
    return _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)
end
