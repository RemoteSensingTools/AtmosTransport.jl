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
# bugs — an ORD=5 limiter call that collapsed to 2-point averaging, and ORD=6
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
the same sign, otherwise zero. Used in ORD=4.
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
# ORD=5: Colella–Woodward fourth-order PPM edge interpolation
#
# Colella & Woodward (1984); the base interpolation of FV3 xppm/yppm for ord>=5
# ---------------------------------------------------------------------------

"""
    _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)

Compute PPM cell-edge values (q_L, q_R) for ORD=5 via the 4th-order interpolation

    q_{i-1/2} = (7/12)(q_im + q_i) - (1/12)(q_imm + q_ip)   (= q_L)
    q_{i+1/2} = (7/12)(q_i + q_ip) - (1/12)(q_im + q_ipp)   (= q_R)

(Colella & Woodward 1984; FV3 xppm/yppm `ord>=5` base). The face kernels later
apply only Colella–Woodward extremum flattening through `_apply_monotonicity`;
they do not apply Huynh's second constraint. This function therefore returns
the *unlimited* fourth-order edge values. It is exact for constant and linear
fields and fourth-order accurate for smooth data.
"""
@inline function _ppm_edge_values_ord5(q_imm::FT, q_im::FT, q_i::FT, q_ip::FT, q_ipp::FT) where FT
    q_L = _ppm_edge_value(q_imm, q_im, q_i, q_ip, FT)
    q_R = _ppm_edge_value(q_im, q_i, q_ip, q_ipp, FT)
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
# ORD=7: cubed-sphere edge treatment
#
# Putman & Lin (2007) Appendix C, Eq. 47 at the panel edge; FV3 `tp_core.F90`
# (`xppm`/`yppm`, `iord < 8` branch) for the one-sided edges beside it.
# ---------------------------------------------------------------------------

"""
    _ppm_face_edge_value_ord7_discontinuous(q_left_0, q_left_1, q_right_0, q_right_1)
    _ppm_face_edge_value_ord7_discontinuous(FT, q_left_0, q_left_1, q_right_0, q_right_1)

Compute edge value at a gnomonic CS face discontinuity. Averages two one-sided
second-order extrapolations (Putman & Lin Appendix C, Eq. 47):

    q_edge = ½ [(3/2 q_left_0 − ½ q_left_1) + (3/2 q_right_0 − ½ q_right_1)]

with `q_left_0`, `q_left_1` the first and second cell on one side of the edge
and `q_right_0`, `q_right_1` on the other. FV3 weights the two extrapolations
by the cell widths `dxa`; on the equiangular grid they are equal at a panel
edge. The `FT` method also accepts the Lin-Rood adjoint's dual numbers.
"""
@inline _ppm_face_edge_value_ord7_discontinuous(q_left_0::FT, q_left_1::FT,
                                                q_right_0::FT, q_right_1::FT) where FT =
    _ppm_face_edge_value_ord7_discontinuous(FT, q_left_0, q_left_1, q_right_0, q_right_1)

@inline function _ppm_face_edge_value_ord7_discontinuous(::Type{FT}, q_left_0, q_left_1,
                                                         q_right_0, q_right_1) where FT
    extrap_left = FT(3/2) * q_left_0 - q_left_1 / FT(2)
    extrap_right = FT(3/2) * q_right_0 - q_right_1 / FT(2)
    return (extrap_left + extrap_right) / FT(2)
end

"""
    _fv3_one_sided_edge_weights(FT) -> (c1, c2, c3) = (-2/14, 11/14, 5/14)

Weights of FV3's one-sided edge value beside a cube edge (`tp_core.F90`,
parameters `c1`, `c2`, `c3`): the edge between cells 1 and 2 is
`c3 q_1 + c2 q_2 + c1 q_3`, built from cells 1-3 on the panel's own side only.
Exact for linear profiles.
"""
@inline _fv3_one_sided_edge_weights(::Type{FT}) where {FT} =
    (FT(-2) / FT(14), FT(11) / FT(14), FT(5) / FT(14))

"""
    _ppm_face_edge_value_ord7(FT, q_mm, q_m, q_p, q_pp, face, Nc)

Unlimited ORD=7 edge value of panel-local face `face` (face `f` lies between
cells `f-1` and `f`; the panel seams are faces `1` and `Nc + 1`) from the cells
`f-2, f-1, f, f+1` = `(q_mm, q_m, q_p, q_pp)`. Every face has exactly one value,
used by both cells beside it, as FV3's `al(i)` (`tp_core.F90`, `xppm`/`yppm`,
`iord < 8` on the cube):

| face            | value                                                            | FV3            |
|:----------------|:-----------------------------------------------------------------|:---------------|
| `1`, `Nc + 1`   | seam: `_ppm_face_edge_value_ord7_discontinuous(q_m, q_mm, q_p, q_pp)` | `al(1)`, `al(npx)` |
| `2`, `Nc + 2`   | one-sided from the cells after the seam: `c3 q_m + c2 q_p + c1 q_pp` | `al(2)`, `al(npx+1)` |
| `0`, `Nc`       | one-sided from the cells before the seam: `c1 q_mm + c2 q_m + c3 q_p` | `al(0)`, `al(npx-1)` |
| otherwise       | fourth-order `7/12 (q_m + q_p) - 1/12 (q_mm + q_pp)`              | `al(i)`        |

so no edge of the two cells beside a seam is built from a stencil that
crosses it (`c1, c2, c3` from `_fv3_one_sided_edge_weights`). Faces `0` and
`Nc + 2` are the outer edges of the halo cells that donate across a seam.
Requires `Nc ≥ 3`, which every Lin-Rood run satisfies: its halo width is 3
(`required_halo_width(::LinRoodPPMScheme)`) and the halo exchange rejects
`Hp > Nc`. Generic in the cell-value type so the adjoint can pass its dual
numbers.
"""
@inline function _ppm_face_edge_value_ord7(::Type{FT}, q_mm, q_m, q_p, q_pp,
                                           face::Integer, Nc::Integer) where FT
    if face == 1 || face == Nc + 1
        return _ppm_face_edge_value_ord7_discontinuous(FT, q_m, q_mm, q_p, q_pp)
    end
    c1, c2, c3 = _fv3_one_sided_edge_weights(FT)
    if face == 2 || face == Nc + 2
        return c3 * q_m + c2 * q_p + c1 * q_pp
    elseif face == 0 || face == Nc
        return c1 * q_mm + c2 * q_m + c3 * q_p
    end
    return _ppm_edge_value(q_mm, q_m, q_p, q_pp, FT)
end

"""
    _ppm_edge_values_ord7(FT, c_m3, c_m2, c_m1, c_0, c_p1, c_p2, face, Nc)
        -> (q_L_m, q_R_m, q_L_0, q_R_0)

Unlimited ORD=7 edges of the two cells beside face `face`: the left cell
`face-1` spans faces `face-1` and `face`, the right cell `face` spans `face` and
`face+1` (`_ppm_face_edge_value_ord7`). The shared face carries one value,
`q_R_m == q_L_0`. Away from the seams this equals two `_ppm_edge_values_ord5`
calls bit for bit.
"""
@inline function _ppm_edge_values_ord7(::Type{FT}, c_m3, c_m2, c_m1, c_0, c_p1, c_p2,
                                       face::Integer, Nc::Integer) where FT
    q_lo  = _ppm_face_edge_value_ord7(FT, c_m3, c_m2, c_m1, c_0, face - 1, Nc)
    q_mid = _ppm_face_edge_value_ord7(FT, c_m2, c_m1, c_0, c_p1, face, Nc)
    q_hi  = _ppm_face_edge_value_ord7(FT, c_m1, c_0, c_p1, c_p2, face + 1, Nc)
    return (q_lo, q_mid, q_mid, q_hi)
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
    # ORD=7 uses the same 4th-order interior reconstruction as ORD=5; the Lin-Rood
    # face kernels build its cube-edge values per face (`_ppm_edge_values_ord7`).
    return _ppm_edge_values_ord5(q_imm, q_im, q_i, q_ip, q_ipp)
end
