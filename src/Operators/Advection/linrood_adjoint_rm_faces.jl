# Lin-Rood adjoint kernels of the rm-input PPM face reconstruction (X and Y, ORD 5 and 7).
# Split from linrood_adjoint_kernels.jl (refactor phase 4); included by Advection.jl in this order.

# ===========================================================================
# Adjoints of the rm-input PPM face kernels (ORD=5)
#
# Forward kernels `_ppm_x_face_kernel!` and `_ppm_y_face_kernel!`
# (LinRood.jl) at ORD=5 fold `_safe_mixing_ratio` into the
# face computation: `c_n = rm_n / m_n` (zero below the
# `100·eps(FT)` threshold) feeds the same downstream
# `_ppm_edge_values_ord5 → _apply_monotonicity → _ppm_face_value`
# chain as the `_from_q` variants. The adjoint therefore needs to
# distribute the face seed into BOTH `lambda_rm` and `lambda_m` at
# the six stencil cells, with the donor-cell `m_donor` additionally
# feeding `_ppm_face_value` directly via `α = F / m_donor`.
#
# Strategy: run the d6 chain TWICE, once with the rm tangent
# `dc_n = 1/m_n · e_n` and once with the m tangent
# `dc_n = -rm_n / m_n² · e_n`. The first run returns `∂f/∂rm_n`; the
# second returns the chain-rule part of `∂f/∂m_n` (i.e., the
# `c = rm/m` coupling). The donor-cell m_donor additionally
# contributes `∂α/∂m_donor = -F / m_donor²` (for a donor above the mass
# floor and |F| < m_donor; zero where α is clamped to ±1), which we add
# analytically via `_courant_fraction`.
# ===========================================================================

# d6-AD safe-mixing-ratio: returns the D6{FT} value `rm_n / m_n` with the
# requested tangent. Mirrors the forward `_safe_mixing_ratio`
# 100·eps threshold by returning a zero-gradient D6 when m_n is too
# small.
@inline function _safe_mixing_ratio_d6(rm_n::FT, m_n::FT,
                                        tangent::NTuple{6, FT}) where {FT}
    if m_n > FT(100) * eps(FT)
        return _d6_pack(rm_n / m_n, tangent)
    else
        return _d6_const(FT, zero(FT))
    end
end

# Pre-compute the rm-input chain once at a state; return the 6-tuple
# face partials w.r.t. one cell-attribute (either rm or m), driven by
# the input `tangents` (one length-6 tuple per stencil cell).
@inline function _linrood_ppm_face_chain_rm_ord5(
    F::FT, m_l::FT, m_r::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
    tan_m3::NTuple{6, FT}, tan_m2::NTuple{6, FT}, tan_m1::NTuple{6, FT},
    tan_0::NTuple{6, FT},  tan_p1::NTuple{6, FT}, tan_p2::NTuple{6, FT},
) where {FT}
    c_m3 = _safe_mixing_ratio_d6(rm_m3, m_m3, tan_m3)
    c_m2 = _safe_mixing_ratio_d6(rm_m2, m_m2, tan_m2)
    c_m1 = _safe_mixing_ratio_d6(rm_m1, m_m1, tan_m1)
    c_0  = _safe_mixing_ratio_d6(rm_0,  m_0,  tan_0)
    c_p1 = _safe_mixing_ratio_d6(rm_p1, m_p1, tan_p1)
    c_p2 = _safe_mixing_ratio_d6(rm_p2, m_p2, tan_p2)

    q_L_m, q_R_m = _ppm_edge_values_ord5_d6(c_m3, c_m2, c_m1, c_0, c_p1)
    q_L_0, q_R_0 = _ppm_edge_values_ord5_d6(c_m2, c_m1, c_0, c_p1, c_p2)
    q_L_m, q_R_m = _apply_monotonicity_d6(q_L_m, q_R_m, c_m1)
    q_L_0, q_R_0 = _apply_monotonicity_d6(q_L_0, q_R_0, c_0)
    face = _ppm_face_value(F, m_l, m_r, c_m1, c_0,
                              q_L_m, q_R_m, q_L_0, q_R_0)
    return face.g
end

# Full rm-input face Jacobian: returns
# `(∂f/∂rm_-3..p2, ∂f/∂m_-3..p2)` for the six stencil cells.
@inline function _linrood_ppm_face_from_rm_grad_ord5(
    F::FT, m_l::FT, m_r::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
) where {FT}
    floor_thresh = FT(100) * eps(FT)
    # rm pass: dc_n = (1/m_n) · e_n   (zero if m_n below threshold)
    inv_m_m3 = m_m3 > floor_thresh ? one(FT) / m_m3 : zero(FT)
    inv_m_m2 = m_m2 > floor_thresh ? one(FT) / m_m2 : zero(FT)
    inv_m_m1 = m_m1 > floor_thresh ? one(FT) / m_m1 : zero(FT)
    inv_m_0  = m_0  > floor_thresh ? one(FT) / m_0  : zero(FT)
    inv_m_p1 = m_p1 > floor_thresh ? one(FT) / m_p1 : zero(FT)
    inv_m_p2 = m_p2 > floor_thresh ? one(FT) / m_p2 : zero(FT)

    tan_rm_m3 = ntuple(i -> i == 1 ? inv_m_m3 : zero(FT), Val(6))
    tan_rm_m2 = ntuple(i -> i == 2 ? inv_m_m2 : zero(FT), Val(6))
    tan_rm_m1 = ntuple(i -> i == 3 ? inv_m_m1 : zero(FT), Val(6))
    tan_rm_0  = ntuple(i -> i == 4 ? inv_m_0  : zero(FT), Val(6))
    tan_rm_p1 = ntuple(i -> i == 5 ? inv_m_p1 : zero(FT), Val(6))
    tan_rm_p2 = ntuple(i -> i == 6 ? inv_m_p2 : zero(FT), Val(6))

    grad_rm = _linrood_ppm_face_chain_rm_ord5(
        F, m_l, m_r,
        rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
        m_m3, m_m2, m_m1, m_0, m_p1, m_p2,
        tan_rm_m3, tan_rm_m2, tan_rm_m1, tan_rm_0, tan_rm_p1, tan_rm_p2)

    # m pass: dc_n = (-rm_n / m_n²) · e_n
    neg_rm_over_m2_m3 = m_m3 > floor_thresh ? -rm_m3 / (m_m3 * m_m3) : zero(FT)
    neg_rm_over_m2_m2 = m_m2 > floor_thresh ? -rm_m2 / (m_m2 * m_m2) : zero(FT)
    neg_rm_over_m2_m1 = m_m1 > floor_thresh ? -rm_m1 / (m_m1 * m_m1) : zero(FT)
    neg_rm_over_m2_0  = m_0  > floor_thresh ? -rm_0  / (m_0  * m_0)  : zero(FT)
    neg_rm_over_m2_p1 = m_p1 > floor_thresh ? -rm_p1 / (m_p1 * m_p1) : zero(FT)
    neg_rm_over_m2_p2 = m_p2 > floor_thresh ? -rm_p2 / (m_p2 * m_p2) : zero(FT)

    tan_m_m3 = ntuple(i -> i == 1 ? neg_rm_over_m2_m3 : zero(FT), Val(6))
    tan_m_m2 = ntuple(i -> i == 2 ? neg_rm_over_m2_m2 : zero(FT), Val(6))
    tan_m_m1 = ntuple(i -> i == 3 ? neg_rm_over_m2_m1 : zero(FT), Val(6))
    tan_m_0  = ntuple(i -> i == 4 ? neg_rm_over_m2_0  : zero(FT), Val(6))
    tan_m_p1 = ntuple(i -> i == 5 ? neg_rm_over_m2_p1 : zero(FT), Val(6))
    tan_m_p2 = ntuple(i -> i == 6 ? neg_rm_over_m2_p2 : zero(FT), Val(6))

    grad_m_chain = _linrood_ppm_face_chain_rm_ord5(
        F, m_l, m_r,
        rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
        m_m3, m_m2, m_m1, m_0, m_p1, m_p2,
        tan_m_m3, tan_m_m2, tan_m_m1, tan_m_0, tan_m_p1, tan_m_p2)

    # Donor-mass alpha contribution. Forward `_ppm_face_value` uses
    #   α = clamp(F / m_donor, -1, 1)   (above the mass floor; else 0)
    # where m_donor = m_l when F ≥ 0 (donor is the cell to the "lo"
    # side of the face, i.e. stencil position 3 = c_m1) and m_donor =
    # m_r when F < 0 (donor = stencil position 4 = c_0). The chain
    # rule contribution
    #   ∂face/∂m_donor |_{α-only} = (∂face/∂α) · (∂α/∂m_donor)
    # adds to the corresponding cell's `∂f/∂m`. We compute
    # ∂face/∂α analytically by differentiating the parabolic-integral
    # form once more.
    # Recompute the limited (q_L, q_R) at the donor cell using the
    # forward chain on plain FT values so we can build the α
    # contribution exactly.
    extra_m_m1 = zero(FT)
    extra_m_0  = zero(FT)
    if F >= zero(FT)
        if m_m1 > floor_thresh
            bl_lo, br_lo, b0_lo, _ = _ppm_face_value_donor_state_lo(
                F, rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
                m_m3, m_m2, m_m1, m_0, m_p1, m_p2)
            _ = bl_lo  # forward chain reads bl, br, b0; ∂face/∂α only uses br and b0
            alpha, dalpha_dm = _courant_fraction(F, m_m1)
            # face = c_lo + (1 - α)(br_lo - α·b0_lo)
            #       ∂face/∂α = -(br_lo - α·b0_lo) + (1 - α)·(-b0_lo)
            #                = -br_lo + (2α - 1)·b0_lo
            dface_dalpha = -br_lo + (FT(2) * alpha - one(FT)) * b0_lo
            extra_m_m1 = dface_dalpha * dalpha_dm
        end
    else
        if m_0 > floor_thresh
            bl_hi, _, b0_hi, _ = _ppm_face_value_donor_state_hi(
                F, rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
                m_m3, m_m2, m_m1, m_0, m_p1, m_p2)
            alpha, dalpha_dm = _courant_fraction(F, m_0)
            # face = c_hi + (1 + α)(bl_hi + α·b0_hi)
            #       ∂face/∂α = (bl_hi + α·b0_hi) + (1 + α)·b0_hi
            #                = bl_hi + b0_hi + 2·α·b0_hi
            dface_dalpha = bl_hi + b0_hi + FT(2) * alpha * b0_hi
            extra_m_0 = dface_dalpha * dalpha_dm
        end
    end

    grad_m = (grad_m_chain[1], grad_m_chain[2],
              grad_m_chain[3] + extra_m_m1,
              grad_m_chain[4] + extra_m_0,
              grad_m_chain[5], grad_m_chain[6])

    return (grad_rm, grad_m)
end

# Helper: compute the forward parabolic-integral coefficients
# `(bl, br, b0, c)` at the donor cell for the F >= 0 branch.
# Mirrors `_ppm_face_value` exactly (no d6 propagation).
@inline function _ppm_face_value_donor_state_lo(
    F::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
) where {FT}
    _ = F
    _ = (rm_p2, m_p2)  # not part of the q_L_m/q_R_m stencil
    floor_thresh = FT(100) * eps(FT)
    c_m3v = m_m3 > floor_thresh ? rm_m3 / m_m3 : zero(FT)
    c_m2v = m_m2 > floor_thresh ? rm_m2 / m_m2 : zero(FT)
    c_m1v = m_m1 > floor_thresh ? rm_m1 / m_m1 : zero(FT)
    c_0v  = m_0  > floor_thresh ? rm_0  / m_0  : zero(FT)
    c_p1v = m_p1 > floor_thresh ? rm_p1 / m_p1 : zero(FT)
    q_L_m_v, q_R_m_v = _ppm_edge_values(c_m3v, c_m2v, c_m1v, c_0v, c_p1v, Val(5))
    q_L_m_v, q_R_m_v = _apply_monotonicity(q_L_m_v, q_R_m_v, c_m1v)
    bl = q_L_m_v - c_m1v
    br = q_R_m_v - c_m1v
    b0 = bl + br
    return (bl, br, b0, c_m1v)
end

@inline function _ppm_face_value_donor_state_hi(
    F::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
) where {FT}
    _ = F
    _ = (rm_m3, m_m3)  # not part of the q_L_0/q_R_0 stencil
    floor_thresh = FT(100) * eps(FT)
    c_m2v = m_m2 > floor_thresh ? rm_m2 / m_m2 : zero(FT)
    c_m1v = m_m1 > floor_thresh ? rm_m1 / m_m1 : zero(FT)
    c_0v  = m_0  > floor_thresh ? rm_0  / m_0  : zero(FT)
    c_p1v = m_p1 > floor_thresh ? rm_p1 / m_p1 : zero(FT)
    c_p2v = m_p2 > floor_thresh ? rm_p2 / m_p2 : zero(FT)
    q_L_0_v, q_R_0_v = _ppm_edge_values(c_m2v, c_m1v, c_0v, c_p1v, c_p2v, Val(5))
    q_L_0_v, q_R_0_v = _apply_monotonicity(q_L_0_v, q_R_0_v, c_0v)
    bl = q_L_0_v - c_0v
    br = q_R_0_v - c_0v
    b0 = bl + br
    return (bl, br, b0, c_0v)
end

# ORD=7 donor-state helpers. At panel-edge faces (`face_idx ∈ {1, Nc+1}`)
# the forward `_apply_ord7_boundary` rewrites `q_R_m` and `q_L_0` with
# the same discontinuous value `q_bdy` before monotonicity. The donor
# cell's limited `(q_L, q_R)` therefore differs from ORD=5 at the
# boundary, and we need to recompute `(bl, br, b0)` against the
# corrected edges. At interior faces these helpers return the same
# `(bl, br, b0, c)` as the ORD=5 versions (compile-time eliminated).
@inline function _ppm_face_value_donor_state_lo_ord7(
    F::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
    face_idx::Integer, Nc::Integer,
) where {FT}
    _ = F
    _ = (rm_p2, m_p2)
    floor_thresh = FT(100) * eps(FT)
    c_m3v = m_m3 > floor_thresh ? rm_m3 / m_m3 : zero(FT)
    c_m2v = m_m2 > floor_thresh ? rm_m2 / m_m2 : zero(FT)
    c_m1v = m_m1 > floor_thresh ? rm_m1 / m_m1 : zero(FT)
    c_0v  = m_0  > floor_thresh ? rm_0  / m_0  : zero(FT)
    c_p1v = m_p1 > floor_thresh ? rm_p1 / m_p1 : zero(FT)
    q_L_m_v, q_R_m_v = _ppm_edge_values(c_m3v, c_m2v, c_m1v, c_0v, c_p1v, Val(5))
    # Reuse a small helper for the boundary correction at the value level
    # — mirrors `_apply_ord7_boundary` exactly (LinRood.jl:186).
    if face_idx == 1 || face_idx == Nc + 1
        # q_R_m is rewritten with q_bdy at boundary (q_L_m unchanged).
        # Use the donor-state stencil's own (c_m2, c_m1, c_0, c_p1).
        q_bdy = _ppm_face_edge_value_ord7_discontinuous(c_m1v, c_m2v, c_0v, c_p1v)
        q_R_m_v = q_bdy
    end
    q_L_m_v, q_R_m_v = _apply_monotonicity(q_L_m_v, q_R_m_v, c_m1v)
    bl = q_L_m_v - c_m1v
    br = q_R_m_v - c_m1v
    b0 = bl + br
    return (bl, br, b0, c_m1v)
end

@inline function _ppm_face_value_donor_state_hi_ord7(
    F::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
    face_idx::Integer, Nc::Integer,
) where {FT}
    _ = F
    _ = (rm_m3, m_m3)
    floor_thresh = FT(100) * eps(FT)
    c_m2v = m_m2 > floor_thresh ? rm_m2 / m_m2 : zero(FT)
    c_m1v = m_m1 > floor_thresh ? rm_m1 / m_m1 : zero(FT)
    c_0v  = m_0  > floor_thresh ? rm_0  / m_0  : zero(FT)
    c_p1v = m_p1 > floor_thresh ? rm_p1 / m_p1 : zero(FT)
    c_p2v = m_p2 > floor_thresh ? rm_p2 / m_p2 : zero(FT)
    q_L_0_v, q_R_0_v = _ppm_edge_values(c_m2v, c_m1v, c_0v, c_p1v, c_p2v, Val(5))
    if face_idx == 1 || face_idx == Nc + 1
        # q_L_0 is rewritten with q_bdy at boundary (q_R_0 unchanged).
        q_bdy = _ppm_face_edge_value_ord7_discontinuous(c_m1v, c_m2v, c_0v, c_p1v)
        q_L_0_v = q_bdy
    end
    q_L_0_v, q_R_0_v = _apply_monotonicity(q_L_0_v, q_R_0_v, c_0v)
    bl = q_L_0_v - c_0v
    br = q_R_0_v - c_0v
    b0 = bl + br
    return (bl, br, b0, c_0v)
end

# ORD=7 from-rm chain: same structure as ORD=5 with the d6 boundary
# correction inserted between `_ppm_edge_values_ord5_d6` and
# `_apply_monotonicity_d6`. Returns the 6-tuple `∂face/∂(c-tangent)` for
# one cell-attribute pass (rm OR m), driven by `tangents`.
@inline function _linrood_ppm_face_chain_rm_ord7(
    F::FT, m_l::FT, m_r::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
    tan_m3::NTuple{6, FT}, tan_m2::NTuple{6, FT}, tan_m1::NTuple{6, FT},
    tan_0::NTuple{6, FT},  tan_p1::NTuple{6, FT}, tan_p2::NTuple{6, FT},
    face_idx::Integer, Nc::Integer,
) where {FT}
    c_m3 = _safe_mixing_ratio_d6(rm_m3, m_m3, tan_m3)
    c_m2 = _safe_mixing_ratio_d6(rm_m2, m_m2, tan_m2)
    c_m1 = _safe_mixing_ratio_d6(rm_m1, m_m1, tan_m1)
    c_0  = _safe_mixing_ratio_d6(rm_0,  m_0,  tan_0)
    c_p1 = _safe_mixing_ratio_d6(rm_p1, m_p1, tan_p1)
    c_p2 = _safe_mixing_ratio_d6(rm_p2, m_p2, tan_p2)

    q_L_m, q_R_m = _ppm_edge_values_ord5_d6(c_m3, c_m2, c_m1, c_0, c_p1)
    q_L_0, q_R_0 = _ppm_edge_values_ord5_d6(c_m2, c_m1, c_0, c_p1, c_p2)
    q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary_d6(
        q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, face_idx, Nc)
    q_L_m, q_R_m = _apply_monotonicity_d6(q_L_m, q_R_m, c_m1)
    q_L_0, q_R_0 = _apply_monotonicity_d6(q_L_0, q_R_0, c_0)
    face = _ppm_face_value(F, m_l, m_r, c_m1, c_0,
                              q_L_m, q_R_m, q_L_0, q_R_0)
    return face.g
end

# Full from-rm face Jacobian at ORD=7: returns
# `(∂f/∂rm_-3..p2, ∂f/∂m_-3..p2)` for the six stencil cells. Mirrors
# `_linrood_ppm_face_from_rm_grad_ord5` with the boundary-aware chain
# and donor-state helpers.
@inline function _linrood_ppm_face_from_rm_grad_ord7(
    F::FT, m_l::FT, m_r::FT,
    rm_m3::FT, rm_m2::FT, rm_m1::FT, rm_0::FT, rm_p1::FT, rm_p2::FT,
    m_m3::FT,  m_m2::FT,  m_m1::FT,  m_0::FT,  m_p1::FT,  m_p2::FT,
    face_idx::Integer, Nc::Integer,
) where {FT}
    floor_thresh = FT(100) * eps(FT)
    # rm pass: dc_n = (1/m_n) · e_n
    inv_m_m3 = m_m3 > floor_thresh ? one(FT) / m_m3 : zero(FT)
    inv_m_m2 = m_m2 > floor_thresh ? one(FT) / m_m2 : zero(FT)
    inv_m_m1 = m_m1 > floor_thresh ? one(FT) / m_m1 : zero(FT)
    inv_m_0  = m_0  > floor_thresh ? one(FT) / m_0  : zero(FT)
    inv_m_p1 = m_p1 > floor_thresh ? one(FT) / m_p1 : zero(FT)
    inv_m_p2 = m_p2 > floor_thresh ? one(FT) / m_p2 : zero(FT)

    tan_rm_m3 = ntuple(i -> i == 1 ? inv_m_m3 : zero(FT), Val(6))
    tan_rm_m2 = ntuple(i -> i == 2 ? inv_m_m2 : zero(FT), Val(6))
    tan_rm_m1 = ntuple(i -> i == 3 ? inv_m_m1 : zero(FT), Val(6))
    tan_rm_0  = ntuple(i -> i == 4 ? inv_m_0  : zero(FT), Val(6))
    tan_rm_p1 = ntuple(i -> i == 5 ? inv_m_p1 : zero(FT), Val(6))
    tan_rm_p2 = ntuple(i -> i == 6 ? inv_m_p2 : zero(FT), Val(6))

    grad_rm = _linrood_ppm_face_chain_rm_ord7(
        F, m_l, m_r,
        rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
        m_m3, m_m2, m_m1, m_0, m_p1, m_p2,
        tan_rm_m3, tan_rm_m2, tan_rm_m1, tan_rm_0, tan_rm_p1, tan_rm_p2,
        face_idx, Nc)

    # m pass: dc_n = (-rm_n / m_n²) · e_n
    neg_rm_over_m2_m3 = m_m3 > floor_thresh ? -rm_m3 / (m_m3 * m_m3) : zero(FT)
    neg_rm_over_m2_m2 = m_m2 > floor_thresh ? -rm_m2 / (m_m2 * m_m2) : zero(FT)
    neg_rm_over_m2_m1 = m_m1 > floor_thresh ? -rm_m1 / (m_m1 * m_m1) : zero(FT)
    neg_rm_over_m2_0  = m_0  > floor_thresh ? -rm_0  / (m_0  * m_0)  : zero(FT)
    neg_rm_over_m2_p1 = m_p1 > floor_thresh ? -rm_p1 / (m_p1 * m_p1) : zero(FT)
    neg_rm_over_m2_p2 = m_p2 > floor_thresh ? -rm_p2 / (m_p2 * m_p2) : zero(FT)

    tan_m_m3 = ntuple(i -> i == 1 ? neg_rm_over_m2_m3 : zero(FT), Val(6))
    tan_m_m2 = ntuple(i -> i == 2 ? neg_rm_over_m2_m2 : zero(FT), Val(6))
    tan_m_m1 = ntuple(i -> i == 3 ? neg_rm_over_m2_m1 : zero(FT), Val(6))
    tan_m_0  = ntuple(i -> i == 4 ? neg_rm_over_m2_0  : zero(FT), Val(6))
    tan_m_p1 = ntuple(i -> i == 5 ? neg_rm_over_m2_p1 : zero(FT), Val(6))
    tan_m_p2 = ntuple(i -> i == 6 ? neg_rm_over_m2_p2 : zero(FT), Val(6))

    grad_m_chain = _linrood_ppm_face_chain_rm_ord7(
        F, m_l, m_r,
        rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
        m_m3, m_m2, m_m1, m_0, m_p1, m_p2,
        tan_m_m3, tan_m_m2, tan_m_m1, tan_m_0, tan_m_p1, tan_m_p2,
        face_idx, Nc)

    # Donor-mass α contribution. Use the ORD=7 donor-state helpers so
    # `(bl, br, b0)` reflects the boundary-corrected (q_L, q_R) at
    # panel-edge donor cells.
    extra_m_m1 = zero(FT)
    extra_m_0  = zero(FT)
    if F >= zero(FT)
        if m_m1 > floor_thresh
            bl_lo, br_lo, b0_lo, _ = _ppm_face_value_donor_state_lo_ord7(
                F, rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
                m_m3, m_m2, m_m1, m_0, m_p1, m_p2, face_idx, Nc)
            _ = bl_lo
            alpha, dalpha_dm = _courant_fraction(F, m_m1)
            dface_dalpha = -br_lo + (FT(2) * alpha - one(FT)) * b0_lo
            extra_m_m1 = dface_dalpha * dalpha_dm
        end
    else
        if m_0 > floor_thresh
            bl_hi, _, b0_hi, _ = _ppm_face_value_donor_state_hi_ord7(
                F, rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
                m_m3, m_m2, m_m1, m_0, m_p1, m_p2, face_idx, Nc)
            alpha, dalpha_dm = _courant_fraction(F, m_0)
            dface_dalpha = bl_hi + b0_hi + FT(2) * alpha * b0_hi
            extra_m_0 = dface_dalpha * dalpha_dm
        end
    end

    grad_m = (grad_m_chain[1], grad_m_chain[2],
              grad_m_chain[3] + extra_m_m1,
              grad_m_chain[4] + extra_m_0,
              grad_m_chain[5], grad_m_chain[6])

    return (grad_rm, grad_m)
end

# ---------------------------------------------------------------------------
# rm-input face kernels (X and Y, ORD=5).
# ---------------------------------------------------------------------------

@kernel function _ppm_x_face_kernel_adjoint_ord5!(
    lambda_rm, lambda_m,
    @Const(lambda_fx_face), @Const(rm), @Const(m), @Const(am),
    Hp, Nc,
)
    iif, j, k = @index(Global, NTuple)
    _ = Nc
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        rm_m3 = rm[ii_l - 2, jj, k]; m_m3 = m[ii_l - 2, jj, k]
        rm_m2 = rm[ii_l - 1, jj, k]; m_m2 = m[ii_l - 1, jj, k]
        rm_m1 = rm[ii_l,     jj, k]; m_m1 = m[ii_l,     jj, k]
        rm_0  = rm[ii_r,     jj, k]; m_0  = m[ii_r,     jj, k]
        rm_p1 = rm[ii_r + 1, jj, k]; m_p1 = m[ii_r + 1, jj, k]
        rm_p2 = rm[ii_r + 2, jj, k]; m_p2 = m[ii_r + 2, jj, k]

        F = am[iif, j, k]
        grad_rm, grad_m = _linrood_ppm_face_from_rm_grad_ord5(
            F, m_m1, m_0,
            rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
            m_m3,  m_m2,  m_m1,  m_0,  m_p1,  m_p2,
        )

        bar = lambda_fx_face[iif, j, k]
        @atomic lambda_rm[ii_l - 2, jj, k] += bar * grad_rm[1]
        @atomic lambda_rm[ii_l - 1, jj, k] += bar * grad_rm[2]
        @atomic lambda_rm[ii_l,     jj, k] += bar * grad_rm[3]
        @atomic lambda_rm[ii_r,     jj, k] += bar * grad_rm[4]
        @atomic lambda_rm[ii_r + 1, jj, k] += bar * grad_rm[5]
        @atomic lambda_rm[ii_r + 2, jj, k] += bar * grad_rm[6]
        @atomic lambda_m[ii_l - 2, jj, k]  += bar * grad_m[1]
        @atomic lambda_m[ii_l - 1, jj, k]  += bar * grad_m[2]
        @atomic lambda_m[ii_l,     jj, k]  += bar * grad_m[3]
        @atomic lambda_m[ii_r,     jj, k]  += bar * grad_m[4]
        @atomic lambda_m[ii_r + 1, jj, k]  += bar * grad_m[5]
        @atomic lambda_m[ii_r + 2, jj, k]  += bar * grad_m[6]
    end
end

# ORD=7 from-rm x-face adjoint kernel. Mirrors the ORD=5 kernel with
# `face_idx = iif` driving the discontinuous boundary correction at
# panel-edge faces. The donor-state helpers also dispatch on (iif, Nc)
# so the α-contribution `(bl, br, b0)` reflects the boundary-corrected
# limited edges at panel-edge donor cells.
@kernel function _ppm_x_face_kernel_adjoint_ord7!(
    lambda_rm, lambda_m,
    @Const(lambda_fx_face), @Const(rm), @Const(m), @Const(am),
    Hp, Nc,
)
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        rm_m3 = rm[ii_l - 2, jj, k]; m_m3 = m[ii_l - 2, jj, k]
        rm_m2 = rm[ii_l - 1, jj, k]; m_m2 = m[ii_l - 1, jj, k]
        rm_m1 = rm[ii_l,     jj, k]; m_m1 = m[ii_l,     jj, k]
        rm_0  = rm[ii_r,     jj, k]; m_0  = m[ii_r,     jj, k]
        rm_p1 = rm[ii_r + 1, jj, k]; m_p1 = m[ii_r + 1, jj, k]
        rm_p2 = rm[ii_r + 2, jj, k]; m_p2 = m[ii_r + 2, jj, k]

        F = am[iif, j, k]
        grad_rm, grad_m = _linrood_ppm_face_from_rm_grad_ord7(
            F, m_m1, m_0,
            rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
            m_m3,  m_m2,  m_m1,  m_0,  m_p1,  m_p2,
            iif, Nc,
        )

        bar = lambda_fx_face[iif, j, k]
        @atomic lambda_rm[ii_l - 2, jj, k] += bar * grad_rm[1]
        @atomic lambda_rm[ii_l - 1, jj, k] += bar * grad_rm[2]
        @atomic lambda_rm[ii_l,     jj, k] += bar * grad_rm[3]
        @atomic lambda_rm[ii_r,     jj, k] += bar * grad_rm[4]
        @atomic lambda_rm[ii_r + 1, jj, k] += bar * grad_rm[5]
        @atomic lambda_rm[ii_r + 2, jj, k] += bar * grad_rm[6]
        @atomic lambda_m[ii_l - 2, jj, k]  += bar * grad_m[1]
        @atomic lambda_m[ii_l - 1, jj, k]  += bar * grad_m[2]
        @atomic lambda_m[ii_l,     jj, k]  += bar * grad_m[3]
        @atomic lambda_m[ii_r,     jj, k]  += bar * grad_m[4]
        @atomic lambda_m[ii_r + 1, jj, k]  += bar * grad_m[5]
        @atomic lambda_m[ii_r + 2, jj, k]  += bar * grad_m[6]
    end
end

"""
    apply_ppm_x_face_adjoint!(lambda_rm, lambda_m, lambda_fx_face, rm, m, am,
                                mesh, ::Val{ORD})

Discrete transpose of `_ppm_x_face_kernel!` (LinRood.jl:270) at ORD=5
or ORD=7. Folds `_safe_mixing_ratio` into the d6-AD chain and includes
the donor-mass `α = F / m_donor` contribution. Atomic accumulation on
shared cells.

`ORD=7` applies the linear discontinuous-edge boundary correction at
panel-edge faces (`face_idx ∈ {1, Nc+1}`) and recomputes the donor
α-contribution against the corrected limited `(q_L, q_R)`. Interior
faces are bit-equal to ORD=5.
"""
function apply_ppm_x_face_adjoint!(lambda_rm, lambda_m, lambda_fx_face,
                                   rm, m, am,
                                   mesh::CubedSphereMesh,
                                   ::Val{ORD}=Val(5)) where {ORD}
    (ORD == 5 || ORD == 7) || throw(ArgumentError(
        "LinRoodPPMScheme adjoint supports ORD ∈ {5, 7}; got ORD=$ORD."))
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(rm, 3)
    backend = get_backend(lambda_rm)
    k! = ORD == 5 ?
        _ppm_x_face_kernel_adjoint_ord5!(backend, 256) :
        _ppm_x_face_kernel_adjoint_ord7!(backend, 256)
    k!(lambda_rm, lambda_m, lambda_fx_face, rm, m, am, Hp, Nc;
       ndrange=(Nc + 1, Nc, Nz))
    synchronize(backend)
    return nothing
end

@kernel function _ppm_y_face_kernel_adjoint_ord5!(
    lambda_rm, lambda_m,
    @Const(lambda_fy_face), @Const(rm), @Const(m), @Const(bm),
    Hp, Nc,
)
    i, jf, k = @index(Global, NTuple)
    _ = Nc
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        rm_m3 = rm[ii, jj_b - 2, k]; m_m3 = m[ii, jj_b - 2, k]
        rm_m2 = rm[ii, jj_b - 1, k]; m_m2 = m[ii, jj_b - 1, k]
        rm_m1 = rm[ii, jj_b,     k]; m_m1 = m[ii, jj_b,     k]
        rm_0  = rm[ii, jj_a,     k]; m_0  = m[ii, jj_a,     k]
        rm_p1 = rm[ii, jj_a + 1, k]; m_p1 = m[ii, jj_a + 1, k]
        rm_p2 = rm[ii, jj_a + 2, k]; m_p2 = m[ii, jj_a + 2, k]

        F = bm[i, jf, k]
        grad_rm, grad_m = _linrood_ppm_face_from_rm_grad_ord5(
            F, m_m1, m_0,
            rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
            m_m3,  m_m2,  m_m1,  m_0,  m_p1,  m_p2,
        )

        bar = lambda_fy_face[i, jf, k]
        @atomic lambda_rm[ii, jj_b - 2, k] += bar * grad_rm[1]
        @atomic lambda_rm[ii, jj_b - 1, k] += bar * grad_rm[2]
        @atomic lambda_rm[ii, jj_b,     k] += bar * grad_rm[3]
        @atomic lambda_rm[ii, jj_a,     k] += bar * grad_rm[4]
        @atomic lambda_rm[ii, jj_a + 1, k] += bar * grad_rm[5]
        @atomic lambda_rm[ii, jj_a + 2, k] += bar * grad_rm[6]
        @atomic lambda_m[ii, jj_b - 2, k]  += bar * grad_m[1]
        @atomic lambda_m[ii, jj_b - 1, k]  += bar * grad_m[2]
        @atomic lambda_m[ii, jj_b,     k]  += bar * grad_m[3]
        @atomic lambda_m[ii, jj_a,     k]  += bar * grad_m[4]
        @atomic lambda_m[ii, jj_a + 1, k]  += bar * grad_m[5]
        @atomic lambda_m[ii, jj_a + 2, k]  += bar * grad_m[6]
    end
end

# ORD=7 from-rm y-face adjoint kernel.
@kernel function _ppm_y_face_kernel_adjoint_ord7!(
    lambda_rm, lambda_m,
    @Const(lambda_fy_face), @Const(rm), @Const(m), @Const(bm),
    Hp, Nc,
)
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        rm_m3 = rm[ii, jj_b - 2, k]; m_m3 = m[ii, jj_b - 2, k]
        rm_m2 = rm[ii, jj_b - 1, k]; m_m2 = m[ii, jj_b - 1, k]
        rm_m1 = rm[ii, jj_b,     k]; m_m1 = m[ii, jj_b,     k]
        rm_0  = rm[ii, jj_a,     k]; m_0  = m[ii, jj_a,     k]
        rm_p1 = rm[ii, jj_a + 1, k]; m_p1 = m[ii, jj_a + 1, k]
        rm_p2 = rm[ii, jj_a + 2, k]; m_p2 = m[ii, jj_a + 2, k]

        F = bm[i, jf, k]
        grad_rm, grad_m = _linrood_ppm_face_from_rm_grad_ord7(
            F, m_m1, m_0,
            rm_m3, rm_m2, rm_m1, rm_0, rm_p1, rm_p2,
            m_m3,  m_m2,  m_m1,  m_0,  m_p1,  m_p2,
            jf, Nc,
        )

        bar = lambda_fy_face[i, jf, k]
        @atomic lambda_rm[ii, jj_b - 2, k] += bar * grad_rm[1]
        @atomic lambda_rm[ii, jj_b - 1, k] += bar * grad_rm[2]
        @atomic lambda_rm[ii, jj_b,     k] += bar * grad_rm[3]
        @atomic lambda_rm[ii, jj_a,     k] += bar * grad_rm[4]
        @atomic lambda_rm[ii, jj_a + 1, k] += bar * grad_rm[5]
        @atomic lambda_rm[ii, jj_a + 2, k] += bar * grad_rm[6]
        @atomic lambda_m[ii, jj_b - 2, k]  += bar * grad_m[1]
        @atomic lambda_m[ii, jj_b - 1, k]  += bar * grad_m[2]
        @atomic lambda_m[ii, jj_b,     k]  += bar * grad_m[3]
        @atomic lambda_m[ii, jj_a,     k]  += bar * grad_m[4]
        @atomic lambda_m[ii, jj_a + 1, k]  += bar * grad_m[5]
        @atomic lambda_m[ii, jj_a + 2, k]  += bar * grad_m[6]
    end
end

"""
    apply_ppm_y_face_adjoint!(lambda_rm, lambda_m, lambda_fy_face, rm, m, bm,
                                mesh, ::Val{ORD})

Discrete transpose of `_ppm_y_face_kernel!` (LinRood.jl:241) at ORD=5
or ORD=7. See `apply_ppm_x_face_adjoint!` for the contract.
"""
function apply_ppm_y_face_adjoint!(lambda_rm, lambda_m, lambda_fy_face,
                                   rm, m, bm,
                                   mesh::CubedSphereMesh,
                                   ::Val{ORD}=Val(5)) where {ORD}
    (ORD == 5 || ORD == 7) || throw(ArgumentError(
        "LinRoodPPMScheme adjoint supports ORD ∈ {5, 7}; got ORD=$ORD."))
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(rm, 3)
    backend = get_backend(lambda_rm)
    k! = ORD == 5 ?
        _ppm_y_face_kernel_adjoint_ord5!(backend, 256) :
        _ppm_y_face_kernel_adjoint_ord7!(backend, 256)
    k!(lambda_rm, lambda_m, lambda_fy_face, rm, m, bm, Hp, Nc;
       ndrange=(Nc, Nc + 1, Nz))
    synchronize(backend)
    return nothing
end
