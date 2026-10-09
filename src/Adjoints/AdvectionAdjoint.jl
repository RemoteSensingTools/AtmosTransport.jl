# ---------------------------------------------------------------------------
# Kernelized adjoint building blocks for CS split-sweep advection.
#
# Reverse-mode of the per-direction sweep kernels in
# `src/Operators/Advection/`. The file collects:
#
#   * Per-scheme face-coefficient helpers (`_upwind_face_coeffs`,
#     `_slopes_no_limiter_face_coeffs`, `_ppm_no_limiter_face_coeffs`,
#     and `_ppm_face_coeffs`, the rm-input variant for the monotone and
#     CW84 limiters).
#   * Per-direction interior + face-edge adjoint update helpers
#     (`_add_x_face_adjoint!`, `_add_y_face_adjoint!`, `_add_z_face_adjoint!`).
#   * Per-direction sweep adjoint kernels (`_cs_xsweep_adjoint_kernel!`,
#     `_cs_ysweep_adjoint_kernel!`, `_cs_zsweep_adjoint_kernel!`).
#   * The driver `_adjoint_scheme_sweep!` for both linear and monotone
#     PPM schemes; LinRood is handled out-of-line in `LinRoodTape.jl`.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kernelized adjoint building blocks
# ---------------------------------------------------------------------------

@inline _wrap_periodic(idx, N) = mod1(idx, N)

@inline function _upwind_face_coeffs(F, m_l, m_r)
    FT = typeof(F)
    if F >= zero(FT)
        return clamp(F / max(m_l, eps(FT)), zero(FT), one(FT)), zero(FT)
    else
        return zero(FT), clamp(F / max(m_r, eps(FT)), -one(FT), zero(FT))
    end
end

@inline function _slopes_no_limiter_face_coeffs(F, m_ll, m_l, m_r, m_rr)
    FT = typeof(F)
    m_floor = eps(FT)
    mll = max(m_ll, m_floor)
    ml = max(m_l, m_floor)
    mr = max(m_r, m_floor)
    mrr = max(m_rr, m_floor)
    if F >= zero(FT)
        α = clamp(F / ml, zero(FT), one(FT))
        β = α * (one(FT) - α) * ml / FT(4)
        return -β / mll, α, β / mr, zero(FT)
    else
        α = clamp(F / mr, -one(FT), zero(FT))
        β = -α * (one(FT) + α) * mr / FT(4)
        return zero(FT), -β / ml, α, β / mrr
    end
end

@inline function _ppm_no_limiter_face_coeffs(F, m_ll, m_l, m_r, m_rr)
    FT = typeof(F)
    tw12 = FT(1) / FT(12)
    m_floor = eps(FT)
    mll = max(m_ll, m_floor)
    ml = max(m_l, m_floor)
    mr = max(m_r, m_floor)
    mrr = max(m_rr, m_floor)
    if F >= zero(FT)
        α = clamp(F / ml, zero(FT), one(FT))
        β = α * (one(FT) - α) * ml
        c_ll = β * (-tw12) / mll
        c_l  = α + β * (FT(-5) * tw12) / ml
        c_r  = β * (FT(7) * tw12) / mr
        c_rr = β * (-tw12) / mrr
        return c_ll, c_l, c_r, c_rr
    else
        α = clamp(F / mr, -one(FT), zero(FT))
        β = -α * (one(FT) + α) * mr
        c_ll = β * tw12 / mll
        c_l  = β * (FT(-7) * tw12) / ml
        c_r  = α + β * (FT(5) * tw12) / mr
        c_rr = β * tw12 / mrr
        return c_ll, c_l, c_r, c_rr
    end
end

@inline _d6_zero(::Type{FT}) where {FT} =
    (zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))

@inline _d6_basis(::Type{FT}, n::Int, scale) where {FT} =
    (n == 1 ? FT(scale) : zero(FT),
     n == 2 ? FT(scale) : zero(FT),
     n == 3 ? FT(scale) : zero(FT),
     n == 4 ? FT(scale) : zero(FT),
     n == 5 ? FT(scale) : zero(FT),
     n == 6 ? FT(scale) : zero(FT))

@inline _d6_add(a, b) =
    (a[1] + b[1], a[2] + b[2], a[3] + b[3],
     a[4] + b[4], a[5] + b[5], a[6] + b[6])

@inline _d6_sub(a, b) =
    (a[1] - b[1], a[2] - b[2], a[3] - b[3],
     a[4] - b[4], a[5] - b[5], a[6] - b[6])

@inline _d6_scale(a, s) =
    (s * a[1], s * a[2], s * a[3],
     s * a[4], s * a[5], s * a[6])

@inline function _ppm_edge_value_ad(c_ll, d_ll, c_l, d_l, c_r, d_r, c_rr, d_rr)
    FT = typeof(c_ll)
    seven_twelfths = FT(7) / FT(12)
    one_twelfth = FT(1) / FT(12)
    value = seven_twelfths * (c_l + c_r) - one_twelfth * (c_ll + c_rr)
    deriv = _d6_sub(_d6_scale(_d6_add(d_l, d_r), seven_twelfths),
                    _d6_scale(_d6_add(d_ll, d_rr), one_twelfth))
    return value, deriv
end

@inline _ppm_edge_value_ad(c_ll, d_ll, c_l, d_l, c_r, d_r, c_rr, d_rr, ::MonotoneLimiter) =
    _ppm_edge_value_ad(c_ll, d_ll, c_l, d_l, c_r, d_r, c_rr, d_rr)

# CW84 edge (`_ppm_edge_value(…, ::CW84Limiter)`) with van Leer-limited slopes.
@inline function _ppm_edge_value_ad(c_ll, d_ll, c_l, d_l, c_r, d_r, c_rr, d_rr, ::CW84Limiter)
    FT = typeof(c_l)
    δc_l, dδc_l = _limited_slope_monotone_ad(c_ll, d_ll, c_l, d_l, c_r, d_r)
    δc_r, dδc_r = _limited_slope_monotone_ad(c_l, d_l, c_r, d_r, c_rr, d_rr)
    value = (c_l + c_r) / FT(2) - (δc_r - δc_l) / FT(6)
    deriv = _d6_sub(_d6_scale(_d6_add(d_l, d_r), FT(1) / FT(2)),
                    _d6_scale(_d6_sub(dδc_r, dδc_l), FT(1) / FT(6)))
    return value, deriv
end

# `_limited_slope(c_m, c_0, c_p, MonotoneLimiter())`: the minmod picks the
# argument of smallest magnitude when all three share a sign.
@inline function _limited_slope_monotone_ad(c_m, d_m, c_0, d_0, c_p, d_p)
    FT = typeof(c_0)
    sc = (c_p - c_m) / 2
    sp = 2 * (c_p - c_0)
    sm = 2 * (c_0 - c_m)
    same_sign = ((sc > zero(FT)) & (sp > zero(FT)) & (sm > zero(FT))) |
                ((sc < zero(FT)) & (sp < zero(FT)) & (sm < zero(FT)))
    same_sign || return zero(FT), _d6_zero(FT)
    if abs(sc) <= abs(sp) && abs(sc) <= abs(sm)
        return sc, _d6_scale(_d6_sub(d_p, d_m), FT(1) / FT(2))
    elseif abs(sp) <= abs(sm)
        return sp, _d6_scale(_d6_sub(d_p, d_0), FT(2))
    else
        return sm, _d6_scale(_d6_sub(d_0, d_m), FT(2))
    end
end

@inline function _ppm_limit_profile_monotone_ad(q_L, dq_L, c_bar, dc_bar, q_R, dq_R)
    FT = typeof(c_bar)
    is_extremum = (q_R - c_bar) * (c_bar - q_L) <= zero(FT)
    dc = q_R - q_L
    c6 = FT(6) * (c_bar - (q_L + q_R) / FT(2))
    needs_left_fix = dc * c6 > dc * dc
    needs_right_fix = -(dc * dc) > dc * c6

    if is_extremum
        return c_bar, dc_bar, c_bar, dc_bar
    end

    q_L_new = q_L
    dq_L_new = dq_L
    if needs_left_fix
        q_L_new = FT(3) * c_bar - FT(2) * q_R
        dq_L_new = _d6_sub(_d6_scale(dc_bar, FT(3)), _d6_scale(dq_R, FT(2)))
    end

    if needs_right_fix
        q_R_new = FT(3) * c_bar - FT(2) * q_L_new
        dq_R_new = _d6_sub(_d6_scale(dc_bar, FT(3)), _d6_scale(dq_L_new, FT(2)))
        return q_L_new, dq_L_new, q_R_new, dq_R_new
    end

    return q_L_new, dq_L_new, q_R, dq_R
end

@inline _limited_moment_monotone_ad(sx, dsx, _rm_cell, _drm_cell) = (sx, dsx)

# Curvature part of the PPM moment (`_ppm_raw_moments`): none for the monotone
# limiter, α·b₀ with b₀ = (q_L − c) + (q_R − c) for CW84.
@inline _ppm_curvature_ad(_α, _q_L, _dq_L, c, _dc, _q_R, _dq_R, ::MonotoneLimiter) =
    (zero(c), _d6_zero(typeof(c)))

@inline function _ppm_curvature_ad(α, q_L, dq_L, c, dc, q_R, dq_R, ::CW84Limiter)
    FT = typeof(c)
    b0 = (q_L - c) + (q_R - c)
    db0 = _d6_sub(_d6_add(dq_L, dq_R), _d6_scale(dc, FT(2)))
    return α * b0, _d6_scale(db0, α)
end

"""
    _ppm_face_coeffs(F, m_3, …, m_pp, rm_3, …, rm_pp, interior_l, interior_r, limiter) → NTuple{6}

Derivatives of the structured PPM face flux (`_xface_tracer_flux` and its y/z
twins) with respect to the tracer masses of the six stencil cells, for the
monotone and CW84 limiters. The branches (limiters, flow direction) are those
of the forward at the taped tracer masses.
"""
@inline function _ppm_face_coeffs(F,
                                  m_3, m_2, m_1, m_0, m_p, m_pp,
                                  rm_3, rm_2, rm_1, rm_0, rm_p, rm_pp,
                                  interior_l::Bool, interior_r::Bool, limiter)
    FT = typeof(F)
    m_floor = eps(FT)
    m3 = max(m_3, m_floor)
    m2 = max(m_2, m_floor)
    m1 = max(m_1, m_floor)
    m0 = max(m_0, m_floor)
    mp = max(m_p, m_floor)
    mpp = max(m_pp, m_floor)

    c_3 = rm_3 / m3
    c_2 = rm_2 / m2
    c_1 = rm_1 / m1
    c_0 = rm_0 / m0
    c_p = rm_p / mp
    c_pp = rm_pp / mpp

    dc_3 = _d6_basis(FT, 1, inv(m3))
    dc_2 = _d6_basis(FT, 2, inv(m2))
    dc_1 = _d6_basis(FT, 3, inv(m1))
    dc_0 = _d6_basis(FT, 4, inv(m0))
    dc_p = _d6_basis(FT, 5, inv(mp))
    dc_pp = _d6_basis(FT, 6, inv(mpp))

    e_left, de_left = _ppm_edge_value_ad(c_3, dc_3, c_2, dc_2, c_1, dc_1, c_0, dc_0, limiter)
    e_face, de_face = _ppm_edge_value_ad(c_2, dc_2, c_1, dc_1, c_0, dc_0, c_p, dc_p, limiter)
    e_right, de_right = _ppm_edge_value_ad(c_1, dc_1, c_0, dc_0, c_p, dc_p, c_pp, dc_pp, limiter)

    qLl, dqLl, qRl, dqRl =
        _ppm_limit_profile_monotone_ad(e_left, de_left, c_1, dc_1, e_face, de_face)
    qLr, dqLr, qRr, dqRr =
        _ppm_limit_profile_monotone_ad(e_face, de_face, c_0, dc_0, e_right, de_right)

    α_pos = clamp(F / m1, zero(FT), one(FT))
    α_neg = clamp(F / m0, -one(FT), zero(FT))

    κ_l, dκ_l = _ppm_curvature_ad(α_pos, qLl, dqLl, c_1, dc_1, qRl, dqRl, limiter)
    sx_l_raw = m1 * ((qRl - c_1) - κ_l)
    dsx_l_raw = _d6_scale(_d6_sub(_d6_sub(dqRl, dc_1), dκ_l), m1)
    sx_l, dsx_l = interior_l ?
        _limited_moment_monotone_ad(sx_l_raw, dsx_l_raw, rm_1, _d6_basis(FT, 3, one(FT))) :
        (zero(FT), _d6_zero(FT))

    κ_r, dκ_r = _ppm_curvature_ad(α_neg, qLr, dqLr, c_0, dc_0, qRr, dqRr, limiter)
    sx_r_raw = m0 * ((c_0 - qLr) - κ_r)
    dsx_r_raw = _d6_scale(_d6_sub(_d6_sub(dc_0, dqLr), dκ_r), m0)
    sx_r, dsx_r = interior_r ?
        _limited_moment_monotone_ad(sx_r_raw, dsx_r_raw, rm_0, _d6_basis(FT, 4, one(FT))) :
        (zero(FT), _d6_zero(FT))

    if F >= zero(FT)
        drm_l = _d6_basis(FT, 3, one(FT))
        return _d6_add(_d6_scale(drm_l, α_pos),
                       _d6_scale(dsx_l, α_pos * (one(FT) - α_pos)))
    else
        drm_r = _d6_basis(FT, 4, one(FT))
        return _d6_sub(_d6_scale(drm_r, α_neg),
                       _d6_scale(dsx_r, α_neg * (one(FT) + α_neg)))
    end
end

@inline function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
                                      ::UpwindScheme, Nx)
    i_l = _wrap_periodic(face_i - Int32(1), Nx)
    i_r = _wrap_periodic(face_i, Nx)
    c_l, c_r = _upwind_face_coeffs(F, m[i_l, j, k], m[i_r, j, k])
    @atomic lambda_in[i_l, j, k] += scale * c_l
    @atomic lambda_in[i_r, j, k] += scale * c_r
    return nothing
end

@inline function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
                                      ::SlopesScheme{NoLimiter}, Nx)
    i_ll = _wrap_periodic(face_i - Int32(2), Nx)
    i_l  = _wrap_periodic(face_i - Int32(1), Nx)
    i_r  = _wrap_periodic(face_i, Nx)
    i_rr = _wrap_periodic(face_i + Int32(1), Nx)
    c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
        F, m[i_ll, j, k], m[i_l, j, k], m[i_r, j, k], m[i_rr, j, k])
    @atomic lambda_in[i_ll, j, k] += scale * c_ll
    @atomic lambda_in[i_l,  j, k] += scale * c_l
    @atomic lambda_in[i_r,  j, k] += scale * c_r
    @atomic lambda_in[i_rr, j, k] += scale * c_rr
    return nothing
end

@inline function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
                                      ::PPMScheme{NoLimiter}, Nx)
    i_ll = _wrap_periodic(face_i - Int32(2), Nx)
    i_l  = _wrap_periodic(face_i - Int32(1), Nx)
    i_r  = _wrap_periodic(face_i, Nx)
    i_rr = _wrap_periodic(face_i + Int32(1), Nx)
    c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
        F, m[i_ll, j, k], m[i_l, j, k], m[i_r, j, k], m[i_rr, j, k])
    @atomic lambda_in[i_ll, j, k] += scale * c_ll
    @atomic lambda_in[i_l,  j, k] += scale * c_l
    @atomic lambda_in[i_r,  j, k] += scale * c_r
    @atomic lambda_in[i_rr, j, k] += scale * c_rr
    return nothing
end

@inline function _add_x_face_adjoint!(lambda_in, m, rm, face_i, j, k, F, scale,
                                      scheme::PPMScheme{<:_PPMTapedLimiter}, Nx)
    i_3  = _wrap_periodic(face_i - Int32(3), Nx)
    i_2  = _wrap_periodic(face_i - Int32(2), Nx)
    i_1  = _wrap_periodic(face_i - Int32(1), Nx)
    i_0  = _wrap_periodic(face_i, Nx)
    i_p  = _wrap_periodic(face_i + Int32(1), Nx)
    i_pp = _wrap_periodic(face_i + Int32(2), Nx)
    c = _ppm_face_coeffs(
        F,
        m[i_3, j, k], m[i_2, j, k], m[i_1, j, k],
        m[i_0, j, k], m[i_p, j, k], m[i_pp, j, k],
        rm[i_3, j, k], rm[i_2, j, k], rm[i_1, j, k],
        rm[i_0, j, k], rm[i_p, j, k], rm[i_pp, j, k],
        true, true, scheme.limiter)
    @atomic lambda_in[i_3,  j, k] += scale * c[1]
    @atomic lambda_in[i_2,  j, k] += scale * c[2]
    @atomic lambda_in[i_1,  j, k] += scale * c[3]
    @atomic lambda_in[i_0,  j, k] += scale * c[4]
    @atomic lambda_in[i_p,  j, k] += scale * c[5]
    @atomic lambda_in[i_pp, j, k] += scale * c[6]
    return nothing
end

@inline function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
                                      ::UpwindScheme, Ny)
    FT = typeof(F)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    jl = max(face_j - Int32(1), Int32(1))
    jr = min(face_j, Ny)
    c_l, c_r = _upwind_face_coeffs(F, m[i, jl, k], m[i, jr, k])
    @atomic lambda_in[i, jl, k] += scale * c_l
    @atomic lambda_in[i, jr, k] += scale * c_r
    return nothing
end

@inline function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
                                      ::SlopesScheme{NoLimiter}, Ny)
    FT = typeof(F)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    jll = max(face_j - Int32(2), Int32(1))
    jl  = max(face_j - Int32(1), Int32(1))
    jr  = min(face_j, Ny)
    jrr = min(face_j + Int32(1), Ny)
    interior_l = (jl > Int32(1)) & (jl < Ny)
    interior_r = (jr > Int32(1)) & (jr < Ny)
    if F >= zero(FT)
        if interior_l
            c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
                F, m[i, jll, k], m[i, jl, k], m[i, jr, k], m[i, jrr, k])
        else
            c_ll = zero(FT)
            c_l, _ = _upwind_face_coeffs(F, m[i, jl, k], m[i, jr, k])
            c_r = zero(FT)
            c_rr = zero(FT)
        end
    else
        if interior_r
            c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
                F, m[i, jll, k], m[i, jl, k], m[i, jr, k], m[i, jrr, k])
        else
            c_ll = zero(FT)
            c_l = zero(FT)
            _, c_r = _upwind_face_coeffs(F, m[i, jl, k], m[i, jr, k])
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, jll, k] += scale * c_ll
    @atomic lambda_in[i, jl,  k] += scale * c_l
    @atomic lambda_in[i, jr,  k] += scale * c_r
    @atomic lambda_in[i, jrr, k] += scale * c_rr
    return nothing
end

@inline function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
                                      ::PPMScheme{NoLimiter}, Ny)
    FT = typeof(F)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    jll = max(face_j - Int32(2), Int32(1))
    jl  = max(face_j - Int32(1), Int32(1))
    jr  = min(face_j, Ny)
    jrr = min(face_j + Int32(1), Ny)
    interior_l = (jl > Int32(2)) & (jl < Ny - Int32(1))
    interior_r = (jr > Int32(2)) & (jr < Ny - Int32(1))
    if F >= zero(FT)
        if interior_l
            c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
                F, m[i, jll, k], m[i, jl, k], m[i, jr, k], m[i, jrr, k])
        else
            c_ll = zero(FT)
            c_l = clamp(F / max(m[i, jl, k], eps(FT)), zero(FT), one(FT))
            c_r = zero(FT)
            c_rr = zero(FT)
        end
    else
        if interior_r
            c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
                F, m[i, jll, k], m[i, jl, k], m[i, jr, k], m[i, jrr, k])
        else
            c_ll = zero(FT)
            c_l = zero(FT)
            c_r = clamp(F / max(m[i, jr, k], eps(FT)), -one(FT), zero(FT))
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, jll, k] += scale * c_ll
    @atomic lambda_in[i, jl,  k] += scale * c_l
    @atomic lambda_in[i, jr,  k] += scale * c_r
    @atomic lambda_in[i, jrr, k] += scale * c_rr
    return nothing
end

@inline function _add_y_face_adjoint!(lambda_in, m, rm, i, face_j, k, F, scale,
                                      scheme::PPMScheme{<:_PPMTapedLimiter}, Ny)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    j3l = max(face_j - Int32(3), Int32(1))
    jll = max(face_j - Int32(2), Int32(1))
    jl  = max(face_j - Int32(1), Int32(1))
    jr  = min(face_j, Ny)
    jrr = min(face_j + Int32(1), Ny)
    j3r = min(face_j + Int32(2), Ny)
    interior_l = (jl > Int32(2)) & (jl < Ny - Int32(1))
    interior_r = (jr > Int32(2)) & (jr < Ny - Int32(1))
    c = _ppm_face_coeffs(
        F,
        m[i, j3l, k], m[i, jll, k], m[i, jl, k],
        m[i, jr, k], m[i, jrr, k], m[i, j3r, k],
        rm[i, j3l, k], rm[i, jll, k], rm[i, jl, k],
        rm[i, jr, k], rm[i, jrr, k], rm[i, j3r, k],
        interior_l, interior_r, scheme.limiter)
    @atomic lambda_in[i, j3l, k] += scale * c[1]
    @atomic lambda_in[i, jll, k] += scale * c[2]
    @atomic lambda_in[i, jl,  k] += scale * c[3]
    @atomic lambda_in[i, jr,  k] += scale * c[4]
    @atomic lambda_in[i, jrr, k] += scale * c[5]
    @atomic lambda_in[i, j3r, k] += scale * c[6]
    return nothing
end

@inline function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
                                      ::UpwindScheme, Nz)
    FT = typeof(F)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    kl = max(face_k - Int32(1), Int32(1))
    kr = min(face_k, Nz)
    c_l, c_r = _upwind_face_coeffs(F, m[i, j, kl], m[i, j, kr])
    @atomic lambda_in[i, j, kl] += scale * c_l
    @atomic lambda_in[i, j, kr] += scale * c_r
    return nothing
end

@inline function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
                                      ::SlopesScheme{NoLimiter}, Nz)
    FT = typeof(F)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    kll = max(face_k - Int32(2), Int32(1))
    kl  = max(face_k - Int32(1), Int32(1))
    kr  = min(face_k, Nz)
    krr = min(face_k + Int32(1), Nz)
    interior_l = (kl > Int32(1)) & (kl < Nz)
    interior_r = (kr > Int32(1)) & (kr < Nz)
    if F >= zero(FT)
        if interior_l
            c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
                F, m[i, j, kll], m[i, j, kl], m[i, j, kr], m[i, j, krr])
        else
            c_ll = zero(FT)
            c_l, _ = _upwind_face_coeffs(F, m[i, j, kl], m[i, j, kr])
            c_r = zero(FT)
            c_rr = zero(FT)
        end
    else
        if interior_r
            c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
                F, m[i, j, kll], m[i, j, kl], m[i, j, kr], m[i, j, krr])
        else
            c_ll = zero(FT)
            c_l = zero(FT)
            _, c_r = _upwind_face_coeffs(F, m[i, j, kl], m[i, j, kr])
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, j, kll] += scale * c_ll
    @atomic lambda_in[i, j, kl]  += scale * c_l
    @atomic lambda_in[i, j, kr]  += scale * c_r
    @atomic lambda_in[i, j, krr] += scale * c_rr
    return nothing
end

@inline function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
                                      ::PPMScheme{NoLimiter, SameAsHorizontal}, Nz)
    FT = typeof(F)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    kll = max(face_k - Int32(2), Int32(1))
    kl  = max(face_k - Int32(1), Int32(1))
    kr  = min(face_k, Nz)
    krr = min(face_k + Int32(1), Nz)
    interior_l = (kl > Int32(2)) & (kl < Nz - Int32(1))
    interior_r = (kr > Int32(2)) & (kr < Nz - Int32(1))
    if F >= zero(FT)
        if interior_l
            c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
                F, m[i, j, kll], m[i, j, kl], m[i, j, kr], m[i, j, krr])
        else
            c_ll = zero(FT)
            c_l = clamp(F / max(m[i, j, kl], eps(FT)), zero(FT), one(FT))
            c_r = zero(FT)
            c_rr = zero(FT)
        end
    else
        if interior_r
            c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
                F, m[i, j, kll], m[i, j, kl], m[i, j, kr], m[i, j, krr])
        else
            c_ll = zero(FT)
            c_l = zero(FT)
            c_r = clamp(F / max(m[i, j, kr], eps(FT)), -one(FT), zero(FT))
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, j, kll] += scale * c_ll
    @atomic lambda_in[i, j, kl]  += scale * c_l
    @atomic lambda_in[i, j, kr]  += scale * c_r
    @atomic lambda_in[i, j, krr] += scale * c_rr
    return nothing
end

@inline function _add_z_face_adjoint!(lambda_in, m, rm, i, j, face_k, F, scale,
                                      scheme::CSAdjointNonlinearScheme, Nz)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    k3l = max(face_k - Int32(3), Int32(1))
    kll = max(face_k - Int32(2), Int32(1))
    kl  = max(face_k - Int32(1), Int32(1))
    kr  = min(face_k, Nz)
    krr = min(face_k + Int32(1), Nz)
    k3r = min(face_k + Int32(2), Nz)
    interior_l = (kl > Int32(2)) & (kl < Nz - Int32(1))
    interior_r = (kr > Int32(2)) & (kr < Nz - Int32(1))
    c = _ppm_face_coeffs(
        F,
        m[i, j, k3l], m[i, j, kll], m[i, j, kl],
        m[i, j, kr], m[i, j, krr], m[i, j, k3r],
        rm[i, j, k3l], rm[i, j, kll], rm[i, j, kl],
        rm[i, j, kr], rm[i, j, krr], rm[i, j, k3r],
        interior_l, interior_r, scheme.limiter)
    @atomic lambda_in[i, j, k3l] += scale * c[1]
    @atomic lambda_in[i, j, kll] += scale * c[2]
    @atomic lambda_in[i, j, kl]  += scale * c[3]
    @atomic lambda_in[i, j, kr]  += scale * c[4]
    @atomic lambda_in[i, j, krr] += scale * c[5]
    @atomic lambda_in[i, j, k3r] += scale * c[6]
    return nothing
end

@kernel function _cs_xsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(am),
                                            scheme, Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Nx = Int32(Nc + 2 * Hp)
        _add_x_face_adjoint!(lambda_in, m, Int32(i),     j, k, flux_scale * am[i,     j, k],  bar, scheme, Nx)
        _add_x_face_adjoint!(lambda_in, m, Int32(i) + 1, j, k, flux_scale * am[i + 1, j, k], -bar, scheme, Nx)
    end
end

@kernel function _cs_ysweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(bm),
                                            scheme, Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Ny = Int32(Nc + 2 * Hp)
        _add_y_face_adjoint!(lambda_in, m, i, Int32(j),     k, flux_scale * bm[i, j,     k],  bar, scheme, Ny)
        _add_y_face_adjoint!(lambda_in, m, i, Int32(j) + 1, k, flux_scale * bm[i, j + 1, k], -bar, scheme, Ny)
    end
end

@kernel function _cs_zsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(cm),
                                            scheme, Nc, Hp, Nz, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        _add_z_face_adjoint!(lambda_in, m, i, j, Int32(k),     flux_scale * cm[i, j, k],     bar, scheme, Int32(Nz))
        _add_z_face_adjoint!(lambda_in, m, i, j, Int32(k) + 1, flux_scale * cm[i, j, k + 1], -bar, scheme, Int32(Nz))
    end
end

@kernel function _cs_xsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(rm), @Const(am),
                                            scheme::PPMScheme{<:_PPMTapedLimiter},
                                            Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Nx = Int32(Nc + 2 * Hp)
        _add_x_face_adjoint!(lambda_in, m, rm, Int32(i),     j, k, flux_scale * am[i,     j, k],  bar, scheme, Nx)
        _add_x_face_adjoint!(lambda_in, m, rm, Int32(i) + 1, j, k, flux_scale * am[i + 1, j, k], -bar, scheme, Nx)
    end
end

@kernel function _cs_ysweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(rm), @Const(bm),
                                            scheme::PPMScheme{<:_PPMTapedLimiter},
                                            Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Ny = Int32(Nc + 2 * Hp)
        _add_y_face_adjoint!(lambda_in, m, rm, i, Int32(j),     k, flux_scale * bm[i, j,     k],  bar, scheme, Ny)
        _add_y_face_adjoint!(lambda_in, m, rm, i, Int32(j) + 1, k, flux_scale * bm[i, j + 1, k], -bar, scheme, Ny)
    end
end

@kernel function _cs_zsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(rm), @Const(cm),
                                            scheme::CSAdjointNonlinearScheme,
                                            Nc, Hp, Nz, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        _add_z_face_adjoint!(lambda_in, m, rm, i, j, Int32(k),     flux_scale * cm[i, j, k],     bar, scheme, Int32(Nz))
        _add_z_face_adjoint!(lambda_in, m, rm, i, j, Int32(k) + 1, flux_scale * cm[i, j, k + 1], -bar, scheme, Int32(Nz))
    end
end

function _adjoint_scheme_sweep!(lambda_panels, m_before, flux_panels,
                                direction::Symbol, scheme::CSAdjointLinearScheme,
                                mesh::CubedSphereMesh, ws::CSAdjointWorkspace,
                                flux_scale)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(lambda_panels[1], 3)
    horizontal = direction === :x || direction === :y
    axis = direction === :x ? Val(1) : Val(2)
    horizontal && _cache_cs_seam_seeds!(ws.seam_seeds, lambda_panels, mesh, axis)
    @inbounds for p in 1:6
        fill!(ws.lambda_A, zero(eltype(ws.lambda_A)))
        backend = get_backend(lambda_panels[p])
        panel_flux = horizontal ? _CSInteriorFlux(flux_panels[p], mesh, axis) : flux_panels[p]
        if direction === :x
            kernel! = _cs_xsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], panel_flux,
                    scheme, Int32(Nc), Int32(Hp), eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :y
            kernel! = _cs_ysweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], panel_flux,
                    scheme, Int32(Nc), Int32(Hp), eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :z
            kernel! = _cs_zsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], panel_flux,
                    scheme, Int32(Nc), Int32(Hp), Int32(Nz), eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        else
            throw(ArgumentError("unknown CS adjoint sweep direction $direction"))
        end
        synchronize(backend)
        copyto!(lambda_panels[p], ws.lambda_A)
    end
    horizontal && _apply_cs_seam_adjoint!(lambda_panels, m_before, nothing,
        flux_panels, ws.seam_seeds, mesh, scheme, axis, flux_scale)
    return nothing
end

# PPM adjoints cover the default vertical reconstruction only; the FV3 vertical
# profile (`FV3ScalarProfile`) has no adjoint yet.
function _adjoint_scheme_sweep!(lambda_panels, m_before, rm_before, flux_panels,
                                direction::Symbol, scheme::CSAdjointNonlinearScheme,
                                mesh::CubedSphereMesh, ws::CSAdjointWorkspace,
                                flux_scale)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(lambda_panels[1], 3)
    horizontal = direction === :x || direction === :y
    axis = direction === :x ? Val(1) : Val(2)
    horizontal && _cache_cs_seam_seeds!(ws.seam_seeds, lambda_panels, mesh, axis)
    @inbounds for p in 1:6
        fill!(ws.lambda_A, zero(eltype(ws.lambda_A)))
        backend = get_backend(lambda_panels[p])
        panel_flux = horizontal ? _CSInteriorFlux(flux_panels[p], mesh, axis) : flux_panels[p]
        if direction === :x
            kernel! = _cs_xsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], rm_before[p],
                    panel_flux, scheme, Int32(Nc), Int32(Hp),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :y
            kernel! = _cs_ysweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], rm_before[p],
                    panel_flux, scheme, Int32(Nc), Int32(Hp),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :z
            kernel! = _cs_zsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], rm_before[p],
                    panel_flux, scheme, Int32(Nc), Int32(Hp), Int32(Nz),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        else
            throw(ArgumentError("unknown CS adjoint sweep direction $direction"))
        end
        synchronize(backend)
        copyto!(lambda_panels[p], ws.lambda_A)
    end
    horizontal && _apply_cs_seam_adjoint!(lambda_panels, m_before, rm_before,
        flux_panels, ws.seam_seeds, mesh, scheme, axis, flux_scale)
    return nothing
end
