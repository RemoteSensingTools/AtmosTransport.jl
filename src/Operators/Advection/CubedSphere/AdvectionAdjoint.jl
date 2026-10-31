# ---------------------------------------------------------------------------
# Kernelized adjoint building blocks for CS split-sweep advection.
#
# Reverse-mode of the per-direction sweep kernels in
# `src/Operators/Advection/`. The file collects:
#
#   * Per-scheme face-coefficient helpers (`_upwind_face_coeffs`,
#     `_slopes_no_limiter_face_coeffs`, `_ppm_no_limiter_face_coeffs`,
#     and `_ppm_face_coeffs` for the monotone and CW84 limiters, which reads
#     the limiter branches the tape recorded instead of the tracer mass).
#   * The limiter branches of a limited PPM sweep (`PPMLimiterCodes`,
#     `record_ppm_limiter_codes!`): one byte per cell, computed when the tape
#     is recorded.
#   * Per-direction interior + face-edge adjoint update helpers
#     (`_add_x_face_adjoint!`, `_add_y_face_adjoint!`, `_add_z_face_adjoint!`).
#   * Per-direction sweep adjoint kernels (`_cs_xsweep_adjoint_kernel!`,
#     `_cs_ysweep_adjoint_kernel!`, `_cs_zsweep_adjoint_kernel!`).
#   * The driver `_adjoint_scheme_sweep!` for both linear and limited
#     PPM schemes and the FV3 vertical profile (whose column kernel lives in
#     `Operators/Advection/vertical_fv3_profile_adjoint.jl`); the LinRood
#     horizontal record is handled out-of-line in `LinRoodTape.jl`.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kernelized adjoint building blocks
# ---------------------------------------------------------------------------

@inline function _upwind_face_coeffs(F, m_l, m_r)
    FT = typeof(F)
    if F >= zero(FT)
        return clamp(F / max(m_l, _advection_mass_floor(FT)), zero(FT), one(FT)), zero(FT)
    else
        return zero(FT), clamp(F / max(m_r, _advection_mass_floor(FT)), -one(FT), zero(FT))
    end
end

@inline function _slopes_no_limiter_face_coeffs(F, m_ll, m_l, m_r, m_rr)
    FT = typeof(F)
    m_floor = _advection_mass_floor(FT)
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
    tw12 = last(_ppm_edge_weights(FT))
    m_floor = _advection_mass_floor(FT)
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

# ---------------------------------------------------------------------------
# Limited PPM (monotone, CW84): the limiter branches on the tape
#
# For fixed limiter branches the face flux of a limited PPM sweep is linear in
# the tracer mass, and its derivative depends on the tracer mass only through
# the branches (theory/adjoint_status.md, "Limiter branches on the tape"). So
# the tape keeps one byte per cell and sweep, the branch code, instead of the
# tracer mass entering the sweep, and the reverse applies the derivative of
# the recorded branch:
#
#   bits 0-1  the profile branch of the cell (`_ppm_profile_branch`):
#             0 extremum (flattened), 1 left edge reset, 2 right edge reset,
#             3 unchanged;
#   bits 2-3  the minmod branch of the cell's van Leer slope in the CW84 edges
#             (`_limited_slope_branch`): 0 zero, 1 centred, 2 forward,
#             3 backward (always 0 for the monotone limiter).
#
# The codes are computed when the tape is recorded, by `PPMLimiterCodes`,
# with the predicates and the values the reverse used to evaluate on the
# staged tracer mass: the mixing ratios `rm / max(m, floor)`, the edges of
# `_ppm_edge_value` (monotone) or CW84 eq. 1.7 with the minmod slopes below,
# and the stencil of each cell clamped to the axis as the reverse clamps it
# (y at the padded panel edges, z at the column ends; x reads only cells whose
# stencil lies inside the halo). The minmod is zero whenever two of its three
# cells are equal, so a clamped stencil gives the same code at every face that
# reads the cell. The reverse then needs no tracer mass: `_ppm_face_coeffs`
# applies the derivatives below with the operation order of the forward
# linearization, bit for bit.
# ---------------------------------------------------------------------------

# The minmod of the van Leer slope (`_limited_slope(…, MonotoneLimiter())`)
# and its branch: 0 zero (the three one-sided slopes do not share a sign),
# 1 centred, 2 forward, 3 backward.
@inline function _limited_slope_branch(c_m, c_0, c_p)
    FT = typeof(c_0)
    sc = (c_p - c_m) / 2
    sp = 2 * (c_p - c_0)
    sm = 2 * (c_0 - c_m)
    same_sign = ((sc > zero(FT)) & (sp > zero(FT)) & (sm > zero(FT))) |
                ((sc < zero(FT)) & (sp < zero(FT)) & (sm < zero(FT)))
    same_sign || return zero(FT), 0x00
    abs(sc) <= abs(sp) && abs(sc) <= abs(sm) && return sc, 0x01
    abs(sp) <= abs(sm) && return sp, 0x02
    return sm, 0x03
end

# Edge value between `c_l` and `c_r` of a limited PPM sweep (the forward's
# `_ppm_edge_value(…, limiter)`): the fourth-order edge for the monotone
# limiter, CW84 eq. 1.7 with van Leer-limited slopes for CW84.
@inline _limited_ppm_edge_value(c_ll, c_l, c_r, c_rr, ::MonotoneLimiter) =
    _ppm_edge_value(c_ll, c_l, c_r, c_rr, typeof(c_ll))

@inline function _limited_ppm_edge_value(c_ll, c_l, c_r, c_rr, ::CW84Limiter)
    FT = typeof(c_l)
    δc_l, _ = _limited_slope_branch(c_ll, c_l, c_r)
    δc_r, _ = _limited_slope_branch(c_l, c_r, c_rr)
    return (c_l + c_r) / FT(2) - (δc_r - δc_l) / FT(6)
end

# The branch of the CW84 monotonicity constraints (`_ppm_limit_profile`) for a
# cell with mean `c_bar` and edges `q_L`, `q_R`: 0 extremum, 1 left edge reset,
# 2 right edge reset, 3 unchanged. Both resets cannot apply: that would need
# dc² < dc·c6 < −dc².
@inline function _ppm_profile_branch(q_L, c_bar, q_R)
    FT = typeof(c_bar)
    is_extremum = (q_R - c_bar) * (c_bar - q_L) <= zero(FT)
    dc = q_R - q_L
    c6 = FT(6) * (c_bar - (q_L + q_R) / FT(2))
    needs_left_fix = dc * c6 > dc * dc
    needs_right_fix = -(dc * dc) > dc * c6
    is_extremum && return 0x00
    needs_left_fix && return 0x01
    needs_right_fix && return 0x02
    return 0x03
end

@inline _profile_code(code::UInt8) = code & 0x03
@inline _slope_code(code::UInt8) = (code >> 0x02) & 0x03

# The minmod branch stored with a cell: the CW84 edges read it, the monotone
# edges do not.
@inline _slope_branch_code(_c_m, _c_0, _c_p, ::MonotoneLimiter) = 0x00
@inline _slope_branch_code(c_m, c_0, c_p, ::CW84Limiter) = last(_limited_slope_branch(c_m, c_0, c_p))

# Cell `(i, j, k)` moved by `o` along axis `D` of an axis of length `n`,
# clamped to it.
@inline _axis_cell(::Val{1}, i, j, k, o, n) = (clamp(i + o, 1, n), j, k)
@inline _axis_cell(::Val{2}, i, j, k, o, n) = (i, clamp(j + o, 1, n), k)
@inline _axis_cell(::Val{3}, i, j, k, o, n) = (i, j, clamp(k + o, 1, n))

Base.@propagate_inbounds function _mixing_ratio_at(rm, m, axis, i, j, k, o, n)
    I = _axis_cell(axis, i, j, k, o, n)
    return rm[I...] / max(m[I...], _advection_mass_floor(eltype(m)))
end

"""
    _ppm_limiter_code(m, rm, i, j, k, Val(D), limiter) -> UInt8

The limiter branch code of cell `(i, j, k)` of a limited PPM sweep along axis
`D` (1 x, 2 y, 3 z) with air mass `m` and tracer mass `rm` entering the sweep:
the profile branch of the cell (bits 0-1) and, for CW84, the minmod branch of
its edge slope (bits 2-3). The stencil `c_{-2} … c_{+2}` is clamped to the
axis.
"""
Base.@propagate_inbounds function _ppm_limiter_code(m, rm, i, j, k, axis::Val{D}, limiter) where D
    n = size(m, D)
    c_mm = _mixing_ratio_at(rm, m, axis, i, j, k, -2, n)
    c_m = _mixing_ratio_at(rm, m, axis, i, j, k, -1, n)
    c_0 = _mixing_ratio_at(rm, m, axis, i, j, k, 0, n)
    c_p = _mixing_ratio_at(rm, m, axis, i, j, k, 1, n)
    c_pp = _mixing_ratio_at(rm, m, axis, i, j, k, 2, n)
    q_L = _limited_ppm_edge_value(c_mm, c_m, c_0, c_p, limiter)
    q_R = _limited_ppm_edge_value(c_m, c_0, c_p, c_pp, limiter)
    return _ppm_profile_branch(q_L, c_0, q_R) | (_slope_branch_code(c_m, c_0, c_p, limiter) << 0x02)
end

"""
    PPMLimiterCodes{D}(limiter) <: AbstractPointOp

The limiter branch codes of a limited PPM sweep along axis `D` for all six
panels in one launch: at `(i, j, k, p)`, `ctx.codes[p][i, j, k] =
_ppm_limiter_code(ctx.m[p], ctx.rm[p], i, j, k, Val(D), limiter)`, over every
cell of the halo-padded panels.
"""
struct PPMLimiterCodes{D, L} <: AbstractPointOp
    limiter::L
end
PPMLimiterCodes{D}(limiter::L) where {D, L} = PPMLimiterCodes{D, L}(limiter)

index_space(::PPMLimiterCodes, ctx) = (size(ctx.m[1])..., 6)

@inline function apply_point!(op::PPMLimiterCodes{D}, ctx, i, j, k, p) where D
    m = select_panel(ctx.m, p)
    rm = select_panel(ctx.rm, p)
    codes = select_panel(ctx.codes, p)
    @inbounds codes[i, j, k] = _ppm_limiter_code(m, rm, i, j, k, Val(D), op.limiter)
    return nothing
end

"""
    record_ppm_limiter_codes!(codes, panels_m, panels_rm, direction, scheme) -> codes

Write the limiter branch codes of a limited PPM sweep (`:x`, `:y` or `:z`)
entering with `panels_m`, `panels_rm` into `codes` (six `UInt8` panels of the
air mass's shape): one launch for the six panels, queued without a host
synchronization. The tape recorder stages them in place of the tracer mass.
"""
function record_ppm_limiter_codes!(codes::NTuple{6}, panels_m::NTuple{6}, panels_rm::NTuple{6},
                                   direction::Symbol, scheme::CSAdjointNonlinearScheme)
    for p in 1:6
        size(codes[p]) == size(panels_m[p]) == size(panels_rm[p]) || throw(DimensionMismatch(
            "limiter codes, air mass and tracer mass of panel $p have shapes " *
            "$(size(codes[p])), $(size(panels_m[p])) and $(size(panels_rm[p]))"))
    end
    op = direction === :x ? PPMLimiterCodes{1}(scheme.limiter) :
         direction === :y ? PPMLimiterCodes{2}(scheme.limiter) :
         direction === :z ? PPMLimiterCodes{3}(scheme.limiter) :
         throw(ArgumentError("unknown CS sweep direction $(repr(direction))"))
    launch!(op, (; codes, m = panels_m, rm = panels_rm), get_backend(panels_m[1]); sync = false)
    return codes
end

# Derivatives (`d6` tuples over the six stencil cells) of a limited PPM face
# flux at recorded branches. Each mirrors the derivative part of the forward
# linearization in operation order.

# Edge between cells `l` and `r`: the fourth-order weights (monotone), or CW84
# eq. 1.7 with the minmod slopes of cells `l` and `r` (codes `code_l`,
# `code_r`).
@inline function _limited_ppm_edge_deriv(d_ll, d_l, d_r, d_rr, _code_l, _code_r, ::MonotoneLimiter)
    seven_twelfths, one_twelfth = _ppm_edge_weights(eltype(d_l))    # forward `_ppm_edge_value` weights
    return _d6_sub(_d6_scale(_d6_add(d_l, d_r), seven_twelfths),
                   _d6_scale(_d6_add(d_ll, d_rr), one_twelfth))
end

@inline function _limited_ppm_edge_deriv(d_ll, d_l, d_r, d_rr, code_l, code_r, ::CW84Limiter)
    FT = eltype(d_l)
    dδc_l = _limited_slope_deriv(_slope_code(code_l), d_ll, d_l, d_r)
    dδc_r = _limited_slope_deriv(_slope_code(code_r), d_l, d_r, d_rr)
    return _d6_sub(_d6_scale(_d6_add(d_l, d_r), FT(1) / FT(2)),
                   _d6_scale(_d6_sub(dδc_r, dδc_l), FT(1) / FT(6)))
end

# Minmod slope of branch `code` (`_limited_slope_branch`).
@inline function _limited_slope_deriv(code::UInt8, d_m, d_0, d_p)
    FT = eltype(d_0)
    code == 0x01 && return _d6_scale(_d6_sub(d_p, d_m), FT(1) / FT(2))
    code == 0x02 && return _d6_scale(_d6_sub(d_p, d_0), FT(2))
    code == 0x03 && return _d6_scale(_d6_sub(d_0, d_m), FT(2))
    return _d6_zero(FT)
end

# Edges of a cell after the monotonicity constraints of branch `code`
# (`_ppm_profile_branch`).
@inline function _ppm_profile_deriv(code::UInt8, dq_L, dc_bar, dq_R)
    FT = eltype(dc_bar)
    code == 0x00 && return dc_bar, dc_bar
    code == 0x01 && return _d6_sub(_d6_scale(dc_bar, FT(3)), _d6_scale(dq_R, FT(2))), dq_R
    code == 0x02 && return dq_L, _d6_sub(_d6_scale(dc_bar, FT(3)), _d6_scale(dq_L, FT(2)))
    return dq_L, dq_R
end

# Curvature part of the PPM moment (`_ppm_raw_moments`): none for the monotone
# limiter, α·b₀ with b₀ = (q_L − c) + (q_R − c) for CW84.
@inline _ppm_curvature_deriv(_α, _dq_L, dc, _dq_R, ::MonotoneLimiter) = _d6_zero(eltype(dc))

@inline function _ppm_curvature_deriv(α, dq_L, dc, dq_R, ::CW84Limiter)
    FT = eltype(dc)
    db0 = _d6_sub(_d6_add(dq_L, dq_R), _d6_scale(dc, FT(2)))
    return _d6_scale(db0, α)
end

"""
    _ppm_face_coeffs(F, m_3, …, m_pp, code_2, code_1, code_0, code_p,
                     interior_l, interior_r, limiter) → NTuple{6}

Derivatives of the structured PPM face flux (`_xface_tracer_flux` and its y/z
twins) with respect to the tracer masses of the six stencil cells `3 2 1 | 0 p
pp` (the face between cells `1` and `0`), for the monotone and CW84 limiters,
at the limiter branches the tape recorded for cells `2 1 0 p`
(`_ppm_limiter_code`): the profile branches of cells `1` and `0`, and for
CW84 the minmod branches of the edge slopes of cells `2 1 0 p`. The flow
direction is that of `F`.
"""
@inline function _ppm_face_coeffs(F,
                                  m_3, m_2, m_1, m_0, m_p, m_pp,
                                  code_2::UInt8, code_1::UInt8, code_0::UInt8, code_p::UInt8,
                                  interior_l::Bool, interior_r::Bool, limiter)
    FT = typeof(F)
    m_floor = _advection_mass_floor(FT)
    m3 = max(m_3, m_floor)
    m2 = max(m_2, m_floor)
    m1 = max(m_1, m_floor)
    m0 = max(m_0, m_floor)
    mp = max(m_p, m_floor)
    mpp = max(m_pp, m_floor)

    dc_3 = _d6_basis(FT, 1, inv(m3))
    dc_2 = _d6_basis(FT, 2, inv(m2))
    dc_1 = _d6_basis(FT, 3, inv(m1))
    dc_0 = _d6_basis(FT, 4, inv(m0))
    dc_p = _d6_basis(FT, 5, inv(mp))
    dc_pp = _d6_basis(FT, 6, inv(mpp))

    de_left = _limited_ppm_edge_deriv(dc_3, dc_2, dc_1, dc_0, code_2, code_1, limiter)
    de_face = _limited_ppm_edge_deriv(dc_2, dc_1, dc_0, dc_p, code_1, code_0, limiter)
    de_right = _limited_ppm_edge_deriv(dc_1, dc_0, dc_p, dc_pp, code_0, code_p, limiter)

    dqLl, dqRl = _ppm_profile_deriv(_profile_code(code_1), de_left, dc_1, de_face)
    dqLr, dqRr = _ppm_profile_deriv(_profile_code(code_0), de_face, dc_0, de_right)

    α_pos = clamp(F / m1, zero(FT), one(FT))
    α_neg = clamp(F / m0, -one(FT), zero(FT))

    dκ_l = _ppm_curvature_deriv(α_pos, dqLl, dc_1, dqRl, limiter)
    dsx_l = interior_l ? _d6_scale(_d6_sub(_d6_sub(dqRl, dc_1), dκ_l), m1) : _d6_zero(FT)

    dκ_r = _ppm_curvature_deriv(α_neg, dqLr, dc_0, dqRr, limiter)
    dsx_r = interior_r ? _d6_scale(_d6_sub(_d6_sub(dc_0, dqLr), dκ_r), m0) : _d6_zero(FT)

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

Base.@propagate_inbounds function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
                                      ::UpwindScheme, Nx)
    i_l = _wrap_periodic(face_i - Int32(1), Nx)
    i_r = _wrap_periodic(face_i, Nx)
    c_l, c_r = _upwind_face_coeffs(F, m[i_l, j, k], m[i_r, j, k])
    @atomic lambda_in[i_l, j, k] += scale * c_l
    @atomic lambda_in[i_r, j, k] += scale * c_r
    return nothing
end

Base.@propagate_inbounds function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
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

Base.@propagate_inbounds function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
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

# Limited PPM: `codes` are the sweep's limiter branch codes (`PPMLimiterCodes`).
Base.@propagate_inbounds function _add_x_face_adjoint!(lambda_in, m, codes, face_i, j, k, F, scale,
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
        codes[i_2, j, k], codes[i_1, j, k], codes[i_0, j, k], codes[i_p, j, k],
        true, true, scheme.limiter)
    @atomic lambda_in[i_3,  j, k] += scale * c[1]
    @atomic lambda_in[i_2,  j, k] += scale * c[2]
    @atomic lambda_in[i_1,  j, k] += scale * c[3]
    @atomic lambda_in[i_0,  j, k] += scale * c[4]
    @atomic lambda_in[i_p,  j, k] += scale * c[5]
    @atomic lambda_in[i_pp, j, k] += scale * c[6]
    return nothing
end

Base.@propagate_inbounds function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
                                      ::UpwindScheme, Ny)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    jl = max(face_j - Int32(1), Int32(1))
    jr = min(face_j, Ny)
    c_l, c_r = _upwind_face_coeffs(F, m[i, jl, k], m[i, jr, k])
    @atomic lambda_in[i, jl, k] += scale * c_l
    @atomic lambda_in[i, jr, k] += scale * c_r
    return nothing
end

Base.@propagate_inbounds function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
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

Base.@propagate_inbounds function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
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
            c_l = clamp(F / max(m[i, jl, k], _advection_mass_floor(FT)), zero(FT), one(FT))
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
            c_r = clamp(F / max(m[i, jr, k], _advection_mass_floor(FT)), -one(FT), zero(FT))
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, jll, k] += scale * c_ll
    @atomic lambda_in[i, jl,  k] += scale * c_l
    @atomic lambda_in[i, jr,  k] += scale * c_r
    @atomic lambda_in[i, jrr, k] += scale * c_rr
    return nothing
end

Base.@propagate_inbounds function _add_y_face_adjoint!(lambda_in, m, codes, i, face_j, k, F, scale,
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
        codes[i, jll, k], codes[i, jl, k], codes[i, jr, k], codes[i, jrr, k],
        interior_l, interior_r, scheme.limiter)
    @atomic lambda_in[i, j3l, k] += scale * c[1]
    @atomic lambda_in[i, jll, k] += scale * c[2]
    @atomic lambda_in[i, jl,  k] += scale * c[3]
    @atomic lambda_in[i, jr,  k] += scale * c[4]
    @atomic lambda_in[i, jrr, k] += scale * c[5]
    @atomic lambda_in[i, j3r, k] += scale * c[6]
    return nothing
end

Base.@propagate_inbounds function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
                                      ::UpwindScheme, Nz)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    kl = max(face_k - Int32(1), Int32(1))
    kr = min(face_k, Nz)
    c_l, c_r = _upwind_face_coeffs(F, m[i, j, kl], m[i, j, kr])
    @atomic lambda_in[i, j, kl] += scale * c_l
    @atomic lambda_in[i, j, kr] += scale * c_r
    return nothing
end

Base.@propagate_inbounds function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
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

Base.@propagate_inbounds function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
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
            c_l = clamp(F / max(m[i, j, kl], _advection_mass_floor(FT)), zero(FT), one(FT))
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
            c_r = clamp(F / max(m[i, j, kr], _advection_mass_floor(FT)), -one(FT), zero(FT))
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, j, kll] += scale * c_ll
    @atomic lambda_in[i, j, kl]  += scale * c_l
    @atomic lambda_in[i, j, kr]  += scale * c_r
    @atomic lambda_in[i, j, krr] += scale * c_rr
    return nothing
end

Base.@propagate_inbounds function _add_z_face_adjoint!(lambda_in, m, codes, i, j, face_k, F, scale,
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
        codes[i, j, kll], codes[i, j, kl], codes[i, j, kr], codes[i, j, krr],
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
        _add_x_face_adjoint!(lambda_in, m, Int32(i),     j, k, flux_scale * am[i,     j, k],  bar, scheme, _HaloPaddedAxis())
        _add_x_face_adjoint!(lambda_in, m, Int32(i) + 1, j, k, flux_scale * am[i + 1, j, k], -bar, scheme, _HaloPaddedAxis())
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
                                            @Const(m), @Const(codes), @Const(am),
                                            scheme::PPMScheme{<:_PPMTapedLimiter},
                                            Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        _add_x_face_adjoint!(lambda_in, m, codes, Int32(i),     j, k, flux_scale * am[i,     j, k],  bar, scheme, _HaloPaddedAxis())
        _add_x_face_adjoint!(lambda_in, m, codes, Int32(i) + 1, j, k, flux_scale * am[i + 1, j, k], -bar, scheme, _HaloPaddedAxis())
    end
end

@kernel function _cs_ysweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(codes), @Const(bm),
                                            scheme::PPMScheme{<:_PPMTapedLimiter},
                                            Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Ny = Int32(Nc + 2 * Hp)
        _add_y_face_adjoint!(lambda_in, m, codes, i, Int32(j),     k, flux_scale * bm[i, j,     k],  bar, scheme, Ny)
        _add_y_face_adjoint!(lambda_in, m, codes, i, Int32(j) + 1, k, flux_scale * bm[i, j + 1, k], -bar, scheme, Ny)
    end
end

@kernel function _cs_zsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(codes), @Const(cm),
                                            scheme::CSAdjointNonlinearScheme,
                                            Nc, Hp, Nz, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        _add_z_face_adjoint!(lambda_in, m, codes, i, j, Int32(k),     flux_scale * cm[i, j, k],     bar, scheme, Int32(Nz))
        _add_z_face_adjoint!(lambda_in, m, codes, i, j, Int32(k) + 1, flux_scale * cm[i, j, k + 1], -bar, scheme, Int32(Nz))
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
    # The face adjoints read the halo without bounds checks or a periodic wrap.
    horizontal && _validate_halo_for_scheme(scheme, Hp)
    horizontal && _cache_cs_seam_seeds!(ws.seam_seeds, lambda_panels, mesh, axis)
    # Every panel reuses `ws.lambda_A`: its zeroing, scatter and copy-back run
    # in queue order on a device, so no host synchronization is needed.
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
        copyto!(lambda_panels[p], ws.lambda_A)
    end
    horizontal && _apply_cs_seam_adjoint!(lambda_panels, m_before, nothing,
        flux_panels, ws.seam_seeds, mesh, scheme, axis, flux_scale)
    return nothing
end

# Split-sweep PPM adjoints cover the default vertical reconstruction only; the
# FV3 vertical profile has its own reverse (below), used by Lin–Rood. A limited
# PPM sweep's reverse reads the air mass entering the sweep and the limiter
# branch codes the tape recorded for it (`record_ppm_limiter_codes!`), six
# `UInt8` panels of the air mass's shape.
function _adjoint_scheme_sweep!(lambda_panels, m_before, codes, flux_panels,
                                direction::Symbol, scheme::CSAdjointNonlinearScheme,
                                mesh::CubedSphereMesh, ws::CSAdjointWorkspace,
                                flux_scale)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(lambda_panels[1], 3)
    horizontal = direction === :x || direction === :y
    axis = direction === :x ? Val(1) : Val(2)
    # The face adjoints read the halo without bounds checks or a periodic wrap.
    horizontal && _validate_halo_for_scheme(scheme, Hp)
    horizontal && _cache_cs_seam_seeds!(ws.seam_seeds, lambda_panels, mesh, axis)
    @inbounds for p in 1:6
        fill!(ws.lambda_A, zero(eltype(ws.lambda_A)))
        backend = get_backend(lambda_panels[p])
        panel_flux = horizontal ? _CSInteriorFlux(flux_panels[p], mesh, axis) : flux_panels[p]
        if direction === :x
            kernel! = _cs_xsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], codes[p],
                    panel_flux, scheme, Int32(Nc), Int32(Hp),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :y
            kernel! = _cs_ysweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], codes[p],
                    panel_flux, scheme, Int32(Nc), Int32(Hp),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :z
            kernel! = _cs_zsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], codes[p],
                    panel_flux, scheme, Int32(Nc), Int32(Hp), Int32(Nz),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        else
            throw(ArgumentError("unknown CS adjoint sweep direction $direction"))
        end
        copyto!(lambda_panels[p], ws.lambda_A)
    end
    horizontal && _apply_cs_seam_adjoint!(lambda_panels, m_before, codes,
        flux_panels, ws.seam_seeds, mesh, scheme, axis, flux_scale)
    return nothing
end

# Reverse of an FV3-profile vertical sweep. Only Lin–Rood's vertical sweeps
# record this scheme (the production Lin–Rood palindrome's `record_sweep!`
# hooks), so the record is always a Z sweep.
# The transpose is column-local and in place on `lambda_panels`
# (`_sweep_z_panels_fv3_adjoint!`); `m_before` and `rm_before` are the state
# that entered the forward sweep.
function _adjoint_scheme_sweep!(lambda_panels, m_before, rm_before, flux_panels,
                                direction::Symbol, scheme::CSAdjointFV3Vertical,
                                mesh::CubedSphereMesh, ws::CSAdjointWorkspace,
                                flux_scale)
    direction === :z || throw(ArgumentError(
        "an FV3-profile adjoint record must be a vertical (:z) sweep, got $(repr(direction))"))
    _sweep_z_panels_fv3_adjoint!(lambda_panels, rm_before, m_before, flux_panels,
                                 scheme.vertical, ws.column_scratch, mesh;
                                 flux_scale)
    return nothing
end
