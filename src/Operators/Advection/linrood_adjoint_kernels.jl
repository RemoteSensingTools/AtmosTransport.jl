# ---------------------------------------------------------------------------
# Lin-Rood adjoint kernels.
#
# Reverse-mode kernels for the LinRood cubed-sphere PPM path. Each kernel
# here is the discrete transpose of the matching forward kernel in
# `LinRood.jl` for fixed velocities (`am`, `bm`) — the tracer state and
# face mixing ratios are the differentiated inputs, the velocity is
# treated as a parameter from the meteo tape.
#
# The naming convention is `<forward_kernel>_adjoint!`. Each adjoint
# kernel takes adjoint arrays sized identically to the forward output
# (read-only) plus adjoint accumulators for each forward input
# (atomically incremented). The forward velocity fields are passed
# through unchanged.
#
# Reference:
#   docs/src/theory/adjoint_status.md       — shipped adjoint surface
# ---------------------------------------------------------------------------

# LinRood.jl imports `@kernel, @index, @Const, synchronize, get_backend`
# but not `@atomic` — adjoint face accumulators need it, so we pull it
# in alongside.
using KernelAbstractions: @atomic

# ---------------------------------------------------------------------------
# Adjoint of `_linrood_update_kernel!` (LinRood.jl)
#
# Forward (mass-space):
#   rm_new[ii, jj, k] = rm[ii, jj, k]
#                     + am_w * (fx_in[i,   j, k] + fx_out[i,   j, k]) / 2
#                     - am_e * (fx_in[i+1, j, k] + fx_out[i+1, j, k]) / 2
#                     + bm_s * (fy_in[i, j,   k] + fy_out[i, j,   k]) / 2
#                     - bm_n * (fy_in[i, j+1, k] + fy_out[i, j+1, k]) / 2
#   m_new [ii, jj, k] = m [ii, jj, k] + (am_w - am_e) + (bm_s - bm_n)
#
# For fixed velocities `(am, bm)`, the forward map
#   F :  (rm, m, fx_in, fx_out, fy_in, fy_out) -> (rm_new, m_new)
# is linear. Its transpose, applied to (lambda_rm_new, lambda_m_new),
# accumulates atomically into the six adjoint inputs.
#
# Coefficients (with `half = 0.5`):
#   ∂rm_new/∂rm[ii,jj,k]           = 1
#   ∂rm_new/∂fx_in [i,   j, k]     = +half · am_w
#   ∂rm_new/∂fx_out[i,   j, k]     = +half · am_w
#   ∂rm_new/∂fx_in [i+1, j, k]     = -half · am_e
#   ∂rm_new/∂fx_out[i+1, j, k]     = -half · am_e
#   ∂rm_new/∂fy_in [i, j,   k]     = +half · bm_s
#   ∂rm_new/∂fy_out[i, j,   k]     = +half · bm_s
#   ∂rm_new/∂fy_in [i, j+1, k]     = -half · bm_n
#   ∂rm_new/∂fy_out[i, j+1, k]     = -half · bm_n
#   ∂m_new /∂m [ii,jj,k]           = 1
# All other partials are zero. `m_new` has no tracer dependence; `rm_new`
# has no air-mass dependence. Face writes share neighbouring cells, so
# face accumulations use `@atomic`.
# ---------------------------------------------------------------------------

@kernel function _linrood_update_kernel_adjoint!(
    lambda_rm, lambda_m,
    lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
    @Const(lambda_rm_new), @Const(lambda_m_new),
    @Const(am), @Const(bm), Hp,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(lambda_rm_new)
        half = FT(0.5)

        bar_rm = lambda_rm_new[ii, jj, k]
        bar_m  = lambda_m_new[ii, jj, k]

        # rm[ii,jj,k] receives bar_rm; m[ii,jj,k] receives bar_m.
        # Each interior cell is touched by exactly one thread, so no
        # atomic needed for the cell-centre accumulations.
        lambda_rm[ii, jj, k] += bar_rm
        lambda_m[ii, jj, k]  += bar_m

        am_w = am[i, j, k]
        am_e = am[i + 1, j, k]
        bm_s = bm[i, j, k]
        bm_n = bm[i, j + 1, k]

        wx_w =  half * am_w * bar_rm
        wx_e = -half * am_e * bar_rm
        wy_s =  half * bm_s * bar_rm
        wy_n = -half * bm_n * bar_rm

        # Face writes: each face index can be touched by two interior
        # cells (left/right or below/above), hence the atomic adds.
        @atomic lambda_fx_in[i, j, k]      += wx_w
        @atomic lambda_fx_out[i, j, k]     += wx_w
        @atomic lambda_fx_in[i + 1, j, k]  += wx_e
        @atomic lambda_fx_out[i + 1, j, k] += wx_e
        @atomic lambda_fy_in[i, j, k]      += wy_s
        @atomic lambda_fy_out[i, j, k]     += wy_s
        @atomic lambda_fy_in[i, j + 1, k]  += wy_n
        @atomic lambda_fy_out[i, j + 1, k] += wy_n
    end
end

"""
    apply_linrood_update_adjoint!(lambda_rm, lambda_m,
                                   lambda_fx_in, lambda_fx_out,
                                   lambda_fy_in, lambda_fy_out,
                                   lambda_rm_new, lambda_m_new,
                                   am, bm, mesh)

Apply the discrete transpose of `_linrood_update_kernel!` for one panel:
accumulate the adjoint of `(rm_new, m_new)` into the adjoint inputs
`(rm, m, fx_in, fx_out, fy_in, fy_out)` for fixed velocities `(am, bm)`.

All `lambda_*` adjoint accumulators are read-and-modified (atomically for
face arrays) — callers are responsible for initialising them to zero
before the call. The kernel only touches interior `(i, j)` indices
`1..Nc`; halo cells of `lambda_rm`/`lambda_m` and face cells outside
`1..Nc+1` are left untouched.
"""
function apply_linrood_update_adjoint!(lambda_rm, lambda_m,
                                       lambda_fx_in, lambda_fx_out,
                                       lambda_fy_in, lambda_fy_out,
                                       lambda_rm_new, lambda_m_new,
                                       am, bm,
                                       mesh::CubedSphereMesh)
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_rm_new, 3)
    backend = get_backend(lambda_rm)
    k! = _linrood_update_kernel_adjoint!(backend, 256)
    k!(lambda_rm, lambda_m,
       lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
       lambda_rm_new, lambda_m_new, am, bm, Hp;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Adjoint of `_pre_advect_y_kernel!` (LinRood.jl:364).
#
# Forward:
#   bm_s   = bm[i, j, k]
#   bm_n   = bm[i, j+1, k]
#   rm_new = rm[ii, jj, k] + bm_s · fy_face[i, j, k]
#                          - bm_n · fy_face[i, j+1, k]
#   m_new  = m [ii, jj, k] + bm_s - bm_n
#   q_i[ii, jj, k] = m_new > thresh ? rm_new / m_new : 0
#
# where `thresh = 100 · eps(FT)` mirrors `_safe_mixing_ratio`. For
# `m_new > thresh` the operator is smooth in `(rm, m, fy_face)`; below
# threshold the output is exactly zero, so all adjoint contributions are
# zero. The adjoint then is:
#
#   inv_m_new = 1 / m_new      (when m_new > thresh, else 0)
#   lambda_rm     [ii, jj, k]   += lambda_q_i · inv_m_new
#   lambda_fy_face[i, j, k]     += lambda_q_i · bm_s · inv_m_new
#   lambda_fy_face[i, j+1, k]   += lambda_q_i · (-bm_n) · inv_m_new
#   lambda_m      [ii, jj, k]   += lambda_q_i · (-q_i) · inv_m_new
#
# Face writes share neighbour cells along the Y direction, hence the
# `@atomic` accumulation on `lambda_fy_face`. Cell-centred writes are
# unique per thread.
# ---------------------------------------------------------------------------

@kernel function _pre_advect_y_kernel_adjoint!(
    lambda_rm, lambda_m, lambda_fy_face,
    @Const(lambda_q_i), @Const(rm), @Const(m), @Const(bm), @Const(fy_face), Hp,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(lambda_q_i)
        thresh = FT(100) * eps(FT)

        bm_s = bm[i, j,     k]
        bm_n = bm[i, j + 1, k]
        m_new = m[ii, jj, k] + bm_s - bm_n

        if m_new > thresh
            rm_new = rm[ii, jj, k] +
                     bm_s * fy_face[i, j, k] - bm_n * fy_face[i, j + 1, k]
            inv_m_new = one(FT) / m_new
            q_i = rm_new * inv_m_new

            bar = lambda_q_i[ii, jj, k]
            scaled = bar * inv_m_new

            lambda_rm[ii, jj, k] += scaled
            lambda_m[ii, jj, k]  += -q_i * scaled

            @atomic lambda_fy_face[i, j,     k] +=  bm_s * scaled
            @atomic lambda_fy_face[i, j + 1, k] += -bm_n * scaled
        end
        # m_new <= thresh: q_i = 0 deterministically, so gradient = 0.
    end
end

"""
    apply_pre_advect_y_adjoint!(lambda_rm, lambda_m, lambda_fy_face,
                                  lambda_q_i, rm, m, bm, fy_face, mesh)

Discrete transpose of `_pre_advect_y_kernel!` for one panel: accumulate
`lambda_q_i` into the adjoint accumulators of `(rm, m, fy_face)` for
fixed velocity `bm`. The small-`m_new` zeroing exactly mirrors
`_safe_mixing_ratio` (LinRood-style 100·eps threshold).

All `lambda_*` accumulators are read-modified; callers initialise them
to zero before the call. Face writes use `@atomic` for shared
neighbour-cell accumulation along Y.
"""
function apply_pre_advect_y_adjoint!(lambda_rm, lambda_m, lambda_fy_face,
                                     lambda_q_i, rm, m, bm, fy_face,
                                     mesh::CubedSphereMesh)
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_q_i, 3)
    backend = get_backend(lambda_rm)
    k! = _pre_advect_y_kernel_adjoint!(backend, 256)
    k!(lambda_rm, lambda_m, lambda_fy_face,
       lambda_q_i, rm, m, bm, fy_face, Hp;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Adjoint of `_pre_advect_x_kernel!` (LinRood.jl:377).
#
# Identical structure to `_pre_advect_y_kernel!` with the directions
# transposed: `am`/`fx_face` replace `bm`/`fy_face`, and the face
# neighbour is `(i+1, j, k)` rather than `(i, j+1, k)`.
# ---------------------------------------------------------------------------

@kernel function _pre_advect_x_kernel_adjoint!(
    lambda_rm, lambda_m, lambda_fx_face,
    @Const(lambda_q_j), @Const(rm), @Const(m), @Const(am), @Const(fx_face), Hp,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(lambda_q_j)
        thresh = FT(100) * eps(FT)

        am_w = am[i,     j, k]
        am_e = am[i + 1, j, k]
        m_new = m[ii, jj, k] + am_w - am_e

        if m_new > thresh
            rm_new = rm[ii, jj, k] +
                     am_w * fx_face[i, j, k] - am_e * fx_face[i + 1, j, k]
            inv_m_new = one(FT) / m_new
            q_j = rm_new * inv_m_new

            bar = lambda_q_j[ii, jj, k]
            scaled = bar * inv_m_new

            lambda_rm[ii, jj, k] += scaled
            lambda_m[ii, jj, k]  += -q_j * scaled

            @atomic lambda_fx_face[i,     j, k] +=  am_w * scaled
            @atomic lambda_fx_face[i + 1, j, k] += -am_e * scaled
        end
    end
end

"""
    apply_pre_advect_x_adjoint!(lambda_rm, lambda_m, lambda_fx_face,
                                  lambda_q_j, rm, m, am, fx_face, mesh)

Discrete transpose of `_pre_advect_x_kernel!` for one panel. See
`apply_pre_advect_y_adjoint!` for the contract — same structure with X
substituted for Y.
"""
function apply_pre_advect_x_adjoint!(lambda_rm, lambda_m, lambda_fx_face,
                                     lambda_q_j, rm, m, am, fx_face,
                                     mesh::CubedSphereMesh)
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_q_j, 3)
    backend = get_backend(lambda_rm)
    k! = _pre_advect_x_kernel_adjoint!(backend, 256)
    k!(lambda_rm, lambda_m, lambda_fx_face,
       lambda_q_j, rm, m, am, fx_face, Hp;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ===========================================================================
# Adjoints of the two `_from_q` PPM face kernels (ORD=5)
#
# The forward kernels `_ppm_x_face_from_q_kernel!` and
# `_ppm_y_face_from_q_kernel!` (LinRood.jl:299, 325) compute a PPM
# parabolic-integral face value from a 6-cell q stencil at fixed
# velocity (am/bm). Both share the chain
#   _ppm_edge_values → _apply_monotonicity → _ppm_face_value
# where each step is piecewise-smooth in q. The composition is rational
# and branch-rich, so we differentiate via a small 6-cell forward-AD
# wrapper `D6{FT}` and propagate `(value, ∂/∂q_n)` pairs through the
# chain. The resulting 6-tuple of face partials is then multiplied by
# the adjoint seed `lambda_face` and atomically accumulated into
# `lambda_q` at the six stencil cells.
# ===========================================================================

# D6 value-tangent pair: `value` plus a 6-component gradient w.r.t. the
# six PPM stencil cells. Immutable + tuple-of-FT, so it works
# unchanged on CPU and CUDA backends.
struct D6{FT}
    v :: FT
    g :: NTuple{6, FT}
end

@inline _d6_const(::Type{FT}, x::FT) where {FT} =
    D6{FT}(x, ntuple(_ -> zero(FT), Val(6)))
@inline _d6_var(x::FT, ::Val{N}) where {FT, N} =
    D6{FT}(x, ntuple(i -> i == N ? one(FT) : zero(FT), Val(6)))
@inline _d6_pack(v::FT, g::NTuple{6, FT}) where {FT} = D6{FT}(v, g)

@inline Base.:+(a::D6{FT}, b::D6{FT}) where {FT} = _d6_pack(a.v + b.v, a.g .+ b.g)
@inline Base.:+(a::D6{FT}, b::FT) where {FT}     = _d6_pack(a.v + b, a.g)
@inline Base.:+(a::FT, b::D6{FT}) where {FT}     = _d6_pack(a + b.v, b.g)
@inline Base.:-(a::D6{FT}, b::D6{FT}) where {FT} = _d6_pack(a.v - b.v, a.g .- b.g)
@inline Base.:-(a::D6{FT}, b::FT) where {FT}     = _d6_pack(a.v - b, a.g)
@inline Base.:-(a::FT, b::D6{FT}) where {FT}     = _d6_pack(a - b.v, .-b.g)
@inline Base.:-(a::D6{FT}) where {FT}            = _d6_pack(-a.v, .-a.g)
@inline Base.:*(a::D6{FT}, b::FT) where {FT}     = _d6_pack(a.v * b, a.g .* b)
@inline Base.:*(a::FT, b::D6{FT}) where {FT}     = _d6_pack(a * b.v, b.g .* a)
@inline Base.:*(a::D6{FT}, b::D6{FT}) where {FT} =
    _d6_pack(a.v * b.v, a.v .* b.g .+ b.v .* a.g)
@inline Base.:/(a::D6{FT}, b::FT) where {FT}     = _d6_pack(a.v / b, a.g ./ b)
@inline function Base.:/(a::D6{FT}, b::D6{FT}) where {FT}
    inv_bv = one(FT) / b.v
    qv = a.v * inv_bv
    qg = inv_bv .* (a.g .- qv .* b.g)
    return _d6_pack(qv, qg)
end
@inline Base.abs(a::D6{FT}) where {FT} =
    a.v >= zero(FT) ? a : _d6_pack(-a.v, .-a.g)

# Forward-only comparisons: branches are taken on `.v`. Used by
# `huynh_second_constraint_d6`, `apply_monotonicity_d6`,
# `ppm_face_value_d6` to mirror the forward branch decisions.

# ---------------------------------------------------------------------------
# d6-AD versions of the LinRood forward chain helpers.
#
# Each function below has the same name as its forward counterpart with
# a `_d6` suffix and accepts `D6{FT}` arguments. Numerical thresholds
# (`10·eps(FT)`, `100·eps(FT)`) come from the forward implementations
# in `ppm_subgrid_distributions.jl` and `LinRood.jl`.
# ---------------------------------------------------------------------------

# d6 mirror of the forward `_ppm_edge_values_ord5` (the 4th-order PPM edge
# interpolation). The formula is purely linear in the q-stencil, so the dual
# carries an exact gradient with no clamp branches.
@inline function _ppm_edge_values_ord5_d6(
    q_imm::D6{FT}, q_im::D6{FT}, q_i::D6{FT}, q_ip::D6{FT}, q_ipp::D6{FT},
) where {FT}
    p1 = FT(7) / FT(12)
    p2 = -FT(1) / FT(12)
    q_L = (q_im + q_i) * p1 + (q_imm + q_ip) * p2
    q_R = (q_i + q_ip) * p1 + (q_im + q_ipp) * p2
    return (q_L, q_R)
end

@inline function _apply_monotonicity_d6(
    q_L::D6{FT}, q_R::D6{FT}, c::D6{FT},
) where {FT}
    diff_R = q_R - c
    diff_L = c - q_L
    if (diff_R.v * diff_L.v) <= zero(FT)
        return (c, c)
    end
    return (q_L, q_R)
end

# The face value itself is the forward `_ppm_face_value` (LinRood.jl), generic in
# the mixing-ratio type: for the LinRood adjoint the velocity tape supplies fixed
# `(F, m_lo, m_hi)`, and the D6 arithmetic propagates the q-stencil sensitivities.

# Full chain on a 6-cell stencil of q values. Returns the 6-component
# gradient `∂face/∂q_n` for n = -3, -2, -1, 0, +1, +2.
@inline function _linrood_ppm_face_from_q_grad_ord5(
    F::FT, m_l::FT, m_r::FT,
    q_m3::FT, q_m2::FT, q_m1::FT, q_0::FT, q_p1::FT, q_p2::FT,
) where {FT}
    c_m3 = _d6_var(q_m3, Val(1))
    c_m2 = _d6_var(q_m2, Val(2))
    c_m1 = _d6_var(q_m1, Val(3))
    c_0  = _d6_var(q_0,  Val(4))
    c_p1 = _d6_var(q_p1, Val(5))
    c_p2 = _d6_var(q_p2, Val(6))

    q_L_m, q_R_m = _ppm_edge_values_ord5_d6(c_m3, c_m2, c_m1, c_0, c_p1)
    q_L_0, q_R_0 = _ppm_edge_values_ord5_d6(c_m2, c_m1, c_0, c_p1, c_p2)
    q_L_m, q_R_m = _apply_monotonicity_d6(q_L_m, q_R_m, c_m1)
    q_L_0, q_R_0 = _apply_monotonicity_d6(q_L_0, q_R_0, c_0)

    face = _ppm_face_value(F, m_l, m_r, c_m1, c_0,
                              q_L_m, q_R_m, q_L_0, q_R_0)
    return face.g  # NTuple{6, FT} = (∂f/∂q_m3, ..., ∂f/∂q_p2)
end

# ---------------------------------------------------------------------------
# ORD=7 discontinuous-edge boundary correction.
#
# At gnomonic CS face boundaries (`face_idx == 1` or `face_idx == Nc + 1`)
# the forward `_apply_ord7_boundary` (LinRood.jl:186) overrides `q_R_m`
# and `q_L_0` with the same discontinuous edge value `q_bdy`. The forward
# formula (`_ppm_face_edge_value_ord7_discontinuous` at
# `ppm_subgrid_distributions.jl:168`) is LINEAR in the four stencil
# cells (c_m1, c_m2, c_0, c_p1):
#
#     extrap_left  = (3/2) c_m1 - (1/2) c_m2
#     extrap_right = (3/2) c_0  - (1/2) c_p1
#     q_bdy        = (extrap_left + extrap_right) / 2
#                  = (3/4) c_m1 - (1/4) c_m2 + (3/4) c_0 - (1/4) c_p1
#
# So `∂q_bdy/∂c_m2 = -1/4`, `∂q_bdy/∂c_m1 = +3/4`, `∂q_bdy/∂c_0 = +3/4`,
# `∂q_bdy/∂c_p1 = -1/4`, and `c_m3`/`c_p2` do not contribute. The d6
# wrapper handles this automatically because the D6 algebra is closed
# under linear combinations.
#
# Returns the (possibly overridden) edge tuple `(q_L_m, q_R_m, q_L_0,
# q_R_0)` in d6 form. For `face_idx` in the interior, this returns its
# arguments unchanged — same compile-time elimination pattern as the
# forward `_apply_ord7_boundary` for `ORD != 7`.
@inline function _apply_ord7_boundary_d6(
    q_L_m::D6{FT}, q_R_m::D6{FT}, q_L_0::D6{FT}, q_R_0::D6{FT},
    c_m1::D6{FT}, c_m2::D6{FT}, c_0::D6{FT}, c_p1::D6{FT},
    face_idx::Integer, Nc::Integer,
) where {FT}
    if face_idx == 1 || face_idx == Nc + 1
        half = one(FT) / FT(2)
        three_halves = FT(3) / FT(2)
        extrap_left  = three_halves * c_m1 - c_m2 * half
        extrap_right = three_halves * c_0  - c_p1 * half
        q_bdy = (extrap_left + extrap_right) * half
        return (q_L_m, q_bdy, q_bdy, q_R_0)
    end
    return (q_L_m, q_R_m, q_L_0, q_R_0)
end

# ORD=7 from-q grad. Same chain as ORD=5 with the discontinuous boundary
# correction inserted between `_ppm_edge_values_ord5_d6` and
# `_apply_monotonicity_d6`. Identical to `_linrood_ppm_face_from_q_grad_ord5`
# at interior faces (since `_apply_ord7_boundary_d6` is the identity there).
@inline function _linrood_ppm_face_from_q_grad_ord7(
    F::FT, m_l::FT, m_r::FT,
    q_m3::FT, q_m2::FT, q_m1::FT, q_0::FT, q_p1::FT, q_p2::FT,
    face_idx::Integer, Nc::Integer,
) where {FT}
    c_m3 = _d6_var(q_m3, Val(1))
    c_m2 = _d6_var(q_m2, Val(2))
    c_m1 = _d6_var(q_m1, Val(3))
    c_0  = _d6_var(q_0,  Val(4))
    c_p1 = _d6_var(q_p1, Val(5))
    c_p2 = _d6_var(q_p2, Val(6))

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

# ---------------------------------------------------------------------------
# Adjoint of `_ppm_x_face_from_q_kernel!` (LinRood.jl:299) for ORD=5.
#
# Forward: `fx_face[iif, j, k] = _ppm_face_value(am[iif, j, k], …,
# q[ii_l-2..ii_r+2, jj, k])`. The stencil spans 6 cells in i at fixed j.
# Adjoint accumulates lambda_fx_face[iif, j, k] into lambda_q at those
# six stencil cells via `@atomic` (multiple faces can write to the same
# cell).
# ---------------------------------------------------------------------------

@kernel function _ppm_x_face_from_q_kernel_adjoint_ord5!(
    lambda_q,
    @Const(lambda_fx_face), @Const(q), @Const(am), @Const(m),
    Hp, Nc,
)
    iif, j, k = @index(Global, NTuple)
    _ = Nc  # signature parity with forward kernels — only used at dispatch
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        q_m3 = q[ii_l - 2, jj, k]
        q_m2 = q[ii_l - 1, jj, k]
        q_m1 = q[ii_l,     jj, k]
        q_0  = q[ii_r,     jj, k]
        q_p1 = q[ii_r + 1, jj, k]
        q_p2 = q[ii_r + 2, jj, k]

        F   = am[iif, j, k]
        m_l = m[ii_l, jj, k]
        m_r = m[ii_r, jj, k]

        grad = _linrood_ppm_face_from_q_grad_ord5(F, m_l, m_r,
                                                  q_m3, q_m2, q_m1, q_0, q_p1, q_p2)
        bar = lambda_fx_face[iif, j, k]

        @atomic lambda_q[ii_l - 2, jj, k] += bar * grad[1]
        @atomic lambda_q[ii_l - 1, jj, k] += bar * grad[2]
        @atomic lambda_q[ii_l,     jj, k] += bar * grad[3]
        @atomic lambda_q[ii_r,     jj, k] += bar * grad[4]
        @atomic lambda_q[ii_r + 1, jj, k] += bar * grad[5]
        @atomic lambda_q[ii_r + 2, jj, k] += bar * grad[6]
    end
end

# ORD=7 from-q x-face adjoint kernel. Identical to the ORD=5 kernel
# except the per-thread grad function takes `face_idx (= iif)` and `Nc`
# so it can apply the discontinuous boundary correction at panel edges
# (`iif == 1` or `iif == Nc + 1`). Compile-time eliminates to the ORD=5
# computation at interior faces.
@kernel function _ppm_x_face_from_q_kernel_adjoint_ord7!(
    lambda_q,
    @Const(lambda_fx_face), @Const(q), @Const(am), @Const(m),
    Hp, Nc,
)
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        q_m3 = q[ii_l - 2, jj, k]
        q_m2 = q[ii_l - 1, jj, k]
        q_m1 = q[ii_l,     jj, k]
        q_0  = q[ii_r,     jj, k]
        q_p1 = q[ii_r + 1, jj, k]
        q_p2 = q[ii_r + 2, jj, k]

        F   = am[iif, j, k]
        m_l = m[ii_l, jj, k]
        m_r = m[ii_r, jj, k]

        grad = _linrood_ppm_face_from_q_grad_ord7(F, m_l, m_r,
                                                  q_m3, q_m2, q_m1, q_0, q_p1, q_p2,
                                                  iif, Nc)
        bar = lambda_fx_face[iif, j, k]

        @atomic lambda_q[ii_l - 2, jj, k] += bar * grad[1]
        @atomic lambda_q[ii_l - 1, jj, k] += bar * grad[2]
        @atomic lambda_q[ii_l,     jj, k] += bar * grad[3]
        @atomic lambda_q[ii_r,     jj, k] += bar * grad[4]
        @atomic lambda_q[ii_r + 1, jj, k] += bar * grad[5]
        @atomic lambda_q[ii_r + 2, jj, k] += bar * grad[6]
    end
end

"""
    apply_ppm_x_face_from_q_adjoint!(lambda_q, lambda_fx_face, q, am, m,
                                       mesh, ::Val{ORD})

Discrete transpose of `_ppm_x_face_from_q_kernel!` (LinRood.jl:299) for
one panel at ORD=5 or ORD=7 (LinRoodPPMScheme). The donor-mass
denominator `m_l`/`m_r` in `_ppm_face_value` and the velocity `am` are
treated as fixed parameters from the tape — the adjoint propagates only
the q-stencil sensitivity. Atomic writes on `lambda_q` because multiple
faces share each cell.

`ORD=7` dispatches to a kernel that applies the linear discontinuous
boundary correction (`_apply_ord7_boundary_d6`) at panel-edge faces
(`face_idx ∈ {1, Nc+1}`) before the monotonicity step; interior faces
are bit-equal to ORD=5.
"""
function apply_ppm_x_face_from_q_adjoint!(lambda_q, lambda_fx_face,
                                          q, am, m,
                                          mesh::CubedSphereMesh,
                                          ::Val{ORD}=Val(5)) where {ORD}
    (ORD == 5 || ORD == 7) || throw(ArgumentError(
        "LinRoodPPMScheme adjoint supports ORD ∈ {5, 7}; got ORD=$ORD."))
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(q, 3)
    backend = get_backend(lambda_q)
    k! = ORD == 5 ?
        _ppm_x_face_from_q_kernel_adjoint_ord5!(backend, 256) :
        _ppm_x_face_from_q_kernel_adjoint_ord7!(backend, 256)
    k!(lambda_q, lambda_fx_face, q, am, m, Hp, Nc;
       ndrange=(Nc + 1, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Adjoint of `_ppm_y_face_from_q_kernel!` (LinRood.jl:325) for ORD=5.
#
# Same chain as the X variant with the stencil running along j at fixed
# i.
# ---------------------------------------------------------------------------

@kernel function _ppm_y_face_from_q_kernel_adjoint_ord5!(
    lambda_q,
    @Const(lambda_fy_face), @Const(q), @Const(bm), @Const(m),
    Hp, Nc,
)
    i, jf, k = @index(Global, NTuple)
    _ = Nc
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        q_m3 = q[ii, jj_b - 2, k]
        q_m2 = q[ii, jj_b - 1, k]
        q_m1 = q[ii, jj_b,     k]
        q_0  = q[ii, jj_a,     k]
        q_p1 = q[ii, jj_a + 1, k]
        q_p2 = q[ii, jj_a + 2, k]

        F   = bm[i, jf, k]
        m_l = m[ii, jj_b, k]
        m_r = m[ii, jj_a, k]

        grad = _linrood_ppm_face_from_q_grad_ord5(F, m_l, m_r,
                                                  q_m3, q_m2, q_m1, q_0, q_p1, q_p2)
        bar = lambda_fy_face[i, jf, k]

        @atomic lambda_q[ii, jj_b - 2, k] += bar * grad[1]
        @atomic lambda_q[ii, jj_b - 1, k] += bar * grad[2]
        @atomic lambda_q[ii, jj_b,     k] += bar * grad[3]
        @atomic lambda_q[ii, jj_a,     k] += bar * grad[4]
        @atomic lambda_q[ii, jj_a + 1, k] += bar * grad[5]
        @atomic lambda_q[ii, jj_a + 2, k] += bar * grad[6]
    end
end

# ORD=7 from-q y-face adjoint kernel. Mirrors the ORD=5 kernel with
# `face_idx = jf` driving the discontinuous boundary correction at
# panel-edge faces.
@kernel function _ppm_y_face_from_q_kernel_adjoint_ord7!(
    lambda_q,
    @Const(lambda_fy_face), @Const(q), @Const(bm), @Const(m),
    Hp, Nc,
)
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        q_m3 = q[ii, jj_b - 2, k]
        q_m2 = q[ii, jj_b - 1, k]
        q_m1 = q[ii, jj_b,     k]
        q_0  = q[ii, jj_a,     k]
        q_p1 = q[ii, jj_a + 1, k]
        q_p2 = q[ii, jj_a + 2, k]

        F   = bm[i, jf, k]
        m_l = m[ii, jj_b, k]
        m_r = m[ii, jj_a, k]

        grad = _linrood_ppm_face_from_q_grad_ord7(F, m_l, m_r,
                                                  q_m3, q_m2, q_m1, q_0, q_p1, q_p2,
                                                  jf, Nc)
        bar = lambda_fy_face[i, jf, k]

        @atomic lambda_q[ii, jj_b - 2, k] += bar * grad[1]
        @atomic lambda_q[ii, jj_b - 1, k] += bar * grad[2]
        @atomic lambda_q[ii, jj_b,     k] += bar * grad[3]
        @atomic lambda_q[ii, jj_a,     k] += bar * grad[4]
        @atomic lambda_q[ii, jj_a + 1, k] += bar * grad[5]
        @atomic lambda_q[ii, jj_a + 2, k] += bar * grad[6]
    end
end

"""
    apply_ppm_y_face_from_q_adjoint!(lambda_q, lambda_fy_face, q, bm, m,
                                       mesh, ::Val{ORD})

Discrete transpose of `_ppm_y_face_from_q_kernel!` (LinRood.jl:325) for
one panel at ORD=5 or ORD=7. See `apply_ppm_x_face_from_q_adjoint!` for
the contract.
"""
function apply_ppm_y_face_from_q_adjoint!(lambda_q, lambda_fy_face,
                                          q, bm, m,
                                          mesh::CubedSphereMesh,
                                          ::Val{ORD}=Val(5)) where {ORD}
    (ORD == 5 || ORD == 7) || throw(ArgumentError(
        "LinRoodPPMScheme adjoint supports ORD ∈ {5, 7}; got ORD=$ORD."))
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(q, 3)
    backend = get_backend(lambda_q)
    k! = ORD == 5 ?
        _ppm_y_face_from_q_kernel_adjoint_ord5!(backend, 256) :
        _ppm_y_face_from_q_kernel_adjoint_ord7!(backend, 256)
    k!(lambda_q, lambda_fy_face, q, bm, m, Hp, Nc;
       ndrange=(Nc, Nc + 1, Nz))
    synchronize(backend)
    return nothing
end
