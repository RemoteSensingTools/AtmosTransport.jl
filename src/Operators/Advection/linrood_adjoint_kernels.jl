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

# ===========================================================================
# Single-panel, zero-halo LinRood horizontal adjoint.
#
# Composes the six kernel adjoints above into a single-step
# reverse pass that mirrors the forward `fv_tp_2d_cs!` (LinRood.jl:695)
# for ONE panel with all halos held at zero. Cross-panel halo / corner
# adjoint (`_adjoint_fill_panel_halos!`, `copy_corners` reverse) is
# deferred to the full tape integration.
#
# The forward path captured in this composition:
#   Phase 1: q_buf = safe_mixing_ratio(rm, m)         (init A: state = c)
#            fy_in = ppm_y_face(rm, m, bm)
#            q_buf[interior] = pre_advect_y(rm, m, bm, fy_in)  (state B: int=q*, halo=c)
#   Phase 2: fx_out = ppm_x_face_from_q(q_buf_B, am, m)
#            fx_in  = ppm_x_face(rm, m, am)
#            q_buf = safe_mixing_ratio(rm, m)         (re-init A')
#            q_buf[interior] = pre_advect_x(rm, m, am, fx_in)  (state C: int=q', halo=c)
#   Phase 3: fy_out = ppm_y_face_from_q(q_buf_C, bm, m)
#            (rm_new, m_new) = linrood_update(rm, m, am, bm, fx_*, fy_*)
# ===========================================================================

# Reverse of `safe_mixing_ratio(rm[h], m[h])` restricted to the halo
# region (i.e., the union of {i in 1..N, j in 1..N, k} with at least
# one of i ∉ Hp+1..Hp+Nc OR j ∉ Hp+1..Hp+Nc). Adds the chain-rule
# contribution
#     lambda_rm[h] += lambda_q[h] / m[h]
#     lambda_m [h] += lambda_q[h] * (-rm[h]/m[h]²)
# with the same `100·eps` threshold guard.
@kernel function _safe_mixing_ratio_halo_adjoint_kernel!(
    lambda_rm, lambda_m, @Const(lambda_q), @Const(rm), @Const(m), Nc, Hp,
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(lambda_rm)
    thresh = FT(100) * eps(FT)
    @inbounds begin
        is_interior = (Hp + 1 <= i <= Hp + Nc) && (Hp + 1 <= j <= Hp + Nc)
        m_h = m[i, j, k]
        if !is_interior && m_h > thresh
            inv_m = one(FT) / m_h
            bar = lambda_q[i, j, k]
            lambda_rm[i, j, k] += bar * inv_m
            lambda_m[i, j, k]  += bar * (-rm[i, j, k] * inv_m * inv_m)
        end
    end
end

function _accumulate_safe_mixing_ratio_halo_adjoint!(
    lambda_rm, lambda_m, lambda_q, rm, m, mesh::CubedSphereMesh,
)
    backend = get_backend(lambda_rm)
    kernel! = _safe_mixing_ratio_halo_adjoint_kernel!(backend, 256)
    kernel!(lambda_rm, lambda_m, lambda_q, rm, m, mesh.Nc, mesh.Hp;
            ndrange=size(lambda_rm))
    synchronize(backend)
    return nothing
end

# Helper to zero the interior of an array in [Hp+1..Hp+Nc] × [Hp+1..Hp+Nc].
function _zero_interior!(arr, mesh::CubedSphereMesh)
    Nc = mesh.Nc; Hp = mesh.Hp
    fill!(view(arr, (Hp + 1):(Hp + Nc), (Hp + 1):(Hp + Nc), :), zero(eltype(arr)))
    return nothing
end

# The forward update preserves halos. Their output adjoints therefore carry
# through unchanged, in addition to the reconstructed-face contributions.
@kernel function _linrood_halo_adjoint_kernel!(
    lambda_rm, lambda_m, @Const(halo_rm), @Const(halo_m), Nc, Hp,
)
    i, j, k = @index(Global, NTuple)
    if !(Hp + 1 <= i <= Hp + Nc && Hp + 1 <= j <= Hp + Nc)
        @inbounds begin
            lambda_rm[i, j, k] += halo_rm[i, j, k]
            lambda_m[i, j, k] += halo_m[i, j, k]
        end
    end
end

function _accumulate_linrood_halo_adjoint!(lambda_rm, lambda_m, halo_rm, halo_m,
                                            mesh::CubedSphereMesh)
    backend = get_backend(lambda_rm)
    kernel! = _linrood_halo_adjoint_kernel!(backend, 256)
    kernel!(lambda_rm, lambda_m, halo_rm, halo_m, mesh.Nc, mesh.Hp;
            ndrange=size(lambda_rm))
    synchronize(backend)
    return nothing
end

"""
    apply_linrood_horizontal_adjoint_single_panel!(
        lambda_rm, lambda_m,
        lambda_rm_new, lambda_m_new,
        rm, m, am, bm,
        q_buf_phase2, q_buf_phase3,
        fx_in, fx_out, fy_in,
        mesh, ::Val{ORD})

Per-panel reverse composition used by `fv_tp_2d_cs!`. With `face_adjoints`,
the six-panel caller supplies seeds after reversing the final update and
shared-seam projection. Without them, this includes the uncoupled local
update used by standalone kernel tests. Outer-face Courant denominators
are fixed meteorology; this is not a complete derivative with respect to air mass.
Reads the forward tape inputs `(rm, m, am, bm, q_buf_phase2,
q_buf_phase3, fx_in, fx_out, fy_in)` and the adjoint seed
`(lambda_rm_new, lambda_m_new)`, then accumulates into the input
adjoints `(lambda_rm, lambda_m)`. Internally allocates the
intermediate face / q_buf adjoints.

Tape inputs:
- `q_buf_phase2` — state B of q_buf (interior = q*, halo = c=rm/m from
  init A); produced by phase 1's pre_advect_y.
- `q_buf_phase3` — state C of q_buf (interior = q', halo = c=rm/m from
  re-init A'); produced by phase 2's pre_advect_x.
- `fx_in`, `fx_out`, `fy_in` — face mixing ratios captured at the end
  of phase 2 / phase 3.

`fy_out` is recomputed inside the kernel; it isn't a tape input.
Supports `Val(5)` (default) and `Val(7)`. Each per-direction face
adjoint dispatches on `Val(ORD)` so the ORD=7 panel-edge boundary
correction propagates through to `(lambda_rm, lambda_m)`.
"""
function apply_linrood_horizontal_adjoint_single_panel!(
    lambda_rm, lambda_m,
    lambda_rm_new, lambda_m_new,
    rm, m, am, bm,
    q_buf_phase2, q_buf_phase3,
    fx_in, fx_out, fy_in,
    mesh::CubedSphereMesh,
    ::Val{ORD}=Val(5);
    face_adjoints=nothing,
) where {ORD}
    (ORD == 5 || ORD == 7) || throw(ArgumentError(
        "LinRoodPPMScheme adjoint supports ORD ∈ {5, 7}; got ORD=$ORD."))
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(lambda_rm, 3)
    N = Nc + 2Hp
    FT = eltype(lambda_rm)

    # A six-panel caller reverses the update and shared-face projection first,
    # then passes the coupled face seeds. Standalone panel callers retain the
    # uncoupled composition used by the local kernel derivative tests.
    if face_adjoints === nothing
        lambda_fx_in  = similar(lambda_rm, FT, (Nc + 1, Nc, Nz)); fill!(lambda_fx_in,  zero(FT))
        lambda_fx_out = similar(lambda_rm, FT, (Nc + 1, Nc, Nz)); fill!(lambda_fx_out, zero(FT))
        lambda_fy_in  = similar(lambda_rm, FT, (Nc, Nc + 1, Nz)); fill!(lambda_fy_in,  zero(FT))
        lambda_fy_out = similar(lambda_rm, FT, (Nc, Nc + 1, Nz)); fill!(lambda_fy_out, zero(FT))
        apply_linrood_update_adjoint!(
            lambda_rm, lambda_m,
            lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
            lambda_rm_new, lambda_m_new, am, bm, mesh,
        )
    else
        lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out = face_adjoints
    end
    lambda_q_buf = similar(lambda_rm, FT, (N, N, Nz)); fill!(lambda_q_buf, zero(FT))

    # yq_face_adjoint: lambda_fy_out → lambda_q_buf (q_buf at phase 3 state C)
    apply_ppm_y_face_from_q_adjoint!(
        lambda_q_buf, lambda_fy_out, q_buf_phase3, bm, m, mesh, Val(ORD),
    )

    # ── Reverse Phase 2 ────────────────────────────────────────────
    # `q_buf` interior at end of phase 2 was overwritten by pre_advect_x
    # from rm, m, am, fx_in. Adjoint: lambda_q_buf[interior] feeds the
    # pre_advect_x reverse, then is zeroed because the forward
    # overwrote it.
    apply_pre_advect_x_adjoint!(
        lambda_rm, lambda_m, lambda_fx_in,
        lambda_q_buf, rm, m, am, fx_in, mesh,
    )
    _zero_interior!(lambda_q_buf, mesh)

    # `q_buf` halo at end of phase 2 was set by the re-init A'
    # (init_q_buf over the entire haloed N×N from c=rm/m). Adjoint:
    # the halo portion of lambda_q_buf goes back into lambda_rm,
    # lambda_m via the safe_mixing_ratio chain rule.
    _accumulate_safe_mixing_ratio_halo_adjoint!(
        lambda_rm, lambda_m, lambda_q_buf, rm, m, mesh,
    )
    # `lambda_q_buf` is now fully consumed; clear it for the next phase.
    fill!(lambda_q_buf, zero(FT))

    # x_face_adjoint: lambda_fx_in → lambda_rm, lambda_m
    apply_ppm_x_face_adjoint!(
        lambda_rm, lambda_m, lambda_fx_in, rm, m, am, mesh, Val(ORD),
    )

    # xq_face_adjoint: lambda_fx_out → lambda_q_buf (q_buf at phase 2 state B)
    apply_ppm_x_face_from_q_adjoint!(
        lambda_q_buf, lambda_fx_out, q_buf_phase2, am, m, mesh, Val(ORD),
    )

    # ── Reverse Phase 1 ────────────────────────────────────────────
    # `q_buf` interior at end of phase 1 was overwritten by pre_advect_y.
    apply_pre_advect_y_adjoint!(
        lambda_rm, lambda_m, lambda_fy_in,
        lambda_q_buf, rm, m, bm, fy_in, mesh,
    )
    _zero_interior!(lambda_q_buf, mesh)

    # `q_buf` halo at end of phase 1 was set by the init A.
    _accumulate_safe_mixing_ratio_halo_adjoint!(
        lambda_rm, lambda_m, lambda_q_buf, rm, m, mesh,
    )
    fill!(lambda_q_buf, zero(FT))

    # y_face_adjoint: lambda_fy_in → lambda_rm, lambda_m
    apply_ppm_y_face_adjoint!(
        lambda_rm, lambda_m, lambda_fy_in, rm, m, bm, mesh, Val(ORD),
    )

    return nothing
end

# ===========================================================================
# Single-panel, multi-substep LinRood horizontal adjoint
# with ZERO cross-panel halos.
#
# Builds on the single-panel composition above to support a
# sequence of forward substeps (each with its own `am`/`bm` tape),
# replayed in reverse order. Six-panel orchestration and cross-panel
# halo adjoint integration into `cs_surface_emission_footprint` are
# deferred — this path ships a parallel API
# `apply_linrood_multi_substep_adjoint!` that takes a single-panel
# meteo tape and produces gradients of an objective over the substep
# sequence.
# ===========================================================================

# Per-substep tape entry produced by the forward pass: holds the
# state at the START of the substep plus the intermediate q_buf
# snapshots needed by the reverse pass. `ORD` (5 or 7) binds the
# entry to the LinRood scheme order that produced it; the reverse
# pass (`apply_linrood_multi_substep_adjoint!`) reads it from the
# tape's element type and dispatches the face-kernel adjoints to
# the matching ORD=5 or ORD=7 kernel — guaranteeing the adjoint
# matches the forward path at panel-edge faces. Same pattern as
# `_CSLinRoodHorizRecord` in `src/Adjoints/LinRoodTape.jl`.
#
# (Known issue: before this binding, a tape recorded
# at ORD=7 silently reversed with ORD=5 if the caller did not pass
# `Val(7)` to `apply_linrood_multi_substep_adjoint!` — reproduced
# with a smooth one-step FD/VJP check, default error ~1.34e-4 vs
# correct error ~6.9e-10.)
struct LinRoodHorizontalTapeEntry{FT, A3, A3x, A3y, ORD}
    rm     :: A3   # rm at substep start (haloed)
    m      :: A3   # m  at substep start (haloed)
    q_buf_phase2 :: A3   # q_buf state B (end of phase 1)
    q_buf_phase3 :: A3   # q_buf state C (end of phase 2)
    fx_in  :: A3x  # fx_in face (computed in phase 2)
    fx_out :: A3x  # fx_out face (computed in phase 2)
    fy_in  :: A3y  # fy_in face (computed in phase 1)
end

"""
    record_linrood_substep!(rm, m, am, bm, mesh; ord=Val(5)) -> (tape_entry, rm_new, m_new)

Run one forward LinRood horizontal substep ON ONE PANEL with halos
held at their input values (no cross-panel transfer) and return a
`LinRoodHorizontalTapeEntry` plus the updated `(rm_new, m_new)`. The
input `rm`, `m` are NOT mutated.

`ord` selects the PPM order (Val(5) or Val(7)) that the face kernels
use. ORD=7 forwards through the discontinuous-edge boundary correction
at panel-edge faces.
"""
function record_linrood_substep!(rm, m, am, bm,
                                  mesh::CubedSphereMesh{FT};
                                  ord::Val{ORD}=Val(5)) where {FT, ORD}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm, 3)
    N = Nc + 2Hp
    backend = get_backend(rm)

    init_k!    = _init_q_buf_kernel!(backend, 256)
    y_face_k!  = _ppm_y_face_kernel!(backend, 256)
    x_face_k!  = _ppm_x_face_kernel!(backend, 256)
    xq_face_k! = _ppm_x_face_from_q_kernel!(backend, 256)
    yq_face_k! = _ppm_y_face_from_q_kernel!(backend, 256)
    pre_y_k!   = _pre_advect_y_kernel!(backend, 256)
    pre_x_k!   = _pre_advect_x_kernel!(backend, 256)
    update_k!  = _linrood_update_kernel!(backend, 256)

    # Backend-aware allocations (`similar` honors the input array's
    # storage type so device inputs stay on-device).
    fy_in  = similar(rm, FT, (Nc, Nc + 1, Nz));  fill!(fy_in,  zero(FT))
    fy_out = similar(rm, FT, (Nc, Nc + 1, Nz));  fill!(fy_out, zero(FT))
    fx_in  = similar(rm, FT, (Nc + 1, Nc, Nz));  fill!(fx_in,  zero(FT))
    fx_out = similar(rm, FT, (Nc + 1, Nc, Nz));  fill!(fx_out, zero(FT))
    q_buf  = similar(rm, FT, (N, N, Nz));        fill!(q_buf,  zero(FT))

    # Phase 1
    init_k!(q_buf, rm, m; ndrange=(N, N, Nz))
    synchronize(backend)
    y_face_k!(fy_in, rm, m, bm, Hp, Nc, Val(ORD);
              ndrange=(Nc, Nc + 1, Nz))
    pre_y_k!(q_buf, rm, m, bm, fy_in, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    q_buf_phase2 = copy(q_buf)

    # Phase 2
    xq_face_k!(fx_out, q_buf_phase2, am, m, Hp, Nc, Val(ORD);
               ndrange=(Nc + 1, Nc, Nz))
    x_face_k!(fx_in, rm, m, am, Hp, Nc, Val(ORD);
              ndrange=(Nc + 1, Nc, Nz))
    synchronize(backend)
    init_k!(q_buf, rm, m; ndrange=(N, N, Nz))
    synchronize(backend)
    pre_x_k!(q_buf, rm, m, am, fx_in, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    q_buf_phase3 = copy(q_buf)

    # Phase 3
    yq_face_k!(fy_out, q_buf_phase3, bm, m, Hp, Nc, Val(ORD);
               ndrange=(Nc, Nc + 1, Nz))
    rm_new = copy(rm)
    m_new  = copy(m)
    update_buf_rm = similar(rm, FT, (N, N, Nz));  fill!(update_buf_rm, zero(FT))
    update_buf_m  = similar(rm, FT, (N, N, Nz));  fill!(update_buf_m,  zero(FT))
    update_k!(update_buf_rm, update_buf_m, rm, m, am, bm,
              fx_in, fx_out, fy_in, fy_out, Hp;
              ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    _copy_interior!(rm_new, update_buf_rm, Nc, Hp, Nz)
    _copy_interior!(m_new, update_buf_m, Nc, Hp, Nz)

    entry = LinRoodHorizontalTapeEntry{FT, typeof(rm), typeof(fx_in), typeof(fy_in), ORD}(
        copy(rm), copy(m), q_buf_phase2, q_buf_phase3, fx_in, fx_out, fy_in)
    return (entry, rm_new, m_new)
end

"""
    apply_linrood_multi_substep_adjoint!(
        lambda_rm0, lambda_m0,
        lambda_rm_final, lambda_m_final,
        tape, am_steps, bm_steps,
        mesh)

Reverse over `length(tape)` substeps. `tape[t]` is the
`LinRoodHorizontalTapeEntry` produced by `record_linrood_substep!`
for substep `t` on one panel. `am_steps[t]` / `bm_steps[t]` are the
matching velocity tapes.

Accumulates the gradient of the final-state objective
`⟨lambda_rm_final, rm_final⟩ + ⟨lambda_m_final, m_final⟩` w.r.t. the
substep-0 state into `(lambda_rm0, lambda_m0)`. Pure single-panel
zero-cross-panel-halo path — the single-substep scope extended over
substeps.

The PPM scheme order `ORD` is read from the tape's element type
(`LinRoodHorizontalTapeEntry{…, ORD}`), so a tape recorded at
ORD=7 always reverses with the ORD=7 face-kernel adjoints — the
ORD is not a separate kwarg that could drift from the forward
pass.
"""
function apply_linrood_multi_substep_adjoint!(
    lambda_rm0, lambda_m0,
    lambda_rm_final, lambda_m_final,
    tape::AbstractVector{<:LinRoodHorizontalTapeEntry{FT_, A3_, A3x_, A3y_, ORD}},
    am_steps, bm_steps,
    mesh::CubedSphereMesh,
) where {FT_, A3_, A3x_, A3y_, ORD}
    nsteps = length(tape)
    @assert length(am_steps) == nsteps
    @assert length(bm_steps) == nsteps
    FT = eltype(lambda_rm0)

    # Working lambda for the running state; starts at the final-time
    # seed, ends at substep-0 (which we copy into lambda_rm0,
    # lambda_m0). Each substep's reverse READS lambda_rm/m (the
    # adjoint of the substep output) and WRITES into the substep-input
    # adjoint accumulators.
    lambda_rm = copy(lambda_rm_final)
    lambda_m  = copy(lambda_m_final)

    for t in nsteps:-1:1
        entry = tape[t]
        # Allocate fresh accumulators for the substep-input adjoint
        # on the same backend as `lambda_rm` / `lambda_m`.
        sub_lambda_rm = similar(lambda_rm); fill!(sub_lambda_rm, zero(FT))
        sub_lambda_m  = similar(lambda_m);  fill!(sub_lambda_m,  zero(FT))
        apply_linrood_horizontal_adjoint_single_panel!(
            sub_lambda_rm, sub_lambda_m,
            lambda_rm, lambda_m,
            entry.rm, entry.m, am_steps[t], bm_steps[t],
            entry.q_buf_phase2, entry.q_buf_phase3,
            entry.fx_in, entry.fx_out, entry.fy_in,
            mesh, Val(ORD),
        )
        # The substep output's adjoint outside the interior is the
        # ``carry-over'' adjoint from the previous reverse step. Inside
        # the interior, the substep overwrites the state, so the carry
        # interior is zero. We pass the interior-only lambda_rm/m into
        # the substep adjoint, and add the halo carry to the substep-
        # input adjoint at the end.
        _accumulate_linrood_halo_adjoint!(sub_lambda_rm, sub_lambda_m,
                                         lambda_rm, lambda_m, mesh)
        # The substep adjoint's output IS the substep-input adjoint;
        # shift it into the running lambda for the next (earlier)
        # substep.
        copyto!(lambda_rm, sub_lambda_rm)
        copyto!(lambda_m,  sub_lambda_m)
    end

    # Final running lambda IS the gradient at substep-0.
    lambda_rm0 .+= lambda_rm
    lambda_m0 .+= lambda_m
    return nothing
end
