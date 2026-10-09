# Lin-Rood horizontal adjoint of one panel (zero halo), single and multi-substep.
# Split from linrood_adjoint_kernels.jl (refactor phase 4); included by Advection.jl in this order.

# ===========================================================================
# Single-panel, zero-halo LinRood horizontal adjoint.
#
# Composes the six kernel adjoints (linrood_adjoint_kernels.jl,
# linrood_adjoint_rm_faces.jl) into a single-step
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
