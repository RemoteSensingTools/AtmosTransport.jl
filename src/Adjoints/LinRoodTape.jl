# ---------------------------------------------------------------------------
# LinRood adjoint tape integration.
#
# Wires the per-kernel LinRood adjoints
# (in `src/Operators/Advection/linrood_adjoint_*.jl`) into the
# CS surface-emission-footprint reverse pass managed by Adjoints.jl.
# Provides:
#
#   * `_CSLinRoodHorizRecord` — per-substep tape record holding all 6
#     panels' input state and the two intermediate `q_buf` snapshots
#     needed by the reverse pass.
#   * the record is built by the recorder hooks of the production
#     `fv_tp_2d_cs!` (`record_linrood_horizontal!`,
#     Footprint/TapeRecorder.jl), so the tape holds the states the
#     production operator computed.
#   * `_apply_cs_linrood_horizontal_adjoint!` — reverse of one
#     `_CSLinRoodHorizRecord`: per-panel single-panel adjoint
#     composition followed by
#     `_adjoint_fill_panel_halos!` for cross-panel halo
#     redistribution.
#
# Status (RESOLVED 2026-06-02): the LinRood reverse-mode adjoint is correct.
#   LinRoodPPMScheme applies a monotonicity limiter (`_apply_monotonicity`:
#   clip the PPM reconstruction to first order at local extrema), so the
#   forward map is mildly NONLINEAR. This reverse is the transpose of its
#   tangent-linear model — the limiter switch is decided by the forward base
#   value and the active branch is propagated linearly, which is exactly right.
#
#   The former "~7.4e-4-per-substep footprint-vs-centered-FD residual" was a
#   FINITE-DIFFERENCE ARTIFACT, not an adjoint error. It appears only when the
#   forward trajectory sits ON limiter switch points — e.g. a zero IC plus a
#   localized emission, where central FD straddles a kink and returns the
#   AVERAGE of the two one-sided derivatives while the adjoint returns one side.
#   The mismatch is eps-INDEPENDENT (constant from FD step 2e-4 down to 2e-7),
#   the signature of a kink rather than roundoff or a transpose error.
#
#   Evidence (all reproduced this session):
#     * The tape is recorded from production `fv_tp_2d_cs!` itself (its
#       recorder hooks), so the forward the reverse transposes is production.
#     * The single-substep VJP matches central FD to 1e-7 (incl. m-coupling).
#     * On a SMOOTH base field the full footprint identity matches central FD
#       to ~1e-11 for BOTH ORD=5 and ORD=7, the same as the linear PPM control,
#       including upward vertical flux with a surface-layer objective
#       (test/core/test_cs_linrood_footprint_fd.jl). Only the non-smooth
#       zero-IC case shows the ~1.4e-3 kink artifact.
#     * Issue #10 (fixed): the Z half-sweep records used to stage the air mass
#       AFTER the sweep. The upwind reverse needs the pre-sweep donor mass, so
#       the identity held only to ~1e-8 for column means and ~1e-4 for a
#       surface layer at vertical Courant 0.08 (error ~C_z^2, zero for purely
#       downward flux). The ~1e-8 had been read as FD noise. The production
#       palindrome's `record_sweep!` hook runs before each Z sweep.
#   The split-sweep (PPM/Upwind/Slopes) footprint test validates at a zero IC
#   precisely because those schemes are NoLimiter (linear). The LinRood adjoint
#   is production-ready for 4D-Var; FD validation must use a smooth base field
#   (standard practice for limited-scheme adjoints — see
#   test/diagnostic/test_linrood_adjoint_integration.jl `_seed_smooth_cs_ic!`).
#   * ORD ∈ {5, 7} (LinRoodPPMScheme(5) and LinRoodPPMScheme(7) are both
#     supported). `_CSLinRoodHorizRecord`
#     binds ORD as a type parameter; the reverse pass reads it via
#     dispatch and forwards `Val(ORD)` to the face-kernel adjoints.
#   * Vertical sweeps: `scheme.vertical`, upwind or the FV3 `kord = 8` profile
#     (OF-94). The FV3 records stage the tracer mass entering the sweep and the
#     reverse applies the transpose at the forward's limiter branches
#     (`Operators/Advection/vertical_fv3_profile_adjoint.jl`), the same
#     frozen-limiter linearization as the horizontal PPM.
# ---------------------------------------------------------------------------

# Forward kernels + adjoint wrappers from Operators.Advection. Imported at
# Adjoints.jl module scope; this file is `include`d inside that module.

# LinRood records hold raw panel tuples rather than per-policy tape
# slots, so any non-`:device` storage request is currently a footgun:
# the forward pass would silently keep the tape on the source backend
# while the user thought they had opted into mmap eviction. Reject
# explicitly until storage plumbing reaches `_CSLinRoodHorizRecord`.
_linrood_validate_tape_storage(::DeviceCSTapeStorage) = nothing
function _linrood_validate_tape_storage(storage::Symbol)
    storage === :device || _linrood_storage_unsupported(storage)
    return nothing
end
_linrood_validate_tape_storage(storage) = _linrood_storage_unsupported(storage)

function _linrood_storage_unsupported(storage)
    throw(ArgumentError(
        "LinRoodPPMScheme reverse tape currently only supports " *
        "tape_storage = :device / DeviceCSTapeStorage(); got " *
        repr(storage) * ". The LinRood per-substep record stores " *
        "panel tuples directly rather than per-policy slots, so " *
        "mmap / pinned-host eviction is not yet wired through " *
        "(Plan 26 follow-up)."))
end

# Per-substep LinRood horizontal tape record. The forward state is
# stored ONCE per substep for all six panels. `ORD` (5 or 7) binds the
# record to the LinRood scheme order that built it; the reverse pass
# (`_apply_cs_linrood_horizontal_adjoint!`) reads it from the type and
# dispatches the face-kernel adjoints to the matching ORD=5 or ORD=7
# kernel — guaranteeing the adjoint matches the forward path at the
# panel-edge boundary correction.
struct _CSLinRoodHorizRecord{FT, A3, A3x, A3y, P, ORD}
    panels_rm    :: NTuple{6, P}     # rm at substep start, post-halo-fill
    panels_m     :: NTuple{6, P}     # m  at substep start, post-halo-fill
    panels_q_buf_phase2 :: NTuple{6, P}   # state B (post-phase-1 pre_advect_y)
    panels_q_buf_phase3 :: NTuple{6, P}   # state C (post-phase-2 pre_advect_x)
    panels_fx_in  :: NTuple{6, A3x}
    panels_fx_out :: NTuple{6, A3x}
    panels_fy_in  :: NTuple{6, A3y}
    panels_am     :: NTuple{6, A3x}
    panels_bm     :: NTuple{6, A3y}
    flux_scale    :: FT
end

# The production Lin-Rood operator (`fv_tp_2d_cs!`) builds the record through
# its recorder hooks (`record_linrood_horizontal!`, Footprint/TapeRecorder.jl):
# rm and m after the halo and corner fills, the predictor buffer after phases
# 1 and 2, the inner faces before and the outer X faces after the seam
# projection, and the interior face fluxes it read.


# Reverse of one LinRood horizontal substep. Mutates the `lambda_panels_rm`,
# `lambda_panels_m` adjoint accumulators IN PLACE. The record's `ORD` type
# parameter selects the face-kernel adjoint variant (Val(5) or Val(7))
# so the reverse pass matches the forward path's panel-edge boundary
# behaviour.
function _apply_cs_linrood_horizontal_adjoint!(
    lambda_panels_rm, lambda_panels_m,
    record::_CSLinRoodHorizRecord{FT, A3, A3x, A3y, P, ORD},
    mesh::CubedSphereMesh{FT},
) where {FT, A3, A3x, A3y, P, ORD}
    # Reverse the final update on all panels before reversing their shared
    # face projection. A seam couples seeds from both neighboring cells.
    input_lambda_rm = map(a -> fill!(similar(a), zero(FT)), lambda_panels_rm)
    input_lambda_m = map(a -> fill!(similar(a), zero(FT)), lambda_panels_m)
    face_seeds = map((record.panels_fx_in, record.panels_fx_in,
                      record.panels_fy_in, record.panels_fy_in)) do panels
        map(a -> fill!(similar(a), zero(FT)), panels)
    end
    for p in 1:6
        apply_linrood_update_adjoint!(
            input_lambda_rm[p], input_lambda_m[p],
            face_seeds[1][p], face_seeds[2][p], face_seeds[3][p], face_seeds[4][p],
            lambda_panels_rm[p], lambda_panels_m[p],
            record.panels_am[p], record.panels_bm[p], mesh,
        )
    end
    _share_lr_seam_faces!(face_seeds..., mesh)

    # Reverse the remaining per-panel reconstruction and predictor stages.
    for p in 1:6
        sub_lambda_rm = input_lambda_rm[p]
        sub_lambda_m = input_lambda_m[p]
        # The substep adjoint maps lambda(rm_new, m_new) — which is the
        # CURRENT lambda_panels_rm[p], lambda_panels_m[p] — into
        # (sub_lambda_rm, sub_lambda_m), the adjoint w.r.t. the
        # substep INPUT state (rm0, m0 of the substep).
        apply_linrood_horizontal_adjoint_single_panel!(
            sub_lambda_rm, sub_lambda_m,
            lambda_panels_rm[p], lambda_panels_m[p],
            record.panels_rm[p], record.panels_m[p],
            record.panels_am[p], record.panels_bm[p],
            record.panels_q_buf_phase2[p], record.panels_q_buf_phase3[p],
            record.panels_fx_in[p], record.panels_fx_out[p],
            record.panels_fy_in[p],
            mesh, Val(ORD);
            face_adjoints=ntuple(i -> face_seeds[i][p], 4),
        )
        # Carry-over: substep output's halo lambda is NOT overwritten
        # by the substep update (which only touches interior cells).
        # Add the halo carry from the OUTPUT lambda back into the
        # substep-input adjoint.
        _accumulate_linrood_halo_adjoint!(
            sub_lambda_rm, sub_lambda_m, lambda_panels_rm[p], lambda_panels_m[p], mesh)
        # Replace the running lambda with the substep-input adjoint.
        copyto!(lambda_panels_rm[p], sub_lambda_rm)
        copyto!(lambda_panels_m[p],  sub_lambda_m)
    end

    # Step 2: cross-panel halo adjoint. The forward path filled halos at
    # the start of the substep; the reverse aggregates each panel's
    # halo lambda contributions into the corresponding neighbour
    # panel's interior cells.
    _adjoint_fill_panel_halos!(lambda_panels_rm, mesh; dir=0)
    _adjoint_fill_panel_halos!(lambda_panels_m,  mesh; dir=0)
    return nothing
end
