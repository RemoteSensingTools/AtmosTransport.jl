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
#   * `_record_cs_linrood_tape` — forward recording function that
#     replays `fv_tp_2d_cs!` phase-by-phase, captures snapshots, and
#     returns the operations list + final air-mass panels.
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
#     * Forward replay is bit-identical to production `fv_tp_2d_cs!` — the
#       single-H comparator (`_record_linrood_horizontal_substep!`,
#       record_ops=false, vs `fv_tp_2d_cs!` on identical stripped fluxes) and
#       the full-step comparator (vs `_linrood_run_forward_step!`) both give 0.0.
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
#       downward flux). The ~1e-8 had been read as FD noise. Z sweeps now go
#       through `_record_and_sweep_z!` (TapeRecording.jl), which records first.
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

_scale_linrood_flux_panels(panels::NTuple{6}, flux_scale) =
    isone(flux_scale) ? panels :
    ntuple(p -> flux_scale .* panels[p], Val(6))

# Run one LinRood horizontal substep across all six panels, replicating
# the forward `fv_tp_2d_cs!` (linrood_horizontal.jl). Updates `panels_rm`,
# `panels_m` in place. With `record_ops = true` (default) captures
# per-phase snapshots and returns a `_CSLinRoodHorizRecord` for the
# reverse pass; with `record_ops = false` (used by the strided
# checkpoint propagation pass) skips every state
# snapshot and returns `nothing`, leaving only the face / q_buf
# scratch buffers that are required to run the kernels themselves
# (they go out of scope at function exit, so peak memory is bounded
# by one substep's worth of scratch rather than the full tape).
function _record_linrood_horizontal_substep!(
    panels_rm, panels_m,
    panels_am, panels_bm,
    mesh::CubedSphereMesh{FT},
    flux_scale,
    ord::Val{ORD} = Val(DEFAULT_LINROOD_PPM_ORDER);
    record_ops::Bool = true,
) where {FT, ORD}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    N = Nc + 2 * Hp
    backend = get_backend(panels_rm[1])

    # Strip the driver's Hp halo from the flux panels so the forward replay AND
    # the stored tape (read back by `_apply_cs_linrood_horizontal_adjoint!`) use
    # the same interior faces the corrected production forward does. The record's
    # typed `A3x`/`A3y` flux fields materialise these views into owned copies on
    # construction, so the reverse pass reads stable data. No-op for unpadded
    # callers. Must match the strip in `_linrood_run_forward_step!` (FD path) and
    # `_cs_transport_step!(::CSLinRoodStyle)` (production) — otherwise the FD,
    # forward-replay, and adjoint paths would disagree by a Hp-cell flux shift.
    panels_am = ntuple(p -> _cs_flux_x_interior(panels_am[p], Nc, Hp), Val(6))
    panels_bm = ntuple(p -> _cs_flux_y_interior(panels_bm[p], Nc, Hp), Val(6))
    fs = FT(flux_scale)
    panels_am = _scale_linrood_flux_panels(panels_am, fs)
    panels_bm = _scale_linrood_flux_panels(panels_bm, fs)

    init_k!    = _init_q_buf_kernel!(backend, 256)
    y_face_k!  = _ppm_y_face_kernel!(backend, 256)
    x_face_k!  = _ppm_x_face_kernel!(backend, 256)
    xq_face_k! = _ppm_x_face_from_q_kernel!(backend, 256)
    yq_face_k! = _ppm_y_face_from_q_kernel!(backend, 256)
    pre_y_k!   = _pre_advect_y_kernel!(backend, 256)
    pre_x_k!   = _pre_advect_x_kernel!(backend, 256)
    update_k!  = _linrood_update_kernel!(backend, 256)

    # ── Halo fills (pre-phase-1) ─────────────────────────────────────
    fill_panel_halos!(panels_rm, mesh)
    fill_panel_halos!(panels_m,  mesh)
    copy_corners!(panels_rm, mesh, 2)
    copy_corners!(panels_m,  mesh, 2)

    # Capture rm/m state at the start of the substep (post halo + corner).
    # Only allocated in recording mode — propagation mode (`record_ops =
    # false`) skips these snapshots since the reverse pass doesn't run.
    panels_rm_tape = record_ops ? ntuple(p -> copy(panels_rm[p]), Val(6)) : nothing
    panels_m_tape  = record_ops ? ntuple(p -> copy(panels_m[p]),  Val(6)) : nothing

    # Allocate per-panel face / q_buf buffers.
    panels_fy_in  = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (Nc, Nc + 1, Nz)); fill!(b, zero(FT)); b
    end, Val(6))
    panels_fy_out = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (Nc, Nc + 1, Nz)); fill!(b, zero(FT)); b
    end, Val(6))
    panels_fx_in  = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (Nc + 1, Nc, Nz)); fill!(b, zero(FT)); b
    end, Val(6))
    panels_fx_out = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (Nc + 1, Nc, Nz)); fill!(b, zero(FT)); b
    end, Val(6))
    panels_q_buf  = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (N, N, Nz)); fill!(b, zero(FT)); b
    end, Val(6))

    # ── Phase 1: init q_buf, y_face, pre_y ───────────────────────────
    for p in 1:6
        init_k!(panels_q_buf[p], panels_rm[p], panels_m[p];
                ndrange=(N, N, Nz))
    end
    synchronize(backend)
    for p in 1:6
        y_face_k!(panels_fy_in[p], panels_rm[p], panels_m[p], panels_bm[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        pre_y_k!(panels_q_buf[p], panels_rm[p], panels_m[p], panels_bm[p],
                 panels_fy_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # Snapshot q_buf state B (post-phase-1). Skipped in propagation mode.
    panels_q_buf_phase2 = record_ops ?
        ntuple(p -> copy(panels_q_buf[p]), Val(6)) : nothing

    # ── Phase 2: x-corners, xq_face / x_face, re-init, pre_x ─────────
    copy_corners!(panels_q_buf, mesh, 1)
    copy_corners!(panels_rm,    mesh, 1)
    copy_corners!(panels_m,     mesh, 1)

    for p in 1:6
        xq_face_k!(panels_fx_out[p], panels_q_buf[p], panels_am[p], panels_m[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
        x_face_k!(panels_fx_in[p], panels_rm[p], panels_m[p], panels_am[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    end
    synchronize(backend)

    for p in 1:6
        init_k!(panels_q_buf[p], panels_rm[p], panels_m[p];
                ndrange=(N, N, Nz))
    end
    synchronize(backend)
    for p in 1:6
        pre_x_k!(panels_q_buf[p], panels_rm[p], panels_m[p], panels_am[p],
                 panels_fx_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # Snapshot q_buf state C (post-phase-2). Skipped in propagation mode.
    panels_q_buf_phase3 = record_ops ?
        ntuple(p -> copy(panels_q_buf[p]), Val(6)) : nothing

    # ── Phase 3: y-corners, yq_face, update ──────────────────────────
    copy_corners!(panels_q_buf, mesh, 2)

    for p in 1:6
        yq_face_k!(panels_fy_out[p], panels_q_buf[p], panels_bm[p], panels_m[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
    end
    synchronize(backend)

    # The transverse predictors used the unsynchronized inner face values.
    # Preserve those inputs for their reverse pass before sharing seam faces.
    panels_fx_in_tape = record_ops ? map(copy, panels_fx_in) : nothing
    panels_fy_in_tape = record_ops ? map(copy, panels_fy_in) : nothing
    _share_lr_seam_faces!(panels_fx_in, panels_fx_out, panels_fy_in, panels_fy_out, mesh)

    # Apply the update in-place using a temporary destination.
    rm_buf = similar(panels_rm[1]); fill!(rm_buf, zero(FT))
    m_buf  = similar(panels_m[1]);  fill!(m_buf,  zero(FT))
    for p in 1:6
        update_k!(rm_buf, m_buf,
                  panels_rm[p], panels_m[p], panels_am[p], panels_bm[p],
                  panels_fx_in[p], panels_fx_out[p],
                  panels_fy_in[p], panels_fy_out[p], Hp;
                  ndrange=(Nc, Nc, Nz))
        synchronize(backend)
        _copy_interior!(panels_rm[p], rm_buf, Nc, Hp, Nz)
        _copy_interior!(panels_m[p], m_buf, Nc, Hp, Nz)
    end

    record_ops || return nothing

    A3  = typeof(panels_rm_tape[1])
    A3x = typeof(panels_fx_in[1])
    A3y = typeof(panels_fy_in[1])
    P   = A3
    return _CSLinRoodHorizRecord{FT, A3, A3x, A3y, P, ORD}(
        panels_rm_tape, panels_m_tape,
        panels_q_buf_phase2, panels_q_buf_phase3,
        panels_fx_in_tape, panels_fx_out, panels_fy_in_tape,
        panels_am, panels_bm,
        fs,
    )
end


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

# ---------------------------------------------------------------------------
# Top-level tape recording for LinRoodPPMScheme. Records one LinRood
# horizontal record + Z sweeps per substep, plus optional diffusion +
# convection records (matching the existing CS tracer-tape contract).
# ---------------------------------------------------------------------------
function _record_cs_linrood_tape(panels_rm0, panels_m0,
                                  panels_am_steps, panels_bm_steps,
                                  panels_cm_steps,
                                  mesh::CubedSphereMesh{FT},
                                  scheme::LinRoodPPMScheme{ORD};
                                  flux_scale = one(FT),
                                  dt = one(FT),
                                  cfl_limit = DEFAULT_CFL_LIMIT,
                                  base_emission_rates = nothing,
                                  diffusion_op = NoDiffusion(),
                                  diffusion_workspace = nothing,
                                  diffusion_meteo = nothing,
                                  convection_op = NoConvection(),
                                  convection_forcing = nothing,
                                  convection_workspace = nothing,
                                  tape_storage = :device,
                                  step_offset::Int = 0,
                                  record_ops::Bool = true) where {FT, ORD}
    _ = cfl_limit  # LinRood doesn't subcycle horizontally — single substep per step

    # `step_offset` and `record_ops` support
    # strided checkpointing — `step_offset` shifts `_CSMidpointRecord`
    # indices into absolute step numbers for window invocations;
    # `record_ops = false` is the propagation pass that runs every
    # forward kernel (horizontal substep, Z half-sweeps, diffusion,
    # emissions, convection) but elides each `_stage_panels` /
    # `push!(ops, ...)` site. Default 0 / true keeps the FullCheckpoint
    # path bit-exact.

    # ORD ∈ {5, 7}: the adjoint kernels carry `Val(ORD)` end-to-end
    # (record type binds it; `_apply_cs_linrood_horizontal_adjoint!`
    # reads it from the record). The ORD=7 reverse pass applies the
    # cube-edge values of the faces at and next to the panel edges
    # (`_ppm_edge_values_ord7`) so the tape and the FD reference match.
    (ORD == 5 || ORD == 7) || throw(ArgumentError(
        "LinRoodPPMScheme adjoint tape supports ORD ∈ {5, 7}; got " *
        "ORD=$(ORD)."))

    # The `_CSLinRoodHorizRecord` struct holds raw `NTuple{6, P}`
    # references rather than per-policy slots, so the LinRood tape is
    # device-resident. Until the storage policy is plumbed through that
    # record, any non-`:device` storage request would be silently
    # ignored, leaving the mmap tape with `cursor=0, records=0` and
    # the LinRood tape entirely device-resident — a latent OOM trap
    # for large LinRood footprints. Reject explicitly so the failure
    # mode is loud. Skipped in propagation mode (no tape needed).
    record_ops && _linrood_validate_tape_storage(tape_storage)
    storage = record_ops ? _tape_storage(tape_storage) : nothing
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    dt_ft = FT(dt)

    # Mutable copies of the panel state — updated in place by the
    # forward replay.
    panels_rm = ntuple(p -> copy(panels_rm0[p]), Val(6))
    panels_m  = ntuple(p -> copy(panels_m0[p]),  Val(6))

    ops = Any[]
    ws = CSAdvectionWorkspace(mesh, panels_rm[1];
                              column_scratch = needs_column_scratch(scheme))

    # The Z (vertical) sweep MUST match the production LinRood forward: both
    # `_strang_split_linrood_ppm_cs!` and the FD reference `_linrood_run_forward_step!`
    # sweep with `scheme.vertical` through `_sweep_z_panels!`. Recording a
    # different Z scheme makes the adjoint the transpose of a DIFFERENT forward
    # (once: a monotone-PPM Z recorded under an upwind forward gave a ~0.14%
    # footprint-vs-FD error as soon as horizontal transport built vertical
    # structure). `UpwindScheme ∈ CSAdjointLinearScheme` stages only the air mass;
    # the FV3 profile (`CSAdjointFV3Vertical`) also stages the tracer mass entering
    # the sweep, from which the reverse replays the limiter branches.
    z_scheme = scheme.vertical

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]

        # Production LinRood Strang palindrome (linrood_horizontal.jl,
        # `_strang_split_linrood_ppm_cs!`):
        #     H → Z_half → center (surface_flux_center!) → Z_half → H
        # No step-start halo record: H refills the halos it reads.
        # The tape mirrors this exactly so that the FD-reference forward
        # in `_linrood_run_forward_step!` and the recorded forward are
        # the same operator (up to numerical rounding).

        absolute_step = step + step_offset

        # LinRood horizontal substep (first half of the palindrome).
        record_a = _record_linrood_horizontal_substep!(
            panels_rm, panels_m, panels_am, panels_bm, mesh, FT(flux_scale), Val(ORD);
            record_ops = record_ops)
        record_ops && push!(ops, record_a)

        # Z half-sweep. The record holds the state entering the sweep: the reverse
        # coefficients divide by the pre-sweep donor mass (issue #10), and the FV3
        # profile replays its limiter branches from the pre-sweep tracer mass.
        _record_and_sweep_z!(ops, z_scheme, panels_rm, panels_m, panels_cm, ws, mesh,
                             FT(flux_scale), storage; record_ops)

        # Palindrome center between the two Z halves: diffusion and the
        # emission midpoint in the coupling's order (`surface_flux_center!`).
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                    "diffusion_op")
        diffusion_ws_step = diffusion_op_step isa NoDiffusion ? nothing :
            _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                   "diffusion_workspace")
        _record_and_apply_center!(ops, panels_rm, panels_m, mesh, storage, dt_ft,
                                  absolute_step, diffusion_op_step, diffusion_ws_step,
                                  diffusion_meteo,
                                  base_emission_rates === nothing ? nothing :
                                      base_emission_rates[step];
                                  record_ops)

        # Z half-sweep (second half), recorded before it is applied.
        _record_and_sweep_z!(ops, z_scheme, panels_rm, panels_m, panels_cm, ws, mesh,
                             FT(flux_scale), storage; record_ops)

        # LinRood horizontal substep (second half of the palindrome).
        record_b = _record_linrood_horizontal_substep!(
            panels_rm, panels_m, panels_am, panels_bm, mesh, FT(flux_scale), Val(ORD);
            record_ops = record_ops)
        record_ops && push!(ops, record_b)

        # Convection (optional, post-transport).
        _record_and_apply_convection!(ops, panels_rm, panels_m, mesh, storage, dt_ft,
                                      convection_op,
                                      _step_convection_forcing(convection_op,
                                                               convection_forcing,
                                                               step, nsteps),
                                      convection_workspace; record_ops)
    end

    return ops, panels_rm, panels_m
end

# ---------------------------------------------------------------------------
# Forward driver for the FD-reference path inside `_run_cs_footprint_forward`
# / `_run_cs_observations_forward`. The standard `strang_split_cs!` doesn't
# know how to dispatch LinRoodPPMScheme (no per-direction face kernels);
# this helper bridges to `strang_split_linrood_ppm!` which IS the right
# forward for LinRood.
# ---------------------------------------------------------------------------
function _linrood_run_forward_step!(panels_rm, panels_m,
                                     panels_am, panels_bm, panels_cm,
                                     mesh::CubedSphereMesh{FT},
                                     scheme::LinRoodPPMScheme{ORD},
                                     ws::CSAdvectionWorkspace,
                                     midpoint!;
                                     flux_scale = one(FT)) where {FT, ORD}
    Nz = size(panels_rm[1], 3)
    # The footprint/inversion driver passes Hp-padded flux panels, but the
    # LinRood kernels index the interior faces; strip the halo so this
    # FD-reference forward runs the same (corrected) transport the production
    # `_cs_transport_step!(::CSLinRoodStyle)` and the tape do. No-op for unpadded
    # callers. See `_cs_flux_x_interior` / `_cs_flux_y_interior`.
    Nc, Hp = mesh.Nc, mesh.Hp
    panels_am = ntuple(p -> _cs_flux_x_interior(panels_am[p], Nc, Hp), Val(6))
    panels_bm = ntuple(p -> _cs_flux_y_interior(panels_bm[p], Nc, Hp), Val(6))
    fs = FT(flux_scale)
    panels_am = _scale_linrood_flux_panels(panels_am, fs)
    panels_bm = _scale_linrood_flux_panels(panels_bm, fs)
    panels_cm = _scale_linrood_flux_panels(panels_cm, fs)
    array_type = typeof(parent(panels_rm[1]))
    ws_lr = LinRoodWorkspace(mesh; FT = FT, Nz = Nz, array_type = array_type)
    # Palindrome H → Z → (midpoint/emissions) → Z → H, matching
    # `_strang_split_linrood_ppm_cs!`, with the scheme's vertical sweep (`ws`
    # carries the column scratch the FV3 profile needs).
    fv_tp_2d_cs!(panels_rm, panels_m, panels_am, panels_bm, mesh, Val(ORD), ws, ws_lr)
    _sweep_z_panels!(panels_rm, panels_m, panels_cm, mesh, scheme.vertical, ws)
    if midpoint! !== nothing
        midpoint!()
    end
    _sweep_z_panels!(panels_rm, panels_m, panels_cm, mesh, scheme.vertical, ws)
    fv_tp_2d_cs!(panels_rm, panels_m, panels_am, panels_bm, mesh, Val(ORD), ws, ws_lr)
    return nothing
end
