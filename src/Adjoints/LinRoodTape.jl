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
#     * The full footprint identity matches FD to ~1e-8 for BOTH ORD=5 and
#       ORD=7, at every nsteps, on a SMOOTH or CONSTANT base field. Only the
#       non-smooth zero-IC case shows the ~1.4e-3 kink artifact.
#   The split-sweep (PPM/Upwind/Slopes) footprint test validates at a zero IC
#   precisely because those schemes are NoLimiter (linear). The LinRood adjoint
#   is production-ready for 4D-Var; FD validation must use a smooth base field
#   (standard practice for limited-scheme adjoints — see
#   test/diagnostic/test_linrood_adjoint_integration.jl `_seed_smooth_cs_ic!`).
#   * ORD ∈ {5, 7} (LinRoodPPMScheme(5) and LinRoodPPMScheme(7) are both
#     supported). `_CSLinRoodHorizRecord`
#     binds ORD as a type parameter; the reverse pass reads it via
#     dispatch and forwards `Val(ORD)` to the face-kernel adjoints.
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
    ord::Val{ORD} = Val(5);
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
                                  cfl_limit = 0.95,
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
    # emissions, convection) but elides each `_stage_panels_strict` /
    # `push!(ops, ...)` site. Default 0 / true keeps the FullCheckpoint
    # path bit-exact.

    # ORD ∈ {5, 7}: the adjoint kernels carry `Val(ORD)` end-to-end
    # (record type binds it; `_apply_cs_linrood_horizontal_adjoint!`
    # reads it from the record). The ORD=7 reverse-pass applies the
    # discontinuous-edge boundary correction at panel-edge faces
    # (`face_idx ∈ {1, Nc+1}`) so the tape and the FD reference match.
    (ORD == 5 || ORD == 7) || throw(ArgumentError(
        "LinRoodPPMScheme adjoint tape supports ORD ∈ {5, 7}; got " *
        "ORD=$(ORD)."))

    # LinRoodPPMScheme stages its forward state through
    # `_stage_panels_strict` (which hardcodes `DeviceCSTapeStorage()`)
    # — the `_CSLinRoodHorizRecord` struct holds raw `NTuple{6, P}`
    # references rather than per-policy slots. Until the LinRood tape
    # is refactored to plumb the storage policy through, any
    # non-`:device` storage request would be silently
    # ignored, leaving the mmap tape with `cursor=0, records=0` and
    # the LinRood tape entirely device-resident — a latent OOM trap
    # for large LinRood footprints. Reject explicitly so the failure
    # mode is loud. Skipped in propagation mode (no tape needed).
    record_ops && _linrood_validate_tape_storage(tape_storage)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    dt_ft = FT(dt)

    # Mutable copies of the panel state — updated in place by the
    # forward replay.
    panels_rm = ntuple(p -> copy(panels_rm0[p]), Val(6))
    panels_m  = ntuple(p -> copy(panels_m0[p]),  Val(6))

    ops = Any[]
    ws = CSAdvectionWorkspace(mesh, panels_rm[1])
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_rm[1], 3)

    # The Z (vertical) sweep MUST match the production LinRood forward: both
    # `_strang_split_linrood_ppm_cs!` and the FD reference `_linrood_run_forward_step!`
    # do the vertical sweep via `_sweep_z!`, which uses `UpwindScheme()`
    # (CubedSphereStrang.jl). Recording a different Z scheme here (the old
    # `PPMScheme(MonotoneLimiter())`) makes the adjoint the transpose of a
    # DIFFERENT forward — masked while horizontal fluxes were trivial (the field
    # stayed vertically uniform, where Upwind and PPM Z agree), but a ~0.14%
    # footprint-vs-FD error once real horizontal transport develops vertical
    # structure. Use Upwind so the tape, the FD reference, and production agree.
    z_scheme = UpwindScheme()

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]

        # Production LinRood Strang palindrome (linrood_horizontal.jl,
        # `_strang_split_linrood_ppm_cs!`):
        #     H → Z_half → midpoint/diffusion/emissions → Z_half → H
        # The tape mirrors this exactly so that the FD-reference forward
        # in `_linrood_run_forward_step!` and the recorded forward are
        # the same operator (up to numerical rounding).

        absolute_step = step + step_offset

        # LinRood horizontal substep (first half of the palindrome).
        record_a = _record_linrood_horizontal_substep!(
            panels_rm, panels_m, panels_am, panels_bm, mesh, FT(flux_scale), Val(ORD);
            record_ops = record_ops)
        record_ops && push!(ops, record_a)

        # Z half-sweep.
        for p in 1:6
            _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                            z_scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz;
                            flux_scale = FT(flux_scale))
        end
        if record_ops
            # `UpwindScheme ∈ CSAdjointLinearScheme`: its Z adjoint is linear, so
            # record `nothing` for rm (the 4-arg linear reverse path). Storing rm
            # would route to the `PPMScheme{MonotoneLimiter}` 5-arg reverse, which
            # rejects Upwind.
            push!(ops, _CSSweepRecord(:z, z_scheme,
                                      _stage_panels_strict(panels_m),
                                      nothing,
                                      panels_cm, FT(flux_scale)))
        end

        # Diffusion + midpoint + emissions (between the two Z halves).
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                    "diffusion_op")
        if diffusion_op_step isa NoDiffusion
            record_ops && push!(ops, _CSMidpointRecord(absolute_step))
            base_emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, base_emission_rates[step], dt_ft, mesh)
        else
            diffusion_ws_step = _diffusion_sequence_at(diffusion_workspace, step,
                                                       nsteps,
                                                       "diffusion_workspace")
            half_dt = dt_ft / FT(2)
            if record_ops
                panels_m_midpoint = _stage_panels_strict(panels_m)
                push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                              panels_m_midpoint, half_dt))
            end
            apply_vertical_diffusion_vmr!(
                panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                half_dt, diffusion_meteo; halo_width = mesh.Hp)
            record_ops && push!(ops, _CSMidpointRecord(absolute_step))
            base_emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, base_emission_rates[step], dt_ft, mesh)
            if record_ops
                push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                              panels_m_midpoint, half_dt))
            end
            apply_vertical_diffusion_vmr!(
                panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                half_dt, diffusion_meteo; halo_width = mesh.Hp)
        end

        # Z half-sweep (second half).
        for p in 1:6
            _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                            z_scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz;
                            flux_scale = FT(flux_scale))
        end
        if record_ops
            # `UpwindScheme ∈ CSAdjointLinearScheme`: its Z adjoint is linear, so
            # record `nothing` for rm (the 4-arg linear reverse path). Storing rm
            # would route to the `PPMScheme{MonotoneLimiter}` 5-arg reverse, which
            # rejects Upwind.
            push!(ops, _CSSweepRecord(:z, z_scheme,
                                      _stage_panels_strict(panels_m),
                                      nothing,
                                      panels_cm, FT(flux_scale)))
        end

        # LinRood horizontal substep (second half of the palindrome).
        record_b = _record_linrood_horizontal_substep!(
            panels_rm, panels_m, panels_am, panels_bm, mesh, FT(flux_scale), Val(ORD);
            record_ops = record_ops)
        record_ops && push!(ops, record_b)

        # Convection (optional, post-transport).
        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            if record_ops
                push!(ops, _CSConvectionRecord(convection_op, forcing_step,
                                               _stage_panels_strict(panels_m),
                                               dt_ft))
            end
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt_ft,
                                          convection_workspace, mesh)
        end
    end

    return ops, panels_rm, panels_m
end

# Internal helper: build a DeviceCSTapeStorage-staged version of
# panels for the tape. The existing `_stage_panels(storage, panels)`
# requires a `storage` argument; we don't have one in scope for the
# LinRood path because the standalone API doesn't expose it. Stage
# strictly: just copy in-place onto the same backend.
function _stage_panels_strict(panels::NTuple{6})
    return _stage_panels(DeviceCSTapeStorage(), panels)
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
    # `_strang_split_linrood_ppm_cs!`.
    fv_tp_2d_cs!(panels_rm, panels_m, panels_am, panels_bm, mesh, Val(ORD), ws, ws_lr)
    _sweep_z!(panels_rm, panels_m, panels_cm, mesh, ws)
    if midpoint! !== nothing
        midpoint!()
    end
    _sweep_z!(panels_rm, panels_m, panels_cm, mesh, ws)
    fv_tp_2d_cs!(panels_rm, panels_m, panels_am, panels_bm, mesh, Val(ORD), ws, ws_lr)
    return nothing
end
