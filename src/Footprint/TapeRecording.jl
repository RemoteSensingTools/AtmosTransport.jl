# ---------------------------------------------------------------------------
# Forward-tape recording for the CS adjoint pipeline.
#
# Collects the per-op records the forward pass appends to the tape:
#
#   * `_tape_byte_estimate` — diagnostic sizing.
#   * `_record_sweep!` — pushes a `_CSSweepRecord` for one per-direction
#     advection sweep (linear or monotone-PPM dispatch).
#   * `_record_and_sweep_z!` — records, then applies, one vertical sweep;
#     shared by all three tapes (including Lin-Rood) so the record always
#     holds the state entering the sweep.
#   * `_record_cs_mass_tape` — air-mass tape only (linear schemes).
#   * `_record_cs_tracer_tape` — air-mass + tracer-mass tape
#     (monotone PPM and other nonlinear schemes).
#   * `_record_cs_adjoint_tape` — top-level dispatch that picks the right
#     recorder for the requested scheme; LinRood goes through
#     `_record_cs_linrood_tape` in `LinRoodTape.jl`.
#
# Loaded into the `Adjoints` module from `Adjoints.jl`. Horizontal records
# use the same paired physical-seam groups as the production forward pass;
# their stored inputs also supply the shared reconstruction's reverse pass.
# ---------------------------------------------------------------------------

function _tape_byte_estimate(panels_m0,
                             panels_am_steps,
                             panels_bm_steps,
                             panels_cm_steps,
                             mesh::CubedSphereMesh,
                             scheme::CSAdjointLinRoodScheme;
                             cfl_limit = DEFAULT_CFL_LIMIT,
                             flux_scale = one(eltype(panels_m0[1])),
                             diffusion_op = NoDiffusion(),
                             convection_op = NoConvection())
    _ = (mesh, scheme, cfl_limit, flux_scale)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                      panels_cm_steps)

    # Each Lin-Rood step records two horizontal operators and two vertical
    # sweeps. A horizontal record retains four full-panel states (rm, m, and
    # the two q_buf phases) plus three face-panel tuples (fx_in, fx_out,
    # fy_in); a vertical sweep retains the full-panel m state (upwind) or m and
    # rm (FV3 profile). Thus its base tape is ten (upwind) or twelve (FV3)
    # full-panel tuples and six face-panel tuples per step, not the split-sweep
    # layout used by the generic method.
    sweep_records = 4nsteps
    halo_records = 0
    midpoint_records = nsteps
    diffusion_records = 0
    diffusion_state_records = 0
    @inbounds for step in 1:nsteps
        if !(_diffusion_sequence_at(diffusion_op, step, nsteps,
                                    "diffusion_op") isa NoDiffusion)
            diffusion_records += 2
            diffusion_state_records += 1
        end
    end
    convection_records = convection_op isa NoConvection ? 0 : nsteps
    state_records = (8 + 2 * _staged_tuples_per_sweep(scheme.vertical)) * nsteps +
                    diffusion_state_records + convection_records
    total_records = sweep_records + midpoint_records + diffusion_records +
                    convection_records
    bytes_per_state = _bytes_per_panel_tuple(panels_m0)
    Nc = mesh.Nc
    Nz = size(panels_m0[1], 3)
    bytes_per_face_tuple = 6 * (Nc + 1) * Nc * Nz * sizeof(eltype(panels_m0[1]))
    state_bytes = state_records * bytes_per_state +
                  6nsteps * bytes_per_face_tuple
    return CSTapeByteEstimate(nsteps, sweep_records, halo_records,
                              midpoint_records, diffusion_records,
                              convection_records, state_records,
                              total_records, bytes_per_state,
                              state_bytes)
end

function _tape_byte_estimate(panels_m0,
                             panels_am_steps,
                             panels_bm_steps,
                             panels_cm_steps,
                             mesh::CubedSphereMesh,
                             scheme::CSAdjointSupportedScheme;
                             cfl_limit = DEFAULT_CFL_LIMIT,
                             flux_scale = one(eltype(panels_m0[1])),
                             diffusion_op = NoDiffusion(),
                             convection_op = NoConvection())
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                      panels_cm_steps)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    dummy_rm = ntuple(p -> similar(panels_m[p]), 6)
    @inbounds for p in 1:6
        fill!(dummy_rm[p], zero(FT))
    end
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    cfl_ft = FT(cfl_limit)
    fs = FT(flux_scale)

    sweep_records = 0
    halo_records = 0
    midpoint_records = 0
    diffusion_records = 0
    diffusion_state_records = 0
    convection_records = 0

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_pal = _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)

        # (X Y Z)^n | midpoint | (Z Y X)^n, as `strang_split_cs!`: 6n sweeps;
        # halo fills: 2 per forward pass plus one before each later X pass,
        # 4 per reverse pass.
        sweep_records += 6n_pal
        halo_records += 7n_pal - 1
        midpoint_records += 1
        if !(_diffusion_sequence_at(diffusion_op, step, nsteps,
                                    "diffusion_op") isa NoDiffusion)
            diffusion_records += 2
            diffusion_state_records += 1
        end
        convection_op isa NoConvection || (convection_records += 1)

        fs_sub = fs / FT(n_pal)

        for pass in 1:n_pal
            pass > 1 && fill_panel_halos!(panels_m, mesh; dir=1)
            _sweep_cs_horizontal!(dummy_rm, panels_m, panels_am, mesh,
                                  scheme, ws, Val(1); flux_scale=fs_sub)
            fill_panel_halos!(panels_m, mesh; dir=1)
            _sweep_cs_horizontal!(dummy_rm, panels_m, panels_bm, mesh,
                                  scheme, ws, Val(2); flux_scale=fs_sub)
            fill_panel_halos!(panels_m, mesh; dir=2)
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_sub)
            end
        end
        for _ in 1:n_pal
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_sub)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
            _sweep_cs_horizontal!(dummy_rm, panels_m, panels_bm, mesh,
                                  scheme, ws, Val(2); flux_scale=fs_sub)
            fill_panel_halos!(panels_m, mesh; dir=2)
            fill_panel_halos!(panels_m, mesh; dir=1)
            _sweep_cs_horizontal!(dummy_rm, panels_m, panels_am, mesh,
                                  scheme, ws, Val(1); flux_scale=fs_sub)
            fill_panel_halos!(panels_m, mesh; dir=1)
        end
    end

    # `sweep_state_records` counts panel STAGINGS (m for linear,
    # m + rm for nonlinear). `state_records` is the total number of
    # staged panel tuples on the tape — i.e. how many full panel
    # snapshots get written. Diffusion stores ONE staged m-panel
    # tuple per step but contributes TWO `_CSDiffusionRecord` ops
    # (the palindrome's `V(dt/2) → emissions → V(dt/2)`); see
    # `_record_cs_mass_tape` for the staging path.
    sweep_state_records = _staged_tuples_per_sweep(scheme) * sweep_records
    state_records = sweep_state_records + diffusion_state_records + convection_records

    # `total_records` is the total number of OPS the tape will hold
    # (every op the reverse loop dispatches on). Sum of all op
    # counts — sweep + halo + midpoint + diffusion + convection —
    # NOT `state_records + halo + midpoint`, since `state_records`
    # double-counts nonlinear staging vs op count and under-counts
    # the diffusion palindrome's two ops sharing one staged state.
    total_records = sweep_records + halo_records + midpoint_records +
                    diffusion_records + convection_records
    bytes_per_state = _bytes_per_panel_tuple(panels_m0)
    return CSTapeByteEstimate(nsteps, sweep_records, halo_records,
                              midpoint_records, diffusion_records,
                              convection_records, state_records,
                              total_records, bytes_per_state,
                              state_records * bytes_per_state)
end

"""
    cs_tape_byte_estimate(panels_m0, panels_am_steps, panels_bm_steps,
                          panels_cm_steps, mesh, scheme;
                          cfl_limit = DEFAULT_CFL_LIMIT,
                          flux_scale = one(eltype(panels_m0[1])),
                          diffusion_op = NoDiffusion(),
                          convection_op = NoConvection())
        -> CSTapeByteEstimate

Statically size the tape that `cs_surface_emission_footprint` will produce
for the given problem. Reports per-record-class counts and the total
`state_bytes` — the raw panel-data cost of the tape, independent of
storage policy.

The same `state_bytes` figure applies to all three storage policies
because none of them compress the payload:

  * `tape_storage = :device` — `state_bytes` is the device-resident
    RAM cost; the reverse loop holds the full tape on the backend.
  * `tape_storage = :pinned_host` — `state_bytes` is the pinned host
    RAM cost; only one staged panel set is mirrored on the device at
    a time via the shared read cache.
  * `tape_storage = :mmap` — `state_bytes` is the on-disk footprint
    of `records.bin`. The shape-keyed device cache holds at most one
    `NTuple{6, T}` per distinct shape signature on top of that.

`bytes_per_state` is the size of one full-panel state tuple. For split-sweep
tapes, multiply it by `state_records` to recover `state_bytes`. Lin-Rood's
`state_bytes` additionally includes the six retained face-panel tuples per
step, whose dimensions differ from a full state. Halo / midpoint records carry
only scalar metadata and are counted in `total_records` but NOT in
`state_bytes`.
"""
cs_tape_byte_estimate(args...; kwargs...) =
    _tape_byte_estimate(args...; kwargs...)

# ---------------------------------------------------------------------------
# Per-step tape recorders
# ---------------------------------------------------------------------------

# Full-panel tuples a sweep record stages: the air mass, plus the tracer mass
# for the schemes whose reverse replays limiter branches.
_staged_tuples_per_sweep(::CSAdjointLinearScheme) = 1
_staged_tuples_per_sweep(::_RmTapedSweepScheme) = 2

function _record_sweep!(ops, direction::Symbol, scheme::CSAdjointLinearScheme,
                        panels_m, panels_flux, flux_scale, tape_storage)
    push!(ops, _CSSweepRecord(direction, scheme,
                              _stage_panels(tape_storage, panels_m),
                              nothing,
                              panels_flux, flux_scale))
    return nothing
end

function _record_sweep!(ops, direction::Symbol, scheme::_RmTapedSweepScheme,
                        panels_m, panels_rm, panels_flux, flux_scale, tape_storage)
    push!(ops, _CSSweepRecord(direction, scheme,
                              _stage_panels(tape_storage, panels_m),
                              _stage_panels(tape_storage, panels_rm),
                              panels_flux, flux_scale))
    return nothing
end

# A linear sweep's reverse needs only the air mass. This method accepts and
# ignores the tracer mass so code that carries `rm` can use one signature for
# every scheme family.
_record_sweep!(ops, direction::Symbol, scheme::CSAdjointLinearScheme,
               panels_m, panels_rm, panels_flux, flux_scale, tape_storage) =
    _record_sweep!(ops, direction, scheme, panels_m, panels_flux, flux_scale,
                   tape_storage)

# Record one vertical (Z) sweep, then apply it to all six panels. The mass,
# tracer and Lin-Rood tapes all run their Z sweeps through this function, so the
# record always comes before the sweep.
#
# The order matters. The reverse Z kernel (`_cs_zsweep_adjoint_kernel!`)
# rebuilds each face coefficient from the donor air mass that the forward kernel
# divided by, e.g. alpha = F / m_donor for upwind (`_upwind_face_coeffs`). That
# is the mass entering the sweep. A record taken after the sweep holds the
# updated mass m + F[k] - F[k+1] (F = flux_scale * cm) instead, which puts an
# O(C_z) relative error in every coefficient (issue #10). `panels_rm` is always
# swept; it is staged only for the taped-limiter PPM schemes and the FV3 profile
# (see the `_record_sweep!` methods above). The FV3 profile sweeps through its
# six-panel method, which needs `ws` built with column scratch.
function _record_and_sweep_z!(ops, scheme, panels_rm, panels_m, panels_cm, ws,
                              mesh::CubedSphereMesh, flux_scale, tape_storage;
                              record_ops::Bool)
    record_ops && _record_sweep!(ops, :z, scheme, panels_m, panels_rm, panels_cm,
                                 flux_scale, tape_storage)
    _sweep_z_panels!(panels_rm, panels_m, panels_cm, mesh, scheme, ws; flux_scale)
    return nothing
end

function _record_cs_mass_tape(panels_m0,
                              panels_am_steps,
                              panels_bm_steps,
                              panels_cm_steps,
                              mesh::CubedSphereMesh,
                              scheme::CSAdjointLinearScheme;
                              cfl_limit,
                              flux_scale,
                              dt,
                              diffusion_op = NoDiffusion(),
                              diffusion_workspace = nothing,
                              convection_op = NoConvection(),
                              convection_forcing = nothing,
                              tape_storage = :device,
                              step_offset::Int = 0,
                              record_ops::Bool = true)
    # `step_offset` shifts `_CSMidpointRecord(step)` indices when the
    # caller is recording one window of a strided checkpoint
    # schedule rather than the whole run. Default 0 keeps the
    # single-tape `FullCheckpoint` path bit-exact.
    #
    # `record_ops = false` is the propagation-only mode used by the
    # forward checkpoint pass in `_propagate_mass_checkpoints`: every
    # `_record_sweep!` / `_stage_panels` / `push!(ops, ...)` site is
    # short-circuited so the kernel calls (`_sweep_x/y/z_panel!`,
    # `fill_panel_halos!`) still propagate `panels_m` forward but no
    # tape storage is allocated. Diffusion and convection branches
    # never mutate `panels_m` in the mass-tape recorder (they only
    # exist to schedule the reverse-pass adjoint), so eliding their
    # record pushes preserves the forward trajectory bit-for-bit.
    FT = eltype(panels_m0[1])
    storage = record_ops ? _tape_storage(tape_storage) : nothing
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    dummy_rm = ntuple(p -> similar(panels_m[p]), 6)
    @inbounds for p in 1:6
        fill!(dummy_rm[p], zero(FT))
    end
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    ops = Any[]
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    cfl_ft = FT(cfl_limit)
    fs = FT(flux_scale)

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_pal = _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)
        fs_sub = fs / FT(n_pal)

        # Forward half (X Y Z)^n_pal, as `strang_split_cs!`.
        for pass in 1:n_pal
            if pass > 1
                fill_panel_halos!(panels_m, mesh; dir=1)
                record_ops && push!(ops, _CSHaloRecord(1))
            end
            record_ops && _record_sweep!(ops, :x, scheme, panels_m, panels_am, fs_sub, storage)
            _sweep_cs_horizontal!(dummy_rm, panels_m, panels_am, mesh,
                                  scheme, ws, Val(1); flux_scale=fs_sub)
            fill_panel_halos!(panels_m, mesh; dir=1)
            record_ops && push!(ops, _CSHaloRecord(1))

            record_ops && _record_sweep!(ops, :y, scheme, panels_m, panels_bm, fs_sub, storage)
            _sweep_cs_horizontal!(dummy_rm, panels_m, panels_bm, mesh,
                                  scheme, ws, Val(2); flux_scale=fs_sub)
            fill_panel_halos!(panels_m, mesh; dir=2)
            record_ops && push!(ops, _CSHaloRecord(2))

            _record_and_sweep_z!(ops, scheme, dummy_rm, panels_m, panels_cm, ws, mesh,
                                 fs_sub, storage; record_ops)
        end

        # Diffusion + midpoint records are pure scheduling metadata for
        # the reverse pass — the mass-tape recorder never applies
        # diffusion to `panels_m` itself. Skip the entire block (and
        # the `_stage_panels` allocation it would have triggered) when
        # `record_ops = false`.
        if record_ops
            diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                       "diffusion_op")
            absolute_step = step + step_offset
            if diffusion_op_step isa NoDiffusion
                push!(ops, _CSMidpointRecord(absolute_step))
            else
                diffusion_ws_step = _diffusion_sequence_at(diffusion_workspace, step,
                                                           nsteps,
                                                           "diffusion_workspace")
                panels_m_midpoint = _stage_panels(storage, panels_m)
                half_dt = FT(dt) / FT(2)
                push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                              panels_m_midpoint, half_dt))
                push!(ops, _CSMidpointRecord(absolute_step))
                push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                              panels_m_midpoint, half_dt))
            end
        end

        # Reverse half (Z Y X)^n_pal.
        for _ in 1:n_pal
            _record_and_sweep_z!(ops, scheme, dummy_rm, panels_m, panels_cm, ws, mesh,
                                 fs_sub, storage; record_ops)

            fill_panel_halos!(panels_m, mesh; dir=2)
            record_ops && push!(ops, _CSHaloRecord(2))
            record_ops && _record_sweep!(ops, :y, scheme, panels_m, panels_bm, fs_sub, storage)
            _sweep_cs_horizontal!(dummy_rm, panels_m, panels_bm, mesh,
                                  scheme, ws, Val(2); flux_scale=fs_sub)
            fill_panel_halos!(panels_m, mesh; dir=2)
            record_ops && push!(ops, _CSHaloRecord(2))

            fill_panel_halos!(panels_m, mesh; dir=1)
            record_ops && push!(ops, _CSHaloRecord(1))
            record_ops && _record_sweep!(ops, :x, scheme, panels_m, panels_am, fs_sub, storage)
            _sweep_cs_horizontal!(dummy_rm, panels_m, panels_am, mesh,
                                  scheme, ws, Val(1); flux_scale=fs_sub)
            fill_panel_halos!(panels_m, mesh; dir=1)
            record_ops && push!(ops, _CSHaloRecord(1))
        end

        # Convection record is also pure reverse-pass metadata in the
        # mass-tape recorder (forward propagation of `panels_m`
        # through convection happens in `_record_cs_tracer_tape`, not
        # here). Skip entirely under propagation-only mode.
        if record_ops && !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            push!(ops, _CSConvectionRecord(convection_op, forcing_step,
                                           _stage_panels(storage, panels_m),
                                           FT(dt)))
        end
    end

    return ops, panels_m
end

function _record_cs_tracer_tape(panels_rm0,
                                panels_m0,
                                panels_am_steps,
                                panels_bm_steps,
                                panels_cm_steps,
                                mesh::CubedSphereMesh,
                                scheme::CSAdjointNonlinearScheme;
                                cfl_limit,
                                flux_scale,
                                dt,
                                base_emission_rates = nothing,
                                diffusion_op = NoDiffusion(),
                                diffusion_workspace = nothing,
                                diffusion_meteo = nothing,
                                convection_op = NoConvection(),
                                convection_forcing = nothing,
                                convection_workspace = nothing,
                                tape_storage = :device,
                                step_offset::Int = 0,
                                record_ops::Bool = true)
    # `step_offset` and `record_ops` behave as in `_record_cs_mass_tape`:
    # step_offset shifts `_CSMidpointRecord` indices into absolute step
    # numbers for windowed invocations; record_ops = false short-circuits
    # every `_record_sweep!` / `_stage_panels` / `push!(ops, ...)` site.
    #
    # Unlike the mass tape, the tracer tape **applies diffusion and
    # convection forward** (they mutate `panels_rm` and, for convection,
    # `panels_m`). The propagation pass keeps those `apply_vertical_diffusion_vmr!`
    # / `_add_surface_rates!` / `_apply_cs_convection_forward!` calls so
    # the rm trajectory at each checkpoint matches the FullCheckpoint
    # forward path; only the staging + record-push sites are elided.
    #
    # Returns `(ops, panels_rm, panels_m)` — the third value lets a
    # strided-checkpoint driver chain the post-window rm state into the
    # next window's recording call. The dispatcher `_record_cs_adjoint_tape`
    # discards `panels_rm` to preserve the public 2-tuple contract.
    storage = record_ops ? _tape_storage(tape_storage) : nothing
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    _validate_emission_rates(base_emission_rates, nsteps, mesh, "base_emission_rates")
    panels_rm = _copy_panel_tuple(panels_rm0)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_rm, mesh; dir=0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    ops = Any[]
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    FT = eltype(panels_m[1])
    cfl_ft = FT(cfl_limit)
    fs = FT(flux_scale)
    dt_ft = FT(dt)

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_pal = _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)
        fs_sub = fs / FT(n_pal)

        # Forward half (X Y Z)^n_pal, as `strang_split_cs!`.
        for pass in 1:n_pal
            if pass > 1
                fill_panel_halos!(panels_rm, mesh; dir=1)
                fill_panel_halos!(panels_m, mesh; dir=1)
                record_ops && push!(ops, _CSHaloRecord(1))
            end
            record_ops && _record_sweep!(ops, :x, scheme, panels_m, panels_rm, panels_am, fs_sub, storage)
            _sweep_cs_horizontal!(panels_rm, panels_m, panels_am, mesh,
                                  scheme, ws, Val(1); flux_scale=fs_sub)
            fill_panel_halos!(panels_rm, mesh; dir=1)
            fill_panel_halos!(panels_m, mesh; dir=1)
            record_ops && push!(ops, _CSHaloRecord(1))

            record_ops && _record_sweep!(ops, :y, scheme, panels_m, panels_rm, panels_bm, fs_sub, storage)
            _sweep_cs_horizontal!(panels_rm, panels_m, panels_bm, mesh,
                                  scheme, ws, Val(2); flux_scale=fs_sub)
            fill_panel_halos!(panels_rm, mesh; dir=2)
            fill_panel_halos!(panels_m, mesh; dir=2)
            record_ops && push!(ops, _CSHaloRecord(2))

            _record_and_sweep_z!(ops, scheme, panels_rm, panels_m, panels_cm, ws, mesh,
                                 fs_sub, storage; record_ops)
        end

        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        absolute_step = step + step_offset
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
                panels_m_midpoint = _stage_panels(storage, panels_m)
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

        # Reverse half (Z Y X)^n_pal.
        for _ in 1:n_pal
            _record_and_sweep_z!(ops, scheme, panels_rm, panels_m, panels_cm, ws, mesh,
                                 fs_sub, storage; record_ops)

            fill_panel_halos!(panels_rm, mesh; dir=2)
            fill_panel_halos!(panels_m, mesh; dir=2)
            record_ops && push!(ops, _CSHaloRecord(2))
            record_ops && _record_sweep!(ops, :y, scheme, panels_m, panels_rm, panels_bm, fs_sub, storage)
            _sweep_cs_horizontal!(panels_rm, panels_m, panels_bm, mesh,
                                  scheme, ws, Val(2); flux_scale=fs_sub)
            fill_panel_halos!(panels_rm, mesh; dir=2)
            fill_panel_halos!(panels_m, mesh; dir=2)
            record_ops && push!(ops, _CSHaloRecord(2))

            fill_panel_halos!(panels_rm, mesh; dir=1)
            fill_panel_halos!(panels_m, mesh; dir=1)
            record_ops && push!(ops, _CSHaloRecord(1))
            record_ops && _record_sweep!(ops, :x, scheme, panels_m, panels_rm, panels_am, fs_sub, storage)
            _sweep_cs_horizontal!(panels_rm, panels_m, panels_am, mesh,
                                  scheme, ws, Val(1); flux_scale=fs_sub)
            fill_panel_halos!(panels_rm, mesh; dir=1)
            fill_panel_halos!(panels_m, mesh; dir=1)
            record_ops && push!(ops, _CSHaloRecord(1))
        end

        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            if record_ops
                push!(ops, _CSConvectionRecord(convection_op, forcing_step,
                                               _stage_panels(storage, panels_m),
                                               dt_ft))
            end
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt_ft,
                                          convection_workspace, mesh)
        end
    end

    return ops, panels_rm, panels_m
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointLinearScheme;
                                 base_emission_rates = nothing,
                                 diffusion_meteo = nothing,
                                 convection_workspace = nothing,
                                 kwargs...)
    return _record_cs_mass_tape(panels_m0,
                                panels_am_steps, panels_bm_steps, panels_cm_steps,
                                mesh, scheme; kwargs...)
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointLinRoodScheme; kwargs...)
    # `_record_cs_linrood_tape` returns `(ops, panels_rm, panels_m)` —
    # the third value lets the stride driver chain rm across windows.
    # The public dispatcher contract is `(ops, final_m)`; drop the
    # intermediate rm tuple here.
    ops, _, final_m = _record_cs_linrood_tape(panels_rm0, panels_m0,
                                               panels_am_steps, panels_bm_steps,
                                               panels_cm_steps, mesh, scheme; kwargs...)
    return ops, final_m
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointNonlinearScheme; kwargs...)
    # `_record_cs_tracer_tape` returns `(ops, panels_rm, panels_m)` —
    # the third value lets the stride driver chain rm across windows.
    # The public dispatcher contract is `(ops, final_m)`; drop the
    # intermediate rm tuple here.
    ops, _, final_m = _record_cs_tracer_tape(panels_rm0, panels_m0,
                                              panels_am_steps, panels_bm_steps,
                                              panels_cm_steps, mesh, scheme; kwargs...)
    return ops, final_m
end
