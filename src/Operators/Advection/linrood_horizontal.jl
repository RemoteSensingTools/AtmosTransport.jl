# Lin-Rood horizontal advection drivers (rm- and q-space) and the Lin-Rood Strang split.
# Split from LinRood.jl (refactor phase 4); included by Advection.jl in this order.

# ---------------------------------------------------------------------------
# Main Lin-Rood Horizontal Advection
# ---------------------------------------------------------------------------

"""
    fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                  mesh, ::Val{ORD}, ws, ws_lr; damp_coeff=0.0)

Lin-Rood horizontal advection for cubed-sphere grids.
Averages X-first and Y-first PPM orderings, then shares the final seam
mixing-ratio estimates between neighboring panels before their flux updates.
The transverse formulation follows FV3; the explicit seam projection is
AtmosTransport's conservative interface coupling.
"""
function fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                       mesh::CubedSphereMesh, ::Val{ORD}, ws, ws_lr::LinRoodWorkspace;
                       damp_coeff=0.0) where ORD
    Nc = mesh.Nc; Hp = mesh.Hp; Nz = size(rm_panels[1], 3)
    N = Nc + 2Hp
    backend = get_backend(rm_panels[1])

    # Optional divergence damping
    damp_coeff > 0 && apply_divergence_damping_cs!(rm_panels, m_panels, mesh, ws, damp_coeff)

    # Pre-instantiate all kernels (avoid repeated compilation inside loops)
    init_k!    = _init_q_buf_kernel!(backend, 256)
    y_face_k!  = _ppm_y_face_kernel!(backend, 256)
    x_face_k!  = _ppm_x_face_kernel!(backend, 256)
    xq_face_k! = _ppm_x_face_from_q_kernel!(backend, 256)
    yq_face_k! = _ppm_y_face_from_q_kernel!(backend, 256)
    pre_y_k!   = _pre_advect_y_kernel!(backend, 256)
    pre_x_k!   = _pre_advect_x_kernel!(backend, 256)
    update_k!  = _linrood_update_kernel!(backend, 256)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Edge halos + Y-corners → inner Y-PPM + pre-advect q_i
    # ═══════════════════════════════════════════════════════════════════════
    fill_panel_halos!(rm_panels, mesh)
    fill_panel_halos!(m_panels, mesh)
    copy_corners!(rm_panels, mesh, 2)
    copy_corners!(m_panels, mesh, 2)

    # Initialize q_buf with original mixing ratio (halos persist for outer PPM)
    for p in eachindex(ws_lr.q_buf)
        init_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p]; ndrange=(N, N, Nz))
    end
    synchronize(backend)

    for p in eachindex(ws_lr.fy_in)
        y_face_k!(ws_lr.fy_in[p], rm_panels[p], m_panels[p], bm_panels[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        pre_y_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p], bm_panels[p],
                 ws_lr.fy_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: X-corners → outer X-PPM on q_i + inner X-PPM + pre-advect q_j
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, mesh, 1)
    copy_corners!(rm_panels, mesh, 1)
    copy_corners!(m_panels, mesh, 1)

    for p in eachindex(ws_lr.fx_out)
        xq_face_k!(ws_lr.fx_out[p], ws_lr.q_buf[p], am_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
        x_face_k!(ws_lr.fx_in[p], rm_panels[p], m_panels[p], am_panels[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    end
    synchronize(backend)

    # Re-initialize q_buf (halos retain original q; interior overwritten with q_j)
    for p in eachindex(ws_lr.q_buf)
        init_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p]; ndrange=(N, N, Nz))
    end
    synchronize(backend)

    for p in eachindex(ws_lr.q_buf)
        pre_x_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p], am_panels[p],
                 ws_lr.fx_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Y-corners on q_j → outer Y-PPM → averaged update
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, mesh, 2)

    for p in eachindex(ws_lr.fx_in)
        yq_face_k!(ws_lr.fy_out[p], ws_lr.q_buf[p], bm_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
    end
    _share_lr_seam_faces!(ws_lr.fx_in, ws_lr.fx_out, ws_lr.fy_in, ws_lr.fy_out, mesh)
    for p in eachindex(ws_lr.fx_in)
        update_k!(ws.rm_A, ws.m_A,
                  rm_panels[p], m_panels[p], am_panels[p], bm_panels[p],
                  ws_lr.fx_in[p], ws_lr.fx_out[p], ws_lr.fy_in[p], ws_lr.fy_out[p],
                  Hp; ndrange=(Nc, Nc, Nz))
        synchronize(backend)  # required: ws.rm_A/m_A reused across panels
        _copy_interior!(rm_panels[p], ws.rm_A, Nc, Hp, Nz)
        _copy_interior!(m_panels[p], ws.m_A, Nc, Hp, Nz)
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Q-space Lin-Rood Horizontal Advection (GCHP-aligned)
#
# Operates on mixing ratio q and carrier air mass m instead of tracer mass rm.
# Air mass evolves via mass flux divergence; q is updated as:
#   q_new = (q*m1 + tracer_flux_div) / m2
# This uses the mass-weighted form of GCHP's tracer_2d update.
#
# m_panels supplies CFL fractions and is updated by the final divergence.
# Phase 3 uses per-panel output buffers (ws_lr.q_out, ws_lr.dp_out)
# for parallel kernel launch across all 6 panels (no sequential sync).
# ---------------------------------------------------------------------------

"""
    fv_tp_2d_cs_q!(q_panels, m_panels, am_panels, bm_panels,
                     mesh, ::Val{ORD}, ws, ws_lr; damp_coeff=0.0)

Q-space Lin-Rood horizontal advection. Evolves `q` (mixing ratio) and `m`
(air mass, same role as pressure thickness) in-place. `m_panels` is both
read (for CFL fraction in PPM face values) and written (mass divergence update).
"""
function fv_tp_2d_cs_q!(q_panels, m_panels, am_panels, bm_panels,
                          mesh::CubedSphereMesh, ::Val{ORD}, ws, ws_lr::LinRoodWorkspace;
                          damp_coeff=0.0) where ORD
    Nc = mesh.Nc; Hp = mesh.Hp; Nz = size(q_panels[1], 3)
    N = Nc + 2Hp
    backend = get_backend(q_panels[1])

    # Pre-instantiate kernels
    yq_face_k!  = _ppm_y_face_from_q_kernel!(backend, 256)
    xq_face_k!  = _ppm_x_face_from_q_kernel!(backend, 256)
    pre_y_q_k!  = _pre_advect_y_q_kernel!(backend, 256)
    pre_x_q_k!  = _pre_advect_x_q_kernel!(backend, 256)
    update_q_k! = _linrood_update_q_kernel!(backend, 256)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Edge halos + Y-corners → inner Y-PPM + pre-advect q_i
    # ═══════════════════════════════════════════════════════════════════════
    fill_panel_halos!(q_panels, mesh)
    fill_panel_halos!(m_panels, mesh)
    copy_corners!(q_panels, mesh, 2)
    copy_corners!(m_panels, mesh, 2)

    # Initialize q_buf with current q (halos included)
    for p in eachindex(ws_lr.q_buf)
        copyto!(ws_lr.q_buf[p], q_panels[p])
    end

    for p in eachindex(ws_lr.fy_in)
        yq_face_k!(ws_lr.fy_in[p], q_panels[p], bm_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        pre_y_q_k!(ws_lr.q_buf[p], q_panels[p], m_panels[p], bm_panels[p],
                   ws_lr.fy_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: X-corners → outer X-PPM on q_i + inner X-PPM + pre-advect q_j
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, mesh, 1)
    copy_corners!(q_panels, mesh, 1)
    copy_corners!(m_panels, mesh, 1)

    for p in eachindex(ws_lr.fx_out)
        xq_face_k!(ws_lr.fx_out[p], ws_lr.q_buf[p], am_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
        xq_face_k!(ws_lr.fx_in[p], q_panels[p], am_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    end
    synchronize(backend)

    # Re-initialize q_buf from original q, then overwrite interior with q_j
    for p in eachindex(ws_lr.q_buf)
        copyto!(ws_lr.q_buf[p], q_panels[p])
    end

    for p in eachindex(ws_lr.q_buf)
        pre_x_q_k!(ws_lr.q_buf[p], q_panels[p], m_panels[p], am_panels[p],
                   ws_lr.fx_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Y-corners on q_j → outer Y-PPM → averaged update (PARALLEL)
    #
    # Both q AND m are updated by the kernel. m evolves via mass flux
    # divergence (same as rm-space _linrood_update_kernel!). q is updated
    # as q_new = (q*m1 + flux_div) / m2. Per-panel buffers allow all 6
    # panels to launch concurrently.
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, mesh, 2)

    for p in eachindex(ws_lr.fx_in)
        yq_face_k!(ws_lr.fy_out[p], ws_lr.q_buf[p], bm_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
    end
    _share_lr_seam_faces!(ws_lr.fx_in, ws_lr.fx_out, ws_lr.fy_in, ws_lr.fy_out, mesh)
    for p in eachindex(ws_lr.fx_in)
        # q_out gets q_new, dp_out gets m_new (reusing dp_out buffer for m)
        update_q_k!(ws_lr.q_out[p], ws_lr.dp_out[p],
                    q_panels[p], m_panels[p], am_panels[p], bm_panels[p],
                    ws_lr.fx_in[p], ws_lr.fx_out[p], ws_lr.fy_in[p], ws_lr.fy_out[p],
                    Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # Copy results back — both q and m are updated
    for p in 1:6
        _copy_interior!(q_panels[p], ws_lr.q_out[p], Nc, Hp, Nz)
        _copy_interior!(m_panels[p], ws_lr.dp_out[p], Nc, Hp, Nz)
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Strang Split: Lin-Rood Horizontal + Vertical Z-sweep
# ---------------------------------------------------------------------------

"""
    strang_split_linrood_ppm!(rm_panels, m_panels, am_panels, bm_panels, cm_panels,
                               mesh, ::Val{ORD}, ws, ws_lr; cfl_limit=0.95, damp_coeff=0.0)

Full 3D advection: Horizontal(LR) → Z → Z → Horizontal(LR).
"""
function strang_split_linrood_ppm!(rm_panels, m_panels, am_panels, bm_panels, cm_panels,
                                    mesh::CubedSphereMesh, ::Val{ORD}, ws, ws_lr::LinRoodWorkspace;
                                    cfl_limit=0.95, damp_coeff=0.0) where ORD
    fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                  mesh, Val(ORD), ws, ws_lr; damp_coeff)
    _sweep_z!(rm_panels, m_panels, cm_panels, mesh, ws)
    _sweep_z!(rm_panels, m_panels, cm_panels, mesh, ws)
    fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                  mesh, Val(ORD), ws, ws_lr; damp_coeff=0.0)
    return nothing
end

function _strang_split_linrood_ppm_cs!(rm_panels, m_panels, am_panels, bm_panels, cm_panels,
                                       mesh::CubedSphereMesh, ::Val{ORD},
                                       ws::CSLinRoodAdvectionWorkspace;
                                       cfl_limit=0.95, midpoint! = nothing,
                                       damp_coeff=0.0,
                                       vertical::AbstractAdvectionScheme = UpwindScheme()) where ORD
    _ = cfl_limit
    fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                 mesh, Val(ORD), ws.cs, ws.linrood; damp_coeff)
    _sweep_z_panels!(rm_panels, m_panels, cm_panels, mesh, vertical, ws.cs)
    midpoint! === nothing || midpoint!()
    _sweep_z_panels!(rm_panels, m_panels, cm_panels, mesh, vertical, ws.cs)
    fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                 mesh, Val(ORD), ws.cs, ws.linrood; damp_coeff = 0.0)
    return nothing
end

export LinRoodWorkspace, fv_tp_2d_cs!, fv_tp_2d_cs_q!, strang_split_linrood_ppm!
export CSLinRoodAdvectionWorkspace
export apply_divergence_damping_cs!
# Adjoint kernels. The forward kernels above are paired with
# reverse-mode kernels defined in `linrood_adjoint_*.jl`, included
# alongside this file from `Advection.jl`. Re-export below for `Adjoints`.
export apply_linrood_update_adjoint!,
       apply_pre_advect_x_adjoint!,
       apply_pre_advect_y_adjoint!,
       apply_ppm_x_face_from_q_adjoint!,
       apply_ppm_y_face_from_q_adjoint!,
       apply_ppm_x_face_adjoint!,
       apply_ppm_y_face_adjoint!
