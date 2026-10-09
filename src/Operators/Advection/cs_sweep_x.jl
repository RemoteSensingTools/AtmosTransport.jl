# Cubed-sphere X sweeps: per-panel, multi-tracer and ping-pong variants with paired seam transfers.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

"""Higher-order X-sweep via KA kernel dispatching on scheme (Slopes, PPM, etc.).

Requires sufficient halo padding: Hp >= 2 for SlopesScheme, Hp >= 3 for PPMScheme.
The `_xface_tracer_flux` reconstruction reads neighbors up to `face_i ± 3` for PPM,
which stays within the halo-padded array when Hp is large enough. The periodic wrap
in `_wrap_periodic` is a safety net that should never trigger with correct Hp.
"""
function _sweep_x_panel!(rm, m, am, scheme::AbstractAdvectionScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_xsweep_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_x, :cs_kernel_sync_x) do
        kernel!(rm_A, rm, m_A, m, am, scheme, Int32(Nc), Int32(Hp), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_x) do
        _copy_interior!(rm, rm_A, Nc, Hp, Nz)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_x_panel_mt!(rm_4d, m, am, scheme::AbstractAdvectionScheme,
                            rm_4d_A, m_A, Nc, Hp, Nz, Nt;
                            flux_scale = one(eltype(m)))
    _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm_4d)
    workgroupsize = _cs_packed_sweep_workgroupsize(backend, scheme, FT)
    kernel! = _cs_xsweep_mt_kernel!(backend, workgroupsize)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_x_mt, :cs_kernel_sync_x_mt) do
        kernel!(rm_4d_A, rm_4d, m_A, m, am, scheme, Int32(Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_x_mt) do
        _copy_interior!(rm_4d, rm_4d_A, Nc, Hp, Nz, Nt)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_x_panel_mt_pingpong!(rm_4d_out, m_out, rm_4d, m, am,
                                     scheme::AbstractAdvectionScheme,
                                     Nc, Hp, Nz, Nt;
                                     flux_scale = one(eltype(m)))
    _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm_4d)
    workgroupsize = _cs_packed_sweep_workgroupsize(backend, scheme, FT)
    kernel! = _cs_xsweep_mt_kernel!(backend, workgroupsize)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_x_mt, :cs_kernel_sync_x_mt) do
        kernel!(rm_4d_out, rm_4d, m_out, m, am, scheme, Int32(Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    return nothing
end

"""Gamma-clamped upwind X-sweep kernel (positivity-safe at CFL > 1).

Donor for the left face at `i` is `i-1` when the face flux is positive
(eastward) and `i` otherwise. The right face at `i+1` is symmetric.
`_gamma_clamped_x_flux` returns 0 when `m_donor ≤ 0`, so the sweep
degrades gracefully on cells the upstream binary already drained
negative — no NaN, just no transport in that subcycle.
"""
@kernel function _cs_xsweep_upwind_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                            @Const(am), Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        am_l = flux_scale * am[i,     j, k]
        am_r = flux_scale * am[i + 1, j, k]

        mi   = m[i,     j, k]
        mim1 = m[i - 1, j, k]
        rim1 = rm[i - 1, j, k]
        ri   = rm[i,    j, k]
        fl = ifelse(am_l >= zero(am_l),
                    _gamma_clamped_x_flux(am_l, mim1, rim1),
                    _gamma_clamped_x_flux(am_l, mi,   ri))

        mip1 = m[i + 1, j, k]
        rip1 = rm[i + 1, j, k]
        fr = ifelse(am_r >= zero(am_r),
                    _gamma_clamped_x_flux(am_r, mi,   ri),
                    _gamma_clamped_x_flux(am_r, mip1, rip1))

        rm_new[i, j, k] = ri + fl - fr
        m_new[i, j, k]  = mi + am_l - am_r
    end
end

@kernel function _cs_xsweep_mt_upwind_kernel!(rm_new_4d, @Const(rm_4d),
                                               m_new, @Const(m),
                                               @Const(am), Nc, Hp, Nt, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        am_l = flux_scale * am[i,     j, k]
        am_r = flux_scale * am[i + 1, j, k]

        mi   = m[i,     j, k]
        mim1 = m[i - 1, j, k]
        mip1 = m[i + 1, j, k]
        m_new[i, j, k] = mi + am_l - am_r
        for t in Int32(1):Int32(Nt)
            ri   = rm_4d[i,     j, k, t]
            rim1 = rm_4d[i - 1, j, k, t]
            rip1 = rm_4d[i + 1, j, k, t]
            fl = ifelse(am_l >= zero(am_l),
                        _gamma_clamped_x_flux(am_l, mim1, rim1),
                        _gamma_clamped_x_flux(am_l, mi,   ri))
            fr = ifelse(am_r >= zero(am_r),
                        _gamma_clamped_x_flux(am_r, mi,   ri),
                        _gamma_clamped_x_flux(am_r, mip1, rip1))
            rm_new_4d[i, j, k, t] = ri + fl - fr
        end
    end
end

"""Gamma-clamped upwind X-sweep: positivity-safe even at CFL > 1."""
function _sweep_x_panel!(rm, m, am, scheme::UpwindScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_xsweep_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_x, :cs_kernel_sync_x) do
        kernel!(rm_A, rm, m_A, m, am, Int32(Nc), Int32(Hp), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_x) do
        _copy_interior!(rm, rm_A, Nc, Hp, Nz)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_x_panel_mt!(rm_4d, m, am, scheme::UpwindScheme,
                            rm_4d_A, m_A, Nc, Hp, Nz, Nt;
                            flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    kernel! = _cs_xsweep_mt_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_x_mt, :cs_kernel_sync_x_mt) do
        kernel!(rm_4d_A, rm_4d, m_A, m, am, Int32(Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_x_mt) do
        _copy_interior!(rm_4d, rm_4d_A, Nc, Hp, Nz, Nt)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_x_panel_mt_pingpong!(rm_4d_out, m_out, rm_4d, m, am,
                                     scheme::UpwindScheme,
                                     Nc, Hp, Nz, Nt;
                                     flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    kernel! = _cs_xsweep_mt_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_x_mt, :cs_kernel_sync_x_mt) do
        kernel!(rm_4d_out, rm_4d, m_out, m, am, Int32(Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    return nothing
end

function _sweep_x_panels_mt_pingpong!(panels_rm_4d_out::NTuple{6},
                                      panels_m_out::NTuple{6},
                                      panels_rm_4d::NTuple{6},
                                      panels_m::NTuple{6},
                                      panels_am::NTuple{6},
                                      mesh::CubedSphereMesh,
                                      scheme::AbstractAdvectionScheme;
                                      flux_scale = one(eltype(panels_m[1])),
                                      seam_flux = nothing)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    cache = seam_flux === nothing ?
        similar(panels_rm_4d[1], eltype(panels_m[1]), Nc, Nz, Nt + 1, 12) : seam_flux
    _cache_cs_seams!(cache, panels_rm_4d, panels_m, panels_am, mesh,
                     scheme, Val(1), flux_scale)
    for p in 1:6
        _sweep_x_panel_mt_pingpong!(panels_rm_4d_out[p], panels_m_out[p],
                                   panels_rm_4d[p], panels_m[p], _CSInteriorFlux(panels_am[p], mesh, Val(1)),
                                   scheme, Nc, Hp, Nz, Nt; flux_scale)
    end
    _apply_cs_seams!(panels_rm_4d_out, panels_m_out, cache, mesh, Val(1))
    return panels_rm_4d_out, panels_m_out
end
