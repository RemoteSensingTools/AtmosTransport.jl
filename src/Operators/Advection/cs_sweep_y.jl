# Cubed-sphere Y sweeps: per-panel, multi-tracer and ping-pong variants with paired seam transfers.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

"""Higher-order Y-sweep via KA kernel dispatching on scheme (Slopes, PPM, etc.)."""
function _sweep_y_panel!(rm, m, bm, scheme::AbstractAdvectionScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_ysweep_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_y, :cs_kernel_sync_y) do
        kernel!(rm_A, rm, m_A, m, bm, scheme, Int32(Nc), Int32(Hp), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_y) do
        _copy_interior!(rm, rm_A, Nc, Hp, Nz)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_y_panel_mt!(rm_4d, m, bm, scheme::AbstractAdvectionScheme,
                            rm_4d_A, m_A, Nc, Hp, Nz, Nt;
                            flux_scale = one(eltype(m)))
    _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm_4d)
    workgroupsize = _cs_packed_sweep_workgroupsize(backend, scheme, FT)
    kernel! = _cs_ysweep_mt_kernel!(backend, workgroupsize)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_y_mt, :cs_kernel_sync_y_mt) do
        kernel!(rm_4d_A, rm_4d, m_A, m, bm, scheme, Int32(Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_y_mt) do
        _copy_interior!(rm_4d, rm_4d_A, Nc, Hp, Nz, Nt)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_y_panel_mt_pingpong!(rm_4d_out, m_out, rm_4d, m, bm,
                                     scheme::AbstractAdvectionScheme,
                                     Nc, Hp, Nz, Nt;
                                     flux_scale = one(eltype(m)))
    _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm_4d)
    workgroupsize = _cs_packed_sweep_workgroupsize(backend, scheme, FT)
    kernel! = _cs_ysweep_mt_kernel!(backend, workgroupsize)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_y_mt, :cs_kernel_sync_y_mt) do
        kernel!(rm_4d_out, rm_4d, m_out, m, bm, scheme, Int32(Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    return nothing
end

"""Gamma-clamped upwind Y-sweep kernel."""
@kernel function _cs_ysweep_upwind_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                            @Const(bm), Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bm_s = flux_scale * bm[i, j,     k]
        bm_n = flux_scale * bm[i, j + 1, k]

        mi   = m[i, j,     k]
        mjm1 = m[i, j - 1, k]
        rjm1 = rm[i, j - 1, k]
        ri   = rm[i, j,    k]
        fs = ifelse(bm_s >= zero(bm_s),
                    _gamma_clamped_x_flux(bm_s, mjm1, rjm1),
                    _gamma_clamped_x_flux(bm_s, mi,   ri))

        mjp1 = m[i, j + 1, k]
        rjp1 = rm[i, j + 1, k]
        fn = ifelse(bm_n >= zero(bm_n),
                    _gamma_clamped_x_flux(bm_n, mi,   ri),
                    _gamma_clamped_x_flux(bm_n, mjp1, rjp1))

        rm_new[i, j, k] = ri + fs - fn
        m_new[i, j, k]  = mi + bm_s - bm_n
    end
end

@kernel function _cs_ysweep_mt_upwind_kernel!(rm_new_4d, @Const(rm_4d),
                                               m_new, @Const(m),
                                               @Const(bm), Nc, Hp, Nt, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bm_s = flux_scale * bm[i, j,     k]
        bm_n = flux_scale * bm[i, j + 1, k]

        mi   = m[i, j,     k]
        mjm1 = m[i, j - 1, k]
        mjp1 = m[i, j + 1, k]
        m_new[i, j, k] = mi + bm_s - bm_n
        for t in Int32(1):Int32(Nt)
            ri   = rm_4d[i, j,     k, t]
            rjm1 = rm_4d[i, j - 1, k, t]
            rjp1 = rm_4d[i, j + 1, k, t]
            fs = ifelse(bm_s >= zero(bm_s),
                        _gamma_clamped_x_flux(bm_s, mjm1, rjm1),
                        _gamma_clamped_x_flux(bm_s, mi,   ri))
            fn = ifelse(bm_n >= zero(bm_n),
                        _gamma_clamped_x_flux(bm_n, mi,   ri),
                        _gamma_clamped_x_flux(bm_n, mjp1, rjp1))
            rm_new_4d[i, j, k, t] = ri + fs - fn
        end
    end
end

function _sweep_y_panel!(rm, m, bm, scheme::UpwindScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_ysweep_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_y, :cs_kernel_sync_y) do
        kernel!(rm_A, rm, m_A, m, bm, Int32(Nc), Int32(Hp), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_y) do
        _copy_interior!(rm, rm_A, Nc, Hp, Nz)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_y_panel_mt!(rm_4d, m, bm, scheme::UpwindScheme,
                            rm_4d_A, m_A, Nc, Hp, Nz, Nt;
                            flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    kernel! = _cs_ysweep_mt_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_y_mt, :cs_kernel_sync_y_mt) do
        kernel!(rm_4d_A, rm_4d, m_A, m, bm, Int32(Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_y_mt) do
        _copy_interior!(rm_4d, rm_4d_A, Nc, Hp, Nz, Nt)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_y_panel_mt_pingpong!(rm_4d_out, m_out, rm_4d, m, bm,
                                     scheme::UpwindScheme,
                                     Nc, Hp, Nz, Nt;
                                     flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    kernel! = _cs_ysweep_mt_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_y_mt, :cs_kernel_sync_y_mt) do
        kernel!(rm_4d_out, rm_4d, m_out, m, bm, Int32(Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    return nothing
end

function _sweep_y_panels_mt_pingpong!(panels_rm_4d_out::NTuple{6},
                                      panels_m_out::NTuple{6},
                                      panels_rm_4d::NTuple{6},
                                      panels_m::NTuple{6},
                                      panels_bm::NTuple{6},
                                      mesh::CubedSphereMesh,
                                      scheme::AbstractAdvectionScheme;
                                      flux_scale = one(eltype(panels_m[1])),
                                      seam_flux = nothing)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    cache = seam_flux === nothing ?
        similar(panels_rm_4d[1], eltype(panels_m[1]), Nc, Nz, Nt + 1, 12) : seam_flux
    _cache_cs_seams!(cache, panels_rm_4d, panels_m, panels_bm, mesh,
                     scheme, Val(2), flux_scale)
    for p in 1:6
        _sweep_y_panel_mt_pingpong!(panels_rm_4d_out[p], panels_m_out[p],
                                   panels_rm_4d[p], panels_m[p], _CSInteriorFlux(panels_bm[p], mesh, Val(2)),
                                   scheme, Nc, Hp, Nz, Nt; flux_scale)
    end
    _apply_cs_seams!(panels_rm_4d_out, panels_m_out, cache, mesh, Val(2))
    return panels_rm_4d_out, panels_m_out
end
