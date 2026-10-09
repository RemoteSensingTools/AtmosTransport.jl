# Cubed-sphere Z sweeps (panel-local) and interior copies.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

"""Higher-order Z-sweep via KA kernel dispatching on scheme (Slopes, PPM, etc.).

Z boundary: `_zface_tracer_flux` handles k=1 (TOA) and k=Nz+1 (surface) boundaries
by falling back to upwind at the domain edges.
"""
function _sweep_z_panel!(rm, m, cm, scheme::AbstractAdvectionScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    # Z-direction does not need halo validation (vertical boundaries are closed, not halo-exchanged)
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_zsweep_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_z, :cs_kernel_sync_z) do
        kernel!(rm_A, rm, m_A, m, cm, scheme, Int32(Nz), Int32(Hp), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_z) do
        _copy_interior!(rm, rm_A, Nc, Hp, Nz)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_z_panel_mt!(rm_4d, m, cm, scheme::AbstractAdvectionScheme,
                            rm_4d_A, m_A, Nc, Hp, Nz, Nt;
                            flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    workgroupsize = _cs_packed_sweep_workgroupsize(backend, scheme, FT)
    kernel! = _cs_zsweep_mt_kernel!(backend, workgroupsize)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_z_mt, :cs_kernel_sync_z_mt) do
        kernel!(rm_4d_A, rm_4d, m_A, m, cm, scheme, Int32(Nz), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_z_mt) do
        _copy_interior!(rm_4d, rm_4d_A, Nc, Hp, Nz, Nt)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_z_panel_mt_pingpong!(rm_4d_out, m_out, rm_4d, m, cm,
                                     scheme::AbstractAdvectionScheme,
                                     Nc, Hp, Nz, Nt;
                                     flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    workgroupsize = _cs_packed_sweep_workgroupsize(backend, scheme, FT)
    kernel! = _cs_zsweep_mt_kernel!(backend, workgroupsize)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_z_mt, :cs_kernel_sync_z_mt) do
        kernel!(rm_4d_out, rm_4d, m_out, m, cm, scheme, Int32(Nz), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    return nothing
end

"""Gamma-clamped upwind Z-sweep kernel.

Closed boundary at k=1 (TOA) and k=Nz+1 (surface): both face fluxes
zero out at the domain edges via `ifelse(at_boundary, ...)`.
"""
@kernel function _cs_zsweep_upwind_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                            @Const(cm), Nz, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        cm_t = flux_scale * cm[i, j, k]
        cm_b = flux_scale * cm[i, j, k + 1]

        mi = m[i, j, k]
        ri = rm[i, j, k]
        zero_FT = zero(cm_t)

        # Top face (k): donor is k-1 if cm_t > 0 (downward), else k.
        # k=1 → no top face (closed TOA).
        kt   = max(k - Int32(1), Int32(1))
        m_kt = m[i, j, kt]
        r_kt = rm[i, j, kt]
        ft_in = ifelse(cm_t >= zero_FT,
                       _gamma_clamped_x_flux(cm_t, m_kt, r_kt),
                       _gamma_clamped_x_flux(cm_t, mi,   ri))
        ft = ifelse(k > Int32(1), ft_in, zero_FT)

        # Bottom face (k+1): donor is k if cm_b > 0 (downward), else k+1.
        # k=Nz → no bottom face (closed surface).
        kb   = min(k + Int32(1), Nz)
        m_kb = m[i, j, kb]
        r_kb = rm[i, j, kb]
        fb_in = ifelse(cm_b >= zero_FT,
                       _gamma_clamped_x_flux(cm_b, mi,   ri),
                       _gamma_clamped_x_flux(cm_b, m_kb, r_kb))
        fb = ifelse(k < Nz, fb_in, zero_FT)

        rm_new[i, j, k] = ri + ft - fb
        m_new[i, j, k]  = mi + cm_t - cm_b
    end
end

@kernel function _cs_zsweep_mt_upwind_kernel!(rm_new_4d, @Const(rm_4d),
                                               m_new, @Const(m),
                                               @Const(cm), Nz, Hp, Nt, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        cm_t = flux_scale * cm[i, j, k]
        cm_b = flux_scale * cm[i, j, k + 1]

        mi = m[i, j, k]
        zero_FT = zero(cm_t)
        kt   = max(k - Int32(1), Int32(1))
        kb   = min(k + Int32(1), Nz)
        m_kt = m[i, j, kt]
        m_kb = m[i, j, kb]
        m_new[i, j, k] = mi + cm_t - cm_b
        for t in Int32(1):Int32(Nt)
            ri   = rm_4d[i, j, k,  t]
            r_kt = rm_4d[i, j, kt, t]
            r_kb = rm_4d[i, j, kb, t]
            ft_in = ifelse(cm_t >= zero_FT,
                           _gamma_clamped_x_flux(cm_t, m_kt, r_kt),
                           _gamma_clamped_x_flux(cm_t, mi,   ri))
            ft = ifelse(k > Int32(1), ft_in, zero_FT)
            fb_in = ifelse(cm_b >= zero_FT,
                           _gamma_clamped_x_flux(cm_b, mi,   ri),
                           _gamma_clamped_x_flux(cm_b, m_kb, r_kb))
            fb = ifelse(k < Nz, fb_in, zero_FT)
            rm_new_4d[i, j, k, t] = ri + ft - fb
        end
    end
end

function _sweep_z_panel!(rm, m, cm, scheme::UpwindScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_zsweep_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_z, :cs_kernel_sync_z) do
        kernel!(rm_A, rm, m_A, m, cm, Int32(Nz), Int32(Hp), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_z) do
        _copy_interior!(rm, rm_A, Nc, Hp, Nz)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_z_panel_mt!(rm_4d, m, cm, scheme::UpwindScheme,
                            rm_4d_A, m_A, Nc, Hp, Nz, Nt;
                            flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    kernel! = _cs_zsweep_mt_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_z_mt, :cs_kernel_sync_z_mt) do
        kernel!(rm_4d_A, rm_4d, m_A, m, cm, Int32(Nz), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    _profiled_copy!(:cs_copyback_z_mt) do
        _copy_interior!(rm_4d, rm_4d_A, Nc, Hp, Nz, Nt)
        _copy_interior!(m, m_A, Nc, Hp, Nz)
    end
    return nothing
end

function _sweep_z_panel_mt_pingpong!(rm_4d_out, m_out, rm_4d, m, cm,
                                     scheme::UpwindScheme,
                                     Nc, Hp, Nz, Nt;
                                     flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    kernel! = _cs_zsweep_mt_upwind_kernel!(backend, 256)
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_z_mt, :cs_kernel_sync_z_mt) do
        kernel!(rm_4d_out, rm_4d, m_out, m, cm, Int32(Nz), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    return nothing
end

@kernel function _copy_interior_3d_kernel!(dst, @Const(src), Hp)
    ii, jj, kk = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dst[i, j, kk] = src[i, j, kk]
    end
end

@kernel function _copy_interior_4d_kernel!(dst, @Const(src), Hp)
    ii, jj, kk, tt = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dst[i, j, kk, tt] = src[i, j, kk, tt]
    end
end

"""Copy interior region from buffer back to array.

Replaces the prior `dst[r, r, 1:Nz] .= src[r, r, 1:Nz]` broadcast, which
launched GPUArrays.jl's generic `gpu_getindex_kernel` (~50 % of GPU time on a
C180 full-physics run, vs ~12% for a direct kernel). The custom kernels above
do one device-local read/write per cell with no intermediate temporary.

The shared `src` workspace buffer (e.g. `rm_A` / `m_A`) is reused for the next
panel's sweep, so panel p's copy-back must precede panel p+1's sweep. On GPU
that ordering is guaranteed by the single issue-ordered stream, so no host
barrier is needed — the periodic GPU sync lands at the `fill_panel_halos!`
boundary. On the CPU backend we synchronize defensively, mirroring
HaloExchange.jl's `KA_CPU` gate. The per-panel host `synchronize` previously
here was the dominant launch-bound bubble (GPU profiling 2026-06-13).
"""
function _copy_interior!(dst, src, Nc, Hp, Nz)
    backend = get_backend(dst)
    kernel! = _copy_interior_3d_kernel!(backend, 256)
    kernel!(dst, src, Int32(Hp); ndrange = (Nc, Nc, Nz))
    backend isa KA_CPU && synchronize(backend)
    return nothing
end

function _copy_interior!(dst::AbstractArray{<:Any, 4}, src::AbstractArray{<:Any, 4},
                         Nc, Hp, Nz, Nt)
    backend = get_backend(dst)
    kernel! = _copy_interior_4d_kernel!(backend, 256)
    kernel!(dst, src, Int32(Hp); ndrange = (Nc, Nc, Nz, Nt))
    backend isa KA_CPU && synchronize(backend)
    return nothing
end
