# Cubed-sphere panel sweeps, shared part: higher-order KA kernels, halo check, gamma-clamped upwind flux, profiled launches.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend, CPU as KA_CPU

# =========================================================================
# CS panel sweep kernels
#
# These launch on ndrange=(Nc, Nc, Nz) and read/write the interior region
# of halo-padded arrays. The Hp offset is added to all indices.
# =========================================================================

# These KA kernels dispatch on `scheme` via _xface_tracer_flux and work for
# UpwindScheme, SlopesScheme, and PPMScheme. They are used by the higher-order
# _sweep_x/y/z_panel! methods (AbstractAdvectionScheme fallback) via KA_CPU().
# On GPU backends, they can be launched directly with the appropriate backend.
# The UpwindScheme specialization uses hand-written gamma-clamped loops instead
# (positivity-safe even at CFL > 1).

"""X-sweep kernel on one CS panel. Interior i ∈ [1,Nc], neighbors via halo."""
@kernel function _cs_xsweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                     @Const(am), scheme, Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        # Map to halo-padded indices
        i = ii + Hp
        j = jj + Hp
        # Face fluxes: am has same halo-padded layout
        am_l = flux_scale * am[i, j, k]
        am_r = flux_scale * am[i + 1, j, k]
        # Reconstruction on the full halo-padded array (Nx = Nc + 2Hp for stencil)
        Nx_padded = Int32(Nc + 2 * Hp)
        flux_L = _xface_tracer_flux(Int32(i), j, k, rm, m, am_l, scheme, Nx_padded)
        flux_R = _xface_tracer_flux(Int32(i) + Int32(1), j, k, rm, m, am_r, scheme, Nx_padded)
        rm_new[i, j, k] = rm[i, j, k] + flux_L - flux_R
        m_new[i, j, k]  = m[i, j, k]  + am_l - am_r
    end
end

"""Y-sweep kernel on one CS panel. Interior j ∈ [1,Nc], halo provides neighbors."""
@kernel function _cs_ysweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                     @Const(bm), scheme, Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bm_s = flux_scale * bm[i, j, k]
        bm_n = flux_scale * bm[i, j + 1, k]
        Ny_padded = Int32(Nc + 2 * Hp)
        flux_S = _yface_tracer_flux(i, Int32(j), k, rm, m, bm_s, scheme, Ny_padded)
        flux_N = _yface_tracer_flux(i, Int32(j) + Int32(1), k, rm, m, bm_n, scheme, Ny_padded)
        rm_new[i, j, k] = rm[i, j, k] + flux_S - flux_N
        m_new[i, j, k]  = m[i, j, k]  + bm_s - bm_n
    end
end

"""Z-sweep kernel on one CS panel. Same as LatLon z-kernel but with Hp offset."""
@kernel function _cs_zsweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                     @Const(cm), scheme, Nz, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        cm_t = flux_scale * cm[i, j, k]
        cm_b = flux_scale * cm[i, j, k + 1]
        flux_T = _zface_tracer_flux(i, j, Int32(k), rm, m, cm_t, scheme, Int32(Nz))
        flux_B = _zface_tracer_flux(i, j, Int32(k) + Int32(1), rm, m, cm_b, scheme, Int32(Nz))
        rm_new[i, j, k] = rm[i, j, k] + flux_T - flux_B
        m_new[i, j, k]  = m[i, j, k]  + cm_t - cm_b
    end
end

"""Packed-tracer X-sweep kernel on one CS panel."""
@kernel function _cs_xsweep_mt_kernel!(rm_new_4d, @Const(rm_4d),
                                        m_new, @Const(m),
                                        @Const(am), scheme, Nc, Hp, Nt, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        am_l = flux_scale * am[i, j, k]
        am_r = flux_scale * am[i + 1, j, k]
        Nx_padded = Int32(Nc + 2 * Hp)
        m_new[i, j, k] = m[i, j, k] + am_l - am_r
        for t in Int32(1):Int32(Nt)
            rm_t = TracerView(rm_4d, t)
            flux_L = _xface_tracer_flux(Int32(i), j, k, rm_t, m, am_l, scheme, Nx_padded)
            flux_R = _xface_tracer_flux(Int32(i) + Int32(1), j, k, rm_t, m, am_r, scheme, Nx_padded)
            rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_L - flux_R
        end
    end
end

"""Packed-tracer Y-sweep kernel on one CS panel."""
@kernel function _cs_ysweep_mt_kernel!(rm_new_4d, @Const(rm_4d),
                                        m_new, @Const(m),
                                        @Const(bm), scheme, Nc, Hp, Nt, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bm_s = flux_scale * bm[i, j, k]
        bm_n = flux_scale * bm[i, j + 1, k]
        Ny_padded = Int32(Nc + 2 * Hp)
        m_new[i, j, k] = m[i, j, k] + bm_s - bm_n
        for t in Int32(1):Int32(Nt)
            rm_t = TracerView(rm_4d, t)
            flux_S = _yface_tracer_flux(i, Int32(j), k, rm_t, m, bm_s, scheme, Ny_padded)
            flux_N = _yface_tracer_flux(i, Int32(j) + Int32(1), k, rm_t, m, bm_n, scheme, Ny_padded)
            rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_S - flux_N
        end
    end
end

"""Packed-tracer Z-sweep kernel on one CS panel."""
@kernel function _cs_zsweep_mt_kernel!(rm_new_4d, @Const(rm_4d),
                                        m_new, @Const(m),
                                        @Const(cm), scheme, Nz, Hp, Nt, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        cm_t = flux_scale * cm[i, j, k]
        cm_b = flux_scale * cm[i, j, k + 1]
        m_new[i, j, k] = m[i, j, k] + cm_t - cm_b
        for t in Int32(1):Int32(Nt)
            rm_t = TracerView(rm_4d, t)
            flux_T = _zface_tracer_flux(i, j, Int32(k), rm_t, m, cm_t, scheme, Int32(Nz))
            flux_B = _zface_tracer_flux(i, j, Int32(k) + Int32(1), rm_t, m, cm_b, scheme, Int32(Nz))
            rm_new_4d[i, j, k, t] = rm_4d[i, j, k, t] + flux_T - flux_B
        end
    end
end

# =========================================================================
# Shared support of the per-panel sweeps in cs_sweep_{x,y,z}.jl
#
# Dispatch strategy of those sweeps:
#   UpwindScheme  → hand-written gamma-clamped loops (positivity-safe)
#   SlopesScheme  → KA kernel via _xface_tracer_flux dispatch (needs Hp ≥ 2)
#   PPMScheme     → KA kernel via _xface_tracer_flux dispatch (needs Hp ≥ 3)
# =========================================================================

"""Validate that the halo width Hp is sufficient for the advection scheme's stencil."""
@inline function _validate_halo_for_scheme(scheme::AbstractAdvectionScheme, Hp::Int)
    min_hp = required_halo_width(scheme)
    Hp >= min_hp || error("CS panel sweep with $(typeof(scheme)) requires Hp ≥ $min_hp, got Hp=$Hp. " *
                          "Construct CubedSphereMesh with Hp=$min_hp.")
    return nothing
end

# =========================================================================
# Gamma-clamped tracer flux (from legacy src/Advection/cubed_sphere_mass_flux.jl)
#
# For face flux F through a donor cell with mass m_donor:
#   gamma = clamp(F / m_donor, 0, 1)  (positive F) or clamp(F / m_donor, -1, 0)
#   F_tracer = gamma * rm_donor
#
# When CFL = |F|/m_donor > 1, gamma is clamped to ±1, reducing tracer transport
# to at most the entire donor cell content. Mass update m_new = m + F_in - F_out
# is EXACT (no clamping on mass). Only the tracer flux is limited.
#
# This guarantees rm_new ≥ 0 when rm_src ≥ 0, and preserves mass conservation
# exactly. It's the TM5/FV3/GCHP standard approach for high-CFL cells.
# =========================================================================

"""
    _gamma_clamped_x_flux(F, m_donor, rm_donor) -> tracer_flux

Gamma-clamped upwind tracer flux (legacy cubed_sphere_mass_flux.jl pattern).

Given mass flux `F` [kg] through a face, donor cell mass `m_donor` [kg],
and conservative donor tracer storage `rm_donor` [carrier-air kg]:

    γ = clamp(F / m_donor, {0, 1} or {-1, 0})
    tracer_flux = γ × rm_donor

This ensures:
- When CFL = |F|/m ≤ 1 (normal): `γ = F/m`, recovering first-order upwind.
- When CFL > 1 (overshooting): `γ` is clamped to ±1, so the tracer flux
  never exceeds the donor cell's total tracer storage. This guarantees
  `rm_new ≥ 0` when `rm ≥ 0` (positivity preservation).
- Mass update `m_new = m + F_west − F_east` is EXACT (unclamped), so total
  mass is conserved. Only the tracer distribution is limited.

The gamma clamping should ideally not be needed if CFL < 1 via the
subcycling pilot. It's a safety net for preprocessing flux-inconsistency
(see CLAUDE.md: clamps should ideally not be needed).
"""
@inline function _gamma_clamped_x_flux(F::FT, m_donor::FT, rm_donor::FT) where FT
    m_donor > zero(FT) || return zero(FT)
    # γ = F/m clamped to [0, 1] for positive flux, [-1, 0] for negative
    gamma = F >= zero(FT) ?
        clamp(F / m_donor, zero(FT), one(FT)) :
        clamp(F / m_donor, -one(FT), zero(FT))
    return gamma * rm_donor
end

# Backend extensions can choose a tile for the packed panel kernels without
# changing their per-cell arithmetic or the CPU/other-scheme launch defaults.
@inline _cs_packed_sweep_workgroupsize(backend, scheme, ::Type) = 256

@inline _cs_gpu_profile_enabled() =
    SectionTimer.is_enabled() &&
    lowercase(get(ENV, "ATMOSTR_PROFILE_GPU", "")) in ("1", "true", "on", "yes")

@inline function _profiled_launch_and_sync!(launch!::F, backend, launch_section::Symbol,
                                            sync_section::Symbol) where {F}
    if _cs_gpu_profile_enabled()
        t0 = time_ns()
        launch!()
        SectionTimer.record_sample!(launch_section, Float64(time_ns() - t0))
        t1 = time_ns()
        synchronize(backend)
        SectionTimer.record_sample!(sync_section, Float64(time_ns() - t1))
    else
        launch!()
        # Intra-stream workspace dependency only: panel p's sweep/copy-back and
        # panel p+1's sweep share `rm_4d_A`/`m_A`, but they are issued on one
        # ordered GPU stream, so no host barrier is needed on GPU — the periodic
        # sync lands at the `fill_panel_halos!` boundary. The per-kernel host
        # `synchronize` here was the dominant launch-bound bubble (GPU profiling
        # 2026-06-13: ~3 host barriers per panel per sweep). Keep it on the CPU
        # backend defensively, mirroring HaloExchange.jl's `KA_CPU` gate.
        backend isa KA_CPU && synchronize(backend)
    end
    return nothing
end

@inline function _profiled_copy!(copy!::F, section::Symbol) where {F}
    if _cs_gpu_profile_enabled()
        SectionTimer.time_section(copy!, section)
    else
        copy!()
    end
    return nothing
end
