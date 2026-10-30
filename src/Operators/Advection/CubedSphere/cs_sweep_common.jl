# Cubed-sphere panel sweeps, shared part: higher-order KA kernels, halo check, gamma-clamped upwind flux, profiled launches.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend, CPU as KA_CPU

# =========================================================================
# CS panel sweep kernels
#
# These launch on ndrange=(Nc, Nc, Nz) and read/write the interior region
# of halo-padded arrays. The Hp offset is added to all indices.
# =========================================================================

# Packed generic kernels call the shared face reconstruction. Upwind uses
# the gamma-clamped kernels in cs_sweep_{x,y,z}.jl. Host launchers below retain
# each scheme's original argument types, timer sections and workgroups.

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
        m_new[i, j, k] = m[i, j, k] + am_l - am_r
        for t in Int32(1):Int32(Nt)
            rm_t = TracerView(rm_4d, t)
            flux_L = _xface_tracer_flux(Int32(i), j, k, rm_t, m, am_l, scheme, _HaloPaddedAxis())
            flux_R = _xface_tracer_flux(Int32(i) + Int32(1), j, k, rm_t, m, am_r, scheme, _HaloPaddedAxis())
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
        # ordered GPU stream, so no host barrier is needed on GPU — the packed
        # palindrome synchronizes once at its end (`strang_split_cs_mt!`), and
        # unpacked callers at their `fill_panel_halos!` (sync = true). The per-kernel host
        # `synchronize` here was the dominant launch-bound bubble (GPU profiling
        # 2026-06-13: ~3 host barriers per panel per sweep). On the CPU backend
        # kernels already run synchronously and this call is a no-op; it is kept
        # so the loop reads the same on every backend.
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

# Host-only axis dispatch. Kernels, argument types, workgroups and timer labels
# are the same as the former directional launch wrappers.
@inline _cs_panel_kernel(::Val{1}, ::AbstractAdvectionScheme) = _cs_xsweep_mt_kernel!
@inline _cs_panel_kernel(::Val{2}, ::AbstractAdvectionScheme) = _cs_ysweep_mt_kernel!
@inline _cs_panel_kernel(::Val{3}, ::AbstractAdvectionScheme) = _cs_zsweep_mt_kernel!
@inline _cs_panel_kernel(::Val{1}, ::UpwindScheme) = _cs_xsweep_mt_upwind_kernel!
@inline _cs_panel_kernel(::Val{2}, ::UpwindScheme) = _cs_ysweep_mt_upwind_kernel!
@inline _cs_panel_kernel(::Val{3}, ::UpwindScheme) = _cs_zsweep_mt_upwind_kernel!
@inline _cs_panel_sections(::Val{1}) = (:cs_kernel_launch_x_mt, :cs_kernel_sync_x_mt)
@inline _cs_panel_sections(::Val{2}) = (:cs_kernel_launch_y_mt, :cs_kernel_sync_y_mt)
@inline _cs_panel_sections(::Val{3}) = (:cs_kernel_launch_z_mt, :cs_kernel_sync_z_mt)

function _sweep_cs_panel_mt_pingpong!(rm_out, m_out, rm, m, flux,
                                      scheme::AbstractAdvectionScheme,
                                      Nc, Hp, Nz, Nt, direction::Val{D};
                                      flux_scale = one(eltype(m))) where D
    D == 3 || _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm)
    workgroupsize = _cs_packed_sweep_workgroupsize(backend, scheme, FT)
    kernel! = _cs_panel_kernel(direction, scheme)(backend, workgroupsize)
    launch_section, sync_section = _cs_panel_sections(direction)
    _profiled_launch_and_sync!(backend, launch_section, sync_section) do
        kernel!(rm_out, rm, m_out, m, flux, scheme,
                Int32(D == 3 ? Nz : Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    return nothing
end

function _sweep_cs_panel_mt_pingpong!(rm_out, m_out, rm, m, flux,
                                      scheme::UpwindScheme,
                                      Nc, Hp, Nz, Nt, direction::Val{D};
                                      flux_scale = one(eltype(m))) where D
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_panel_kernel(direction, scheme)(backend, 256)
    launch_section, sync_section = _cs_panel_sections(direction)
    _profiled_launch_and_sync!(backend, launch_section, sync_section) do
        kernel!(rm_out, rm, m_out, m, flux,
                Int32(D == 3 ? Nz : Nc), Int32(Hp), Int32(Nt), FT(flux_scale);
                ndrange=(Nc, Nc, Nz))
    end
    return nothing
end

function _sweep_cs_horizontal_pingpong!(rm_out::NTuple{6}, m_out::NTuple{6},
                                        rm::NTuple{6}, m::NTuple{6}, flux::NTuple{6},
                                        mesh::CubedSphereMesh, scheme::AbstractAdvectionScheme,
                                        direction::Val{D};
                                        flux_scale = one(eltype(m[1])),
                                        seam_flux = nothing) where D
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(m[1], 3)
    Nt = size(rm[1], 4)
    cache = seam_flux === nothing ?
        similar(rm[1], eltype(m[1]), Nc, Nz, Nt + 1, 12) : seam_flux
    _cache_cs_seams!(cache, rm, m, flux, mesh, scheme, direction, flux_scale)
    for p in 1:6
        _sweep_cs_panel_mt_pingpong!(rm_out[p], m_out[p], rm[p], m[p],
            _CSInteriorFlux(flux[p], mesh, direction), scheme, Nc, Hp, Nz, Nt,
            direction; flux_scale)
    end
    _apply_cs_seams!(rm_out, m_out, cache, mesh, direction)
    return rm_out, m_out
end
