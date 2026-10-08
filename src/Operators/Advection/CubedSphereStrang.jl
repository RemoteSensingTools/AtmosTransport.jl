# ---------------------------------------------------------------------------
# Cubed-sphere Strang splitting orchestrator for src
#
# Performs X → Y → Z → Z → Y → X dimensionally-split advection on 6
# gnomonic panels with halo exchange between horizontal sweeps.
#
# Panel-interior kernels reuse the SAME reconstruction functions
# (_xface_tracer_flux, _yface_tracer_flux, _zface_tracer_flux) as the
# LatLon path. The only CS-specific logic is:
#   1. Halo exchange after each horizontal sweep (fill_panel_halos!)
#   2. Kernel launch on interior indices with Hp offset
#   3. Per-panel loop over 6 panels
#   4. Paired physical-seam transfers within each horizontal group
#
# The panel arrays have layout (Nc+2Hp, Nc+2Hp, Nz) with interior at
# [Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]. The reconstruction stencil reads into
# the halo region naturally.
#
# Each global X/Y group includes its panel-interior faces plus physical seams
# owned by the lower-numbered panel's corresponding local axis. CubedSphereSeams
# caches those transfers before any panel update and applies them with opposite
# signs to both neighbors, including rotated X/Y contacts. Each group therefore
# conserves air and tracer mass for mirrored input fluxes. Vertical sweeps are
# panel-local and closed. Temporal accuracy at seams needs separate validation.
#
# References:
#   Strang (1968) — symmetric splitting for second-order accuracy
#   Putman & Lin (2007) — FV3 cubed-sphere transport
# ---------------------------------------------------------------------------

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
# Per-panel sweep functions (double-buffered)
#
# Dispatch strategy:
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

# =========================================================================
# CS workspace — pre-allocated buffers for one panel
# =========================================================================

"""
    CSAdvectionWorkspace{FT, A, P3, A4, P4}

Pre-allocated cubed-sphere transport workspace.

- `rm_A`, `m_A` are the halo-padded single-tracer advection ping-pong
  buffers shared across panels.
- `rm_4d_A` is the packed-tracer panel buffer used by the production
  split-sweep path so CS follows the same packed `tracers_raw` paradigm
  as structured grids.
- `m_pp_buf`, `rm_4d_pp_buf` are full-panel spare buffers for the packed
  ping-pong path, avoiding the per-sweep copy-back kernels.
- `seam_flux` holds one air and N-tracer transfer per physical panel edge;
  its storage scales with edge length, not panel area. Lin–Rood workspaces
  disable this split-sweep cache (`seam_transport=false`).
- `column_scratch` is the per-column, per-tracer working storage of the FV3
  vertical profile (`Nc × Nc × (Nz+1) × 3 max(Nt_s, 1)`, `Nt_s =
  column_scratch_tracers`, the packed tracer count by default); it is empty
  unless the workspace is built with `column_scratch=true`.
- `max_subcycles` tracks this workspace's high-water mark for CFL diagnostics;
  keeping it with the workspace prevents unrelated simulations sharing state.
"""
struct CSAdvectionWorkspace{FT, A <: AbstractArray{FT, 3},
                            P3 <: NTuple{6, <:AbstractArray{FT, 3}},
                            A4 <: AbstractArray{FT, 4},
                            P4 <: NTuple{6, <:AbstractArray{FT, 4}}}
    rm_A       :: A
    m_A        :: A
    rm_4d_A    :: A4
    m_pp_buf   :: P3
    rm_4d_pp_buf :: P4
    seam_flux  :: A4
    column_scratch :: A4
    max_subcycles :: Base.RefValue{NTuple{3, Int}}
end

function CSAdvectionWorkspace(mesh::CubedSphereMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              array_type::Type{<:AbstractArray} = Array,
                              n_tracers::Integer = 0,
                              seam_transport::Bool = true,
                              column_scratch::Bool = false,
                              column_scratch_tracers::Integer = n_tracers)
    N = mesh.Nc + 2 * mesh.Hp
    Nt = Int(n_tracers)
    Nt >= 0 || throw(ArgumentError("CSAdvectionWorkspace: n_tracers must be non-negative, got $n_tracers"))
    rm_A = array_type(zeros(FT, N, N, Nz))
    m_A  = array_type(zeros(FT, N, N, Nz))
    rm_4d_A = array_type(zeros(FT, N, N, Nz, Nt))
    m_pp_buf = Nt > 0 ? ntuple(_ -> array_type(zeros(FT, N, N, Nz)), 6) :
                         ntuple(_ -> m_A, 6)
    rm_4d_pp_buf = Nt > 0 ? ntuple(_ -> array_type(zeros(FT, N, N, Nz, Nt)), 6) :
                            ntuple(_ -> rm_4d_A, 6)
    seam_flux = similar(rm_4d_A, FT, mesh.Nc, Nz, max(Nt, 1) + 1, seam_transport ? 12 : 0)
    scratch = _column_scratch(rm_4d_A, mesh.Nc, Nz, Int(column_scratch_tracers), column_scratch)
    return CSAdvectionWorkspace{FT, typeof(rm_A),
                                typeof(m_pp_buf), typeof(rm_4d_A),
                                typeof(rm_4d_pp_buf)}(
        rm_A, m_A, rm_4d_A, m_pp_buf, rm_4d_pp_buf, seam_flux, scratch,
        Ref((1, 1, 1)))
end

function CSAdvectionWorkspace(mesh::CubedSphereMesh,
                              prototype::AbstractArray{FT, 3};
                              n_tracers::Integer = 0,
                              seam_transport::Bool = true,
                              column_scratch::Bool = false,
                              column_scratch_tracers::Integer = n_tracers) where {FT <: AbstractFloat}
    N = mesh.Nc + 2 * mesh.Hp
    Nz = size(prototype, 3)
    Nt = Int(n_tracers)
    Nt >= 0 || throw(ArgumentError("CSAdvectionWorkspace: n_tracers must be non-negative, got $n_tracers"))
    rm_A = similar(prototype, FT, N, N, Nz)
    m_A = similar(prototype, FT, N, N, Nz)
    rm_4d_A = similar(prototype, FT, N, N, Nz, Nt)
    m_pp_buf = Nt > 0 ? ntuple(_ -> similar(prototype, FT, N, N, Nz), 6) :
                         ntuple(_ -> m_A, 6)
    rm_4d_pp_buf = Nt > 0 ? ntuple(_ -> similar(prototype, FT, N, N, Nz, Nt), 6) :
                            ntuple(_ -> rm_4d_A, 6)
    seam_flux = similar(rm_4d_A, FT, mesh.Nc, Nz, max(Nt, 1) + 1, seam_transport ? 12 : 0)
    scratch = _column_scratch(rm_4d_A, mesh.Nc, Nz, Int(column_scratch_tracers), column_scratch)
    return CSAdvectionWorkspace{FT, typeof(rm_A),
                                typeof(m_pp_buf), typeof(rm_4d_A),
                                typeof(rm_4d_pp_buf)}(
        rm_A, m_A, rm_4d_A, m_pp_buf, rm_4d_pp_buf, seam_flux, scratch,
        Ref((1, 1, 1)))
end

# Per-column, per-tracer working storage of the FV3 vertical profile
# (`FV3Column` in `vertical_fv3_profile.jl`); empty unless a scheme needs it.
_column_scratch(prototype, Nc, Nz, Nt, needed::Bool) =
    needed ? similar(prototype, Nc, Nc, Nz + 1, 3 * max(Nt, 1)) : similar(prototype, 0, 0, 0, 0)

function Adapt.adapt_structure(to, ws::CSAdvectionWorkspace{FT}) where FT
    rm_A = Adapt.adapt(to, ws.rm_A)
    m_A = Adapt.adapt(to, ws.m_A)
    rm_4d_A = Adapt.adapt(to, ws.rm_4d_A)
    m_pp_buf = Adapt.adapt(to, ws.m_pp_buf)
    rm_4d_pp_buf = Adapt.adapt(to, ws.rm_4d_pp_buf)
    seam_flux = Adapt.adapt(to, ws.seam_flux)
    column_scratch = Adapt.adapt(to, ws.column_scratch)
    return CSAdvectionWorkspace{FT, typeof(rm_A),
                                typeof(m_pp_buf), typeof(rm_4d_A),
                                typeof(rm_4d_pp_buf)}(
        rm_A, m_A, rm_4d_A, m_pp_buf, rm_4d_pp_buf, seam_flux, column_scratch,
        Ref(ws.max_subcycles[]))
end

@inline function _record_cs_subcycle_growth!(workspace::CSAdvectionWorkspace,
                                              n_x::Int, n_y::Int, n_z::Int)
    mx, my, mz = workspace.max_subcycles[]
    if n_x > mx || n_y > my || n_z > mz
        workspace.max_subcycles[] = (max(mx, n_x), max(my, n_y), max(mz, n_z))
        @info "strang_split_cs! subcycle count grew" n_x n_y n_z
    end
    return nothing
end

# =========================================================================
# Public API: strang_split_cs!
# =========================================================================

# =========================================================================
# CFL-based subcycle count
# =========================================================================

"""Static CFL subcycle count from initial mass (no evolving-mass pilot).

The per-cell positivity bound for one Strang half-sweep is

    outgoing_mass_per_substep = max(0, −F_lo) + max(0, F_hi)
    cfl                       = outgoing_mass_per_substep / m

Both faces can carry mass *out of* the cell simultaneously at a
divergent stagnation point; the previous formulation
`max(|F_lo|, |F_hi|) / m` only measured the larger of the two and
under-estimated by up to 2× at exactly the cells where positivity
fails (both faces outgoing).  The new formula is a *correctness
refinement* — Lin-Rood 1996's positivity criterion — and is what the
runtime actually needs to subcycle on; it is not a strict tightening
in every direction (e.g. a pure-inflow cell with `F_lo > 0, F_hi < 0`
gets `outgoing = 0` here vs. `max(|F_lo|, |F_hi|)` under the old
formula — but that's exactly right: a cell receiving on both faces
loses no mass and needs no subcycling).

Sign convention for each face `F_lo` (lower-index face) and `F_hi`
(higher-index face): positive flux means mass flows in the +index
direction, so the cell loses mass when `F_lo < 0` (out the low side)
or `F_hi > 0` (out the high side). The same convention holds for
all three directions in this code path.

Implementation runs every `step!`, so we use a backend-portable
broadcast + `mapreduce(max, …)` formulation:

- on `Array` it lowers to vectorised SIMD reductions (no allocation),
- on `CuArray` it dispatches to CUDA's parallel reduction (no host
  round-trip).

The `m <= 0` guard from the original scalar loop is preserved via
`ifelse`; in practice `m > 0` always holds and the guard is
defensive.
"""
function _cs_static_subcycle_count(panels_flux::NTuple{6}, panels_m::NTuple{6},
                                    Nc::Int, Hp::Int, Nz::Int, cfl_limit::Real,
                                    direction::Symbol;
                                    flux_scale = one(eltype(panels_m[1])))
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    iL = Hp + 1
    iH = Hp + Nc
    max_cfl = zero(FT)
    @inbounds for p in 1:6
        m_p = panels_m[p]
        F_p = panels_flux[p]
        m_int = view(m_p, iL:iH, iL:iH, 1:Nz)
        F_lo, F_hi = if direction === :x
            (view(F_p, iL    :iH,     iL:iH,     1:Nz),
             view(F_p, iL + 1:iH + 1, iL:iH,     1:Nz))
        elseif direction === :y
            (view(F_p, iL:iH,     iL    :iH,     1:Nz),
             view(F_p, iL:iH,     iL + 1:iH + 1, 1:Nz))
        else  # :z
            (view(F_p, iL:iH, iL:iH, 1    :Nz),
             view(F_p, iL:iH, iL:iH, 2:Nz + 1))
        end
        zero_FT = zero(FT)
        cfl_panel = mapreduce(max, m_int, F_lo, F_hi; init = zero_FT) do mi, fl, fh
            fls = fs * fl
            fhs = fs * fh
            outgoing = max(zero_FT, -fls) + max(zero_FT, fhs)
            ifelse(mi > zero_FT, outgoing / mi, zero_FT)
        end
        max_cfl = max(max_cfl, cfl_panel)
    end
    max_cfl <= cfl_limit && return 1
    return ceil(Int, max_cfl / cfl_limit)
end

"""Static palindrome CFL subcycle count from initial mass.

This is the runtime-side second line of defense for the CS Strang sequence.
The actual sequence applies each direction twice (`X-Y-Z-Z-Y-X`), so a
per-direction CFL pilot can under-estimate cells where moderate outgoing flux
exists in several directions at once. This budget is still a static proxy, not
an evolving-mass proof, but it is conservative with respect to the old
direction-isolated metric and matches the preprocessor's adaptive schedule
gate.
"""
function _cs_static_palindrome_subcycle_count(panels_am::NTuple{6},
                                              panels_bm::NTuple{6},
                                              panels_cm::NTuple{6},
                                              panels_m::NTuple{6},
                                              Nc::Int, Hp::Int, Nz::Int,
                                              cfl_limit::Real;
                                              flux_scale = one(eltype(panels_m[1])),
                                              max_n_sub::Int = 4096)
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    iL = Hp + 1
    iH = Hp + Nc
    max_cfl = zero(FT)
    @inbounds for p in 1:6
        m_int = view(panels_m[p], iL:iH, iL:iH, 1:Nz)
        ax_lo = view(panels_am[p], iL    :iH,     iL:iH,     1:Nz)
        ax_hi = view(panels_am[p], iL + 1:iH + 1, iL:iH,     1:Nz)
        by_lo = view(panels_bm[p], iL:iH,     iL    :iH,     1:Nz)
        by_hi = view(panels_bm[p], iL:iH,     iL + 1:iH + 1, 1:Nz)
        cz_lo = view(panels_cm[p], iL:iH, iL:iH, 1    :Nz)
        cz_hi = view(panels_cm[p], iL:iH, iL:iH, 2:Nz + 1)
        zero_FT = zero(FT)
        cfl_panel = mapreduce(max, m_int, ax_lo, ax_hi, by_lo, by_hi,
                              cz_lo, cz_hi; init = zero_FT) do mi, axl, axh, byl, byh, czl, czh
            out_x = max(zero_FT, -(fs * axl)) + max(zero_FT, fs * axh)
            out_y = max(zero_FT, -(fs * byl)) + max(zero_FT, fs * byh)
            out_z = max(zero_FT, -(fs * czl)) + max(zero_FT, fs * czh)
            outgoing_half = out_x + out_y + out_z
            outgoing = outgoing_half + outgoing_half
            ifelse(mi > zero_FT, outgoing / mi, zero_FT)
        end
        max_cfl = max(max_cfl, cfl_panel)
    end
    max_cfl <= cfl_limit && return 1
    n_sub = ceil(Int, max_cfl / cfl_limit)
    n_sub <= max_n_sub ||
        error("cubed-sphere palindrome subcycling exceeded max_n_sub=$(max_n_sub)")
    return n_sub
end


# NOTE: Evolving-mass pilot functions (_cs_x/y/z_pilot_subcycle_count) were
# removed — they were dead code (`strang_split_cs!` uses the shared static
# palindrome budget above).
# The static pilot is sufficient because the gamma-clamped sweep handles CFL > 1
# safely. If evolving-mass pilots are needed in the future, see git history.

"""
    strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                     mesh, scheme, workspace; flux_scale=1, cfl_limit=0.95)

Perform one Strang-split advection step on a 6-panel cubed-sphere field
with automatic CFL-based subcycling per direction.

## Splitting sequence

    X sweep (n_x subcycles)
    → fill_panel_halos!(dir=1)     ← exchange halos between panels (X direction)
    → Y sweep (n_y subcycles)
    → fill_panel_halos!(dir=2)     ← exchange halos between panels (Y direction)
    → Z sweep (n_z subcycles)      ← first Z half-step
    → Z sweep (n_z subcycles)      ← second Z half-step (palindrome)
    → fill_panel_halos!(dir=2)
    → Y sweep (n_y subcycles)
    → fill_panel_halos!(dir=1)
    → X sweep (n_x subcycles)

This sequence follows Strang's X-Y-Z-Z-Y-X ordering. Second-order splitting
requires sufficiently accurate subflows; the palindrome alone does not prove
the transport's temporal order. Horizontal groups pair physical seam transfers
before halo exchange supplies neighboring values for the next reconstruction.

## Subcycling

All six legs use one count from the initial-mass palindrome budget
`2 * (out_x + out_y + out_z) / m_start`, with flux divided by that count.
A supplied `subcycle_count` uses the binary's schedule; setting
`ATMOSTR_ASSERT_CS_BINARY_CFL=1` checks it against the runtime budget.

## Panel array layout

Each panel's rm and m arrays are `(Nc+2Hp, Nc+2Hp, Nz)` with Hp-wide halos.
Interior cells are at indices `[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]`. The sweep
kernels only update interior cells; halo regions are filled by
`fill_panel_halos!` from adjacent panels.

## Arguments

- `panels_rm`, `panels_m`: NTuple{6} of 3D arrays `(Nc+2Hp, Nc+2Hp, Nz)` —
  tracer mass and air mass. Modified in-place.
- `panels_am`, `panels_bm`, `panels_cm`: NTuple{6} of flux arrays.
  `am[Nc+2Hp+1, Nc+2Hp, Nz]`, `bm[Nc+2Hp, Nc+2Hp+1, Nz]`,
  `cm[Nc+2Hp, Nc+2Hp, Nz+1]`. Read-only.
- `mesh`: `CubedSphereMesh` with Nc, Hp, and panel connectivity.
- `scheme`: advection scheme — `UpwindScheme()` uses gamma-clamped upwind
  for tracer face transfers. `SlopesScheme()` and `PPMScheme()` use the generic
  KA kernels with `_xface_tracer_flux` dispatch. Higher-order schemes
  require `mesh.Hp ≥ 2` (Slopes) or `mesh.Hp ≥ 3` (PPM).
- `workspace`: pre-allocated `CSAdvectionWorkspace` buffers.
- `flux_scale`: overall scaling applied to all fluxes (default 1.0).
- `cfl_limit`: maximum CFL per subcycle pass (default 0.95).
"""
function strang_split_cs!(panels_rm::NTuple{6},
                          panels_m::NTuple{6},
                          panels_am::NTuple{6},
                          panels_bm::NTuple{6},
                          panels_cm::NTuple{6},
                          mesh::CubedSphereMesh,
                          scheme,
                          workspace::CSAdvectionWorkspace;
                          flux_scale = one(eltype(panels_m[1])),
                          cfl_limit::Real = 0.95,
                          subcycle_count::Union{Nothing, Integer} = nothing,
                          midpoint! = nothing)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_rm[1], 3)
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    cfl_ft = convert(FT, cfl_limit)

    n_pal = if subcycle_count === nothing
        # Budget all six legs against initial carrier mass. Face-local tracer
        # clamping does not replace a safe total-outflow budget.
        SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)
    else
        n = Int(subcycle_count)
        n >= 1 || throw(ArgumentError("strang_split_cs!: subcycle_count must be ≥ 1, got $(subcycle_count)"))
        if get(ENV, "ATMOSTR_ASSERT_CS_BINARY_CFL", "0") == "1"
            required = SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
                panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
                flux_scale = fs)
            required <= n || throw(ArgumentError(
                "strang_split_cs!: binary substep contract requested " *
                "subcycle_count=$n, but runtime CFL assertion requires " *
                "$required. Regenerate the binary or disable " *
                "ATMOSTR_ASSERT_CS_BINARY_CFL for diagnostic runs."))
        end
        n
    end
    n_x = n_pal
    n_y = n_pal
    n_z = n_pal

    _record_cs_subcycle_growth!(workspace, n_x, n_y, n_z)

    fs_x = fs / FT(n_x)
    fs_y = fs / FT(n_y)
    fs_z = fs / FT(n_z)

    # ---- X sweep (subcycled) ----
    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_cs_horizontal!(panels_rm, panels_m, panels_am, mesh,
                              scheme, workspace, Val(1); flux_scale=fs_x)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm, mesh; dir=1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,  mesh; dir=1)
    end

    # ---- Y sweep (subcycled) ----
    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_cs_horizontal!(panels_rm, panels_m, panels_bm, mesh,
                              scheme, workspace, Val(2); flux_scale=fs_y)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm, mesh; dir=2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,  mesh; dir=2)
    end

    # ---- Z sweep × 2 (subcycled) ----
    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        _sweep_z_panels!(panels_rm, panels_m, panels_cm, mesh, scheme, workspace;
                         flux_scale = fs_z)
    end

    midpoint! === nothing || midpoint!()

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        _sweep_z_panels!(panels_rm, panels_m, panels_cm, mesh, scheme, workspace;
                         flux_scale = fs_z)
    end

    # ---- Reverse: Y sweep (subcycled) ----
    SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm, mesh; dir=2)
    SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,  mesh; dir=2)
    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_cs_horizontal!(panels_rm, panels_m, panels_bm, mesh,
                              scheme, workspace, Val(2); flux_scale=fs_y)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm, mesh; dir=2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,  mesh; dir=2)
    end

    # ---- Reverse: X sweep (subcycled) ----
    SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm, mesh; dir=1)
    SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,  mesh; dir=1)
    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_cs_horizontal!(panels_rm, panels_m, panels_am, mesh,
                              scheme, workspace, Val(1); flux_scale=fs_x)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm, mesh; dir=1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,  mesh; dir=1)
    end

    return nothing
end

@inline function _check_cs_packed_workspace(workspace::CSAdvectionWorkspace, Nt::Int)
    size(workspace.rm_4d_A, 4) >= Nt || throw(ArgumentError(
        "CSAdvectionWorkspace was built for $(size(workspace.rm_4d_A, 4)) packed tracers, " *
        "but the state has $Nt. Rebuild the workspace with `n_tracers = ntracers(state)` " *
        "or construct the `TransportModel` without overriding `workspace`."))
    size(workspace.rm_4d_pp_buf[1], 4) >= Nt || throw(ArgumentError(
        "CSAdvectionWorkspace ping-pong buffers were built for " *
        "$(size(workspace.rm_4d_pp_buf[1], 4)) packed tracers, but the state has $Nt. " *
        "Rebuild the workspace with `n_tracers = ntracers(state)`."))
    size(workspace.m_pp_buf[1], 3) == size(workspace.m_A, 3) || throw(ArgumentError(
        "CSAdvectionWorkspace mass ping-pong buffers are not allocated for packed transport. " *
        "Rebuild the workspace with `n_tracers = ntracers(state)`."))
    return nothing
end

"""
    strang_split_cs_mt!(panels_rm_4d, panels_m, panels_am, panels_bm, panels_cm,
                        mesh, scheme, workspace; ...)

Packed-tracer cubed-sphere split-sweep transport. This is the production CS
path for `CSSplitSweepStyle` schemes: air mass is advanced once per sweep and
all tracers in each panel's fourth dimension are updated inside the same panel
kernel. The sequence and CFL contract match [`strang_split_cs!`](@ref).
"""
function _strang_split_cs_mt_copyback!(panels_rm_4d::NTuple{6},
                                       panels_m::NTuple{6},
                                       panels_am::NTuple{6},
                                       panels_bm::NTuple{6},
                                       panels_cm::NTuple{6},
                                       mesh::CubedSphereMesh,
                                       scheme::AbstractAdvectionScheme,
                                       workspace::CSAdvectionWorkspace;
                                       flux_scale = one(eltype(panels_m[1])),
                                       cfl_limit::Real = 0.95,
                                       subcycle_count::Union{Nothing, Integer} = nothing,
                                       midpoint! = nothing)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    _check_cs_packed_workspace(workspace, Nt)
    rm_4d_A, m_A = workspace.rm_4d_A, workspace.m_A
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    cfl_ft = convert(FT, cfl_limit)

    n_pal = if subcycle_count === nothing
        SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)
    else
        n = Int(subcycle_count)
        n >= 1 || throw(ArgumentError("strang_split_cs_mt!: subcycle_count must be ≥ 1, got $(subcycle_count)"))
        if get(ENV, "ATMOSTR_ASSERT_CS_BINARY_CFL", "0") == "1"
            required = SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
                panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
                flux_scale = fs)
            required <= n || throw(ArgumentError(
                "strang_split_cs_mt!: binary substep contract requested " *
                "subcycle_count=$n, but runtime CFL assertion requires $required."))
        end
        n
    end
    n_x = n_pal
    n_y = n_pal
    n_z = n_pal

    _record_cs_subcycle_growth!(workspace, n_x, n_y, n_z)

    fs_x = fs / FT(n_x)
    fs_y = fs / FT(n_y)
    fs_z = fs / FT(n_z)

    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_cs_horizontal!(panels_rm_4d, panels_m, panels_am, mesh,
                              scheme, workspace, Val(1); flux_scale=fs_x)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm_4d, mesh; dir = 1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,     mesh; dir = 1)
    end

    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_cs_horizontal!(panels_rm_4d, panels_m, panels_bm, mesh,
                              scheme, workspace, Val(2); flux_scale=fs_y)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm_4d, mesh; dir = 2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,     mesh; dir = 2)
    end

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        for p in 1:6
            _sweep_z_panel_mt!(panels_rm_4d[p], panels_m[p], panels_cm[p],
                               scheme, rm_4d_A, m_A, Nc, Hp, Nz, Nt; flux_scale = fs_z)
        end
    end

    midpoint! === nothing || midpoint!()

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        for p in 1:6
            _sweep_z_panel_mt!(panels_rm_4d[p], panels_m[p], panels_cm[p],
                               scheme, rm_4d_A, m_A, Nc, Hp, Nz, Nt; flux_scale = fs_z)
        end
    end

    SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm_4d, mesh; dir = 2)
    SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,     mesh; dir = 2)
    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_cs_horizontal!(panels_rm_4d, panels_m, panels_bm, mesh,
                              scheme, workspace, Val(2); flux_scale=fs_y)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(panels_rm_4d, mesh; dir = 2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(panels_m,     mesh; dir = 2)
    end

    SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm_4d, mesh; dir = 1)
    SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,     mesh; dir = 1)
    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_cs_horizontal!(panels_rm_4d, panels_m, panels_am, mesh,
                              scheme, workspace, Val(1); flux_scale=fs_x)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(panels_rm_4d, mesh; dir = 1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(panels_m,     mesh; dir = 1)
    end

    return nothing
end

function strang_split_cs_mt!(panels_rm_4d::NTuple{6},
                             panels_m::NTuple{6},
                             panels_am::NTuple{6},
                             panels_bm::NTuple{6},
                             panels_cm::NTuple{6},
                             mesh::CubedSphereMesh,
                             scheme::AbstractAdvectionScheme,
                             workspace::CSAdvectionWorkspace;
                             flux_scale = one(eltype(panels_m[1])),
                             cfl_limit::Real = 0.95,
                             subcycle_count::Union{Nothing, Integer} = nothing,
                             midpoint! = nothing)
    strang_split_cs_mt_pingpong!(panels_rm_4d, panels_m,
                                 workspace.rm_4d_pp_buf, workspace.m_pp_buf,
                                 panels_am, panels_bm, panels_cm, mesh, scheme,
                                 workspace; flux_scale, cfl_limit,
                                 subcycle_count, midpoint!)
    return nothing
end

"""
    strang_split_cs_mt_pingpong!(panels_rm_4d, panels_m, panels_rm_4d_buf, panels_m_buf,
                                 panels_am, panels_bm, panels_cm, mesh, scheme, workspace; ...)

Packed-tracer CS split-sweep that writes each sweep directly into alternate
panel buffers and swaps active/inactive tuples between sweeps. This removes the
per-sweep copy-back kernels while keeping the existing KA sweep kernels. The
final active `(rm, m)` tuple is returned.
"""
function strang_split_cs_mt_pingpong!(panels_rm_4d::NTuple{6},
                                      panels_m::NTuple{6},
                                      panels_rm_4d_buf::NTuple{6},
                                      panels_m_buf::NTuple{6},
                                      panels_am::NTuple{6},
                                      panels_bm::NTuple{6},
                                      panels_cm::NTuple{6},
                                      mesh::CubedSphereMesh,
                                      scheme::AbstractAdvectionScheme,
                                      workspace::CSAdvectionWorkspace;
                                      flux_scale = one(eltype(panels_m[1])),
                                      cfl_limit::Real = 0.95,
                                      subcycle_count::Union{Nothing, Integer} = nothing,
                                      midpoint! = nothing)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    _check_cs_packed_workspace(workspace, Nt)
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    cfl_ft = convert(FT, cfl_limit)

    n_pal = if subcycle_count === nothing
        SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
            panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
            flux_scale = fs)
    else
        n = Int(subcycle_count)
        n >= 1 || throw(ArgumentError("strang_split_cs_mt_pingpong!: subcycle_count must be ≥ 1, got $(subcycle_count)"))
        if get(ENV, "ATMOSTR_ASSERT_CS_BINARY_CFL", "0") == "1"
            required = SectionTimer.@section :cs_cfl_x _cs_static_palindrome_subcycle_count(
                panels_am, panels_bm, panels_cm, panels_m, Nc, Hp, Nz, cfl_ft;
                flux_scale = fs)
            required <= n || throw(ArgumentError(
                "strang_split_cs_mt_pingpong!: binary substep contract requested " *
                "subcycle_count=$n, but runtime CFL assertion requires $required."))
        end
        n
    end
    n_x = n_pal
    n_y = n_pal
    n_z = n_pal

    _record_cs_subcycle_growth!(workspace, n_x, n_y, n_z)

    fs_x = fs / FT(n_x)
    fs_y = fs / FT(n_y)
    fs_z = fs / FT(n_z)

    active_rm = panels_rm_4d
    active_m = panels_m
    spare_rm = panels_rm_4d_buf
    spare_m = panels_m_buf

    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_x_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_am, mesh, scheme; flux_scale = fs_x,
                                     seam_flux = workspace.seam_flux)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(spare_rm, mesh; dir = 1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(spare_m,  mesh; dir = 1)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_y_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_bm, mesh, scheme; flux_scale = fs_y,
                                     seam_flux = workspace.seam_flux)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(spare_rm, mesh; dir = 2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(spare_m,  mesh; dir = 2)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        _sweep_z_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_cm, mesh, scheme, workspace;
                                     flux_scale = fs_z)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    # The midpoint operators (diffusion / surface flux) must act on the CURRENT
    # active ping-pong buffer, not `state.tracers_raw` (which is stale mid-
    # palindrome). A 0-argument `midpoint!()` would silently mutate the wrong
    # array, so require the buffer-aware 2-arg form and fail loudly otherwise.
    if midpoint! !== nothing
        applicable(midpoint!, active_rm, active_m) || throw(ArgumentError(
            "strang_split_cs_mt_pingpong! requires a buffer-aware `midpoint!` " *
            "accepting (active_rm, active_m); a 0-argument midpoint! would mutate " *
            "the wrong (stale) buffer mid-palindrome."))
        midpoint!(active_rm, active_m)
    end

    SectionTimer.@section :cs_sweep_z for _ in 1:n_z
        _sweep_z_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_cm, mesh, scheme, workspace;
                                     flux_scale = fs_z)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(active_rm, mesh; dir = 2)
    SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(active_m,  mesh; dir = 2)
    SectionTimer.@section :cs_sweep_y for _ in 1:n_y
        _sweep_y_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_bm, mesh, scheme; flux_scale = fs_y,
                                     seam_flux = workspace.seam_flux)
        SectionTimer.@section :cs_halo_rm_y fill_panel_halos!(spare_rm, mesh; dir = 2)
        SectionTimer.@section :cs_halo_m_y  fill_panel_halos!(spare_m,  mesh; dir = 2)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(active_rm, mesh; dir = 1)
    SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(active_m,  mesh; dir = 1)
    SectionTimer.@section :cs_sweep_x for _ in 1:n_x
        _sweep_x_panels_mt_pingpong!(spare_rm, spare_m, active_rm, active_m,
                                     panels_am, mesh, scheme; flux_scale = fs_x,
                                     seam_flux = workspace.seam_flux)
        SectionTimer.@section :cs_halo_rm_x fill_panel_halos!(spare_rm, mesh; dir = 1)
        SectionTimer.@section :cs_halo_m_x  fill_panel_halos!(spare_m,  mesh; dir = 1)
        active_rm, spare_rm = spare_rm, active_rm
        active_m, spare_m = spare_m, active_m
    end

    return active_rm, active_m
end

"""Packed-tracer Z-sweep of all six panels into the ping-pong buffers."""
function _sweep_z_panels_mt_pingpong!(panels_rm_4d_out::NTuple{6},
                                      panels_m_out::NTuple{6},
                                      panels_rm_4d::NTuple{6},
                                      panels_m::NTuple{6},
                                      panels_cm::NTuple{6},
                                      mesh::CubedSphereMesh,
                                      scheme::AbstractAdvectionScheme,
                                      ::CSAdvectionWorkspace;
                                      flux_scale = one(eltype(panels_m[1])))
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    for p in 1:6
        _sweep_z_panel_mt_pingpong!(panels_rm_4d_out[p], panels_m_out[p],
                                   panels_rm_4d[p], panels_m[p], panels_cm[p],
                                   scheme, Nc, Hp, Nz, Nt; flux_scale)
    end
    return panels_rm_4d_out, panels_m_out
end

"""Single-tracer Z-sweep of all six panels (the split-sweep `strang_split_cs!` path)."""
function _sweep_z_panels!(panels_rm::NTuple{6}, panels_m::NTuple{6}, panels_cm::NTuple{6},
                          mesh::CubedSphereMesh, scheme::AbstractAdvectionScheme,
                          workspace::CSAdvectionWorkspace;
                          flux_scale = one(eltype(panels_m[1])))
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_rm[1], 3)
    for p in 1:6
        _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p], scheme,
                        workspace.rm_A, workspace.m_A, Nc, Hp, Nz; flux_scale)
    end
    return nothing
end

"""
    _sweep_z!(rm_panels, m_panels, cm_panels, mesh, ws)

Upwind Z-sweep of all six panels (`_sweep_z_panel!` with `UpwindScheme()`), used
by the Lin-Rood adjoint tape and `strang_split_linrood_ppm!`. FV3 itself remaps
tracers vertically with PPM; see `LinRoodPPMScheme`'s `vertical` option.
"""
function _sweep_z!(rm_panels, m_panels, cm_panels,
                   mesh::CubedSphereMesh, ws::CSAdvectionWorkspace)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(rm_panels[1], 3)
    for p in 1:6
        _sweep_z_panel!(rm_panels[p], m_panels[p], cm_panels[p],
                         UpwindScheme(), ws.rm_A, ws.m_A, Nc, Hp, Nz)
    end
    return nothing
end

export strang_split_cs!, strang_split_cs_mt!, CSAdvectionWorkspace
