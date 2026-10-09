# ---------------------------------------------------------------------------
# Lin-Rood / COSMIC Cross-Term Advection for Cubed-Sphere Grids
#
# Implements FV3's fv_tp_2d algorithm (Putman & Lin 2007, Lin & Rood 1996):
# Both X-then-Y and Y-then-X orderings are computed from the ORIGINAL field,
# and fluxes are averaged. This eliminates the directional splitting error
# that causes wave artifacts at CS panel boundaries.
#
# Reference: tp_core.F90 in FV3/fvdycore, fv_tracer2d.F90 in GCHP
# ---------------------------------------------------------------------------

# KA imports available from parent module
using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# ---------------------------------------------------------------------------
# Flux-panel halo normalization (runtime entry only)
#
# The LinRood flux kernels index `am`/`bm` with the *interior* face index
# (`am[i]`, `am[iif]`, no Hp offset) — the unpadded convention the kernels, the
# adjoint tape (LinRoodTape), and their kernel/footprint tests are all written
# for. The production runtime driver, however, Hp-pads the flux panels
# (`transport_binary/cubed_sphere_driver.jl`) exactly like the cell panels,
# so `am` arrives as (Nc+1+2Hp, Nc+2Hp, ·) and `bm` as (Nc+2Hp, Nc+1+2Hp, ·).
# Strip the halo to the interior faces at the runtime operator boundary
# (`_cs_transport_step!(::CSLinRoodStyle)`) so the kernels read the correct cell.
# No flux halo is ever read — the cross-term only needs interior faces;
# neighbour-panel information is carried by the padded `rm`/`m` cell halos.
# Without this the fluxes are shifted by Hp cells, which still conserves *total*
# mass (telescoping) and leaves uniform fields uniform — so unit + footprint
# tests pass — but corrupts real fields, driving air mass negative first at
# panel edges (C180 GEOS: panel-1 j=Nc → NaN by t=3h). The size guard makes this
# a no-op for the already-unpadded callers (footprint/adjoint tape, kernel
# tests), which build their own consistently-shifted fluxes and must stay
# untouched.
@inline _cs_flux_x_interior(am, Nc::Int, Hp::Int) =
    (Hp > 0 && size(am, 1) == Nc + 1 + 2Hp) ?
        view(am, (Hp + 1):(Hp + Nc + 1), (Hp + 1):(Hp + Nc), :) : am
@inline _cs_flux_y_interior(bm, Nc::Int, Hp::Int) =
    (Hp > 0 && size(bm, 2) == Nc + 1 + 2Hp) ?
        view(bm, (Hp + 1):(Hp + Nc), (Hp + 1):(Hp + Nc + 1), :) : bm

# ---------------------------------------------------------------------------
# Divergence Damping (del-2 diffusion on mixing ratio)
#
# FV3-style horizontal diffusion to suppress grid imprinting at panel boundaries.
# Conservative flux-form Laplacian: face fluxes telescope exactly -> mass conserving.
# Applied once before the first Strang sweep (not per-subcycle).
#
# Reference: tp_core.F90:deln_flux (simplified to del-2 for cubed-sphere)
# ---------------------------------------------------------------------------

@kernel function _divergence_damping_cs_kernel!(rm_new, @Const(rm), @Const(m), damp, Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)

        m_ij = m[ii, jj, k]
        c_ij = _safe_mixing_ratio(rm[ii, jj, k], m_ij)

        m_xm = m[ii - 1, jj, k]; c_xm = _safe_mixing_ratio(rm[ii - 1, jj, k], m_xm)
        m_xp = m[ii + 1, jj, k]; c_xp = _safe_mixing_ratio(rm[ii + 1, jj, k], m_xp)
        m_ym = m[ii, jj - 1, k]; c_ym = _safe_mixing_ratio(rm[ii, jj - 1, k], m_ym)
        m_yp = m[ii, jj + 1, k]; c_yp = _safe_mixing_ratio(rm[ii, jj + 1, k], m_yp)

        m_face_xm = FT(0.5) * (m_xm + m_ij)
        m_face_xp = FT(0.5) * (m_xp + m_ij)
        m_face_ym = FT(0.5) * (m_ym + m_ij)
        m_face_yp = FT(0.5) * (m_yp + m_ij)

        diff = m_face_xm * (c_xm - c_ij) + m_face_xp * (c_xp - c_ij) +
               m_face_ym * (c_ym - c_ij) + m_face_yp * (c_yp - c_ij)

        rm_new[ii, jj, k] = rm[ii, jj, k] + FT(damp) * diff
    end
end

"""
    apply_divergence_damping_cs!(rm_panels, m_panels, mesh, ws, damp_coeff)

Conservative del-2 divergence damping on tracer panels. Mass-conserving
flux-form Laplacian diffusion on mixing ratio (c = rm/m).
Typical `damp_coeff` values: 0.02-0.05 for mild panel-boundary noise.
"""
function apply_divergence_damping_cs!(rm_panels, m_panels,
                                      mesh::CubedSphereMesh, ws, damp_coeff)
    FT = eltype(rm_panels[1])
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(rm_panels[1], 3)

    fill_panel_halos!(rm_panels, mesh)
    fill_panel_halos!(m_panels, mesh)

    for p in 1:6
        backend = get_backend(rm_panels[p])
        k! = _divergence_damping_cs_kernel!(backend, 256)
        k!(ws.rm_A, rm_panels[p], m_panels[p], FT(damp_coeff), Hp;
           ndrange=(Nc, Nc, Nz))
        synchronize(backend)
        _copy_interior!(rm_panels[p], ws.rm_A, Nc, Hp, Nz)
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

struct LinRoodWorkspace{FT, A3h <: AbstractArray{FT,3},
                         A3x <: AbstractArray{FT,3},
                         A3y <: AbstractArray{FT,3}}
    q_buf  :: NTuple{6, A3h}   # pre-advected mixing ratio (haloed)
    fx_in  :: NTuple{6, A3x}   # inner X face values (Nc+1 × Nc × Nz)
    fx_out :: NTuple{6, A3x}   # outer X face values
    fy_in  :: NTuple{6, A3y}   # inner Y face values (Nc × Nc+1 × Nz)
    fy_out :: NTuple{6, A3y}   # outer Y face values (per-panel for multi-GPU)
    # Per-panel output buffers for parallel Phase 3
    q_out  :: NTuple{6, A3h}   # q output buffer per panel (haloed)
    dp_out :: NTuple{6, A3h}   # dp output buffer per panel (haloed)
end

function LinRoodWorkspace(mesh::CubedSphereMesh; FT::Type{<:AbstractFloat}=Float64,
                           Nz::Int,
                           array_type::Type{<:AbstractArray} = Array)
    Nc = mesh.Nc
    Hp = mesh.Hp
    N  = Nc + 2Hp

    q_buf  = ntuple(_ -> array_type(zeros(FT, N, N, Nz)), 6)
    fx_in  = ntuple(_ -> array_type(zeros(FT, Nc + 1, Nc, Nz)), 6)
    fx_out = ntuple(_ -> array_type(zeros(FT, Nc + 1, Nc, Nz)), 6)
    fy_in  = ntuple(_ -> array_type(zeros(FT, Nc, Nc + 1, Nz)), 6)
    fy_out = ntuple(_ -> array_type(zeros(FT, Nc, Nc + 1, Nz)), 6)
    q_out  = ntuple(_ -> array_type(zeros(FT, N, N, Nz)), 6)
    dp_out = ntuple(_ -> array_type(zeros(FT, N, N, Nz)), 6)

    return LinRoodWorkspace(q_buf, fx_in, fx_out, fy_in, fy_out, q_out, dp_out)
end

function LinRoodWorkspace(mesh::CubedSphereMesh,
                          prototype::AbstractArray{FT, 3}) where {FT <: AbstractFloat}
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(prototype, 3)
    N  = Nc + 2Hp

    q_buf  = ntuple(_ -> similar(prototype, FT, N, N, Nz), 6)
    fx_in  = ntuple(_ -> similar(prototype, FT, Nc + 1, Nc, Nz), 6)
    fx_out = ntuple(_ -> similar(prototype, FT, Nc + 1, Nc, Nz), 6)
    fy_in  = ntuple(_ -> similar(prototype, FT, Nc, Nc + 1, Nz), 6)
    fy_out = ntuple(_ -> similar(prototype, FT, Nc, Nc + 1, Nz), 6)
    q_out  = ntuple(_ -> similar(prototype, FT, N, N, Nz), 6)
    dp_out = ntuple(_ -> similar(prototype, FT, N, N, Nz), 6)

    return LinRoodWorkspace(q_buf, fx_in, fx_out, fy_in, fy_out, q_out, dp_out)
end

function Adapt.adapt_structure(to, ws::LinRoodWorkspace{FT}) where FT
    q_buf = Adapt.adapt(to, ws.q_buf)
    fx_in = Adapt.adapt(to, ws.fx_in)
    fx_out = Adapt.adapt(to, ws.fx_out)
    fy_in = Adapt.adapt(to, ws.fy_in)
    fy_out = Adapt.adapt(to, ws.fy_out)
    q_out = Adapt.adapt(to, ws.q_out)
    dp_out = Adapt.adapt(to, ws.dp_out)
    return LinRoodWorkspace(q_buf, fx_in, fx_out, fy_in, fy_out, q_out, dp_out)
end

struct CSLinRoodAdvectionWorkspace{CSW, LRW}
    cs      :: CSW
    linrood :: LRW
end

function CSLinRoodAdvectionWorkspace(mesh::CubedSphereMesh, Nz::Int;
                                     FT::Type{<:AbstractFloat} = Float64,
                                     array_type::Type{<:AbstractArray} = Array,
                                     n_tracers::Integer = 0,
                                     column_scratch::Bool = false)
    cs = CSAdvectionWorkspace(mesh, Nz; FT, array_type, n_tracers, seam_transport=false,
                              column_scratch, column_scratch_tracers = 1)   # tracers run one by one
    lr = LinRoodWorkspace(mesh; FT = FT, Nz = Nz, array_type = array_type)
    return CSLinRoodAdvectionWorkspace{typeof(cs), typeof(lr)}(cs, lr)
end

function CSLinRoodAdvectionWorkspace(mesh::CubedSphereMesh,
                                     prototype::AbstractArray{FT, 3};
                                     n_tracers::Integer = 0,
                                     column_scratch::Bool = false) where {FT <: AbstractFloat}
    cs = CSAdvectionWorkspace(mesh, prototype; n_tracers, seam_transport=false, column_scratch,
                              column_scratch_tracers = 1)                    # tracers run one by one
    lr = LinRoodWorkspace(mesh, prototype)
    return CSLinRoodAdvectionWorkspace{typeof(cs), typeof(lr)}(cs, lr)
end

function Adapt.adapt_structure(to, workspace::CSLinRoodAdvectionWorkspace)
    cs = Adapt.adapt(to, workspace.cs)
    linrood = Adapt.adapt(to, workspace.linrood)
    return CSLinRoodAdvectionWorkspace{typeof(cs), typeof(linrood)}(cs, linrood)
end

# ---------------------------------------------------------------------------
# Shared PPM helpers (inline, GPU-safe)
# ---------------------------------------------------------------------------

"""Apply ORD=7 discontinuous boundary treatment at panel edges (compile-time eliminated for ORD≠7)."""
@inline function _apply_ord7_boundary(q_L_m, q_R_m, q_L_0, q_R_0,
                                       c_m1, c_m2, c_0, c_p1,
                                       face_idx, Nc, ::Val{ORD}) where ORD
    if ORD == 7
        if face_idx == 1 || face_idx == Nc + 1
            q_bdy = _ppm_face_edge_value_ord7_discontinuous(c_m1, c_m2, c_0, c_p1)
            q_R_m = q_bdy
            q_L_0 = q_bdy
        end
    end
    return q_L_m, q_R_m, q_L_0, q_R_0
end

"""Apply Colella-Woodward monotonicity: flatten reconstruction at local extrema."""
@inline function _apply_monotonicity(q_L, q_R, c)
    FT = typeof(c)
    if (q_R - c) * (c - q_L) <= zero(FT)
        return c, c
    end
    return q_L, q_R
end

"""
    _courant_fraction(F, m_donor) -> (α, ∂α/∂m_donor)

Fraction of the donor cell that crosses a face in one sweep, `α = F / m_donor`,
clamped to the donor cell (`|α| ≤ 1`) as in FV3's `xppm`/`yppm`; zero for an
empty donor (mass below `100 eps`). The derivative `∂α/∂m_donor = −F / m_donor²`
(used by the adjoint) applies for `|F| < m_donor`; where `α` is clamped to ±1 it
no longer depends on the donor mass and the derivative is zero.
"""
@inline function _courant_fraction(F::FT, m_donor::FT) where FT
    m_donor > 100 * eps(FT) || return (zero(FT), zero(FT))
    alpha = F / m_donor
    dalpha_dm = abs(alpha) < one(FT) ? -F / (m_donor * m_donor) : zero(FT)
    return (clamp(alpha, -one(FT), one(FT)), dalpha_dm)     # clamp keeps a NaN a NaN
end

"""
    _ppm_face_value(flux, m_lo, m_hi, c_lo, c_hi, q_L_lo, q_R_lo, q_L_hi, q_R_hi)

Upwind PPM face value: the mean mixing ratio of the donor-cell part that crosses
the face, the full parabolic integral of FV3's `xppm`/`yppm`,

    flux ≥ 0:  face = c + (1 − α)(br − α·b0)        (donor = lo cell)
    flux < 0:  face = c + (1 + α)(bl + α·b0)        (donor = hi cell)

with `bl = q_L − c`, `br = q_R − c`, `b0 = bl + br` and the clamped Courant
fraction `α` of [`_courant_fraction`](@ref). Generic in the type of the mixing
ratios, so the Lin-Rood adjoint evaluates the same function on its dual numbers.
"""
@inline function _ppm_face_value(flux::FT, m_lo::FT, m_hi::FT, c_lo, c_hi,
                                 q_L_lo, q_R_lo, q_L_hi, q_R_hi) where FT
    if flux >= zero(FT)
        alpha = first(_courant_fraction(flux, m_lo))
        bl = q_L_lo - c_lo
        br = q_R_lo - c_lo
        b0 = bl + br
        return c_lo + (one(FT) - alpha) * (br - alpha * b0)
    else
        alpha = first(_courant_fraction(flux, m_hi))
        bl = q_L_hi - c_hi
        br = q_R_hi - c_hi
        b0 = bl + br
        return c_hi + (one(FT) + alpha) * (bl + alpha * b0)
    end
end

# ---------------------------------------------------------------------------
# PPM Face Value Kernels (mixing ratio at cell faces)
#
# Two variants per direction:
#   _ppm_{x,y}_face_kernel!     — reads rm/m (original field)
#   _ppm_{x,y}_face_from_q_kernel! — reads pre-computed mixing ratio q
# ---------------------------------------------------------------------------

@kernel function _ppm_y_face_kernel!(
    fy_face, @Const(rm), @Const(m), @Const(bm), Hp, Nc, ::Val{ORD}
) where ORD
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        c_m3 = _safe_mixing_ratio(rm[ii, jj_b - 2, k], m[ii, jj_b - 2, k])
        c_m2 = _safe_mixing_ratio(rm[ii, jj_b - 1, k], m[ii, jj_b - 1, k])
        c_m1 = _safe_mixing_ratio(rm[ii, jj_b,     k], m[ii, jj_b,     k])
        c_0  = _safe_mixing_ratio(rm[ii, jj_a,     k], m[ii, jj_a,     k])
        c_p1 = _safe_mixing_ratio(rm[ii, jj_a + 1, k], m[ii, jj_a + 1, k])
        c_p2 = _safe_mixing_ratio(rm[ii, jj_a + 2, k], m[ii, jj_a + 2, k])

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, jf, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fy_face[i, jf, k] = _ppm_face_value(
            bm[i, jf, k], m[ii, jj_b, k], m[ii, jj_a, k],
            c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

@kernel function _ppm_x_face_kernel!(
    fx_face, @Const(rm), @Const(m), @Const(am), Hp, Nc, ::Val{ORD}
) where ORD
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        c_m3 = _safe_mixing_ratio(rm[ii_l - 2, jj, k], m[ii_l - 2, jj, k])
        c_m2 = _safe_mixing_ratio(rm[ii_l - 1, jj, k], m[ii_l - 1, jj, k])
        c_m1 = _safe_mixing_ratio(rm[ii_l,     jj, k], m[ii_l,     jj, k])
        c_0  = _safe_mixing_ratio(rm[ii_r,     jj, k], m[ii_r,     jj, k])
        c_p1 = _safe_mixing_ratio(rm[ii_r + 1, jj, k], m[ii_r + 1, jj, k])
        c_p2 = _safe_mixing_ratio(rm[ii_r + 2, jj, k], m[ii_r + 2, jj, k])

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, iif, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fx_face[iif, j, k] = _ppm_face_value(
            am[iif, j, k], m[ii_l, jj, k], m[ii_r, jj, k],
            c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

@kernel function _ppm_x_face_from_q_kernel!(
    fx_face, @Const(q), @Const(am), @Const(m), Hp, Nc, ::Val{ORD}
) where ORD
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        c_m3 = q[ii_l - 2, jj, k]; c_m2 = q[ii_l - 1, jj, k]
        c_m1 = q[ii_l,     jj, k]; c_0  = q[ii_r,     jj, k]
        c_p1 = q[ii_r + 1, jj, k]; c_p2 = q[ii_r + 2, jj, k]

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, iif, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fx_face[iif, j, k] = _ppm_face_value(
            am[iif, j, k], m[ii_l, jj, k], m[ii_r, jj, k],
            c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

@kernel function _ppm_y_face_from_q_kernel!(
    fy_face, @Const(q), @Const(bm), @Const(m), Hp, Nc, ::Val{ORD}
) where ORD
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        c_m3 = q[ii, jj_b - 2, k]; c_m2 = q[ii, jj_b - 1, k]
        c_m1 = q[ii, jj_b,     k]; c_0  = q[ii, jj_a,     k]
        c_p1 = q[ii, jj_a + 1, k]; c_p2 = q[ii, jj_a + 2, k]

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, jf, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fy_face[i, jf, k] = _ppm_face_value(
            bm[i, jf, k], m[ii, jj_b, k], m[ii, jj_a, k],
            c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

# ---------------------------------------------------------------------------
# q_buf initialization kernel (mixing ratio from rm/m, full haloed domain)
# ---------------------------------------------------------------------------

@kernel function _init_q_buf_kernel!(q_buf, @Const(rm), @Const(m))
    i, j, k = @index(Global, NTuple)
    @inbounds q_buf[i, j, k] = _safe_mixing_ratio(rm[i, j, k], m[i, j, k])
end

# ---------------------------------------------------------------------------
# Pre-advection kernels (advective-form transport for cross-term)
# ---------------------------------------------------------------------------

@kernel function _pre_advect_y_kernel!(
    q_i, @Const(rm), @Const(m), @Const(bm), @Const(fy_face), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(rm)
        bm_s = bm[i, j, k];  bm_n = bm[i, j + 1, k]
        m1 = m[ii, jj, k]
        mass_div = bm_s - bm_n
        max_outflow = FT(0.9) * m1
        scale = mass_div < -max_outflow && m1 > zero(FT) ? max_outflow / (-mass_div) : one(FT)
        rm_new = rm[ii, jj, k] + scale * (bm_s * fy_face[i, j, k] - bm_n * fy_face[i, j + 1, k])
        m_new  = m1 + scale * mass_div
        q_i[ii, jj, k] = _safe_mixing_ratio(rm_new, m_new)
    end
end

@kernel function _pre_advect_x_kernel!(
    q_j, @Const(rm), @Const(m), @Const(am), @Const(fx_face), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(rm)
        am_w = am[i, j, k];  am_e = am[i + 1, j, k]
        m1 = m[ii, jj, k]
        mass_div = am_w - am_e
        max_outflow = FT(0.9) * m1
        scale = mass_div < -max_outflow && m1 > zero(FT) ? max_outflow / (-mass_div) : one(FT)
        rm_new = rm[ii, jj, k] + scale * (am_w * fx_face[i, j, k] - am_e * fx_face[i + 1, j, k])
        m_new  = m1 + scale * mass_div
        q_j[ii, jj, k] = _safe_mixing_ratio(rm_new, m_new)
    end
end

# ---------------------------------------------------------------------------
# Combined update kernel (applies averaged x+y fluxes simultaneously)
# ---------------------------------------------------------------------------

@kernel function _linrood_update_kernel!(
    rm_new, m_new, @Const(rm), @Const(m), @Const(am), @Const(bm),
    @Const(fx_in), @Const(fx_out), @Const(fy_in), @Const(fy_out), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(rm)
        half = FT(0.5)

        avg_fx_w = half * (fx_out[i,   j, k] + fx_in[i,   j, k])
        avg_fx_e = half * (fx_out[i+1, j, k] + fx_in[i+1, j, k])
        avg_fy_s = half * (fy_out[i, j,   k] + fy_in[i, j,   k])
        avg_fy_n = half * (fy_out[i, j+1, k] + fy_in[i, j+1, k])

        am_w = am[i, j, k];  am_e = am[i+1, j, k]
        bm_s = bm[i, j, k];  bm_n = bm[i, j+1, k]

        rm_new[ii, jj, k] = rm[ii, jj, k] +
            (am_w * avg_fx_w - am_e * avg_fx_e) +
            (bm_s * avg_fy_s - bm_n * avg_fy_n)
        m_new[ii, jj, k] = m[ii, jj, k] +
            (am_w - am_e) + (bm_s - bm_n)
    end
end

# ---------------------------------------------------------------------------
# Q-space kernels for GCHP-aligned transport
# ---------------------------------------------------------------------------

"""Convert tracer panels from rm (mass) to q (mixing ratio) in-place: data /= m."""
function rm_to_q_panels!(panels::NTuple{6}, m_panels::NTuple{6}, mesh)
    Nc = mesh.Nc; Hp = mesh.Hp; Nz = size(panels[1], 3)
    backend = get_backend(panels[1])
    k! = _rm_to_q_kernel!(backend, 256)
    for p in 1:6
        k!(panels[p], m_panels[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)
end

"""Convert tracer panels from q (mixing ratio) to rm (mass) in-place: data *= m."""
function q_to_rm_panels!(panels::NTuple{6}, m_panels::NTuple{6}, mesh)
    Nc = mesh.Nc; Hp = mesh.Hp; Nz = size(panels[1], 3)
    backend = get_backend(panels[1])
    k! = _q_to_rm_kernel!(backend, 256)
    for p in 1:6
        k!(panels[p], m_panels[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)
end

@kernel function _rm_to_q_kernel!(data, @Const(m), Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i; jj = Hp + j
        FT = eltype(data)
        mk = m[ii, jj, k]
        data[ii, jj, k] = mk > FT(100) * eps(FT) ? data[ii, jj, k] / mk : zero(FT)
    end
end

@kernel function _q_to_rm_kernel!(data, @Const(m), Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i; jj = Hp + j
        data[ii, jj, k] *= m[ii, jj, k]
    end
end

"""Compute dp (pressure thickness, Pa) from air mass m: dp = m × g / area."""
function compute_dp_from_m_panels!(dp_panels::NTuple{6}, m_panels::NTuple{6},
                                     area_panels, gravity, mesh)
    Nc = mesh.Nc; Hp = mesh.Hp; Nz = size(dp_panels[1], 3)
    backend = get_backend(dp_panels[1])
    k! = _dp_from_m_kernel!(backend, 256)
    for p in 1:6
        k!(dp_panels[p], m_panels[p], area_panels[p], gravity, Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)
end

"""Set air mass m from dp (pressure thickness): m = dp × area / g."""
function set_m_from_dp_panels!(m_panels::NTuple{6}, dp_panels::NTuple{6},
                                 area_panels, gravity, mesh)
    Nc = mesh.Nc; Hp = mesh.Hp; Nz = size(m_panels[1], 3)
    backend = get_backend(m_panels[1])
    k! = _m_from_dp_kernel!(backend, 256)
    for p in 1:6
        k!(m_panels[p], dp_panels[p], area_panels[p], gravity, Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)
end

@kernel function _dp_from_m_kernel!(dp, @Const(m), @Const(area), g, Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i; jj = Hp + j
        dp[ii, jj, k] = m[ii, jj, k] * g / area[i, j]
    end
end

@kernel function _m_from_dp_kernel!(m, @Const(dp), @Const(area), g, Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i; jj = Hp + j
        m[ii, jj, k] = dp[ii, jj, k] * area[i, j] / g
    end
end

# --- Q-space Lin-Rood update kernel (GCHP tracer_2d:543-549) ---

@kernel function _linrood_update_q_kernel!(
    q_new, m_new, @Const(q), @Const(m), @Const(am), @Const(bm),
    @Const(fx_in), @Const(fx_out), @Const(fy_in), @Const(fy_out), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(q)
        half = FT(0.5)

        avg_fx_w = half * (fx_out[i,   j, k] + fx_in[i,   j, k])
        avg_fx_e = half * (fx_out[i+1, j, k] + fx_in[i+1, j, k])
        avg_fy_s = half * (fy_out[i, j,   k] + fy_in[i, j,   k])
        avg_fy_n = half * (fy_out[i, j+1, k] + fy_in[i, j+1, k])

        am_w = am[i, j, k];  am_e = am[i+1, j, k]
        bm_s = bm[i, j, k];  bm_n = bm[i, j+1, k]

        m1 = m[ii, jj, k]

        # CFL guard: clamp mass flux divergence to prevent m_new < 0.
        # At thin TOA levels (k≥70), air mass m ≈ 0.01-0.5 kg but horizontal
        # fluxes can be 0.1-1 kg → CFL > 1 → m_new < 0 → q inverts → blowup.
        # This cell-local fallback is not conservative when activated. Callers
        # must supply CFL-safe fluxes; it is not a substitute for subcycling.
        mass_div = (am_w - am_e) + (bm_s - bm_n)
        max_outflow = FT(0.9) * m1
        scale = mass_div < -max_outflow && m1 > zero(FT) ? max_outflow / (-mass_div) : one(FT)

        m2 = m1 + scale * mass_div

        # GCHP: q_new = (q*m1 + flux_div) / m2
        rm_old = q[ii, jj, k] * m1
        tracer_div = (am_w * avg_fx_w - am_e * avg_fx_e) +
                     (bm_s * avg_fy_s - bm_n * avg_fy_n)
        rm_new = rm_old + scale * tracer_div

        q_new[ii, jj, k] = m2 > FT(100) * eps(FT) ? rm_new / m2 : zero(FT)
        m_new[ii, jj, k] = m2
    end
end

# --- Q-space pre-advect kernels ---

@kernel function _pre_advect_y_q_kernel!(
    q_i, @Const(q), @Const(m), @Const(bm), @Const(fy_face), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(q)
        bm_s = bm[i, j, k];  bm_n = bm[i, j + 1, k]
        m1 = m[ii, jj, k]

        # CFL guard (same as _linrood_update_q_kernel!)
        mass_div = bm_s - bm_n
        max_outflow = FT(0.9) * m1
        scale = mass_div < -max_outflow && m1 > zero(FT) ? max_outflow / (-mass_div) : one(FT)

        m_new = m1 + scale * mass_div
        tracer_div = bm_s * fy_face[i, j, k] - bm_n * fy_face[i, j + 1, k]
        rm_new = q[ii, jj, k] * m1 + scale * tracer_div
        q_i[ii, jj, k] = m_new > FT(100) * eps(FT) ? rm_new / m_new : zero(FT)
    end
end

@kernel function _pre_advect_x_q_kernel!(
    q_j, @Const(q), @Const(m), @Const(am), @Const(fx_face), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(q)
        am_w = am[i, j, k];  am_e = am[i + 1, j, k]
        m1 = m[ii, jj, k]

        # CFL guard (same as _linrood_update_q_kernel!)
        mass_div = am_w - am_e
        max_outflow = FT(0.9) * m1
        scale = mass_div < -max_outflow && m1 > zero(FT) ? max_outflow / (-mass_div) : one(FT)

        m_new = m1 + scale * mass_div
        tracer_div = am_w * fx_face[i, j, k] - am_e * fx_face[i + 1, j, k]
        rm_new = q[ii, jj, k] * m1 + scale * tracer_div
        q_j[ii, jj, k] = m_new > FT(100) * eps(FT) ? rm_new / m_new : zero(FT)
    end
end

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
# reverse-mode kernels defined in `linrood_adjoint_kernels.jl`, included
# alongside this file from `Advection.jl`. Re-export below for `Adjoints`.
export apply_linrood_update_adjoint!,
       apply_pre_advect_x_adjoint!,
       apply_pre_advect_y_adjoint!,
       apply_ppm_x_face_from_q_adjoint!,
       apply_ppm_y_face_from_q_adjoint!,
       apply_ppm_x_face_adjoint!,
       apply_ppm_y_face_adjoint!
