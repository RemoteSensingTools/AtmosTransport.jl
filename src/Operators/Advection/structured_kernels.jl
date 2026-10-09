# ---------------------------------------------------------------------------
# Universal structured-grid kernel shells
#
# Design principle (Oceananigans-inspired):
# ─────────────────────────────────────────
# Each kernel is a THIN indexing shell (5 lines of logic) that:
#
#   1. Reads the face mass flux from the staggered-grid flux array
#   2. Calls @inline face-flux functions that dispatch on scheme type
#   3. Updates tracer mass: rm_new = rm + F_in - F_out
#   4. Updates air mass:    m_new  = m  + F_in - F_out
#
# Julia specializes the full kernel at compile time for each concrete
# scheme type.  On GPU, this produces a single monomorphic code path
# with zero branches — the scheme type is compiled away entirely.
#
# Mass conservation identity
# ──────────────────────────
# The mass update in each kernel is a telescoping sum:
#
#   For x:  m_new[i,j,k] = m[i,j,k] + am[i,j,k] - am[i+1,j,k]
#
# Summing over all i (with periodic BCs: am[Nx+1] = am[1]):
#
#   Σᵢ m_new[i,j,k] = Σᵢ m[i,j,k] + am[1] - am[Nx+1] = Σᵢ m[i,j,k]
#
# This holds identically (to machine precision) for each j,k slice.
# The same telescoping applies to the tracer mass rm, guaranteeing
# exact mass conservation regardless of the reconstruction scheme.
#
# For y and z (closed BCs): bm[i,1,k] = bm[i,Ny+1,k] = 0 and
# cm[i,j,1] = cm[i,j,Nz+1] = 0, so the boundary terms vanish.
#
# Double buffering
# ────────────────
# Each kernel writes to separate output arrays (rm_new, m_new) while
# reading from the input arrays (rm, m) marked with @Const.  The caller
# (sweep_x!/y!/z! in sweeps.jl) copies the output back to the
# input arrays after synchronization.  This double-buffer pattern is
# ESSENTIAL for correctness: in-place updates would violate the stencil
# read-before-write contract and break mass conservation by ~10% per step.
#
# Workgroup size
# ──────────────
# The sweep functions launch kernels with workgroup size 256 and
# ndrange = size(m) = (Nx, Ny, Nz).  KernelAbstractions maps this
# to a 3D grid of work items.  On GPU, threads within a warp access
# consecutive i-indices, giving coalesced memory access for the
# innermost array dimension (column-major layout).
#
# References
# ----------
# - Ramadhan et al. (2020), Oceananigans.jl — thin kernel shells
# - KernelAbstractions.jl documentation (kernel launch, @index, @Const)
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const

"""
    _xsweep_kernel!(rm_new, rm, m_new, m, am, scheme, Nx)

X-direction advection kernel for structured grids with periodic boundaries.

One thread per grid cell `(i, j, k)`.  Computes the tracer flux through
the left face (`face_i = i`) and right face (`face_i = i+1`), then updates
tracer and air mass via the divergence form:

```math
r_{m,\\text{new}}[i,j,k] = r_m[i,j,k] + F_q^L - F_q^R
```
```math
m_{\\text{new}}[i,j,k] = m[i,j,k] + a_m[i,j,k] - a_m[i+1,j,k]
```

The face flux function `_xface_tracer_flux` dispatches on `scheme` type.
"""
@kernel function _xsweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                  @Const(am), scheme, Nx, flux_scale)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # am[i, j, k] = mass flux through the WEST face of cell (i, j, k).
        # Positive am = eastward mass transport [kg per substep].
        # flux_scale = 1/n_sub for subcycled passes.
        am_l = flux_scale * am[i, j, k]         # flux into cell from west face
        am_r = flux_scale * am[i + 1, j, k]     # flux out of cell through east face
        # Tracer flux: F_q = χ_donor × |am| (upwind), or higher-order (slopes/PPM).
        flux_L = _xface_tracer_flux(Int32(i), j, k, rm, m, am_l, scheme, Nx)
        flux_R = _xface_tracer_flux(Int32(i) + Int32(1), j, k, rm, m, am_r, scheme, Nx)
        # Divergence form: new = old + (left flux in) − (right flux out).
        # Telescoping: Σ_i (flux_L − flux_R) = am[1] − am[Nx+1] = 0 (periodic).
        rm_new[i, j, k] = rm[i, j, k] + flux_L - flux_R
        m_new[i, j, k]  = m[i, j, k]  + am_l - am_r
    end
end

"""
    _ysweep_kernel!(rm_new, rm, m_new, m, bm, scheme, Ny)

Y-direction advection kernel with closed boundaries at the poles.

Computes fluxes through south face (`face_j = j`) and north face
(`face_j = j+1`).  Boundary faces (`j=1`, `j=Ny+1`) return zero flux
via the `_yface_tracer_flux` convention.

```math
r_{m,\\text{new}}[i,j,k] = r_m[i,j,k] + F_q^S - F_q^N
```
```math
m_{\\text{new}}[i,j,k] = m[i,j,k] + b_m[i,j,k] - b_m[i,j+1,k]
```
"""
@kernel function _ysweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                  @Const(bm), scheme, Ny, flux_scale)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        bm_s = flux_scale * bm[i, j, k]
        bm_n = flux_scale * bm[i, j + 1, k]
        flux_S = _yface_tracer_flux(i, Int32(j), k, rm, m, bm_s, scheme, Ny)
        flux_N = _yface_tracer_flux(i, Int32(j) + Int32(1), k, rm, m, bm_n, scheme, Ny)
        rm_new[i, j, k] = rm[i, j, k] + flux_S - flux_N
        m_new[i, j, k]  = m[i, j, k]  + bm_s - bm_n
    end
end

"""
    _zsweep_kernel!(rm_new, rm, m_new, m, cm, scheme, Nz)

Z-direction advection kernel with closed boundaries at TOA and surface.

Computes fluxes through top face (`face_k = k`) and bottom face
(`face_k = k+1`).  Boundary faces (`k=1`, `k=Nz+1`) return zero flux.

```math
r_{m,\\text{new}}[i,j,k] = r_m[i,j,k] + F_q^T - F_q^B
```
```math
m_{\\text{new}}[i,j,k] = m[i,j,k] + c_m[i,j,k] - c_m[i,j,k+1]
```
"""
@kernel function _zsweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                  @Const(cm), scheme, Nz, flux_scale)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        # cm[i, j, k] = vertical mass flux through the TOP face of cell (i, j, k).
        # Positive cm = downward (toward surface). cm[k=1] = 0 (TOA boundary),
        # cm[k=Nz+1] = 0 (surface boundary).
        cm_t = flux_scale * cm[i, j, k]         # flux entering from above (top face)
        cm_b = flux_scale * cm[i, j, k + 1]     # flux leaving below (bottom face)
        flux_T = _zface_tracer_flux(i, j, Int32(k), rm, m, cm_t, scheme, Nz)
        flux_B = _zface_tracer_flux(i, j, Int32(k) + Int32(1), rm, m, cm_b, scheme, Nz)
        # Divergence form: new = old + (top flux in) − (bottom flux out).
        rm_new[i, j, k] = rm[i, j, k] + flux_T - flux_B
        m_new[i, j, k]  = m[i, j, k]  + cm_t - cm_b
    end
end
