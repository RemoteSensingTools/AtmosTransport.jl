# ---------------------------------------------------------------------------
# Face tracer flux functions — the scheme dispatch point
#
# Each concrete advection scheme implements three @inline functions:
#
#   _xface_tracer_flux(face_i, j, k, rm, m, F, scheme, Nx)  (x: periodic)
#   _yface_tracer_flux(i, face_j, k, rm, m, F, scheme, Ny)  (y: closed)
#   _zface_tracer_flux(i, j, face_k, rm, m, F, scheme, Nz)  (z: closed)
#
# These are the ONLY functions that change between schemes.  The universal
# kernel shells (structured_kernels.jl) call them, and Julia specializes
# the full kernel at compile time for each concrete scheme type — the GPU
# sees a single monomorphic code path.
#
# Architecture (Oceananigans-inspired):
#
#   kernel shell (universal, 5 lines)
#     └── _xface_tracer_flux(face_i, j, k, rm, m, F, scheme, Nx)
#           ├── _wrap_periodic(idx, N)      [x-direction only]
#           ├── rm[...] / m[...]            [mixing ratios from stencil]
#           ├── _limited_slope(...)          [linear/quadratic schemes]
#           ├── _limited_moment(...)         [linear/quadratic schemes]
#           └── _slopes_face_flux(...)       [Courant-fraction formula]
#
# All functions are @inline, branchless (ifelse), and GPU-safe.
# All arithmetic uses `eltype(rm)` to stay type-stable for Float32/Float64.
#
# Boundary conditions
# -------------------
# - X (longitude): periodic — indices wrapped via mod1(idx, Nx)
# - Y (latitude):  closed (no-flux) at j=1 (south pole) and j=Ny+1 (north pole)
#                   Slopes zeroed at boundary cells (j=1, j=Ny)
# - Z (vertical):  closed (no-flux) at k=1 (TOA) and k=Nz+1 (surface)
#                   Slopes zeroed at boundary cells (k=1, k=Nz)
#                   eps-floored mass to prevent division by zero in thin layers
#
# References
# ----------
# - Russell & Lerner (1981), J. Appl. Meteor., 20:1483–1498.
# - van Leer (1977), J. Comp. Phys., 23:276–299.
# - TM5 source: advectx__slopes, advecty__slopes (deps/tm5/base/src/)
# ---------------------------------------------------------------------------

# =========================================================================
# Shared helpers
# =========================================================================

"""
    _wrap_periodic(idx, N) → Int

Wrap a 1-based index into ``[1, N]`` under periodic boundary conditions.

Uses Julia's `mod1` to handle arbitrary offsets, including negative indices
from wide stencils (e.g., `face_i - 2` for the 4-cell SlopesScheme stencil,
or `face_i - 3` for PPM's 6-cell stencil).

# Examples
```julia
_wrap_periodic( 0, 36)  # → 36  (one left of domain start)
_wrap_periodic(-1, 36)  # → 35  (two left of domain start)
_wrap_periodic(37, 36)  # → 1   (one right of domain end)
_wrap_periodic(38, 36)  # → 2   (two right of domain end)
```
"""
@inline _wrap_periodic(idx, N) = mod1(idx, N)

"""
    _mixing_ratio(rm, m, i, j, k) → FT

Compute mixing ratio ``c = r_m / m`` with an `eps(FT)` floor on mass
to prevent division by zero in cells with vanishingly small air mass
(e.g., the uppermost model level near the TOA).

Used by `UpwindScheme` face flux functions.  The `SlopesScheme` functions
use raw `rm/m` division (no floor) in the horizontal because air mass is
always well above eps for horizontal cells, and use an explicit `eps` floor
in the vertical direction where thin layers can occur.
"""
@inline function _mixing_ratio(rm, m, i, j, k)
    FT = eltype(rm)
    return rm[i, j, k] / max(m[i, j, k], eps(FT))
end

"""
    _mixing_ratio_2d(rm, m, c, k) → FT

Safe mixing ratio for 2D `(cell, level)` arrays used by face-indexed
(unstructured) grids.  Same eps floor as `_mixing_ratio`.
"""
@inline function _mixing_ratio_2d(rm, m, c, k)
    FT = eltype(rm)
    return rm[c, k] / max(m[c, k], eps(FT))
end


# =========================================================================
# UpwindScheme (constant reconstruction, order 0)
# =========================================================================
#
# The face tracer flux is simply the mass flux times the donor-cell
# mixing ratio:
#
#       F_q = F · c_donor
#
# where the donor cell is determined by the sign of the mass flux F:
#
#       c_donor = c_L  if F ≥ 0   (flow is left-to-right)
#                 c_R  if F < 0   (flow is right-to-left)
#
# This is first-order accurate (Godunov 1959) and strongly diffusive,
# but unconditionally monotone and the simplest reference scheme.

"""
    _xface_tracer_flux(face_i, j, k, rm, m, F, ::UpwindScheme, Nx) → FT

Tracer mass flux ``F_q`` [kg] through x-face `face_i` for first-order upwind.

# Stencil
```
     face_i
       │
  ┌────┼────┐
  │ iL │ iR │
  └────┼────┘
```
- `iL = mod1(face_i - 1, Nx)` — left cell (periodic)
- `iR = mod1(face_i, Nx)` — right cell (periodic)

# Boundary conditions
Periodic in x: indices wrapped via `mod1`.
"""
@inline function _xface_tracer_flux(face_i, j, k, rm, m, F, ::UpwindScheme, Nx)
    FT = eltype(rm)
    il = _wrap_periodic(face_i - Int32(1), Nx)
    ir = _wrap_periodic(face_i, Nx)
    c_l = _mixing_ratio(rm, m, il, j, k)
    c_r = _mixing_ratio(rm, m, ir, j, k)
    return ifelse(F >= zero(FT), F * c_l, F * c_r)
end

"""
    _yface_tracer_flux(i, face_j, k, rm, m, F, ::UpwindScheme, Ny) → FT

Tracer mass flux through y-face `face_j` for first-order upwind.

# Boundary conditions
Closed (no-flux) at `face_j = 1` (south pole) and `face_j = Ny + 1` (north pole).
Returns zero at boundaries.
"""
@inline function _yface_tracer_flux(i, face_j, k, rm, m, F, ::UpwindScheme, Ny)
    FT = eltype(rm)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    jl = max(face_j - Int32(1), Int32(1))
    jr = min(face_j, Ny)
    c_l = _mixing_ratio(rm, m, i, jl, k)
    c_r = _mixing_ratio(rm, m, i, jr, k)
    flux = ifelse(F >= zero(FT), F * c_l, F * c_r)
    return ifelse(at_boundary, zero(FT), flux)
end

"""
    _zface_tracer_flux(i, j, face_k, rm, m, F, ::UpwindScheme, Nz) → FT

Tracer mass flux through z-face `face_k` for first-order upwind.

# Boundary conditions
Closed (no-flux) at `face_k = 1` (TOA) and `face_k = Nz + 1` (surface).
Returns zero at boundaries.
"""
@inline function _zface_tracer_flux(i, j, face_k, rm, m, F, ::UpwindScheme, Nz)
    FT = eltype(rm)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    kl = max(face_k - Int32(1), Int32(1))
    kr = min(face_k, Nz)
    c_l = _mixing_ratio(rm, m, i, j, kl)
    c_r = _mixing_ratio(rm, m, i, j, kr)
    flux = ifelse(F >= zero(FT), F * c_l, F * c_r)
    return ifelse(at_boundary, zero(FT), flux)
end


# =========================================================================
# SlopesScheme (linear reconstruction, order 1)
# =========================================================================
#
# Derivation of the Courant-fraction flux formula
# ------------------------------------------------
#
# Consider a face between cell L (left) and cell R (right) with mass
# flux F > 0 (flow from L to R).  The Courant fraction is:
#
#     α = F / m_L                                              (1)
#
# Cell L has a piecewise-linear subcell profile of the mixing ratio:
#
#     c(x) = c_L + s_L · (x - x_L) / Δx                      (2)
#
# where s_L is the limited slope (see limiters.jl).  The tracer flux
# is the integral of c(x) · dm over the right-most fraction α of L:
#
#     F_q = m_L ∫₍₁₋α₎¹ c(ξ) dξ     where ξ = (x - x_L⁻) / Δx
#
#         = m_L [ c_L · α + s_L · (α - α²/2 - α/2 + α/2) ]
#
# Working in terms of tracer mass rm = m·c and the edge-offset moment
# sx = m·s/2:
#
#     F_q = α · (rm_L + (1 - α) · sx_L)                       (3)
#
# For F < 0 (flow from R to L), the donor is cell R, swept from its
# left edge:
#
#     F_q = α · (rm_R - (1 + α) · sx_R)    where α = F/m_R < 0  (4)
#
# Equations (3) and (4) are implemented in _slopes_face_flux below.
# This is equivalent to TM5's advectx__slopes (Russell & Lerner 1981,
# eqs. 8–12) and the van Leer MUSCL flux (van Leer 1977, eq. 40).
#
# Stencil
# -------
# Each face needs slopes for both the left and right donor cells.
# Each slope needs three consecutive mixing ratios (c_{i-1}, c_i, c_{i+1}).
# For a face between cells L and R, the full stencil is:
#
#     ┌─────┬─────┬─────┬─────┐
#     │ iLL │ iL  │ iR  │ iRR │
#     └─────┴──┬──┴──┬──┴─────┘
#              face        face+1
#              (left)      (right of iR)
#
# iLL, iL  → slope for left donor
# iL, iR   → shared by both slopes
# iR, iRR  → slope for right donor
#
# Total stencil width: 4 cells (2 per side of the face).

"""
    _slopes_face_flux(F, m_l, rm_l, sx_l, m_r, rm_r, sx_r) → FT

Courant-fraction weighted tracer flux through a face with linear
reconstruction (Russell & Lerner 1981, eqs. 8–12). Shared by
`SlopesScheme` and `PPMScheme` (PPM precomputes `sx` from its limited
edge profile and reuses this kernel).

# Arguments
- `F`:    mass flux through the face [kg] (positive = left-to-right)
- `m_l`:  air mass in the left cell [kg]
- `rm_l`: conservative tracer storage in the left cell [carrier-air kg]
- `sx_l`: limited edge-offset first moment of tracer storage [carrier-air kg]
- `m_r`, `rm_r`, `sx_r`: same for the right cell

# Equations

For ``F \\geq 0`` (left cell is donor, Courant fraction ``\\alpha = F / m_L``):

```math
F_q = \\alpha \\bigl(r_{m,L} + (1 - \\alpha)\\, s_{x,L}\\bigr)
```

For ``F < 0`` (right cell is donor, ``\\alpha = F / m_R < 0``):

```math
F_q = \\alpha \\bigl(r_{m,R} - (1 + \\alpha)\\, s_{x,R}\\bigr)
```

Both branches are always evaluated (branchless `ifelse` for GPU safety);
the unused branch's result is discarded.

# Courant-fraction clamping

`α_pos` is clamped to ``[0, 1]`` and `α_neg` to ``[-1, 0]`` before the
multiplication. For well-behaved cells (CFL ≤ 1) the clamp is a no-op
and the formula reduces exactly to Russell-Lerner. When the upstream
mass goes very small or negative — `m → eps(FT)` via the `m_floor` guard
in the calling reconstruction — the unclamped `α = F/eps` blows up and
overflows through the `α·(rm + (1−α)·sx)` polynomial, producing NaN.
The clamp saturates `α` at ±1, giving `f = ±rm_donor`, i.e. pure upwind
transport of the donor's full tracer content. This is the same
positivity safety net `_gamma_clamped_x_flux` provides for
`UpwindScheme`. Mass conservation is preserved because both adjacent
cells see the same clamped α (computed from the shared face's m_donor
and the shared F).

The cost is locally upwind-degraded accuracy in cells where the runtime
CFL exceeds 1; for the well-behaved interior the slopes / PPM physics
is bit-identical to the pre-clamp behavior.
"""
@inline function _slopes_face_flux(F, m_l, rm_l, sx_l, m_r, rm_r, sx_r)
    FT = eltype(F)
    # Defensive α clamp. With the LL→CS mass-regrid bug fixed (extensive `m`
    # converted via density), the clamp is bit-exactly a no-op on the C180
    # ERA5 Dec-2 path: empirical test shows
    # max|Δ| = 0 between clamp-on and clamp-off PPM runs. The clamp is kept
    # as defensive in depth for two cases that can still produce |α| > 1:
    #   (1) Panel-edge cells where the Poisson balance solver corrects
    #       continuity by setting individual face fluxes that exceed cell
    #       mass over the window (e.g. C180 panel-4 south edge cell at
    #       lat 44°N has window-CFL ≈ 1.6 after balance).
    #   (2) Future coarser grids or different met sources where per-substep
    #       CFL exceeds 1 before the runtime's `_cs_static_palindrome_subcycle_count`
    #       can subdivide.
    # Without the clamp, `α = F/eps` blows up and the polynomial overflows.
    # With the clamp, α saturates at ±1 → degrades to upwind locally,
    # mass-conservation preserved (both adjacent cells see the same α).
    # TODO: replace with adaptive per-Strang sub-stepping that scales fluxes
    # by 1/n_palindrome and runs n_palindrome inner Strang calls, matching
    # the FV3 (`fv_tracer2d.F90:133-191`) / TM5 (Krol et al. 2005 §2.3)
    # convention. At that point the clamp becomes dead code and can drop.
    α_pos = clamp(F / m_l, zero(FT), one(FT))
    α_neg = clamp(F / m_r, -one(FT), zero(FT))
    f_pos = α_pos * (rm_l + (one(FT) - α_pos) * sx_l)
    f_neg = α_neg * (rm_r - (one(FT) + α_neg) * sx_r)
    return ifelse(F >= zero(FT), f_pos, f_neg)
end

"""
    _xface_tracer_flux(face_i, j, k, rm, m, F, scheme::SlopesScheme, Nx) → FT

Tracer mass flux through x-face `face_i` for van Leer slopes advection.

# Stencil (periodic in x)
```
  ┌─────┬─────┬─────┬─────┐
  │ iLL │ iL  │ iR  │ iRR │
  └─────┴──┬──┴──┬──┴─────┘
           face_i
```
All indices wrapped via `mod1(·, Nx)` for periodic boundaries.

# Algorithm
1. Load 4 mixing ratios: ``c = r_m / m`` at iLL, iL, iR, iRR
2. Compute limited slopes for left donor (iL) and right donor (iR)
3. Convert slopes to edge-offset moments: ``s_x = m \\cdot s / 2``
4. Apply moment limiter
5. Evaluate Courant-fraction flux via `_slopes_face_flux`
"""
@inline function _xface_tracer_flux(face_i, j, k, rm, m, F, scheme::SlopesScheme, Nx)
    FT = eltype(rm)
    limiter = scheme.limiter

    i_ll = _wrap_periodic(face_i - Int32(2), Nx)
    i_l  = _wrap_periodic(face_i - Int32(1), Nx)
    i_r  = _wrap_periodic(face_i, Nx)
    i_rr = _wrap_periodic(face_i + Int32(1), Nx)

    # Floor prevents NaN if cell mass → 0 (consistent with z-direction at line 399)
    m_floor = eps(eltype(rm))
    c_ll = rm[i_ll, j, k] / max(m[i_ll, j, k], m_floor)
    c_l  = rm[i_l,  j, k] / max(m[i_l,  j, k], m_floor)
    c_r  = rm[i_r,  j, k] / max(m[i_r,  j, k], m_floor)
    c_rr = rm[i_rr, j, k] / max(m[i_rr, j, k], m_floor)

    sc_l = _limited_slope(c_ll, c_l, c_r, limiter)
    sx_l = _limited_moment(max(m[i_l, j, k], m_floor) * (sc_l / FT(2)), rm[i_l, j, k], limiter)

    sc_r = _limited_slope(c_l, c_r, c_rr, limiter)
    sx_r = _limited_moment(max(m[i_r, j, k], m_floor) * (sc_r / FT(2)), rm[i_r, j, k], limiter)

    return _slopes_face_flux(F, max(m[i_l, j, k], m_floor), rm[i_l, j, k], sx_l,
                                max(m[i_r, j, k], m_floor), rm[i_r, j, k], sx_r)
end

"""
    _yface_tracer_flux(i, face_j, k, rm, m, F, scheme::SlopesScheme, Ny) → FT

Tracer mass flux through y-face `face_j` for van Leer slopes advection.

# Boundary conditions
- Closed (no-flux) at `face_j ≤ 1` (south pole) and `face_j > Ny` (north pole)
- Slopes zeroed at boundary cells (`j = 1` and `j = Ny`) to prevent
  extrapolation beyond the domain.  Interior check: `1 < j < Ny`.
- Stencil indices clamped via `max`/`min` (ghost cells mirror the boundary)

# Stencil (closed boundaries)
```
  ┌─────┬─────┬─────┬─────┐
  │ jLL │ jL  │ jR  │ jRR │
  └─────┴──┬──┴──┬──┴─────┘
          face_j
```
Indices clamped: `jLL = max(face_j-2, 1)`, `jRR = min(face_j+1, Ny)`.
"""
@inline function _yface_tracer_flux(i, face_j, k, rm, m, F, scheme::SlopesScheme, Ny)
    FT = eltype(rm)
    limiter = scheme.limiter
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)

    jl  = max(face_j - Int32(1), Int32(1))
    jr  = min(face_j, Ny)
    jll = max(face_j - Int32(2), Int32(1))
    jrr = min(face_j + Int32(1), Ny)

    m_floor = eps(FT)  # prevent NaN if cell mass → 0 (consistent with z-direction)
    c_ll = rm[i, jll, k] / max(m[i, jll, k], m_floor)
    c_l  = rm[i, jl,  k] / max(m[i, jl,  k], m_floor)
    c_r  = rm[i, jr,  k] / max(m[i, jr,  k], m_floor)
    c_rr = rm[i, jrr, k] / max(m[i, jrr, k], m_floor)

    interior_l = (jl > Int32(1)) & (jl < Ny)
    sc_l = _limited_slope(c_ll, c_l, c_r, limiter)
    sx_l = _limited_moment(max(m[i, jl, k], m_floor) * (sc_l / FT(2)), rm[i, jl, k], limiter)
    sx_l = ifelse(interior_l, sx_l, zero(FT))

    interior_r = (jr > Int32(1)) & (jr < Ny)
    sc_r = _limited_slope(c_l, c_r, c_rr, limiter)
    sx_r = _limited_moment(max(m[i, jr, k], m_floor) * (sc_r / FT(2)), rm[i, jr, k], limiter)
    sx_r = ifelse(interior_r, sx_r, zero(FT))

    flux = _slopes_face_flux(F, max(m[i, jl, k], m_floor), rm[i, jl, k], sx_l,
                                max(m[i, jr, k], m_floor), rm[i, jr, k], sx_r)
    return ifelse(at_boundary, zero(FT), flux)
end

"""
    _zface_tracer_flux(i, j, face_k, rm, m, F, scheme::SlopesScheme, Nz) → FT

Tracer mass flux through z-face `face_k` for van Leer slopes advection.

# Boundary conditions
Same as y-direction: closed at `face_k ≤ 1` (TOA) and `face_k > Nz` (surface),
slopes zeroed at boundary levels (`k = 1`, `k = Nz`).

# Mass floor
Unlike the horizontal directions, the z-direction uses an `eps(FT)` floor
on mass in both the mixing ratio computation and the Courant-fraction
formula.  This prevents NaN from division by zero in the uppermost model
level where air mass can be extremely small (pressure thickness ~500 Pa
at TOA vs ~50,000 Pa near the surface).

Matches the convention in the legacy `_rl_z_kernel!` (RussellLerner.jl).
"""
@inline function _zface_tracer_flux(i, j, face_k, rm, m, F, scheme::SlopesScheme, Nz)
    FT = eltype(rm)
    m_floor = eps(FT)
    limiter = scheme.limiter
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)

    kl  = max(face_k - Int32(1), Int32(1))
    kr  = min(face_k, Nz)
    kll = max(face_k - Int32(2), Int32(1))
    krr = min(face_k + Int32(1), Nz)

    c_ll = rm[i, j, kll] / max(m[i, j, kll], m_floor)
    c_l  = rm[i, j, kl]  / max(m[i, j, kl],  m_floor)
    c_r  = rm[i, j, kr]  / max(m[i, j, kr],  m_floor)
    c_rr = rm[i, j, krr] / max(m[i, j, krr], m_floor)

    interior_l = (kl > Int32(1)) & (kl < Nz)
    sc_l = _limited_slope(c_ll, c_l, c_r, limiter)
    sx_l = _limited_moment(max(m[i, j, kl], m_floor) * (sc_l / FT(2)), rm[i, j, kl], limiter)
    sx_l = ifelse(interior_l, sx_l, zero(FT))

    interior_r = (kr > Int32(1)) & (kr < Nz)
    sc_r = _limited_slope(c_l, c_r, c_rr, limiter)
    sx_r = _limited_moment(max(m[i, j, kr], m_floor) * (sc_r / FT(2)), rm[i, j, kr], limiter)
    sx_r = ifelse(interior_r, sx_r, zero(FT))

    flux = _slopes_face_flux(F, max(m[i, j, kl], m_floor), rm[i, j, kl], sx_l,
                                max(m[i, j, kr], m_floor), rm[i, j, kr], sx_r)
    return ifelse(at_boundary, zero(FT), flux)
end


# =========================================================================
# Face-indexed flux functions (for ReducedGaussian / unstructured grids)
# =========================================================================
#
# These functions operate on 2D arrays (cell, level) where cell indices
# come from a face connectivity table.  Currently only UpwindScheme is
# implemented for face-indexed topologies.

"""
    _hface_tracer_flux(rm, m, F, c_left, c_right, k, ::UpwindScheme) → FT

Horizontal tracer mass flux for face-indexed (unstructured) grids.

Cell indices `c_left` and `c_right` are looked up from the mesh
connectivity table by the caller.  The flux formula is identical to
the structured upwind, but indexing uses `(cell, level)` layout.
"""
@inline function _hface_tracer_flux(rm, m, F, c_left, c_right, k, ::UpwindScheme)
    FT = eltype(rm)
    c_l = _mixing_ratio_2d(rm, m, c_left, k)
    c_r = _mixing_ratio_2d(rm, m, c_right, k)
    return ifelse(F >= zero(FT), F * c_l, F * c_r)
end

"""
    _vface_tracer_flux(rm, m, F, c, k_above, k_below, ::UpwindScheme) → FT

Vertical tracer mass flux at interface between levels `k_above` and
`k_below` for face-indexed grids.  Upwind donor selection based on
sign of `F`.
"""
@inline function _vface_tracer_flux(rm, m, F, c, k_above, k_below, ::UpwindScheme)
    FT = eltype(rm)
    c_above = _mixing_ratio_2d(rm, m, c, k_above)
    c_below = _mixing_ratio_2d(rm, m, c, k_below)
    return ifelse(F >= zero(FT), F * c_above, F * c_below)
end


# =========================================================================
# PPMScheme (quadratic reconstruction, order 2)
# =========================================================================
#
# The Piecewise Parabolic Method (Colella & Woodward 1984; Putman & Lin
# 2007) reconstructs a parabolic subcell profile in each cell.
#
# Edge interpolation (CW84 eq. 1.6, 4th-order accurate):
#
#     e_{i+1/2} = (7/12)(c_i + c_{i+1}) - (1/12)(c_{i-1} + c_{i+2})
#
# The parabolic profile in cell i:
#
#     q(ξ) = q_L + ξ·(Δq + q₆·(1-ξ))
#
# where Δq = q_R - q_L, q₆ = 6·(c̄ - (q_L + q_R)/2), ξ ∈ [0,1].
#
# Face flux formula:
# ------------------
# Uses the SAME Courant-fraction formula as SlopesScheme (via
# _slopes_face_flux), but with PPM edge-offset moments instead of
# limited slopes:
#
#     For F > 0 (left donor): sx = m_L · (q_R_L - c̄_L)
#     For F < 0 (right donor): sx = m_R · (c̄_R - q_L_R)
#
# Here (q_R - c̄) is the displacement from the cell mean to the
# PPM-interpolated right edge, generalizing the linear half-slope
# (q_R = c̄ + slope/2 for a linear profile → q_R - c̄ = slope/2).
#
# This is equivalent to integrating a linear profile between c̄ and
# the outflow edge — second-order in α but benefiting from the
# 4th-order edge accuracy.  This matches Putman & Lin (2007) and
# the production PPM code in src/Advection/latlon_mass_flux_ppm.jl.
#
# Stencil: 6 cells per face (3 per side of the face)
#
# References:
# - Colella & Woodward (1984), J. Comp. Phys., 54:174–201.
# - Putman & Lin (2007), J. Comp. Phys., 227:55–78.
# - Lin & Rood (1996), Mon. Wea. Rev., 124:2046–2070.

# ---- PPM helper functions ------------------------------------------------

"""
    _ppm_edge_value(c_ll, c_l, c_r, c_rr) → FT

Compute the 4th-order accurate edge value at the interface between
cells `c_l` and `c_r` (Colella & Woodward 1984, eq. 1.6):

```math
e_{i+1/2} = \\frac{7}{12}(c_i + c_{i+1}) - \\frac{1}{12}(c_{i-1} + c_{i+2})
```

Stencil: 4 cells `[c_ll, c_l | c_r, c_rr]` with the interface between `c_l` and `c_r`.
"""
@inline function _ppm_edge_value(c_ll, c_l, c_r, c_rr)
    FT = typeof(c_ll)
    seven_twelfths = FT(7) / FT(12)
    one_twelfth    = FT(1) / FT(12)
    return seven_twelfths * (c_l + c_r) - one_twelfth * (c_ll + c_rr)
end

"""
    _ppm_edge_value(c_ll, c_l, c_r, c_rr, limiter) → FT

Edge value between `c_l` and `c_r` for `PPMScheme{L}`. The default limiters use
the unlimited fourth-order edge above. `CW84Limiter` uses CW84 eq. 1.7 with van
Leer-limited slopes (eq. 1.8, the minmod of `_limited_slope`),

```math
e_{i+1/2} = \\frac{c_i + c_{i+1}}{2} - \\frac{δc_{i+1} - δc_i}{6},
```

which lies between `c_l` and `c_r` and equals the fourth-order edge where
neither slope is limited.
"""
@inline _ppm_edge_value(c_ll, c_l, c_r, c_rr, ::AbstractLimiter) =
    _ppm_edge_value(c_ll, c_l, c_r, c_rr)

@inline function _ppm_edge_value(c_ll, c_l, c_r, c_rr, ::CW84Limiter)
    FT = typeof(c_l)
    δc_l = _limited_slope(c_ll, c_l, c_r, MonotoneLimiter())
    δc_r = _limited_slope(c_l, c_r, c_rr, MonotoneLimiter())
    return (c_l + c_r) / FT(2) - (δc_r - δc_l) / FT(6)
end

"""
    _ppm_limit_profile(q_L, c_bar, q_R, ::MonotoneLimiter) → (q_L, q_R)

Apply Colella & Woodward (1984) monotonicity constraints to the PPM
edge values of a cell with mean `c_bar`.

# Conditions (CW84 eqs. 1.10a–c):

1. **Local extremum**: if ``(q_R - \\bar{c})(\\bar{c} - q_L) \\leq 0``,
   the cell is a local extremum → flatten: ``q_L = q_R = \\bar{c}``

2. **Left overshoot**: if ``\\Delta q \\cdot q_6 > (\\Delta q)^2``
   (the parabola's minimum lies inside the cell on the left side),
   adjust ``q_L = 3\\bar{c} - 2 q_R``

3. **Right overshoot**: if ``-(\\Delta q)^2 > \\Delta q \\cdot q_6``,
   adjust ``q_R = 3\\bar{c} - 2 q_L``

where ``\\Delta q = q_R - q_L`` and ``q_6 = 6(\\bar{c} - (q_L + q_R)/2)``.

All conditions are evaluated branchlessly via `ifelse` for GPU safety.
"""
@inline function _ppm_limit_profile(q_L, c_bar, q_R, ::MonotoneLimiter)
    FT = typeof(c_bar)

    is_extremum = (q_R - c_bar) * (c_bar - q_L) <= zero(FT)

    dc = q_R - q_L
    c6 = FT(6) * (c_bar - (q_L + q_R) / FT(2))
    needs_left_fix  = dc * c6 > dc * dc
    needs_right_fix = -(dc * dc) > dc * c6

    q_L_new = ifelse(is_extremum, c_bar,
              ifelse(needs_left_fix, FT(3) * c_bar - FT(2) * q_R, q_L))
    q_R_new = ifelse(is_extremum, c_bar,
              ifelse(needs_right_fix, FT(3) * c_bar - FT(2) * q_L_new, q_R))

    return q_L_new, q_R_new
end

"""
    _ppm_limit_profile(q_L, c_bar, q_R, ::NoLimiter) → (q_L, q_R)

No limiting — return raw 4th-order edge values unchanged.
May produce new extrema (oscillatory) near sharp gradients.
"""
@inline _ppm_limit_profile(q_L, c_bar, q_R, ::NoLimiter) = (q_L, q_R)

"""
    _ppm_limit_profile(q_L, c_bar, q_R, ::PositivityLimiter) → (q_L, q_R)

Clamp edge values to be non-negative.  Weaker than `MonotoneLimiter`
but sufficient for species that must remain ≥ 0.
"""
@inline function _ppm_limit_profile(q_L, c_bar, q_R, ::PositivityLimiter)
    FT = typeof(c_bar)
    return max(q_L, zero(FT)), max(q_R, zero(FT))
end

"""
    _ppm_limit_profile(q_L, c_bar, q_R, ::CW84Limiter) → (q_L, q_R)

The monotonicity conditions of `MonotoneLimiter` (CW84 eq. 1.10).
"""
@inline _ppm_limit_profile(q_L, c_bar, q_R, ::CW84Limiter) =
    _ppm_limit_profile(q_L, c_bar, q_R, MonotoneLimiter())

"""
    _ppm_raw_moments(F, m_l, c_l, q_L_l, q_R_l, m_r, c_r, q_L_r, q_R_r, limiter) → (sx_l, sx_r)

Moments of the two donor cells of a face for `_slopes_face_flux`: the left cell
(outflow through its right edge when `F ≥ 0`) and the right cell (outflow
through its left edge when `F < 0`). `m_l` and `m_r` are the floored masses.

The default limiters use the limited outflow edge as a linear
(Russell–Lerner) slope: `sx_l = m_l (q_R_l − c_l)`, `sx_r = m_r (c_r − q_L_r)`.

`CW84Limiter` adds the parabola's curvature `b₀ = (q_L − c) + (q_R − c)` over
the swept fraction `α` (clamped like in `_slopes_face_flux`),

    sx_l = m_l ((q_R_l − c_l) − α₊ b₀_l),   sx_r = m_r ((c_r − q_L_r) − α₋ b₀_r),

so that `_slopes_face_flux` returns `F` times the mean of the donor parabola
over the swept part of the cell (CW84 eq. 1.12).
"""
@inline _ppm_raw_moments(F, m_l, c_l, q_L_l, q_R_l, m_r, c_r, q_L_r, q_R_r, ::AbstractLimiter) =
    (m_l * (q_R_l - c_l), m_r * (c_r - q_L_r))

@inline function _ppm_raw_moments(F, m_l, c_l, q_L_l, q_R_l, m_r, c_r, q_L_r, q_R_r, ::CW84Limiter)
    α_pos = clamp(F / m_l, zero(F), one(F))
    α_neg = clamp(F / m_r, -one(F), zero(F))
    b0_l = (q_L_l - c_l) + (q_R_l - c_l)
    b0_r = (q_L_r - c_r) + (q_R_r - c_r)
    return m_l * ((q_R_l - c_l) - α_pos * b0_l), m_r * ((c_r - q_L_r) - α_neg * b0_r)
end

# ---- PPM x-face flux (periodic) -----------------------------------------

"""
    _xface_tracer_flux(face_i, j, k, rm, m, F, scheme::PPMScheme, Nx) → FT

Tracer mass flux through x-face `face_i` for PPM advection.

# Stencil (periodic in x, 6 cells)
```
  ┌──────┬──────┬──────┬──────┬──────┬──────┐
  │ i₋₃  │ i₋₂  │ i₋₁  │ i₀   │ i₊₁  │ i₊₂  │
  └──────┴──────┴───┬──┴──┬───┴──────┴──────┘
              e_left  face_i  e_right
                   (e_face)
```
- Left donor cell = `i₋₁ = face_i - 1`
- Right donor cell = `i₀ = face_i`
- `e_left`: left edge of left donor (at face_i - 1)
- `e_face`: shared interface edge (at face_i)
- `e_right`: right edge of right donor (at face_i + 1)

# Algorithm
1. Load 6 mixing ratios from the stencil
2. Compute 3 CW84 4th-order edge values
3. Apply monotone profile limiting to each donor cell
4. Compute edge-offset moments (PPM analogue of slopes `sx`)
5. Evaluate Courant-fraction flux via `_slopes_face_flux`
"""
@inline function _xface_tracer_flux(face_i, j, k, rm, m, F, scheme::PPMScheme, Nx)
    FT = eltype(rm)
    limiter = scheme.limiter

    i_3  = _wrap_periodic(face_i - Int32(3), Nx)
    i_2  = _wrap_periodic(face_i - Int32(2), Nx)
    i_1  = _wrap_periodic(face_i - Int32(1), Nx)
    i_0  = _wrap_periodic(face_i, Nx)
    i_p  = _wrap_periodic(face_i + Int32(1), Nx)
    i_pp = _wrap_periodic(face_i + Int32(2), Nx)

    # Floor prevents NaN if cell mass → 0 (consistent with all other schemes)
    m_floor = eps(FT)
    c_3  = rm[i_3,  j, k] / max(m[i_3,  j, k], m_floor)
    c_2  = rm[i_2,  j, k] / max(m[i_2,  j, k], m_floor)
    c_1  = rm[i_1,  j, k] / max(m[i_1,  j, k], m_floor)
    c_0  = rm[i_0,  j, k] / max(m[i_0,  j, k], m_floor)
    c_p  = rm[i_p,  j, k] / max(m[i_p,  j, k], m_floor)
    c_pp = rm[i_pp, j, k] / max(m[i_pp, j, k], m_floor)

    e_left  = _ppm_edge_value(c_3, c_2, c_1, c_0, limiter)
    e_face  = _ppm_edge_value(c_2, c_1, c_0, c_p, limiter)
    e_right = _ppm_edge_value(c_1, c_0, c_p, c_pp, limiter)

    q_L_l, q_R_l = _ppm_limit_profile(e_left, c_1, e_face, limiter)
    q_L_r, q_R_r = _ppm_limit_profile(e_face, c_0, e_right, limiter)

    m_l = max(m[i_1, j, k], m_floor)
    m_r = max(m[i_0, j, k], m_floor)
    sx_l, sx_r = _ppm_raw_moments(F, m_l, c_1, q_L_l, q_R_l, m_r, c_0, q_L_r, q_R_r, limiter)
    sx_l = _limited_moment(sx_l, rm[i_1, j, k], limiter)
    sx_r = _limited_moment(sx_r, rm[i_0, j, k], limiter)

    return _slopes_face_flux(F, m_l, rm[i_1, j, k], sx_l, m_r, rm[i_0, j, k], sx_r)
end

# ---- PPM y-face flux (closed boundaries) --------------------------------

"""
    _yface_tracer_flux(i, face_j, k, rm, m, F, scheme::PPMScheme, Ny) → FT

Tracer mass flux through y-face `face_j` for PPM advection.

# Boundary conditions
- Closed (no-flux) at `face_j ≤ 1` and `face_j > Ny`
- PPM edge offsets zeroed when a donor cell is within 2 cells of a
  boundary (`jl ≤ 2` or `jl ≥ Ny-1`), effectively falling back to
  upwind there.  This matches the production PPM code's pole fallback.
- Stencil indices clamped via `max`/`min`.
"""
@inline function _yface_tracer_flux(i, face_j, k, rm, m, F, scheme::PPMScheme, Ny)
    FT = eltype(rm)
    limiter = scheme.limiter
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)

    j3l = max(face_j - Int32(3), Int32(1))
    jll = max(face_j - Int32(2), Int32(1))
    jl  = max(face_j - Int32(1), Int32(1))
    jr  = min(face_j, Ny)
    jrr = min(face_j + Int32(1), Ny)
    j3r = min(face_j + Int32(2), Ny)

    m_floor = eps(FT)  # prevent NaN if cell mass → 0 (consistent with z-direction)
    c_3l = rm[i, j3l, k] / max(m[i, j3l, k], m_floor)
    c_ll = rm[i, jll, k] / max(m[i, jll, k], m_floor)
    c_l  = rm[i, jl,  k] / max(m[i, jl,  k], m_floor)
    c_r  = rm[i, jr,  k] / max(m[i, jr,  k], m_floor)
    c_rr = rm[i, jrr, k] / max(m[i, jrr, k], m_floor)
    c_3r = rm[i, j3r, k] / max(m[i, j3r, k], m_floor)

    e_left  = _ppm_edge_value(c_3l, c_ll, c_l, c_r, limiter)
    e_face  = _ppm_edge_value(c_ll, c_l, c_r, c_rr, limiter)
    e_right = _ppm_edge_value(c_l, c_r, c_rr, c_3r, limiter)

    q_L_l, q_R_l = _ppm_limit_profile(e_left, c_l, e_face, limiter)
    q_L_r, q_R_r = _ppm_limit_profile(e_face, c_r, e_right, limiter)

    m_l = max(m[i, jl, k], m_floor)
    m_r = max(m[i, jr, k], m_floor)
    sx_l, sx_r = _ppm_raw_moments(F, m_l, c_l, q_L_l, q_R_l, m_r, c_r, q_L_r, q_R_r, limiter)

    interior_l = (jl > Int32(2)) & (jl < Ny - Int32(1))
    sx_l = ifelse(interior_l, _limited_moment(sx_l, rm[i, jl, k], limiter), zero(FT))

    interior_r = (jr > Int32(2)) & (jr < Ny - Int32(1))
    sx_r = ifelse(interior_r, _limited_moment(sx_r, rm[i, jr, k], limiter), zero(FT))

    flux = _slopes_face_flux(F, m_l, rm[i, jl, k], sx_l, m_r, rm[i, jr, k], sx_r)
    return ifelse(at_boundary, zero(FT), flux)
end

# ---- PPM z-face flux (closed boundaries, eps floor) ---------------------

"""
    _zface_tracer_flux(i, j, face_k, rm, m, F, scheme::PPMScheme, Nz) → FT

Tracer mass flux through z-face `face_k` for PPM advection.

# Boundary conditions
Same as y-direction: closed at `face_k ≤ 1` (TOA) and `face_k > Nz`
(surface), with PPM edge offsets zeroed at boundary levels.

Uses `eps(FT)` floor on mass to prevent NaN from division by zero
in the thin uppermost model levels.
"""
@inline function _zface_tracer_flux(i, j, face_k, rm, m, F, scheme::PPMScheme, Nz)
    FT = eltype(rm)
    m_floor = eps(FT)
    limiter = scheme.limiter
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)

    k3l = max(face_k - Int32(3), Int32(1))
    kll = max(face_k - Int32(2), Int32(1))
    kl  = max(face_k - Int32(1), Int32(1))
    kr  = min(face_k, Nz)
    krr = min(face_k + Int32(1), Nz)
    k3r = min(face_k + Int32(2), Nz)

    c_3l = rm[i, j, k3l] / max(m[i, j, k3l], m_floor)
    c_ll = rm[i, j, kll] / max(m[i, j, kll], m_floor)
    c_l  = rm[i, j, kl]  / max(m[i, j, kl],  m_floor)
    c_r  = rm[i, j, kr]  / max(m[i, j, kr],  m_floor)
    c_rr = rm[i, j, krr] / max(m[i, j, krr], m_floor)
    c_3r = rm[i, j, k3r] / max(m[i, j, k3r], m_floor)

    e_left  = _ppm_edge_value(c_3l, c_ll, c_l, c_r, limiter)
    e_face  = _ppm_edge_value(c_ll, c_l, c_r, c_rr, limiter)
    e_right = _ppm_edge_value(c_l, c_r, c_rr, c_3r, limiter)

    q_L_l, q_R_l = _ppm_limit_profile(e_left, c_l, e_face, limiter)
    q_L_r, q_R_r = _ppm_limit_profile(e_face, c_r, e_right, limiter)

    m_l = max(m[i, j, kl], m_floor)
    m_r = max(m[i, j, kr], m_floor)
    sx_l, sx_r = _ppm_raw_moments(F, m_l, c_l, q_L_l, q_R_l, m_r, c_r, q_L_r, q_R_r, limiter)

    interior_l = (kl > Int32(2)) & (kl < Nz - Int32(1))
    sx_l = ifelse(interior_l, _limited_moment(sx_l, rm[i, j, kl], limiter), zero(FT))

    interior_r = (kr > Int32(2)) & (kr < Nz - Int32(1))
    sx_r = ifelse(interior_r, _limited_moment(sx_r, rm[i, j, kr], limiter), zero(FT))

    flux = _slopes_face_flux(F, m_l, rm[i, j, kl], sx_l, m_r, rm[i, j, kr], sx_r)
    return ifelse(at_boundary, zero(FT), flux)
end
