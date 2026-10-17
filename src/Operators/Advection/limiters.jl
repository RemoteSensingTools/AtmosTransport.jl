# ---------------------------------------------------------------------------
# Slope and moment limiters — @inline, branchless, GPU-safe
#
# These functions enforce monotonicity or positivity constraints on the
# piecewise-linear (or higher) subcell reconstruction.  They are the
# key ingredient that makes second-order advection stable and non-oscillatory.
#
# Why branchless?
# ---------------
# GPU warps execute in lockstep (SIMT).  An `if/else` branch causes warp
# divergence — threads that take different paths are serialized.  Using
# `ifelse(cond, a, b)` ensures ALL threads compute BOTH values and select
# one, keeping the warp fully utilized.  The compiler eliminates dead
# computation when the condition is statically known (e.g., scheme type).
#
# References
# ----------
# - van Leer (1977), "Towards the ultimate conservative difference
#   scheme. IV", J. Comp. Phys., 23:276–299.
#   → Original MUSCL slope limiter framework.
#
# - Sweby (1984), "High resolution schemes using flux limiters for
#   hyperbolic conservation laws", SIAM J. Numer. Anal., 21:995–1011.
#   → TVD (total variation diminishing) conditions for flux limiters.
#
# - Zalesak (1979), "Fully multidimensional flux-corrected transport
#   algorithms for fluids", J. Comp. Phys., 31:335–362.
#   → Flux-corrected transport and positivity preservation.
#
# - Russell & Lerner (1981), "A new finite-differencing scheme for the
#   tracer transport equation", J. Appl. Meteor., 20:1483–1498.
#   → Slopes advection for atmospheric tracers (TM5 heritage).
# ---------------------------------------------------------------------------

# ---- Minmod primitives ---------------------------------------------------

"""
    _minmod3(a, b, c) → typeof(a)

Three-argument minmod: returns the value with smallest magnitude if all
three share the same sign, zero otherwise.

```math
\\text{minmod}(a, b, c) = \\begin{cases}
  \\min(a, b, c)  & \\text{if } a, b, c > 0 \\\\
  \\max(a, b, c)  & \\text{if } a, b, c < 0 \\\\
  0              & \\text{otherwise}
\\end{cases}
```

This is the core primitive of the van Leer / Sweby TVD framework.
The three arguments are typically:

1. The centered slope: ``s_c = (c_{i+1} - c_{i-1}) / 2``
2. Twice the right one-sided slope: ``2 (c_{i+1} - c_i)``
3. Twice the left one-sided slope: ``2 (c_i - c_{i-1})``

The factor of 2 comes from the Sweby TVD region boundary — it is the
maximum slope that still yields monotone face values when the Courant
number reaches 1 (Sweby 1984, Theorem 4.1).
"""
@inline function _minmod3(a, b, c)
    all_pos = (a > zero(a)) & (b > zero(b)) & (c > zero(c))
    all_neg = (a < zero(a)) & (b < zero(b)) & (c < zero(c))
    return ifelse(all_pos, min(a, b, c),
           ifelse(all_neg, max(a, b, c), zero(a)))
end

# ---- Slope limiters (operating on mixing ratio) --------------------------
#
# Given three consecutive cell-mean mixing ratios (c_left, c_center, c_right),
# compute a limited slope for the center cell.  The slope is the coefficient
# in the piecewise-linear reconstruction:
#
#   q(x) = c_center + slope · (x - x_center) / Δx
#
# In mass-flux advection, this slope is converted to an edge-offset moment
# sx = m · slope / 2, which is then used in the Courant-fraction flux formula
# (see reconstruction.jl).

"""
    _limited_slope(c_left, c_center, c_right, ::MonotoneLimiter)

Van Leer / Sweby monotone slope limiter (van Leer 1977; Sweby 1984).

Computes the centered slope and limits it against twice the one-sided
slopes using the three-argument minmod:

```math
s = \\text{minmod}\\!\\left(
    \\frac{c_{i+1} - c_{i-1}}{2},\\;
    2(c_{i+1} - c_i),\\;
    2(c_i - c_{i-1})
\\right)
```

This ensures the reconstruction is TVD (total variation diminishing):
the reconstructed face values ``c_i \\pm s/2`` lie between the
neighboring cell means, preventing new extrema.

TM5 does not use this limiter: its slopes advection carries the slope as a
prognostic moment and only clamps it to ``|r_{xm}| \\le r_m``.
"""
@inline function _limited_slope(c_left, c_center, c_right, ::MonotoneLimiter)
    sc = (c_right - c_left) / 2
    return _minmod3(sc,
                    2 * (c_right  - c_center),
                    2 * (c_center - c_left))
end

"""
    _limited_slope(c_left, c_center, c_right, ::NoLimiter)

Unlimited centered slope: ``s = (c_{i+1} - c_{i-1}) / 2``.

Second-order accurate but may introduce new extrema (Gibbs-type
oscillations) near sharp gradients.  Useful for smooth test problems
and as a reference for limiter comparison.
"""
@inline function _limited_slope(c_left, c_center, c_right, ::NoLimiter)
    return (c_right - c_left) / 2
end

"""
    _limited_slope(c_left, c_center, c_right, ::PositivityLimiter)

Positivity-preserving slope limiter.

Keeps the sign of the centered slope and clips its magnitude so that both
face values ``c_i \\pm s/2`` stay non-negative, i.e. ``|s| \\le 2 c_i``:

```math
s = \\operatorname{sign}(s_c)\\, \\min\\bigl(|s_c|,\\; 2 \\max(c_i, 0)\\bigr),
\\qquad s_c = \\frac{c_{i+1} - c_{i-1}}{2}
```

The rule treats rising and falling profiles alike, so the reconstruction of a
mirrored field is the mirror of the reconstruction. In moment form it is the
zero-referenced clamp ``|s_x| \\le r_m`` of `_limited_moment`. It does not
bound the reconstruction by the neighboring means: weaker than
`MonotoneLimiter` (allows new maxima) but keeps reconstructed values
non-negative — important for species mixing ratios that must remain ≥ 0.
"""
@inline function _limited_slope(c_left, c_center, c_right, ::PositivityLimiter)
    sc = (c_right - c_left) / 2
    return sign(sc) * min(abs(sc), 2 * max(c_center, zero(c_center)))
end

# ---- Moment limiters (operating on tracer mass) --------------------------
#
# After computing the slope in mixing-ratio space, we convert to the edge-offset
# moment sx = m · slope / 2. Monotonicity is already enforced relative to
# neighbouring values. A second bound relative to tracer zero breaks signed
# tracers and constant-offset invariance, so only the explicit
# `PositivityLimiter` applies a zero-referenced moment clamp.

"""
    _limited_moment(sx, rm_cell, ::MonotoneLimiter)

Return the moment produced by the monotone mixing-ratio reconstruction.

The slope/profile limiter has already bounded reconstructed values by local
neighbouring mixing ratios. Applying an additional ``[-r_m, r_m]`` clamp would
anchor the result to zero: it reverses its bounds when ``r_m < 0`` and makes
transporting ``q-q_0`` differ from transporting ``q``. Keeping the moment
unchanged makes the default monotone scheme valid for signed tracers and
equivariant under a constant VMR offset.
"""
@inline _limited_moment(sx, _rm_cell, ::MonotoneLimiter) = sx

"""
    _limited_moment(sx, rm_cell, ::NoLimiter)

No moment limiting — returns ``s_x`` unchanged.
"""
@inline _limited_moment(sx, rm_cell, ::NoLimiter) = sx

"""
    _limited_moment(sx, rm_cell, ::CW84Limiter)

No moment clamp: the limited CW84 parabola already bounds the swept mean by
neighbouring cell means (see `MonotoneLimiter`).
"""
@inline _limited_moment(sx, _rm_cell, ::CW84Limiter) = sx

"""
    _limited_moment(sx, rm_cell, ::PositivityLimiter)

Clamp ``s_x`` to ``[-r_m, r_m]``. This limiter requires a non-negative
tracer field and is intentionally incompatible with signed tracers. A
non-positive cell mean returns a zero moment instead of reversing the clamp
bounds; this is a misuse guard, not a signed-tracer transport policy.
"""
@inline function _limited_moment(sx, rm_cell, ::PositivityLimiter)
    limited = max(min(sx, rm_cell), -rm_cell)
    return ifelse(rm_cell > zero(rm_cell), limited, zero(sx))
end
