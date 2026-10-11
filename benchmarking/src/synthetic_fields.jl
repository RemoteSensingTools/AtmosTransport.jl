# Synthetic cubed-sphere air mass and face mass fluxes of the benchmark cases.
# Base only, so the test suite checks them without loading AtmosTransport.
#
# Arrays follow the harness layout: per panel, air mass `(Nc+2Hp, Nc+2Hp, Nz)`,
# x-face fluxes `am` `(Nc+2Hp+1, Nc+2Hp, Nz)`, y-face fluxes `bm`
# `(Nc+2Hp, Nc+2Hp+1, Nz)`, and level-interface fluxes `cm`
# `(Nc+2Hp, Nc+2Hp, Nz+1)`, with interior cells at `Hp+1:Hp+Nc`. Fluxes are
# mass per sweep; positive values point toward higher index (downward for
# `cm`). Only interior cells and faces are written; halos stay zero.

"Bound on a synthetic case's palindrome Courant number; the harness refuses a case above it."
const MAX_SYNTHETIC_COURANT = 0.5

"Bound on the relative air-mass drift of a synthetic case over its steps (Float32 rounding is far below it)."
const MAX_SYNTHETIC_MASS_DRIFT = 1e-3

"Largest stream-function face flux as a fraction of the smaller adjacent cell mass."
const SYNTHETIC_GYRE_FLUX_FRACTION = 0.05

"Bound on the divergent part's face fluxes and on `cm`, as fractions of the smaller adjacent cell mass."
const SYNTHETIC_DIVERGENT_FLUX_FRACTION = 0.01

"""
    fill_synthetic_mass!(panels_m, Nc, Hp)

Smooth air mass of about 1e9 kg per cell, varying by a few percent across
each panel and with level. Writes interior cells only; fill the halos with
the cubed-sphere halo exchange afterwards.
"""
function fill_synthetic_mass!(panels_m, Nc::Integer, Hp::Integer)
    @inbounds for p in 1:6
        m = panels_m[p]
        FT = eltype(m)
        Nz = size(m, 3)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            x = FT(i - 1) / max(FT(Nc - 1), one(FT))
            y = FT(j - 1) / max(FT(Nc - 1), one(FT))
            z = FT(k - 1) / max(FT(Nz - 1), one(FT))
            m[Hp + i, Hp + j, k] = FT(1e9) * (FT(1) + FT(0.05) * sinpi(FT(2) * x) +
                                   FT(0.04) * cospi(FT(2) * y) + FT(0.02) * z +
                                   FT(0.001) * p)
        end
    end
    return panels_m
end

# Unit-amplitude flux patterns of one panel, in Float64, evaluated per face
# from 1-D factors, so building the fields needs O(Nc + Nz) extra memory:
#  - gyre part `_am_s`, `_bm_s`: discrete curl of a stream function ψ on cell
#    corners that vanishes on the panel boundary (one closed gyre per panel
#    and level), so every cell's horizontal divergence is exactly zero and no
#    flux crosses a panel edge;
#  - divergent part `_am_d`, `_bm_d`: a pattern that vanishes on the panel
#    edges, times g(k) with Σₖ g(k) = 0;
#  - vertical part: cm[1] = 0, cm[k+1] = cm[k] − div_h(k) of the divergent
#    part (continuity); cm[Nz+1] = 0 because the column sum of the divergence
#    is zero, so the rounding residual there is dropped.
# Indices are 1-based within the panel interior: corners and faces 1:Nc+1,
# cells 1:Nc.
struct _PanelPattern
    amplitude::Float64
    edge_sine::Vector{Float64}     # sinpi((i - 1) / Nc) at corners and faces, zero on the edges
    gyre_level::Vector{Float64}    # stream-function amplitude by level
    g::Vector{Float64}             # divergent-part level factor, zero column sum
    x_wave::Vector{Float64}        # cross-face factor of the x-face divergent part
    y_wave::Vector{Float64}        # cross-face factor of the y-face divergent part
end

function _PanelPattern(p::Integer, Nc::Integer, Nz::Integer)
    corner = [(i - 1) / Nc for i in 1:(Nc + 1)]
    centre = [(i - 0.5) / Nc for i in 1:Nc]
    level = [(k - 0.5) / Nz for k in 1:Nz]
    g = [cospi(z) for z in level]
    g .-= sum(g) / Nz
    return _PanelPattern((isodd(p) ? 1.0 : -1.0) * (1 + 0.1 * (p - 1)),
                         [sinpi(c) for c in corner],
                         [1 + 0.5 * cospi(z) for z in level], g,
                         [cospi(2 * c + 0.1 * p) for c in centre],
                         [cospi(2 * c - 0.1 * p) for c in centre])
end

@inline _ψ(P::_PanelPattern, i, j, k) = P.amplitude * P.gyre_level[k] * P.edge_sine[i] * P.edge_sine[j]
@inline _am_s(P::_PanelPattern, i, j, k) = _ψ(P, i, j + 1, k) - _ψ(P, i, j, k)
@inline _bm_s(P::_PanelPattern, i, j, k) = _ψ(P, i, j, k) - _ψ(P, i + 1, j, k)
@inline _am_d(P::_PanelPattern, i, j, k) = P.g[k] * P.edge_sine[i] * P.x_wave[j]
@inline _bm_d(P::_PanelPattern, i, j, k) = 0.5 * P.g[k] * P.edge_sine[j] * P.y_wave[i]
@inline _divergence_d(P::_PanelPattern, i, j, k) =
    _am_d(P, i + 1, j, k) - _am_d(P, i, j, k) + _bm_d(P, i, j + 1, k) - _bm_d(P, i, j, k)

# Smaller mass of the two cells sharing an interior x face, y face, or level interface.
@inline _x_donor(m, Hp, Nc, i, j, k) = min(m[Hp + max(i - 1, 1), Hp + j, k], m[Hp + min(i, Nc), Hp + j, k])
@inline _y_donor(m, Hp, Nc, i, j, k) = min(m[Hp + i, Hp + max(j - 1, 1), k], m[Hp + i, Hp + min(j, Nc), k])
@inline _z_donor(m, Hp, i, j, k) = min(m[Hp + i, Hp + j, k - 1], m[Hp + i, Hp + j, k])

# Largest |flux| / donor mass of one panel's gyre faces, divergent faces, and
# vertical interfaces (unit amplitudes), streamed without storing a field.
function _panel_fractions(P::_PanelPattern, m, Nc, Hp, Nz)
    gyre = divergent = vertical = 0.0
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        donor = _x_donor(m, Hp, Nc, i, j, k)
        gyre = max(gyre, abs(_am_s(P, i, j, k)) / donor)
        divergent = max(divergent, abs(_am_d(P, i, j, k)) / donor)
    end
    @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        donor = _y_donor(m, Hp, Nc, i, j, k)
        gyre = max(gyre, abs(_bm_s(P, i, j, k)) / donor)
        divergent = max(divergent, abs(_bm_d(P, i, j, k)) / donor)
    end
    @inbounds for j in 1:Nc, i in 1:Nc
        c = 0.0
        for k in 1:(Nz - 1)
            c -= _divergence_d(P, i, j, k)                 # cm at interface k + 1
            vertical = max(vertical, abs(c) / _z_donor(m, Hp, i, j, k + 1))
        end
    end
    return gyre, divergent, vertical
end

"""
    fill_synthetic_fluxes!(am, bm, cm, panels_m, Nc, Hp)

Static face mass fluxes that leave every cell's air mass unchanged by a full
transport step, so all steps of a case do the same work at the same Courant
number:

- horizontal: the discrete curl of a stream function that vanishes on each
  panel's boundary (zero divergence in every cell, no flux across panel
  edges), scaled so the largest face flux moves
  `SYNTHETIC_GYRE_FLUX_FRACTION` of the smaller adjacent cell mass; plus a
  divergent, level-dependent part that vanishes on panel edges and sums to
  zero over each column;
- vertical: `cm` from continuity with the divergent part, zero at the model
  top and the surface.

The divergent part is scaled so that neither its face fluxes nor `cm` exceed
`SYNTHETIC_DIVERGENT_FLUX_FRACTION` of the adjacent mass. The resulting
Courant numbers do not depend on the grid size beyond those bounds. A single
level has no divergent part and no vertical flux. `Nc >= 2` is required: a
`C1` panel has no interior corner, so its circulation would be zero.

The fields are evaluated face by face from 1-D factors (two streaming
passes: normalisation, then filling), so the extra memory is O(Nc + Nz).
"""
function fill_synthetic_fluxes!(am, bm, cm, panels_m, Nc::Integer, Hp::Integer)
    Nc >= 2 || throw(ArgumentError(
        "synthetic fluxes need Nc >= 2 (C2 or larger); a C$(Nc) panel has no interior corner, " *
        "so its closed circulation would be zero"))
    Nz = size(panels_m[1], 3)
    gyre_fraction = divergent_fraction = vertical_fraction = 0.0
    for p in 1:6
        gyre, divergent, vertical = _panel_fractions(_PanelPattern(p, Nc, Nz), panels_m[p], Nc, Hp, Nz)
        gyre_fraction = max(gyre_fraction, gyre)
        divergent_fraction = max(divergent_fraction, divergent)
        vertical_fraction = max(vertical_fraction, vertical)
    end
    # A component that is identically zero (the divergent part on one level)
    # keeps a zero scale instead of dividing by zero.
    divergent_max = max(divergent_fraction, vertical_fraction)
    gyre_scale = gyre_fraction > 0 ? SYNTHETIC_GYRE_FLUX_FRACTION / gyre_fraction : 0.0
    divergent_scale = divergent_max > 0 ? SYNTHETIC_DIVERGENT_FLUX_FRACTION / divergent_max : 0.0
    for p in 1:6
        P = _PanelPattern(p, Nc, Nz)
        a, b, c = am[p], bm[p], cm[p]
        FT = eltype(a)
        fill!(a, zero(FT))
        fill!(b, zero(FT))
        fill!(c, zero(FT))
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
            a[Hp + i, Hp + j, k] = FT(gyre_scale * _am_s(P, i, j, k) + divergent_scale * _am_d(P, i, j, k))
        end
        @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
            b[Hp + i, Hp + j, k] = FT(gyre_scale * _bm_s(P, i, j, k) + divergent_scale * _bm_d(P, i, j, k))
        end
        @inbounds for j in 1:Nc, i in 1:Nc
            column = 0.0
            for k in 1:(Nz - 1)
                column -= _divergence_d(P, i, j, k)
                c[Hp + i, Hp + j, k + 1] = FT(divergent_scale * column)
            end
        end
    end
    return nothing
end

"""
    synthetic_courant(panels_m, am, bm, cm, Nc, Hp) -> Float64

Palindrome Courant number of the fields, as the cubed-sphere transport's
static subcycling budget computes it: the largest over interior cells of
twice the summed outgoing face fluxes of all three directions divided by the
cell mass. The transport subcycles once this exceeds its CFL limit (0.95).
"""
function synthetic_courant(panels_m, am, bm, cm, Nc::Integer, Hp::Integer)
    Nz = size(panels_m[1], 3)
    worst = 0.0
    @inbounds for p in 1:6, k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
        out = max(0, -am[p][i, j, k]) + max(0, am[p][i + 1, j, k]) +
              max(0, -bm[p][i, j, k]) + max(0, bm[p][i, j + 1, k]) +
              max(0, -cm[p][i, j, k]) + max(0, cm[p][i, j, k + 1])
        worst = max(worst, 2 * Float64(out) / Float64(panels_m[p][i, j, k]))
    end
    return worst
end

"""
    check_synthetic_courant(courant, case_name; stage)

Throw an `ErrorException` naming the case and the stage when the palindrome
Courant number exceeds `MAX_SYNTHETIC_COURANT`.
"""
function check_synthetic_courant(courant::Real, case_name::AbstractString; stage::AbstractString)
    courant <= MAX_SYNTHETIC_COURANT || error(
        "synthetic case $(case_name): palindrome Courant number $(round(courant; sigdigits = 3)) " *
        "$(stage) exceeds $(MAX_SYNTHETIC_COURANT); the synthetic fields are not a sane, " *
        "steady benchmark workload (see benchmarking/src/synthetic_fields.jl)")
    return nothing
end
