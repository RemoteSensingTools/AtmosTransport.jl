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

# Unit-amplitude flux patterns of one panel, in Float64, interior only:
#  - `am_s`, `bm_s`: discrete curl of a stream function ψ on cell corners that
#    vanishes on the panel boundary (one closed gyre per panel and level), so
#    every cell's horizontal divergence is exactly zero and no flux crosses a
#    panel edge;
#  - `am_d`, `bm_d`: a divergent pattern that vanishes on the panel edges,
#    times g(k) with Σₖ g(k) = 0;
#  - `cm_d`: vertical fluxes closing continuity for the divergent part,
#    cm[1] = 0, cm[k+1] = cm[k] − div_h(k); cm[Nz+1] = 0 because the column
#    sum of the divergence is zero.
function _unit_panel_fluxes(p::Integer, Nc::Integer, Nz::Integer)
    corner = [(i - 1) / Nc for i in 1:(Nc + 1)]
    centre = [(i - 0.5) / Nc for i in 1:Nc]
    level = [(k - 0.5) / Nz for k in 1:Nz]
    gyre_level = [1 + 0.5 * cospi(z) for z in level]
    g = [cospi(z) for z in level]
    g .-= sum(g) / Nz
    amplitude = (isodd(p) ? 1.0 : -1.0) * (1 + 0.1 * (p - 1))
    ψ(i, j, k) = amplitude * gyre_level[k] * sinpi(corner[i]) * sinpi(corner[j])
    am_s = [ψ(i, j + 1, k) - ψ(i, j, k) for i in 1:(Nc + 1), j in 1:Nc, k in 1:Nz]
    bm_s = [ψ(i, j, k) - ψ(i + 1, j, k) for i in 1:Nc, j in 1:(Nc + 1), k in 1:Nz]
    am_d = [g[k] * sinpi(corner[i]) * cospi(2 * centre[j] + 0.1 * p)
            for i in 1:(Nc + 1), j in 1:Nc, k in 1:Nz]
    bm_d = [0.5 * g[k] * sinpi(corner[j]) * cospi(2 * centre[i] - 0.1 * p)
            for i in 1:Nc, j in 1:(Nc + 1), k in 1:Nz]
    cm_d = zeros(Nc, Nc, Nz + 1)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        divergence = am_d[i + 1, j, k] - am_d[i, j, k] + bm_d[i, j + 1, k] - bm_d[i, j, k]
        cm_d[i, j, k + 1] = cm_d[i, j, k] - divergence
    end
    cm_d[:, :, Nz + 1] .= 0
    return am_s, bm_s, am_d, bm_d, cm_d
end

# Largest |flux| / (smaller adjacent cell mass) of the interior faces.
function _max_face_fraction(am, bm, cm, m, Nc, Hp, Nz)
    r = 0.0
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        donor = min(m[Hp + max(i - 1, 1), Hp + j, k], m[Hp + min(i, Nc), Hp + j, k])
        r = max(r, abs(am[i, j, k]) / donor)
    end
    @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        donor = min(m[Hp + i, Hp + max(j - 1, 1), k], m[Hp + i, Hp + min(j, Nc), k])
        r = max(r, abs(bm[i, j, k]) / donor)
    end
    cmax = 0.0
    @inbounds for k in 2:Nz, j in 1:Nc, i in 1:Nc
        donor = min(m[Hp + i, Hp + j, k - 1], m[Hp + i, Hp + j, k])
        cmax = max(cmax, abs(cm[i, j, k]) / donor)
    end
    return r, cmax
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
Courant numbers do not depend on the grid size beyond those bounds.
"""
function fill_synthetic_fluxes!(am, bm, cm, panels_m, Nc::Integer, Hp::Integer)
    Nz = size(panels_m[1], 3)
    gyre_fraction = 0.0
    divergent_fraction = 0.0
    vertical_fraction = 0.0
    for p in 1:6
        am_s, bm_s, am_d, bm_d, cm_d = _unit_panel_fluxes(p, Nc, Nz)
        m = panels_m[p]
        gyre_fraction = max(gyre_fraction, first(_max_face_fraction(am_s, bm_s, cm_d, m, Nc, Hp, Nz)))
        rd, rc = _max_face_fraction(am_d, bm_d, cm_d, m, Nc, Hp, Nz)
        divergent_fraction = max(divergent_fraction, rd)
        vertical_fraction = max(vertical_fraction, rc)
    end
    gyre_scale = SYNTHETIC_GYRE_FLUX_FRACTION / gyre_fraction
    divergent_scale = SYNTHETIC_DIVERGENT_FLUX_FRACTION / max(divergent_fraction, vertical_fraction)
    interior = (Hp + 1):(Hp + Nc)
    faces = (Hp + 1):(Hp + Nc + 1)
    for p in 1:6
        am_s, bm_s, am_d, bm_d, cm_d = _unit_panel_fluxes(p, Nc, Nz)
        FT = eltype(am[p])
        fill!(am[p], zero(FT))
        fill!(bm[p], zero(FT))
        fill!(cm[p], zero(FT))
        am[p][faces, interior, :] .= FT.(gyre_scale .* am_s .+ divergent_scale .* am_d)
        bm[p][interior, faces, :] .= FT.(gyre_scale .* bm_s .+ divergent_scale .* bm_d)
        cm[p][interior, interior, :] .= FT.(divergent_scale .* cm_d)
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
