# GEOS native CS preprocessing: DELP ↔ air mass, surface pressure, pressure-fixer mass evolution, residual smoothing.
# Split from cubed_sphere_geos.jl (refactor phase 4); included by Preprocessing.jl in this order.

"""
    _delp_pa_to_air_mass_kg!(m_kg, m_pa, cell_areas, inv_g) -> m_kg

In-place: convert pressure thickness in Pa to cell air mass in kg per
`m_kg[i, j, k] = m_pa[i, j, k] × cell_areas[i, j] × inv_g`. Cell areas are
in m² and apply identically to every CS panel by symmetry.
"""
function _delp_pa_to_air_mass_kg!(m_kg::AbstractArray{FT, 3},
                                  m_pa::AbstractArray{FT, 3},
                                  cell_areas::AbstractMatrix{FT},
                                  inv_g::FT) where {FT}
    Nx, Ny, Nz = size(m_kg)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m_kg[i, j, k] = m_pa[i, j, k] * cell_areas[i, j] * inv_g
    end
    return m_kg
end

"""
    _ps_from_air_mass!(ps, m, area, g, Nc, Nz)

Set `ps[i,j] = (Σ_k m[i,j,k]) · g / area[i,j]` (Pa). Used to keep the
binary's stored `ps` consistent with the chained pressure-fixer mass.
"""
function _ps_from_air_mass!(ps::AbstractMatrix{FT},
                            m::AbstractArray{FT, 3},
                            cell_areas::AbstractMatrix{FT},
                            g::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for k in 1:Nz
            s += m[i, j, k]
        end
        ps[i, j] = s * g / cell_areas[i, j]
    end
    return ps
end

"""
    _evolve_mass_pressure_fixer!(m_next, m_cur, am_v4, bm_v4, ΔB, two_steps, Nc, Nz)

FV3 pressure-fixer mass evolution (restored from commit e648bf3f):

    pit       = Σ_k (am[i,j,k] − am[i+1,j,k] + bm[i,j,k] − bm[i,j+1,k])
    m_next[k] = m_cur[k] + two_steps · ΔB[k] · pit

This is the endpoint implied by `compute_cs_cm_pressure_fixer!`'s closure
`cm[k+1]−cm[k] = C_k − ΔB[k]·pit`, so the window replay closes to roundoff.
The `two_steps` factor cancels the per-window flux scaling (`am ∝ 1/steps`), so
`m_next` is independent of the chosen substep count. Globally mass-conserving
(Σ_cells pit = 0 on the closed sphere), so the dry-mass pin is a no-op here.
"""
function _evolve_mass_pressure_fixer!(
        m_next::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        m_cur::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        am_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        bm_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        ΔB::AbstractVector,
        two_steps::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for p in 1:CS_PANEL_COUNT
        am = am_v4[p]; bm = bm_v4[p]
        m  = m_cur[p]; mn = m_next[p]
        for j in 1:Nc, i in 1:Nc
            pit = zero(FT)
            for k in 1:Nz
                pit += (am[i, j, k] - am[i + 1, j, k]) +
                       (bm[i, j, k] - bm[i, j + 1, k])
            end
            for k in 1:Nz
                mn[i, j, k] = m[i, j, k] + two_steps * FT(ΔB[k]) * pit
            end
        end
    end
    return nothing
end

"""
    _smooth_cs_residual_panels!(field, niter, w, Nc, Nz)

In-place horizontal Jacobi smoothing of the moisture-source residual, applied
INDEPENDENTLY and IDENTICALLY to each vertical level of every panel. The stencil
is the 4-neighbour interior average with weight `w`; panel-edge cells average
only their in-panel neighbours (no cross-panel exchange — the SH-UTLS fingering
lives in panel interiors, and a missing seam neighbour leaves the gate identity
untouched). Because the operator is LINEAR and LEVEL-INDEPENDENT,
`Σ_k smooth(rₖ) = smooth(Σ_k rₖ)`; with a zero column-integral residual this is
`smooth(0) = 0`, so the column closure (and thus surface pressure) is preserved
exactly while only the grid-scale per-layer structure is damped.
"""
function _smooth_cs_residual_panels!(field::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                     niter::Int, w::FT, Nc::Int, Nz::Int) where FT
    niter <= 0 && return nothing
    scratch = Array{FT}(undef, Nc, Nc)
    @inbounds for p in 1:CS_PANEL_COUNT
        f = field[p]
        for k in 1:Nz
            for _ in 1:niter
                for j in 1:Nc, i in 1:Nc
                    scratch[i, j] = f[i, j, k]
                end
                for j in 1:Nc, i in 1:Nc
                    s = zero(FT); n = 0
                    i > 1  && (s += scratch[i - 1, j]; n += 1)
                    i < Nc && (s += scratch[i + 1, j]; n += 1)
                    j > 1  && (s += scratch[i, j - 1]; n += 1)
                    j < Nc && (s += scratch[i, j + 1]; n += 1)
                    f[i, j, k] = (one(FT) - w) * scratch[i, j] + w * (s / FT(n))
                end
            end
        end
    end
    return nothing
end

"""
    _smooth_cs_columns!(field, niter, w, Nc)

In-place horizontal Jacobi low-pass of a per-panel 2D column field (e.g. the
column dry-mass drift), GLOBAL-SUM-PRESERVING (a uniform offset restores the
total after smoothing, so a zero-sum input stays zero-sum and the global dry
mass lands exactly on the analyzed target). Per-panel (no cross-panel exchange);
keeps the LARGE-SCALE part of the field and damps grid scales.
"""
function _smooth_cs_columns!(field::NTuple{CS_PANEL_COUNT, Array{FT, 2}},
                             niter::Int, w::FT, Nc::Int) where FT
    niter <= 0 && return nothing
    total_before = 0.0
    @inbounds for p in 1:CS_PANEL_COUNT, v in field[p]
        total_before += Float64(v)
    end
    scratch = Array{FT}(undef, Nc, Nc)
    @inbounds for p in 1:CS_PANEL_COUNT
        f = field[p]
        for _ in 1:niter
            copyto!(scratch, f)
            for j in 1:Nc, i in 1:Nc
                s = zero(FT); n = 0
                i > 1  && (s += scratch[i - 1, j]; n += 1)
                i < Nc && (s += scratch[i + 1, j]; n += 1)
                j > 1  && (s += scratch[i, j - 1]; n += 1)
                j < Nc && (s += scratch[i, j + 1]; n += 1)
                f[i, j] = (one(FT) - w) * scratch[i, j] + w * (s / FT(n))
            end
        end
    end
    total_after = 0.0
    @inbounds for p in 1:CS_PANEL_COUNT, v in field[p]
        total_after += Float64(v)
    end
    offset = FT((total_before - total_after) / (CS_PANEL_COUNT * Nc * Nc))
    @inbounds for p in 1:CS_PANEL_COUNT, idx in eachindex(field[p])
        field[p][idx] += offset
    end
    return nothing
end
