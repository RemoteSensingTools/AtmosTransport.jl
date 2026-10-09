# CS preprocessing: native CS mass fluxes → v4 face-staggered fluxes, FV3-style pressure-fixer cm.
# Split from cs_transport_helpers.jl (refactor phase 4); included by Preprocessing.jl in this order.

# ---------------------------------------------------------------------------
# Native CS source mass flux → v4 face-staggered (with panel halo sync).
# ---------------------------------------------------------------------------

"""
    geos_native_to_face_flux!(am_v4, bm_v4, am_native, bm_native, conn, Nc, Nz, scale)

Convert GCHP-convention cell-centered mass fluxes to v4 face-staggered
arrays.

GCHP semantic: `MFXC[i, j, k]` is the eastward mass flux at the **east face**
of cell `(i, j, k)`. The v4 convention has `am[i, j, k]` as the flux through
face index `i` (where `i=1` is the west boundary, `i=Nc+1` is the east
boundary, and `i=2..Nc` are interior faces between cells `i-1` and `i`). The
mapping is therefore:

    am_v4[i+1, j, k] = MFXC[i, j, k] * scale     for i = 1..Nc

Likewise for MFYC → bm. The west halo `am_v4[1, :, :]` and south halo
`bm_v4[:, 1, :]` come from the corresponding neighbor panel's NORTH or
EAST canonical (the same physical face), with a sign flip when both edges
sit at outflow. This is a *one-way* propagation: we never overwrite a
canonical (Nc+1) face. (`sync_all_cs_boundary_mirrors!` is bidirectional
and would clobber correctly-filled canonicals when paired with a
zero-initialized halo on the partner panel.)

`scale` is multiplied into every face value (typically `dt_factor / g` for
unit conversion to the v4 binary's `kg per substep`).
"""
function geos_native_to_face_flux!(
        am_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        bm_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        am_native::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        bm_native::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        conn::PanelConnectivity, Nc::Int, Nz::Int, scale::FT) where {FT}
    # 1. Interior + outflow canonicals: fill am[2..Nc+1, j, k] = MFXC[1..Nc, j, k]
    #    and bm[i, 2..Nc+1, k] = MFYC[i, 1..Nc, k]. West/south halos at index 1
    #    stay zero; they are filled in step 2.
    @inbounds for p in 1:CS_PANEL_COUNT
        ap, an = am_v4[p], am_native[p]
        bp, bn = bm_v4[p], bm_native[p]
        for k in 1:Nz, j in 1:Nc
            ap[1, j, k] = zero(FT)
            for i in 1:Nc
                ap[i + 1, j, k] = an[i, j, k] * scale
            end
        end
        for k in 1:Nz, i in 1:Nc
            bp[i, 1, k] = zero(FT)
            for j in 1:Nc
                bp[i, j + 1, k] = bn[i, j, k] * scale
            end
        end
    end
    # 2. Pull west/south halos from each panel's neighbor canonical.
    _propagate_cs_outflow_to_halo!(am_v4, bm_v4, conn, Nc, Nz)
    return nothing
end

"""
    _propagate_cs_outflow_to_halo!(am, bm, conn, Nc, Nz)

For every panel `p`, fill the WEST halo (`am[p][1, :, :]`) and SOUTH halo
(`bm[p][:, 1, :]`) from the same physical face on the neighbor panel.
Mirrors the geometry-aware face-location and sign logic of
`sync_all_cs_boundary_mirrors!` but in one direction only — never
overwrites a canonical face.

Assumption: every cross-panel boundary has at least one side at the
outflow boundary (NORTH or EAST). True for both gnomonic and GEOS-native
panel conventions.
"""
function _propagate_cs_outflow_to_halo!(
        am::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        bm::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        conn::PanelConnectivity, Nc::Int, Nz::Int) where {FT}
    @inbounds for p in 1:CS_PANEL_COUNT
        for e in (EDGE_WEST, EDGE_SOUTH)
            ne = conn.neighbors[p][e]
            q  = ne.panel
            ori = ne.orientation
            eq  = reciprocal_edge(conn, p, e)
            for s in 1:Nc
                t = ori == 0 ? s : Nc + 1 - s
                can_dir, can_i, can_j = _cs_edge_face_location(eq, t, Nc)
                mir_dir, mir_i, mir_j = _cs_edge_face_location(e, s, Nc)
                can_at_outflow = (can_dir == 1 && can_i == Nc + 1) ||
                                 (can_dir == 2 && can_j == Nc + 1)
                mir_at_outflow = (mir_dir == 1 && mir_i == Nc + 1) ||
                                 (mir_dir == 2 && mir_j == Nc + 1)
                msign = (can_at_outflow == mir_at_outflow) ? FT(-1) : FT(1)
                for k in 1:Nz
                    canonical = can_dir == 1 ? am[q][can_i, can_j, k] :
                                               bm[q][can_i, can_j, k]
                    mirror_val = msign * canonical
                    if mir_dir == 1
                        am[p][mir_i, mir_j, k] = mirror_val
                    else
                        bm[p][mir_i, mir_j, k] = mirror_val
                    end
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# FV3-style pressure-fixer cm for native GEOS sources.
# ---------------------------------------------------------------------------

"""
    compute_cs_cm_pressure_fixer!(cm_panels, am_v4, bm_v4, ΔB, Nc, Nz)

Diagnose the vertical mass flux `cm` from the column horizontal flux
convergence using FV3's pressure-fixer rule:

    pit[i,j]      = Σ_k (am[i,j,k] − am[i+1,j,k] + bm[i,j,k] − bm[i,j+1,k])
    cm[i,j,1]     = 0
    cm[i,j,k+1]   = cm[i,j,k] + (am[i,j,k] − am[i+1,j,k] + bm[i,j,k] − bm[i,j+1,k])
                              − ΔB[k] · pit[i,j]

Σ ΔB[k] = 1 by construction of a hybrid sigma-pressure coordinate, so
`cm[Nz+1] = 0` exactly: the surface no-flux boundary condition is enforced
without any per-cell residual redistribution. This is the same formulation
used by the historical GEOS-FP CS forward runner (commit `76fa489`,
`compute_cm_panel_cpu!`) and matches GFDL FV3's discrete continuity.

Use this on native-source paths (GEOS-IT, GEOS-FP) where dry-mass
horizontal flux divergence is NOT closed by `m_next − m_cur` per cell
(vertical moisture transport contributes). For sources where the closure
DOES hold by construction (Poisson-balanced spectral → CS), use
`diagnose_cs_cm!` instead — it derives `cm` from `(am, bm, dm)` directly.
"""
function compute_cs_cm_pressure_fixer!(cm_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       am_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       bm_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       ΔB::AbstractVector,
                                       Nc::Int, Nz::Int) where {FT}
    @assert length(ΔB) == Nz "ΔB must have one entry per layer (length=$(Nz))"
    @inbounds for p in 1:CS_PANEL_COUNT
        am = am_v4[p]; bm = bm_v4[p]; cm = cm_panels[p]
        for j in 1:Nc, i in 1:Nc
            pit = zero(FT)
            for k in 1:Nz
                pit += (am[i, j, k] - am[i + 1, j, k]) +
                       (bm[i, j, k] - bm[i, j + 1, k])
            end
            cm[i, j, 1] = zero(FT)
            acc = zero(FT)
            for k in 1:Nz
                conv_k = (am[i, j, k] - am[i + 1, j, k]) +
                         (bm[i, j, k] - bm[i, j + 1, k])
                acc += conv_k - FT(ΔB[k]) * pit
                cm[i, j, k + 1] = acc
            end
        end
    end
    return nothing
end
