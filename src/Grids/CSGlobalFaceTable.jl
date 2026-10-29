# Cubed-sphere Poisson balance: global cell and face indexing (the face table).
# Owned by Grids; preprocessing consumes the exported face topology.

# ---------------------------------------------------------------------------
# Global cell and face indexing
# ---------------------------------------------------------------------------

"""
    CSGlobalFaceTable

Global face table for a 6-panel cubed sphere.

# Fields
- `face_left  :: Vector{Int32}` — global cell index on the "left" side of each face
- `face_right :: Vector{Int32}` — global cell index on the "right" side of each face
- `nf :: Int` — total number of faces (= 12 × Nc²)
- `nc :: Int` — total number of cells (= 6 × Nc²)
- `Nc :: Int` — cells per panel edge

## Per-face back-mapping to per-panel am/bm arrays

- `face_panel :: Vector{Int32}` — which panel owns this face (1-6)
- `face_dir   :: Vector{Int32}` — direction: 1 = x-face (am), 2 = y-face (bm)
- `face_idx_i :: Vector{Int32}` — the `i` index into `am[i, j, k]` or `bm[i, j, k]`
- `face_idx_j :: Vector{Int32}` — the `j` index

For cross-panel faces, there's also a **mirror** entry on the neighbor panel:

- `mirror_panel :: Vector{Int32}` — 0 for interior faces; neighbor panel for cross-panel
- `mirror_dir   :: Vector{Int32}` — direction of the mirror entry
- `mirror_idx_i :: Vector{Int32}` — the `i` index of the mirror entry
- `mirror_idx_j :: Vector{Int32}` — the `j` index of the mirror entry

Sign convention: flux positive → mass flows from `face_left` to `face_right`.
"""
struct CSGlobalFaceTable
    face_left    :: Vector{Int32}
    face_right   :: Vector{Int32}
    face_panel   :: Vector{Int32}
    face_dir     :: Vector{Int32}
    face_idx_i   :: Vector{Int32}
    face_idx_j   :: Vector{Int32}
    mirror_panel :: Vector{Int32}
    mirror_dir   :: Vector{Int32}
    mirror_idx_i :: Vector{Int32}
    mirror_idx_j :: Vector{Int32}
    mirror_sign  :: Vector{Int32}   # +1 or -1: sign flip when writing mirror entry
    nf :: Int
    nc :: Int
    Nc :: Int
end

"""
    build_cs_global_face_table(Nc, conn) -> CSGlobalFaceTable

Build the global face table for a C`Nc` cubed sphere with panel connectivity
`conn`. Enumerates all `12Nc²` unique faces (interior + cross-panel).

Cross-panel faces are created only from outgoing edges (north, east) to avoid
double-counting. The canonical entry is on the outgoing panel, the mirror on
the incoming panel.
"""
function build_cs_global_face_table(Nc::Int, conn::PanelConnectivity)
    max_nf = 12 * Nc^2
    fl = Vector{Int32}(undef, max_nf)
    fr = Vector{Int32}(undef, max_nf)
    fp = Vector{Int32}(undef, max_nf)
    fd = Vector{Int32}(undef, max_nf)
    fi = Vector{Int32}(undef, max_nf)
    fj = Vector{Int32}(undef, max_nf)
    mp = zeros(Int32, max_nf)
    md = zeros(Int32, max_nf)
    mi = zeros(Int32, max_nf)
    mj = zeros(Int32, max_nf)
    ms = ones(Int32, max_nf)    # mirror_sign: default +1

    nf = 0

    # --- Phase 1: Interior faces (within each panel) ---
    for p in 1:6
        # Interior x-faces: am[i, j] for i ∈ 2:Nc, j ∈ 1:Nc
        for j in 1:Nc, i in 2:Nc
            nf += 1
            fl[nf] = cs_flat_index(i - 1, j, p, Nc)
            fr[nf] = cs_flat_index(i,     j, p, Nc)
            fp[nf] = Int32(p)
            fd[nf] = Int32(1)   # x-face → am
            fi[nf] = Int32(i)
            fj[nf] = Int32(j)
        end
        # Interior y-faces: bm[i, j] for i ∈ 1:Nc, j ∈ 2:Nc
        for j in 2:Nc, i in 1:Nc
            nf += 1
            fl[nf] = cs_flat_index(i, j - 1, p, Nc)
            fr[nf] = cs_flat_index(i, j,     p, Nc)
            fp[nf] = Int32(p)
            fd[nf] = Int32(2)   # y-face → bm
            fi[nf] = Int32(i)
            fj[nf] = Int32(j)
        end
    end

    n_interior = nf

    # --- Phase 2: Cross-panel faces ---
    # Iterate ALL 4 edges of each panel.  De-duplicate by keeping only the
    # (p, e) side where p < q (the lower-numbered panel is canonical).
    # This correctly enumerates all 12 physical edges regardless of which
    # combination of NORTH/SOUTH/EAST/WEST they land on.
    for p in 1:6
        for e in (EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST)
            q   = conn.neighbors[p][e].panel
            ori = conn.neighbors[p][e].orientation
            eq  = reciprocal_edge(conn, p, e)

            # De-duplicate: each physical edge is shared by (p, q).
            # Keep only the p < q side.
            p > q && continue

            for s in 1:Nc
                t = ori == 0 ? s : Nc + 1 - s

                cp = cs_edge_cell(e, s, Nc)
                cq = cs_edge_cell(eq, t, Nc)

                gp = cs_flat_index(cp[1], cp[2], p, Nc)
                gq = cs_flat_index(cq[1], cq[2], q, Nc)

                can_dir, can_i, can_j = cs_edge_face_location(e, s, Nc)
                mir_dir, mir_i, mir_j = cs_edge_face_location(eq, t, Nc)

                # Face-table convention: fl = cell on the lower-index side of
                # the face, fr = cell on the higher-index side.
                # For NORTH/EAST canonicals (outflow at Nc+1): fl = P-cell (inside),
                #   fr = Q-cell (outside).
                # For SOUTH/WEST canonicals (inflow at 1): fl = Q-cell (outside),
                #   fr = P-cell (inside) — swap because the P-cell is on the
                #   higher-index side of the boundary face.
                can_at_outflow = (can_dir == 1 && can_i == Nc + 1) ||
                                 (can_dir == 2 && can_j == Nc + 1)

                nf += 1
                if can_at_outflow
                    fl[nf] = gp
                    fr[nf] = gq
                else
                    fl[nf] = gq
                    fr[nf] = gp
                end
                fp[nf] = Int32(p)
                fd[nf] = Int32(can_dir)
                fi[nf] = Int32(can_i)
                fj[nf] = Int32(can_j)
                mp[nf] = Int32(q)
                md[nf] = Int32(mir_dir)
                mi[nf] = Int32(mir_i)
                mj[nf] = Int32(mir_j)

                # Mirror sign: determines whether the mirror value must be
                # negated so that per-panel flux telescoping conserves mass.
                #
                # The advection kernel computes:
                #   m_new = m + flux_in(j=1 or i=1) - flux_out(j=Nc+1 or i=Nc+1)
                # For global mass conservation, the canonical and mirror must
                # cancel.  When both are at the SAME position type (both inflow
                # or both outflow), negate; when at OPPOSITE types, keep as-is.
                ms[nf] = seam_mirror_sign(e, eq, Int32)
            end
        end
    end

    n_cross = nf - n_interior
    nc = 6 * Nc^2

    @assert nf == 12 * Nc^2 "Expected $(12*Nc^2) faces, got $nf " *
        "(interior=$n_interior, cross=$n_cross)"

    return CSGlobalFaceTable(
        fl[1:nf], fr[1:nf], fp[1:nf], fd[1:nf], fi[1:nf], fj[1:nf],
        mp[1:nf], md[1:nf], mi[1:nf], mj[1:nf], ms[1:nf],
        nf, nc, Nc,
    )
end

export CSGlobalFaceTable, build_cs_global_face_table
