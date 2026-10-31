# Shared rectangular-block palindrome positivity scan for LL and CS.

"""
    verify_substep_positivity_cs!(m, am, bm, cm; cfl_limit = DEFAULT_POSITIVITY_CFL_LIMIT,
                                  m_next = nothing)

Verify the per-substep horizontal+vertical positivity contract that the runtime's
`_cs_static_palindrome_subcycle_count` depends on. For every interior cell on every panel
(any rectangular block; the lat-lon gate `verify_substep_positivity_ll!` passes
its window as one block):

  1. The cell air mass itself must be positive (`m > 0`), a preprocessing
     invariant enforced here (the runtime CFL scan skips such cells). A
     non-positive cell mass is reported with `ratio = Inf` regardless of flux
     magnitude.
  2. The combined Strang-palindrome outgoing budget
     `2 * (out_x + out_y + out_z)` must not exceed `cfl_limit * m_ref`, where
     `m_ref = min(m, m_next)` when `m_next` is supplied by a caller that wants
     endpoint tightening, and `m_ref = m` otherwise. The factor of 2 is
     required because the CS runtime applies the
     direction sequence `X-Y-Z-Z-Y-X` for every met substep.

Returns a NamedTuple `(direction, ratio, ratio_xy, ratio_z, location, ok)`:
* `direction :: Union{Symbol, Nothing}` — the dominant contributor among
  `:x`, `:y`, `:z`, or `nothing` when no cell was inspected.
* `ratio :: Float64` — worst observed palindrome outgoing budget over the
  reference mass, or `Inf` if any inspected cell had invalid mass or flux.
* `ratio_xy :: Float64` — worst observed horizontal-only palindrome outgoing
  budget over the reference mass.
* `ratio_z :: Float64` — worst observed vertical-only palindrome outgoing
  budget over the reference mass.
* `location :: NTuple{4, Int}` — `(panel, i, j, k)` of the worst cell.
* `ok :: Bool` — `true` iff `ratio <= cfl_limit`.

The replay gate (`verify_write_replay_cs!`) only checks endpoint continuity. A
binary that drives a cell mass negative mid-sweep can still pass replay because
the cell re-fills from inflow before the window ends — but the runtime cannot
recover. This gate is the actual contract the runtime depends on.

Panel arrays are stored without halos at preprocessing time.
"""
function verify_substep_positivity_cs!(m::NTuple{NP, <:AbstractArray{FT, 3}},
                                       am::NTuple{NP, <:AbstractArray},
                                       bm::NTuple{NP, <:AbstractArray},
                                       cm::NTuple{NP, <:AbstractArray};
                                       cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                                       m_next::Union{Nothing, NTuple{NP, <:AbstractArray}} = nothing) where {FT, NP}
    validate_cfl_limit(cfl_limit, "verify_substep_positivity_cs!")
    # All buffers share the same interior extent `(Ni, Nj, Nz)`; derive from m.
    Ni = size(m[1], 1)
    Nj = size(m[1], 2)
    Nz = size(m[1], 3)
    iL, jL = 1, 1
    iH, jH = Ni, Nj
    worst_dir = nothing
    worst_ratio = 0.0
    worst_ratio_xy = 0.0
    worst_ratio_z = 0.0
    worst_loc = (0, 0, 0, 0)
    for p in 1:NP
        m_p = m[p]
        m_next_p = m_next === nothing ? nothing : m_next[p]
        m_int = view(m_p, iL:iH, jL:jH, 1:Nz)
        m_next_int = m_next_p === nothing ? nothing :
                     view(m_next_p, iL:iH, jL:jH, 1:Nz)
        am_lo = view(am[p], iL    :iH,     jL:jH,     1:Nz)
        am_hi = view(am[p], iL + 1:iH + 1, jL:jH,     1:Nz)
        bm_lo = view(bm[p], iL:iH,     jL    :jH,     1:Nz)
        bm_hi = view(bm[p], iL:iH,     jL + 1:jH + 1, 1:Nz)
        cm_lo = view(cm[p], iL:iH, jL:jH, 1    :Nz)
        cm_hi = view(cm[p], iL:iH, jL:jH, 2:Nz + 1)
        for k in 1:Nz, j in 1:Nj, i in 1:Ni
            mi = m_int[i, j, k]
            mi_next = m_next_int === nothing ? mi : m_next_int[i, j, k]
            ax_l = am_lo[i, j, k]
            ax_h = am_hi[i, j, k]
            by_l = bm_lo[i, j, k]
            by_h = bm_hi[i, j, k]
            cz_l = cm_lo[i, j, k]
            cz_h = cm_hi[i, j, k]
            if !isfinite(mi) || mi <= zero(FT) ||
               !isfinite(mi_next) || mi_next <= zero(FT) ||
               !isfinite(ax_l) || !isfinite(ax_h) ||
               !isfinite(by_l) || !isfinite(by_h) ||
               !isfinite(cz_l) || !isfinite(cz_h)
                if !isinf(worst_ratio)
                    worst_ratio = Inf
                    worst_ratio_xy = Inf
                    worst_ratio_z = Inf
                    worst_dir = :x
                    worst_loc = (p, i, j, k)
                end
                continue
            end
            out_x = max(zero(FT), -ax_l) + max(zero(FT), ax_h)
            out_y = max(zero(FT), -by_l) + max(zero(FT), by_h)
            out_z = max(zero(FT), -cz_l) + max(zero(FT), cz_h)
            m_ref = min(mi, mi_next)
            outgoing_xy = FT(2) * (out_x + out_y)
            outgoing_z = FT(2) * out_z
            outgoing = outgoing_xy + outgoing_z
            ratio_xy = outgoing_xy / m_ref
            ratio_z = outgoing_z / m_ref
            ratio = outgoing / m_ref
            worst_ratio_xy = max(worst_ratio_xy, Float64(ratio_xy))
            worst_ratio_z = max(worst_ratio_z, Float64(ratio_z))
            if ratio > worst_ratio
                worst_ratio = ratio
                worst_dir = out_x >= out_y && out_x >= out_z ? :x :
                            out_y >= out_z ? :y : :z
                worst_loc = (p, i, j, k)
            end
        end
    end
    return (direction = worst_dir, ratio = Float64(worst_ratio),
            ratio_xy = Float64(worst_ratio_xy),
            ratio_z = Float64(worst_ratio_z),
            location = worst_loc, ok = worst_ratio <= cfl_limit)
end
