# Palindrome CFL subcycle count: the static outflow budget shared by every
# split-sweep topology, and its cubed-sphere evaluation.
# Split from CubedSphereStrang.jl (refactor phase 4); included by Advection.jl in this order.

# =========================================================================
# Palindrome outflow budget (all split-sweep topologies)
# =========================================================================
#
# One transport step applies every direction twice: X Y Z | midpoint | Z Y X
# for structured and cubed-sphere grids, H V | midpoint | V H on face-indexed
# grids. When the stored fluxes are too large for one palindrome, the runtime
# runs the subcycled composition
#
#     (X Y Z)^n  midpoint  (Z Y X)^n        every pass with flux / n
#
# (`strang_split_mt!`, `strang_split_cs_mt!`, the
# face-indexed `apply!`). Every topology takes `n` from the same budget: with
# `out_d = max(0, -F_lo) + max(0, F_hi)` the cell's outflow and
# `F_lo - F_hi` its net inflow through its two `d` faces for one full-flux
# application,
#
#     2 (out_x + out_y + out_z) / (n m_ref)  <=  cfl_limit,
#     m_ref = min(m, m_end),   m_end = m + 2 Σ_d (F_lo - F_hi),
#
# `m_end` being the cell's mass after the whole palindrome. This is the
# reference mass of the write-time positivity gate
# (`verify_substep_positivity_cs!`, `m_ref = min(m, m_next)`), with the step's
# own end mass in place of the window's.
#
# Guarantee. The midpoint operators do not change air mass, and every block
# of passes (one `X Y Z` or `Z Y X`) changes a cell's mass by the same
# `Σ_d (F_lo - F_hi) / n`, so the masses at block boundaries lie on the line
# from `m` to `m_end` and are all `>= m_ref`. Inside a block the passes remove
# at most `out / n <= cfl_limit m_ref / 2` in total. Hence every pass removes
# at most `cfl_limit / 2` of the mass the cell holds when the pass starts, and
# the air mass stays `>= (1 - cfl_limit / 2) m_ref > 0` throughout the step.
# Budgeting against the step-start mass alone (the earlier rule) does not
# bound a cell that drains within the step: its later blocks start well
# below `m`.
#
# A cell with outflow whose reference mass is not positive (`m <= 0`, or a
# flux set that would leave it with `m_end <= 0`) has no valid count; the
# count then throws instead of running.

"""
    _outflow(F_lo, F_hi)

Mass a cell loses through its two faces of one direction, `F_lo` the lower- and
`F_hi` the higher-index face flux (positive toward higher index):
`max(0, -F_lo) + max(0, F_hi)`. Both faces count at a divergent point (Lin &
Rood 1996).
"""
@inline _outflow(F_lo, F_hi) = max(zero(F_lo), -F_lo) + max(zero(F_hi), F_hi)

"""
    _net_inflow(F_lo, F_hi)

Mass a cell gains through its two faces of one direction for one application
(same convention as `_outflow`): `F_lo - F_hi`.
"""
@inline _net_inflow(F_lo, F_hi) = F_lo - F_hi

"""
    _palindrome_outflow_ratio(m, out, net_in)

Outflow of one palindrome relative to the cell's reference mass,
`2 out / min(m, m + 2 net_in)`, where `out` is the cell's outflow and `net_in`
its net inflow, each summed over all directions for one application of each.
Cells without outflow contribute zero; a cell with outflow whose reference
mass is not positive (or a non-finite flux) gives `Inf`, which
`_palindrome_subcycle_count` rejects.
"""
@inline function _palindrome_outflow_ratio(m, out, net_in)
    m_ref = min(m, m + (net_in + net_in))
    return ifelse(m_ref > zero(m_ref), (out + out) / m_ref,
                  ifelse(out == zero(out), zero(m_ref), oftype(m_ref, Inf)))
end

"""
    _palindrome_subcycle_count(max_ratio, cfl_limit, max_n_sub, topology) -> Int

Number of subcycled passes `n` of a palindrome whose largest outflow ratio
(`_palindrome_outflow_ratio`) is `max_ratio`: the smallest `n` with
`max_ratio / n <= cfl_limit`. `cfl_limit = Inf` disables subcycling. Throws
an `ArgumentError` for a non-finite `max_ratio` (invalid flux set) or when
`n` would exceed `max_n_sub`.
"""
function _palindrome_subcycle_count(max_ratio, cfl_limit, max_n_sub::Integer, topology)
    cfl_limit > zero(cfl_limit) || throw(ArgumentError(
        "$(topology) advection requires cfl_limit > 0, got $(cfl_limit)"))
    isinf(cfl_limit) && return 1
    isfinite(max_ratio) || throw(ArgumentError(
        "$(topology) palindrome subcycling: a cell with outflow has non-positive air " *
        "mass at the start or end of the step (min(m, m + 2 (in - out)) <= 0), or a " *
        "flux or mass is not finite (outflow ratio $(max_ratio)); the flux set is invalid"))
    max_ratio <= cfl_limit && return 1
    n_sub = ceil(Int, max_ratio / cfl_limit)
    n_sub <= max_n_sub || throw(ArgumentError(
        "$(topology) palindrome subcycling exceeded max_n_sub=$(max_n_sub) " *
        "(outflow ratio $(max_ratio), cfl_limit $(cfl_limit))"))
    return n_sub
end

# CFL subcycle counts of the structured (lat-lon) and face-indexed (reduced-Gaussian)
# palindromes: the shared palindrome outflow budget of cs_subcycling.jl evaluated on
# each topology's flux layout.
# Split from StrangSplitting.jl (refactor phase 4); included by Advection.jl in this order.

"""
    _structured_palindrome_subcycle_count(am, bm, cm, m, cfl_limit;
                                          max_n_sub = DEFAULT_MAX_SUBCYCLES) -> Int

Subcycle count `n` of one structured (lat-lon) transport step: the smallest `n`
with `2 (out_x + out_y + out_z) / (n m_ref) <= cfl_limit` in every cell,
`m_ref = min(m, m + 2 (in - out))`, the palindrome budget the cubed sphere uses
(`_palindrome_subcycle_count`).

For x, face flux `am[i, j, k]` lies between cells `i-1` and `i` (positive =
rightward), so cell `i` loses `max(0, -am[i]) + max(0, am[i+1])`; y and z are
analogous (positive `cm` = downward); its net inflow is `am[i] - am[i+1]`. One
reduction over device-resident views, so device arrays stay on the device; an
invalid flux set (a cell with outflow and `m_ref <= 0`) throws.
"""
function _structured_palindrome_subcycle_count(am::AbstractArray{FT,3}, bm::AbstractArray{FT,3},
                                               cm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                               cfl_limit::FT;
                                               max_n_sub::Int = DEFAULT_MAX_SUBCYCLES) where FT
    isinf(cfl_limit) && return 1
    Nx, Ny, Nz = size(m)
    max_ratio = mapreduce(max, m,
                          view(am, 1:Nx, :, :), view(am, 2:Nx+1, :, :),
                          view(bm, :, 1:Ny, :), view(bm, :, 2:Ny+1, :),
                          view(cm, :, :, 1:Nz), view(cm, :, :, 2:Nz+1);
                          init = zero(FT)) do mi, axl, axh, byl, byh, czl, czh
        _palindrome_outflow_ratio(mi, _outflow(axl, axh) + _outflow(byl, byh) + _outflow(czl, czh),
                                  _net_inflow(axl, axh) + _net_inflow(byl, byh) + _net_inflow(czl, czh))
    end
    return _palindrome_subcycle_count(max_ratio, cfl_limit, max_n_sub, "lat-lon")
end

"""
    _face_indexed_palindrome_subcycle_count(horizontal_flux, cm, m, mesh, cfl_limit;
                                            max_n_sub = DEFAULT_MAX_SUBCYCLES) -> Int

Subcycle count `n` of one face-indexed (reduced-Gaussian) transport step
`H V | midpoint | V H`: the palindrome budget `2 (out_h + out_v) / (n m_ref) <=
cfl_limit`, `m_ref = min(m, m + 2 (in - out))` (`_palindrome_subcycle_count`).
`out_h` sums the outflow through the cell's interior faces (boundary stubs
carry no runtime flux), `out_v` is the outflow through its top and bottom faces
(`cm[:, k]` is the top face of level `k`, positive = downward); the net inflow
sums the same faces with sign.

The face connectivity lives on the host, so device arrays are copied to the
host once per step for this count.
"""
function _face_indexed_palindrome_subcycle_count(horizontal_flux::AbstractArray{FT,2},
                                                 cm::AbstractArray{FT,2},
                                                 m::AbstractArray{FT,2},
                                                 mesh::AbstractHorizontalMesh,
                                                 cfl_limit::FT;
                                                 max_n_sub::Int = DEFAULT_MAX_SUBCYCLES) where FT
    isinf(cfl_limit) && return 1
    hflux = host_array(horizontal_flux)
    cm_h = host_array(cm)
    m_h = host_array(m)
    nc, Nz = size(m_h)
    outgoing = zeros(FT, nc)
    net_in = zeros(FT, nc)
    max_ratio = zero(FT)
    @inbounds for k in 1:Nz
        fill!(outgoing, zero(FT))
        fill!(net_in, zero(FT))
        for f in 1:nfaces(mesh)
            left, right = face_cells(mesh, f)
            if left > 0 && right > 0
                flux = hflux[f, k]
                if flux >= zero(FT)
                    outgoing[left] += flux
                else
                    outgoing[right] -= flux
                end
                net_in[left] -= flux
                net_in[right] += flux
            end
        end
        for c in 1:nc
            out = outgoing[c] + _outflow(cm_h[c, k], cm_h[c, k + 1])
            net = net_in[c] + _net_inflow(cm_h[c, k], cm_h[c, k + 1])
            max_ratio = max(max_ratio, _palindrome_outflow_ratio(m_h[c, k], out, net))
        end
    end
    return _palindrome_subcycle_count(max_ratio, cfl_limit, max_n_sub, "face-indexed")
end

# Host view of an array for the connectivity loop: no copy for host arrays.
