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
    hflux = _host_array(horizontal_flux)
    cm_h = _host_array(cm)
    m_h = _host_array(m)
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
_host_array(a::Array) = a
_host_array(a::AbstractArray) = Array(a)
