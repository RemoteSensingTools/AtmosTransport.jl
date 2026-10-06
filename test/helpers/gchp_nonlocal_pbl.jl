# Synthetic cubed-sphere column forcing for the GEOS-Chem non-local PBL tests.
# A column with a 1.5 km boundary layer: 6.5 K/km lapse rate up to a 210 K
# tropopause, moist near the surface, wind shear only below 3 km.

using AtmosTransport.State: GCHPNonlocalPBLField, refresh_gchp_nonlocal_pbl!

const NONLOCAL_NC, NONLOCAL_NZ, NONLOCAL_HP = 2, 40, 1

function synthetic_pbl_window(; hflux, eflux = 100.0, pblh = 1500.0, area = 1e10,
                              Nc = NONLOCAL_NC, Nz = NONLOCAL_NZ, Hp = NONLOCAL_HP)
    g = 9.80665
    p_edges = collect(range(1.0, 1e5; length = Nz + 1))           # top-down [Pa]
    p_mid = (p_edges[1:end-1] .+ p_edges[2:end]) ./ 2
    z_mid = @. -7000 * log(p_mid / 1e5)                         # rough heights
    T = @. max(295 - 0.0065z_mid, 210.0)
    qv = @. 0.012 * exp(-z_mid / 2000)
    u = @. ifelse(z_mid < 3000, 10 * z_mid / 3000, 10.0)
    m_dry = @. (p_edges[2:end] - p_edges[1:end-1]) * area / g * (1 - qv)
    column(x) = ntuple(_ -> repeat(reshape(x, 1, 1, Nz), Nc, Nc, 1), 6)
    padded(x) = ntuple(_ -> repeat(reshape(x, 1, 1, Nz), Nc + 2Hp, Nc + 2Hp, 1), 6)
    surface(x) = ntuple(_ -> fill(x, Nc, Nc), 6)
    sfc = (pblh = surface(pblh), ustar = surface(0.4), hflux = surface(hflux),
           t2m = surface(T[end]), eflux = surface(eflux))
    vdiff = (u = column(u), v = column(zero(u)), t = column(T), qv = column(qv))
    vertical = (A = p_edges, B = zero(p_edges))                     # pure pressure levels
    return sfc, vdiff, padded(m_dry), fill(area, Nc, Nc), (; p_mid, z_mid, T, qv, vertical)
end

# Convert every array in a (nested) tuple of panels with `f` (element type or array type).
convert_panels(f, x::Union{Tuple, NamedTuple}) = map(y -> convert_panels(f, y), x)
convert_panels(f, x::AbstractArray) = f(x)

function refreshed_pbl_field(::Type{T}, adapt = identity; kw...) where T
    sfc, vdiff, m, areas, col = synthetic_pbl_window(; kw...)
    to(x) = convert_panels(a -> adapt(T.(a)), x)
    field = adapt(GCHPNonlocalPBLField(NONLOCAL_NC, NONLOCAL_NZ, T))
    refresh_gchp_nonlocal_pbl!(field, to(sfc), to(vdiff), to(m), T.(areas), col.vertical;
                               halo_width = NONLOCAL_HP)
    return field, col
end
