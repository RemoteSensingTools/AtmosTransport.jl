# ---------------------------------------------------------------------------
# Vertical sweep with FV3's tracer profile (scalar_profile, kord = 8, iv = 0)
#
# GEOS-Chem High Performance remaps tracers vertically with the piecewise
# parabolic profile of FV3's `scalar_profile` (`fv_mapz.F90`, `kord_tr = 8`).
# Here the same profile drives an Eulerian flux-form vertical sweep:
#
#   rm_k ← rm_k + f_{k−½} − f_{k+½},   f = F · q̄_swept,
#
# where F is the air-mass flux through the interface (positive downward) and
# q̄_swept the mean of the donor layer's parabola over the swept fraction
# α = |F| / m_donor. Each interface flux is evaluated once per column and used
# for both adjacent layers, so the column tracer mass telescopes exactly.
#
# Vertical index: k = 1 is the model top, k = Nz the surface; interface e lies
# above layer e. Within a layer, s ∈ [0, 1] runs from the top (q_L) to the
# bottom (q_R) edge, and the parabola is
#
#   q(s) = q_L + s [(q_R − q_L) + q_6 (1 − s)],   q_6 = 3 (2 q̄ − q_L − q_R).
#
# The arithmetic follows `fv_mapz.F90` line by line (S.-J. Lin, NOAA/GFDL),
# including its operation order, so the profile matches FV3 to rounding.
# `FV3ScalarProfile{true}` is FV3's positive-definite profile (`iv = 0`, used
# by GCHP for all tracers); `FV3ScalarProfile{false}` is its profile for signed
# fields (`iv = 1`), which omits the three non-negativity steps and is
# symmetric under q → −q except where neighbouring layer means are exactly
# equal (FV3 resolves those ties with its local-minimum branch).
# ---------------------------------------------------------------------------

@inline _layer_mass(m, i, j, k) = max(m[i, j, k], eps(eltype(m)))

# The positive-definite steps of `iv = 0`; the signed profile skips them.
@inline _nonnegative(::FV3ScalarProfile{true}, q) = max(zero(q), q)
@inline _nonnegative(::FV3ScalarProfile{false}, q) = q

"""
    FV3Column(scratch, ii, jj, t)

Working storage of the FV3 profile for interior column `(ii, jj)` and tracer
`t`. Each tracer owns three slices of `scratch` (size `Nc × Nc × (Nz+1) × 3Nt`):
the edge value at interface `k`, the elimination coefficient of the tridiagonal
solve, and the mean mixing ratio of layer `k`.
"""
struct FV3Column{A}
    scratch :: A
    ii :: Int
    jj :: Int
    t  :: Int
end
Base.@propagate_inbounds _edge(c::FV3Column, k) = c.scratch[c.ii, c.jj, k, 3c.t - 2]
Base.@propagate_inbounds _elim(c::FV3Column, k) = c.scratch[c.ii, c.jj, k, 3c.t - 1]
Base.@propagate_inbounds _qbar(c::FV3Column, k) = c.scratch[c.ii, c.jj, k, 3c.t]
Base.@propagate_inbounds _set_edge!(c::FV3Column, k, v) = (c.scratch[c.ii, c.jj, k, 3c.t - 2] = v)
Base.@propagate_inbounds _set_elim!(c::FV3Column, k, v) = (c.scratch[c.ii, c.jj, k, 3c.t - 1] = v)
Base.@propagate_inbounds _set_qbar!(c::FV3Column, k, v) = (c.scratch[c.ii, c.jj, k, 3c.t] = v)

"""
    _fv3_unlimited_edges!(col::FV3Column, rm, m, i, j, Nz)

Store the layer means `rm / m` of column `(i, j)` and solve FV3's compact
tridiagonal system for its edge values. The system is fourth-order accurate on
a non-uniform grid whose spacing is the layer air mass; the end rows impose the
boundary conditions of `scalar_profile` (iv ≠ −2).
"""
@inline function _fv3_unlimited_edges!(col::FV3Column, rm, m, i, j, Nz)
    FT = eltype(col.scratch)
    for k in 1:Nz
        _set_qbar!(col, k, rm[i, j, k] / _layer_mass(m, i, j, k))
    end

    grat = _layer_mass(m, i, j, 2) / _layer_mass(m, i, j, 1)
    bet  = grat * (grat + FT(0.5))
    _set_edge!(col, 1, ((grat + grat) * (grat + 1) * _qbar(col, 1) + _qbar(col, 2)) / bet)
    _set_elim!(col, 1, (1 + grat * (grat + FT(1.5))) / bet)

    d4 = zero(FT)
    for k in 2:Nz
        d4  = _layer_mass(m, i, j, k - 1) / _layer_mass(m, i, j, k)
        bet = 2 + d4 + d4 - _elim(col, k - 1)
        _set_edge!(col, k, (3 * (_qbar(col, k - 1) + d4 * _qbar(col, k)) - _edge(col, k - 1)) / bet)
        _set_elim!(col, k, d4 / bet)
    end

    a_bot = 1 + d4 * (d4 + FT(1.5))
    _set_edge!(col, Nz + 1,
               (2 * d4 * (d4 + 1) * _qbar(col, Nz) + _qbar(col, Nz - 1) - a_bot * _edge(col, Nz)) /
               (d4 * (d4 + FT(0.5)) - a_bot * _elim(col, Nz)))

    for k in Nz:-1:1
        _set_edge!(col, k, _edge(col, k) - _elim(col, k) * _edge(col, k + 1))
    end
    return nothing
end

"""
    _fv3_edge(col::FV3Column, e, Nz, profile) -> q_e

Edge value at interface `e` after FV3's large-scale constraints: an edge lies
between its two neighbouring layer means unless the edge sits at a local
extremum of the layer means, where it may overshoot on the extremum side only.
The positive-definite profile also keeps an edge at a local minimum
non-negative. The top (`e = 1`) and surface (`e = Nz + 1`) edges are not
constrained here.
"""
@inline function _fv3_edge(col::FV3Column, e, Nz, profile)
    q_e = _edge(col, e)
    (e == 1 || e == Nz + 1) && return q_e

    q_up, q_dn = _qbar(col, e - 1), _qbar(col, e)
    lo, hi = min(q_up, q_dn), max(q_up, q_dn)
    (e == 2 || e == Nz) && return max(min(q_e, hi), lo)

    δ_above = q_up - _qbar(col, e - 2)
    δ_below = _qbar(col, e + 1) - q_dn
    if δ_above * δ_below > 0
        return max(min(q_e, hi), lo)          # monotone: stay between the means
    elseif δ_above > 0
        return max(q_e, lo)                   # local maximum
    else
        return _nonnegative(profile, min(q_e, hi))   # local minimum
    end
end

"""
    _fv3_monotone_limit(q, q_L, q_R, q_6, flatten) -> (q_L, q_R, q_6)

Standard PPM monotonicity limiter (`cs_limiters`, iv = 1 and 2): a flagged
extremum is flattened to the layer mean; otherwise an edge is moved so the
parabola has no interior extremum.
"""
@inline function _fv3_monotone_limit(q, q_L, q_R, q_6, flatten::Bool)
    flatten && return q, q, zero(q)
    da1  = q_R - q_L
    da2  = da1^2
    a6da = q_6 * da1
    if a6da < -da2
        q_6 = 3 * (q_L - q)
        return q_L, q_L - q_6, q_6
    elseif a6da > da2
        q_6 = 3 * (q_R - q)
        return q_R - q_6, q_R, q_6
    end
    return q_L, q_R, q_6
end

"""
    _fv3_positive_limit(q, q_L, q_R, q_6) -> (q_L, q_R, q_6)

Positive-definite limiter (`cs_limiters`, iv = 0): if the parabola dips below
zero inside the layer, flatten it (both edges above the mean) or move one edge
so the minimum touches zero at the other edge.
"""
@inline function _fv3_positive_limit(q, q_L, q_R, q_6)
    FT = typeof(q)
    q <= 0 && return q, q, zero(FT)
    r12 = one(FT) / 12                       # FV3 multiplies by a rounded 1/12
    if abs(q_R - q_L) < -q_6 && q + FT(0.25) * (q_R - q_L)^2 / q_6 + q_6 * r12 < 0
        if q < q_R && q < q_L
            return q, q, zero(FT)
        elseif q_R > q_L
            q_6 = 3 * (q_L - q)
            return q_L, q_L - q_6, q_6
        else
            q_6 = 3 * (q_R - q)
            return q_R - q_6, q_R, q_6
        end
    end
    return q_L, q_R, q_6
end

"""
    _fv3_layer_profile(col::FV3Column, k, Nz, profile) -> (q_L, q_R, q_6)

Limited parabola of layer `k`, following `scalar_profile` (kord = 8; `iv = 0`
for the positive-definite `profile`, `iv = 1` for the signed one):

- k = 1 and k = Nz: the outer edge is made non-negative (positive-definite
  profile only), then the monotone limiter flattens the layer if its mean is
  not between its edges;
- k = 2 and k = Nz − 1: monotone limiter, flattened at a local extremum of the
  layer means;
- 3 ≤ k ≤ Nz − 2: Huynh's second constraint on both edges, then the
  positive-definite limiter (positive-definite profile only).
"""
@inline function _fv3_layer_profile(col::FV3Column, k, Nz, profile)
    FT  = eltype(col.scratch)
    q   = _qbar(col, k)
    q_L = _fv3_edge(col, k, Nz, profile)
    q_R = _fv3_edge(col, k + 1, Nz, profile)

    if k == 1 || k == Nz
        q_L = ifelse(k == 1, _nonnegative(profile, q_L), q_L)
        q_R = ifelse(k == Nz, _nonnegative(profile, q_R), q_R)
        q_6 = 3 * (2 * q - (q_L + q_R))
        return _fv3_monotone_limit(q, q_L, q_R, q_6, (q - q_L) * (q - q_R) >= 0)
    end

    δ_k  = q - _qbar(col, k - 1)
    δ_k1 = _qbar(col, k + 1) - q
    if k == 2 || k == Nz - 1
        q_6 = 3 * (2 * q - (q_L + q_R))
        return _fv3_monotone_limit(q, q_L, q_R, q_6, δ_k * δ_k1 < 0)
    end

    # Huynh's (1996) second constraint, as in fv_mapz.F90 for kord < 9
    δ_km1 = _qbar(col, k - 1) - _qbar(col, k - 2)
    δ_k2  = _qbar(col, k + 2) - _qbar(col, k + 1)
    pmp_1 = q - 2 * δ_k1
    lac_1 = pmp_1 + FT(1.5) * δ_k2
    q_L = min(max(q_L, min(q, pmp_1, lac_1)), max(q, pmp_1, lac_1))
    pmp_2 = q + 2 * δ_k
    lac_2 = pmp_2 - FT(1.5) * δ_km1
    q_R = min(max(q_R, min(q, pmp_2, lac_2)), max(q, pmp_2, lac_2))
    q_6 = 3 * (2 * q - (q_L + q_R))
    return _fv3_interior_limit(profile, q, q_L, q_R, q_6)
end

@inline _fv3_interior_limit(::FV3ScalarProfile{true}, q, q_L, q_R, q_6) =
    _fv3_positive_limit(q, q_L, q_R, q_6)
@inline _fv3_interior_limit(::FV3ScalarProfile{false}, q, q_L, q_R, q_6) = (q_L, q_R, q_6)

# Mean of the parabola over the bottom fraction α of the layer (s ∈ [1 − α, 1])
# and over the top fraction (s ∈ [0, α]).
@inline _parabola_bottom_mean((q_L, q_R, q_6), α) = q_R - α / 2 * ((q_R - q_L) - (1 - 2 * α / 3) * q_6)
@inline _parabola_top_mean((q_L, q_R, q_6), α)    = q_L + α / 2 * ((q_R - q_L) + (1 - 2 * α / 3) * q_6)

"""
    _fv3_interface_flux(F, above, m_above, below, m_below) -> tracer flux

Tracer mass through an interior interface with air-mass flux `F` (positive
downward): `F` times the mean of the donor profile over the swept fraction
`α = |F| / m_donor`. Beyond a Courant number of one (a CFL violation) the
flux is capped at the donor's whole tracer content, as in `_slopes_face_flux`.
"""
@inline function _fv3_interface_flux(F, above, m_above, below, m_below)
    if F >= 0
        F_donor = min(F, m_above)
        return F_donor * _parabola_bottom_mean(above, F_donor / m_above)
    else
        F_donor = max(F, -m_below)
        return F_donor * _parabola_top_mean(below, -F_donor / m_below)
    end
end

"""
Vertical sweep of one cubed-sphere panel with the FV3 profile; one work item
per column and tracer. The `t = 1` items also update the air mass, exactly as
every other sweep does. The surface and model-top interfaces carry no tracer
flux.
"""
@kernel function _cs_zsweep_fv3_kernel!(rm_out, @Const(rm), m_out, @Const(m), @Const(cm),
                                        scratch, profile, Nz, Hp, flux_scale)
    ii, jj, t = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        if t == 1
            for k in Int32(1):Nz
                cm_t = flux_scale * cm[i, j, k]
                cm_b = flux_scale * cm[i, j, k + Int32(1)]
                m_out[i, j, k] = m[i, j, k] + cm_t - cm_b
            end
        end
        col  = FV3Column(scratch, ii, jj, t)
        rm_t = TracerView(rm, Int32(t))
        _fv3_unlimited_edges!(col, rm_t, m, i, j, Nz)
        above = _fv3_layer_profile(col, Int32(1), Nz, profile)
        f_top = zero(eltype(m))
        for k in Int32(1):(Nz - Int32(1))
            below = _fv3_layer_profile(col, k + Int32(1), Nz, profile)
            f_bot = _fv3_interface_flux(flux_scale * cm[i, j, k + Int32(1)],
                                        above, _layer_mass(m, i, j, k),
                                        below, _layer_mass(m, i, j, k + Int32(1)))
            rm_out[i, j, k, t] = rm_t[i, j, k] + f_top - f_bot
            f_top, above = f_bot, below
        end
        rm_out[i, j, Nz, t] = rm_t[i, j, Nz] + f_top
    end
end

const _FV3VerticalPPM = PPMScheme{<:AbstractLimiter, <:FV3ScalarProfile}

"""
    needs_column_scratch(scheme) -> Bool

Whether `scheme`'s vertical sweep needs per-column working storage in
`CSAdvectionWorkspace` (`Nc × Nc × (Nz+1) × 3Nt`; see `FV3Column`).
"""
needs_column_scratch(::AbstractAdvectionScheme) = false
needs_column_scratch(::_FV3VerticalPPM) = true

function _check_column_scratch(scratch, Nc, Nz, Nt)
    size(scratch)[1:3] == (Nc, Nc, Nz + 1) && size(scratch, 4) >= 3Nt || throw(ArgumentError(
        "PPMScheme with FV3ScalarProfile needs CSAdvectionWorkspace column scratch of " *
        "size $((Nc, Nc, Nz + 1, 3Nt)), got $(size(scratch)). Build the workspace with " *
        "`column_scratch = true` (TransportModel does this automatically)."))
    Nz >= 4 || throw(ArgumentError(
        "PPMScheme with FV3ScalarProfile needs at least 4 layers, got Nz = $Nz."))
    return nothing
end

function _sweep_z_panel_fv3!(rm_4d_out, m_out, rm_4d, m, cm, scratch,
                             profile::FV3ScalarProfile, Nc, Hp, Nz, Nt;
                             flux_scale = one(eltype(m)))
    FT = eltype(m)
    backend = get_backend(rm_4d)
    kernel! = _cs_zsweep_fv3_kernel!(backend, (32, 4, 1))
    _profiled_launch_and_sync!(backend, :cs_kernel_launch_z_mt, :cs_kernel_sync_z_mt) do
        kernel!(rm_4d_out, rm_4d, m_out, m, cm, scratch, profile,
                Int32(Nz), Int32(Hp), FT(flux_scale); ndrange = (Nc, Nc, Nt))
    end
    return nothing
end

# Packed-tracer ping-pong path (production).
function _sweep_z_panels_mt_pingpong!(panels_rm_4d_out::NTuple{6},
                                      panels_m_out::NTuple{6},
                                      panels_rm_4d::NTuple{6},
                                      panels_m::NTuple{6},
                                      panels_cm::NTuple{6},
                                      mesh::CubedSphereMesh,
                                      scheme::_FV3VerticalPPM,
                                      workspace::CSAdvectionWorkspace;
                                      flux_scale = one(eltype(panels_m[1])))
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    Nt = size(panels_rm_4d[1], 4)
    _check_column_scratch(workspace.column_scratch, Nc, Nz, Nt)
    for p in 1:6
        _sweep_z_panel_fv3!(panels_rm_4d_out[p], panels_m_out[p],
                            panels_rm_4d[p], panels_m[p], panels_cm[p],
                            workspace.column_scratch, scheme.vertical, Nc, Hp, Nz, Nt;
                            flux_scale)
    end
    return panels_rm_4d_out, panels_m_out
end

# Single-tracer path: the 3-D panels are viewed as one-tracer 4-D arrays, the
# sweep writes into the workspace buffers and the interior is copied back.
function _sweep_z_panels!(panels_rm::NTuple{6}, panels_m::NTuple{6}, panels_cm::NTuple{6},
                          mesh::CubedSphereMesh, scheme::_FV3VerticalPPM,
                          workspace::CSAdvectionWorkspace;
                          flux_scale = one(eltype(panels_m[1])))
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    _check_column_scratch(workspace.column_scratch, Nc, Nz, 1)
    as_4d(a) = reshape(a, size(a)..., 1)
    for p in 1:6
        _sweep_z_panel_fv3!(as_4d(workspace.rm_A), workspace.m_A,
                            as_4d(panels_rm[p]), panels_m[p], panels_cm[p],
                            workspace.column_scratch, scheme.vertical, Nc, Hp, Nz, 1;
                            flux_scale)
        _profiled_copy!(:cs_copyback_z) do
            _copy_interior!(panels_rm[p], workspace.rm_A, Nc, Hp, Nz)
            _copy_interior!(panels_m[p], workspace.m_A, Nc, Hp, Nz)
        end
    end
    return nothing
end

# The structured (lat-lon) vertical sweep has no FV3 profile; refuse rather
# than silently use the horizontal face flux.
_require_structured_vertical(::AbstractAdvectionScheme) = nothing
_require_structured_vertical(::_FV3VerticalPPM) = throw(ArgumentError(
    "PPMScheme with FV3ScalarProfile is implemented for cubed-sphere grids only."))

# The per-panel sweeps have no column scratch; reaching them with the FV3
# profile would silently fall back to the horizontal face flux.
_fv3_needs_panel_set(f) = throw(ArgumentError(
    "PPMScheme with FV3ScalarProfile runs through `strang_split_cs!` or " *
    "`strang_split_cs_mt!`; the per-panel `$f` has no column scratch."))
_sweep_z_panel!(rm, m, cm, ::_FV3VerticalPPM, args...; kwargs...) =
    _fv3_needs_panel_set(:_sweep_z_panel!)
_sweep_z_panel_mt!(rm_4d, m, cm, ::_FV3VerticalPPM, args...; kwargs...) =
    _fv3_needs_panel_set(:_sweep_z_panel_mt!)
_sweep_z_panel_mt_pingpong!(rm_4d_out, m_out, rm_4d, m, cm, ::_FV3VerticalPPM, args...; kwargs...) =
    _fv3_needs_panel_set(:_sweep_z_panel_mt_pingpong!)
