"""
    TM5BLDiffParameters(FT=Float64; lmax)

Precision-specific constants and bottom-up mixing depth for live TM5 Calc_Kzz.
"""
struct TM5BLDiffParameters{FT,C,G}
    constants::C
    geometry::G
    lmax::Int
end
function TM5BLDiffParameters(::Type{FT}=Float64; lmax::Integer) where {FT<:AbstractFloat}
    lmax >= 1 || throw(ArgumentError("TM5 lmax must be positive"))
    c = map(FT, TM5_BLDIFF_CONSTANTS)
    g = map(FT, TM5_GPH_CONSTANTS)
    return TM5BLDiffParameters{FT,typeof(c),typeof(g)}(c, g, Int(lmax))
end

"Number of bottom layers below a reference mid-pressure cutoff (top-down hybrid coefficients)."
function pbl_layer_count(A, B, p_min, p_surface_ref)
    p_ref(k) = (A[k] + A[k + 1] + (B[k] + B[k + 1]) * p_surface_ref) / 2
    Nz = length(A) - 1
    k = findlast(k -> p_ref(k) < p_min, 1:Nz)
    return max(1, Nz - something(k, 0))
end

# diffusion_kernels.F90:1026–1037: preserve Fortran SIGN's signed-zero rule.
@inline tm5_jqif(a, b) = !signbit(a - b)

"TM5 friction velocity, with fractional land cover and independent ECMWF/land roughness roles."
@inline function tm5_ustar(s, c)
    FT = typeof(c.gravity)
    sea = max(c.sea_ustar_min, sqrt(sqrt(s.ewss^2 + s.nsss^2) / c.sea_density))
    land = zero(FT)
    if s.lsm > zero(FT)
        zl = max(FT(0.01), s.fsr)
        zh = min(s.sr, FT(0.03))
        land = c.karman * sqrt(s.u10^2 + s.v10^2) / log(FT(10) / zh) *
               log(FT(75) / zh) / log(FT(10) / zl)
    end
    return s.lsm * land + (one(FT) - s.lsm) * sea
end

# Stored face amounts correspond to one application in the palindrome.
@inline tm5_wind_from_faces(west,east,south,north,m,dx,dy,seconds) =
    (dx*(east+west)*typeof(m)(0.5)/m/seconds,dy*(north+south)*typeof(m)(0.5)/m/seconds)

@inline function _tm5_pbl_pass(z, theta_v, u, v, theta_s, parcel, ustr, c, start)
    FT = typeof(theta_s)
    previous = zero(FT)
    h = z[length(z)]
    crossing = 0
    @inbounds for l in start:length(z)
        v2 = (u[l] - u[1])^2 + (v[l] - v[1])^2 + c.bulk_friction * ustr * ustr
        v2 = max(v2, c.wind2_min)
        rib = c.gravity * (theta_v[l] - parcel) * (z[l] - z[1]) / (theta_s * v2)
        rib += copysign(c.signed_offset, rib)
        if !tm5_jqif(c.ri_critical, rib)
            lower = l == 1 ? zero(FT) : z[l - 1]
            h = lower + (c.ri_critical - previous) / (rib - previous) * (z[l] - lower)
            crossing = l
            break
        end
        previous = rib
    end
    return h, crossing
end

"""
    tm5_calc_kzz_column!(D, out, column, params)

Allocation-free bottom-up live Calc_Kzz column, usable inside KA kernels.
`column` contains `p`, `p_geometry` (L+1 interface pressures), `t`, `q`,
`mass`, `u`, `v` (L centres), and `surface` (lsm fraction, sr/fsr metres,
u10/v10, ewss/nsss, downward-positive sshf/slhf, oro geopotential).
Normally the two pressure profiles are identical; separate profiles also
represent TM5's prognostic-geometry/midpoint-mass convention.
`out` supplies scratch vectors `g` (L+1), `pf`, `theta`, `theta_v`, `z` (L),
`ri`, `shear2`, `k_free`, `kvh` (L-1). D has L entries, with the last zero.
Returns scratch intermediates plus friction velocity, buoyancy flux, Obukhov
length, both PBL-height passes, crossing indices and guarded-interface count.

Sources: diffusion_kernels.F90:595–988; meteo__pclim.F90:6006–6020.
Unused singular branches are selected away. Non-finite entrainment keeps the
BL/free-atmosphere K; finite negative entrainment is retained and clipped
only when forming D. No threshold, ceiling or extra floor is applied.
"""
@inline function tm5_calc_kzz_column!(D, out, column, params::TM5BLDiffParameters{FT}) where FT
    c, geo = params.constants, params.geometry
    p, pg, t, q, m, u, v, s = column.p, column.p_geometry, column.t, column.q,
                                column.mass, column.u, column.v, column.surface
    g, pf, theta, thv, z = out.g, out.pf, out.theta, out.theta_v, out.z
    ri, shear2, k_free, kvh = out.ri, out.shear2, out.k_free, out.kvh
    L = length(t)
    @inbounds begin
        g[1] = s.oro / c.gravity
        for l in 1:L-1
            tv = t[l] * (one(FT) + geo.virtual_factor * q[l])
            g[l + 1] = g[l] + tv * geo.r_air * log(pg[l] / pg[l + 1]) / c.gravity
        end
        g[L + 1] = geo.top_height
        chi = one(FT) / (c.r_air / c.r_vap) - one(FT)
        for l in 1:L
            pf[l] = (p[l] + p[l + 1]) * FT(0.5)
            theta[l] = t[l] * (c.p_ref / pf[l])^(c.r_air / c.cp_air)
            thv[l] = theta[l] * (one(FT) + chi * q[l])
            f = (p[l] - pf[l]) / (p[l] - p[l + 1])
            z[l] = (g[l] - g[1]) * (one(FT) - f) + (g[l + 1] - g[1]) * f
        end
        ustar = tm5_ustar(s, c)
        ustr = max(ustar, c.ustar_min)
        w_theta = -s.sshf * (g[2] - g[1]) * c.gravity / (p[1] - p[2]) / c.cp_air
        w_q = -s.slhf * (g[2] - g[1]) * c.gravity / (p[1] - p[2]) / c.l_vap
        theta_s = thv[1]
        hv = w_theta + chi * theta[1] * w_q
        l_obukhov = -theta_s * ustr^3 / (c.gravity * c.karman * (hv + copysign(c.signed_offset, hv)))
        unstable = tm5_jqif(hv, zero(FT))
        h_pass1, crossing1 = _tm5_pbl_pass(z, thv, u, v, theta_s, theta_s, ustr, c, 1)
        third = one(FT) / FT(3)
        binh = c.surface_fraction * c.beta
        wsc = ustr * (unstable ? one(FT) - binh * h_pass1 / l_obukhov : one(FT))^third
        theta_excess = hv * c.fak / wsc
        theta_parcel = theta_s + (unstable ? theta_excess : zero(FT))
        h_pass2_raw, crossing2 = _tm5_pbl_pass(z, thv, u, v, theta_s, theta_parcel, ustr, c, 2)
        h = max(h_pass2_raw, c.h_min)
        w_star = (abs(hv) * c.gravity * h / theta_s)^third
        guarded = 0
        guarded_active = 0
        for l in 1:L-1
            zi = g[l + 1] - g[1]
            lambda = tm5_jqif(FT(1000), zi) ? FT(300) : FT(30) + FT(270) * exp(one(FT) - zi / FT(1000))
            length2 = (one(FT) / (one(FT) / lambda + one(FT) / (c.karman * zi)))^2
            dz = z[l + 1] - z[l]
            shear2[l] = max((u[l + 1] - u[l])^2 + (v[l + 1] - v[l])^2, c.shear2_min) / dz^2
            dtheta = thv[l + 1] - thv[l]
            alpha = (log(p[l + 1]) - log(pf[l])) / (log(pf[l + 1]) - log(pf[l]))
            th_ifc = thv[l] + alpha * dtheta
            ri[l] = (c.gravity / th_ifc * dtheta / dz) / shear2[l]
            kn = length2 * sqrt(shear2[l])
            stable_ri = tm5_jqif(ri[l], zero(FT))
            fs = stable_ri ? one(FT) / (one(FT) + FT(10) * ri[l] * (one(FT) + FT(8) * ri[l])) :
                            sqrt(max(one(FT) - FT(18) * ri[l], zero(FT)))
            k_free[l] = max(c.k_free_min, kn * fs)
            inside = tm5_jqif(h, zi)
            kb = k_free[l]
            if inside && !unstable
                ell2 = (one(FT) / (one(FT) / (c.karman * zi) + one(FT) / FT(450)))^2
                f = stable_ri ? one(FT) / (one(FT) + FT(10) * ri[l] * sqrt(one(FT) + ri[l])) : one(FT)
                kb = max(ell2 * sqrt(shear2[l]) * f, c.k_bl_min)
            elseif inside && unstable
                eta = zi / h
                shape = (one(FT) - eta)^2
                a = one(FT) - binh * h / l_obukhov
                wm = ustr * a^third
                km = tm5_jqif(c.surface_fraction, eta) ?
                    ustr * c.karman * zi * shape * (one(FT) - c.beta * (zi / l_obukhov))^third :
                    wm * c.karman * zi * shape
                pr = a^third / sqrt(a) + c.surface_fraction * c.karman * c.fakn * w_star / (ustr * a^third)
                kb = max(km / pr, c.k_bl_min)
            end
            ke = FT(0.2) * hv / (dtheta / dz)
            guarded += !isfinite(ke)
            entrainment = unstable && tm5_jqif(h, z[l]) && tm5_jqif(z[l + 1], h)
            guarded_active += entrainment && !isfinite(ke) && l < params.lmax
            kvh[l] = entrainment && isfinite(ke) ? ke : kb
            D[l] = l < params.lmax ? max(zero(FT), kvh[l]) * FT(2) * (m[l + 1] + m[l]) / (g[l + 2] - g[l])^2 : zero(FT)
        end
        D[L] = zero(FT)
        return (; out..., ustar, ustr, w_theta, w_q, theta_s, hv, l_obukhov,
                  theta_excess, theta_parcel, w_star, h_pass1, h_pass2_raw, h,
                  crossing1, crossing2, guarded, guarded_active)
    end
end
