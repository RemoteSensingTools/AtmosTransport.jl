# Reference transcription of FV3's vertical tracer profile, used to check the
# production kernel in `src/Operators/Advection/vertical_fv3_profile.jl`.
#
# `scalar_profile` (kord = 8; iv = 0 positive definite, iv = 1 signed) and
# `cs_limiters` follow `fv_mapz.F90` of
# the GFDL FV3 dynamical core as shipped with GCHP 14.7
# (`GCHP/src/GCHP_GridComp/FVdycoreCubed_GridComp/fvdycore/model/fv_mapz.F90`),
# with the column index running from the model top (1) to the surface (km).
# It was written separately from the production kernel, for the 2026-10 1-D
# column study, and stores the profile in FV3's layout: a4[1, k] layer mean,
# a4[2, k] top edge, a4[3, k] bottom edge, a4[4, k] curvature.

function fv3_cs_limiters!(a4, k, extm, iv)
    if iv == 0
        if a4[1, k] <= 0
            a4[2, k] = a4[1, k]; a4[3, k] = a4[1, k]; a4[4, k] = 0
        elseif abs(a4[3, k] - a4[2, k]) < -a4[4, k]
            if a4[1, k] + 0.25 * (a4[3, k] - a4[2, k])^2 / a4[4, k] + a4[4, k] / 12 < 0
                if a4[1, k] < a4[3, k] && a4[1, k] < a4[2, k]
                    a4[3, k] = a4[1, k]; a4[2, k] = a4[1, k]; a4[4, k] = 0
                elseif a4[3, k] > a4[2, k]
                    a4[4, k] = 3 * (a4[2, k] - a4[1, k]); a4[3, k] = a4[2, k] - a4[4, k]
                else
                    a4[4, k] = 3 * (a4[3, k] - a4[1, k]); a4[2, k] = a4[3, k] - a4[4, k]
                end
            end
        end
    else
        flat = iv == 1 ? (a4[1, k] - a4[2, k]) * (a4[1, k] - a4[3, k]) >= 0 : extm
        if flat
            a4[2, k] = a4[1, k]; a4[3, k] = a4[1, k]; a4[4, k] = 0
        else
            da1 = a4[3, k] - a4[2, k]; da2 = da1^2; a6da = a4[4, k] * da1
            if a6da < -da2
                a4[4, k] = 3 * (a4[2, k] - a4[1, k]); a4[3, k] = a4[2, k] - a4[4, k]
            elseif a6da > da2
                a4[4, k] = 3 * (a4[3, k] - a4[1, k]); a4[2, k] = a4[3, k] - a4[4, k]
            end
        end
    end
end

function fv3_scalar_profile_kord8(qbar::AbstractVector{T}, delp::AbstractVector{T}; iv = 0) where T
    km = length(qbar); a4 = zeros(T, 4, km); a4[1, :] .= qbar
    q = zeros(T, km + 1); gam = zeros(T, km + 1)
    grat = delp[2] / delp[1]; bet = grat * (grat + T(0.5))
    q[1] = ((grat + grat) * (grat + 1) * a4[1, 1] + a4[1, 2]) / bet
    gam[1] = (1 + grat * (grat + T(1.5))) / bet
    d4 = zero(T)
    for k in 2:km
        d4 = delp[k-1] / delp[k]; bet = 2 + d4 + d4 - gam[k-1]
        q[k] = (3 * (a4[1, k-1] + d4 * a4[1, k]) - q[k-1]) / bet; gam[k] = d4 / bet
    end
    a_bot = 1 + d4 * (d4 + T(1.5))
    q[km+1] = (2 * d4 * (d4 + 1) * a4[1, km] + a4[1, km-1] - a_bot * q[km]) /
              (d4 * (d4 + T(0.5)) - a_bot * gam[km])
    for k in km:-1:1
        q[k] -= gam[k] * q[k+1]
    end
    q[2] = min(q[2], max(a4[1, 1], a4[1, 2])); q[2] = max(q[2], min(a4[1, 1], a4[1, 2]))
    for k in 2:km
        gam[k] = a4[1, k] - a4[1, k-1]
    end
    for k in 3:km-1
        if gam[k-1] * gam[k+1] > 0
            q[k] = min(q[k], max(a4[1, k-1], a4[1, k])); q[k] = max(q[k], min(a4[1, k-1], a4[1, k]))
        elseif gam[k-1] > 0
            q[k] = max(q[k], min(a4[1, k-1], a4[1, k]))
        else
            q[k] = min(q[k], max(a4[1, k-1], a4[1, k]))
            iv == 0 && (q[k] = max(zero(T), q[k]))
        end
    end
    q[km] = min(q[km], max(a4[1, km-1], a4[1, km])); q[km] = max(q[km], min(a4[1, km-1], a4[1, km]))
    for k in 1:km
        a4[2, k] = q[k]; a4[3, k] = q[k+1]
    end
    extm = falses(km)
    for k in 1:km
        extm[k] = (k == 1 || k == km) ? (a4[2, k] - a4[1, k]) * (a4[3, k] - a4[1, k]) > 0 :
                                         gam[k] * gam[k+1] < 0
    end
    iv == 0 && (a4[2, 1] = max(zero(T), a4[2, 1]))
    a4[4, 1] = 3 * (2a4[1, 1] - (a4[2, 1] + a4[3, 1])); fv3_cs_limiters!(a4, 1, extm[1], 1)
    a4[4, 2] = 3 * (2a4[1, 2] - (a4[2, 2] + a4[3, 2])); fv3_cs_limiters!(a4, 2, extm[2], 2)
    for k in 3:km-2
        pmp_1 = a4[1, k] - 2gam[k+1]; lac_1 = pmp_1 + T(1.5) * gam[k+2]
        a4[2, k] = min(max(a4[2, k], min(a4[1, k], pmp_1, lac_1)), max(a4[1, k], pmp_1, lac_1))
        pmp_2 = a4[1, k] + 2gam[k]; lac_2 = pmp_2 - T(1.5) * gam[k-1]
        a4[3, k] = min(max(a4[3, k], min(a4[1, k], pmp_2, lac_2)), max(a4[1, k], pmp_2, lac_2))
        a4[4, k] = 3 * (2a4[1, k] - (a4[2, k] + a4[3, k]))
        iv == 0 && fv3_cs_limiters!(a4, k, extm[k], 0)
    end
    iv == 0 && (a4[3, km] = max(zero(T), a4[3, km]))
    for k in km-1:km
        a4[4, k] = 3 * (2a4[1, k] - (a4[2, k] + a4[3, k]))
        fv3_cs_limiters!(a4, k, extm[k], k == km - 1 ? 2 : 1)
    end
    return a4
end

# Mean of layer l's parabola between fractional positions s1 and s2 (FV3 `map1_q2`).
fv3_parabola_mean(a4, l, s1, s2) =
    a4[2, l] + (a4[4, l] + a4[3, l] - a4[2, l]) * (s1 + s2) / 2 - a4[4, l] * (s1 * (s1 + s2) + s2^2) / 3
